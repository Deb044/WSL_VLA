"""Independent, resumable model-zoo training runs.

One run trains one task with one seed from the shared frozen base. Everything
that is easy to conflate is kept explicit and separate:

* **optimizer updates** (parameter updates) versus **micro steps** (gradient
  evaluations) versus **sample exposure** (training windows consumed);
* **compact adapter snapshots** (``adapter.npz`` + ``metadata.json``, the zoo
  population) versus the **resume state** (optimizer moments, random-generator
  counters, dataset cursor), which lives in a different directory and is never
  part of the population;
* **structural validity** (the artifact reloads and round-trips) versus
  **behavioural evaluation** (paired frozen-base/adapter rollouts), recorded by
  a later command.

Failed and poorly performing runs are never deleted; they stay on disk with
their history and failure record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..adapters.latent import factors_to_state
from ..adapters.packing import (
    PackedAdapter,
    PackedAdapterMetadata,
    load_packed_adapter,
    pack_low_rank_adapter,
    save_packed_adapter,
)
from ..contracts import POPULATION_SAMPLE_SCHEMA_VERSION, canonical_sha256, require_schema
from ..data.streams import TaskTrainingStream, TransitionCursor
from ..experiments.manifest import SampleIdentity, ZooBudget
from ..experiments.provenance import (  # noqa: F401  (writers re-exported for callers)
    atomic_write_bytes,
    atomic_write_json,
    sha256_array_tree,
    sha256_file,
)
from .bridge import assert_base_parameters_unchanged, build_adapter_spec_and_factors
from .training import initial_adapter_state, materialize_policy_params, octo_diffusion_loss


RESUME_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 2
COMPONENT_KEYS = ("vision", "language", "action_readout", "action_head")


class ResumeMismatchError(RuntimeError):
    """A saved resume state belongs to a different configuration or dataset."""


def _jax_imports():
    try:
        import flax.serialization
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError as exc:  # pragma: no cover - research environment only
        raise RuntimeError("zoo training requires the JAX/Flax/Optax research stack") from exc
    return flax.serialization, jax, jnp, optax


def _component_key(encoded_path: str) -> str:
    from .training import _leaf_branch

    located = _leaf_branch(encoded_path)
    if located is None:
        raise ValueError(f"not an adapter leaf: {encoded_path}")
    component = located[0].split("/", 1)[0]
    return "action_readout" if component == "action" else component


def component_square_sums(tree: Mapping[str, Any]) -> dict[str, Any]:
    """Sum of squares per component; works on traced JAX values."""

    import jax.numpy as jnp

    totals = {name: jnp.zeros((), dtype=jnp.float32) for name in COMPONENT_KEYS}
    for encoded, value in tree["transformer"].items():
        key = _component_key(encoded)
        totals[key] = totals[key] + jnp.sum(jnp.square(value))
    for pair in tree["diffusion"].values():
        totals["action_head"] = (
            totals["action_head"]
            + jnp.sum(jnp.square(pair["down"]))
            + jnp.sum(jnp.square(pair["up"]))
        )
    return totals


def effective_update_norms(spec: Any, factors: Mapping[str, tuple[Any, Any]]) -> dict[str, float]:
    """Frobenius norm of the dense effective update per component and site."""

    from ..adapters.packing import factors_to_updates
    from ..contracts import SITE_HEAD_KERNEL

    totals = {name: 0.0 for name in COMPONENT_KEYS}
    updates = factors_to_updates(spec, factors)
    for entry in spec.entries:
        key = (
            "action_head"
            if entry.site == SITE_HEAD_KERNEL
            else ("action_readout" if entry.component.value == "action" else entry.component.value)
        )
        totals[key] += float(np.square(np.asarray(updates[entry.parameter_path], np.float64)).sum())
    return {name: float(np.sqrt(value)) for name, value in totals.items()}


@dataclass(frozen=True)
class ZooRunIdentity:
    """Everything a resume state and a snapshot must agree on."""

    task_id: str
    suite: str
    task_index: int
    instruction: str
    seed: int
    manifest_name: str
    zoo_sha256: str
    budget: Mapping[str, Any]
    snapshot_updates: tuple[int, ...]
    purpose: str
    dataset_sha256: str
    train_episode_ids: tuple[str, ...]
    held_out_episode_ids: tuple[str, ...]
    base_sha256: str
    base_revision: str
    architecture_sha256: str
    preprocessing_sha256: str
    action_convention_sha256: str
    adapter_spec_sha256: str
    environment_profile: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "suite": self.suite,
            "task_index": int(self.task_index),
            "instruction": self.instruction,
            "seed": int(self.seed),
            "manifest_name": self.manifest_name,
            "zoo_sha256": self.zoo_sha256,
            "budget": dict(self.budget),
            "snapshot_updates": [int(value) for value in self.snapshot_updates],
            "purpose": self.purpose,
            "dataset_sha256": self.dataset_sha256,
            "train_episode_ids": list(self.train_episode_ids),
            "held_out_episode_ids": list(self.held_out_episode_ids),
            "base_sha256": self.base_sha256,
            "base_revision": self.base_revision,
            "architecture_sha256": self.architecture_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
            "action_convention_sha256": self.action_convention_sha256,
            "adapter_spec_sha256": self.adapter_spec_sha256,
            "environment_profile": self.environment_profile,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def fixed_held_out_batches(
    stream: TaskTrainingStream, *, batch_size: int, batches: int, text_processor: Any
) -> list[dict]:
    """A fixed, evenly spaced sample of held-out-episode windows.

    The same batches (and, in :class:`ZooTrainer`, the same noise keys) are
    reused at every evaluation, so the held-out loss is comparable across
    training time and against the frozen base.
    """

    index = stream.transition_index()
    wanted = min(batch_size * batches, len(index))
    if wanted < batch_size:
        raise ValueError("held-out episodes hold fewer windows than one batch")
    wanted -= wanted % batch_size
    positions = np.round(np.linspace(0, len(index) - 1, wanted)).astype(np.int64)
    examples = [stream.example(*(int(v) for v in index[int(position)])) for position in positions]
    return [
        stream.collate(examples[start : start + batch_size], text_processor)
        for start in range(0, wanted, batch_size)
    ]


@dataclass
class ZooTrainer:
    """Adapter-only training with explicit counters and an exact resume path."""

    bundle: Any
    stream: TaskTrainingStream
    held_out_stream: TaskTrainingStream
    budget: ZooBudget
    identity: ZooRunIdentity
    token_width: int
    sequence_window: int
    gradient_clip_norm: float = 1.0

    update_step: int = field(default=0, init=False)
    micro_step: int = field(default=0, init=False)
    history: list[dict] = field(default_factory=list, init=False)
    snapshots: list[dict] = field(default_factory=list, init=False)
    # Timing is split so that a short run does not report compilation as
    # throughput: gradient steps (the first one of each process includes JIT
    # compilation), held-out evaluation, and the per-update callback
    # (snapshots, the sampling check, resume writes).
    train_seconds: float = field(default=0.0, init=False)
    compile_seconds: float = field(default=0.0, init=False)
    compiled_steps: int = field(default=0, init=False)
    evaluation_seconds: float = field(default=0.0, init=False)
    callback_seconds: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        serialization, jax, jnp, optax = _jax_imports()
        self.budget.validate()
        self._serialization = serialization
        self._jax = jax
        self._jnp = jnp
        bundle = self.bundle
        self.spec, _ = build_adapter_spec_and_factors(bundle, token_width=self.token_width)
        if self.spec.sha256() != self.identity.adapter_spec_sha256:
            raise ValueError("run identity names a different AdapterSpec than the live model")
        self.adapter_state = initial_adapter_state(bundle)
        accumulation = self.budget.gradient_accumulation_steps
        self._optimizer = optax.MultiSteps(
            optax.chain(
                optax.clip_by_global_norm(self.gradient_clip_norm),
                optax.adamw(self.budget.learning_rate),
            ),
            every_k_schedule=accumulation,
        )
        self.optimizer_state = self._optimizer.init(self.adapter_state)
        self.cursor = TransitionCursor(
            self.stream.transition_count, self.budget.micro_batch_size, self.identity.seed
        )
        self._transitions = self.stream.transition_index()
        self._text_processor = bundle.pretrained_model.text_processor
        # Noise for micro step ``i`` is fold_in(base, i): the random-generator
        # state is just the counter, which the resume file stores.
        self._train_key = jax.random.PRNGKey(self.identity.seed)
        self._eval_key = jax.random.PRNGKey(self.identity.seed + 1_000_003)
        self._sampling_key = jax.random.PRNGKey(self.identity.seed + 2_000_003)
        self._base_sampling: dict[str, Any] | None = None
        self._compiled = False
        self._held_out_batches = fixed_held_out_batches(
            self.held_out_stream,
            batch_size=self.budget.micro_batch_size,
            batches=self.budget.held_out_loss_batches,
            text_processor=self._text_processor,
        )
        optimizer = self._optimizer

        # The frozen base enters compiled code as an argument (see
        # ``materialize_policy_params``); gradients are taken only with respect
        # to the adapter state, so the base cannot receive an update.
        self._base_params = bundle.research_model.params

        def loss_fn(base, state, batch, key, train):
            return octo_diffusion_loss(bundle, state, batch, key, train=train, base_params=base)

        @jax.jit
        def train_step(base, state, opt_state, batch, key):
            (loss, _), gradients = jax.value_and_grad(
                lambda value: loss_fn(base, value, batch, key, True), has_aux=True
            )(state)
            updates, next_opt_state = optimizer.update(gradients, opt_state, state)
            next_state = optax.apply_updates(state, updates)
            return (
                next_state,
                next_opt_state,
                loss,
                component_square_sums(gradients),
                component_square_sums(updates),
            )

        @jax.jit
        def eval_step(base, state, batch, key):
            return loss_fn(base, state, batch, key, False)[0]

        self._train_step = train_step
        self._eval_step = eval_step
        self.base_held_out_loss = self.held_out_loss(initial_adapter_state(bundle))

    # ------------------------------------------------------------------ metrics
    def held_out_loss(self, state: Any | None = None) -> float:
        """Mean diffusion loss on the fixed held-out batches with fixed noise."""

        jax = self._jax
        state = self.adapter_state if state is None else state
        losses = [
            float(
                self._eval_step(
                    self._base_params, state, batch, jax.random.fold_in(self._eval_key, index)
                )
            )
            for index, batch in enumerate(self._held_out_batches)
        ]
        return float(np.mean(losses))

    def held_out_sampling(self, state: Any | None = None) -> dict[str, Any]:
        """Open-loop action sampling on the fixed held-out windows (no simulator).

        The diffusion loss is 1.0 for a denoiser that predicts zero noise. Near
        or above that value the sampler does not remove noise: samples drift to
        the head's clamp and a rollout executes saturated commands. This check
        exposes that before any rollout by reporting how far sampled commands
        are from the demonstrated ones and how often they leave the controller
        bounds. Noise keys are fixed, so values are comparable across training
        time and against the frozen base.
        """

        from .bridge import unnormalization_statistics

        jax = self._jax
        bundle = self.bundle
        convention = bundle.action_convention
        state = self.adapter_state if state is None else state
        model = bundle.research_model.replace(
            params=materialize_policy_params(bundle, state, base_params=self._base_params)
        )
        statistics = unnormalization_statistics(convention)
        low = np.asarray(convention.low, dtype=np.float64)
        high = np.asarray(convention.high, dtype=np.float64)
        clamp = getattr(bundle.pretrained_model.module.heads["action"], "max_action", None)
        executed = bundle.preprocessing.executed_actions_per_plan
        absolute_error = 0.0
        executed_error = 0.0
        valid = 0
        executed_valid = 0
        outside = 0
        saturated = 0
        commands = 0
        for index, batch in enumerate(self._held_out_batches):
            sampled = np.asarray(
                jax.device_get(
                    model.sample_actions(
                        batch["observation"],
                        batch["task"],
                        unnormalization_statistics=statistics,
                        rng=jax.random.fold_in(self._sampling_key, index),
                    )
                ),
                dtype=np.float64,
            )[..., : convention.action_dim]
            # Octo samples the chunk of the most recent window step.
            target = np.asarray(batch["action"], dtype=np.float64)[:, -1]
            mask = np.asarray(batch["action_pad_mask"], dtype=bool)[:, -1]
            if sampled.shape != target.shape:
                raise ValueError(
                    f"sampled action chunk {sampled.shape} differs from targets {target.shape}"
                )
            if not np.isfinite(sampled).all():
                raise FloatingPointError("the policy sampled non-finite actions")
            difference = np.abs(sampled - target)
            absolute_error += float(difference[mask].sum())
            valid += int(mask.sum())
            executed_error += float(difference[:, :executed][mask[:, :executed]].sum())
            executed_valid += int(mask[:, :executed].sum())
            outside += int(((sampled < low - 1e-6) | (sampled > high + 1e-6)).sum())
            if clamp is not None:
                saturated += int((np.abs(sampled) >= float(clamp) - 1e-3).sum())
            commands += sampled.size
        return {
            "windows": len(self._held_out_batches) * self.budget.micro_batch_size,
            "mean_absolute_error": absolute_error / valid if valid else None,
            "executed_command_mean_absolute_error": (
                executed_error / executed_valid if executed_valid else None
            ),
            "out_of_bounds_fraction": outside / commands,
            "saturated_fraction": None if clamp is None else saturated / commands,
            "head_clamp": None if clamp is None else float(clamp),
            "noise": "fixed sampling keys per held-out batch",
        }

    def base_held_out_sampling(self) -> dict[str, Any]:
        """The same open-loop check for the frozen base (computed once)."""

        if self._base_sampling is None:
            self._base_sampling = self.held_out_sampling(initial_adapter_state(self.bundle))
        return self._base_sampling

    @property
    def sample_exposure(self) -> int:
        return self.micro_step * self.budget.micro_batch_size

    # ----------------------------------------------------------------- training
    def _next_batch(self) -> dict:
        indices = self.cursor.next_indices()
        examples = [
            self.stream.example(*(int(v) for v in self._transitions[int(position)]))
            for position in indices
        ]
        return self.stream.collate(examples, self._text_processor)

    def train_until(
        self,
        target_update: int,
        *,
        on_update: Callable[["ZooTrainer"], None] | None = None,
        log_every: int = 100,
    ) -> None:
        """Advance to ``target_update`` optimizer updates.

        ``on_update`` runs after every completed optimizer update (snapshot and
        resume writers hook in here). Held-out loss is recorded at the manifest
        cadence and at every snapshot stage.
        """

        jax = self._jax
        accumulation = self.budget.gradient_accumulation_steps
        stages = set(self.identity.snapshot_updates)
        gradient_sums = {name: 0.0 for name in COMPONENT_KEYS}
        losses: list[float] = []
        while self.update_step < target_update:
            tick = perf_counter()
            batch = self._next_batch()
            key = jax.random.fold_in(self._train_key, self.micro_step)
            (
                self.adapter_state,
                self.optimizer_state,
                loss,
                gradient_squares,
                update_squares,
            ) = self._train_step(
                self._base_params, self.adapter_state, self.optimizer_state, batch, key
            )
            self.micro_step += 1
            loss = float(loss)  # waits for the device, so the timing below is real
            elapsed = perf_counter() - tick
            self.train_seconds += elapsed
            if not self._compiled:
                self._compiled = True
                self.compile_seconds += elapsed
                self.compiled_steps += 1
            if not np.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at micro step {self.micro_step}"
                )
            losses.append(loss)
            for name in COMPONENT_KEYS:
                gradient_sums[name] += float(gradient_squares[name])
            if self.micro_step % accumulation != 0:
                continue
            self.update_step += 1
            step = self.update_step
            evaluate = step % self.budget.held_out_loss_every == 0 or step in stages
            if step == 1 or step % log_every == 0 or evaluate or step == target_update:
                record = {
                    "optimizer_update": step,
                    "micro_step": self.micro_step,
                    "sample_exposure": self.sample_exposure,
                    "train_loss": float(np.mean(losses)),
                    "gradient_norm": {
                        name: float(np.sqrt(gradient_sums[name] / accumulation))
                        for name in COMPONENT_KEYS
                    },
                    "update_norm": {
                        name: float(np.sqrt(float(update_squares[name])))
                        for name in COMPONENT_KEYS
                    },
                }
                if evaluate:
                    tick = perf_counter()
                    record["held_out_loss"] = self.held_out_loss()
                    record["held_out_loss_frozen_base"] = self.base_held_out_loss
                    self.evaluation_seconds += perf_counter() - tick
                self.history.append(record)
            gradient_sums = {name: 0.0 for name in COMPONENT_KEYS}
            losses = []
            if on_update is not None:
                tick = perf_counter()
                on_update(self)
                self.callback_seconds += perf_counter() - tick

    # ---------------------------------------------------------------- snapshots
    def packed_adapter(self, snapshot_index: int) -> PackedAdapter:
        host_state = self._jax.device_get(self.adapter_state)
        spec, factors = build_adapter_spec_and_factors(
            self.bundle, token_width=self.token_width, adapter_state=host_state
        )
        if spec != self.spec:
            raise AssertionError("adapter layout changed during training")
        first = next(iter(factors.values()))[0]
        return pack_low_rank_adapter(
            spec,
            factors,
            metadata=PackedAdapterMetadata(
                task_id=self.identity.task_id,
                suite=self.identity.suite,
                seed=self.identity.seed,
                optimizer_updates=self.update_step,
                snapshot_index=snapshot_index,
                sample_exposure=self.sample_exposure,
                source_dtype=str(np.asarray(first).dtype),
            ),
            sequence_window=self.sequence_window,
        )

    def save_snapshot(self, directory: Path, *, snapshot_index: int) -> dict:
        """Write one compact, reloadable adapter snapshot and verify the reload."""

        if directory.exists():
            raise FileExistsError(f"refusing to overwrite an existing zoo sample: {directory}")
        packed = self.packed_adapter(snapshot_index)
        staging = directory.with_name(f".{directory.name}.partial")
        if staging.exists():
            for item in staging.iterdir():
                item.unlink()
            staging.rmdir()
        staging.mkdir(parents=True)
        adapter_path = staging / "adapter.npz"
        save_packed_adapter(packed, adapter_path)
        reloaded = load_packed_adapter(adapter_path)
        if not np.array_equal(reloaded.tokens, packed.tokens):
            raise AssertionError("adapter snapshot does not reload to the saved tokens")
        base_check = assert_base_parameters_unchanged(self.bundle)
        last = next(
            (item for item in reversed(self.history) if "held_out_loss" in item), None
        )
        sampling = {
            "adapter": self.held_out_sampling(),
            "frozen_base": self.base_held_out_sampling(),
        }
        metadata = {
            "schema_version": POPULATION_SAMPLE_SCHEMA_VERSION,
            "synthetic": False,
            "purpose": self.identity.purpose,
            "sample_id": SampleIdentity(
                self.identity.task_id, self.identity.seed, snapshot_index, self.update_step
            ).sample_id,
            "run_id": f"{self.identity.task_id}/seed_{self.identity.seed}",
            "task_id": self.identity.task_id,
            "suite": self.identity.suite,
            "task_index": self.identity.task_index,
            "seed": self.identity.seed,
            "snapshot_index": snapshot_index,
            "optimizer_updates": self.update_step,
            "micro_steps": self.micro_step,
            "sample_exposure": self.sample_exposure,
            "effective_batch_size": self.budget.effective_batch_size,
            "manifest_name": self.identity.manifest_name,
            "zoo_sha256": self.identity.zoo_sha256,
            "run_identity_sha256": self.identity.sha256(),
            "base_sha256": self.identity.base_sha256,
            "base_revision": self.identity.base_revision,
            "architecture_sha256": self.identity.architecture_sha256,
            "preprocessing_sha256": self.identity.preprocessing_sha256,
            "action_convention_sha256": self.identity.action_convention_sha256,
            "adapter_spec_sha256": self.identity.adapter_spec_sha256,
            "layout_sha256": packed.layout.sha256(),
            "dataset_sha256": self.identity.dataset_sha256,
            "train_episode_ids_sha256": canonical_sha256(list(self.identity.train_episode_ids)),
            "adapter_sha256": sha256_file(adapter_path),
            "adapter_state_sha256": sha256_array_tree(self._jax.device_get(self.adapter_state)),
            "effective_update_norm": effective_update_norms(self.spec, packed.factors),
            "held_out_loss": None if last is None else last["held_out_loss"],
            "held_out_loss_frozen_base": self.base_held_out_loss,
            # Open-loop sampling on held-out demonstration windows; see
            # ``ZooTrainer.held_out_sampling``. Not a substitute for rollouts.
            "held_out_sampling": sampling,
            "validation": {
                # Reload and round-trip succeeded. Behaviour is unknown until
                # scripts/eval/evaluate_zoo_behaviour.py writes behaviour.json.
                "structurally_valid": True,
                "base_parameters_unchanged": base_check["base_sha256"],
                "behaviourally_evaluated": False,
            },
        }
        atomic_write_json(staging / "metadata.json", metadata)
        os.replace(staging, directory)
        record = {
            "snapshot_index": snapshot_index,
            "optimizer_updates": self.update_step,
            "directory": str(directory),
            "adapter_sha256": metadata["adapter_sha256"],
            "held_out_loss": metadata["held_out_loss"],
            "held_out_sampling": sampling["adapter"],
        }
        self.snapshots.append(record)
        return metadata

    def ensure_snapshot(self, directory: Path, *, snapshot_index: int) -> dict:
        """Save a snapshot, or adopt an identical one left by an interrupted run.

        A crash between writing a snapshot and writing the resume state leaves
        the snapshot on disk while training restarts before it. Reaching the
        stage again must reproduce the same adapter state exactly; anything
        else means the resumed run diverged and is reported, never overwritten.
        """

        if not directory.exists():
            return self.save_snapshot(directory, snapshot_index=snapshot_index)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        current = sha256_array_tree(self._jax.device_get(self.adapter_state))
        if (
            metadata.get("adapter_state_sha256") != current
            or metadata.get("run_identity_sha256") != self.identity.sha256()
            or sha256_file(directory / "adapter.npz") != metadata.get("adapter_sha256")
        ):
            raise ResumeMismatchError(
                f"existing snapshot {directory} differs from the resumed training state; "
                "the run is not reproducing itself. Preserve it and start a new run root."
            )
        if not any(item["optimizer_updates"] == self.update_step for item in self.snapshots):
            self.snapshots.append(
                {
                    "snapshot_index": snapshot_index,
                    "optimizer_updates": self.update_step,
                    "directory": str(directory),
                    "adapter_sha256": metadata["adapter_sha256"],
                    "held_out_loss": metadata.get("held_out_loss"),
                    "held_out_sampling": metadata.get("held_out_sampling", {}).get("adapter"),
                }
            )
        return metadata

    # ------------------------------------------------------------------- resume
    def save_resume(self, directory: Path) -> None:
        """Persist optimizer moments, generator counters, and the dataset cursor."""

        jax = self._jax
        payload = {
            "adapter_state": jax.device_get(self.adapter_state),
            "optimizer_state": jax.device_get(self.optimizer_state),
        }
        state_bytes = self._serialization.to_bytes(payload)
        atomic_write_bytes(directory / "state.msgpack", state_bytes)
        atomic_write_json(
            directory / "resume.json",
            {
                "schema_version": RESUME_SCHEMA_VERSION,
                "run_identity": self.identity.to_dict(),
                "run_identity_sha256": self.identity.sha256(),
                "update_step": self.update_step,
                "micro_step": self.micro_step,
                "random_generator": {
                    "train_seed": self.identity.seed,
                    "next_micro_step": self.micro_step,
                    "derivation": "jax.random.fold_in(PRNGKey(train_seed), micro_step)",
                },
                "dataset_cursor": self.cursor.state(),
                "state_sha256": __import__("hashlib").sha256(state_bytes).hexdigest(),
                "history": self.history,
                "snapshots": self.snapshots,
                "train_seconds": self.train_seconds,
                "compile_seconds": self.compile_seconds,
                "compiled_steps": self.compiled_steps,
                "evaluation_seconds": self.evaluation_seconds,
                "callback_seconds": self.callback_seconds,
                "base_held_out_loss": self.base_held_out_loss,
            },
        )

    def restore(self, directory: Path) -> bool:
        """Resume from ``directory`` if a state exists; reject any mismatch."""

        meta_path = directory / "resume.json"
        state_path = directory / "state.msgpack"
        if not meta_path.is_file() and not state_path.is_file():
            return False
        if not meta_path.is_file() or not state_path.is_file():
            raise ResumeMismatchError(f"incomplete resume state in {directory}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema_version") != RESUME_SCHEMA_VERSION:
            raise ResumeMismatchError("unsupported resume-state schema")
        if meta["run_identity_sha256"] != self.identity.sha256():
            current = self.identity.to_dict()
            differing = sorted(
                key for key, value in meta["run_identity"].items() if current.get(key) != value
            )
            raise ResumeMismatchError(
                "resume state was written for a different configuration or dataset; "
                f"differing fields: {differing}. Start a new run directory instead."
            )
        state_bytes = state_path.read_bytes()
        if __import__("hashlib").sha256(state_bytes).hexdigest() != meta["state_sha256"]:
            raise ResumeMismatchError("resume state file does not match its recorded hash")
        template = {
            "adapter_state": self._jax.device_get(self.adapter_state),
            "optimizer_state": self._jax.device_get(self.optimizer_state),
        }
        restored = self._serialization.from_bytes(template, state_bytes)
        self.adapter_state = restored["adapter_state"]
        self.optimizer_state = restored["optimizer_state"]
        self.update_step = int(meta["update_step"])
        self.micro_step = int(meta["micro_step"])
        self.cursor = TransitionCursor.restore(
            meta["dataset_cursor"],
            transition_count=self.stream.transition_count,
            batch_size=self.budget.micro_batch_size,
            seed=self.identity.seed,
        )
        self.history = list(meta["history"])
        self.snapshots = list(meta["snapshots"])
        self.train_seconds = float(meta["train_seconds"])
        self.compile_seconds = float(meta.get("compile_seconds", 0.0))
        self.compiled_steps = int(meta.get("compiled_steps", 0))
        self.evaluation_seconds = float(meta.get("evaluation_seconds", 0.0))
        self.callback_seconds = float(meta.get("callback_seconds", 0.0))
        if abs(float(meta["base_held_out_loss"]) - self.base_held_out_loss) > 1e-4 * max(
            1.0, abs(self.base_held_out_loss)
        ):
            raise ResumeMismatchError(
                "frozen-base held-out loss changed since the resume state was written"
            )
        return True

    def run_summary(self, *, completed: bool, failures: Sequence[Mapping[str, Any]] = ()) -> dict:
        throughput = self.micro_step / self.train_seconds if self.train_seconds > 0 else None
        # Steady state: every gradient step except the one per process that compiled.
        steady_steps = self.micro_step - self.compiled_steps
        steady_seconds = self.train_seconds - self.compile_seconds
        steady = steady_steps / steady_seconds if steady_steps > 0 and steady_seconds > 0 else None
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "completed": bool(completed),
            "failures": [dict(item) for item in failures],
            "run_identity": self.identity.to_dict(),
            "run_identity_sha256": self.identity.sha256(),
            "optimizer_updates": self.update_step,
            "micro_steps": self.micro_step,
            "sample_exposure": self.sample_exposure,
            "effective_batch_size": self.budget.effective_batch_size,
            "held_out_loss_frozen_base": self.base_held_out_loss,
            "held_out_sampling_frozen_base": self._base_sampling,
            "snapshots": self.snapshots,
            "history": self.history,
            "resources": {
                # Gradient steps only; includes the compiling first step of each process.
                "train_seconds": self.train_seconds,
                "compile_seconds": self.compile_seconds,
                "held_out_evaluation_seconds": self.evaluation_seconds,
                "snapshot_and_resume_seconds": self.callback_seconds,
                "micro_steps_per_second": throughput,
                # Use this one to size longer runs: it excludes compilation.
                "steady_micro_steps_per_second": steady,
                "seconds_per_optimizer_update": (
                    self.train_seconds / self.update_step if self.update_step else None
                ),
                **resource_peaks(),
            },
        }


def resource_peaks() -> dict[str, int | None]:
    """Peak host and accelerator memory of this process, where measurable."""

    peak_ram = None
    try:
        import resource

        peak_ram = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except (ImportError, AttributeError):  # pragma: no cover - non-POSIX hosts
        pass
    peak_vram = None
    try:
        import jax

        values = [
            int(stats["peak_bytes_in_use"])
            for device in jax.devices()
            if (stats := device.memory_stats()) and "peak_bytes_in_use" in stats
        ]
        peak_vram = max(values) if values else None
    except Exception:  # pragma: no cover - backend dependent
        pass
    return {"peak_ram_bytes": peak_ram, "peak_vram_bytes": peak_vram}


def load_zoo_adapter(sample_directory: str | Path, bundle: Any, *, token_width: int):
    """Reload one zoo sample into a deployable low-rank adapter state.

    Verifies the sample metadata, the file hash, that the stored AdapterSpec is
    the live model's, and that the frozen base is still the pinned checkpoint.
    Returns ``(adapter_state, packed_adapter, metadata)``.
    """

    directory = Path(sample_directory)
    metadata_path = directory / "metadata.json"
    adapter_path = directory / "adapter.npz"
    if not metadata_path.is_file() or not adapter_path.is_file():
        raise FileNotFoundError(f"incomplete zoo sample: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require_schema(
        metadata.get("schema_version"), POPULATION_SAMPLE_SCHEMA_VERSION, "zoo sample metadata"
    )
    if metadata.get("synthetic", True):
        raise ValueError(f"synthetic zoo sample is forbidden: {directory}")
    if sha256_file(adapter_path) != metadata.get("adapter_sha256"):
        raise ValueError(f"adapter hash mismatch: {directory}")
    packed = load_packed_adapter(adapter_path)
    live_spec, _ = build_adapter_spec_and_factors(bundle, token_width=token_width)
    if packed.spec != live_spec:
        raise ValueError(
            f"zoo sample {directory} was trained for a different base, architecture, "
            "routing, or action convention than the live model"
        )
    for name, expected in (
        ("base_sha256", bundle.base_sha256),
        ("architecture_sha256", bundle.architecture_sha256),
        ("preprocessing_sha256", bundle.preprocessing.sha256()),
        ("action_convention_sha256", bundle.action_convention.sha256()),
    ):
        if metadata.get(name) != expected:
            raise ValueError(f"zoo sample {directory} differs from the live model on {name}")
    state = factors_to_state(bundle, packed.spec, packed.factors)
    assert_base_parameters_unchanged(bundle)
    # Materialization exercises every shape check of the deployment path.
    materialize_policy_params(bundle, state)
    return state, packed, metadata
