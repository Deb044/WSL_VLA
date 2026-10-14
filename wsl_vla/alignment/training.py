"""Paired alignment training: ``aligned`` versus ``reconstruction_only``.

The two primary conditions differ in exactly one thing: the control drops the
contrastive terms. Architecture, initialization (same seed), the sequence of
training batches, optimizer exposure, evidence classification over training-
task identities, and checkpoint selection by validation *reconstruction* are
identical, and each of those is fingerprinted in the checkpoint so the pairing
can be audited afterwards.
"""

from __future__ import annotations

import hashlib
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import COMPONENT_NAMES, canonical_sha256
from ..experiments.manifest import ALIGNMENT_CONDITIONS, ExperimentManifest
from ..experiments.provenance import sha256_array_tree
from .archive import AlignmentArchive
from .mapping import select_ridge_mapper
from .models import (
    AlignmentSystem,
    bidirectional_token_contrastive_loss,
    component_balanced_reconstruction_loss,
    require_jax,
)
from .windows import LayoutBinding, floored_entry_scales


CONFIG_KEYS_FROM_CONFIG = (
    "latent_dim",
    "transformer_width",
    "encoder_layers",
    "decoder_layers",
    "attention_heads",
    "mlp_ratio",
    "evidence_hidden_dim",
    "initial_temperature",
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "gradient_accumulation_steps",
    "reconstruction_weight",
    "contrastive_weight",
    "auxiliary_classification_weight",
    "rms_floor",
    "window_chunk",
)
CONFIG_KEYS_FROM_MANIFEST = (
    "tasks_per_batch",
    "samples_per_task",
    "optimizer_updates",
    "validation_every",
)
OVERRIDABLE_KEYS = (*CONFIG_KEYS_FROM_CONFIG, *CONFIG_KEYS_FROM_MANIFEST)
INTEGER_KEYS = {
    "latent_dim", "transformer_width", "encoder_layers", "decoder_layers", "attention_heads",
    "mlp_ratio", "evidence_hidden_dim", "gradient_accumulation_steps", "tasks_per_batch",
    "samples_per_task", "optimizer_updates", "validation_every", "window_chunk",
}


def resolve_alignment_config(
    *,
    alignment_config: Mapping[str, Any],
    manifest: ExperimentManifest,
    archive: AlignmentArchive,
    condition: str,
    seed: int,
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Fully resolved effective configuration; nothing is defaulted in code.

    Precedence is command line > experiment manifest (batch composition and
    update budget) / research configuration (model and optimizer). A value that
    is missing everywhere is an error, and every value records where it came
    from.
    """

    if condition not in ALIGNMENT_CONDITIONS:
        raise ValueError(f"condition must be one of {ALIGNMENT_CONDITIONS}")
    if seed not in manifest.alignment.seeds:
        raise ValueError(f"alignment seed {seed} is not declared by experiment {manifest.name!r}")
    unknown = sorted(set(overrides) - set(OVERRIDABLE_KEYS))
    if unknown:
        raise ValueError(f"unknown alignment overrides: {unknown}")
    manifest_values = manifest.alignment.to_dict()
    resolved: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for key in OVERRIDABLE_KEYS:
        if overrides.get(key) is not None:
            value, source = overrides[key], "command_line"
        elif key in CONFIG_KEYS_FROM_MANIFEST:
            value, source = manifest_values[key], "experiment_manifest"
        elif key in alignment_config:
            value, source = alignment_config[key], "research_config"
        else:
            raise KeyError(f"alignment setting {key!r} is not configured anywhere")
        resolved[key] = int(value) if key in INTEGER_KEYS else float(value)
        sources[key] = source
    train_tasks = archive.train_task_ids
    requested_tasks = resolved["tasks_per_batch"]
    effective_tasks = min(requested_tasks, len(train_tasks))
    if effective_tasks < 2:
        raise ValueError("alignment batches need at least two distinct training tasks")
    samples_available = min(
        int(np.sum(archive.sample_task_rows[archive.indices("train")] == row))
        for row in archive.task_rows("train")
    )
    if resolved["samples_per_task"] > samples_available:
        raise ValueError(
            f"samples_per_task={resolved['samples_per_task']} exceeds the "
            f"{samples_available} adapter samples available per training task"
        )
    modality_weights = {
        name: float(alignment_config["modality_alignment_weights"][name]) for name in COMPONENT_NAMES
    }
    if resolved["reconstruction_weight"] <= 0 or resolved["contrastive_weight"] <= 0:
        raise ValueError(
            "reconstruction and contrastive weights must be positive; the control is "
            "selected with condition=reconstruction_only, not by zeroing a weight"
        )
    if resolved["transformer_width"] % resolved["attention_heads"]:
        raise ValueError("transformer_width must be divisible by attention_heads")
    sizes = archive.layout.embedding_sizes
    config = {
        "condition": condition,
        "seed": int(seed),
        "experiment": manifest.name,
        "fold": archive.manifest["fold"]["name"],
        "integrity_only": archive.integrity_only,
        **resolved,
        "requested_tasks_per_batch": requested_tasks,
        "tasks_per_batch": effective_tasks,
        "micro_batch_size": effective_tasks * resolved["samples_per_task"],
        # The only difference between the two primary conditions.
        "effective_contrastive_weight": (
            resolved["contrastive_weight"] if condition == "aligned" else 0.0
        ),
        "modality_alignment_weights": modality_weights,
        "ridge_alpha_candidates": [float(v) for v in alignment_config["ridge_alpha_candidates"]],
        "token_width": archive.layout.token_width,
        "sequence_window": archive.layout.sequence_window,
        "embedding_sizes": [
            int(sizes[name]) for name in ("component", "layer", "entry", "row", "column_window")
        ],
        "train_task_count": len(train_tasks),
        "evidence_feature_dims": {
            name: int(archive.evidence[name].shape[-1]) for name in COMPONENT_NAMES
        },
        "negative_set": (
            "the tasks inside one micro-batch; gradient accumulation adds optimizer "
            "exposure, not negatives"
        ),
        "checkpoint_selection": "minimum validation component-balanced reconstruction loss",
        "sources": sources,
    }
    config["sha256"] = canonical_sha256(config)
    return config


MODEL_CONFIG_KEYS = (
    "token_width", "latent_dim", "transformer_width", "encoder_layers", "decoder_layers",
    "attention_heads", "mlp_ratio", "evidence_hidden_dim", "embedding_sizes",
    "train_task_count", "initial_temperature",
)


def build_alignment_model(config: Mapping[str, Any]) -> AlignmentSystem:
    """Construct the model from a resolved configuration (the only constructor path)."""

    require_jax()
    missing = [key for key in MODEL_CONFIG_KEYS if key not in config]
    if missing:
        raise KeyError(f"resolved alignment configuration lacks model fields: {missing}")
    return AlignmentSystem(
        token_width=int(config["token_width"]),
        latent_dim=int(config["latent_dim"]),
        width=int(config["transformer_width"]),
        encoder_layers=int(config["encoder_layers"]),
        decoder_layers=int(config["decoder_layers"]),
        heads=int(config["attention_heads"]),
        mlp_ratio=int(config["mlp_ratio"]),
        evidence_hidden_dim=int(config["evidence_hidden_dim"]),
        embedding_sizes=tuple(int(value) for value in config["embedding_sizes"]),
        train_task_count=int(config["train_task_count"]),
        initial_temperature=float(config["initial_temperature"]),
    )


def initialize_alignment_params(model: AlignmentSystem, config: Mapping[str, Any]) -> Any:
    """Initialize from the seed alone, on one tiny window.

    Parameter shapes depend only on the resolved configuration, so the same
    seed gives both conditions bit-identical initial parameters, and reloading
    a checkpoint never needs a full-size forward pass.
    """

    import jax
    import jax.numpy as jnp

    window = 2
    identities = {
        name: jnp.zeros((1, window), dtype=jnp.int32)
        for name in ("component", "layer", "entry", "row", "column_window")
    }
    dims = config["evidence_feature_dims"]
    evidence = {
        "vision": jnp.zeros((1, 1, dims["vision"])),
        "vision_mask": jnp.ones((1, 1), dtype=bool),
        "language": jnp.zeros((1, 1, dims["language"])),
        "language_mask": jnp.ones((1, 1), dtype=bool),
        "action": jnp.zeros((1, 1, 1, dims["action"])),
        "action_mask": jnp.ones((1, 1, 1), dtype=bool),
    }
    variables = model.init(
        jax.random.PRNGKey(int(config["seed"])),
        jnp.zeros((1, window, int(config["token_width"]))),
        jnp.ones((1, window), dtype=bool),
        identities,
        evidence,
    )
    return variables["params"]


def build_batch_schedule(
    train_indices: Sequence[int],
    sample_task_rows: np.ndarray,
    *,
    tasks_per_batch: int,
    samples_per_task: int,
    micro_batches: int,
    seed: int,
) -> np.ndarray:
    """Deterministic ``[micro_batches, tasks * samples]`` sample indices.

    Every batch holds ``tasks_per_batch`` *distinct* tasks and
    ``samples_per_task`` distinct adapter samples of each. Tasks are drawn from
    reshuffled passes over the training tasks, so they are visited equally
    often. The schedule depends only on its arguments - never on the condition.
    """

    indices = np.asarray(train_indices, dtype=np.int64)
    rows = np.asarray(sample_task_rows)[indices]
    tasks = np.unique(rows)
    if tasks_per_batch < 2 or tasks_per_batch > len(tasks):
        raise ValueError("tasks_per_batch must lie between 2 and the number of training tasks")
    members = {int(task): indices[rows == task] for task in tasks}
    if min(len(value) for value in members.values()) < samples_per_task:
        raise ValueError("a training task has fewer adapter samples than samples_per_task")
    rng = np.random.default_rng(seed)
    queue: list[int] = []
    schedule = np.empty((micro_batches, tasks_per_batch * samples_per_task), dtype=np.int64)
    for batch in range(micro_batches):
        chosen: list[int] = []
        while len(chosen) < tasks_per_batch:
            if not queue:
                queue = [int(task) for task in rng.permutation(tasks)]
            candidate = queue.pop(0)
            if candidate not in chosen:
                chosen.append(candidate)
        schedule[batch] = np.concatenate(
            [rng.choice(members[task], size=samples_per_task, replace=False) for task in chosen]
        )
    return schedule


def schedule_sha256(schedule: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(schedule).tobytes()).hexdigest()


class AlignmentObjective:
    """Loss and evaluation functions for one resolved configuration."""

    def __init__(self, config: Mapping[str, Any], archive: AlignmentArchive):
        require_jax()
        import jax
        import jax.numpy as jnp
        import optax

        self.config = config
        self.model = build_alignment_model(config)
        self.entry_scales = floored_entry_scales(archive.entry_rms, floor=float(config["rms_floor"]))
        self.binding = LayoutBinding(
            archive.layout, self.entry_scales, window_chunk=int(config["window_chunk"])
        )
        self.positions = {
            name: jnp.asarray(archive.layout.component_positions(name)) for name in COMPONENT_NAMES
        }
        self._jax = jax
        self._jnp = jnp
        self._optax = optax

    def loss(self, params, batch, *, contrastive_weight: float):
        """Total loss and metrics. ``contrastive_weight == 0`` builds no contrastive graph."""

        jnp, optax = self._jnp, self._optax
        config = self.config
        reconstruction, latents, targets = self.binding.autoencode(
            self.model, params, batch["tokens"]
        )
        reconstruction_loss, per_component = component_balanced_reconstruction_loss(
            reconstruction, targets, self.binding.token_mask, self.binding.component_ids
        )
        embeddings = self.model.apply({"params": params}, batch["evidence"], method=self.model.encode_evidence)
        logits = self.model.apply({"params": params}, embeddings, method=self.model.classify)
        classification = jnp.mean(
            jnp.stack(
                [
                    jnp.mean(
                        optax.softmax_cross_entropy_with_integer_labels(
                            logits[name], batch["class_labels"]
                        )
                    )
                    for name in COMPONENT_NAMES
                ]
            )
        )
        total = (
            float(config["reconstruction_weight"]) * reconstruction_loss
            + float(config["auxiliary_classification_weight"]) * classification
        )
        metrics = {
            "reconstruction": reconstruction_loss,
            "classification": classification,
            **{f"reconstruction_{name}": per_component[name] for name in COMPONENT_NAMES},
        }
        if contrastive_weight > 0:
            temperatures = self.model.apply({"params": params}, method=self.model.temperatures)
            contrastive = jnp.zeros((), dtype=reconstruction_loss.dtype)
            for name in COMPONENT_NAMES:
                stream = latents[:, self.positions[name]]
                value = bidirectional_token_contrastive_loss(
                    stream,
                    jnp.ones(stream.shape[:2], dtype=bool),
                    embeddings[name],
                    batch["task_labels"],
                    temperature=temperatures[name],
                )
                metrics[f"contrastive_{name}"] = value
                contrastive = contrastive + float(config["modality_alignment_weights"][name]) * value
            metrics["contrastive"] = contrastive
            total = total + contrastive_weight * contrastive
        metrics["total"] = total
        return total, metrics

    def reconstruction_per_sample(self, params, tokens):
        """``[batch, components]`` normalized reconstruction error (no gradients)."""

        jnp = self._jnp
        reconstruction, _, targets = self.binding.autoencode(self.model, params, tokens)
        mask = self.binding.token_mask.astype(reconstruction.dtype)
        per_token = jnp.sum(jnp.square(reconstruction - targets) * mask, axis=-1)
        valid = jnp.sum(mask, axis=-1)
        values = []
        for index in range(len(COMPONENT_NAMES)):
            selected = jnp.asarray(self.binding.component_ids == index, dtype=reconstruction.dtype)
            values.append(jnp.sum(per_token * selected, axis=-1) / jnp.sum(valid * selected))
        return jnp.stack(values, axis=-1)


def make_batch(archive: AlignmentArchive, sample_indices: np.ndarray) -> dict[str, Any]:
    """Host batch: tokens from the memory-mapped shards, evidence gathered per task."""

    rows = archive.sample_task_rows[sample_indices]
    return {
        "tokens": archive.tokens(sample_indices),
        "evidence": archive.task_evidence(rows),
        "task_labels": rows.astype(np.int32),
        "class_labels": archive.task_train_class[rows].astype(np.int32),
    }


def evaluate_reconstruction(
    objective: AlignmentObjective, eval_step, params, archive: AlignmentArchive,
    sample_indices: np.ndarray, *, batch_size: int,
) -> dict[str, Any]:
    """Mean per-sample, per-component normalized reconstruction over a split."""

    indices = np.asarray(sample_indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("cannot evaluate reconstruction on an empty split")
    rows = np.zeros((len(indices), len(COMPONENT_NAMES)), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        # Keep one compiled batch shape: repeat the tail sample, then drop it.
        padded = np.concatenate([chunk, np.repeat(chunk[-1:], batch_size - len(chunk))])
        values = np.asarray(eval_step(params, archive.tokens(padded)))
        rows[start : start + len(chunk)] = values[: len(chunk)]
    per_component = rows.mean(axis=0)
    return {
        "loss": float(per_component.mean()),
        "per_component": {name: float(per_component[i]) for i, name in enumerate(COMPONENT_NAMES)},
        "samples": int(len(indices)),
    }


def train_alignment(
    config: Mapping[str, Any],
    archive: AlignmentArchive,
    *,
    log: Any = print,
) -> dict[str, Any]:
    """Train one condition and return its best parameters and audit fingerprints."""

    require_jax()
    import jax
    import optax

    objective = AlignmentObjective(config, archive)
    params = initialize_alignment_params(objective.model, config)
    initial_sha256 = sha256_array_tree(jax.device_get(params))
    accumulation = int(config["gradient_accumulation_steps"])
    updates_total = int(config["optimizer_updates"])
    inner = optax.chain(
        optax.clip_by_global_norm(float(config["gradient_clip_norm"])),
        optax.adamw(float(config["learning_rate"]), weight_decay=float(config["weight_decay"])),
    )
    optimizer = optax.MultiSteps(inner, every_k_schedule=accumulation) if accumulation > 1 else inner
    optimizer_state = optimizer.init(params)
    contrastive_weight = float(config["effective_contrastive_weight"])

    @jax.jit
    def train_step(current, opt_state, batch):
        (_, metrics), gradients = jax.value_and_grad(
            lambda value: objective.loss(value, batch, contrastive_weight=contrastive_weight),
            has_aux=True,
        )(current)
        updates, next_state = optimizer.update(gradients, opt_state, current)
        return optax.apply_updates(current, updates), next_state, metrics

    eval_step = jax.jit(objective.reconstruction_per_sample)
    train_indices = archive.indices("train")
    validation_indices = archive.indices("validation")
    schedule = build_batch_schedule(
        train_indices,
        archive.sample_task_rows,
        tasks_per_batch=int(config["tasks_per_batch"]),
        samples_per_task=int(config["samples_per_task"]),
        micro_batches=updates_total * accumulation,
        seed=int(config["seed"]),
    )
    batch_size = int(config["micro_batch_size"])
    best = {"loss": np.inf, "update": 0, "params": jax.device_get(params)}
    history = []
    interval: list[dict[str, float]] = []
    started = perf_counter()
    for micro in range(len(schedule)):
        params, optimizer_state, metrics = train_step(
            params, optimizer_state, make_batch(archive, schedule[micro])
        )
        values = {name: float(value) for name, value in metrics.items()}
        if not np.isfinite(values["total"]):
            raise FloatingPointError(f"non-finite alignment loss at micro step {micro + 1}")
        interval.append(values)
        if (micro + 1) % accumulation:
            continue
        update = (micro + 1) // accumulation
        if update % int(config["validation_every"]) and update != updates_total:
            continue
        validation = evaluate_reconstruction(
            objective, eval_step, params, archive, validation_indices, batch_size=batch_size
        )
        record = {
            "optimizer_update": update,
            "micro_step": micro + 1,
            "train": {
                name: float(np.mean([row[name] for row in interval])) for name in interval[0]
            },
            "validation_reconstruction": validation["loss"],
            "validation_reconstruction_per_component": validation["per_component"],
            "seconds": perf_counter() - started,
        }
        history.append(record)
        interval = []
        if validation["loss"] < best["loss"]:
            best = {"loss": validation["loss"], "update": update, "params": jax.device_get(params)}
        log(
            f"[{config['condition']}] update {update}/{updates_total} "
            f"train_rec={record['train']['reconstruction']:.5g} "
            f"val_rec={validation['loss']:.5g} best={best['loss']:.5g}@{best['update']}"
        )
    seconds = perf_counter() - started
    return {
        "objective": objective,
        "params": best["params"],
        "selected_update": int(best["update"]),
        "best_validation_reconstruction": float(best["loss"]),
        "history": history,
        "initial_params_sha256": initial_sha256,
        "batch_schedule_sha256": schedule_sha256(schedule),
        "optimizer_exposure": {
            "optimizer_updates": updates_total,
            "micro_batches": int(len(schedule)),
            "adapter_samples_seen": int(schedule.size),
        },
        "resources": {
            "train_seconds": seconds,
            "seconds_per_optimizer_update": seconds / updates_total,
        },
    }


def fit_alignment_geometry(
    objective: AlignmentObjective,
    params: Any,
    archive: AlignmentArchive,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Task-mean latents, evidence embeddings, ridge mappers, and empirical shells.

    * Mapper targets are complete modality latent sequences averaged over each
      training task's seeds and snapshots; the ridge strength is selected on
      validation tasks only.
    * Shell centres and radii use training latents only.
    """

    import jax
    import jax.numpy as jnp

    model, binding, layout = objective.model, objective.binding, archive.layout
    device_params = jax.device_put(params)
    encode_compiled = jax.jit(lambda values, tokens: binding.encode(model, values, tokens))
    embed_compiled = jax.jit(
        lambda values, evidence: model.apply(
            {"params": values}, evidence, method=model.encode_evidence
        )
    )

    def encode(tokens):
        return encode_compiled(device_params, tokens)

    def embed(evidence):
        return embed_compiled(device_params, evidence)

    task_count = len(archive.task_ids)
    latent_dim = int(objective.config["latent_dim"])
    sums = np.zeros((task_count, layout.token_count, latent_dim), dtype=np.float32)
    counts = np.zeros(task_count, dtype=np.int64)
    all_indices = np.arange(len(archive.sample_ids))

    def batches(indices):
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            padded = np.concatenate([chunk, np.repeat(chunk[-1:], batch_size - len(chunk))])
            yield chunk, np.asarray(encode(archive.tokens(padded)))[: len(chunk)]

    for chunk, latents in batches(all_indices):
        for sample, value in zip(chunk, latents):
            row = archive.sample_task_rows[sample]
            sums[row] += value
            counts[row] += 1
    if np.any(counts == 0):
        raise ValueError("an archive task has no adapter samples")
    task_means = sums / counts[:, None, None]

    train_indices = archive.indices("train")
    center = np.zeros((layout.token_count, latent_dim), dtype=np.float64)
    for _, latents in batches(train_indices):
        center += latents.sum(axis=0)
    center /= len(train_indices)
    radius = np.zeros(layout.token_count, dtype=np.float64)
    for _, latents in batches(train_indices):
        radius += np.linalg.norm(latents - center[None], axis=-1).sum(axis=0)
    radius = np.maximum(radius / len(train_indices), 1e-6)

    evidence = {
        name: np.asarray(value)
        for name, value in embed(
            {name: jnp.asarray(value) for name, value in archive.task_evidence(np.arange(task_count)).items()}
        ).items()
    }
    train_rows = archive.task_rows("train")
    validation_rows = archive.task_rows("validation")
    mappers = {}
    mapper_selection = {}
    shells = {}
    for name in COMPONENT_NAMES:
        positions = layout.component_positions(name)
        mapper, errors = select_ridge_mapper(
            evidence[name][train_rows],
            task_means[train_rows][:, positions],
            evidence[name][validation_rows],
            task_means[validation_rows][:, positions],
            ridge_alphas=objective.config["ridge_alpha_candidates"],
        )
        mappers[name] = mapper
        mapper_selection[name] = {
            "selected_ridge_alpha": mapper.ridge_alpha,
            "validation_error_by_alpha": {str(alpha): value for alpha, value in errors.items()},
            "selected_on": (
                "training tasks (integrity run only)"
                if archive.integrity_only
                else "validation tasks"
            ),
            "fit_rows": "one task-mean target per training task",
        }
        shells[name] = {
            "center": center[positions].astype(np.float32),
            "radius": radius[positions].astype(np.float32),
        }
    return {
        "task_mean_latents": task_means,
        "task_evidence_embeddings": evidence,
        "mappers": mappers,
        "mapper_selection": mapper_selection,
        "shells": shells,
        "shell_fit": {"split": "train", "samples": int(len(train_indices))},
    }
