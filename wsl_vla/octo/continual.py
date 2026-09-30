"""JAX training stages for shared-state Octo continual experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..adapters.latent import effective_updates_to_state, unpack_effective_tokens_jax
from ..alignment.checkpoint import decode_alignment_latents, refine_alignment_latents
from ..contracts import AdapterSpec, AlignmentCheckpoint
from ..experiments.provenance import sha256_array_tree
from .training import adapter_value_and_grad


def _jax_imports():
    try:
        import jax
        import optax
    except ImportError as exc:  # pragma: no cover - WSL research environment
        raise RuntimeError("continual Octo training requires JAX and Optax") from exc
    return jax, optax


def tree_nbytes(tree: Any) -> int:
    """Exact storage of array leaves, excluding Python container overhead."""

    try:
        import jax

        leaves = jax.tree_util.tree_leaves(tree)
    except ImportError:
        leaves = _python_tree_leaves(tree)
    total = 0
    for leaf in leaves:
        array = np.asarray(leaf)
        if array.dtype.kind in "biufc":
            total += int(array.nbytes)
    return total


def _python_tree_leaves(tree: Any) -> list[Any]:
    if isinstance(tree, Mapping):
        result = []
        for key in sorted(tree):
            result.extend(_python_tree_leaves(tree[key]))
        return result
    if isinstance(tree, (tuple, list)):
        result = []
        for value in tree:
            result.extend(_python_tree_leaves(value))
        return result
    return [tree]


def _validate_batch(batch: Mapping[str, Any], index: int = 0) -> None:
    required = {"observation", "task", "action", "action_pad_mask"}
    if required - set(batch):
        raise ValueError(f"batch {index} lacks official Octo fields")


def _validate_batches(batches: Sequence[Mapping[str, Any]], steps: int) -> None:
    if steps <= 0:
        raise ValueError("training steps must be positive")
    if not batches:
        raise ValueError("training requires at least one real Octo batch")
    for index, batch in enumerate(batches):
        _validate_batch(batch, index)


@dataclass(frozen=True)
class LoRAStageResult:
    adapter_state: Any
    checkpoint_sha256: str
    final_task_loss: float
    update_steps: int
    micro_steps: int
    adapter_bytes: int


def train_shared_lora_stage(
    bundle: Any,
    previous_adapter_state: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]] | Callable[[], Iterable[Mapping[str, Any]]],
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
) -> LoRAStageResult:
    """Continue one adapter state; optimizer moments reset at task boundaries."""

    if steps <= 0:
        raise ValueError("training steps must be positive")
    if not callable(batches):
        _validate_batches(batches, steps)
    if learning_rate <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("learning rate and accumulation steps must be positive")
    jax, optax = _jax_imports()
    adapter_state = previous_adapter_state
    optimizer = optax.MultiSteps(
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate)),
        every_k_schedule=gradient_accumulation_steps,
    )
    optimizer_state = optimizer.init(adapter_state)

    @jax.jit
    def step_fn(state, opt_state, batch, key):
        (loss_and_metrics, gradients) = adapter_value_and_grad(bundle, state, batch, key)
        loss, _ = loss_and_metrics
        updates, next_opt_state = optimizer.update(gradients, opt_state, state)
        return optax.apply_updates(state, updates), next_opt_state, loss

    key = jax.random.PRNGKey(seed)
    micro_steps = 0
    update_steps = 0
    final_loss = None
    def stream():
        if callable(batches):
            while True:
                produced = False
                for batch in batches():
                    produced = True
                    _validate_batch(batch)
                    yield batch
                if not produced:
                    raise ValueError("batch factory produced no LoRA training batches")
        else:
            while True:
                yield from batches

    iterator = stream()
    while update_steps < steps:
        batch = next(iterator)
        key, step_key = jax.random.split(key)
        adapter_state, optimizer_state, final_loss = step_fn(
            adapter_state, optimizer_state, batch, step_key
        )
        micro_steps += 1
        if micro_steps % gradient_accumulation_steps == 0:
            update_steps += 1
    host_state = jax.device_get(adapter_state)
    return LoRAStageResult(
        adapter_state=adapter_state,
        checkpoint_sha256=sha256_array_tree(host_state),
        final_task_loss=float(final_loss),
        update_steps=update_steps,
        micro_steps=micro_steps,
        adapter_bytes=tree_nbytes(host_state),
    )


@dataclass(frozen=True)
class LatentStageResult:
    latents: Mapping[str, Any]
    adapter_state: Mapping[str, Any]
    checkpoint_sha256: str
    final_task_loss: float
    final_regularization_loss: float
    latent_drift: Mapping[str, float]
    adapter_bytes: int


def train_shared_latent_stage(
    checkpoint: AlignmentCheckpoint,
    *,
    bundle: Any,
    adapter_spec: AdapterSpec,
    previous_latents: Mapping[str, Any],
    batches: list[tuple[Mapping[str, Any], Any]],
    gammas: Mapping[str, float],
    steps: int,
    learning_rate: float,
) -> LatentStageResult:
    """Refine the previous stage's one shared latent policy in official Octo."""

    jax, _ = _jax_imports()
    refined, history = refine_alignment_latents(
        checkpoint,
        bundle=bundle,
        adapter_spec=adapter_spec,
        initial=previous_latents,
        batches=batches,
        gammas=gammas,
        steps=steps,
        learning_rate=learning_rate,
    )
    decoded = decode_alignment_latents(checkpoint, refined)
    updates = unpack_effective_tokens_jax(decoded, checkpoint.token_mask, adapter_spec)
    adapter_state = effective_updates_to_state(bundle, adapter_spec, updates)
    host_state = jax.device_get(adapter_state)
    host_latents = jax.device_get(refined)
    drift = {
        name: float(
            np.linalg.norm(
                np.asarray(host_latents[name]) - np.asarray(previous_latents[name])
            )
        )
        for name in ("vision", "language", "action")
    }
    final = history[-1]
    return LatentStageResult(
        latents=refined,
        adapter_state=adapter_state,
        checkpoint_sha256=sha256_array_tree(host_state),
        final_task_loss=float(final["task_loss"]),
        final_regularization_loss=float(final["regularizer"]),
        latent_drift=drift,
        adapter_bytes=tree_nbytes(host_state),
    )
