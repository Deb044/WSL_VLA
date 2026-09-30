"""Reload and apply a complete Methodology 1 alignment checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..adapters.latent import decoded_token_task_loss
from ..contracts import AdapterSpec, AlignmentCheckpoint
from .mapping import LinearRidgeMapper
from .models import (
    AlignmentSystem,
    EmpiricalShell,
    assert_task_loss_gradients,
    project_to_empirical_shell,
    refine_latents,
)


COMPONENT_INDEX = {"vision": 0, "language": 1, "action": 2}


def _research_imports():
    try:
        import flax.serialization
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - WSL research environment
        raise RuntimeError("alignment checkpoint loading requires JAX/Flax") from exc
    return flax.serialization, jax, jnp


def load_alignment_checkpoint(directory: str | Path) -> AlignmentCheckpoint:
    serialization, jax, jnp = _research_imports()
    root = Path(directory)
    required = {
        "alignment_params.msgpack",
        "linear_mappers.npz",
        "empirical_shells.npz",
        "token_layout.npz",
        "metadata.json",
    }
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"alignment checkpoint lacks files: {sorted(missing)}")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported alignment checkpoint schema")
    model_config = metadata["model"]
    model = AlignmentSystem(
        token_width=int(model_config["token_width"]),
        vision_feature_dim=int(model_config["vision_feature_dim"]),
        language_feature_dim=int(model_config["language_feature_dim"]),
        action_feature_dim=int(model_config["action_feature_dim"]),
        latent_dim=int(model_config["latent_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["layers"]),
        heads=int(model_config["heads"]),
        max_tokens=int(model_config["max_tokens"]),
        max_layers=int(model_config["max_layers"]),
        task_count=int(model_config["task_count"]),
        initial_temperature=float(model_config["initial_temperature"]),
    )
    with np.load(root / "token_layout.npz", allow_pickle=False) as archive:
        token_mask = np.array(archive["token_mask"], copy=True)
        component_ids = np.array(archive["component_ids"], copy=True)
        layer_ids = np.array(archive["layer_ids"], copy=True)
    token_count, token_width = token_mask.shape
    dummy = {
        "tokens": jnp.zeros((1, token_count, token_width), dtype=jnp.float32),
        "token_mask": jnp.asarray(token_mask[None]),
        "component_ids": jnp.asarray(component_ids[None]),
        "layer_ids": jnp.asarray(layer_ids[None]),
        "vision_features": jnp.zeros((1, 1, model_config["vision_feature_dim"])),
        "vision_mask": jnp.ones((1, 1), dtype=bool),
        "language_features": jnp.zeros((1, model_config["language_feature_dim"])),
        "action_features": jnp.zeros((1, model_config["action_feature_dim"])),
    }
    variables = model.init(jax.random.PRNGKey(0), *dummy.values(), train=False)
    params = serialization.from_bytes(
        variables["params"], (root / "alignment_params.msgpack").read_bytes()
    )

    mappers = {}
    with np.load(root / "linear_mappers.npz", allow_pickle=False) as archive:
        for name in COMPONENT_INDEX:
            mappers[name] = LinearRidgeMapper(
                coefficient=np.array(archive[f"{name}_coefficient"], copy=True),
                intercept=np.array(archive[f"{name}_intercept"], copy=True),
                latent_shape=tuple(int(value) for value in archive[f"{name}_latent_shape"]),
                ridge_alpha=float(metadata["selected_ridge_alphas"][name]),
            )
    shells = {}
    with np.load(root / "empirical_shells.npz", allow_pickle=False) as archive:
        for name in COMPONENT_INDEX:
            shells[name] = EmpiricalShell(
                center=jnp.asarray(archive[f"{name}_center"]),
                radius=jnp.asarray(archive[f"{name}_radius"]),
            )
    checkpoint = AlignmentCheckpoint(
        params=params,
        model=model,
        mappers=mappers,
        shells=shells,
        token_mask=token_mask,
        component_ids=component_ids,
        layer_ids=layer_ids,
        metadata=metadata,
    )
    checkpoint.validate()
    return checkpoint


def encode_and_map_evidence(
    checkpoint: AlignmentCheckpoint,
    *,
    vision_features: Any,
    vision_mask: Any,
    language_features: Any,
    action_features: Any,
) -> dict[str, Any]:
    """Project evidence, predict full modality sequences, and enter the shell."""

    _, _, jnp = _research_imports()
    evidence = checkpoint.model.apply(
        {"params": checkpoint.params},
        jnp.asarray(vision_features)[None],
        jnp.asarray(vision_mask)[None],
        jnp.asarray(language_features)[None],
        jnp.asarray(action_features)[None],
        method=checkpoint.model.encode_evidence,
    )
    initial = {}
    for name in COMPONENT_INDEX:
        projected = np.asarray(evidence[name][0])
        mapped = checkpoint.mappers[name].predict(projected)
        initial[name] = project_to_empirical_shell(jnp.asarray(mapped), checkpoint.shells[name])
    return initial


def compose_latent_sequence(
    checkpoint: AlignmentCheckpoint, latents: Mapping[str, Any]
):
    _, _, jnp = _research_imports()
    if set(latents) != set(COMPONENT_INDEX):
        raise ValueError("latents must contain vision, language, and action")
    latent_dim = next(iter(latents.values())).shape[-1]
    sequence = jnp.zeros((len(checkpoint.component_ids), latent_dim), dtype=jnp.float32)
    for name, component_index in COMPONENT_INDEX.items():
        positions = np.flatnonzero(
            (checkpoint.component_ids == component_index)
            & np.any(checkpoint.token_mask, axis=-1)
        )
        if latents[name].shape != (len(positions), latent_dim):
            raise ValueError(f"{name} latent shape differs from checkpoint layout")
        sequence = sequence.at[positions].set(latents[name])
    return sequence


def decode_alignment_latents(checkpoint: AlignmentCheckpoint, latents: Mapping[str, Any]):
    _, _, jnp = _research_imports()
    sequence = compose_latent_sequence(checkpoint, latents)
    decoded = checkpoint.model.apply(
        {"params": checkpoint.params},
        sequence[None],
        jnp.asarray(checkpoint.token_mask[None]),
        jnp.asarray(checkpoint.component_ids[None]),
        jnp.asarray(checkpoint.layer_ids[None]),
        train=False,
        method=checkpoint.model.decode,
    )
    return decoded[0]


def refine_alignment_latents(
    checkpoint: AlignmentCheckpoint,
    *,
    bundle: Any,
    adapter_spec: AdapterSpec,
    initial: Mapping[str, Any],
    batches: Any,
    gammas: Mapping[str, float],
    steps: int,
    learning_rate: float,
):
    """Refine mapped latents through decoder, Octo, and official diffusion loss."""

    def task_loss(values, batch_and_rng):
        batch, rng = batch_and_rng
        decoded = decode_alignment_latents(checkpoint, values)
        loss_and_metrics = decoded_token_task_loss(
            bundle,
            adapter_spec,
            decoded,
            checkpoint.token_mask,
            batch,
            rng,
            train=True,
        )
        return loss_and_metrics[0]

    # Publication runs fail before optimization if any modality is detached
    # from the real decoded-adapter -> Octo -> diffusion-loss computation.
    first_batch = next(iter(batches())) if callable(batches) else batches[0]
    assert_task_loss_gradients(
        initial,
        task_loss=lambda values: task_loss(values, first_batch),
    )
    return refine_latents(
        initial,
        task_loss=task_loss,
        batches=batches,
        gammas=gammas,
        steps=steps,
        learning_rate=learning_rate,
    )
