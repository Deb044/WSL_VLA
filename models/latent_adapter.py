"""Differentiable path from decoded effective-update tokens to Octo adapters."""

from __future__ import annotations

from typing import Any, Mapping

from core.contracts import AdapterSpec
from models.octo_training import (
    extract_decoded_transformer_params,
    initial_adapter_state,
    octo_diffusion_loss,
)


def _jax():
    try:
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - research environment only
        raise RuntimeError("decoded adapter application requires JAX") from exc
    return jnp


def unpack_effective_tokens_jax(tokens: Any, mask: Any, spec: AdapterSpec) -> dict[str, Any]:
    """Invert canonical row-window packing using JAX operations."""

    jnp = _jax()
    if tokens.ndim != 2 or mask.shape != tokens.shape:
        raise ValueError("decoded tokens and mask must have shape [tokens, width]")
    if tokens.shape[1] != spec.token_width:
        raise ValueError("decoded token width differs from AdapterSpec")
    cursor = 0
    updates = {}
    for entry in spec.entries:
        chunks_per_row = (entry.output_dim + spec.token_width - 1) // spec.token_width
        rows = []
        for _ in range(entry.input_dim):
            chunks = [tokens[cursor + offset] * mask[cursor + offset] for offset in range(chunks_per_row)]
            cursor += chunks_per_row
            rows.append(jnp.concatenate(chunks)[: entry.output_dim])
        updates[entry.parameter_path] = jnp.stack(rows)
    if cursor != tokens.shape[0]:
        raise ValueError("decoded token count differs from AdapterSpec")
    return updates


def effective_updates_to_state(bundle: Any, spec: AdapterSpec, updates: Mapping[str, Any]):
    """Build a policy state that applies each decoded effective update exactly."""

    if {entry.parameter_path for entry in spec.entries} != set(updates):
        raise ValueError("decoded update paths differ from AdapterSpec")
    state = initial_adapter_state(bundle)
    decoded_transformer = extract_decoded_transformer_params(
        bundle.research_model.params
    )
    decoded_diffusion = {}
    for entry in spec.entries:
        update = updates[entry.parameter_path]
        if entry.parameter_path in state["diffusion"]:
            decoded_diffusion[entry.parameter_path] = update
            continue
        prefix, name = entry.parameter_path.rsplit("/", 1)
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(f"cannot parse transformer adapter path: {entry.parameter_path}")
        component, layer = parts
        component = component.removeprefix("adapter_")
        decoded_key = f"{prefix}/decoded_{component}_{layer}_kernel"
        if decoded_key not in decoded_transformer:
            raise ValueError(f"decoded path is not an Octo adapter: {entry.parameter_path}")
        decoded_transformer[decoded_key] = update
    return {
        "transformer": state["transformer"],
        "diffusion": state["diffusion"],
        "decoded_transformer": decoded_transformer,
        "decoded_diffusion": decoded_diffusion,
    }


def decoded_token_task_loss(
    bundle: Any,
    spec: AdapterSpec,
    decoded_tokens: Any,
    token_mask: Any,
    batch: Mapping[str, Any],
    rng: Any,
    *,
    train: bool,
):
    updates = unpack_effective_tokens_jax(decoded_tokens, token_mask, spec)
    state = effective_updates_to_state(bundle, spec, updates)
    return octo_diffusion_loss(bundle, state, batch, rng, train=train)
