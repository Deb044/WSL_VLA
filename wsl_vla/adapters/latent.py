"""Paths from effective updates (dense or factored) to Octo adapter state.

Two deployment routes exist and must not be confused:

* :func:`factors_to_state` installs rank-``r`` factors through the same low-rank
  path the zoo adapters use. This is the primary, rank-matched route for zoo
  adapters, autoencoder reconstructions, and generated adapters.
* :func:`effective_updates_to_state` installs dense updates through the
  zero-initialized ``decoded_*`` injection kernels. It is differentiable with
  respect to decoded tokens (latent refinement) and is otherwise a diagnostic:
  an unrestricted dense update has more capacity than a zoo adapter.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts import SITE_HEAD_KERNEL, AdapterSpec
from ..octo.training import (
    extract_decoded_transformer_params,
    initial_adapter_state,
    octo_diffusion_loss,
)
from .packing import effective_update, unpack_tokens, validate_factor_shapes


def _decoded_key(parameter_path: str) -> str:
    prefix, name = parameter_path.rsplit("/", 1)
    component, separator, layer = name.removeprefix("adapter_").rpartition("_")
    if not name.startswith("adapter_") or not separator or not layer.isdigit():
        raise ValueError(f"cannot parse transformer adapter path: {parameter_path}")
    return f"{prefix}/decoded_{component}_{layer}_kernel"


def factors_to_state(
    bundle: Any, spec: AdapterSpec, factors: Mapping[str, tuple[Any, Any]]
) -> dict[str, Any]:
    """Low-rank adapter state from ``{parameter_path: (down, up)}`` factors."""

    validate_factor_shapes(spec, factors)
    template = initial_adapter_state(bundle)
    transformer = {}
    diffusion = {}
    for entry in spec.entries:
        down, up = factors[entry.parameter_path]
        if entry.site == SITE_HEAD_KERNEL:
            if entry.parameter_path not in template["diffusion"]:
                raise ValueError(f"not an Octo action-head kernel: {entry.parameter_path}")
            diffusion[entry.parameter_path] = {"down": down, "up": up}
        else:
            transformer[f"{entry.parameter_path}_down/kernel"] = down
            transformer[f"{entry.parameter_path}_up/kernel"] = up
    if set(transformer) != set(template["transformer"]) or set(diffusion) != set(
        template["diffusion"]
    ):
        raise ValueError("factor paths do not cover the live Octo adapter layout")
    return {"transformer": transformer, "diffusion": diffusion}


def effective_updates_from_state(bundle: Any, spec: AdapterSpec, state: Mapping[str, Any]):
    """Recover the true dense update from low-rank and decoded state leaves."""

    from ..octo.bridge import build_adapter_spec_and_factors

    observed_spec, factors = build_adapter_spec_and_factors(
        bundle, token_width=spec.token_width, adapter_state=state
    )
    if observed_spec != spec:
        raise ValueError("adapter state layout differs from AdapterSpec")
    updates = {}
    decoded_transformer = state.get("decoded_transformer", {})
    decoded_diffusion = state.get("decoded_diffusion", {})
    for entry in spec.entries:
        down, up = factors[entry.parameter_path]
        value = effective_update(down, up, alpha=spec.alpha, rank=entry.rank)
        if entry.parameter_path in decoded_diffusion:
            value = value + decoded_diffusion[entry.parameter_path]
        elif decoded_transformer and entry.site != SITE_HEAD_KERNEL:
            decoded_key = _decoded_key(entry.parameter_path)
            if decoded_key not in decoded_transformer:
                raise ValueError(f"decoded transformer state lacks {decoded_key}")
            value = value + decoded_transformer[decoded_key]
        updates[entry.parameter_path] = value
    return updates


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
    return unpack_tokens(spec, jnp.where(jnp.asarray(mask), tokens, 0.0), xp=jnp)


def effective_updates_to_state(bundle: Any, spec: AdapterSpec, updates: Mapping[str, Any]):
    """Build a policy state that applies each dense effective update exactly."""

    if {entry.parameter_path for entry in spec.entries} != set(updates):
        raise ValueError("decoded update paths differ from AdapterSpec")
    state = initial_adapter_state(bundle)
    decoded_transformer = extract_decoded_transformer_params(bundle.research_model.params)
    decoded_diffusion = {}
    for entry in spec.entries:
        update = updates[entry.parameter_path]
        if entry.site == SITE_HEAD_KERNEL:
            if entry.parameter_path not in state["diffusion"]:
                raise ValueError(f"decoded path is not an Octo head kernel: {entry.parameter_path}")
            decoded_diffusion[entry.parameter_path] = update
            continue
        decoded_key = _decoded_key(entry.parameter_path)
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
