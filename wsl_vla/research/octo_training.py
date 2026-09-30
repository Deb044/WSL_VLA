"""Adapter-only official Octo task loss and optimization primitives."""

from __future__ import annotations

from typing import Any, Mapping

from .flax_adapters import apply_dense_kernel_updates, apply_external_adapters, path_string


def _imports():
    try:
        from flax import traverse_util
        from flax.core import freeze, unfreeze
        import jax
    except ImportError as exc:  # pragma: no cover - research environment only
        raise RuntimeError("official Octo training requires the JAX/Flax research extra") from exc
    return traverse_util, freeze, unfreeze, jax


def extract_internal_adapter_params(params: Any) -> dict[str, Any]:
    """Extract patched transformer leaves so the optimizer cannot touch the base."""

    traverse_util, _, unfreeze, _ = _imports()
    flat = traverse_util.flatten_dict(unfreeze(params))
    selected = {
        path_string(path): value
        for path, value in flat.items()
        if any(
            str(part).startswith("adapter_")
            and (str(part).endswith("_down") or str(part).endswith("_up"))
            for part in path
        )
    }
    if not selected:
        raise ValueError("Octo parameter tree contains no internal research adapters")
    return selected


def apply_internal_adapter_params(base_params: Any, adapters: Mapping[str, Any]) -> Any:
    """Functionally replace only known adapter leaves in an immutable base tree."""

    traverse_util, freeze, unfreeze, _ = _imports()
    flat = traverse_util.flatten_dict(unfreeze(base_params))
    expected = {
        path_string(path)
        for path in flat
        if any(
            str(part).startswith("adapter_")
            and (str(part).endswith("_down") or str(part).endswith("_up"))
            for part in path
        )
    }
    if set(adapters) != expected:
        raise ValueError(
            f"internal adapter keys differ; missing={sorted(expected-set(adapters))}, "
            f"extra={sorted(set(adapters)-expected)}"
        )
    for encoded, value in adapters.items():
        path = tuple(encoded.split("/"))
        if value.shape != flat[path].shape:
            raise ValueError(f"internal adapter shape differs at {encoded}")
        flat[path] = value
    return freeze(traverse_util.unflatten_dict(flat))


def extract_decoded_transformer_params(params: Any) -> dict[str, Any]:
    traverse_util, _, unfreeze, _ = _imports()
    flat = traverse_util.flatten_dict(unfreeze(params))
    decoded = {
        path_string(path): value
        for path, value in flat.items()
        if any(str(part).startswith("decoded_") and str(part).endswith("_kernel") for part in path)
    }
    if not decoded:
        raise ValueError("Octo parameter tree contains no decoded-update injection paths")
    return decoded


def initial_adapter_state(bundle: Any) -> dict[str, Any]:
    """Low-rank-only state used for model-zoo and sequential LoRA training."""

    return {
        "transformer": extract_internal_adapter_params(bundle.research_model.params),
        "diffusion": bundle.diffusion_factors,
    }


def materialize_policy_params(bundle: Any, adapter_state: Mapping[str, Any]) -> Any:
    low_rank_keys = {"transformer", "diffusion"}
    generated_keys = low_rank_keys | {"decoded_transformer", "decoded_diffusion"}
    if set(adapter_state) not in (low_rank_keys, generated_keys):
        raise ValueError(
            "adapter_state must be low-rank-only or include both decoded update groups"
        )
    params = apply_internal_adapter_params(
        bundle.research_model.params, adapter_state["transformer"]
    )
    params = apply_external_adapters(
        params, adapter_state["diffusion"], alpha=bundle.adapter_alpha
    )
    if set(adapter_state) == low_rank_keys:
        return params
    # Internal decoded parameters are absolute zero-based overrides; diffusion
    # decoded values are additive updates to official Dense kernels.
    traverse_util, freeze, unfreeze, _ = _imports()
    flat = traverse_util.flatten_dict(unfreeze(params))
    expected_decoded = {
        path_string(path)
        for path in flat
        if any(str(part).startswith("decoded_") and str(part).endswith("_kernel") for part in path)
    }
    if set(adapter_state["decoded_transformer"]) != expected_decoded:
        raise ValueError("decoded transformer update keys differ from the patched model")
    for encoded, value in adapter_state["decoded_transformer"].items():
        path = tuple(encoded.split("/"))
        if value.shape != flat[path].shape:
            raise ValueError(f"decoded transformer shape differs at {encoded}")
        flat[path] = value
    params = freeze(traverse_util.unflatten_dict(flat))
    return apply_dense_kernel_updates(params, adapter_state["decoded_diffusion"])


def octo_diffusion_loss(
    bundle: Any,
    adapter_state: Mapping[str, Any],
    batch: Mapping[str, Any],
    rng: Any,
    *,
    train: bool,
):
    """Official Octo loss with gradients restricted to adapter state."""

    params = materialize_policy_params(bundle, adapter_state)
    bound = bundle.research_model.module.bind(
        {"params": params}, rngs={"dropout": rng}
    )
    transformer_outputs = bound.octo_transformer(
        batch["observation"],
        batch["task"],
        batch["observation"]["timestep_pad_mask"],
        train=train,
    )
    return bound.heads["action"].loss(
        transformer_outputs,
        batch["action"],
        batch["observation"]["timestep_pad_mask"],
        batch["action_pad_mask"],
        train=train,
    )


def adapter_value_and_grad(bundle: Any, adapter_state: Mapping[str, Any], batch, rng):
    """Compute official task loss and adapter-only gradients for one batch."""

    _, _, _, jax = _imports()
    return jax.value_and_grad(
        lambda state: octo_diffusion_loss(
            bundle, state, batch, rng, train=True
        ),
        has_aux=True,
    )(adapter_state)
