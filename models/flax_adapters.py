"""Differentiable external adapters for Octo's diffusion action head."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _imports():
    try:
        from flax import traverse_util
        from flax.core import freeze, unfreeze
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - research environment only
        raise RuntimeError("Flax/JAX research dependencies are required") from exc
    return traverse_util, freeze, unfreeze, jax, jnp


def path_string(path: Sequence[Any]) -> str:
    return "/".join(str(part) for part in path)


def discover_diffusion_kernel_paths(params: Any) -> tuple[tuple[str, ...], ...]:
    """Find 2-D Dense kernels under the official action diffusion head."""

    traverse_util, _, unfreeze, _, _ = _imports()
    flat = traverse_util.flatten_dict(unfreeze(params))
    paths = []
    for path, value in flat.items():
        lowered = tuple(str(part).lower() for part in path)
        if (
            path[-1] == "kernel"
            and getattr(value, "ndim", None) == 2
            and "diffusion_model" in lowered
        ):
            paths.append(tuple(str(part) for part in path))
    if not paths:
        raise ValueError("no Dense kernels found under Octo's action diffusion head")
    return tuple(sorted(paths))


def initialize_external_adapters(
    params: Any,
    paths: Sequence[Sequence[str]],
    *,
    rank: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Create random-down/zero-up factors, preserving the base output exactly."""

    if rank <= 0:
        raise ValueError("rank must be positive")
    traverse_util, _, unfreeze, jax, jnp = _imports()
    flat = traverse_util.flatten_dict(unfreeze(params))
    keys = jax.random.split(jax.random.PRNGKey(seed), len(paths))
    factors = {}
    for key, requested_path in zip(keys, paths):
        path = tuple(requested_path)
        if path not in flat:
            raise ValueError(f"adapter target does not exist: {path_string(path)}")
        kernel = flat[path]
        if kernel.ndim != 2:
            raise ValueError(f"adapter target is not a matrix: {path_string(path)}")
        factors[path_string(path)] = {
            "down": jax.random.normal(key, (kernel.shape[0], rank), dtype=kernel.dtype) * 0.02,
            "up": jnp.zeros((rank, kernel.shape[1]), dtype=kernel.dtype),
        }
    return factors


def apply_external_adapters(
    base_params: Any,
    factors: Mapping[str, Mapping[str, Any]],
    *,
    alpha: float,
) -> Any:
    """Return a parameter tree with differentiable effective updates applied."""

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    traverse_util, freeze, unfreeze, _, _ = _imports()
    flat = traverse_util.flatten_dict(unfreeze(base_params))
    for encoded_path, pair in factors.items():
        path = tuple(encoded_path.split("/"))
        if path not in flat:
            raise ValueError(f"adapter target does not exist: {encoded_path}")
        down, up = pair["down"], pair["up"]
        if down.ndim != 2 or up.ndim != 2 or down.shape[1] != up.shape[0]:
            raise ValueError(f"invalid low-rank factors for {encoded_path}")
        if (down.shape[0], up.shape[1]) != flat[path].shape:
            raise ValueError(f"factor shape does not match {encoded_path}")
        flat[path] = flat[path] + (alpha / down.shape[1]) * (down @ up)
    return freeze(traverse_util.unflatten_dict(flat))


def apply_dense_kernel_updates(base_params: Any, updates: Mapping[str, Any]) -> Any:
    """Apply decoded effective updates directly to selected Flax kernels."""

    traverse_util, freeze, unfreeze, _, _ = _imports()
    flat = traverse_util.flatten_dict(unfreeze(base_params))
    for encoded_path, update in updates.items():
        path = tuple(encoded_path.split("/"))
        if path not in flat:
            raise ValueError(f"dense update target does not exist: {encoded_path}")
        if update.shape != flat[path].shape:
            raise ValueError(f"dense update shape does not match {encoded_path}")
        flat[path] = flat[path] + update
    return freeze(traverse_util.unflatten_dict(flat))
