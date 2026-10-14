"""Adapter-only official Octo task loss and optimization primitives."""

from __future__ import annotations

from typing import Any, Mapping

from ..adapters.flax import apply_dense_kernel_updates, apply_external_adapters, path_string


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


def materialize_policy_params(
    bundle: Any, adapter_state: Mapping[str, Any], *, base_params: Any | None = None
) -> Any:
    """Functional policy parameters: frozen base plus one adapter state.

    ``base_params`` lets a compiled function receive the frozen base as an
    argument. Closing over ``bundle.research_model.params`` inside ``jax.jit``
    would instead bake ~0.5 GB of weights into every compiled program as
    constants, which exhausts host memory on small machines.
    """

    from .bridge import activate_research_bundle

    # Every graph that applies an adapter state is traced through here, so this
    # is where the bundle makes sure Octo instantiates *its* patched transformer.
    activate_research_bundle(bundle)
    low_rank_keys = {"transformer", "diffusion"}
    generated_keys = low_rank_keys | {"decoded_transformer", "decoded_diffusion"}
    if set(adapter_state) not in (low_rank_keys, generated_keys):
        raise ValueError(
            "adapter_state must be low-rank-only or include both decoded update groups"
        )
    base = bundle.research_model.params if base_params is None else base_params
    params = apply_internal_adapter_params(base, adapter_state["transformer"])
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
    base_params: Any | None = None,
):
    """Official Octo loss with gradients restricted to adapter state."""

    params = materialize_policy_params(bundle, adapter_state, base_params=base_params)
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


def _leaf_branch(encoded_path: str) -> tuple[str, str] | None:
    """Map an adapter leaf path to ``(branch, factor)``.

    Branches are ``<component>/block_<layer>`` for transformer residuals and
    ``action/head/<kernel path>`` for diffusion-head kernels.
    """

    from .bridge import ADAPTER_LEAF

    for part in encoded_path.split("/"):
        match = ADAPTER_LEAF.fullmatch(part)
        if match:
            return f"{match.group(1)}/block_{int(match.group(2)):02d}", match.group(3)
    return None


def branch_norms(tree: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """L2 norm of every adapter branch's ``down`` and ``up`` leaves.

    ``tree`` has the low-rank adapter-state structure (state, gradient, or
    update): ``{"transformer": {path: array}, "diffusion": {path: {down, up}}}``.
    """

    import numpy as np

    norms: dict[str, dict[str, float]] = {}
    for encoded, value in tree["transformer"].items():
        located = _leaf_branch(encoded)
        if located is None:
            raise ValueError(f"not an adapter leaf: {encoded}")
        branch, factor = located
        norms.setdefault(branch, {})[factor] = float(np.linalg.norm(np.asarray(value)))
    for encoded, pair in tree["diffusion"].items():
        norms[f"action/head/{encoded}"] = {
            "down": float(np.linalg.norm(np.asarray(pair["down"]))),
            "up": float(np.linalg.norm(np.asarray(pair["up"]))),
        }
    return norms


def component_norms(tree: Mapping[str, Any]) -> dict[str, float]:
    """Root-sum-square norm per component, splitting action into its two sites."""

    import numpy as np

    totals = {"vision": 0.0, "language": 0.0, "action_readout": 0.0, "action_head": 0.0}
    for branch, factors in branch_norms(tree).items():
        component = branch.split("/", 1)[0]
        key = (
            "action_head"
            if branch.startswith("action/head/")
            else ("action_readout" if component == "action" else component)
        )
        totals[key] += sum(value**2 for value in factors.values())
    return {name: float(np.sqrt(value)) for name, value in totals.items()}


def assert_adapter_gradient_reachability(
    bundle: Any,
    batch: Mapping[str, Any],
    rng: Any,
    *,
    probe_step: float = 1e-2,
    minimum_norm: float = 1e-12,
    extra_draws: int = 4,
) -> dict[str, Any]:
    """Gate: the task loss must be able to train every adapter branch.

    With zero-initialized ``up`` factors the correct expectation is asymmetric:

    * every ``up`` factor must receive a non-zero gradient immediately;
    * every ``down`` gradient is exactly zero at initialization (it is
      multiplied by the zero ``up`` factor) and must become non-zero once the
      ``up`` factors have moved.

    A branch that fails either condition can never leave its initial value.

    The diffusion loss is random: Octo draws timesteps and noise from ``rng``,
    and one draw can give a reachable branch an exactly zero gradient. The
    head's time embedding gets none when every drawn timestep is 0, which some
    keys produce on a small batch (seen with JAX 0.7's random streams). A
    branch that looks dead is therefore checked again with up to
    ``extra_draws`` keys folded from ``rng``; it fails the gate only if no
    draw reaches it. ``redrawn`` in the report names any branch that needed
    another draw.
    """

    _, _, _, jax = _imports()
    keys = [rng, *(jax.random.fold_in(rng, draw) for draw in range(1, extra_draws + 1))]

    def gradients_of(adapter_state, draw):
        (_, _), gradients = adapter_value_and_grad(bundle, adapter_state, batch, keys[draw])
        gradients = jax.device_get(gradients)
        norms = branch_norms(gradients)
        non_finite = [name for name, value in norms.items() if not all(map(_finite, value.values()))]
        if non_finite:
            raise AssertionError(f"non-finite adapter gradients: {non_finite}")
        return norms, gradients

    def reach(adapter_state, factor):
        """Gradient norms per branch; branches at zero are redrawn."""

        norms, gradients = gradients_of(adapter_state, 0)
        trees = {0: gradients}
        source = dict.fromkeys(norms, 0)
        for draw in range(1, len(keys)):
            dead = [name for name, value in norms.items() if value[factor] <= minimum_norm]
            if not dead:
                break
            other, trees[draw] = gradients_of(adapter_state, draw)
            for name in dead:
                if other[name][factor] > minimum_norm:
                    norms[name], source[name] = other[name], draw
        dead = sorted(name for name, value in norms.items() if value[factor] <= minimum_norm)
        return norms, source, trees, dead

    state = initial_adapter_state(bundle)
    initial, initial_source, trees, dead_up = reach(state, "up")
    if dead_up:
        raise AssertionError(
            "the task loss cannot reach these zero-initialized adapter branches "
            f"(zero up-factor gradient in {len(keys)} draws): {dead_up}"
        )

    # Move only the up factors along their gradient (from the draw that
    # reached each branch), then re-check the down factors.
    def up_gradient(branch, part, path):
        return trees[initial_source[branch]][part][path]

    probed = {
        "transformer": {
            path: (
                value - probe_step * up_gradient(_leaf_branch(path)[0], "transformer", path)
                if path.endswith("_up/kernel")
                else value
            )
            for path, value in state["transformer"].items()
        },
        "diffusion": {
            path: {
                "down": pair["down"],
                "up": pair["up"]
                - probe_step * up_gradient(f"action/head/{path}", "diffusion", path)["up"],
            }
            for path, pair in state["diffusion"].items()
        },
    }
    after, after_source, _, dead_down = reach(probed, "down")
    if dead_down:
        raise AssertionError(
            "down factors stay unreachable after the up factors moved "
            f"(zero gradient in {len(keys)} draws): {dead_down}"
        )
    return {
        "branches": len(initial),
        "initial_up_gradient_norm": {name: value["up"] for name, value in initial.items()},
        "initial_down_gradient_norm": {name: value["down"] for name, value in initial.items()},
        "post_update_down_gradient_norm": {name: value["down"] for name, value in after.items()},
        "minimum_initial_up_gradient_norm": min(value["up"] for value in initial.values()),
        "maximum_initial_down_gradient_norm": max(value["down"] for value in initial.values()),
        "minimum_post_update_down_gradient_norm": min(value["down"] for value in after.values()),
        "redrawn": {
            "up": {name: draw for name, draw in sorted(initial_source.items()) if draw},
            "down": {name: draw for name, draw in sorted(after_source.items()) if draw},
        },
    }


def _finite(value: float) -> bool:
    import math

    return math.isfinite(value)
