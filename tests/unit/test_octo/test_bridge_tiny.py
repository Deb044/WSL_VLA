"""The patched Octo graph, tested on a toy model built from Octo's own classes."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("octo")

import flax.traverse_util  # noqa: E402
from flax.core import freeze, unfreeze  # noqa: E402
import jax  # noqa: E402
from octo.model.octo_model import OctoModel  # noqa: E402

from tests import octo_tiny as tiny  # noqa: E402
from wsl_vla.adapters.latent import (  # noqa: E402
    effective_updates_from_state,
    effective_updates_to_state,
    factors_to_state,
)
from wsl_vla.adapters.packing import (  # noqa: E402
    PackedAdapterMetadata,
    load_packed_adapter,
    pack_low_rank_adapter,
    save_packed_adapter,
    unpack_effective_updates,
)
from wsl_vla.adapters.rank import project_updates_to_rank, update_rank  # noqa: E402
from wsl_vla.contracts import (  # noqa: E402
    COMPONENT_NAMES,
    FROZEN_ROUTE,
    PLACEMENT_PRE_BLOCK,
    SITE_BLOCK_INPUT,
    SITE_HEAD_KERNEL,
)
from wsl_vla.octo.bridge import (  # noqa: E402
    ADAPTER_LEAF,
    FROZEN_COMPONENT_ID,
    activate_research_bundle,
    assert_base_parameters_unchanged,
    assert_zero_adapter_equivalence,
    base_parameter_sha256,
    build_adapter_spec_and_factors,
    build_research_bundle,
    build_token_routing,
    install_octo_modality_patch,
    research_model_config,
    restore_official_octo,
    trace_adapter_routing,
    unnormalization_statistics,
)
from wsl_vla.octo.training import (  # noqa: E402
    adapter_value_and_grad,
    assert_adapter_gradient_reachability,
    branch_norms,
    component_norms,
    initial_adapter_state,
    materialize_policy_params,
)

pytestmark = pytest.mark.research

LAYERS = 2
VISION, LANGUAGE, ACTION = (COMPONENT_NAMES.index(name) for name in ("vision", "language", "action"))


def expected_component_ids(proprio_tokens: int = 0) -> np.ndarray:
    """Token routing of the toy model: language prefix, then each timestep."""

    step = (
        [VISION] * (tiny.PRIMARY_TOKENS + tiny.WRIST_TOKENS)
        + [FROZEN_COMPONENT_ID] * proprio_tokens
        + [LANGUAGE] * tiny.LANGUAGE_TOKENS
        + [ACTION]
    )
    return np.asarray([LANGUAGE] * tiny.LANGUAGE_TOKENS + step * tiny.WINDOW)


def trained_like_state(bundle, seed: int = 0, scale: float = 0.05) -> dict:
    """An adapter state whose every factor is non-zero, as after training."""

    rng = np.random.default_rng(seed)
    state = jax.device_get(initial_adapter_state(bundle))

    def noisy(value):
        return rng.normal(scale=scale, size=np.shape(value)).astype(np.float32)

    return {
        "transformer": {
            path: noisy(value) if path.endswith("_up/kernel") else np.asarray(value)
            for path, value in state["transformer"].items()
        },
        "diffusion": {
            path: {"down": np.asarray(pair["down"]), "up": noisy(pair["up"])}
            for path, pair in state["diffusion"].items()
        },
    }


def with_only(state: dict, component: str, layer: int, seed: int = 0) -> dict:
    """``state`` with one transformer branch's up factor made non-zero."""

    rng = np.random.default_rng(seed)
    transformer = dict(state["transformer"])
    (path,) = [key for key in transformer if key.endswith(f"adapter_{component}_{layer}_up/kernel")]
    transformer[path] = rng.normal(scale=0.5, size=np.shape(transformer[path])).astype(np.float32)
    return {"transformer": transformer, "diffusion": state["diffusion"]}


def policy_outputs(bundle, state, *, seed: int = 3):
    """Transformer readout tokens and sampled actions for the example batch."""

    model = bundle.research_model.replace(params=materialize_policy_params(bundle, state))
    observations = bundle.pretrained_model.example_batch["observation"]
    tasks = bundle.pretrained_model.example_batch["task"]
    outputs = model.run_transformer(
        observations, tasks, observations["timestep_pad_mask"], train=False
    )
    actions = model.sample_actions(
        observations,
        tasks,
        unnormalization_statistics=unnormalization_statistics(bundle.action_convention),
        rng=jax.random.PRNGKey(seed),
    )
    return np.asarray(outputs["readout_action"].tokens), np.asarray(actions)


# --------------------------------------------------------------------- routing
def test_routing_table_is_derived_from_the_checkpoint_configuration(tiny_bundle):
    assert dict(tiny_bundle.token_routing) == {
        "obs_primary": "vision",
        "obs_wrist": "vision",
        "task_language": "language",
        "obs_task_language": "language",
        "readout_action": "action",
    }


def test_live_graph_applies_the_routing_table(tiny_bundle):
    audit = trace_adapter_routing(tiny_bundle)

    np.testing.assert_array_equal(audit["component_ids"], expected_component_ids())
    assert audit["token_counts"] == {
        "vision": tiny.WINDOW * (tiny.PRIMARY_TOKENS + tiny.WRIST_TOKENS),
        "language": tiny.LANGUAGE_TOKENS * (1 + tiny.WINDOW),
        "action": tiny.WINDOW,
        FROZEN_ROUTE: 0,
    }
    assert sorted(audit["residuals"]) == list(range(LAYERS))
    # Zero-initialized adapters add nothing anywhere.
    assert all(not np.any(value) for value in audit["residuals"].values())


def test_unknown_tokenizer_type_is_rejected(tiny_t5):
    config = tiny.tiny_model_config(tiny_t5)
    config["observation_tokenizers"]["depth"] = {
        "module": "somewhere",
        "name": "PointCloudTokenizer",
        "args": (),
        "kwargs": {},
    }
    with pytest.raises(ValueError, match="unrecognized Octo observation tokenizer 'depth'"):
        build_token_routing(config)


def test_routing_requires_every_modality(tiny_t5):
    config = tiny.tiny_model_config(tiny_t5)
    config["task_tokenizers"] = {}
    with pytest.raises(ValueError, match=r"routes no token group to: \['language'\]"):
        build_token_routing(config)


def test_readout_without_an_action_head_is_frozen(tiny_t5):
    config = tiny.tiny_model_config(tiny_t5)
    config["readouts"]["value"] = 1
    routing = build_token_routing(config)
    assert routing["readout_action"] == "action"
    assert routing["readout_value"] == FROZEN_ROUTE


def test_patch_rejects_unknown_destinations():
    with pytest.raises(ValueError, match="unknown destinations"):
        install_octo_modality_patch(rank=2, alpha=4.0, routing={"obs_primary": "proprio"})
    with pytest.raises(ValueError, match="rank and alpha must be positive"):
        install_octo_modality_patch(rank=0, alpha=4.0, routing={"obs_primary": "vision"})


def test_unrouted_token_group_fails_when_the_graph_is_traced(tiny_t5):
    pretrained = tiny.build_tiny_pretrained(tiny_t5)
    routing = dict(build_token_routing(pretrained.config["model"]))
    del routing["obs_wrist"]
    install_octo_modality_patch(rank=2, alpha=4.0, routing=routing)
    try:
        with pytest.raises(ValueError, match="unrecognized Octo token group 'obs_wrist'"):
            OctoModel.from_config(
                research_model_config(pretrained.config, rank=2, alpha=4.0, routing=routing),
                pretrained.example_batch,
                pretrained.text_processor,
            )
    finally:
        restore_official_octo()


def test_a_module_refuses_a_transformer_patch_it_was_not_built_for(tiny_t5, tiny_bundle):
    """Octo reads its block transformer from a module global; a mismatch must be loud."""

    pretrained = tiny.build_tiny_pretrained(tiny_t5)
    routing = build_token_routing(pretrained.config["model"])
    install_octo_modality_patch(rank=2, alpha=4.0, routing=routing)
    try:
        # An official model traced while a patch is installed.
        with pytest.raises(RuntimeError, match="not built for the research transformer"):
            OctoModel.from_config(
                pretrained.config, pretrained.example_batch, pretrained.text_processor
            )
        # A research module for another scale (same shapes, so nothing else would notice).
        with pytest.raises(RuntimeError, match="not built for the research transformer"):
            OctoModel.from_config(
                research_model_config(pretrained.config, rank=2, alpha=8.0, routing=routing),
                pretrained.example_batch,
                pretrained.text_processor,
            )
    finally:
        restore_official_octo()
    # A research module traced by official Octo is rejected by Octo itself.
    with pytest.raises(TypeError):
        OctoModel.from_config(
            research_model_config(pretrained.config, rank=2, alpha=4.0, routing=routing),
            pretrained.example_batch,
            pretrained.text_processor,
        )


def test_proprioception_tokens_are_frozen_in_the_graph(tiny_t5):
    """A LowdimObsTokenizer group gets no modality residual, whatever the adapters hold."""

    pretrained = tiny.build_tiny_pretrained(tiny_t5, proprio=True)
    routing = build_token_routing(pretrained.config["model"])
    assert routing["obs_proprio"] == FROZEN_ROUTE
    install_octo_modality_patch(rank=2, alpha=4.0, routing=routing)
    try:
        research = OctoModel.from_config(
            research_model_config(pretrained.config, rank=2, alpha=4.0, routing=routing),
            pretrained.example_batch,
            pretrained.text_processor,
            rng=jax.random.PRNGKey(1),
        )
        rng = np.random.default_rng(0)
        flat = flax.traverse_util.flatten_dict(unfreeze(research.params))
        for path, value in flat.items():
            if any(ADAPTER_LEAF.fullmatch(str(part)) for part in path):
                flat[path] = rng.normal(size=value.shape).astype(np.float32)
        observations = research.example_batch["observation"]
        _, captured = research.module.apply(
            {"params": freeze(flax.traverse_util.unflatten_dict(flat))},
            observations,
            research.example_batch["task"],
            observations["timestep_pad_mask"],
            train=False,
            method="octo_transformer",
            mutable=["intermediates"],
        )
    finally:
        restore_official_octo()
    found = {
        str(path[-1]): np.asarray(values[0])
        for path, values in flax.traverse_util.flatten_dict(
            unfreeze(captured["intermediates"])
        ).items()
    }
    component_ids = found["adapter_component_ids"]
    np.testing.assert_array_equal(
        component_ids, expected_component_ids(proprio_tokens=tiny.PROPRIO_DIM)
    )
    frozen = component_ids == FROZEN_COMPONENT_ID
    assert frozen.sum() == tiny.WINDOW * tiny.PROPRIO_DIM
    for layer in range(LAYERS):
        residual = found[f"adapter_residual_{layer}"]
        assert not np.any(residual[:, frozen])
        assert np.all(np.abs(residual[:, ~frozen]).max(axis=-1) > 0)


def test_checkpoint_that_consumes_proprioception_is_rejected(tiny_t5):
    pretrained = tiny.build_tiny_pretrained(tiny_t5, proprio=True)
    with pytest.raises(ValueError, match="no shared preprocessing for a proprioception tokenizer"):
        build_research_bundle(
            pretrained,
            base_revision="tiny-octo-test",
            rank=2,
            alpha=4.0,
            seed=0,
            preprocessing=tiny.TINY_PREPROCESSING,
        )


# ----------------------------------------------------------------- equivalence
def test_zero_adapters_reproduce_official_outputs_and_sampled_actions(tiny_bundle):
    report = assert_zero_adapter_equivalence(tiny_bundle)

    # The gate's own tolerance (atol 1e-5 plus rtol 1e-5 of the magnitude):
    # JAX 0.4.20 reproduces both exactly; JAX 0.7 fuses the patched graph
    # differently and the diffusion sampler compounds the float32 residue to
    # about 1e-5 in the toy model's actions.
    for name in ("transformer_outputs", "sampled_actions"):
        assert report[name]["max_reference_abs"] > 0.1
        assert report[name]["max_abs_diff"] <= 1e-5 * (1 + report[name]["max_reference_abs"])
    assert report["sampled_actions"]["random_key_seed"] == tiny_bundle.equivalence_seed


def test_equivalence_gate_detects_a_non_zero_adapter(tiny_bundle):
    """The gate compares real outputs: an adapter that leaks into the base path fails it."""

    state = with_only(jax.device_get(initial_adapter_state(tiny_bundle)), "action", LAYERS - 1)
    leaked = tiny_bundle.research_model.replace(
        params=materialize_policy_params(tiny_bundle, state)
    )
    from dataclasses import replace

    with pytest.raises(AssertionError, match="zero-adapter equivalence failed"):
        assert_zero_adapter_equivalence(replace(tiny_bundle, research_model=leaked))


def test_research_model_inherits_every_pretrained_parameter(tiny_bundle):
    assert base_parameter_sha256(tiny_bundle.research_model.params) == tiny_bundle.base_sha256
    assert_base_parameters_unchanged(tiny_bundle)


def test_patched_parameter_names_are_a_superset_of_the_official_tree(tiny_bundle):
    official = set(flax.traverse_util.flatten_dict(unfreeze(tiny_bundle.pretrained_model.params)))
    patched = set(flax.traverse_util.flatten_dict(unfreeze(tiny_bundle.research_model.params)))
    extra = patched - official
    assert official <= patched
    assert extra
    assert all(
        any(str(part).startswith(("adapter_", "decoded_")) for part in path) for path in extra
    )


# ------------------------------------------------------------------- placement
@pytest.mark.parametrize("component", COMPONENT_NAMES)
def test_every_layer_adapter_reaches_the_action_readout(tiny_bundle, component):
    """Regression: post-block placement cut final-layer vision/language adapters off.

    With pre-block residuals a change to any modality's adapter in any layer,
    including the last, must move the action readout and the sampled action.
    """

    base = jax.device_get(initial_adapter_state(tiny_bundle))
    reference_readout, reference_actions = policy_outputs(tiny_bundle, base)
    for layer in range(LAYERS):
        readout, actions = policy_outputs(tiny_bundle, with_only(base, component, layer))
        assert np.max(np.abs(readout - reference_readout)) > 1e-5, (component, layer)
        assert np.max(np.abs(actions - reference_actions)) > 1e-6, (component, layer)


@pytest.mark.parametrize("component", COMPONENT_NAMES)
def test_a_modality_residual_touches_only_its_own_tokens(tiny_bundle, component):
    base = jax.device_get(initial_adapter_state(tiny_bundle))
    index = COMPONENT_NAMES.index(component)
    for layer in range(LAYERS):
        audit = trace_adapter_routing(tiny_bundle, with_only(base, component, layer))
        own = audit["component_ids"] == index
        residual = audit["residuals"][layer]
        assert np.all(np.abs(residual[:, own]).max(axis=-1) > 0)
        assert not np.any(residual[:, ~own])
        for other_layer in set(range(LAYERS)) - {layer}:
            assert not np.any(audit["residuals"][other_layer])


def test_modality_residuals_read_the_same_block_input(tiny_bundle):
    """The three branches of one block are independent of evaluation order."""

    base = jax.device_get(initial_adapter_state(tiny_bundle))
    for layer in range(LAYERS):
        combined = base
        separate = np.zeros(())
        for seed, component in enumerate(COMPONENT_NAMES):
            single = with_only(base, component, layer, seed=seed)
            separate = separate + trace_adapter_routing(tiny_bundle, single)["residuals"][layer]
            combined = with_only(combined, component, layer, seed=seed)
        together = trace_adapter_routing(tiny_bundle, combined)["residuals"][layer]
        np.testing.assert_allclose(together, separate, rtol=1e-6, atol=1e-7)


# -------------------------------------------------------------------- gradients
def example_training_batch(bundle):
    stream = tiny.tiny_stream(bundle)
    return stream.collate(
        [stream.example(0, 1), stream.example(1, 3), stream.example(2, 0)],
        bundle.pretrained_model.text_processor,
    )


def test_every_adapter_branch_is_trainable(tiny_bundle):
    report = assert_adapter_gradient_reachability(
        tiny_bundle, example_training_batch(tiny_bundle), jax.random.PRNGKey(0)
    )

    transformer_branches = {
        f"{component}/block_{layer:02d}" for component in COMPONENT_NAMES for layer in range(LAYERS)
    }
    assert transformer_branches <= set(report["initial_up_gradient_norm"])
    assert report["branches"] == len(transformer_branches) + len(
        tiny_bundle.diffusion_kernel_paths
    )
    # Zero-init semantics: up factors learn immediately, down factors are
    # exactly zero until the up factors have moved.
    assert report["minimum_initial_up_gradient_norm"] > 1e-8
    assert report["maximum_initial_down_gradient_norm"] == 0.0
    assert report["minimum_post_update_down_gradient_norm"] > 1e-10
    final = f"block_{LAYERS - 1:02d}"
    assert report["initial_up_gradient_norm"][f"vision/{final}"] > 1e-8
    assert report["initial_up_gradient_norm"][f"language/{final}"] > 1e-8


def test_gradient_gate_redraws_a_branch_that_one_random_draw_leaves_at_zero(
    tiny_bundle, monkeypatch
):
    """Octo samples diffusion timesteps from the key; if every drawn timestep is
    0, the head's time embedding gets no gradient although it is reachable."""

    import wsl_vla.octo.training as training

    first = jax.random.PRNGKey(0)
    target = next(path for path in initial_adapter_state(tiny_bundle)["diffusion"]
                  if "time_preprocess" in path)
    real = training.adapter_value_and_grad

    def zero_target(gradients):
        diffusion = dict(gradients["diffusion"])
        diffusion[target] = {key: value * 0 for key, value in diffusion[target].items()}
        return {**gradients, "diffusion": diffusion}

    def unlucky_first_draw(bundle, state, batch, rng):
        (loss, aux), gradients = real(bundle, state, batch, rng)
        if np.array_equal(np.asarray(rng), np.asarray(first)):
            gradients = zero_target(gradients)
        return (loss, aux), gradients

    batch = example_training_batch(tiny_bundle)
    monkeypatch.setattr(training, "adapter_value_and_grad", unlucky_first_draw)
    report = training.assert_adapter_gradient_reachability(tiny_bundle, batch, first)
    assert report["redrawn"] == {"up": {f"action/head/{target}": 1}, "down": {f"action/head/{target}": 1}}
    assert report["minimum_initial_up_gradient_norm"] > 1e-8
    assert report["minimum_post_update_down_gradient_norm"] > 1e-10

    def never_reached(bundle, state, batch, rng):
        (loss, aux), gradients = real(bundle, state, batch, rng)
        return (loss, aux), zero_target(gradients)

    monkeypatch.setattr(training, "adapter_value_and_grad", never_reached)
    with pytest.raises(AssertionError, match=r"zero up-factor gradient in 5 draws\): \['action/head/"):
        training.assert_adapter_gradient_reachability(tiny_bundle, batch, first)


def test_gradient_gate_reports_unreachable_branches(tiny_bundle):
    """A batch whose every timestep is padding gives the loss no signal at all."""

    batch = example_training_batch(tiny_bundle)
    batch["action_pad_mask"] = np.zeros_like(batch["action_pad_mask"])
    with pytest.raises(AssertionError, match="cannot reach these zero-initialized adapter"):
        assert_adapter_gradient_reachability(tiny_bundle, batch, jax.random.PRNGKey(0))


def test_gradients_exist_only_for_adapter_state(tiny_bundle):
    state = initial_adapter_state(tiny_bundle)
    (loss, _), gradients = adapter_value_and_grad(
        tiny_bundle, state, example_training_batch(tiny_bundle), jax.random.PRNGKey(1)
    )

    assert np.isfinite(float(loss))
    assert set(gradients) == {"transformer", "diffusion"}
    assert set(gradients["transformer"]) == set(state["transformer"])
    norms = component_norms(jax.device_get(gradients))
    assert set(norms) == {"vision", "language", "action_readout", "action_head"}
    assert all(value > 0 for value in norms.values())
    assert len(branch_norms(jax.device_get(gradients))) == 3 * LAYERS + len(
        tiny_bundle.diffusion_kernel_paths
    )
    assert_base_parameters_unchanged(tiny_bundle)


# ---------------------------------------------------------------- spec and I/O
def test_adapter_spec_records_placement_routing_and_sites(tiny_bundle):
    spec, factors = build_adapter_spec_and_factors(tiny_bundle, token_width=tiny.TOKEN_WIDTH)

    assert spec.placement == PLACEMENT_PRE_BLOCK
    assert dict(spec.token_routing) == dict(tiny_bundle.token_routing)
    assert spec.base_sha256 == tiny_bundle.base_sha256
    assert spec.architecture_sha256 == tiny_bundle.architecture_sha256
    assert spec.action_convention == tiny_bundle.action_convention
    block_entries = [entry for entry in spec.entries if entry.site == SITE_BLOCK_INPUT]
    head_entries = [entry for entry in spec.entries if entry.site == SITE_HEAD_KERNEL]
    assert len(block_entries) == 3 * LAYERS
    assert {(entry.component.value, entry.layer) for entry in block_entries} == {
        (component, layer) for component in COMPONENT_NAMES for layer in range(LAYERS)
    }
    assert len(head_entries) == len(tiny_bundle.diffusion_kernel_paths)
    assert all(entry.component.value == "action" for entry in head_entries)
    assert set(factors) == {entry.parameter_path for entry in spec.entries}
    # A different rank is a different architecture of adapters, hence another spec.
    assert spec.sha256() != build_adapter_spec_and_factors(tiny_bundle, token_width=4)[0].sha256()


def test_saved_adapter_reloads_to_the_same_policy(tiny_bundle, tmp_path):
    state = trained_like_state(tiny_bundle)
    spec, factors = build_adapter_spec_and_factors(
        tiny_bundle, token_width=tiny.TOKEN_WIDTH, adapter_state=state
    )
    packed = pack_low_rank_adapter(
        spec,
        factors,
        metadata=PackedAdapterMetadata(
            task_id="libero_spatial_0",
            suite="libero_spatial",
            seed=17,
            optimizer_updates=4,
            snapshot_index=0,
            sample_exposure=16,
            source_dtype="float32",
        ),
        sequence_window=tiny.SEQUENCE_WINDOW,
    )
    save_packed_adapter(packed, tmp_path / "adapter.npz")
    reloaded = load_packed_adapter(tmp_path / "adapter.npz")
    restored = factors_to_state(tiny_bundle, reloaded.spec, reloaded.factors)

    assert reloaded.spec == spec
    for path, value in state["transformer"].items():
        np.testing.assert_array_equal(np.asarray(restored["transformer"][path]), value)
    expected_readout, expected_actions = policy_outputs(tiny_bundle, state)
    readout, actions = policy_outputs(tiny_bundle, restored)
    np.testing.assert_array_equal(readout, expected_readout)
    np.testing.assert_array_equal(actions, expected_actions)
    # The trained adapter is a real change, not the frozen base.
    base_readout, _ = policy_outputs(tiny_bundle, initial_adapter_state(tiny_bundle))
    assert np.max(np.abs(expected_readout - base_readout)) > 1e-4
    assert_base_parameters_unchanged(tiny_bundle)


def test_dense_and_rank_matched_deployment_agree_with_the_low_rank_adapter(tiny_bundle):
    state = trained_like_state(tiny_bundle, seed=4)
    spec, factors = build_adapter_spec_and_factors(
        tiny_bundle, token_width=tiny.TOKEN_WIDTH, adapter_state=state
    )
    packed = pack_low_rank_adapter(
        spec,
        factors,
        metadata=PackedAdapterMetadata(
            task_id="libero_spatial_0",
            suite="libero_spatial",
            seed=17,
            optimizer_updates=4,
            snapshot_index=0,
            sample_exposure=16,
            source_dtype="float32",
        ),
        sequence_window=tiny.SEQUENCE_WINDOW,
    )
    updates = unpack_effective_updates(packed)
    expected_readout, expected_actions = policy_outputs(tiny_bundle, state)

    # Dense path: the unpacked tokens applied through the decoded-update kernels.
    dense_state = effective_updates_to_state(tiny_bundle, spec, updates)
    dense_readout, dense_actions = policy_outputs(tiny_bundle, dense_state)
    np.testing.assert_allclose(dense_readout, expected_readout, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(dense_actions, expected_actions, rtol=1e-4, atol=1e-5)
    recovered = effective_updates_from_state(tiny_bundle, spec, dense_state)
    for path, value in updates.items():
        np.testing.assert_allclose(np.asarray(recovered[path]), value, rtol=1e-6, atol=1e-7)

    # Rank-matched path: the truncated SVD of a rank-r update loses nothing and
    # deploys through the same low-rank route as a zoo adapter.
    projection = project_updates_to_rank(spec, updates)
    assert projection.relative_error is not None and projection.relative_error < 1e-5
    for entry in spec.entries:
        assert update_rank(projection.updates[entry.parameter_path]) <= entry.effective_rank_bound
    ranked_state = factors_to_state(tiny_bundle, spec, projection.factors)
    assert set(ranked_state) == {"transformer", "diffusion"}
    ranked_readout, ranked_actions = policy_outputs(tiny_bundle, ranked_state)
    np.testing.assert_allclose(ranked_readout, expected_readout, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(ranked_actions, expected_actions, rtol=1e-4, atol=1e-5)


def test_rank_matching_removes_excess_capacity_before_deployment(tiny_bundle):
    """A dense (full-rank) decoded update is deployed at the zoo's rank, not its own."""

    spec, _ = build_adapter_spec_and_factors(tiny_bundle, token_width=tiny.TOKEN_WIDTH)
    rng = np.random.default_rng(2)
    dense = {
        entry.parameter_path: rng.normal(scale=0.05, size=(entry.input_dim, entry.output_dim))
        for entry in spec.entries
    }
    projection = project_updates_to_rank(spec, dense)

    assert projection.relative_error is not None and projection.relative_error > 0.1
    over_capacity = 0
    for entry in spec.entries:
        # Kernels narrower than the rank are already within the bound.
        over_capacity += update_rank(dense[entry.parameter_path]) > entry.effective_rank_bound
        assert update_rank(projection.updates[entry.parameter_path]) <= entry.effective_rank_bound
    assert over_capacity >= 3 * LAYERS
    state = factors_to_state(tiny_bundle, spec, projection.factors)
    recovered = effective_updates_from_state(tiny_bundle, spec, state)
    for entry in spec.entries:
        np.testing.assert_allclose(
            np.asarray(recovered[entry.parameter_path]),
            projection.updates[entry.parameter_path],
            rtol=1e-4,
            atol=1e-6,
        )


def test_adapter_state_with_foreign_keys_is_rejected(tiny_bundle):
    state = jax.device_get(initial_adapter_state(tiny_bundle))
    with pytest.raises(ValueError, match="low-rank-only or include both decoded"):
        materialize_policy_params(tiny_bundle, {**state, "decoded_transformer": {}})
    broken = {"transformer": dict(state["transformer"]), "diffusion": state["diffusion"]}
    broken["transformer"].pop(next(iter(broken["transformer"])))
    with pytest.raises(ValueError, match="internal adapter keys differ"):
        materialize_policy_params(tiny_bundle, broken)


# ------------------------------------------------------------ several bundles
def test_each_bundle_traces_its_own_adapter_graph(tiny_t5, tiny_bundle):
    """Loading a second bundle must not change what the first one builds."""

    other = tiny.build_tiny_bundle(tiny_t5, rank=3, seed=5)
    assert other.adapter_rank == 3 and tiny_bundle.adapter_rank == 2
    assert other.base_sha256 != tiny_bundle.base_sha256

    def fresh_trace(bundle):
        # A batch size no earlier test used forces a new trace of the graph.
        stream = tiny.tiny_stream(bundle, count=5, seed=9)
        batch = stream.collate(
            [stream.example(index, 2) for index in range(5)],
            bundle.pretrained_model.text_processor,
        )
        state = with_only(jax.device_get(initial_adapter_state(bundle)), "vision", 0)
        model = bundle.research_model.replace(params=materialize_policy_params(bundle, state))
        outputs = model.run_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=False,
        )
        return np.asarray(outputs["readout_action"].tokens)

    # ``other`` was built last, so Octo's module global currently names its
    # rank-3 transformer; tracing the first bundle must still build rank 2.
    assert fresh_trace(tiny_bundle).shape == (5, tiny.WINDOW, 1, tiny.TOKEN_DIM)
    assert fresh_trace(other).shape == (5, tiny.WINDOW, 1, tiny.TOKEN_DIM)
    assert fresh_trace(tiny_bundle).shape == (5, tiny.WINDOW, 1, tiny.TOKEN_DIM)
    assert_zero_adapter_equivalence(tiny_bundle)
    assert_zero_adapter_equivalence(other)
    activate_research_bundle(tiny_bundle)
    assert trace_adapter_routing(tiny_bundle)["token_counts"]["action"] == tiny.WINDOW
