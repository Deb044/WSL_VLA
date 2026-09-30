from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from wsl_vla.adapters.packing import (
    load_packed_adapter,
    PackedAdapterMetadata,
    pack_low_rank_adapter,
    save_packed_adapter,
    unpack_effective_updates,
)
from wsl_vla.alignment.mapping import fit_linear_ridge_mapper
from wsl_vla.contracts import AdapterEntry, AdapterSpec, Component
from wsl_vla.experiments.protocol import (
    expand_population_runs,
    load_yaml,
    publication_folds,
    validate_reference_tasks,
    validate_research_config,
)


def make_spec() -> AdapterSpec:
    return AdapterSpec(
        base_model_id="test/model",
        base_revision="revision",
        base_sha256="a" * 64,
        alpha=4.0,
        token_width=3,
        entries=(
            AdapterEntry(Component.VISION, 0, "block0/vision", 4, 5, 2),
            AdapterEntry(Component.ACTION, 1, "block1/action", 3, 4, 2),
        ),
    )


def make_metadata() -> PackedAdapterMetadata:
    return PackedAdapterMetadata(
        task_id="libero_spatial_0",
        suite="libero_spatial",
        seed=17,
        checkpoint_stage=1600,
        checkpoint_fraction=0.8,
        source_dtype="float32",
    )


def test_adapter_spec_roundtrip():
    spec = make_spec()
    assert AdapterSpec.from_dict(spec.to_dict()) == spec


def test_effective_update_packing_is_factor_basis_invariant():
    rng = np.random.default_rng(4)
    spec = make_spec()
    factors = {}
    transformed = {}
    change = np.array([[2.0, 0.25], [0.0, 0.5]])
    inverse = np.linalg.inv(change)
    for entry in spec.entries:
        down = rng.normal(size=(entry.input_dim, entry.rank))
        up = rng.normal(size=(entry.rank, entry.output_dim))
        factors[entry.parameter_path] = (down, up)
        transformed[entry.parameter_path] = (down @ change, inverse @ up)

    packed = pack_low_rank_adapter(spec, factors, metadata=make_metadata())
    repacked = pack_low_rank_adapter(spec, transformed, metadata=make_metadata())
    np.testing.assert_allclose(packed.tokens, repacked.tokens, atol=1e-6)
    reconstructed = unpack_effective_updates(packed)
    for entry in spec.entries:
        down, up = factors[entry.parameter_path]
        expected = (spec.alpha / entry.rank) * down @ up
        np.testing.assert_allclose(reconstructed[entry.parameter_path], expected, atol=1e-6)


def test_packing_rejects_missing_parameter_path():
    with pytest.raises(ValueError, match="factor paths differ"):
        pack_low_rank_adapter(make_spec(), {}, metadata=make_metadata())


def test_packed_adapter_roundtrip_preserves_reloadable_factors(tmp_path):
    spec = make_spec()
    factors = {
        entry.parameter_path: (
            np.ones((entry.input_dim, entry.rank), dtype=np.float32),
            np.ones((entry.rank, entry.output_dim), dtype=np.float32),
        )
        for entry in spec.entries
    }
    path = tmp_path / "adapter.npz"
    original = pack_low_rank_adapter(spec, factors, metadata=make_metadata())
    save_packed_adapter(original, path)
    restored = load_packed_adapter(path)
    np.testing.assert_array_equal(restored.tokens, original.tokens)
    np.testing.assert_array_equal(restored.column_offsets, original.column_offsets)
    assert restored.metadata == original.metadata
    for key in factors:
        np.testing.assert_array_equal(restored.factors[key][0], factors[key][0])
        np.testing.assert_array_equal(restored.factors[key][1], factors[key][1])


def test_linear_mapper_recovers_affine_relation():
    rng = np.random.default_rng(9)
    evidence = rng.normal(size=(30, 4))
    weight = rng.normal(size=(4, 12))
    bias = rng.normal(size=(12,))
    latents = (evidence @ weight + bias).reshape(30, 3, 4)
    mapper = fit_linear_ridge_mapper(evidence, latents, ridge_alpha=1e-10)
    np.testing.assert_allclose(mapper.predict(evidence), latents, atol=1e-7)


def test_locked_protocol_expands_to_360_population_checkpoints():
    config = load_yaml("configs/research/base.yaml")
    tasks = load_yaml("configs/reference_tasks.yaml")
    validate_research_config(config)
    validate_reference_tasks(tasks)
    runs = expand_population_runs(tasks, config)
    assert len(runs) == 360
    assert len(publication_folds()) == 4
    assert all(len(fold.train_suites) == 3 for fold in publication_folds())
