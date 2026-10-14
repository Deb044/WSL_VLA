"""Token layout identities, factor/token round trips, and sequence windows."""

import json

import numpy as np
import pytest

from tests.synthetic import SEQUENCE_WINDOW, synthetic_factors, synthetic_spec
from wsl_vla.adapters.packing import (
    PackedAdapterMetadata,
    build_token_layout,
    factors_to_updates,
    from_windows,
    load_packed_adapter,
    pack_effective_updates,
    pack_low_rank_adapter,
    save_packed_adapter,
    to_windows,
    unpack_effective_updates,
    unpack_tokens,
)
from wsl_vla.contracts import COMPONENT_NAMES, AdapterSpec, LegacyArtifactError


def metadata(**changes) -> PackedAdapterMetadata:
    values = dict(
        task_id="libero_spatial_0", suite="libero_spatial", seed=17, optimizer_updates=8,
        snapshot_index=2, sample_exposure=32, source_dtype="float32",
    )
    values.update(changes)
    return PackedAdapterMetadata(**values)


def test_layout_carries_parameter_row_column_and_sequence_window_identities():
    spec = synthetic_spec()
    layout = build_token_layout(spec, sequence_window=SEQUENCE_WINDOW)
    # 6 transformer entries of 8 rows x 2 column windows, a 5x12 head kernel
    # (2 windows per row), and a 4x1 head kernel (1 window per row).
    assert layout.token_count == 6 * 16 + 10 + 4
    assert set(np.unique(layout.column_window_ids)) == {0, 1}
    assert layout.row_ids.max() == 7
    assert layout.entry_ids.max() == len(spec.entries) - 1
    # Head kernels are numbered after the transformer blocks, not on top of them.
    head = [i for i, entry in enumerate(spec.entries) if entry.site == "action_head_kernel"]
    assert sorted(set(layout.layer_ids[np.isin(layout.entry_ids, head)])) == [2, 3]
    # The 12-wide head kernel leaves no padding (2 x 6); the 1-wide one pads 5 of 6.
    narrow = layout.entry_ids == head[-1]
    assert layout.token_mask[narrow].sum(axis=1).tolist() == [1, 1, 1, 1]
    # Every token sits in exactly one window of its own modality stream.
    for name in COMPONENT_NAMES:
        positions = layout.component_positions(name)
        windows = layout.component_windows(name)
        assert len(windows) == -(-len(positions) // SEQUENCE_WINDOW)
        assert set(layout.sequence_window_ids[positions]) == set(windows.tolist())
    assert layout.window_valid.sum() == layout.token_count
    assert build_token_layout(spec, sequence_window=SEQUENCE_WINDOW).sha256() == layout.sha256()
    assert build_token_layout(spec, sequence_window=4).sha256() != layout.sha256()


def test_factors_tokens_and_dense_updates_round_trip():
    rng = np.random.default_rng(3)
    spec = synthetic_spec()
    factors = synthetic_factors(spec, rng)
    packed = pack_low_rank_adapter(spec, factors, metadata=metadata(), sequence_window=SEQUENCE_WINDOW)
    updates = unpack_effective_updates(packed)
    for entry in spec.entries:
        down, up = factors[entry.parameter_path]
        np.testing.assert_allclose(
            updates[entry.parameter_path], (spec.alpha / entry.rank) * down @ up, rtol=1e-5, atol=1e-6
        )
    np.testing.assert_array_equal(pack_effective_updates(spec, updates), packed.tokens)
    assert not packed.tokens[~packed.mask].any()


def test_packing_is_invariant_to_the_factor_basis():
    rng = np.random.default_rng(4)
    spec = synthetic_spec()
    factors = synthetic_factors(spec, rng)
    change = np.array([[2.0, 0.25], [0.0, 0.5]], dtype=np.float32)
    rotated = {
        path: (down @ change, np.linalg.inv(change) @ up) for path, (down, up) in factors.items()
    }
    first = pack_low_rank_adapter(spec, factors, metadata=metadata(), sequence_window=SEQUENCE_WINDOW)
    second = pack_low_rank_adapter(spec, rotated, metadata=metadata(), sequence_window=SEQUENCE_WINDOW)
    np.testing.assert_allclose(first.tokens, second.tokens, atol=1e-5)


def test_sequence_windows_restore_the_complete_sequence_exactly():
    rng = np.random.default_rng(5)
    spec = synthetic_spec()
    layout = build_token_layout(spec, sequence_window=SEQUENCE_WINDOW)
    tokens = rng.normal(size=(3, layout.token_count, spec.token_width)).astype(np.float32)
    windows = to_windows(layout, tokens)
    assert windows.shape == (3, layout.window_count, SEQUENCE_WINDOW, spec.token_width)
    assert not windows[:, ~layout.window_valid].any()  # padding is zero, never data
    np.testing.assert_array_equal(from_windows(layout, windows), tokens)
    identities = to_windows(layout, layout.entry_ids, feature_dims=0)
    np.testing.assert_array_equal(from_windows(layout, identities, feature_dims=0), layout.entry_ids)
    with pytest.raises(ValueError, match="token axis"):
        to_windows(layout, tokens[:, :-1])


def test_window_round_trip_is_identical_under_jax():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    spec = synthetic_spec()
    layout = build_token_layout(spec, sequence_window=SEQUENCE_WINDOW)
    tokens = jax.random.normal(jax.random.PRNGKey(0), (2, layout.token_count, spec.token_width))
    restored = from_windows(layout, to_windows(layout, tokens, xp=jnp), xp=jnp)
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(tokens))
    updates = unpack_tokens(spec, tokens[0], xp=jnp)
    assert updates[spec.entries[0].parameter_path].shape == (
        spec.entries[0].input_dim, spec.entries[0].output_dim,
    )


def test_packed_adapter_reloads_and_rejects_tampering_and_legacy_files(tmp_path):
    rng = np.random.default_rng(6)
    spec = synthetic_spec()
    factors = synthetic_factors(spec, rng)
    packed = pack_low_rank_adapter(spec, factors, metadata=metadata(), sequence_window=SEQUENCE_WINDOW)
    path = tmp_path / "adapter.npz"
    save_packed_adapter(packed, path)
    restored = load_packed_adapter(path)
    np.testing.assert_array_equal(restored.tokens, packed.tokens)
    assert restored.metadata == packed.metadata and restored.spec == spec
    assert restored.layout.sha256() == packed.layout.sha256()
    for key, (down, up) in factors.items():
        np.testing.assert_array_equal(restored.factors[key][0], down)
        np.testing.assert_array_equal(restored.factors[key][1], up)

    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    tampered = dict(arrays)
    tampered["tokens"] = arrays["tokens"] + 0.5 * packed.mask
    np.savez_compressed(tmp_path / "tampered.npz", **tampered)
    with pytest.raises(ValueError, match="not the effective update of the stored factors"):
        load_packed_adapter(tmp_path / "tampered.npz")

    legacy_metadata = {
        key: value
        for key, value in json.loads(bytes(arrays["packed_adapter_metadata_json"]).decode()).items()
        if key != "schema_version"
    }
    legacy = dict(arrays)
    legacy["packed_adapter_metadata_json"] = np.frombuffer(
        json.dumps(legacy_metadata).encode(), dtype=np.uint8
    )
    np.savez_compressed(tmp_path / "legacy.npz", **legacy)
    with pytest.raises(LegacyArtifactError, match="version-one packed adapter"):
        load_packed_adapter(tmp_path / "legacy.npz")


def test_adapter_spec_records_placement_routing_and_action_convention():
    spec = synthetic_spec()
    payload = spec.to_dict()
    assert payload["placement"] == "pre_block_residual"
    assert payload["token_routing"]["obs_primary"] == "vision"
    assert payload["action_convention"]["policy_normalization"] == "identity"
    assert AdapterSpec.from_dict(payload) == spec
    with pytest.raises(LegacyArtifactError, match="schema 1"):
        AdapterSpec.from_dict({**payload, "schema_version": 1})
    with pytest.raises(ValueError, match="unsupported adapter placement"):
        AdapterSpec.from_dict({**payload, "placement": "post_block_residual"})
    with pytest.raises(ValueError, match="unknown destinations"):
        AdapterSpec.from_dict(
            {**payload, "token_routing": {**payload["token_routing"], "obs_depth": "depth"}}
        )
    changed = AdapterSpec.from_dict(
        {**payload, "token_routing": {**payload["token_routing"], "obs_proprio": "frozen"}}
    )
    assert changed.sha256() != spec.sha256()


def test_factors_to_updates_rejects_missing_paths():
    spec = synthetic_spec()
    with pytest.raises(ValueError, match="factor paths differ"):
        factors_to_updates(spec, {})
