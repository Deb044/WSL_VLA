"""Rank-matched deployment: enforcement, reported error, reloadable factors."""

import numpy as np
import pytest

from tests.synthetic import synthetic_factors, synthetic_spec
from wsl_vla.adapters.packing import factors_to_updates
from wsl_vla.adapters.rank import (
    project_updates_to_rank,
    relative_error,
    top_singular_triplets,
    update_rank,
)


def test_projection_enforces_each_entry_rank_and_reports_the_error():
    rng = np.random.default_rng(0)
    spec = synthetic_spec()
    dense = {
        entry.parameter_path: rng.normal(size=(entry.input_dim, entry.output_dim))
        for entry in spec.entries
    }
    projection = project_updates_to_rank(spec, dense)
    for entry in spec.entries:
        projected = projection.updates[entry.parameter_path]
        assert update_rank(projected) <= entry.effective_rank_bound
        down, up = projection.factors[entry.parameter_path]
        assert down.shape == (entry.input_dim, entry.rank) and up.shape == (entry.rank, entry.output_dim)
        # The exported factors reproduce the deployed update: they are reloadable.
        np.testing.assert_allclose(
            (spec.alpha / entry.rank) * down @ up, projected, rtol=1e-4, atol=1e-5
        )
        # Truncated SVD is the best rank-r approximation.
        singular = np.linalg.svd(dense[entry.parameter_path], compute_uv=False)
        keep = entry.effective_rank_bound
        expected = np.sqrt(np.square(singular[keep:]).sum() / np.square(singular).sum())
        assert projection.entry_relative_error[entry.parameter_path] == pytest.approx(expected, abs=1e-6)
    report = projection.report()
    assert 0 < report["relative_error"] < 1
    assert set(report["component_relative_error"]) == {"vision", "language", "action"}
    # The 4x1 kernel has rank 1 < nominal rank 2: it is represented exactly.
    assert projection.entry_relative_error["h/time/kernel"] == pytest.approx(0.0, abs=1e-6)


def test_projection_is_exact_for_updates_that_are_already_rank_matched():
    rng = np.random.default_rng(1)
    spec = synthetic_spec()
    updates = factors_to_updates(spec, synthetic_factors(spec, rng))
    projection = project_updates_to_rank(spec, updates)
    assert projection.relative_error == pytest.approx(0.0, abs=1e-5)
    for path, value in updates.items():
        np.testing.assert_allclose(projection.updates[path], value, rtol=1e-4, atol=1e-5)


def test_zero_update_has_undefined_relative_error_not_a_manufactured_score():
    spec = synthetic_spec()
    zeros = {
        entry.parameter_path: np.zeros((entry.input_dim, entry.output_dim)) for entry in spec.entries
    }
    projection = project_updates_to_rank(spec, zeros)
    assert projection.relative_error is None
    assert all(value is None for value in projection.entry_relative_error.values())
    assert sorted(projection.report()["undefined_entries"]) == sorted(zeros)
    assert relative_error(0.0, 0.0) is None and relative_error(1.0, 4.0) == 0.5


def test_projection_rejects_wrong_shapes_and_non_finite_updates():
    spec = synthetic_spec()
    good = {
        entry.parameter_path: np.ones((entry.input_dim, entry.output_dim)) for entry in spec.entries
    }
    bad = dict(good)
    first = spec.entries[0].parameter_path
    bad[first] = np.ones((3, 3))
    with pytest.raises(ValueError, match="bad update shape"):
        project_updates_to_rank(spec, bad)
    bad[first] = np.full_like(good[first], np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        project_updates_to_rank(spec, bad)


@pytest.mark.parametrize("shape", [(96, 96), (160, 40), (40, 160), (12, 9), (16, 1)])
@pytest.mark.parametrize("kind", ["dense", "low_rank", "zero"])
def test_leading_triplets_match_a_full_svd(shape, kind):
    rng = np.random.default_rng(7)
    if kind == "dense":
        matrix = rng.normal(size=shape)
    elif kind == "low_rank":
        matrix = rng.normal(size=(shape[0], 3)) @ rng.normal(size=(3, shape[1]))
    else:
        matrix = np.zeros(shape)
    keep = min(4, *shape)
    left, singular, right = top_singular_triplets(matrix, keep)
    full_left, full_singular, full_right = np.linalg.svd(matrix, full_matrices=False)
    np.testing.assert_allclose(singular, full_singular[:keep], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        (left * singular) @ right,
        (full_left[:, :keep] * full_singular[:keep]) @ full_right[:keep],
        rtol=1e-8,
        atol=1e-8,
    )
