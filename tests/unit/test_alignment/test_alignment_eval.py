"""Scale-aware diagnostics metrics and task-balanced galleries."""

import numpy as np
import pytest

from tests.synthetic import synthetic_spec
from wsl_vla.alignment.evaluation import (
    assert_finite_report,
    component_update_error,
    gallery_retrieval,
    pooled_token_embedding,
    relative_to_training_mean,
    summarize_task_metric,
)


def random_updates(spec, rng, scale=1.0):
    return {
        entry.parameter_path: scale * rng.normal(size=(entry.input_dim, entry.output_dim))
        for entry in spec.entries
    }


def test_update_error_is_a_scale_aware_ratio_per_component():
    rng = np.random.default_rng(0)
    spec = synthetic_spec()
    target = random_updates(spec, rng)
    noise = random_updates(spec, rng, scale=0.1)
    predicted = {path: target[path] + noise[path] for path in target}
    error = component_update_error(spec, predicted, target)
    assert set(error) == {"vision", "language", "action", "macro"}
    assert all(0.05 < error[name] < 0.2 for name in ("vision", "language", "action"))
    assert error["macro"] == pytest.approx(np.mean([error[n] for n in ("vision", "language", "action")]))
    # Rescaling prediction and target together leaves every ratio unchanged.
    scaled = component_update_error(
        spec, {p: 1e4 * v for p, v in predicted.items()}, {p: 1e4 * v for p, v in target.items()}
    )
    for name in error:
        assert scaled[name] == pytest.approx(error[name], rel=1e-9)
    assert component_update_error(spec, target, target)["macro"] == 0.0


def test_zero_denominators_are_undefined_not_scored():
    rng = np.random.default_rng(1)
    spec = synthetic_spec()
    target = random_updates(spec, rng)
    for entry in spec.entries:
        if entry.component.value == "language":
            target[entry.parameter_path] = np.zeros_like(target[entry.parameter_path])
    predicted = random_updates(spec, rng)
    error = component_update_error(spec, predicted, target)
    assert error["language"] is None
    assert error["macro"] == pytest.approx(np.mean([error["vision"], error["action"]]))
    all_zero = {path: np.zeros_like(value) for path, value in target.items()}
    assert component_update_error(spec, predicted, all_zero)["macro"] is None
    with pytest.raises(FloatingPointError):
        component_update_error(spec, {p: np.full_like(v, np.nan) for p, v in predicted.items()}, target)


def test_latent_error_is_reported_relative_to_the_training_mean_predictor():
    target = np.array([[1.0, 2.0], [3.0, 4.0]])
    training_mean = np.zeros((2, 2))
    perfect = relative_to_training_mean(target, target, training_mean)
    assert perfect["relative_to_training_mean"] == 0.0
    same_as_mean = relative_to_training_mean(training_mean, target, training_mean)
    assert same_as_mean["relative_to_training_mean"] == pytest.approx(1.0)
    worse = relative_to_training_mean(-target, target, training_mean)
    assert worse["relative_to_training_mean"] == pytest.approx(4.0)
    # A different latent scale does not change the ratio.
    rescaled = relative_to_training_mean(-7 * target, 7 * target, training_mean)
    assert rescaled["relative_to_training_mean"] == pytest.approx(4.0)
    undefined = relative_to_training_mean(target + 1, target, target)
    assert undefined["relative_to_training_mean"] is None
    assert undefined["training_mean_squared_error"] == 0.0


def test_task_summaries_average_within_task_first_and_count_undefined_samples():
    summary = summarize_task_metric(
        {"a": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "b": [3.0], "c": [None, None]}
    )
    # Task-balanced: (1 + 3) / 2, not the sample mean 9/7; the undefined task is not a zero.
    assert summary["task_balanced_mean"] == pytest.approx(2.0)
    assert summary["tasks"] == 2 and summary["undefined_samples"] == 2
    assert summarize_task_metric({"a": [None]})["task_balanced_mean"] is None


def test_gallery_retrieval_reports_chance_and_requires_one_entry_per_task():
    rng = np.random.default_rng(2)
    gallery = rng.normal(size=(5, 12))
    labels = [f"task_{index}" for index in range(5)]
    result = gallery_retrieval(gallery + 0.01 * rng.normal(size=gallery.shape), gallery, labels, labels)
    assert result["top1_accuracy"] == 1.0 and result["chance_top1"] == pytest.approx(0.2)
    assert result["gallery_size"] == 5 and result["mean_reciprocal_rank"] == 1.0
    assert all(result["per_query_correct"].values())
    random = gallery_retrieval(rng.normal(size=(5, 12)), gallery, labels, labels)
    assert 0.0 <= random["top1_accuracy"] <= 1.0
    with pytest.raises(ValueError, match="exactly one entry per task"):
        gallery_retrieval(gallery, np.concatenate([gallery, gallery[:1]]), labels, labels + ["task_0"])
    with pytest.raises(ValueError, match="no gallery entry"):
        gallery_retrieval(gallery[:1], gallery[1:], ["task_0"], labels[1:])


def test_centering_removes_the_component_shared_by_every_task():
    rng = np.random.default_rng(3)
    shared = 100.0 * rng.normal(size=12)
    distinct = rng.normal(size=(4, 12))
    labels = [f"task_{index}" for index in range(4)]
    queries = shared + distinct + 0.05 * rng.normal(size=distinct.shape)
    centred = gallery_retrieval(queries, shared + distinct, labels, labels, center=shared)
    assert centred["top1_accuracy"] == 1.0
    # Latent sequences are compared as whole sequences, whatever their shape.
    sequence = gallery_retrieval(
        (shared + distinct).reshape(4, 3, 4), (shared + distinct).reshape(4, 3, 4), labels, labels,
        center=shared.reshape(3, 4),
    )
    assert sequence["top1_accuracy"] == 1.0


def test_pooled_embedding_is_scale_free_per_token():
    tokens = np.array([[3.0, 0.0], [0.0, 0.5]])
    np.testing.assert_allclose(pooled_token_embedding(tokens), [0.5, 0.5])
    np.testing.assert_allclose(pooled_token_embedding(10 * tokens), [0.5, 0.5])


def test_non_finite_report_values_are_an_integrity_failure():
    assert_finite_report({"a": [1.0, None, {"b": 2}], "c": "text"})
    with pytest.raises(FloatingPointError, match="report.a\\[1\\].b"):
        assert_finite_report({"a": [1.0, {"b": float("nan")}]})
