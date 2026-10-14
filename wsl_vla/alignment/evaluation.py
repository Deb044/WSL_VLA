"""Offline alignment diagnostics.

Three outcomes are kept apart and never folded into one pass/fail bit:

``integrity``
    Whether the comparison is valid at all (paired checkpoints, matching
    hashes, finite numbers). Only an integrity failure fails a command.
``inconclusive``
    The data cannot distinguish the conditions (too few tasks, or an interval
    containing zero).
``negative finding``
    The aligned condition is reliably *not* better. This is a scientific
    result and is reported as such; infrastructure still passes.

All normalized metrics are scale-aware ratios. When a denominator is zero the
metric is reported as undefined (``None``) with a count, never as 0 or infinity.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..adapters.rank import relative_error
from ..contracts import COMPONENT_NAMES, AdapterSpec


INTEGRITY_PASS = "pass"
INTEGRITY_FAIL = "fail"


def component_update_error(
    spec: AdapterSpec,
    predicted: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, float | None]:
    """Root relative squared error per component, plus their mean (``macro``).

    ``predicted`` and ``target`` are dense effective updates in original units.
    Multiplying both by the same factor leaves every value unchanged. A
    component whose target is exactly zero has no relative error: it is
    ``None`` and is excluded from the macro average.
    """

    if set(predicted) != set(target) or set(target) != {e.parameter_path for e in spec.entries}:
        raise ValueError("update paths differ from AdapterSpec")
    error = {name: 0.0 for name in COMPONENT_NAMES}
    reference = {name: 0.0 for name in COMPONENT_NAMES}
    for entry in spec.entries:
        truth = np.asarray(target[entry.parameter_path], dtype=np.float64)
        guess = np.asarray(predicted[entry.parameter_path], dtype=np.float64)
        if guess.shape != truth.shape:
            raise ValueError(f"update shape mismatch at {entry.parameter_path}")
        if not np.isfinite(guess).all() or not np.isfinite(truth).all():
            raise FloatingPointError(f"non-finite update at {entry.parameter_path}")
        error[entry.component.value] += float(np.square(guess - truth).sum())
        reference[entry.component.value] += float(np.square(truth).sum())
    result: dict[str, float | None] = {
        name: relative_error(error[name], reference[name]) for name in COMPONENT_NAMES
    }
    defined = [value for value in result.values() if value is not None]
    result["macro"] = float(np.mean(defined)) if defined else None
    return result


def relative_to_training_mean(
    prediction: Any, target: Any, training_mean: Any
) -> dict[str, float | None]:
    """Squared latent error relative to always predicting the training mean.

    A ratio below 1 means the prediction beats the training-mean predictor. If
    the target equals the training mean the baseline error is zero and the
    ratio is undefined.
    """

    truth = np.asarray(target, dtype=np.float64)
    error = float(np.square(np.asarray(prediction, np.float64) - truth).sum())
    baseline = float(np.square(np.asarray(training_mean, np.float64) - truth).sum())
    return {
        "squared_error": error,
        "training_mean_squared_error": baseline,
        "relative_to_training_mean": None if baseline <= 0.0 else error / baseline,
    }


def summarize_task_metric(values: Mapping[str, Sequence[float | None]]) -> dict[str, Any]:
    """Task-balanced mean of a metric that may be undefined for some samples.

    Samples are averaged within their task first, then tasks are averaged with
    equal weight. Undefined samples are counted, not replaced by a number.
    """

    task_means = {}
    undefined = 0
    for task, samples in values.items():
        defined = [float(value) for value in samples if value is not None]
        undefined += len(samples) - len(defined)
        if defined:
            task_means[task] = float(np.mean(defined))
    return {
        "task_balanced_mean": float(np.mean(list(task_means.values()))) if task_means else None,
        "per_task": task_means,
        "tasks": len(task_means),
        "undefined_samples": int(undefined),
    }


def pooled_token_embedding(latents: Any) -> np.ndarray:
    """Mean of L2-normalized token latents for one sample (``[tokens, dim]``)."""

    values = np.asarray(latents, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
    return (values / norms).mean(axis=0)


def gallery_retrieval(
    queries: Any,
    gallery: Any,
    query_labels: Sequence[str],
    gallery_labels: Sequence[str],
    *,
    center: Any | None = None,
) -> dict[str, Any]:
    """Top-1 cosine retrieval of each query's own task from a task-balanced gallery.

    The gallery must hold exactly one entry per task so every task is equally
    represented; chance is then ``1 / gallery_size`` and is reported explicitly.
    ``center`` (for example the training-mean latent) is subtracted from both
    sides first, because latents of all tasks share a large common component
    that would otherwise dominate the cosine.
    """

    left = np.asarray(queries, dtype=np.float64).reshape(len(query_labels), -1)
    right = np.asarray(gallery, dtype=np.float64).reshape(len(gallery_labels), -1)
    if left.shape[1] != right.shape[1]:
        raise ValueError("queries and gallery live in different spaces")
    if len(set(gallery_labels)) != len(gallery_labels):
        raise ValueError("a task-balanced gallery holds exactly one entry per task")
    missing = sorted(set(query_labels) - set(gallery_labels))
    if missing:
        raise ValueError(f"queries have no gallery entry: {missing}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise FloatingPointError("retrieval embeddings must be finite")
    if center is not None:
        offset = np.asarray(center, dtype=np.float64).reshape(1, -1)
        left, right = left - offset, right - offset
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    similarity = left @ right.T
    labels = list(gallery_labels)
    correct = []
    reciprocal = []
    for row, label in enumerate(query_labels):
        order = np.argsort(-similarity[row], kind="stable")
        rank = int(np.flatnonzero(order == labels.index(label))[0]) + 1
        correct.append(rank == 1)
        reciprocal.append(1.0 / rank)
    return {
        "top1_accuracy": float(np.mean(correct)),
        "mean_reciprocal_rank": float(np.mean(reciprocal)),
        "chance_top1": 1.0 / len(labels),
        "queries": len(query_labels),
        "gallery_size": len(labels),
        "per_query_correct": {label: bool(hit) for label, hit in zip(query_labels, correct)},
    }


def assert_finite_report(report: Any, path: str = "report") -> None:
    """Integrity check: every number in a diagnostics report is finite or ``None``."""

    if isinstance(report, Mapping):
        for key, value in report.items():
            assert_finite_report(value, f"{path}.{key}")
    elif isinstance(report, (list, tuple)):
        for index, value in enumerate(report):
            assert_finite_report(value, f"{path}[{index}]")
    elif isinstance(report, (float, np.floating)) and not np.isfinite(report):
        raise FloatingPointError(f"non-finite diagnostic value at {path}")
