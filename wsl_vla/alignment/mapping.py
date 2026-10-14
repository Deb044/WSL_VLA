"""WeightCLIP-style evidence-to-latent ridge mapper.

The mapper is fitted on **task-balanced** data: one row per training task, whose
target is the mean latent over that task's seeds and snapshots. Correlated
snapshots of one task therefore count once, and a task with more snapshots
cannot dominate the regression.

Latent targets are complete modality sequences (hundreds of thousands of
numbers) while a fold has tens of tasks, so the predictor is stored in the
equivalent *sample-weight* form::

    weights(x) = (x - x_mean) @ projection          # [tasks]
    predict(x) = y_mean + weights(x) @ y_centered   # complete latent sequence

``projection`` is ``(Xc^T Xc + a I)^-1 Xc^T`` computed through whichever of the
feature-space or sample-space systems is smaller; both give the same ridge
solution with an unpenalized intercept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LinearRidgeMapper:
    projection: np.ndarray
    x_mean: np.ndarray
    y_mean: np.ndarray
    y_centered: np.ndarray
    ridge_alpha: float

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return tuple(self.y_mean.shape)

    def sample_weights(self, evidence: np.ndarray) -> np.ndarray:
        """Weight of each training task's centered target in the prediction."""

        values = np.atleast_2d(np.asarray(evidence, dtype=np.float64))
        if values.shape[1] != self.x_mean.shape[0]:
            raise ValueError("evidence width differs from the fitted mapper")
        return (values - self.x_mean) @ self.projection

    def predict(self, evidence: np.ndarray) -> np.ndarray:
        single = np.asarray(evidence).ndim == 1
        weights = self.sample_weights(evidence)
        result = self.y_mean + np.tensordot(weights, self.y_centered, axes=1)
        return result[0] if single else result


def fit_linear_ridge_mapper(
    evidence: np.ndarray,
    latents: np.ndarray,
    *,
    ridge_alpha: float,
) -> LinearRidgeMapper:
    """Ridge regression from ``[samples, features]`` to latents of any trailing shape."""

    x = np.asarray(evidence, dtype=np.float64)
    y = np.asarray(latents, dtype=np.float64)
    if x.ndim != 2 or y.ndim < 2 or x.shape[0] != y.shape[0]:
        raise ValueError("evidence and latents must share a sample dimension")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha cannot be negative")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("mapper inputs must be finite")
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    centered_x = x - x_mean
    samples, features = centered_x.shape
    if features <= samples:
        gram = centered_x.T @ centered_x + ridge_alpha * np.eye(features)
        projection = np.linalg.lstsq(gram, centered_x.T, rcond=None)[0]
    else:
        kernel = centered_x @ centered_x.T + ridge_alpha * np.eye(samples)
        projection = centered_x.T @ np.linalg.pinv(kernel, hermitian=True)
    return LinearRidgeMapper(
        projection=projection,
        x_mean=x_mean,
        y_mean=y_mean.astype(np.float32),
        y_centered=(y - y_mean).astype(np.float32),
        ridge_alpha=float(ridge_alpha),
    )


def task_mean_targets(
    latents: np.ndarray, task_rows: Sequence[int], *, tasks: Sequence[int]
) -> np.ndarray:
    """Average sample latents within each task (seeds and snapshots)."""

    values = np.asarray(latents)
    rows = np.asarray(task_rows)
    if values.shape[0] != rows.shape[0]:
        raise ValueError("one task row is required per latent sample")
    means = []
    for task in tasks:
        members = rows == task
        if not members.any():
            raise ValueError(f"task {task} has no samples to average")
        means.append(values[members].mean(axis=0))
    return np.stack(means)


def normalized_mapper_error(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean over tasks of the per-task squared error (task-balanced)."""

    difference = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    return float(np.mean(np.square(difference).reshape(difference.shape[0], -1).mean(axis=1)))


def select_ridge_mapper(
    train_evidence: np.ndarray,
    train_targets: np.ndarray,
    validation_evidence: np.ndarray,
    validation_targets: np.ndarray,
    *,
    ridge_alphas: Sequence[float],
) -> tuple[LinearRidgeMapper, Mapping[float, float]]:
    """Fit one mapper per candidate and keep the best on *validation tasks* only.

    Every array has one row per task (task-mean targets). Ties resolve to the
    larger ridge strength, the more conservative mapper.
    """

    if not ridge_alphas:
        raise ValueError("at least one ridge strength candidate is required")
    errors: dict[float, float] = {}
    best: LinearRidgeMapper | None = None
    best_error = np.inf
    for alpha in sorted(float(value) for value in ridge_alphas):
        mapper = fit_linear_ridge_mapper(train_evidence, train_targets, ridge_alpha=alpha)
        error = normalized_mapper_error(mapper.predict(validation_evidence), validation_targets)
        errors[alpha] = error
        if error <= best_error:
            best, best_error = mapper, error
    assert best is not None
    return best, errors
