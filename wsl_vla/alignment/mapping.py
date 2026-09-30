"""WeightCLIP-style prompt-to-latent linear ridge mapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearRidgeMapper:
    coefficient: np.ndarray
    intercept: np.ndarray
    latent_shape: tuple[int, ...]
    ridge_alpha: float

    def predict(self, evidence: np.ndarray) -> np.ndarray:
        evidence = np.asarray(evidence, dtype=np.float64)
        single = evidence.ndim == 1
        evidence = np.atleast_2d(evidence)
        flat = evidence @ self.coefficient + self.intercept
        result = flat.reshape((evidence.shape[0], *self.latent_shape))
        return result[0] if single else result


def fit_linear_ridge_mapper(
    evidence: np.ndarray,
    latents: np.ndarray,
    *,
    ridge_alpha: float,
) -> LinearRidgeMapper:
    evidence = np.asarray(evidence, dtype=np.float64)
    latents = np.asarray(latents, dtype=np.float64)
    if evidence.ndim != 2 or latents.ndim < 2 or evidence.shape[0] != latents.shape[0]:
        raise ValueError("evidence and latents must share a sample dimension")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha cannot be negative")
    x_mean = evidence.mean(axis=0)
    y = latents.reshape(latents.shape[0], -1)
    y_mean = y.mean(axis=0)
    centered_x = evidence - x_mean
    centered_y = y - y_mean
    gram = centered_x.T @ centered_x
    regularized = gram + ridge_alpha * np.eye(gram.shape[0])
    coefficient = np.linalg.solve(regularized, centered_x.T @ centered_y)
    intercept = y_mean - x_mean @ coefficient
    return LinearRidgeMapper(
        coefficient=coefficient,
        intercept=intercept,
        latent_shape=tuple(latents.shape[1:]),
        ridge_alpha=ridge_alpha,
    )
