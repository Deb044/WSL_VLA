"""Held-out retrieval and mapper gates for aligned latent spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def multi_positive_retrieval_accuracy(
    weight_embeddings: np.ndarray,
    evidence_embeddings: np.ndarray,
    task_labels: np.ndarray,
) -> float:
    """Top-1 cosine retrieval where any same-task sample is a positive."""

    weights = np.array(weight_embeddings, dtype=np.float64, copy=True)
    evidence = np.array(evidence_embeddings, dtype=np.float64, copy=True)
    labels = np.asarray(task_labels)
    if weights.ndim != 2 or evidence.shape != weights.shape or labels.shape != (weights.shape[0],):
        raise ValueError("embeddings must be matching [samples, dim] arrays with one label each")
    if not np.isfinite(weights).all() or not np.isfinite(evidence).all():
        raise ValueError("retrieval embeddings must be finite")
    weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
    evidence /= np.maximum(np.linalg.norm(evidence, axis=1, keepdims=True), 1e-12)
    nearest = np.argmax(weights @ evidence.T, axis=1)
    return float(np.mean(labels[nearest] == labels))


@dataclass(frozen=True)
class AlignmentGateResult:
    aligned_retrieval: float
    reconstruction_only_retrieval: float
    aligned_mapper_mse: float
    reconstruction_only_mapper_mse: float

    @property
    def passed(self) -> bool:
        return (
            self.aligned_retrieval > self.reconstruction_only_retrieval
            and self.aligned_mapper_mse < self.reconstruction_only_mapper_mse
        )


def alignment_advantage_gate(
    *,
    aligned_weight_embeddings: np.ndarray,
    aligned_evidence_embeddings: np.ndarray,
    reconstruction_weight_embeddings: np.ndarray,
    reconstruction_evidence_embeddings: np.ndarray,
    task_labels: np.ndarray,
    aligned_mapper_prediction: np.ndarray,
    reconstruction_mapper_prediction: np.ndarray,
    mapper_target: np.ndarray,
) -> AlignmentGateResult:
    result = AlignmentGateResult(
        aligned_retrieval=multi_positive_retrieval_accuracy(
            aligned_weight_embeddings, aligned_evidence_embeddings, task_labels
        ),
        reconstruction_only_retrieval=multi_positive_retrieval_accuracy(
            reconstruction_weight_embeddings,
            reconstruction_evidence_embeddings,
            task_labels,
        ),
        aligned_mapper_mse=float(
            np.mean(np.square(np.asarray(aligned_mapper_prediction) - mapper_target))
        ),
        reconstruction_only_mapper_mse=float(
            np.mean(np.square(np.asarray(reconstruction_mapper_prediction) - mapper_target))
        ),
    )
    if not result.passed:
        raise AssertionError(f"aligned space did not beat reconstruction-only baseline: {result}")
    return result
