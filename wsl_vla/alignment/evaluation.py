"""Held-out retrieval and mapper gates for aligned latent spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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


@dataclass(frozen=True)
class ModalityAlignmentMetrics:
    """Validation metrics for one evidence/adapter component."""

    retrieval_accuracy: float
    mapper_mse: float
    sample_count: int
    task_count: int


@dataclass(frozen=True)
class AlignmentCheckpointComparison:
    """Predeclared macro comparison plus diagnostic per-modality results."""

    aligned: Mapping[str, ModalityAlignmentMetrics]
    reconstruction_only: Mapping[str, ModalityAlignmentMetrics]
    gate: AlignmentGateResult

    @property
    def passed(self) -> bool:
        return self.gate.passed


def pool_modality_token_embeddings(
    token_embeddings: np.ndarray,
    token_valid: np.ndarray,
) -> np.ndarray:
    """Pool normalized token embeddings exactly like reverse InfoNCE retrieval."""

    embeddings = np.asarray(token_embeddings, dtype=np.float64)
    valid = np.asarray(token_valid, dtype=bool)
    if embeddings.ndim != 3 or valid.shape != embeddings.shape[:2]:
        raise ValueError("token embeddings must be [samples, tokens, dim] with a matching mask")
    if not np.isfinite(embeddings).all() or np.any(valid.sum(axis=1) == 0):
        raise ValueError("token embeddings must be finite and every sample needs a valid token")
    normalized = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-12
    )
    pooled = np.sum(normalized * valid[..., None], axis=1) / valid.sum(
        axis=1, keepdims=True
    )
    return pooled / np.maximum(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-12)


def modality_alignment_metrics(
    *,
    weight_token_embeddings: np.ndarray,
    token_valid: np.ndarray,
    evidence_embeddings: np.ndarray,
    task_labels: np.ndarray,
    mapper_prediction: np.ndarray,
) -> ModalityAlignmentMetrics:
    """Measure held-out task retrieval and full-sequence mapper error."""

    token_embeddings = np.asarray(weight_token_embeddings, dtype=np.float64)
    valid = np.asarray(token_valid, dtype=bool)
    evidence = np.asarray(evidence_embeddings, dtype=np.float64)
    labels = np.asarray(task_labels)
    prediction = np.asarray(mapper_prediction, dtype=np.float64)
    if prediction.shape != token_embeddings.shape:
        raise ValueError("mapper prediction must cover the complete modality token sequence")
    if evidence.shape != (token_embeddings.shape[0], token_embeddings.shape[-1]):
        raise ValueError("evidence embeddings must match sample and latent dimensions")
    if labels.shape != (token_embeddings.shape[0],):
        raise ValueError("one task label is required per sample")
    squared_error = np.square(prediction - token_embeddings) * valid[..., None]
    denominator = int(valid.sum()) * token_embeddings.shape[-1]
    return ModalityAlignmentMetrics(
        retrieval_accuracy=multi_positive_retrieval_accuracy(
            pool_modality_token_embeddings(token_embeddings, valid), evidence, labels
        ),
        mapper_mse=float(squared_error.sum() / denominator),
        sample_count=int(token_embeddings.shape[0]),
        task_count=int(np.unique(labels).size),
    )


def compare_alignment_checkpoints(
    *,
    aligned: Mapping[str, ModalityAlignmentMetrics],
    reconstruction_only: Mapping[str, ModalityAlignmentMetrics],
    enforce: bool = True,
) -> AlignmentCheckpointComparison:
    """Apply the locked macro-average gate across the three modalities."""

    required = {"vision", "language", "action"}
    if set(aligned) != required or set(reconstruction_only) != required:
        raise ValueError("both checkpoints require vision, language, and action metrics")
    for name in required:
        if (
            aligned[name].sample_count != reconstruction_only[name].sample_count
            or aligned[name].task_count != reconstruction_only[name].task_count
        ):
            raise ValueError(f"{name} metrics were not evaluated on the same population")
    aligned_retrieval = float(np.mean([aligned[name].retrieval_accuracy for name in sorted(required)]))
    reconstruction_retrieval = float(
        np.mean([reconstruction_only[name].retrieval_accuracy for name in sorted(required)])
    )
    aligned_mse = float(np.mean([aligned[name].mapper_mse for name in sorted(required)]))
    reconstruction_mse = float(
        np.mean([reconstruction_only[name].mapper_mse for name in sorted(required)])
    )
    gate = AlignmentGateResult(
        aligned_retrieval=aligned_retrieval,
        reconstruction_only_retrieval=reconstruction_retrieval,
        aligned_mapper_mse=aligned_mse,
        reconstruction_only_mapper_mse=reconstruction_mse,
    )
    if enforce and not gate.passed:
        raise AssertionError(f"aligned space did not beat reconstruction-only baseline: {gate}")
    return AlignmentCheckpointComparison(
        aligned=dict(aligned), reconstruction_only=dict(reconstruction_only), gate=gate
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
