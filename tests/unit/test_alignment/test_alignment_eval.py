import numpy as np
import pytest

from wsl_vla.alignment.evaluation import (
    ModalityAlignmentMetrics,
    alignment_advantage_gate,
    compare_alignment_checkpoints,
    modality_alignment_metrics,
    pool_modality_token_embeddings,
)


def test_alignment_gate_requires_both_retrieval_and_mapping_improvement():
    labels = np.array([0, 1])
    aligned = np.eye(2)
    reconstruction = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = alignment_advantage_gate(
        aligned_weight_embeddings=aligned,
        aligned_evidence_embeddings=aligned,
        reconstruction_weight_embeddings=reconstruction,
        reconstruction_evidence_embeddings=aligned,
        task_labels=labels,
        aligned_mapper_prediction=np.zeros((2, 2)),
        reconstruction_mapper_prediction=np.ones((2, 2)),
        mapper_target=np.zeros((2, 2)),
    )
    assert result.passed
    with pytest.raises(AssertionError):
        alignment_advantage_gate(
            aligned_weight_embeddings=reconstruction,
            aligned_evidence_embeddings=aligned,
            reconstruction_weight_embeddings=aligned,
            reconstruction_evidence_embeddings=aligned,
            task_labels=labels,
            aligned_mapper_prediction=np.zeros((2, 2)),
            reconstruction_mapper_prediction=np.ones((2, 2)),
            mapper_target=np.zeros((2, 2)),
        )


def test_modality_metrics_pool_tokens_and_mask_mapper_error():
    token_embeddings = np.array(
        [
            [[1.0, 0.0], [0.0, 4.0], [50.0, 50.0]],
            [[0.0, 1.0], [-3.0, 0.0], [50.0, 50.0]],
        ]
    )
    valid = np.array([[True, True, False], [True, True, False]])
    pooled = pool_modality_token_embeddings(token_embeddings, valid)
    np.testing.assert_allclose(np.linalg.norm(pooled, axis=1), 1.0)
    prediction = token_embeddings.copy()
    prediction[:, 2] = -999.0
    metrics = modality_alignment_metrics(
        weight_token_embeddings=token_embeddings,
        token_valid=valid,
        evidence_embeddings=pooled,
        task_labels=np.array([0, 1]),
        mapper_prediction=prediction,
    )
    assert metrics.retrieval_accuracy == 1.0
    assert metrics.mapper_mse == 0.0
    assert metrics.sample_count == 2
    assert metrics.task_count == 2


def test_checkpoint_comparison_uses_predeclared_macro_gate():
    aligned = {
        name: ModalityAlignmentMetrics(0.8, 0.1, 54, 6)
        for name in ("vision", "language", "action")
    }
    reconstruction = {
        name: ModalityAlignmentMetrics(0.5, 0.4, 54, 6)
        for name in ("vision", "language", "action")
    }
    result = compare_alignment_checkpoints(
        aligned=aligned, reconstruction_only=reconstruction
    )
    assert result.passed
    failed = compare_alignment_checkpoints(
        aligned=reconstruction, reconstruction_only=aligned, enforce=False
    )
    assert not failed.passed
    with pytest.raises(AssertionError):
        compare_alignment_checkpoints(
            aligned=reconstruction, reconstruction_only=aligned
        )


def test_checkpoint_comparison_rejects_different_validation_populations():
    aligned = {
        name: ModalityAlignmentMetrics(0.8, 0.1, 54, 6)
        for name in ("vision", "language", "action")
    }
    reconstruction = dict(aligned)
    reconstruction["action"] = ModalityAlignmentMetrics(0.8, 0.1, 45, 5)
    with pytest.raises(ValueError, match="same population"):
        compare_alignment_checkpoints(
            aligned=aligned, reconstruction_only=reconstruction
        )
