import numpy as np
import pytest

from wsl_vla.alignment.evaluation import alignment_advantage_gate


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
