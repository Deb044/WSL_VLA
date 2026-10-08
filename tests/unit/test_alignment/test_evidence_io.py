from __future__ import annotations

import json

import numpy as np
import pytest

from wsl_vla.alignment.evidence_io import load_task_evidence, save_task_evidence
from wsl_vla.contracts import TaskEvidence


def make_evidence(*, synthetic: bool = False) -> TaskEvidence:
    return TaskEvidence(
        task_id="libero_spatial_0",
        suite="libero_spatial",
        vision_features=np.arange(12, dtype=np.float32).reshape(3, 4),
        vision_mask=np.array([True, True, False]),
        language_features=np.arange(4, dtype=np.float32),
        language_mask=np.array([True, True, True, False]),
        action_statistics=np.arange(42, dtype=np.float32),
        episode_ids=("demo_0", "demo_2"),
        preprocessing_version="octo-small-1.5:v1",
        raw_feature_references={"dataset": "sha256:abc"},
        synthetic=synthetic,
    )


def test_task_evidence_round_trip_preserves_provenance(tmp_path):
    path = tmp_path / "evidence.npz"
    save_task_evidence(make_evidence(), path)
    loaded = load_task_evidence(path)
    assert loaded.task_id == "libero_spatial_0"
    assert loaded.episode_ids == ("demo_0", "demo_2")
    assert loaded.preprocessing_version == "octo-small-1.5:v1"
    assert loaded.raw_feature_references == {"dataset": "sha256:abc"}
    np.testing.assert_array_equal(loaded.vision_mask, np.array([True, True, False]))


def test_synthetic_evidence_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError, match="synthetic evidence is forbidden"):
        save_task_evidence(make_evidence(synthetic=True), tmp_path / "evidence.npz")


def test_missing_provenance_metadata_is_rejected(tmp_path):
    path = tmp_path / "evidence.npz"
    np.savez_compressed(
        path,
        vision_features=np.ones((1, 2)),
        language_features=np.ones(2),
        action_statistics=np.ones(6),
    )
    with pytest.raises(ValueError, match="evidence archive lacks arrays"):
        load_task_evidence(path)
