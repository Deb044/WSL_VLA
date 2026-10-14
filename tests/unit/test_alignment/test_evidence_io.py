from __future__ import annotations

import numpy as np
import pytest

from tests.synthetic import synthetic_evidence
from wsl_vla.alignment.evidence_io import load_task_evidence, save_task_evidence
from wsl_vla.contracts import LegacyArtifactError


def make_evidence(**changes):
    rng = np.random.default_rng(0)
    evidence = synthetic_evidence("libero_spatial_0", rng, rng.normal(size=8))
    for name, value in changes.items():
        setattr(evidence, name, value)
    return evidence


def test_task_evidence_round_trip_preserves_provenance(tmp_path):
    path = tmp_path / "evidence.npz"
    original = make_evidence()
    save_task_evidence(original, path)
    loaded = load_task_evidence(path)
    assert loaded.task_id == "libero_spatial_0" and loaded.episode_ids == original.episode_ids
    assert loaded.feature_extractor_sha256 == original.feature_extractor_sha256
    assert loaded.preprocessing_sha256 == original.preprocessing_sha256
    assert loaded.dataset_sha256 == original.dataset_sha256
    np.testing.assert_array_equal(loaded.vision_episode_index, original.vision_episode_index)
    np.testing.assert_array_equal(loaded.action_chunk_mask, original.action_chunk_mask)
    np.testing.assert_array_equal(loaded.language_mask, [True, True, False])


def test_synthetic_evidence_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError, match="synthetic evidence is forbidden"):
        save_task_evidence(make_evidence(synthetic=True), tmp_path / "evidence.npz")


def test_every_training_episode_must_contribute_cached_frames(tmp_path):
    evidence = make_evidence()
    evidence.vision_episode_index = np.zeros_like(evidence.vision_episode_index)
    with pytest.raises(ValueError, match="every training episode"):
        save_task_evidence(evidence, tmp_path / "evidence.npz")


def test_version_one_evidence_is_rejected_as_legacy(tmp_path):
    import json

    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        vision_features=np.ones((1, 2)),
        language_features=np.ones(2),
        action_statistics=np.ones(42),
        evidence_metadata_json=np.frombuffer(
            json.dumps({"schema_version": 1, "task_id": "t"}).encode(), dtype=np.uint8
        ),
    )
    with pytest.raises(LegacyArtifactError, match="version-one evidence"):
        load_task_evidence(path)
