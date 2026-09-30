import json

import numpy as np
import pytest

from wsl_vla.adapters.packing import (
    PackedAdapterMetadata,
    pack_low_rank_adapter,
    save_packed_adapter,
)
from wsl_vla.alignment.evidence_io import save_task_evidence
from wsl_vla.contracts import AdapterEntry, AdapterSpec, Component, TaskEvidence
from wsl_vla.experiments.provenance import sha256_file
from wsl_vla.experiments.zoo_validation import validate_research_population


def _population(tmp_path):
    spec = AdapterSpec(
        base_model_id="test/model",
        base_revision="revision",
        base_sha256="a" * 64,
        alpha=2.0,
        token_width=2,
        entries=(AdapterEntry(Component.VISION, 0, "vision/kernel", 2, 2, 1),),
    )
    factors = {"vision/kernel": (np.ones((2, 1)), np.ones((1, 2)))}
    expected = set()
    for fraction, step in ((0.8, 8), (1.0, 10)):
        directory = tmp_path / "libero_spatial" / "libero_spatial_0" / "seed_17" / f"step_{step}"
        directory.mkdir(parents=True)
        adapter = pack_low_rank_adapter(
            spec,
            factors,
            metadata=PackedAdapterMetadata(
                task_id="libero_spatial_0",
                suite="libero_spatial",
                seed=17,
                checkpoint_stage=step,
                checkpoint_fraction=fraction,
                source_dtype="float64",
            ),
        )
        save_packed_adapter(adapter, directory / "adapter.npz")
        evidence = TaskEvidence(
            task_id="libero_spatial_0",
            suite="libero_spatial",
            vision_features=np.ones((1, 2)),
            language_features=np.ones(2),
            action_statistics=np.ones(4),
            episode_ids=("demo_0",),
            preprocessing_version="test",
            synthetic=False,
        )
        save_task_evidence(evidence, directory / "evidence.npz")
        metadata = {
            "schema_version": 1,
            "synthetic": False,
            "task_id": "libero_spatial_0",
            "suite": "libero_spatial",
            "task_index": 0,
            "seed": 17,
            "checkpoint_stage": step,
            "checkpoint_fraction": fraction,
            "split": "train",
            "base_sha256": "a" * 64,
            "base_revision": "revision",
            "dataset_sha256": "b" * 64,
            "adapter_sha256": sha256_file(directory / "adapter.npz"),
            "evidence_sha256": sha256_file(directory / "evidence.npz"),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        expected.add(("libero_spatial", 0, 17, fraction))
    return expected


def test_research_population_validation_checks_complete_cross_file_contract(tmp_path):
    expected = _population(tmp_path)
    report = validate_research_population(tmp_path, expected_identities=expected)
    assert report["sample_count"] == 2
    assert report["task_count"] == 1
    assert report["base_sha256"] == "a" * 64


def test_research_population_validation_rejects_tampering(tmp_path):
    expected = _population(tmp_path)
    evidence = next(tmp_path.rglob("evidence.npz"))
    evidence.write_bytes(evidence.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        validate_research_population(tmp_path, expected_identities=expected)
