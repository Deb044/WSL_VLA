"""Versioned, no-pickle persistence for real task evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..contracts import (
    TASK_EVIDENCE_SCHEMA_VERSION,
    LegacyArtifactError,
    TaskEvidence,
    require_schema,
)


ARRAY_FIELDS = (
    "vision_features",
    "vision_episode_index",
    "vision_timestep",
    "language_features",
    "language_mask",
    "action_chunk_features",
    "action_chunk_mask",
)


def save_task_evidence(evidence: TaskEvidence, path: str | Path) -> None:
    """Persist evidence together with episode and preprocessing provenance."""

    evidence.validate_for_research()
    metadata = {
        "schema_version": evidence.schema_version,
        "task_id": evidence.task_id,
        "suite": evidence.suite,
        "instruction": evidence.instruction,
        "episode_ids": list(evidence.episode_ids),
        "feature_extractor_sha256": dict(evidence.feature_extractor_sha256),
        "preprocessing_sha256": evidence.preprocessing_sha256,
        "action_convention_sha256": evidence.action_convention_sha256,
        "base_sha256": evidence.base_sha256,
        "base_revision": evidence.base_revision,
        "architecture_sha256": evidence.architecture_sha256,
        "dataset_sha256": evidence.dataset_sha256,
        "extraction": dict(evidence.extraction),
        "synthetic": evidence.synthetic,
    }
    arrays = {name: np.asarray(getattr(evidence, name)) for name in ARRAY_FIELDS}
    arrays["evidence_metadata_json"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the target and renamed, so a reader never sees a partial file.
    temporary = target.with_name(f".{target.name}.writing.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_task_evidence(path: str | Path) -> TaskEvidence:
    """Load and validate one evidence artifact without permitting pickle data."""

    with np.load(path, allow_pickle=False) as archive:
        if "evidence_metadata_json" not in archive.files:
            raise ValueError(f"evidence archive has no metadata: {path}")
        metadata = json.loads(bytes(archive["evidence_metadata_json"]).decode("utf-8"))
        if metadata.get("schema_version") == 1 or "action_statistics" in archive.files:
            raise LegacyArtifactError(
                f"{path} is version-one evidence (single action-summary vector, per-task "
                "normalization). It is inspection-only; re-extract evidence."
            )
        require_schema(metadata.get("schema_version"), TASK_EVIDENCE_SCHEMA_VERSION, "TaskEvidence")
        missing = set(ARRAY_FIELDS) - set(archive.files)
        if missing:
            raise ValueError(f"evidence archive lacks arrays: {sorted(missing)}")
        arrays = {name: np.array(archive[name], copy=True) for name in ARRAY_FIELDS}
    evidence = TaskEvidence(
        task_id=str(metadata["task_id"]),
        suite=str(metadata["suite"]),
        instruction=str(metadata["instruction"]),
        episode_ids=tuple(str(value) for value in metadata["episode_ids"]),
        feature_extractor_sha256={
            str(key): str(value) for key, value in metadata["feature_extractor_sha256"].items()
        },
        preprocessing_sha256=str(metadata["preprocessing_sha256"]),
        action_convention_sha256=str(metadata["action_convention_sha256"]),
        base_sha256=str(metadata["base_sha256"]),
        base_revision=str(metadata["base_revision"]),
        architecture_sha256=str(metadata["architecture_sha256"]),
        dataset_sha256=str(metadata["dataset_sha256"]),
        extraction=dict(metadata.get("extraction", {})),
        synthetic=bool(metadata.get("synthetic", True)),
        schema_version=int(metadata["schema_version"]),
        **arrays,
    )
    evidence.validate_for_research()
    return evidence
