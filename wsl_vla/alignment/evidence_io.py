"""Versioned, no-pickle persistence for real task evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..contracts import SCHEMA_VERSION, TaskEvidence


def save_task_evidence(evidence: TaskEvidence, path: str | Path) -> None:
    """Persist evidence together with episode and preprocessing provenance."""

    evidence.validate_for_research()
    metadata = {
        "schema_version": evidence.schema_version,
        "task_id": evidence.task_id,
        "suite": evidence.suite,
        "episode_ids": list(evidence.episode_ids),
        "preprocessing_version": evidence.preprocessing_version,
        "raw_feature_references": dict(evidence.raw_feature_references),
        "synthetic": evidence.synthetic,
        "optional_arrays": [
            name
            for name in (
                "vision_mask",
                "language_mask",
                "projected_visual",
                "projected_language",
                "projected_action",
            )
            if getattr(evidence, name) is not None
        ],
    }
    arrays = {
        "vision_features": np.asarray(evidence.vision_features),
        "language_features": np.asarray(evidence.language_features),
        "action_statistics": np.asarray(evidence.action_statistics),
        "evidence_metadata_json": np.frombuffer(
            json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
        ),
    }
    for name in metadata["optional_arrays"]:
        arrays[name] = np.asarray(getattr(evidence, name))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)


def load_task_evidence(path: str | Path) -> TaskEvidence:
    """Load and validate one evidence artifact without permitting pickle data."""

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "vision_features",
            "language_features",
            "action_statistics",
            "evidence_metadata_json",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"evidence archive lacks arrays: {sorted(missing)}")
        metadata = json.loads(bytes(archive["evidence_metadata_json"]).decode("utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence schema {metadata.get('schema_version')}")
        optional = set(metadata.get("optional_arrays", ()))
        supported_optional = {
            "vision_mask",
            "language_mask",
            "projected_visual",
            "projected_language",
            "projected_action",
        }
        if not optional <= supported_optional:
            raise ValueError(f"unknown optional evidence arrays: {sorted(optional-supported_optional)}")
        missing_optional = optional - set(archive.files)
        if missing_optional:
            raise ValueError(f"declared evidence arrays are missing: {sorted(missing_optional)}")
        arrays = {name: np.array(archive[name], copy=True) for name in optional}
        evidence = TaskEvidence(
            task_id=str(metadata["task_id"]),
            suite=str(metadata["suite"]),
            vision_features=np.array(archive["vision_features"], copy=True),
            language_features=np.array(archive["language_features"], copy=True),
            action_statistics=np.array(archive["action_statistics"], copy=True),
            episode_ids=tuple(str(value) for value in metadata["episode_ids"]),
            preprocessing_version=str(metadata["preprocessing_version"]),
            raw_feature_references={
                str(key): str(value)
                for key, value in metadata.get("raw_feature_references", {}).items()
            },
            synthetic=bool(metadata.get("synthetic", True)),
            schema_version=int(metadata["schema_version"]),
            **arrays,
        )
    evidence.validate_for_research()
    return evidence
