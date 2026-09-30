"""Fail-closed validation for the official packed-adapter population."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from ..adapters.packing import load_packed_adapter
from ..alignment.evidence_io import load_task_evidence
from .provenance import sha256_file


PopulationIdentity = tuple[str, int, int, float]


def locked_population_identities(
    suites: Iterable[str],
    *,
    task_count: int = 10,
    seeds: Iterable[int] = (17, 42, 73),
    fractions: Iterable[float] = (0.8, 0.9, 1.0),
) -> set[PopulationIdentity]:
    return {
        (suite, task_index, int(seed), float(fraction))
        for suite in suites
        for task_index in range(task_count)
        for seed in seeds
        for fraction in fractions
    }


def validate_research_population(
    population_root: str | Path,
    *,
    expected_identities: set[PopulationIdentity],
) -> dict:
    """Validate identities, hashes, evidence, base, and adapter schema."""
    root = Path(population_root)
    sample_dirs = sorted(path.parent for path in root.rglob("adapter.npz"))
    if len(sample_dirs) != len(expected_identities):
        raise ValueError(
            f"research population requires {len(expected_identities)} adapters, "
            f"found {len(sample_dirs)}"
        )

    observed: set[PopulationIdentity] = set()
    base_hashes: set[str] = set()
    base_revisions: set[str] = set()
    spec_hashes: set[str] = set()
    task_provenance: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for sample_dir in sample_dirs:
        metadata_path = sample_dir / "metadata.json"
        evidence_path = sample_dir / "evidence.npz"
        if not metadata_path.is_file() or not evidence_path.is_file():
            raise FileNotFoundError(f"incomplete research population sample: {sample_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1 or metadata.get("synthetic", True):
            raise ValueError(f"invalid or synthetic population metadata: {sample_dir}")
        identity = (
            str(metadata.get("suite")),
            int(metadata.get("task_index", -1)),
            int(metadata.get("seed", -1)),
            float(metadata.get("checkpoint_fraction", -1)),
        )
        if identity in observed:
            raise ValueError(f"duplicate research population identity: {identity}")
        observed.add(identity)
        if sha256_file(sample_dir / "adapter.npz") != metadata.get("adapter_sha256"):
            raise ValueError(f"adapter hash mismatch: {sample_dir}")
        if sha256_file(evidence_path) != metadata.get("evidence_sha256"):
            raise ValueError(f"evidence hash mismatch: {sample_dir}")

        packed = load_packed_adapter(sample_dir / "adapter.npz")
        evidence = load_task_evidence(evidence_path)
        task_id = f"{identity[0]}_{identity[1]}"
        if (
            metadata.get("task_id") != task_id
            or packed.metadata.task_id != task_id
            or packed.metadata.suite != identity[0]
            or packed.metadata.seed != identity[2]
            or packed.metadata.checkpoint_fraction != identity[3]
            or evidence.task_id != task_id
            or evidence.suite != identity[0]
        ):
            raise ValueError(f"cross-file population identity mismatch: {sample_dir}")
        expected_split = "validation" if identity[1] in {8, 9} else "train"
        if metadata.get("split") != expected_split:
            raise ValueError(f"invalid locked train/validation split: {sample_dir}")
        base_hash = str(metadata.get("base_sha256", ""))
        if packed.spec.base_sha256 != base_hash:
            raise ValueError(f"packed adapter uses a different base: {sample_dir}")
        base_hashes.add(base_hash)
        base_revisions.add(str(metadata.get("base_revision", "")))
        spec_hashes.add(
            hashlib.sha256(
                json.dumps(packed.spec.to_dict(), sort_keys=True).encode("utf-8")
            ).hexdigest()
        )
        task_provenance.setdefault(identity[:2], set()).add(
            (str(metadata.get("dataset_sha256", "")), metadata["evidence_sha256"])
        )

    if observed != expected_identities:
        raise ValueError(
            "population identities differ from the locked protocol; "
            f"missing={len(expected_identities-observed)}, extra={len(observed-expected_identities)}"
        )
    if len(base_hashes) != 1 or "" in base_hashes:
        raise ValueError("population does not share one valid base checkpoint hash")
    if len(base_revisions) != 1 or "" in base_revisions:
        raise ValueError("population does not share one base checkpoint revision")
    if len(spec_hashes) != 1:
        raise ValueError("population contains incompatible adapter specifications")
    inconsistent = [task for task, values in task_provenance.items() if len(values) != 1]
    if inconsistent or any("" in pair for values in task_provenance.values() for pair in values):
        raise ValueError(f"task data/evidence provenance is inconsistent: {inconsistent}")
    return {
        "schema_version": 1,
        "ok": True,
        "sample_count": len(sample_dirs),
        "task_count": len(task_provenance),
        "base_sha256": next(iter(base_hashes)),
        "base_revision": next(iter(base_revisions)),
        "adapter_spec_sha256": next(iter(spec_hashes)),
    }
