"""Read-only inspection of version-one (pre-repair) research artifacts.

Legacy zoo samples, evidence, archives, and alignment checkpoints stay on disk
as diagnostics. Nothing here converts them: corrected research commands reject
them by schema version, and this module only *describes* them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


LEGACY_REASONS = (
    "adapters were applied after each transformer block, so the final-layer vision "
    "and language adapters could not reach the action readout and never trained",
    "each task used its own action mean/std, so adapters live in different output "
    "coordinates and small-variance dimensions were clipped by the diffusion head",
    "action evidence was one task-level summary vector with velocity/jerk labels",
    "the alignment autoencoder used a token-wise MLP decoder over the whole sequence",
    "the reconstruction-only control had evidence classification disabled",
)


def _json_member(archive: Any, name: str) -> dict:
    return json.loads(bytes(archive[name]).decode("utf-8"))


def inspect_legacy_adapter(path: str | Path) -> dict[str, Any]:
    """Describe one version-one ``adapter.npz`` without constructing research objects."""

    with np.load(path, allow_pickle=False) as archive:
        spec = _json_member(archive, "adapter_spec_json")
        metadata = _json_member(archive, "packed_adapter_metadata_json")
        tokens = np.asarray(archive["tokens"])
        entry_ids = np.asarray(archive["entry_ids"])
    norms = {}
    for index, entry in enumerate(spec["entries"]):
        norms[entry["parameter_path"]] = float(np.linalg.norm(tokens[entry_ids == index]))
    return {
        "path": str(path),
        "adapter_spec_schema": spec.get("schema_version"),
        "packed_schema": metadata.get("schema_version", 1),
        "task_id": metadata.get("task_id"),
        "seed": metadata.get("seed"),
        "checkpoint_stage": metadata.get("checkpoint_stage"),
        "entries": len(spec["entries"]),
        "tokens": int(tokens.shape[0]),
        "identically_zero_entries": sorted(
            path.rsplit("/", 1)[-1] for path, value in norms.items() if value == 0.0
        ),
    }


def inspect_legacy_tree(root: str | Path) -> dict[str, Any]:
    """Summarize every legacy artifact below ``root`` (never modifies anything)."""

    base = Path(root)
    adapters = []
    for path in sorted(base.rglob("adapter.npz")):
        if "experiments" in path.relative_to(base).parts[:1]:
            continue
        try:
            record = inspect_legacy_adapter(path)
        except Exception as exc:  # legacy files are heterogeneous; report, do not fail
            record = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
        adapters.append(record)
    gates = []
    for path in sorted(base.rglob("alignment_advantage.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        gates.append(
            {
                "path": str(path),
                "held_out_suite": payload.get("held_out_suite"),
                "legacy_gate_passed": payload.get("passed"),
                "macro": payload.get("macro"),
                "validation_tasks": payload.get("task_count"),
            }
        )
    checkpoints = []
    for path in sorted(base.rglob("metadata.json")):
        if "experiments" in path.relative_to(base).parts[:1]:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == 1 and "model" in payload and "history" in payload:
            checkpoints.append(
                {
                    "path": str(path.parent),
                    "parameters_present": (path.parent / "alignment_params.msgpack").is_file(),
                    "contrastive_weight": payload["model"].get("contrastive_weight"),
                    "classification_weight": payload["model"].get("classification_weight"),
                    "epochs": len(payload["history"]),
                }
            )
    evidence = [
        str(path)
        for path in sorted(base.rglob("*.npz"))
        if path.name != "adapter.npz" and "evidence" in path.parts and "experiments" not in path.parts
    ]
    zero = sorted({name for record in adapters for name in record.get("identically_zero_entries", [])})
    return {
        "root": str(base),
        "status": "legacy version-one artifacts: inspection only, rejected by corrected commands",
        "why_legacy": list(LEGACY_REASONS),
        "adapter_samples": len(adapters),
        "adapters_with_dead_entries": sum(bool(r.get("identically_zero_entries")) for r in adapters),
        "entries_identically_zero_in_some_sample": zero,
        "evidence_files": len(evidence),
        "alignment_checkpoints": checkpoints,
        "alignment_gate_reports": gates,
        "adapters": adapters,
    }
