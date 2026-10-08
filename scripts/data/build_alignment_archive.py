#!/usr/bin/env python3
"""Assemble validated model-zoo samples into a no-pickle alignment NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.adapters.packing import load_packed_adapter
from wsl_vla.alignment.evidence_io import load_task_evidence
from wsl_vla.experiments.protocol import REQUIRED_SUITES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("population_root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--held-out-suite", required=True, choices=REQUIRED_SUITES)
    parser.add_argument("--validation-task-indices", default="8,9")
    args = parser.parse_args()
    validation_indices = {int(value) for value in args.validation_task_indices.split(",")}
    if not validation_indices or any(index < 0 or index >= 10 for index in validation_indices):
        raise ValueError("validation task indices must lie in [0, 9]")

    sample_dirs = sorted(path.parent for path in Path(args.population_root).rglob("adapter.npz"))
    if not sample_dirs:
        raise FileNotFoundError("no adapter.npz samples found")

    samples = []
    expected_base = None
    expected_spec = None
    task_labels: dict[str, int] = {}
    included_dirs = []
    excluded_held_out = 0
    identities = set()
    for sample_dir in sample_dirs:
        metadata_path = sample_dir / "metadata.json"
        evidence_path = sample_dir / "evidence.npz"
        if not metadata_path.is_file() or not evidence_path.is_file():
            raise FileNotFoundError(f"incomplete population sample: {sample_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1:
            raise ValueError(f"unsupported sample schema in {sample_dir}")
        if metadata.get("synthetic", True):
            raise ValueError(f"synthetic sample is forbidden: {sample_dir}")
        if metadata.get("split") not in {"train", "validation"}:
            raise ValueError(f"sample split must be train or validation: {sample_dir}")
        if metadata.get("suite") == args.held_out_suite:
            excluded_held_out += 1
            continue
        if metadata.get("suite") not in set(REQUIRED_SUITES) - {args.held_out_suite}:
            raise ValueError(f"unknown or invalid training suite: {sample_dir}")
        task_index = int(metadata.get("task_index", -1))
        expected_split = "validation" if task_index in validation_indices else "train"
        if metadata.get("split") != expected_split:
            raise ValueError(
                f"split mismatch in {sample_dir}: task index {task_index} must be {expected_split}"
            )
        identity = (
            metadata.get("suite"),
            task_index,
            int(metadata.get("seed", -1)),
            float(metadata.get("checkpoint_fraction", -1)),
        )
        if identity in identities:
            raise ValueError(f"duplicate population identity {identity}")
        identities.add(identity)
        base = metadata.get("base_sha256")
        if expected_base is None:
            expected_base = base
        if base != expected_base:
            raise ValueError("population contains adapters from different base checkpoints")

        packed = load_packed_adapter(sample_dir / "adapter.npz")
        if (
            packed.metadata.task_id != metadata.get("task_id")
            or packed.metadata.suite != metadata.get("suite")
            or packed.metadata.seed != int(metadata.get("seed", -1))
            or packed.metadata.checkpoint_fraction
            != float(metadata.get("checkpoint_fraction", -1))
        ):
            raise ValueError(f"packed adapter identity differs from sidecar metadata: {sample_dir}")
        spec_digest = hashlib.sha256(
            json.dumps(packed.spec.to_dict(), sort_keys=True).encode()
        ).hexdigest()
        if expected_spec is None:
            expected_spec = spec_digest
        if spec_digest != expected_spec:
            raise ValueError("population contains incompatible adapter specifications")

        evidence_record = load_task_evidence(evidence_path)
        if evidence_record.task_id != metadata.get("task_id"):
            raise ValueError(f"evidence task identity differs in {sample_dir}")
        if evidence_record.suite != metadata.get("suite"):
            raise ValueError(f"evidence suite identity differs in {sample_dir}")
        if evidence_record.vision_mask is None:
            raise ValueError(f"visual evidence mask is required: {sample_dir}")
        evidence = {
            "vision_features": evidence_record.vision_features,
            "vision_mask": evidence_record.vision_mask,
            "language_features": evidence_record.language_features,
            "action_features": evidence_record.action_statistics,
        }
        task_id = metadata["task_id"]
        task_labels.setdefault(task_id, len(task_labels))
        samples.append((packed, evidence, metadata, task_labels[task_id]))
        included_dirs.append(sample_dir)

    max_tokens = max(sample[0].tokens.shape[0] for sample in samples)
    token_width = samples[0][0].tokens.shape[1]
    max_vision = max(sample[1]["vision_features"].shape[0] for sample in samples)
    vision_dim = samples[0][1]["vision_features"].shape[-1]
    n = len(samples)
    if n != 270:
        raise ValueError(f"one alignment fold requires exactly 270 samples, found {n}")
    expected_identities = {
        (suite, task_index, seed, fraction)
        for suite in set(REQUIRED_SUITES) - {args.held_out_suite}
        for task_index in range(10)
        for seed in (17, 42, 73)
        for fraction in (0.8, 0.9, 1.0)
    }
    if identities != expected_identities:
        missing = expected_identities - identities
        extra = identities - expected_identities
        raise ValueError(
            f"population identities differ from locked protocol; missing={len(missing)}, extra={len(extra)}"
        )
    tokens = np.zeros((n, max_tokens, token_width), dtype=np.float32)
    token_mask = np.zeros_like(tokens, dtype=bool)
    component_ids = np.zeros((n, max_tokens), dtype=np.int32)
    layer_ids = np.zeros((n, max_tokens), dtype=np.int32)
    vision_features = np.zeros((n, max_vision, vision_dim), dtype=np.float32)
    vision_mask = np.zeros((n, max_vision), dtype=bool)
    language_features = []
    action_features = []
    labels = []
    splits = []

    for index, (packed, evidence, metadata, label) in enumerate(samples):
        count = packed.tokens.shape[0]
        tokens[index, :count] = packed.tokens
        token_mask[index, :count] = packed.mask
        component_ids[index, :count] = packed.component_ids
        layer_ids[index, :count] = packed.layer_ids
        vision_count = evidence["vision_features"].shape[0]
        vision_features[index, :vision_count] = evidence["vision_features"]
        vision_mask[index, :vision_count] = evidence["vision_mask"]
        language_features.append(evidence["language_features"])
        action_features.append(evidence["action_features"])
        labels.append(label)
        splits.append(metadata["split"].encode())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        tokens=tokens,
        token_mask=token_mask,
        component_ids=component_ids,
        layer_ids=layer_ids,
        vision_features=vision_features,
        vision_mask=vision_mask,
        language_features=np.stack(language_features).astype(np.float32),
        action_features=np.stack(action_features).astype(np.float32),
        task_labels=np.asarray(labels, dtype=np.int32),
        split=np.asarray(splits),
    )
    manifest = {
        "schema_version": 1,
        "base_sha256": expected_base,
        "adapter_spec_sha256": expected_spec,
        "adapter_spec": samples[0][0].spec.to_dict(),
        "held_out_suite": args.held_out_suite,
        "validation_task_indices": sorted(validation_indices),
        "sample_count": n,
        "task_count": len(task_labels),
        "task_labels": task_labels,
        "source_samples": [str(path) for path in included_dirs],
        "excluded_held_out_samples": excluded_held_out,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"ok": True, "output": str(output), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
