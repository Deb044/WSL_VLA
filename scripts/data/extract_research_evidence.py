#!/usr/bin/env python3
"""Cache frozen-Octo evidence for one task of an experiment manifest.

Only the task's *training* episodes (manifest episode split) are read. The
output stores frozen visual and language features tagged with episode ids, and
per-episode command-chunk summaries; nothing is trained, pooled over episodes,
or standardized here - that happens per fold during alignment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.alignment.evidence_io import save_task_evidence
from wsl_vla.alignment.octo_evidence import extract_task_evidence
from wsl_vla.data.streams import load_split_episodes, task_stream_from_bundle
from wsl_vla.experiments.manifest import task_id as make_task_id
from wsl_vla.experiments.provenance import sha256_file
from wsl_vla.experiments.runtime import (
    add_experiment_arguments,
    environment_record,
    input_hashes,
    load_experiment,
    require_gpu_runtime_settings,
    task_data_file,
)
from wsl_vla.octo.bridge import load_research_octo_from_config
from wsl_vla.octo.zoo import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    require_gpu_runtime_settings()

    config, tasks, manifest, paths = load_experiment(args)
    identifier = make_task_id(args.suite, args.task_index)
    if identifier not in manifest.tasks:
        raise ValueError(f"{identifier} is not a task of experiment {manifest.name!r}")
    output = paths.evidence_file(identifier)
    sidecar = output.with_suffix(".manifest.json")
    environment = environment_record()
    data_file, instruction = task_data_file(
        args.data_root or config["data"]["root"], tasks, identifier
    )
    dataset_sha256 = sha256_file(data_file)
    if output.exists():
        # The evidence file is published only after its sidecar, so an existing
        # file is a finished extraction: reuse it if it is this request's.
        recorded = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
        request = {
            "evidence_sha256": sha256_file(output),
            "evidence_identity": manifest.evidence_identity(),
            "dataset_sha256": dataset_sha256,
            "task_id": identifier,
        }
        changed = sorted(key for key, value in request.items() if recorded.get(key) != value)
        if changed:
            raise FileExistsError(
                f"{output} was extracted with different {changed}; evidence is never overwritten"
            )
        print(json.dumps({"ok": True, "already_complete": str(output)}, indent=2))
        return 0
    # Evidence is adapter-independent; the bundle seed only names unused
    # zero-initialized adapters.
    bundle = load_research_octo_from_config(config, args.checkpoint, seed=0)
    split, episodes = load_split_episodes(
        data_file,
        preprocessing=bundle.preprocessing,
        split_seed=manifest.episode_split.seed,
        train_fraction=manifest.episode_split.train_fraction,
        validation_fraction=manifest.episode_split.validation_fraction,
        splits=("train",),
    )
    stream = task_stream_from_bundle(
        bundle, task_id=identifier, instruction=instruction, episodes=episodes["train"]
    )
    evidence = extract_task_evidence(
        bundle,
        stream,
        suite=args.suite,
        dataset_sha256=dataset_sha256,
        frames_per_episode=int(manifest.evidence["frames_per_episode"]),
        action_chunk_length=int(manifest.evidence["action_chunk_length"]),
        batch_size=args.batch_size,
    )
    if evidence.episode_ids != split.train:
        raise AssertionError("evidence episodes differ from the manifest training split")
    staged = output.with_name(f".{output.name}.partial")
    save_task_evidence(evidence, staged)
    atomic_write_json(
        sidecar,
        {
            "schema_version": evidence.schema_version,
            "task_id": identifier,
            "suite": args.suite,
            "instruction": instruction,
            "evidence_sha256": sha256_file(staged),
            "evidence_identity": manifest.evidence_identity(),
            "experiment": manifest.name,
            "dataset_sha256": dataset_sha256,
            "train_episode_ids": list(split.train),
            "excluded_validation_episode_ids": list(split.validation),
            "excluded_test_episode_ids": list(split.test),
            "bundle": bundle.identity(),
            "extraction": dict(evidence.extraction),
            "shapes": {
                "vision_features": list(evidence.vision_features.shape),
                "language_features": list(evidence.language_features.shape),
                "action_chunk_features": list(evidence.action_chunk_features.shape),
                "valid_action_chunks": int(evidence.action_chunk_mask.sum()),
                "valid_language_tokens": int(evidence.language_mask.sum()),
            },
            "environment": environment,
            "inputs": input_hashes(args),
            "command": list(sys.argv),
        },
    )
    os.replace(staged, output)
    print(json.dumps({"ok": True, "output": str(output), "manifest": str(sidecar)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
