#!/usr/bin/env python3
"""Assemble one fold's alignment archive from verified zoo samples and evidence.

The fold, its task splits, and every expected sample count come from the
experiment manifest. Adapter and evidence hashes are re-checked here, the
held-out suite is excluded, and the result is a directory of memory-mapped
token shards with task evidence stored once per task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.alignment.archive import build_alignment_archive, open_alignment_archive
from wsl_vla.experiments.runtime import add_experiment_arguments, load_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser, checkpoint=False)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--samples-per-shard", type=int, default=32)
    args = parser.parse_args()

    _, _, manifest, paths = load_experiment(args)
    fold = manifest.fold(args.fold)
    output = paths.archive(fold.name)
    already_built = (output / "archive_manifest.json").is_file()
    # Building twice from the same inputs returns the existing archive; an
    # archive built from different inputs is refused, never overwritten.
    archive_manifest = build_alignment_archive(
        manifest, paths, fold, output, samples_per_shard=args.samples_per_shard
    )
    # Re-open through the reader so the written index, hashes, and leakage
    # rules are verified before any trainer depends on them.
    archive = open_alignment_archive(output)
    summary = {
        "ok": True,
        "already_complete": already_built,
        "output": str(output),
        "fold": fold.name,
        "held_out_suite": fold.held_out_suite,
        "integrity_only": archive.integrity_only,
        "split_counts": archive_manifest["split_counts"],
        "train_tasks": list(archive.train_task_ids),
        "validation_tasks": [archive.task_ids[row] for row in archive.task_rows("validation")],
        "tokens_per_sample": archive.layout.token_count,
        "sequence_windows_per_sample": archive.layout.window_count,
        "excluded_held_out_tasks": archive_manifest["excluded_held_out_tasks"],
        "archive_manifest_sha256": archive.manifest_sha256,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
