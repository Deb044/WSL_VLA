#!/usr/bin/env python3
"""Create a complete immutable manifest before launching a research run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.contracts import RunManifest
from wsl_vla.data.libero import load_suite_manifest
from wsl_vla.experiments.protocol import load_yaml, validate_reference_tasks, validate_research_config
from wsl_vla.experiments.provenance import (
    capture_environment,
    capture_hardware,
    git_state,
    sha256_file,
    write_manifest_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.base_sha256) != 64:
        raise ValueError("base-sha256 must be the verified Octo parameter-tree digest")
    config = load_yaml(args.config)
    tasks = load_yaml(args.tasks)
    validate_research_config(config)
    validate_reference_tasks(tasks)
    if args.suite not in tasks["suites"]:
        raise ValueError(f"unknown reference suite: {args.suite}")
    data_root = Path(args.data_root or config["data"]["root"])
    dataset_files = load_suite_manifest(
        data_root / args.suite, tasks["suites"][args.suite]
    )
    commit, dirty = git_state(REPO_ROOT)
    manifest = RunManifest(
        run_id=args.run_id,
        command=sys.argv,
        git_commit=commit,
        git_dirty=dirty,
        base_model_id=config["model"]["id"],
        base_revision=config["model"]["revision"],
        base_sha256=args.base_sha256,
        dataset_sha256={str(path.resolve()): sha256_file(path) for path in dataset_files},
        task_order=tasks["suites"][args.suite],
        seeds=config["evaluation"]["training_seeds"],
        configuration=config,
        hardware=capture_hardware(),
        environment=capture_environment(
            ("jax", "jaxlib", "flax", "optax", "octo", "libero", "numpy", "h5py")
        ),
    )
    write_manifest_atomic(manifest, args.output)
    print(json.dumps({"ok": True, "output": str(Path(args.output).resolve()), "git_dirty": dirty}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
