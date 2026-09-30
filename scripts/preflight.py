#!/usr/bin/env python3
"""Fail-closed preflight for the official Octo/JAX research pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import (
    StrictLiberoHDF5,
    forbid_synthetic_research_output,
    load_suite_manifest,
)
from models.octo_model import OCTO_GIT_REVISION
from core.protocol import (
    expand_population_runs,
    load_yaml,
    publication_folds,
    validate_reference_tasks,
    validate_research_config,
)
from core.provenance import capture_environment, capture_hardware, sha256_file


RESEARCH_MODULES = ("jax", "flax", "optax", "octo", "h5py", "yaml")
RESEARCH_DISTRIBUTIONS = ("jax", "flax", "optax", "octo", "h5py", "PyYAML")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default="research_results/preflight")
    parser.add_argument(
        "--host-inspection-only",
        action="store_true",
        help="Report incompatibilities without requiring Linux research dependencies.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    tasks = load_yaml(args.tasks)
    validate_research_config(config)
    validate_reference_tasks(tasks)
    population = expand_population_runs(tasks, config)
    folds = publication_folds()

    problems: list[str] = []
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        problems.append(f"Python {platform.python_version()} is unsupported; use 3.10 or 3.11")
    if platform.system() != "Linux":
        problems.append(f"host is {platform.system()}, not Linux/WSL2")
    missing = [name for name in RESEARCH_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        problems.append(f"missing research packages: {', '.join(missing)}")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() not in {"false", "0"}:
        problems.append("set XLA_PYTHON_CLIENT_PREALLOCATE=false for the <=8 GB profile")

    raw_data_root = Path(args.data_root or config["data"]["root"])
    if not raw_data_root.exists() and (Path(__file__).resolve().parents[1] / raw_data_root).exists():
        data_root = Path(__file__).resolve().parents[1] / raw_data_root
    else:
        data_root = raw_data_root
    indexed_files = []
    for suite in tasks["suites"]:
        suite_dir = data_root / suite
        try:
            files = load_suite_manifest(suite_dir, tasks["suites"][suite])
        except Exception as exc:
            problems.append(str(exc))
            continue
        for path in files:
            try:
                dataset = StrictLiberoHDF5(
                    path,
                    require_wrist_camera=config["data"]["require_wrist_camera"],
                    required_proprio_keys=tuple(config["data"]["proprio_keys"]),
                )
                split = dataset.split(seed=config["data"]["split_seed"])
                indexed_files.append(
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "episodes": len(dataset.episode_ids),
                        "split_sizes": {
                            "train": len(split.train),
                            "validation": len(split.validation),
                            "test": len(split.test),
                        },
                    }
                )
            except Exception as exc:
                problems.append(f"{path}: {exc}")

    base_checkpoint = None
    if args.base_checkpoint:
        base_path = Path(args.base_checkpoint)
        if not base_path.is_file():
            problems.append(f"base checkpoint file is missing: {base_path}")
        else:
            base_checkpoint = {"path": str(base_path), "sha256": sha256_file(base_path)}

    forbid_synthetic_research_output(
        smoke_test=args.smoke_test,
        output_directory=args.output_dir,
    )
    report = {
        "ok": not problems,
        "problems": problems,
        "octo_git_revision": OCTO_GIT_REVISION,
        "expected_population_checkpoints": len(population),
        "folds": [fold.__dict__ for fold in folds],
        "indexed_data": indexed_files,
        "base_checkpoint": base_checkpoint,
        "environment": capture_environment(RESEARCH_DISTRIBUTIONS),
        "hardware": capture_hardware(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.host_inspection_only:
        return 0
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
