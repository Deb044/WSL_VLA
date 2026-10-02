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

from wsl_vla.data.libero import (
    StrictLiberoHDF5,
    forbid_synthetic_research_output,
    load_suite_manifest,
)
from wsl_vla.experiments.protocol import (
    LIBERO_GIT_REVISION,
    expand_population_runs,
    load_yaml,
    publication_folds,
    validate_reference_tasks,
    validate_research_config,
)
from wsl_vla.experiments.provenance import (
    assert_installed_vcs_revision,
    capture_environment,
    capture_environment_lock,
    capture_hardware,
    sha256_file,
)
from wsl_vla.experiments.gamma_selection import validate_gamma_candidate_config
from wsl_vla.octo.bridge import OCTO_GIT_REVISION, OCTO_UPSTREAM_REVISION


RESEARCH_MODULES = ("jax", "flax", "optax", "octo", "libero", "h5py", "yaml")
RESEARCH_DISTRIBUTIONS = (
    "jax", "jaxlib", "flax", "optax", "orbax-checkpoint", "octo", "libero",
    "tensorflow", "transformers", "robosuite", "mujoco", "h5py", "PyYAML",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument(
        "--gamma-candidates", default="configs/research/gamma_candidates.yaml"
    )
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
    gamma_candidates = load_yaml(args.gamma_candidates)
    validate_gamma_candidate_config(gamma_candidates)
    population = expand_population_runs(tasks, config)
    folds = publication_folds()

    problems: list[str] = []
    if sys.version_info[:2] != (3, 11):
        problems.append(f"Python {platform.python_version()} is unsupported; use 3.11")
    if platform.system() != "Linux":
        problems.append(f"host is {platform.system()}, not Linux/WSL2")
    missing = [name for name in RESEARCH_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        problems.append(f"missing research packages: {', '.join(missing)}")
    installed_revisions = {}
    for distribution, expected in (
        ("octo", OCTO_GIT_REVISION),
        ("libero", LIBERO_GIT_REVISION),
    ):
        if distribution in missing:
            continue
        try:
            installed_revisions[distribution] = assert_installed_vcs_revision(
                distribution, expected
            )
        except RuntimeError as exc:
            problems.append(str(exc))
    accelerator = None
    if "jax" not in missing:
        import jax

        accelerator = {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        }
        if accelerator["backend"] != "gpu":
            problems.append(
                f"JAX backend is {accelerator['backend']}, not gpu; Blackwell needs the "
                "CUDA 12.8+ jax build installed by scripts/setup_research_env.sh"
            )
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() not in {"false", "0"}:
        problems.append("set XLA_PYTHON_CLIENT_PREALLOCATE=false for the <=8 GB profile")

    data_root = Path(args.data_root or config["data"]["root"])
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
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    environment_lock_path = output_directory / "environment_lock.json"
    environment_lock_path.write_text(
        json.dumps(capture_environment_lock(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "ok": not problems,
        "problems": problems,
        "octo_git_revision": OCTO_GIT_REVISION,
        "octo_upstream_revision": OCTO_UPSTREAM_REVISION,
        "octo_patches": {
            path.name: sha256_file(path)
            for path in sorted(Path("third_party/octo").glob("*.patch"))
        },
        "accelerator": accelerator,
        "libero_git_revision": LIBERO_GIT_REVISION,
        "installed_vcs_revisions": installed_revisions,
        "expected_population_checkpoints": len(population),
        "folds": [fold.__dict__ for fold in folds],
        "indexed_data": indexed_files,
        "base_checkpoint": base_checkpoint,
        "gamma_candidates": {
            "path": str(Path(args.gamma_candidates).resolve()),
            "sha256": sha256_file(args.gamma_candidates),
            "proposed_count": len(gamma_candidates["proposed"]),
            "uniform_count": len(gamma_candidates["uniform"]),
            "patience_count": len(gamma_candidates["early_stopping_patience"]),
        },
        "inputs": {
            "config_sha256": sha256_file(args.config),
            "tasks_sha256": sha256_file(args.tasks),
            "requirements_research_sha256": sha256_file("requirements-research.txt"),
        },
        "environment_lock": {
            "path": str(environment_lock_path.resolve()),
            "sha256": sha256_file(environment_lock_path),
        },
        "environment": capture_environment(RESEARCH_DISTRIBUTIONS),
        "hardware": capture_hardware(),
    }
    report_path = output_directory / "preflight.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.host_inspection_only:
        return 0
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
