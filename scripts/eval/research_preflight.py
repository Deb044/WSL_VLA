#!/usr/bin/env python3
"""Fail-closed environment and device preflight for the Octo/JAX research pipeline.

Run profile 1. Reports which locked environment profile the interpreter
matches, whether JAX sees a GPU, the pinned Octo/LIBERO revisions, the
experiment manifest's derived sample counts, and which of the manifest's
demonstration files are present and well formed. Missing prerequisites are
listed explicitly; ``ok`` is true only when nothing is missing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.data.libero import StrictLiberoHDF5, forbid_synthetic_research_output
from wsl_vla.experiments.manifest import load_experiment_manifest, split_task_id
from wsl_vla.experiments.protocol import (
    LIBERO_GIT_REVISION,
    load_yaml,
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
from wsl_vla.experiments.runtime import (
    DETERMINISTIC_XLA_FLAG,
    ENVIRONMENT_PACKAGES,
    deterministic_gpu_ops,
    task_data_file,
)
from wsl_vla.octo.compat import ENVIRONMENT_PROFILES, detect_environment_profile


RESEARCH_MODULES = ("jax", "flax", "optax", "octo", "h5py", "yaml")
SIMULATOR_MODULES = ("libero", "robosuite", "mujoco")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="experiment manifest YAML")
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--research-root", default="research_results")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default="research_results/preflight")
    parser.add_argument(
        "--host-inspection-only",
        action="store_true",
        help="Report incompatibilities without failing (exit 0).",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    tasks = load_yaml(args.tasks)
    validate_research_config(config)
    validate_reference_tasks(tasks)
    manifest = load_experiment_manifest(args.manifest, reference_tasks=tasks)

    problems: list[str] = []
    profile = detect_environment_profile()
    if profile["profile"] is None:
        for name, mismatches in profile["mismatches"].items():
            problems.append(f"environment differs from the {name} profile: " + "; ".join(mismatches))
    if platform.system() != "Linux":
        problems.append(f"host is {platform.system()}, not Linux/WSL2")
    missing = [name for name in RESEARCH_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        problems.append(f"missing research packages: {', '.join(missing)}")
    missing_simulator = [
        name for name in SIMULATOR_MODULES if importlib.util.find_spec(name) is None
    ]
    simulator = {"available": not missing_simulator, "missing": missing_simulator}
    # A host without the simulator can still train and align; it just cannot run
    # rollout jobs. That is a stated limitation of the host, not a failed gate.
    limitations: list[str] = []
    if profile["profile"] is not None and "unverified" in str(profile["hardware_status"]):
        # A locked profile that has not passed the integrity run on its target
        # hardware keeps that label in every preflight report.
        limitations.append(
            f"environment profile {profile['profile']!r} is {profile['hardware_status']}; run "
            "the integrity profile on this hardware before relying on results from it"
        )
    if missing_simulator:
        limitations.append(
            "LIBERO simulator stack is incomplete; rollout and camera-orientation jobs "
            "cannot run on this host: " + ", ".join(missing_simulator)
        )
    else:
        try:
            simulator["libero_revision"] = assert_installed_vcs_revision(
                "libero", LIBERO_GIT_REVISION
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
            problems.append(f"JAX backend is {accelerator['backend']}, not gpu")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() not in {"false", "0"}:
        problems.append("set XLA_PYTHON_CLIENT_PREALLOCATE=false before research commands")
    if not deterministic_gpu_ops():
        limitations.append(
            f"XLA_FLAGS lacks {DETERMINISTIC_XLA_FLAG}: GPU kernels may be chosen differently "
            "per process, so identical runs can differ by float rounding and a resumed run is "
            "not bit-identical to an uninterrupted one"
        )

    data_root = Path(args.data_root or config["data"]["root"])
    indexed = []
    source = (
        int(config["preprocessing"]["source_camera_height"]),
        int(config["preprocessing"]["source_camera_width"]),
    )
    for identifier in manifest.tasks:
        try:
            data_file, _ = task_data_file(data_root, tasks, identifier)
            dataset = StrictLiberoHDF5(data_file, require_wrist_camera=True)
            split = dataset.split(
                seed=manifest.episode_split.seed,
                train_fraction=manifest.episode_split.train_fraction,
                validation_fraction=manifest.episode_split.validation_fraction,
            )
            convention = dataset.image_convention
            if convention != config["preprocessing"]["demonstration_image_convention"]:
                problems.append(
                    f"{identifier}: demonstrations use image convention {convention!r}, "
                    "which differs from the preprocessing config"
                )
            indexed.append(
                {
                    "task_id": identifier,
                    "path": str(data_file),
                    "sha256": sha256_file(data_file),
                    "episodes": len(dataset.episode_ids),
                    "image_convention": convention,
                    "split_sizes": {
                        "train": len(split.train),
                        "validation": len(split.validation),
                        "test": len(split.test),
                    },
                }
            )
        except Exception as exc:
            problems.append(f"{identifier}: {exc}")

    forbid_synthetic_research_output(smoke_test=args.smoke_test, output_directory=args.output_dir)
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    environment_lock_path = output_directory / "environment_lock.json"
    environment_lock_path.write_text(
        json.dumps(capture_environment_lock(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    legacy_root = Path(args.research_root)
    legacy = sorted(
        str(path.relative_to(legacy_root))
        for name in ("population", "evidence", "alignment")
        for path in [legacy_root / name]
        if path.is_dir()
    )
    report = {
        "ok": not problems,
        "problems": problems,
        "limitations": limitations,
        "environment_profile": profile,
        "locked_profiles": ENVIRONMENT_PROFILES,
        "accelerator": accelerator,
        "simulator": simulator,
        "experiment": {
            "name": manifest.name,
            "profile": manifest.profile,
            "manifest_sha256": manifest.sha256(),
            "zoo_budget_frozen": manifest.zoo.frozen,
            "expected": manifest.derived_counts(),
            "suites": sorted({split_task_id(i)[0] for i in manifest.tasks}),
        },
        "preprocessing": dict(config["preprocessing"]),
        "configured_source_resolution": list(source),
        "indexed_data": indexed,
        "legacy_version_one_directories": legacy,
        "legacy_note": (
            "Directories listed above hold version-one artifacts. They are inspection-only "
            "(scripts/reporting/inspect_legacy_artifacts.py); corrected experiments write "
            "under research_results/experiments/<name>."
        ),
        "octo_patches": {
            path.name: sha256_file(path)
            for path in sorted((REPO_ROOT / "third_party/octo").glob("*.patch"))
        },
        "inputs": {
            "config_sha256": sha256_file(args.config),
            "tasks_sha256": sha256_file(args.tasks),
            "manifest_file_sha256": sha256_file(args.manifest),
        },
        "environment_lock": {
            "path": str(environment_lock_path.resolve()),
            "sha256": sha256_file(environment_lock_path),
        },
        "environment": capture_environment(ENVIRONMENT_PACKAGES),
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
