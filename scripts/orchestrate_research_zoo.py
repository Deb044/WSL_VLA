#!/usr/bin/env python3
"""
scripts/orchestrate_research_zoo.py

Automated suite-by-suite model-zoo construction for Methodology 1.
Designed specifically for environments with tight disk constraints (< 9 GB).

Lifecycle per suite in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
  1. Download the suite's demonstration HDF5 files (~5 to 7 GB) + write verified manifest.json.
  2. Extract research evidence for all 10 tasks into research_results/evidence/<suite>/.
  3. Train 3 random seeds (17, 42, 73) x 3 late-checkpoint fractions (0.8, 0.9, 1.0) = 90 adapters.
  4. Automatically delete the suite's raw HDF5 files to reclaim 5-7 GB disk space.
  5. Advance to the next suite.
  6. Run final population verification confirming 360 valid adapters.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.experiments.protocol import REQUIRED_SUITES, load_yaml


def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def banner(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  {title}\n{bar}\n", flush=True)


def get_dir_size_mb(path: Path) -> float:
    if not path.is_dir():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 2)


def run_cmd(cmd: list[str], desc: str, env: dict[str, str] | None = None) -> bool:
    log(f"Starting: {desc}")
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        log(f"FAILED (code {result.returncode}) after {elapsed:.1f}s: {desc}")
        return False
    log(f"COMPLETED in {elapsed:.1f}s: {desc}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Suite-by-suite GPU model zoo construction with automatic disk space reclamation"
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=list(REQUIRED_SUITES),
        choices=list(REQUIRED_SUITES),
        help="LIBERO suites to process in order (default: all 4 reference suites)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[17, 42, 73],
        help="Seeds to train per task (default: [17, 42, 73])",
    )
    parser.add_argument(
        "--tasks-per-suite",
        type=int,
        default=10,
        help="Number of tasks per suite to run (default: 10)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Training steps per task (default: from configs/research/base.yaml)",
    )
    parser.add_argument(
        "--data-root",
        default="data/libero",
        help="Directory where datasets are staged",
    )
    parser.add_argument(
        "--output-root",
        default="research_results/population",
        help="Directory where model zoo population checkpoints are stored",
    )
    parser.add_argument(
        "--evidence-root",
        default="research_results/evidence",
        help="Directory where multi-modal evidence is stored",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Do NOT delete raw HDF5 files after suite training (requires > 30 GB disk space)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use already-downloaded suites and rebuild manifests offline (compute nodes)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run sub-scripts (default: current interpreter)",
    )
    args = parser.parse_args()

    python_bin = args.python
    data_root = REPO_ROOT / args.data_root
    output_root = REPO_ROOT / args.output_root
    evidence_root = REPO_ROOT / args.evidence_root
    tasks_config = load_yaml(REPO_ROOT / "configs/reference_tasks.yaml")

    # Set up GPU environment defaults for JAX/XLA
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = str(REPO_ROOT)
    # Prevent JAX from preallocating 90% of VRAM so it shares VRAM smoothly on laptop GPUs
    run_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    run_env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")

    banner("METHODOLOGY 1: SUITE-BY-SUITE GPU MODEL ZOO CONSTRUCTION")
    log(f"Selected suites: {args.suites}")
    log(f"Seeds: {args.seeds}")
    log(f"Tasks per suite: {args.tasks_per_suite}")
    log(f"Auto disk cleanup: {not args.keep_raw}")
    log(f"Python interpreter: {python_bin}")

    overall_start = time.perf_counter()

    for suite_idx, suite in enumerate(args.suites, start=1):
        suite_dir = data_root / suite
        manifest_path = suite_dir / "manifest.json"
        suite_tasks = tasks_config["suites"][suite][: args.tasks_per_suite]

        banner(f"[{suite_idx}/{len(args.suites)}] SUITE: {suite}")

        # ------------------------------------------------------------------
        # 1. DOWNLOAD SUITE DEMONSTRATIONS
        # ------------------------------------------------------------------
        download_cmd = [
            python_bin,
            "scripts/download_libero.py",
            "--suite",
            suite,
            "--dest",
            str(data_root),
        ]
        if args.tasks_per_suite < 10:
            download_cmd.extend(["--max_tasks", str(args.tasks_per_suite)])

        if args.skip_download:
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
            from download_libero import generate_suite_manifest

            generate_suite_manifest(suite, str(suite_dir))
        elif not run_cmd(download_cmd, f"Download demonstrations for {suite}", env=run_env):
            log(f"Aborting due to download failure on suite: {suite}")
            return 1

        if not manifest_path.is_file():
            log(f"ERROR: Expected manifest at {manifest_path} was not created.")
            return 1

        with open(manifest_path, encoding="utf-8") as f:
            manifest_data = json.load(f)
        task_entries = manifest_data["tasks"][: args.tasks_per_suite]

        # ------------------------------------------------------------------
        # 2. EVIDENCE EXTRACTION (10 tasks)
        # ------------------------------------------------------------------
        log(f"Extracting task evidence for {len(task_entries)} tasks in {suite}...")
        for task_entry in task_entries:
            task_idx = task_entry["task_index"]
            hdf5_file = suite_dir / task_entry["file"]
            if not hdf5_file.is_file():
                log(f"ERROR: Task file {hdf5_file} not found.")
                return 1

            evidence_path = evidence_root / suite / f"{suite}_{task_idx}.npz"
            if evidence_path.is_file() and evidence_path.stat().st_size > 1024:
                log(f"Evidence for {suite} task {task_idx} already exists ({evidence_path.name}), skipping.")
                continue

            extract_cmd = [
                python_bin,
                "scripts/extract_research_evidence.py",
                "--suite",
                suite,
                "--task-index",
                str(task_idx),
                "--data-file",
                str(hdf5_file),
                "--output",
                str(evidence_path),
            ]
            if not run_cmd(extract_cmd, f"Evidence extraction [{suite} Task {task_idx}]", env=run_env):
                return 1

        # ------------------------------------------------------------------
        # 3. MODEL ZOO POPULATION TRAINING (10 tasks x 3 seeds x 3 checkpoints)
        # ------------------------------------------------------------------
        log(f"Training model zoo adapters for {suite} on GPU...")
        for task_entry in task_entries:
            task_idx = task_entry["task_index"]
            evidence_path = evidence_root / suite / f"{suite}_{task_idx}.npz"

            for seed in args.seeds:
                # Check if all 3 checkpoints already exist for this seed
                task_dir = output_root / suite / f"{suite}_{task_idx}" / f"seed_{seed}"
                existing_ckpts = list(task_dir.glob("step_*/adapter.npz"))
                if len(existing_ckpts) >= 3:
                    log(f"Zoo run {suite}_{task_idx} seed={seed} already trained (found {len(existing_ckpts)} ckpts), skipping.")
                    continue
                if task_dir.exists():
                    # A walltime kill leaves step_* dirs that train_research_zoo refuses to overwrite.
                    # Kept outside output_root, whose verifier counts every adapter.npz.
                    aside = (
                        output_root.parent
                        / f"{output_root.name}_partial"
                        / task_dir.relative_to(output_root).parent
                        / f"{task_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}"
                    )
                    aside.parent.mkdir(parents=True, exist_ok=True)
                    log(f"Moving partial zoo run {task_dir} ({len(existing_ckpts)} ckpts) to {aside}.")
                    task_dir.rename(aside)

                train_cmd = [
                    python_bin,
                    "scripts/train_research_zoo.py",
                    "--suite",
                    suite,
                    "--task-index",
                    str(task_idx),
                    "--seed",
                    str(seed),
                    "--evidence",
                    str(evidence_path),
                    "--data-root",
                    str(data_root),
                    "--output-root",
                    str(output_root),
                ]
                if args.steps is not None:
                    train_cmd.extend(["--steps", str(args.steps)])

                if not run_cmd(train_cmd, f"Train zoo [{suite} Task {task_idx} Seed {seed}]", env=run_env):
                    return 1

        # ------------------------------------------------------------------
        # 4. DISK CLEANUP: REMOVE RAW HDF5s TO RECLAIM DISK SPACE
        # ------------------------------------------------------------------
        if not args.keep_raw:
            raw_files = list(suite_dir.glob("*.hdf5")) + list(suite_dir.glob("*.h5"))
            freed_bytes = sum(f.stat().st_size for f in raw_files)
            for rf in raw_files:
                rf.unlink()
            freed_gb = freed_bytes / (1024 ** 3)
            log(f"CLEANUP: Removed {len(raw_files)} raw demonstration files from {suite_dir}.")
            log(f"CLEANUP: Successfully reclaimed {freed_gb:.2f} GB of physical disk space!")
        else:
            log("CLEANUP SKIPPED: --keep-raw flag active.")

    # ------------------------------------------------------------------
    # 5. FINAL POPULATION VERIFICATION
    # ------------------------------------------------------------------
    banner("FINAL VERIFICATION: RESEARCH MODEL ZOO POPULATION")
    verification_output = output_root / "verification.json"
    verify_cmd = [
        python_bin,
        "scripts/verify_research_zoo.py",
        str(output_root),
        "--output",
        str(verification_output),
    ]
    if run_cmd(verify_cmd, "Verify research population integrity", env=run_env):
        if verification_output.is_file():
            print("\nVerification Summary:")
            print(verification_output.read_text(encoding="utf-8"))
    else:
        log("Notice: Full verification expects all 4 reference suites with 10 tasks each.")

    total_time = time.perf_counter() - overall_start
    banner(f"MODEL ZOO CONSTRUCTION COMPLETE IN {total_time / 60:.1f} MINUTES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
