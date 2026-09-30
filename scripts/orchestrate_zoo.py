#!/usr/bin/env python3
"""
scripts/orchestrate_zoo.py

Legacy PyTorch proxy smoke-test orchestrator. Not valid for research results.
Designed to operate strictly within tight storage limits (e.g. 14 GB).
Supports HF_TOKEN via environment variable, CLI argument, or .env file to bypass rate limits.

Workflow for each suite in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
  1. Download the suite's demonstration HDF5 files (~3.5 to 5 GB).
  2. Train the 10 tasks with the proxy model and factorized LoRA.
  3. Serialize smoke checkpoints to ./smoke_results/model_zoo/task_*.pt.
  4. Clean up the raw HDF5 files for that suite to reclaim disk space.
  5. Repeat for the next suite.
  6. Final integrity verification confirming 40 valid checkpoints.
"""
import os
import sys
import shutil
import argparse
import subprocess
from typing import Optional

# Ensure repo root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wsl_vla.smoke_guard import require_explicit_smoke_test, write_smoke_marker


SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def get_hf_token(token_arg: Optional[str] = None) -> Optional[str]:
    """Retrieves Hugging Face token from CLI argument, environment variable, or .env file."""
    if token_arg:
        return token_arg
    if os.environ.get("HF_TOKEN"):
        return os.environ.get("HF_TOKEN")
    if os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return os.environ.get("HUGGING_FACE_HUB_TOKEN")

    for env_path in [".env", "../.env"]:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def run_command(cmd, desc, env=None):
    print(f"\n======================================================================")
    print(f" {desc}")
    print(f" Executing: {' '.join(cmd)}")
    print(f"======================================================================")
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f"[ERROR] Failed during: {desc} (Exit code {res.returncode})")
        return False
    return True


def orchestrate_zoo(
    python_bin: str = sys.executable,
    steps_per_task: int = 2000,
    keep_datasets: bool = False,
    start_suite: str = None,
    hf_token: Optional[str] = None,
    smoke_test: bool = False,
):
    require_explicit_smoke_test(smoke_test, output_directories=("smoke_results/model_zoo",))
    token = get_hf_token(hf_token)
    run_env = os.environ.copy()
    if token:
        run_env["HF_TOKEN"] = token
        run_env["HUGGING_FACE_HUB_TOKEN"] = token
        print("  [Auth] HF_TOKEN detected: Rate limits removed across all Hugging Face requests.")

    print("======================================================================")
    print(" Starting 40-Task Model Zoo Suite-by-Suite Construction Pipeline")
    print(" Target Model: LEGACY PYTORCH PROXY (NOT OFFICIAL OCTO; SMOKE TEST ONLY)")
    print(f" Steps per task: {steps_per_task}")
    print(f" Storage Strategy: Clean raw HDF5 after each suite to conserve disk")
    print("======================================================================")

    data_dir = "./data/libero"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs("./smoke_results/model_zoo", exist_ok=True)
    write_smoke_marker("./smoke_results/model_zoo", command="scripts/orchestrate_zoo.py")

    suite_list = list(SUITES)
    if start_suite and start_suite in suite_list:
        idx = suite_list.index(start_suite)
        suite_list = suite_list[idx:]

    for suite_idx, suite in enumerate(suite_list, 1):
        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f" [SUITE {suite_idx}/{len(suite_list)}] Processing: '{suite}'")
        print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        suite_data_path = os.path.join(data_dir, suite)

        # 1. Download suite demonstrations
        dl_cmd = [
            python_bin,
            "scripts/download_libero.py",
            "--suite", suite,
            "--dest", data_dir,
        ]
        if token:
            dl_cmd.extend(["--token", token])

        if not run_command(dl_cmd, f"Downloading dataset for {suite}", env=run_env):
            print(f"[WARNING] Download script exited with non-zero code. Attempting to proceed with available data...")

        # 2. Train the 10 tasks in this suite
        train_cmd = [
            python_bin,
            "scripts/train_zoo.py",
            "--suite", suite,
            "--max_steps", str(steps_per_task),
            "--smoke-test",
        ]
        if not run_command(train_cmd, f"Training 10 tasks for {suite}", env=run_env):
            print(f"[ERROR] Training failed on suite {suite}. Halting.")
            sys.exit(1)

        # 3. Reclaim disk space by deleting raw HDF5 files for this suite
        if not keep_datasets:
            print(f"\n  [Reclaiming Disk Space] Removing raw HDF5 files from: {suite_data_path}...")
            if os.path.exists(suite_data_path):
                shutil.rmtree(suite_data_path, ignore_errors=True)
            print(f"  [Disk Space Reclaimed] Finished cleanup for {suite}.")
        else:
            print(f"  [Notice] Keeping raw HDF5 dataset on disk.")

    # 4. Final Verification
    print("\n======================================================================")
    print(" Running Final Model Zoo Verification on All 40 Tasks")
    print("======================================================================")
    verify_cmd = [
        python_bin,
        "scripts/verify_zoo.py",
        "--dir", "./smoke_results/model_zoo",
        "--expected", "40",
        "--smoke-test",
    ]
    run_command(verify_cmd, "Final Population Verification", env=run_env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000, help="Optimizer steps per task (e.g. 2000)")
    parser.add_argument("--keep_data", action="store_true", help="Keep raw HDF5 datasets after training")
    parser.add_argument("--start_from", type=str, default=None, help="Resume from a specific suite")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (or set HF_TOKEN env var)")
    parser.add_argument("--smoke-test", action="store_true", help="Acknowledge this is the legacy PyTorch proxy")
    args = parser.parse_args()

    orchestrate_zoo(
        python_bin=sys.executable,
        steps_per_task=args.steps,
        keep_datasets=args.keep_data,
        start_suite=args.start_from,
        hf_token=args.token,
        smoke_test=args.smoke_test,
    )
