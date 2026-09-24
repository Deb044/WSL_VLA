#!/usr/bin/env python3
"""
scripts/download_libero_subset.py

Downloads a real subset of the LIBERO robotics demonstration dataset from Hugging Face.
Usage:
    python scripts/download_libero_subset.py --suite libero_spatial --max_tasks 3
"""
import os
import argparse
import shutil
import yaml
from huggingface_hub import hf_hub_download, HfApi

REPO_ID = "yifengzhu-hf/LIBERO-datasets"

def download_subset(dest_dir: str = "./data/libero", suite: str = "libero_spatial", max_tasks: int = 3):
    os.makedirs(dest_dir, exist_ok=True)
    print("=" * 70)
    print(f" Downloading real LIBERO demonstrations (Suite: {suite}, Max tasks: {max_tasks})")
    print(f" Target Directory: {os.path.abspath(dest_dir)}")
    print("=" * 70)

    # Load tasks configuration
    with open("configs/tasks_config.yaml", "r") as f:
        tasks_cfg = yaml.safe_load(f)

    if suite not in tasks_cfg["suites"]:
        print(f"[ERROR] Suite '{suite}' not found in configs/tasks_config.yaml.")
        return

    suite_tasks = tasks_cfg["suites"][suite]["tasks"][:max_tasks]

    api = HfApi()
    all_repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

    downloaded = 0
    for task_info in suite_tasks:
        task_id = task_info["id"]
        task_name = task_info["name"]
        target_dest = os.path.join(dest_dir, f"{task_id}.hdf5")

        if os.path.exists(target_dest) and os.path.getsize(target_dest) > 1024 * 1024:
            print(f"  [EXISTS] {task_id}.hdf5 ({os.path.getsize(target_dest) / (1024*1024):.1f} MB). Skipping download.")
            downloaded += 1
            continue

        # Look for matching file in repo
        matched_file = None
        for rf in all_repo_files:
            if suite in rf and task_name in rf and rf.endswith(".hdf5"):
                matched_file = rf
                break

        if matched_file:
            print(f"  [DOWNLOADING] {task_id} ({task_name[:40]}...) from Hugging Face...")
            cached_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=matched_file,
                repo_type="dataset",
            )
            # Copy to target destination
            shutil.copyfile(cached_path, target_dest)
            size_mb = os.path.getsize(target_dest) / (1024 * 1024)
            print(f"    -> Saved to {target_dest} ({size_mb:.1f} MB)")
            downloaded += 1
        else:
            print(f"  [WARN] No matching remote file found for {task_id}")

    print("=" * 70)
    print(f" Finished downloading {downloaded}/{len(suite_tasks)} task demonstrations.")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download real LIBERO demonstration subset")
    parser.add_argument("--dest_dir", type=str, default="./data/libero")
    parser.add_argument("--suite", type=str, default="libero_spatial", help="libero_spatial, libero_object, libero_goal, libero_10")
    parser.add_argument("--max_tasks", type=int, default=2, help="Number of task HDF5 files to download")
    args = parser.parse_args()

    download_subset(dest_dir=args.dest_dir, suite=args.suite, max_tasks=args.max_tasks)
