#!/usr/bin/env python3
"""
scripts/download_libero.py

Downloads LIBERO benchmark demonstration datasets from Hugging Face
(yifengzhu-hf/LIBERO-datasets).
Supports filtering by suite (libero_spatial, libero_object, libero_goal, libero_10, all),
restricting to a max number of tasks per suite, and Hugging Face token authentication.
"""
import os
import sys
import argparse
from typing import Optional, List
import yaml
from huggingface_hub import HfApi, hf_hub_download

ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
REPO_ID = "yifengzhu-hf/LIBERO-datasets"


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


def download_suite(
    suite_name: str,
    dest_dir: str = "./data/libero",
    max_tasks: Optional[int] = None,
    token: Optional[str] = None,
):
    target_suite_dir = os.path.join(dest_dir, suite_name)
    os.makedirs(target_suite_dir, exist_ok=True)
    os.makedirs(dest_dir, exist_ok=True)

    print("=" * 70)
    print(f" Fetching demonstrations for suite: '{suite_name}' from {REPO_ID}")
    print("=" * 70)

    api = HfApi(token=token)
    repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

    # Match files in the repository corresponding to the requested suite
    suite_files = sorted([
        f for f in repo_files
        if f.startswith(f"{suite_name}/") and f.endswith(".hdf5")
    ])
    if not suite_files:
        suite_files = sorted([
            f for f in repo_files
            if suite_name in f and f.endswith(".hdf5")
        ])

    print(f"  Found {len(suite_files)} task demonstration file(s) for '{suite_name}'.")
    if max_tasks is not None:
        suite_files = suite_files[:max_tasks]
        print(f"  [Task Limit] Restricting to first {max_tasks} task(s).")

    for idx, remote_path in enumerate(suite_files):
        filename = os.path.basename(remote_path)
        local_path = os.path.join(target_suite_dir, filename)

        if os.path.exists(local_path) and os.path.getsize(local_path) > 1024 * 1024:
            print(f"  [{idx+1}/{len(suite_files)}] Already downloaded: {filename} ({os.path.getsize(local_path) / (1024**2):.1f} MB)")
            continue

        print(f"  [{idx+1}/{len(suite_files)}] Downloading: {filename}...")
        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=remote_path,
                local_dir=dest_dir,
                local_dir_use_symlinks=False,
                token=token,
            )
            file_size_mb = os.path.getsize(downloaded) / (1024 ** 2)
            print(f"       -> Saved ({file_size_mb:.1f} MB)")
        except Exception as e:
            print(f"       -> Failed downloading {filename}: {e}")

    print(f" [SUCCESS] Suite '{suite_name}' staging complete in: {target_suite_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Download LIBERO benchmark demonstration datasets")
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
        choices=ALL_SUITES + ["all"],
        help="LIBERO suite to download or 'all'",
    )
    parser.add_argument("--dest", type=str, default="./data/libero", help="Destination directory for datasets")
    parser.add_argument("--max_tasks", type=int, default=None, help="Maximum number of tasks to download per suite")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = get_hf_token(args.token)
    if token:
        print("  [Auth] Authenticated Hugging Face token detected.")
    else:
        print("  [Notice] Running unauthenticated. Provide --token or set HF_TOKEN to avoid rate limits.")

    suites_to_download = ALL_SUITES if args.suite == "all" else [args.suite]
    for s in suites_to_download:
        download_suite(
            suite_name=s,
            dest_dir=args.dest,
            max_tasks=args.max_tasks,
            token=token,
        )


if __name__ == "__main__":
    main()
