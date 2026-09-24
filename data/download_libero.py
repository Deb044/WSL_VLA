"""
data/download_libero.py

Downloads only a single LIBERO suite (or specific tasks) from Hugging Face
(yifengzhu-hf/LIBERO-datasets) to conserve disk space.
Supports HF_TOKEN via environment variable, CLI argument, or .env file to bypass rate limits.
"""
import os
import sys
import argparse
from typing import Optional
from huggingface_hub import HfApi, hf_hub_download


def get_hf_token(token_arg: Optional[str] = None) -> Optional[str]:
    """Retrieves Hugging Face token from CLI argument, environment variable, or .env file."""
    if token_arg:
        return token_arg
    if os.environ.get("HF_TOKEN"):
        return os.environ.get("HF_TOKEN")
    if os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Check for .env file in current or parent directory
    for env_path in [".env", "../.env"]:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def download_libero_subset(
    suite_name: str = "libero_spatial",
    dest_dir: str = "./data/libero",
    max_tasks: int = None,
    hf_token: Optional[str] = None,
):
    token = get_hf_token(hf_token)
    if token:
        print("  [Auth] Hugging Face Token detected: Authenticated requests active (no rate limits).")
    else:
        print("  [Notice] Running unauthenticated. Provide --token or set HF_TOKEN to avoid rate limits.")

    repo_id = "yifengzhu-hf/LIBERO-datasets"
    target_suite_dir = os.path.join(dest_dir, suite_name)
    os.makedirs(target_suite_dir, exist_ok=True)

    print("======================================================================")
    print(f" Fetching file list for suite: '{suite_name}' from {repo_id}")
    print("======================================================================")

    api = HfApi(token=token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    # Filter files belonging to requested suite
    suite_files = sorted([
        f for f in repo_files
        if f.startswith(f"{suite_name}/") and f.endswith(".hdf5")
    ])

    if not suite_files:
        suite_files = sorted([
            f for f in repo_files
            if suite_name in f and f.endswith(".hdf5")
        ])

    print(f"  Found {len(suite_files)} task files in '{suite_name}'.")

    if max_tasks is not None:
        suite_files = suite_files[:max_tasks]
        print(f"  [Disk Saver Mode] Restricted to first {max_tasks} task(s) (~{max_tasks * 300} MB).")
    else:
        print(f"  Downloading all {len(suite_files)} tasks in {suite_name} (~3.2 GB total).")

    for idx, remote_path in enumerate(suite_files):
        filename = os.path.basename(remote_path)
        local_path = os.path.join(target_suite_dir, filename)

        if os.path.exists(local_path) and os.path.getsize(local_path) > 1024 * 1024:
            print(f"  [{idx+1}/{len(suite_files)}] Already downloaded: {filename} ({os.path.getsize(local_path) / (1024**2):.1f} MB)")
            continue

        print(f"  [{idx+1}/{len(suite_files)}] Downloading: {filename}...")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=remote_path,
                local_dir=dest_dir,
                local_dir_use_symlinks=False,
                token=token,
            )
            file_size_mb = os.path.getsize(downloaded) / (1024 ** 2)
            print(f"       -> Saved ({file_size_mb:.1f} MB)")
        except Exception as e:
            print(f"       -> Failed to download {filename}: {e}")

    print("\n======================================================================")
    print(f" [SUCCESS] Dataset staging complete! Files saved in: {target_suite_dir}")
    print("======================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download single LIBERO suite or subset")
    parser.add_argument("--suite", type=str, default="libero_spatial", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--dest", type=str, default="./data/libero")
    parser.add_argument("--max_tasks", type=int, default=None, help="Limit to N tasks to minimize disk usage")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    download_libero_subset(
        suite_name=args.suite,
        dest_dir=args.dest,
        max_tasks=args.max_tasks,
        hf_token=args.token,
    )
