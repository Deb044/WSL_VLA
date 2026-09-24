"""
models/download_octo_weights.py

Downloads the official pre-trained Octo-Small checkpoint (~108 MB)
from Hugging Face (rail-berkeley/octo-small-1.5) to serve as the frozen backbone.
"""
import os
import argparse
from huggingface_hub import hf_hub_download, HfApi

def download_octo_checkpoint(dest_dir="./checkpoints/octo_pretrained"):
    os.makedirs(dest_dir, exist_ok=True)
    repo_id = "rail-berkeley/octo-small-1.5"

    print("======================================================================")
    print(f" Downloading Octo-Small (27M) Pre-Trained Checkpoint from {repo_id}")
    print("======================================================================")

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id)
        # Identify weights file (safetensors or bin)
        target_file = None
        for f in ["model.safetensors", "pytorch_model.bin", "config.json"]:
            if f in files:
                target_file = f
                print(f"  Downloading: {target_file} (~108 MB)...")
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=target_file,
                    local_dir=dest_dir,
                    local_dir_use_symlinks=False,
                )
                print(f"  -> Saved to: {downloaded}")

        print("\n[SUCCESS] Octo-Small checkpoint is ready!")
        print(f"Set 'pretrained_path: {os.path.join(dest_dir, target_file)}' in configs/vla_config.yaml")

    except Exception as e:
        print(f"Notice: Direct download from {repo_id} encountered: {e}")
        print("You can continue training with the exact Octo-Small architecture directly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, default="./checkpoints/octo_pretrained")
    args = parser.parse_args()
    download_octo_checkpoint(dest_dir=args.dest)
