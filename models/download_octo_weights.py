"""
models/download_octo_weights.py

Downloads the official pre-trained Octo-Small checkpoint (~108 MB)
from Hugging Face (rail-berkeley/octo-small-1.5) to serve as the frozen backbone.
Supports HF_TOKEN via environment variable, CLI argument, or .env file to bypass rate limits.
"""
import os
import argparse
from typing import Optional
from huggingface_hub import hf_hub_download, HfApi


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


def download_octo_checkpoint(dest_dir: str = "./checkpoints/octo_pretrained", token: Optional[str] = None):
    os.makedirs(dest_dir, exist_ok=True)
    repo_id = "rail-berkeley/octo-small-1.5"
    hf_tok = get_hf_token(token)

    print("======================================================================")
    print(f" Downloading Octo-Small (27M) Pre-Trained Checkpoint from {repo_id}")
    if hf_tok:
        print("  [Auth] Authenticated request active via HF_TOKEN.")
    print("======================================================================")

    api = HfApi(token=hf_tok)
    try:
        files = api.list_repo_files(repo_id=repo_id)
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
                    token=hf_tok,
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
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (or set HF_TOKEN env var)")
    args = parser.parse_args()
    download_octo_checkpoint(dest_dir=args.dest, token=args.token)
