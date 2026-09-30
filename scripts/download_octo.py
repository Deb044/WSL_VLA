#!/usr/bin/env python3
"""
scripts/download_octo.py

Downloads the official pre-trained Octo-Small 1.5 checkpoint (~108 MB)
from Hugging Face (rail-berkeley/octo-small-1.5) to serve as the frozen backbone.
Supports HF_TOKEN via environment variable, CLI argument, or .env file.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.octo_model import OCTO_MODEL_ID, OCTO_MODEL_REVISION


def get_hf_token(token_arg: Optional[str] = None) -> Optional[str]:
    """Retrieves Hugging Face token from CLI argument, environment variable, or .env file."""
    if token_arg:
        return token_arg
    if os.environ.get("HF_TOKEN"):
        return os.environ.get("HF_TOKEN")
    if os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return os.environ.get("HUGGING_FACE_HUB_TOKEN")

    for env_path in [".env", "../.env"]:
        candidate = Path(env_path)
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def download_octo_checkpoint(
    dest_dir: str = "./checkpoints/octo_pretrained",
    token: Optional[str] = None,
    revision: str = OCTO_MODEL_REVISION,
) -> str:
    """Download official Octo-Small 1.5 checkpoint repository snapshot."""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    hf_tok = get_hf_token(token)

    print("=" * 70)
    print(f" Downloading Official Octo-Small 1.5 Checkpoint from {OCTO_MODEL_ID}")
    print(f" Revision: {revision}")
    if hf_tok:
        print("  [Auth] Authenticated request active via HF_TOKEN.")
    print("=" * 70)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download Octo weights") from exc

    downloaded = snapshot_download(
        repo_id=OCTO_MODEL_ID,
        revision=revision,
        local_dir=str(dest_path),
        local_dir_use_symlinks=False,
        token=hf_tok,
    )
    print(f"\n[SUCCESS] Official Octo-Small 1.5 checkpoint downloaded to: {downloaded}")
    return str(downloaded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download official pre-trained Octo-Small 1.5 weights")
    parser.add_argument("--dest", type=str, default="./checkpoints/octo_pretrained", help="Destination directory")
    parser.add_argument("--revision", type=str, default=OCTO_MODEL_REVISION, help="Pinned model revision")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (or set HF_TOKEN env var)")
    args = parser.parse_args()
    download_octo_checkpoint(dest_dir=args.dest, token=args.token, revision=args.revision)
