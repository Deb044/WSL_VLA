#!/usr/bin/env python3
"""Fill the Hugging Face cache with everything research jobs read from the hub.

Cluster jobs run with ``HF_HUB_OFFLINE=1`` and compute nodes may have no
internet, so run this once on a login node, with the ``HF_HOME`` the jobs use:

    HF_HOME=/scratch/$USER/hf_cache python scripts/setup/prefetch_research_cache.py

It downloads the pinned Octo-Small 1.5 checkpoint and the tokenizer and model
configuration that Octo loads by name (``t5-base``; no weights, they are part
of the checkpoint), then checks in a separate process that all of it loads
with the hub switched off. It needs huggingface_hub and transformers only, so
JAX and a GPU are not required. The LIBERO demonstrations are fetched
separately (``scripts/setup/download_libero.py``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.octo.bridge import OCTO_MODEL_ID, OCTO_MODEL_REVISION


def hub_model_names(checkpoint_config: dict) -> list[str]:
    """Hub models the checkpoint configuration loads by name (e.g. t5-base)."""

    names = {checkpoint_config["text_processor"]["kwargs"]["tokenizer_name"]}
    for tokenizer in checkpoint_config["model"]["task_tokenizers"].values():
        encoder = tokenizer.get("kwargs", {}).get("encoder")
        if encoder:
            names.add(encoder)
    return sorted(names)


def fetch(checkpoint: str | None) -> dict:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    if checkpoint is None:
        checkpoint = snapshot_download(repo_id=OCTO_MODEL_ID, revision=OCTO_MODEL_REVISION)
    config = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
    names = hub_model_names(config)
    for name in names:
        AutoConfig.from_pretrained(name)
        AutoTokenizer.from_pretrained(name)
    return {"checkpoint": str(checkpoint), "hub_models": names}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-offline", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_offline:
        print(json.dumps(fetch(None)))
        return 0

    fetched = fetch(None)
    # Load everything again in a fresh process with the hub switched off,
    # exactly as the jobs will.
    offline = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    check = subprocess.run(
        [sys.executable, __file__, "--check-offline"],
        env=offline, capture_output=True, text=True,
    )
    report = {
        "ok": check.returncode == 0,
        "hf_home": os.environ.get("HF_HOME", "(default: ~/.cache/huggingface)"),
        "octo_checkpoint": {"id": OCTO_MODEL_ID, "revision": OCTO_MODEL_REVISION, "path": fetched["checkpoint"]},
        "hub_models_loaded_by_name": fetched["hub_models"],
    }
    if check.returncode != 0:
        report["offline_check_error"] = check.stderr.strip().splitlines()[-5:]
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
