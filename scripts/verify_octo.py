#!/usr/bin/env python3
"""Confirm that the patched Octo architecture preserves official output exactly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.octo_model import (
    OCTO_MODEL_ID,
    assert_zero_adapter_equivalence,
    install_octo_modality_patch,
    load_research_octo,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=f"hf://{OCTO_MODEL_ID}")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    install_octo_modality_patch(rank=args.rank, alpha=args.alpha)
    bundle = load_research_octo(
        checkpoint=args.checkpoint,
        rank=args.rank,
        alpha=args.alpha,
        seed=args.seed,
    )
    assert_zero_adapter_equivalence(bundle, atol=args.tolerance)
    print(f"PASS: zero-adapter equivalence confirmed (tolerance: {args.tolerance:.2e})")
    print(f"Base parameter SHA-256 hash: {bundle.base_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
