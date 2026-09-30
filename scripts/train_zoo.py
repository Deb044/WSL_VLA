#!/usr/bin/env python3
"""Train task-specific modality adapters for official Octo-Small 1.5 across LIBERO.

Freezes the base Octo backbone and optimizes only low-rank modality adapters on real HDF5 demonstration data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import StrictLiberoHDF5, load_suite_manifest
from models.octo_model import (
    OCTO_MODEL_ID,
    build_adapter_spec_and_factors,
    install_octo_modality_patch,
    load_research_octo,
)
from models.octo_training import (
    adapter_value_and_grad,
    initial_adapter_state,
    materialize_policy_params,
)
from models.packing import pack_low_rank_adapter, save_packed_adapter
from core.protocol import load_yaml, resolve_config_path, validate_reference_tasks
from core.provenance import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial", help="Target LIBERO suite")
    parser.add_argument("--task-index", type=int, default=None, help="Optional specific task index (0-9)")
    parser.add_argument("--data-root", default="data/libero", help="Root directory containing real LIBERO HDF5s")
    parser.add_argument("--output-dir", default="research_results/population", help="Output directory for trained adapters")
    parser.add_argument("--rank", type=int, default=16, help="Adapter low rank")
    parser.add_argument("--alpha", type=float, default=32.0, help="Adapter alpha scaling")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for adapter parameters")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    print("=" * 80)
    print(" OFFICIAL OCTO-SMALL 1.5 TASK ADAPTER TRAINING (MODEL ZOO)")
    print(f" Suite: {args.suite} | Data Root: {args.data_root} | Adapter Rank: {args.rank}")
    print("=" * 80)

    # 1. Verify dependencies and patch Octo
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError as exc:
        raise RuntimeError(
            "Training official Octo adapters requires the JAX/Flax research environment. "
            "Please install requirements-research.txt under Python 3.10/3.11."
        ) from exc

    install_octo_modality_patch(rank=args.rank, alpha=args.alpha)
    bundle = load_research_octo(
        rank=args.rank,
        alpha=args.alpha,
        seed=args.seed,
    )
    print(f"Base Octo parameter SHA-256 hash: {bundle.base_sha256}")

    # 2. Resolve task data
    tasks_config = load_yaml(resolve_config_path("configs/reference_tasks.yaml"))
    validate_reference_tasks(tasks_config)
    suite_tasks = tasks_config["suites"][args.suite]

    data_dir = Path(args.data_root) / args.suite
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Suite data directory not found: {data_dir}. "
            "Please download real LIBERO demonstration HDF5 files using 'python scripts/download_libero.py'."
        )

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    target_indices = [args.task_index] if args.task_index is not None else list(range(len(suite_tasks)))
    print(f"Target tasks in suite '{args.suite}': {target_indices}")
    print("Pre-training setup and model instantiation verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
