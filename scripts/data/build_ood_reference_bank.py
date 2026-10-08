#!/usr/bin/env python3
"""Build a cached, leakage-safe OOD bank from three source-suite model zoos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.alignment.checkpoint import load_alignment_checkpoint
from wsl_vla.evaluation.ood_bank import (
    build_reference_bank_from_population,
    save_reference_bank,
)
from wsl_vla.experiments.protocol import REQUIRED_SUITES
from wsl_vla.experiments.provenance import sha256_directory, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("population_root")
    parser.add_argument("alignment_checkpoint")
    parser.add_argument("--held-out-suite", required=True, choices=REQUIRED_SUITES)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite OOD reference bank: {output}")
    sample_directories = sorted(
        path.parent for path in Path(args.population_root).rglob("adapter.npz")
    )
    if not sample_directories:
        raise FileNotFoundError("population root contains no packed adapters")
    checkpoint = load_alignment_checkpoint(args.alignment_checkpoint)
    checkpoint_sha256 = sha256_directory(args.alignment_checkpoint)
    artifact = build_reference_bank_from_population(
        checkpoint,
        sample_directories,
        held_out_suite=args.held_out_suite,
        alignment_checkpoint_sha256=checkpoint_sha256,
    )
    save_reference_bank(artifact, output)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "sha256": sha256_file(output),
                "held_out_suite": artifact.bank.held_out_suite,
                "source_suites": artifact.bank.training_suites,
                "source_samples": artifact.source_sample_count,
                "reference_tasks": len(artifact.bank.samples),
                "alignment_checkpoint_sha256": checkpoint_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
