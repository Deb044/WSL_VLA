#!/usr/bin/env python3
"""Validate and aggregate all four-fold, three-seed OOD experiments."""

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

from wsl_vla.evaluation.ood_reporting import ood_study_report
from wsl_vla.evaluation.records import (
    export_typed_records_parquet,
    load_ood_evaluation_record_tree,
)
from wsl_vla.experiments.protocol import REQUIRED_SUITES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-parquet")
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 73))
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    records = load_ood_evaluation_record_tree(args.records)
    report = ood_study_report(
        records,
        suites=REQUIRED_SUITES,
        seeds=args.seeds,
        resamples=args.bootstrap_resamples,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_parquet:
        export_typed_records_parquet(records, args.output_parquet)
    print(json.dumps({"ok": True, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
