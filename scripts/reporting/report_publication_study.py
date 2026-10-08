#!/usr/bin/env python3
"""Validate and report the complete four-fold, three-seed continual study."""

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

from wsl_vla.evaluation.publication import publication_study_report
from wsl_vla.evaluation.records import (
    export_evaluation_parquet,
    load_evaluation_record_tree,
)
from wsl_vla.experiments.conditions import PRIMARY_CONDITIONS
from wsl_vla.experiments.protocol import REQUIRED_SUITES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-parquet")
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 73))
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    records = load_evaluation_record_tree(args.records)
    report = publication_study_report(
        records,
        suites=REQUIRED_SUITES,
        seeds=args.seeds,
        conditions=PRIMARY_CONDITIONS,
        oracle_condition="independent_adapter_oracle",
        task_count=10,
        resamples=args.bootstrap_resamples,
    )
    target = Path(args.output_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_parquet:
        export_evaluation_parquet(records, args.output_parquet)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(target),
                "run_count": report["design"]["expected_run_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
