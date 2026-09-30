#!/usr/bin/env python3
"""Compute publication metrics from immutable rollout JSONL records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsl_vla.evaluation.metrics import continual_learning_metrics
from wsl_vla.evaluation.records import (
    export_evaluation_parquet,
    load_evaluation_records,
    records_to_success_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--output-json")
    parser.add_argument("--output-parquet")
    args = parser.parse_args()

    records = load_evaluation_records(args.records)
    matrix = records_to_success_matrix(
        records,
        suite=args.suite,
        seed=args.seed,
        condition=args.condition,
        task_count=args.task_count,
    )
    report = {
        "suite": args.suite,
        "seed": args.seed,
        "condition": args.condition,
        "success_matrix": matrix.tolist(),
        "metrics": continual_learning_metrics(matrix),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    if args.output_parquet:
        export_evaluation_parquet(records, args.output_parquet)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
