#!/usr/bin/env python3
"""Compute publication metrics from immutable rollout JSONL records."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsl_vla.evaluation.metrics import (
    aggregate_seed_metrics,
    continual_learning_metrics,
    drift_degradation_correlations,
)
from wsl_vla.evaluation.records import (
    export_evaluation_parquet,
    load_evaluation_records,
    records_to_success_matrix,
)
from wsl_vla.evaluation.reporting import plot_metric_intervals, plot_success_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records")
    parser.add_argument("--suite", required=True)
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--output-json")
    parser.add_argument("--output-parquet")
    parser.add_argument("--output-figure")
    parser.add_argument("--drift-correlations", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()

    records = load_evaluation_records(args.records)
    if args.seed is not None:
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
        if args.output_figure:
            plot_success_matrix(
                matrix,
                args.output_figure,
                title=f"{args.suite}: {args.condition}, seed {args.seed}",
            )
    else:
        per_seed, estimates = aggregate_seed_metrics(
            records,
            suite=args.suite,
            condition=args.condition,
            seeds=args.seeds,
            task_count=args.task_count,
            resamples=args.bootstrap_resamples,
        )
        report = {
            "suite": args.suite,
            "seeds": args.seeds,
            "condition": args.condition,
            "per_seed_metrics": per_seed,
            "aggregate_metrics": {name: asdict(value) for name, value in estimates.items()},
        }
        if args.drift_correlations:
            correlations = drift_degradation_correlations(
                records,
                condition=args.condition,
                resamples=args.bootstrap_resamples,
            )
            report["drift_correlations"] = {
                modality: {method: asdict(value) for method, value in methods.items()}
                for modality, methods in correlations.items()
            }
        if args.output_figure:
            plot_metric_intervals(
                estimates,
                args.output_figure,
                title=f"{args.suite}: {args.condition} across seeds",
            )
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
