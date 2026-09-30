#!/usr/bin/env python3
"""Validate and aggregate the four-fold component-swap study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsl_vla.evaluation.component_reporting import component_swap_study_report
from wsl_vla.evaluation.records import (
    export_typed_records_parquet,
    load_component_swap_record_tree,
)
from wsl_vla.experiments.protocol import REQUIRED_SUITES


def save_scatter_figure(report: dict, output_path: str | Path) -> None:
    """Render the three prespecified drift/drop scatter plots from stored records."""
    import matplotlib.pyplot as plt

    samples = report["transition_samples"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for axis, modality in zip(axes, ("vision", "language", "action")):
        axis.scatter(
            [row["latent_drift"][modality] for row in samples],
            [row["success_rate_drop"][modality] for row in samples],
            alpha=0.7,
        )
        axis.set_title(modality.capitalize())
        axis.set_xlabel("Latent drift")
        axis.set_ylabel("Success-rate drop")
        axis.grid(alpha=0.2)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-parquet")
    parser.add_argument("--output-figure")
    parser.add_argument("--condition", default="proposed_asymmetric")
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 73))
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    records = load_component_swap_record_tree(args.records)
    report = component_swap_study_report(
        records,
        suites=REQUIRED_SUITES,
        seeds=args.seeds,
        condition=args.condition,
        resamples=args.bootstrap_resamples,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_parquet:
        export_typed_records_parquet(records, args.output_parquet)
    if args.output_figure:
        save_scatter_figure(report, args.output_figure)
    print(json.dumps({"ok": True, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
