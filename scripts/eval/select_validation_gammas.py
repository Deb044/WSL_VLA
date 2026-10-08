#!/usr/bin/env python3
"""Freeze gamma and patience values from complete non-test validation records."""

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

from wsl_vla.experiments.gamma_selection import (
    load_gamma_validation_records,
    save_gamma_selection_with_report,
    select_gamma_configuration,
)
from wsl_vla.experiments.protocol import REQUIRED_SUITES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records")
    parser.add_argument("--held-out-suite", required=True, choices=REQUIRED_SUITES)
    parser.add_argument("--seeds", default="17,42,73")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    output = Path(args.output)
    report = Path(args.report) if args.report else output.with_suffix(".report.json")
    if output.exists() or report.exists():
        raise FileExistsError("refusing to overwrite frozen gamma-selection artifacts")
    seeds = tuple(int(value) for value in args.seeds.split(","))
    selection, details = select_gamma_configuration(
        load_gamma_validation_records(args.records),
        held_out_suite=args.held_out_suite,
        seeds=seeds,
    )
    save_gamma_selection_with_report(
        selection,
        details,
        selection_path=output,
        report_path=report,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "selection": str(output),
                "report": str(report),
                "proposed": dict(selection.proposed),
                "uniform": selection.uniform,
                "early_stopping_patience": selection.early_stopping_patience,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
