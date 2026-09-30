#!/usr/bin/env python3
"""Validate the complete official 360-sample Octo adapter population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsl_vla.experiments.protocol import REQUIRED_SUITES
from wsl_vla.experiments.zoo_validation import (
    locked_population_identities,
    validate_research_population,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("population_root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = validate_research_population(
        args.population_root,
        expected_identities=locked_population_identities(REQUIRED_SUITES),
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
