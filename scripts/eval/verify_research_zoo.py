#!/usr/bin/env python3
"""Validate the complete official 360-sample Octo adapter population."""

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
