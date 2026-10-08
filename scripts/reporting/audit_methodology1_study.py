#!/usr/bin/env python3
"""Report whether a Methodology 1 artifact tree is publication-complete."""

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

from wsl_vla.experiments.study_audit import audit_methodology1_study


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("research_root")
    parser.add_argument("--output")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return success after writing a progress report even when artifacts are missing",
    )
    args = parser.parse_args()
    report = audit_methodology1_study(args.research_root)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["ready"] or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
