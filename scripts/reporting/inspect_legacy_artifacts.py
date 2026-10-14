#!/usr/bin/env python3
"""Describe version-one (pre-repair) artifacts without migrating them.

Legacy zoo samples, evidence, and alignment checkpoints remain readable for
diagnostics only. This command reports what exists and why it is legacy; it
never writes inside the inspected tree, and corrected research commands reject
those artifacts by schema version.
"""

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

from wsl_vla.experiments.legacy import inspect_legacy_tree


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="research_results")
    parser.add_argument("--output", help="optional JSON report path outside the legacy tree")
    args = parser.parse_args()
    report = inspect_legacy_tree(args.root)
    if args.output:
        output = Path(args.output).resolve()
        legacy = Path(args.root).resolve()
        if legacy in output.parents and "experiments" not in output.relative_to(legacy).parts[:1]:
            raise ValueError("write the report outside the legacy artifact directories")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "adapters"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
