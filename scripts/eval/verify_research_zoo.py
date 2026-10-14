#!/usr/bin/env python3
"""Validate the zoo population of one experiment against its manifest.

Expected samples, seeds, and snapshot stages come from the manifest. The report
separates *structurally valid* samples (reloadable, hashes and identities
consistent) from *behaviourally evaluated* ones (paired frozen-base/adapter
rollouts recorded in ``behaviour.json``), and lists incomplete or failed runs,
which are preserved rather than hidden.
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

from wsl_vla.alignment.archive import validate_population
from wsl_vla.experiments.runtime import add_experiment_arguments, load_experiment
from wsl_vla.octo.zoo import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser, checkpoint=False)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="report progress of a population that is still being trained")
    parser.add_argument("--output", help="default: <population>/verification.json")
    args = parser.parse_args()

    _, _, manifest, paths = load_experiment(args)
    report = validate_population(manifest, paths, require_complete=not args.allow_incomplete)
    runs = []
    behaviour = []
    for identifier, seed in manifest.runs():
        summary_path = paths.run_directory(identifier, seed) / "run.json"
        if not summary_path.is_file():
            runs.append({"run": f"{identifier}/seed_{seed}", "status": "missing"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "run": f"{identifier}/seed_{seed}",
                "status": "completed" if summary.get("completed") else "failed",
                "failures": summary.get("failures", []),
                "held_out_loss_frozen_base": summary.get("held_out_loss_frozen_base"),
            }
        )
    for sample, directory, _ in report["records"]:
        path = directory / "behaviour.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            behaviour.append(
                {
                    "sample_id": sample.sample_id,
                    "adapter_success_rate": payload["adapter"]["success_rate"],
                    "frozen_base_success_rate": payload["frozen_base"]["success_rate"],
                    "beats_frozen_base": payload["beats_frozen_base"],
                }
            )
    payload = {
        "schema_version": 2,
        "ok": not report["missing_samples"],
        "experiment": manifest.name,
        "manifest_sha256": manifest.sha256(),
        "expected": manifest.derived_counts(),
        "expected_samples": report["expected_samples"],
        "present_samples": report["present_samples"],
        "missing_samples": report["missing_samples"],
        "structurally_valid_samples": report["structurally_valid_samples"],
        "behaviourally_evaluated_samples": len(behaviour),
        "behaviour": behaviour,
        "runs": runs,
        "incomplete_or_failed_runs": [item for item in runs if item["status"] != "completed"],
        "shared": report["shared"],
    }
    output = Path(args.output) if args.output else paths.population / "verification.json"
    atomic_write_json(output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key not in {"runs", "behaviour"}},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
