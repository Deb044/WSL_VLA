#!/usr/bin/env python3
"""Aggregate generated-policy rollouts of one experiment.

Validation (development) rollouts and held-out rollouts are reported in separate
sections and are never pooled. Success rates are task-balanced; uncertainty is
a two-level bootstrap over tasks and training runs (a zoo run combined with an
alignment seed), so correlated episodes and snapshots are not counted as
independent evidence.
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

from wsl_vla.evaluation.generated_policies import summarize_rollouts
from wsl_vla.experiments.runtime import add_experiment_arguments, load_experiment
from wsl_vla.octo.zoo import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser, checkpoint=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-tasks", type=int, default=6)
    args = parser.parse_args()

    config, _, manifest, paths = load_experiment(args)
    report = {
        "schema_version": 1,
        "experiment": manifest.name,
        "profile": manifest.profile,
        "manifest_sha256": manifest.sha256(),
        "sections": {},
        "note": (
            "Validation rollouts are development diagnostics. Only held-out rollouts of a "
            "frozen full-profile experiment are publication evidence."
        ),
    }
    for scope in ("validation", "held_out"):
        records = []
        runs = []
        for run_manifest_path in sorted((paths.root / "rollouts" / scope).glob("*/seed_*/manifest.json")):
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if run_manifest["manifest_sha256"] != manifest.sha256():
                raise ValueError(
                    f"{run_manifest_path.parent} was produced under a different manifest"
                )
            entry = {
                "directory": str(run_manifest_path.parent),
                "fold": run_manifest["fold"],
                "alignment_seed": run_manifest["alignment_seed"],
                "tier": run_manifest["tier"],
                "finished": run_manifest["finished"],
                "failures": run_manifest["failures"],
            }
            runs.append(entry)
            if not run_manifest["finished"]:
                continue
            with (run_manifest_path.parent / "episodes.jsonl").open(encoding="utf-8") as stream:
                records.extend(json.loads(line) for line in stream if line.strip())
        section = {"runs": runs, "episodes": len(records)}
        if records:
            section.update(
                summarize_rollouts(
                    records,
                    resamples=int(config["evaluation"]["bootstrap_resamples"]),
                    confidence=float(config["evaluation"]["confidence_level"]),
                    seed=manifest.rollouts.seed,
                    minimum_tasks=args.minimum_tasks,
                )
            )
        report["sections"][scope] = section
    report["publication_evidence"] = bool(
        manifest.publication_evidence and report["sections"]["held_out"]["episodes"]
    )
    atomic_write_json(Path(args.output), report)
    print(
        json.dumps(
            {
                "ok": True,
                "output": args.output,
                "episodes": {scope: value["episodes"] for scope, value in report["sections"].items()},
                "publication_evidence": report["publication_evidence"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
