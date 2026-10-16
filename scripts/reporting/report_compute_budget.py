#!/usr/bin/env python3
"""Measured compute of an executed zoo/alignment experiment, projected to another manifest.

Reads the results of an experiment that ran (e.g. the pilot): its job records,
zoo runs, behaviour and rollout episodes, and alignment metadata. Reports the
measured unit costs, the exact workload of a target manifest (e.g. full), and
the job-hours and wall-clock that follow. Unit costs that were not measured yet
(for example alignment before the alignment stage ran) are reported as missing,
never filled in.

    python scripts/reporting/report_compute_budget.py \\
        --manifest configs/research/manifests/pilot.yaml \\
        --research-root /scratch/$USER/research_results \\
        --target-manifest configs/research/manifests/full.yaml \\
        --output /scratch/$USER/research_results/experiments/pilot/compute_budget.json
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

from wsl_vla.experiments.compute_budget import (
    collect_measurements,
    experiment_workload,
    project_workload,
    rollout_condition_count,
    wall_clock,
)
from wsl_vla.experiments.manifest import load_experiment_manifest
from wsl_vla.experiments.protocol import load_yaml, validate_reference_tasks
from wsl_vla.experiments.provenance import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="manifest of the experiment that ran")
    parser.add_argument("--research-root", default="research_results")
    parser.add_argument("--target-manifest", help="manifest to project to (default: the measured one)")
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--runners", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--training-slots", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()

    tasks = load_yaml(args.tasks)
    validate_reference_tasks(tasks)
    config = load_yaml(args.config)
    measured_manifest = load_experiment_manifest(args.manifest, reference_tasks=tasks)
    target = load_experiment_manifest(args.target_manifest or args.manifest, reference_tasks=tasks)

    def workload(manifest):
        steps = manifest.rollouts.max_episode_steps or int(config["simulator"]["max_episode_steps"])
        return experiment_workload(manifest, max_episode_steps=steps, conditions=rollout_condition_count(manifest))

    measured = collect_measurements(measured_manifest, args.research_root)
    report = {"measured": measured, "workload": {}, "projection": {}}
    for label, manifest in (("measured_experiment", measured_manifest), ("target", target)):
        counts = workload(manifest)
        report["workload"][label] = counts
        cases = {
            "measured_episode_length": project_workload(counts, measured),
            "every_episode_to_horizon": project_workload(
                counts, measured, episode_policy_steps=counts["max_steps_per_episode"]
            ),
        }
        report["projection"][label] = {
            name: {
                **projection,
                "wall_clock": wall_clock(
                    counts, projection, runners=args.runners, workers=args.workers, training_slots=args.training_slots
                ),
            }
            for name, projection in cases.items()
        }
    if args.output:
        atomic_write_json(Path(args.output), report)
    print(json.dumps(report if not args.output else {"ok": True, "output": args.output, "projection": report["projection"]}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
