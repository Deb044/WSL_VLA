#!/usr/bin/env python3
"""Write the resumable job manifest of one zoo/alignment experiment.

The output is a JSONL file of jobs with stable ids, argument arrays,
dependencies, expected outputs, and resource needs, ordered so that every job
follows its dependencies. Generating it runs nothing. Execute it (here or on
collaborator hardware) with ``scripts/train/run_job_manifest.py``.
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

from wsl_vla.experiments.manifest import ExperimentPaths, load_experiment_manifest
from wsl_vla.experiments.protocol import load_yaml, validate_reference_tasks
from wsl_vla.experiments.provenance import atomic_write_json
from wsl_vla.experiments.zoo_jobs import (
    build_experiment_job_plan,
    summarize_job_plan,
    write_job_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--research-root", default="research_results")
    parser.add_argument("--data-root")
    parser.add_argument("--python", default="python")
    parser.add_argument("--output", help="default: <experiment root>/jobs/jobs.jsonl")
    args = parser.parse_args()

    tasks = load_yaml(args.tasks)
    validate_reference_tasks(tasks)
    manifest = load_experiment_manifest(args.manifest, reference_tasks=tasks)
    paths = ExperimentPaths(Path(args.research_root), manifest.name)
    jobs = build_experiment_job_plan(
        manifest,
        manifest_path=args.manifest,
        research_root=args.research_root,
        python=args.python,
        data_root=args.data_root,
    )
    output = Path(args.output) if args.output else paths.jobs / "jobs.jsonl"
    # Both writes are atomic: every runner of a multi-runner launch replans.
    write_job_plan(jobs, output)
    summary = summarize_job_plan(jobs, manifest)
    atomic_write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps({"ok": True, "output": str(output), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
