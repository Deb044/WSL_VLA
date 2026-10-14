#!/usr/bin/env python3
"""Build the zoo of one experiment manifest through the resumable job runner.

This is a convenience wrapper: it (optionally) downloads exactly the
demonstration files the manifest names, writes the experiment's job manifest,
and executes the gate, evidence, zoo, and zoo-verification stages. Tasks,
seeds, budgets, and snapshot stages come from the manifest. Interrupted zoo
runs resume from their saved state, failed runs keep their logs and are
retried on the next invocation, and one failed run does not stop the others.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
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
from wsl_vla.experiments.zoo_jobs import (
    build_experiment_job_plan,
    run_job_plan,
    summarize_job_plan,
    write_job_plan,
)

ZOO_STAGES = ("gates", "evidence", "zoo", "zoo_behaviour", "zoo_verification")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--research-root", default="research_results")
    parser.add_argument("--data-root", default="data/libero")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--download", action="store_true",
                        help="download the manifest's demonstration files first")
    parser.add_argument("--skip-simulator", action="store_true",
                        help="skip paired rollouts and camera checks on hosts without LIBERO")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = load_yaml(args.tasks)
    validate_reference_tasks(tasks)
    manifest = load_experiment_manifest(args.manifest, reference_tasks=tasks)
    environment = dict(os.environ)
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    if args.download and not args.dry_run:
        subprocess.run(
            [args.python, "scripts/setup/download_libero.py", "--manifest", args.manifest,
             "--dest", args.data_root],
            cwd=str(REPO_ROOT),
            env=environment,
            check=True,
        )
    paths = ExperimentPaths(Path(args.research_root), manifest.name)
    jobs = build_experiment_job_plan(
        manifest,
        manifest_path=args.manifest,
        research_root=args.research_root,
        python=args.python,
        data_root=args.data_root,
    )
    plan_path = paths.jobs / "jobs.jsonl"
    write_job_plan(jobs, plan_path)
    plan_path.with_suffix(".summary.json").write_text(
        json.dumps(summarize_job_plan(jobs, manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = run_job_plan(
        jobs,
        state_directory=paths.jobs,
        working_directory=REPO_ROOT,
        stages=ZOO_STAGES,
        skip_simulator=args.skip_simulator,
        dry_run=args.dry_run,
        environment=environment,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "events"}, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
