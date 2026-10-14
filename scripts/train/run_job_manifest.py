#!/usr/bin/env python3
"""Execute a job manifest written by ``plan_zoo_alignment_jobs.py``
(or ``plan_methodology1_jobs.py``, which appends the downstream study).

The runner is resumable and dependency-aware: completed jobs are skipped, a job
whose dependency failed is reported as blocked instead of being run, and one
failure does not stop independent jobs. Each job logs to
``<state dir>/logs``; ``failure_summary.json`` lists every failed and blocked
job with the tail of its log. Re-running the same command retries failures.

Several runners may execute one job manifest at the same time, for example one
per GPU slice in separate scheduler jobs, as long as they share the state
directory: each job runs once, runners wait for dependencies that another
runner is executing, and the jobs of a killed runner are taken over. SIGTERM
(a scheduler's time limit) or Ctrl-C terminates this runner's jobs and records
them as interrupted; they continue from their resume state on the next run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.experiments.job_plan import DOWNSTREAM_STAGES
from wsl_vla.experiments.runtime import DETERMINISTIC_XLA_FLAG
from wsl_vla.experiments.zoo_jobs import STAGES as CORE_STAGES, read_job_plan, run_job_plan

STAGES = (*CORE_STAGES, *DOWNSTREAM_STAGES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", help="jobs.jsonl written by plan_zoo_alignment_jobs.py")
    parser.add_argument("--state-dir", help="default: the directory of the job manifest")
    parser.add_argument("--stages", help=f"comma-separated subset of {', '.join(STAGES)}")
    parser.add_argument("--jobs-only", help="comma-separated job ids")
    parser.add_argument("--skip-simulator", action="store_true",
                        help="do not run jobs that need the LIBERO simulator on this host")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--allow-nondeterministic-gpu",
        action="store_true",
        help=f"never add {DETERMINISTIC_XLA_FLAG} to XLA_FLAGS. By default it is added for "
        "every job except alignment training (where it costs about 36x); without it identical "
        "runs may differ by float rounding",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="run every job with HF_HUB_OFFLINE=1: the pinned checkpoint and tokenizer are "
        "read from the local Hugging Face cache, so a hub outage cannot stall or alter a run "
        "(requires the gates stage, or any earlier run, to have populated the cache)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="jobs this runner executes at once (default 1). Rollouts are CPU-bound, so a GPU "
        "slice with several CPUs can run a training job and rollouts side by side",
    )
    parser.add_argument(
        "--training-slots",
        type=int,
        default=1,
        help="of those, GPU-training jobs (zoo, alignment, downstream training) at once "
        "(default 1: two trainings on one GPU slice share its compute and memory)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="how often to look for work finished by other runners (default 30)",
    )
    parser.add_argument(
        "--stale-after-minutes",
        type=float,
        default=15.0,
        help="take over a job whose runner has not sent a heartbeat for this long (default 15)",
    )
    args = parser.parse_args()

    plan_path = Path(args.jobs)
    jobs = read_job_plan(plan_path)
    stages = None if not args.stages else tuple(args.stages.split(","))
    if stages and set(stages) - set(STAGES):
        raise ValueError(f"unknown stages: {sorted(set(stages) - set(STAGES))}")
    environment = dict(os.environ)
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")

    stop = threading.Event()

    def request_stop(signum, frame):
        print(f"received signal {signum}: stopping this runner's jobs", file=sys.stderr, flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    summary = run_job_plan(
        jobs,
        state_directory=args.state_dir or plan_path.parent,
        working_directory=REPO_ROOT,
        stages=stages,
        job_ids=None if not args.jobs_only else tuple(args.jobs_only.split(",")),
        skip_simulator=args.skip_simulator,
        skip_gpu=args.skip_gpu,
        dry_run=args.dry_run,
        keep_going=not args.stop_on_failure,
        environment=environment,
        deterministic_gpu=not args.allow_nondeterministic_gpu,
        workers=args.workers,
        training_slots=args.training_slots,
        poll_seconds=args.poll_seconds,
        stale_after_seconds=60.0 * args.stale_after_minutes,
        stop=stop,
    )
    printable = {key: value for key, value in summary.items() if key != "events"}
    if args.dry_run:
        printable["would_run"] = [
            event["job_id"] for event in summary["events"] if event["status"] == "would_run"
        ]
    print(json.dumps(printable, indent=2))
    if summary["failed"]:
        return 1
    return 3 if summary.get("interrupted") else 0


if __name__ == "__main__":
    raise SystemExit(main())
