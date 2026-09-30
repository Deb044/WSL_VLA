#!/usr/bin/env python3
"""Generate the locked Methodology 1 dependency graph as JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsl_vla.data.libero import load_suite_manifest
from wsl_vla.experiments.gamma_selection import validate_gamma_candidate_config
from wsl_vla.experiments.job_plan import build_methodology1_job_plan, summarize_job_plan
from wsl_vla.experiments.protocol import REQUIRED_SUITES, load_yaml, validate_reference_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", required=True)
    parser.add_argument("--data-root", default="data/libero")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--gamma-candidates", default="configs/research/gamma_candidates.yaml")
    parser.add_argument("--python", default="python")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    args = parser.parse_args()
    tasks = load_yaml(args.tasks)
    validate_reference_tasks(tasks)
    candidates = load_yaml(args.gamma_candidates)
    validate_gamma_candidate_config(candidates)
    data_root = Path(args.data_root)
    data_files = {
        suite: load_suite_manifest(data_root / suite, tasks["suites"][suite])
        for suite in REQUIRED_SUITES
    }
    jobs = build_methodology1_job_plan(
        research_root=args.research_root,
        data_files=data_files,
        gamma_candidates=candidates,
        python=args.python,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs), encoding="utf-8")
    summary = summarize_job_plan(jobs)
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "plan": str(output), "summary": str(summary_path), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
