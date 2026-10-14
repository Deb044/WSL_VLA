#!/usr/bin/env python3
"""Generate the Methodology 1 dependency graph (corrected core + downstream study).

The core stages (gates, evidence, zoo, archives, paired alignment, diagnostics,
generated-policy rollouts) are exactly those of ``plan_zoo_alignment_jobs.py``
for the given experiment manifest. The downstream study (``m1:*`` jobs) is
appended and waits for the integrity of the paired alignment diagnostics, never
for a favourable alignment outcome. Execute the result with
``scripts/train/run_job_manifest.py``.

The manifest must have a frozen zoo budget and one leave-one-suite-out fold per
reference suite (``configs/research/manifests/full.yaml`` once frozen).
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

from wsl_vla.experiments.gamma_selection import validate_gamma_candidate_config
from wsl_vla.experiments.job_plan import build_methodology1_job_plan, summarize_job_plan
from wsl_vla.experiments.manifest import ExperimentPaths, load_experiment_manifest
from wsl_vla.experiments.protocol import load_yaml, validate_reference_tasks
from wsl_vla.experiments.zoo_jobs import write_job_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--research-root", default="research_results")
    parser.add_argument("--data-root")
    parser.add_argument("--gamma-candidates", default="configs/research/gamma_candidates.yaml")
    parser.add_argument("--python", default="python")
    parser.add_argument("--output", help="default: <experiment root>/jobs/methodology1.jsonl")
    parser.add_argument("--summary")
    args = parser.parse_args()

    tasks = load_yaml(args.tasks)
    validate_reference_tasks(tasks)
    candidates = load_yaml(args.gamma_candidates)
    validate_gamma_candidate_config(candidates)
    manifest = load_experiment_manifest(args.manifest, reference_tasks=tasks)
    jobs = build_methodology1_job_plan(
        manifest,
        manifest_path=args.manifest,
        research_root=args.research_root,
        gamma_candidates=candidates,
        python=args.python,
        data_root=args.data_root,
    )
    paths = ExperimentPaths(Path(args.research_root), manifest.name)
    output = Path(args.output) if args.output else paths.jobs / "methodology1.jsonl"
    write_job_plan(jobs, output)
    summary = {"experiment": manifest.name, "manifest_sha256": manifest.sha256(),
               **summarize_job_plan(jobs)}
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "plan": str(output), "summary": str(summary_path), **summary},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
