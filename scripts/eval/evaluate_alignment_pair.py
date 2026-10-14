#!/usr/bin/env python3
"""Paired offline diagnostics: aligned versus reconstruction-only.

This command replaces the old "alignment advantage gate". It never requires
alignment to win:

* exit 0 - the comparison is valid (integrity passed), whatever it found;
* exit 2 - an integrity failure (unpaired checkpoints, hash mismatch, leakage,
  non-finite values).

Each comparison is labelled ``inconclusive``, ``supports_first_condition``, or
``negative_finding``. ``--scope validation`` evaluates the fold's validation
tasks and may be re-run freely during development. ``--scope held_out``
evaluates the held-out suite, requires a frozen zoo budget, and freezes the
experiment manifest: held-out numbers cannot steer later configuration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from wsl_vla.alignment.checkpoint import load_alignment_checkpoint
from wsl_vla.alignment.diagnostics import run_pair_diagnostics
from wsl_vla.experiments.manifest import record_held_out_results
from wsl_vla.experiments.provenance import git_state
from wsl_vla.experiments.runtime import (
    add_experiment_arguments,
    environment_record,
    input_hashes,
    load_experiment,
)
from wsl_vla.octo.zoo import atomic_write_json


def strip_per_task(report: dict) -> dict:
    return {
        **report,
        "conditions": {
            name: {key: value for key, value in condition.items() if key != "per_task"}
            for name, condition in report["conditions"].items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser, checkpoint=False)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, required=True, help="alignment seed")
    parser.add_argument("--scope", choices=("validation", "held_out"), default="validation")
    parser.add_argument("--minimum-tasks", type=int, default=6,
                        help="fewer evaluated tasks than this is always labelled inconclusive")
    args = parser.parse_args()

    config, _, manifest, paths = load_experiment(args)
    fold = manifest.fold(args.fold)
    if args.scope == "held_out":
        manifest.require_frozen_budget("held-out alignment diagnostics")
        tasks = fold.held_out_tasks
        if not tasks:
            raise ValueError(f"fold {fold.name} defines no held-out tasks")
    else:
        tasks = fold.validation_tasks
    output = paths.diagnostics(fold.name, args.seed, args.scope) / "alignment_pair.json"
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    confidence = float(config["evaluation"]["confidence_level"])
    try:
        environment = environment_record()
        aligned = load_alignment_checkpoint(paths.alignment_checkpoint(fold.name, args.seed, "aligned"))
        control = load_alignment_checkpoint(
            paths.alignment_checkpoint(fold.name, args.seed, "reconstruction_only")
        )
        for checkpoint in (aligned, control):
            if checkpoint.metadata["experiment_manifest_sha256"] != manifest.sha256():
                raise ValueError("alignment checkpoint was trained under a different manifest")
        # Everything the report depends on; an existing report is reused only
        # for the identical request and is never overwritten otherwise.
        request = {
            "manifest_sha256": manifest.sha256(),
            "fold": fold.name,
            "alignment_seed": args.seed,
            "scope": args.scope,
            "evaluated_tasks": list(tasks),
            "minimum_tasks": args.minimum_tasks,
            "bootstrap_resamples": resamples,
            "confidence_level": confidence,
            "checkpoints": {
                "aligned": dict(aligned.metadata["files"]),
                "reconstruction_only": dict(control.metadata["files"]),
            },
        }
        if output.exists():
            recorded = json.loads(output.read_text(encoding="utf-8"))
            if recorded.get("request") != request:
                raise FileExistsError(
                    f"{output} was produced for different inputs; diagnostics are never overwritten"
                )
            status = recorded["integrity"]["status"]
            print(json.dumps({"ok": status == "pass", "already_complete": str(output),
                              "integrity": status}, indent=2))
            return 0 if status == "pass" else 2
        report = run_pair_diagnostics(
            manifest=manifest,
            paths=paths,
            aligned=aligned,
            control=control,
            evaluated_tasks=tasks,
            scope=args.scope,
            minimum_tasks=args.minimum_tasks,
            resamples=resamples,
            confidence=confidence,
        )
    except FileExistsError:
        raise
    except Exception as exc:
        failure = {
            "integrity": {"status": "fail", "error": f"{type(exc).__name__}: {exc}"},
            "fold": args.fold,
            "alignment_seed": args.seed,
            "scope": args.scope,
            "manifest_sha256": manifest.sha256(),
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "command": list(sys.argv),
        }
        # An integrity failure is evidence about the experiment, so it is kept:
        # one line per failed attempt, beside where the report would have been.
        output.parent.mkdir(parents=True, exist_ok=True)
        with (output.parent / "integrity_failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(failure, sort_keys=True) + "\n")
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2
    commit, dirty = git_state(REPO_ROOT)
    report["request"] = request
    report["provenance"] = {
        "command": list(sys.argv),
        "git_commit": commit,
        "git_dirty": dirty,
        "environment": environment,
        "inputs": input_hashes(args),
        "manifest_sha256": manifest.sha256(),
    }
    atomic_write_json(output, report)
    if args.scope == "held_out":
        record_held_out_results(manifest, paths, artifact=str(output.relative_to(paths.root)))
    summary = strip_per_task(report)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "integrity": summary["integrity"]["status"],
                "evaluated_tasks": len(report["evaluated_tasks"]),
                "comparisons": {
                    name: {
                        "evidence": value["evidence"],
                        "mean_difference": value["paired_difference"]["mean"],
                        "ci": [value["paired_difference"]["ci_low"], value["paired_difference"]["ci_high"]],
                    }
                    for name, value in report["comparisons"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
