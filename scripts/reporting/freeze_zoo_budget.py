#!/usr/bin/env python3
"""Freeze the zoo training budget of an experiment before held-out evaluation.

Budget validation trains the manifest's source inspection tasks to the largest
inspection stage and rolls out the inspection snapshots
(``train_research_zoo.py --budget-inspection`` followed by
``evaluate_zoo_behaviour.py --budget-inspection``). This command records the
chosen budget in a new manifest file with ``zoo.frozen: true``. Held-out
diagnostics and rollouts refuse to run on an unfrozen manifest, and a manifest
can no longer change once the experiment holds held-out results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.experiments.manifest import ExperimentPaths, load_experiment_manifest
from wsl_vla.experiments.protocol import load_yaml, validate_reference_tasks


def inspection_evidence(paths: ExperimentPaths) -> list[dict]:
    rows = []
    for path in sorted(paths.budget_inspection.rglob("behaviour.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "sample_id": payload["sample_id"],
                "adapter_success_rate": payload["adapter"]["success_rate"],
                "frozen_base_success_rate": payload["frozen_base"]["success_rate"],
                "rollout_count": payload["rollout_count"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--research-root", default="research_results")
    parser.add_argument("--optimizer-updates", type=int, required=True)
    parser.add_argument("--snapshot-updates", required=True,
                        help="comma-separated snapshot stages ending at --optimizer-updates")
    parser.add_argument("--output", required=True, help="path of the frozen manifest to write")
    parser.add_argument("--rationale", required=True, help="why this budget was chosen")
    parser.add_argument(
        "--without-inspection-evidence",
        action="store_true",
        help="freeze although no budget-inspection rollouts exist (recorded in the file)",
    )
    args = parser.parse_args()

    tasks = load_yaml(args.tasks)
    validate_reference_tasks(tasks)
    manifest = load_experiment_manifest(args.manifest, reference_tasks=tasks)
    paths = ExperimentPaths(Path(args.research_root), manifest.name)
    if paths.held_out_marker.is_file():
        raise RuntimeError(
            f"experiment {manifest.name!r} already holds held-out results; its budget can "
            "no longer change. Start a new experiment name."
        )
    evidence = inspection_evidence(paths)
    if not evidence and not args.without_inspection_evidence:
        raise RuntimeError(
            "no budget-inspection rollouts found under "
            f"{paths.budget_inspection}; run them or pass --without-inspection-evidence"
        )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    payload["zoo"]["optimizer_updates"] = args.optimizer_updates
    payload["zoo"]["snapshot_updates"] = [int(value) for value in args.snapshot_updates.split(",")]
    payload["zoo"]["frozen"] = True
    header = [
        f"# Frozen zoo budget for experiment {manifest.name!r}.",
        f"# Source manifest: {args.manifest} (sha256 {manifest.sha256()})",
        f"# Rationale: {args.rationale}",
        "# Budget-inspection evidence (source tasks only):",
        *(f"#   {json.dumps(row, sort_keys=True)}" for row in evidence),
    ]
    if not evidence:
        header.append("#   NONE - frozen without inspection rollouts by explicit request.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(header) + "\n" + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    frozen = load_experiment_manifest(output, reference_tasks=tasks)
    changed = frozen.zoo_sha256() != manifest.zoo_sha256()
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "manifest_sha256": frozen.sha256(),
                "zoo_training_identity_changed": changed,
                "note": (
                    "The budget differs from the one existing zoo samples were trained with; "
                    "retrain the zoo under the frozen manifest."
                    if changed
                    else "Existing zoo samples remain valid under the frozen manifest."
                ),
                "inspection_evidence_rows": len(evidence),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
