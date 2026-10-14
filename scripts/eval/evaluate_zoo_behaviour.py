#!/usr/bin/env python3
"""Paired frozen-base versus adapter rollouts for one zoo run.

A zoo sample is *structurally valid* once it reloads; this command makes it
*behaviourally evaluated*. Both policies run on identical initial states and
action-sampling seeds. The result is written to ``behaviour.json`` beside each
evaluated snapshot and never edits the snapshot itself. A snapshot that does not
beat the frozen base is recorded and kept, not discarded.

With ``--budget-inspection`` the inspection snapshots of a manifest source task
are evaluated instead, which is how the zoo budget is validated before it is
frozen.

Re-running the command is safe: snapshots that already carry a ``behaviour.json``
for the same rollout settings are skipped, and a record made with different
settings is reported as a conflict instead of being overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

from wsl_vla.evaluation.libero_rollout import LiberoRolloutEvaluator, OctoLiberoPolicy
from wsl_vla.evaluation.paired import clustered_bootstrap
from wsl_vla.experiments.manifest import task_id as make_task_id
from wsl_vla.experiments.provenance import git_state
from wsl_vla.experiments.runtime import (
    add_experiment_arguments,
    environment_record,
    input_hashes,
    load_experiment,
    require_gpu_runtime_settings,
)
from wsl_vla.octo.bridge import (
    assert_base_parameters_unchanged,
    assert_zero_adapter_equivalence,
    load_research_octo_from_config,
)
from wsl_vla.octo.training import initial_adapter_state
from wsl_vla.octo.zoo import atomic_write_json, load_zoo_adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True, help="zoo training seed")
    parser.add_argument("--updates", help="comma-separated snapshot stages (default: all)")
    parser.add_argument("--tier", choices=("development", "final"), default="development")
    parser.add_argument("--budget-inspection", action="store_true")
    args = parser.parse_args()
    require_gpu_runtime_settings()

    config, tasks, manifest, paths = load_experiment(args)
    identifier = make_task_id(args.suite, args.task_index)
    if identifier not in manifest.tasks or args.seed not in manifest.seeds:
        raise ValueError(f"{identifier} seed {args.seed} is not a run of {manifest.name!r}")
    if args.budget_inspection:
        if identifier not in manifest.zoo.inspection_tasks:
            raise ValueError(f"{identifier} is not a budget-inspection source task")
        stages = manifest.zoo.inspection_updates
        root = paths.budget_inspection
    else:
        stages = manifest.zoo.snapshot_updates
        root = paths.population
    if args.updates:
        requested = tuple(int(value) for value in args.updates.split(","))
        if set(requested) - set(stages):
            raise ValueError(f"requested stages {requested} are not snapshot stages {stages}")
        stages = requested
    run_directory = root / args.suite / identifier / f"seed_{args.seed}"
    rollout_count = manifest.rollouts.count(args.tier)
    max_steps = manifest.rollouts.max_episode_steps or int(config["simulator"]["max_episode_steps"])
    request = {
        "tier": args.tier,
        "rollout_count": rollout_count,
        "rollout_seed": manifest.rollouts.seed,
        "max_episode_steps": max_steps,
        "manifest_sha256": manifest.sha256(),
    }
    sample_directories = {}
    for stage in stages:
        directory = run_directory / f"update_{stage}"
        existing = directory / "behaviour.json"
        if not existing.exists():
            sample_directories[stage] = directory
            continue
        recorded = json.loads(existing.read_text(encoding="utf-8"))
        changed = sorted(key for key, value in request.items() if recorded.get(key) != value)
        if changed:
            raise FileExistsError(
                f"{existing} was recorded with different {changed}; behaviour records are "
                "never overwritten"
            )
    if not sample_directories:
        print(json.dumps({"ok": True, "already_complete": str(run_directory)}, indent=2))
        return 0

    environment = environment_record()
    bundle = load_research_octo_from_config(config, args.checkpoint, seed=args.seed)
    assert_zero_adapter_equivalence(bundle)
    token_width = int(config["adapter"]["token_width"])
    policies = {"frozen_base": OctoLiberoPolicy(bundle, initial_adapter_state(bundle))}
    metadata = {}
    for stage, directory in sample_directories.items():
        state, _, sample_metadata = load_zoo_adapter(directory, bundle, token_width=token_width)
        policies[f"update_{stage}"] = OctoLiberoPolicy(bundle, state)
        metadata[stage] = sample_metadata
    evaluator = LiberoRolloutEvaluator(
        suite=args.suite,
        expected_instructions=tuple(tasks["suites"][args.suite]),
        preprocessing=bundle.preprocessing,
        convention=bundle.action_convention,
        warmup_steps=int(config["simulator"]["warmup_steps"]),
        max_steps=max_steps,
    )
    summaries = evaluator.evaluate_policies(
        policies,
        task_index=args.task_index,
        rollout_count=rollout_count,
        seed=manifest.rollouts.seed,
    )
    base_check = assert_base_parameters_unchanged(bundle)
    commit, dirty = git_state(REPO_ROOT)
    base = summaries["frozen_base"]
    results = {}
    for stage, directory in sample_directories.items():
        adapter = summaries[f"update_{stage}"]
        differences = [
            float(left.success) - float(right.success)
            for left, right in zip(adapter.episodes, base.episodes)
        ]
        # One task and one training run: the difference is descriptive. Its
        # uncertainty is aggregated across tasks and runs by the report command.
        paired = clustered_bootstrap(
            {identifier: {f"seed_{args.seed}": differences}},
            resamples=int(config["evaluation"]["bootstrap_resamples"]),
            confidence=float(config["evaluation"]["confidence_level"]),
            seed=manifest.rollouts.seed,
        )
        payload = {
            "schema_version": 1,
            "sample_id": metadata[stage]["sample_id"],
            "adapter_sha256": metadata[stage]["adapter_sha256"],
            "purpose": "budget_inspection" if args.budget_inspection else "zoo",
            **request,
            "pairing": "identical initial states and action-sampling seeds",
            "frozen_base": {
                "successes": base.successes,
                "success_rate": base.success_rate,
                "controller_clipping": base.clipping,
                "episodes": [asdict(episode) for episode in base.episodes],
            },
            "adapter": {
                "successes": adapter.successes,
                "success_rate": adapter.success_rate,
                "controller_clipping": adapter.clipping,
                "policy_checkpoint_sha256": adapter.policy_checkpoint_sha256,
                "episodes": [asdict(episode) for episode in adapter.episodes],
            },
            "paired_success_difference": paired,
            "beats_frozen_base": adapter.successes > base.successes,
            "base_parameters_unchanged": base_check["base_sha256"],
            "provenance": {
                "command": list(sys.argv),
                "git_commit": commit,
                "git_dirty": dirty,
                "environment": environment,
                "inputs": input_hashes(args),
                "manifest_sha256": manifest.sha256(),
            },
        }
        atomic_write_json(directory / "behaviour.json", payload)
        results[f"update_{stage}"] = {
            "adapter_success_rate": adapter.success_rate,
            "frozen_base_success_rate": base.success_rate,
            "clipped_step_fraction": adapter.clipping["clipped_step_fraction"],
        }
    print(json.dumps({"ok": True, "run": str(run_directory), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
