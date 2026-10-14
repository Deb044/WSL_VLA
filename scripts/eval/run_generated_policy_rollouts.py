#!/usr/bin/env python3
"""Roll out reconstructed and evidence-generated policies in LIBERO.

For every evaluated task this runs, on identical initial states and with
identical action-sampling seeds: the frozen base, the independently trained
zoo adapter, the autoencoder reconstruction and the mapper-generated adapter of
both alignment conditions, the nearest-neighbour adapter, and the training-mean
adapter. Every generated update is rank-matched before deployment.

``--scope validation`` (development) evaluates the fold's validation tasks.
``--scope held_out`` evaluates the held-out suite; it requires a frozen zoo
budget and freezes the experiment manifest. Episode-level outcomes are stored
in ``episodes.jsonl``; ``summary.json`` holds task-balanced success rates,
paired differences, and clustered confidence intervals.

The run is resumable. Re-running the same command after an interruption
continues after the last completed task; a finished run is reported as complete
and left untouched; an interrupted run with a different configuration is moved
to ``failed_attempts/`` (never deleted) before a fresh run starts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.alignment.checkpoint import assert_paired_checkpoints, load_alignment_checkpoint
from wsl_vla.alignment.diagnostics import PopulationIndex
from wsl_vla.evaluation.generated_policies import (
    build_policy_states,
    condition_names,
    episode_records,
    summarize_rollouts,
)
from wsl_vla.evaluation.libero_rollout import LiberoRolloutEvaluator, OctoLiberoPolicy
from wsl_vla.evaluation.rollout_runs import open_rollout_run
from wsl_vla.experiments.manifest import (
    ALIGNMENT_CONDITIONS,
    record_held_out_results,
    split_task_id,
)
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
    build_adapter_spec_and_factors,
    load_research_octo_from_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, required=True, help="alignment seed")
    parser.add_argument("--scope", choices=("validation", "held_out"), default="validation")
    parser.add_argument("--tier", choices=("development", "final"), default="development")
    parser.add_argument("--conditions", help="comma-separated subset of policy conditions")
    parser.add_argument("--task-ids", help="comma-separated subset of the scope's task ids")
    parser.add_argument("--minimum-tasks", type=int, default=6)
    args = parser.parse_args()
    require_gpu_runtime_settings()

    config, tasks, manifest, paths = load_experiment(args)
    fold = manifest.fold(args.fold)
    if args.scope == "held_out":
        manifest.require_frozen_budget("held-out rollouts")
        scope_tasks = fold.held_out_tasks
    else:
        scope_tasks = fold.validation_tasks
    selected = tuple(args.task_ids.split(",")) if args.task_ids else tuple(scope_tasks)
    unknown = sorted(set(selected) - set(scope_tasks))
    if unknown or not selected:
        raise ValueError(f"tasks outside the {args.scope} scope of fold {fold.name}: {unknown}")
    available = condition_names(ALIGNMENT_CONDITIONS)
    conditions = tuple(args.conditions.split(",")) if args.conditions else available
    output = paths.rollouts(args.scope, fold.name, args.seed)
    rollout_count = manifest.rollouts.count(args.tier)
    max_steps = manifest.rollouts.max_episode_steps or int(config["simulator"]["max_episode_steps"])

    environment = environment_record()
    checkpoints = {
        condition: load_alignment_checkpoint(
            paths.alignment_checkpoint(fold.name, args.seed, condition)
        )
        for condition in ALIGNMENT_CONDITIONS
    }
    assert_paired_checkpoints(checkpoints["aligned"], checkpoints["reconstruction_only"])
    for checkpoint in checkpoints.values():
        if checkpoint.metadata["experiment_manifest_sha256"] != manifest.sha256():
            raise ValueError("alignment checkpoint was trained under a different manifest")
    bundle = load_research_octo_from_config(config, args.checkpoint, seed=0)
    equivalence = assert_zero_adapter_equivalence(bundle)
    spec, _ = build_adapter_spec_and_factors(bundle, token_width=int(config["adapter"]["token_width"]))
    if spec != checkpoints["aligned"].adapter_spec:
        raise ValueError("live Octo adapter layout differs from the alignment checkpoints")
    train_tasks = tuple(checkpoints["aligned"].metadata["train_task_ids"])
    evaluated = PopulationIndex(manifest, paths, selected)
    training = PopulationIndex(manifest, paths, train_tasks).load_all_means()

    commit, dirty = git_state(REPO_ROOT)
    warmup_steps = int(config["simulator"]["warmup_steps"])
    # Everything that determines the episodes. A run resumes only under the
    # identical identity; provenance of each attempt is recorded separately.
    identity = {
        "experiment": manifest.name,
        "manifest_sha256": manifest.sha256(),
        "fold": fold.name,
        "alignment_seed": args.seed,
        "scope": args.scope,
        "tier": args.tier,
        "tasks": list(selected),
        "conditions": list(conditions),
        "rollouts_per_task": rollout_count,
        "rollout_seed": manifest.rollouts.seed,
        "max_episode_steps": max_steps,
        "warmup_steps": warmup_steps,
        "bundle": bundle.identity(),
        "alignment_checkpoints": {
            condition: dict(checkpoint.metadata["files"])
            for condition, checkpoint in checkpoints.items()
        },
    }
    relative = output.relative_to(paths.root)
    run = open_rollout_run(
        output,
        identity=identity,
        tasks=selected,
        conditions=conditions,
        rollouts_per_task=rollout_count,
        preserve_root=paths.root / "failed_attempts" / relative,
        attempt={
            "command": list(sys.argv),
            "git_commit": commit,
            "git_dirty": dirty,
            "environment": environment,
            "inputs": input_hashes(args),
            "gates": {"zero_adapter_equivalence": equivalence},
        },
        details={
            "experiment": manifest.name,
            "manifest_sha256": manifest.sha256(),
            "profile": manifest.profile,
            "integrity_only": bool(checkpoints["aligned"].metadata["integrity_only"]),
            "publication_evidence": manifest.publication_evidence
            and args.scope == "held_out"
            and args.tier == "final",
            "fold": fold.name,
            "alignment_seed": args.seed,
            "scope": args.scope,
            "tier": args.tier,
            "rollout_seed": manifest.rollouts.seed,
            "max_episode_steps": max_steps,
            "pairing": "identical initial states and action-sampling seeds across conditions",
        },
    )
    if run.status == "complete":
        print(json.dumps({"ok": True, "already_complete": str(output)}, indent=2))
        return 0

    evaluators: dict[str, LiberoRolloutEvaluator] = {}
    started = perf_counter()
    try:
        for identifier in selected:
            if identifier in run.completed_tasks:
                continue
            suite, task_index = split_task_id(identifier)
            if suite not in evaluators:
                evaluators[suite] = LiberoRolloutEvaluator(
                    suite=suite,
                    expected_instructions=tuple(tasks["suites"][suite]),
                    preprocessing=bundle.preprocessing,
                    convention=bundle.action_convention,
                    warmup_steps=warmup_steps,
                    max_steps=max_steps,
                )
            states = build_policy_states(
                bundle=bundle,
                spec=spec,
                manifest=manifest,
                paths=paths,
                task=evaluated.load(identifier),
                training=training,
                checkpoints=checkpoints,
                conditions=conditions,
            )
            policies = {
                name: OctoLiberoPolicy(bundle, item["state"]) for name, item in states.items()
            }
            summaries = evaluators[suite].evaluate_policies(
                policies,
                task_index=task_index,
                rollout_count=rollout_count,
                seed=manifest.rollouts.seed,
            )
            training_run = manifest.reference_sample(identifier).run_id
            run.record_task(
                identifier,
                [
                    record
                    for name, summary in summaries.items()
                    for record in episode_records(
                        summary,
                        experiment=manifest.name,
                        fold=fold.name,
                        alignment_seed=args.seed,
                        scope=args.scope,
                        condition=name,
                        training_run=f"{training_run}|alignment_seed_{args.seed}",
                    )
                ],
                {
                    name: {
                        **item["record"],
                        "policy_checkpoint_sha256": policies[name].checkpoint_sha256,
                    }
                    for name, item in states.items()
                },
            )
        base_check = assert_base_parameters_unchanged(bundle)
        summary = summarize_rollouts(
            run.records,
            resamples=int(config["evaluation"]["bootstrap_resamples"]),
            confidence=float(config["evaluation"]["confidence_level"]),
            seed=manifest.rollouts.seed,
            minimum_tasks=args.minimum_tasks,
        )
        summary.update(
            {
                "experiment": manifest.name,
                "fold": fold.name,
                "alignment_seed": args.seed,
                "scope": args.scope,
                "tier": args.tier,
                "rollouts_per_task": rollout_count,
                "tasks": list(selected),
                "integrity": {"status": "pass", "base_parameters": base_check},
                "attempts": len(run.manifest["attempts"]),
                "wall_time_seconds_last_attempt": perf_counter() - started,
            }
        )
        run.finish(summary)
        if args.scope == "held_out":
            record_held_out_results(manifest, paths, artifact=str(relative))
    except BaseException as exc:
        run.fail(exc)
        raise
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "status": run.status,
                "episodes": len(run.records),
                "task_balanced_success_rate": {
                    name: value["task_balanced_success_rate"]["mean"]
                    for name, value in summary["conditions"].items()
                },
                "wall_time_seconds_last_attempt": summary["wall_time_seconds_last_attempt"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
