#!/usr/bin/env python3
"""Train one independent Octo-Small zoo run (one task, one seed) from the shared base.

Task, seed, optimizer-update budget, snapshot stages, and the episode split all
come from the experiment manifest. The run is resumable: optimizer state,
random-generator counters, and the dataset cursor are saved separately from the
compact adapter snapshots, and a resume is refused unless the configuration and
data hashes match.
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

from wsl_vla.data.streams import load_split_episodes, task_stream_from_bundle
from wsl_vla.experiments.manifest import task_id as make_task_id
from wsl_vla.experiments.provenance import git_state, sha256_file
from wsl_vla.experiments.runtime import (
    add_experiment_arguments,
    environment_record,
    input_hashes,
    load_experiment,
    require_gpu_runtime_settings,
    task_data_file,
)
from wsl_vla.octo.bridge import (
    assert_base_parameters_unchanged,
    assert_zero_adapter_equivalence,
    build_adapter_spec_and_factors,
    load_research_octo_from_config,
)
from wsl_vla.octo.training import assert_adapter_gradient_reachability
from wsl_vla.octo.zoo import ZooRunIdentity, ZooTrainer, atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root")
    parser.add_argument(
        "--budget-inspection",
        action="store_true",
        help="Train a manifest inspection task to the largest inspection stage and save "
        "inspection snapshots outside the zoo population (budget validation).",
    )
    parser.add_argument("--resume-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.resume_every <= 0 or args.log_every <= 0:
        raise ValueError("resume and log cadences must be positive")
    require_gpu_runtime_settings()

    config, tasks, manifest, paths = load_experiment(args)
    identifier = make_task_id(args.suite, args.task_index)
    if identifier not in manifest.tasks:
        raise ValueError(f"{identifier} is not a task of experiment {manifest.name!r}")
    if args.seed not in manifest.seeds:
        raise ValueError(f"seed {args.seed} is not a zoo seed of experiment {manifest.name!r}")
    if args.budget_inspection:
        if identifier not in manifest.zoo.inspection_tasks:
            raise ValueError(
                f"{identifier} is not a budget-inspection source task of {manifest.name!r}"
            )
        stages = manifest.zoo.inspection_updates
        purpose = "budget_inspection"
        output_root = paths.budget_inspection
        resume_root = paths.resume / "budget_inspection"
    else:
        stages = manifest.zoo.snapshot_updates
        purpose = "zoo"
        output_root = paths.population
        resume_root = paths.resume / "zoo"
    relative = Path(args.suite) / identifier / f"seed_{args.seed}"
    run_directory = output_root / relative
    resume_directory = resume_root / relative
    summary_path = run_directory / "run.json"

    environment = environment_record()
    data_file, instruction = task_data_file(
        args.data_root or config["data"]["root"], tasks, identifier
    )
    dataset_sha256 = sha256_file(data_file)
    bundle = load_research_octo_from_config(config, args.checkpoint, seed=args.seed)
    equivalence = assert_zero_adapter_equivalence(bundle)
    split, episodes = load_split_episodes(
        data_file,
        preprocessing=bundle.preprocessing,
        split_seed=manifest.episode_split.seed,
        train_fraction=manifest.episode_split.train_fraction,
        validation_fraction=manifest.episode_split.validation_fraction,
        splits=("train", "validation"),
    )
    stream = task_stream_from_bundle(
        bundle, task_id=identifier, instruction=instruction, episodes=episodes["train"]
    )
    held_out_stream = task_stream_from_bundle(
        bundle, task_id=identifier, instruction=instruction, episodes=episodes["validation"]
    )
    token_width = int(config["adapter"]["token_width"])
    spec, _ = build_adapter_spec_and_factors(bundle, token_width=token_width)
    identity = ZooRunIdentity(
        task_id=identifier,
        suite=args.suite,
        task_index=args.task_index,
        instruction=instruction,
        seed=args.seed,
        manifest_name=manifest.name,
        zoo_sha256=manifest.zoo_sha256(),
        budget=manifest.zoo.training_identity(),
        snapshot_updates=tuple(stages),
        purpose=purpose,
        dataset_sha256=dataset_sha256,
        train_episode_ids=split.train,
        held_out_episode_ids=split.validation,
        base_sha256=bundle.base_sha256,
        base_revision=bundle.base_revision,
        architecture_sha256=bundle.architecture_sha256,
        preprocessing_sha256=bundle.preprocessing.sha256(),
        action_convention_sha256=bundle.action_convention.sha256(),
        adapter_spec_sha256=spec.sha256(),
        environment_profile=str(environment["profile"]),
    )
    if summary_path.is_file():
        recorded = json.loads(summary_path.read_text(encoding="utf-8"))
        if recorded.get("completed"):
            # A finished run is reused only if it is this run: same manifest,
            # budget, data, base, architecture, preprocessing, and adapter layout.
            if recorded.get("run_identity_sha256") != identity.sha256():
                current = identity.to_dict()
                differing = sorted(
                    key
                    for key, value in recorded.get("run_identity", {}).items()
                    if current.get(key) != value
                )
                raise FileExistsError(
                    f"{run_directory} holds a completed zoo run with a different identity "
                    f"(differing fields: {differing}). Zoo runs are never overwritten; use a new "
                    "experiment manifest name or research root."
                )
            print(json.dumps({"ok": True, "already_complete": str(summary_path)}, indent=2))
            return 0
    trainer = ZooTrainer(
        bundle=bundle,
        stream=stream,
        held_out_stream=held_out_stream,
        budget=manifest.zoo,
        identity=identity,
        token_width=token_width,
        sequence_window=int(config["adapter"]["sequence_window"]),
        gradient_clip_norm=1.0,
    )
    import jax

    gradient_gate = assert_adapter_gradient_reachability(
        bundle, trainer._held_out_batches[0], jax.random.PRNGKey(args.seed)
    )
    resumed = trainer.restore(resume_directory)
    if resumed and trainer.update_step in stages:
        # The resume state of a stage is written before its snapshot. A crash
        # between the two leaves the snapshot missing (written now, from the
        # exactly restored state) or complete (adopted after an exact check).
        trainer.ensure_snapshot(
            run_directory / f"update_{trainer.update_step}",
            snapshot_index=stages.index(trainer.update_step),
        )
    commit, dirty = git_state(REPO_ROOT)
    provenance = {
        "command": list(sys.argv),
        "git_commit": commit,
        "git_dirty": dirty,
        "environment": environment,
        "inputs": input_hashes(args),
        "manifest_sha256": manifest.sha256(),
        "bundle": bundle.identity(),
        "controller_actions": stream.normalization.to_dict(),
        "resumed_from_update": trainer.update_step if resumed else None,
        "gates": {
            "zero_adapter_equivalence": equivalence,
            "gradient_reachability": {
                key: gradient_gate[key]
                for key in (
                    "branches",
                    "minimum_initial_up_gradient_norm",
                    "maximum_initial_down_gradient_norm",
                    "minimum_post_update_down_gradient_norm",
                )
            },
        },
    }

    def on_update(current: ZooTrainer) -> None:
        step = current.update_step
        is_stage = step in stages
        # Resume state first: a snapshot is then never newer than the state a
        # restart resumes from, so no snapshot has to be recomputed. (Recomputing
        # is exact on CPU but not guaranteed on GPU without deterministic ops.)
        if is_stage or step % args.resume_every == 0:
            current.save_resume(resume_directory)
        if is_stage:
            sample_directory = run_directory / f"update_{step}"
            current.ensure_snapshot(sample_directory, snapshot_index=stages.index(step))

    try:
        trainer.train_until(max(stages), on_update=on_update, log_every=args.log_every)
        base_check = assert_base_parameters_unchanged(bundle)
        summary = trainer.run_summary(completed=True)
        summary["provenance"] = {**provenance, "base_parameters_after_training": base_check}
        atomic_write_json(summary_path, summary)
        trainer.save_resume(resume_directory)
    except BaseException as exc:
        # Preserve the failed run: history so far, snapshots already written,
        # and the failure itself stay on disk for diagnosis.
        summary = trainer.run_summary(
            completed=False,
            failures=[{"type": type(exc).__name__, "message": str(exc)}],
        )
        summary["provenance"] = provenance
        atomic_write_json(summary_path, summary)
        raise
    print(
        json.dumps(
            {
                "ok": True,
                "summary": str(summary_path),
                "samples": [item["directory"] for item in trainer.snapshots],
                "held_out_loss_frozen_base": trainer.base_held_out_loss,
                "held_out_loss_final": trainer.held_out_loss(),
                "steady_micro_steps_per_second": summary["resources"][
                    "steady_micro_steps_per_second"
                ],
                "compile_seconds": summary["resources"]["compile_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
