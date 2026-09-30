#!/usr/bin/env python3
"""Gate scaling on real one-task Octo learning and deterministic LIBERO rollouts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsl_vla.contracts import RunManifest
from wsl_vla.data.libero import StrictLiberoHDF5, load_suite_manifest
from wsl_vla.data.octo_batches import ActionNormalization
from wsl_vla.data.streams import TaskTrainingStream, make_octo_batch_factory
from wsl_vla.evaluation.gates import one_task_learning_gate
from wsl_vla.evaluation.libero_rollout import LiberoRolloutEvaluator, OctoLiberoPolicy
from wsl_vla.experiments.protocol import (
    LIBERO_GIT_REVISION,
    REQUIRED_SUITES,
    load_yaml,
    validate_reference_tasks,
    validate_research_config,
)
from wsl_vla.experiments.provenance import (
    assert_installed_vcs_revision,
    capture_environment,
    capture_hardware,
    git_state,
    sha256_array_tree,
    sha256_file,
    write_manifest_atomic,
)
from wsl_vla.octo.bridge import (
    OCTO_GIT_REVISION,
    assert_zero_adapter_equivalence,
    load_research_octo,
)
from wsl_vla.octo.continual import train_shared_lora_stage
from wsl_vla.octo.training import initial_adapter_state


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def summary_payload(summary) -> dict:
    return {
        "suite": summary.suite,
        "task_id": summary.task_id,
        "task_index": summary.task_index,
        "policy_checkpoint_sha256": summary.policy_checkpoint_sha256,
        "rollout_count": summary.rollout_count,
        "successes": summary.successes,
        "success_rate": summary.success_rate,
        "episodes": [asdict(episode) for episode in summary.episodes],
    }


def build_task_stream(*, task_id, instruction, data_file, config, bundle):
    dataset = StrictLiberoHDF5(
        data_file,
        require_wrist_camera=bool(config["data"]["require_wrist_camera"]),
        required_proprio_keys=tuple(config["data"]["proprio_keys"]),
    )
    split = dataset.split(seed=int(config["data"]["split_seed"]))
    episodes = tuple(dataset.iter_episodes(split.train))
    normalization = ActionNormalization.fit(episodes)
    example = bundle.pretrained_model.example_batch
    observation = example["observation"]
    primary = np.asarray(observation["image_primary"])
    stream = TaskTrainingStream(
        task_id=task_id,
        instruction=instruction,
        episodes=episodes,
        normalization=normalization,
        window_size=int(np.asarray(observation["timestep_pad_mask"]).shape[1]),
        action_horizon=int(np.asarray(example["action"]).shape[-2]),
        image_size=(int(primary.shape[-3]), int(primary.shape[-2])),
    )
    return stream


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=REQUIRED_SUITES)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--output-root", default="research_results/gates/one_task")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--rollouts", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("the official one-task gate must run under Linux/WSL2")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() not in {"false", "0"}:
        raise RuntimeError("set XLA_PYTHON_CLIENT_PREALLOCATE=false for the <=8 GB profile")
    if not 0 <= args.task_index < 10:
        raise ValueError("task index must lie in [0, 9]")

    try:
        import flax.serialization
        import jax
    except ImportError as exc:  # pragma: no cover - official environment
        raise RuntimeError("the one-task gate requires the research dependencies") from exc

    config = load_yaml(args.config)
    tasks = load_yaml(args.tasks)
    validate_research_config(config)
    validate_reference_tasks(tasks)
    if args.seed not in config["evaluation"]["training_seeds"]:
        raise ValueError("seed is outside the locked evaluation protocol")
    assert_installed_vcs_revision("octo", OCTO_GIT_REVISION)
    assert_installed_vcs_revision("libero", LIBERO_GIT_REVISION)

    suite_tasks = tuple(tasks["suites"][args.suite])
    data_root = Path(args.data_root or config["data"]["root"])
    data_files = load_suite_manifest(data_root / args.suite, suite_tasks)
    data_file = data_files[args.task_index]
    task_id = f"{args.suite}_{args.task_index}"
    instruction = suite_tasks[args.task_index]
    steps = args.steps or int(config["continual_learning"]["train_steps_per_task"])
    batch_size = args.batch_size or int(config["continual_learning"]["micro_batch_size"])
    rollouts = args.rollouts or int(config["continual_learning"]["development_rollouts"])
    if min(steps, batch_size, rollouts) <= 0 or args.learning_rate <= 0:
        raise ValueError("steps, batch size, rollouts, and learning rate must be positive")

    run_id = f"one-task-{args.suite}-{args.task_index}-{args.seed}-{uuid.uuid4().hex[:12]}"
    run_root = Path(args.output_root) / args.suite / f"task_{args.task_index}" / f"seed_{args.seed}"
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite an existing gate run: {run_root}")
    run_root.mkdir(parents=True)
    commit, dirty = git_state(Path(__file__).resolve().parents[1])
    manifest = None
    manifest_path = run_root / "manifest.json"

    try:
        bundle = load_research_octo(
            args.checkpoint,
            rank=int(config["adapter"]["rank"]),
            alpha=float(config["adapter"]["alpha"]),
            seed=args.seed,
        )
        assert_zero_adapter_equivalence(bundle)
        stream = build_task_stream(
            task_id=task_id,
            instruction=instruction,
            data_file=data_file,
            config=config,
            bundle=bundle,
        )
        manifest = RunManifest(
            run_id=run_id,
            command=tuple(sys.argv),
            git_commit=commit,
            git_dirty=dirty,
            base_model_id=str(config["model"]["id"]),
            base_revision=bundle.base_revision,
            base_sha256=bundle.base_sha256,
            dataset_sha256={task_id: sha256_file(data_file)},
            task_order=(task_id,),
            seeds=(args.seed,),
            configuration={
                "gate": "one_task_learning",
                "research_config": str(Path(args.config).resolve()),
                "research_config_sha256": sha256_file(args.config),
                "task_config": str(Path(args.tasks).resolve()),
                "task_config_sha256": sha256_file(args.tasks),
                "task_index": args.task_index,
                "instruction": instruction,
                "adapter": dict(config["adapter"]),
                "steps": steps,
                "learning_rate": args.learning_rate,
                "micro_batch_size": batch_size,
                "gradient_accumulation_steps": int(
                    config["continual_learning"]["gradient_accumulation_steps"]
                ),
                "rollouts": rollouts,
                "determinism_repetitions": 2,
                "action_normalization": stream.normalization.to_dict(),
            },
            hardware=capture_hardware(),
            environment=capture_environment(("jax", "flax", "optax", "octo", "libero")),
        )
        write_manifest_atomic(manifest, manifest_path)
        evaluator = LiberoRolloutEvaluator(
            suite=args.suite,
            expected_instructions=suite_tasks,
            camera_size=(
                int(config["simulator"]["camera_height"]),
                int(config["simulator"]["camera_width"]),
            ),
            warmup_steps=int(config["simulator"]["warmup_steps"]),
            max_steps=int(config["simulator"]["max_episode_steps"]),
        )

        def evaluate(adapter_state):
            policy = OctoLiberoPolicy(
                bundle,
                adapter_state,
                action_mean=stream.normalization.mean,
                action_std=stream.normalization.std,
            )
            return evaluator.evaluate_task(
                policy,
                task_index=args.task_index,
                rollout_count=rollouts,
                seed=args.seed,
            )

        baseline_state = initial_adapter_state(bundle)
        baseline_first = evaluate(baseline_state)
        baseline_repeat = evaluate(baseline_state)
        factory = make_octo_batch_factory(
            stream,
            batch_size=batch_size,
            seed=args.seed,
            text_processor=bundle.pretrained_model.text_processor,
            example_batch=bundle.pretrained_model.example_batch,
        )
        trained = train_shared_lora_stage(
            bundle,
            baseline_state,
            factory,
            seed=args.seed,
            steps=steps,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=int(
                config["continual_learning"]["gradient_accumulation_steps"]
            ),
        )
        if trained.checkpoint_sha256 == sha256_array_tree(jax.device_get(baseline_state)):
            raise AssertionError("adapter training did not change the zero-adapter state")
        adapted_first = evaluate(trained.adapter_state)
        adapted_repeat = evaluate(trained.adapter_state)
        result = one_task_learning_gate(
            baseline_first=baseline_first,
            baseline_repeat=baseline_repeat,
            adapted_first=adapted_first,
            adapted_repeat=adapted_repeat,
            enforce=False,
        )
        state_path = run_root / "trained_adapter.msgpack"
        atomic_bytes(
            state_path,
            flax.serialization.to_bytes(jax.device_get(trained.adapter_state)),
        )
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "passed": result.passed,
            "decision": asdict(result),
            "training": {
                "final_task_loss": trained.final_task_loss,
                "update_steps": trained.update_steps,
                "micro_steps": trained.micro_steps,
                "adapter_bytes": trained.adapter_bytes,
                "checkpoint_sha256": trained.checkpoint_sha256,
                "state_file": str(state_path.resolve()),
                "state_file_sha256": sha256_file(state_path),
                "training_episode_ids": [episode.episode_id for episode in stream.episodes],
                "action_normalization": stream.normalization.to_dict(),
            },
            "baseline_first": summary_payload(baseline_first),
            "baseline_repeat": summary_payload(baseline_repeat),
            "adapted_first": summary_payload(adapted_first),
            "adapted_repeat": summary_payload(adapted_repeat),
        }
        atomic_bytes(
            run_root / "one_task_gate.json",
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        write_manifest_atomic(manifest, manifest_path)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if result.passed else 1
    except Exception as exc:
        if manifest is not None:
            manifest.failures.append({"type": type(exc).__name__, "message": str(exc)})
            manifest.finished_at = datetime.now(timezone.utc).isoformat()
            write_manifest_atomic(manifest, manifest_path)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
