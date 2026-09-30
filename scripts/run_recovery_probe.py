#!/usr/bin/env python3
"""Measure reference-style recovery efficiency from consecutive continual checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsl_vla.data.libero import StrictLiberoHDF5, load_suite_manifest
from wsl_vla.data.octo_batches import ActionNormalization
from wsl_vla.data.streams import TaskTrainingStream, make_octo_batch_factory
from wsl_vla.evaluation.libero_rollout import LiberoRolloutEvaluator, OctoLiberoPolicy
from wsl_vla.evaluation.records import load_evaluation_records
from wsl_vla.evaluation.recovery import (
    RecoveryCurvePoint,
    recovery_efficiency,
    recovery_step_schedule,
)
from wsl_vla.experiments.protocol import (
    LIBERO_GIT_REVISION,
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
    sha256_directory,
    sha256_file,
)
from wsl_vla.octo.bridge import (
    OCTO_GIT_REVISION,
    assert_zero_adapter_equivalence,
    load_research_octo,
)
from wsl_vla.octo.continual import train_shared_lora_stage


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def task_stream(*, task_id, instruction, data_file, config, bundle):
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
    return TaskTrainingStream(
        task_id=task_id,
        instruction=instruction,
        episodes=episodes,
        normalization=normalization,
        window_size=int(np.asarray(observation["timestep_pad_mask"]).shape[1]),
        action_horizon=int(np.asarray(example["action"]).shape[-2]),
        image_size=(int(primary.shape[-3]), int(primary.shape[-2])),
    )


def load_stage_adapter(run_root: Path, stage: int, *, bundle):
    try:
        import flax.serialization
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - official environment
        raise RuntimeError("recovery probing requires JAX/Flax") from exc
    matches = tuple((run_root / "checkpoints").glob(f"stage_{stage:02d}_*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one checkpoint directory for stage {stage}")
    metadata_path = matches[0] / "metadata.json"
    state_path = matches[0] / "state.msgpack"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("base_sha256") != bundle.base_sha256:
        raise ValueError("continual stage metadata is incompatible with the live Octo base")
    if metadata.get("state_file_sha256") != sha256_file(state_path):
        raise ValueError("continual stage state file hash mismatch")
    payload = flax.serialization.msgpack_restore(state_path.read_bytes())
    if "adapter_state" not in payload:
        raise ValueError("continual stage state lacks adapter_state")
    state = jax.tree_util.tree_map(jnp.asarray, payload["adapter_state"])
    if sha256_array_tree(jax.device_get(state)) != metadata.get("checkpoint_sha256"):
        raise ValueError("continual stage adapter identity mismatch")
    return state, metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("continual_run")
    parser.add_argument("--transitions", type=int, nargs="+", default=tuple(range(9)))
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--output-root", default="research_results/recovery")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("official recovery probes must run under Linux/WSL2")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() not in {"false", "0"}:
        raise RuntimeError("set XLA_PYTHON_CLIENT_PREALLOCATE=false for the <=8 GB profile")
    transitions = tuple(int(value) for value in args.transitions)
    if transitions != tuple(sorted(set(transitions))) or any(not 0 <= value < 9 for value in transitions):
        raise ValueError("transitions must be unique increasing task indices in [0, 8]")

    config = load_yaml(args.config)
    tasks = load_yaml(args.tasks)
    validate_research_config(config)
    validate_reference_tasks(tasks)
    assert_installed_vcs_revision("octo", OCTO_GIT_REVISION)
    assert_installed_vcs_revision("libero", LIBERO_GIT_REVISION)
    run_root = Path(args.continual_run)
    manifest_path = run_root / "manifest.json"
    record_path = run_root / "evaluation.jsonl"
    if not manifest_path.is_file() or not record_path.is_file():
        raise FileNotFoundError("recovery requires a completed continual run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("failures") or not manifest.get("finished_at"):
        raise ValueError("recovery requires a successfully completed continual run")
    suite = str(manifest["task_order"][0]).rsplit("_", 1)[0]
    seed = int(manifest["seeds"][0])
    condition = str(manifest["configuration"]["condition"]["name"])
    optimization_space = str(
        manifest["configuration"]["condition"]["optimization_space"]
    )
    if optimization_space != "lora":
        raise ValueError(
            "the reference recovery probe currently requires a LoRA-space continual run"
        )
    if bool(manifest["configuration"]["condition"].get("independent_per_task")):
        raise ValueError("recovery requires one evolving shared adapter, not the oracle protocol")
    if len(manifest["task_order"]) != 10 or manifest["configuration"].get("task_count", 10) != 10:
        raise ValueError("recovery requires a complete ten-task continual run")
    suite_tasks = tuple(tasks["suites"][suite])
    data_root = Path(args.data_root or config["data"]["root"])
    data_files = load_suite_manifest(data_root / suite, suite_tasks)
    records = load_evaluation_records(record_path)
    if any(item.run_id != manifest["run_id"] for item in records):
        raise ValueError("continual records and manifest use different run identities")

    bundle = load_research_octo(
        args.checkpoint,
        rank=int(config["adapter"]["rank"]),
        alpha=float(config["adapter"]["alpha"]),
        seed=seed,
    )
    assert_zero_adapter_equivalence(bundle)
    if bundle.base_sha256 != manifest["base_sha256"]:
        raise ValueError("continual run and live Octo use different base checkpoints")
    evaluator = LiberoRolloutEvaluator(
        suite=suite,
        expected_instructions=suite_tasks,
        camera_size=(
            int(config["simulator"]["camera_height"]),
            int(config["simulator"]["camera_width"]),
        ),
        warmup_steps=int(config["simulator"]["warmup_steps"]),
        max_steps=int(config["simulator"]["max_episode_steps"]),
    )
    output = Path(args.output_root) / suite / f"seed_{seed}" / condition
    if output.exists():
        raise FileExistsError(f"refusing to overwrite recovery results: {output}")
    output.mkdir(parents=True)
    learning_rate = float(manifest["configuration"]["learning_rate"])
    batch_size = int(manifest["configuration"]["micro_batch_size"])
    accumulation = int(manifest["configuration"]["gradient_accumulation_steps"])
    if min(learning_rate, batch_size, accumulation) <= 0:
        raise ValueError("source continual run has invalid optimization settings")
    probe_id = f"recovery-{suite}-{seed}-{condition}-{uuid.uuid4().hex[:12]}"
    results = []
    source_run_sha256 = sha256_directory(run_root)

    for task_index in transitions:
        task_id = f"{suite}_{task_index}"
        original = next(
            item
            for item in records
            if item.training_stage == task_index
            and item.evaluated_task_index == task_index
            and item.condition == condition
        )
        forgotten = next(
            item
            for item in records
            if item.training_stage == task_index + 1
            and item.evaluated_task_index == task_index
            and item.condition == condition
        )
        if original.training_update_steps <= 0:
            raise ValueError("continual records lack original training update counts")
        state, stage_metadata = load_stage_adapter(run_root, task_index + 1, bundle=bundle)
        if (
            stage_metadata.get("condition") != condition
            or stage_metadata.get("task_id") != f"{suite}_{task_index + 1}"
        ):
            raise ValueError("stage k+1 metadata differs from the requested transition")
        if stage_metadata["checkpoint_sha256"] != forgotten.checkpoint_sha256:
            raise ValueError("stage k+1 checkpoint differs from its evaluation record")
        stream = task_stream(
            task_id=task_id,
            instruction=suite_tasks[task_index],
            data_file=data_files[task_index],
            config=config,
            bundle=bundle,
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
                task_index=task_index,
                rollout_count=original.rollout_count,
                seed=seed,
            )

        initial_summary = evaluate(state)
        if (
            initial_summary.successes != forgotten.successes
            or initial_summary.initialization_indices != tuple(forgotten.initialization_indices)
            or tuple(item.episode_seed for item in initial_summary.episodes)
            != tuple(forgotten.rollout_seeds)
        ):
            raise AssertionError("step-zero recovery rollout does not reproduce stage k+1")
        points = [RecoveryCurvePoint(0, initial_summary.successes, initial_summary.rollout_count)]
        point_payloads = [
            {
                **asdict(points[0]),
                "success_rate": points[0].success_rate,
                "task_loss": None,
                "checkpoint_sha256": initial_summary.policy_checkpoint_sha256,
                "initialization_indices": list(initial_summary.initialization_indices),
                "rollout_seeds": [item.episode_seed for item in initial_summary.episodes],
            }
        ]
        schedule = recovery_step_schedule(original.training_update_steps)
        if original.success_rate > 0 and points[0].success_rate < original.success_rate:
            factory = make_octo_batch_factory(
                stream,
                batch_size=batch_size,
                seed=seed * 10_000 + task_index,
                text_processor=bundle.pretrained_model.text_processor,
                example_batch=bundle.pretrained_model.example_batch,
            )

            def checkpoint_sink(update_steps, adapter_state, task_loss):
                summary = evaluate(adapter_state)
                point = RecoveryCurvePoint(
                    update_steps, summary.successes, summary.rollout_count
                )
                points.append(point)
                point_payloads.append(
                    {
                        **asdict(point),
                        "success_rate": point.success_rate,
                        "task_loss": task_loss,
                        "checkpoint_sha256": summary.policy_checkpoint_sha256,
                        "initialization_indices": list(summary.initialization_indices),
                        "rollout_seeds": [item.episode_seed for item in summary.episodes],
                    }
                )
                return point.success_rate >= original.success_rate

            train_shared_lora_stage(
                bundle,
                state,
                factory,
                seed=seed * 10_000 + task_index,
                steps=original.training_update_steps,
                learning_rate=learning_rate,
                gradient_accumulation_steps=accumulation,
                checkpoint_steps=schedule,
                checkpoint_sink=checkpoint_sink,
            )
        efficiency = recovery_efficiency(
            original_peak_success_rate=original.success_rate,
            original_training_steps=original.training_update_steps,
            curve=points,
        )
        results.append(
            {
                "schema_version": 1,
                "probe_id": probe_id,
                "source_run_id": manifest["run_id"],
                "source_run_sha256": source_run_sha256,
                "suite": suite,
                "seed": seed,
                "condition": condition,
                "recovered_task_index": task_index,
                "recovered_task_id": task_id,
                "post_task_index": task_index + 1,
                "original_peak_success_rate": original.success_rate,
                "post_task_success_rate": forgotten.success_rate,
                "original_training_steps": original.training_update_steps,
                "recovered_at_steps": efficiency.recovered_at_steps,
                "recovery_step_ratio": efficiency.recovery_step_ratio,
                "recovered": efficiency.recovered,
                "eligible": efficiency.eligible,
                "schedule": list(schedule),
                "curve": point_payloads,
                "dataset_sha256": sha256_file(data_files[task_index]),
                "training_episode_ids": [item.episode_id for item in stream.episodes],
            }
        )
        atomic_json(output / f"transition_{task_index:02d}.json", results[-1])

    summary = {
        "schema_version": 1,
        "probe_id": probe_id,
        "source_run": str(run_root.resolve()),
        "suite": suite,
        "seed": seed,
        "condition": condition,
        "transition_count": len(results),
        "eligible_count": sum(item["eligible"] for item in results),
        "recovered_count": sum(item["recovered"] for item in results),
        "unrecovered_eligible_count": sum(
            item["eligible"] and not item["recovered"] for item in results
        ),
        "mean_recovery_step_ratio_recovered_only": (
            float(np.mean([item["recovery_step_ratio"] for item in results if item["recovered"]]))
            if any(item["recovered"] for item in results)
            else None
        ),
        "results": results,
        "provenance": {
            "command": sys.argv,
            "git": dict(zip(("commit", "dirty"), git_state(Path(__file__).resolve().parents[1]))),
            "hardware": capture_hardware(),
            "environment": capture_environment(("jax", "flax", "optax", "octo", "libero")),
            "base_sha256": bundle.base_sha256,
            "source_run_sha256": source_run_sha256,
            "config_sha256": sha256_file(args.config),
            "tasks_sha256": sha256_file(args.tasks),
            "learning_rate": learning_rate,
            "micro_batch_size": batch_size,
            "gradient_accumulation_steps": accumulation,
        },
    }
    atomic_json(output / "recovery_summary.json", summary)
    print(json.dumps({"ok": True, **{key: summary[key] for key in ("probe_id", "transition_count", "eligible_count", "recovered_count", "unrecovered_eligible_count", "mean_recovery_step_ratio_recovered_only")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
