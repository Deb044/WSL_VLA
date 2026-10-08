#!/usr/bin/env python3
"""Evaluate one gamma/patience candidate on two locked source-suite tasks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import numpy as np

def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.alignment.checkpoint import encode_and_map_evidence, load_alignment_checkpoint
from wsl_vla.alignment.evidence_io import load_task_evidence
from wsl_vla.contracts import AdapterSpec, RunManifest
from wsl_vla.data.libero import StrictLiberoHDF5, load_suite_manifest
from wsl_vla.data.octo_batches import ActionNormalization
from wsl_vla.data.streams import TaskTrainingStream, make_octo_batch_factory
from wsl_vla.evaluation.libero_rollout import LiberoRolloutEvaluator, OctoLiberoPolicy
from wsl_vla.evaluation.metrics import continual_learning_metrics
from wsl_vla.evaluation.records import append_evaluation_record, records_to_success_matrix
from wsl_vla.evaluation.sequential import StageUpdate, run_sequential_protocol
from wsl_vla.experiments.gamma_selection import (
    GammaValidationRecord,
    append_gamma_validation_record,
)
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
    sha256_directory,
    sha256_file,
    write_manifest_atomic,
)
from wsl_vla.octo.bridge import (
    OCTO_GIT_REVISION,
    assert_zero_adapter_equivalence,
    build_adapter_spec_and_factors,
    load_research_octo,
)
from wsl_vla.octo.continual import train_shared_latent_stage
from wsl_vla.octo.training import initial_adapter_state


@dataclass(frozen=True)
class ValidationPolicyState:
    adapter_state: Mapping[str, Any]
    latents: Mapping[str, Any] | None


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out-suite", required=True, choices=REQUIRED_SUITES)
    parser.add_argument("--source-suite", required=True, choices=REQUIRED_SUITES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--family", required=True, choices=("proposed", "uniform"))
    parser.add_argument("--gamma-vision", required=True, type=float)
    parser.add_argument("--gamma-language", required=True, type=float)
    parser.add_argument("--gamma-action", required=True, type=float)
    parser.add_argument("--patience", required=True, type=int)
    parser.add_argument("--alignment-checkpoint", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--evidence-template", default="{suite}/{task_id}.npz")
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--output-root", default="research_results/gamma_validation")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--rollouts", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def _stream_pair(task_id, instruction, data_file, config, bundle):
    dataset = StrictLiberoHDF5(
        data_file,
        require_wrist_camera=bool(config["data"]["require_wrist_camera"]),
        required_proprio_keys=tuple(config["data"]["proprio_keys"]),
    )
    split = dataset.split(seed=int(config["data"]["split_seed"]))
    training_episodes = tuple(dataset.iter_episodes(split.train))
    validation_episodes = tuple(dataset.iter_episodes(split.validation))
    if not validation_episodes:
        raise ValueError("gamma early stopping requires validation episodes")
    normalization = ActionNormalization.fit(training_episodes)
    example = bundle.pretrained_model.example_batch
    observation = example["observation"]
    primary = np.asarray(observation["image_primary"])
    common = dict(
        task_id=task_id,
        instruction=instruction,
        normalization=normalization,
        window_size=int(np.asarray(observation["timestep_pad_mask"]).shape[1]),
        action_horizon=int(np.asarray(example["action"]).shape[-2]),
        image_size=(int(primary.shape[-3]), int(primary.shape[-2])),
    )
    return (
        TaskTrainingStream(episodes=training_episodes, **common),
        TaskTrainingStream(episodes=validation_episodes, **common),
    )


def main() -> int:
    args = _parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("official gamma validation must run under Linux/WSL2")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() not in {"false", "0"}:
        raise RuntimeError("set XLA_PYTHON_CLIENT_PREALLOCATE=false for the <=8 GB profile")
    if args.source_suite == args.held_out_suite:
        raise ValueError("held-out test suite cannot be used for gamma selection")
    import jax

    config = load_yaml(args.config)
    tasks = load_yaml(args.tasks)
    validate_research_config(config)
    validate_reference_tasks(tasks)
    if args.seed not in config["evaluation"]["training_seeds"]:
        raise ValueError("seed is outside the locked three-seed protocol")
    assert_installed_vcs_revision("octo", OCTO_GIT_REVISION)
    assert_installed_vcs_revision("libero", LIBERO_GIT_REVISION)
    gammas = {
        "vision": float(args.gamma_vision),
        "language": float(args.gamma_language),
        "action": float(args.gamma_action),
    }
    if min(gammas.values()) < 0:
        raise ValueError("gamma values cannot be negative")
    if args.family == "uniform" and len(set(gammas.values())) != 1:
        raise ValueError("uniform validation requires equal modality gammas")
    if args.family == "proposed" and not (
        gammas["vision"] > gammas["action"]
        and gammas["language"] > gammas["action"]
    ):
        raise ValueError("proposed validation requires vision/language gamma > action gamma")
    if args.patience <= 0 or args.early_stopping_min_delta < 0:
        raise ValueError("patience must be positive and min_delta non-negative")

    alignment = load_alignment_checkpoint(args.alignment_checkpoint)
    alignment_sha256 = sha256_directory(args.alignment_checkpoint)
    if alignment.metadata.get("held_out_suite") != args.held_out_suite:
        raise ValueError("alignment checkpoint belongs to another held-out fold")
    if float(alignment.metadata.get("model", {}).get("contrastive_weight", 0)) <= 0:
        raise ValueError("gamma validation requires the aligned latent checkpoint")
    validation_indices = tuple(int(value) for value in alignment.metadata["validation_task_indices"])
    if validation_indices != (8, 9):
        raise ValueError("gamma validation requires locked source task indices 8 and 9")
    instructions = tuple(tasks["suites"][args.source_suite][index] for index in validation_indices)
    task_ids = tuple(f"{args.source_suite}_{index}" for index in validation_indices)
    suite_files = load_suite_manifest(
        Path(args.data_root or config["data"]["root"]) / args.source_suite,
        tasks["suites"][args.source_suite],
    )
    data_files = tuple(suite_files[index] for index in validation_indices)
    dataset_hashes = {task_id: sha256_file(path) for task_id, path in zip(task_ids, data_files)}

    bundle = load_research_octo(
        args.checkpoint,
        rank=int(config["adapter"]["rank"]),
        alpha=float(config["adapter"]["alpha"]),
        seed=args.seed,
    )
    assert_zero_adapter_equivalence(bundle)
    adapter_spec = AdapterSpec.from_dict(alignment.metadata["adapter_spec"])
    runtime_spec, _ = build_adapter_spec_and_factors(
        bundle, token_width=int(config["adapter"]["token_width"])
    )
    if adapter_spec != runtime_spec or bundle.base_sha256 != alignment.metadata["base_sha256"]:
        raise ValueError("live Octo base/layout differs from alignment checkpoint")

    max_steps = args.max_steps or int(config["continual_learning"]["train_steps_per_task"])
    batch_size = args.batch_size or int(config["continual_learning"]["micro_batch_size"])
    rollouts = args.rollouts or int(config["continual_learning"]["development_rollouts"])
    if min(max_steps, batch_size, rollouts) <= 0:
        raise ValueError("steps, batch size, and rollouts must be positive")
    candidate_payload = {
        "held_out_suite": args.held_out_suite,
        "source_suite": args.source_suite,
        "seed": args.seed,
        "family": args.family,
        "gammas": gammas,
        "patience": args.patience,
        "min_delta": args.early_stopping_min_delta,
        "alignment_checkpoint_sha256": alignment_sha256,
    }
    candidate_id = hashlib.sha256(
        json.dumps(candidate_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    run_id = f"gamma-{args.held_out_suite}-{args.source_suite}-{args.seed}-{candidate_id}"
    run_root = (
        Path(args.output_root)
        / args.held_out_suite
        / args.source_suite
        / f"seed_{args.seed}"
        / candidate_id
    )
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite gamma validation run: {run_root}")
    run_root.mkdir(parents=True)
    evaluation_path = run_root / "evaluation.jsonl"
    gamma_record_path = run_root / "gamma_validation.jsonl"
    commit, dirty = git_state(REPO_ROOT)
    manifest = RunManifest(
        run_id=run_id,
        command=tuple(sys.argv),
        git_commit=commit,
        git_dirty=dirty,
        base_model_id=str(config["model"]["id"]),
        base_revision=bundle.base_revision,
        base_sha256=bundle.base_sha256,
        dataset_sha256=dataset_hashes,
        task_order=task_ids,
        seeds=(args.seed,),
        configuration={
            **candidate_payload,
            "research_config": str(Path(args.config).resolve()),
            "research_config_sha256": sha256_file(args.config),
            "task_config": str(Path(args.tasks).resolve()),
            "task_config_sha256": sha256_file(args.tasks),
            "max_steps_per_stage": max_steps,
            "learning_rate": args.learning_rate,
            "micro_batch_size": batch_size,
            "rollouts_per_cell": rollouts,
        },
        hardware=capture_hardware(),
        environment=capture_environment(("jax", "flax", "optax", "octo", "libero")),
    )
    manifest_path = run_root / "manifest.json"
    write_manifest_atomic(manifest, manifest_path)

    evaluator = LiberoRolloutEvaluator(
        suite=args.source_suite,
        expected_instructions=tasks["suites"][args.source_suite],
        camera_size=(
            int(config["simulator"]["camera_height"]),
            int(config["simulator"]["camera_width"]),
        ),
        warmup_steps=int(config["simulator"]["warmup_steps"]),
        max_steps=int(config["simulator"]["max_episode_steps"]),
    )
    normalizations = {}
    update_steps = {}

    def train_stage(previous, task_id, local_stage):
        train_stream, validation_stream = _stream_pair(
            task_id,
            instructions[local_stage],
            data_files[local_stage],
            config,
            bundle,
        )
        normalizations[task_id] = train_stream.normalization
        relative = args.evidence_template.format(
            suite=args.source_suite,
            task_id=task_id,
            task_index=validation_indices[local_stage],
        )
        evidence = load_task_evidence(Path(args.evidence_root) / relative)
        if evidence.task_id != task_id or evidence.suite != args.source_suite:
            raise ValueError("gamma-validation evidence identity mismatch")
        if evidence.vision_mask is None:
            raise ValueError("gamma-validation visual evidence requires a mask")
        if evidence.raw_feature_references.get("dataset_sha256") != dataset_hashes[task_id]:
            raise ValueError("gamma-validation evidence uses different task data")
        if evidence.raw_feature_references.get("base_sha256") != bundle.base_sha256:
            raise ValueError("gamma-validation evidence uses a different Octo base")
        if evidence.episode_ids != tuple(item.episode_id for item in train_stream.episodes):
            raise ValueError("gamma-validation evidence uses different training episodes")
        initial_latents = previous.latents
        if initial_latents is None:
            initial_latents = encode_and_map_evidence(
                alignment,
                vision_features=evidence.vision_features,
                vision_mask=evidence.vision_mask,
                language_features=evidence.language_features,
                action_features=evidence.action_statistics,
            )
        train_factory = make_octo_batch_factory(
            train_stream,
            batch_size=batch_size,
            seed=args.seed * 100 + local_stage,
            text_processor=bundle.pretrained_model.text_processor,
            example_batch=bundle.pretrained_model.example_batch,
        )
        validation_factory = make_octo_batch_factory(
            validation_stream,
            batch_size=batch_size,
            seed=args.seed * 100 + local_stage + 50_000,
            text_processor=bundle.pretrained_model.text_processor,
            example_batch=bundle.pretrained_model.example_batch,
        )
        train_key = jax.random.PRNGKey(args.seed * 100 + local_stage)
        validation_key = jax.random.PRNGKey(args.seed * 100 + local_stage + 50_000)
        train_counter = 0
        validation_counter = 0

        def batches():
            nonlocal train_counter
            for batch in train_factory():
                key = jax.random.fold_in(train_key, train_counter)
                train_counter += 1
                yield batch, key

        def validation_batches():
            nonlocal validation_counter
            for batch in validation_factory():
                key = jax.random.fold_in(validation_key, validation_counter)
                validation_counter += 1
                yield batch, key

        result = train_shared_latent_stage(
            alignment,
            bundle=bundle,
            adapter_spec=adapter_spec,
            previous_latents=initial_latents,
            batches=batches,
            validation_batches=validation_batches,
            gammas=gammas,
            steps=max_steps,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        )
        update_steps[local_stage] = result.update_steps
        evidence_size = sum(
            int(np.asarray(value).nbytes)
            for value in (
                evidence.vision_features,
                evidence.vision_mask,
                evidence.language_features,
                evidence.action_statistics,
            )
        )
        return StageUpdate(
            state=ValidationPolicyState(result.adapter_state, result.latents),
            checkpoint_sha256=result.checkpoint_sha256,
            latent_drift=result.latent_drift,
            task_loss=result.final_task_loss,
            regularization_loss=result.final_regularization_loss,
            adapter_bytes=result.adapter_bytes,
            evidence_bytes=evidence_size,
            update_steps=result.update_steps,
            micro_steps=result.update_steps,
        )

    global_task_index = {
        task_id: validation_indices[local_stage]
        for local_stage, task_id in enumerate(task_ids)
    }

    def rollout(state, task_id, count, rollout_seed):
        normalization = normalizations[task_id]
        policy = OctoLiberoPolicy(
            bundle,
            state.adapter_state,
            action_mean=normalization.mean,
            action_std=normalization.std,
        )
        return evaluator.evaluate_task(
            policy,
            task_index=global_task_index[task_id],
            rollout_count=count,
            seed=rollout_seed,
        ).as_outcome()

    try:
        _, records = run_sequential_protocol(
            initial_state=ValidationPolicyState(initial_adapter_state(bundle), None),
            task_ids=task_ids,
            train_stage=train_stage,
            rollout=rollout,
            run_id=run_id,
            suite=args.source_suite,
            seed=args.seed,
            condition=f"gamma_validation_{args.family}_{candidate_id}",
            rollout_count=rollouts,
            record_sink=lambda record: append_evaluation_record(record, evaluation_path),
        )
        matrix = records_to_success_matrix(
            records,
            suite=args.source_suite,
            seed=args.seed,
            condition=f"gamma_validation_{args.family}_{candidate_id}",
            task_count=2,
        )
        metrics = continual_learning_metrics(matrix)
        gamma_record = GammaValidationRecord(
            run_id=run_id,
            held_out_suite=args.held_out_suite,
            source_suite=args.source_suite,
            seed=args.seed,
            family=args.family,
            gammas=gammas,
            early_stopping_patience=args.patience,
            average_success_rate=metrics["average_success_rate"],
            negative_backward_transfer=metrics["negative_backward_transfer"],
            validation_task_ids=task_ids,
            rollout_count=rollouts,
            stage_update_steps=tuple(update_steps[index] for index in range(2)),
            max_steps_per_stage=max_steps,
            alignment_checkpoint_sha256=alignment_sha256,
        )
        append_gamma_validation_record(gamma_record, gamma_record_path)
        manifest.configuration["action_normalization"] = {
            task_id: normalizations[task_id].to_dict() for task_id in task_ids
        }
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        write_manifest_atomic(manifest, manifest_path)
        print(json.dumps({"ok": True, **asdict(gamma_record)}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        manifest.failures.append({"type": type(exc).__name__, "message": str(exc)})
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        write_manifest_atomic(manifest, manifest_path)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
