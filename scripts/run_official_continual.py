#!/usr/bin/env python3
"""Run one locked Methodology 1 continual-learning condition on official Octo."""

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
import tempfile
import uuid
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsl_vla.adapters.packing import effective_update
from wsl_vla.alignment.checkpoint import (
    encode_and_map_evidence,
    load_alignment_checkpoint,
)
from wsl_vla.alignment.evidence_io import load_task_evidence
from wsl_vla.contracts import AdapterSpec, RunManifest
from wsl_vla.data.libero import StrictLiberoHDF5, load_suite_manifest
from wsl_vla.data.octo_batches import ActionNormalization
from wsl_vla.data.streams import (
    TaskTrainingStream,
    make_octo_batch_factory,
    sample_replay_transitions,
)
from wsl_vla.evaluation.libero_rollout import LiberoRolloutEvaluator, OctoLiberoPolicy
from wsl_vla.evaluation.gates import two_task_success_matrix
from wsl_vla.evaluation.records import append_evaluation_record
from wsl_vla.evaluation.sequential import (
    StageUpdate,
    run_independent_oracle_protocol,
    run_sequential_protocol,
)
from wsl_vla.experiments.conditions import (
    PRIMARY_CONDITIONS,
    PerTaskReplayMemory,
    load_gamma_selection,
    resolve_condition_spec,
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
    sha256_array_tree,
    sha256_file,
    write_manifest_atomic,
)
from wsl_vla.octo.bridge import (
    OCTO_GIT_REVISION,
    assert_zero_adapter_equivalence,
    build_adapter_spec_and_factors,
    load_research_octo,
)
from wsl_vla.octo.continual import train_shared_latent_stage, train_shared_lora_stage
from wsl_vla.octo.training import initial_adapter_state


@dataclass(frozen=True)
class PolicyState:
    adapter_state: Mapping[str, Any]
    latents: Mapping[str, Any] | None = None


def _atomic_bytes(path: Path, payload: bytes) -> None:
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


def _spec_sha256(spec: AdapterSpec) -> str:
    return hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True).encode()).hexdigest()


def _validate_alignment_checkpoint(checkpoint, *, suite: str, bundle, aligned: bool):
    metadata = checkpoint.metadata
    if metadata.get("held_out_suite") != suite:
        raise ValueError("alignment checkpoint belongs to a different held-out suite")
    if metadata.get("base_sha256") != bundle.base_sha256:
        raise ValueError("alignment checkpoint and Octo bundle use different bases")
    contrastive = float(metadata.get("model", {}).get("contrastive_weight", -1))
    if aligned and contrastive <= 0:
        raise ValueError("aligned condition requires a positive contrastive loss checkpoint")
    if not aligned and contrastive != 0:
        raise ValueError("reconstruction-only condition requires contrastive_weight=0")
    spec = AdapterSpec.from_dict(metadata["adapter_spec"])
    if metadata.get("adapter_spec_sha256") != _spec_sha256(spec):
        raise ValueError("alignment checkpoint AdapterSpec digest is inconsistent")
    return spec


def _task_streams(
    *,
    task_id: str,
    instruction: str,
    data_file: Path,
    config: Mapping[str, Any],
    bundle: Any,
    require_validation: bool,
) -> tuple[TaskTrainingStream, TaskTrainingStream | None]:
    dataset = StrictLiberoHDF5(
        data_file,
        require_wrist_camera=bool(config["data"]["require_wrist_camera"]),
        required_proprio_keys=tuple(config["data"]["proprio_keys"]),
    )
    split = dataset.split(seed=int(config["data"]["split_seed"]))
    training_episodes = tuple(dataset.iter_episodes(split.train))
    validation_episodes = (
        tuple(dataset.iter_episodes(split.validation)) if require_validation else ()
    )
    if require_validation and not validation_episodes:
        raise ValueError("latent early stopping requires validation episodes")
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
        TaskTrainingStream(episodes=validation_episodes, **common)
        if require_validation
        else None,
    )


def _component_drift(bundle, previous, current, *, token_width: int) -> dict[str, float]:
    previous_spec, previous_factors = build_adapter_spec_and_factors(
        bundle, token_width=token_width, adapter_state=previous
    )
    current_spec, current_factors = build_adapter_spec_and_factors(
        bundle, token_width=token_width, adapter_state=current
    )
    if previous_spec != current_spec:
        raise ValueError("adapter layout changed between continual stages")
    sums = {"vision": 0.0, "language": 0.0, "action": 0.0}
    for entry in previous_spec.entries:
        old_down, old_up = previous_factors[entry.parameter_path]
        new_down, new_up = current_factors[entry.parameter_path]
        old = effective_update(old_down, old_up, alpha=previous_spec.alpha, rank=entry.rank)
        new = effective_update(new_down, new_up, alpha=current_spec.alpha, rank=entry.rank)
        sums[entry.component.value] += float(np.square(np.asarray(new) - np.asarray(old)).sum())
    return {name: float(np.sqrt(value)) for name, value in sums.items()}


def _memory_peaks() -> tuple[int | None, int | None]:
    peak_ram = None
    try:
        import resource

        # Linux reports KiB; this command is deliberately Linux/WSL-only.
        peak_ram = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except (ImportError, AttributeError):
        pass
    peak_vram = None
    try:
        import jax

        values = [
            int(stats["peak_bytes_in_use"])
            for device in jax.devices()
            if (stats := device.memory_stats()) and "peak_bytes_in_use" in stats
        ]
        peak_vram = max(values) if values else None
    except (ImportError, TypeError, KeyError):
        pass
    return peak_vram, peak_ram


def _save_stage(
    root: Path,
    *,
    stage: int,
    task_id: str,
    update: StageUpdate[PolicyState],
    bundle: Any,
    condition: str,
) -> None:
    try:
        import flax.serialization
        import jax
    except ImportError as exc:  # pragma: no cover - official environment
        raise RuntimeError("stage persistence requires JAX/Flax") from exc
    if sha256_array_tree(jax.device_get(update.state.adapter_state)) != update.checkpoint_sha256:
        raise ValueError("stage state does not match its declared checkpoint identity")
    directory = root / "checkpoints" / f"stage_{stage:02d}_{task_id}"
    directory.mkdir(parents=True, exist_ok=False)
    payload = {
        "adapter_state": jax.device_get(update.state.adapter_state),
        "latents": None if update.state.latents is None else jax.device_get(update.state.latents),
    }
    state_path = directory / "state.msgpack"
    _atomic_bytes(state_path, flax.serialization.to_bytes(payload))
    metadata = {
        "schema_version": 1,
        "condition": condition,
        "stage": stage,
        "task_id": task_id,
        "base_sha256": bundle.base_sha256,
        "checkpoint_sha256": update.checkpoint_sha256,
        "state_file_sha256": sha256_file(state_path),
        "latent_drift": dict(update.latent_drift),
        "task_loss": update.task_loss,
        "regularization_loss": update.regularization_loss,
        "update_steps": update.update_steps,
        "micro_steps": update.micro_steps,
    }
    _atomic_bytes(
        directory / "metadata.json",
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=REQUIRED_SUITES)
    parser.add_argument("--condition", required=True, choices=PRIMARY_CONDITIONS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gamma-selection")
    parser.add_argument("--alignment-checkpoint")
    parser.add_argument("--reconstruction-checkpoint")
    parser.add_argument("--evidence-root")
    parser.add_argument(
        "--evidence-template",
        default="{suite}/{task_id}.npz",
        help="Path below evidence-root; supports {suite}, {task_id}, and {task_index}.",
    )
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--output-root", default="research_results/continual")
    parser.add_argument("--tier", choices=("development", "publication"), default="development")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--task-count",
        type=int,
        choices=(2, 10),
        default=10,
        help="Use 2 only for the official pre-scaling shared-state gate.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("official continual experiments must run under Linux/WSL2")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() not in {"false", "0"}:
        raise RuntimeError("set XLA_PYTHON_CLIENT_PREALLOCATE=false for the <=8 GB profile")

    import jax

    config = load_yaml(args.config)
    tasks = load_yaml(args.tasks)
    validate_research_config(config)
    validate_reference_tasks(tasks)
    if args.seed not in config["evaluation"]["training_seeds"]:
        raise ValueError("seed is outside the locked three-seed evaluation protocol")
    assert_installed_vcs_revision("octo", OCTO_GIT_REVISION)
    assert_installed_vcs_revision("libero", LIBERO_GIT_REVISION)

    selection = (
        load_gamma_selection(args.gamma_selection, held_out_suite=args.suite)
        if args.gamma_selection
        else None
    )
    condition = resolve_condition_spec(args.condition, selection)
    if args.task_count == 2 and args.condition != "sequential_no_regularization":
        raise ValueError("the two-task scaling gate uses sequential_no_regularization")
    suite_tasks = tuple(tasks["suites"][args.suite])
    task_ids = tuple(f"{args.suite}_{index}" for index in range(args.task_count))
    data_root = Path(args.data_root or config["data"]["root"])
    data_files = load_suite_manifest(data_root / args.suite, suite_tasks)[: args.task_count]
    dataset_hashes = {task_id: sha256_file(path) for task_id, path in zip(task_ids, data_files)}

    bundle = load_research_octo(
        args.checkpoint,
        rank=int(config["adapter"]["rank"]),
        alpha=float(config["adapter"]["alpha"]),
        seed=args.seed,
    )
    assert_zero_adapter_equivalence(bundle)
    runtime_spec, _ = build_adapter_spec_and_factors(
        bundle, token_width=int(config["adapter"]["token_width"])
    )

    alignment = None
    adapter_spec = None
    if condition.optimization_space == "latent":
        selected_path = (
            args.reconstruction_checkpoint
            if condition.name == "reconstruction_only_latent"
            else args.alignment_checkpoint
        )
        if not selected_path or not args.evidence_root:
            raise ValueError("latent conditions require their alignment checkpoint and evidence-root")
        alignment = load_alignment_checkpoint(selected_path)
        adapter_spec = _validate_alignment_checkpoint(
            alignment,
            suite=args.suite,
            bundle=bundle,
            aligned=condition.name != "reconstruction_only_latent",
        )
        if adapter_spec != runtime_spec:
            raise ValueError("alignment AdapterSpec differs from the live Octo adapter layout")

    run_id = f"{args.suite}-{args.seed}-{args.condition}-{uuid.uuid4().hex[:12]}"
    protocol_root = "two_task_gate" if args.task_count == 2 else "ten_task_study"
    run_root = (
        Path(args.output_root)
        / protocol_root
        / args.suite
        / f"seed_{args.seed}"
        / args.condition
    )
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite an existing research run: {run_root}")
    run_root.mkdir(parents=True)
    record_path = run_root / "evaluation.jsonl"
    repo_root = Path(__file__).resolve().parents[1]
    commit, dirty = git_state(repo_root)
    rollouts = int(
        config["continual_learning"][
            "publication_rollouts" if args.tier == "publication" else "development_rollouts"
        ]
    )
    steps = args.steps or int(config["continual_learning"]["train_steps_per_task"])
    batch_size = args.batch_size or int(config["continual_learning"]["micro_batch_size"])
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
            "tier": args.tier,
            "research_config": str(Path(args.config).resolve()),
            "research_config_sha256": sha256_file(args.config),
            "task_config": str(Path(args.tasks).resolve()),
            "task_config_sha256": sha256_file(args.tasks),
            "condition": asdict(condition),
            "steps_per_task": steps,
            "learning_rate": args.learning_rate,
            "micro_batch_size": batch_size,
            "gradient_accumulation_steps": int(
                config["continual_learning"]["gradient_accumulation_steps"]
            ),
            "rollouts_per_cell": rollouts,
            "task_count": args.task_count,
            "scaling_gate": args.task_count == 2,
            "gamma_selection": (
                str(Path(args.gamma_selection).resolve()) if args.gamma_selection else None
            ),
            "gamma_selection_sha256": (
                sha256_file(args.gamma_selection) if args.gamma_selection else None
            ),
            "early_stopping_patience": (
                selection.early_stopping_patience if selection is not None else None
            ),
        },
        hardware=capture_hardware(),
        environment=capture_environment(("jax", "flax", "optax", "octo", "libero")),
    )
    manifest_path = run_root / "manifest.json"
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
    normalizations: dict[str, ActionNormalization] = {}
    replay_memory = (
        PerTaskReplayMemory(
            transitions_per_task=condition.replay_per_previous_task, seed=args.seed
        )
        if condition.replay_per_previous_task
        else None
    )

    def evidence_path(task_id: str, task_index: int) -> Path:
        relative = args.evidence_template.format(
            suite=args.suite, task_id=task_id, task_index=task_index
        )
        path = Path(args.evidence_root) / relative
        if not path.is_file():
            raise FileNotFoundError(f"task evidence is missing: {path}")
        return path

    def train_stage(previous: PolicyState, task_id: str, stage: int):
        stream, validation_stream = _task_streams(
            task_id=task_id,
            instruction=suite_tasks[stage],
            data_file=data_files[stage],
            config=config,
            bundle=bundle,
            require_validation=condition.optimization_space == "latent",
        )
        normalizations[task_id] = stream.normalization
        replay = ()
        if replay_memory is not None:
            replay = replay_memory.prior_transitions(
                task_ids[: stage + 1], current_task_id=task_id
            )
        factory = make_octo_batch_factory(
            stream,
            batch_size=batch_size,
            seed=args.seed * 100 + stage,
            text_processor=bundle.pretrained_model.text_processor,
            example_batch=bundle.pretrained_model.example_batch,
            replay=replay,
        )

        evidence_bytes = 0
        if condition.optimization_space in {"lora", "oracle"}:
            result = train_shared_lora_stage(
                bundle,
                previous.adapter_state,
                factory,
                seed=args.seed * 100 + stage,
                steps=steps,
                learning_rate=args.learning_rate,
                gradient_accumulation_steps=int(
                    config["continual_learning"]["gradient_accumulation_steps"]
                ),
            )
            drift = _component_drift(
                bundle,
                previous.adapter_state,
                result.adapter_state,
                token_width=int(config["adapter"]["token_width"]),
            )
            next_state = PolicyState(result.adapter_state)
            regularization_loss = None
        else:
            task_evidence = load_task_evidence(evidence_path(task_id, stage))
            if task_evidence.task_id != task_id or task_evidence.suite != args.suite:
                raise ValueError("evidence identity differs from the current task")
            if task_evidence.vision_mask is None:
                raise ValueError("latent continual training requires a visual evidence mask")
            if task_evidence.raw_feature_references.get("dataset_sha256") != dataset_hashes[task_id]:
                raise ValueError("evidence and continual run use different task data")
            if task_evidence.raw_feature_references.get("base_sha256") != bundle.base_sha256:
                raise ValueError("evidence and continual run use different Octo bases")
            expected_episodes = tuple(episode.episode_id for episode in stream.episodes)
            if task_evidence.episode_ids != expected_episodes:
                raise ValueError("evidence and continual run use different training episodes")
            evidence_bytes = sum(
                int(np.asarray(value).nbytes)
                for value in (
                    task_evidence.vision_features,
                    task_evidence.vision_mask,
                    task_evidence.language_features,
                    task_evidence.action_statistics,
                )
            )
            initial_latents = previous.latents
            if initial_latents is None:
                initial_latents = encode_and_map_evidence(
                    alignment,
                    vision_features=task_evidence.vision_features,
                    vision_mask=task_evidence.vision_mask,
                    language_features=task_evidence.language_features,
                    action_features=task_evidence.action_statistics,
                )
            key = jax.random.PRNGKey(args.seed * 100 + stage)
            counter = 0

            def latent_batches():
                nonlocal counter
                for batch in factory():
                    batch_key = jax.random.fold_in(key, counter)
                    counter += 1
                    yield batch, batch_key

            if validation_stream is None:
                raise AssertionError("latent condition lacks validation stream")
            validation_factory = make_octo_batch_factory(
                validation_stream,
                batch_size=batch_size,
                seed=args.seed * 100 + stage + 50_000,
                text_processor=bundle.pretrained_model.text_processor,
                example_batch=bundle.pretrained_model.example_batch,
            )
            validation_key = jax.random.PRNGKey(args.seed * 100 + stage + 50_000)
            validation_counter = 0

            def validation_batches():
                nonlocal validation_counter
                for batch in validation_factory():
                    batch_key = jax.random.fold_in(
                        validation_key, validation_counter
                    )
                    validation_counter += 1
                    yield batch, batch_key

            result = train_shared_latent_stage(
                alignment,
                bundle=bundle,
                adapter_spec=adapter_spec,
                previous_latents=initial_latents,
                batches=latent_batches,
                gammas=condition.gammas,
                steps=steps,
                learning_rate=args.learning_rate,
                validation_batches=validation_batches,
                early_stopping_patience=selection.early_stopping_patience,
            )
            drift = result.latent_drift
            next_state = PolicyState(result.adapter_state, result.latents)
            regularization_loss = result.final_regularization_loss

        if replay_memory is not None:
            sampled = sample_replay_transitions(
                stream,
                count=condition.replay_per_previous_task,
                seed=args.seed * 10_000 + stage,
            )
            replay_memory.add_task(task_id, sampled)
            _atomic_bytes(
                run_root / "replay_memory.json",
                (json.dumps(replay_memory.provenance(), indent=2, sort_keys=True) + "\n").encode(),
            )
        peak_vram, peak_ram = _memory_peaks()
        return StageUpdate(
            state=next_state,
            checkpoint_sha256=result.checkpoint_sha256,
            latent_drift=drift,
            task_loss=result.final_task_loss,
            regularization_loss=regularization_loss,
            adapter_bytes=result.adapter_bytes,
            evidence_bytes=evidence_bytes,
            peak_vram_bytes=peak_vram,
            peak_ram_bytes=peak_ram,
            update_steps=result.update_steps,
            micro_steps=getattr(result, "micro_steps", result.update_steps),
        )

    task_index_by_id = {task_id: index for index, task_id in enumerate(task_ids)}

    def rollout(state: PolicyState, task_id: str, count: int, seed: int):
        normalization = normalizations.get(task_id)
        if normalization is None:
            raise ValueError(f"no training-split action normalization for {task_id}")
        policy = OctoLiberoPolicy(
            bundle,
            state.adapter_state,
            action_mean=normalization.mean,
            action_std=normalization.std,
        )
        return evaluator.evaluate_task(
            policy,
            task_index=task_index_by_id[task_id],
            rollout_count=count,
            seed=seed,
        ).as_outcome()

    initial = PolicyState(initial_adapter_state(bundle))
    try:
        common = dict(
            initial_state=initial,
            task_ids=task_ids,
            rollout=rollout,
            run_id=run_id,
            suite=args.suite,
            seed=args.seed,
            rollout_count=rollouts,
            stage_sink=lambda stage, task_id, update: _save_stage(
                run_root,
                stage=stage,
                task_id=task_id,
                update=update,
                bundle=bundle,
                condition=args.condition,
            ),
            record_sink=lambda record: append_evaluation_record(record, record_path),
        )
        if condition.independent_per_task:
            _, records = run_independent_oracle_protocol(train_task=train_stage, **common)
        else:
            _, records = run_sequential_protocol(
                train_stage=train_stage, condition=args.condition, **common
            )
        if args.task_count == 2:
            matrix = two_task_success_matrix(records)
            _atomic_bytes(
                run_root / "two_task_gate.json",
                (
                    json.dumps(
                        {
                            "schema_version": 1,
                            "passed": True,
                            "run_id": run_id,
                            "suite": args.suite,
                            "seed": args.seed,
                            "condition": args.condition,
                            "matrix_axes": {
                                "rows": "training_stage",
                                "columns": "evaluated_task_index",
                            },
                            "success_rate_matrix": [
                                [None if np.isnan(value) else float(value) for value in row]
                                for row in matrix
                            ],
                            "expected_missing_cell": [0, 1],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode(),
            )
        manifest.configuration["action_normalization"] = {
            task_id: normalizations[task_id].to_dict() for task_id in task_ids
        }
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        write_manifest_atomic(manifest, manifest_path)
        print(json.dumps({"ok": True, "run_id": run_id, "records": len(records)}, indent=2))
        return 0
    except Exception as exc:
        manifest.failures.append({"type": type(exc).__name__, "message": str(exc)})
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        write_manifest_atomic(manifest, manifest_path)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
