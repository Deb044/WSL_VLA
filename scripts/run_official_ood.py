#!/usr/bin/env python3
"""Run all five locked OOD adapter methods on one held-out LIBERO suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from wsl_vla.adapters.latent import effective_updates_to_state, unpack_effective_tokens_jax
from wsl_vla.alignment.checkpoint import (
    decode_alignment_latents,
    encode_and_map_evidence,
    encode_evidence_prompts,
    load_alignment_checkpoint,
)
from wsl_vla.alignment.evidence_io import load_task_evidence
from wsl_vla.contracts import AdapterSpec, RunManifest
from wsl_vla.data.libero import StrictLiberoHDF5, load_suite_manifest
from wsl_vla.data.octo_batches import ActionNormalization
from wsl_vla.data.streams import TaskTrainingStream, make_octo_batch_factory
from wsl_vla.evaluation.libero_rollout import LiberoRolloutEvaluator, OctoLiberoPolicy
from wsl_vla.evaluation.ood import OOD_METHODS, OODCandidate, run_ood_protocol
from wsl_vla.evaluation.ood_bank import load_reference_bank
from wsl_vla.evaluation.records import append_ood_evaluation_record
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
from wsl_vla.octo.continual import train_shared_latent_stage, train_shared_lora_stage, tree_nbytes


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


def _task_stream(task_id, instruction, data_file, config, bundle) -> TaskTrainingStream:
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


def _decoded_state(checkpoint, bundle, adapter_spec: AdapterSpec, latents):
    decoded = decode_alignment_latents(checkpoint, latents)
    updates = unpack_effective_tokens_jax(decoded, checkpoint.token_mask, adapter_spec)
    return effective_updates_to_state(bundle, adapter_spec, updates)


def _save_candidate(root: Path, *, task_id: str, method: str, state, metadata) -> None:
    try:
        import flax.serialization
        import jax
    except ImportError as exc:  # pragma: no cover - official environment
        raise RuntimeError("OOD candidate persistence requires JAX/Flax") from exc
    directory = root / "checkpoints" / task_id / method
    directory.mkdir(parents=True, exist_ok=False)
    host_state = jax.device_get(state)
    identity = sha256_array_tree(host_state)
    if identity != metadata["checkpoint_sha256"]:
        raise ValueError("OOD candidate state differs from its checkpoint identity")
    state_path = directory / "adapter_state.msgpack"
    _atomic_bytes(state_path, flax.serialization.to_bytes(host_state))
    sidecar = {
        "schema_version": 1,
        "task_id": task_id,
        "method": method,
        "state_file_sha256": sha256_file(state_path),
        **metadata,
    }
    _atomic_bytes(
        directory / "metadata.json",
        (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode(),
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=REQUIRED_SUITES)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--alignment-checkpoint", required=True)
    parser.add_argument("--reference-bank", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument(
        "--evidence-template", default="{suite}/{task_id}.npz"
    )
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--output-root", default="research_results/ood")
    parser.add_argument("--tier", choices=("development", "publication"), default="development")
    parser.add_argument("--adaptation-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("official OOD experiments must run under Linux/WSL2")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() not in {"false", "0"}:
        raise RuntimeError("set XLA_PYTHON_CLIENT_PREALLOCATE=false for the <=8 GB profile")
    import jax

    config = load_yaml(args.config)
    tasks = load_yaml(args.tasks)
    validate_research_config(config)
    validate_reference_tasks(tasks)
    if args.seed not in config["evaluation"]["training_seeds"]:
        raise ValueError("seed is outside the locked three-seed protocol")
    assert_installed_vcs_revision("octo", OCTO_GIT_REVISION)
    assert_installed_vcs_revision("libero", LIBERO_GIT_REVISION)

    suite_tasks = tuple(tasks["suites"][args.suite])
    task_ids = tuple(f"{args.suite}_{index}" for index in range(10))
    data_root = Path(args.data_root or config["data"]["root"])
    data_files = load_suite_manifest(data_root / args.suite, suite_tasks)
    dataset_hashes = {task_id: sha256_file(path) for task_id, path in zip(task_ids, data_files)}
    alignment = load_alignment_checkpoint(args.alignment_checkpoint)
    alignment_sha256 = sha256_directory(args.alignment_checkpoint)
    if alignment.metadata.get("held_out_suite") != args.suite:
        raise ValueError("alignment checkpoint belongs to another held-out suite")
    if float(alignment.metadata.get("model", {}).get("contrastive_weight", 0)) <= 0:
        raise ValueError("primary OOD evaluation requires the aligned checkpoint")
    adapter_spec = AdapterSpec.from_dict(alignment.metadata["adapter_spec"])

    bank_artifact = load_reference_bank(args.reference_bank)
    bank = bank_artifact.bank
    if bank.held_out_suite != args.suite:
        raise ValueError("OOD reference bank belongs to another held-out suite")
    if bank_artifact.alignment_checkpoint_sha256 != alignment_sha256:
        raise ValueError("OOD reference bank was built with another alignment checkpoint")
    if bank_artifact.base_sha256 != alignment.metadata["base_sha256"]:
        raise ValueError("OOD reference bank and alignment checkpoint use different bases")
    expected_sources = tuple(sorted(set(REQUIRED_SUITES) - {args.suite}))
    if bank.training_suites != expected_sources:
        raise ValueError("OOD reference bank does not contain the exact three source suites")

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
    if runtime_spec != adapter_spec or bundle.base_sha256 != bank_artifact.base_sha256:
        raise ValueError("live Octo adapter layout/base differs from OOD artifacts")

    run_id = f"ood-{args.suite}-{args.seed}-{uuid.uuid4().hex[:12]}"
    run_root = Path(args.output_root) / args.suite / f"seed_{args.seed}"
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite an existing OOD run: {run_root}")
    run_root.mkdir(parents=True)
    record_path = run_root / "evaluation.jsonl"
    rollouts = int(
        config["continual_learning"][
            "publication_rollouts" if args.tier == "publication" else "development_rollouts"
        ]
    )
    adaptation_steps = args.adaptation_steps or int(
        config["continual_learning"]["train_steps_per_task"]
    )
    batch_size = args.batch_size or int(config["continual_learning"]["micro_batch_size"])
    repo_root = Path(__file__).resolve().parents[1]
    commit, dirty = git_state(repo_root)
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
            "methods": OOD_METHODS,
            "matched_adaptation_steps": adaptation_steps,
            "adaptation_gradient_accumulation_steps": 1,
            "learning_rate": args.learning_rate,
            "micro_batch_size": batch_size,
            "rollouts_per_method_task": rollouts,
            "alignment_checkpoint_sha256": alignment_sha256,
            "reference_bank_sha256": sha256_file(args.reference_bank),
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
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    cached_id = None
    cached_stream = None
    cached_evidence = None
    cached_mapped = None
    cached_prompts = None

    def task_inputs(task_id: str):
        nonlocal cached_id, cached_stream, cached_evidence, cached_mapped, cached_prompts
        if cached_id == task_id:
            return cached_stream, cached_evidence, cached_mapped, cached_prompts
        index = task_index[task_id]
        stream = _task_stream(
            task_id, suite_tasks[index], data_files[index], config, bundle
        )
        relative = args.evidence_template.format(
            suite=args.suite, task_id=task_id, task_index=index
        )
        evidence_path = Path(args.evidence_root) / relative
        evidence = load_task_evidence(evidence_path)
        if evidence.task_id != task_id or evidence.suite != args.suite:
            raise ValueError("OOD evidence identity differs from requested task")
        if evidence.vision_mask is None:
            raise ValueError("OOD visual evidence requires a validity mask")
        if evidence.raw_feature_references.get("dataset_sha256") != dataset_hashes[task_id]:
            raise ValueError("OOD evidence and run use different task data")
        if evidence.raw_feature_references.get("base_sha256") != bundle.base_sha256:
            raise ValueError("OOD evidence and run use different Octo bases")
        if evidence.episode_ids != tuple(item.episode_id for item in stream.episodes):
            raise ValueError("OOD evidence and adaptation use different train episodes")
        mapped = encode_and_map_evidence(
            alignment,
            vision_features=evidence.vision_features,
            vision_mask=evidence.vision_mask,
            language_features=evidence.language_features,
            action_features=evidence.action_statistics,
        )
        prompts = encode_evidence_prompts(
            alignment,
            vision_features=evidence.vision_features,
            vision_mask=evidence.vision_mask,
            language_features=evidence.language_features,
            action_features=evidence.action_statistics,
        )
        cached_id, cached_stream, cached_evidence = task_id, stream, evidence
        cached_mapped = mapped
        cached_prompts = {name: np.asarray(value) for name, value in prompts.items()}
        return cached_stream, cached_evidence, cached_mapped, cached_prompts

    def evidence_bytes(evidence) -> int:
        return sum(
            int(np.asarray(value).nbytes)
            for value in (
                evidence.vision_features,
                evidence.vision_mask,
                evidence.language_features,
                evidence.action_statistics,
            )
        )

    def finish_candidate(task_id, method, state, *, steps=0, loss=None, evidence_size=0):
        host_state = jax.device_get(state)
        identity = sha256_array_tree(host_state)
        size = tree_nbytes(host_state)
        metadata = {
            "checkpoint_sha256": identity,
            "base_sha256": bundle.base_sha256,
            "alignment_checkpoint_sha256": alignment_sha256,
            "adaptation_steps": steps,
            "adaptation_loss": loss,
            "adapter_bytes": size,
            "evidence_bytes": evidence_size,
        }
        _save_candidate(
            run_root, task_id=task_id, method=method, state=state, metadata=metadata
        )
        return OODCandidate(
            state=state,
            checkpoint_sha256=identity,
            adaptation_steps=steps,
            adaptation_loss=loss,
            adapter_bytes=size,
            evidence_bytes=evidence_size,
        )

    def mapped_zero_shot(task_id, instruction, seed):
        _, evidence, mapped, _ = task_inputs(task_id)
        del instruction, seed
        return finish_candidate(
            task_id,
            "mapped_zero_shot",
            _decoded_state(alignment, bundle, adapter_spec, mapped),
            evidence_size=evidence_bytes(evidence),
        )

    def mapped_latent_refinement(task_id, instruction, seed):
        stream, evidence, mapped, _ = task_inputs(task_id)
        del instruction
        factory = make_octo_batch_factory(
            stream,
            batch_size=batch_size,
            seed=seed,
            text_processor=bundle.pretrained_model.text_processor,
            example_batch=bundle.pretrained_model.example_batch,
        )
        key = jax.random.PRNGKey(seed)
        counter = 0

        def batches():
            nonlocal counter
            for batch in factory():
                batch_key = jax.random.fold_in(key, counter)
                counter += 1
                yield batch, batch_key

        result = train_shared_latent_stage(
            alignment,
            bundle=bundle,
            adapter_spec=adapter_spec,
            previous_latents=mapped,
            batches=batches,
            gammas={name: 0.0 for name in ("vision", "language", "action")},
            steps=adaptation_steps,
            learning_rate=args.learning_rate,
        )
        return finish_candidate(
            task_id,
            "mapped_latent_refinement",
            result.adapter_state,
            steps=result.update_steps,
            loss=result.final_task_loss,
            evidence_size=evidence_bytes(evidence),
        )

    def mapped_weight_finetune(task_id, instruction, seed):
        stream, evidence, mapped, _ = task_inputs(task_id)
        del instruction
        initial = _decoded_state(alignment, bundle, adapter_spec, mapped)
        factory = make_octo_batch_factory(
            stream,
            batch_size=batch_size,
            seed=seed,
            text_processor=bundle.pretrained_model.text_processor,
            example_batch=bundle.pretrained_model.example_batch,
        )
        result = train_shared_lora_stage(
            bundle,
            initial,
            factory,
            seed=seed,
            steps=adaptation_steps,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=1,
        )
        return finish_candidate(
            task_id,
            "mapped_weight_finetune",
            result.adapter_state,
            steps=result.update_steps,
            loss=result.final_task_loss,
            evidence_size=evidence_bytes(evidence),
        )

    def nearest_neighbor(task_id, instruction, seed):
        _, evidence, _, prompts = task_inputs(task_id)
        del instruction, seed
        _, latents = bank.nearest_neighbor_latents(prompts)
        return finish_candidate(
            task_id,
            "nearest_neighbor",
            _decoded_state(alignment, bundle, adapter_spec, latents),
            evidence_size=evidence_bytes(evidence),
        )

    def mean_latent(task_id, instruction, seed):
        _, evidence, _, _ = task_inputs(task_id)
        del instruction, seed
        return finish_candidate(
            task_id,
            "mean_latent",
            _decoded_state(alignment, bundle, adapter_spec, bank.mean_latents()),
            evidence_size=evidence_bytes(evidence),
        )

    builders = {
        "mapped_zero_shot": mapped_zero_shot,
        "mapped_latent_refinement": mapped_latent_refinement,
        "mapped_weight_finetune": mapped_weight_finetune,
        "nearest_neighbor": nearest_neighbor,
        "mean_latent": mean_latent,
    }

    def rollout(state: Mapping[str, Any], task_id: str, count: int, seed: int):
        stream, _, _, _ = task_inputs(task_id)
        policy = OctoLiberoPolicy(
            bundle,
            state,
            action_mean=stream.normalization.mean,
            action_std=stream.normalization.std,
        )
        return evaluator.evaluate_task(
            policy,
            task_index=task_index[task_id],
            rollout_count=count,
            seed=seed,
        ).as_outcome()

    try:
        records = run_ood_protocol(
            task_ids=task_ids,
            instructions=suite_tasks,
            candidate_builders=builders,
            rollout=rollout,
            run_id=run_id,
            held_out_suite=args.suite,
            seed=args.seed,
            rollout_count=rollouts,
            matched_adaptation_steps=adaptation_steps,
            source_training_suites=bank.training_suites,
            alignment_checkpoint_sha256=alignment_sha256,
            record_sink=lambda record: append_ood_evaluation_record(record, record_path),
        )
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
