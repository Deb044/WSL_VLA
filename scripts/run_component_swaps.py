#!/usr/bin/env python3
"""Run consecutive-stage modality swaps for one completed continual experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsl_vla.adapters.latent import effective_updates_from_state, effective_updates_to_state
from wsl_vla.contracts import RunManifest
from wsl_vla.data.libero import StrictLiberoHDF5, load_suite_manifest
from wsl_vla.data.octo_batches import ActionNormalization
from wsl_vla.evaluation.components import run_consecutive_component_swaps
from wsl_vla.evaluation.libero_rollout import LiberoRolloutEvaluator, OctoLiberoPolicy
from wsl_vla.evaluation.records import append_component_swap_record
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


def _load_stage(directory: Path):
    try:
        import flax.serialization
    except ImportError as exc:  # pragma: no cover - official environment
        raise RuntimeError("component analysis requires JAX/Flax") from exc
    metadata_path = directory / "metadata.json"
    state_path = directory / "state.msgpack"
    if not metadata_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"incomplete continual stage checkpoint: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported continual stage checkpoint schema")
    if metadata.get("state_file_sha256") != sha256_file(state_path):
        raise ValueError("continual stage checkpoint file digest mismatch")
    payload = flax.serialization.msgpack_restore(state_path.read_bytes())
    if set(payload) != {"adapter_state", "latents"}:
        raise ValueError("continual stage checkpoint payload is incomplete")
    identity = sha256_array_tree(payload["adapter_state"])
    if identity != metadata.get("checkpoint_sha256"):
        raise ValueError("continual stage adapter identity mismatch")
    return payload, metadata


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("continual_run")
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--tasks", default="configs/reference_tasks.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--output-root", default="research_results/component_swaps")
    parser.add_argument("--tier", choices=("development", "publication"), default="development")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("official component analysis must run under Linux/WSL2")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() not in {"false", "0"}:
        raise RuntimeError("set XLA_PYTHON_CLIENT_PREALLOCATE=false for the <=8 GB profile")
    import jax

    config = load_yaml(args.config)
    tasks = load_yaml(args.tasks)
    validate_research_config(config)
    validate_reference_tasks(tasks)
    assert_installed_vcs_revision("octo", OCTO_GIT_REVISION)
    assert_installed_vcs_revision("libero", LIBERO_GIT_REVISION)

    source_root = Path(args.continual_run)
    source_manifest_path = source_root / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError("continual run manifest is missing")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source.get("schema_version") != 1 or source.get("finished_at") is None:
        raise ValueError("component analysis requires one completed continual run")
    if source.get("failures"):
        raise ValueError("component analysis cannot consume a failed continual run")
    # The suite is an explicit task-order property; do not trust run-id parsing.
    task_order = tuple(str(value) for value in source["task_order"])
    matching_suites = [
        name
        for name in REQUIRED_SUITES
        if task_order == tuple(f"{name}_{index}" for index in range(10))
    ]
    if len(matching_suites) != 1:
        raise ValueError("continual manifest task order is not one exact reference suite")
    suite = matching_suites[0]
    seed_values = tuple(int(value) for value in source["seeds"])
    if len(seed_values) != 1:
        raise ValueError("component analysis requires one continual training seed")
    seed = seed_values[0]
    condition = str(source["configuration"]["condition"]["name"])
    if condition == "independent_adapter_oracle":
        raise ValueError("independent adapters are not one evolving sequential checkpoint")
    data_root = Path(args.data_root or config["data"]["root"])
    data_files = load_suite_manifest(data_root / suite, tasks["suites"][suite])
    dataset_hashes = {
        task_id: sha256_file(path) for task_id, path in zip(task_order, data_files)
    }
    if dataset_hashes != source["dataset_sha256"]:
        raise ValueError("component analysis data differs from continual training data")
    bundle = load_research_octo(
        args.checkpoint,
        rank=int(config["adapter"]["rank"]),
        alpha=float(config["adapter"]["alpha"]),
        seed=seed,
    )
    assert_zero_adapter_equivalence(bundle)
    if bundle.base_sha256 != source["base_sha256"]:
        raise ValueError("component analysis uses a different Octo base")
    spec, _ = build_adapter_spec_and_factors(
        bundle, token_width=int(config["adapter"]["token_width"])
    )

    stage_directories = sorted((source_root / "checkpoints").glob("stage_*"))
    if len(stage_directories) != 10:
        raise ValueError("component analysis requires all ten continual stage checkpoints")
    stages = []
    for expected_stage, directory in enumerate(stage_directories):
        payload, metadata = _load_stage(directory)
        if metadata.get("stage") != expected_stage:
            raise ValueError("continual stage checkpoints are missing or out of order")
        if metadata.get("task_id") != task_order[expected_stage]:
            raise ValueError("continual stage task identity differs from manifest")
        if metadata.get("condition") != condition or metadata.get("base_sha256") != bundle.base_sha256:
            raise ValueError("continual stage provenance differs from source manifest")
        stages.append((payload, metadata))

    normalizations = {}
    for task_id, data_file in zip(task_order, data_files):
        dataset = StrictLiberoHDF5(
            data_file,
            require_wrist_camera=bool(config["data"]["require_wrist_camera"]),
            required_proprio_keys=tuple(config["data"]["proprio_keys"]),
        )
        split = dataset.split(seed=int(config["data"]["split_seed"]))
        normalizations[task_id] = ActionNormalization.fit(dataset.iter_episodes(split.train))

    evaluator = LiberoRolloutEvaluator(
        suite=suite,
        expected_instructions=tasks["suites"][suite],
        camera_size=(
            int(config["simulator"]["camera_height"]),
            int(config["simulator"]["camera_width"]),
        ),
        warmup_steps=int(config["simulator"]["warmup_steps"]),
        max_steps=int(config["simulator"]["max_episode_steps"]),
    )
    task_index = {task_id: index for index, task_id in enumerate(task_order)}
    rollouts = int(
        config["continual_learning"][
            "publication_rollouts" if args.tier == "publication" else "development_rollouts"
        ]
    )
    run_id = f"swaps-{suite}-{seed}-{condition}-{uuid.uuid4().hex[:12]}"
    output_root = Path(args.output_root) / suite / f"seed_{seed}" / condition
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite component analysis: {output_root}")
    output_root.mkdir(parents=True)
    records_path = output_root / "component_swaps.jsonl"
    commit, dirty = git_state(Path(__file__).resolve().parents[1])
    manifest = RunManifest(
        run_id=run_id,
        command=tuple(sys.argv),
        git_commit=commit,
        git_dirty=dirty,
        base_model_id=str(config["model"]["id"]),
        base_revision=bundle.base_revision,
        base_sha256=bundle.base_sha256,
        dataset_sha256=dataset_hashes,
        task_order=task_order,
        seeds=(seed,),
        configuration={
            "analysis": "consecutive_component_swaps",
            "condition": condition,
            "source_run_id": source["run_id"],
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "rollouts_per_swap_task": rollouts,
            "evaluated_tasks": "all_seen",
        },
        hardware=capture_hardware(),
        environment=capture_environment(("jax", "flax", "octo", "libero")),
    )
    manifest_path = output_root / "manifest.json"
    write_manifest_atomic(manifest, manifest_path)

    def rollout(state, task_id, count, rollout_seed):
        normalization = normalizations[task_id]
        policy = OctoLiberoPolicy(
            bundle,
            state,
            action_mean=normalization.mean,
            action_std=normalization.std,
        )
        return evaluator.evaluate_task(
            policy,
            task_index=task_index[task_id],
            rollout_count=count,
            seed=rollout_seed,
        ).as_outcome()

    try:
        record_count = 0
        for current_stage in range(1, 10):
            previous_payload, previous_metadata = stages[current_stage - 1]
            current_payload, current_metadata = stages[current_stage]
            previous_updates = effective_updates_from_state(
                bundle, spec, previous_payload["adapter_state"]
            )
            current_updates = effective_updates_from_state(
                bundle, spec, current_payload["adapter_state"]
            )

            def materialize(updates):
                state = effective_updates_to_state(bundle, spec, updates)
                return state, sha256_array_tree(jax.device_get(state))

            for evaluated_task_id in task_order[: current_stage + 1]:
                records = run_consecutive_component_swaps(
                    previous_updates=previous_updates,
                    current_updates=current_updates,
                    current_state=current_payload["adapter_state"],
                    spec=spec,
                    materialize_swapped_state=materialize,
                    rollout=rollout,
                    run_id=run_id,
                    suite=suite,
                    seed=seed,
                    condition=condition,
                    previous_stage=current_stage - 1,
                    current_stage=current_stage,
                    evaluated_task_id=evaluated_task_id,
                    rollout_count=rollouts,
                    previous_checkpoint_sha256=previous_metadata["checkpoint_sha256"],
                    current_checkpoint_sha256=current_metadata["checkpoint_sha256"],
                    record_sink=lambda record: append_component_swap_record(
                        record, records_path
                    ),
                )
                record_count += len(records)
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        write_manifest_atomic(manifest, manifest_path)
        print(json.dumps({"ok": True, "run_id": run_id, "records": record_count}, indent=2))
        return 0
    except Exception as exc:
        manifest.failures.append({"type": type(exc).__name__, "message": str(exc)})
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        write_manifest_atomic(manifest, manifest_path)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
