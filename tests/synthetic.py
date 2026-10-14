"""Small synthetic zoo populations for contract tests (no Octo, no simulator).

Everything written here goes through the same writers the research commands
use, so archive, alignment, and evaluation tests exercise the real contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from wsl_vla.adapters.packing import (
    PackedAdapterMetadata,
    pack_low_rank_adapter,
    save_packed_adapter,
)
from wsl_vla.alignment.evidence import action_chunk_evidence
from wsl_vla.alignment.evidence_io import save_task_evidence
from wsl_vla.contracts import (
    LIBERO_ACTION_CONVENTION,
    POPULATION_SAMPLE_SCHEMA_VERSION,
    SITE_BLOCK_INPUT,
    SITE_HEAD_KERNEL,
    AdapterEntry,
    AdapterSpec,
    Component,
    TaskEvidence,
    canonical_sha256,
)
from wsl_vla.experiments.manifest import (
    ExperimentManifest,
    ExperimentPaths,
    bind_experiment,
    manifest_from_dict,
    split_task_id,
)
from wsl_vla.experiments.provenance import sha256_file


BASE_SHA = "a" * 64
ARCHITECTURE_SHA = "b" * 64
PREPROCESSING_SHA = "c" * 64
DATASET_SHA = "d" * 64
SEQUENCE_WINDOW = 8
EPISODE_IDS = ("demo_0", "demo_1", "demo_2")


def synthetic_spec(token_width: int = 6) -> AdapterSpec:
    entries = []
    for component in ("vision", "language", "action"):
        for layer in range(2):
            entries.append(
                AdapterEntry(
                    Component(component), layer, f"t/adapter_{component}_{layer}", 8, 8, 2,
                    SITE_BLOCK_INPUT,
                )
            )
    entries.append(AdapterEntry(Component.ACTION, 0, "h/dense_0/kernel", 5, 12, 2, SITE_HEAD_KERNEL))
    entries.append(AdapterEntry(Component.ACTION, 1, "h/time/kernel", 4, 1, 2, SITE_HEAD_KERNEL))
    entries.sort(key=lambda e: (e.component.value, e.site, e.layer, e.parameter_path))
    return AdapterSpec(
        base_model_id="test/model",
        base_revision="revision",
        base_sha256=BASE_SHA,
        alpha=4.0,
        token_width=token_width,
        entries=tuple(entries),
        architecture_sha256=ARCHITECTURE_SHA,
        token_routing=(
            ("obs_primary", "vision"),
            ("readout_action", "action"),
            ("task_language", "language"),
        ),
        action_convention=LIBERO_ACTION_CONVENTION,
    )


def synthetic_factors(spec: AdapterSpec, rng: np.random.Generator, *, scale: float = 1.0):
    return {
        entry.parameter_path: (
            (scale * rng.normal(size=(entry.input_dim, entry.rank))).astype(np.float32),
            rng.normal(size=(entry.rank, entry.output_dim)).astype(np.float32),
        )
        for entry in spec.entries
    }


def synthetic_manifest_dict(
    *,
    name: str = "unit",
    profile: str = "pilot",
    suites: Sequence[str] = ("libero_spatial", "libero_object", "libero_goal", "libero_10"),
    held_out: str = "libero_spatial",
    train_indices: Sequence[int] = (0, 1),
    validation_indices: Sequence[int] = (8,),
    seeds: Sequence[int] = (17, 42),
    snapshots: Sequence[int] = (4, 6, 8),
    frozen: bool = True,
) -> dict:
    indices = sorted({*train_indices, *validation_indices})
    sources = [suite for suite in suites if suite != held_out]
    runs = len(suites) * len(indices) * len(seeds)
    per_task = len(seeds) * len(snapshots)
    return {
        "schema_version": 1,
        "name": name,
        "profile": profile,
        "publication_evidence": False,
        "description": "synthetic contract-test experiment",
        "tasks": {suite: list(indices) for suite in suites},
        "seeds": list(seeds),
        "zoo": {
            "optimizer_updates": snapshots[-1],
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 3e-4,
            "snapshot_updates": list(snapshots),
            "held_out_loss_every": 4,
            "held_out_loss_batches": 2,
            "frozen": frozen,
        },
        "episode_split": {"seed": 5, "train_fraction": 0.7, "validation_fraction": 0.15},
        "evidence": {"frames_per_episode": 2, "action_chunk_length": 4},
        "folds": [
            {
                "name": f"holdout_{held_out}",
                "held_out_suite": held_out,
                "train_tasks": {suite: list(train_indices) for suite in sources},
                "validation_tasks": {suite: list(validation_indices) for suite in sources},
                "held_out_tasks": {held_out: list(indices)},
                "selection": "validation_tasks",
            }
        ],
        "alignment": {
            "seeds": [0],
            "conditions": ["aligned", "reconstruction_only"],
            "tasks_per_batch": 3,
            "samples_per_task": 2,
            "optimizer_updates": 6,
            "validation_every": 3,
        },
        "rollouts": {"development": 2, "final": 2, "seed": 17, "zoo_reference_seed": seeds[0]},
        "expected": {
            "independent_runs": runs,
            "adapter_samples": runs * len(snapshots),
            "folds": {
                f"holdout_{held_out}": {
                    "train_tasks": len(sources) * len(train_indices),
                    "train_samples": len(sources) * len(train_indices) * per_task,
                    "validation_tasks": len(sources) * len(validation_indices),
                    "validation_samples": len(sources) * len(validation_indices) * per_task,
                    "held_out_tasks": len(indices),
                    "held_out_samples": len(indices) * per_task,
                }
            },
        },
    }


def synthetic_manifest(**overrides) -> ExperimentManifest:
    manifest = manifest_from_dict(synthetic_manifest_dict(**overrides))
    manifest.validate()
    return manifest


def synthetic_evidence(identifier: str, rng: np.random.Generator, signal: np.ndarray) -> TaskEvidence:
    suite, _ = split_task_id(identifier)
    frames = 2 * len(EPISODE_IDS)
    actions = [
        np.clip(0.3 * signal[:2] + 0.1 * rng.normal(size=(9 + index, 2)), -1, 1)
        for index in range(len(EPISODE_IDS))
    ]
    chunks, chunk_mask = action_chunk_evidence(actions, chunk_length=4)
    return TaskEvidence(
        task_id=identifier,
        suite=suite,
        instruction=f"do {identifier}",
        episode_ids=EPISODE_IDS,
        vision_features=(signal[None, :5] + 0.05 * rng.normal(size=(frames, 5))).astype(np.float32),
        vision_episode_index=np.repeat(np.arange(len(EPISODE_IDS)), 2).astype(np.int32),
        vision_timestep=np.tile(np.asarray([0, 5]), len(EPISODE_IDS)).astype(np.int32),
        language_features=np.tile(signal[None, :4], (3, 1)).astype(np.float32),
        language_mask=np.asarray([True, True, False]),
        action_chunk_features=chunks,
        action_chunk_mask=chunk_mask,
        feature_extractor_sha256={"vision": "e" * 64, "language": "f" * 64},
        preprocessing_sha256=PREPROCESSING_SHA,
        action_convention_sha256=LIBERO_ACTION_CONVENTION.sha256(),
        base_sha256=BASE_SHA,
        base_revision="revision",
        architecture_sha256=ARCHITECTURE_SHA,
        dataset_sha256=DATASET_SHA,
        extraction={"frames_per_episode": 2, "action_chunk_length": 4},
        synthetic=False,
    )


def write_synthetic_population(
    root: Path,
    manifest: ExperimentManifest,
    *,
    spec: AdapterSpec | None = None,
    seed: int = 0,
    task_noise: float = 0.05,
) -> tuple[ExperimentPaths, AdapterSpec]:
    """Write every manifest sample plus one evidence file per task."""

    spec = spec or synthetic_spec()
    paths = bind_experiment(manifest, root)
    rng = np.random.default_rng(seed)
    task_signal = {identifier: rng.normal(size=8) for identifier in manifest.tasks}
    task_factors = {identifier: synthetic_factors(spec, rng) for identifier in manifest.tasks}
    for identifier in manifest.tasks:
        evidence = synthetic_evidence(identifier, rng, task_signal[identifier])
        path = paths.evidence_file(identifier)
        save_task_evidence(evidence, path)
        path.with_suffix(".manifest.json").write_text(
            json.dumps(
                {
                    "evidence_sha256": sha256_file(path),
                    "evidence_identity": manifest.evidence_identity(),
                    "task_id": identifier,
                }
            ),
            encoding="utf-8",
        )
    layout_sha = None
    for sample in manifest.samples():
        factors = {
            path: (down + task_noise * rng.normal(size=down.shape).astype(np.float32), up)
            for path, (down, up) in task_factors[sample.task_id].items()
        }
        packed = pack_low_rank_adapter(
            spec,
            factors,
            metadata=PackedAdapterMetadata(
                task_id=sample.task_id,
                suite=sample.suite,
                seed=sample.seed,
                optimizer_updates=sample.optimizer_updates,
                snapshot_index=sample.snapshot_index,
                sample_exposure=manifest.zoo.sample_exposure(sample.optimizer_updates),
                source_dtype="float32",
            ),
            sequence_window=SEQUENCE_WINDOW,
        )
        layout_sha = packed.layout.sha256()
        directory = paths.sample_directory(sample)
        directory.mkdir(parents=True)
        save_packed_adapter(packed, directory / "adapter.npz")
        metadata = {
            "schema_version": POPULATION_SAMPLE_SCHEMA_VERSION,
            "synthetic": False,
            "purpose": "zoo",
            "sample_id": sample.sample_id,
            "run_id": sample.run_id,
            "task_id": sample.task_id,
            "suite": sample.suite,
            "task_index": sample.task_index,
            "seed": sample.seed,
            "snapshot_index": sample.snapshot_index,
            "optimizer_updates": sample.optimizer_updates,
            "sample_exposure": manifest.zoo.sample_exposure(sample.optimizer_updates),
            "manifest_name": manifest.name,
            "zoo_sha256": manifest.zoo_sha256(),
            "base_sha256": BASE_SHA,
            "base_revision": "revision",
            "architecture_sha256": ARCHITECTURE_SHA,
            "preprocessing_sha256": PREPROCESSING_SHA,
            "action_convention_sha256": LIBERO_ACTION_CONVENTION.sha256(),
            "adapter_spec_sha256": spec.sha256(),
            "layout_sha256": layout_sha,
            "dataset_sha256": DATASET_SHA,
            "train_episode_ids_sha256": canonical_sha256(list(EPISODE_IDS)),
            "adapter_sha256": sha256_file(directory / "adapter.npz"),
            "validation": {"structurally_valid": True, "behaviourally_evaluated": False},
        }
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return paths, spec


SMALL_ALIGNMENT_CONFIG = {
    "latent_dim": 8,
    "transformer_width": 16,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "attention_heads": 2,
    "mlp_ratio": 2,
    "evidence_hidden_dim": 16,
    "initial_temperature": 0.07,
    "learning_rate": 3e-3,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,
    "gradient_accumulation_steps": 1,
    "reconstruction_weight": 1.0,
    "contrastive_weight": 1.0,
    "auxiliary_classification_weight": 1.0,
    "modality_alignment_weights": {"vision": 1.0, "language": 1.0, "action": 1.0},
    "rms_floor": 1e-6,
    "window_chunk": 0,
    "ridge_alpha_candidates": [0.01, 1.0],
}
