"""Shared command-line plumbing for the zoo and alignment experiment commands."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

from ..data.libero import load_suite_manifest
from ..octo.compat import (
    ENVIRONMENT_PROFILES,
    detect_environment_profile,
    require_environment_profile,
)
from .manifest import (
    ExperimentManifest,
    ExperimentPaths,
    bind_experiment,
    load_experiment_manifest,
    split_task_id,
)
from .protocol import load_yaml, validate_reference_tasks, validate_research_config
from .provenance import capture_environment, capture_hardware, sha256_file


DEFAULT_CONFIG = "configs/research/base.yaml"
DEFAULT_TASKS = "configs/reference_tasks.yaml"
DEFAULT_RESEARCH_ROOT = "research_results"
DEFAULT_CHECKPOINT = "hf://rail-berkeley/octo-small-1.5"
ENVIRONMENT_PACKAGES = (
    "jax", "jaxlib", "flax", "optax", "octo", "libero", "robosuite", "mujoco", "numpy",
)


def add_experiment_arguments(parser, *, checkpoint: bool = True) -> None:
    parser.add_argument("--manifest", required=True, help="experiment manifest YAML")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--research-root", default=DEFAULT_RESEARCH_ROOT)
    if checkpoint:
        parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)


def load_experiment(args) -> tuple[dict, dict, ExperimentManifest, ExperimentPaths]:
    """Load and validate config, reference tasks, and manifest; bind the run root."""

    config = load_yaml(args.config)
    tasks = load_yaml(args.tasks)
    validate_research_config(config)
    validate_reference_tasks(tasks)
    manifest = load_experiment_manifest(args.manifest, reference_tasks=tasks)
    paths = bind_experiment(manifest, args.research_root)
    return config, tasks, manifest, paths


# XLA on GPU may pick different kernels in different processes (autotuning),
# so two runs with the same seed can differ by float rounding. With this flag
# two processes produced bit-identical zoo runs on the verified profile, at no
# measurable cost. The job runner and the Slurm scripts set it; it is recorded
# in every environment record.
DETERMINISTIC_XLA_FLAG = "--xla_gpu_deterministic_ops=true"


def deterministic_gpu_ops(environment: Mapping[str, str] | None = None) -> bool:
    flags = (os.environ if environment is None else environment).get("XLA_FLAGS", "")
    return DETERMINISTIC_XLA_FLAG in flags.split()


def with_deterministic_gpu_ops(environment: Mapping[str, str]) -> dict[str, str]:
    """``environment`` with the deterministic-ops flag added to ``XLA_FLAGS``."""

    result = dict(environment)
    if not deterministic_gpu_ops(result):
        result["XLA_FLAGS"] = " ".join(
            part for part in (result.get("XLA_FLAGS", ""), DETERMINISTIC_XLA_FLAG) if part
        )
    return result


def require_gpu_runtime_settings() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("official research commands must run under Linux/WSL2")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() not in {"false", "0"}:
        raise RuntimeError("set XLA_PYTHON_CLIENT_PREALLOCATE=false before research commands")
    if not deterministic_gpu_ops():
        print(
            f"warning: XLA_FLAGS lacks {DETERMINISTIC_XLA_FLAG}; GPU results of identical "
            "runs may differ by float rounding (recorded in the environment record)",
            file=sys.stderr,
        )


def environment_record(*, require: bool = True) -> dict[str, Any]:
    """Locked-profile identity plus installed versions and hardware."""

    report = detect_environment_profile()
    if require:
        require_environment_profile()
    profile = report["profile"]
    return {
        "profile": profile,
        "hardware_status": report["hardware_status"],
        "deterministic_gpu_ops": deterministic_gpu_ops(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "locked": None if profile is None else ENVIRONMENT_PROFILES[profile],
        "observed": report["observed"],
        "packages": capture_environment(ENVIRONMENT_PACKAGES),
        "hardware": capture_hardware(),
    }


def task_data_file(
    data_root: str | Path, tasks: Mapping[str, Any], identifier: str
) -> tuple[Path, str]:
    """Resolve a task's demonstration file and instruction from the suite manifest."""

    suite, index = split_task_id(identifier)
    files = load_suite_manifest(Path(data_root) / suite, tasks["suites"][suite], allow_missing=True)
    data_file = files[index]
    if data_file is None:
        raise FileNotFoundError(
            f"demonstrations for {identifier} are not downloaded under {Path(data_root) / suite}"
        )
    return data_file, tasks["suites"][suite][index]


def input_hashes(args) -> dict[str, str]:
    return {
        "research_config_sha256": sha256_file(args.config),
        "task_config_sha256": sha256_file(args.tasks),
        "manifest_file_sha256": sha256_file(args.manifest),
    }


# --------------------------------------------------------------------------
# Helpers shared by the continual, OOD, gamma-validation, swap, and recovery
# runners. They keep those entry points on the corrected contracts: shared
# preprocessing, the fixed controller-action convention, schema-2 evidence, and
# schema-2 alignment checkpoints.
# --------------------------------------------------------------------------
def task_streams_from_config(
    bundle: Any,
    *,
    task_id: str,
    instruction: str,
    data_file: str | Path,
    config: Mapping[str, Any],
    splits: tuple[str, ...] = ("train",),
) -> dict[str, Any]:
    """Validated task streams for the requested episode splits of one task."""

    from ..data.streams import load_split_episodes, task_stream_from_bundle

    fractions = config["data"]["episode_split"]
    _, episodes = load_split_episodes(
        data_file,
        preprocessing=bundle.preprocessing,
        split_seed=int(config["data"]["split_seed"]),
        train_fraction=float(fractions[0]),
        validation_fraction=float(fractions[1]),
        splits=splits,
    )
    streams = {}
    for name in splits:
        if not episodes[name]:
            raise ValueError(f"{task_id} has no {name} episodes")
        streams[name] = task_stream_from_bundle(
            bundle, task_id=task_id, instruction=instruction, episodes=episodes[name]
        )
    return streams


def rollout_evaluator(bundle: Any, config: Mapping[str, Any], suite: str, instructions) -> Any:
    """LIBERO evaluator rendering at the shared preprocessing's source resolution."""

    from ..evaluation.libero_rollout import LiberoRolloutEvaluator

    return LiberoRolloutEvaluator(
        suite=suite,
        expected_instructions=tuple(instructions),
        preprocessing=bundle.preprocessing,
        convention=bundle.action_convention,
        warmup_steps=int(config["simulator"]["warmup_steps"]),
        max_steps=int(config["simulator"]["max_episode_steps"]),
    )


def load_checked_task_evidence(
    path: str | Path,
    *,
    task_id: str,
    suite: str,
    dataset_sha256: str,
    bundle: Any,
    episode_ids: tuple[str, ...],
) -> Any:
    """Load schema-2 evidence and require it to match this run's data and base."""

    from ..alignment.evidence_io import load_task_evidence

    evidence = load_task_evidence(path)
    if evidence.task_id != task_id or evidence.suite != suite:
        raise ValueError("evidence identity differs from the requested task")
    if evidence.dataset_sha256 != dataset_sha256:
        raise ValueError("evidence and run use different task data")
    for name, expected in (
        ("base_sha256", bundle.base_sha256),
        ("architecture_sha256", bundle.architecture_sha256),
        ("preprocessing_sha256", bundle.preprocessing.sha256()),
        ("action_convention_sha256", bundle.action_convention.sha256()),
    ):
        if getattr(evidence, name) != expected:
            raise ValueError(f"evidence and run differ on {name}")
    if tuple(evidence.episode_ids) != tuple(episode_ids):
        raise ValueError("evidence and run use different training episodes")
    return evidence


def evidence_nbytes(evidence: Any) -> int:
    """Stored size of one task's cached evidence arrays."""

    import numpy as np

    return int(
        sum(
            np.asarray(getattr(evidence, name)).nbytes
            for name in (
                "vision_features",
                "language_features",
                "language_mask",
                "action_chunk_features",
                "action_chunk_mask",
            )
        )
    )


def validate_alignment_checkpoint_for_run(
    checkpoint: Any, *, held_out_suite: str, bundle: Any, runtime_spec: Any, condition: str
) -> Any:
    """Require a schema-2 checkpoint of the right fold, condition, and adapter layout."""

    metadata = checkpoint.metadata
    if metadata.get("held_out_suite") != held_out_suite:
        raise ValueError("alignment checkpoint belongs to a different held-out suite")
    if metadata.get("integrity_only"):
        raise ValueError("integrity-only alignment checkpoints cannot drive an experiment")
    if metadata["shared"]["base_sha256"] != bundle.base_sha256:
        raise ValueError("alignment checkpoint and Octo bundle use different bases")
    if checkpoint.condition != condition:
        raise ValueError(
            f"alignment checkpoint is the {checkpoint.condition!r} condition, expected {condition!r}"
        )
    if checkpoint.adapter_spec != runtime_spec:
        raise ValueError("alignment AdapterSpec differs from the live Octo adapter layout")
    return checkpoint.adapter_spec
