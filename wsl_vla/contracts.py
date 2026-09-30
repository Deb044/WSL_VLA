"""Versioned public data contracts for publication experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1


class Component(str, Enum):
    VISION = "vision"
    LANGUAGE = "language"
    ACTION = "action"


@dataclass(frozen=True)
class AdapterEntry:
    """One low-rank update target in a frozen base model."""

    component: Component
    layer: int
    parameter_path: str
    input_dim: int
    output_dim: int
    rank: int

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if min(self.input_dim, self.output_dim, self.rank) <= 0:
            raise ValueError("adapter dimensions and rank must be positive")
        if not self.parameter_path:
            raise ValueError("parameter_path cannot be empty")


@dataclass(frozen=True)
class AdapterSpec:
    """Immutable adapter layout tied to one exact pretrained base."""

    base_model_id: str
    base_revision: str
    base_sha256: str
    alpha: float
    token_width: int
    entries: tuple[AdapterEntry, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported AdapterSpec schema {self.schema_version}")
        if len(self.base_sha256) != 64:
            raise ValueError("base_sha256 must be a SHA-256 hex digest")
        if self.alpha <= 0 or self.token_width <= 0:
            raise ValueError("alpha and token_width must be positive")
        paths = [entry.parameter_path for entry in self.entries]
        if len(paths) != len(set(paths)):
            raise ValueError("adapter parameter paths must be unique")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for source, entry in zip(self.entries, payload["entries"]):
            entry["component"] = source.component.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdapterSpec":
        version = int(payload.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported AdapterSpec schema {version}")
        entries = tuple(
            AdapterEntry(
                component=Component(item["component"]),
                layer=int(item["layer"]),
                parameter_path=str(item["parameter_path"]),
                input_dim=int(item["input_dim"]),
                output_dim=int(item["output_dim"]),
                rank=int(item["rank"]),
            )
            for item in payload["entries"]
        )
        return cls(
            base_model_id=str(payload["base_model_id"]),
            base_revision=str(payload["base_revision"]),
            base_sha256=str(payload["base_sha256"]),
            alpha=float(payload["alpha"]),
            token_width=int(payload["token_width"]),
            entries=entries,
            schema_version=version,
        )


@dataclass
class TaskEvidence:
    """Unprojected evidence for one task/checkpoint pairing."""

    task_id: str
    suite: str
    vision_features: np.ndarray
    language_features: np.ndarray
    action_statistics: np.ndarray
    episode_ids: tuple[str, ...]
    preprocessing_version: str
    raw_feature_references: Mapping[str, str] = field(default_factory=dict)
    vision_mask: np.ndarray | None = None
    language_mask: np.ndarray | None = None
    projected_visual: np.ndarray | None = None
    projected_language: np.ndarray | None = None
    projected_action: np.ndarray | None = None
    synthetic: bool = False
    schema_version: int = SCHEMA_VERSION

    def validate_for_research(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported TaskEvidence schema {self.schema_version}")
        if self.synthetic:
            raise ValueError("synthetic evidence is forbidden in research runs")
        if not self.episode_ids or len(self.episode_ids) != len(set(self.episode_ids)):
            raise ValueError("episode_ids must be non-empty and unique")
        for name, value in (
            ("vision_features", self.vision_features),
            ("language_features", self.language_features),
            ("action_statistics", self.action_statistics),
        ):
            array = np.asarray(value)
            if array.size == 0 or not np.isfinite(array).all():
                raise ValueError(f"{name} must be non-empty and finite")
        for name, value in (
            ("projected_visual", self.projected_visual),
            ("projected_language", self.projected_language),
            ("projected_action", self.projected_action),
        ):
            if value is not None and (np.asarray(value).size == 0 or not np.isfinite(value).all()):
                raise ValueError(f"{name} must be non-empty and finite when present")
        if self.vision_mask is not None and np.asarray(self.vision_mask).shape[0] != self.vision_features.shape[0]:
            raise ValueError("vision_mask must match the visual evidence sample axis")


@dataclass(frozen=True)
class AlignmentCheckpointMetadata:
    base_sha256: str
    adapter_spec_sha256: str
    train_suites: tuple[str, ...]
    validation_task_indices: tuple[int, ...]
    test_suite: str
    seeds: tuple[int, ...]
    schema_version: int = SCHEMA_VERSION


@dataclass
class AlignmentCheckpoint:
    """Reloaded alignment system and all inference-time geometric artifacts."""

    params: Any
    model: Any
    mappers: Mapping[str, Any]
    shells: Mapping[str, Any]
    token_mask: np.ndarray
    component_ids: np.ndarray
    layer_ids: np.ndarray
    metadata: Mapping[str, Any]
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported AlignmentCheckpoint schema {self.schema_version}")
        if set(self.mappers) != {"vision", "language", "action"}:
            raise ValueError("alignment checkpoint requires all three modality mappers")
        if set(self.shells) != {"vision", "language", "action"}:
            raise ValueError("alignment checkpoint requires all three empirical shells")
        if self.token_mask.ndim != 2 or self.component_ids.shape != self.token_mask.shape[:1]:
            raise ValueError("alignment token layout is inconsistent")
        if self.layer_ids.shape != self.component_ids.shape:
            raise ValueError("alignment layer layout is inconsistent")


@dataclass
class RunManifest:
    run_id: str
    command: Sequence[str]
    git_commit: str
    git_dirty: bool
    base_model_id: str
    base_revision: str
    base_sha256: str
    dataset_sha256: Mapping[str, str]
    task_order: Sequence[str]
    seeds: Sequence[int]
    configuration: Mapping[str, Any]
    hardware: Mapping[str, Any]
    environment: Mapping[str, str]
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str | None = None
    failures: list[Mapping[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class EvaluationRecord:
    run_id: str
    suite: str
    seed: int
    condition: str
    training_stage: int
    evaluated_task_index: int
    evaluated_task_id: str
    rollout_count: int
    successes: int
    checkpoint_sha256: str
    latent_drift: Mapping[str, float]
    wall_time_seconds: float
    initialization_indices: Sequence[int]
    rollout_seeds: Sequence[int]
    evaluation_wall_time_seconds: float
    task_loss: float | None = None
    regularization_loss: float | None = None
    adapter_bytes: int = 0
    evidence_bytes: int = 0
    peak_vram_bytes: int | None = None
    peak_ram_bytes: int | None = None
    component_swap_condition: str | None = None
    failure: Mapping[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def success_rate(self) -> float:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported EvaluationRecord schema {self.schema_version}")
        if self.rollout_count <= 0:
            raise ValueError("rollout_count must be positive")
        if not 0 <= self.successes <= self.rollout_count:
            raise ValueError("successes must lie in [0, rollout_count]")
        if len(self.initialization_indices) != self.rollout_count:
            raise ValueError("every rollout must record its fixed initialization index")
        if len(set(self.initialization_indices)) != self.rollout_count:
            raise ValueError("fixed initialization indices cannot repeat within a cell")
        if len(self.rollout_seeds) != self.rollout_count:
            raise ValueError("every rollout must record its deterministic seed")
        if min(self.wall_time_seconds, self.evaluation_wall_time_seconds) < 0:
            raise ValueError("training and evaluation wall times cannot be negative")
        if min(self.adapter_bytes, self.evidence_bytes) < 0:
            raise ValueError("memory sizes cannot be negative")
        return self.successes / self.rollout_count


@dataclass(frozen=True)
class OODEvaluationRecord:
    run_id: str
    held_out_suite: str
    task_id: str
    seed: int
    method: str
    adaptation_steps: int
    rollout_count: int
    successes: int
    checkpoint_sha256: str
    wall_time_seconds: float
    initialization_indices: Sequence[int]
    rollout_seeds: Sequence[int]
    source_training_suites: Sequence[str]
    alignment_checkpoint_sha256: str
    adaptation_loss: float | None = None
    adapter_bytes: int = 0
    evidence_bytes: int = 0
    failure: Mapping[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def success_rate(self) -> float:
        if self.adaptation_steps < 0 or self.rollout_count <= 0:
            raise ValueError("adaptation steps must be non-negative and rollouts positive")
        if not 0 <= self.successes <= self.rollout_count:
            raise ValueError("successes must lie in [0, rollout_count]")
        if len(self.initialization_indices) != self.rollout_count:
            raise ValueError("OOD records require every fixed initialization index")
        if len(set(self.initialization_indices)) != self.rollout_count:
            raise ValueError("OOD initialization indices cannot repeat")
        if len(self.rollout_seeds) != self.rollout_count:
            raise ValueError("OOD records require every rollout seed")
        if self.held_out_suite in self.source_training_suites:
            raise ValueError("held-out suite leaked into OOD reference data")
        if len(self.checkpoint_sha256) != 64 or len(self.alignment_checkpoint_sha256) != 64:
            raise ValueError("OOD checkpoint identities must be SHA-256 digests")
        if min(self.wall_time_seconds, self.adapter_bytes, self.evidence_bytes) < 0:
            raise ValueError("OOD runtime and memory values cannot be negative")
        return self.successes / self.rollout_count


@dataclass(frozen=True)
class ComponentSwapRecord:
    run_id: str
    suite: str
    seed: int
    condition: str
    transition_stage: int
    evaluated_task_id: str
    swap_condition: str
    rollout_count: int
    baseline_successes: int
    swapped_successes: int
    initialization_indices: Sequence[int]
    previous_checkpoint_sha256: str
    current_checkpoint_sha256: str
    swapped_checkpoint_sha256: str
    latent_drift: Mapping[str, float]
    rollout_seeds: Sequence[int]
    baseline_wall_time_seconds: float
    swapped_wall_time_seconds: float
    schema_version: int = SCHEMA_VERSION

    @property
    def success_rate_drop(self) -> float:
        if self.transition_stage <= 0:
            raise ValueError("component swaps require a consecutive post-task transition")
        if self.swap_condition not in {
            "vision",
            "language",
            "action",
            "vision_language",
            "full",
        }:
            raise ValueError("unknown component swap condition")
        if self.rollout_count <= 0:
            raise ValueError("component swap rollout_count must be positive")
        if not 0 <= self.baseline_successes <= self.rollout_count:
            raise ValueError("baseline successes lie outside the rollout count")
        if not 0 <= self.swapped_successes <= self.rollout_count:
            raise ValueError("swapped successes lie outside the rollout count")
        if len(self.initialization_indices) != self.rollout_count:
            raise ValueError("component swaps require fixed initialization identities")
        if len(set(self.initialization_indices)) != self.rollout_count:
            raise ValueError("component-swap initialization identities cannot repeat")
        if len(self.rollout_seeds) != self.rollout_count:
            raise ValueError("component swaps require every rollout seed")
        if min(self.baseline_wall_time_seconds, self.swapped_wall_time_seconds) < 0:
            raise ValueError("component-swap wall times cannot be negative")
        for digest in (
            self.previous_checkpoint_sha256,
            self.current_checkpoint_sha256,
            self.swapped_checkpoint_sha256,
        ):
            if len(digest) != 64:
                raise ValueError("component swap checkpoint identity is not SHA-256")
        return (self.baseline_successes - self.swapped_successes) / self.rollout_count
