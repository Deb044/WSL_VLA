"""Versioned public data contracts for publication experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


# Unchanged artifacts (run manifests and rollout/OOD/component-swap records)
# keep the original schema. Every artifact whose meaning changed with the
# 2026-10 zoo/alignment repair carries its own version so that one artifact can
# evolve without invalidating the others.
SCHEMA_VERSION = 1
ADAPTER_SPEC_SCHEMA_VERSION = 2
PACKED_ADAPTER_SCHEMA_VERSION = 2
TASK_EVIDENCE_SCHEMA_VERSION = 2
POPULATION_SAMPLE_SCHEMA_VERSION = 2
ALIGNMENT_ARCHIVE_SCHEMA_VERSION = 2
ALIGNMENT_CHECKPOINT_SCHEMA_VERSION = 2
ROLLOUT_EPISODE_SCHEMA_VERSION = 1

COMPONENT_NAMES = ("vision", "language", "action")
FROZEN_ROUTE = "frozen"
PLACEMENT_PRE_BLOCK = "pre_block_residual"
SITE_BLOCK_INPUT = "transformer_block_input"
SITE_HEAD_KERNEL = "action_head_kernel"


class LegacyArtifactError(ValueError):
    """A version-one artifact was offered to a corrected research command.

    Version-one zoo, evidence, archive, and alignment artifacts were produced
    with post-block adapters (dead final-layer vision/language updates),
    per-task action normalization, and the single action-summary vector. They
    stay readable through ``wsl_vla.experiments.legacy`` for diagnostics, but
    are never migrated into a corrected experiment.
    """


def canonical_sha256(payload: Any) -> str:
    """SHA-256 of a JSON-serializable payload in canonical key order."""

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def require_schema(observed: Any, expected: int, artifact: str) -> None:
    """Reject an artifact whose individual schema version is not the current one."""

    if observed == expected:
        return
    if isinstance(observed, int) and 0 < observed < expected:
        raise LegacyArtifactError(
            f"{artifact} has schema {observed}; corrected research commands require "
            f"schema {expected}. Legacy artifacts are inspection-only "
            "(scripts/reporting/inspect_legacy_artifacts.py); regenerate them instead."
        )
    raise ValueError(f"unsupported {artifact} schema {observed!r}; expected {expected}")


class Component(str, Enum):
    VISION = "vision"
    LANGUAGE = "language"
    ACTION = "action"


@dataclass(frozen=True)
class ActionConvention:
    """One fixed policy-action coordinate system shared by every zoo task.

    The policy is trained and sampled directly in controller coordinates
    (identity normalization), so adapters from different tasks live in one
    comparable output space and a generated adapter needs no task statistics.
    """

    name: str
    controller: str
    dimension_names: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    gripper_dimension: int
    gripper_open: float
    gripper_close: float
    policy_normalization: str = "identity"

    def __post_init__(self) -> None:
        count = len(self.dimension_names)
        if count == 0 or len(set(self.dimension_names)) != count:
            raise ValueError("action dimension names must be unique and non-empty")
        if len(self.low) != count or len(self.high) != count:
            raise ValueError("controller bounds must cover every action dimension")
        if any(not low < high for low, high in zip(self.low, self.high)):
            raise ValueError("controller lower bounds must lie below upper bounds")
        if not 0 <= self.gripper_dimension < count:
            raise ValueError("gripper dimension lies outside the action vector")
        if self.gripper_open == self.gripper_close:
            raise ValueError("gripper open/close commands must differ")
        if self.policy_normalization != "identity":
            raise ValueError("the zoo protocol fixes identity policy normalization")

    @property
    def action_dim(self) -> int:
        return len(self.dimension_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "controller": self.controller,
            "dimension_names": list(self.dimension_names),
            "low": [float(value) for value in self.low],
            "high": [float(value) for value in self.high],
            "gripper_dimension": int(self.gripper_dimension),
            "gripper_open": float(self.gripper_open),
            "gripper_close": float(self.gripper_close),
            "policy_normalization": self.policy_normalization,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionConvention":
        return cls(
            name=str(payload["name"]),
            controller=str(payload["controller"]),
            dimension_names=tuple(str(value) for value in payload["dimension_names"]),
            low=tuple(float(value) for value in payload["low"]),
            high=tuple(float(value) for value in payload["high"]),
            gripper_dimension=int(payload["gripper_dimension"]),
            gripper_open=float(payload["gripper_open"]),
            gripper_close=float(payload["gripper_close"]),
            policy_normalization=str(payload["policy_normalization"]),
        )

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


# LIBERO's OSC_POSE controller consumes six delta-pose commands and one gripper
# command, each in [-1, 1]; -1 opens and +1 closes the gripper. Demonstration
# HDF5 files store exactly these commands and ``env.step`` consumes them.
LIBERO_ACTION_CONVENTION = ActionConvention(
    name="libero_osc_pose_delta_v1",
    controller="OSC_POSE",
    dimension_names=("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"),
    low=(-1.0,) * 7,
    high=(1.0,) * 7,
    gripper_dimension=6,
    gripper_open=-1.0,
    gripper_close=1.0,
)


@dataclass(frozen=True)
class AdapterEntry:
    """One low-rank update target in a frozen base model."""

    component: Component
    layer: int
    parameter_path: str
    input_dim: int
    output_dim: int
    rank: int
    site: str = SITE_BLOCK_INPUT

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if min(self.input_dim, self.output_dim, self.rank) <= 0:
            raise ValueError("adapter dimensions and rank must be positive")
        if not self.parameter_path:
            raise ValueError("parameter_path cannot be empty")
        if self.site not in {SITE_BLOCK_INPUT, SITE_HEAD_KERNEL}:
            raise ValueError(f"unknown adapter site: {self.site}")
        if self.site == SITE_HEAD_KERNEL and self.component is not Component.ACTION:
            raise ValueError("action-head kernels belong to the action component")

    @property
    def effective_rank_bound(self) -> int:
        """Largest rank a deployed update for this entry may have."""

        return min(self.rank, self.input_dim, self.output_dim)


@dataclass(frozen=True)
class AdapterSpec:
    """Immutable adapter layout tied to one exact pretrained base.

    ``token_routing`` is the token-conditioned factorization used to adapt the
    proposal to Octo's single shared transformer: every token group named by the
    checkpoint is routed to exactly one modality adapter or is explicitly frozen.
    """

    base_model_id: str
    base_revision: str
    base_sha256: str
    alpha: float
    token_width: int
    entries: tuple[AdapterEntry, ...]
    architecture_sha256: str
    token_routing: tuple[tuple[str, str], ...]
    action_convention: ActionConvention
    placement: str = PLACEMENT_PRE_BLOCK
    schema_version: int = ADAPTER_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_schema(self.schema_version, ADAPTER_SPEC_SCHEMA_VERSION, "AdapterSpec")
        for name, value in (
            ("base_sha256", self.base_sha256),
            ("architecture_sha256", self.architecture_sha256),
        ):
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")
        if self.alpha <= 0 or self.token_width <= 0:
            raise ValueError("alpha and token_width must be positive")
        if self.placement != PLACEMENT_PRE_BLOCK:
            raise ValueError(
                f"unsupported adapter placement {self.placement!r}; corrected zoo "
                f"adapters are applied as {PLACEMENT_PRE_BLOCK}"
            )
        paths = [entry.parameter_path for entry in self.entries]
        if len(paths) != len(set(paths)):
            raise ValueError("adapter parameter paths must be unique")
        if not self.token_routing:
            raise ValueError("AdapterSpec requires an explicit token routing table")
        groups = [group for group, _ in self.token_routing]
        if groups != sorted(groups) or len(groups) != len(set(groups)):
            raise ValueError("token routing must list each group once in sorted order")
        allowed = {*COMPONENT_NAMES, FROZEN_ROUTE}
        unknown = sorted({route for _, route in self.token_routing} - allowed)
        if unknown:
            raise ValueError(f"token routing uses unknown destinations: {unknown}")

    @property
    def routing(self) -> dict[str, str]:
        return dict(self.token_routing)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_model_id": self.base_model_id,
            "base_revision": self.base_revision,
            "base_sha256": self.base_sha256,
            "architecture_sha256": self.architecture_sha256,
            "alpha": float(self.alpha),
            "token_width": int(self.token_width),
            "placement": self.placement,
            "token_routing": {group: route for group, route in self.token_routing},
            "action_convention": self.action_convention.to_dict(),
            "entries": [
                {
                    "component": entry.component.value,
                    "layer": int(entry.layer),
                    "parameter_path": entry.parameter_path,
                    "input_dim": int(entry.input_dim),
                    "output_dim": int(entry.output_dim),
                    "rank": int(entry.rank),
                    "site": entry.site,
                }
                for entry in self.entries
            ],
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdapterSpec":
        require_schema(
            payload.get("schema_version"), ADAPTER_SPEC_SCHEMA_VERSION, "AdapterSpec"
        )
        entries = tuple(
            AdapterEntry(
                component=Component(item["component"]),
                layer=int(item["layer"]),
                parameter_path=str(item["parameter_path"]),
                input_dim=int(item["input_dim"]),
                output_dim=int(item["output_dim"]),
                rank=int(item["rank"]),
                site=str(item["site"]),
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
            architecture_sha256=str(payload["architecture_sha256"]),
            token_routing=tuple(
                sorted((str(group), str(route)) for group, route in payload["token_routing"].items())
            ),
            action_convention=ActionConvention.from_dict(payload["action_convention"]),
            placement=str(payload["placement"]),
            schema_version=int(payload["schema_version"]),
        )


@dataclass
class TaskEvidence:
    """Cached frozen-Octo features and command summaries for one task.

    Nothing here is trained or standardized: visual aggregation, evidence
    projection, and fold-specific standardization all happen during alignment.
    """

    task_id: str
    suite: str
    instruction: str
    episode_ids: tuple[str, ...]
    vision_features: np.ndarray
    vision_episode_index: np.ndarray
    vision_timestep: np.ndarray
    language_features: np.ndarray
    language_mask: np.ndarray
    action_chunk_features: np.ndarray
    action_chunk_mask: np.ndarray
    feature_extractor_sha256: Mapping[str, str]
    preprocessing_sha256: str
    action_convention_sha256: str
    base_sha256: str
    base_revision: str
    architecture_sha256: str
    dataset_sha256: str
    extraction: Mapping[str, Any] = field(default_factory=dict)
    synthetic: bool = False
    schema_version: int = TASK_EVIDENCE_SCHEMA_VERSION

    def validate_for_research(self) -> None:
        require_schema(self.schema_version, TASK_EVIDENCE_SCHEMA_VERSION, "TaskEvidence")
        if self.synthetic:
            raise ValueError("synthetic evidence is forbidden in research runs")
        if not self.task_id or not self.suite or not self.instruction.strip():
            raise ValueError("evidence task, suite, and instruction identities are required")
        if not self.episode_ids or len(self.episode_ids) != len(set(self.episode_ids)):
            raise ValueError("episode_ids must be non-empty and unique")
        episodes = len(self.episode_ids)
        vision = np.asarray(self.vision_features)
        episode_index = np.asarray(self.vision_episode_index)
        timestep = np.asarray(self.vision_timestep)
        if vision.ndim != 2 or vision.shape[0] == 0:
            raise ValueError("vision_features must have shape [frames, feature_dim]")
        if episode_index.shape != (vision.shape[0],) or timestep.shape != episode_index.shape:
            raise ValueError("every cached visual frame needs an episode index and timestep")
        if episode_index.min() < 0 or episode_index.max() >= episodes or timestep.min() < 0:
            raise ValueError("visual frame identities lie outside the training episodes")
        if set(np.unique(episode_index).tolist()) != set(range(episodes)):
            raise ValueError("every training episode must contribute visual evidence")
        language = np.asarray(self.language_features)
        language_mask = np.asarray(self.language_mask)
        if language.ndim != 2 or language_mask.shape != language.shape[:1]:
            raise ValueError("language features must be [tokens, dim] with a token mask")
        if language_mask.dtype != bool or not language_mask.any():
            raise ValueError("language_mask must be boolean with at least one valid token")
        chunks = np.asarray(self.action_chunk_features)
        chunk_mask = np.asarray(self.action_chunk_mask)
        if chunks.ndim != 3 or chunks.shape[0] != episodes:
            raise ValueError("action chunks must be [episodes, chunks, feature_dim]")
        if chunk_mask.shape != chunks.shape[:2] or chunk_mask.dtype != bool:
            raise ValueError("action_chunk_mask must be a boolean [episodes, chunks] array")
        if not chunk_mask.any(axis=1).all():
            raise ValueError("every training episode needs at least one valid action chunk")
        for name, array, valid in (
            ("vision_features", vision, None),
            ("language_features", language, language_mask),
            ("action_chunk_features", chunks, chunk_mask),
        ):
            selected = array if valid is None else array[valid]
            if not np.isfinite(selected).all():
                raise ValueError(f"{name} contains non-finite valid entries")
        if set(self.feature_extractor_sha256) != {"vision", "language"}:
            raise ValueError("vision and language feature-extractor hashes are required")
        for name, value in (
            *((f"feature_extractor_sha256[{key}]", digest)
              for key, digest in self.feature_extractor_sha256.items()),
            ("preprocessing_sha256", self.preprocessing_sha256),
            ("action_convention_sha256", self.action_convention_sha256),
            ("base_sha256", self.base_sha256),
            ("architecture_sha256", self.architecture_sha256),
            ("dataset_sha256", self.dataset_sha256),
        ):
            if len(str(value)) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")


@dataclass
class AlignmentCheckpoint:
    """Reloaded alignment system and all inference-time geometric artifacts."""

    params: Any
    model: Any
    config: Mapping[str, Any]
    adapter_spec: AdapterSpec
    layout: Any
    entry_scales: np.ndarray
    standardization: Any
    mappers: Mapping[str, Any]
    shells: Mapping[str, Any]
    metadata: Mapping[str, Any]
    schema_version: int = ALIGNMENT_CHECKPOINT_SCHEMA_VERSION

    @property
    def token_mask(self) -> np.ndarray:
        return self.layout.token_mask

    @property
    def component_ids(self) -> np.ndarray:
        return self.layout.component_ids

    @property
    def layer_ids(self) -> np.ndarray:
        return self.layout.layer_ids

    @property
    def condition(self) -> str:
        return str(self.config["condition"])

    def validate(self) -> None:
        require_schema(
            self.schema_version, ALIGNMENT_CHECKPOINT_SCHEMA_VERSION, "AlignmentCheckpoint"
        )
        if set(self.mappers) != set(COMPONENT_NAMES):
            raise ValueError("alignment checkpoint requires all three modality mappers")
        if set(self.shells) != set(COMPONENT_NAMES):
            raise ValueError("alignment checkpoint requires all three empirical shells")
        if self.token_mask.ndim != 2 or self.component_ids.shape != self.token_mask.shape[:1]:
            raise ValueError("alignment token layout is inconsistent")
        if self.layer_ids.shape != self.component_ids.shape:
            raise ValueError("alignment layer layout is inconsistent")
        scales = np.asarray(self.entry_scales)
        if scales.shape != (len(self.adapter_spec.entries),):
            raise ValueError("alignment checkpoint needs one RMS scale per adapter entry")
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("per-entry RMS scales must be finite and positive")
        if self.config.get("condition") not in {"aligned", "reconstruction_only"}:
            raise ValueError("alignment checkpoint condition is not a primary condition")


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
    training_update_steps: int = 0
    training_micro_steps: int = 0
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
        if (
            self.training_update_steps < 0
            or self.training_micro_steps < self.training_update_steps
        ):
            raise ValueError("training step counts are inconsistent")
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
