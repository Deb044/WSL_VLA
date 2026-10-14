"""Validation and deterministic expansion of the publication protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from ..contracts import LIBERO_ACTION_CONVENTION, PLACEMENT_PRE_BLOCK
from ..data.splits import SuiteFold, leave_one_suite_out_folds


REQUIRED_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)
LIBERO_GIT_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def validate_reference_tasks(payload: Mapping[str, Any]) -> None:
    suites = payload.get("suites")
    if not isinstance(suites, Mapping) or tuple(suites.keys()) != REQUIRED_SUITES:
        raise ValueError(f"reference tasks must define suites in order {REQUIRED_SUITES}")
    for suite, tasks in suites.items():
        if not isinstance(tasks, list) or len(tasks) != 10:
            raise ValueError(f"{suite} must contain exactly ten ordered tasks")
        if len(set(tasks)) != 10 or any(not isinstance(task, str) or not task.strip() for task in tasks):
            raise ValueError(f"{suite} tasks must be ten unique non-empty strings")


REQUIRED_ALIGNMENT_KEYS = (
    "latent_dim",
    "transformer_width",
    "encoder_layers",
    "decoder_layers",
    "attention_heads",
    "mlp_ratio",
    "evidence_hidden_dim",
    "initial_temperature",
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "gradient_accumulation_steps",
    "reconstruction_weight",
    "contrastive_weight",
    "auxiliary_classification_weight",
    "modality_alignment_weights",
    "rms_floor",
    "window_chunk",
    "ridge_alpha_candidates",
)
REQUIRED_PREPROCESSING_KEYS = (
    "source_camera_height",
    "source_camera_width",
    "demonstration_image_convention",
    "image_transform",
    "simulator_alignment",
    "executed_actions_per_plan",
)
RESEARCH_CONFIG_SCHEMA_VERSION = 2


def validate_research_config(config: Mapping[str, Any]) -> None:
    """Validate the model/optimizer configuration shared by every experiment.

    Population sizes, seeds, snapshot stages, and folds are deliberately absent:
    they belong to the experiment manifest (``wsl_vla.experiments.manifest``).
    """

    if config.get("schema_version") != RESEARCH_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"research config schema_version must be {RESEARCH_CONFIG_SCHEMA_VERSION}"
        )
    if not config.get("data", {}).get("require_real_data", False):
        raise ValueError("publication config must require real data")
    if config.get("data", {}).get("synthetic_smoke_test", True):
        raise ValueError("publication config cannot enable synthetic smoke data")
    if config.get("model", {}).get("octo_git_revision") is None:
        raise ValueError("Octo git revision must be pinned")
    if config.get("simulator", {}).get("git_revision") != LIBERO_GIT_REVISION:
        raise ValueError("LIBERO git revision differs from the locked research version")
    simulator = config.get("simulator", {})
    if int(simulator.get("max_episode_steps", 0)) <= 0 or int(simulator.get("warmup_steps", -1)) < 0:
        raise ValueError("LIBERO rollout horizons are invalid")
    if "camera_height" in simulator or "camera_width" in simulator:
        raise ValueError(
            "simulator camera size is no longer configurable separately; rollouts render "
            "at preprocessing.source_camera_* so they match the demonstrations"
        )
    adapter = config.get("adapter", {})
    if min(int(adapter.get("rank", 0)), int(adapter.get("token_width", 0)),
           int(adapter.get("sequence_window", 0))) <= 0 or float(adapter.get("alpha", 0)) <= 0:
        raise ValueError("adapter rank, alpha, token width, and sequence window must be positive")
    if adapter.get("placement") != PLACEMENT_PRE_BLOCK:
        raise ValueError(f"adapter placement must be {PLACEMENT_PRE_BLOCK}")
    preprocessing = config.get("preprocessing", {})
    missing = [key for key in REQUIRED_PREPROCESSING_KEYS if key not in preprocessing]
    if missing:
        raise ValueError(f"preprocessing config lacks required keys: {missing}")
    if config.get("action_convention") != LIBERO_ACTION_CONVENTION.name:
        raise ValueError(
            f"action_convention must be the fixed {LIBERO_ACTION_CONVENTION.name}"
        )
    alignment = config.get("alignment", {})
    missing = [key for key in REQUIRED_ALIGNMENT_KEYS if key not in alignment]
    if missing:
        raise ValueError(
            f"alignment config lacks required keys {missing}; there are no architecture "
            "defaults in code"
        )
    modality_weights = alignment["modality_alignment_weights"]
    if set(modality_weights) != {"vision", "language", "action"}:
        raise ValueError("alignment must define all three modality loss weights")
    if any(float(value) < 0 for value in modality_weights.values()):
        raise ValueError("modality alignment weights cannot be negative")
    if float(alignment["reconstruction_weight"]) <= 0:
        raise ValueError("reconstruction weight must be positive")
    if float(alignment["rms_floor"]) <= 0:
        raise ValueError("the per-entry RMS floor must be positive")
    if not alignment["ridge_alpha_candidates"] or min(alignment["ridge_alpha_candidates"]) < 0:
        raise ValueError("ridge strength candidates must be non-negative and non-empty")
    conditions = set(config.get("continual_learning", {}).get("primary_conditions", []))
    required_conditions = {
        "sequential_no_regularization",
        "replay_10",
        "replay_100",
        "uniform_regularization",
        "proposed_asymmetric",
        "direction_inverted",
        "reconstruction_only_latent",
        "independent_adapter_oracle",
    }
    if conditions != required_conditions:
        raise ValueError("primary condition set differs from the locked publication protocol")


def publication_folds() -> tuple[SuiteFold, ...]:
    return leave_one_suite_out_folds(REQUIRED_SUITES)
