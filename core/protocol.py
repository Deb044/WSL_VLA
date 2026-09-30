"""Validation and deterministic expansion of the publication protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from data.splits import SuiteFold, leave_one_suite_out_folds


REQUIRED_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


@dataclass(frozen=True)
class PopulationRun:
    task_id: str
    suite: str
    task_index: int
    seed: int
    checkpoint_fraction: float


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a config path relative to cwd or repository root."""
    target = Path(path)
    if target.is_file():
        return target
    repo_root = Path(__file__).resolve().parent.parent
    if (repo_root / target).is_file():
        return repo_root / target
    if (repo_root / "configs" / target.name).is_file():
        return repo_root / "configs" / target.name
    return target


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_config_path(path)
    with resolved.open("r", encoding="utf-8") as stream:
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


def validate_research_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("research config schema_version must be 1")
    if not config.get("data", {}).get("require_real_data", False):
        raise ValueError("publication config must require real data")
    if config.get("data", {}).get("synthetic_smoke_test", True):
        raise ValueError("publication config cannot enable synthetic smoke data")
    seeds = config.get("population", {}).get("seeds", [])
    fractions = config.get("population", {}).get("late_checkpoint_fractions", [])
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("publication model zoo requires three unique seeds")
    if len(fractions) != 3 or sorted(fractions) != list(fractions):
        raise ValueError("publication model zoo requires three ordered late checkpoints")
    if config.get("model", {}).get("octo_git_revision") is None:
        raise ValueError("Octo git revision must be pinned")
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


def expand_population_runs(
    task_payload: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[PopulationRun, ...]:
    validate_reference_tasks(task_payload)
    validate_research_config(config)
    seeds = config["population"]["seeds"]
    fractions = config["population"]["late_checkpoint_fractions"]
    runs = []
    for suite, tasks in task_payload["suites"].items():
        for task_index, task in enumerate(tasks):
            task_id = f"{suite}_{task_index}"
            for seed in seeds:
                for fraction in fractions:
                    runs.append(PopulationRun(task_id, suite, task_index, seed, fraction))
    if len(runs) != 360:
        raise AssertionError(f"expected 360 adapter checkpoints, got {len(runs)}")
    return tuple(runs)


def publication_folds() -> tuple[SuiteFold, ...]:
    return leave_one_suite_out_folds(REQUIRED_SUITES)
