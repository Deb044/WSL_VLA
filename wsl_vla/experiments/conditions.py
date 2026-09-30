"""Locked continual-learning conditions and deterministic replay memory."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MODALITIES = ("vision", "language", "action")
PRIMARY_CONDITIONS = (
    "sequential_no_regularization",
    "replay_10",
    "replay_100",
    "uniform_regularization",
    "proposed_asymmetric",
    "direction_inverted",
    "reconstruction_only_latent",
    "independent_adapter_oracle",
)


@dataclass(frozen=True)
class GammaSelection:
    """Hyperparameters selected without looking at the held-out stream."""

    held_out_suite: str
    validation_suites: tuple[str, ...]
    proposed: Mapping[str, float]
    uniform: float
    early_stopping_patience: int

    def validate(self) -> None:
        if self.held_out_suite in self.validation_suites:
            raise ValueError("the held-out test suite cannot select hyperparameters")
        if len(self.validation_suites) != 3 or len(set(self.validation_suites)) != 3:
            raise ValueError("gamma selection requires three distinct training suites")
        if set(self.proposed) != set(MODALITIES):
            raise ValueError("proposed gammas must define every modality")
        values = {name: float(value) for name, value in self.proposed.items()}
        if min(*values.values(), float(self.uniform)) < 0:
            raise ValueError("regularization strengths cannot be negative")
        if not values["vision"] > values["action"]:
            raise ValueError("proposed gamma_vision must exceed gamma_action")
        if not values["language"] > values["action"]:
            raise ValueError("proposed gamma_language must exceed gamma_action")
        if self.early_stopping_patience <= 0:
            raise ValueError("early stopping must be selected on validation data")


def save_gamma_selection(selection: GammaSelection, path: str | Path) -> None:
    """Persist the frozen validation result consumed by held-out test runs."""

    selection.validate()
    payload = {
        "schema_version": 1,
        "held_out_suite": selection.held_out_suite,
        "validation_suites": list(selection.validation_suites),
        "proposed": {name: float(selection.proposed[name]) for name in MODALITIES},
        "uniform": float(selection.uniform),
        "early_stopping_patience": int(selection.early_stopping_patience),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_gamma_selection(path: str | Path, *, held_out_suite: str) -> GammaSelection:
    """Load hyperparameters and require they were selected for this exact fold."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported gamma-selection schema")
    selection = GammaSelection(
        held_out_suite=str(payload["held_out_suite"]),
        validation_suites=tuple(str(value) for value in payload["validation_suites"]),
        proposed={str(key): float(value) for key, value in payload["proposed"].items()},
        uniform=float(payload["uniform"]),
        early_stopping_patience=int(payload["early_stopping_patience"]),
    )
    selection.validate()
    if selection.held_out_suite != held_out_suite:
        raise ValueError(
            "gamma selection belongs to a different held-out suite: "
            f"{selection.held_out_suite} != {held_out_suite}"
        )
    return selection


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    optimization_space: str
    replay_per_previous_task: int
    gammas: Mapping[str, float]
    aligned_latent_space: bool | None
    independent_per_task: bool = False

    def validate(self) -> None:
        if self.name not in PRIMARY_CONDITIONS:
            raise ValueError(f"unknown publication condition: {self.name}")
        if self.optimization_space not in {"lora", "latent", "oracle"}:
            raise ValueError("optimization_space must be lora, latent, or oracle")
        if self.replay_per_previous_task not in {0, 10, 100}:
            raise ValueError("publication replay size must be 0, 10, or 100")
        if set(self.gammas) != set(MODALITIES):
            raise ValueError("condition must define all modality gammas")
        if any(float(value) < 0 for value in self.gammas.values()):
            raise ValueError("condition gammas cannot be negative")
        if self.independent_per_task != (self.optimization_space == "oracle"):
            raise ValueError("only the oracle may keep independent per-task state")


def locked_condition_specs(selection: GammaSelection) -> tuple[ConditionSpec, ...]:
    """Expand validation-selected values into all eight required comparisons."""

    selection.validate()
    proposed = {name: float(selection.proposed[name]) for name in MODALITIES}
    zero = {name: 0.0 for name in MODALITIES}
    uniform = {name: float(selection.uniform) for name in MODALITIES}
    inverted = {
        "vision": proposed["action"],
        "language": proposed["action"],
        "action": max(proposed["vision"], proposed["language"]),
    }
    specs = (
        ConditionSpec(PRIMARY_CONDITIONS[0], "lora", 0, zero, None),
        ConditionSpec(PRIMARY_CONDITIONS[1], "lora", 10, zero, None),
        ConditionSpec(PRIMARY_CONDITIONS[2], "lora", 100, zero, None),
        ConditionSpec(PRIMARY_CONDITIONS[3], "latent", 0, uniform, True),
        ConditionSpec(PRIMARY_CONDITIONS[4], "latent", 0, proposed, True),
        ConditionSpec(PRIMARY_CONDITIONS[5], "latent", 0, inverted, True),
        ConditionSpec(PRIMARY_CONDITIONS[6], "latent", 0, proposed, False),
        ConditionSpec(PRIMARY_CONDITIONS[7], "oracle", 0, zero, None, True),
    )
    for spec in specs:
        spec.validate()
    if tuple(spec.name for spec in specs) != PRIMARY_CONDITIONS:
        raise AssertionError("publication condition order changed")
    return specs


@dataclass(frozen=True)
class ReplayTransition:
    task_id: str
    episode_id: str
    timestep: int
    payload: Any

    def __post_init__(self) -> None:
        if not self.task_id or not self.episode_id or self.timestep < 0:
            raise ValueError("replay transition identity is incomplete")


class PerTaskReplayMemory:
    """A fixed, auditable sample of 10 or 100 transitions per prior task."""

    def __init__(self, *, transitions_per_task: int, seed: int) -> None:
        if transitions_per_task not in {10, 100}:
            raise ValueError("publication replay memory must store 10 or 100 transitions")
        self.transitions_per_task = int(transitions_per_task)
        self.seed = int(seed)
        self._by_task: dict[str, tuple[ReplayTransition, ...]] = {}

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._by_task)

    @property
    def transition_count(self) -> int:
        return sum(len(values) for values in self._by_task.values())

    def add_task(self, task_id: str, transitions: Iterable[ReplayTransition]) -> None:
        if task_id in self._by_task:
            raise ValueError(f"replay task already frozen: {task_id}")
        candidates = tuple(transitions)
        if any(item.task_id != task_id for item in candidates):
            raise ValueError("replay candidates contain a different task identity")
        identities = [(item.episode_id, item.timestep) for item in candidates]
        if len(identities) != len(set(identities)):
            raise ValueError("replay candidates contain duplicate transitions")
        if len(candidates) < self.transitions_per_task:
            raise ValueError(
                f"task {task_id} has {len(candidates)} transitions; "
                f"{self.transitions_per_task} are required"
            )
        digest = hashlib.sha256(f"{self.seed}:{task_id}".encode("utf-8")).digest()
        task_seed = int.from_bytes(digest[:8], "big", signed=False)
        indices = np.random.default_rng(task_seed).choice(
            len(candidates), size=self.transitions_per_task, replace=False
        )
        self._by_task[task_id] = tuple(candidates[int(index)] for index in indices)

    def prior_transitions(self, seen_task_ids: Sequence[str], *, current_task_id: str):
        """Return only frozen examples from tasks before the current stage."""

        if not seen_task_ids or seen_task_ids[-1] != current_task_id:
            raise ValueError("seen_task_ids must end with the current task")
        prior = tuple(seen_task_ids[:-1])
        unknown = set(prior) - set(self._by_task)
        if unknown:
            raise ValueError(f"replay memory lacks prior tasks: {sorted(unknown)}")
        unexpected = set(self._by_task) - set(prior)
        if unexpected:
            raise ValueError(f"replay memory contains future/unseen tasks: {sorted(unexpected)}")
        return tuple(item for task_id in prior for item in self._by_task[task_id])

    def provenance(self) -> dict[str, list[dict[str, int | str]]]:
        return {
            task_id: [
                {"episode_id": item.episode_id, "timestep": item.timestep}
                for item in transitions
            ]
            for task_id, transitions in self._by_task.items()
        }
