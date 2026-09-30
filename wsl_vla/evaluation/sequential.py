"""Protocol-safe sequential continual-learning experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Generic, Mapping, Sequence, TypeVar

from ..contracts import EvaluationRecord


State = TypeVar("State")


@dataclass(frozen=True)
class StageUpdate(Generic[State]):
    state: State
    checkpoint_sha256: str
    latent_drift: Mapping[str, float]
    task_loss: float | None = None
    regularization_loss: float | None = None
    adapter_bytes: int = 0
    evidence_bytes: int = 0
    peak_vram_bytes: int | None = None
    peak_ram_bytes: int | None = None


@dataclass(frozen=True)
class RolloutOutcome:
    successes: int
    initialization_indices: tuple[int, ...]
    rollout_seeds: tuple[int, ...]


TrainStage = Callable[[State, str, int], StageUpdate[State]]
Rollout = Callable[[State, str, int, int], RolloutOutcome]


def run_sequential_protocol(
    *,
    initial_state: State,
    task_ids: Sequence[str],
    train_stage: TrainStage[State],
    rollout: Rollout[State],
    run_id: str,
    suite: str,
    seed: int,
    condition: str,
    rollout_count: int,
) -> tuple[State, tuple[EvaluationRecord, ...]]:
    """Train one evolving state and evaluate that same state on all seen tasks.

    The interface intentionally offers no per-task checkpoint lookup. This makes
    substituting stored task adapters during continual evaluation impossible.
    """

    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("task_ids must be non-empty and unique")
    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")

    state = initial_state
    records: list[EvaluationRecord] = []
    for stage, task_id in enumerate(task_ids):
        started = perf_counter()
        update = train_stage(state, task_id, stage)
        if len(update.checkpoint_sha256) != 64:
            raise ValueError("train_stage must return a SHA-256 checkpoint identity")
        state = update.state
        train_seconds = perf_counter() - started
        for evaluated_index, evaluated_task in enumerate(task_ids[: stage + 1]):
            evaluation_started = perf_counter()
            outcome = rollout(state, evaluated_task, rollout_count, seed)
            evaluation_seconds = perf_counter() - evaluation_started
            if not isinstance(outcome, RolloutOutcome):
                raise TypeError("publication rollouts must return initialization provenance")
            record = EvaluationRecord(
                run_id=run_id,
                suite=suite,
                seed=seed,
                condition=condition,
                training_stage=stage,
                evaluated_task_index=evaluated_index,
                evaluated_task_id=evaluated_task,
                rollout_count=rollout_count,
                successes=int(outcome.successes),
                checkpoint_sha256=update.checkpoint_sha256,
                latent_drift=dict(update.latent_drift),
                wall_time_seconds=train_seconds,
                initialization_indices=outcome.initialization_indices,
                rollout_seeds=outcome.rollout_seeds,
                evaluation_wall_time_seconds=evaluation_seconds,
                task_loss=update.task_loss,
                regularization_loss=update.regularization_loss,
                adapter_bytes=update.adapter_bytes,
                evidence_bytes=update.evidence_bytes,
                peak_vram_bytes=update.peak_vram_bytes,
                peak_ram_bytes=update.peak_ram_bytes,
            )
            _ = record.success_rate
            records.append(record)
    return state, tuple(records)
