"""Protocol-safe sequential continual-learning experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Generic, Mapping, Sequence, TypeVar

from core.contracts import EvaluationRecord


State = TypeVar("State")


@dataclass(frozen=True)
class StageUpdate(Generic[State]):
    state: State
    checkpoint_sha256: str
    latent_drift: Mapping[str, float]


TrainStage = Callable[[State, str, int], StageUpdate[State]]
Rollout = Callable[[State, str, int, int], int]


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
            successes = int(rollout(state, evaluated_task, rollout_count, seed))
            record = EvaluationRecord(
                run_id=run_id,
                suite=suite,
                seed=seed,
                condition=condition,
                training_stage=stage,
                evaluated_task_index=evaluated_index,
                evaluated_task_id=evaluated_task,
                rollout_count=rollout_count,
                successes=successes,
                checkpoint_sha256=update.checkpoint_sha256,
                latent_drift=dict(update.latent_drift),
                wall_time_seconds=train_seconds,
            )
            _ = record.success_rate
            records.append(record)
    return state, tuple(records)
