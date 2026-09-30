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
    update_steps: int = 0
    micro_steps: int = 0


@dataclass(frozen=True)
class RolloutOutcome:
    successes: int
    initialization_indices: tuple[int, ...]
    rollout_seeds: tuple[int, ...]
    checkpoint_sha256: str | None = None


TrainStage = Callable[[State, str, int], StageUpdate[State]]
Rollout = Callable[[State, str, int, int], RolloutOutcome]
StageSink = Callable[[int, str, StageUpdate[State]], None]
RecordSink = Callable[[EvaluationRecord], None]


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
    stage_sink: StageSink[State] | None = None,
    record_sink: RecordSink | None = None,
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
        if stage_sink is not None:
            stage_sink(stage, task_id, update)
        for evaluated_index, evaluated_task in enumerate(task_ids[: stage + 1]):
            evaluation_started = perf_counter()
            outcome = rollout(state, evaluated_task, rollout_count, seed)
            evaluation_seconds = perf_counter() - evaluation_started
            if not isinstance(outcome, RolloutOutcome):
                raise TypeError("publication rollouts must return initialization provenance")
            if (
                outcome.checkpoint_sha256 is not None
                and outcome.checkpoint_sha256 != update.checkpoint_sha256
            ):
                raise ValueError("rollout policy differs from the trained shared checkpoint")
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
                training_update_steps=update.update_steps,
                training_micro_steps=update.micro_steps,
            )
            _ = record.success_rate
            records.append(record)
            if record_sink is not None:
                record_sink(record)
    return state, tuple(records)


def run_independent_oracle_protocol(
    *,
    initial_state: State,
    task_ids: Sequence[str],
    train_task: TrainStage[State],
    rollout: Rollout[State],
    run_id: str,
    suite: str,
    seed: int,
    rollout_count: int,
    stage_sink: StageSink[State] | None = None,
    record_sink: RecordSink | None = None,
) -> tuple[Mapping[str, State], tuple[EvaluationRecord, ...]]:
    """Evaluate an explicit independent-adapter upper bound.

    Each task is trained from the identical initial state. At later stages the
    evaluator deliberately selects that task's own stored adapter. Keeping this
    exceptional behavior in a separate function prevents it from leaking into
    any shared-state continual condition.
    """

    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("task_ids must be non-empty and unique")
    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")

    updates: dict[str, StageUpdate[State]] = {}
    train_seconds: dict[str, float] = {}
    records: list[EvaluationRecord] = []
    for stage, task_id in enumerate(task_ids):
        started = perf_counter()
        update = train_task(initial_state, task_id, stage)
        train_seconds[task_id] = perf_counter() - started
        if len(update.checkpoint_sha256) != 64:
            raise ValueError("train_task must return a SHA-256 checkpoint identity")
        updates[task_id] = update
        if stage_sink is not None:
            stage_sink(stage, task_id, update)

        for evaluated_index, evaluated_task in enumerate(task_ids[: stage + 1]):
            selected = updates[evaluated_task]
            evaluation_started = perf_counter()
            outcome = rollout(selected.state, evaluated_task, rollout_count, seed)
            evaluation_seconds = perf_counter() - evaluation_started
            if not isinstance(outcome, RolloutOutcome):
                raise TypeError("publication rollouts must return initialization provenance")
            if (
                outcome.checkpoint_sha256 is not None
                and outcome.checkpoint_sha256 != selected.checkpoint_sha256
            ):
                raise ValueError("oracle rollout policy differs from its stored task checkpoint")
            record = EvaluationRecord(
                run_id=run_id,
                suite=suite,
                seed=seed,
                condition="independent_adapter_oracle",
                training_stage=stage,
                evaluated_task_index=evaluated_index,
                evaluated_task_id=evaluated_task,
                rollout_count=rollout_count,
                successes=int(outcome.successes),
                checkpoint_sha256=selected.checkpoint_sha256,
                latent_drift=dict(selected.latent_drift),
                wall_time_seconds=train_seconds[evaluated_task],
                initialization_indices=outcome.initialization_indices,
                rollout_seeds=outcome.rollout_seeds,
                evaluation_wall_time_seconds=evaluation_seconds,
                task_loss=selected.task_loss,
                regularization_loss=selected.regularization_loss,
                adapter_bytes=selected.adapter_bytes,
                evidence_bytes=selected.evidence_bytes,
                peak_vram_bytes=selected.peak_vram_bytes,
                peak_ram_bytes=selected.peak_ram_bytes,
                training_update_steps=selected.update_steps,
                training_micro_steps=selected.micro_steps,
            )
            _ = record.success_rate
            records.append(record)
            if record_sink is not None:
                record_sink(record)
    return {task_id: update.state for task_id, update in updates.items()}, tuple(records)
