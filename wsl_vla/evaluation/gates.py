"""Acceptance gates that stop invalid research runs before expensive scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import EvaluationRecord
from .libero_rollout import TaskRolloutSummary


@dataclass(frozen=True)
class OneTaskLearningGateResult:
    baseline_successes: int
    adapted_successes: int
    rollout_count: int
    baseline_deterministic: bool
    adapted_deterministic: bool

    @property
    def improved(self) -> bool:
        return self.adapted_successes > self.baseline_successes

    @property
    def passed(self) -> bool:
        return self.improved and self.baseline_deterministic and self.adapted_deterministic


def rollout_summaries_match(
    first: TaskRolloutSummary,
    second: TaskRolloutSummary,
    *,
    return_tolerance: float = 1e-6,
) -> bool:
    """Compare all rollout outcomes that determine a reported success cell."""

    if return_tolerance < 0:
        raise ValueError("return tolerance cannot be negative")
    if (
        first.suite != second.suite
        or first.task_id != second.task_id
        or first.task_index != second.task_index
        or first.policy_checkpoint_sha256 != second.policy_checkpoint_sha256
        or len(first.episodes) != len(second.episodes)
    ):
        return False
    for left, right in zip(first.episodes, second.episodes):
        if (
            left.task_id != right.task_id
            or left.task_index != right.task_index
            or left.instruction != right.instruction
            or left.initialization_index != right.initialization_index
            or left.episode_seed != right.episode_seed
            or left.steps != right.steps
            or left.success != right.success
            or not np.isclose(
                left.episode_return,
                right.episode_return,
                atol=return_tolerance,
                rtol=0.0,
            )
        ):
            return False
    return True


def one_task_learning_gate(
    *,
    baseline_first: TaskRolloutSummary,
    baseline_repeat: TaskRolloutSummary,
    adapted_first: TaskRolloutSummary,
    adapted_repeat: TaskRolloutSummary,
    enforce: bool = True,
) -> OneTaskLearningGateResult:
    """Require strict rollout improvement and reproducibility at fixed seeds."""

    summaries = (baseline_first, baseline_repeat, adapted_first, adapted_repeat)
    identity = {(item.suite, item.task_id, item.task_index, item.rollout_count) for item in summaries}
    if len(identity) != 1:
        raise ValueError("one-task gate summaries must cover the same task and rollout count")
    result = OneTaskLearningGateResult(
        baseline_successes=baseline_first.successes,
        adapted_successes=adapted_first.successes,
        rollout_count=baseline_first.rollout_count,
        baseline_deterministic=rollout_summaries_match(
            baseline_first, baseline_repeat
        ),
        adapted_deterministic=rollout_summaries_match(adapted_first, adapted_repeat),
    )
    if enforce and not result.passed:
        raise AssertionError(f"one-task learning gate failed: {result}")
    return result


def two_task_success_matrix(records: tuple[EvaluationRecord, ...]) -> np.ndarray:
    """Build the locked lower-triangular 2-task scaling-gate matrix."""

    expected = {(0, 0), (1, 0), (1, 1)}
    observed = {
        (record.training_stage, record.evaluated_task_index) for record in records
    }
    if observed != expected or len(records) != len(expected):
        raise ValueError(
            "two-task gate requires exactly the stage/task cells (0,0), (1,0), and (1,1)"
        )
    identities = {(record.run_id, record.suite, record.seed, record.condition) for record in records}
    if len(identities) != 1:
        raise ValueError("two-task gate records must belong to one run")
    matrix = np.full((2, 2), np.nan, dtype=np.float64)
    for record in records:
        matrix[record.training_stage, record.evaluated_task_index] = record.success_rate
    return matrix
