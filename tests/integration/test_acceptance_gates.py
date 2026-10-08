from dataclasses import replace

import numpy as np
import pytest

from wsl_vla.contracts import EvaluationRecord
from wsl_vla.evaluation.gates import (
    one_task_learning_gate,
    rollout_summaries_match,
    two_task_success_matrix,
)
from wsl_vla.evaluation.libero_rollout import EpisodeRollout, TaskRolloutSummary


def summary(successes: int, checkpoint: str = "a" * 64) -> TaskRolloutSummary:
    episodes = tuple(
        EpisodeRollout(
            task_id="libero_spatial_0",
            task_index=0,
            instruction="do the task",
            initialization_index=index,
            episode_seed=100 + index,
            steps=20 + index,
            success=index < successes,
            episode_return=float(index < successes),
        )
        for index in range(4)
    )
    return TaskRolloutSummary(
        suite="libero_spatial",
        task_id="libero_spatial_0",
        task_index=0,
        policy_checkpoint_sha256=checkpoint,
        episodes=episodes,
    )


def test_one_task_gate_requires_improvement_and_determinism():
    baseline = summary(1)
    adapted = summary(2, "b" * 64)
    result = one_task_learning_gate(
        baseline_first=baseline,
        baseline_repeat=baseline,
        adapted_first=adapted,
        adapted_repeat=adapted,
    )
    assert result.passed
    no_improvement = one_task_learning_gate(
        baseline_first=baseline,
        baseline_repeat=baseline,
        adapted_first=baseline,
        adapted_repeat=baseline,
        enforce=False,
    )
    assert not no_improvement.passed


def test_one_task_gate_detects_nondeterministic_episode_outcome():
    baseline = summary(1)
    adapted = summary(2, "b" * 64)
    changed_episode = replace(adapted.episodes[0], steps=adapted.episodes[0].steps + 1)
    changed = replace(adapted, episodes=(changed_episode, *adapted.episodes[1:]))
    assert not rollout_summaries_match(adapted, changed)
    with pytest.raises(AssertionError, match="one-task learning gate failed"):
        one_task_learning_gate(
            baseline_first=baseline,
            baseline_repeat=baseline,
            adapted_first=adapted,
            adapted_repeat=changed,
        )


def test_one_task_gate_rejects_mismatched_task_population():
    baseline = summary(1)
    adapted = summary(2, "b" * 64)
    wrong_task = replace(adapted, task_index=1, task_id="libero_spatial_1")
    with pytest.raises(ValueError, match="same task"):
        one_task_learning_gate(
            baseline_first=baseline,
            baseline_repeat=baseline,
            adapted_first=adapted,
            adapted_repeat=wrong_task,
        )


def evaluation_record(stage: int, task: int, successes: int) -> EvaluationRecord:
    return EvaluationRecord(
        run_id="run",
        suite="libero_spatial",
        seed=17,
        condition="sequential_no_regularization",
        training_stage=stage,
        evaluated_task_index=task,
        evaluated_task_id=f"libero_spatial_{task}",
        rollout_count=4,
        successes=successes,
        checkpoint_sha256="a" * 64,
        latent_drift={"vision": 0.0, "language": 0.0, "action": 0.0},
        wall_time_seconds=1.0,
        initialization_indices=(0, 1, 2, 3),
        rollout_seeds=(10, 11, 12, 13),
        evaluation_wall_time_seconds=1.0,
    )


def test_two_task_gate_requires_complete_lower_triangle():
    records = (
        evaluation_record(0, 0, 1),
        evaluation_record(1, 0, 2),
        evaluation_record(1, 1, 3),
    )
    matrix = two_task_success_matrix(records)
    assert matrix[0, 0] == 0.25
    assert matrix[1, 0] == 0.5
    assert matrix[1, 1] == 0.75
    assert np.isnan(matrix[0, 1])
    with pytest.raises(ValueError, match="exactly"):
        two_task_success_matrix(records[:-1])
