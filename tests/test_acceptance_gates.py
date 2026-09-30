from dataclasses import replace

import pytest

from wsl_vla.evaluation.gates import one_task_learning_gate, rollout_summaries_match
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
