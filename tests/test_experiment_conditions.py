from __future__ import annotations

import pytest

from wsl_vla.experiments.conditions import (
    GammaSelection,
    PerTaskReplayMemory,
    PRIMARY_CONDITIONS,
    ReplayTransition,
    locked_condition_specs,
)


def selection():
    return GammaSelection(
        held_out_suite="libero_10",
        validation_suites=("libero_spatial", "libero_object", "libero_goal"),
        proposed={"vision": 1.0, "language": 2.0, "action": 0.1},
        uniform=0.5,
        early_stopping_patience=20,
    )


def transitions(task_id, count=120):
    return tuple(
        ReplayTransition(task_id, f"episode_{index // 4}", index % 4, index)
        for index in range(count)
    )


def test_all_locked_conditions_are_explicit_and_direction_is_inverted():
    specs = locked_condition_specs(selection())
    assert tuple(item.name for item in specs) == PRIMARY_CONDITIONS
    by_name = {item.name: item for item in specs}
    assert by_name["replay_10"].replay_per_previous_task == 10
    assert by_name["replay_100"].replay_per_previous_task == 100
    assert by_name["proposed_asymmetric"].gammas["vision"] > by_name[
        "proposed_asymmetric"
    ].gammas["action"]
    assert by_name["direction_inverted"].gammas["action"] > by_name[
        "direction_inverted"
    ].gammas["vision"]
    assert not by_name["reconstruction_only_latent"].aligned_latent_space
    assert by_name["independent_adapter_oracle"].independent_per_task


def test_test_suite_cannot_select_gammas():
    invalid = GammaSelection(
        held_out_suite="libero_10",
        validation_suites=("libero_spatial", "libero_goal", "libero_10"),
        proposed={"vision": 1.0, "language": 1.0, "action": 0.1},
        uniform=1.0,
        early_stopping_patience=10,
    )
    with pytest.raises(ValueError, match="held-out"):
        locked_condition_specs(invalid)


def test_replay_memory_freezes_exact_deterministic_per_task_samples():
    left = PerTaskReplayMemory(transitions_per_task=10, seed=17)
    right = PerTaskReplayMemory(transitions_per_task=10, seed=17)
    for memory in (left, right):
        memory.add_task("task_0", transitions("task_0"))
    assert left.provenance() == right.provenance()
    assert left.transition_count == 10
    assert len(left.prior_transitions(("task_0", "task_1"), current_task_id="task_1")) == 10


def test_replay_memory_rejects_future_tasks_and_insufficient_samples():
    memory = PerTaskReplayMemory(transitions_per_task=10, seed=17)
    with pytest.raises(ValueError, match="10 are required"):
        memory.add_task("task_0", transitions("task_0", 9))
    memory.add_task("task_0", transitions("task_0"))
    memory.add_task("task_1", transitions("task_1"))
    with pytest.raises(ValueError, match="future/unseen"):
        memory.prior_transitions(("task_0", "task_2"), current_task_id="task_2")
