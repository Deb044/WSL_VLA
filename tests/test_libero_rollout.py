from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from wsl_vla.evaluation.libero_rollout import (
    LiberoRolloutEvaluator,
    build_octo_observation,
    fixed_initialization_indices,
    validate_benchmark_task_order,
)


@dataclass
class FakeTask:
    language: str
    problem_folder: str = "problem"
    bddl_file: str = "task.bddl"


class FakeSuite:
    def __init__(self, instructions, init_count=60):
        self.tasks = [FakeTask(value) for value in instructions]
        self.init_states = np.arange(init_count * 2).reshape(init_count, 2)

    def get_task(self, index):
        return self.tasks[index]

    def get_task_init_states(self, index):
        return self.init_states + index


class FakeEnvironment:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.step_count = 0
        self.closed = False
        self.seeds = []
        self.__class__.instances.append(self)

    def seed(self, seed):
        self.seeds.append(seed)

    def reset(self):
        self.step_count = 0

    def set_init_state(self, state):
        self.state = np.array(state)
        return {"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8)}

    def step(self, action):
        self.step_count += 1
        success = self.step_count == 7
        return (
            {"agentview_image": np.full((4, 4, 3), self.step_count, dtype=np.uint8)},
            float(success),
            success,
            {},
        )

    def close(self):
        self.closed = True


class FakePolicy:
    checkpoint_sha256 = "a" * 64

    def __init__(self):
        self.resets = []

    def reset(self, *, instruction, seed):
        self.resets.append((instruction, seed))

    def action(self, observation):
        return np.zeros(7, dtype=np.float32)


def instructions():
    return tuple(f"Do task {index}." for index in range(10))


def test_fixed_initializations_are_unique_deterministic_and_fail_closed():
    first = fixed_initialization_indices(50, 20, seed=17)
    assert first == fixed_initialization_indices(50, 20, seed=17)
    assert len(first) == len(set(first)) == 20
    assert first != fixed_initialization_indices(50, 20, seed=18)
    with pytest.raises(ValueError, match="only 10"):
        fixed_initialization_indices(10, 20, seed=17)


def test_task_order_validation_allows_only_presentation_differences():
    expected = instructions()
    validate_benchmark_task_order(
        FakeSuite([value.lower().rstrip(".") for value in expected]), expected
    )
    changed = list(expected)
    changed[3] = "Do a different task."
    with pytest.raises(ValueError, match="index 3"):
        validate_benchmark_task_order(FakeSuite(changed), expected)


def test_evaluator_uses_fixed_states_warmup_and_closes_environment(tmp_path):
    FakeEnvironment.instances.clear()
    folder = tmp_path / "problem"
    folder.mkdir()
    (folder / "task.bddl").write_text("task", encoding="utf-8")
    evaluator = LiberoRolloutEvaluator(
        suite="libero_spatial",
        expected_instructions=instructions(),
        benchmark_suite=FakeSuite(instructions()),
        env_factory=FakeEnvironment,
        bddl_root=tmp_path,
        warmup_steps=5,
        max_steps=5,
        camera_size=(4, 4),
    )
    policy = FakePolicy()
    result = evaluator.evaluate_task(policy, task_index=2, rollout_count=3, seed=17)
    assert result.successes == 3
    assert result.rollout_count == 3
    assert result.initialization_indices == fixed_initialization_indices(60, 3, seed=17)
    outcome = result.as_outcome()
    assert outcome.successes == 3
    assert outcome.initialization_indices == result.initialization_indices
    assert len(policy.resets) == 3
    assert all(episode.steps == 2 for episode in result.episodes)
    assert FakeEnvironment.instances[0].closed


def test_octo_observation_left_pads_and_resizes_all_modalities():
    example = {
        "image_primary": np.zeros((2, 3, 8, 8, 3), dtype=np.uint8),
        "image_wrist": np.zeros((2, 3, 8, 8, 3), dtype=np.uint8),
        "proprio": np.zeros((2, 3, 9), dtype=np.float32),
        "timestep_pad_mask": np.ones((2, 3), dtype=bool),
    }
    history = [
        {
            "agentview_image": np.full((4, 4, 3), 7, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.full((4, 4, 3), 8, dtype=np.uint8),
            "robot0_eef_pos": np.arange(3),
            "robot0_eef_quat": np.arange(4),
            "robot0_gripper_qpos": np.arange(2),
        }
    ]
    result = build_octo_observation(history, example)
    assert result["image_primary"].shape == (1, 3, 8, 8, 3)
    assert result["image_primary"][0, -1].mean() == 7
    assert result["image_wrist"][0, -1].mean() == 8
    np.testing.assert_array_equal(result["timestep_pad_mask"], [[False, False, True]])
    np.testing.assert_array_equal(result["proprio"][0, -1], np.r_[np.arange(3), np.arange(4), np.arange(2)])


def test_octo_observation_rejects_checkpoint_inputs_it_cannot_supply():
    with pytest.raises(ValueError, match="unsupported checkpoint observation key"):
        build_octo_observation(
            [{"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8)}],
            {
                "timestep_pad_mask": np.ones((1, 1), dtype=bool),
                "depth": np.zeros((1, 1, 4, 4, 1)),
            },
        )
