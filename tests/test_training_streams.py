from __future__ import annotations

import numpy as np
import pytest

from wsl_vla.data.libero import LiberoEpisode
from wsl_vla.data.octo_batches import ActionNormalization
from wsl_vla.data.streams import (
    TaskTrainingStream,
    make_octo_batch_factory,
    sample_replay_transitions,
)


def episodes(offset=0):
    result = []
    for episode_index in range(3):
        length = 5
        actions = np.arange(length * 7, dtype=np.float32).reshape(length, 7) + offset
        observations = {
            "agentview_rgb": np.full((length, 4, 4, 3), episode_index, dtype=np.uint8),
            "eye_in_hand_rgb": np.full((length, 4, 4, 3), episode_index + 1, dtype=np.uint8),
            "robot0_eef_pos": np.ones((length, 3), dtype=np.float32),
            "robot0_eef_quat": np.ones((length, 4), dtype=np.float32),
            "robot0_gripper_qpos": np.ones((length, 2), dtype=np.float32),
        }
        result.append(LiberoEpisode(f"episode_{episode_index}", actions, observations))
    return tuple(result)


def stream(task_id="task_0", offset=0):
    values = episodes(offset)
    return TaskTrainingStream(
        task_id=task_id,
        instruction="do the task",
        episodes=values,
        normalization=ActionNormalization.fit(values),
        window_size=2,
        action_horizon=2,
        image_size=(4, 4),
    )


def example_batch():
    return {
        "observation": {
            "image_primary": np.zeros((1, 2, 4, 4, 3), dtype=np.uint8),
            "image_wrist": np.zeros((1, 2, 4, 4, 3), dtype=np.uint8),
            "proprio": np.zeros((1, 2, 9), dtype=np.float32),
            "timestep_pad_mask": np.ones((1, 2), dtype=bool),
        },
        "task": {"language_instruction": np.asarray([b"task"])},
        "action": np.zeros((1, 2, 2, 7), dtype=np.float32),
    }


def test_replay_sampling_is_deterministic_identifiable_and_exact():
    left = sample_replay_transitions(stream(), count=10, seed=7)
    right = sample_replay_transitions(stream(), count=10, seed=7)
    assert [(x.episode_id, x.timestep) for x in left] == [
        (x.episode_id, x.timestep) for x in right
    ]
    assert len({(x.episode_id, x.timestep) for x in left}) == 10


def test_batch_factory_mixes_only_prior_replay_and_is_memory_bounded():
    current = stream("task_1", offset=100)
    replay = sample_replay_transitions(stream("task_0"), count=10, seed=3)
    factory = make_octo_batch_factory(
        current,
        batch_size=2,
        seed=5,
        text_processor=None,
        example_batch=example_batch(),
        replay=replay,
        shuffle_buffer=3,
    )
    batch = next(iter(factory()))
    assert batch["action"].shape == (2, 2, 2, 7)
    assert batch["observation"]["image_primary"].shape == (2, 2, 4, 4, 3)
    with pytest.raises(ValueError, match="only prior"):
        make_octo_batch_factory(
            current,
            batch_size=2,
            seed=5,
            text_processor=None,
            example_batch=example_batch(),
            replay=sample_replay_transitions(current, count=10, seed=3),
        )
