import numpy as np

from wsl_vla.data.libero import LiberoEpisode
from wsl_vla.data.octo_batches import (
    ActionNormalization,
    collate_octo_examples,
    conform_batch_to_octo_example,
    iter_octo_examples,
)


def test_octo_windows_preserve_episode_boundaries_and_action_masks():
    observations = {
        "agentview_rgb": np.zeros((3, 4, 4, 3), dtype=np.uint8),
        "eye_in_hand_rgb": np.ones((3, 4, 4, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros((3, 3), dtype=np.float32),
        "robot0_eef_quat": np.zeros((3, 4), dtype=np.float32),
        "robot0_gripper_qpos": np.zeros((3, 2), dtype=np.float32),
    }
    first = LiberoEpisode("demo_0", np.array([[0.0], [1.0], [2.0]], dtype=np.float32), observations)
    second = LiberoEpisode("demo_1", np.array([[10.0], [11.0], [12.0]], dtype=np.float32), observations)
    normalization = ActionNormalization.fit((first, second))
    np.testing.assert_allclose(
        normalization.denormalize(normalization.normalize(first.actions)), first.actions
    )
    examples = list(
        iter_octo_examples(
            (first,),
            language_instruction="do the task",
            normalization=normalization,
            window_size=2,
            action_horizon=2,
            image_size=(4, 4),
        )
    )
    assert examples[0]["observation"]["timestep_pad_mask"].tolist() == [False, True]
    assert examples[-1]["action_pad_mask"][-1, :, 0].tolist() == [True, False]
    batch = collate_octo_examples(examples[:2])
    assert batch["action"].shape == (2, 2, 2, 1)


def test_conform_batch_drops_only_checkpoint_unsupported_observations():
    batch = {
        "observation": {
            "image_primary": np.zeros((2, 1, 4, 4, 3), dtype=np.uint8),
            "timestep_pad_mask": np.ones((2, 1), dtype=bool),
            "proprio": np.zeros((2, 1, 8), dtype=np.float32),
        },
        "task": {"language_instruction": np.zeros((2, 4), dtype=np.int32)},
        "action": np.zeros((2, 1, 2, 7), dtype=np.float32),
        "action_pad_mask": np.ones((2, 1, 2, 7), dtype=bool),
    }
    example = {
        "observation": {
            "image_primary": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8),
            "timestep_pad_mask": np.ones((1, 1), dtype=bool),
        },
        "task": {"language_instruction": np.zeros((1, 4), dtype=np.int32)},
        "action": np.zeros((1, 1, 2, 7), dtype=np.float32),
    }
    conformed = conform_batch_to_octo_example(batch, example)
    assert set(conformed["observation"]) == {"image_primary", "timestep_pad_mask"}
    assert "proprio" not in conformed["observation"]
