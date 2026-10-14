"""Checkpoint-free stand-ins for the shared preprocessing contract tests."""

from __future__ import annotations

import numpy as np

from wsl_vla.contracts import LIBERO_ACTION_CONVENTION
from wsl_vla.data.libero import LiberoEpisode
from wsl_vla.data.preprocessing import CameraSpec, PreprocessingSpec


SOURCE = (8, 8)


class FakeTextProcessor:
    """Deterministic tokenizer with the same output structure as Octo's HFTokenizer."""

    length = 4

    def encode(self, strings):
        input_ids = np.zeros((len(strings), self.length), dtype=np.int32)
        attention = np.zeros((len(strings), self.length), dtype=np.int32)
        for row, text in enumerate(strings):
            codes = [ord(char) % 97 + 1 for char in text][: self.length]
            input_ids[row, : len(codes)] = codes
            attention[row, : len(codes)] = 1
        return {"input_ids": input_ids, "attention_mask": attention}


def preprocessing_spec(
    *,
    image_transform: str = "flip_vertical",
    simulator_alignment: str = "identity",
    window_size: int = 2,
    action_horizon: int = 3,
    executed_actions_per_plan: int = 1,
) -> PreprocessingSpec:
    return PreprocessingSpec(
        window_size=window_size,
        action_horizon=action_horizon,
        action_dim=7,
        cameras=(
            CameraSpec("image_primary", "agentview_rgb", "agentview_image", 16, 16),
            CameraSpec("image_wrist", "eye_in_hand_rgb", "robot0_eye_in_hand_image", 8, 8),
        ),
        source_height=SOURCE[0],
        source_width=SOURCE[1],
        source_image_convention="opengl",
        image_transform=image_transform,
        simulator_alignment=simulator_alignment,
        executed_actions_per_plan=executed_actions_per_plan,
        consumed_observations=("image_primary", "image_wrist"),
        unconsumed_observations=(),
        proprio_consumed=False,
        text_processor="f" * 64,
    )


def example_batch(spec: PreprocessingSpec) -> dict:
    window = spec.window_size
    return {
        "observation": {
            "image_primary": np.zeros((1, window, 16, 16, 3), dtype=np.uint8),
            "image_wrist": np.zeros((1, window, 8, 8, 3), dtype=np.uint8),
            "timestep_pad_mask": np.ones((1, window), dtype=bool),
            "timestep": np.zeros((1, window), dtype=np.int32),
            "task_completed": np.zeros((1, window, spec.action_horizon), dtype=bool),
            "pad_mask_dict": {
                "image_primary": np.ones((1, window), dtype=bool),
                "image_wrist": np.ones((1, window), dtype=bool),
                "timestep": np.ones((1, window), dtype=bool),
            },
        },
        "task": {
            "image_primary": np.zeros((1, 16, 16, 3), dtype=np.uint8),
            "language_instruction": {
                "input_ids": np.zeros((1, 4), dtype=np.int32),
                "attention_mask": np.zeros((1, 4), dtype=np.int32),
            },
            "pad_mask_dict": {
                "image_primary": np.zeros((1,), dtype=bool),
                "language_instruction": np.ones((1,), dtype=bool),
            },
        },
        "action": np.zeros((1, window, spec.action_horizon, 7), dtype=np.float32),
    }


def controller_actions(rng: np.random.Generator, length: int) -> np.ndarray:
    """Valid LIBERO commands: deltas inside the bounds, gripper exactly -1 or +1."""

    actions = rng.uniform(-0.9, 0.9, size=(length, 7)).astype(np.float32)
    actions[:, LIBERO_ACTION_CONVENTION.gripper_dimension] = np.where(
        rng.random(length) < 0.5, -1.0, 1.0
    )
    return actions


def episode(index: int, *, length: int = 6, seed: int = 0) -> LiberoEpisode:
    rng = np.random.default_rng(seed * 1000 + index)
    frames = lambda: rng.integers(0, 255, size=(length, *SOURCE, 3), dtype=np.uint8)  # noqa: E731
    return LiberoEpisode(
        episode_id=f"demo_{index}",
        actions=controller_actions(rng, length),
        observations={"agentview_rgb": frames(), "eye_in_hand_rgb": frames()},
    )


def episodes(count: int = 3, *, length: int = 6, seed: int = 0) -> tuple[LiberoEpisode, ...]:
    return tuple(episode(index, length=length + index, seed=seed) for index in range(count))
