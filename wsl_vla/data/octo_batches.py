"""Convert complete LIBERO episodes into official Octo training batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from .libero import LiberoEpisode


@dataclass(frozen=True)
class ActionNormalization:
    mean: np.ndarray
    std: np.ndarray
    count: int

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "count": int(self.count),
        }

    @classmethod
    def fit(cls, episodes: Iterable[LiberoEpisode], epsilon: float = 1e-6):
        arrays = [np.asarray(episode.actions, dtype=np.float64) for episode in episodes]
        if not arrays:
            raise ValueError("cannot fit action normalization without training episodes")
        values = np.concatenate(arrays, axis=0)
        std = values.std(axis=0)
        if np.any(std <= epsilon):
            raise ValueError("training action statistics contain a constant dimension")
        return cls(values.mean(axis=0).astype(np.float32), std.astype(np.float32), len(values))

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        values = (np.asarray(actions, dtype=np.float32) - self.mean) / self.std
        if not np.isfinite(values).all():
            raise ValueError("normalized actions contain non-finite values")
        return values

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        return np.asarray(actions, dtype=np.float32) * self.std + self.mean


def _resize_rgb(images: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    values = np.asarray(images)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("camera observations must have shape [time,height,width,3]")
    if values.shape[1:3] == size:
        return values.astype(np.uint8, copy=False)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - installed by official Octo
        raise RuntimeError("Pillow is required to resize LIBERO observations") from exc
    resized = [
        np.asarray(Image.fromarray(frame.astype(np.uint8)).resize(size[::-1], Image.Resampling.BILINEAR))
        for frame in values
    ]
    return np.stack(resized).astype(np.uint8)


def iter_octo_examples(
    episodes: Iterable[LiberoEpisode],
    *,
    language_instruction: str,
    normalization: ActionNormalization,
    window_size: int,
    action_horizon: int,
    image_size: tuple[int, int] = (256, 256),
    include_wrist: bool = True,
    proprio_keys: tuple[str, ...] = (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ),
) -> Iterator[dict]:
    """Yield left-padded observation windows and right-padded action chunks."""

    if window_size <= 0 or action_horizon <= 0 or not language_instruction.strip():
        raise ValueError("window/action horizons must be positive and language non-empty")
    for episode in episodes:
        primary = _resize_rgb(episode.observations["agentview_rgb"], image_size)
        wrist = None
        if include_wrist:
            if "eye_in_hand_rgb" not in episode.observations:
                raise ValueError(f"{episode.episode_id} has no wrist camera")
            wrist = _resize_rgb(episode.observations["eye_in_hand_rgb"], image_size)
        missing_proprio = set(proprio_keys) - set(episode.observations)
        if missing_proprio:
            raise ValueError(
                f"{episode.episode_id} lacks proprioception keys {sorted(missing_proprio)}"
            )
        proprio = np.concatenate(
            [np.asarray(episode.observations[key], dtype=np.float32) for key in proprio_keys],
            axis=-1,
        )
        actions = normalization.normalize(episode.actions)
        if len(primary) != len(actions) or (wrist is not None and len(wrist) != len(actions)):
            raise ValueError(f"camera/action length mismatch in {episode.episode_id}")
        for timestep in range(len(actions)):
            observation = {
                "image_primary": np.zeros((window_size, *image_size, 3), dtype=np.uint8),
                "timestep_pad_mask": np.zeros(window_size, dtype=bool),
                "proprio": np.zeros((window_size, proprio.shape[-1]), dtype=np.float32),
            }
            if wrist is not None:
                observation["image_wrist"] = np.zeros(
                    (window_size, *image_size, 3), dtype=np.uint8
                )
            start = max(0, timestep - window_size + 1)
            observed_count = timestep - start + 1
            destination = slice(window_size - observed_count, window_size)
            observation["image_primary"][destination] = primary[start : timestep + 1]
            observation["timestep_pad_mask"][destination] = True
            observation["proprio"][destination] = proprio[start : timestep + 1]
            if wrist is not None:
                observation["image_wrist"][destination] = wrist[start : timestep + 1]

            action = np.zeros(
                (window_size, action_horizon, actions.shape[-1]), dtype=np.float32
            )
            action_mask = np.zeros_like(action, dtype=bool)
            for window_offset, source_timestep in enumerate(range(start, timestep + 1), window_size - observed_count):
                end = min(len(actions), source_timestep + action_horizon)
                count = end - source_timestep
                action[window_offset, :count] = actions[source_timestep:end]
                action_mask[window_offset, :count] = True
            yield {
                "observation": observation,
                "task": {"language_instruction": language_instruction.encode("utf-8")},
                "action": action,
                "action_pad_mask": action_mask,
                "episode_id": episode.episode_id,
                "timestep": timestep,
            }


def collate_octo_examples(examples: Sequence[Mapping], text_processor=None) -> dict:
    """Stack examples and apply Octo's official language tokenizer."""

    if not examples:
        raise ValueError("cannot collate an empty batch")
    batch = {
        "observation": {
            key: np.stack([item["observation"][key] for item in examples])
            for key in examples[0]["observation"]
        },
        "task": {
            "language_instruction": np.asarray(
                [item["task"]["language_instruction"] for item in examples]
            )
        },
        "action": np.stack([item["action"] for item in examples]),
        "action_pad_mask": np.stack([item["action_pad_mask"] for item in examples]),
    }
    if text_processor is not None:
        try:
            from octo.utils.train_utils import process_text
        except ImportError as exc:  # pragma: no cover - research environment only
            raise RuntimeError("official Octo is required for language tokenization") from exc
        batch = process_text(batch, text_processor)
    return batch


def conform_batch_to_octo_example(batch: Mapping, example_batch: Mapping) -> dict:
    """Select and shape-check exactly the inputs supported by a checkpoint.

    LIBERO loaders may expose useful extra observations such as proprioception.
    Official Octo checkpoints differ in which tokenizers they enable, so extras
    are dropped only after they have been loaded and validated from the dataset.
    Missing checkpoint-required inputs fail closed.
    """

    required_top = {"observation", "task", "action", "action_pad_mask"}
    if required_top - set(batch):
        raise ValueError(f"batch lacks Octo fields: {sorted(required_top-set(batch))}")
    if not {"observation", "task", "action"} <= set(example_batch):
        raise ValueError("official Octo example batch is incomplete")

    result = {}
    for family in ("observation", "task"):
        expected = example_batch[family]
        missing = set(expected) - set(batch[family])
        if missing:
            raise ValueError(f"batch lacks checkpoint-required {family} keys: {sorted(missing)}")
        result[family] = {key: batch[family][key] for key in expected}
        for key, value in result[family].items():
            actual_shape = np.asarray(value).shape
            expected_shape = np.asarray(expected[key]).shape
            if actual_shape[1:] != expected_shape[1:]:
                raise ValueError(
                    f"{family}/{key} shape {actual_shape[1:]} differs from checkpoint {expected_shape[1:]}"
                )
    for name in ("action", "action_pad_mask"):
        value = np.asarray(batch[name])
        if name == "action" and value.shape[1:] != np.asarray(example_batch["action"]).shape[1:]:
            raise ValueError("action shape differs from official Octo checkpoint")
        result[name] = value
    return result
