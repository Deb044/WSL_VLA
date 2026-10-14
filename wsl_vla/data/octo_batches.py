"""Convert complete LIBERO episodes into official Octo training batches.

All model inputs are built by :mod:`wsl_vla.data.preprocessing`, the same code
the rollout policy uses. This module only iterates episodes, validates actions
against the fixed controller convention, and stacks examples.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..contracts import LIBERO_ACTION_CONVENTION, ActionConvention
from .actions import (  # noqa: F401  (re-exported for existing callers)
    ActionNormalization,
    controller_action_normalization,
    validate_controller_actions,
)
from .libero import LiberoEpisode
from .preprocessing import (
    PreprocessingSpec,
    action_window,
    camera_frames,
    observation_window,
    resize_rgb,
    task_batch,
)


# Kept for callers that only need the resize primitive.
_resize_rgb = resize_rgb


def octo_example(
    episode: LiberoEpisode,
    timestep: int,
    *,
    language_instruction: str,
    spec: PreprocessingSpec,
    example_observation: Mapping[str, Any],
) -> dict:
    """One training example: history window ending at ``timestep`` plus targets.

    Only the frames inside the window are preprocessed, so random access across
    episodes costs no more than sequential iteration. The caller is responsible
    for having validated ``episode.actions`` against the controller convention.
    """

    length = len(episode.actions)
    if not 0 <= timestep < length:
        raise IndexError(f"timestep {timestep} lies outside {episode.episode_id}")
    start = max(0, timestep - spec.window_size + 1)
    frames = {}
    for camera in spec.cameras:
        if camera.dataset_key not in episode.observations:
            raise ValueError(f"{episode.episode_id} lacks camera {camera.dataset_key}")
        raw = episode.observations[camera.dataset_key]
        if len(raw) != length:
            raise ValueError(f"camera/action length mismatch in {episode.episode_id}")
        frames[camera.octo_key] = camera_frames(
            raw[start : timestep + 1], camera, spec, source="dataset"
        )
    action, action_mask = action_window(episode.actions, timestep, spec)
    return {
        "observation": observation_window(frames, spec, example_observation),
        "task": {"language_instruction": language_instruction},
        "action": action,
        "action_pad_mask": action_mask,
        "episode_id": episode.episode_id,
        "timestep": int(timestep),
    }


def iter_octo_examples(
    episodes: Iterable[LiberoEpisode],
    *,
    language_instruction: str,
    spec: PreprocessingSpec,
    example_observation: Mapping[str, Any],
    convention: ActionConvention = LIBERO_ACTION_CONVENTION,
    timesteps: Sequence[int] | None = None,
) -> Iterator[dict]:
    """Yield left-padded observation windows and right-padded action chunks.

    Targets stay in controller coordinates. Each episode's commands are checked
    against the controller bounds first; nothing is clipped or rescaled.
    ``timesteps`` restricts the yielded steps (every episode must contain them).
    """

    if not language_instruction.strip():
        raise ValueError("language instruction cannot be empty")
    if spec.action_dim != convention.action_dim:
        raise ValueError("checkpoint action dimension differs from the controller convention")
    for episode in episodes:
        validate_controller_actions(
            episode.actions, convention, source=f"episode {episode.episode_id}"
        )
        selected = range(len(episode.actions)) if timesteps is None else timesteps
        for timestep in selected:
            yield octo_example(
                episode,
                int(timestep),
                language_instruction=language_instruction,
                spec=spec,
                example_observation=example_observation,
            )


def _stack(values: Sequence[Any]) -> Any:
    if isinstance(values[0], Mapping):
        return {key: _stack([value[key] for value in values]) for key in values[0]}
    return np.stack([np.asarray(value) for value in values])


def collate_octo_examples(
    examples: Sequence[Mapping], *, text_processor: Any, example_task: Mapping[str, Any]
) -> dict:
    """Stack examples and build the language task with Octo's own tokenizer."""

    if not examples:
        raise ValueError("cannot collate an empty batch")
    return {
        "observation": _stack([item["observation"] for item in examples]),
        "task": task_batch(
            [item["task"]["language_instruction"] for item in examples],
            text_processor,
            example_task,
        ),
        "action": np.stack([item["action"] for item in examples]),
        "action_pad_mask": np.stack([item["action_pad_mask"] for item in examples]),
    }


def _assert_tree_matches(value: Any, expected: Any, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be a mapping like the checkpoint example")
        if set(value) != set(expected):
            raise ValueError(
                f"{path} keys differ from the checkpoint: "
                f"missing={sorted(set(expected)-set(value))}, extra={sorted(set(value)-set(expected))}"
            )
        for key in expected:
            _assert_tree_matches(value[key], expected[key], f"{path}/{key}")
        return
    actual_shape = np.asarray(value).shape
    expected_shape = np.asarray(expected).shape
    if actual_shape[1:] != expected_shape[1:]:
        raise ValueError(
            f"{path} shape {actual_shape[1:]} differs from checkpoint {expected_shape[1:]}"
        )


def conform_batch_to_octo_example(batch: Mapping, example_batch: Mapping) -> dict:
    """Fail closed unless a batch has exactly the checkpoint's inputs.

    The shared preprocessing already builds checkpoint-exact inputs, so this is
    a structural check rather than a repair step: no key is added, dropped, or
    resized here.
    """

    required_top = {"observation", "task", "action", "action_pad_mask"}
    if required_top - set(batch):
        raise ValueError(f"batch lacks Octo fields: {sorted(required_top-set(batch))}")
    if not {"observation", "task", "action"} <= set(example_batch):
        raise ValueError("official Octo example batch is incomplete")
    _assert_tree_matches(batch["observation"], example_batch["observation"], "observation")
    _assert_tree_matches(batch["task"], example_batch["task"], "task")
    action = np.asarray(batch["action"])
    if action.shape[1:] != np.asarray(example_batch["action"]).shape[1:]:
        raise ValueError("action shape differs from official Octo checkpoint")
    mask = np.asarray(batch["action_pad_mask"])
    if mask.shape != action.shape or mask.dtype != bool:
        raise ValueError("action_pad_mask must be a boolean array shaped like action")
    sizes = {
        action.shape[0],
        np.asarray(batch["observation"]["timestep_pad_mask"]).shape[0],
    }
    if len(sizes) != 1:
        raise ValueError("observation and action batch sizes differ")
    return {
        "observation": batch["observation"],
        "task": batch["task"],
        "action": action,
        "action_pad_mask": mask,
    }
