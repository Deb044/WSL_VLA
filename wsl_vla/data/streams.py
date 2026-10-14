"""Memory-bounded real-LIBERO streams for Octo training.

Two consumers share the same example builder:

* :class:`TransitionCursor` gives the model-zoo trainer a deterministic stream
  whose position is two integers, so an interrupted run resumes exactly;
* :func:`make_octo_batch_factory` gives the continual runners epoch factories
  with optional replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..contracts import LIBERO_ACTION_CONVENTION, ActionConvention
from ..experiments.conditions import ReplayTransition
from .actions import ActionNormalization, controller_action_normalization
from .libero import LiberoEpisode, StrictLiberoHDF5
from .octo_batches import (
    collate_octo_examples,
    conform_batch_to_octo_example,
    octo_example,
)
from .preprocessing import PreprocessingSpec
from .splits import EpisodeSplit


@dataclass(frozen=True)
class TaskTrainingStream:
    """One task's validated episodes plus the shared preprocessing identity."""

    task_id: str
    instruction: str
    episodes: tuple[LiberoEpisode, ...]
    preprocessing: PreprocessingSpec
    example_batch: Mapping[str, Any]
    convention: ActionConvention = LIBERO_ACTION_CONVENTION
    normalization: ActionNormalization = field(init=False)

    def __post_init__(self) -> None:
        if not self.task_id or not self.instruction.strip() or not self.episodes:
            raise ValueError("task training stream identity/data is incomplete")
        identifiers = [episode.episode_id for episode in self.episodes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("task training stream repeats an episode")
        # Validates every command against the controller bounds; never clips.
        object.__setattr__(
            self,
            "normalization",
            controller_action_normalization(self.episodes, self.convention),
        )

    @property
    def transition_count(self) -> int:
        return sum(len(episode.actions) for episode in self.episodes)

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(episode.episode_id for episode in self.episodes)

    def transition_index(self) -> np.ndarray:
        """``[transitions, 2]`` array of (episode index, timestep) in canonical order."""

        return np.asarray(
            [
                (episode_index, timestep)
                for episode_index, episode in enumerate(self.episodes)
                for timestep in range(len(episode.actions))
            ],
            dtype=np.int64,
        )

    def example(self, episode_index: int, timestep: int) -> dict:
        return octo_example(
            self.episodes[int(episode_index)],
            int(timestep),
            language_instruction=self.instruction,
            spec=self.preprocessing,
            example_observation=self.example_batch["observation"],
        )

    def iter_examples(self, episode_order: Sequence[int] | None = None) -> Iterator[dict]:
        order = tuple(episode_order) if episode_order is not None else tuple(range(len(self.episodes)))
        if set(order) != set(range(len(self.episodes))) or len(order) != len(self.episodes):
            raise ValueError("episode_order must be a permutation of complete episodes")
        for index in order:
            for timestep in range(len(self.episodes[index].actions)):
                yield self.example(index, timestep)

    def collate(self, examples: Sequence[Mapping], text_processor: Any) -> dict:
        batch = collate_octo_examples(
            examples, text_processor=text_processor, example_task=self.example_batch["task"]
        )
        return conform_batch_to_octo_example(batch, self.example_batch)


def load_split_episodes(
    data_file: str | Path,
    *,
    preprocessing: PreprocessingSpec,
    split_seed: int,
    train_fraction: float,
    validation_fraction: float,
    splits: Sequence[str] = ("train",),
) -> tuple[EpisodeSplit, dict[str, tuple[LiberoEpisode, ...]]]:
    """Load only the consumed cameras of the requested episode splits."""

    dataset = StrictLiberoHDF5(data_file, require_wrist_camera=False)
    if dataset.image_convention != preprocessing.source_image_convention:
        raise ValueError(
            f"{data_file} was generated with the {dataset.image_convention!r} image "
            f"convention, but the shared preprocessing declares "
            f"{preprocessing.source_image_convention!r}"
        )
    split = dataset.split(
        seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    unknown = set(splits) - {"train", "validation", "test"}
    if unknown:
        raise ValueError(f"unknown episode splits: {sorted(unknown)}")
    loaded = {
        name: tuple(
            dataset.iter_episodes(
                getattr(split, name), observation_keys=preprocessing.dataset_camera_keys
            )
        )
        for name in splits
    }
    return split, loaded


def task_stream_from_bundle(
    bundle: Any,
    *,
    task_id: str,
    instruction: str,
    episodes: Sequence[LiberoEpisode],
) -> TaskTrainingStream:
    return TaskTrainingStream(
        task_id=task_id,
        instruction=instruction,
        episodes=tuple(episodes),
        preprocessing=bundle.preprocessing,
        example_batch=bundle.pretrained_model.example_batch,
        convention=bundle.action_convention,
    )


@dataclass
class TransitionCursor:
    """Deterministic shuffled stream whose whole state is ``(epoch, position)``.

    The permutation of epoch ``e`` depends only on ``(seed, e)``, so the stream
    needs no saved generator state: restoring the two counters reproduces every
    later batch exactly. Batches are always full (an epoch tail continues into
    the next epoch), which keeps one compiled train step.
    """

    transition_count: int
    batch_size: int
    seed: int
    epoch: int = 0
    position: int = 0

    def __post_init__(self) -> None:
        if self.transition_count <= 0 or self.batch_size <= 0 or self.seed < 0:
            raise ValueError("cursor needs positive transition and batch counts")
        if self.epoch < 0 or not 0 <= self.position < self.transition_count:
            raise ValueError("cursor position lies outside its epoch")

    def _order(self, epoch: int) -> np.ndarray:
        return np.random.default_rng([self.seed, epoch]).permutation(self.transition_count)

    @property
    def samples_consumed(self) -> int:
        return self.epoch * self.transition_count + self.position

    def next_indices(self) -> np.ndarray:
        chosen: list[int] = []
        while len(chosen) < self.batch_size:
            order = self._order(self.epoch)
            take = min(self.batch_size - len(chosen), self.transition_count - self.position)
            chosen.extend(int(value) for value in order[self.position : self.position + take])
            self.position += take
            if self.position == self.transition_count:
                self.epoch += 1
                self.position = 0
        return np.asarray(chosen, dtype=np.int64)

    def state(self) -> dict[str, int]:
        return {
            "transition_count": int(self.transition_count),
            "batch_size": int(self.batch_size),
            "seed": int(self.seed),
            "epoch": int(self.epoch),
            "position": int(self.position),
        }

    @classmethod
    def restore(
        cls, state: Mapping[str, int], *, transition_count: int, batch_size: int, seed: int
    ) -> "TransitionCursor":
        expected = {"transition_count": transition_count, "batch_size": batch_size, "seed": seed}
        for name, value in expected.items():
            if int(state[name]) != int(value):
                raise ValueError(f"dataset cursor was saved for a different {name}")
        return cls(
            transition_count, batch_size, seed, int(state["epoch"]), int(state["position"])
        )


def sample_replay_transitions(
    stream: TaskTrainingStream,
    *,
    count: int,
    seed: int,
) -> tuple[ReplayTransition, ...]:
    """Uniformly select identifiable transitions without retaining all windows."""

    if count <= 0 or count > stream.transition_count:
        raise ValueError("replay count must fit the task's real transition population")
    index = stream.transition_index()
    selected = np.sort(
        np.random.default_rng(seed).choice(stream.transition_count, size=count, replace=False)
    )
    result = []
    for position in selected:
        episode_index, timestep = (int(value) for value in index[int(position)])
        example = stream.example(episode_index, timestep)
        result.append(
            ReplayTransition(
                task_id=stream.task_id,
                episode_id=str(example["episode_id"]),
                timestep=int(example["timestep"]),
                payload=example,
            )
        )
    result.sort(key=lambda item: (item.episode_id, item.timestep))
    return tuple(result)


def _buffered_shuffle(
    examples: Iterable[dict], *, rng: np.random.Generator, buffer_size: int
) -> Iterator[dict]:
    if buffer_size <= 0:
        raise ValueError("shuffle buffer must be positive")
    buffer = []
    for item in examples:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            index = int(rng.integers(len(buffer)))
            yield buffer.pop(index)
    rng.shuffle(buffer)
    yield from buffer


def make_octo_batch_factory(
    stream: TaskTrainingStream,
    *,
    batch_size: int,
    seed: int,
    text_processor: Any,
    replay: Sequence[ReplayTransition] = (),
    shuffle_buffer: int = 256,
):
    """Return repeatable epoch factories with bounded image memory.

    When replay is non-empty, half of each micro-batch (rounded down, at least
    one current sample) is drawn from the current task and the rest from the
    frozen prior-task memory.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if replay and batch_size < 2:
        raise ValueError("replay requires micro-batch size at least two")
    if any(item.task_id == stream.task_id for item in replay):
        raise ValueError("replay input must contain only prior tasks")
    epoch = 0

    def factory():
        nonlocal epoch
        rng = np.random.default_rng(seed + epoch)
        epoch += 1
        episode_order = rng.permutation(len(stream.episodes))
        current = _buffered_shuffle(
            stream.iter_examples(episode_order), rng=rng, buffer_size=shuffle_buffer
        )
        current_per_batch = batch_size if not replay else max(1, batch_size // 2)
        replay_per_batch = batch_size - current_per_batch
        replay_order = rng.permutation(len(replay)) if replay else np.empty(0, dtype=int)
        replay_cursor = 0
        pending = []
        for example in current:
            pending.append(example)
            if len(pending) < current_per_batch:
                continue
            if replay_per_batch:
                for _ in range(replay_per_batch):
                    if replay_cursor >= len(replay_order):
                        replay_order = rng.permutation(len(replay))
                        replay_cursor = 0
                    pending.append(replay[int(replay_order[replay_cursor])].payload)
                    replay_cursor += 1
            yield stream.collate(pending, text_processor)
            pending = []
        if pending:
            while len(pending) < batch_size and replay:
                if replay_cursor >= len(replay_order):
                    replay_order = rng.permutation(len(replay))
                    replay_cursor = 0
                pending.append(replay[int(replay_order[replay_cursor])].payload)
                replay_cursor += 1
            yield stream.collate(pending, text_processor)

    return factory
