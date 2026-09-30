"""Memory-bounded real-LIBERO streams for continual Octo training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from ..experiments.conditions import ReplayTransition
from .libero import LiberoEpisode
from .octo_batches import (
    ActionNormalization,
    collate_octo_examples,
    conform_batch_to_octo_example,
    iter_octo_examples,
)


@dataclass(frozen=True)
class TaskTrainingStream:
    task_id: str
    instruction: str
    episodes: tuple[LiberoEpisode, ...]
    normalization: ActionNormalization
    window_size: int
    action_horizon: int
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        if not self.task_id or not self.instruction or not self.episodes:
            raise ValueError("task training stream identity/data is incomplete")
        if min(self.window_size, self.action_horizon, *self.image_size) <= 0:
            raise ValueError("task training stream dimensions must be positive")

    @property
    def transition_count(self) -> int:
        return sum(len(episode.actions) for episode in self.episodes)

    def iter_examples(self, episode_order: Sequence[int] | None = None) -> Iterator[dict]:
        order = tuple(episode_order) if episode_order is not None else tuple(range(len(self.episodes)))
        if set(order) != set(range(len(self.episodes))) or len(order) != len(self.episodes):
            raise ValueError("episode_order must be a permutation of complete episodes")
        for index in order:
            yield from iter_octo_examples(
                (self.episodes[index],),
                language_instruction=self.instruction,
                normalization=self.normalization,
                window_size=self.window_size,
                action_horizon=self.action_horizon,
                image_size=self.image_size,
                include_wrist=True,
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
    selected = set(
        int(value)
        for value in np.random.default_rng(seed).choice(
            stream.transition_count, size=count, replace=False
        )
    )
    result = []
    for global_index, example in enumerate(stream.iter_examples()):
        if global_index not in selected:
            continue
        result.append(
            ReplayTransition(
                task_id=stream.task_id,
                episode_id=str(example["episode_id"]),
                timestep=int(example["timestep"]),
                payload=example,
            )
        )
    if len(result) != count:
        raise AssertionError("replay sampling did not resolve every selected transition")
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
    example_batch: dict,
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
            batch = collate_octo_examples(pending, text_processor=text_processor)
            yield conform_batch_to_octo_example(batch, example_batch)
            pending = []
        if pending:
            while len(pending) < batch_size and replay:
                if replay_cursor >= len(replay_order):
                    replay_order = rng.permutation(len(replay))
                    replay_cursor = 0
                pending.append(replay[int(replay_order[replay_cursor])].payload)
                replay_cursor += 1
            batch = collate_octo_examples(pending, text_processor=text_processor)
            yield conform_batch_to_octo_example(batch, example_batch)

    return factory
