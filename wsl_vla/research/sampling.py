"""Deterministic task-balanced batches for multi-positive alignment."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def task_balanced_index_batches(
    indices: Iterable[int],
    task_labels: np.ndarray,
    *,
    tasks_per_batch: int,
    samples_per_task: int,
    seed: int,
    shuffle: bool,
) -> tuple[np.ndarray, ...]:
    """Build batches with repeated positives and at least two task identities.

    Every source index appears at least once. A task's final incomplete positive
    group wraps deterministically within that task rather than borrowing a false
    positive from another task.
    """

    if tasks_per_batch < 2:
        raise ValueError("contrastive batches require at least two task identities")
    if samples_per_task < 2:
        raise ValueError("multi-positive batches require at least two samples per task")
    selected = np.asarray(tuple(indices), dtype=np.int64)
    labels = np.asarray(task_labels)
    if selected.ndim != 1 or not len(selected):
        raise ValueError("indices must be a non-empty one-dimensional collection")
    if np.any(selected < 0) or np.any(selected >= len(labels)):
        raise ValueError("batch index lies outside task_labels")

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in selected:
        grouped[int(labels[index])].append(int(index))
    if len(grouped) < tasks_per_batch:
        raise ValueError("split has too few task identities for a contrastive batch")

    rng = np.random.default_rng(seed)
    chunks: dict[int, list[np.ndarray]] = {}
    for label, members in grouped.items():
        values = np.asarray(members, dtype=np.int64)
        if shuffle:
            rng.shuffle(values)
        task_chunks = []
        for start in range(0, len(values), samples_per_task):
            chunk = values[start : start + samples_per_task]
            if len(chunk) < samples_per_task:
                needed = samples_per_task - len(chunk)
                chunk = np.concatenate([chunk, np.resize(values, needed)])
            task_chunks.append(chunk)
        chunks[label] = task_chunks

    schedule = [
        label
        for label in sorted(chunks)
        for _ in range(len(chunks[label]))
    ]
    if shuffle:
        rng.shuffle(schedule)
    batches = []
    cursors = {label: 0 for label in chunks}
    while schedule:
        chosen = []
        for label in list(schedule):
            if label not in chosen:
                chosen.append(label)
            if len(chosen) == tasks_per_batch:
                break
        if len(chosen) < tasks_per_batch:
            # Reuse task chunks only to complete the final mixed-task batch.
            for label in sorted(chunks):
                if label not in chosen:
                    chosen.append(label)
                if len(chosen) == tasks_per_batch:
                    break
        parts = []
        for label in chosen:
            cursor = cursors[label] % len(chunks[label])
            parts.append(chunks[label][cursor])
            cursors[label] += 1
            if label in schedule:
                schedule.remove(label)
        batch = np.concatenate(parts)
        if shuffle:
            rng.shuffle(batch)
        batches.append(batch)
    return tuple(batches)
