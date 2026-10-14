"""Evidence construction: command-chunk summaries and fold standardization.

Action evidence is a set of per-episode, non-overlapping chunk summaries rather
than one task-level vector. Each chunk is summarized by the mean and standard
deviation of its commands and of their first, second, and third differences.
These are **command differences**: LIBERO actions are controller commands (delta
poses and a gripper command), so their differences are not physical velocity,
acceleration, or jerk and are never labelled as such.

Standardization is fitted on the source fold's training tasks only and applied
unchanged to validation and held-out tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import COMPONENT_NAMES, TaskEvidence


MAX_DIFFERENCE_ORDER = 3
MIN_CHUNK_LENGTH = MAX_DIFFERENCE_ORDER + 1
ACTION_SUMMARY_FIELDS = (
    "command_mean",
    "command_std",
    "difference1_mean",
    "difference1_std",
    "difference2_mean",
    "difference2_std",
    "difference3_mean",
    "difference3_std",
)


def chunk_command_summary(chunk: np.ndarray) -> np.ndarray:
    """Summarize one chunk of real commands; differences stay inside the chunk."""

    values = np.asarray(chunk, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < MIN_CHUNK_LENGTH:
        raise ValueError(
            f"an action chunk needs at least {MIN_CHUNK_LENGTH} commands of shape [time, dim]"
        )
    if not np.isfinite(values).all():
        raise ValueError("action chunk contains non-finite commands")
    parts = []
    current = values
    for order in range(MAX_DIFFERENCE_ORDER + 1):
        if order:
            current = np.diff(current, axis=0)
        parts.extend([current.mean(axis=0), current.std(axis=0)])
    return np.concatenate(parts).astype(np.float32)


def episode_chunk_summaries(actions: np.ndarray, *, chunk_length: int) -> np.ndarray:
    """Non-overlapping chunk summaries ``[chunks, 8 * action_dim]`` of one episode.

    A trailing remainder is kept as its own shorter chunk when it still holds
    ``MIN_CHUNK_LENGTH`` commands and is dropped otherwise; no command is ever
    padded, repeated, or borrowed from another episode.
    """

    values = np.asarray(actions)
    if chunk_length < MIN_CHUNK_LENGTH:
        raise ValueError(f"chunk_length must be at least {MIN_CHUNK_LENGTH}")
    if values.ndim != 2 or values.shape[0] < MIN_CHUNK_LENGTH:
        raise ValueError(
            f"an episode needs at least {MIN_CHUNK_LENGTH} commands of shape [time, dim]"
        )
    summaries = [
        chunk_command_summary(values[start : start + chunk_length])
        for start in range(0, values.shape[0], chunk_length)
        if values.shape[0] - start >= MIN_CHUNK_LENGTH
    ]
    return np.stack(summaries)


def action_chunk_evidence(
    episode_actions: Sequence[np.ndarray], *, chunk_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Masked ``[episodes, chunks, features]`` summaries and their validity mask."""

    if not episode_actions:
        raise ValueError("action evidence requires at least one training episode")
    per_episode = [
        episode_chunk_summaries(actions, chunk_length=chunk_length) for actions in episode_actions
    ]
    widths = {item.shape[1] for item in per_episode}
    if len(widths) != 1:
        raise ValueError("episodes disagree on the action dimension")
    maximum = max(item.shape[0] for item in per_episode)
    features = np.zeros((len(per_episode), maximum, widths.pop()), dtype=np.float32)
    mask = np.zeros((len(per_episode), maximum), dtype=bool)
    for index, item in enumerate(per_episode):
        features[index, : item.shape[0]] = item
        mask[index, : item.shape[0]] = True
    return features, mask


@dataclass(frozen=True)
class FeatureStandardization:
    """Per-feature affine standardization with unit scale for constant features."""

    mean: np.ndarray
    scale: np.ndarray
    constant: np.ndarray

    def apply(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.shape[-1] != self.mean.shape[0]:
            raise ValueError("feature width differs from the fitted standardization")
        return (values - self.mean) / self.scale


def fit_feature_standardization(
    task_feature_sets: Sequence[np.ndarray], *, epsilon: float = 1e-6
) -> FeatureStandardization:
    """Task-balanced mean/scale from the valid rows of each *training* task.

    Every task contributes equally regardless of how many frames, tokens, or
    chunks it has. A feature whose balanced standard deviation is at most
    ``epsilon`` is constant across the training fold; it keeps scale 1 so that
    an unseen task with a different value is shifted, not divided by ~0.
    """

    if not task_feature_sets:
        raise ValueError("standardization needs at least one training task")
    first = np.zeros(0)
    second = np.zeros(0)
    for index, rows in enumerate(task_feature_sets):
        values = np.asarray(rows, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("each task must supply a non-empty [rows, features] array")
        if not np.isfinite(values).all():
            raise ValueError("standardization inputs must be finite")
        if index == 0:
            first = np.zeros(values.shape[1])
            second = np.zeros(values.shape[1])
        elif values.shape[1] != first.shape[0]:
            raise ValueError("tasks disagree on the feature width")
        first += values.mean(axis=0)
        second += np.square(values).mean(axis=0)
    count = len(task_feature_sets)
    mean = first / count
    deviation = np.sqrt(np.maximum(second / count - np.square(mean), 0.0))
    constant = deviation <= epsilon
    return FeatureStandardization(
        mean=mean.astype(np.float32),
        scale=np.where(constant, 1.0, deviation).astype(np.float32),
        constant=constant,
    )


def valid_evidence_rows(evidence: TaskEvidence) -> dict[str, np.ndarray]:
    """Valid ``[rows, features]`` evidence of one task; padding is excluded."""

    return {
        "vision": np.asarray(evidence.vision_features),
        "language": np.asarray(evidence.language_features)[np.asarray(evidence.language_mask)],
        "action": np.asarray(evidence.action_chunk_features)[
            np.asarray(evidence.action_chunk_mask)
        ],
    }


@dataclass(frozen=True)
class EvidenceStandardization:
    """Fold-specific standardization of all three evidence modalities."""

    modalities: Mapping[str, FeatureStandardization]
    fit_task_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if set(self.modalities) != set(COMPONENT_NAMES):
            raise ValueError("standardization must cover vision, language, and action")
        if not self.fit_task_ids or len(set(self.fit_task_ids)) != len(self.fit_task_ids):
            raise ValueError("standardization must record its unique training tasks")

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {
            "fit_task_ids": np.asarray(self.fit_task_ids, dtype=np.str_)
        }
        for name, value in self.modalities.items():
            arrays[f"{name}_mean"] = value.mean
            arrays[f"{name}_scale"] = value.scale
            arrays[f"{name}_constant"] = value.constant
        return arrays

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, Any]) -> "EvidenceStandardization":
        return cls(
            modalities={
                name: FeatureStandardization(
                    mean=np.array(arrays[f"{name}_mean"], dtype=np.float32),
                    scale=np.array(arrays[f"{name}_scale"], dtype=np.float32),
                    constant=np.array(arrays[f"{name}_constant"], dtype=bool),
                )
                for name in COMPONENT_NAMES
            },
            fit_task_ids=tuple(str(value) for value in np.asarray(arrays["fit_task_ids"])),
        )

    def apply(self, evidence: TaskEvidence) -> dict[str, np.ndarray]:
        """Standardized arrays with padded entries reset to exactly zero."""

        language_mask = np.asarray(evidence.language_mask, dtype=bool)
        chunk_mask = np.asarray(evidence.action_chunk_mask, dtype=bool)
        language = self.modalities["language"].apply(evidence.language_features)
        action = self.modalities["action"].apply(evidence.action_chunk_features)
        return {
            "vision": self.modalities["vision"].apply(evidence.vision_features),
            "vision_mask": np.ones(len(evidence.vision_features), dtype=bool),
            "language": np.where(language_mask[:, None], language, 0.0).astype(np.float32),
            "language_mask": language_mask,
            "action": np.where(chunk_mask[..., None], action, 0.0).astype(np.float32),
            "action_mask": chunk_mask,
        }


def fit_evidence_standardization(
    evidence_by_task: Mapping[str, TaskEvidence], train_task_ids: Sequence[str]
) -> EvidenceStandardization:
    """Fit on the named training tasks only; other tasks are never read."""

    train = tuple(train_task_ids)
    missing = sorted(set(train) - set(evidence_by_task))
    if missing:
        raise ValueError(f"training tasks lack evidence: {missing}")
    rows = [valid_evidence_rows(evidence_by_task[task_id]) for task_id in train]
    return EvidenceStandardization(
        modalities={
            name: fit_feature_standardization([item[name] for item in rows])
            for name in COMPONENT_NAMES
        },
        fit_task_ids=train,
    )
