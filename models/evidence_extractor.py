"""
models/evidence_extractor.py

Multi-Modal Task Evidence Extractor (e_vis, e_lang, e_act).
Compresses demonstration frames, language instruction, and action trajectories into task prompts.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from models.contrastive_alignment import (
    DeepSetsEvidenceEncoder,
    EvidenceProjector,
)
from models.weight_autoencoder import require_jax


def masked_action_statistics(
    actions: np.ndarray,
    mask: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Return per-dimension mean, std, velocity and jerk statistics.

    ``actions`` is ``[episodes, time, action_dim]`` and ``mask`` is
    ``[episodes, time]``. Differences never cross episode boundaries. Velocity
    and jerk entries are included only when all source timesteps are valid.
    """
    values = np.asarray(actions, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    norm_mean = np.asarray(mean, dtype=np.float64)
    norm_std = np.asarray(std, dtype=np.float64)
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("actions/mask must have shapes [E,T,D] and [E,T]")
    if norm_mean.shape != values.shape[-1:] or norm_std.shape != values.shape[-1:]:
        raise ValueError("normalization statistics must match action_dim")
    if np.any(norm_std <= epsilon):
        raise ValueError("action standard deviations must be greater than epsilon")
    if not valid.any():
        raise ValueError("at least one action timestep must be valid")

    normalized = (values - norm_mean) / norm_std
    flat = normalized[valid]
    velocity = np.diff(normalized, axis=1)
    velocity_mask = valid[:, 1:] & valid[:, :-1]
    jerk = np.diff(velocity, axis=1)
    jerk_mask = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]

    def moments(array: np.ndarray, item_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = array[item_mask]
        if selected.size == 0:
            zeros = np.zeros(values.shape[-1], dtype=np.float64)
            return zeros, zeros
        return selected.mean(axis=0), selected.std(axis=0)

    action_mean, action_std = flat.mean(axis=0), flat.std(axis=0)
    velocity_mean, velocity_std = moments(velocity, velocity_mask)
    jerk_mean, jerk_std = moments(jerk, jerk_mask)
    result = np.concatenate(
        [action_mean, action_std, velocity_mean, velocity_std, jerk_mean, jerk_std]
    )
    if not np.isfinite(result).all():
        raise ValueError("action evidence contains non-finite values")
    return result.astype(np.float32)


def pad_feature_set(
    features: list[np.ndarray] | tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Pad variable-length episode features without inventing valid tokens."""
    if not features:
        raise ValueError("features cannot be empty")
    arrays = [np.asarray(item) for item in features]
    if any(item.ndim != 2 for item in arrays):
        raise ValueError("each feature set must have shape [tokens, width]")
    widths = {item.shape[1] for item in arrays}
    if len(widths) != 1 or any(item.shape[0] == 0 for item in arrays):
        raise ValueError("feature widths must match and token sets must be non-empty")
    maximum = max(item.shape[0] for item in arrays)
    output = np.zeros((len(arrays), maximum, arrays[0].shape[1]), dtype=arrays[0].dtype)
    mask = np.zeros((len(arrays), maximum), dtype=bool)
    for index, item in enumerate(arrays):
        output[index, : item.shape[0]] = item
        mask[index, : item.shape[0]] = True
    return output, mask


__all__ = [
    "DeepSetsEvidenceEncoder",
    "EvidenceProjector",
    "masked_action_statistics",
    "pad_feature_set",
    "require_jax",
]
