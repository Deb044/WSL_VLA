"""Command-chunk evidence, episode isolation, and training-only standardization."""

import numpy as np
import pytest

from tests.synthetic import synthetic_evidence
from wsl_vla.alignment.evidence import (
    ACTION_SUMMARY_FIELDS,
    MIN_CHUNK_LENGTH,
    action_chunk_evidence,
    chunk_command_summary,
    episode_chunk_summaries,
    fit_evidence_standardization,
    fit_feature_standardization,
)


def test_chunk_summary_holds_command_moments_and_three_command_differences():
    # A cubic command has constant third differences and zero spread there.
    time = np.arange(8, dtype=np.float64)
    chunk = np.stack([time**3, np.full(8, 2.0)], axis=1)
    summary = chunk_command_summary(chunk).reshape(len(ACTION_SUMMARY_FIELDS), 2)
    fields = dict(zip(ACTION_SUMMARY_FIELDS, summary))
    np.testing.assert_allclose(fields["command_mean"], [np.mean(time**3), 2.0])
    np.testing.assert_allclose(fields["difference3_mean"], [6.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(fields["difference3_std"], [0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(fields["difference1_mean"][1], 0.0)
    assert all("velocity" not in name and "jerk" not in name for name in ACTION_SUMMARY_FIELDS)
    with pytest.raises(ValueError, match=f"at least {MIN_CHUNK_LENGTH}"):
        chunk_command_summary(chunk[:3])


def test_chunks_are_non_overlapping_and_a_short_tail_is_dropped_not_padded():
    actions = np.arange(22, dtype=np.float32).reshape(11, 2)
    summaries = episode_chunk_summaries(actions, chunk_length=4)
    # 11 commands -> chunks [0:4], [4:8]; the 3-command tail cannot hold a third difference.
    assert summaries.shape == (2, 16)
    np.testing.assert_allclose(summaries[1], chunk_command_summary(actions[4:8]))
    with_tail = episode_chunk_summaries(np.arange(24, dtype=np.float32).reshape(12, 2), chunk_length=4)
    assert with_tail.shape[0] == 3


def test_action_evidence_never_crosses_episode_boundaries_or_reads_padding():
    rng = np.random.default_rng(0)
    first = rng.normal(size=(9, 2)).astype(np.float32)
    second = rng.normal(size=(17, 2)).astype(np.float32)
    features, mask = action_chunk_evidence([first, second], chunk_length=4)
    assert features.shape == (2, 4, 16) and mask.tolist() == [[True, True, False, False], [True] * 4]
    assert not features[0, 2:].any()
    # Changing one episode's commands (or making a huge jump at its boundary)
    # leaves every other episode's summaries untouched.
    changed, _ = action_chunk_evidence([first + 1000.0, second], chunk_length=4)
    np.testing.assert_array_equal(changed[1], features[1])
    alone, _ = action_chunk_evidence([second], chunk_length=4)
    np.testing.assert_array_equal(alone[0], features[1])


def test_standardization_is_task_balanced_and_keeps_unit_scale_for_constants():
    many = np.concatenate([np.zeros((1000, 1)), np.full((1000, 1), 5.0)], axis=1)
    few = np.concatenate([np.full((2, 1), 4.0), np.full((2, 1), 5.0)], axis=1)
    fitted = fit_feature_standardization([many, few])
    # Each task counts once: the mean is 2, not ~0 as a pooled mean would give.
    np.testing.assert_allclose(fitted.mean, [2.0, 5.0])
    np.testing.assert_allclose(fitted.scale, [2.0, 1.0])
    assert fitted.constant.tolist() == [False, True]
    # A constant feature keeps unit scale: an unseen value is shifted, not blown up.
    np.testing.assert_allclose(fitted.apply(np.array([[2.0, 7.0]])), [[0.0, 2.0]])


def test_evidence_standardization_is_fitted_on_training_tasks_only():
    rng = np.random.default_rng(1)
    tasks = {
        name: synthetic_evidence(name, rng, rng.normal(size=8))
        for name in ("libero_object_0", "libero_object_1", "libero_goal_8")
    }
    train = ("libero_object_0", "libero_object_1")
    fitted = fit_evidence_standardization(tasks, train)
    assert fitted.fit_task_ids == train
    # Replacing the validation task's evidence cannot change the statistics.
    shifted = dict(tasks)
    shifted["libero_goal_8"] = synthetic_evidence("libero_goal_8", rng, 100 + rng.normal(size=8))
    refit = fit_evidence_standardization(shifted, train)
    for name in ("vision", "language", "action"):
        np.testing.assert_array_equal(refit.modalities[name].mean, fitted.modalities[name].mean)
        np.testing.assert_array_equal(refit.modalities[name].scale, fitted.modalities[name].scale)
    # ...and it is applied unchanged, with padding reset to zero.
    applied = fitted.apply(shifted["libero_goal_8"])
    assert not applied["language"][~applied["language_mask"]].any()
    assert not applied["action"][~applied["action_mask"]].any()
    restored = fitted.from_arrays(fitted.to_arrays())
    np.testing.assert_array_equal(restored.modalities["action"].scale, fitted.modalities["action"].scale)
    with pytest.raises(ValueError, match="lack evidence"):
        fit_evidence_standardization(tasks, ("libero_object_0", "missing_task_3"))
