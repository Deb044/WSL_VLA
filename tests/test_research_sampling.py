from __future__ import annotations

import numpy as np
import pytest

from wsl_vla.research.sampling import task_balanced_index_batches


def test_batches_have_multiple_tasks_and_multiple_positives():
    labels = np.repeat(np.arange(3), 3)
    batches = task_balanced_index_batches(
        range(9), labels, tasks_per_batch=2, samples_per_task=2, seed=7, shuffle=True
    )
    covered = set()
    for batch in batches:
        unique, counts = np.unique(labels[batch], return_counts=True)
        assert len(unique) == 2
        assert np.all(counts == 2)
        covered.update(batch.tolist())
    assert covered == set(range(9))


def test_batch_schedule_is_seed_deterministic():
    labels = np.repeat(np.arange(4), 4)
    left = task_balanced_index_batches(
        range(16), labels, tasks_per_batch=2, samples_per_task=2, seed=12, shuffle=True
    )
    right = task_balanced_index_batches(
        range(16), labels, tasks_per_batch=2, samples_per_task=2, seed=12, shuffle=True
    )
    assert all(np.array_equal(a, b) for a, b in zip(left, right))


def test_invalid_contrastive_batch_is_rejected():
    with pytest.raises(ValueError, match="at least two task identities"):
        task_balanced_index_batches(
            range(4), np.array([0, 0, 1, 1]), tasks_per_batch=1,
            samples_per_task=2, seed=0, shuffle=False
        )
