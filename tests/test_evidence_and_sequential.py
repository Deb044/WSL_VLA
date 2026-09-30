import hashlib

import numpy as np
import pytest

from wsl_vla.alignment.evidence import masked_action_statistics, pad_feature_set
from wsl_vla.evaluation.sequential import StageUpdate, run_sequential_protocol


def test_action_statistics_respect_episode_boundaries_and_masks():
    actions = np.array([[[0.0], [1.0], [2.0]], [[100.0], [101.0], [999.0]]])
    mask = np.array([[True, True, True], [True, True, False]])
    stats = masked_action_statistics(actions, mask, mean=np.array([0.0]), std=np.array([1.0]))
    assert stats.shape == (6,)
    assert stats[2] == pytest.approx(1.0)  # no spurious 98-unit cross-episode jump
    assert stats[4] == pytest.approx(0.0)


def test_pad_feature_set_preserves_validity_mask():
    padded, mask = pad_feature_set((np.ones((2, 3)), np.ones((1, 3)) * 2))
    assert padded.shape == (2, 2, 3)
    assert mask.tolist() == [[True, True], [True, False]]
    assert padded[1, 1].tolist() == [0.0, 0.0, 0.0]


def test_sequential_runner_evaluates_current_state_not_task_snapshots():
    observed = []

    def train(state, task_id, stage):
        new_state = state + (task_id,)
        digest = hashlib.sha256("|".join(new_state).encode()).hexdigest()
        return StageUpdate(new_state, digest, {"vision": float(stage)})

    def rollout(state, task_id, count, seed):
        observed.append((state, task_id))
        return min(count, len(state))

    state, records = run_sequential_protocol(
        initial_state=(),
        task_ids=("a", "b", "c"),
        train_stage=train,
        rollout=rollout,
        run_id="test",
        suite="suite",
        seed=7,
        condition="sequential_lora",
        rollout_count=10,
    )
    assert state == ("a", "b", "c")
    assert len(records) == 6
    assert observed[-3:] == [(("a", "b", "c"), "a"), (("a", "b", "c"), "b"), (("a", "b", "c"), "c")]
