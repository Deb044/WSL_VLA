import hashlib

import numpy as np
import pytest

from wsl_vla.alignment.evidence import masked_action_statistics, pad_feature_set
from wsl_vla.evaluation.sequential import (
    RolloutOutcome,
    StageUpdate,
    run_independent_oracle_protocol,
    run_sequential_protocol,
)


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
        return RolloutOutcome(
            successes=min(count, len(state)),
            initialization_indices=tuple(range(count)),
            rollout_seeds=tuple(seed * 100 + index for index in range(count)),
        )

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
    assert records[-1].initialization_indices == tuple(range(10))
    assert observed[-3:] == [(("a", "b", "c"), "a"), (("a", "b", "c"), "b"), (("a", "b", "c"), "c")]


def test_sequential_runner_rejects_rollouts_without_initialization_provenance():
    def train(state, task_id, stage):
        return StageUpdate(state, "a" * 64, {})

    with pytest.raises(TypeError, match="initialization provenance"):
        run_sequential_protocol(
            initial_state=None,
            task_ids=("a",),
            train_stage=train,
            rollout=lambda state, task, count, seed: count,
            run_id="test",
            suite="suite",
            seed=1,
            condition="condition",
            rollout_count=2,
        )


def test_sequential_runner_rejects_a_different_rollout_checkpoint():
    def train(state, task_id, stage):
        return StageUpdate(state, "a" * 64, {})

    with pytest.raises(ValueError, match="differs from the trained"):
        run_sequential_protocol(
            initial_state=None,
            task_ids=("a",),
            train_stage=train,
            rollout=lambda state, task, count, seed: RolloutOutcome(
                1, (0,), (seed,), "b" * 64
            ),
            run_id="test",
            suite="suite",
            seed=1,
            condition="condition",
            rollout_count=1,
        )


def test_sequential_runner_persists_stages_and_records_incrementally():
    events = []

    def train(state, task_id, stage):
        return StageUpdate(state + 1, "a" * 64, {})

    def rollout(state, task_id, count, seed):
        return RolloutOutcome(
            successes=1,
            initialization_indices=(0,),
            rollout_seeds=(seed,),
        )

    run_sequential_protocol(
        initial_state=0,
        task_ids=("a", "b"),
        train_stage=train,
        rollout=rollout,
        run_id="test",
        suite="suite",
        seed=1,
        condition="condition",
        rollout_count=1,
        stage_sink=lambda stage, task, update: events.append(
            ("stage", stage, task, update.state)
        ),
        record_sink=lambda record: events.append(
            ("record", record.training_stage, record.evaluated_task_id)
        ),
    )
    assert events == [
        ("stage", 0, "a", 1),
        ("record", 0, "a"),
        ("stage", 1, "b", 2),
        ("record", 1, "a"),
        ("record", 1, "b"),
    ]


def test_oracle_trains_each_task_from_same_initial_state_and_selects_own_adapter():
    trained_from = []
    evaluated = []

    def train(state, task_id, stage):
        trained_from.append((state, task_id))
        task_state = f"{state}:{task_id}"
        return StageUpdate(
            task_state,
            hashlib.sha256(task_state.encode()).hexdigest(),
            {},
        )

    def rollout(state, task_id, count, seed):
        evaluated.append((state, task_id))
        return RolloutOutcome(1, (0,), (seed,))

    states, records = run_independent_oracle_protocol(
        initial_state="base",
        task_ids=("a", "b"),
        train_task=train,
        rollout=rollout,
        run_id="oracle",
        suite="suite",
        seed=9,
        rollout_count=1,
    )
    assert trained_from == [("base", "a"), ("base", "b")]
    assert states == {"a": "base:a", "b": "base:b"}
    assert evaluated == [("base:a", "a"), ("base:a", "a"), ("base:b", "b")]
    assert {record.condition for record in records} == {"independent_adapter_oracle"}
