from __future__ import annotations

import numpy as np
import pytest

from tests.fakes import FakeTextProcessor, episodes, example_batch, preprocessing_spec
from wsl_vla.data.streams import (
    TaskTrainingStream,
    TransitionCursor,
    make_octo_batch_factory,
    sample_replay_transitions,
)


def stream(task_id="task_0", seed=0):
    spec = preprocessing_spec()
    return TaskTrainingStream(
        task_id=task_id,
        instruction=f"do {task_id}",
        episodes=episodes(3, seed=seed),
        preprocessing=spec,
        example_batch=example_batch(spec),
    )


def test_replay_sampling_is_deterministic_identifiable_and_exact():
    left = sample_replay_transitions(stream(), count=10, seed=7)
    right = sample_replay_transitions(stream(), count=10, seed=7)
    assert [(x.episode_id, x.timestep) for x in left] == [
        (x.episode_id, x.timestep) for x in right
    ]
    assert len({(x.episode_id, x.timestep) for x in left}) == 10


def test_batch_factory_mixes_only_prior_replay_and_is_memory_bounded():
    current = stream("task_1", seed=1)
    replay = sample_replay_transitions(stream("task_0"), count=10, seed=3)
    factory = make_octo_batch_factory(
        current,
        batch_size=2,
        seed=5,
        text_processor=FakeTextProcessor(),
        replay=replay,
        shuffle_buffer=3,
    )
    batch = next(iter(factory()))
    assert batch["action"].shape == (2, 2, 3, 7)
    assert batch["observation"]["image_primary"].shape == (2, 2, 16, 16, 3)
    with pytest.raises(ValueError, match="only prior"):
        make_octo_batch_factory(
            current,
            batch_size=2,
            seed=5,
            text_processor=FakeTextProcessor(),
            replay=sample_replay_transitions(current, count=10, seed=3),
        )


def test_stream_validates_actions_and_reports_identity_normalization():
    current = stream()
    assert current.normalization.to_dict()["kind"] == "identity"
    assert current.normalization.count == current.transition_count == 6 + 7 + 8
    assert current.transition_index().shape == (21, 2)


def test_transition_cursor_resumes_exactly_and_never_emits_a_partial_batch():
    reference = TransitionCursor(transition_count=7, batch_size=3, seed=11)
    batches = [reference.next_indices() for _ in range(9)]
    assert all(len(batch) == 3 for batch in batches)
    # Every epoch is a permutation: 27 draws cover the 7 transitions 3-4 times each.
    counts = np.bincount(np.concatenate(batches), minlength=7)
    assert counts.min() >= 3 and counts.max() <= 4
    resumed = TransitionCursor(transition_count=7, batch_size=3, seed=11)
    for _ in range(4):
        resumed.next_indices()
    restored = TransitionCursor.restore(
        resumed.state(), transition_count=7, batch_size=3, seed=11
    )
    for expected in batches[4:]:
        np.testing.assert_array_equal(restored.next_indices(), expected)
    with pytest.raises(ValueError, match="different seed"):
        TransitionCursor.restore(resumed.state(), transition_count=7, batch_size=3, seed=12)
    with pytest.raises(ValueError, match="different transition_count"):
        TransitionCursor.restore(resumed.state(), transition_count=8, batch_size=3, seed=11)
