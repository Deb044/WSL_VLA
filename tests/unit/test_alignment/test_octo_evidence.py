from __future__ import annotations

import numpy as np
import pytest

from wsl_vla.alignment.octo_evidence import evidence_frame_schedule, masked_pool_tokens


def test_masked_pool_ignores_padding_tokens():
    tokens = np.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = np.array([[True, True, False]])
    np.testing.assert_allclose(masked_pool_tokens(tokens, mask), [[2.0, 3.0]])


def test_entirely_masked_token_group_is_rejected():
    with pytest.raises(ValueError, match="entirely masked"):
        masked_pool_tokens(np.ones((1, 2, 3)), np.zeros((1, 2), dtype=bool))


def test_frame_schedule_spans_the_episode_without_duplicates():
    np.testing.assert_array_equal(evidence_frame_schedule(100, 4), [0, 33, 66, 99])
    np.testing.assert_array_equal(evidence_frame_schedule(3, 8), [0, 1, 2])
    assert evidence_frame_schedule(1, 4).tolist() == [0]
    with pytest.raises(ValueError):
        evidence_frame_schedule(0, 4)
