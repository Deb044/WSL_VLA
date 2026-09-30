from __future__ import annotations

import numpy as np
import pytest

from wsl_vla.research.octo_evidence import (
    flatten_visual_token_groups,
    masked_pool_tokens,
)


def test_masked_language_pool_ignores_padding():
    tokens = np.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = np.array([[True, True, False]])
    np.testing.assert_allclose(masked_pool_tokens(tokens, mask), [2.0, 3.0])


def test_visual_groups_are_combined_in_stable_view_order():
    groups = {
        "image_wrist": (np.array([[[[5.0, 6.0]]]]), np.array([[[True]]])),
        "image_primary": (
            np.array([[[[1.0, 2.0], [3.0, 4.0]]]]),
            np.array([[[True, False]]]),
        ),
    }
    features, mask = flatten_visual_token_groups(groups)
    np.testing.assert_array_equal(features, np.array([[1.0, 2.0], [5.0, 6.0]]))
    np.testing.assert_array_equal(mask, np.ones(2, dtype=bool))


def test_visual_subsampling_is_deterministic_and_spans_demonstration():
    tokens = np.arange(20, dtype=np.float32).reshape(1, 1, 10, 2)
    features, _ = flatten_visual_token_groups(
        {"image_primary": (tokens, np.ones((1, 1, 10), dtype=bool))},
        maximum_tokens=3,
    )
    np.testing.assert_array_equal(features, tokens.reshape(10, 2)[[0, 4, 9]])


def test_entirely_masked_language_is_rejected():
    with pytest.raises(ValueError, match="entirely masked"):
        masked_pool_tokens(np.ones((1, 2, 3)), np.zeros((1, 2), dtype=bool))
