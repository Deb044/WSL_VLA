from __future__ import annotations

import numpy as np
import pytest

from wsl_vla.octo.continual import _validate_batches, tree_nbytes


def test_tree_nbytes_counts_numeric_array_leaves():
    tree = {
        "a": np.zeros((2, 3), dtype=np.float32),
        "b": {"c": np.ones(4, dtype=np.int16)},
    }
    assert tree_nbytes(tree) == 2 * 3 * 4 + 4 * 2


def test_official_stage_batches_fail_closed_on_missing_fields():
    valid = {
        "observation": {},
        "task": {},
        "action": np.zeros(1),
        "action_pad_mask": np.ones(1),
    }
    _validate_batches([valid], 1)
    with pytest.raises(ValueError, match="official Octo fields"):
        _validate_batches([{"observation": {}}], 1)
    with pytest.raises(ValueError, match="positive"):
        _validate_batches([valid], 0)
