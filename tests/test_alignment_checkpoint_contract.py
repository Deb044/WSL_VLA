from __future__ import annotations

import numpy as np
import pytest

from wsl_vla.research.contracts import AlignmentCheckpoint


def make_checkpoint():
    return AlignmentCheckpoint(
        params={},
        model=object(),
        mappers={name: object() for name in ("vision", "language", "action")},
        shells={name: object() for name in ("vision", "language", "action")},
        token_mask=np.ones((3, 4), dtype=bool),
        component_ids=np.array([0, 1, 2]),
        layer_ids=np.array([0, 0, 0]),
        metadata={"base_sha256": "a" * 64},
    )


def test_alignment_checkpoint_requires_complete_modality_geometry():
    checkpoint = make_checkpoint()
    checkpoint.validate()
    del checkpoint.mappers["action"]
    with pytest.raises(ValueError, match="all three modality mappers"):
        checkpoint.validate()


def test_alignment_checkpoint_rejects_inconsistent_layout():
    checkpoint = make_checkpoint()
    checkpoint.layer_ids = np.array([0, 0])
    with pytest.raises(ValueError, match="layer layout"):
        checkpoint.validate()
