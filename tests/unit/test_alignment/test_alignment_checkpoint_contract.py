from __future__ import annotations

import numpy as np
import pytest

from tests.synthetic import synthetic_spec
from wsl_vla.adapters.packing import build_token_layout
from wsl_vla.contracts import AlignmentCheckpoint, LegacyArtifactError


def make_checkpoint(**changes):
    spec = synthetic_spec()
    values = dict(
        params={},
        model=object(),
        config={"condition": "aligned", "seed": 0},
        adapter_spec=spec,
        layout=build_token_layout(spec, sequence_window=8),
        entry_scales=np.ones(len(spec.entries), dtype=np.float32),
        standardization=object(),
        mappers={name: object() for name in ("vision", "language", "action")},
        shells={name: object() for name in ("vision", "language", "action")},
        metadata={"base_sha256": "a" * 64},
    )
    values.update(changes)
    return AlignmentCheckpoint(**values)


def test_alignment_checkpoint_requires_complete_modality_geometry():
    checkpoint = make_checkpoint()
    checkpoint.validate()
    assert checkpoint.condition == "aligned"
    assert checkpoint.token_mask.shape == (checkpoint.layout.token_count, 6)
    del checkpoint.mappers["action"]
    with pytest.raises(ValueError, match="all three modality mappers"):
        checkpoint.validate()


def test_alignment_checkpoint_requires_persisted_positive_scales_per_entry():
    with pytest.raises(ValueError, match="one RMS scale per adapter entry"):
        make_checkpoint(entry_scales=np.ones(3, dtype=np.float32)).validate()
    scales = np.ones(len(synthetic_spec().entries), dtype=np.float32)
    scales[0] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        make_checkpoint(entry_scales=scales).validate()


def test_alignment_checkpoint_rejects_other_conditions_and_legacy_schema():
    with pytest.raises(ValueError, match="not a primary condition"):
        make_checkpoint(config={"condition": "contrastive_only"}).validate()
    with pytest.raises(LegacyArtifactError):
        make_checkpoint(schema_version=1).validate()
