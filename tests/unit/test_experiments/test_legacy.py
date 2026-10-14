"""Version-one artifacts stay inspectable and are never migrated."""

import json

import numpy as np
import pytest

from wsl_vla.adapters.packing import load_packed_adapter
from wsl_vla.contracts import AdapterSpec, LegacyArtifactError, require_schema
from wsl_vla.experiments.legacy import inspect_legacy_adapter, inspect_legacy_tree


def write_legacy_adapter(path):
    spec = {
        "schema_version": 1,
        "base_model_id": "m", "base_revision": "r", "base_sha256": "a" * 64,
        "alpha": 16.0, "token_width": 4,
        "entries": [
            {"component": "vision", "layer": 10, "parameter_path": "t/adapter_vision_10",
             "input_dim": 2, "output_dim": 4, "rank": 1},
            {"component": "vision", "layer": 11, "parameter_path": "t/adapter_vision_11",
             "input_dim": 2, "output_dim": 4, "rank": 1},
        ],
    }
    metadata = {"task_id": "libero_spatial_0", "suite": "libero_spatial", "seed": 17,
                "checkpoint_stage": 2000, "checkpoint_fraction": 1.0, "source_dtype": "float32"}
    tokens = np.zeros((4, 4), dtype=np.float32)
    tokens[:2] = 1.0  # the final-layer entry never trained
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        tokens=tokens,
        mask=np.ones((4, 4), dtype=bool),
        entry_ids=np.array([0, 0, 1, 1], dtype=np.int32),
        adapter_spec_json=np.frombuffer(json.dumps(spec).encode(), dtype=np.uint8),
        packed_adapter_metadata_json=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
    )
    return spec


def test_legacy_adapter_is_described_but_rejected_by_research_loaders(tmp_path):
    path = tmp_path / "population" / "s" / "t" / "seed_17" / "step_2000" / "adapter.npz"
    spec = write_legacy_adapter(path)
    report = inspect_legacy_adapter(path)
    assert report["adapter_spec_schema"] == 1 and report["packed_schema"] == 1
    assert report["identically_zero_entries"] == ["adapter_vision_11"]
    with pytest.raises(LegacyArtifactError):
        load_packed_adapter(path)
    with pytest.raises(LegacyArtifactError):
        AdapterSpec.from_dict(spec)


def test_legacy_tree_inspection_never_touches_corrected_experiments(tmp_path):
    write_legacy_adapter(tmp_path / "population" / "s" / "t" / "seed_17" / "step_2000" / "adapter.npz")
    write_legacy_adapter(tmp_path / "experiments" / "pilot" / "population" / "x" / "adapter.npz")
    gate = tmp_path / "alignment" / "libero_spatial" / "alignment_advantage.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps({"held_out_suite": "libero_spatial", "passed": False, "task_count": 6}))
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    report = inspect_legacy_tree(tmp_path)
    assert report["adapter_samples"] == 1 and report["adapters_with_dead_entries"] == 1
    assert report["entries_identically_zero_in_some_sample"] == ["adapter_vision_11"]
    assert report["alignment_gate_reports"][0]["legacy_gate_passed"] is False
    assert "never trained" in report["why_legacy"][0]
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before


def test_schema_guard_distinguishes_legacy_from_unknown_versions():
    require_schema(2, 2, "artifact")
    with pytest.raises(LegacyArtifactError, match="inspection-only"):
        require_schema(1, 2, "artifact")
    with pytest.raises(ValueError, match="unsupported artifact schema 3"):
        require_schema(3, 2, "artifact")
    with pytest.raises(ValueError, match="unsupported artifact schema None"):
        require_schema(None, 2, "artifact")
