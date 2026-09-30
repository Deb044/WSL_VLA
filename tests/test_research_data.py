from __future__ import annotations

from pathlib import Path
import json

import h5py
import numpy as np
import pytest

from wsl_vla.research.data import (
    StrictLiberoHDF5,
    forbid_synthetic_research_output,
    load_suite_manifest,
)
from wsl_vla.research.splits import split_episodes


def make_hdf5(path: Path, episodes: int = 10) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for index in range(episodes):
            demo = data.create_group(f"demo_{index}")
            demo.create_dataset("actions", data=np.zeros((4, 7), dtype=np.float32))
            obs = demo.create_group("obs")
            obs.create_dataset("agentview_rgb", data=np.zeros((4, 8, 8, 3), dtype=np.uint8))


def test_real_hdf5_is_episode_split_without_leakage(tmp_path):
    path = tmp_path / "task.hdf5"
    make_hdf5(path)
    dataset = StrictLiberoHDF5(path)
    split = dataset.split(seed=42)
    split.validate()
    assert set(split.train) | set(split.validation) | set(split.test) == set(dataset.episode_ids)
    assert len(list(dataset.iter_episodes(split.test))) == len(split.test)


def test_missing_real_data_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        StrictLiberoHDF5(tmp_path / "missing.hdf5")


def test_synthetic_smoke_output_cannot_enter_research_results():
    with pytest.raises(ValueError, match="cannot write"):
        forbid_synthetic_research_output(
            smoke_test=True,
            output_directory="research_results/records",
        )


def test_episode_split_is_deterministic_and_seeded():
    episodes = [f"demo_{index}" for index in range(20)]
    first = split_episodes(episodes, seed=1)
    second = split_episodes(episodes, seed=1)
    third = split_episodes(episodes, seed=2)
    assert first == second
    assert first != third


def test_suite_manifest_locks_file_to_instruction_order(tmp_path):
    instructions = [f"task {index}" for index in range(2)]
    for index in range(2):
        (tmp_path / f"task_{index}.hdf5").touch()
    payload = {
        "schema_version": 1,
        "tasks": [
            {
                "task_index": index,
                "instruction": instruction,
                "file": f"task_{index}.hdf5",
            }
            for index, instruction in enumerate(instructions)
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert len(load_suite_manifest(tmp_path, instructions)) == 2
    payload["tasks"][0]["instruction"] = "wrong"
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="order/instruction"):
        load_suite_manifest(tmp_path, instructions)
