"""
data/dataset.py

Strict real-data LIBERO HDF5 loader for robotic demonstration trajectories.
Synthetic data generation is strictly forbidden.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Mapping, Optional
import h5py
import numpy as np

from data.splits import EpisodeSplit, split_episodes

REQUIRED_OBSERVATIONS = ("agentview_rgb",)


@dataclass(frozen=True)
class LiberoEpisode:
    episode_id: str
    actions: np.ndarray
    observations: Mapping[str, np.ndarray]


class StrictLiberoHDF5:
    """Fail-closed loader that preserves complete episode boundaries from real LIBERO HDF5 files."""

    def __init__(
        self,
        path: str | Path,
        *,
        require_wrist_camera: bool = False,
        required_proprio_keys: tuple[str, ...] = (),
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"real LIBERO HDF5 file not found: {self.path}")
        self.required_observations = list(REQUIRED_OBSERVATIONS)
        if require_wrist_camera:
            self.required_observations.append("eye_in_hand_rgb")
        self.required_observations.extend(required_proprio_keys)
        self._episode_ids = self._inspect()

    def _inspect(self) -> tuple[str, ...]:
        with h5py.File(self.path, "r") as handle:
            if "data" not in handle:
                raise ValueError(f"{self.path} has no /data group")
            episode_ids = tuple(sorted(handle["data"].keys()))
            if not episode_ids:
                raise ValueError(f"{self.path} contains no demonstrations")
            for episode_id in episode_ids:
                demo = handle["data"][episode_id]
                if "actions" not in demo or "obs" not in demo:
                    raise ValueError(f"{episode_id} lacks actions or observations")
                actions = demo["actions"]
                if actions.ndim != 2 or actions.shape[0] < 2:
                    raise ValueError(f"{episode_id} has invalid action shape {actions.shape}")
                for key in self.required_observations:
                    if key not in demo["obs"]:
                        raise ValueError(f"{episode_id} lacks required observation {key}")
                    if demo["obs"][key].shape[0] != actions.shape[0]:
                        raise ValueError(f"{episode_id}/{key} length differs from actions")
            return episode_ids

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return self._episode_ids

    def split(self, *, seed: int) -> EpisodeSplit:
        return split_episodes(self.episode_ids, seed=seed)

    def iter_episodes(self, episode_ids: tuple[str, ...] | None = None) -> Iterator[LiberoEpisode]:
        selected = episode_ids or self.episode_ids
        unknown = set(selected) - set(self.episode_ids)
        if unknown:
            raise KeyError(f"unknown episode ids: {sorted(unknown)}")
        with h5py.File(self.path, "r") as handle:
            for episode_id in selected:
                demo = handle["data"][episode_id]
                observations = {
                    key: np.asarray(value)
                    for key, value in demo["obs"].items()
                }
                yield LiberoEpisode(
                    episode_id=episode_id,
                    actions=np.asarray(demo["actions"], dtype=np.float32),
                    observations=observations,
                )


def forbid_synthetic_research_output(*, smoke_test: bool, output_directory: str | Path) -> None:
    """Guards against any synthetic data entering research results."""
    output = Path(output_directory)
    if smoke_test and "research_results" in {part.lower() for part in output.parts}:
        raise ValueError("smoke-test or synthetic runs cannot write into research_results")


def load_suite_manifest(
    suite_directory: str | Path,
    expected_instructions: tuple[str, ...] | list[str],
) -> tuple[Path, ...]:
    """Resolve ten ordered task files from an explicit, checked sidecar."""
    root = Path(suite_directory)
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"suite data manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("tasks"), list):
        raise ValueError(f"unsupported suite manifest: {path}")
    tasks = payload["tasks"]
    if len(tasks) != len(expected_instructions):
        raise ValueError(f"suite manifest task count differs from protocol: {path}")
    resolved = []
    for index, (entry, instruction) in enumerate(zip(tasks, expected_instructions)):
        if entry.get("task_index") != index or entry.get("instruction") != instruction:
            raise ValueError(f"suite manifest order/instruction mismatch at task {index}: {path}")
        filename = entry.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"suite manifest contains unsafe task filename: {filename!r}")
        task_path = root / filename
        if not task_path.is_file() or task_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise FileNotFoundError(f"suite task file is missing: {task_path}")
        resolved.append(task_path)
    return tuple(resolved)
