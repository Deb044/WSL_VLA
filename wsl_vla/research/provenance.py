"""Hashing, host capture, and atomic manifest persistence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import RunManifest


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array_tree(tree: Mapping[str, Any]) -> str:
    """Hash a flat or nested array tree deterministically."""

    digest = hashlib.sha256()

    def walk(prefix: tuple[str, ...], value: Any) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                walk((*prefix, str(key)), value[key])
            return
        array = np.asarray(value)
        digest.update("/".join(prefix).encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())

    walk((), tree)
    return digest.hexdigest()


def git_state(repository: str | Path) -> tuple[str, bool]:
    root = str(repository)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    return commit, dirty


def capture_environment(packages: Iterable[str]) -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def capture_hardware() -> dict[str, Any]:
    hardware: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        hardware["nvidia_gpus"] = output.splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        hardware["nvidia_gpus"] = []
    return hardware


def write_manifest_atomic(manifest: RunManifest, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(manifest), indent=2, sort_keys=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
