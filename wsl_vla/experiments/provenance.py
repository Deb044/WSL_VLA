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

from ..contracts import RunManifest


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    """Hash a directory artifact by relative path and file content."""

    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"artifact directory is missing: {root}")
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"artifact directory is empty: {root}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(item)))
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


def installed_vcs_commit(distribution_name: str) -> str:
    """Read the immutable commit recorded by a PEP 610 VCS installation."""

    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required distribution is not installed: {distribution_name}") from exc
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        raise RuntimeError(
            f"{distribution_name} was not installed from a provenance-bearing VCS URL"
        )
    try:
        payload = json.loads(direct_url)
        commit = payload["vcs_info"]["commit_id"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{distribution_name} has invalid direct_url.json") from exc
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError(f"{distribution_name} has no full 40-character VCS commit")
    return commit


def assert_installed_vcs_revision(distribution_name: str, expected: str) -> str:
    if len(expected) != 40:
        raise ValueError("expected VCS revision must be a full 40-character commit")
    actual = installed_vcs_commit(distribution_name)
    if actual != expected:
        raise RuntimeError(
            f"{distribution_name} revision mismatch: installed={actual}, expected={expected}"
        )
    return actual


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
