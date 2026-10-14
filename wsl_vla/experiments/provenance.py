"""Hashing, host capture, and atomic manifest persistence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..contracts import RunManifest


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Write a file so that readers see the old content or the new, never a part."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # mkstemp creates 0600 files; artifacts are handed to collaborators.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(temporary, 0o666 & ~umask)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


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


def capture_environment_lock() -> dict[str, Any]:
    """Capture every installed distribution without leaking index credentials."""
    distributions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            distributions[str(name).lower()] = distribution.version
    return {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "distributions": dict(sorted(distributions.items())),
    }


def _editable_checkout_commit(distribution_name: str, url: str) -> str:
    """Return HEAD of an editable git checkout whose tracked files are unmodified."""

    from urllib.parse import unquote, urlparse

    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise RuntimeError(f"{distribution_name} editable install is not a local checkout")
    checkout = unquote(parsed.path)
    try:
        commit = subprocess.check_output(
            ["git", "-C", checkout, "rev-parse", "HEAD"], text=True, timeout=30
        ).strip()
        modified = subprocess.check_output(
            ["git", "-C", checkout, "status", "--porcelain", "--untracked-files=no"],
            text=True,
            timeout=30,
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"{distribution_name} editable install is not a readable git checkout"
        ) from exc
    if modified:
        raise RuntimeError(f"{distribution_name} editable checkout has modified tracked files")
    return commit


def _legacy_editable_checkout(distribution: Any) -> Path | None:
    """Source checkout of a legacy editable install, or ``None`` if it is not one."""

    metadata = getattr(distribution, "_path", None)
    if metadata is None:
        return None
    metadata = Path(str(metadata))
    if metadata.suffix != ".egg-info" or not (metadata.parent / ".git").exists():
        return None
    return metadata.parent.resolve()


def installed_vcs_commit(distribution_name: str) -> str:
    """Read the immutable commit recorded by a PEP 610 VCS installation."""

    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required distribution is not installed: {distribution_name}") from exc
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        # A legacy editable install ("setup.py develop", which pip still uses for
        # projects without a pyproject.toml such as LIBERO) keeps its metadata in
        # an .egg-info inside the source checkout and has no direct_url.json.
        # The checkout itself is then the evidence: read its clean HEAD.
        checkout = _legacy_editable_checkout(distribution)
        if checkout is None:
            raise RuntimeError(
                f"{distribution_name} was not installed from a provenance-bearing VCS URL"
            )
        commit = _editable_checkout_commit(distribution_name, checkout.as_uri())
        if len(commit) != 40:
            raise RuntimeError(f"{distribution_name} has no full 40-character VCS commit")
        return commit
    try:
        payload = json.loads(direct_url)
        if "vcs_info" in payload:
            commit = payload["vcs_info"]["commit_id"]
        elif payload["dir_info"]["editable"] is True:
            # LIBERO only imports from an editable checkout; pip then records
            # the directory rather than the commit, so read the clean checkout.
            commit = _editable_checkout_commit(distribution_name, payload["url"])
        else:
            raise KeyError("vcs_info")
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
