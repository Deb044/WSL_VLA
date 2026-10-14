"""Compatibility layer between the two locked research environments.

The pinned Octo-Small checkpoint is used through two separately locked stacks:

``blackwell``
    Python 3.11, JAX 0.7.1 (``jax[cuda12]``, CUDA >= 12.8 for sm_120), Flax
    0.12. Upstream Octo ``241fb35`` does not import on this JAX, so
    ``third_party/octo/*.patch`` ports it and
    ``scripts/setup/build_patched_octo.sh`` rebuilds the port as the
    deterministic commit ``a4cc964``.

``legacy-gpu``
    Python 3.10, JAX 0.4.20 (CUDA 11 pip wheels), Flax 0.7.5, and *unmodified*
    upstream Octo ``241fb35``. This is the stack Octo was released against.

The port only replaces JAX aliases that were removed (``jax.tree_map`` ->
``jax.tree_util.tree_map`` and similar). It must never change the pretrained
architecture; :func:`architecture_identity` hashes the model configuration, the
complete parameter tree structure (paths, shapes, dtypes), and the example
batch structure, so both profiles must report the same architecture hash for
the same checkpoint. Everything version-specific that this repository itself
needs lives in this module.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
import sys
from typing import Any, Iterator, Mapping

import numpy as np

from ..contracts import canonical_sha256


OCTO_UPSTREAM_REVISION = "241fb3514b7c40957a86d869fecb7c7fc353f540"
OCTO_PATCHED_REVISION = "a4cc964b7e77f8d8b19f533a0dfa95d653501ab7"

ENVIRONMENT_PROFILES: dict[str, dict[str, Any]] = {
    "blackwell": {
        "python": "3.11",
        "jax": "0.7.1",
        "flax": "0.12.0",
        "optax": "0.2.8",
        "octo_revision": OCTO_PATCHED_REVISION,
        "octo_source": f"upstream {OCTO_UPSTREAM_REVISION[:7]} + third_party/octo patches",
        "requirements": "envs/requirements/research-blackwell.txt",
        "jax_install_path": "pip install 'jax[cuda12]==0.7.1' (NVIDIA CUDA 12 wheels)",
        "hardware_status": "unverified on Blackwell for the corrected pipeline",
    },
    "legacy-gpu": {
        "python": "3.10",
        "jax": "0.4.20",
        "flax": "0.7.5",
        "optax": "0.1.5",
        "octo_revision": OCTO_UPSTREAM_REVISION,
        "octo_source": f"unmodified upstream {OCTO_UPSTREAM_REVISION[:7]}",
        "requirements": "envs/requirements/research-legacy-gpu.txt",
        "jax_install_path": (
            "pip install 'jax[cuda11_pip]==0.4.20' "
            "-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        ),
        "hardware_status": "verified on an RTX 4050 Laptop GPU (sm_89, CUDA 11 wheels)",
    },
}


class UnsupportedEnvironmentError(RuntimeError):
    """The interpreter does not match either locked research profile."""


def _installed(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def detect_environment_profile() -> dict[str, Any]:
    """Report which locked profile this interpreter matches, and every mismatch.

    Never raises: preflight records the report, and research commands call
    :func:`require_environment_profile` to fail closed.
    """

    observed = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "jax": _installed("jax"),
        "flax": _installed("flax"),
        "optax": _installed("optax"),
    }
    octo_revision = None
    octo_error = None
    try:
        from ..experiments.provenance import installed_vcs_commit

        octo_revision = installed_vcs_commit("octo")
    except RuntimeError as exc:
        octo_error = str(exc)
    observed["octo_revision"] = octo_revision
    candidates = {}
    for name, profile in ENVIRONMENT_PROFILES.items():
        candidates[name] = [
            f"{key}: installed {observed[key]!r}, locked {profile[key]!r}"
            for key in ("python", "jax", "flax", "optax", "octo_revision")
            if observed[key] != profile[key]
        ]
    matched = [name for name, problems in candidates.items() if not problems]
    return {
        "profile": matched[0] if matched else None,
        "observed": observed,
        "octo_error": octo_error,
        "mismatches": candidates,
        "hardware_status": (
            ENVIRONMENT_PROFILES[matched[0]]["hardware_status"] if matched else None
        ),
    }


def require_environment_profile() -> str:
    """Return the active locked profile name or raise with every mismatch."""

    report = detect_environment_profile()
    if report["profile"] is None:
        details = "; ".join(
            f"[{name}] " + ", ".join(problems) for name, problems in report["mismatches"].items()
        )
        raise UnsupportedEnvironmentError(
            "the interpreter matches neither locked research environment "
            f"({', '.join(ENVIRONMENT_PROFILES)}): {details}. Install one with "
            "envs/scripts/setup_research.sh --profile <name>."
        )
    return str(report["profile"])


@contextmanager
def true_float32_matmul() -> Iterator[None]:
    """Run matmuls in real float32 (GPUs otherwise use TF32 on Ampere and later).

    JAX 0.4 spells this precision ``float32``; newer releases also accept
    ``highest``. The equivalence gate needs it on every profile.
    """

    import jax

    for name in ("float32", "highest"):
        try:
            context = jax.default_matmul_precision(name)
            context.__enter__()
        except ValueError:
            continue
        try:
            yield
        finally:
            context.__exit__(None, None, None)
        return
    raise UnsupportedEnvironmentError("this JAX offers no float32 matmul precision setting")


def tree_structure(tree: Any) -> list[list[Any]]:
    """Sorted ``[path, shape, dtype]`` rows of an array tree (no values)."""

    rows: list[list[Any]] = []

    def walk(prefix: tuple[str, ...], value: Any) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                walk((*prefix, str(key)), value[key])
            return
        array = np.asarray(value)
        rows.append(["/".join(prefix), list(array.shape), str(array.dtype)])

    walk((), tree)
    return rows


def architecture_identity(pretrained_model: Any) -> dict[str, Any]:
    """Value-free description of the pretrained architecture."""

    config = pretrained_model.config
    return {
        "model": config["model"],
        "text_processor": config.get("text_processor"),
        "parameters": tree_structure(pretrained_model.params),
        "example_batch": tree_structure(pretrained_model.example_batch),
    }


def architecture_sha256(pretrained_model: Any) -> str:
    return canonical_sha256(architecture_identity(pretrained_model))
