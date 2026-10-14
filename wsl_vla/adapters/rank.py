"""Rank-matched deployment of decoded effective updates.

A zoo adapter entry has rank at most ``r``. A decoded update is dense, so
deploying it unchanged would give generated policies more capacity than the
adapters they are compared with. Primary comparisons therefore project every
decoded update back to its entry's maximum rank with a truncated SVD and deploy
the resulting reloadable factors; the unrestricted update stays a diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..contracts import COMPONENT_NAMES, AdapterSpec


def relative_error(error_squared: float, reference_squared: float) -> float | None:
    """Root relative squared error, or ``None`` when the reference is zero.

    A zero denominator has no normalized error; returning 0 or infinity would
    manufacture a score.
    """

    if reference_squared <= 0.0:
        return None
    return float(np.sqrt(error_squared / reference_squared))


@dataclass(frozen=True)
class RankProjection:
    """Truncated-SVD factors and the error they introduce."""

    factors: Mapping[str, tuple[np.ndarray, np.ndarray]]
    updates: Mapping[str, np.ndarray]
    entry_relative_error: Mapping[str, float | None]
    component_relative_error: Mapping[str, float | None]
    relative_error: float | None

    def report(self) -> dict[str, Any]:
        return {
            "relative_error": self.relative_error,
            "component_relative_error": dict(self.component_relative_error),
            "max_entry_relative_error": max(
                (value for value in self.entry_relative_error.values() if value is not None),
                default=None,
            ),
            "undefined_entries": sorted(
                path for path, value in self.entry_relative_error.items() if value is None
            ),
        }


def _blas_limit(threads: int = 4):
    """Cap BLAS threads for the many small factorizations below.

    On many-core hosts the default thread count makes small LAPACK calls several
    times slower (thread contention); ``threadpoolctl`` is optional.
    """

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - optional dependency
        from contextlib import nullcontext

        return nullcontext()
    return threadpool_limits(limits=threads)


def top_singular_triplets(dense: np.ndarray, keep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leading ``keep`` singular triplets ``(U, s, Vt)`` of a dense matrix.

    Small problems use a full SVD. Larger ones find the leading right (or left)
    singular subspace from the smaller Gram matrix and then take an exact thin
    SVD of the matrix restricted to that subspace, which restores the accuracy
    lost by squaring and is far cheaper than a full SVD when ``keep`` is small.
    """

    rows, cols = dense.shape
    small = min(rows, cols)
    if keep >= small or small <= 4 * keep:
        left, singular, right = np.linalg.svd(dense, full_matrices=False)
        return left[:, :keep], singular[:keep], right[:keep]
    from scipy.linalg import eigh

    if cols <= rows:
        _, basis = eigh(dense.T @ dense, subset_by_index=[cols - keep, cols - 1])
        left, singular, mixing = np.linalg.svd(dense @ basis, full_matrices=False)
        return left, singular, (basis @ mixing.T).T
    _, basis = eigh(dense @ dense.T, subset_by_index=[rows - keep, rows - 1])
    mixing, singular, right = np.linalg.svd(basis.T @ dense, full_matrices=False)
    return basis @ mixing, singular, right


def project_updates_to_rank(
    spec: AdapterSpec, updates: Mapping[str, Any]
) -> RankProjection:
    """Project each dense update onto its entry's maximum rank.

    Returns ``(down, up)`` factors with the zoo's shapes ``[in, rank]`` and
    ``[rank, out]`` such that ``(alpha / rank) * down @ up`` is the best
    rank-``r`` approximation in Frobenius norm. Entries whose kernel is smaller
    than the nominal rank are padded with zero directions.
    """

    if {entry.parameter_path for entry in spec.entries} != set(updates):
        raise ValueError("update paths differ from AdapterSpec")
    factors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    projected: dict[str, np.ndarray] = {}
    entry_error: dict[str, float | None] = {}
    component_error = {name: 0.0 for name in COMPONENT_NAMES}
    component_reference = {name: 0.0 for name in COMPONENT_NAMES}
    for entry in spec.entries:
        dense = np.asarray(updates[entry.parameter_path], dtype=np.float64)
        if dense.shape != (entry.input_dim, entry.output_dim):
            raise ValueError(f"bad update shape for {entry.parameter_path}: {dense.shape}")
        if not np.isfinite(dense).all():
            raise ValueError(f"decoded update is non-finite at {entry.parameter_path}")
        keep = entry.effective_rank_bound
        with _blas_limit():
            left, singular, right = top_singular_triplets(dense, keep)
        root = np.sqrt(singular[:keep] * entry.rank / spec.alpha)
        down = np.zeros((entry.input_dim, entry.rank), dtype=np.float64)
        up = np.zeros((entry.rank, entry.output_dim), dtype=np.float64)
        down[:, :keep] = left[:, :keep] * root
        up[:keep] = root[:, None] * right[:keep]
        approximation = (spec.alpha / entry.rank) * (down @ up)
        error = float(np.square(dense - approximation).sum())
        reference = float(np.square(dense).sum())
        factors[entry.parameter_path] = (down.astype(np.float32), up.astype(np.float32))
        projected[entry.parameter_path] = approximation.astype(np.float32)
        entry_error[entry.parameter_path] = relative_error(error, reference)
        component_error[entry.component.value] += error
        component_reference[entry.component.value] += reference
    return RankProjection(
        factors=factors,
        updates=projected,
        entry_relative_error=entry_error,
        component_relative_error={
            name: relative_error(component_error[name], component_reference[name])
            for name in COMPONENT_NAMES
        },
        relative_error=relative_error(
            sum(component_error.values()), sum(component_reference.values())
        ),
    )


def update_rank(matrix: Any, *, tolerance: float = 1e-6) -> int:
    """Numerical rank relative to the largest singular value."""

    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if singular.size == 0 or singular[0] == 0.0:
        return 0
    return int(np.sum(singular > tolerance * singular[0]))
