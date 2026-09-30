"""Create publication artifacts only from immutable evaluation records."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def plot_success_matrix(
    matrix: np.ndarray,
    path: str | Path,
    *,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - research extra
        raise RuntimeError("matplotlib is required for publication figures") from exc
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("success matrix figure requires a square matrix")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis")
    axis.set_title(title)
    axis.set_xlabel("Evaluated task")
    axis.set_ylabel("Training stage")
    axis.set_xticks(range(values.shape[1]))
    axis.set_yticks(range(values.shape[0]))
    figure.colorbar(image, ax=axis, label="Success rate")
    figure.savefig(target, dpi=300, metadata={"Creator": "WSL_VLA record reporter"})
    plt.close(figure)


def plot_metric_intervals(
    estimates: Mapping[str, object],
    path: str | Path,
    *,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - research extra
        raise RuntimeError("matplotlib is required for publication figures") from exc
    names = tuple(estimates)
    centers = np.asarray([estimates[name].estimate for name in names])
    lower = centers - np.asarray([estimates[name].ci_low for name in names])
    upper = np.asarray([estimates[name].ci_high for name in names]) - centers
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    axis.errorbar(range(len(names)), centers, yerr=np.stack([lower, upper]), fmt="o", capsize=4)
    axis.set_xticks(range(len(names)), names, rotation=25, ha="right")
    axis.set_title(title)
    axis.set_ylabel("Estimate with 95% bootstrap CI")
    axis.axhline(0, color="black", linewidth=0.8)
    figure.savefig(target, dpi=300, metadata={"Creator": "WSL_VLA record reporter"})
    plt.close(figure)
