"""Consecutive-checkpoint component swaps and drift measurements."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from core.contracts import AdapterSpec, Component


def validate_consecutive_stages(previous_stage: int, current_stage: int) -> None:
    if previous_stage < 0 or current_stage != previous_stage + 1:
        raise ValueError("component swaps are valid only for consecutive checkpoints")


def swap_effective_updates(
    previous: Mapping[str, np.ndarray],
    current: Mapping[str, np.ndarray],
    spec: AdapterSpec,
    *,
    components: Sequence[Component | str],
    previous_stage: int,
    current_stage: int,
) -> dict[str, np.ndarray]:
    """Insert selected previous-stage components into the current policy."""

    validate_consecutive_stages(previous_stage, current_stage)
    expected = {entry.parameter_path for entry in spec.entries}
    if set(previous) != expected or set(current) != expected:
        raise ValueError("checkpoint update paths differ from AdapterSpec")
    selected = {Component(item) for item in components}
    if not selected:
        raise ValueError("at least one component must be selected")
    by_path = {entry.parameter_path: entry.component for entry in spec.entries}
    return {
        path: np.asarray(previous[path]).copy()
        if by_path[path] in selected
        else np.asarray(current[path]).copy()
        for path in sorted(expected)
    }


def component_drift(
    previous: Mapping[str, np.ndarray],
    current: Mapping[str, np.ndarray],
    spec: AdapterSpec,
) -> dict[str, float]:
    """Root-sum-square effective-update drift for each modality."""

    expected = {entry.parameter_path for entry in spec.entries}
    if set(previous) != expected or set(current) != expected:
        raise ValueError("checkpoint update paths differ from AdapterSpec")
    totals = {component: 0.0 for component in Component}
    for entry in spec.entries:
        delta = np.asarray(current[entry.parameter_path]) - np.asarray(
            previous[entry.parameter_path]
        )
        if not np.isfinite(delta).all():
            raise ValueError("checkpoint drift contains non-finite values")
        totals[entry.component] += float(np.sum(np.square(delta)))
    return {component.value: float(np.sqrt(value)) for component, value in totals.items()}
