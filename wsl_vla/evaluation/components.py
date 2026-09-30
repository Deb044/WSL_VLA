"""Consecutive-checkpoint component swaps and drift measurements."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..contracts import AdapterSpec, Component, ComponentSwapRecord
from .sequential import RolloutOutcome


SWAP_CONDITIONS = {
    "vision": (Component.VISION,),
    "language": (Component.LANGUAGE,),
    "action": (Component.ACTION,),
    "vision_language": (Component.VISION, Component.LANGUAGE),
    "full": (Component.VISION, Component.LANGUAGE, Component.ACTION),
}


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


def run_consecutive_component_swaps(
    *,
    previous_updates: Mapping[str, np.ndarray],
    current_updates: Mapping[str, np.ndarray],
    current_state: Any,
    spec: AdapterSpec,
    materialize_swapped_state: Callable[[Mapping[str, np.ndarray]], tuple[Any, str]],
    rollout: Callable[[Any, str, int, int], RolloutOutcome],
    run_id: str,
    suite: str,
    seed: int,
    condition: str,
    previous_stage: int,
    current_stage: int,
    evaluated_task_id: str,
    rollout_count: int,
    previous_checkpoint_sha256: str,
    current_checkpoint_sha256: str,
) -> tuple[ComponentSwapRecord, ...]:
    """Swap only the immediately preceding checkpoint on identical rollouts."""

    validate_consecutive_stages(previous_stage, current_stage)
    baseline = rollout(current_state, evaluated_task_id, rollout_count, seed)
    if not isinstance(baseline, RolloutOutcome):
        raise TypeError("component-swap rollouts require initialization provenance")
    drift = component_drift(previous_updates, current_updates, spec)
    records = []
    for swap_name, components in SWAP_CONDITIONS.items():
        swapped_updates = swap_effective_updates(
            previous_updates,
            current_updates,
            spec,
            components=components,
            previous_stage=previous_stage,
            current_stage=current_stage,
        )
        swapped_state, swapped_sha256 = materialize_swapped_state(swapped_updates)
        outcome = rollout(swapped_state, evaluated_task_id, rollout_count, seed)
        if not isinstance(outcome, RolloutOutcome):
            raise TypeError("component-swap rollouts require initialization provenance")
        if outcome.initialization_indices != baseline.initialization_indices:
            raise ValueError("component swaps must use baseline initialization identities")
        record = ComponentSwapRecord(
            run_id=run_id,
            suite=suite,
            seed=seed,
            condition=condition,
            transition_stage=current_stage,
            evaluated_task_id=evaluated_task_id,
            swap_condition=swap_name,
            rollout_count=rollout_count,
            baseline_successes=baseline.successes,
            swapped_successes=outcome.successes,
            initialization_indices=outcome.initialization_indices,
            previous_checkpoint_sha256=previous_checkpoint_sha256,
            current_checkpoint_sha256=current_checkpoint_sha256,
            swapped_checkpoint_sha256=swapped_sha256,
            latent_drift=drift,
        )
        _ = record.success_rate_drop
        records.append(record)
    return tuple(records)
