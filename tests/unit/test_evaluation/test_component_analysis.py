import numpy as np
import pytest

from wsl_vla.contracts import AdapterEntry, AdapterSpec, Component
from wsl_vla.evaluation.components import (
    SWAP_CONDITIONS,
    component_drift,
    run_consecutive_component_swaps,
    swap_effective_updates,
)
from wsl_vla.evaluation.sequential import RolloutOutcome


def _spec():
    return AdapterSpec(
        base_model_id="model",
        base_revision="revision",
        base_sha256="a" * 64,
        alpha=1.0,
        token_width=2,
        entries=tuple(
            AdapterEntry(component, index, component.value, 1, 2, 1)
            for index, component in enumerate(Component)
        ),
    )


def test_swaps_only_selected_components_between_consecutive_stages():
    spec = _spec()
    previous = {entry.parameter_path: np.zeros((1, 2)) for entry in spec.entries}
    current = {entry.parameter_path: np.ones((1, 2)) for entry in spec.entries}
    swapped = swap_effective_updates(
        previous,
        current,
        spec,
        components=("vision", "language"),
        previous_stage=2,
        current_stage=3,
    )
    assert np.all(swapped["vision"] == 0)
    assert np.all(swapped["language"] == 0)
    assert np.all(swapped["action"] == 1)
    assert component_drift(previous, current, spec)["action"] == pytest.approx(np.sqrt(2))


def test_rejects_arbitrary_cross_task_swap():
    spec = _spec()
    values = {entry.parameter_path: np.zeros((1, 2)) for entry in spec.entries}
    with pytest.raises(ValueError, match="consecutive"):
        swap_effective_updates(
            values,
            values,
            spec,
            components=("vision",),
            previous_stage=0,
            current_stage=2,
        )


def test_component_swap_protocol_uses_locked_swaps_and_same_initializations():
    spec = _spec()
    previous = {entry.parameter_path: np.zeros((1, 2)) for entry in spec.entries}
    current = {entry.parameter_path: np.ones((1, 2)) for entry in spec.entries}

    def materialize(updates):
        marker = float(sum(value.sum() for value in updates.values()))
        return marker, f"{int(marker):064x}"[-64:]

    def rollout(state, task, count, seed):
        return RolloutOutcome(
            successes=min(count, int(state) % (count + 1)),
            initialization_indices=tuple(range(count)),
            rollout_seeds=tuple(seed + index for index in range(count)),
        )

    records = run_consecutive_component_swaps(
        previous_updates=previous,
        current_updates=current,
        current_state=100.0,
        spec=spec,
        materialize_swapped_state=materialize,
        rollout=rollout,
        run_id="run",
        suite="suite",
        seed=17,
        condition="proposed_asymmetric",
        previous_stage=2,
        current_stage=3,
        evaluated_task_id="task_1",
        rollout_count=5,
        previous_checkpoint_sha256="a" * 64,
        current_checkpoint_sha256="b" * 64,
    )
    assert tuple(record.swap_condition for record in records) == tuple(SWAP_CONDITIONS)
    assert all(record.initialization_indices == tuple(range(5)) for record in records)
