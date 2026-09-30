import numpy as np
import pytest

from wsl_vla.contracts import AdapterEntry, AdapterSpec, Component
from wsl_vla.evaluation.components import component_drift, swap_effective_updates


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
