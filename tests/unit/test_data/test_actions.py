import numpy as np
import pytest

from tests.fakes import controller_actions, episodes
from wsl_vla.contracts import LIBERO_ACTION_CONVENTION, ActionConvention
from wsl_vla.data.actions import (
    ActionNormalization,
    ClippingMonitor,
    ControllerActionError,
    assert_head_preserves_targets,
    controller_action_normalization,
    validate_controller_actions,
)


def test_convention_is_identity_in_controller_coordinates_and_hashable():
    convention = LIBERO_ACTION_CONVENTION
    assert convention.action_dim == 7 and convention.policy_normalization == "identity"
    assert convention.low == (-1.0,) * 7 and convention.high == (1.0,) * 7
    assert (convention.gripper_open, convention.gripper_close) == (-1.0, 1.0)
    assert ActionConvention.from_dict(convention.to_dict()) == convention
    changed = ActionConvention.from_dict({**convention.to_dict(), "gripper_open": 0.0})
    assert changed.sha256() != convention.sha256()


def test_identity_normalization_cannot_be_refitted_per_task():
    rng = np.random.default_rng(0)
    actions = controller_actions(rng, 9)
    normalization = controller_action_normalization(episodes(2))
    np.testing.assert_array_equal(normalization.normalize(actions), actions)
    np.testing.assert_array_equal(normalization.denormalize(actions), actions)
    assert normalization.to_dict()["kind"] == "identity"
    with pytest.raises(ControllerActionError, match="identity action normalization"):
        ActionNormalization(np.full(7, 0.1, np.float32), np.ones(7, np.float32), 1)
    assert not hasattr(ActionNormalization, "fit")


def test_demonstration_commands_are_validated_without_being_altered():
    rng = np.random.default_rng(1)
    actions = controller_actions(rng, 12)
    original = actions.copy()
    report = validate_controller_actions(actions)
    np.testing.assert_array_equal(actions, original)
    assert report["count"] == 12 and max(report["maximum"]) <= 1.0
    too_large = actions.copy()
    too_large[3, 2] = -1.2
    with pytest.raises(ControllerActionError, match="dz"):
        validate_controller_actions(too_large)
    wrong_gripper = actions.copy()
    wrong_gripper[:, 6] = 0.0  # a {0, 1} gripper convention is not this controller's
    with pytest.raises(ControllerActionError, match="gripper commands outside the convention"):
        validate_controller_actions(wrong_gripper)
    with pytest.raises(ControllerActionError, match="shape"):
        validate_controller_actions(actions[:, :6])
    non_finite = actions.copy()
    non_finite[0, 0] = np.nan
    with pytest.raises(ControllerActionError, match="non-finite"):
        validate_controller_actions(non_finite)


def test_head_clip_is_proven_inactive_for_the_controller_bounds():
    assert_head_preserves_targets(LIBERO_ACTION_CONVENTION, head_max_action=5.0)
    with pytest.raises(ControllerActionError, match="clipped silently"):
        assert_head_preserves_targets(LIBERO_ACTION_CONVENTION, head_max_action=0.5)


def test_deployment_clipping_is_applied_and_its_frequency_reported():
    monitor = ClippingMonitor()
    inside = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    outside = np.array([1.5, -0.2, -2.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(monitor.clip(inside), inside)
    clipped = monitor.clip(outside)
    np.testing.assert_allclose(clipped, [1.0, -0.2, -1.0, 0.0, 0.0, 0.0, 1.0], rtol=1e-6)
    monitor.clip(inside)
    report = monitor.report()
    assert report["steps"] == 3 and report["clipped_steps"] == 1
    assert report["clipped_step_fraction"] == pytest.approx(1 / 3)
    assert report["clipped_fraction_per_dimension"]["dx"] == pytest.approx(1 / 3)
    assert report["clipped_fraction_per_dimension"]["dy"] == 0.0
    assert report["max_excess"] == pytest.approx(1.0)
    with pytest.raises(ControllerActionError):
        monitor.clip(np.zeros(6, dtype=np.float32))
