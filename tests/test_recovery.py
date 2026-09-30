import pytest

from wsl_vla.evaluation.recovery import (
    RecoveryCurvePoint,
    recovery_efficiency,
    recovery_step_schedule,
)


def test_recovery_schedule_is_locked_and_handles_short_training_runs():
    schedule = recovery_step_schedule(1000)
    assert schedule[:5] == (20, 40, 60, 80, 100)
    assert schedule[-1] == 1000
    assert recovery_step_schedule(3) == (1, 2, 3)


def test_recovery_efficiency_uses_first_point_regaining_original_peak():
    result = recovery_efficiency(
        original_peak_success_rate=0.75,
        original_training_steps=1000,
        curve=(
            RecoveryCurvePoint(0, 1, 4),
            RecoveryCurvePoint(20, 2, 4),
            RecoveryCurvePoint(40, 3, 4),
            RecoveryCurvePoint(60, 4, 4),
        ),
    )
    assert result.recovered
    assert result.recovered_at_steps == 40
    assert result.recovery_step_ratio == pytest.approx(0.04)


def test_recovery_efficiency_reports_unrecovered_without_fabricating_ratio():
    result = recovery_efficiency(
        original_peak_success_rate=1.0,
        original_training_steps=100,
        curve=(RecoveryCurvePoint(0, 0, 4), RecoveryCurvePoint(100, 3, 4)),
    )
    assert not result.recovered
    assert result.recovery_step_ratio is None
    with pytest.raises(ValueError, match="step-zero"):
        recovery_efficiency(
            original_peak_success_rate=1.0,
            original_training_steps=100,
            curve=(RecoveryCurvePoint(10, 1, 4),),
        )


def test_recovery_efficiency_excludes_tasks_never_learned_initially():
    result = recovery_efficiency(
        original_peak_success_rate=0.0,
        original_training_steps=100,
        curve=(RecoveryCurvePoint(0, 0, 4),),
    )
    assert not result.eligible
    assert not result.recovered
    assert result.recovery_step_ratio is None
