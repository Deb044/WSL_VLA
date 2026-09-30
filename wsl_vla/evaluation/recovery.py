"""Consecutive-checkpoint recovery curves matching the reference VLA protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


DEFAULT_RECOVERY_FRACTIONS = (
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.60,
    0.80,
    1.00,
)


@dataclass(frozen=True)
class RecoveryCurvePoint:
    update_steps: int
    successes: int
    rollout_count: int

    @property
    def success_rate(self) -> float:
        if self.update_steps < 0 or self.rollout_count <= 0:
            raise ValueError("recovery curve step and rollout count are invalid")
        if not 0 <= self.successes <= self.rollout_count:
            raise ValueError("recovery successes lie outside the rollout count")
        return self.successes / self.rollout_count


@dataclass(frozen=True)
class RecoveryEfficiency:
    original_peak_success_rate: float
    original_training_steps: int
    recovered_at_steps: int | None
    recovery_step_ratio: float | None
    curve: tuple[RecoveryCurvePoint, ...]

    @property
    def recovered(self) -> bool:
        return self.recovered_at_steps is not None


def recovery_step_schedule(
    original_training_steps: int,
    fractions: Sequence[float] = DEFAULT_RECOVERY_FRACTIONS,
) -> tuple[int, ...]:
    """Create a fixed, monotonic evaluation grid ending at original training time."""

    if original_training_steps <= 0:
        raise ValueError("original training steps must be positive")
    values = tuple(float(value) for value in fractions)
    if not values or any(value <= 0 or value > 1 for value in values):
        raise ValueError("recovery fractions must lie in (0, 1]")
    if values != tuple(sorted(set(values))) or values[-1] != 1.0:
        raise ValueError("recovery fractions must be unique, increasing, and end at 1.0")
    return tuple(
        sorted(
            {
                min(original_training_steps, max(1, int(round(original_training_steps * value))))
                for value in values
            }
        )
    )


def recovery_efficiency(
    *,
    original_peak_success_rate: float,
    original_training_steps: int,
    curve: Sequence[RecoveryCurvePoint],
) -> RecoveryEfficiency:
    """Find the first measured point that regains the original peak success rate."""

    if not 0 <= original_peak_success_rate <= 1:
        raise ValueError("original peak success rate must lie in [0, 1]")
    if original_training_steps <= 0:
        raise ValueError("original training steps must be positive")
    points = tuple(curve)
    if not points or points[0].update_steps != 0:
        raise ValueError("recovery curve must include the post-forgetting step-zero point")
    steps = tuple(point.update_steps for point in points)
    if steps != tuple(sorted(set(steps))):
        raise ValueError("recovery curve steps must be unique and increasing")
    if steps[-1] > original_training_steps:
        raise ValueError("recovery curve exceeds the original training budget")
    for point in points:
        _ = point.success_rate
    recovered_at = next(
        (
            point.update_steps
            for point in points
            if point.success_rate >= original_peak_success_rate
        ),
        None,
    )
    return RecoveryEfficiency(
        original_peak_success_rate=float(original_peak_success_rate),
        original_training_steps=int(original_training_steps),
        recovered_at_steps=recovered_at,
        recovery_step_ratio=(
            None if recovered_at is None else recovered_at / original_training_steps
        ),
        curve=points,
    )
