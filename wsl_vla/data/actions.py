"""The single LIBERO controller-action convention used by the whole zoo.

Policies are trained and sampled directly in controller coordinates (identity
normalization). Training targets are *validated* against the controller bounds
and never clipped; clipping happens only at deployment, where its frequency is
measured and reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ..contracts import LIBERO_ACTION_CONVENTION, ActionConvention


BOUND_TOLERANCE = 1e-6


class ControllerActionError(ValueError):
    """Demonstration or policy actions violate the fixed controller convention."""


@dataclass(frozen=True)
class ActionNormalization:
    """Identity policy normalization in controller coordinates.

    The class keeps the ``mean``/``std``/``count`` shape consumed by Octo's
    sampler and by run manifests, but it can no longer be fitted per task:
    per-task statistics put every zoo adapter in a different output coordinate
    system and pushed small-variance dimensions past the diffusion head's
    internal clip.
    """

    mean: np.ndarray
    std: np.ndarray
    count: int
    convention: ActionConvention = LIBERO_ACTION_CONVENTION

    def __post_init__(self) -> None:
        dim = self.convention.action_dim
        if np.shape(self.mean) != (dim,) or np.shape(self.std) != (dim,):
            raise ControllerActionError("action normalization must match the controller")
        if np.any(np.asarray(self.mean) != 0) or np.any(np.asarray(self.std) != 1):
            raise ControllerActionError(
                "the zoo protocol fixes identity action normalization (mean 0, std 1)"
            )

    @classmethod
    def identity(
        cls, convention: ActionConvention = LIBERO_ACTION_CONVENTION, *, count: int = 0
    ) -> "ActionNormalization":
        dim = convention.action_dim
        return cls(
            np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32), int(count), convention
        )

    def to_dict(self) -> dict:
        return {
            "kind": "identity",
            "convention": self.convention.name,
            "convention_sha256": self.convention.sha256(),
            "mean": np.asarray(self.mean).tolist(),
            "std": np.asarray(self.std).tolist(),
            "count": int(self.count),
        }

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ControllerActionError("actions contain non-finite values")
        return values

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        return np.asarray(actions, dtype=np.float32)


def validate_controller_actions(
    actions: Any,
    convention: ActionConvention = LIBERO_ACTION_CONVENTION,
    *,
    source: str = "actions",
) -> dict[str, Any]:
    """Check demonstration commands against the controller without altering them.

    Raises :class:`ControllerActionError` for a wrong dimension count, a
    non-finite value, a command outside the controller bounds, or a gripper
    command that is neither the open nor the close value. Returns the observed
    range so the caller can record it.
    """

    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != convention.action_dim:
        raise ControllerActionError(
            f"{source} must have shape [time, {convention.action_dim}], got {values.shape}"
        )
    if values.shape[0] == 0:
        raise ControllerActionError(f"{source} contains no actions")
    if not np.isfinite(values).all():
        raise ControllerActionError(f"{source} contains non-finite commands")
    low = np.asarray(convention.low)
    high = np.asarray(convention.high)
    below = values < low - BOUND_TOLERANCE
    above = values > high + BOUND_TOLERANCE
    if below.any() or above.any():
        dims = sorted(
            convention.dimension_names[index]
            for index in np.flatnonzero((below | above).any(axis=0))
        )
        raise ControllerActionError(
            f"{source} leaves the {convention.controller} bounds in dimensions {dims} "
            f"(observed range [{values.min():.6g}, {values.max():.6g}]). Training targets "
            "are never clipped silently; fix the data or change the declared convention."
        )
    gripper = values[:, convention.gripper_dimension]
    levels = np.asarray([convention.gripper_open, convention.gripper_close])
    if np.any(np.min(np.abs(gripper[:, None] - levels[None]), axis=1) > BOUND_TOLERANCE):
        raise ControllerActionError(
            f"{source} uses gripper commands outside the convention "
            f"{{open={convention.gripper_open}, close={convention.gripper_close}}}: "
            f"{np.unique(np.round(gripper, 6)).tolist()[:8]}"
        )
    return {
        "count": int(values.shape[0]),
        "minimum": values.min(axis=0).tolist(),
        "maximum": values.max(axis=0).tolist(),
        "gripper_close_fraction": float(
            np.mean(np.abs(gripper - convention.gripper_close) <= BOUND_TOLERANCE)
        ),
    }


def controller_action_normalization(
    episodes: Iterable[Any],
    convention: ActionConvention = LIBERO_ACTION_CONVENTION,
) -> ActionNormalization:
    """Validate every training episode and return the fixed identity transform."""

    count = 0
    seen = False
    for episode in episodes:
        seen = True
        report = validate_controller_actions(
            episode.actions, convention, source=f"episode {episode.episode_id}"
        )
        count += report["count"]
    if not seen:
        raise ControllerActionError("no training episodes were supplied")
    return ActionNormalization.identity(convention, count=count)


def assert_head_preserves_targets(
    convention: ActionConvention, *, head_max_action: float
) -> None:
    """Octo's diffusion loss clips targets to ``+-max_action``; prove it is inactive."""

    extreme = max(max(abs(value) for value in convention.low), max(abs(v) for v in convention.high))
    if extreme > head_max_action:
        raise ControllerActionError(
            f"controller bound {extreme} exceeds the action head's internal clip "
            f"{head_max_action}; training targets would be clipped silently"
        )


@dataclass
class ClippingMonitor:
    """Deployment-time controller clipping with per-dimension frequency."""

    convention: ActionConvention = LIBERO_ACTION_CONVENTION
    steps: int = 0
    clipped_steps: int = 0
    clipped_per_dimension: np.ndarray = field(default=None)  # type: ignore[assignment]
    max_excess: float = 0.0

    def __post_init__(self) -> None:
        if self.clipped_per_dimension is None:
            self.clipped_per_dimension = np.zeros(self.convention.action_dim, dtype=np.int64)

    def clip(self, action: Any) -> np.ndarray:
        raw = np.asarray(action, dtype=np.float32)
        if raw.shape != (self.convention.action_dim,) or not np.isfinite(raw).all():
            raise ControllerActionError(
                f"policy must produce one finite {self.convention.action_dim}-D action, "
                f"got shape {raw.shape}"
            )
        low = np.asarray(self.convention.low, dtype=np.float32)
        high = np.asarray(self.convention.high, dtype=np.float32)
        clipped = np.clip(raw, low, high)
        changed = np.abs(clipped - raw) > BOUND_TOLERANCE
        self.steps += 1
        self.clipped_steps += int(changed.any())
        self.clipped_per_dimension += changed
        self.max_excess = max(self.max_excess, float(np.max(np.abs(clipped - raw))))
        return clipped

    def report(self) -> dict[str, Any]:
        steps = max(self.steps, 1)
        return {
            "steps": int(self.steps),
            "clipped_steps": int(self.clipped_steps),
            "clipped_step_fraction": self.clipped_steps / steps if self.steps else None,
            "clipped_fraction_per_dimension": {
                name: (int(count) / steps if self.steps else None)
                for name, count in zip(
                    self.convention.dimension_names, self.clipped_per_dimension
                )
            },
            "max_excess": float(self.max_excess),
        }
