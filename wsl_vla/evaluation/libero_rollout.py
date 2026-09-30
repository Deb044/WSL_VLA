"""Deterministic official-LIBERO rollouts for Octo research policies.

The simulator imports are deliberately lazy: lightweight contract tests do not
need MuJoCo, while publication commands fail closed when the official LIBERO
package or benchmark assets are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ..data.octo_batches import _resize_rgb
from ..experiments.provenance import sha256_array_tree


def canonical_instruction(value: str) -> str:
    """Normalize harmless presentation differences, not semantic content."""

    return re.sub(r"\s+", " ", value.strip().rstrip(".")).casefold()


def fixed_initialization_indices(
    available: int, rollout_count: int, *, seed: int
) -> tuple[int, ...]:
    """Select a reproducible, non-repeating subset of official init states."""

    if available <= 0:
        raise ValueError("the benchmark task has no initialization states")
    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")
    if rollout_count > available:
        raise ValueError(
            f"requested {rollout_count} rollouts but only {available} fixed init states exist"
        )
    permutation = np.random.default_rng(seed).permutation(available)
    return tuple(int(index) for index in permutation[:rollout_count])


@dataclass(frozen=True)
class EpisodeRollout:
    task_id: str
    task_index: int
    instruction: str
    initialization_index: int
    episode_seed: int
    steps: int
    success: bool
    episode_return: float


@dataclass(frozen=True)
class TaskRolloutSummary:
    suite: str
    task_id: str
    task_index: int
    policy_checkpoint_sha256: str
    episodes: tuple[EpisodeRollout, ...]

    @property
    def rollout_count(self) -> int:
        return len(self.episodes)

    @property
    def successes(self) -> int:
        return sum(episode.success for episode in self.episodes)

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            raise ValueError("a rollout summary cannot be empty")
        return self.successes / self.rollout_count

    @property
    def initialization_indices(self) -> tuple[int, ...]:
        return tuple(episode.initialization_index for episode in self.episodes)


class RolloutPolicy(Protocol):
    checkpoint_sha256: str

    def reset(self, *, instruction: str, seed: int) -> None: ...

    def action(self, observation: Mapping[str, Any]) -> np.ndarray: ...


def validate_benchmark_task_order(
    benchmark_suite: Any, expected_instructions: Sequence[str]
) -> None:
    """Require the installed LIBERO benchmark to match the locked protocol."""

    if len(expected_instructions) != 10:
        raise ValueError("a reference LIBERO suite must contain exactly ten tasks")
    actual = [str(benchmark_suite.get_task(index).language) for index in range(10)]
    for index, (observed, expected) in enumerate(zip(actual, expected_instructions)):
        if canonical_instruction(observed) != canonical_instruction(expected):
            raise ValueError(
                f"LIBERO task order mismatch at index {index}: "
                f"installed={observed!r}, protocol={expected!r}"
            )


class LiberoRolloutEvaluator:
    """Evaluate one policy state in one official LIBERO suite, one env at a time."""

    def __init__(
        self,
        *,
        suite: str,
        expected_instructions: Sequence[str],
        camera_size: tuple[int, int] = (256, 256),
        warmup_steps: int = 5,
        max_steps: int = 600,
        benchmark_suite: Any | None = None,
        env_factory: Any | None = None,
        bddl_root: str | Path | None = None,
    ) -> None:
        if suite not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
            raise ValueError(f"unsupported reference suite: {suite}")
        if min(camera_size) <= 0 or warmup_steps < 0 or max_steps <= 0:
            raise ValueError("invalid LIBERO rollout dimensions or horizons")
        if benchmark_suite is None or env_factory is None or bddl_root is None:
            try:
                from libero.libero import benchmark, get_libero_path
                from libero.libero.envs import OffScreenRenderEnv
            except ImportError as exc:  # pragma: no cover - WSL research environment
                raise RuntimeError(
                    "official LIBERO is required for rollout evaluation"
                ) from exc
            benchmark_suite = benchmark.get_benchmark_dict()[suite]()
            env_factory = OffScreenRenderEnv
            bddl_root = get_libero_path("bddl_files")
        self.suite = suite
        self.expected_instructions = tuple(expected_instructions)
        self.camera_size = tuple(int(value) for value in camera_size)
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.benchmark_suite = benchmark_suite
        self.env_factory = env_factory
        self.bddl_root = Path(bddl_root)
        validate_benchmark_task_order(benchmark_suite, self.expected_instructions)

    def evaluate_task(
        self,
        policy: RolloutPolicy,
        *,
        task_index: int,
        rollout_count: int,
        seed: int,
    ) -> TaskRolloutSummary:
        if not 0 <= task_index < len(self.expected_instructions):
            raise IndexError(f"task_index {task_index} is outside the reference suite")
        checkpoint_sha256 = str(policy.checkpoint_sha256)
        if len(checkpoint_sha256) != 64:
            raise ValueError("policy checkpoint identity must be a SHA-256 digest")
        task = self.benchmark_suite.get_task(task_index)
        instruction = self.expected_instructions[task_index]
        init_states = np.asarray(self.benchmark_suite.get_task_init_states(task_index))
        indices = fixed_initialization_indices(len(init_states), rollout_count, seed=seed)
        bddl_file = self.bddl_root / task.problem_folder / task.bddl_file
        if not bddl_file.is_file():
            raise FileNotFoundError(f"LIBERO BDDL task file is missing: {bddl_file}")

        env = self.env_factory(
            bddl_file_name=os.fspath(bddl_file),
            camera_heights=self.camera_size[0],
            camera_widths=self.camera_size[1],
        )
        episodes: list[EpisodeRollout] = []
        zero_action = np.zeros(7, dtype=np.float32)
        try:
            for ordinal, initialization_index in enumerate(indices):
                episode_seed = int(seed * 100_000 + task_index * 1_000 + ordinal)
                env.seed(episode_seed)
                env.reset()
                observation = env.set_init_state(init_states[initialization_index])
                for _ in range(self.warmup_steps):
                    observation, _, done, _ = env.step(zero_action)
                    if done:
                        raise RuntimeError("LIBERO episode terminated during physics warmup")
                policy.reset(instruction=instruction, seed=episode_seed)
                episode_return = 0.0
                success = False
                steps = 0
                for steps in range(1, self.max_steps + 1):
                    action = np.asarray(policy.action(observation), dtype=np.float32)
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError(
                            f"policy must produce one finite 7-D LIBERO action, got {action.shape}"
                        )
                    observation, reward, done, info = env.step(action)
                    episode_return += float(reward)
                    success = bool(
                        float(reward) > 0
                        or done
                        or (isinstance(info, Mapping) and info.get("success", False))
                    )
                    if success:
                        break
                episodes.append(
                    EpisodeRollout(
                        task_id=f"{self.suite}_{task_index}",
                        task_index=task_index,
                        instruction=instruction,
                        initialization_index=initialization_index,
                        episode_seed=episode_seed,
                        steps=steps,
                        success=success,
                        episode_return=episode_return,
                    )
                )
        finally:
            env.close()
        return TaskRolloutSummary(
            suite=self.suite,
            task_id=f"{self.suite}_{task_index}",
            task_index=task_index,
            policy_checkpoint_sha256=checkpoint_sha256,
            episodes=tuple(episodes),
        )


def build_octo_observation(
    history: Sequence[Mapping[str, Any]],
    example_observation: Mapping[str, Any],
    *,
    proprio_keys: Sequence[str] = (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ),
) -> dict[str, np.ndarray]:
    """Convert raw LIBERO observations to one left-padded Octo batch."""

    if not history:
        raise ValueError("observation history cannot be empty")
    if "timestep_pad_mask" not in example_observation:
        raise ValueError("Octo checkpoint has no timestep_pad_mask")
    window = int(np.asarray(example_observation["timestep_pad_mask"]).shape[1])
    selected = history[-window:]
    start = window - len(selected)
    result: dict[str, np.ndarray] = {}
    raw_image_keys = {
        "image_primary": "agentview_image",
        "image_wrist": "robot0_eye_in_hand_image",
    }
    for key, expected_value in example_observation.items():
        expected = np.asarray(expected_value)
        if key == "timestep_pad_mask":
            value = np.zeros((1, window), dtype=bool)
            value[0, start:] = True
            result[key] = value
            continue
        if key in raw_image_keys:
            raw_key = raw_image_keys[key]
            missing = [index for index, item in enumerate(selected) if raw_key not in item]
            if missing:
                raise ValueError(f"LIBERO observations lack {raw_key} at history entries {missing}")
            image_size = (int(expected.shape[-3]), int(expected.shape[-2]))
            resized = _resize_rgb(
                np.stack([np.asarray(item[raw_key]) for item in selected]), image_size
            )
            value = np.zeros((1, window, *expected.shape[2:]), dtype=expected.dtype)
            value[0, start:] = resized.astype(expected.dtype, copy=False)
            result[key] = value
            continue
        if key == "proprio":
            rows = []
            for item in selected:
                missing = set(proprio_keys) - set(item)
                if missing:
                    raise ValueError(f"LIBERO observation lacks proprio keys {sorted(missing)}")
                rows.append(
                    np.concatenate([np.asarray(item[name]).reshape(-1) for name in proprio_keys])
                )
            stacked = np.stack(rows).astype(expected.dtype, copy=False)
            if stacked.shape[1:] != expected.shape[2:]:
                raise ValueError(
                    f"LIBERO proprio shape {stacked.shape[1:]} differs from Octo {expected.shape[2:]}"
                )
            value = np.zeros((1, window, *expected.shape[2:]), dtype=expected.dtype)
            value[0, start:] = stacked
            result[key] = value
            continue
        raise ValueError(f"unsupported checkpoint observation key for LIBERO: {key}")
    return result


class OctoLiberoPolicy:
    """Official Octo sampling wrapper for one immutable adapter state."""

    def __init__(
        self,
        bundle: Any,
        adapter_state: Mapping[str, Any],
        *,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        argmax: bool = False,
        temperature: float = 1.0,
    ) -> None:
        try:
            import jax
        except ImportError as exc:  # pragma: no cover - WSL research environment
            raise RuntimeError("official Octo rollout requires JAX") from exc
        from ..octo.training import materialize_policy_params

        mean = np.asarray(action_mean, dtype=np.float32)
        std = np.asarray(action_std, dtype=np.float32)
        if mean.shape != (7,) or std.shape != (7,) or np.any(std <= 0):
            raise ValueError("LIBERO action statistics must be positive 7-D mean/std arrays")
        params = materialize_policy_params(bundle, adapter_state)
        self.model = bundle.research_model.replace(params=params)
        self.checkpoint_sha256 = sha256_array_tree(params)
        self.unnormalization_statistics = {"mean": mean, "std": std}
        self.argmax = bool(argmax)
        self.temperature = float(temperature)
        self._jax = jax
        self._history: list[Mapping[str, Any]] = []
        self._tasks = None
        self._rng = None

    def reset(self, *, instruction: str, seed: int) -> None:
        if not instruction.strip():
            raise ValueError("LIBERO instruction cannot be empty")
        self._history = []
        self._tasks = self.model.create_tasks(texts=[instruction])
        self._rng = self._jax.random.PRNGKey(seed)

    def action(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self._tasks is None or self._rng is None:
            raise RuntimeError("reset must be called before Octo action sampling")
        self._history.append(observation)
        inputs = build_octo_observation(
            self._history, self.model.example_batch["observation"]
        )
        self._rng, sample_rng = self._jax.random.split(self._rng)
        actions = self.model.sample_actions(
            inputs,
            self._tasks,
            unnormalization_statistics=self.unnormalization_statistics,
            rng=sample_rng,
            argmax=self.argmax,
            temperature=self.temperature,
        )
        values = np.asarray(self._jax.device_get(actions))
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[-1] < 7:
            raise ValueError(f"Octo returned invalid action tensor {values.shape}")
        return values[0, 0, :7].astype(np.float32, copy=False)
