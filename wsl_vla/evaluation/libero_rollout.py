"""Deterministic official-LIBERO rollouts for Octo research policies.

The simulator imports are deliberately lazy: lightweight contract tests do not
need MuJoCo, while publication commands fail closed when the official LIBERO
package or benchmark assets are unavailable.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ..contracts import LIBERO_ACTION_CONVENTION, ActionConvention
from ..data.actions import ClippingMonitor
from ..data.preprocessing import PreprocessingSpec, camera_frames, observation_window, task_batch
from ..experiments.provenance import sha256_array_tree


@contextmanager
def robosuite_import_environment():
    """Hide a non-numeric CUDA_VISIBLE_DEVICES while robosuite is imported.

    At import, robosuite asserts that MUJOCO_EGL_DEVICE_ID is a substring of
    CUDA_VISIBLE_DEVICES. Slurm gives MIG slices a UUID there, and the EGL
    device is Mesa's software renderer, so the two are unrelated. The variable
    is restored afterwards because JAX needs it to find the GPU.
    """

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    hide = visible is not None and not visible.replace(",", "").isdigit()
    if hide:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        yield
    finally:
        if hide:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


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
    clipped_steps: int = 0
    clipped_step_fraction: float | None = None
    max_clip_excess: float = 0.0


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

    @property
    def clipping(self) -> dict[str, Any]:
        """Controller-bound clipping frequency over every executed step."""

        steps = sum(episode.steps for episode in self.episodes)
        clipped = sum(episode.clipped_steps for episode in self.episodes)
        return {
            "steps": steps,
            "clipped_steps": clipped,
            "clipped_step_fraction": clipped / steps if steps else None,
            "max_excess": max((episode.max_clip_excess for episode in self.episodes), default=0.0),
        }

    def as_outcome(self):
        """Convert to the strict shared-state runner contract without a cycle."""

        from .sequential import RolloutOutcome

        return RolloutOutcome(
            successes=self.successes,
            initialization_indices=self.initialization_indices,
            rollout_seeds=tuple(episode.episode_seed for episode in self.episodes),
            checkpoint_sha256=self.policy_checkpoint_sha256,
        )


class RolloutPolicy(Protocol):
    checkpoint_sha256: str

    def reset(self, *, instruction: str, seed: int) -> None: ...

    def action(self, observation: Mapping[str, Any]) -> np.ndarray: ...


def benchmark_task_indices(
    benchmark_suite: Any, expected_instructions: Sequence[str]
) -> tuple[int, ...]:
    """Map each protocol task to the installed LIBERO benchmark's task index.

    The protocol orders a suite's tasks by its own list (configs/reference_tasks.yaml),
    which differs from LIBERO's order for libero_object and libero_10. Tasks are
    matched by instruction, and every protocol task must match exactly one
    benchmark task.
    """

    if len(expected_instructions) != 10:
        raise ValueError("a reference LIBERO suite must contain exactly ten tasks")
    installed = [canonical_instruction(str(benchmark_suite.get_task(index).language)) for index in range(10)]
    indices = []
    for index, expected in enumerate(expected_instructions):
        matches = [position for position, value in enumerate(installed) if value == canonical_instruction(expected)]
        if len(matches) != 1:
            raise ValueError(
                f"protocol task {index} ({expected!r}) matches {len(matches)} installed LIBERO tasks"
            )
        indices.append(matches[0])
    return tuple(indices)


class LiberoRolloutEvaluator:
    """Evaluate policy states in one official LIBERO suite, one env at a time.

    Cameras render at the demonstration storage resolution named by the shared
    preprocessing; the policy applies the same orientation and resize steps that
    produced its training inputs. Every policy action is clipped to the
    controller bounds here, and the clipping frequency is recorded per episode.
    """

    def __init__(
        self,
        *,
        suite: str,
        expected_instructions: Sequence[str],
        preprocessing: PreprocessingSpec,
        convention: ActionConvention = LIBERO_ACTION_CONVENTION,
        warmup_steps: int = 5,
        max_steps: int = 600,
        benchmark_suite: Any | None = None,
        env_factory: Any | None = None,
        bddl_root: str | Path | None = None,
        simulator_image_convention: str | None = None,
    ) -> None:
        if suite not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
            raise ValueError(f"unsupported reference suite: {suite}")
        if warmup_steps < 0 or max_steps <= 0:
            raise ValueError("invalid LIBERO rollout horizons")
        if benchmark_suite is None or env_factory is None or bddl_root is None:
            try:
                with robosuite_import_environment():
                    import robosuite.macros as macros
                    from libero.libero import benchmark, get_libero_path
                    from libero.libero.envs import OffScreenRenderEnv
            except ImportError as exc:  # pragma: no cover - WSL research environment
                raise RuntimeError(
                    "official LIBERO is required for rollout evaluation"
                ) from exc
            benchmark_suite = benchmark.get_benchmark_dict()[suite]()
            env_factory = OffScreenRenderEnv
            bddl_root = get_libero_path("bddl_files")
            simulator_image_convention = str(macros.IMAGE_CONVENTION)
        if simulator_image_convention is None:
            raise ValueError(
                "an injected simulator must declare its image convention so rollout "
                "frames can be checked against demonstration generation"
            )
        # Rollout cameras must be oriented exactly like the demonstrations.
        preprocessing.assert_simulator_convention(simulator_image_convention)
        self.suite = suite
        self.expected_instructions = tuple(expected_instructions)
        self.preprocessing = preprocessing
        self.convention = convention
        self.camera_size = (int(preprocessing.source_height), int(preprocessing.source_width))
        self.simulator_image_convention = simulator_image_convention
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.benchmark_suite = benchmark_suite
        self.env_factory = env_factory
        self.bddl_root = Path(bddl_root)
        # Protocol task index -> the installed benchmark's index of the same task.
        self.benchmark_indices = benchmark_task_indices(benchmark_suite, self.expected_instructions)

    def evaluate_task(
        self,
        policy: RolloutPolicy,
        *,
        task_index: int,
        rollout_count: int,
        seed: int,
    ) -> TaskRolloutSummary:
        return self.evaluate_policies(
            {"policy": policy}, task_index=task_index, rollout_count=rollout_count, seed=seed
        )["policy"]

    def evaluate_policies(
        self,
        policies: Mapping[str, RolloutPolicy],
        *,
        task_index: int,
        rollout_count: int,
        seed: int,
    ) -> dict[str, TaskRolloutSummary]:
        """Paired evaluation: every policy sees the same init states and seeds.

        The initialization indices and episode seeds depend only on ``seed``,
        the task, and the rollout ordinal - never on the policy - so outcomes
        of different conditions are paired episode by episode.
        """

        if not policies:
            raise ValueError("at least one policy is required")
        if not 0 <= task_index < len(self.expected_instructions):
            raise IndexError(f"task_index {task_index} is outside the reference suite")
        identities = {name: str(policy.checkpoint_sha256) for name, policy in policies.items()}
        if any(len(value) != 64 for value in identities.values()):
            raise ValueError("policy checkpoint identity must be a SHA-256 digest")
        benchmark_index = self.benchmark_indices[task_index]
        task = self.benchmark_suite.get_task(benchmark_index)
        instruction = self.expected_instructions[task_index]
        init_states = np.asarray(self.benchmark_suite.get_task_init_states(benchmark_index))
        indices = fixed_initialization_indices(len(init_states), rollout_count, seed=seed)
        bddl_file = self.bddl_root / task.problem_folder / task.bddl_file
        if not bddl_file.is_file():
            raise FileNotFoundError(f"LIBERO BDDL task file is missing: {bddl_file}")

        env = self.env_factory(
            bddl_file_name=os.fspath(bddl_file),
            camera_heights=self.camera_size[0],
            camera_widths=self.camera_size[1],
        )
        episodes: dict[str, list[EpisodeRollout]] = {name: [] for name in policies}
        zero_action = np.zeros(self.convention.action_dim, dtype=np.float32)
        try:
            for ordinal, initialization_index in enumerate(indices):
                episode_seed = int(seed * 100_000 + task_index * 1_000 + ordinal)
                for name, policy in policies.items():
                    env.seed(episode_seed)
                    env.reset()
                    observation = env.set_init_state(init_states[initialization_index])
                    for _ in range(self.warmup_steps):
                        observation, _, done, _ = env.step(zero_action)
                        if done:
                            raise RuntimeError("LIBERO episode terminated during physics warmup")
                    policy.reset(instruction=instruction, seed=episode_seed)
                    monitor = ClippingMonitor(self.convention)
                    episode_return = 0.0
                    success = False
                    steps = 0
                    for steps in range(1, self.max_steps + 1):
                        action = monitor.clip(policy.action(observation))
                        observation, reward, done, info = env.step(action)
                        episode_return += float(reward)
                        success = bool(
                            float(reward) > 0
                            or done
                            or (isinstance(info, Mapping) and info.get("success", False))
                        )
                        if success:
                            break
                    clipping = monitor.report()
                    episodes[name].append(
                        EpisodeRollout(
                            task_id=f"{self.suite}_{task_index}",
                            task_index=task_index,
                            instruction=instruction,
                            initialization_index=initialization_index,
                            episode_seed=episode_seed,
                            steps=steps,
                            success=success,
                            episode_return=episode_return,
                            clipped_steps=clipping["clipped_steps"],
                            clipped_step_fraction=clipping["clipped_step_fraction"],
                            max_clip_excess=clipping["max_excess"],
                        )
                    )
        finally:
            env.close()
        return {
            name: TaskRolloutSummary(
                suite=self.suite,
                task_id=f"{self.suite}_{task_index}",
                task_index=task_index,
                policy_checkpoint_sha256=identities[name],
                episodes=tuple(episodes[name]),
            )
            for name in policies
        }


def build_octo_observation(
    history: Sequence[Mapping[str, Any]],
    spec: PreprocessingSpec,
    example_observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert raw LIBERO simulator observations to one left-padded Octo batch.

    This is :func:`wsl_vla.data.preprocessing.observation_window` applied to
    simulator frames; training batches call the same function on demonstration
    frames, so the two paths cannot drift apart.
    """

    if not history:
        raise ValueError("observation history cannot be empty")
    selected = history[-spec.window_size :]
    frames = {}
    for camera in spec.cameras:
        missing = [
            index for index, item in enumerate(selected) if camera.simulator_key not in item
        ]
        if missing:
            raise ValueError(
                f"LIBERO observations lack {camera.simulator_key} at history entries {missing}"
            )
        frames[camera.octo_key] = camera_frames(
            np.stack([np.asarray(item[camera.simulator_key]) for item in selected]),
            camera,
            spec,
            source="simulator",
        )
    window = observation_window(frames, spec, example_observation)
    return _add_batch_axis(window)


def _add_batch_axis(tree: Any) -> Any:
    if isinstance(tree, Mapping):
        return {key: _add_batch_axis(value) for key, value in tree.items()}
    return np.asarray(tree)[None]


class OctoLiberoPolicy:
    """Official Octo sampling wrapper for one immutable adapter state.

    Actions are sampled directly in controller coordinates (identity
    normalization). The sampling key is derived only from the episode seed, so
    two policies evaluated on the same episode consume identical noise.
    """

    def __init__(
        self,
        bundle: Any,
        adapter_state: Mapping[str, Any],
        *,
        argmax: bool = False,
        temperature: float = 1.0,
    ) -> None:
        try:
            import jax
        except ImportError as exc:  # pragma: no cover - WSL research environment
            raise RuntimeError("official Octo rollout requires JAX") from exc
        from ..octo.bridge import unnormalization_statistics
        from ..octo.training import materialize_policy_params

        params = materialize_policy_params(bundle, adapter_state)
        self.model = bundle.research_model.replace(params=params)
        self._bundle = bundle
        self.preprocessing = bundle.preprocessing
        self.convention = bundle.action_convention
        # Records and stage sidecars identify the small reloadable adapter, not
        # a second hash of the repeatedly materialized frozen base parameters.
        self.checkpoint_sha256 = sha256_array_tree(jax.device_get(adapter_state))
        self.unnormalization_statistics = unnormalization_statistics(self.convention)
        self.argmax = bool(argmax)
        self.temperature = float(temperature)
        self._jax = jax
        self._history: list[Mapping[str, Any]] = []
        self._pending: list[np.ndarray] = []
        self._tasks = None
        self._rng = None

    def reset(self, *, instruction: str, seed: int) -> None:
        if not instruction.strip():
            raise ValueError("LIBERO instruction cannot be empty")
        self._history = []
        self._pending = []
        self._tasks = task_batch(
            [instruction], self.model.text_processor, self.model.example_batch["task"]
        )
        self._rng = self._jax.random.PRNGKey(seed)

    def action(self, observation: Mapping[str, Any]) -> np.ndarray:
        from ..octo.bridge import activate_research_bundle

        if self._tasks is None or self._rng is None:
            raise RuntimeError("reset must be called before Octo action sampling")
        keys = [camera.simulator_key for camera in self.preprocessing.cameras]
        self._history.append({key: observation[key] for key in keys if key in observation})
        del self._history[: -self.preprocessing.window_size]
        if self._pending:
            return self._pending.pop(0)
        inputs = build_octo_observation(
            self._history, self.preprocessing, self.model.example_batch["observation"]
        )
        self._rng, sample_rng = self._jax.random.split(self._rng)
        activate_research_bundle(self._bundle)
        actions = self.model.sample_actions(
            inputs,
            self._tasks,
            unnormalization_statistics=self.unnormalization_statistics,
            rng=sample_rng,
            argmax=self.argmax,
            temperature=self.temperature,
        )
        values = np.asarray(self._jax.device_get(actions))
        dim = self.convention.action_dim
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[-1] < dim:
            raise ValueError(f"Octo returned invalid action tensor {values.shape}")
        plan = values[0, : self.preprocessing.executed_actions_per_plan, :dim].astype(np.float32)
        self._pending = [row.copy() for row in plan[1:]]
        return plan[0].copy()
