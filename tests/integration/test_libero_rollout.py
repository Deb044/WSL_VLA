"""Rollout evaluator and policy wrapper against a scripted stand-in simulator.

No MuJoCo or LIBERO is needed here: the evaluator accepts an injected benchmark
and environment factory. The policy-level tests use the toy official Octo model
and are skipped when the research environment is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from tests.fakes import SOURCE, episode, example_batch, preprocessing_spec
from wsl_vla.contracts import LIBERO_ACTION_CONVENTION
from wsl_vla.data.octo_batches import octo_example
from wsl_vla.evaluation.libero_rollout import (
    LiberoRolloutEvaluator,
    build_octo_observation,
    fixed_initialization_indices,
    validate_benchmark_task_order,
)


@dataclass
class FakeTask:
    language: str
    problem_folder: str = "problem"
    bddl_file: str = "task.bddl"


class FakeSuite:
    def __init__(self, instructions, init_count=60):
        self.tasks = [FakeTask(value) for value in instructions]
        self.init_states = np.arange(init_count * 2).reshape(init_count, 2)

    def get_task(self, index):
        return self.tasks[index]

    def get_task_init_states(self, index):
        return self.init_states + index


def frame(value: int) -> np.ndarray:
    return np.full((*SOURCE, 3), value % 251, dtype=np.uint8)


class FakeEnvironment:
    """Succeeds on the seventh step after a reset; records everything it is told."""

    instances: list["FakeEnvironment"] = []
    success_step = 7

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.step_count = 0
        self.closed = False
        self.seeds: list[int] = []
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.__class__.instances.append(self)

    def _observation(self):
        return {
            "agentview_image": frame(self.step_count),
            "robot0_eye_in_hand_image": frame(self.step_count + 100),
        }

    def seed(self, seed):
        self.seeds.append(seed)

    def reset(self):
        self.step_count = 0

    def set_init_state(self, state):
        self.states.append(np.array(state))
        return self._observation()

    def step(self, action):
        self.actions.append(np.array(action))
        self.step_count += 1
        success = self.step_count == self.success_step
        return self._observation(), float(success), success, {}

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self, checkpoint="a", command=None):
        self.checkpoint_sha256 = checkpoint * 64
        self.resets: list[tuple[str, int]] = []
        self.command = np.zeros(7, dtype=np.float32) if command is None else command

    def reset(self, *, instruction, seed):
        self.resets.append((instruction, seed))

    def action(self, observation):
        return self.command.copy()


def instructions():
    return tuple(f"Do task {index}." for index in range(10))


def make_evaluator(tmp_path, *, spec=None, convention="opengl", **overrides):
    FakeEnvironment.instances.clear()
    folder = tmp_path / "problem"
    folder.mkdir(exist_ok=True)
    (folder / "task.bddl").write_text("task", encoding="utf-8")
    settings = {
        "suite": "libero_spatial",
        "expected_instructions": instructions(),
        "preprocessing": preprocessing_spec() if spec is None else spec,
        "benchmark_suite": FakeSuite(instructions()),
        "env_factory": FakeEnvironment,
        "bddl_root": tmp_path,
        "simulator_image_convention": convention,
        "warmup_steps": 5,
        "max_steps": 5,
    }
    settings.update(overrides)
    return LiberoRolloutEvaluator(**settings)


# -------------------------------------------------------------- determinism
def test_fixed_initializations_are_unique_deterministic_and_fail_closed():
    first = fixed_initialization_indices(50, 20, seed=17)
    assert first == fixed_initialization_indices(50, 20, seed=17)
    assert len(first) == len(set(first)) == 20
    assert first != fixed_initialization_indices(50, 20, seed=18)
    with pytest.raises(ValueError, match="only 10"):
        fixed_initialization_indices(10, 20, seed=17)


def test_task_order_validation_allows_only_presentation_differences():
    expected = instructions()
    validate_benchmark_task_order(
        FakeSuite([value.lower().rstrip(".") for value in expected]), expected
    )
    changed = list(expected)
    changed[3] = "Do a different task."
    with pytest.raises(ValueError, match="index 3"):
        validate_benchmark_task_order(FakeSuite(changed), expected)


# ---------------------------------------------------------------- evaluator
def test_evaluator_renders_at_the_demonstration_resolution_and_closes_the_environment(tmp_path):
    evaluator = make_evaluator(tmp_path)
    policy = FakePolicy()
    result = evaluator.evaluate_task(policy, task_index=2, rollout_count=3, seed=17)

    env = FakeEnvironment.instances[0]
    assert env.kwargs["camera_heights"] == SOURCE[0] and env.kwargs["camera_widths"] == SOURCE[1]
    assert env.closed
    assert result.successes == 3 and result.rollout_count == 3
    assert result.initialization_indices == fixed_initialization_indices(60, 3, seed=17)
    # Five warm-up steps, then success on the second policy step.
    assert all(item.steps == 2 for item in result.episodes)
    assert len(policy.resets) == 3
    outcome = result.as_outcome()
    assert outcome.successes == 3
    assert outcome.initialization_indices == result.initialization_indices


def test_rollout_cameras_must_match_demonstration_generation(tmp_path):
    # Demonstrations: OpenGL convention; a simulator rendering OpenCV rows is
    # upside down relative to them unless the alignment is declared.
    with pytest.raises(ValueError, match="camera orientation mismatch"):
        make_evaluator(tmp_path, convention="opencv")
    make_evaluator(
        tmp_path, convention="opencv", spec=preprocessing_spec(simulator_alignment="flip_vertical")
    )
    with pytest.raises(ValueError, match="camera orientation mismatch"):
        make_evaluator(tmp_path, spec=preprocessing_spec(simulator_alignment="flip_vertical"))
    with pytest.raises(ValueError, match="must declare its image convention"):
        make_evaluator(tmp_path, convention=None)
    with pytest.raises(ValueError, match="unknown simulator image convention"):
        make_evaluator(tmp_path, convention="direct3d")


def test_policies_are_evaluated_on_identical_initial_states_and_seeds(tmp_path):
    evaluator = make_evaluator(tmp_path)
    first, second = FakePolicy("a"), FakePolicy("b")
    summaries = evaluator.evaluate_policies(
        {"frozen_base": first, "adapter": second}, task_index=1, rollout_count=4, seed=17
    )

    base, adapter = summaries["frozen_base"], summaries["adapter"]
    assert base.initialization_indices == adapter.initialization_indices
    assert [item.episode_seed for item in base.episodes] == [
        item.episode_seed for item in adapter.episodes
    ]
    assert len({item.episode_seed for item in base.episodes}) == 4
    assert first.resets == second.resets
    env = FakeEnvironment.instances[0]
    # The single environment is re-seeded and reset to the same state for each policy.
    assert env.seeds[0::2] == env.seeds[1::2]
    for left, right in zip(env.states[0::2], env.states[1::2]):
        np.testing.assert_array_equal(left, right)
    assert base.policy_checkpoint_sha256 != adapter.policy_checkpoint_sha256

    # Another seed gives another, equally shared, set of episodes.
    other = make_evaluator(tmp_path).evaluate_policies(
        {"frozen_base": FakePolicy("a")}, task_index=1, rollout_count=4, seed=18
    )["frozen_base"]
    assert other.initialization_indices != base.initialization_indices


def test_out_of_bounds_commands_are_clipped_and_counted(tmp_path):
    evaluator = make_evaluator(tmp_path)
    command = np.zeros(7, dtype=np.float32)
    command[0], command[6] = 3.5, -1.0
    saturated = FakePolicy("c", command=command)
    inside = FakePolicy("d")
    summaries = evaluator.evaluate_policies(
        {"saturated": saturated, "inside": inside}, task_index=0, rollout_count=2, seed=17
    )

    env = FakeEnvironment.instances[0]
    executed = np.stack(env.actions)
    assert executed.max() <= LIBERO_ACTION_CONVENTION.high[0]
    assert executed.min() >= LIBERO_ACTION_CONVENTION.low[0]
    report = summaries["saturated"].clipping
    assert report["clipped_steps"] == report["steps"] == 4
    assert report["clipped_step_fraction"] == 1.0
    assert report["max_excess"] == pytest.approx(2.5)
    for item in summaries["saturated"].episodes:
        assert item.clipped_steps == item.steps and item.max_clip_excess == pytest.approx(2.5)
    assert summaries["inside"].clipping["clipped_steps"] == 0
    assert summaries["inside"].clipping["clipped_step_fraction"] == 0.0


def test_evaluator_rejects_bad_requests(tmp_path):
    evaluator = make_evaluator(tmp_path)
    with pytest.raises(ValueError, match="at least one policy"):
        evaluator.evaluate_policies({}, task_index=0, rollout_count=1, seed=17)
    with pytest.raises(IndexError):
        evaluator.evaluate_task(FakePolicy(), task_index=10, rollout_count=1, seed=17)
    short = FakePolicy()
    short.checkpoint_sha256 = "abc"
    with pytest.raises(ValueError, match="SHA-256"):
        evaluator.evaluate_task(short, task_index=0, rollout_count=1, seed=17)
    with pytest.raises(ValueError, match="unsupported reference suite"):
        make_evaluator(tmp_path, suite="libero_90")


# -------------------------------------------------------------- observation
def test_rollout_observation_is_the_training_observation_for_the_same_frames():
    spec = preprocessing_spec()
    example = example_batch(spec)
    demonstration = episode(0, length=5)
    history = [
        {
            "agentview_image": demonstration.observations["agentview_rgb"][step],
            "robot0_eye_in_hand_image": demonstration.observations["eye_in_hand_rgb"][step],
        }
        for step in range(4)
    ]
    rollout = build_octo_observation(history, spec, example["observation"])
    training = octo_example(
        demonstration,
        3,
        language_instruction="do the task",
        spec=spec,
        example_observation=example["observation"],
    )["observation"]

    assert set(rollout) == set(training)
    for key, value in training.items():
        if isinstance(value, dict):
            for name, mask in value.items():
                np.testing.assert_array_equal(rollout[key][name][0], mask)
        else:
            np.testing.assert_array_equal(rollout[key][0], value)
            assert rollout[key].dtype == np.asarray(value).dtype


def test_rollout_observation_left_pads_the_first_step():
    spec = preprocessing_spec()
    example = example_batch(spec)
    result = build_octo_observation(
        [{"agentview_image": frame(5), "robot0_eye_in_hand_image": frame(9)}],
        spec,
        example["observation"],
    )

    np.testing.assert_array_equal(result["timestep_pad_mask"], [[False, True]])
    assert result["image_primary"].shape == (1, 2, 16, 16, 3)
    assert not result["image_primary"][0, 0].any() and result["image_primary"][0, 1].mean() == 5
    assert result["image_wrist"][0, 1].mean() == 9
    np.testing.assert_array_equal(result["pad_mask_dict"]["image_primary"], [[False, True]])
    assert result["task_completed"].shape == (1, 2, spec.action_horizon)


def test_rollout_observation_rejects_missing_cameras_and_other_resolutions():
    spec = preprocessing_spec()
    example = example_batch(spec)
    with pytest.raises(ValueError, match="lack robot0_eye_in_hand_image"):
        build_octo_observation([{"agentview_image": frame(1)}], spec, example["observation"])
    wrong = {
        "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    with pytest.raises(ValueError, match="must use the same camera resolution"):
        build_octo_observation([wrong], spec, example["observation"])
    with pytest.raises(ValueError, match="history cannot be empty"):
        build_octo_observation([], spec, example["observation"])


# ----------------------------------------------------- policy on toy Octo
@pytest.mark.research
def test_policy_samples_the_action_the_model_gives_the_matching_training_window(tiny_bundle):
    import jax

    from tests import octo_tiny as tiny
    from wsl_vla.evaluation.libero_rollout import OctoLiberoPolicy
    from wsl_vla.octo.bridge import unnormalization_statistics
    from wsl_vla.octo.training import initial_adapter_state, materialize_policy_params

    state = initial_adapter_state(tiny_bundle)
    policy = OctoLiberoPolicy(tiny_bundle, state)
    demonstration = episode(0, length=5)
    stream = tiny.tiny_stream(tiny_bundle)
    policy.reset(instruction=stream.instruction, seed=123)
    actions = [
        policy.action(
            {
                "agentview_image": demonstration.observations["agentview_rgb"][step],
                "robot0_eye_in_hand_image": demonstration.observations["eye_in_hand_rgb"][step],
            }
        )
        for step in range(3)
    ]

    # Reproduce step two by hand from the *training* preprocessing of the same frames.
    example = octo_example(
        demonstration,
        2,
        language_instruction=stream.instruction,
        spec=tiny_bundle.preprocessing,
        example_observation=tiny_bundle.pretrained_model.example_batch["observation"],
    )
    batch = stream.collate([{**example}], tiny_bundle.pretrained_model.text_processor)
    key = jax.random.PRNGKey(123)
    for _ in range(3):
        key, sample_key = jax.random.split(key)
    model = tiny_bundle.research_model.replace(
        params=materialize_policy_params(tiny_bundle, state)
    )
    expected = np.asarray(
        model.sample_actions(
            batch["observation"],
            batch["task"],
            unnormalization_statistics=unnormalization_statistics(tiny_bundle.action_convention),
            rng=sample_key,
            argmax=policy.argmax,
            temperature=policy.temperature,
        )
    )[0, 0]

    np.testing.assert_array_equal(actions[2], expected.astype(np.float32))
    assert all(item.shape == (7,) and item.dtype == np.float32 for item in actions)
    # Identity normalization: the controller receives the head's samples unscaled.
    np.testing.assert_array_equal(policy.unnormalization_statistics["mean"], np.zeros(7))
    np.testing.assert_array_equal(policy.unnormalization_statistics["std"], np.ones(7))


@pytest.mark.research
def test_policies_share_noise_and_are_identified_by_their_adapter(tiny_bundle):
    import jax

    from tests import octo_tiny as tiny
    from wsl_vla.evaluation.libero_rollout import OctoLiberoPolicy
    from wsl_vla.experiments.provenance import sha256_array_tree
    from wsl_vla.octo.training import initial_adapter_state

    base_state = jax.device_get(initial_adapter_state(tiny_bundle))
    changed = {
        "transformer": {
            path: (np.full_like(value, 0.3) if path.endswith("_up/kernel") else value)
            for path, value in base_state["transformer"].items()
        },
        "diffusion": base_state["diffusion"],
    }
    observation = {
        "agentview_image": frame(40),
        "robot0_eye_in_hand_image": frame(90),
    }

    def run(state, seed):
        policy = OctoLiberoPolicy(tiny_bundle, state)
        policy.reset(instruction=tiny.INSTRUCTION, seed=seed)
        return policy, np.stack([policy.action(observation) for _ in range(3)])

    base, base_actions = run(base_state, 7)
    _, repeat_actions = run(base_state, 7)
    _, other_seed = run(base_state, 8)
    adapted, adapted_actions = run(changed, 7)

    np.testing.assert_array_equal(base_actions, repeat_actions)
    assert not np.array_equal(base_actions, other_seed)
    assert not np.array_equal(base_actions, adapted_actions)
    assert base.checkpoint_sha256 == sha256_array_tree(base_state)
    assert adapted.checkpoint_sha256 == sha256_array_tree(changed)
    assert base.checkpoint_sha256 != adapted.checkpoint_sha256
    with pytest.raises(RuntimeError, match="reset must be called"):
        OctoLiberoPolicy(tiny_bundle, base_state).action(observation)


@pytest.mark.research
def test_policy_executes_the_configured_number_of_actions_per_plan(tiny_t5):
    from tests import octo_tiny as tiny
    from wsl_vla.evaluation.libero_rollout import OctoLiberoPolicy
    from wsl_vla.octo.training import initial_adapter_state

    bundle = tiny.build_tiny_bundle(
        tiny_t5, preprocessing={**tiny.TINY_PREPROCESSING, "executed_actions_per_plan": 2}
    )
    assert bundle.preprocessing.executed_actions_per_plan == 2
    policy = OctoLiberoPolicy(bundle, initial_adapter_state(bundle))
    policy.reset(instruction=tiny.INSTRUCTION, seed=3)
    observation = {"agentview_image": frame(1), "robot0_eye_in_hand_image": frame(2)}

    first = policy.action(observation)
    assert len(policy._pending) == 1
    queued = policy._pending[0].copy()
    second = policy.action(observation)
    np.testing.assert_array_equal(second, queued)
    assert not policy._pending and not np.array_equal(first, second)
    # The third call plans again from the updated history.
    policy.action(observation)
    assert len(policy._pending) == 1

    # A different plan length is a different preprocessing identity.
    single = replace(bundle.preprocessing, executed_actions_per_plan=1)
    assert single.sha256() != bundle.preprocessing.sha256()
