"""Training batches and rollout inputs must come out of the same preprocessing."""

import numpy as np
import pytest

from tests.fakes import FakeTextProcessor, episode, example_batch, preprocessing_spec
from wsl_vla.data.octo_batches import octo_example
from wsl_vla.data.preprocessing import (
    apply_image_transform,
    camera_frames,
    preprocessing_spec_from_checkpoint,
    task_batch,
    text_processor_fingerprint,
)
from wsl_vla.evaluation.libero_rollout import build_octo_observation


def simulator_history(demo, spec, upto: int):
    """Simulator observations that render exactly what the demonstration stored."""

    return [
        {camera.simulator_key: demo.observations[camera.dataset_key][t] for camera in spec.cameras}
        for t in range(upto + 1)
    ]


@pytest.mark.parametrize("image_transform", ["identity", "flip_vertical"])
def test_rollout_inputs_equal_training_inputs_for_the_same_frames(image_transform):
    spec = preprocessing_spec(image_transform=image_transform)
    example = example_batch(spec)
    demo = episode(0, length=5)
    for timestep in range(len(demo.actions)):
        train = octo_example(
            demo, timestep, language_instruction="go", spec=spec,
            example_observation=example["observation"],
        )["observation"]
        rollout = build_octo_observation(
            simulator_history(demo, spec, timestep), spec, example["observation"]
        )
        assert set(rollout) == set(train)
        for key, value in train.items():
            if isinstance(value, dict):
                for name in value:
                    np.testing.assert_array_equal(rollout[key][name][0], value[name])
            else:
                np.testing.assert_array_equal(rollout[key][0], value)


def test_orientation_is_applied_identically_and_flip_is_not_identity():
    upright = preprocessing_spec(image_transform="flip_vertical")
    stored = preprocessing_spec(image_transform="identity")
    demo = episode(2, length=3)
    camera = upright.cameras[1]  # wrist: no resize, so pixels can be compared exactly
    raw = demo.observations[camera.dataset_key]
    flipped = camera_frames(raw, camera, upright, source="dataset")
    np.testing.assert_array_equal(flipped, raw[:, ::-1])
    np.testing.assert_array_equal(camera_frames(raw, camera, stored, source="dataset"), raw)
    assert upright.sha256() != stored.sha256()
    assert upright.presents_upright_frames and not stored.presents_upright_frames


def test_simulator_alignment_brings_renders_into_demonstration_orientation():
    spec = preprocessing_spec(image_transform="identity", simulator_alignment="flip_vertical")
    demo = episode(3, length=2)
    camera = spec.cameras[1]
    stored = demo.observations[camera.dataset_key]
    rendered_opencv = apply_image_transform(stored, "flip_vertical")
    np.testing.assert_array_equal(
        camera_frames(rendered_opencv, camera, spec, source="simulator"),
        camera_frames(stored, camera, spec, source="dataset"),
    )
    spec.assert_simulator_convention("opencv")
    with pytest.raises(ValueError, match="camera orientation mismatch"):
        spec.assert_simulator_convention("opengl")
    with pytest.raises(ValueError, match="camera orientation mismatch"):
        preprocessing_spec(simulator_alignment="identity").assert_simulator_convention("opencv")


def test_rollout_resolution_must_match_demonstration_resolution():
    spec = preprocessing_spec()
    camera = spec.cameras[0]
    with pytest.raises(ValueError, match="same camera resolution"):
        camera_frames(np.zeros((1, 16, 16, 3), np.uint8), camera, spec, source="simulator")


def test_language_task_inputs_are_shared_and_goal_images_masked_out():
    spec = preprocessing_spec()
    example = example_batch(spec)
    tasks = task_batch(["pick the bowl", "open the drawer"], FakeTextProcessor(), example["task"])
    assert tasks["language_instruction"]["input_ids"].shape == (2, 4)
    assert tasks["pad_mask_dict"]["language_instruction"].tolist() == [True, True]
    assert tasks["pad_mask_dict"]["image_primary"].tolist() == [False, False]
    assert not tasks["image_primary"].any()
    with pytest.raises(ValueError, match="non-empty language instruction"):
        task_batch(["  "], FakeTextProcessor(), example["task"])


MODEL_CONFIG = {
    "observation_tokenizers": {
        "primary": {"name": "ImageTokenizer", "kwargs": {"obs_stack_keys": ["image_primary"]}},
        "wrist": {"name": "ImageTokenizer", "kwargs": {"obs_stack_keys": ["image_wrist"]}},
    }
}
SETTINGS = dict(
    text_processor=FakeTextProcessor(),
    source_resolution=(8, 8),
    source_image_convention="opengl",
    image_transform="flip_vertical",
    simulator_alignment="identity",
)


def test_spec_records_what_the_checkpoint_consumes_and_never_claims_proprio():
    example = example_batch(preprocessing_spec())
    example["observation"]["proprio"] = np.zeros((1, 2, 8), dtype=np.float32)
    spec = preprocessing_spec_from_checkpoint(
        model_config=MODEL_CONFIG, example_batch=example, text_processor_config={"t": 1}, **SETTINGS
    )
    assert spec.consumed_observations == ("image_primary", "image_wrist")
    assert spec.unconsumed_observations == ("proprio",)
    assert not spec.proprio_consumed
    assert (spec.window_size, spec.action_horizon, spec.action_dim) == (2, 3, 7)
    with_proprio = {
        "observation_tokenizers": {
            **MODEL_CONFIG["observation_tokenizers"],
            "proprio": {"name": "LowdimObsTokenizer", "kwargs": {"obs_keys": ["proprio"]}},
        }
    }
    with pytest.raises(ValueError, match="proprioception tokenizer"):
        preprocessing_spec_from_checkpoint(
            model_config=with_proprio, example_batch=example, text_processor_config={"t": 1},
            **SETTINGS,
        )
    unknown = {"observation_tokenizers": {"depth": {"name": "DepthTokenizer", "kwargs": {}}}}
    with pytest.raises(ValueError, match="unrecognized observation tokenizer"):
        preprocessing_spec_from_checkpoint(
            model_config=unknown, example_batch=example, text_processor_config={"t": 1}, **SETTINGS
        )


def test_preprocessing_identity_follows_what_the_tokenizer_produces():
    """Octo fetches tokenizer files by name; a changed tokenizer must change the identity."""

    example = example_batch(preprocessing_spec())

    class ShiftedTokenizer(FakeTextProcessor):
        def encode(self, strings):
            encoded = super().encode(strings)
            return {**encoded, "input_ids": encoded["input_ids"] + 1}

    def build(processor, config=None):
        return preprocessing_spec_from_checkpoint(
            model_config=MODEL_CONFIG,
            example_batch=example,
            text_processor_config={"name": "t5-base"} if config is None else config,
            **{**SETTINGS, "text_processor": processor},
        )

    reference = build(FakeTextProcessor())
    assert build(FakeTextProcessor()).sha256() == reference.sha256()
    # Same name and configuration, different ids for the same instructions.
    assert build(ShiftedTokenizer()).sha256() != reference.sha256()
    assert build(FakeTextProcessor(), config={"name": "other"}).sha256() != reference.sha256()
    assert text_processor_fingerprint(FakeTextProcessor(), {"name": "t5-base"}) == (
        reference.text_processor
    )
