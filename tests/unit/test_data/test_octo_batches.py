import numpy as np
import pytest

from tests.fakes import FakeTextProcessor, episode, episodes, example_batch, preprocessing_spec
from wsl_vla.data.libero import LiberoEpisode
from wsl_vla.data.octo_batches import (
    collate_octo_examples,
    conform_batch_to_octo_example,
    iter_octo_examples,
)


def test_octo_windows_preserve_episode_boundaries_and_action_masks():
    spec = preprocessing_spec()
    example = example_batch(spec)
    first = episode(0, length=3)
    examples = list(
        iter_octo_examples(
            (first,),
            language_instruction="do the task",
            spec=spec,
            example_observation=example["observation"],
        )
    )
    assert len(examples) == 3
    # The first window has one real frame, left-padded and masked.
    assert examples[0]["observation"]["timestep_pad_mask"].tolist() == [False, True]
    assert examples[0]["observation"]["pad_mask_dict"]["image_primary"].tolist() == [False, True]
    assert not examples[0]["observation"]["image_primary"][0].any()
    # The last step can only supervise the single action that still exists.
    assert examples[-1]["action_pad_mask"][-1, :, 0].tolist() == [True, False, False]
    np.testing.assert_array_equal(examples[-1]["action"][-1, 0], first.actions[-1])
    batch = collate_octo_examples(
        examples[:2], text_processor=FakeTextProcessor(), example_task=example["task"]
    )
    assert batch["action"].shape == (2, 2, 3, 7)
    assert batch["task"]["pad_mask_dict"]["language_instruction"].tolist() == [True, True]
    assert batch["task"]["pad_mask_dict"]["image_primary"].tolist() == [False, False]
    conform_batch_to_octo_example(batch, example)


def test_targets_stay_in_controller_coordinates_and_are_never_rescaled():
    spec = preprocessing_spec()
    example = example_batch(spec)
    first = episode(1, length=5)
    examples = list(
        iter_octo_examples(
            (first,), language_instruction="go", spec=spec, example_observation=example["observation"]
        )
    )
    for timestep, item in enumerate(examples):
        np.testing.assert_array_equal(item["action"][-1, 0], first.actions[timestep])


def test_out_of_bounds_training_targets_are_rejected_not_clipped():
    spec = preprocessing_spec()
    example = example_batch(spec)
    bad = episode(0, length=4)
    actions = bad.actions.copy()
    actions[2, 0] = 1.5
    with pytest.raises(ValueError, match="never clipped silently"):
        list(
            iter_octo_examples(
                (LiberoEpisode(bad.episode_id, actions, bad.observations),),
                language_instruction="go",
                spec=spec,
                example_observation=example["observation"],
            )
        )


def test_conform_batch_fails_closed_on_extra_or_missing_checkpoint_inputs():
    spec = preprocessing_spec()
    example = example_batch(spec)
    examples = list(
        iter_octo_examples(
            episodes(1), language_instruction="go", spec=spec,
            example_observation=example["observation"],
        )
    )
    batch = collate_octo_examples(
        examples[:2], text_processor=FakeTextProcessor(), example_task=example["task"]
    )
    extra = {**batch, "observation": {**batch["observation"], "proprio": np.zeros((2, 2, 8))}}
    with pytest.raises(ValueError, match="keys differ from the checkpoint"):
        conform_batch_to_octo_example(extra, example)
    missing = {
        **batch,
        "observation": {k: v for k, v in batch["observation"].items() if k != "image_wrist"},
    }
    with pytest.raises(ValueError, match="keys differ from the checkpoint"):
        conform_batch_to_octo_example(missing, example)
