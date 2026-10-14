"""Token-routing contract of the Octo bridge (no Octo installation required).

The routing table is derived from the checkpoint configuration alone, so these
checks run in any environment. The patched graph itself is exercised on a toy
official Octo model in ``tests/unit/test_octo``.
"""

from __future__ import annotations

import copy
from dataclasses import fields

import pytest

from wsl_vla.contracts import COMPONENT_NAMES, FROZEN_ROUTE
from wsl_vla.octo.bridge import (
    ResearchOctoBundle,
    adapter_marker,
    build_token_routing,
    research_model_config,
)


def spec(name: str, **kwargs) -> dict:
    return {"module": "octo.model.components", "name": name, "args": [], "kwargs": kwargs}


# The token-producing structure of rail-berkeley/octo-small-1.5.
OCTO_SMALL = {
    "observation_tokenizers": {
        "primary": spec("ImageTokenizer", obs_stack_keys=["image_primary"]),
        "wrist": spec("ImageTokenizer", obs_stack_keys=["image_wrist"]),
    },
    "task_tokenizers": {"language": spec("LanguageTokenizer", encoder="t5-base")},
    "heads": {"action": spec("DiffusionActionHead", readout_key="readout_action")},
    "readouts": {"action": 1},
    "repeat_task_tokens": True,
    "transformer_kwargs": {"num_layers": 12},
}


def test_octo_token_groups_are_routed_to_the_intended_modality():
    assert build_token_routing(OCTO_SMALL) == {
        "obs_primary": "vision",
        "obs_wrist": "vision",
        "task_language": "language",
        "obs_task_language": "language",
        "readout_action": "action",
    }


def test_task_tokens_are_not_repeated_unless_the_checkpoint_repeats_them():
    config = {**OCTO_SMALL, "repeat_task_tokens": False}
    assert "obs_task_language" not in build_token_routing(config)


def test_unknown_octo_tokenizer_fails_closed():
    config = copy.deepcopy(OCTO_SMALL)
    config["observation_tokenizers"]["tactile"] = spec("TactileTokenizer")
    with pytest.raises(ValueError, match="unrecognized Octo observation tokenizer 'tactile'"):
        build_token_routing(config)


def test_proprioception_and_unused_readouts_are_frozen_not_guessed():
    config = copy.deepcopy(OCTO_SMALL)
    config["observation_tokenizers"]["proprio"] = spec("LowdimObsTokenizer", obs_keys=["proprio"])
    config["readouts"]["value"] = 1
    routing = build_token_routing(config)

    assert routing["obs_proprio"] == FROZEN_ROUTE
    assert routing["readout_value"] == FROZEN_ROUTE
    assert routing["readout_action"] == "action"


@pytest.mark.parametrize("component", COMPONENT_NAMES)
def test_checkpoint_must_route_tokens_to_every_modality(component):
    config = copy.deepcopy(OCTO_SMALL)
    if component == "vision":
        config["observation_tokenizers"] = {}
    elif component == "language":
        config["task_tokenizers"] = {}
    else:
        config["heads"] = {}
    with pytest.raises(ValueError, match="routes no token group to"):
        build_token_routing(config)


def test_adapter_identity_is_written_into_the_research_module_config():
    routing = build_token_routing(OCTO_SMALL)
    official = {"model": copy.deepcopy(OCTO_SMALL), "text_processor": None}
    research = research_model_config(official, rank=8, alpha=16.0, routing=routing)

    marker = research["model"]["transformer_kwargs"]["research_adapter"]
    assert marker == adapter_marker(rank=8, alpha=16.0, routing=routing)
    assert "research_adapter" not in official["model"]["transformer_kwargs"]
    # Rank, scale, and routing each change the identity a compiled graph is keyed on.
    assert marker != adapter_marker(rank=4, alpha=16.0, routing=routing)
    assert marker != adapter_marker(rank=8, alpha=8.0, routing=routing)
    assert marker != adapter_marker(
        rank=8, alpha=16.0, routing={**routing, "obs_wrist": FROZEN_ROUTE}
    )


def test_bundle_retains_prepatch_references_for_both_equivalence_checks():
    names = {field.name for field in fields(ResearchOctoBundle)}
    assert {
        "reference_transformer_outputs",
        "reference_sampled_actions",
        "equivalence_seed",
        "token_routing",
        "action_convention",
        "preprocessing",
        "patched_block_transformer",
    } <= names
