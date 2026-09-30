from __future__ import annotations

from dataclasses import fields

import pytest

from wsl_vla.octo.bridge import ResearchOctoBundle, _group_component


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("task_language_instruction", 1),
        ("obs_task_language_instruction", 1),
        ("obs_primary", 0),
        ("obs_wrist", 0),
        ("readout_action", 2),
    ],
)
def test_octo_token_groups_are_routed_to_the_intended_modality(name, expected):
    assert _group_component(name) == expected


def test_unknown_octo_token_group_fails_closed():
    with pytest.raises(ValueError, match="unrecognized Octo token group"):
        _group_component("other_tokens")


def test_bundle_retains_a_prepatch_reference_output():
    assert "reference_transformer_outputs" in {field.name for field in fields(ResearchOctoBundle)}
