from __future__ import annotations

import pytest

from wsl_vla.experiments.gamma_selection import (
    GammaValidationRecord,
    select_gamma_configuration,
    validate_gamma_candidate_config,
)


SOURCES = ("libero_spatial", "libero_object", "libero_goal")
SEEDS = (17, 42, 73)


def test_gamma_candidate_search_space_is_explicit_and_direction_safe():
    payload = {
        "schema_version": 1,
        "early_stopping_patience": [10, 20],
        "proposed": [
            {"name": "asymmetric", "vision": 1.0, "language": 2.0, "action": 0.1}
        ],
        "uniform": [
            {"name": "uniform", "vision": 0.5, "language": 0.5, "action": 0.5}
        ],
    }
    validate_gamma_candidate_config(payload)
    payload["proposed"][0]["action"] = 3.0
    with pytest.raises(ValueError, match="preserve the hypothesis"):
        validate_gamma_candidate_config(payload)


def records_for(family, gammas, patience, score):
    result = []
    for suite in SOURCES:
        for seed in SEEDS:
            result.append(
                GammaValidationRecord(
                    run_id=f"{family}-{patience}-{suite}-{seed}-{score}",
                    held_out_suite="libero_10",
                    source_suite=suite,
                    seed=seed,
                    family=family,
                    gammas=gammas,
                    early_stopping_patience=patience,
                    average_success_rate=score,
                    negative_backward_transfer=0.1,
                    validation_task_ids=(f"{suite}_8", f"{suite}_9"),
                    rollout_count=20,
                    stage_update_steps=(100, 80),
                    max_steps_per_stage=100,
                    alignment_checkpoint_sha256="a" * 64,
                )
            )
    return result


def test_selection_requires_complete_non_test_cells_and_freezes_joint_patience():
    records = []
    records += records_for(
        "proposed", {"vision": 1.0, "language": 2.0, "action": 0.1}, 10, 0.7
    )
    records += records_for(
        "uniform", {"vision": 0.5, "language": 0.5, "action": 0.5}, 10, 0.6
    )
    records += records_for(
        "proposed", {"vision": 2.0, "language": 2.0, "action": 0.2}, 20, 0.8
    )
    records += records_for(
        "uniform", {"vision": 1.0, "language": 1.0, "action": 1.0}, 20, 0.75
    )
    selection, report = select_gamma_configuration(
        records, held_out_suite="libero_10", seeds=SEEDS
    )
    assert selection.early_stopping_patience == 20
    assert selection.proposed == {"vision": 2.0, "language": 2.0, "action": 0.2}
    assert selection.uniform == 1.0
    assert report["selection_objective"] == (
        "mean_success_rate_minus_negative_backward_transfer"
    )

    with pytest.raises(ValueError, match="incomplete gamma candidate"):
        select_gamma_configuration(
            records[:-1], held_out_suite="libero_10", seeds=SEEDS
        )


def test_gamma_validation_record_rejects_test_suite_and_wrong_direction():
    record = records_for(
        "proposed", {"vision": 1.0, "language": 1.0, "action": 0.1}, 10, 0.7
    )[0]
    broken = GammaValidationRecord(
        **{
            **record.__dict__,
            "source_suite": "libero_10",
        }
    )
    with pytest.raises(ValueError, match="non-held-out"):
        broken.validate()
    inverted = GammaValidationRecord(
        **{
            **record.__dict__,
            "gammas": {"vision": 0.1, "language": 0.1, "action": 1.0},
        }
    )
    with pytest.raises(ValueError, match="hypothesized direction"):
        inverted.validate()
