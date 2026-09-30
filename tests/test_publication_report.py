from dataclasses import replace

import numpy as np
import pytest

from wsl_vla.contracts import EvaluationRecord
from wsl_vla.evaluation.publication import (
    crossed_bootstrap_mean,
    publication_study_report,
)


def record(*, suite, seed, condition, stage, task, successes):
    return EvaluationRecord(
        run_id=f"{condition}-{suite}-{seed}",
        suite=suite,
        seed=seed,
        condition=condition,
        training_stage=stage,
        evaluated_task_index=task,
        evaluated_task_id=f"{suite}_{task}",
        rollout_count=4,
        successes=successes,
        checkpoint_sha256="a" * 64,
        latent_drift={"vision": 0.1, "language": 0.2, "action": 0.3},
        wall_time_seconds=2.0,
        initialization_indices=(0, 1, 2, 3),
        rollout_seeds=(10, 11, 12, 13),
        evaluation_wall_time_seconds=1.0,
        adapter_bytes=100,
        evidence_bytes=20,
        peak_vram_bytes=1000,
        peak_ram_bytes=2000,
        training_update_steps=10,
        training_micro_steps=20,
    )


def complete_records():
    rows = []
    for condition in ("method", "oracle"):
        for suite in ("suite_a", "suite_b"):
            for seed in (17, 42):
                for stage, task in ((0, 0), (1, 0), (1, 1)):
                    successes = 3 if condition == "method" else 2
                    rows.append(
                        record(
                            suite=suite,
                            seed=seed,
                            condition=condition,
                            stage=stage,
                            task=task,
                            successes=successes,
                        )
                    )
    return tuple(rows)


def test_publication_report_requires_and_summarizes_complete_factorial_design():
    report = publication_study_report(
        complete_records(),
        suites=("suite_a", "suite_b"),
        seeds=(17, 42),
        conditions=("method", "oracle"),
        oracle_condition="oracle",
        task_count=2,
        resamples=100,
    )
    assert report["design"]["expected_run_count"] == 8
    assert report["aggregate"]["method"]["average_success_rate"]["estimate"] == 0.75
    assert report["aggregate"]["method"]["forward_transfer"]["estimate"] == 0.25
    resources = report["per_run"]["method"]["suite_a"]["17"]["resources"]
    assert resources["training_wall_time_seconds"] == 4.0
    assert resources["adapter_bytes_per_task"] == 50.0


def test_publication_report_rejects_missing_or_mixed_runs():
    rows = complete_records()
    with pytest.raises(ValueError, match="incomplete run"):
        publication_study_report(
            rows[:-1],
            suites=("suite_a", "suite_b"),
            seeds=(17, 42),
            conditions=("method", "oracle"),
            oracle_condition="oracle",
            task_count=2,
            resamples=10,
        )
    mixed = list(rows)
    mixed[1] = replace(mixed[1], run_id="different")
    with pytest.raises(ValueError, match="run identity"):
        publication_study_report(
            mixed,
            suites=("suite_a", "suite_b"),
            seeds=(17, 42),
            conditions=("method", "oracle"),
            oracle_condition="oracle",
            task_count=2,
            resamples=10,
        )


def test_crossed_bootstrap_reports_undefined_zero_baseline_runs():
    estimate = crossed_bootstrap_mean(
        np.array([[np.nan, 0.5], [np.nan, 1.0]]), resamples=100
    )
    assert estimate["estimate"] == 0.75
    assert estimate["sample_count"] == 2
    assert estimate["excluded_undefined"] == 2
    undefined = crossed_bootstrap_mean(np.full((2, 2), np.nan), resamples=10)
    assert undefined["estimate"] is None
    assert undefined["excluded_undefined"] == 4
