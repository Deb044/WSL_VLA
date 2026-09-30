from __future__ import annotations

import numpy as np
import pytest

from wsl_vla.contracts import EvaluationRecord
from wsl_vla.evaluation.metrics import (
    average_forgetting,
    average_success_rate,
    aggregate_seed_metrics,
    bootstrap_correlation,
    drift_degradation_correlations,
    forward_transfer,
    negative_backward_transfer,
    normalized_negative_backward_transfer,
    recovery_step_ratio,
)


MATRIX = np.array(
    [
        [0.8, np.nan, np.nan],
        [0.6, 0.9, np.nan],
        [0.4, 0.8, 0.7],
    ]
)


def test_continual_learning_metric_formulas():
    assert average_success_rate(MATRIX) == pytest.approx((0.4 + 0.8 + 0.7) / 3)
    expected_nbt = (((0.8 - 0.6) + (0.8 - 0.4)) / 2 + (0.9 - 0.8) + 0) / 3
    assert negative_backward_transfer(MATRIX) == pytest.approx(expected_nbt)
    expected_normalized = (((0.8 - 0.6) / 0.8 + (0.8 - 0.4) / 0.8) / 2 + (0.9 - 0.8) / 0.9) / 3
    assert normalized_negative_backward_transfer(MATRIX) == pytest.approx(expected_normalized)
    assert average_forgetting(MATRIX) == pytest.approx(((0.8 - 0.4) + (0.9 - 0.8)) / 2)
    assert forward_transfer(MATRIX, np.array([0.7, 0.7, 0.6])) == pytest.approx(0.4 / 3)
    assert recovery_step_ratio(50, 1000) == pytest.approx(0.05)


def test_zero_initial_success_is_excluded_from_normalized_nbt():
    matrix = np.array([[0.0, np.nan], [0.0, 0.5]])
    assert normalized_negative_backward_transfer(matrix) == pytest.approx(0.0)


def test_bootstrap_correlation_reports_both_methods():
    x = np.arange(10, dtype=float)
    y = x * 2 + np.array([0, 1, 0, -1, 0, 1, 0, -1, 0, 1])
    for method in ("pearson", "spearman"):
        result = bootstrap_correlation(x, y, method=method, resamples=200, seed=4)
        assert result.sample_count == 10
        assert result.ci_low <= result.estimate <= result.ci_high
        assert result.estimate > 0.9


def test_drift_correlation_uses_one_sample_per_stage_transition():
    records = []
    diagonal = [0.9, 0.85, 0.8, 0.75]
    for seed in (1, 2):
        for stage in range(4):
            drift = float(stage + seed / 10)
            for task in range(stage + 1):
                success = diagonal[task] if task == stage else diagonal[task] - 0.05 * stage
                records.append(
                    EvaluationRecord(
                        run_id=f"run-{seed}",
                        suite="suite",
                        seed=seed,
                        condition="proposed",
                        training_stage=stage,
                        evaluated_task_index=task,
                        evaluated_task_id=f"task-{task}",
                        rollout_count=20,
                        successes=round(success * 20),
                        checkpoint_sha256="a" * 64,
                        latent_drift={name: drift for name in ("vision", "language", "action")},
                        wall_time_seconds=1.0,
                        initialization_indices=tuple(range(20)),
                        rollout_seeds=tuple(range(20)),
                        evaluation_wall_time_seconds=1.0,
                    )
                )
    result = drift_degradation_correlations(
        records, condition="proposed", resamples=100, seed=3
    )
    assert result["vision"]["pearson"].sample_count == 6
    assert result["action"]["spearman"].estimate > 0


def test_aggregate_metrics_requires_and_reports_all_independent_seeds():
    records = []
    for seed, values in ((17, ((8,), (6, 9))), (42, ((7,), (5, 8))), (73, ((9,), (7, 9)))):
        for stage, row in enumerate(values):
            for task, successes in enumerate(row):
                records.append(
                    EvaluationRecord(
                        run_id=f"run-{seed}",
                        suite="suite",
                        seed=seed,
                        condition="condition",
                        training_stage=stage,
                        evaluated_task_index=task,
                        evaluated_task_id=f"task-{task}",
                        rollout_count=10,
                        successes=successes,
                        checkpoint_sha256="a" * 64,
                        latent_drift={name: float(stage) for name in ("vision", "language", "action")},
                        wall_time_seconds=1.0,
                        initialization_indices=tuple(range(10)),
                        rollout_seeds=tuple(range(10)),
                        evaluation_wall_time_seconds=1.0,
                    )
                )
    per_seed, aggregate = aggregate_seed_metrics(
        records,
        suite="suite",
        condition="condition",
        seeds=(17, 42, 73),
        task_count=2,
        resamples=100,
    )
    assert set(per_seed) == {17, 42, 73}
    assert aggregate["average_success_rate"].sample_count == 3
    with pytest.raises(ValueError, match="record seeds differ"):
        aggregate_seed_metrics(
            records,
            suite="suite",
            condition="condition",
            seeds=(17, 42),
            task_count=2,
        )
