from __future__ import annotations

import numpy as np
import pytest

from core.metrics import (
    average_forgetting,
    average_success_rate,
    bootstrap_correlation,
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
