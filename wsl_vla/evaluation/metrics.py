"""Publication metrics for continual learning and drift validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


def _validate_success_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("success matrix must be square [stage, task]")
    if np.any((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)):
        raise ValueError("finite success rates must lie in [0, 1]")
    for stage in range(values.shape[0]):
        if not np.isfinite(values[stage, : stage + 1]).all():
            raise ValueError("the observed lower triangle cannot contain missing values")
    return values


def average_success_rate(matrix: np.ndarray) -> float:
    values = _validate_success_matrix(matrix)
    final = values[-1]
    observed = final[np.isfinite(final)]
    return float(observed.mean())


def negative_backward_transfer(matrix: np.ndarray) -> float:
    """Reference-paper NBT, with the final task contributing zero.

    For task ``k``, average ``c[k,k] - c[t,k]`` over later stages ``t``.
    The paper averages over all K tasks, so task K has no later stages and is
    assigned zero rather than introducing an undefined denominator.
    """

    values = _validate_success_matrix(matrix)
    per_task: list[float] = []
    for task in range(values.shape[0]):
        later = values[task + 1 :, task]
        per_task.append(float(np.mean(values[task, task] - later)) if later.size else 0.0)
    return float(np.mean(per_task))


def normalized_negative_backward_transfer(matrix: np.ndarray) -> float:
    """NBT divided by performance when each task was first learned.

    Tasks whose initial success is exactly zero are excluded because relative
    forgetting is undefined. The final task contributes zero when its diagonal
    success is non-zero, matching the reference NBT convention.
    """

    values = _validate_success_matrix(matrix)
    per_task: list[float] = []
    for task in range(values.shape[0]):
        initial = values[task, task]
        if initial == 0:
            continue
        later = values[task + 1 :, task]
        per_task.append(float(np.mean((initial - later) / initial)) if later.size else 0.0)
    return float(np.mean(per_task)) if per_task else float("nan")


def average_forgetting(matrix: np.ndarray) -> float:
    values = _validate_success_matrix(matrix)
    final_stage = values.shape[0] - 1
    drops = []
    for task in range(final_stage):
        best = float(np.max(values[task:, task]))
        drops.append(best - float(values[final_stage, task]))
    return float(np.mean(drops)) if drops else 0.0


def forward_transfer(matrix: np.ndarray, independent_baseline: np.ndarray) -> float:
    values = _validate_success_matrix(matrix)
    baseline = np.asarray(independent_baseline, dtype=np.float64)
    if baseline.shape != (values.shape[0],):
        raise ValueError("independent_baseline must have one value per task")
    return float(np.mean(np.diag(values) - baseline))


def recovery_step_ratio(recovery_steps: float, original_training_steps: float) -> float:
    if recovery_steps < 0 or original_training_steps <= 0:
        raise ValueError("step counts must be non-negative and denominator positive")
    return float(recovery_steps / original_training_steps)


def _correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    if method == "spearman":
        return float(stats.spearmanr(x, y).statistic)
    raise ValueError(f"unknown correlation method: {method}")


@dataclass(frozen=True)
class CorrelationEstimate:
    estimate: float
    ci_low: float
    ci_high: float
    sample_count: int
    method: str


@dataclass(frozen=True)
class MeanEstimate:
    estimate: float
    ci_low: float
    ci_high: float
    sample_count: int


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> CorrelationEstimate:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 3:
        raise ValueError("correlation inputs must be matching vectors with at least 3 values")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("correlation inputs must be finite")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("correlation is undefined for constant inputs")

    estimate = _correlation(x, y, method)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(resamples):
        indices = rng.integers(0, x.size, size=x.size)
        bx, by = x[indices], y[indices]
        if np.ptp(bx) == 0 or np.ptp(by) == 0:
            continue
        samples.append(_correlation(bx, by, method))
    if not samples:
        raise ValueError("bootstrap produced no defined correlation samples")
    alpha = 1 - confidence
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return CorrelationEstimate(estimate, float(low), float(high), x.size, method)


def bootstrap_mean(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> MeanEstimate:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap mean requires at least two finite values")
    if not 0 < confidence < 1 or resamples <= 0:
        raise ValueError("bootstrap confidence/resamples are invalid")
    rng = np.random.default_rng(seed)
    samples = np.mean(
        values[rng.integers(0, values.size, size=(resamples, values.size))], axis=1
    )
    alpha = 1 - confidence
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return MeanEstimate(float(np.mean(values)), float(low), float(high), values.size)


def aggregate_seed_metrics(
    records: Iterable[Any],
    *,
    suite: str,
    condition: str,
    seeds: Sequence[int],
    task_count: int = 10,
    confidence: float = 0.95,
    resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> tuple[dict[int, dict[str, float]], dict[str, MeanEstimate]]:
    """Calculate metrics per seed, then bootstrap uncertainty across seeds."""

    from .records import records_to_success_matrix

    requested = tuple(int(value) for value in seeds)
    if len(requested) < 2 or len(set(requested)) != len(requested):
        raise ValueError("aggregate reporting requires distinct independent seeds")
    record_tuple = tuple(records)
    available = {
        record.seed
        for record in record_tuple
        if record.suite == suite and record.condition == condition
    }
    if available != set(requested):
        raise ValueError(
            f"record seeds differ from requested seeds: records={sorted(available)}, "
            f"requested={sorted(requested)}"
        )
    per_seed = {}
    for value in requested:
        matrix = records_to_success_matrix(
            record_tuple,
            suite=suite,
            seed=value,
            condition=condition,
            task_count=task_count,
        )
        per_seed[value] = continual_learning_metrics(matrix)
    names = tuple(next(iter(per_seed.values())))
    aggregate = {
        name: bootstrap_mean(
            np.asarray([per_seed[value][name] for value in requested]),
            confidence=confidence,
            resamples=resamples,
            seed=bootstrap_seed,
        )
        for name in names
    }
    return per_seed, aggregate


def continual_learning_metrics(
    matrix: np.ndarray,
    *,
    independent_baseline: np.ndarray | None = None,
) -> dict[str, float]:
    result = {
        "average_success_rate": average_success_rate(matrix),
        "negative_backward_transfer": negative_backward_transfer(matrix),
        "normalized_negative_backward_transfer": normalized_negative_backward_transfer(matrix),
        "average_forgetting": average_forgetting(matrix),
    }
    if independent_baseline is not None:
        result["forward_transfer"] = forward_transfer(matrix, independent_baseline)
    return result


def drift_degradation_correlations(
    records: Iterable[Any],
    *,
    condition: str,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, CorrelationEstimate]]:
    """Correlate drift and forgetting once per seed/suite/stage transition.

    Behavioural degradation is averaged across tasks learned before a stage.
    This avoids treating one stage's repeated drift value as independent once
    per evaluated task.
    """

    selected = [record for record in records if record.condition == condition]
    by_run: dict[tuple[str, str, int], dict[tuple[int, int], Any]] = {}
    for record in selected:
        run_key = (record.run_id, record.suite, record.seed)
        cell = (record.training_stage, record.evaluated_task_index)
        if cell in by_run.setdefault(run_key, {}):
            raise ValueError(f"duplicate evaluation cell for drift analysis: {run_key}/{cell}")
        by_run[run_key][cell] = record

    samples = []
    for run_key, cells in by_run.items():
        for stage in sorted({stage for stage, _ in cells}):
            if stage == 0:
                continue
            drops = []
            for task in range(stage):
                previous = cells.get((stage - 1, task))
                current = cells.get((stage, task))
                if previous is None or current is None:
                    raise ValueError(
                        f"incomplete consecutive cells for drift analysis: {run_key}/stage {stage}"
                    )
                drops.append(previous.success_rate - current.success_rate)
            stage_records = [cells[(stage, task)] for task in range(stage + 1)]
            drifts = {tuple(sorted(item.latent_drift.items())) for item in stage_records}
            if len(drifts) != 1:
                raise ValueError("latent drift differs across cells from one training stage")
            samples.append((dict(next(iter(drifts))), float(np.mean(drops))))
    if len(samples) < 3:
        raise ValueError("drift analysis requires at least three stage-transition samples")

    degradation = np.asarray([item[1] for item in samples], dtype=np.float64)
    result = {}
    for modality in ("vision", "language", "action"):
        drift = np.asarray([item[0][modality] for item in samples], dtype=np.float64)
        result[modality] = {
            method: bootstrap_correlation(
                drift,
                degradation,
                method=method,
                resamples=resamples,
                confidence=confidence,
                seed=seed,
            )
            for method in ("pearson", "spearman")
        }
    return result
