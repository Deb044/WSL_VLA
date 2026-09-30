"""Completeness checks and crossed-fold summaries for the final continual study."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..contracts import EvaluationRecord
from .metrics import continual_learning_metrics
from .records import records_to_success_matrix


def crossed_bootstrap_mean(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    """Bootstrap suites and training seeds as crossed experimental factors."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError("crossed bootstrap needs at least two suites and two seeds")
    finite = np.isfinite(array)
    finite_count = int(finite.sum())
    if finite_count == 0:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "sample_count": 0,
            "excluded_undefined": int(array.size),
        }
    estimate = float(np.mean(array[finite]))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(resamples):
        suite_indices = rng.integers(0, array.shape[0], size=array.shape[0])
        seed_indices = rng.integers(0, array.shape[1], size=array.shape[1])
        sample = array[np.ix_(suite_indices, seed_indices)]
        sample = sample[np.isfinite(sample)]
        if sample.size:
            samples.append(float(np.mean(sample)))
    if not samples:
        raise ValueError("crossed bootstrap produced no defined samples")
    alpha = 1 - confidence
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "sample_count": finite_count,
        "excluded_undefined": int(array.size - finite_count),
    }


def _run_resources(
    records: Sequence[EvaluationRecord], *, task_count: int, oracle: bool
) -> dict[str, float | int | None]:
    by_cell = {(item.training_stage, item.evaluated_task_index): item for item in records}
    diagonal = [by_cell[(stage, stage)] for stage in range(task_count)]
    peak_vram = [item.peak_vram_bytes for item in records if item.peak_vram_bytes is not None]
    peak_ram = [item.peak_ram_bytes for item in records if item.peak_ram_bytes is not None]
    if oracle:
        resident_adapter_bytes = sum(item.adapter_bytes for item in diagonal)
    else:
        resident_adapter_bytes = diagonal[-1].adapter_bytes
    return {
        "training_wall_time_seconds": float(sum(item.wall_time_seconds for item in diagonal)),
        "evaluation_wall_time_seconds": float(
            sum(item.evaluation_wall_time_seconds for item in records)
        ),
        "resident_adapter_bytes": int(resident_adapter_bytes),
        "adapter_bytes_per_task": float(resident_adapter_bytes / task_count),
        "evidence_bytes_per_task": float(
            np.mean([item.evidence_bytes for item in diagonal])
        ),
        "peak_vram_bytes": max(peak_vram) if peak_vram else None,
        "peak_ram_bytes": max(peak_ram) if peak_ram else None,
        "training_update_steps": int(sum(item.training_update_steps for item in diagonal)),
        "training_micro_steps": int(sum(item.training_micro_steps for item in diagonal)),
    }


def publication_study_report(
    records: Iterable[EvaluationRecord],
    *,
    suites: Sequence[str],
    seeds: Sequence[int],
    conditions: Sequence[str],
    oracle_condition: str,
    task_count: int = 10,
    confidence: float = 0.95,
    resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict:
    """Require the complete factorial study and summarize without silent omissions."""

    suite_values = tuple(suites)
    seed_values = tuple(int(value) for value in seeds)
    condition_values = tuple(conditions)
    if min(len(suite_values), len(seed_values), len(condition_values), task_count) <= 0:
        raise ValueError("publication study axes cannot be empty")
    if len(set(suite_values)) != len(suite_values):
        raise ValueError("publication suites must be unique")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("publication seeds must be unique")
    if len(set(condition_values)) != len(condition_values):
        raise ValueError("publication conditions must be unique")
    if oracle_condition not in condition_values:
        raise ValueError("oracle condition must be included in the publication conditions")
    record_tuple = tuple(records)
    expected_records = task_count * (task_count + 1) // 2
    runs = {}
    for condition in condition_values:
        for suite in suite_values:
            for seed in seed_values:
                key = (condition, suite, seed)
                selected = tuple(
                    item
                    for item in record_tuple
                    if item.suite == suite
                    and item.seed == seed
                    and item.condition == condition
                )
                if len(selected) != expected_records:
                    raise ValueError(
                        f"incomplete run {condition}/{suite}/seed {seed}: "
                        f"expected {expected_records} records, found {len(selected)}"
                    )
                run_ids = {item.run_id for item in selected}
                rollout_counts = {item.rollout_count for item in selected}
                if len(run_ids) != 1 or len(rollout_counts) != 1:
                    raise ValueError(
                        f"run identity or rollout budget changed within {condition}/{suite}/seed {seed}"
                    )
                runs[key] = selected

    per_run: dict[str, dict[str, dict[str, dict]]] = {}
    metric_grids: dict[str, dict[str, np.ndarray]] = {}
    for condition in condition_values:
        metric_grids[condition] = {}
        per_run[condition] = {}
        for suite_index, suite in enumerate(suite_values):
            per_run[condition][suite] = {}
            for seed_index, seed in enumerate(seed_values):
                selected = runs[(condition, suite, seed)]
                matrix = records_to_success_matrix(
                    selected,
                    suite=suite,
                    seed=seed,
                    condition=condition,
                    task_count=task_count,
                )
                oracle_matrix = records_to_success_matrix(
                    record_tuple,
                    suite=suite,
                    seed=seed,
                    condition=oracle_condition,
                    task_count=task_count,
                )
                metrics = continual_learning_metrics(
                    matrix, independent_baseline=np.diag(oracle_matrix)
                )
                for name, value in metrics.items():
                    metric_grids[condition].setdefault(
                        name,
                        np.full((len(suite_values), len(seed_values)), np.nan),
                    )[suite_index, seed_index] = value
                per_run[condition][suite][str(seed)] = {
                    "metrics": metrics,
                    "resources": _run_resources(
                        selected,
                        task_count=task_count,
                        oracle=condition == oracle_condition,
                    ),
                }

    aggregate = {}
    for condition, grids in metric_grids.items():
        aggregate[condition] = {
            name: crossed_bootstrap_mean(
                values,
                confidence=confidence,
                resamples=resamples,
                seed=bootstrap_seed,
            )
            for name, values in grids.items()
        }
    return {
        "schema_version": 1,
        "design": {
            "suites": list(suite_values),
            "seeds": list(seed_values),
            "conditions": list(condition_values),
            "task_count": task_count,
            "records_per_run": task_count * (task_count + 1) // 2,
            "expected_run_count": len(suite_values) * len(seed_values) * len(condition_values),
            "confidence": confidence,
            "bootstrap_resamples": resamples,
            "bootstrap_unit": "crossed suite and training-seed resampling",
            "normalized_nbt_zero_handling": (
                "tasks with zero diagonal success are excluded within a run; "
                "runs with no defined tasks are reported as undefined"
            ),
        },
        "per_run": per_run,
        "aggregate": aggregate,
    }
