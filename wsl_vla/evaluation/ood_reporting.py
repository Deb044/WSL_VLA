"""Complete-study OOD validation and crossed-fold aggregation."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..contracts import OODEvaluationRecord
from .ood import OOD_METHODS
from .publication import crossed_bootstrap_mean


def ood_study_report(
    records: Iterable[OODEvaluationRecord],
    *,
    suites: Sequence[str],
    seeds: Sequence[int],
    task_count: int = 10,
    confidence: float = 0.95,
    resamples: int = 10_000,
) -> dict:
    suite_values = tuple(suites)
    seed_values = tuple(int(value) for value in seeds)
    values = tuple(records)
    grids = {
        method: np.full((len(suite_values), len(seed_values)), np.nan)
        for method in OOD_METHODS
    }
    per_run = {}
    for suite_index, suite in enumerate(suite_values):
        per_run[suite] = {}
        for seed_index, seed in enumerate(seed_values):
            selected = [
                item for item in values if item.held_out_suite == suite and item.seed == seed
            ]
            expected = task_count * len(OOD_METHODS)
            if len(selected) != expected:
                raise ValueError(
                    f"incomplete OOD run {suite}/seed {seed}: expected {expected}, found {len(selected)}"
                )
            if len({item.run_id for item in selected}) != 1:
                raise ValueError("OOD cells for one suite/seed combine multiple runs")
            identities = {(item.task_id, item.method) for item in selected}
            expected_identities = {
                (f"{suite}_{task}", method)
                for task in range(task_count)
                for method in OOD_METHODS
            }
            if identities != expected_identities:
                raise ValueError("OOD task/method grid is incomplete or duplicated")
            method_rates = {}
            for method in OOD_METHODS:
                method_records = [item for item in selected if item.method == method]
                rate = float(np.mean([item.success_rate for item in method_records]))
                method_rates[method] = rate
                grids[method][suite_index, seed_index] = rate
            per_run[suite][str(seed)] = method_rates
    return {
        "schema_version": 1,
        "design": {
            "suites": list(suite_values),
            "seeds": list(seed_values),
            "methods": list(OOD_METHODS),
            "task_count": task_count,
            "expected_record_count": len(suite_values)
            * len(seed_values)
            * task_count
            * len(OOD_METHODS),
            "bootstrap_unit": "crossed held-out suite and training-seed resampling",
        },
        "per_run_average_success_rate": per_run,
        "aggregate_average_success_rate": {
            method: crossed_bootstrap_mean(
                grid, confidence=confidence, resamples=resamples
            )
            for method, grid in grids.items()
        },
    }
