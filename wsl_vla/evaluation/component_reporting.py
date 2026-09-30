"""Complete-study reporting for consecutive component swaps."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..contracts import ComponentSwapRecord
from .components import SWAP_CONDITIONS
from .metrics import bootstrap_correlation
from .publication import crossed_bootstrap_mean


def _correlation_or_undefined(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    resamples: int,
    clusters: Sequence[str],
) -> dict:
    """Return a correlation result without hiding scientifically valid null cases."""
    try:
        return bootstrap_correlation(
            x, y, method=method, resamples=resamples, clusters=clusters
        ).__dict__
    except ValueError as exc:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "sample_count": int(x.size),
            "cluster_count": len(set(clusters)),
            "method": method,
            "undefined_reason": str(exc),
        }


def component_swap_study_report(
    records: Iterable[ComponentSwapRecord],
    *,
    suites: Sequence[str],
    seeds: Sequence[int],
    condition: str,
    task_count: int = 10,
    resamples: int = 10_000,
) -> dict:
    suites = tuple(suites)
    seeds = tuple(int(value) for value in seeds)
    records = tuple(item for item in records if item.condition == condition)
    grids = {
        name: np.full((len(suites), len(seeds)), np.nan) for name in SWAP_CONDITIONS
    }
    transition_samples = []
    expected_per_run = sum(stage + 1 for stage in range(1, task_count)) * len(SWAP_CONDITIONS)
    for suite_index, suite in enumerate(suites):
        for seed_index, seed in enumerate(seeds):
            selected = [item for item in records if item.suite == suite and item.seed == seed]
            if len(selected) != expected_per_run or len({item.run_id for item in selected}) != 1:
                raise ValueError(
                    f"incomplete component-swap run {suite}/seed {seed}: "
                    f"expected {expected_per_run}, found {len(selected)}"
                )
            identities = {
                (item.transition_stage, item.evaluated_task_id, item.swap_condition)
                for item in selected
            }
            expected = {
                (stage, f"{suite}_{task}", swap)
                for stage in range(1, task_count)
                for task in range(stage + 1)
                for swap in SWAP_CONDITIONS
            }
            if identities != expected:
                raise ValueError("component-swap stage/task/condition grid is invalid")
            for swap in SWAP_CONDITIONS:
                grids[swap][suite_index, seed_index] = np.mean(
                    [item.success_rate_drop for item in selected if item.swap_condition == swap]
                )
            for stage in range(1, task_count):
                stage_rows = [item for item in selected if item.transition_stage == stage]
                drifts = {tuple(sorted(item.latent_drift.items())) for item in stage_rows}
                if len(drifts) != 1:
                    raise ValueError("component drift differs within one transition")
                by_swap = {
                    swap: float(
                        np.mean(
                            [item.success_rate_drop for item in stage_rows if item.swap_condition == swap]
                        )
                    )
                    for swap in SWAP_CONDITIONS
                }
                transition_samples.append(
                    {
                        "suite": suite,
                        "seed": seed,
                        "transition_stage": stage,
                        "cluster": f"{suite}:{seed}",
                        "latent_drift": dict(next(iter(drifts))),
                        "success_rate_drop": by_swap,
                    }
                )
    correlations = {}
    clusters = [item["cluster"] for item in transition_samples]
    for modality in ("vision", "language", "action"):
        x = np.asarray([item["latent_drift"][modality] for item in transition_samples])
        y = np.asarray(
            [item["success_rate_drop"][modality] for item in transition_samples]
        )
        correlations[modality] = {
            method: _correlation_or_undefined(
                x,
                y,
                method=method,
                resamples=resamples,
                clusters=clusters,
            )
            for method in ("pearson", "spearman")
        }
    return {
        "schema_version": 1,
        "design": {
            "suites": list(suites),
            "seeds": list(seeds),
            "condition": condition,
            "records_per_run": expected_per_run,
            "transition_sample_count": len(transition_samples),
        },
        "aggregate_success_rate_drop": {
            name: crossed_bootstrap_mean(grid, resamples=resamples)
            for name, grid in grids.items()
        },
        "drift_drop_correlations": correlations,
        "transition_samples": transition_samples,
    }
