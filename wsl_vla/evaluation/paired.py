"""Task-balanced summaries, paired differences, and clustered uncertainty.

Uncertainty is aggregated by **task** and by **training run**. Snapshots of one
run share an optimization trajectory and are averaged inside their run before
anything is resampled; they are never counted as independent seeds.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


EVIDENCE_SUPPORTS = "supports_first_condition"
EVIDENCE_AGAINST = "negative_finding"
EVIDENCE_INCONCLUSIVE = "inconclusive"


def task_balanced_mean(values: Mapping[str, Mapping[str, Sequence[float]]]) -> float | None:
    """Mean over tasks of the mean over runs of the mean over repeated outcomes.

    ``values[task][run]`` lists that run's outcomes (episodes, or snapshots of
    one run). Returns ``None`` when no task has any outcome.
    """

    task_means = []
    for runs in values.values():
        run_means = [float(np.mean(outcomes)) for outcomes in runs.values() if len(outcomes)]
        if run_means:
            task_means.append(float(np.mean(run_means)))
    return float(np.mean(task_means)) if task_means else None


def clustered_bootstrap(
    values: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Two-level bootstrap of a task-balanced mean: tasks, then runs within a task.

    The interval is reported as invalid, with the reason, when the design cannot
    support it: fewer than two tasks means between-task variability is
    unobserved, and an interval is not manufactured from episodes alone.
    """

    if resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("bootstrap needs positive resamples and a confidence in (0, 1)")
    tasks = [
        {run: np.asarray(outcomes, dtype=np.float64) for run, outcomes in runs.items() if len(outcomes)}
        for runs in values.values()
    ]
    tasks = [runs for runs in tasks if runs]
    estimate = task_balanced_mean(values)
    result = {
        "mean": estimate,
        "tasks": len(tasks),
        "runs_per_task": sorted({len(runs) for runs in tasks}),
        "outcomes": int(sum(len(o) for runs in tasks for o in runs.values())),
        "resamples": int(resamples),
        "confidence": float(confidence),
        "ci_low": None,
        "ci_high": None,
        "interval_valid": False,
        "interval_note": None,
    }
    if len(tasks) < 2:
        result["interval_note"] = "fewer than two tasks: between-task uncertainty is unobserved"
        return result
    rng = np.random.default_rng(seed)
    run_means = [np.asarray([outcomes.mean() for outcomes in runs.values()]) for runs in tasks]
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(len(run_means), size=len(run_means))
        draws[index] = np.mean(
            [rng.choice(run_means[task], size=len(run_means[task]), replace=True).mean() for task in chosen]
        )
    alpha = (1.0 - confidence) / 2.0
    result.update(
        ci_low=float(np.quantile(draws, alpha)),
        ci_high=float(np.quantile(draws, 1.0 - alpha)),
        interval_valid=True,
        interval_note=(
            "single training run per task: run-to-run variability is not represented"
            if max(result["runs_per_task"]) < 2
            else None
        ),
    )
    return result


def paired_differences(
    first: Mapping[str, Mapping[str, Sequence[float]]],
    second: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, dict[str, list[float]]]:
    """Outcome-wise ``first - second`` on identical tasks, runs, and repetitions."""

    if set(first) != set(second):
        raise ValueError("paired conditions were not evaluated on the same tasks")
    differences: dict[str, dict[str, list[float]]] = {}
    for task in first:
        if set(first[task]) != set(second[task]):
            raise ValueError(f"paired conditions differ in training runs for task {task}")
        differences[task] = {}
        for run in first[task]:
            left, right = np.asarray(first[task][run]), np.asarray(second[task][run])
            if left.shape != right.shape:
                raise ValueError(f"paired outcomes differ in length for {task}/{run}")
            differences[task][run] = (left - right).tolist()
    return differences


def classify_evidence(
    summary: Mapping[str, Any], *, higher_is_better: bool, minimum_tasks: int
) -> dict[str, Any]:
    """Label a paired difference without ever turning it into a pass/fail gate.

    * ``inconclusive``: too few tasks, no valid interval, or an interval that
      contains zero - the data do not distinguish the conditions;
    * ``supports_first_condition``: the interval excludes zero in the first
      condition's favour;
    * ``negative_finding``: the interval excludes zero against it - a real
      scientific result, not an infrastructure failure.
    """

    reasons = []
    if summary["mean"] is None:
        reasons.append("no defined paired outcomes")
    if summary["tasks"] < minimum_tasks:
        reasons.append(f"{summary['tasks']} tasks is below the declared minimum of {minimum_tasks}")
    if not summary["interval_valid"]:
        reasons.append(summary["interval_note"] or "no valid interval")
    if reasons:
        return {"evidence": EVIDENCE_INCONCLUSIVE, "reasons": reasons}
    low, high = summary["ci_low"], summary["ci_high"]
    if low <= 0.0 <= high:
        return {
            "evidence": EVIDENCE_INCONCLUSIVE,
            "reasons": ["the confidence interval of the paired difference contains zero"],
        }
    favourable = low > 0.0 if higher_is_better else high < 0.0
    return {
        "evidence": EVIDENCE_SUPPORTS if favourable else EVIDENCE_AGAINST,
        "reasons": ["the confidence interval of the paired difference excludes zero"],
    }
