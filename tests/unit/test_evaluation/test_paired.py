"""Task-balanced means, paired differences, and task/run-clustered intervals."""

import pytest

from wsl_vla.evaluation.generated_policies import (
    assert_paired_episodes,
    outcomes_by_condition,
    summarize_rollouts,
)
from wsl_vla.evaluation.paired import (
    classify_evidence,
    clustered_bootstrap,
    paired_differences,
    task_balanced_mean,
)


def test_task_balanced_mean_does_not_let_large_tasks_or_snapshots_dominate():
    values = {
        "task_a": {"run_1": [1.0] * 50, "run_2": [1.0] * 50},
        "task_b": {"run_1": [0.0]},
    }
    assert task_balanced_mean(values) == pytest.approx(0.5)
    # Snapshots of one run are averaged inside the run, never counted as seeds.
    correlated = {"task_a": {"run_1": [1.0, 1.0, 1.0], "run_2": [0.0]}}
    assert task_balanced_mean(correlated) == pytest.approx(0.5)
    assert task_balanced_mean({}) is None


def test_interval_is_not_manufactured_from_a_single_task():
    single = clustered_bootstrap(
        {"task_a": {"run_1": [1.0, 0.0, 1.0, 1.0]}}, resamples=100, confidence=0.95, seed=0
    )
    assert single["mean"] == pytest.approx(0.75)
    assert not single["interval_valid"] and single["ci_low"] is None
    assert "fewer than two tasks" in single["interval_note"]
    several = clustered_bootstrap(
        {f"task_{i}": {"run_1": [float(i % 2)] * 4} for i in range(8)},
        resamples=500, confidence=0.95, seed=0,
    )
    assert several["interval_valid"] and several["ci_low"] <= 0.5 <= several["ci_high"]
    assert several["tasks"] == 8 and several["runs_per_task"] == [1]
    assert "single training run per task" in several["interval_note"]


def test_bootstrap_is_reproducible_and_clustered_by_task():
    values = {
        f"task_{i}": {f"run_{r}": [float((i + r) % 3 == 0)] * 5 for r in range(3)} for i in range(6)
    }
    first = clustered_bootstrap(values, resamples=300, confidence=0.9, seed=7)
    second = clustered_bootstrap(values, resamples=300, confidence=0.9, seed=7)
    assert first == second and first["interval_note"] is None
    # Repeating every episode 10x adds no independent evidence: the interval,
    # which resamples tasks and runs, does not shrink.
    repeated = {
        task: {run: list(outcomes) * 10 for run, outcomes in runs.items()}
        for task, runs in values.items()
    }
    wider = clustered_bootstrap(repeated, resamples=300, confidence=0.9, seed=7)
    assert wider["ci_high"] - wider["ci_low"] == pytest.approx(first["ci_high"] - first["ci_low"])


def test_paired_differences_require_identical_tasks_runs_and_lengths():
    first = {"a": {"r": [1.0, 0.0]}, "b": {"r": [1.0, 1.0]}}
    second = {"a": {"r": [0.0, 0.0]}, "b": {"r": [1.0, 0.0]}}
    assert paired_differences(first, second) == {"a": {"r": [1.0, 0.0]}, "b": {"r": [0.0, 1.0]}}
    with pytest.raises(ValueError, match="same tasks"):
        paired_differences(first, {"a": {"r": [0.0, 0.0]}})
    with pytest.raises(ValueError, match="training runs"):
        paired_differences(first, {"a": {"x": [0.0, 0.0]}, "b": {"r": [1.0, 0.0]}})
    with pytest.raises(ValueError, match="differ in length"):
        paired_differences(first, {"a": {"r": [0.0]}, "b": {"r": [1.0, 0.0]}})


def summary(mean, low, high, tasks=8, valid=True, note=None):
    return {"mean": mean, "ci_low": low, "ci_high": high, "tasks": tasks,
            "interval_valid": valid, "interval_note": note}


def test_outcomes_are_labelled_without_becoming_a_pass_fail_gate():
    assert classify_evidence(summary(0.2, 0.05, 0.3), higher_is_better=True, minimum_tasks=6)[
        "evidence"
    ] == "supports_first_condition"
    assert classify_evidence(summary(-0.2, -0.3, -0.05), higher_is_better=True, minimum_tasks=6)[
        "evidence"
    ] == "negative_finding"
    assert classify_evidence(summary(0.1, -0.05, 0.3), higher_is_better=True, minimum_tasks=6)[
        "evidence"
    ] == "inconclusive"
    # For an error metric a negative difference favours the first condition.
    assert classify_evidence(summary(-0.2, -0.3, -0.05), higher_is_better=False, minimum_tasks=6)[
        "evidence"
    ] == "supports_first_condition"
    few = classify_evidence(summary(0.5, 0.4, 0.6, tasks=3), higher_is_better=True, minimum_tasks=6)
    assert few["evidence"] == "inconclusive" and "below the declared minimum" in few["reasons"][0]
    invalid = classify_evidence(
        summary(0.5, None, None, tasks=1, valid=False, note="fewer than two tasks"),
        higher_is_better=True, minimum_tasks=1,
    )
    assert invalid["evidence"] == "inconclusive"
    undefined = classify_evidence(
        summary(None, None, None, tasks=0, valid=False), higher_is_better=True, minimum_tasks=1
    )
    assert undefined["evidence"] == "inconclusive"


def episode(condition, task, index, success, clipped=0):
    return {
        "condition": condition, "task_id": task, "training_run": f"{task}/seed_17|alignment_seed_0",
        "initialization_index": index, "episode_seed": 1700 + index, "success": success,
        "steps": 10, "clipped_steps": clipped,
    }


def test_rollout_summary_is_task_balanced_paired_and_reports_clipping():
    records = []
    for task, base, adapter in (("t0", [0, 0, 0, 0], [1, 1, 1, 0]), ("t1", [1, 0, 0, 0], [1, 1, 0, 0])):
        for index, (b, a) in enumerate(zip(base, adapter)):
            records.append(episode("frozen_base", task, index, bool(b)))
            records.append(episode("zoo_adapter", task, index, bool(a), clipped=2))
    assert_paired_episodes(records)
    assert outcomes_by_condition(records)["zoo_adapter"]["t0"] == {
        "t0/seed_17|alignment_seed_0": [1.0, 1.0, 1.0, 0.0]
    }
    report = summarize_rollouts(records, resamples=200, confidence=0.95, seed=0, minimum_tasks=2)
    assert report["conditions"]["zoo_adapter"]["task_balanced_success_rate"]["mean"] == pytest.approx(0.625)
    assert report["conditions"]["frozen_base"]["task_balanced_success_rate"]["mean"] == pytest.approx(0.125)
    assert report["conditions"]["zoo_adapter"]["controller_clipping"]["clipped_step_fraction"] == pytest.approx(0.2)
    paired = report["paired_differences"]["zoo_adapter - frozen_base"]
    assert paired["paired_difference"]["mean"] == pytest.approx(0.5)
    assert paired["evidence"] in {"inconclusive", "supports_first_condition", "negative_finding"}
    # A condition evaluated on other initial states cannot be paired.
    records.append(episode("mapper[aligned]", "t0", 9, True))
    with pytest.raises(ValueError, match="same initial states and seeds"):
        assert_paired_episodes(records)
