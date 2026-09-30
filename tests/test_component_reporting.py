import pytest

from wsl_vla.contracts import ComponentSwapRecord
from wsl_vla.evaluation.component_reporting import component_swap_study_report
from wsl_vla.evaluation.components import SWAP_CONDITIONS


def records():
    rows = []
    for suite_index, suite in enumerate(("suite_a", "suite_b")):
        for seed_index, seed in enumerate((17, 42)):
            for stage in (1, 2):
                drift = 0.1 + 0.1 * (suite_index + seed_index + stage)
                for task in range(stage + 1):
                    for swap_index, swap in enumerate(SWAP_CONDITIONS):
                        drop = min(9, int(round(10 * drift)) + swap_index)
                        rows.append(
                            ComponentSwapRecord(
                                run_id=f"{suite}-{seed}",
                                suite=suite,
                                seed=seed,
                                condition="proposed_asymmetric",
                                transition_stage=stage,
                                evaluated_task_id=f"{suite}_{task}",
                                swap_condition=swap,
                                rollout_count=10,
                                baseline_successes=10,
                                swapped_successes=10 - drop,
                                initialization_indices=tuple(range(10)),
                                previous_checkpoint_sha256="a" * 64,
                                current_checkpoint_sha256="b" * 64,
                                swapped_checkpoint_sha256="c" * 64,
                                latent_drift={
                                    "vision": drift,
                                    "language": drift * 2,
                                    "action": drift * 3,
                                },
                                rollout_seeds=tuple(range(10, 20)),
                                baseline_wall_time_seconds=1.0,
                                swapped_wall_time_seconds=1.0,
                            )
                        )
    return tuple(rows)


def test_component_report_requires_complete_runs_and_reports_correlations():
    report = component_swap_study_report(
        records(),
        suites=("suite_a", "suite_b"),
        seeds=(17, 42),
        condition="proposed_asymmetric",
        task_count=3,
        resamples=20,
    )
    assert report["design"]["records_per_run"] == 25
    assert report["design"]["transition_sample_count"] == 8
    assert len(report["transition_samples"]) == 8
    assert report["drift_drop_correlations"]["vision"]["pearson"]["sample_count"] == 8
    with pytest.raises(ValueError, match="incomplete component-swap run"):
        component_swap_study_report(
            records()[:-1],
            suites=("suite_a", "suite_b"),
            seeds=(17, 42),
            condition="proposed_asymmetric",
            task_count=3,
            resamples=10,
        )


def test_component_report_preserves_undefined_constant_correlation():
    rows = []
    for row in records():
        rows.append(
            ComponentSwapRecord(
                **{
                    **row.__dict__,
                    "latent_drift": {"vision": 1.0, "language": 1.0, "action": 1.0},
                }
            )
        )
    report = component_swap_study_report(
        rows,
        suites=("suite_a", "suite_b"),
        seeds=(17, 42),
        condition="proposed_asymmetric",
        task_count=3,
        resamples=10,
    )
    result = report["drift_drop_correlations"]["vision"]["pearson"]
    assert result["estimate"] is None
    assert result["undefined_reason"]
