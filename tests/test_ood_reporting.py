from dataclasses import replace

import pytest

from wsl_vla.contracts import OODEvaluationRecord
from wsl_vla.evaluation.ood import OOD_METHODS
from wsl_vla.evaluation.ood_reporting import ood_study_report


def records():
    result = []
    for suite in ("suite_a", "suite_b"):
        for seed in (17, 42):
            for task in range(2):
                for method_index, method in enumerate(OOD_METHODS):
                    result.append(
                        OODEvaluationRecord(
                            run_id=f"{suite}-{seed}",
                            held_out_suite=suite,
                            task_id=f"{suite}_{task}",
                            seed=seed,
                            method=method,
                            adaptation_steps=(20 if method_index in (1, 2) else 0),
                            rollout_count=4,
                            successes=method_index % 4,
                            checkpoint_sha256="a" * 64,
                            wall_time_seconds=1.0,
                            initialization_indices=(0, 1, 2, 3),
                            rollout_seeds=(10, 11, 12, 13),
                            source_training_suites=("x", "y", "z"),
                            alignment_checkpoint_sha256="b" * 64,
                        )
                    )
    return tuple(result)


def test_ood_report_requires_complete_factorial_grid():
    report = ood_study_report(
        records(), suites=("suite_a", "suite_b"), seeds=(17, 42), task_count=2, resamples=20
    )
    assert report["design"]["expected_record_count"] == 40
    assert report["aggregate_average_success_rate"][OOD_METHODS[1]]["estimate"] == 0.25
    with pytest.raises(ValueError, match="incomplete OOD run"):
        ood_study_report(
            records()[:-1],
            suites=("suite_a", "suite_b"),
            seeds=(17, 42),
            task_count=2,
            resamples=10,
        )


def test_ood_report_rejects_mixed_run_ids():
    values = list(records())
    values[0] = replace(values[0], run_id="mixed")
    with pytest.raises(ValueError, match="multiple runs"):
        ood_study_report(
            values,
            suites=("suite_a", "suite_b"),
            seeds=(17, 42),
            task_count=2,
            resamples=10,
        )
