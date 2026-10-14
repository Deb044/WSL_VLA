import json

from tests.unit.test_experiments.test_job_plan import full_manifest
from wsl_vla.experiments.manifest import ExperimentPaths
from wsl_vla.experiments.study_audit import audit_methodology1_study


def test_study_audit_is_machine_readable_and_fail_closed(tmp_path):
    manifest = full_manifest()
    report = audit_methodology1_study(tmp_path, manifest)

    assert report["ready"] is False
    assert report["passed_checks"] < report["total_checks"]
    assert report["manifest_sha256"] == manifest.sha256()
    assert {check["name"] for check in report["checks"]} >= {
        "preflight",
        "camera_orientation_verified",
        "population_verification",
        "alignment_folds_complete_with_integrity",
        "development_pilot_run_grid",
        "continual_run_grid",
        "ood_run_grid",
        "component_swap_run_grid",
        "recovery_run_grid",
        "publication_reports",
        "publication_provenance_consistency",
    }
    grids = {check["name"]: check for check in report["checks"]}
    assert grids["development_pilot_run_grid"]["expected_runs"] == 24
    assert grids["continual_run_grid"]["expected_runs"] == 96
    assert grids["ood_run_grid"]["expected_runs"] == 12
    assert grids["population_verification"]["expected_samples"] == 360
    json.dumps(report)


def test_audit_requires_integrity_and_records_outcomes_without_requiring_a_win(tmp_path):
    manifest = full_manifest()
    paths = ExperimentPaths(tmp_path, manifest.name)
    fold = manifest.folds[0]
    seed = manifest.alignment.seeds[0]

    def write_report(scope, status, evidence):
        directory = paths.diagnostics(fold.name, seed, scope)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "alignment_pair.json").write_text(
            json.dumps(
                {
                    "integrity": {"status": status},
                    "comparisons": {"mapped.update_error": {"evidence": evidence}},
                }
            )
        )

    def alignment_check():
        report = audit_methodology1_study(tmp_path, manifest)
        return next(
            check
            for check in report["checks"]
            if check["name"] == "alignment_folds_complete_with_integrity"
        )

    # A negative finding with passing integrity is recorded, not reported as a problem.
    write_report("validation", "pass", "negative_finding")
    check = alignment_check()
    key = f"{fold.name}/seed_{seed}/validation"
    report_path = str(paths.diagnostics(fold.name, seed, "validation") / "alignment_pair.json")
    assert check["recorded_outcomes"][key] == {"mapped.update_error": "negative_finding"}
    assert not any(report_path in problem for problem in check["problems"])

    # An integrity failure is a problem even if the comparison looks favourable.
    write_report("validation", "fail", "supports_first_condition")
    check = alignment_check()
    assert key not in check["recorded_outcomes"]
    assert any(
        "integrity is not 'pass'" in problem and report_path in problem
        for problem in check["problems"]
    )
