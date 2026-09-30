import json

from wsl_vla.experiments.study_audit import audit_methodology1_study


def test_study_audit_is_machine_readable_and_fail_closed(tmp_path):
    report = audit_methodology1_study(tmp_path)
    assert report["ready"] is False
    assert report["passed_checks"] < report["total_checks"]
    assert {check["name"] for check in report["checks"]} >= {
        "population_verification",
        "environment_lock",
        "four_alignment_folds",
        "development_pilot_24_run_grid",
        "continual_96_run_grid",
        "ood_12_run_grid",
        "component_swap_12_run_grid",
        "recovery_12_run_grid",
        "publication_reports",
        "publication_provenance_consistency",
    }
    json.dumps(report)
