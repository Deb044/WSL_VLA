from wsl_vla.experiments.job_plan import build_methodology1_job_plan, summarize_job_plan
from wsl_vla.experiments.protocol import REQUIRED_SUITES, load_yaml


def test_full_job_plan_is_complete_deterministic_and_dependency_closed(tmp_path):
    data_files = {
        suite: tuple(tmp_path / suite / f"task_{index}.hdf5" for index in range(10))
        for suite in REQUIRED_SUITES
    }
    candidates = load_yaml("configs/research/gamma_candidates.yaml")
    first = build_methodology1_job_plan(
        research_root=tmp_path / "results",
        data_files=data_files,
        gamma_candidates=candidates,
    )
    second = build_methodology1_job_plan(
        research_root=tmp_path / "results",
        data_files=data_files,
        gamma_candidates=candidates,
    )
    assert first == second
    summary = summarize_job_plan(first)
    assert summary["job_count"] == 1213
    assert summary["jobs_by_stage"]["evidence"] == 40
    assert summary["jobs_by_stage"]["model_zoo"] == 121
    assert summary["jobs_by_stage"]["gamma_validation"] == 864
    assert summary["jobs_by_stage"]["pilot"] == 24
    assert summary["jobs_by_stage"]["continual"] == 96
    ids = {job["job_id"] for job in first}
    assert all(set(job["dependencies"]) <= ids for job in first)
    audit = next(job for job in first if job["job_id"] == "audit")
    assert "report:continual" in audit["dependencies"]
    final = next(job for job in first if job["job_id"].startswith("continual:"))
    assert len([item for item in final["dependencies"] if item.startswith("pilot:")]) == 24
