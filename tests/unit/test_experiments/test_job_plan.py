"""The Methodology 1 plan is the corrected core plan plus a re-pointed downstream study."""

import pytest

from tests.synthetic import synthetic_manifest
from wsl_vla.experiments.job_plan import (
    DOWNSTREAM_STAGES,
    build_methodology1_job_plan,
    summarize_job_plan,
)
from wsl_vla.experiments.manifest import load_experiment_manifest, manifest_from_dict
from wsl_vla.experiments.protocol import REQUIRED_SUITES, load_yaml
from wsl_vla.experiments.zoo_jobs import build_experiment_job_plan


def full_manifest(*, frozen: bool = True):
    """The shipped full profile; its budget is frozen only after validation."""

    tasks = load_yaml("configs/reference_tasks.yaml")
    manifest = load_experiment_manifest(
        "configs/research/manifests/full.yaml", reference_tasks=tasks
    )
    if not frozen:
        return manifest
    payload = manifest.to_dict()
    payload["zoo"]["frozen"] = True
    payload["zoo"]["inspection_updates"] = []
    payload["zoo"]["inspection_tasks"] = []
    frozen_manifest = manifest_from_dict(payload)
    frozen_manifest.validate()
    return frozen_manifest


def plan(tmp_path, manifest):
    return build_methodology1_job_plan(
        manifest,
        manifest_path="configs/research/manifests/full.yaml",
        research_root=tmp_path / "results",
        gamma_candidates=load_yaml("configs/research/gamma_candidates.yaml"),
    )


def test_plan_is_deterministic_dependency_closed_and_extends_the_core_plan(tmp_path):
    manifest = full_manifest()
    first = plan(tmp_path, manifest)
    assert first == plan(tmp_path, manifest)

    core = build_experiment_job_plan(
        manifest,
        manifest_path="configs/research/manifests/full.yaml",
        research_root=tmp_path / "results",
    )
    assert first[: len(core)] == core
    downstream = first[len(core):]
    assert all(job["job_id"].startswith("m1:") for job in downstream)
    assert {job["stage"] for job in downstream} == set(DOWNSTREAM_STAGES)

    summary = summarize_job_plan(first)
    counts = manifest.derived_counts()
    stages = summary["jobs_by_stage"]
    assert stages["zoo"] == counts["independent_runs"] == 120
    assert stages["evidence"] == 40
    assert stages["gamma_validation"] == 864
    assert stages["pilot"] == 24
    assert stages["continual"] == 96
    assert stages["ood"] == 12
    assert "have not been executed" in summary["downstream_status"]
    ids = {job["job_id"] for job in first}
    assert len(ids) == len(first)
    assert all(set(job["dependencies"]) <= ids for job in first)
    audit = next(job for job in first if job["job_id"] == "m1:audit")
    assert {"m1:report:continual", "report:rollouts"} <= set(audit["dependencies"])
    final = next(job for job in first if job["job_id"].startswith("m1:continual:"))
    assert len([item for item in final["dependencies"] if item.startswith("m1:pilot:")]) == 24


def test_downstream_waits_for_alignment_integrity_not_for_alignment_winning(tmp_path):
    manifest = full_manifest()
    jobs = plan(tmp_path, manifest)
    commands = " ".join(token for job in jobs for token in job["command"])

    # The retired "must win" gate is neither run nor depended on.
    assert "run_alignment_advantage_gate" not in commands
    assert "alignment_advantage" not in commands
    seed = manifest.alignment.seeds[0]
    for suite in REQUIRED_SUITES:
        fold = next(item for item in manifest.folds if item.held_out_suite == suite)
        integrity = f"diagnostics:validation:{fold.name}:seed-{seed}"
        bank = next(job for job in jobs if job["job_id"] == f"m1:ood_bank:{suite}")
        assert bank["dependencies"] == [integrity]
        gamma = next(job for job in jobs if job["job_id"].startswith(f"m1:gamma:{suite}:"))
        assert integrity in gamma["dependencies"]
        # Downstream jobs read the corrected experiment layout.
        checkpoint = gamma["command"][gamma["command"].index("--alignment-checkpoint") + 1]
        assert checkpoint.endswith(f"experiments/full/alignment/{fold.name}/seed_{seed}/aligned")
        evidence = gamma["command"][gamma["command"].index("--evidence-root") + 1]
        assert evidence.endswith("experiments/full/evidence")
    # Nothing downstream hard-codes validation indices or the population size.
    assert "--validation-task-indices" not in commands


def test_downstream_study_requires_a_frozen_budget_and_every_fold(tmp_path):
    with pytest.raises(ValueError, match="frozen zoo budget"):
        plan(tmp_path, full_manifest(frozen=False))
    with pytest.raises(ValueError, match="one leave-one-suite-out fold per reference suite"):
        plan(tmp_path, synthetic_manifest(frozen=True))
