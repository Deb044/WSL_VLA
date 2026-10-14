"""Every planned command must be accepted by the script it names.

A job plan is only useful if its commands run. These tests build each script's
argument parser for real (``--help`` in a subprocess) and check that every flag
a job plan emits exists in that parser, so a renamed or conflicting option is
caught here instead of hours into a collaborator's run.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

pytest.importorskip("jax")

from tests.synthetic import synthetic_manifest  # noqa: E402
from tests.unit.test_experiments.test_job_plan import full_manifest  # noqa: E402
from wsl_vla.experiments.job_plan import build_methodology1_job_plan  # noqa: E402
from wsl_vla.experiments.manifest import load_experiment_manifest  # noqa: E402
from wsl_vla.experiments.protocol import load_yaml  # noqa: E402
from wsl_vla.experiments.zoo_jobs import build_experiment_job_plan  # noqa: E402

pytestmark = pytest.mark.research

REPO_ROOT = Path(__file__).resolve().parents[3]
# Entry points a collaborator runs by hand rather than through a job plan.
MANUAL_SCRIPTS = (
    "scripts/reporting/plan_zoo_alignment_jobs.py",
    "scripts/reporting/plan_methodology1_jobs.py",
    "scripts/train/run_job_manifest.py",
    "scripts/train/orchestrate_research_zoo.py",
    "scripts/reporting/freeze_zoo_budget.py",
    "scripts/reporting/inspect_legacy_artifacts.py",
)


def planned_jobs() -> list[dict]:
    """Jobs of a frozen experiment (held-out stages), of the shipped pilot
    (budget-inspection stages), and of the Methodology 1 study (downstream
    stages), so every stage's command is covered."""

    frozen = synthetic_manifest(frozen=True)
    pilot = load_experiment_manifest(
        REPO_ROOT / "configs/research/manifests/pilot.yaml",
        reference_tasks=load_yaml(REPO_ROOT / "configs/reference_tasks.yaml"),
    )
    jobs = []
    for manifest in (frozen, pilot):
        jobs.extend(
            build_experiment_job_plan(
                manifest,
                manifest_path="manifest.yaml",
                research_root="results",
                python="python",
                data_root="data",
            )
        )
    core = {job["job_id"] for job in jobs}
    jobs.extend(
        job
        for job in build_methodology1_job_plan(
            full_manifest(),
            manifest_path="manifest.yaml",
            research_root="results",
            gamma_candidates=load_yaml(REPO_ROOT / "configs/research/gamma_candidates.yaml"),
            data_root="data",
        )
        if job["job_id"].startswith("m1:") and job["job_id"] not in core
    )
    return jobs


def script_of(job: dict) -> str | None:
    command = job["command"]
    return command[1] if len(command) > 1 and command[1].endswith(".py") else None


JOBS = planned_jobs()
PLANNED_SCRIPTS = sorted({script for job in JOBS if (script := script_of(job))})


@lru_cache(maxsize=None)
def help_text(script: str) -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "JAX_PLATFORM_NAME": "cpu", "CUDA_VISIBLE_DEVICES": ""},
    )
    return completed.returncode, completed.stdout + completed.stderr


def test_plan_covers_every_stage_script():
    stages = {job["stage"] for job in JOBS}
    assert {"budget_inspection", "diagnostics_held_out", "rollouts_held_out"} <= stages
    assert {"gamma_validation", "continual", "ood", "mechanistic_probes"} <= stages
    assert len(PLANNED_SCRIPTS) >= 24
    assert all((REPO_ROOT / script).is_file() for script in (*PLANNED_SCRIPTS, *MANUAL_SCRIPTS))


@pytest.mark.parametrize("script", (*PLANNED_SCRIPTS, *MANUAL_SCRIPTS))
def test_script_builds_its_argument_parser(script):
    returncode, output = help_text(script)
    assert returncode == 0, f"{script} --help failed:\n{output[-2000:]}"
    assert "usage:" in output


@pytest.mark.parametrize("script", PLANNED_SCRIPTS)
def test_every_planned_flag_is_accepted_by_the_script(script):
    returncode, output = help_text(script)
    assert returncode == 0, output[-2000:]
    accepted = set(re.findall(r"(?<![\w-])--[a-z0-9][a-z0-9-]*", output))
    used = {
        token
        for job in JOBS
        if script_of(job) == script
        for token in job["command"][2:]
        if token.startswith("--")
    }
    assert used, f"no planned job passes a flag to {script}"
    assert used <= accepted, f"{script} does not accept {sorted(used - accepted)}"
