"""run_job_manifest.py as separate processes sharing one job manifest.

This is how the job manifest runs on a cluster: one runner per GPU slice, each
in its own scheduler job, all pointing at the same state directory.
"""

import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from wsl_vla.experiments.zoo_jobs import CLAIMS_DIRECTORY, RECORDS_DIRECTORY, write_job_plan

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "train" / "run_job_manifest.py"
WRITE_AFTER = "import pathlib, sys, time; time.sleep(float(sys.argv[2])); pathlib.Path(sys.argv[1]).write_text('done')"


def job(tmp_path, job_id, dependencies=(), seconds=0.2):
    output = tmp_path / "out" / f"{job_id}.txt"
    return {
        "schema_version": 2,
        "job_id": job_id,
        "stage": "evidence",
        "command": [sys.executable, "-c", WRITE_AFTER, str(output), str(seconds)],
        "dependencies": list(dependencies),
        "expected_outputs": [str(output)],
        "resources": {"gpu_required": False, "simulator_required": False, "gpu_training": False},
        "environment": {},
    }


def start_runner(plan, *arguments):
    return subprocess.Popen(
        [sys.executable, str(RUNNER), str(plan), "--poll-seconds", "0.1", *arguments],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_two_runner_processes_complete_a_shared_plan_running_each_job_once(tmp_path):
    (tmp_path / "out").mkdir()
    jobs = [job(tmp_path, f"job{index}") for index in range(6)]
    jobs.append(job(tmp_path, "merge", [item["job_id"] for item in jobs]))
    plan = tmp_path / "jobs" / "jobs.jsonl"
    write_job_plan(jobs, plan)
    runners = [start_runner(plan, "--workers", "2") for _ in range(2)]
    for runner in runners:
        _, stderr = runner.communicate(timeout=120)
        assert runner.returncode == 0, stderr
    for item in jobs:
        record = json.loads((plan.parent / RECORDS_DIRECTORY / f"{item['job_id']}.json").read_text())
        assert record["status"] == "completed" and record["attempts"] == 1
        log = Path(record["log"]).read_text()
        assert log.count("=====") == 1  # executed exactly once, by one of the runners
    owners = {
        json.loads((plan.parent / RECORDS_DIRECTORY / f"{item['job_id']}.json").read_text())["runner"]
        for item in jobs
    }
    assert len(owners) == 2  # both processes did part of the work


def test_sigterm_interrupts_the_runner_and_its_job_resumes_on_the_next_run(tmp_path):
    (tmp_path / "out").mkdir()
    plan = tmp_path / "jobs" / "jobs.jsonl"
    write_job_plan([job(tmp_path, "slow", seconds=60)], plan)
    runner = start_runner(plan)
    owner = plan.parent / CLAIMS_DIRECTORY / "slow" / "owner.json"
    deadline = time.time() + 60
    while time.time() < deadline and not (owner.is_file() and "child_pid" in owner.read_text()):
        time.sleep(0.05)
    runner.send_signal(signal.SIGTERM)  # what a scheduler's time limit sends
    stdout, stderr = runner.communicate(timeout=60)
    assert runner.returncode == 3, stderr
    assert json.loads(stdout)["interrupted"] is True
    record = json.loads((plan.parent / RECORDS_DIRECTORY / "slow.json").read_text())
    assert record["status"] == "interrupted"
    assert not (plan.parent / CLAIMS_DIRECTORY / "slow").exists()

    write_job_plan([job(tmp_path, "slow", seconds=0)], plan)
    rerun = start_runner(plan)
    _, stderr = rerun.communicate(timeout=60)
    assert rerun.returncode == 0, stderr
    record = json.loads((plan.parent / RECORDS_DIRECTORY / "slow.json").read_text())
    assert record["status"] == "completed" and record["attempts"] == 2


def test_submit_workers_requests_one_slice_per_runner(tmp_path):
    fake = tmp_path / "bin" / "sbatch"
    fake.parent.mkdir()
    fake.write_text('#!/bin/sh\necho "$@" >> "$SBATCH_CALLS"\n')
    fake.chmod(0o755)
    calls = tmp_path / "calls.txt"
    environment = {"PATH": f"{fake.parent}:/usr/bin:/bin", "SBATCH_CALLS": str(calls), "RENDER": "mesa-egl"}
    subprocess.run(
        ["bash", "jobs/submit_workers.sh", "small", "4", "jobs/03_model_zoo.sh"],
        cwd=tmp_path, env=environment, check=False, capture_output=True,
    )  # wrong directory: the stage script is not found
    assert not calls.exists()
    for script in ("submit_workers.sh", "03_model_zoo.sh"):
        (tmp_path / "jobs").mkdir(exist_ok=True)
        (tmp_path / "jobs" / script).write_text((REPO_ROOT / "jobs" / script).read_text())
    done = subprocess.run(
        ["bash", "jobs/submit_workers.sh", "small", "4", "jobs/03_model_zoo.sh"],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 4 and len(set(lines)) == 1
    assert lines[0].split() == [
        "--partition=gpu_small", "--qos=gpu_small", "--gres=gpu:1g.24gb:1",
        "--cpus-per-task=6", "--mem=24G", "--export=ALL,WORKERS=3,TRAINING_SLOTS=1",
        "jobs/03_model_zoo.sh",
    ]
    assert (tmp_path / "logs").is_dir()
    large = subprocess.run(
        ["bash", "jobs/submit_workers.sh", "large", "1", "jobs/03_model_zoo.sh"],
        cwd=tmp_path, env={**environment, "WORKERS": "10", "TIME": "12:00:00"},
        capture_output=True, text=True,
    )
    assert large.returncode == 0, large.stderr
    last = calls.read_text().splitlines()[-1]
    assert "--gres=gpu:4g.96gb:1" in last and "--time=12:00:00" in last
    assert "WORKERS=10,TRAINING_SLOTS=4" in last
