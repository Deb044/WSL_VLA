"""Resumable job manifests: dependencies, skipping, blocking, failure summaries."""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tests.synthetic import synthetic_manifest
from wsl_vla.experiments.manifest import ExperimentPaths
from wsl_vla.experiments.zoo_jobs import (
    CLAIMS_DIRECTORY,
    FAILURE_SUMMARY,
    RECORDS_DIRECTORY,
    STATE_FILE,
    build_experiment_job_plan,
    read_job_plan,
    run_job_plan,
    summarize_job_plan,
    write_job_plan,
)


def plan(tmp_path, **changes):
    manifest = synthetic_manifest(**changes)
    jobs = build_experiment_job_plan(
        manifest, manifest_path="manifest.yaml", research_root=tmp_path, python="python"
    )
    return manifest, jobs


def test_plan_follows_the_ordered_run_profiles_and_manifest_counts(tmp_path):
    manifest, jobs = plan(tmp_path)
    by_stage = summarize_job_plan(jobs, manifest)["jobs_by_stage"]
    counts = manifest.derived_counts()
    assert by_stage["zoo"] == counts["independent_runs"] == 24
    assert by_stage["evidence"] == len(manifest.tasks) == 12
    assert by_stage["archive"] == 1 and by_stage["alignment"] == 2
    order = [job["stage"] for job in jobs]
    for earlier, later in (
        ("gates", "evidence"), ("zoo", "zoo_verification"), ("zoo_verification", "archive"),
        ("archive", "alignment"), ("alignment", "diagnostics_validation"),
        ("rollouts_validation", "rollouts_held_out"),
    ):
        assert max(i for i, s in enumerate(order) if s == earlier) < min(
            i for i, s in enumerate(order) if s == later
        )
    ids = {job["job_id"]: job for job in jobs}
    alignment = [job for job in jobs if job["stage"] == "alignment"]
    assert {job["command"][-1] for job in alignment} == {"aligned", "reconstruction_only"}
    assert all(job["dependencies"] == alignment[0]["dependencies"] for job in alignment)
    diagnostics = next(job for job in jobs if job["stage"] == "diagnostics_validation")
    assert set(diagnostics["dependencies"]) == {job["job_id"] for job in alignment}
    assert ids["zoo_verification"]["resources"] == {
        "gpu_required": False,
        "simulator_required": False,
        "gpu_training": False,
        "deterministic_gpu": True,
    }
    assert all(job["resources"]["simulator_required"] for job in jobs if job["stage"].startswith("rollouts"))
    training = {job["job_id"] for job in jobs if job["resources"]["gpu_training"]}
    assert training == {
        job["job_id"]
        for job in jobs
        if job["stage"] in ("zoo", "alignment") or job["job_id"].startswith("budget_inspection:train:")
    }


def test_held_out_jobs_exist_only_for_a_frozen_budget(tmp_path):
    manifest, jobs = plan(tmp_path, frozen=False)
    assert not [job for job in jobs if "held_out" in job["stage"]]
    assert "not frozen" in summarize_job_plan(jobs, manifest)["notes"][0]
    _, frozen_jobs = plan(tmp_path, frozen=True)
    held_out = [job for job in frozen_jobs if "held_out" in job["stage"]]
    assert {job["stage"] for job in held_out} == {"diagnostics_held_out", "rollouts_held_out"}
    rollouts = next(job for job in held_out if job["stage"] == "rollouts_held_out")
    assert rollouts["command"][-4:] == ["--scope", "held_out", "--tier", "final"]


def test_plan_round_trips_through_jsonl(tmp_path):
    _, jobs = plan(tmp_path)
    path = tmp_path / "jobs.jsonl"
    write_job_plan(jobs, path)
    assert read_job_plan(path) == list(jobs)


def toy_jobs(tmp_path):
    def job(job_id, deps=(), output=True, simulator=False):
        return {
            "schema_version": 2,
            "job_id": job_id,
            "stage": "zoo",
            "command": ["run", job_id],
            "dependencies": list(deps),
            "expected_outputs": [str(tmp_path / "out" / f"{job_id}.txt")] if output else [],
            "resources": {"gpu_required": False, "simulator_required": simulator},
        }

    return [job("a"), job("b", ["a"]), job("c", ["b"]), job("d"), job("e", ["d"], simulator=True)]


def fake_runner(tmp_path, failing=()):
    calls = []

    def run(command, cwd, stdout, stderr, env):
        job_id = command[1]
        calls.append(job_id)
        stdout.write(f"log of {job_id}\n".encode())
        if job_id in failing:
            return SimpleNamespace(returncode=3)
        target = tmp_path / "out" / f"{job_id}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("done")
        return SimpleNamespace(returncode=0)

    return run, calls


def test_runner_blocks_dependents_keeps_going_and_summarizes_failures(tmp_path):
    jobs = toy_jobs(tmp_path)
    runner, calls = fake_runner(tmp_path, failing={"b"})
    summary = run_job_plan(
        jobs, state_directory=tmp_path / "state", working_directory=tmp_path, runner=runner
    )
    assert calls == ["a", "b", "d", "e"]  # c never ran; d and e are independent of the failure
    assert summary["counts"] == {"blocked": 1, "completed": 3, "failed": 1}
    assert summary["failed"][0]["job_id"] == "b" and summary["failed"][0]["returncode"] == 3
    assert "log of b" in summary["failed"][0]["log_tail"][-1]
    assert summary["blocked"] == [{"job_id": "c", "blocked_by": ["b"]}]
    saved = json.loads((tmp_path / "state" / FAILURE_SUMMARY).read_text())
    assert saved["failed"][0]["job_id"] == "b"


def test_runner_resumes_skipping_completed_work_and_retrying_failures(tmp_path):
    jobs = toy_jobs(tmp_path)
    runner, _ = fake_runner(tmp_path, failing={"b"})
    run_job_plan(jobs, state_directory=tmp_path / "state", working_directory=tmp_path, runner=runner)
    runner, calls = fake_runner(tmp_path)
    summary = run_job_plan(
        jobs, state_directory=tmp_path / "state", working_directory=tmp_path, runner=runner
    )
    assert calls == ["b", "c"]  # a, d, e were complete; the failure and what it blocked rerun
    assert summary["counts"] == {"completed": 5} and not summary["failed"]
    # A completed job whose output disappeared is no longer complete.
    (tmp_path / "out" / "d.txt").unlink()
    runner, calls = fake_runner(tmp_path)
    run_job_plan(jobs, state_directory=tmp_path / "state", working_directory=tmp_path, runner=runner)
    assert calls == ["d"]


def test_runner_reports_missing_outputs_and_host_limitations(tmp_path):
    jobs = toy_jobs(tmp_path)

    def silent(command, cwd, stdout, stderr, env):
        return SimpleNamespace(returncode=0)  # exits cleanly but writes nothing

    summary = run_job_plan(
        jobs[:1], state_directory=tmp_path / "s1", working_directory=tmp_path, runner=silent
    )
    assert summary["failed"][0]["missing_outputs"] == jobs[0]["expected_outputs"]

    runner, calls = fake_runner(tmp_path)
    summary = run_job_plan(
        jobs, state_directory=tmp_path / "s2", working_directory=tmp_path, runner=runner,
        skip_simulator=True,
    )
    assert "e" not in calls
    skipped = summary["skipped"]
    assert skipped[0]["job_id"] == "e" and "simulator" in skipped[0]["reason"]
    assert summary["counts"]["skipped"] == 1  # a skip is reported, never counted as completed


def test_dry_run_executes_nothing(tmp_path):
    jobs = toy_jobs(tmp_path)
    runner, calls = fake_runner(tmp_path)
    summary = run_job_plan(
        jobs, state_directory=tmp_path / "state", working_directory=tmp_path, runner=runner,
        dry_run=True,
    )
    assert not calls and not Path(tmp_path / "state" / FAILURE_SUMMARY).exists()
    assert summary["counts"] == {"blocked": 3, "would_run": 2}


def test_experiment_paths_match_the_planned_outputs(tmp_path):
    manifest, jobs = plan(tmp_path)
    paths = ExperimentPaths(tmp_path, manifest.name)
    zoo = next(job for job in jobs if job["stage"] == "zoo")
    assert zoo["expected_outputs"][0] == str(paths.run_directory(*manifest.runs()[0]) / "run.json")
    assert len(zoo["expected_outputs"]) == 1 + len(manifest.zoo.snapshot_updates)


def test_unit_test_gate_runs_on_cpu_and_other_jobs_inherit_the_runner_environment(tmp_path):
    _, jobs = plan(tmp_path)
    by_id = {job["job_id"]: job for job in jobs}
    assert by_id["gate:unit_tests"]["environment"] == {
        "JAX_PLATFORM_NAME": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
    }
    assert all(not job["environment"] for job in jobs if job["job_id"] != "gate:unit_tests")


def test_runner_applies_job_environment_on_top_of_its_own(tmp_path):
    seen = {}

    def runner(command, cwd, stdout, stderr, env):
        seen[command[1]] = env
        return SimpleNamespace(returncode=0)

    jobs = [
        {**job, "expected_outputs": []} for job in toy_jobs(tmp_path)[:2]
    ]
    jobs[1]["environment"] = {"JAX_PLATFORM_NAME": "cpu", "HF_HUB_OFFLINE": "0"}
    run_job_plan(
        jobs,
        state_directory=tmp_path / "state",
        working_directory=tmp_path,
        environment={"HF_HUB_OFFLINE": "1", "KEEP": "yes"},
        deterministic_gpu=False,
        runner=runner,
    )
    assert seen["a"] == {"HF_HUB_OFFLINE": "1", "KEEP": "yes"}
    assert seen["b"] == {"HF_HUB_OFFLINE": "0", "KEEP": "yes", "JAX_PLATFORM_NAME": "cpu"}

    # Without a runner environment, overrides, or determinism the process environment is inherited.
    seen.clear()
    run_job_plan(
        [{**toy_jobs(tmp_path)[0], "expected_outputs": []}],
        state_directory=tmp_path / "other",
        working_directory=tmp_path,
        deterministic_gpu=False,
        runner=runner,
    )
    assert seen["a"] is None


def test_deterministic_gpu_ops_are_added_per_job_and_never_to_alignment_training(tmp_path):
    _, jobs = plan(tmp_path)
    alignment = [job for job in jobs if job["stage"] == "alignment"]
    assert alignment and not any(job["resources"]["deterministic_gpu"] for job in alignment)
    assert all(job["resources"]["deterministic_gpu"] for job in jobs if job["stage"] != "alignment")

    seen = {}

    def runner(command, cwd, stdout, stderr, env):
        seen[command[1]] = env
        return SimpleNamespace(returncode=0)

    toy = [{**job, "expected_outputs": []} for job in toy_jobs(tmp_path)[:2]]
    toy[1]["resources"] = {**toy[1]["resources"], "deterministic_gpu": False}
    run_job_plan(
        toy,
        state_directory=tmp_path / "state",
        working_directory=tmp_path,
        environment={"XLA_FLAGS": "--xla_dump_to=/x"},
        runner=runner,
    )
    assert seen["a"]["XLA_FLAGS"].split() == ["--xla_dump_to=/x", "--xla_gpu_deterministic_ops=true"]
    assert seen["b"]["XLA_FLAGS"] == "--xla_dump_to=/x"


def test_state_of_the_sequential_runner_is_adopted(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "a.txt").write_text("done")
    (state / STATE_FILE).write_text(
        json.dumps({"jobs": {"a": {"status": "completed", "attempts": 1},
                             "c": {"status": "blocked", "blocked_by": ["b"]}}})
    )
    runner, calls = fake_runner(tmp_path)
    run_job_plan(toy_jobs(tmp_path), state_directory=state, working_directory=tmp_path, runner=runner)
    assert calls == ["b", "c", "d", "e"]
    assert json.loads((state / RECORDS_DIRECTORY / "a.json").read_text())["status"] == "completed"


# ------------------------------------------------------- several runners at once


def toy_job(tmp_path, job_id, dependencies=(), *, training=False, output=True):
    return {
        "schema_version": 2,
        "job_id": job_id,
        "stage": "zoo" if training else "evidence",
        "command": ["run", job_id],
        "dependencies": list(dependencies),
        "expected_outputs": [str(tmp_path / "out" / f"{job_id}.txt")] if output else [],
        "resources": {"gpu_required": True, "simulator_required": False, "gpu_training": training},
    }


class SlowRunner:
    """Fake job execution that takes a while and records how many jobs overlap."""

    def __init__(self, tmp_path, seconds=0.0):
        self.tmp_path = tmp_path
        self.seconds = seconds
        self.lock = threading.Lock()
        self.calls = []
        self.active = {"all": 0, "training": 0}
        self.peak = {"all": 0, "training": 0}

    def __call__(self, command, cwd, stdout, stderr, env):
        job_id = command[1]
        training = int(job_id.startswith("train"))
        with self.lock:
            self.calls.append(job_id)
            self.active["all"] += 1
            self.active["training"] += training
            self.peak = {key: max(self.peak[key], self.active[key]) for key in self.peak}
        time.sleep(self.seconds)
        with self.lock:
            self.active["all"] -= 1
            self.active["training"] -= training
        target = self.tmp_path / "out" / f"{job_id}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("done")
        return SimpleNamespace(returncode=0)


def claim_of_another_runner(state, job_id, *, host="other-node", pid=12345, age=0.0):
    claim = state / CLAIMS_DIRECTORY / job_id
    claim.mkdir(parents=True)
    owner = claim / "owner.json"
    owner.write_text(json.dumps({"host": host, "pid": pid, "token": "theirs", "job_id": job_id}))
    if age:
        stamp = time.time() - age
        os.utime(owner, (stamp, stamp))
    return claim


def finish_elsewhere(state, claim, record, *, output=None, after=0.3):
    """Act as the other runner: write the job's record, then release its claim."""

    def finish():
        time.sleep(after)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("done")
        (state / RECORDS_DIRECTORY).mkdir(parents=True, exist_ok=True)
        (state / RECORDS_DIRECTORY / f"{claim.name}.json").write_text(json.dumps(record))
        shutil.rmtree(claim)

    thread = threading.Thread(target=finish)
    thread.start()
    return thread


def test_two_runners_share_one_plan_and_run_every_job_exactly_once(tmp_path):
    jobs = [toy_job(tmp_path, f"job{index}") for index in range(8)]
    jobs.append(toy_job(tmp_path, "merge", [job["job_id"] for job in jobs]))
    execute = SlowRunner(tmp_path, seconds=0.05)
    summaries = []

    def runner():
        summaries.append(
            run_job_plan(
                jobs, state_directory=tmp_path / "state", working_directory=tmp_path,
                runner=execute, poll_seconds=0.02,
            )
        )

    threads = [threading.Thread(target=runner) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert sorted(execute.calls) == sorted(job["job_id"] for job in jobs)  # each exactly once
    assert execute.peak["all"] == 2  # the two runners worked side by side
    for summary in summaries:
        # A runner may finish while the other still runs the last job.
        assert set(summary["counts"]) <= {"completed", "running"} and not summary["failed"]
    records = json.loads((tmp_path / "state" / STATE_FILE).read_text())["jobs"]
    assert {job_id: record["status"] for job_id, record in records.items()} == {
        job["job_id"]: "completed" for job in jobs
    }
    assert not [path for path in (tmp_path / "state" / CLAIMS_DIRECTORY).iterdir()
                if not path.name.startswith(".")]


def test_a_runner_waits_for_a_dependency_that_another_runner_is_executing(tmp_path):
    state = tmp_path / "state"
    jobs = [toy_job(tmp_path, "a"), toy_job(tmp_path, "b", ["a"])]
    claim = claim_of_another_runner(state, "a")
    helper = finish_elsewhere(
        state, claim, {"status": "completed", "attempts": 1}, output=tmp_path / "out" / "a.txt"
    )
    execute = SlowRunner(tmp_path)
    summary = run_job_plan(
        jobs, state_directory=state, working_directory=tmp_path, runner=execute, poll_seconds=0.02
    )
    helper.join()
    assert execute.calls == ["b"]
    assert summary["counts"] == {"completed": 2}


def test_claims_of_dead_runners_are_taken_over_and_live_claims_respected(tmp_path):
    state = tmp_path / "state"
    jobs = [toy_job(tmp_path, job_id) for job_id in ("silent", "dead", "live")]
    claim_of_another_runner(state, "silent", age=3600)  # another node, no heartbeat for an hour
    exited = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                            capture_output=True, text=True, check=True)
    claim_of_another_runner(state, "dead", host=socket.gethostname(), pid=int(exited.stdout))
    claim_of_another_runner(state, "live")  # another node, fresh heartbeat
    execute = SlowRunner(tmp_path)
    summary = run_job_plan(
        jobs, state_directory=state, working_directory=tmp_path, runner=execute,
        poll_seconds=0.02, stale_after_seconds=900,
    )
    assert sorted(execute.calls) == ["dead", "silent"]
    assert summary["counts"] == {"completed": 2, "running": 1}
    assert (state / CLAIMS_DIRECTORY / "live" / "owner.json").is_file()


def test_failures_are_final_for_runners_that_saw_them_and_retried_by_later_runners(tmp_path):
    state = tmp_path / "state"
    jobs = [toy_job(tmp_path, "a"), toy_job(tmp_path, "b", ["a"])]
    claim = claim_of_another_runner(state, "a")
    helper = finish_elsewhere(state, claim, {"status": "failed", "returncode": 1, "attempts": 1})
    execute = SlowRunner(tmp_path)
    summary = run_job_plan(
        jobs, state_directory=state, working_directory=tmp_path, runner=execute, poll_seconds=0.02
    )
    helper.join()
    assert execute.calls == []  # not retried by a runner that was running when it failed
    assert summary["counts"] == {"blocked": 1, "failed": 1}
    later = SlowRunner(tmp_path)
    summary = run_job_plan(
        jobs, state_directory=state, working_directory=tmp_path, runner=later, poll_seconds=0.02
    )
    assert later.calls == ["a", "b"] and summary["counts"] == {"completed": 2}


def test_a_runner_runs_several_jobs_at_once_but_limits_training_jobs(tmp_path):
    jobs = [toy_job(tmp_path, f"train{index}", training=True) for index in range(3)]
    jobs += [toy_job(tmp_path, f"rollout{index}") for index in range(3)]
    execute = SlowRunner(tmp_path, seconds=0.2)
    summary = run_job_plan(
        jobs, state_directory=tmp_path / "state", working_directory=tmp_path, runner=execute,
        workers=3, training_slots=1, poll_seconds=0.02,
    )
    assert summary["counts"] == {"completed": 6}
    assert execute.peak == {"all": 3, "training": 1}


def sleeping_job(tmp_path):
    return {
        **toy_job(tmp_path, "sleeper", output=False),
        "command": [sys.executable, "-c", "import time; time.sleep(120)"],
    }


def wait_for_job_process(state, job_id):
    owner = state / CLAIMS_DIRECTORY / job_id / "owner.json"
    deadline = time.time() + 60
    while time.time() < deadline:
        if owner.is_file() and "child_pid" in (details := json.loads(owner.read_text())):
            return owner, details
        time.sleep(0.05)
    raise AssertionError("the job process never started")


def test_stopping_a_runner_terminates_its_jobs_and_records_them_as_interrupted(tmp_path):
    state = tmp_path / "state"
    stop = threading.Event()
    result = {}
    thread = threading.Thread(
        target=lambda: result.update(
            run_job_plan([sleeping_job(tmp_path)], state_directory=state,
                         working_directory=tmp_path, stop=stop, poll_seconds=0.05)
        )
    )
    thread.start()
    _, details = wait_for_job_process(state, "sleeper")
    stop.set()
    thread.join(timeout=60)
    assert not thread.is_alive()
    record = json.loads((state / RECORDS_DIRECTORY / "sleeper.json").read_text())
    assert record["status"] == "interrupted" and record["returncode"] != 0
    assert result["interrupted"] and result["counts"] == {"pending": 1}
    assert not (state / CLAIMS_DIRECTORY / "sleeper").exists()
    with pytest.raises(ProcessLookupError):
        os.kill(details["child_pid"], 0)


def test_a_runner_whose_claim_was_taken_over_stops_the_job_and_writes_no_record(tmp_path):
    state = tmp_path / "state"
    result = {}
    thread = threading.Thread(
        target=lambda: result.update(
            run_job_plan([sleeping_job(tmp_path)], state_directory=state,
                         working_directory=tmp_path, poll_seconds=0.05, heartbeat_seconds=0.05)
        )
    )
    thread.start()
    owner, details = wait_for_job_process(state, "sleeper")
    owner.write_text(json.dumps({**details, "token": "another runner"}))  # taken over
    thread.join(timeout=60)
    assert not thread.is_alive()
    assert [event["status"] for event in result["events"]] == ["claim_lost"]
    assert not (state / RECORDS_DIRECTORY / "sleeper.json").exists()
    assert result["counts"] == {"running": 1}  # the job now belongs to the other runner
