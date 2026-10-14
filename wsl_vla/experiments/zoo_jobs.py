"""Resumable job manifests for the zoo and alignment experiments.

``build_experiment_job_plan`` expands one experiment manifest into an ordered,
dependency-annotated list of commands that follows the run profiles of the
repair plan (gates -> evidence and zoo -> archives -> paired alignment ->
offline diagnostics -> rollouts). ``run_job_plan`` executes such a list,
skips work that is already complete, keeps going past failures of independent
jobs, and writes a failure summary. Several runners (one per GPU slice, for
example) may execute the same plan at once; they coordinate through the state
directory alone. Generating a plan never runs an experiment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence
import uuid

from .manifest import ALIGNMENT_CONDITIONS, ExperimentManifest, ExperimentPaths, split_task_id
from .provenance import atomic_write_bytes, atomic_write_json
from .runtime import with_deterministic_gpu_ops


JOB_SCHEMA_VERSION = 2
# Summary of the job records, rewritten by every runner; the records are the truth.
STATE_FILE = "job_state.json"
RECORDS_DIRECTORY = "records"
CLAIMS_DIRECTORY = "claims"
FAILURE_SUMMARY = "failure_summary.json"
# Stages whose jobs train on the GPU, for plans written before resources.gpu_training.
TRAINING_STAGES = ("zoo", "alignment")
STAGES = (
    "gates",
    "evidence",
    "budget_inspection",
    "zoo",
    "zoo_behaviour",
    "zoo_verification",
    "archive",
    "alignment",
    "diagnostics_validation",
    "rollouts_validation",
    "diagnostics_held_out",
    "rollouts_held_out",
    "report",
)


def build_experiment_job_plan(
    manifest: ExperimentManifest,
    *,
    manifest_path: str | Path,
    research_root: str | Path,
    python: str = "python",
    data_root: str | Path | None = None,
) -> tuple[dict, ...]:
    """Expand a manifest into jobs with stable ids, dependencies, and outputs."""

    paths = ExperimentPaths(Path(research_root), manifest.name)
    common = ["--manifest", str(manifest_path), "--research-root", str(research_root)]
    data = [] if data_root is None else ["--data-root", str(data_root)]
    jobs: list[dict] = []

    def add(job_id, stage, command, *, dependencies=(), outputs=(), gpu=True, simulator=False,
            training=False, environment=None, deterministic_gpu=True):
        jobs.append(
            {
                "schema_version": JOB_SCHEMA_VERSION,
                "job_id": job_id,
                "stage": stage,
                "command": [str(value) for value in command],
                "dependencies": sorted(set(dependencies)),
                "expected_outputs": [str(Path(value)) for value in outputs],
                "resources": {
                    "gpu_required": bool(gpu),
                    "simulator_required": bool(simulator),
                    # Trains on the GPU; a runner limits how many run at once
                    # (run_job_plan's training_slots).
                    "gpu_training": bool(training),
                    # Run with XLA's deterministic GPU ops (see run_job_plan).
                    "deterministic_gpu": bool(deterministic_gpu),
                },
                # Variables set for this job on top of the runner's environment.
                "environment": dict(environment or {}),
            }
        )

    def task_args(identifier):
        suite, index = split_task_id(identifier)
        return ["--suite", suite, "--task-index", index]

    gates = paths.root / "gates"
    add(
        "gate:unit_tests",
        "gates",
        [python, "-m", "pytest", "-m", "not integration", "-q"],
        gpu=False,
        # Unit tests check logic and exact reproducibility, so they run on CPU.
        # Behaviour on the accelerator is covered by gate:octo and the run itself.
        environment={"JAX_PLATFORM_NAME": "cpu", "CUDA_VISIBLE_DEVICES": ""},
    )
    add(
        "gate:preflight",
        "gates",
        [python, "scripts/eval/research_preflight.py", *common, "--output-dir", gates / "preflight", *data],
        outputs=[gates / "preflight" / "preflight.json"],
        gpu=False,
    )
    add(
        "gate:octo",
        "gates",
        [python, "scripts/eval/verify_official_octo.py", "--output", gates / "octo_gates.json"],
        dependencies=["gate:preflight"],
        outputs=[gates / "octo_gates.json"],
    )
    suites = sorted({split_task_id(identifier)[0] for identifier in manifest.tasks})
    orientation_ids = []
    for suite in suites:
        first = next(i for i in manifest.tasks if split_task_id(i)[0] == suite)
        job_id = f"gate:camera_orientation:{suite}"
        add(
            job_id,
            "gates",
            [python, "scripts/setup/verify_camera_orientation.py", *task_args(first), *data,
             "--output-dir", gates / "camera_orientation" / suite],
            dependencies=["gate:preflight"],
            outputs=[gates / "camera_orientation" / suite / "camera_orientation.json"],
            gpu=False,
            simulator=True,
        )
        orientation_ids.append(job_id)
    gate_ids = ["gate:unit_tests", "gate:octo"]

    evidence_ids = {}
    for identifier in manifest.tasks:
        job_id = f"evidence:{identifier}"
        add(
            job_id,
            "evidence",
            [python, "scripts/data/extract_research_evidence.py", *common, *task_args(identifier), *data],
            dependencies=gate_ids,
            outputs=[paths.evidence_file(identifier)],
        )
        evidence_ids[identifier] = job_id

    inspection_seed = manifest.rollouts.zoo_reference_seed
    for identifier in manifest.zoo.inspection_tasks:
        suite, _ = split_task_id(identifier)
        run = paths.budget_inspection / suite / identifier / f"seed_{inspection_seed}"
        train_id = f"budget_inspection:train:{identifier}"
        add(
            train_id,
            "budget_inspection",
            [python, "scripts/train/train_research_zoo.py", *common, *task_args(identifier),
             "--seed", inspection_seed, "--budget-inspection", *data],
            dependencies=gate_ids,
            outputs=[run / "run.json"],
            training=True,
        )
        add(
            f"budget_inspection:rollouts:{identifier}",
            "budget_inspection",
            [python, "scripts/eval/evaluate_zoo_behaviour.py", *common, *task_args(identifier),
             "--seed", inspection_seed, "--budget-inspection"],
            dependencies=[train_id, *orientation_ids],
            outputs=[run / f"update_{stage}" / "behaviour.json" for stage in manifest.zoo.inspection_updates],
            simulator=True,
        )

    zoo_ids = {}
    final_stage = manifest.zoo.snapshot_updates[-1]
    for identifier, seed in manifest.runs():
        job_id = f"zoo:{identifier}:seed-{seed}"
        run = paths.run_directory(identifier, seed)
        add(
            job_id,
            "zoo",
            [python, "scripts/train/train_research_zoo.py", *common, *task_args(identifier),
             "--seed", seed, *data],
            dependencies=gate_ids,
            outputs=[run / "run.json"]
            + [run / f"update_{stage}" / "adapter.npz" for stage in manifest.zoo.snapshot_updates],
            training=True,
        )
        zoo_ids[(identifier, seed)] = job_id
        add(
            f"zoo_behaviour:{identifier}:seed-{seed}",
            "zoo_behaviour",
            [python, "scripts/eval/evaluate_zoo_behaviour.py", *common, *task_args(identifier),
             "--seed", seed, "--updates", final_stage],
            dependencies=[job_id, *orientation_ids],
            outputs=[run / f"update_{final_stage}" / "behaviour.json"],
            simulator=True,
        )
    add(
        "zoo_verification",
        "zoo_verification",
        [python, "scripts/eval/verify_research_zoo.py", *common],
        dependencies=list(zoo_ids.values()),
        outputs=[paths.population / "verification.json"],
        gpu=False,
    )

    held_out_allowed = manifest.zoo.frozen
    for fold in manifest.folds:
        fold_tasks = (*fold.train_tasks, *fold.validation_tasks)
        archive_id = f"archive:{fold.name}"
        add(
            archive_id,
            "archive",
            [python, "scripts/data/build_alignment_archive.py", *common, "--fold", fold.name],
            dependencies=["zoo_verification", *(evidence_ids[t] for t in set(fold_tasks))],
            outputs=[paths.archive(fold.name) / "archive_manifest.json"],
            gpu=False,
        )
        for seed in manifest.alignment.seeds:
            pair = []
            for condition in ALIGNMENT_CONDITIONS:
                job_id = f"alignment:{fold.name}:seed-{seed}:{condition}"
                add(
                    job_id,
                    "alignment",
                    [python, "scripts/train/train_research_alignment.py", *common, "--fold", fold.name,
                     "--seed", seed, "--condition", condition],
                    dependencies=[archive_id],
                    outputs=[paths.alignment_checkpoint(fold.name, seed, condition) / "metadata.json"],
                    training=True,
                    # Deterministic GPU ops made alignment training about 36 times
                    # slower (scatter-heavy backward pass). The paired comparison
                    # does not need bitwise reproducibility: both conditions share
                    # initialization and batch schedule by construction.
                    deterministic_gpu=False,
                )
                pair.append(job_id)
            validation_diagnostics = f"diagnostics:validation:{fold.name}:seed-{seed}"
            add(
                validation_diagnostics,
                "diagnostics_validation",
                [python, "scripts/eval/evaluate_alignment_pair.py", *common, "--fold", fold.name,
                 "--seed", seed, "--scope", "validation"],
                dependencies=pair,
                outputs=[paths.diagnostics(fold.name, seed, "validation") / "alignment_pair.json"],
            )
            validation_rollouts = f"rollouts:validation:{fold.name}:seed-{seed}"
            add(
                validation_rollouts,
                "rollouts_validation",
                [python, "scripts/eval/run_generated_policy_rollouts.py", *common, "--fold", fold.name,
                 "--seed", seed, "--scope", "validation", "--tier", "development"],
                dependencies=[*pair, *orientation_ids],
                outputs=[paths.rollouts("validation", fold.name, seed) / "summary.json"],
                simulator=True,
            )
            if not (held_out_allowed and fold.held_out_tasks):
                continue
            held_out_evidence = [evidence_ids[t] for t in fold.held_out_tasks]
            add(
                f"diagnostics:held_out:{fold.name}:seed-{seed}",
                "diagnostics_held_out",
                [python, "scripts/eval/evaluate_alignment_pair.py", *common, "--fold", fold.name,
                 "--seed", seed, "--scope", "held_out"],
                dependencies=[validation_diagnostics, *held_out_evidence],
                outputs=[paths.diagnostics(fold.name, seed, "held_out") / "alignment_pair.json"],
            )
            add(
                f"rollouts:held_out:{fold.name}:seed-{seed}",
                "rollouts_held_out",
                [python, "scripts/eval/run_generated_policy_rollouts.py", *common, "--fold", fold.name,
                 "--seed", seed, "--scope", "held_out", "--tier", "final"],
                dependencies=[validation_rollouts, validation_diagnostics, *held_out_evidence],
                outputs=[paths.rollouts("held_out", fold.name, seed) / "summary.json"],
                simulator=True,
            )
    report = paths.root / "reports" / "generated_policy_rollouts.json"
    add(
        "report:rollouts",
        "report",
        [python, "scripts/reporting/report_generated_policy_rollouts.py", *common, "--output", report],
        dependencies=[job["job_id"] for job in jobs if job["stage"].startswith("rollouts_")],
        outputs=[report],
        gpu=False,
    )
    validate_job_plan(jobs)
    return tuple(jobs)


def validate_job_plan(jobs: Sequence[Mapping[str, Any]]) -> None:
    ids = [job["job_id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("job plan contains duplicate job ids")
    known = set(ids)
    seen: set[str] = set()
    for job in jobs:
        missing = set(job["dependencies"]) - known
        if missing:
            raise ValueError(f"job {job['job_id']} depends on unknown jobs {sorted(missing)}")
        if not set(job["dependencies"]) <= seen:
            raise ValueError(f"job {job['job_id']} is listed before one of its dependencies")
        seen.add(job["job_id"])


def summarize_job_plan(jobs: Sequence[Mapping[str, Any]], manifest: ExperimentManifest) -> dict:
    notes = []
    if not manifest.zoo.frozen:
        notes.append(
            "zoo budget is not frozen: held-out diagnostics and rollouts are not planned. "
            "Validate the budget, run scripts/reporting/freeze_zoo_budget.py, and re-plan."
        )
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "experiment": manifest.name,
        "profile": manifest.profile,
        "manifest_sha256": manifest.sha256(),
        "job_count": len(jobs),
        "gpu_job_count": sum(bool(job["resources"]["gpu_required"]) for job in jobs),
        "simulator_job_count": sum(bool(job["resources"]["simulator_required"]) for job in jobs),
        "jobs_by_stage": dict(sorted(Counter(str(job["stage"]) for job in jobs).items())),
        "expected": manifest.derived_counts(),
        "notes": notes,
    }


def write_job_plan(jobs: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    # Atomic: every runner of a multi-runner launch regenerates the same plan.
    atomic_write_bytes(path, "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs).encode())


def read_job_plan(path: str | Path) -> list[dict]:
    jobs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for job in jobs:
        if job.get("schema_version") != JOB_SCHEMA_VERSION:
            raise ValueError(f"unsupported job schema in {path}")
    validate_job_plan(jobs)
    return jobs


def _outputs_exist(job: Mapping[str, Any]) -> bool:
    return bool(job["expected_outputs"]) and all(Path(item).exists() for item in job["expected_outputs"])


def _is_complete(job: Mapping[str, Any], record: Mapping[str, Any] | None) -> bool:
    return bool(record) and record.get("status") == "completed" and (
        _outputs_exist(job) or not job["expected_outputs"]
    )


def _is_training(job: Mapping[str, Any]) -> bool:
    return bool(job["resources"].get("gpu_training", job["stage"] in TRAINING_STAGES))


def _file_name(job_id: str) -> str:
    return job_id.replace("/", "_").replace(":", "__")


# ------------------------------------------------------------------ execution
#
# Several runners may execute one plan at the same time, for example one per
# GPU slice, each a separate scheduler job. They share nothing but the state
# directory, which must be on a filesystem that all of them see:
#
#   records/<job>.json   the last attempt of every job, written atomically by
#                        the runner that holds the job's claim
#   claims/<job>/        exists while a runner executes the job; created with
#                        mkdir, which is atomic on local and network filesystems
#   job_state.json       all records in one file, for reading; never read back
#                        except to adopt the state of the earlier sequential runner


class _Records:
    def __init__(self, root: Path, *, adopt_legacy: bool) -> None:
        self.directory = root / RECORDS_DIRECTORY
        self._legacy: dict[str, dict] = {}
        legacy = root / STATE_FILE
        if not self.directory.is_dir() and legacy.is_file():
            # State written by the sequential runner, before per-job records.
            self._legacy = {
                job_id: record
                for job_id, record in json.loads(legacy.read_text(encoding="utf-8"))["jobs"].items()
                if record.get("status") in ("completed", "failed")
            }
            if adopt_legacy:
                for job_id, record in self._legacy.items():
                    self.write(job_id, record)
                self._legacy = {}

    def read(self, job_id: str) -> dict | None:
        try:
            path = self.directory / f"{_file_name(job_id)}.json"
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self._legacy.get(job_id)

    def read_all(self, job_ids: Sequence[str]) -> dict[str, dict]:
        return {job_id: record for job_id in job_ids if (record := self.read(job_id)) is not None}

    def write(self, job_id: str, record: Mapping[str, Any]) -> None:
        atomic_write_json(self.directory / f"{_file_name(job_id)}.json", record)


def _process_start(pid: int) -> str | None:
    """Start time of a process on this host (tells a live process from a reused pid)."""

    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError, ValueError):
        return None


def _process_alive(pid: Any, start: str | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    current = _process_start(int(pid))
    return start is None or current is None or current == start


@dataclass
class _Claim:
    job_id: str
    path: Path
    token: str
    owner: dict = field(default_factory=dict)
    lost: bool = False


class _Claims:
    """Exclusive job claims shared by every runner of one state directory.

    ``owner.json`` inside a claim names the runner (host, process, scheduler
    job) and is touched as a heartbeat while the job runs. A claim is
    abandoned once its heartbeat is older than ``stale_after`` seconds by the
    filesystem's own clock, so clock differences between nodes do not matter,
    or at once when neither its runner nor its job process lives on this host.
    """

    def __init__(self, root: Path, owner: Mapping[str, Any], stale_after: float) -> None:
        self.directory = root / CLAIMS_DIRECTORY
        self.directory.mkdir(parents=True, exist_ok=True)
        self.owner = dict(owner)
        self.stale_after = float(stale_after)
        self.clock = self.directory / f".clock-{owner['host']}-{owner['pid']}-{uuid.uuid4().hex[:8]}"

    def now(self) -> float:
        """The current time as the filesystem holding the claims sees it."""

        self.clock.touch()
        os.utime(self.clock, None)
        return self.clock.stat().st_mtime

    def _path(self, job_id: str) -> Path:
        return self.directory / _file_name(job_id)

    def owner_of(self, job_id: str) -> dict | None:
        try:
            return json.loads((self._path(job_id) / "owner.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
            return None

    def held_elsewhere(self, job_id: str, now: float) -> bool:
        path = self._path(job_id)
        return path.exists() and not self._stale(path, now)

    def acquire(self, job_id: str, now: float) -> _Claim | None:
        path = self._path(job_id)
        try:
            path.mkdir()
        except FileExistsError:
            if not self._stale(path, now) or not self._discard(path, "stale"):
                return None
            try:
                path.mkdir()
            except FileExistsError:
                return None
        claim = _Claim(job_id, path, uuid.uuid4().hex)
        claim.owner = {
            **self.owner,
            "job_id": job_id,
            "token": claim.token,
            "claimed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(path / "owner.json", claim.owner)
        return claim

    def annotate(self, claim: _Claim, **values: Any) -> None:
        claim.owner.update(values)
        if self.heartbeat(claim):
            atomic_write_json(claim.path / "owner.json", claim.owner)

    def heartbeat(self, claim: _Claim) -> bool:
        """Refresh a held claim; False once another runner has taken it over."""

        current = self.owner_of(claim.job_id)
        if current is None or current.get("token") != claim.token:
            claim.lost = True
            return False
        try:
            os.utime(claim.path / "owner.json", None)
        except FileNotFoundError:
            claim.lost = True
            return False
        return True

    def release(self, claim: _Claim) -> None:
        current = self.owner_of(claim.job_id)
        if current is not None and current.get("token") == claim.token:
            self._discard(claim.path, "released")

    def _stale(self, path: Path, now: float) -> bool:
        try:
            owner = json.loads((path / "owner.json").read_text(encoding="utf-8"))
            beat = (path / "owner.json").stat().st_mtime
        except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
            owner = None  # being created or released right now
            try:
                beat = path.stat().st_mtime
            except FileNotFoundError:
                return False
        if owner and owner.get("host") == self.owner["host"] and not (
            _process_alive(owner.get("pid"), owner.get("process_start"))
            or _process_alive(owner.get("child_pid"), owner.get("child_start"))
        ):
            return True
        return now - beat > self.stale_after

    def _discard(self, path: Path, reason: str) -> bool:
        """Move a claim out of the way atomically; of several runners, one wins."""

        graveyard = self.directory / ".discarded"
        graveyard.mkdir(exist_ok=True)
        target = graveyard / f"{path.name}.{reason}.{uuid.uuid4().hex}"
        try:
            os.rename(path, target)
        except OSError:
            return False
        shutil.rmtree(target, ignore_errors=True)
        return True


class _ProcessRunner:
    """``subprocess.run`` that keeps each job's process, so a stopping runner
    can terminate it and the claim can name it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: dict[int, subprocess.Popen] = {}
        self._started: dict[int, Callable[[int], None]] = {}

    def on_start(self, callback: Callable[[int], None]) -> None:
        with self._lock:
            self._started[threading.get_ident()] = callback

    def __call__(self, command, *, cwd, stdout, stderr, env):
        thread = threading.get_ident()
        process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr, env=env)
        with self._lock:
            self._processes[thread] = process
            callback = self._started.pop(thread, None)
        try:
            if callback is not None:
                try:
                    callback(process.pid)
                except OSError:
                    pass  # the claim records the job process for faster takeover only
            returncode = process.wait()
        finally:
            with self._lock:
                self._processes.pop(thread, None)
        return subprocess.CompletedProcess(command, returncode)

    def signal(self, thread: int | None, *, kill: bool = False) -> None:
        with self._lock:
            process = self._processes.get(thread) if thread is not None else None
        if process is None or process.poll() is not None:
            return
        if kill:
            process.kill()
        else:
            process.terminate()


@dataclass
class _Running:
    job: Mapping[str, Any]
    claim: _Claim
    thread: int | None = None


def run_job_plan(
    jobs: Sequence[Mapping[str, Any]],
    *,
    state_directory: str | Path,
    working_directory: str | Path,
    stages: Sequence[str] | None = None,
    job_ids: Sequence[str] | None = None,
    skip_simulator: bool = False,
    skip_gpu: bool = False,
    dry_run: bool = False,
    keep_going: bool = True,
    environment: Mapping[str, str] | None = None,
    deterministic_gpu: bool = True,
    runner: Any = None,
    workers: int = 1,
    training_slots: int = 1,
    poll_seconds: float = 30.0,
    heartbeat_seconds: float = 60.0,
    stale_after_seconds: float = 900.0,
    stop: threading.Event | None = None,
    termination_grace_seconds: float = 20.0,
) -> dict[str, Any]:
    """Execute jobs in plan order; resumable, dependency-aware, and failure-preserving.

    A job is *complete* when its last attempt succeeded and its expected
    outputs still exist. A job whose dependency failed, was skipped, or is
    blocked is *blocked* (never run). A failed job keeps its log; a runner
    started later retries it and everything it blocked, but no runner retries
    a failure that happened while it was running.

    Up to ``workers`` jobs run at once, at most ``training_slots`` of them
    GPU-training jobs. Other runners may execute the same plan concurrently
    with the same ``state_directory``: each job runs in one runner at a time,
    a runner waits for dependencies that another runner is executing (polling
    every ``poll_seconds``), and the job of a runner that was killed is taken
    over once its claim is stale (no heartbeat for ``stale_after_seconds``).
    Give every runner the same stages: a dependency outside a runner's
    selection blocks its dependents there. Setting ``stop`` makes the runner
    terminate its jobs, record them as interrupted (they rerun later), and
    return.

    With ``deterministic_gpu`` every job whose plan entry allows it runs with
    XLA's deterministic GPU ops, so identical jobs give bit-identical results
    and a resumed zoo run equals an uninterrupted one.
    """

    if workers < 1 or training_slots < 1:
        raise ValueError("workers and training_slots must be at least 1")
    root = Path(state_directory)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    records = _Records(root, adopt_legacy=not dry_run)
    order = [job["job_id"] for job in jobs]
    selected_stages = None if stages is None else set(stages)
    selected_ids = None if job_ids is None else set(job_ids)

    def skip_reason(job: Mapping[str, Any]) -> str | None:
        if not (
            (selected_stages is None or job["stage"] in selected_stages)
            and (selected_ids is None or job["job_id"] in selected_ids)
        ):
            return "not selected"
        if skip_simulator and job["resources"]["simulator_required"]:
            return "requires the LIBERO simulator (skipped on this host)"
        if skip_gpu and job["resources"]["gpu_required"]:
            return "requires a GPU (skipped on this host)"
        return None

    if dry_run:
        return _dry_run(jobs, records.read_all(order), skip_reason)

    host = socket.gethostname()
    scheduler_job = os.environ.get("SLURM_JOB_ID")
    owner = {
        "host": host,
        "pid": os.getpid(),
        "process_start": _process_start(os.getpid()),
        "scheduler_job": scheduler_job,
        "runner": f"{host}:{os.getpid()}" + (f" (job {scheduler_job})" if scheduler_job else ""),
    }
    claims = _Claims(root, owner, stale_after_seconds)
    execute_command = runner if runner is not None else _ProcessRunner()
    processes = execute_command if isinstance(execute_command, _ProcessRunner) else None
    stop = stop if stop is not None else threading.Event()
    # Failures already recorded when this runner started are retried once;
    # failures that happen while it runs are final for it.
    baseline = {job_id: int(r.get("attempts", 0)) for job_id, r in records.read_all(order).items()}
    attempted: set[str] = set()
    running: dict[str, _Running] = {}
    events: list[dict] = []
    skip_logged: set[str] = set()
    lock = threading.Lock()
    wake = threading.Event()
    finished = threading.Event()

    def classify(snapshot: Mapping[str, dict], now: float) -> tuple[dict, dict]:
        view: dict[str, str] = {}
        blocked_by: dict[str, list[str]] = {}
        with lock:
            active = set(running)
        for job in jobs:
            job_id = job["job_id"]
            record = snapshot.get(job_id)
            reason = skip_reason(job)
            if job_id in active:
                view[job_id] = "running"
            elif _is_complete(job, record):
                view[job_id] = "completed"
            elif reason is not None:
                view[job_id] = "skipped"
                if job_id not in skip_logged:
                    skip_logged.add(job_id)
                    with lock:
                        events.append({"job_id": job_id, "status": "skipped", "reason": reason})
            elif record and record.get("status") == "failed" and (
                job_id in attempted or int(record.get("attempts", 0)) > baseline.get(job_id, 0)
            ):
                view[job_id] = "failed"
            else:
                unmet = [d for d in job["dependencies"] if view[d] in ("failed", "skipped", "blocked")]
                if unmet:
                    view[job_id] = "blocked"
                    blocked_by[job_id] = unmet
                elif all(view[d] == "completed" for d in job["dependencies"]):
                    view[job_id] = "elsewhere" if claims.held_elsewhere(job_id, now) else "ready"
                else:
                    view[job_id] = "waiting"
        return view, blocked_by

    def execute(item: _Running) -> None:
        job = item.job
        job_id = job["job_id"]
        previous = records.read(job_id) or {}
        log_path = logs / (_file_name(job_id) + ".log")
        started = datetime.now(timezone.utc).isoformat()
        clock = perf_counter()
        try:
            with log_path.open("ab") as stream:
                stream.write(f"\n===== {started} [{owner['runner']}] :: {' '.join(job['command'])}\n".encode())
                stream.flush()
                overrides = job.get("environment") or {}
                deterministic = deterministic_gpu and job["resources"].get("deterministic_gpu", True)
                if environment is None and not overrides and not deterministic:
                    job_environment = None
                else:
                    job_environment = {
                        **(os.environ if environment is None else environment),
                        **overrides,
                    }
                    if deterministic:
                        job_environment = with_deterministic_gpu_ops(job_environment)
                if processes is not None:
                    processes.on_start(
                        lambda pid: claims.annotate(
                            item.claim, child_pid=pid, child_start=_process_start(pid)
                        )
                    )
                completed = execute_command(
                    job["command"],
                    cwd=str(working_directory),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=job_environment,
                )
            returncode = int(completed.returncode)
        except Exception as exc:  # the job could not be started at all
            with log_path.open("ab") as stream:
                stream.write(f"runner error: {exc!r}\n".encode())
            returncode = -1
        seconds = perf_counter() - clock
        if item.claim.lost:
            # Another runner took the job over; its record is the one that counts.
            event = {"job_id": job_id, "status": "claim_lost", "returncode": returncode, "log": str(log_path)}
        else:
            ok = returncode == 0 and (_outputs_exist(job) or not job["expected_outputs"])
            record = {
                "status": "completed" if ok else ("interrupted" if stop.is_set() else "failed"),
                "returncode": returncode,
                "started_at": started,
                "seconds": seconds,
                "log": str(log_path),
                "attempts": int(previous.get("attempts", 0)) + 1,
                "runner": owner["runner"],
            }
            if returncode == 0 and not ok:
                record["missing_outputs"] = [
                    item for item in job["expected_outputs"] if not Path(item).exists()
                ]
            records.write(job_id, record)
            claims.release(item.claim)
            event = {"job_id": job_id, **record}
        with lock:
            events.append(event)
            attempted.add(job_id)
            running.pop(job_id, None)
        wake.set()

    def heartbeat() -> None:
        while not finished.wait(heartbeat_seconds):
            with lock:
                items = list(running.values())
            for item in items:
                if not claims.heartbeat(item.claim) and processes is not None:
                    processes.signal(item.thread)

    beating = threading.Thread(target=heartbeat, name="job-claim-heartbeat", daemon=True)
    beating.start()
    failed_here = False
    terminating_since = None
    try:
        while True:
            wake.clear()
            now = claims.now()
            view, _ = classify(records.read_all(order), now)
            with lock:
                failed_here = failed_here or any(
                    event["status"] == "failed" for event in events if "returncode" in event
                )
                active = len(running)
            halted = stop.is_set() or (failed_here and not keep_going)
            if stop.is_set() and active and processes is not None:
                # Terminate this runner's jobs; kill those that outlive the grace period.
                with lock:
                    threads = [item.thread for item in running.values()]
                if terminating_since is None:
                    terminating_since = perf_counter()
                    for thread in threads:
                        processes.signal(thread)
                elif perf_counter() - terminating_since > termination_grace_seconds:
                    for thread in threads:
                        processes.signal(thread, kill=True)
            if not halted:
                for job in jobs:
                    job_id = job["job_id"]
                    if view[job_id] != "ready":
                        continue
                    with lock:
                        if len(running) >= workers:
                            break
                        training = sum(_is_training(item.job) for item in running.values())
                    if _is_training(job) and training >= training_slots:
                        continue
                    claim = claims.acquire(job_id, now)
                    if claim is None:
                        view[job_id] = "elsewhere"
                        continue
                    # Another runner may have finished or failed it since the snapshot.
                    record = records.read(job_id)
                    if _is_complete(job, record) or (
                        record
                        and record.get("status") == "failed"
                        and int(record.get("attempts", 0)) > baseline.get(job_id, 0)
                    ):
                        claims.release(claim)
                        continue
                    item = _Running(job, claim)
                    thread = threading.Thread(target=execute, args=(item,), name=f"job {job_id}", daemon=True)
                    with lock:
                        running[job_id] = item
                    thread.start()
                    item.thread = thread.ident
                    view[job_id] = "running"
            with lock:
                active = len(running)
            pending = any(view[job_id] in ("ready", "waiting") for job_id in order)
            if active == 0 and (halted or not pending):
                break
            # Wake when one of this runner's jobs ends, or poll other runners' progress.
            deadline = perf_counter() + (1.0 if stop.is_set() else poll_seconds)
            while not wake.is_set() and not stop.is_set() and perf_counter() < deadline:
                wake.wait(min(1.0, max(0.0, deadline - perf_counter())))
    finally:
        finished.set()
        beating.join(timeout=5)

    snapshot = records.read_all(order)
    view, blocked_by = classify(snapshot, claims.now())
    claims.clock.unlink(missing_ok=True)
    final = {
        job_id: {"elsewhere": "running", "ready": "pending", "waiting": "pending"}.get(state, state)
        for job_id, state in view.items()
    }
    failed = [job_id for job_id in order if final[job_id] == "failed"]
    summary = {
        "schema_version": JOB_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": False,
        "runner": owner["runner"],
        "interrupted": stop.is_set(),
        "counts": dict(sorted(Counter(final.values()).items())),
        "failed": [
            {
                "job_id": job_id,
                "returncode": snapshot[job_id].get("returncode"),
                "log": snapshot[job_id].get("log"),
                "runner": snapshot[job_id].get("runner"),
                "missing_outputs": snapshot[job_id].get("missing_outputs", []),
                "log_tail": _tail(snapshot[job_id].get("log")),
            }
            for job_id in failed
        ],
        "blocked": [
            {"job_id": job_id, "blocked_by": blocked_by[job_id]}
            for job_id in order
            if final[job_id] == "blocked"
        ],
        "skipped": [event for event in events if event["status"] == "skipped"],
        "events": events,
    }
    atomic_write_json(root / FAILURE_SUMMARY, {k: v for k, v in summary.items() if k != "events"})
    atomic_write_json(root / STATE_FILE, {"jobs": snapshot})
    return summary


def _dry_run(jobs, snapshot: Mapping[str, dict], skip_reason) -> dict[str, Any]:
    """What a single runner would start now; executes and writes nothing."""

    status: dict[str, str] = {}
    events = []
    for job in jobs:
        job_id = job["job_id"]
        reason = skip_reason(job)
        if _is_complete(job, snapshot.get(job_id)):
            status[job_id] = "completed"
        elif reason is not None:
            status[job_id] = "skipped"
            events.append({"job_id": job_id, "status": "skipped", "reason": reason})
        elif unmet := [dep for dep in job["dependencies"] if status.get(dep) != "completed"]:
            status[job_id] = "blocked"
            events.append({"job_id": job_id, "status": "blocked", "blocked_by": unmet})
        else:
            status[job_id] = "would_run"
            events.append({"job_id": job_id, "status": "would_run", "command": job["command"]})
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "counts": dict(sorted(Counter(status.values()).items())),
        "failed": [],
        "blocked": [
            {"job_id": event["job_id"], "blocked_by": event["blocked_by"]}
            for event in events
            if event["status"] == "blocked"
        ],
        "skipped": [event for event in events if event["status"] == "skipped"],
        "events": events,
    }


def _tail(path: str | None, lines: int = 15) -> list[str]:
    if not path or not Path(path).is_file():
        return []
    return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
