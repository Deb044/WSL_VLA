"""Compute budget of a zoo/alignment experiment: exact workload, measured unit costs, projection.

Three parts, kept separate so that a report can say which numbers are counted,
which are measured, and which are projected:

* :func:`experiment_workload` counts the work a manifest defines (training
  micro-steps, rollout episodes, alignment updates). It reads only the manifest.
* :func:`collect_measurements` reads the results of an executed experiment (the
  runner's job records, each zoo run's ``run.json``, behaviour and rollout
  episodes, alignment metadata) and reduces them to unit costs, together with the
  hardware they were measured on and the disk space the results occupy.
* :func:`project_workload` multiplies the counts of a target manifest by the
  measured unit costs. Every projected quantity names the measurement it used.
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

from .manifest import ExperimentManifest, ExperimentPaths

COMPUTE_BUDGET_SCHEMA_VERSION = 1
WARMUP_STEPS = 5  # zero-action physics steps before every episode (base.yaml simulator.warmup_steps)


def _stats(values: Iterable[float]) -> dict[str, float] | None:
    data = [float(value) for value in values]
    if not data:
        return None
    return {
        "n": len(data),
        "median": statistics.median(data),
        "mean": statistics.fmean(data),
        "min": min(data),
        "max": max(data),
        "sum": sum(data),
    }


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _record_name(job_id: str) -> str:
    # Same mapping as the job runner's per-job record files.
    return job_id.replace("/", "_").replace(":", "__")


# ------------------------------------------------------------------ workload
def rollout_condition_count(manifest: ExperimentManifest) -> int:
    from ..evaluation.generated_policies import condition_names

    return len(condition_names(manifest.alignment.conditions))


def experiment_workload(
    manifest: ExperimentManifest, *, max_episode_steps: int, conditions: int
) -> dict[str, Any]:
    """Count the work of every zoo/alignment stage; nothing here is estimated."""

    zoo = manifest.zoo
    runs = len(manifest.runs())
    micro_steps = zoo.optimizer_updates * zoo.gradient_accumulation_steps
    development, final = manifest.rollouts.development, manifest.rollouts.final
    inspection_updates = max(zoo.inspection_updates) if zoo.inspection_updates else 0
    alignment_runs = len(manifest.folds) * len(manifest.alignment.seeds) * len(manifest.alignment.conditions)
    fold_seeds = len(manifest.folds) * len(manifest.alignment.seeds)
    validation_episodes = sum(len(f.validation_tasks) for f in manifest.folds) * len(manifest.alignment.seeds) * conditions * development
    held_out_episodes = sum(len(f.held_out_tasks) for f in manifest.folds) * len(manifest.alignment.seeds) * conditions * final
    return {
        "manifest": manifest.name,
        "tasks": len(manifest.tasks),
        "seeds": len(manifest.seeds),
        "max_steps_per_episode": max_episode_steps + WARMUP_STEPS,
        "zoo": {
            "runs": runs,
            "adapter_snapshots": runs * len(zoo.snapshot_updates),
            "optimizer_updates_per_run": zoo.optimizer_updates,
            "micro_batch_size": zoo.micro_batch_size,
            "gradient_accumulation_steps": zoo.gradient_accumulation_steps,
            "micro_steps_per_run": micro_steps,
            "training_samples_per_run": micro_steps * zoo.micro_batch_size,
            "training_samples_total": runs * micro_steps * zoo.micro_batch_size,
            "held_out_loss_evaluations_per_run": zoo.optimizer_updates // zoo.held_out_loss_every,
            "held_out_loss_batches_per_evaluation": zoo.held_out_loss_batches,
        },
        "budget_inspection": {
            "runs": len(zoo.inspection_tasks),
            "optimizer_updates_per_run": inspection_updates,
            "micro_steps_per_run": inspection_updates * zoo.gradient_accumulation_steps,
            "behaviour_episodes_per_run": len(zoo.inspection_updates) * 2 * development,
        },
        "zoo_behaviour": {
            "jobs": runs,
            "episodes_per_job": 2 * development,  # frozen base and adapter, paired
            "episodes_total": runs * 2 * development,
        },
        "evidence": {"jobs": len(manifest.tasks)},
        "archive": {"jobs": len(manifest.folds)},
        "alignment": {
            "runs": alignment_runs,
            "optimizer_updates_per_run": manifest.alignment.optimizer_updates,
            "adapter_samples_per_update": manifest.alignment.tasks_per_batch * manifest.alignment.samples_per_task,
            "validation_every": manifest.alignment.validation_every,
            "optimizer_updates_total": alignment_runs * manifest.alignment.optimizer_updates,
        },
        "diagnostics": {"jobs_validation": fold_seeds, "jobs_held_out": fold_seeds},
        "rollouts_validation": {
            "jobs": fold_seeds,
            "conditions": conditions,
            "episodes_per_condition_task": development,
            "episodes_total": validation_episodes,
        },
        "rollouts_held_out": {
            "jobs": fold_seeds,
            "conditions": conditions,
            "episodes_per_condition_task": final,
            "episodes_total": held_out_episodes,
        },
    }


# --------------------------------------------------------------- measurement
def _records(paths: ExperimentPaths) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted((paths.jobs / "records").glob("*.json")):
        payload = _read(path)
        if payload is not None:
            out[path.stem] = payload
    return out


def _clean_seconds(record: Mapping[str, Any] | None) -> float | None:
    """Wall time of a job that completed in one uninterrupted attempt."""

    if not record or record.get("status") != "completed" or int(record.get("attempts", 1)) != 1:
        return None
    return float(record["seconds"])


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.is_dir() else 0


def _zoo_run(run: Mapping[str, Any], record: Mapping[str, Any] | None) -> dict[str, Any]:
    budget = run["run_identity"]["budget"]
    resources = run.get("resources", {})
    micro_batch = int(budget["micro_batch_size"])
    steady = resources.get("steady_micro_steps_per_second")
    return {
        "task_id": run["run_identity"]["task_id"],
        "seed": run["run_identity"]["seed"],
        "micro_batch_size": micro_batch,
        "micro_steps": run.get("micro_steps"),
        "resumed_from_update": run.get("provenance", {}).get("resumed_from_update"),
        "steady_micro_steps_per_second": steady,
        "steady_samples_per_second": None if steady is None else steady * micro_batch,
        "compile_seconds": resources.get("compile_seconds"),
        "snapshot_and_resume_seconds": resources.get("snapshot_and_resume_seconds"),
        "held_out_evaluation_seconds": resources.get("held_out_evaluation_seconds"),
        "train_seconds": resources.get("train_seconds"),
        "peak_vram_gib": None if resources.get("peak_vram_bytes") is None else resources["peak_vram_bytes"] / 2**30,
        "peak_ram_gib": None if resources.get("peak_ram_bytes") is None else resources["peak_ram_bytes"] / 2**30,
        "job_wall_seconds": _clean_seconds(record),
    }


def _episodes_summary(episodes: list[Mapping[str, Any]]) -> dict[str, Any]:
    steps = [int(item["steps"]) for item in episodes]
    succeeded = [int(item["steps"]) for item in episodes if item.get("success")]
    return {
        "episodes": len(episodes),
        "successes": len(succeeded),
        "policy_steps": _stats(steps),
        "policy_steps_of_successes": _stats(succeeded),
    }


def _rollout_job(episodes: list[Mapping[str, Any]], seconds: float | None) -> dict[str, Any]:
    summary = _episodes_summary(episodes)
    simulator_steps = sum(int(item["steps"]) + WARMUP_STEPS for item in episodes)
    summary["simulator_steps"] = simulator_steps
    summary["tasks"] = len({str(item.get("task_id")) for item in episodes})
    summary["job_wall_seconds"] = seconds
    summary["wall_seconds_per_simulator_step"] = (
        None if seconds is None or not simulator_steps else seconds / simulator_steps
    )
    return summary


def rollout_step_cost(jobs: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Split rollout job wall time into a fixed part per job and a part per simulator step.

    Least squares ``seconds = fixed + per_step * simulator_steps`` over jobs whose
    episode lengths differ (successful episodes end early). Without that spread
    the two cannot be separated and the pooled ratio is returned instead, which
    charges start-up to every step and therefore overstates long jobs.
    """

    timed = [(job["simulator_steps"], job["job_wall_seconds"]) for job in jobs if job.get("job_wall_seconds") and job.get("simulator_steps")]
    if not timed:
        return None
    steps = [item[0] for item in timed]
    seconds = [item[1] for item in timed]
    if len(timed) >= 3 and max(steps) > 1.1 * min(steps):
        mean_steps, mean_seconds = statistics.fmean(steps), statistics.fmean(seconds)
        slope = sum((x - mean_steps) * (y - mean_seconds) for x, y in timed) / sum((x - mean_steps) ** 2 for x in steps)
        fixed = mean_seconds - slope * mean_steps
        if slope > 0 and fixed >= 0:
            residuals = [y - (fixed + slope * x) for x, y in timed]
            return {
                "method": "least squares over jobs",
                "jobs": len(timed),
                "seconds_per_simulator_step": slope,
                "fixed_seconds_per_job": fixed,
                "residual_seconds": _stats(abs(value) for value in residuals),
                "simulator_steps_range": [min(steps), max(steps)],
            }
    return {
        "method": "pooled ratio (start-up not separated: overstates long jobs)",
        "jobs": len(timed),
        "seconds_per_simulator_step": sum(seconds) / sum(steps),
        "fixed_seconds_per_job": 0.0,
        "simulator_steps_range": [min(steps), max(steps)],
    }


def collect_measurements(manifest: ExperimentManifest, research_root: str | Path) -> dict[str, Any]:
    """Unit costs measured in an executed experiment (its results directory)."""

    paths = ExperimentPaths(Path(research_root), manifest.name)
    records = _records(paths)
    hardware: dict[str, Any] = {}

    def remember_environment(payload: Mapping[str, Any]) -> None:
        environment = payload.get("provenance", {}).get("environment", {})
        if environment and not hardware:
            hardware.update(
                {
                    "hardware": environment.get("hardware"),
                    "profile": environment.get("profile"),
                    "packages": environment.get("packages"),
                }
            )

    zoo_runs, inspection_runs = [], []
    for run_path in sorted(paths.population.glob("*/*/seed_*/run.json")):
        run = _read(run_path)
        if run and run.get("completed"):
            remember_environment(run)
            identity = run["run_identity"]
            job = f"zoo:{identity['task_id']}:seed-{identity['seed']}"
            zoo_runs.append(_zoo_run(run, records.get(_record_name(job))))
    for run_path in sorted(paths.budget_inspection.glob("*/*/seed_*/run.json")):
        run = _read(run_path)
        if run and run.get("completed"):
            identity = run["run_identity"]
            job = f"budget_inspection:train:{identity['task_id']}"
            inspection_runs.append(_zoo_run(run, records.get(_record_name(job))))

    behaviour_jobs = []
    for run_directory in sorted(paths.population.glob("*/*/seed_*")):
        files = sorted(run_directory.glob("update_*/behaviour.json"))
        if not files:
            continue
        task_id, seed = run_directory.parent.name, run_directory.name.removeprefix("seed_")
        episodes = []
        for path in files:
            payload = _read(path) or {}
            for policy in ("frozen_base", "adapter"):
                episodes.extend(payload.get(policy, {}).get("episodes", []))
        seconds = _clean_seconds(records.get(_record_name(f"zoo_behaviour:{task_id}:seed-{seed}")))
        behaviour_jobs.append({"task_id": task_id, "seed": seed, **_rollout_job(episodes, seconds)})

    alignment_runs = []
    for metadata_path in sorted((paths.root / "alignment").glob("*/seed_*/*/metadata.json")):
        metadata = _read(metadata_path)
        if not metadata:
            continue
        history = [item for item in metadata.get("history", []) if "seconds" in item]
        steady = None
        if len(history) >= 2:
            updates = history[-1]["optimizer_update"] - history[0]["optimizer_update"]
            if updates > 0:
                steady = (history[-1]["seconds"] - history[0]["seconds"]) / updates
        fold, seed, condition = metadata_path.parts[-4], metadata_path.parts[-3].removeprefix("seed_"), metadata_path.parts[-2]
        alignment_runs.append(
            {
                "fold": fold,
                "seed": seed,
                "condition": condition,
                "optimizer_updates": metadata.get("optimizer_exposure", {}).get("optimizer_updates"),
                "adapter_samples_per_update": metadata.get("config", {}).get("micro_batch_size"),
                "train_task_count": metadata.get("config", {}).get("train_task_count"),
                "steady_seconds_per_update": steady,
                "train_seconds": metadata.get("resources", {}).get("train_seconds"),
                "job_wall_seconds": _clean_seconds(records.get(_record_name(f"alignment:{fold}:seed-{seed}:{condition}"))),
            }
        )

    generated = {}
    for scope in ("validation", "held_out"):
        jobs = []
        for directory in sorted((paths.root / "rollouts" / scope).glob("*/seed_*")):
            if not (directory / "summary.json").is_file():
                continue  # unfinished: its wall time is not that of the whole job
            episodes = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines() if line.strip()]
            fold, seed = directory.parent.name, directory.name.removeprefix("seed_")
            seconds = _clean_seconds(records.get(_record_name(f"rollouts:{scope}:{fold}:seed-{seed}")))
            jobs.append({"fold": fold, "seed": seed, **_rollout_job(episodes, seconds)})
        generated[scope] = jobs

    from datetime import datetime, timedelta

    spans: dict[str, list[datetime]] = {}
    for name, record in records.items():
        if record.get("status") != "completed" or not record.get("started_at"):
            continue
        start = datetime.fromisoformat(record["started_at"])
        end = start + timedelta(seconds=float(record["seconds"]))
        span = spans.setdefault(name.split("__")[0], [start, end])
        span[0], span[1] = min(span[0], start), max(span[1], end)
    observed = {
        stage: {"first_start": start.isoformat(), "last_end": end.isoformat(), "span_hours": (end - start).total_seconds() / 3600}
        for stage, (start, end) in sorted(spans.items())
    }

    stage_seconds: dict[str, list[float]] = {}
    excluded = 0
    for name, record in records.items():
        seconds = _clean_seconds(record)
        stage = name.split("__")[0]
        if seconds is None:
            excluded += record.get("status") == "completed"
            continue
        stage_seconds.setdefault(stage, []).append(seconds)

    def pooled(jobs: list[Mapping[str, Any]]) -> dict[str, Any]:
        timed = [job for job in jobs if job["wall_seconds_per_simulator_step"] is not None]
        steps = sum(job["simulator_steps"] for job in timed)
        return {
            "jobs": len(jobs),
            "jobs_timed": len(timed),
            "cost_model": rollout_step_cost(jobs),
            "wall_seconds_per_simulator_step": None if not steps else sum(job["job_wall_seconds"] for job in timed) / steps,
            "per_job_seconds_per_step": _stats(job["wall_seconds_per_simulator_step"] for job in timed),
            "episodes": sum(job["episodes"] for job in jobs),
            "successes": sum(job["successes"] for job in jobs),
            "mean_policy_steps_per_episode": (
                None
                if not sum(job["episodes"] for job in jobs)
                else sum(job["policy_steps"]["sum"] for job in jobs if job["policy_steps"])
                / sum(job["episodes"] for job in jobs)
            ),
        }

    def summarize_runs(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
        fresh = [run for run in runs if run["resumed_from_update"] is None]
        return {
            "runs": len(runs),
            "uninterrupted_runs": len(fresh),
            **{
                key: _stats(run[key] for run in fresh if run[key] is not None)
                for key in (
                    "steady_micro_steps_per_second",
                    "steady_samples_per_second",
                    "compile_seconds",
                    "snapshot_and_resume_seconds",
                    "held_out_evaluation_seconds",
                    "train_seconds",
                    "job_wall_seconds",
                    "peak_vram_gib",
                    "peak_ram_gib",
                )
            },
        }

    storage = {
        name: _directory_bytes(path) / 2**30
        for name, path in {
            "population": paths.population,
            "zoo_resume": paths.resume,
            "budget_inspection": paths.budget_inspection,
            "evidence": paths.evidence,
            "alignment": paths.root / "alignment",
            "diagnostics": paths.root / "diagnostics",
            "rollouts": paths.root / "rollouts",
            "job_logs": paths.jobs,
        }.items()
    }
    adapters = [p.stat().st_size for p in paths.population.glob("*/*/seed_*/update_*/adapter.npz")]
    return {
        "schema_version": COMPUTE_BUDGET_SCHEMA_VERSION,
        "experiment": manifest.name,
        "research_root": str(research_root),
        "environment": hardware,
        "zoo": {"summary": summarize_runs(zoo_runs), "runs": zoo_runs},
        "budget_inspection": {"summary": summarize_runs(inspection_runs), "runs": inspection_runs},
        "zoo_behaviour": {"summary": pooled(behaviour_jobs), "jobs": behaviour_jobs},
        "alignment": {
            "summary": {
                "runs": len(alignment_runs),
                "steady_seconds_per_update": _stats(r["steady_seconds_per_update"] for r in alignment_runs if r["steady_seconds_per_update"] is not None),
                "job_wall_seconds": _stats(r["job_wall_seconds"] for r in alignment_runs if r["job_wall_seconds"] is not None),
            },
            "runs": alignment_runs,
        },
        "generated_rollouts": {scope: {"summary": pooled(jobs), "jobs": jobs} for scope, jobs in generated.items()},
        "job_wall_seconds_by_stage": {stage: _stats(values) for stage, values in sorted(stage_seconds.items())},
        "completed_jobs_excluded_from_timing": excluded,
        # First start to last end of each stage's completed jobs (last attempt of each);
        # includes any idle time between submissions.
        "observed_stage_spans": observed,
        "storage_gib": storage,
        "adapter_snapshot_bytes": _stats(adapters),
    }


# ---------------------------------------------------------------- projection
def _median(measured: Mapping[str, Any], *keys: str) -> float | None:
    value: Any = measured
    for key in keys:
        if not isinstance(value, Mapping) or value.get(key) is None:
            return None
        value = value[key]
    return value["median"] if isinstance(value, Mapping) else value


def project_workload(
    workload: Mapping[str, Any],
    measured: Mapping[str, Any],
    *,
    episode_policy_steps: float | None = None,
) -> dict[str, Any]:
    """Job-hours of each stage: exact counts times measured unit costs.

    ``episode_policy_steps`` overrides the measured mean episode length (for a
    bound: the horizon). Each entry states the unit cost and its source; an
    entry whose unit cost was not measured is reported as ``None``.
    """

    hours: dict[str, Any] = {}
    max_steps = workload["max_steps_per_episode"]

    # Zoo training: per-run overhead (load, compile, evaluation, snapshots) measured
    # as job wall time minus steady training time, plus micro-steps at the steady rate.
    zoo = workload["zoo"]
    rate = _median(measured, "zoo", "summary", "steady_micro_steps_per_second")
    wall = _median(measured, "zoo", "summary", "job_wall_seconds")
    runs = measured.get("zoo", {}).get("runs", [])
    overheads = [
        run["job_wall_seconds"] - run["micro_steps"] / run["steady_micro_steps_per_second"]
        for run in runs
        if run["job_wall_seconds"] and run["steady_micro_steps_per_second"] and run["micro_steps"] and run["resumed_from_update"] is None
    ]
    overhead = statistics.median(overheads) if overheads else None
    micro_batches = {run["micro_batch_size"] for run in runs}
    per_run = None
    if rate and overhead is not None:
        per_run = overhead + zoo["micro_steps_per_run"] / rate
    hours["zoo_training"] = {
        "jobs": zoo["runs"],
        "seconds_per_job": per_run,
        "job_hours": None if per_run is None else zoo["runs"] * per_run / 3600,
        "unit_cost": {
            "steady_micro_steps_per_second": rate,
            "per_run_overhead_seconds": overhead,
            "measured_micro_batch_sizes": sorted(micro_batches),
            "measured_median_job_seconds": wall,
        },
        "valid": bool(micro_batches) and micro_batches == {zoo["micro_batch_size"]},
    }

    behaviour = (measured.get("zoo_behaviour") or {}).get("summary") or {}
    step_cost = behaviour.get("cost_model") or {}
    per_step = step_cost.get("seconds_per_simulator_step")
    measured_length = behaviour.get("mean_policy_steps_per_episode")
    length = episode_policy_steps or measured_length
    capped = None if length is None else min(length, max_steps - WARMUP_STEPS) + WARMUP_STEPS

    def rollout_entry(episodes, jobs, fixed_per_job, fixed_source):
        steps = None if capped is None else episodes * capped
        seconds = None if per_step is None or steps is None or fixed_per_job is None else steps * per_step + jobs * fixed_per_job
        return {
            "jobs": jobs,
            "episodes": episodes,
            "simulator_steps": steps,
            "job_hours": None if seconds is None else seconds / 3600,
            "unit_cost": {
                "seconds_per_simulator_step": per_step,
                "step_cost_method": step_cost.get("method"),
                "policy_steps_per_episode": length,
                "policy_steps_source": "assumed" if episode_policy_steps else "mean of measured zoo behaviour episodes",
                "fixed_seconds_per_job": fixed_per_job,
                "fixed_source": fixed_source,
            },
        }

    behaviour_jobs = workload["zoo_behaviour"]["jobs"]
    hours["zoo_behaviour"] = rollout_entry(
        workload["zoo_behaviour"]["episodes_total"], behaviour_jobs,
        step_cost.get("fixed_seconds_per_job"), "zoo behaviour fit",
    )

    alignment = workload["alignment"]
    per_update = _median(measured, "alignment", "summary", "steady_seconds_per_update")
    alignment_runs = measured.get("alignment", {}).get("runs", [])
    align_overheads = [
        run["job_wall_seconds"] - run["optimizer_updates"] * run["steady_seconds_per_update"]
        for run in alignment_runs
        if run["job_wall_seconds"] and run["steady_seconds_per_update"] and run["optimizer_updates"]
    ]
    align_overhead = statistics.median(align_overheads) if align_overheads else None
    align_per_run = None if per_update is None or align_overhead is None else align_overhead + alignment["optimizer_updates_per_run"] * per_update
    hours["alignment_training"] = {
        "jobs": alignment["runs"],
        "seconds_per_job": align_per_run,
        "job_hours": None if align_per_run is None else alignment["runs"] * align_per_run / 3600,
        "unit_cost": {"steady_seconds_per_update": per_update, "per_run_overhead_seconds": align_overhead},
    }

    # Generated-policy rollouts reuse the per-step cost (the same Octo policy and
    # simulator); their fixed cost is per task (ten policies are built per task),
    # measured from the validation rollout jobs as wall time minus step time.
    generated_jobs = ((measured.get("generated_rollouts") or {}).get("validation") or {}).get("jobs", [])
    per_task = [
        (job["job_wall_seconds"] - job["simulator_steps"] * per_step) / job["tasks"]
        for job in generated_jobs
        if per_step and job.get("job_wall_seconds") and job.get("tasks")
    ]
    per_task_fixed = statistics.median(per_task) if per_task else None
    if per_task_fixed is not None and per_task_fixed < 0:
        # The per-step cost overstates these jobs (typically a pooled ratio that
        # still contains start-up); charge no fixed cost rather than a negative one.
        per_task_fixed = 0.0
    for scope in ("validation", "held_out"):
        entry = workload[f"rollouts_{scope}"]
        tasks = entry["episodes_total"] // (entry["conditions"] * entry["episodes_per_condition_task"])
        projected = rollout_entry(entry["episodes_total"], tasks, per_task_fixed, "per task, measured validation rollout jobs")
        projected["jobs"] = entry["jobs"]
        projected["tasks"] = tasks
        hours[f"rollouts_{scope}"] = projected

    by_stage = measured.get("job_wall_seconds_by_stage", {})
    for stage, jobs in (
        ("evidence", workload["evidence"]["jobs"]),
        ("archive", workload["archive"]["jobs"]),
        ("diagnostics", workload["diagnostics"]["jobs_validation"] + workload["diagnostics"]["jobs_held_out"]),
    ):
        seconds = (by_stage.get(stage) or {}).get("median")
        hours[stage] = {
            "jobs": jobs,
            "seconds_per_job": seconds,
            "job_hours": None if seconds is None else jobs * seconds / 3600,
            "unit_cost": {"source": f"median measured {stage} job"},
        }
    known = [entry["job_hours"] for entry in hours.values() if entry.get("job_hours") is not None]
    return {"stages": hours, "job_hours_total_of_measured_stages": sum(known), "missing": [k for k, v in hours.items() if v.get("job_hours") is None]}


def wall_clock(
    workload: Mapping[str, Any],
    projection: Mapping[str, Any],
    *,
    runners: int,
    workers: int,
    training_slots: int,
) -> dict[str, Any]:
    """Wall-clock hours per phase on ``runners`` slices, each running ``workers`` jobs
    of which at most ``training_slots`` train. Phases follow the job dependencies:
    zoo (with behaviour rollouts and evidence), alignment, validation, held-out.
    Within a phase jobs are packed greedily, so a phase lasts at least as long as
    its longest job; this is a schedule, not a lower bound on the hardware.
    """

    import math

    stages = projection["stages"]

    def hours(name: str) -> float | None:
        return stages[name]["job_hours"]

    def per_job(name: str, jobs: int) -> float | None:
        total = hours(name)
        return None if total is None or not jobs else total / jobs

    training = runners * training_slots
    slots = runners * workers
    phases: dict[str, float | None] = {}

    zoo, behaviour = hours("zoo_training"), hours("zoo_behaviour")
    zoo_job = per_job("zoo_training", workload["zoo"]["runs"])
    behaviour_job = per_job("zoo_behaviour", workload["zoo_behaviour"]["jobs"])
    if None in (zoo, behaviour, zoo_job, behaviour_job):
        phases["zoo"] = None
    else:
        training_waves = math.ceil(workload["zoo"]["runs"] / training) * zoo_job
        phases["zoo"] = max(training_waves, (zoo + behaviour) / slots) + behaviour_job

    align_job = per_job("alignment_training", workload["alignment"]["runs"])
    phases["alignment"] = None if align_job is None else math.ceil(workload["alignment"]["runs"] / training) * align_job

    for scope in ("validation", "held_out"):
        name = f"rollouts_{scope}"
        jobs = workload[name]["jobs"]
        job = per_job(name, jobs)
        phases[scope] = None if job is None else math.ceil(jobs / slots) * job
    known = [value for value in phases.values() if value is not None]
    return {
        "runners": runners,
        "workers_per_runner": workers,
        "training_slots_per_runner": training_slots,
        "phase_hours": phases,
        "total_hours_of_known_phases": sum(known),
        "unknown_phases": [name for name, value in phases.items() if value is None],
    }
