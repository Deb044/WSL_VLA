from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.synthetic import synthetic_manifest
from wsl_vla.experiments.compute_budget import (
    WARMUP_STEPS,
    collect_measurements,
    experiment_workload,
    project_workload,
    rollout_condition_count,
    rollout_step_cost,
    wall_clock,
)
from wsl_vla.experiments.manifest import ExperimentPaths, load_experiment_manifest
from wsl_vla.experiments.protocol import load_yaml

REPOSITORY = Path(__file__).resolve().parents[3]


def test_full_manifest_workload_is_counted_exactly():
    tasks = load_yaml(REPOSITORY / "configs/reference_tasks.yaml")
    full = load_experiment_manifest(REPOSITORY / "configs/research/manifests/full.yaml", reference_tasks=tasks)
    conditions = rollout_condition_count(full)
    assert conditions == 10
    work = experiment_workload(full, max_episode_steps=600, conditions=conditions)
    assert work["zoo"]["runs"] == full.expected["independent_runs"] == 120
    assert work["zoo"]["adapter_snapshots"] == full.expected["adapter_samples"] == 360
    assert work["zoo"]["training_samples_total"] == 120 * 5000 * 16
    assert work["zoo_behaviour"]["episodes_total"] == 120 * 2 * 20
    assert work["alignment"]["runs"] == 4 * 3 * 2 and work["alignment"]["optimizer_updates_total"] == 480_000
    assert work["rollouts_validation"]["episodes_total"] == 4 * 3 * 6 * 10 * 20
    assert work["rollouts_held_out"]["episodes_total"] == 4 * 3 * 10 * 10 * 50
    assert work["max_steps_per_episode"] == 600 + WARMUP_STEPS


def test_rollout_cost_separates_start_up_from_steps_and_says_when_it_cannot():
    jobs = [{"simulator_steps": steps, "job_wall_seconds": 90 + 0.3 * steps} for steps in (1000, 4000, 9000, 24000)]
    fit = rollout_step_cost(jobs)
    assert fit["method"] == "least squares over jobs"
    assert fit["seconds_per_simulator_step"] == pytest.approx(0.3)
    assert fit["fixed_seconds_per_job"] == pytest.approx(90)
    same_length = [{"simulator_steps": 1000, "job_wall_seconds": 390}] * 3
    pooled = rollout_step_cost(same_length)
    assert pooled["method"].startswith("pooled") and pooled["seconds_per_simulator_step"] == pytest.approx(0.39)
    assert rollout_step_cost([]) is None


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_measured_unit_costs_reproduce_the_executed_experiment(tmp_path):
    manifest = synthetic_manifest()
    paths = ExperimentPaths(tmp_path, manifest.name)
    rate, overhead, per_step, fixed = 20.0, 120.0, 0.3, 60.0
    micro_steps = manifest.zoo.optimizer_updates * manifest.zoo.gradient_accumulation_steps
    final = manifest.zoo.snapshot_updates[-1]
    for index, (identifier, seed) in enumerate(manifest.runs()):
        suite = identifier.rsplit("_", 1)[0]
        run = paths.population / suite / identifier / f"seed_{seed}"
        _write(run / "run.json", {
            "completed": True,
            "micro_steps": micro_steps,
            "run_identity": {"task_id": identifier, "seed": seed, "budget": {"micro_batch_size": manifest.zoo.micro_batch_size}},
            "resources": {"steady_micro_steps_per_second": rate, "peak_vram_bytes": 2**33},
            "provenance": {"resumed_from_update": None, "environment": {"hardware": {"nvidia_gpus": ["test"]}}},
        })
        _write(paths.jobs / "records" / f"zoo__{identifier}__seed-{seed}.json", {
            "status": "completed", "attempts": 1, "seconds": overhead + micro_steps / rate,
            "started_at": "2026-10-15T00:00:00+00:00",
        })
        lengths = [50 + 25 * ((index + k) % 7) for k in range(2 * manifest.rollouts.development)]
        episodes = [{"steps": n, "success": n < 150, "task_id": identifier} for n in lengths]
        half = len(episodes) // 2
        _write(run / f"update_{final}" / "behaviour.json", {
            "frozen_base": {"episodes": episodes[:half]}, "adapter": {"episodes": episodes[half:]},
        })
        steps = sum(n + WARMUP_STEPS for n in lengths)
        _write(paths.jobs / "records" / f"zoo_behaviour__{identifier}__seed-{seed}.json", {
            "status": "completed", "attempts": 1, "seconds": fixed + per_step * steps,
            "started_at": "2026-10-15T01:00:00+00:00",
        })

    measured = collect_measurements(manifest, tmp_path)
    assert measured["zoo"]["summary"]["uninterrupted_runs"] == len(manifest.runs())
    assert measured["zoo"]["summary"]["peak_vram_gib"]["max"] == pytest.approx(8.0)
    fit = measured["zoo_behaviour"]["summary"]["cost_model"]
    assert fit["seconds_per_simulator_step"] == pytest.approx(per_step)
    assert fit["fixed_seconds_per_job"] == pytest.approx(fixed)

    work = experiment_workload(manifest, max_episode_steps=600, conditions=10)
    projection = project_workload(work, measured)
    zoo = projection["stages"]["zoo_training"]
    assert zoo["valid"] and zoo["seconds_per_job"] == pytest.approx(overhead + micro_steps / rate)
    # Projecting the experiment onto itself reproduces its measured job time.
    behaviour_seconds = sum(
        json.loads(p.read_text())["seconds"] for p in (paths.jobs / "records").glob("zoo_behaviour__*.json")
    )
    assert projection["stages"]["zoo_behaviour"]["job_hours"] * 3600 == pytest.approx(behaviour_seconds, rel=1e-6)
    # Not measured yet: reported as missing, never filled in.
    assert projection["stages"]["alignment_training"]["job_hours"] is None
    assert "alignment_training" in projection["missing"]
    horizon = project_workload(work, measured, episode_policy_steps=work["max_steps_per_episode"])
    assert horizon["stages"]["zoo_behaviour"]["job_hours"] > projection["stages"]["zoo_behaviour"]["job_hours"]
    schedule = wall_clock(work, projection, runners=4, workers=3, training_slots=1)
    assert schedule["phase_hours"]["zoo"] > 0 and "alignment" in schedule["unknown_phases"]
