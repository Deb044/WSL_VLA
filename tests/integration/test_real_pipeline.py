"""Real Octo/LIBERO integration test: train, save, reload, reconstruct, map, evaluate.

This drives the *actual* command-line pipeline through the resumable job runner
on a reduced copy of the integrity manifest (two tasks, a few optimizer
updates, one short rollout per policy): gates, evidence extraction, zoo
training with snapshots, paired base-versus-adapter rollouts, zoo
verification, the fold archive, both alignment conditions, paired diagnostics,
and generated-policy rollouts of rank-matched reconstructed and mapped
adapters in the real simulator.

It needs the pinned checkpoint, real demonstrations, a GPU, and the simulator,
and takes roughly 20 to 30 minutes on a 6 GB laptop GPU. Run it with

    pytest tests/integration/test_real_pipeline.py -m integration

from an activated research environment. Missing prerequisites are listed by
name. They skip the test unless ``WSL_VLA_REQUIRE_INTEGRATION=1`` is set, in
which case they fail it. **A skip is not a pass**: the session summary says
explicitly when the real path was not validated.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from wsl_vla.experiments.manifest import ExperimentPaths, manifest_from_dict
from wsl_vla.experiments.protocol import load_yaml, validate_reference_tasks
from wsl_vla.experiments.zoo_jobs import (
    build_experiment_job_plan,
    run_job_plan,
    write_job_plan,
)

pytestmark = [pytest.mark.integration, pytest.mark.research]

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRE = os.environ.get("WSL_VLA_REQUIRE_INTEGRATION", "") == "1"
# Checkpoints are reloaded in this process after the jobs finish; never preallocate.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
NOT_VALIDATED = "REAL OCTO/LIBERO PIPELINE NOT VALIDATED"


def reduced_integrity_manifest() -> dict:
    """The shipped integrity profile with the smallest budgets that still run."""

    payload = copy.deepcopy(load_yaml(REPO_ROOT / "configs/research/manifests/integrity.yaml"))
    payload["name"] = "integration_test"
    payload["description"] = "Reduced integrity profile driven by tests/integration."
    payload["zoo"].update(
        optimizer_updates=4,
        gradient_accumulation_steps=1,
        snapshot_updates=[2, 4],
        held_out_loss_every=2,
        held_out_loss_batches=1,
    )
    payload["alignment"].update(optimizer_updates=6, validation_every=3)
    payload["rollouts"].update(development=1, final=1, max_episode_steps=20)
    payload["expected"]["adapter_samples"] = 4
    fold = payload["expected"]["folds"]["integrity"]
    fold["train_samples"] = fold["validation_samples"] = 4
    return payload


def missing_prerequisites(manifest_payload: dict) -> list[str]:
    """Everything the real pipeline needs that this host lacks, by name."""

    missing: list[str] = []
    try:
        from wsl_vla.octo.compat import detect_environment_profile
    except Exception as exc:  # pragma: no cover - broken installation
        return [f"wsl_vla is not importable: {exc}"]
    profile = detect_environment_profile()
    if profile["profile"] is None:
        details = "; ".join(
            f"[{name}] " + ", ".join(problems) for name, problems in profile["mismatches"].items()
        )
        return [f"no locked research environment is active ({details})"]

    from huggingface_hub import try_to_load_from_cache

    from wsl_vla.octo.bridge import OCTO_MODEL_ID, OCTO_MODEL_REVISION

    for repository, revision, filename in (
        (OCTO_MODEL_ID, OCTO_MODEL_REVISION, "config.json"),
        ("t5-base", None, "spiece.model"),
    ):
        cached = try_to_load_from_cache(repository, filename, revision=revision)
        if not isinstance(cached, str):
            missing.append(
                f"{repository} is not in the Hugging Face cache "
                "(load it once online, e.g. scripts/eval/verify_official_octo.py)"
            )

    tasks = load_yaml(REPO_ROOT / "configs/reference_tasks.yaml")
    config = load_yaml(REPO_ROOT / "configs/research/base.yaml")
    from wsl_vla.experiments.runtime import task_data_file

    for suite, indices in manifest_payload["tasks"].items():
        for index in indices:
            identifier = f"{suite}_{index}"
            try:
                task_data_file(REPO_ROOT / config["data"]["root"], tasks, identifier)
            except Exception as exc:
                missing.append(
                    f"LIBERO demonstrations for {identifier} are unavailable ({exc}); run "
                    "scripts/setup/download_libero.py --manifest "
                    "configs/research/manifests/integrity.yaml"
                )

    # Probed in child processes: this process must not claim the GPU that the
    # pipeline's jobs are about to use.
    child_environment = {
        **os.environ,
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    backend = subprocess.run(
        [sys.executable, "-c", "import jax; print(jax.default_backend())"],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        env=child_environment,
    )
    if backend.stdout.strip().splitlines()[-1:] != ["gpu"]:
        missing.append("JAX sees no GPU (the test would take hours on CPU)")
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mujoco\n"
            "from libero.libero.envs import OffScreenRenderEnv\n"
            "context = mujoco.GLContext(64, 64)\n"
            "context.make_current()\n",
        ],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        env=child_environment,
    )
    if probe.returncode != 0:
        reason = (probe.stderr.strip().splitlines() or ["no output"])[-1]
        missing.append(
            "the LIBERO simulator or an offscreen MuJoCo renderer is unavailable in this "
            f"interpreter ({reason}); activate the research environment so that "
            "LIBERO_CONFIG_PATH and MUJOCO_GL are set"
        )
    return missing


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    payload = reduced_integrity_manifest()
    missing = missing_prerequisites(payload)
    if missing:
        message = f"{NOT_VALIDATED}: " + " | ".join(missing)
        if REQUIRE:
            pytest.fail(message)
        pytest.skip(message)

    root = tmp_path_factory.mktemp("real_pipeline")
    manifest_path = root / "integration_test.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    tasks = load_yaml(REPO_ROOT / "configs/reference_tasks.yaml")
    validate_reference_tasks(tasks)
    manifest = manifest_from_dict(payload)
    manifest.validate()
    research_root = root / "results"
    jobs = build_experiment_job_plan(
        manifest, manifest_path=manifest_path, research_root=research_root, python=sys.executable
    )
    paths = ExperimentPaths(research_root, manifest.name)
    write_job_plan(jobs, paths.jobs / "jobs.jsonl")
    # The environment run_job_manifest.py --offline gives every job; run_job_plan
    # adds deterministic GPU ops per job, as the runner does.
    environment = {
        **os.environ,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    def runner(command, **kwargs):
        # The plan's first gate is this very test suite; it is running already.
        if command[1:3] == ["-m", "pytest"]:
            return SimpleNamespace(returncode=0)
        return subprocess.run(command, **kwargs)

    def run():
        return run_job_plan(
            jobs,
            state_directory=paths.jobs,
            working_directory=REPO_ROOT,
            environment=environment,
            runner=runner,
        )

    summary = run()
    return SimpleNamespace(
        manifest=manifest, paths=paths, jobs=jobs, summary=summary, rerun=run, tasks=tasks
    )


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_job_of_the_plan_completes(pipeline):
    failed = pipeline.summary["failed"]
    assert not failed, "\n\n".join(
        f"{item['job_id']} (exit {item['returncode']}):\n" + "\n".join(item["log_tail"])
        for item in failed
    )
    assert not pipeline.summary["blocked"]
    assert not pipeline.summary["skipped"]
    assert pipeline.summary["counts"] == {"completed": len(pipeline.jobs)}


def test_gates_hold_on_the_real_checkpoint_and_simulator(pipeline):
    gates = pipeline.paths.root / "gates"
    assert read(gates / "preflight" / "preflight.json")["ok"] is True
    octo = read(gates / "octo_gates.json")
    assert octo["ok"] is True
    assert octo["zero_adapter_equivalence"]["transformer_outputs"]["max_abs_diff"] <= 1e-5
    assert octo["zero_adapter_equivalence"]["sampled_actions"]["max_abs_diff"] <= 1e-5
    assert octo["gradient_reachability"]["minimum_initial_up_gradient_norm"] > 0
    assert octo["gradient_reachability"]["maximum_initial_down_gradient_norm"] == 0.0
    assert octo["gradient_reachability"]["minimum_post_update_down_gradient_norm"] > 0
    counts = octo["token_routing_in_live_graph"]["token_counts"]
    assert min(counts["vision"], counts["language"], counts["action"]) > 0
    assert counts["frozen"] == 0
    camera = read(gates / "camera_orientation" / "libero_spatial" / "camera_orientation.json")
    assert camera["status"] == "verified" and camera["verified"] is True


def test_zoo_runs_train_snapshot_and_reload(pipeline):
    paths, manifest = pipeline.paths, pipeline.manifest
    for identifier, seed in manifest.runs():
        run = read(paths.run_directory(identifier, seed) / "run.json")
        assert run["completed"] is True and run["failures"] == []
        assert run["optimizer_updates"] == 4 and run["micro_steps"] == 4
        assert run["sample_exposure"] == 8
        assert run["provenance"]["environment"]["deterministic_gpu_ops"] is True
        gates = run["provenance"]["gates"]
        assert gates["zero_adapter_equivalence"]["sampled_actions"]["max_abs_diff"] <= 1e-5
        assert run["provenance"]["base_parameters_after_training"]["base_sha256"]
        for stage in (2, 4):
            sample = paths.run_directory(identifier, seed) / f"update_{stage}"
            metadata = read(sample / "metadata.json")
            assert metadata["validation"]["structurally_valid"] is True
            assert metadata["optimizer_updates"] == stage
            assert set(metadata["held_out_sampling"]) == {"adapter", "frozen_base"}
            assert all(value > 0 for value in metadata["effective_update_norm"].values())
    verification = read(paths.population / "verification.json")
    assert verification["ok"] is True
    assert verification["present_samples"] == 4 and not verification["missing_samples"]
    assert not verification["incomplete_or_failed_runs"]


def test_adapters_are_rolled_out_against_the_frozen_base_on_paired_episodes(pipeline):
    paths, manifest = pipeline.paths, pipeline.manifest
    for identifier, seed in manifest.runs():
        behaviour = read(paths.run_directory(identifier, seed) / "update_4" / "behaviour.json")
        base, adapter = behaviour["frozen_base"]["episodes"], behaviour["adapter"]["episodes"]
        assert len(base) == len(adapter) == 1
        assert [item["initialization_index"] for item in base] == [
            item["initialization_index"] for item in adapter
        ]
        assert [item["episode_seed"] for item in base] == [item["episode_seed"] for item in adapter]
        assert 0 < base[0]["steps"] <= 20
        assert 0.0 <= behaviour["adapter"]["controller_clipping"]["clipped_step_fraction"] <= 1.0
        assert behaviour["base_parameters_unchanged"]


def test_paired_alignment_checkpoints_reload_and_differ_only_in_the_contrastive_loss(pipeline):
    from wsl_vla.alignment.checkpoint import assert_paired_checkpoints, load_alignment_checkpoint

    paths = pipeline.paths
    archive = read(paths.archive("integrity") / "archive_manifest.json")
    assert archive["split_counts"] == {"train": 4, "validation": 0}
    assert archive["integrity_only"] is True
    aligned = load_alignment_checkpoint(paths.alignment_checkpoint("integrity", 0, "aligned"))
    control = load_alignment_checkpoint(
        paths.alignment_checkpoint("integrity", 0, "reconstruction_only")
    )
    assert_paired_checkpoints(aligned, control)
    assert aligned.metadata["initial_params_sha256"] == control.metadata["initial_params_sha256"]
    assert aligned.metadata["batch_schedule_sha256"] == control.metadata["batch_schedule_sha256"]
    assert aligned.metadata["params_sha256"] != control.metadata["params_sha256"]
    assert aligned.config["effective_contrastive_weight"] > 0
    assert control.config["effective_contrastive_weight"] == 0


def test_diagnostics_pass_integrity_and_draw_no_conclusion_from_two_tasks(pipeline):
    report = read(
        pipeline.paths.diagnostics("integrity", 0, "validation") / "alignment_pair.json"
    )
    assert report["integrity"]["status"] == "pass"
    assert report["integrity_only"] is True and report["publication_evidence"] is False
    assert report["comparisons"]
    assert {item["evidence"] for item in report["comparisons"].values()} == {"inconclusive"}


def test_generated_policies_are_rank_matched_and_evaluated_in_the_simulator(pipeline):
    from wsl_vla.evaluation.generated_policies import condition_names
    from wsl_vla.experiments.manifest import ALIGNMENT_CONDITIONS

    directory = pipeline.paths.rollouts("validation", "integrity", 0)
    run_manifest = read(directory / "manifest.json")
    assert run_manifest["finished"] is True and run_manifest["failures"] == []
    conditions = set(condition_names(ALIGNMENT_CONDITIONS))
    episodes = [
        json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()
    ]
    tasks = set(pipeline.manifest.tasks)
    assert {item["condition"] for item in episodes} == conditions
    assert {item["task_id"] for item in episodes} == tasks
    assert len(episodes) == len(conditions) * len(tasks)
    for task in tasks:
        rows = [item for item in episodes if item["task_id"] == task]
        assert len({(item["initialization_index"], item["episode_seed"]) for item in rows}) == 1
    policies = read(directory / "policies.json")
    for task in tasks:
        assert set(policies[task]) == conditions
        for name, record in policies[task].items():
            if name.split("[")[0] in {"reconstruction", "mapper", "mapper_shell"}:
                # Decoded updates are deployed at the zoo's rank; the cost is reported.
                assert record["rank_matched"] is True
                assert record["rank_projection"]["relative_error"] is not None
    summary = read(directory / "summary.json")
    assert summary["integrity"]["status"] == "pass"
    assert set(summary["conditions"]) == conditions
    assert (pipeline.paths.root / "reports" / "generated_policy_rollouts.json").is_file()


def test_rerunning_the_plan_does_nothing(pipeline):
    again = pipeline.rerun()
    assert again["counts"] == {"completed": len(pipeline.jobs)}
    assert not [event for event in again["events"] if event["status"] not in {"completed"}]
