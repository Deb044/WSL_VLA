"""The experiment manifest is the sole source of selections and expected counts."""

import copy
import json
import threading

import pytest

from tests.synthetic import synthetic_manifest, synthetic_manifest_dict
from wsl_vla.experiments.manifest import (
    ExperimentPaths,
    ManifestError,
    bind_experiment,
    load_experiment_manifest,
    manifest_from_dict,
    record_held_out_results,
)
from wsl_vla.experiments.protocol import load_yaml


REFERENCE = load_yaml("configs/reference_tasks.yaml")


def load(name):
    return load_experiment_manifest(
        f"configs/research/manifests/{name}.yaml", reference_tasks=REFERENCE
    )


def test_run_profiles_derive_the_population_of_the_repair_plan():
    pilot = load("pilot").derived_counts()
    assert pilot["independent_runs"] == 24 and pilot["adapter_samples"] == 72
    assert len(pilot["folds"]) == 4
    assert all(
        (fold["train_samples"], fold["validation_samples"], fold["held_out_samples"]) == (36, 18, 18)
        for fold in pilot["folds"].values()
    )
    full = load("full")
    counts = full.derived_counts()
    assert counts["independent_runs"] == 120 and counts["adapter_samples"] == 360
    assert all(
        (fold["train_tasks"], fold["validation_tasks"], fold["held_out_tasks"]) == (24, 6, 10)
        for fold in counts["folds"].values()
    )
    assert full.alignment.seeds == (0, 1, 2) and len(full.seeds) == 3
    # Source task indices 8 and 9 remain validation tasks in every full fold.
    for fold in full.folds:
        assert {task.rsplit("_", 1)[1] for task in fold.validation_tasks} == {"8", "9"}
    integrity = load("integrity")
    assert integrity.derived_counts()["adapter_samples"] == 6
    assert integrity.alignment.tasks_per_batch == 2 and not integrity.publication_evidence
    assert load("pilot").alignment.tasks_per_batch == 6 and load("pilot").alignment.samples_per_task == 2


def test_pilot_follows_the_ordered_run_profile():
    pilot = load("pilot")
    assert pilot.seeds == (17, 42) and len(pilot.zoo.snapshot_updates) == 3
    assert pilot.zoo.optimizer_updates == 5000 and pilot.zoo.effective_batch_size == 16
    assert pilot.zoo.sample_exposure(5000) == 80_000
    assert pilot.zoo.inspection_updates == (2000, 5000, 10000)
    assert not pilot.zoo.frozen and not pilot.publication_evidence
    assert (pilot.rollouts.development, pilot.rollouts.final) == (20, 50)
    for fold in pilot.folds:
        assert {task.rsplit("_", 1)[1] for task in fold.train_tasks} == {"0", "1"}
        assert {task.rsplit("_", 1)[1] for task in fold.validation_tasks} == {"8"}
        assert all(task.startswith(fold.held_out_suite) for task in fold.held_out_tasks)
    with pytest.raises(ManifestError, match="frozen zoo budget"):
        pilot.require_frozen_budget("held-out evaluation")


def test_declared_counts_must_equal_the_derived_counts():
    payload = synthetic_manifest_dict()
    payload["expected"]["adapter_samples"] += 1
    with pytest.raises(ManifestError, match="declared expected counts differ"):
        manifest_from_dict(payload).validate()


def test_every_snapshot_of_a_task_follows_its_task_split():
    manifest = synthetic_manifest()
    fold = manifest.folds[0]
    train = manifest.fold_samples(fold, "train")
    assert {sample.task_id for sample in train} == set(fold.train_tasks)
    assert len(train) == len(fold.train_tasks) * len(manifest.seeds) * len(manifest.zoo.snapshot_updates)
    assert {sample.run_id for sample in train if sample.task_id == fold.train_tasks[0]} == {
        f"{fold.train_tasks[0]}/seed_{seed}" for seed in manifest.seeds
    }
    sample = train[0]
    assert sample.sample_id == f"{sample.task_id}/seed_{sample.seed}/update_{sample.optimizer_updates}"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p["zoo"].__setitem__("snapshot_updates", [4, 6, 9]), "end at the final optimizer update"),
        (lambda p: p["zoo"].__setitem__("snapshot_updates", [6, 4, 8]), "unique, increasing"),
        (lambda p: p["seeds"].append(p["seeds"][0]), "seeds must be unique"),
        (lambda p: p["alignment"].__setitem__("tasks_per_batch", 1), "at least two task identities"),
        (lambda p: p["alignment"].__setitem__("conditions", ["aligned"]), "two primary alignment conditions"),
        (lambda p: p["folds"][0].__setitem__("selection", "integrity_reuse_training_tasks"), "only in the integrity profile"),
        (lambda p: p["folds"][0].__setitem__("validation_tasks", {}), "no validation tasks"),
        (lambda p: p.__setitem__("publication_evidence", True), "only the full profile"),
        (lambda p: p["rollouts"].__setitem__("zoo_reference_seed", 999), "must be a zoo seed"),
        (lambda p: p["episode_split"].__setitem__("train_fraction", 0.95), "leave test episodes"),
    ],
)
def test_invalid_manifests_are_rejected(mutation, message):
    payload = copy.deepcopy(synthetic_manifest_dict())
    mutation(payload)
    with pytest.raises(ManifestError, match=message):
        manifest_from_dict(payload).validate()


def test_unknown_reference_task_is_rejected():
    payload = synthetic_manifest_dict()
    payload["tasks"]["libero_spatial"] = [0, 1, 8, 42]
    with pytest.raises(ManifestError):
        manifest_from_dict(payload).validate(REFERENCE)


def test_zoo_identity_ignores_bookkeeping_but_tracks_training_settings():
    base = synthetic_manifest(frozen=False)
    frozen = synthetic_manifest(frozen=True)
    assert base.zoo_sha256() == frozen.zoo_sha256()  # freezing keeps trained adapters valid
    assert base.sha256() != frozen.sha256()
    changed = synthetic_manifest_dict()
    changed["zoo"]["learning_rate"] = 1e-3
    assert manifest_from_dict(changed).zoo_sha256() != base.zoo_sha256()
    resplit = synthetic_manifest_dict()
    resplit["episode_split"]["seed"] = 6
    assert manifest_from_dict(resplit).zoo_sha256() != base.zoo_sha256()


def test_held_out_results_freeze_the_experiment_configuration(tmp_path):
    manifest = synthetic_manifest(frozen=True)
    paths = bind_experiment(manifest, tmp_path)
    assert paths.root == tmp_path / "experiments" / manifest.name
    lock = json.loads(paths.lock.read_text())
    assert lock["manifest_sha256"] == manifest.sha256() and lock["history"] == []

    # Before any held-out result exists the manifest may still change (logged).
    changed_payload = synthetic_manifest_dict(frozen=True)
    changed_payload["alignment"]["optimizer_updates"] = 9
    changed = manifest_from_dict(changed_payload)
    bind_experiment(changed, tmp_path)
    assert json.loads(paths.lock.read_text())["history"] == [manifest.sha256()]

    record_held_out_results(changed, paths, artifact="rollouts/held_out/fold/seed_0")
    assert paths.held_out_marker.is_file()
    bind_experiment(changed, tmp_path)  # the same manifest keeps working
    with pytest.raises(ManifestError, match="already holds held-out results"):
        bind_experiment(manifest, tmp_path)
    with pytest.raises(ManifestError, match="different manifest"):
        record_held_out_results(manifest, paths, artifact="other")


def test_held_out_results_recorded_concurrently_are_all_kept(tmp_path):
    # Held-out jobs of different folds can finish at the same moment when
    # several runners share one experiment.
    manifest = synthetic_manifest(frozen=True)
    paths = bind_experiment(manifest, tmp_path)
    artifacts = [f"rollouts/held_out/fold_{index}/seed_0" for index in range(8)]
    threads = [
        threading.Thread(target=record_held_out_results, args=(manifest, paths),
                         kwargs={"artifact": artifact})
        for artifact in artifacts
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(json.loads(paths.held_out_marker.read_text())["artifacts"]) == artifacts


def test_held_out_results_require_a_frozen_budget(tmp_path):
    manifest = synthetic_manifest(frozen=False)
    paths = bind_experiment(manifest, tmp_path)
    with pytest.raises(ManifestError, match="frozen zoo budget"):
        record_held_out_results(manifest, paths, artifact="x")
    assert not paths.held_out_marker.exists()


def test_corrected_experiments_never_share_directories_with_legacy_artifacts(tmp_path):
    paths = ExperimentPaths(tmp_path, "pilot")
    for directory in (paths.population, paths.evidence, paths.archive("f"), paths.jobs):
        assert tmp_path / "experiments" / "pilot" in directory.parents
    assert paths.population != tmp_path / "population"
