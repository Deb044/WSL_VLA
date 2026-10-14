"""Archive -> paired alignment training -> checkpoint -> diagnostics, end to end.

Uses a small synthetic zoo written through the real writers, so every contract
between the stages is exercised without Octo or a simulator.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from tests.synthetic import (  # noqa: E402
    SMALL_ALIGNMENT_CONFIG,
    synthetic_manifest,
    write_synthetic_population,
)
from wsl_vla.adapters.rank import update_rank  # noqa: E402
from wsl_vla.alignment.archive import (  # noqa: E402
    ArchiveIntegrityError,
    build_alignment_archive,
    open_alignment_archive,
)
from wsl_vla.alignment.checkpoint import (  # noqa: E402
    assert_paired_checkpoints,
    compose_latent_sequence,
    decode_latent_sequence,
    encode_evidence_prompts,
    encode_weight_tokens,
    load_alignment_checkpoint,
    map_evidence_embeddings,
    matching_checkpoint_metadata,
    rank_matched_updates,
    save_alignment_checkpoint,
)
from wsl_vla.alignment.diagnostics import load_task_populations, run_pair_diagnostics  # noqa: E402
from wsl_vla.alignment.training import (  # noqa: E402
    AlignmentObjective,
    build_batch_schedule,
    fit_alignment_geometry,
    initialize_alignment_params,
    make_batch,
    resolve_alignment_config,
    train_alignment,
)
from wsl_vla.contracts import COMPONENT_NAMES, LegacyArtifactError  # noqa: E402
from wsl_vla.experiments.manifest import ManifestError, manifest_from_dict  # noqa: E402
from wsl_vla.experiments.provenance import sha256_array_tree  # noqa: E402

pytestmark = pytest.mark.research
EVIDENCE_LABELS = {"inconclusive", "supports_first_condition", "negative_finding"}


def resolve(manifest, archive, condition, **overrides):
    return resolve_alignment_config(
        alignment_config=SMALL_ALIGNMENT_CONFIG,
        manifest=manifest,
        archive=archive,
        condition=condition,
        seed=0,
        overrides=overrides,
    )


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiment")
    manifest = synthetic_manifest()
    paths, spec = write_synthetic_population(root, manifest)
    fold = manifest.folds[0]
    archive_manifest = build_alignment_archive(
        manifest, paths, fold, paths.archive(fold.name), samples_per_shard=16
    )
    archive = open_alignment_archive(paths.archive(fold.name))
    checkpoints = {}
    trainings = {}
    for condition in ("aligned", "reconstruction_only"):
        config = resolve(manifest, archive, condition, optimizer_updates=12, validation_every=4)
        training = train_alignment(config, archive, log=lambda *_: None)
        geometry = fit_alignment_geometry(
            training["objective"], training["params"], archive,
            batch_size=config["micro_batch_size"],
        )
        directory = paths.alignment_checkpoint(fold.name, 0, condition)
        save_alignment_checkpoint(
            directory,
            config=config,
            params=training["params"],
            archive=archive,
            entry_scales=training["objective"].entry_scales,
            geometry=geometry,
            training=training,
            provenance={"test": True},
        )
        checkpoints[condition] = load_alignment_checkpoint(directory)
        trainings[condition] = training
    return {
        "root": root,
        "manifest": manifest,
        "paths": paths,
        "spec": spec,
        "fold": fold,
        "archive": archive,
        "archive_manifest": archive_manifest,
        "checkpoints": checkpoints,
        "trainings": trainings,
    }


# ------------------------------------------------------------------- archive
def test_archive_counts_come_from_the_manifest_and_evidence_is_stored_once_per_task(pipeline):
    manifest, archive, fold = pipeline["manifest"], pipeline["archive"], pipeline["fold"]
    expected = manifest.derived_counts()["folds"][fold.name]
    assert len(archive.indices("train")) == expected["train_samples"] == 36
    assert len(archive.indices("validation")) == expected["validation_samples"] == 18
    # 9 tasks x (2 seeds x 3 snapshots): one evidence row per task, not per sample.
    assert archive.evidence["vision"].shape[0] == len(archive.task_ids) == 9
    assert len(archive.sample_ids) == 54
    assert pipeline["archive_manifest"]["excluded_held_out_tasks"] == list(fold.held_out_tasks)
    assert all("libero_spatial" not in task for task in archive.task_ids)
    # Every seed and snapshot of a task sits in that task's split.
    for task, split in zip(archive.task_ids, archive.task_split):
        rows = [i for i, name in enumerate(archive.sample_ids) if name.startswith(task + "/")]
        assert len(rows) == 6 and {archive.sample_split[i] for i in rows} == {split}


def test_archive_shards_are_memory_mapped_and_match_the_packed_samples(pipeline):
    archive, paths, manifest = pipeline["archive"], pipeline["paths"], pipeline["manifest"]
    assert len(archive._shards) == 4 and all(isinstance(s, np.memmap) for s in archive._shards)
    populations, _ = load_task_populations(manifest, paths, [archive.task_ids[0]])
    first = populations[archive.task_ids[0]]
    index = archive.sample_ids.index(first.samples[0].sample_id)
    np.testing.assert_array_equal(archive.tokens([index])[0], first.tokens[0])


def test_training_only_statistics_ignore_validation_samples(pipeline):
    archive = pipeline["archive"]
    train = archive.tokens(archive.indices("train")).astype(np.float64)
    mask = archive.layout.token_mask
    expected = np.zeros(len(archive.spec.entries))
    for entry in range(len(expected)):
        rows = archive.layout.entry_ids == entry
        expected[entry] = np.sqrt(np.square(train[:, rows]).sum() / (mask[rows].sum() * len(train)))
    np.testing.assert_allclose(archive.entry_rms, expected, rtol=1e-6)
    assert set(archive.standardization.fit_task_ids) == set(archive.train_task_ids)
    assert pipeline["archive_manifest"]["entry_rms"]["fit_split"] == "train"


def test_archive_builder_rechecks_adapter_and_evidence_hashes(pipeline, tmp_path):
    manifest = pipeline["manifest"]
    paths, _ = write_synthetic_population(tmp_path / "copy", manifest)
    fold = manifest.folds[0]
    sample = manifest.fold_samples(fold, "train")[0]
    adapter = paths.sample_directory(sample) / "adapter.npz"
    adapter.write_bytes(adapter.read_bytes() + b"tampered")
    with pytest.raises(ArchiveIntegrityError, match="adapter hash mismatch"):
        build_alignment_archive(manifest, paths, fold, tmp_path / "archive_a")

    paths, _ = write_synthetic_population(tmp_path / "copy2", manifest)
    evidence = paths.evidence_file(fold.train_tasks[0])
    evidence.write_bytes(evidence.read_bytes() + b"tampered")
    with pytest.raises(ArchiveIntegrityError, match="evidence hash mismatch"):
        build_alignment_archive(manifest, paths, fold, tmp_path / "archive_b")


def test_archive_builder_rejects_evidence_from_other_episodes_or_settings(pipeline, tmp_path):
    manifest = pipeline["manifest"]
    fold = manifest.folds[0]
    paths, _ = write_synthetic_population(tmp_path / "episodes", manifest)
    sample = manifest.fold_samples(fold, "train")[0]
    metadata_path = paths.sample_directory(sample) / "metadata.json"
    for other in manifest.samples([sample.task_id]):
        path = paths.sample_directory(other) / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["train_episode_ids_sha256"] = "0" * 64
        path.write_text(json.dumps(metadata))
    assert metadata_path.is_file()
    with pytest.raises(ArchiveIntegrityError, match="not built from the zoo's training episodes"):
        build_alignment_archive(manifest, paths, fold, tmp_path / "archive_c")

    paths, _ = write_synthetic_population(tmp_path / "settings", manifest)
    sidecar = paths.evidence_file(fold.train_tasks[0]).with_suffix(".manifest.json")
    payload = json.loads(sidecar.read_text())
    payload["evidence_identity"]["evidence"]["action_chunk_length"] = 99
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(ArchiveIntegrityError, match="different settings than the manifest"):
        build_alignment_archive(manifest, paths, fold, tmp_path / "archive_d")


def test_incomplete_or_foreign_populations_are_rejected(pipeline, tmp_path):
    manifest = pipeline["manifest"]
    fold = manifest.folds[0]
    paths, _ = write_synthetic_population(tmp_path / "missing", manifest)
    shutil.rmtree(paths.sample_directory(manifest.fold_samples(fold, "validation")[0]))
    with pytest.raises(ArchiveIntegrityError, match="lacks 1 of 54 manifest samples"):
        build_alignment_archive(manifest, paths, fold, tmp_path / "archive_e")

    paths, _ = write_synthetic_population(tmp_path / "budget", manifest)
    other = manifest_from_dict(
        {**json.loads(json.dumps(manifest.to_dict())), "zoo": {**manifest.zoo.to_dict(), "learning_rate": 1e-3}}
    )
    with pytest.raises(ArchiveIntegrityError, match="different budget or episode split"):
        build_alignment_archive(other, paths, other.folds[0], tmp_path / "archive_f")


def test_archive_build_is_atomic_idempotent_and_never_overwrites(pipeline, tmp_path):
    manifest, fold = pipeline["manifest"], pipeline["fold"]
    paths, _ = write_synthetic_population(tmp_path / "population", manifest)
    destination = tmp_path / "archive"

    # A failed build leaves nothing behind that could block or mislead a retry.
    sample = manifest.fold_samples(fold, "train")[-1]
    removed = tmp_path / "removed_sample"
    shutil.move(paths.sample_directory(sample), removed)
    with pytest.raises(ArchiveIntegrityError):
        build_alignment_archive(manifest, paths, fold, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".archive*"))
    shutil.move(removed, paths.sample_directory(sample))

    first = build_alignment_archive(manifest, paths, fold, destination, samples_per_shard=16)
    assert not list(tmp_path.glob(".archive*"))
    before = {item.name: item.stat().st_mtime_ns for item in destination.iterdir()}

    # Same inputs: the finished archive is returned untouched.
    again = build_alignment_archive(manifest, paths, fold, destination, samples_per_shard=16)
    assert again == first
    assert {item.name: item.stat().st_mtime_ns for item in destination.iterdir()} == before

    # Different inputs: a conflict, never an overwrite.
    with pytest.raises(FileExistsError, match="built from different inputs"):
        build_alignment_archive(manifest, paths, fold, destination, samples_per_shard=8)
    assert {item.name: item.stat().st_mtime_ns for item in destination.iterdir()} == before

    # A stale staging directory from an interrupted build is replaced.
    other = tmp_path / "second"
    staging = tmp_path / ".second.partial"
    staging.mkdir()
    (staging / "tokens_000.npy").write_bytes(b"half written")
    build_alignment_archive(manifest, paths, fold, other, samples_per_shard=16)
    assert not staging.exists()
    assert open_alignment_archive(other).manifest_sha256 == open_alignment_archive(
        destination
    ).manifest_sha256

    # A non-empty directory without an archive manifest is not silently adopted.
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "tokens_000.npy").write_bytes(b"unknown")
    with pytest.raises(FileExistsError, match="holds no archive manifest"):
        build_alignment_archive(manifest, paths, fold, foreign)


def test_leakage_is_rejected_by_the_manifest_and_by_the_archive_reader(pipeline, tmp_path):
    manifest = pipeline["manifest"]
    payload = json.loads(json.dumps(manifest.to_dict()))
    leaked = json.loads(json.dumps(payload))
    leaked["folds"][0]["train_tasks"].append("libero_spatial_0")
    leaked["folds"][0]["held_out_tasks"].remove("libero_spatial_0")
    with pytest.raises(ManifestError, match="held-out suite tasks leaked"):
        manifest_from_dict(leaked).validate()
    two_splits = json.loads(json.dumps(payload))
    two_splits["folds"][0]["validation_tasks"].append(two_splits["folds"][0]["train_tasks"][0])
    with pytest.raises(ManifestError, match="more than one alignment split"):
        manifest_from_dict(two_splits).validate()

    # A hand-edited archive index that moves one snapshot into another split, or
    # that names a held-out task, is refused when the archive is opened.
    source = pipeline["archive"].directory
    for mutation, message in (
        (lambda m: m["samples"][0].__setitem__("split", "validation"), "not in its task's alignment split"),
        (lambda m: m["tasks"][0].__setitem__("task_id", "libero_spatial_0"), "held-out suite task"),
    ):
        copy = tmp_path / f"archive_{len(list(tmp_path.iterdir()))}"
        shutil.copytree(source, copy)
        index = json.loads((copy / "archive_manifest.json").read_text())
        mutation(index)
        (copy / "archive_manifest.json").write_text(json.dumps(index))
        with pytest.raises(ArchiveIntegrityError, match=message):
            open_alignment_archive(copy)
    copy = tmp_path / "archive_shard"
    shutil.copytree(source, copy)
    with (copy / "tokens_000.npy").open("r+b") as stream:
        stream.seek(-8, 2)
        stream.write(b"\x01" * 8)
    with pytest.raises(ArchiveIntegrityError, match="hash mismatch"):
        open_alignment_archive(copy)


# ---------------------------------------------------------------- resolution
def test_effective_configuration_is_fully_resolved_with_explicit_precedence(pipeline):
    manifest, archive = pipeline["manifest"], pipeline["archive"]
    config = resolve(manifest, archive, "aligned", latent_dim=4, optimizer_updates=7)
    assert config["latent_dim"] == 4 and config["sources"]["latent_dim"] == "command_line"
    assert config["optimizer_updates"] == 7 and config["sources"]["optimizer_updates"] == "command_line"
    assert config["transformer_width"] == 16 and config["sources"]["transformer_width"] == "research_config"
    assert config["samples_per_task"] == 2 and config["sources"]["samples_per_task"] == "experiment_manifest"
    assert config["train_task_count"] == 6 and config["embedding_sizes"][0] == 3
    assert "gradient accumulation" in config["negative_set"]
    # Nothing is defaulted in code: a missing key is an error, not a silent value.
    incomplete = {k: v for k, v in SMALL_ALIGNMENT_CONFIG.items() if k != "encoder_layers"}
    with pytest.raises(KeyError, match="encoder_layers"):
        resolve_alignment_config(
            alignment_config=incomplete, manifest=manifest, archive=archive,
            condition="aligned", seed=0, overrides={},
        )
    with pytest.raises(ValueError, match="unknown alignment overrides"):
        resolve(manifest, archive, "aligned", hidden_dim=3)
    with pytest.raises(ValueError, match="not declared"):
        resolve_alignment_config(
            alignment_config=SMALL_ALIGNMENT_CONFIG, manifest=manifest, archive=archive,
            condition="aligned", seed=5, overrides={},
        )


def test_six_tasks_per_batch_where_available_and_fewer_otherwise(pipeline):
    manifest, archive = pipeline["manifest"], pipeline["archive"]
    config = resolve(manifest, archive, "aligned", tasks_per_batch=8)
    assert config["requested_tasks_per_batch"] == 8 and config["tasks_per_batch"] == 6
    schedule = build_batch_schedule(
        archive.indices("train"), archive.sample_task_rows,
        tasks_per_batch=6, samples_per_task=2, micro_batches=5, seed=3,
    )
    for batch in schedule:
        rows = archive.sample_task_rows[batch]
        assert len(set(rows)) == 6 and all(np.sum(rows == row) == 2 for row in set(rows))
        assert len(set(batch)) == 12
        assert all(archive.sample_split[i] == "train" for i in batch)
    with pytest.raises(ValueError, match="exceeds"):
        resolve(manifest, archive, "aligned", samples_per_task=7)


# ------------------------------------------------------------------- pairing
def test_paired_conditions_share_initialization_samples_and_exposure(pipeline):
    aligned = pipeline["checkpoints"]["aligned"]
    control = pipeline["checkpoints"]["reconstruction_only"]
    assert_paired_checkpoints(aligned, control)
    for field in ("initial_params_sha256", "batch_schedule_sha256", "optimizer_exposure"):
        assert aligned.metadata[field] == control.metadata[field]
    assert aligned.config["effective_contrastive_weight"] == 1.0
    assert control.config["effective_contrastive_weight"] == 0.0
    assert aligned.config["auxiliary_classification_weight"] == control.config[
        "auxiliary_classification_weight"
    ] == 1.0
    history = pipeline["trainings"]["reconstruction_only"]["history"]
    assert "classification" in history[-1]["train"] and "contrastive" not in history[-1]["train"]
    assert "contrastive" in pipeline["trainings"]["aligned"]["history"][-1]["train"]
    # The trained parameters differ, although everything else was identical.
    assert aligned.metadata["params_sha256"] != control.metadata["params_sha256"]
    with pytest.raises(ValueError, match="first checkpoint must be the aligned"):
        assert_paired_checkpoints(control, aligned)


def test_unpaired_checkpoints_are_an_integrity_failure(pipeline):
    aligned = pipeline["checkpoints"]["aligned"]
    control = pipeline["checkpoints"]["reconstruction_only"]
    saved = control.metadata["batch_schedule_sha256"]
    control.metadata["batch_schedule_sha256"] = "0" * 64
    try:
        with pytest.raises(ValueError, match="differ on batch_schedule_sha256"):
            assert_paired_checkpoints(aligned, control)
    finally:
        control.metadata["batch_schedule_sha256"] = saved
    saved = control.config["latent_dim"]
    control.config["latent_dim"] = saved + 1
    try:
        with pytest.raises(ValueError, match="beyond the contrastive term"):
            assert_paired_checkpoints(aligned, control)
    finally:
        control.config["latent_dim"] = saved


def test_control_keeps_exactly_the_baseline_supervision(pipeline):
    """Disabling contrastive terms must not change reconstruction/classification."""

    manifest, archive = pipeline["manifest"], pipeline["archive"]
    aligned = AlignmentObjective(resolve(manifest, archive, "aligned"), archive)
    control = AlignmentObjective(resolve(manifest, archive, "reconstruction_only"), archive)
    params = initialize_alignment_params(aligned.model, aligned.config)
    control_params = initialize_alignment_params(control.model, control.config)
    assert sha256_array_tree(jax.device_get(params)) == sha256_array_tree(jax.device_get(control_params))
    schedule = build_batch_schedule(
        archive.indices("train"), archive.sample_task_rows,
        tasks_per_batch=6, samples_per_task=2, micro_batches=1, seed=0,
    )
    batch = make_batch(archive, schedule[0])
    (with_total, with_metrics), with_grad = jax.value_and_grad(
        lambda p: aligned.loss(p, batch, contrastive_weight=1.0), has_aux=True
    )(params)
    (without_total, without_metrics), without_grad = jax.value_and_grad(
        lambda p: control.loss(p, batch, contrastive_weight=0.0), has_aux=True
    )(params)
    for name in ("reconstruction", "classification"):
        assert float(with_metrics[name]) == pytest.approx(float(without_metrics[name]), rel=1e-6)
    assert "contrastive" not in without_metrics
    assert float(with_total) == pytest.approx(
        float(without_total) + float(with_metrics["contrastive"]), rel=1e-5
    )
    contrastive_grad = jax.grad(
        lambda p: aligned.loss(p, batch, contrastive_weight=1.0)[1]["contrastive"]
    )(params)
    difference = jax.tree_util.tree_map(lambda a, b, c: a - b - c, with_grad, without_grad, contrastive_grad)
    worst = max(float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree_util.tree_leaves(difference))
    assert worst < 1e-4
    # The control's temperatures receive no gradient at all.
    assert float(jnp.max(jnp.abs(without_grad["log_temperature"]))) == 0.0
    assert float(jnp.max(jnp.abs(with_grad["log_temperature"]))) > 0.0
    # Evidence encoders are trained in both conditions (classification).
    for tree in (with_grad, without_grad):
        assert float(jnp.max(jnp.abs(tree["vision_evidence"]["element_in"]["kernel"]))) > 0.0


def test_auxiliary_classifiers_cover_training_task_identities_only(pipeline):
    checkpoint = pipeline["checkpoints"]["aligned"]
    archive = pipeline["archive"]
    for name in COMPONENT_NAMES:
        assert checkpoint.params[f"{name}_task_classifier"]["kernel"].shape == (8, 6)
    assert sorted(archive.task_train_class[archive.task_rows("train")]) == list(range(6))
    assert set(archive.task_train_class[archive.task_rows("validation")]) == {-1}
    assert checkpoint.params["log_temperature"].shape == (3,)


# ----------------------------------------------------------------- inference
def test_checkpoint_reload_reproduces_parameters_and_outputs(pipeline):
    checkpoint = pipeline["checkpoints"]["aligned"]
    training = pipeline["trainings"]["aligned"]
    assert sha256_array_tree(checkpoint.params) == sha256_array_tree(training["params"])
    assert checkpoint.config["sha256"] == training["objective"].config["sha256"]
    archive = pipeline["archive"]
    tokens = archive.tokens(archive.indices("validation")[:2])
    reloaded = np.asarray(encode_weight_tokens(checkpoint, tokens=tokens))
    objective = training["objective"]
    direct = np.asarray(objective.binding.encode(objective.model, training["params"], jnp.asarray(tokens)))
    np.testing.assert_allclose(reloaded, direct, rtol=1e-5, atol=1e-6)
    again = load_alignment_checkpoint(checkpoint.metadata["directory"])
    np.testing.assert_array_equal(
        np.asarray(encode_weight_tokens(again, tokens=tokens)), reloaded
    )


def test_checkpoint_directory_is_complete_or_absent_and_reused_only_for_the_same_request(
    pipeline,
):
    manifest, archive, paths, fold = (
        pipeline["manifest"], pipeline["archive"], pipeline["paths"], pipeline["fold"]
    )
    directory = paths.alignment_checkpoint(fold.name, 0, "aligned")
    config = pipeline["checkpoints"]["aligned"].config

    # Saving stages its files; nothing partial is left beside the checkpoint.
    assert not list(directory.parent.glob(".*partial"))
    assert sorted(item.name for item in directory.iterdir()) == sorted(
        [
            "alignment_params.msgpack", "entry_scales.npy", "layout.npz", "mappers.npz",
            "metadata.json", "shells.npz", "standardization.npz", "task_bank.npz",
        ]
    )
    # The recorded archive location does not carry a host path.
    assert not json.loads((directory / "metadata.json").read_text())["archive"].startswith("/")

    same = resolve(manifest, archive, "aligned", optimizer_updates=12, validation_every=4)
    assert matching_checkpoint_metadata(directory, config=same, archive=archive)["config"] == (
        json.loads(json.dumps(dict(config), sort_keys=True))
    )
    assert matching_checkpoint_metadata(
        paths.alignment_checkpoint(fold.name, 7, "aligned"), config=same, archive=archive
    ) is None
    changed = resolve(manifest, archive, "aligned", optimizer_updates=13, validation_every=4)
    with pytest.raises(FileExistsError, match="different configuration or archive"):
        matching_checkpoint_metadata(directory, config=changed, archive=archive)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_alignment_checkpoint(
            directory,
            config=config,
            params=pipeline["trainings"]["aligned"]["params"],
            archive=archive,
            entry_scales=pipeline["trainings"]["aligned"]["objective"].entry_scales,
            geometry={},
            training={},
            provenance={},
        )


def test_checkpoint_loader_rejects_tampering_and_version_one(pipeline, tmp_path):
    source = pipeline["checkpoints"]["aligned"].metadata["directory"]
    copy = tmp_path / "tampered"
    shutil.copytree(source, copy)
    scales = np.load(copy / "entry_scales.npy")
    np.save(copy / "entry_scales.npy", scales * 2)
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_alignment_checkpoint(copy)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "metadata.json").write_text(json.dumps({"schema_version": 1, "model": {}}))
    with pytest.raises(LegacyArtifactError, match="version-one alignment checkpoint"):
        load_alignment_checkpoint(legacy)


def test_rms_scales_are_inverted_before_updates_reach_a_policy(pipeline):
    checkpoint = pipeline["checkpoints"]["reconstruction_only"]
    archive = pipeline["archive"]
    tokens = archive.tokens(archive.indices("train")[:1])
    latents = np.asarray(encode_weight_tokens(checkpoint, tokens=tokens))[0]
    decoded = np.asarray(decode_latent_sequence(checkpoint, latents))
    assert decoded.shape == tokens[0].shape
    assert not decoded[~checkpoint.token_mask].any()
    binding = pipeline["trainings"]["reconstruction_only"]["objective"].binding
    normalized = np.asarray(
        binding.decode_normalized(
            checkpoint.model, checkpoint.params, jnp.asarray(latents)[None]
        )
    )[0]
    scale = checkpoint.entry_scales[checkpoint.layout.entry_ids][:, None]
    # Two separately compiled float32 programs: agreement to rounding, not bitwise.
    np.testing.assert_allclose(
        decoded, np.where(checkpoint.token_mask, normalized * scale, 0.0), rtol=1e-4, atol=1e-5
    )
    # Scaling every adapter by a constant scales the RMS and leaves normalized targets unchanged.
    np.testing.assert_allclose(
        np.asarray(binding.normalize(jnp.asarray(tokens)))[0] * scale, tokens[0], rtol=1e-5, atol=1e-7
    )


def test_padding_inside_sequence_windows_cannot_change_valid_latents(pipeline):
    checkpoint = pipeline["checkpoints"]["aligned"]
    layout = checkpoint.layout
    assert (~layout.window_valid).any(), "the synthetic layout must contain padded positions"
    binding = pipeline["trainings"]["aligned"]["objective"].binding
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(layout.window_count, layout.sequence_window, layout.token_width))
    garbage = windows.copy()
    garbage[~layout.window_valid] = 1e3 * rng.normal(size=garbage[~layout.window_valid].shape)

    def encode(values):
        return np.asarray(
            checkpoint.model.apply(
                {"params": checkpoint.params},
                jnp.asarray(values, dtype=jnp.float32),
                binding.window_valid,
                binding.identities,
                method=checkpoint.model.encode,
            )
        )

    clean, dirty = encode(windows), encode(garbage)
    np.testing.assert_allclose(dirty[layout.window_valid], clean[layout.window_valid], rtol=1e-4, atol=1e-5)
    assert not dirty[~layout.window_valid].any()


def test_mapper_targets_are_task_balanced_and_selected_on_validation_tasks(pipeline):
    checkpoint = pipeline["checkpoints"]["aligned"]
    selection = checkpoint.metadata["mapper_selection"]
    for name in COMPONENT_NAMES:
        mapper = checkpoint.mappers[name]
        # One centred target per training task (6), not one per snapshot (36).
        assert mapper.y_centered.shape[0] == 6 and mapper.projection.shape == (8, 6)
        assert selection[name]["selected_on"] == "validation tasks"
        assert selection[name]["selected_ridge_alpha"] in (0.01, 1.0)
        assert set(selection[name]["validation_error_by_alpha"]) == {"0.01", "1.0"}
    assert checkpoint.metadata["shell_fit"] == {"split": "train", "samples": 36}
    assert checkpoint.metadata["entry_scale_fit"]["split"] == "train"


def test_mapped_latents_decode_to_rank_matched_reloadable_updates(pipeline):
    checkpoint = pipeline["checkpoints"]["aligned"]
    manifest, paths, fold = pipeline["manifest"], pipeline["paths"], pipeline["fold"]
    populations, spec = load_task_populations(manifest, paths, fold.validation_tasks[:1])
    task = next(iter(populations.values()))
    embeddings = encode_evidence_prompts(checkpoint, task.evidence)
    plain = map_evidence_embeddings(checkpoint, embeddings, shell_projection=False)
    shell = map_evidence_embeddings(checkpoint, embeddings, shell_projection=True)
    for name in COMPONENT_NAMES:
        distance = np.linalg.norm(np.asarray(shell[name]) - np.asarray(checkpoint.shells[name].center), axis=-1)
        np.testing.assert_allclose(distance, np.asarray(checkpoint.shells[name].radius), rtol=1e-4)
        assert not np.allclose(np.asarray(plain[name]), np.asarray(shell[name]))
    tokens = np.asarray(decode_latent_sequence(checkpoint, compose_latent_sequence(checkpoint, plain)))
    projection = rank_matched_updates(checkpoint, tokens)
    for entry in spec.entries:
        assert update_rank(projection.updates[entry.parameter_path]) <= entry.effective_rank_bound
        down, up = projection.factors[entry.parameter_path]
        assert down.shape == (entry.input_dim, entry.rank)
    assert projection.report()["relative_error"] is not None


# --------------------------------------------------------------- diagnostics
def test_pair_diagnostics_separate_integrity_from_scientific_outcome(pipeline):
    manifest, paths, fold = pipeline["manifest"], pipeline["paths"], pipeline["fold"]
    report = run_pair_diagnostics(
        manifest=manifest,
        paths=paths,
        aligned=pipeline["checkpoints"]["aligned"],
        control=pipeline["checkpoints"]["reconstruction_only"],
        evaluated_tasks=fold.validation_tasks,
        scope="validation",
        minimum_tasks=6,
        resamples=200,
        confidence=0.95,
    )
    assert report["integrity"]["status"] == "pass"
    assert not report["publication_evidence"]
    json.dumps(report)  # the whole report is serializable
    # Three validation tasks are below the declared minimum: every comparison is
    # inconclusive, and that is a valid outcome, not a failure.
    assert {item["evidence"] for item in report["comparisons"].values()} == {"inconclusive"}
    relaxed = run_pair_diagnostics(
        manifest=manifest,
        paths=paths,
        aligned=pipeline["checkpoints"]["aligned"],
        control=pipeline["checkpoints"]["reconstruction_only"],
        evaluated_tasks=fold.validation_tasks,
        scope="validation",
        minimum_tasks=2,
        resamples=200,
        confidence=0.95,
    )
    assert {item["evidence"] for item in relaxed["comparisons"].values()} <= EVIDENCE_LABELS
    for condition in ("aligned", "reconstruction_only"):
        section = report["conditions"][condition]
        assert section["reconstruction_error"]["macro"]["tasks"] == 3
        for variant in ("mapped", "mapped_shell"):
            mapping = section["mapping"][variant]
            retrieval = mapping["evidence_to_mapped_latent_retrieval"]
            assert retrieval["chance_top1"] == pytest.approx(1 / 3) and retrieval["gallery_size"] == 3
            assert mapping["rank_projection_error"]["task_balanced_mean"] is not None
            assert mapping["latent_error_relative_to_training_mean"]["tasks"] == 3
        assert "Descriptive" in section["direct_weight_evidence_cosine_retrieval"]["note"]
    assert set(report["baselines"]["nearest_neighbours"]) == set(fold.validation_tasks)
    assert all(
        item["training_task"] in fold.train_tasks
        for item in report["baselines"]["nearest_neighbours"].values()
    )


def test_held_out_diagnostics_refuse_training_tasks(pipeline):
    manifest, paths, fold = pipeline["manifest"], pipeline["paths"], pipeline["fold"]
    with pytest.raises(ValueError, match="overlap the alignment training tasks"):
        run_pair_diagnostics(
            manifest=manifest,
            paths=paths,
            aligned=pipeline["checkpoints"]["aligned"],
            control=pipeline["checkpoints"]["reconstruction_only"],
            evaluated_tasks=fold.train_tasks[:2],
            scope="held_out",
            minimum_tasks=2,
            resamples=50,
            confidence=0.95,
        )
    report = run_pair_diagnostics(
        manifest=manifest,
        paths=paths,
        aligned=pipeline["checkpoints"]["aligned"],
        control=pipeline["checkpoints"]["reconstruction_only"],
        evaluated_tasks=fold.held_out_tasks,
        scope="held_out",
        minimum_tasks=2,
        resamples=50,
        confidence=0.95,
    )
    assert report["scope"] == "held_out" and report["integrity"]["status"] == "pass"
    assert all("libero_spatial" in task for task in report["evaluated_tasks"])
