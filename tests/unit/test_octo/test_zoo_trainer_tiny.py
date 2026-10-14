"""Zoo training, snapshots, and exact resume on the toy official Octo model."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

pytest.importorskip("octo")

import jax  # noqa: E402

from tests import octo_tiny as tiny  # noqa: E402
from tests.fakes import episode  # noqa: E402
from wsl_vla.contracts import LegacyArtifactError  # noqa: E402
from wsl_vla.data.streams import task_stream_from_bundle  # noqa: E402
from wsl_vla.experiments.provenance import sha256_array_tree  # noqa: E402
from wsl_vla.octo.bridge import (  # noqa: E402
    assert_base_parameters_unchanged,
    unnormalization_statistics,
)
from wsl_vla.octo.training import initial_adapter_state, materialize_policy_params  # noqa: E402
from wsl_vla.octo.zoo import (  # noqa: E402
    COMPONENT_KEYS,
    ResumeMismatchError,
    ZooTrainer,
    load_zoo_adapter,
)

pytestmark = pytest.mark.research

STAGES = (2, 4)


def streams(bundle):
    train = tiny.tiny_stream(bundle, count=4, seed=0)
    held_out = task_stream_from_bundle(
        bundle,
        task_id=train.task_id,
        instruction=train.instruction,
        episodes=tuple(episode(10 + index, length=7, seed=1) for index in range(2)),
    )
    return train, held_out


def make_trainer(bundle, *, budget=None, held_out_on_train=False, **identity) -> ZooTrainer:
    train, held_out = streams(bundle)
    if held_out_on_train:
        held_out = train
    budget = tiny.tiny_budget() if budget is None else budget
    return ZooTrainer(
        bundle=bundle,
        stream=train,
        held_out_stream=held_out,
        budget=budget,
        identity=tiny.tiny_identity(bundle, train, held_out, budget, **identity),
        token_width=tiny.TOKEN_WIDTH,
        sequence_window=tiny.SEQUENCE_WINDOW,
    )


def state_hash(trainer: ZooTrainer) -> str:
    return sha256_array_tree(jax.device_get(trainer.adapter_state))


def run_with_stages(trainer: ZooTrainer, run_directory, resume_directory, *, target=None):
    """Mirror scripts/train/train_research_zoo.py: resume state, then the stage snapshot."""

    def on_update(current: ZooTrainer) -> None:
        step = current.update_step
        current.save_resume(resume_directory)
        if step == STAGES[0] and not (resume_directory.parent / "resume_at_first_stage").exists():
            shutil.copytree(resume_directory, resume_directory.parent / "resume_at_first_stage")
        if step in STAGES:
            current.ensure_snapshot(
                run_directory / f"update_{step}", snapshot_index=STAGES.index(step)
            )

    trainer.train_until(max(STAGES) if target is None else target, on_update=on_update, log_every=1)


@pytest.fixture(scope="module")
def completed(tiny_bundle, tmp_path_factory):
    """One uninterrupted run to the final stage, plus the resume state left at stage one."""

    root = tmp_path_factory.mktemp("zoo")
    trainer = make_trainer(tiny_bundle)
    run_with_stages(trainer, root / "run", root / "resume")
    return {
        "trainer": trainer,
        "root": root,
        "run": root / "run",
        "resume": root / "resume",
        "resume_at_first_stage": root / "resume_at_first_stage",
    }


# ------------------------------------------------------------------ accounting
def test_updates_micro_steps_and_exposure_are_counted_separately(completed):
    trainer = completed["trainer"]
    budget = trainer.budget

    assert budget.effective_batch_size == 4
    assert trainer.update_step == 4
    assert trainer.micro_step == 4 * budget.gradient_accumulation_steps == 8
    assert trainer.sample_exposure == 8 * budget.micro_batch_size == 16
    assert [item["optimizer_update"] for item in trainer.history] == [1, 2, 3, 4]
    for item in trainer.history:
        assert item["micro_step"] == item["optimizer_update"] * 2
        assert item["sample_exposure"] == item["optimizer_update"] * 4
        assert np.isfinite(item["train_loss"])


def test_history_records_per_component_gradient_and_update_norms(completed):
    for item in completed["trainer"].history:
        for field in ("gradient_norm", "update_norm"):
            assert set(item[field]) == set(COMPONENT_KEYS)
            assert all(np.isfinite(value) and value > 0 for value in item[field].values()), item


def test_held_out_loss_is_recorded_at_the_cadence_and_at_every_stage(completed):
    trainer = completed["trainer"]
    evaluated = [item["optimizer_update"] for item in trainer.history if "held_out_loss" in item]

    assert evaluated == [2, 4]
    for item in trainer.history:
        if "held_out_loss" in item:
            assert item["held_out_loss_frozen_base"] == trainer.base_held_out_loss
            assert np.isfinite(item["held_out_loss"])


def test_held_out_loss_uses_fixed_batches_and_fixed_noise(completed, tiny_bundle):
    trainer = completed["trainer"]

    assert trainer.held_out_loss() == trainer.held_out_loss()
    assert trainer.held_out_loss(initial_adapter_state(tiny_bundle)) == trainer.base_held_out_loss
    # Training changed the function being evaluated.
    assert trainer.held_out_loss() != trainer.base_held_out_loss


def test_training_never_changes_the_frozen_base(completed, tiny_bundle):
    report = assert_base_parameters_unchanged(tiny_bundle)
    assert report["research_tree_base_leaves"] == tiny_bundle.base_sha256
    summary = completed["trainer"].run_summary(completed=True)
    assert summary["completed"] and summary["failures"] == []
    assert summary["optimizer_updates"] == 4 and summary["micro_steps"] == 8
    resources = summary["resources"]
    assert resources["micro_steps_per_second"] > 0
    # Compilation is reported apart from steady-state throughput, which is what
    # a longer run should be sized with.
    assert 0 < resources["compile_seconds"] <= resources["train_seconds"]
    assert resources["steady_micro_steps_per_second"] > resources["micro_steps_per_second"]
    assert resources["held_out_evaluation_seconds"] > 0
    assert resources["snapshot_and_resume_seconds"] > 0
    assert [item["optimizer_updates"] for item in summary["snapshots"]] == list(STAGES)


def test_adapter_training_lowers_the_loss_it_optimizes(tiny_bundle):
    """With the evaluation windows drawn from the training episodes, loss must fall."""

    budget = tiny.tiny_budget(
        optimizer_updates=30,
        snapshot_updates=(30,),
        gradient_accumulation_steps=1,
        learning_rate=3e-2,
        held_out_loss_every=10,
    )
    trainer = make_trainer(tiny_bundle, budget=budget, held_out_on_train=True)
    trainer.train_until(30, log_every=10)

    assert trainer.held_out_loss() < 0.97 * trainer.base_held_out_loss
    assert_base_parameters_unchanged(tiny_bundle)


# ------------------------------------------------------------------- snapshots
def test_snapshot_holds_only_the_compact_adapter(completed):
    for step in STAGES:
        directory = completed["run"] / f"update_{step}"
        assert sorted(item.name for item in directory.iterdir()) == ["adapter.npz", "metadata.json"]
        metadata = json.loads((directory / "metadata.json").read_text())
        assert metadata["optimizer_updates"] == step
        assert metadata["micro_steps"] == 2 * step
        assert metadata["sample_exposure"] == 4 * step
        assert metadata["effective_batch_size"] == 4
        assert metadata["snapshot_index"] == STAGES.index(step)
        assert metadata["synthetic"] is False
        assert metadata["validation"] == {
            "structurally_valid": True,
            "base_parameters_unchanged": completed["trainer"].identity.base_sha256,
            "behaviourally_evaluated": False,
        }
        assert set(metadata["effective_update_norm"]) == set(COMPONENT_KEYS)
        assert all(value > 0 for value in metadata["effective_update_norm"].values())
        assert metadata["held_out_loss"] is not None
        assert set(metadata["held_out_sampling"]) == {"adapter", "frozen_base"}


def test_open_loop_sampling_check_reports_error_and_controller_bound_violations(
    completed, tiny_bundle
):
    trainer = completed["trainer"]
    adapter = trainer.held_out_sampling()
    base = trainer.base_held_out_sampling()

    for report in (adapter, base):
        assert report["windows"] == 4
        assert report["mean_absolute_error"] > 0
        assert report["executed_command_mean_absolute_error"] > 0
        assert 0.0 <= report["saturated_fraction"] <= report["out_of_bounds_fraction"] <= 1.0
        assert report["head_clamp"] == 5.0
    # Fixed noise: repeated checks agree, and the frozen base is its own reference.
    assert trainer.held_out_sampling() == adapter
    assert trainer.held_out_sampling(initial_adapter_state(tiny_bundle)) == base
    assert adapter != base
    recorded = json.loads((completed["run"] / "update_4" / "metadata.json").read_text())
    assert recorded["held_out_sampling"] == {"adapter": adapter, "frozen_base": base}
    summary = trainer.run_summary(completed=True)
    assert summary["held_out_sampling_frozen_base"] == base
    assert summary["snapshots"][-1]["held_out_sampling"] == adapter


def test_resume_state_is_kept_apart_from_adapter_snapshots(completed):
    assert sorted(item.name for item in completed["resume"].iterdir()) == [
        "resume.json",
        "state.msgpack",
    ]
    assert not list(completed["run"].rglob("*.msgpack"))
    resume = json.loads((completed["resume"] / "resume.json").read_text())
    assert resume["update_step"] == 4 and resume["micro_step"] == 8
    assert resume["random_generator"]["next_micro_step"] == 8
    assert set(resume["dataset_cursor"]) >= {"epoch", "position"}
    assert resume["run_identity_sha256"] == completed["trainer"].identity.sha256()


def test_saved_snapshot_reloads_to_the_trained_policy(completed, tiny_bundle):
    trainer = completed["trainer"]
    state, packed, metadata = load_zoo_adapter(
        completed["run"] / "update_4", tiny_bundle, token_width=tiny.TOKEN_WIDTH
    )
    live = jax.device_get(trainer.adapter_state)

    assert metadata["adapter_state_sha256"] == state_hash(trainer)
    assert packed.spec == trainer.spec
    for path, value in live["transformer"].items():
        np.testing.assert_array_equal(np.asarray(state["transformer"][path]), np.asarray(value))
    for path, pair in live["diffusion"].items():
        for factor in ("down", "up"):
            np.testing.assert_array_equal(
                np.asarray(state["diffusion"][path][factor]), np.asarray(pair[factor])
            )

    def actions(adapter_state):
        model = tiny_bundle.research_model.replace(
            params=materialize_policy_params(tiny_bundle, adapter_state)
        )
        example = tiny_bundle.pretrained_model.example_batch
        return np.asarray(
            model.sample_actions(
                example["observation"],
                example["task"],
                unnormalization_statistics=unnormalization_statistics(
                    tiny_bundle.action_convention
                ),
                rng=jax.random.PRNGKey(5),
            )
        )

    np.testing.assert_array_equal(actions(state), actions(live))
    assert np.max(np.abs(actions(state) - actions(initial_adapter_state(tiny_bundle)))) > 1e-6


def test_snapshots_are_never_overwritten(completed):
    with pytest.raises(FileExistsError, match="refusing to overwrite an existing zoo sample"):
        completed["trainer"].save_snapshot(completed["run"] / "update_4", snapshot_index=1)


def test_reload_rejects_tampered_foreign_and_legacy_samples(completed, tiny_bundle, tmp_path):
    source = completed["run"] / "update_4"

    def copy(name: str):
        target = tmp_path / name
        shutil.copytree(source, target)
        return target, json.loads((target / "metadata.json").read_text())

    tampered, _ = copy("tampered")
    with open(tampered / "adapter.npz", "ab") as stream:
        stream.write(b"\x01")
    with pytest.raises(ValueError, match="adapter hash mismatch"):
        load_zoo_adapter(tampered, tiny_bundle, token_width=tiny.TOKEN_WIDTH)

    foreign, metadata = copy("foreign")
    metadata["preprocessing_sha256"] = "0" * 64
    (foreign / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="differs from the live model on preprocessing_sha256"):
        load_zoo_adapter(foreign, tiny_bundle, token_width=tiny.TOKEN_WIDTH)

    legacy, metadata = copy("legacy")
    metadata["schema_version"] = 1
    (legacy / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(LegacyArtifactError):
        load_zoo_adapter(legacy, tiny_bundle, token_width=tiny.TOKEN_WIDTH)

    synthetic, metadata = copy("synthetic")
    metadata["synthetic"] = True
    (synthetic / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="synthetic zoo sample is forbidden"):
        load_zoo_adapter(synthetic, tiny_bundle, token_width=tiny.TOKEN_WIDTH)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    shutil.copy(source / "metadata.json", incomplete / "metadata.json")
    with pytest.raises(FileNotFoundError, match="incomplete zoo sample"):
        load_zoo_adapter(incomplete, tiny_bundle, token_width=tiny.TOKEN_WIDTH)

    # Another token width is another packing of the same factors, hence another spec.
    with pytest.raises(ValueError, match="trained for a different base, architecture"):
        load_zoo_adapter(source, tiny_bundle, token_width=4)


# ---------------------------------------------------------------------- resume
def test_resumed_run_reproduces_the_uninterrupted_run(completed, tiny_bundle, tmp_path):
    """Optimizer moments, noise counters, and the data cursor all resume exactly."""

    reference = completed["trainer"]
    resume_directory = tmp_path / "resume"
    shutil.copytree(completed["resume_at_first_stage"], resume_directory)
    resumed = make_trainer(tiny_bundle)

    assert resumed.restore(resume_directory) is True
    assert resumed.update_step == 2 and resumed.micro_step == 4
    assert resumed.cursor.state() == json.loads(
        (resume_directory / "resume.json").read_text()
    )["dataset_cursor"]
    assert state_hash(resumed) != state_hash(reference)

    run_with_stages(resumed, tmp_path / "run", resume_directory)

    assert resumed.update_step == 4 and resumed.micro_step == 8
    assert state_hash(resumed) == state_hash(reference)
    resumed_moments = jax.tree_util.tree_leaves(jax.device_get(resumed.optimizer_state))
    reference_moments = jax.tree_util.tree_leaves(jax.device_get(reference.optimizer_state))
    assert len(resumed_moments) == len(reference_moments) > 0
    for left, right in zip(resumed_moments, reference_moments):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    assert [item["optimizer_update"] for item in resumed.history] == [1, 2, 3, 4]
    for left, right in zip(resumed.history, reference.history):
        assert left["train_loss"] == right["train_loss"]
        assert left.get("held_out_loss") == right.get("held_out_loss")
        assert left["gradient_norm"] == right["gradient_norm"]
    first = json.loads((tmp_path / "run" / "update_4" / "metadata.json").read_text())
    second = json.loads((completed["run"] / "update_4" / "metadata.json").read_text())
    assert first["adapter_state_sha256"] == second["adapter_state_sha256"]
    assert (tmp_path / "run" / "update_4" / "adapter.npz").read_bytes() == (
        completed["run"] / "update_4" / "adapter.npz"
    ).read_bytes()

    # A snapshot left behind by an interrupted run is adopted when the resumed
    # run reaches the same stage with the same state ...
    adopted = resumed.ensure_snapshot(completed["run"] / "update_4", snapshot_index=1)
    assert adopted["adapter_state_sha256"] == state_hash(resumed)
    # ... and reported, never overwritten, when the states differ.
    resumed.adapter_state = jax.tree_util.tree_map(lambda value: value + 1e-3, resumed.adapter_state)
    with pytest.raises(ResumeMismatchError, match="differs from the resumed training state"):
        resumed.ensure_snapshot(completed["run"] / "update_4", snapshot_index=1)


def test_resume_is_refused_for_another_configuration_or_dataset(completed, tiny_bundle):
    other = make_trainer(
        tiny_bundle,
        budget=tiny.tiny_budget(learning_rate=5e-3),
        seed=18,
        dataset_sha256="e" * 64,
    )
    before = state_hash(other)

    with pytest.raises(ResumeMismatchError) as error:
        other.restore(completed["resume"])

    message = str(error.value)
    assert "different configuration or dataset" in message
    for name in ("budget", "dataset_sha256", "seed"):
        assert name in message
    # A refused resume leaves the fresh trainer untouched.
    assert other.update_step == 0 and state_hash(other) == before


def test_resume_rejects_damaged_or_partial_state(completed, tiny_bundle, tmp_path):
    trainer = make_trainer(tiny_bundle)

    assert trainer.restore(tmp_path / "missing") is False

    damaged = tmp_path / "damaged"
    shutil.copytree(completed["resume"], damaged)
    payload = bytearray((damaged / "state.msgpack").read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    (damaged / "state.msgpack").write_bytes(bytes(payload))
    with pytest.raises(ResumeMismatchError, match="does not match its recorded hash"):
        trainer.restore(damaged)

    partial = tmp_path / "partial"
    partial.mkdir()
    shutil.copy(completed["resume"] / "resume.json", partial / "resume.json")
    with pytest.raises(ResumeMismatchError, match="incomplete resume state"):
        trainer.restore(partial)

    stale = tmp_path / "stale"
    shutil.copytree(completed["resume"], stale)
    meta = json.loads((stale / "resume.json").read_text())
    meta["schema_version"] = 0
    (stale / "resume.json").write_text(json.dumps(meta))
    with pytest.raises(ResumeMismatchError, match="unsupported resume-state schema"):
        trainer.restore(stale)
    assert trainer.update_step == 0


def test_run_identity_names_a_different_adapter_layout(tiny_bundle):
    train, held_out = streams(tiny_bundle)
    budget = tiny.tiny_budget()
    identity = tiny.tiny_identity(tiny_bundle, train, held_out, budget)
    from dataclasses import replace

    with pytest.raises(ValueError, match="different AdapterSpec than the live model"):
        ZooTrainer(
            bundle=tiny_bundle,
            stream=train,
            held_out_stream=held_out,
            budget=budget,
            identity=replace(identity, adapter_spec_sha256="0" * 64),
            token_width=tiny.TOKEN_WIDTH,
            sequence_window=tiny.SEQUENCE_WINDOW,
        )


def test_a_crash_between_resume_state_and_snapshot_is_repaired_exactly(
    completed, tiny_bundle, tmp_path
):
    """The resume state of a stage is written first; a restart writes or adopts its snapshot."""

    resume = tmp_path / "resume"
    shutil.copytree(completed["resume_at_first_stage"], resume)
    run = tmp_path / "run"

    # Crash after the resume state of stage one, before its snapshot existed.
    restarted = make_trainer(tiny_bundle)
    assert restarted.restore(resume) and restarted.update_step == STAGES[0]
    written = restarted.ensure_snapshot(run / f"update_{STAGES[0]}", snapshot_index=0)
    reference = json.loads(
        (completed["run"] / f"update_{STAGES[0]}" / "metadata.json").read_text()
    )
    assert written["adapter_state_sha256"] == reference["adapter_state_sha256"]

    # Crash after both were written: the restart adopts the snapshot unchanged.
    again = make_trainer(tiny_bundle)
    assert again.restore(resume)
    before = (run / f"update_{STAGES[0]}" / "adapter.npz").read_bytes()
    adopted = again.ensure_snapshot(run / f"update_{STAGES[0]}", snapshot_index=0)
    assert adopted["adapter_state_sha256"] == written["adapter_state_sha256"]
    assert (run / f"update_{STAGES[0]}" / "adapter.npz").read_bytes() == before
    assert [item["optimizer_updates"] for item in again.snapshots] == [STAGES[0]]
