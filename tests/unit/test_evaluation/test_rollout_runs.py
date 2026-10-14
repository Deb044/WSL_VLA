"""Rollout runs resume per task, are never overwritten, and keep failed attempts."""

from __future__ import annotations

import json

import pytest

from wsl_vla.evaluation.rollout_runs import (
    EPISODES_FILE,
    MANIFEST_FILE,
    POLICIES_FILE,
    SUMMARY_FILE,
    RolloutRunConflict,
    completed_task_records,
    open_rollout_run,
)


TASKS = ("libero_goal_0", "libero_goal_1", "libero_goal_2")
CONDITIONS = ("frozen_base", "zoo_adapter")
ROLLOUTS = 2
IDENTITY = {
    "manifest_sha256": "a" * 64,
    "fold": "holdout_goal",
    "alignment_seed": 0,
    "tasks": list(TASKS),
    "conditions": list(CONDITIONS),
    "rollouts_per_task": ROLLOUTS,
}


def episodes(task: str, *, conditions=CONDITIONS, rollouts=ROLLOUTS) -> list[dict]:
    return [
        {"task_id": task, "condition": condition, "episode_seed": ordinal, "success": ordinal % 2 == 0}
        for condition in conditions
        for ordinal in range(rollouts)
    ]


def start(tmp_path, *, identity=IDENTITY, attempt=None):
    return open_rollout_run(
        tmp_path / "rollouts" / "validation" / "fold" / "seed_0",
        identity=identity,
        tasks=TASKS,
        conditions=CONDITIONS,
        rollouts_per_task=ROLLOUTS,
        preserve_root=tmp_path / "failed_attempts" / "rollouts" / "validation" / "fold" / "seed_0",
        attempt={"command": ["rollouts"], **(attempt or {})},
        details={"experiment": "pilot"},
    )


def test_fresh_run_records_tasks_and_finishes(tmp_path):
    run = start(tmp_path)

    assert run.status == "fresh" and run.completed_tasks == ()
    for task in TASKS:
        run.record_task(task, episodes(task), {"frozen_base": {"kind": "base"}})
    run.finish({"conditions": {}})

    manifest = json.loads((run.directory / MANIFEST_FILE).read_text())
    assert manifest["finished"] is True and manifest["failures"] == []
    assert manifest["completed_tasks"] == list(TASKS)
    assert manifest["experiment"] == "pilot"
    assert len(manifest["attempts"]) == 1 and manifest["attempts"][0]["command"] == ["rollouts"]
    lines = (run.directory / EPISODES_FILE).read_text().splitlines()
    assert len(lines) == len(TASKS) * len(CONDITIONS) * ROLLOUTS
    assert set(json.loads((run.directory / POLICIES_FILE).read_text())) == set(TASKS)
    assert (run.directory / SUMMARY_FILE).is_file()


def test_finished_run_is_recognized_and_left_untouched(tmp_path):
    run = start(tmp_path)
    for task in TASKS:
        run.record_task(task, episodes(task), {})
    run.finish({"conditions": {}})
    before = {item.name: item.read_bytes() for item in run.directory.iterdir()}

    again = start(tmp_path, attempt={"git_commit": "later"})

    assert again.status == "complete"
    assert len(again.records) == len(TASKS) * len(CONDITIONS) * ROLLOUTS
    assert {item.name: item.read_bytes() for item in again.directory.iterdir()} == before


def test_interrupted_run_resumes_after_its_last_complete_task(tmp_path):
    run = start(tmp_path)
    run.record_task(TASKS[0], episodes(TASKS[0]), {"zoo_adapter": {"sha": "1"}})
    # The second task was interrupted: only one condition reached the log, and
    # the final line was cut short.
    with (run.directory / EPISODES_FILE).open("a") as stream:
        for record in episodes(TASKS[1], conditions=CONDITIONS[:1]):
            stream.write(json.dumps(record) + "\n")
        stream.write('{"task_id": "libero_goal_1", "condi')
    run.fail(RuntimeError("simulator crashed"))

    resumed = start(tmp_path, attempt={"git_commit": "retry"})

    assert resumed.status == "resumed"
    assert resumed.completed_tasks == (TASKS[0],)
    assert [record["task_id"] for record in resumed.records] == [TASKS[0]] * 4
    assert set(resumed.policy_records) == {TASKS[0]}
    # The partial task was dropped from the log so it can be rerun whole.
    logged = [json.loads(line) for line in (resumed.directory / EPISODES_FILE).read_text().splitlines()]
    assert {record["task_id"] for record in logged} == {TASKS[0]}
    manifest = json.loads((resumed.directory / MANIFEST_FILE).read_text())
    assert len(manifest["attempts"]) == 2
    assert manifest["failures"][0]["type"] == "RuntimeError"
    assert manifest["failures"][0]["completed_tasks"] == [TASKS[0]]

    for task in TASKS[1:]:
        resumed.record_task(task, episodes(task), {})
    resumed.finish({"conditions": {}})
    assert json.loads((resumed.directory / MANIFEST_FILE).read_text())["finished"] is True
    assert len(resumed.records) == len(TASKS) * len(CONDITIONS) * ROLLOUTS


def test_task_without_policy_provenance_is_rerun(tmp_path):
    run = start(tmp_path)
    run.record_task(TASKS[0], episodes(TASKS[0]), {})
    # Episodes of the next task reached the log but the process died before the
    # policy record and the completion mark were written.
    with (run.directory / EPISODES_FILE).open("a") as stream:
        for record in episodes(TASKS[1]):
            stream.write(json.dumps(record) + "\n")

    resumed = start(tmp_path)

    assert resumed.completed_tasks == (TASKS[0],)
    assert {record["task_id"] for record in resumed.records} == {TASKS[0]}


def test_unfinished_run_with_another_configuration_is_preserved_not_resumed(tmp_path):
    run = start(tmp_path)
    run.record_task(TASKS[0], episodes(TASKS[0]), {})
    run.fail(RuntimeError("interrupted"))

    changed = start(tmp_path, identity={**IDENTITY, "manifest_sha256": "b" * 64})

    assert changed.status == "fresh" and changed.completed_tasks == ()
    assert changed.preserved_attempt is not None
    preserved = changed.preserved_attempt
    assert preserved.parent.name == "seed_0" and preserved.name == "attempt-001"
    assert "failed_attempts" in preserved.parts and "rollouts" in preserved.parts
    old = json.loads((preserved / MANIFEST_FILE).read_text())
    assert old["identity"]["manifest_sha256"] == "a" * 64
    assert old["completed_tasks"] == [TASKS[0]]
    assert len((preserved / EPISODES_FILE).read_text().splitlines()) == 4
    assert not (changed.directory / EPISODES_FILE).exists()
    assert json.loads((changed.directory / MANIFEST_FILE).read_text())[
        "preserved_previous_attempt"
    ] == str(preserved)

    # A second abandoned attempt gets the next slot instead of replacing the first.
    changed.fail(RuntimeError("again"))
    third = start(tmp_path, identity={**IDENTITY, "manifest_sha256": "c" * 64})
    assert third.preserved_attempt.name == "attempt-002"
    assert (preserved / MANIFEST_FILE).is_file()


def test_finished_run_with_another_configuration_is_a_conflict(tmp_path):
    run = start(tmp_path)
    for task in TASKS:
        run.record_task(task, episodes(task), {})
    run.finish({"conditions": {}})
    before = (run.directory / MANIFEST_FILE).read_bytes()

    with pytest.raises(RolloutRunConflict, match="never overwritten"):
        start(tmp_path, identity={**IDENTITY, "rollouts_per_task": 50})

    assert (run.directory / MANIFEST_FILE).read_bytes() == before
    assert not (tmp_path / "failed_attempts").exists()


def test_directory_without_a_manifest_is_preserved(tmp_path):
    directory = tmp_path / "rollouts" / "validation" / "fold" / "seed_0"
    directory.mkdir(parents=True)
    (directory / EPISODES_FILE).write_text(json.dumps(episodes(TASKS[0])[0]) + "\n")

    run = start(tmp_path)

    assert run.status == "fresh"
    assert (run.preserved_attempt / EPISODES_FILE).is_file()


def test_record_task_rejects_duplicates_and_foreign_episodes(tmp_path):
    run = start(tmp_path)
    run.record_task(TASKS[0], episodes(TASKS[0]), {})

    with pytest.raises(ValueError, match="already recorded"):
        run.record_task(TASKS[0], episodes(TASKS[0]), {})
    with pytest.raises(ValueError, match="belong to a different task"):
        run.record_task(TASKS[1], episodes(TASKS[2]), {})


def test_a_task_is_complete_only_with_every_paired_episode():
    records = (
        episodes(TASKS[0])
        + episodes(TASKS[1], rollouts=1)
        + episodes(TASKS[2], conditions=(*CONDITIONS, "unexpected"))
    )

    complete = completed_task_records(
        records, tasks=TASKS, conditions=CONDITIONS, rollouts_per_task=ROLLOUTS
    )

    assert set(complete) == {TASKS[0]}
    assert len(complete[TASKS[0]]) == len(CONDITIONS) * ROLLOUTS


def test_corruption_before_the_final_line_is_not_silently_dropped(tmp_path):
    run = start(tmp_path)
    run.record_task(TASKS[0], episodes(TASKS[0]), {})
    path = run.directory / EPISODES_FILE
    lines = path.read_text().splitlines()
    lines[1] = lines[1][:10]
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="corrupt at line 2"):
        start(tmp_path)
