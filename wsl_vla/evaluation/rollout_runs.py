"""On-disk state of one paired rollout run: resumable, never overwritten.

A rollout job can run for hours, so an interruption must not cost the tasks that
already finished, and a failed attempt must never be silently replaced:

* a run that finished under the same configuration is reported as complete;
* an interrupted run under the same configuration resumes after its last
  completed task (episodes of a task are independent of every other task: the
  initial states and seeds depend only on the task and the rollout ordinal);
* an interrupted run under a *different* configuration is moved, intact, to the
  experiment's ``failed_attempts`` tree before a fresh run starts;
* a *finished* run under a different configuration is a conflict and is left
  untouched.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import canonical_sha256
from ..experiments.provenance import atomic_write_bytes, atomic_write_json


ROLLOUT_RUN_SCHEMA_VERSION = 2
MANIFEST_FILE = "manifest.json"
EPISODES_FILE = "episodes.jsonl"
POLICIES_FILE = "policies.json"
SUMMARY_FILE = "summary.json"


class RolloutRunConflict(RuntimeError):
    """A finished rollout run with another configuration occupies the directory."""


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_episodes(path: Path) -> list[dict[str, Any]]:
    """Episode records, ignoring a final line truncated by an interruption."""

    if not path.is_file():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines) - 1:
                raise ValueError(f"{path} is corrupt at line {number + 1}") from None
    return records


def completed_task_records(
    records: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    conditions: Sequence[str],
    rollouts_per_task: int,
) -> dict[str, list[dict[str, Any]]]:
    """Records of the tasks whose every condition has every paired episode."""

    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(str(record["task_id"]), []).append(dict(record))
    expected = {condition: rollouts_per_task for condition in conditions}
    complete = {}
    for identifier in tasks:
        observed = Counter(str(record["condition"]) for record in by_task.get(identifier, ()))
        if dict(observed) == expected:
            complete[identifier] = by_task[identifier]
    return complete


@dataclass
class RolloutRun:
    """Handle on a run directory; every write goes through here."""

    directory: Path
    identity: dict[str, Any]
    manifest: dict[str, Any]
    status: str  # "fresh", "resumed", or "complete"
    records: list[dict[str, Any]] = field(default_factory=list)
    policy_records: dict[str, Any] = field(default_factory=dict)
    preserved_attempt: Path | None = None

    @property
    def completed_tasks(self) -> tuple[str, ...]:
        return tuple(self.manifest["completed_tasks"])

    def _save_manifest(self) -> None:
        atomic_write_json(self.directory / MANIFEST_FILE, self.manifest)

    def record_task(
        self,
        identifier: str,
        records: Sequence[Mapping[str, Any]],
        policy_record: Mapping[str, Any],
    ) -> None:
        """Persist one finished task: episodes first, then the completion mark."""

        if identifier in self.manifest["completed_tasks"]:
            raise ValueError(f"task {identifier} is already recorded in {self.directory}")
        if any(str(record["task_id"]) != identifier for record in records):
            raise ValueError("episode records belong to a different task")
        with (self.directory / EPISODES_FILE).open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records.extend(dict(record) for record in records)
        self.policy_records[identifier] = dict(policy_record)
        atomic_write_json(self.directory / POLICIES_FILE, self.policy_records)
        self.manifest["completed_tasks"].append(identifier)
        self._save_manifest()

    def fail(self, error: BaseException) -> None:
        self.manifest["failures"].append(
            {
                "attempt": len(self.manifest["attempts"]),
                "type": type(error).__name__,
                "message": str(error),
                "completed_tasks": list(self.manifest["completed_tasks"]),
            }
        )
        self._save_manifest()

    def finish(self, summary: Mapping[str, Any]) -> None:
        atomic_write_json(self.directory / SUMMARY_FILE, summary)
        self.manifest["finished"] = True
        self._save_manifest()


def _preserve(directory: Path, preserve_root: Path) -> Path:
    """Move an abandoned attempt out of the results tree without deleting it."""

    preserve_root.mkdir(parents=True, exist_ok=True)
    index = 1 + sum(1 for item in preserve_root.iterdir() if item.name.startswith("attempt-"))
    target = preserve_root / f"attempt-{index:03d}"
    while target.exists():
        index += 1
        target = preserve_root / f"attempt-{index:03d}"
    os.replace(directory, target)
    return target


def open_rollout_run(
    directory: str | Path,
    *,
    identity: Mapping[str, Any],
    tasks: Sequence[str],
    conditions: Sequence[str],
    rollouts_per_task: int,
    preserve_root: str | Path,
    attempt: Mapping[str, Any],
    details: Mapping[str, Any] | None = None,
) -> RolloutRun:
    """Create, resume, or recognize as complete the run stored in ``directory``.

    ``identity`` is everything that determines the episodes (experiment
    manifest, fold, seeds, tasks, conditions, rollout budget, model and
    checkpoint hashes). ``attempt`` is provenance of this invocation (command,
    code revision, environment) and never affects resumption. ``details`` are
    descriptive fields copied into a new manifest.
    """

    directory = Path(directory)
    identity = json.loads(json.dumps(dict(identity), sort_keys=True))
    digest = canonical_sha256(identity)
    stamped = {"started_at": datetime.now(timezone.utc).isoformat(), **dict(attempt)}
    preserved = None
    existing = _read_json(directory / MANIFEST_FILE)
    occupied = directory.exists() and any(directory.iterdir())

    if occupied and existing is not None and existing.get("identity_sha256") == digest:
        if existing.get("schema_version") != ROLLOUT_RUN_SCHEMA_VERSION:
            raise RolloutRunConflict(f"unsupported rollout-run schema in {directory}")
        if existing.get("finished") and (directory / SUMMARY_FILE).is_file():
            return RolloutRun(
                directory=directory,
                identity=identity,
                manifest=existing,
                status="complete",
                records=_read_episodes(directory / EPISODES_FILE),
                policy_records=_read_json(directory / POLICIES_FILE) or {},
            )
        policies = _read_json(directory / POLICIES_FILE) or {}
        complete = completed_task_records(
            _read_episodes(directory / EPISODES_FILE),
            tasks=tasks,
            conditions=conditions,
            rollouts_per_task=rollouts_per_task,
        )
        # A task counts only if its policy provenance was persisted too.
        kept = [identifier for identifier in tasks if identifier in complete and identifier in policies]
        records = [record for identifier in kept for record in complete[identifier]]
        atomic_write_bytes(
            directory / EPISODES_FILE,
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records).encode(),
        )
        policy_records = {identifier: policies[identifier] for identifier in kept}
        atomic_write_json(directory / POLICIES_FILE, policy_records)
        existing["completed_tasks"] = kept
        existing["finished"] = False
        existing["attempts"] = [*existing.get("attempts", []), stamped]
        run = RolloutRun(
            directory=directory,
            identity=identity,
            manifest=existing,
            status="resumed",
            records=records,
            policy_records=policy_records,
        )
        run._save_manifest()
        return run

    if occupied:
        if existing is not None and existing.get("finished"):
            raise RolloutRunConflict(
                f"{directory} holds a finished rollout run with a different configuration; "
                "finished runs are never overwritten. Use a new experiment manifest name or "
                "research root for the changed configuration."
            )
        preserved = _preserve(directory, Path(preserve_root))

    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        **dict(details or {}),
        "schema_version": ROLLOUT_RUN_SCHEMA_VERSION,
        "identity": identity,
        "identity_sha256": digest,
        "tasks": list(tasks),
        "conditions": list(conditions),
        "rollouts_per_task": int(rollouts_per_task),
        "completed_tasks": [],
        "attempts": [stamped],
        "failures": [],
        "finished": False,
        "preserved_previous_attempt": None if preserved is None else str(preserved),
    }
    run = RolloutRun(
        directory=directory,
        identity=identity,
        manifest=manifest,
        status="fresh",
        preserved_attempt=preserved,
    )
    run._save_manifest()
    return run
