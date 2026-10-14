"""Versioned experiment manifest for the zoo and alignment experiments.

The manifest is the *sole* source of task selections, seeds, snapshot stages,
episode splits, fold assignments, and expected sample counts. Trainers,
archive builders, validators, and evaluators derive every population count
from it; none of them hard-codes 360, 270, 54, ``{8, 9}``, or a seed list.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Mapping, Sequence

import yaml

from ..contracts import canonical_sha256


EXPERIMENT_MANIFEST_SCHEMA_VERSION = 1
PROFILES = ("integrity", "pilot", "full")
ALIGNMENT_CONDITIONS = ("aligned", "reconstruction_only")
SELECTION_VALIDATION = "validation_tasks"
SELECTION_INTEGRITY = "integrity_reuse_training_tasks"
SPLITS = ("train", "validation", "held_out")


class ManifestError(ValueError):
    """The experiment manifest is inconsistent or violates a leakage rule."""


def task_id(suite: str, task_index: int) -> str:
    return f"{suite}_{int(task_index)}"


def split_task_id(identifier: str) -> tuple[str, int]:
    suite, _, index = identifier.rpartition("_")
    if not suite or not index.isdigit():
        raise ManifestError(f"malformed task id: {identifier!r}")
    return suite, int(index)


@dataclass(frozen=True)
class SampleIdentity:
    """One adapter snapshot of one independent training run."""

    task_id: str
    seed: int
    snapshot_index: int
    optimizer_updates: int

    @property
    def suite(self) -> str:
        return split_task_id(self.task_id)[0]

    @property
    def task_index(self) -> int:
        return split_task_id(self.task_id)[1]

    @property
    def run_id(self) -> str:
        """Snapshots of one run are correlated; uncertainty is aggregated by run."""

        return f"{self.task_id}/seed_{self.seed}"

    @property
    def sample_id(self) -> str:
        return f"{self.run_id}/update_{self.optimizer_updates}"

    def relative_directory(self) -> Path:
        return Path(self.suite) / self.task_id / f"seed_{self.seed}" / f"update_{self.optimizer_updates}"


@dataclass(frozen=True)
class ZooBudget:
    """Explicit optimizer-update budget, sample exposure, and snapshot stages."""

    optimizer_updates: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    snapshot_updates: tuple[int, ...]
    held_out_loss_every: int
    held_out_loss_batches: int
    frozen: bool
    inspection_updates: tuple[int, ...] = ()
    inspection_tasks: tuple[str, ...] = ()

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def sample_exposure(self, optimizer_updates: int) -> int:
        """Training windows consumed after ``optimizer_updates`` updates."""

        return int(optimizer_updates) * self.effective_batch_size

    def validate(self) -> None:
        if min(self.optimizer_updates, self.micro_batch_size, self.gradient_accumulation_steps) <= 0:
            raise ManifestError("zoo update count, micro-batch, and accumulation must be positive")
        if self.learning_rate <= 0:
            raise ManifestError("zoo learning rate must be positive")
        stages = self.snapshot_updates
        if not stages or list(stages) != sorted(set(stages)):
            raise ManifestError("snapshot stages must be unique, increasing optimizer-update counts")
        if stages[0] <= 0 or stages[-1] != self.optimizer_updates:
            raise ManifestError(
                "snapshot stages must be positive and end at the final optimizer update"
            )
        if self.held_out_loss_every <= 0 or self.held_out_loss_batches <= 0:
            raise ManifestError("held-out loss cadence and batch count must be positive")
        if list(self.inspection_updates) != sorted(set(self.inspection_updates)):
            raise ManifestError("inspection stages must be unique and increasing")
        if any(value <= 0 for value in self.inspection_updates):
            raise ManifestError("inspection stages must be positive")
        if bool(self.inspection_updates) != bool(self.inspection_tasks):
            raise ManifestError("budget inspection needs both stages and source tasks")

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_updates": int(self.optimizer_updates),
            "micro_batch_size": int(self.micro_batch_size),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "effective_batch_size": int(self.effective_batch_size),
            "learning_rate": float(self.learning_rate),
            "snapshot_updates": [int(value) for value in self.snapshot_updates],
            "held_out_loss_every": int(self.held_out_loss_every),
            "held_out_loss_batches": int(self.held_out_loss_batches),
            "frozen": bool(self.frozen),
            "inspection_updates": [int(value) for value in self.inspection_updates],
            "inspection_tasks": list(self.inspection_tasks),
        }

    def training_identity(self) -> dict[str, Any]:
        """Fields that change a trained adapter (excludes bookkeeping flags)."""

        payload = self.to_dict()
        for key in ("frozen", "inspection_updates", "inspection_tasks"):
            payload.pop(key)
        return payload


@dataclass(frozen=True)
class EpisodeSplitSpec:
    seed: int
    train_fraction: float
    validation_fraction: float

    def validate(self) -> None:
        if not (0 < self.train_fraction < 1 and 0 < self.validation_fraction < 1):
            raise ManifestError("episode split fractions must lie in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ManifestError("train + validation episode fractions must leave test episodes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "train_fraction": float(self.train_fraction),
            "validation_fraction": float(self.validation_fraction),
        }


@dataclass(frozen=True)
class FoldSpec:
    """Task-level alignment split; every seed/snapshot of a task follows its task."""

    name: str
    held_out_suite: str | None
    train_tasks: tuple[str, ...]
    validation_tasks: tuple[str, ...]
    held_out_tasks: tuple[str, ...]
    selection: str

    def split_of(self, identifier: str) -> str | None:
        """Role of a task in this fold. Integrity folds report ``train`` first."""

        if identifier in self.train_tasks:
            return "train"
        if identifier in self.validation_tasks:
            return "validation"
        if identifier in self.held_out_tasks:
            return "held_out"
        return None

    def tasks(self, split: str) -> tuple[str, ...]:
        if split not in SPLITS:
            raise ManifestError(f"unknown alignment split: {split}")
        return getattr(self, f"{split}_tasks")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "held_out_suite": self.held_out_suite,
            "train_tasks": list(self.train_tasks),
            "validation_tasks": list(self.validation_tasks),
            "held_out_tasks": list(self.held_out_tasks),
            "selection": self.selection,
        }


@dataclass(frozen=True)
class AlignmentPlan:
    seeds: tuple[int, ...]
    conditions: tuple[str, ...]
    tasks_per_batch: int
    samples_per_task: int
    optimizer_updates: int
    validation_every: int

    def validate(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ManifestError("alignment seeds must be unique and non-empty")
        if tuple(self.conditions) != ALIGNMENT_CONDITIONS:
            raise ManifestError(
                f"the two primary alignment conditions are {ALIGNMENT_CONDITIONS}, in that order"
            )
        if self.tasks_per_batch < 2 or self.samples_per_task < 1:
            raise ManifestError(
                "alignment batches need at least two task identities and one sample per task"
            )
        if self.optimizer_updates <= 0 or not 0 < self.validation_every <= self.optimizer_updates:
            raise ManifestError("alignment update budget and validation cadence are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": [int(value) for value in self.seeds],
            "conditions": list(self.conditions),
            "tasks_per_batch": int(self.tasks_per_batch),
            "samples_per_task": int(self.samples_per_task),
            "optimizer_updates": int(self.optimizer_updates),
            "validation_every": int(self.validation_every),
        }


@dataclass(frozen=True)
class RolloutPlan:
    development: int
    final: int
    seed: int
    zoo_reference_seed: int
    max_episode_steps: int | None = None

    def validate(self) -> None:
        if min(self.development, self.final) <= 0:
            raise ManifestError("rollout counts must be positive")
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ManifestError("a rollout horizon override must be positive")

    def count(self, tier: str) -> int:
        if tier not in {"development", "final"}:
            raise ManifestError("rollout tier must be development or final")
        return self.development if tier == "development" else self.final

    def to_dict(self) -> dict[str, Any]:
        return {
            "development": int(self.development),
            "final": int(self.final),
            "seed": int(self.seed),
            "zoo_reference_seed": int(self.zoo_reference_seed),
            "max_episode_steps": self.max_episode_steps,
        }


@dataclass(frozen=True)
class ExperimentManifest:
    name: str
    profile: str
    publication_evidence: bool
    description: str
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    zoo: ZooBudget
    episode_split: EpisodeSplitSpec
    evidence: Mapping[str, int]
    folds: tuple[FoldSpec, ...]
    alignment: AlignmentPlan
    rollouts: RolloutPlan
    expected: Mapping[str, Any]
    schema_version: int = EXPERIMENT_MANIFEST_SCHEMA_VERSION

    # ------------------------------------------------------------------ derived
    def runs(self) -> tuple[tuple[str, int], ...]:
        """Independent training runs: one per task and seed."""

        return tuple((identifier, seed) for identifier in self.tasks for seed in self.seeds)

    def samples(self, tasks: Sequence[str] | None = None) -> tuple[SampleIdentity, ...]:
        selected = self.tasks if tasks is None else tuple(tasks)
        unknown = sorted(set(selected) - set(self.tasks))
        if unknown:
            raise ManifestError(f"tasks are not part of this experiment: {unknown}")
        return tuple(
            SampleIdentity(identifier, seed, index, updates)
            for identifier in selected
            for seed in self.seeds
            for index, updates in enumerate(self.zoo.snapshot_updates)
        )

    def fold(self, name: str) -> FoldSpec:
        for fold in self.folds:
            if fold.name == name:
                return fold
        raise ManifestError(f"experiment {self.name!r} has no fold {name!r}")

    def fold_samples(self, fold: FoldSpec | str, split: str) -> tuple[SampleIdentity, ...]:
        spec = self.fold(fold) if isinstance(fold, str) else fold
        return self.samples(spec.tasks(split))

    def derived_counts(self) -> dict[str, Any]:
        return {
            "independent_runs": len(self.runs()),
            "adapter_samples": len(self.samples()),
            "folds": {
                fold.name: {
                    f"{split}_{kind}": (
                        len(fold.tasks(split)) if kind == "tasks" else len(self.fold_samples(fold, split))
                    )
                    for split in SPLITS
                    for kind in ("tasks", "samples")
                }
                for fold in self.folds
            },
        }

    # --------------------------------------------------------------- identities
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "profile": self.profile,
            "publication_evidence": bool(self.publication_evidence),
            "description": self.description,
            "tasks": list(self.tasks),
            "seeds": [int(value) for value in self.seeds],
            "zoo": self.zoo.to_dict(),
            "episode_split": self.episode_split.to_dict(),
            "evidence": {key: int(value) for key, value in sorted(self.evidence.items())},
            "folds": [fold.to_dict() for fold in self.folds],
            "alignment": self.alignment.to_dict(),
            "rollouts": self.rollouts.to_dict(),
            "expected": self.derived_counts(),
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def zoo_identity(self) -> dict[str, Any]:
        """Everything that determines a trained adapter besides task and seed."""

        return {
            "zoo": self.zoo.training_identity(),
            "episode_split": self.episode_split.to_dict(),
        }

    def zoo_sha256(self) -> str:
        return canonical_sha256(self.zoo_identity())

    def evidence_identity(self) -> dict[str, Any]:
        return {
            "evidence": {key: int(value) for key, value in sorted(self.evidence.items())},
            "episode_split": self.episode_split.to_dict(),
        }

    # --------------------------------------------------------------- validation
    def validate(self, reference_tasks: Mapping[str, Any] | None = None) -> None:
        if self.schema_version != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
            raise ManifestError(f"unsupported experiment manifest schema {self.schema_version}")
        if self.profile not in PROFILES:
            raise ManifestError(f"profile must be one of {PROFILES}")
        if not self.name or not self.name.replace("_", "").replace("-", "").isalnum():
            raise ManifestError("experiment name must be a non-empty identifier")
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ManifestError("experiment tasks must be unique and non-empty")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ManifestError("zoo seeds must be unique, non-negative, and non-empty")
        for identifier in self.tasks:
            suite, index = split_task_id(identifier)
            if reference_tasks is not None:
                suites = reference_tasks["suites"]
                if suite not in suites or not 0 <= index < len(suites[suite]):
                    raise ManifestError(f"{identifier} is not a reference benchmark task")
        self.zoo.validate()
        self.episode_split.validate()
        self.alignment.validate()
        self.rollouts.validate()
        if set(self.evidence) != {"frames_per_episode", "action_chunk_length"}:
            raise ManifestError("evidence needs frames_per_episode and action_chunk_length")
        if min(self.evidence.values()) <= 0:
            raise ManifestError("evidence extraction settings must be positive")
        unknown_inspection = sorted(set(self.zoo.inspection_tasks) - set(self.tasks))
        if unknown_inspection:
            raise ManifestError(f"budget inspection names unknown tasks: {unknown_inspection}")
        if self.rollouts.zoo_reference_seed not in self.seeds:
            raise ManifestError("the zoo reference seed for rollouts must be a zoo seed")
        if self.publication_evidence and self.profile != "full":
            raise ManifestError("only the full profile may be declared publication evidence")
        if not self.folds or len({fold.name for fold in self.folds}) != len(self.folds):
            raise ManifestError("fold names must be unique and non-empty")
        for fold in self.folds:
            self._validate_fold(fold)
        declared = _normalize(self.expected)
        derived = _normalize(self.derived_counts())
        if declared != derived:
            raise ManifestError(
                "declared expected counts differ from the counts derived from tasks, "
                f"seeds, snapshots, and folds: declared={declared}, derived={derived}"
            )

    def _validate_fold(self, fold: FoldSpec) -> None:
        known = set(self.tasks)
        groups = {split: fold.tasks(split) for split in SPLITS}
        for split, members in groups.items():
            if len(set(members)) != len(members):
                raise ManifestError(f"fold {fold.name} repeats a task inside {split}")
            unknown = sorted(set(members) - known)
            if unknown:
                raise ManifestError(f"fold {fold.name} {split} names unknown tasks: {unknown}")
        if len(groups["train"]) < 2:
            raise ManifestError(f"fold {fold.name} needs at least two training tasks")
        if fold.selection == SELECTION_INTEGRITY:
            if self.profile != "integrity":
                raise ManifestError(
                    f"fold {fold.name}: reusing training tasks for selection is allowed "
                    "only in the integrity profile"
                )
            if set(groups["validation"]) != set(groups["train"]):
                raise ManifestError(
                    f"fold {fold.name}: integrity selection must reuse exactly the training tasks"
                )
            overlap = set(groups["held_out"]) & set(groups["train"])
        elif fold.selection == SELECTION_VALIDATION:
            if not groups["validation"]:
                raise ManifestError(f"fold {fold.name} has no validation tasks")
            overlap = (
                (set(groups["train"]) & set(groups["validation"]))
                | (set(groups["train"]) & set(groups["held_out"]))
                | (set(groups["validation"]) & set(groups["held_out"]))
            )
        else:
            raise ManifestError(f"fold {fold.name} has unknown selection {fold.selection!r}")
        if overlap:
            raise ManifestError(
                f"fold {fold.name} places tasks in more than one alignment split: {sorted(overlap)}"
            )
        if fold.held_out_suite is not None:
            leaked = sorted(
                identifier
                for split in ("train", "validation")
                for identifier in groups[split]
                if split_task_id(identifier)[0] == fold.held_out_suite
            )
            if leaked:
                raise ManifestError(
                    f"fold {fold.name}: held-out suite tasks leaked into training/selection: {leaked}"
                )
            foreign = sorted(
                identifier
                for identifier in groups["held_out"]
                if split_task_id(identifier)[0] != fold.held_out_suite
            )
            if foreign:
                raise ManifestError(
                    f"fold {fold.name}: held-out tasks outside the held-out suite: {foreign}"
                )
        elif groups["held_out"]:
            raise ManifestError(f"fold {fold.name} lists held-out tasks without a held-out suite")

    def reference_sample(self, identifier: str) -> SampleIdentity:
        """The zoo adapter deployed for a task in rollouts: reference seed, final snapshot."""

        index = len(self.zoo.snapshot_updates) - 1
        return SampleIdentity(
            identifier, self.rollouts.zoo_reference_seed, index, self.zoo.snapshot_updates[index]
        )

    def require_frozen_budget(self, purpose: str) -> None:
        if not self.zoo.frozen:
            raise ManifestError(
                f"{purpose} requires a frozen zoo budget. Validate the budget on source tasks, "
                "then freeze it with scripts/reporting/freeze_zoo_budget.py."
            )


def _normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _tasks_from_mapping(mapping: Mapping[str, Sequence[int]] | None) -> tuple[str, ...]:
    if not mapping:
        return ()
    return tuple(
        task_id(suite, index) for suite, indices in mapping.items() for index in indices
    )


def manifest_from_dict(payload: Mapping[str, Any]) -> ExperimentManifest:
    """Build a manifest from its YAML/JSON form (``{suite: [indices]}`` task maps)."""

    if payload.get("schema_version") != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported experiment manifest schema {payload.get('schema_version')!r}"
        )
    try:
        zoo = payload["zoo"]
        tasks = payload["tasks"]
        folds = []
        for item in payload["folds"]:
            folds.append(
                FoldSpec(
                    name=str(item["name"]),
                    held_out_suite=(
                        None if item.get("held_out_suite") is None else str(item["held_out_suite"])
                    ),
                    train_tasks=_fold_tasks(item["train_tasks"]),
                    validation_tasks=_fold_tasks(item["validation_tasks"]),
                    held_out_tasks=_fold_tasks(item.get("held_out_tasks")),
                    selection=str(item["selection"]),
                )
            )
        return ExperimentManifest(
            name=str(payload["name"]),
            profile=str(payload["profile"]),
            publication_evidence=bool(payload["publication_evidence"]),
            description=str(payload.get("description", "")),
            tasks=_fold_tasks(tasks),
            seeds=tuple(int(value) for value in payload["seeds"]),
            zoo=ZooBudget(
                optimizer_updates=int(zoo["optimizer_updates"]),
                micro_batch_size=int(zoo["micro_batch_size"]),
                gradient_accumulation_steps=int(zoo["gradient_accumulation_steps"]),
                learning_rate=float(zoo["learning_rate"]),
                snapshot_updates=tuple(int(value) for value in zoo["snapshot_updates"]),
                held_out_loss_every=int(zoo["held_out_loss_every"]),
                held_out_loss_batches=int(zoo["held_out_loss_batches"]),
                frozen=bool(zoo["frozen"]),
                inspection_updates=tuple(int(v) for v in zoo.get("inspection_updates", ())),
                inspection_tasks=_fold_tasks(zoo.get("inspection_tasks")),
            ),
            episode_split=EpisodeSplitSpec(
                seed=int(payload["episode_split"]["seed"]),
                train_fraction=float(payload["episode_split"]["train_fraction"]),
                validation_fraction=float(payload["episode_split"]["validation_fraction"]),
            ),
            evidence={key: int(value) for key, value in payload["evidence"].items()},
            folds=tuple(folds),
            alignment=AlignmentPlan(
                seeds=tuple(int(value) for value in payload["alignment"]["seeds"]),
                conditions=tuple(str(value) for value in payload["alignment"]["conditions"]),
                tasks_per_batch=int(payload["alignment"]["tasks_per_batch"]),
                samples_per_task=int(payload["alignment"]["samples_per_task"]),
                optimizer_updates=int(payload["alignment"]["optimizer_updates"]),
                validation_every=int(payload["alignment"]["validation_every"]),
            ),
            rollouts=RolloutPlan(
                development=int(payload["rollouts"]["development"]),
                final=int(payload["rollouts"]["final"]),
                seed=int(payload["rollouts"]["seed"]),
                zoo_reference_seed=int(payload["rollouts"]["zoo_reference_seed"]),
                max_episode_steps=(
                    None
                    if payload["rollouts"].get("max_episode_steps") is None
                    else int(payload["rollouts"]["max_episode_steps"])
                ),
            ),
            expected=dict(payload["expected"]),
        )
    except KeyError as exc:
        raise ManifestError(f"experiment manifest lacks required field {exc.args[0]!r}") from exc


def _fold_tasks(value: Any) -> tuple[str, ...]:
    """Accept ``{suite: [indices]}`` or an explicit list of task ids."""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        return _tasks_from_mapping(value)
    return tuple(str(item) for item in value)


def load_experiment_manifest(
    path: str | Path, *, reference_tasks: Mapping[str, Any] | None = None
) -> ExperimentManifest:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ManifestError(f"{path} must contain a YAML mapping")
    manifest = manifest_from_dict(payload)
    manifest.validate(reference_tasks)
    return manifest


# ------------------------------------------------------------------ run layout
HELD_OUT_MARKER = "HELD_OUT_RESULTS.json"


@dataclass(frozen=True)
class ExperimentPaths:
    """Directory layout of one experiment under a research root.

    Corrected experiments live under ``<research root>/experiments/<name>`` and
    never share a directory with legacy version-one artifacts.
    """

    research_root: Path
    name: str

    @property
    def root(self) -> Path:
        return self.research_root / "experiments" / self.name

    @property
    def lock(self) -> Path:
        return self.root / "manifest.lock.json"

    @property
    def evidence(self) -> Path:
        return self.root / "evidence"

    @property
    def population(self) -> Path:
        return self.root / "population"

    @property
    def resume(self) -> Path:
        return self.root / "zoo_resume"

    @property
    def budget_inspection(self) -> Path:
        return self.root / "budget_inspection"

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"

    def evidence_file(self, identifier: str) -> Path:
        suite, _ = split_task_id(identifier)
        return self.evidence / suite / f"{identifier}.npz"

    def sample_directory(self, sample: SampleIdentity) -> Path:
        return self.population / sample.relative_directory()

    def run_directory(self, identifier: str, seed: int) -> Path:
        suite, _ = split_task_id(identifier)
        return self.population / suite / identifier / f"seed_{seed}"

    def archive(self, fold: str) -> Path:
        return self.root / "alignment" / fold / "archive"

    def alignment_checkpoint(self, fold: str, seed: int, condition: str) -> Path:
        return self.root / "alignment" / fold / f"seed_{seed}" / condition

    def diagnostics(self, fold: str, seed: int, scope: str) -> Path:
        return self.root / "diagnostics" / scope / fold / f"seed_{seed}"

    def rollouts(self, scope: str, fold: str, seed: int) -> Path:
        return self.root / "rollouts" / scope / fold / f"seed_{seed}"

    @property
    def held_out_marker(self) -> Path:
        return self.root / HELD_OUT_MARKER


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(temporary, 0o666 & ~umask)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bind_experiment(
    manifest: ExperimentManifest, research_root: str | Path
) -> ExperimentPaths:
    """Bind a manifest to its experiment directory and enforce immutability.

    The first command records the manifest hash. Later commands must present
    the same manifest. A changed manifest is accepted (and logged) only while
    the experiment holds no held-out result; afterwards the configuration is
    frozen, so held-out numbers cannot steer a configuration change inside the
    experiment that reports them.
    """

    paths = ExperimentPaths(Path(research_root), manifest.name)
    digest = manifest.sha256()
    if not paths.lock.is_file():
        _atomic_json(paths.lock, {"manifest_sha256": digest, "manifest": manifest.to_dict(), "history": []})
        return paths
    recorded = json.loads(paths.lock.read_text(encoding="utf-8"))
    if recorded["manifest_sha256"] == digest:
        return paths
    if paths.held_out_marker.is_file():
        raise ManifestError(
            f"experiment {manifest.name!r} already holds held-out results produced with "
            f"manifest {recorded['manifest_sha256'][:12]}; a changed manifest "
            f"({digest[:12]}) must run as a new experiment name."
        )
    history = [*recorded.get("history", []), recorded["manifest_sha256"]]
    _atomic_json(
        paths.lock, {"manifest_sha256": digest, "manifest": manifest.to_dict(), "history": history}
    )
    return paths


def record_held_out_results(
    manifest: ExperimentManifest, paths: ExperimentPaths, *, artifact: str
) -> None:
    """Mark the experiment as having produced held-out results (freezes the manifest)."""

    manifest.require_frozen_budget("held-out evaluation")
    # Held-out jobs of different folds may finish together when several
    # runners share one experiment, so the update is made under a lock.
    with _directory_lock(paths.held_out_marker.with_name(HELD_OUT_MARKER + ".lock")):
        existing = (
            json.loads(paths.held_out_marker.read_text(encoding="utf-8"))
            if paths.held_out_marker.is_file()
            else {"manifest_sha256": manifest.sha256(), "artifacts": []}
        )
        if existing["manifest_sha256"] != manifest.sha256():
            raise ManifestError("held-out results exist for a different manifest")
        if artifact not in existing["artifacts"]:
            existing["artifacts"].append(artifact)
        _atomic_json(paths.held_out_marker, existing)


@contextmanager
def _directory_lock(path: Path, *, timeout: float = 600.0, stale_after: float = 120.0):
    """Short critical sections shared across processes and hosts (mkdir is atomic).

    A lock older than ``stale_after`` seconds belonged to a process that died
    inside the critical section, which takes milliseconds; it is broken.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > stale_after:
                    broken = path.with_name(f"{path.name}.stale.{os.getpid()}.{random.getrandbits(32)}")
                    path.rename(broken)  # of several processes, only one succeeds
                    broken.rmdir()
                    continue
            except OSError:
                continue  # released or broken meanwhile
            if time.monotonic() > deadline:
                raise ManifestError(f"timed out waiting for lock {path}")
            time.sleep(random.uniform(0.005, 0.05))
    try:
        yield
    finally:
        path.rmdir()
