"""Manifest-indexed, memory-mapped alignment archives.

One archive holds one fold's alignment *training* and *selection* population:

* weight tokens live in fixed-size ``tokens_NNN.npy`` shards that are opened
  memory-mapped, so a 270-sample fold never has to fit in host memory;
* task evidence is stored **once per task**, not once per adapter snapshot;
* the evidence standardization and the per-entry RMS scales are fitted on the
  fold's training tasks only and stored beside the data they describe.

Held-out-suite adapters are never written into an archive. Every adapter and
evidence hash is re-checked while the archive is built, so a stale or edited
zoo sample cannot enter alignment even if no separate validator was run.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from ..adapters.packing import TokenLayout, load_packed_adapter
from ..contracts import (
    ALIGNMENT_ARCHIVE_SCHEMA_VERSION,
    COMPONENT_NAMES,
    POPULATION_SAMPLE_SCHEMA_VERSION,
    AdapterSpec,
    LegacyArtifactError,
    canonical_sha256,
    require_schema,
)
from ..experiments.manifest import (
    SELECTION_INTEGRITY,
    ExperimentManifest,
    ExperimentPaths,
    FoldSpec,
    SampleIdentity,
    split_task_id,
)
from ..experiments.provenance import sha256_file
from .evidence import EvidenceStandardization, fit_evidence_standardization
from .evidence_io import load_task_evidence


ARCHIVE_MANIFEST = "archive_manifest.json"
SHARED_IDENTITY_FIELDS = (
    "base_sha256",
    "base_revision",
    "architecture_sha256",
    "preprocessing_sha256",
    "action_convention_sha256",
    "adapter_spec_sha256",
    "layout_sha256",
    "zoo_sha256",
    "manifest_name",
)


class ArchiveIntegrityError(ValueError):
    """A zoo sample, evidence file, or archive violates an integrity or leakage rule."""


def _load_sample_metadata(directory: Path) -> dict[str, Any]:
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file() or not (directory / "adapter.npz").is_file():
        raise ArchiveIntegrityError(f"incomplete zoo sample: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") == 1:
        raise LegacyArtifactError(
            f"{directory} is a version-one zoo sample; corrected archives accept only "
            "regenerated (schema 2) samples"
        )
    require_schema(
        metadata.get("schema_version"), POPULATION_SAMPLE_SCHEMA_VERSION, "zoo sample metadata"
    )
    if metadata.get("synthetic", True):
        raise ArchiveIntegrityError(f"synthetic zoo sample is forbidden: {directory}")
    return metadata


def population_sample_directories(population_root: Path) -> list[Path]:
    return sorted(path.parent for path in Path(population_root).rglob("adapter.npz"))


def validate_population(
    manifest: ExperimentManifest,
    paths: ExperimentPaths,
    *,
    samples: Sequence[SampleIdentity] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Re-check identities and hashes of zoo samples against the manifest.

    Counts come from the manifest. Returns shared identities plus how many
    samples are merely structurally valid versus behaviourally evaluated.
    """

    expected = tuple(manifest.samples() if samples is None else samples)
    known = {sample.sample_id for sample in manifest.samples()}
    observed_directories = population_sample_directories(paths.population)
    unexpected = []
    for directory in observed_directories:
        metadata = _load_sample_metadata(directory)
        if metadata.get("sample_id") not in known:
            unexpected.append(str(directory))
    if unexpected:
        raise ArchiveIntegrityError(
            f"population holds samples the manifest does not define: {unexpected[:5]}"
        )
    shared: dict[str, set[str]] = {name: set() for name in SHARED_IDENTITY_FIELDS}
    missing = []
    behavioural = 0
    task_provenance: dict[str, set[tuple[str, str]]] = {}
    records = []
    for sample in expected:
        directory = paths.sample_directory(sample)
        if not (directory / "adapter.npz").is_file():
            missing.append(sample.sample_id)
            continue
        metadata = _load_sample_metadata(directory)
        for key, value in (
            ("sample_id", sample.sample_id),
            ("task_id", sample.task_id),
            ("seed", sample.seed),
            ("snapshot_index", sample.snapshot_index),
            ("optimizer_updates", sample.optimizer_updates),
            ("purpose", "zoo"),
        ):
            if metadata.get(key) != value:
                raise ArchiveIntegrityError(
                    f"{directory}: metadata {key}={metadata.get(key)!r}, manifest expects {value!r}"
                )
        if metadata["sample_exposure"] != manifest.zoo.sample_exposure(sample.optimizer_updates):
            raise ArchiveIntegrityError(f"{directory}: sample exposure differs from the manifest")
        if sha256_file(directory / "adapter.npz") != metadata.get("adapter_sha256"):
            raise ArchiveIntegrityError(f"adapter hash mismatch: {directory}")
        if not metadata.get("validation", {}).get("structurally_valid", False):
            raise ArchiveIntegrityError(f"zoo sample is not structurally valid: {directory}")
        behavioural += bool(metadata["validation"].get("behaviourally_evaluated", False)) or (
            directory / "behaviour.json"
        ).is_file()
        for name in SHARED_IDENTITY_FIELDS:
            shared[name].add(str(metadata.get(name, "")))
        task_provenance.setdefault(sample.task_id, set()).add(
            (str(metadata.get("dataset_sha256", "")), str(metadata.get("train_episode_ids_sha256", "")))
        )
        records.append((sample, directory, metadata))
    if missing and require_complete:
        raise ArchiveIntegrityError(
            f"population lacks {len(missing)} of {len(expected)} manifest samples, "
            f"e.g. {missing[:5]}"
        )
    for name, values in shared.items():
        if records and (len(values) != 1 or "" in values):
            raise ArchiveIntegrityError(f"zoo samples disagree on {name}: {sorted(values)[:4]}")
    if records and next(iter(shared["zoo_sha256"])) != manifest.zoo_sha256():
        raise ArchiveIntegrityError(
            "zoo samples were trained with a different budget or episode split than the manifest"
        )
    inconsistent = sorted(task for task, values in task_provenance.items() if len(values) != 1)
    if inconsistent:
        raise ArchiveIntegrityError(
            f"runs of one task used different data or training episodes: {inconsistent}"
        )
    return {
        "expected_samples": len(expected),
        "present_samples": len(records),
        "missing_samples": missing,
        "structurally_valid_samples": len(records),
        "behaviourally_evaluated_samples": int(behavioural),
        "shared": {name: next(iter(values)) for name, values in shared.items() if values},
        "task_provenance": {
            task: {"dataset_sha256": pair[0], "train_episode_ids_sha256": pair[1]}
            for task, values in task_provenance.items()
            for pair in values
        },
        "records": records,
    }


def _load_fold_evidence(
    manifest: ExperimentManifest,
    paths: ExperimentPaths,
    tasks: Sequence[str],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = {}
    hashes = {}
    shared = None
    for identifier in tasks:
        path = paths.evidence_file(identifier)
        sidecar = path.with_suffix(".manifest.json")
        if not path.is_file() or not sidecar.is_file():
            raise ArchiveIntegrityError(f"task evidence or its manifest is missing: {path}")
        declared = json.loads(sidecar.read_text(encoding="utf-8"))
        digest = sha256_file(path)
        if digest != declared.get("evidence_sha256"):
            raise ArchiveIntegrityError(f"evidence hash mismatch: {path}")
        record = load_task_evidence(path)
        suite, _ = split_task_id(identifier)
        if record.task_id != identifier or record.suite != suite:
            raise ArchiveIntegrityError(f"evidence identity differs from its path: {path}")
        expected_settings = manifest.evidence_identity()
        if declared.get("evidence_identity") != expected_settings:
            raise ArchiveIntegrityError(
                f"{path} was extracted with different settings than the manifest declares"
            )
        provenance = population["task_provenance"][identifier]
        if record.dataset_sha256 != provenance["dataset_sha256"]:
            raise ArchiveIntegrityError(
                f"evidence and zoo runs of {identifier} used different demonstration files"
            )
        if canonical_sha256(list(record.episode_ids)) != provenance["train_episode_ids_sha256"]:
            raise ArchiveIntegrityError(
                f"evidence of {identifier} was not built from the zoo's training episodes"
            )
        identity = (
            record.base_sha256,
            record.architecture_sha256,
            record.preprocessing_sha256,
            record.action_convention_sha256,
            tuple(sorted(record.feature_extractor_sha256.items())),
        )
        if shared is None:
            shared = identity
        if identity != shared:
            raise ArchiveIntegrityError("task evidence files disagree on extractor identity")
        for name in ("base_sha256", "architecture_sha256", "preprocessing_sha256",
                     "action_convention_sha256"):
            if getattr(record, name) != population["shared"][name]:
                raise ArchiveIntegrityError(
                    f"evidence of {identifier} and the zoo disagree on {name}"
                )
        evidence[identifier] = record
        hashes[identifier] = digest
    return {"records": evidence, "sha256": hashes, "feature_extractor_sha256": dict(shared[4])}


def _pad_task_arrays(records: Sequence[Any]) -> dict[str, np.ndarray]:
    """Stack per-task evidence with zero padding and explicit masks."""

    count = len(records)
    frames = max(item.vision_features.shape[0] for item in records)
    tokens = max(item.language_features.shape[0] for item in records)
    episodes = max(item.action_chunk_features.shape[0] for item in records)
    chunks = max(item.action_chunk_features.shape[1] for item in records)
    widths = {
        name: {getattr(item, field).shape[-1] for item in records}
        for name, field in (
            ("vision", "vision_features"),
            ("language", "language_features"),
            ("action", "action_chunk_features"),
        )
    }
    if any(len(value) != 1 for value in widths.values()):
        raise ArchiveIntegrityError("tasks disagree on an evidence feature width")
    arrays = {
        "evidence_vision": np.zeros((count, frames, widths["vision"].pop()), np.float32),
        "evidence_vision_mask": np.zeros((count, frames), bool),
        "evidence_language": np.zeros((count, tokens, widths["language"].pop()), np.float32),
        "evidence_language_mask": np.zeros((count, tokens), bool),
        "evidence_action": np.zeros((count, episodes, chunks, widths["action"].pop()), np.float32),
        "evidence_action_mask": np.zeros((count, episodes, chunks), bool),
    }
    for row, item in enumerate(records):
        f = item.vision_features.shape[0]
        arrays["evidence_vision"][row, :f] = item.vision_features
        arrays["evidence_vision_mask"][row, :f] = True
        length = item.language_features.shape[0]
        mask = np.asarray(item.language_mask, bool)
        arrays["evidence_language"][row, :length] = np.where(
            mask[:, None], item.language_features, 0.0
        )
        arrays["evidence_language_mask"][row, :length] = mask
        e, k = item.action_chunk_mask.shape
        chunk_mask = np.asarray(item.action_chunk_mask, bool)
        arrays["evidence_action"][row, :e, :k] = np.where(
            chunk_mask[..., None], item.action_chunk_features, 0.0
        )
        arrays["evidence_action_mask"][row, :e, :k] = chunk_mask
    return arrays


def build_alignment_archive(
    manifest: ExperimentManifest,
    paths: ExperimentPaths,
    fold: FoldSpec | str,
    output: str | Path,
    *,
    samples_per_shard: int = 32,
) -> dict[str, Any]:
    """Assemble one fold's train/selection archive from verified zoo samples.

    The archive is built in a sibling ``.partial`` directory and renamed into
    place only once complete, so ``output`` either holds a finished archive or
    nothing. Building again from the same inputs returns the existing archive's
    manifest; an archive built from different inputs is never overwritten.
    """

    spec_fold = manifest.fold(fold) if isinstance(fold, str) else fold
    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive")
    destination = Path(output)
    integrity_only = spec_fold.selection == SELECTION_INTEGRITY
    task_split = {identifier: "train" for identifier in spec_fold.train_tasks}
    if not integrity_only:
        for identifier in spec_fold.validation_tasks:
            if identifier in task_split:
                raise ArchiveIntegrityError(f"{identifier} is in two alignment splits")
            task_split[identifier] = "validation"
    for identifier in task_split:
        if spec_fold.held_out_suite and split_task_id(identifier)[0] == spec_fold.held_out_suite:
            raise ArchiveIntegrityError(
                f"held-out suite task {identifier} cannot enter alignment training or selection"
            )
        if identifier in spec_fold.held_out_tasks:
            raise ArchiveIntegrityError(f"held-out task {identifier} leaked into the archive")
    tasks = tuple(task_split)
    samples = manifest.samples(tasks)
    population = validate_population(manifest, paths, samples=samples)
    evidence = _load_fold_evidence(manifest, paths, tasks, population)
    train_tasks = tuple(spec_fold.train_tasks)

    request = {
        "experiment_manifest_sha256": manifest.sha256(),
        "fold": spec_fold.to_dict(),
        "samples_per_shard": int(samples_per_shard),
        "samples": {
            sample.sample_id: metadata["adapter_sha256"]
            for sample, _, metadata in population["records"]
        },
        "evidence": {identifier: evidence["sha256"][identifier] for identifier in tasks},
    }
    if (destination / ARCHIVE_MANIFEST).is_file():
        recorded = json.loads((destination / ARCHIVE_MANIFEST).read_text(encoding="utf-8"))
        if _archive_inputs(recorded) == request:
            return recorded
        raise FileExistsError(
            f"{destination} holds an archive built from different inputs; archives are never "
            "overwritten. Use a new experiment manifest name or research root."
        )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"{destination} is not empty but holds no archive manifest; move it aside first"
        )
    staging = destination.with_name(f".{destination.name}.partial")
    if staging.exists():
        # Left by an interrupted build; nothing can depend on it.
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        archive_manifest = _write_archive(
            staging,
            manifest=manifest,
            spec_fold=spec_fold,
            tasks=tasks,
            task_split=task_split,
            train_tasks=train_tasks,
            population=population,
            evidence=evidence,
            integrity_only=integrity_only,
            samples_per_shard=samples_per_shard,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if destination.exists():
        destination.rmdir()
    os.replace(staging, destination)
    return archive_manifest


def _archive_inputs(archive_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The inputs an archive manifest was built from, in request form."""

    return {
        "experiment_manifest_sha256": archive_manifest.get("experiment_manifest_sha256"),
        "fold": archive_manifest.get("fold"),
        "samples_per_shard": archive_manifest.get("samples_per_shard"),
        "samples": {
            item["sample_id"]: item["adapter_sha256"]
            for item in archive_manifest.get("samples", ())
        },
        "evidence": {
            item["task_id"]: item["evidence_sha256"] for item in archive_manifest.get("tasks", ())
        },
    }


def _write_archive(
    destination: Path,
    *,
    manifest: ExperimentManifest,
    spec_fold: FoldSpec,
    tasks: tuple[str, ...],
    task_split: Mapping[str, str],
    train_tasks: tuple[str, ...],
    population: Mapping[str, Any],
    evidence: Mapping[str, Any],
    integrity_only: bool,
    samples_per_shard: int,
) -> dict[str, Any]:
    standardization = fit_evidence_standardization(evidence["records"], train_tasks)
    spec: AdapterSpec | None = None
    layout: TokenLayout | None = None
    entry_square = None
    entry_count = None
    token_shards: list[np.memmap] = []
    index = []
    task_row = {identifier: row for row, identifier in enumerate(tasks)}
    train_class = {identifier: row for row, identifier in enumerate(train_tasks)}
    for position, (sample, directory, metadata) in enumerate(population["records"]):
        packed = load_packed_adapter(directory / "adapter.npz")
        if (
            packed.metadata.task_id != sample.task_id
            or packed.metadata.seed != sample.seed
            or packed.metadata.optimizer_updates != sample.optimizer_updates
            or packed.metadata.snapshot_index != sample.snapshot_index
        ):
            raise ArchiveIntegrityError(
                f"packed adapter identity differs from its metadata: {directory}"
            )
        if packed.spec.sha256() != metadata["adapter_spec_sha256"]:
            raise ArchiveIntegrityError(f"packed AdapterSpec differs from metadata: {directory}")
        if spec is None:
            spec, layout = packed.spec, packed.layout
            entry_square = np.zeros(len(spec.entries), np.float64)
            entry_count = np.zeros(len(spec.entries), np.float64)
            valid_per_entry = np.bincount(
                layout.entry_ids, weights=layout.token_mask.sum(axis=1), minlength=len(spec.entries)
            )
        elif packed.spec != spec or packed.layout.sha256() != layout.sha256():
            raise ArchiveIntegrityError("zoo samples have incompatible adapter layouts")
        shard, row = divmod(position, samples_per_shard)
        if row == 0:
            rows = min(samples_per_shard, len(population["records"]) - position)
            token_shards.append(
                np.lib.format.open_memmap(
                    destination / f"tokens_{shard:03d}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(rows, layout.token_count, layout.token_width),
                )
            )
        token_shards[shard][row] = packed.tokens
        split = task_split[sample.task_id]
        if split == "train":
            # Training-only statistics for the per-entry RMS target scale.
            entry_square += np.bincount(
                layout.entry_ids,
                weights=np.square(packed.tokens.astype(np.float64)).sum(axis=1),
                minlength=len(spec.entries),
            )
            entry_count += valid_per_entry
        index.append(
            {
                "sample_id": sample.sample_id,
                "run_id": sample.run_id,
                "task_id": sample.task_id,
                "seed": sample.seed,
                "snapshot_index": sample.snapshot_index,
                "optimizer_updates": sample.optimizer_updates,
                "split": split,
                "task_row": task_row[sample.task_id],
                "shard": shard,
                "row": row,
                "adapter_sha256": metadata["adapter_sha256"],
            }
        )
    for shard in token_shards:
        shard.flush()
    del token_shards
    if spec is None or layout is None:
        raise ArchiveIntegrityError("the fold selects no zoo samples")
    arrays = _pad_task_arrays([evidence["records"][identifier] for identifier in tasks])
    for name, value in arrays.items():
        np.save(destination / f"{name}.npy", value)
    np.savez_compressed(destination / "layout.npz", **layout.to_arrays())
    np.savez(destination / "standardization.npz", **standardization.to_arrays())
    np.save(destination / "entry_rms.npy", np.sqrt(entry_square / np.maximum(entry_count, 1.0)))

    counts = {
        split: sum(item["split"] == split for item in index) for split in ("train", "validation")
    }
    expected_counts = {
        "train": len(manifest.fold_samples(spec_fold, "train")),
        "validation": 0 if integrity_only else len(manifest.fold_samples(spec_fold, "validation")),
    }
    if counts != expected_counts:
        raise ArchiveIntegrityError(
            f"archive sample counts {counts} differ from the manifest {expected_counts}"
        )
    files = {
        path.name: sha256_file(path)
        for path in sorted(destination.iterdir())
        if path.name != ARCHIVE_MANIFEST
    }
    archive_manifest = {
        "schema_version": ALIGNMENT_ARCHIVE_SCHEMA_VERSION,
        "experiment": manifest.name,
        "experiment_manifest_sha256": manifest.sha256(),
        "profile": manifest.profile,
        "integrity_only": integrity_only,
        "fold": spec_fold.to_dict(),
        "selection": spec_fold.selection,
        "shared": population["shared"],
        "feature_extractor_sha256": evidence["feature_extractor_sha256"],
        "adapter_spec": spec.to_dict(),
        "sequence_window": layout.sequence_window,
        "sample_count": len(index),
        "split_counts": counts,
        "samples_per_shard": samples_per_shard,
        "samples": index,
        "tasks": [
            {
                "task_id": identifier,
                "row": task_row[identifier],
                "split": task_split[identifier],
                "train_class": train_class.get(identifier, -1),
                "evidence_sha256": evidence["sha256"][identifier],
                "dataset_sha256": population["task_provenance"][identifier]["dataset_sha256"],
            }
            for identifier in tasks
        ],
        "standardization": {"fit_split": "train", "fit_task_ids": list(train_tasks)},
        "entry_rms": {"fit_split": "train", "fit_samples": counts["train"]},
        "excluded_held_out_tasks": list(spec_fold.held_out_tasks),
        "files": files,
    }
    (destination / ARCHIVE_MANIFEST).write_text(
        json.dumps(archive_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return archive_manifest


@dataclass
class AlignmentArchive:
    """Read-only, memory-mapped view of one fold archive."""

    directory: Path
    manifest: Mapping[str, Any]
    spec: AdapterSpec
    layout: TokenLayout
    standardization: EvidenceStandardization
    entry_rms: np.ndarray
    task_ids: tuple[str, ...]
    task_split: tuple[str, ...]
    task_train_class: np.ndarray
    sample_task_rows: np.ndarray
    sample_split: tuple[str, ...]
    sample_ids: tuple[str, ...]
    sample_run_ids: tuple[str, ...]
    evidence: Mapping[str, np.ndarray]
    _shards: Sequence[np.ndarray]
    _locations: np.ndarray

    @property
    def manifest_sha256(self) -> str:
        return sha256_file(self.directory / ARCHIVE_MANIFEST)

    @property
    def integrity_only(self) -> bool:
        return bool(self.manifest["integrity_only"])

    @property
    def train_task_ids(self) -> tuple[str, ...]:
        order = np.argsort(np.where(self.task_train_class < 0, np.iinfo(np.int32).max, self.task_train_class))
        return tuple(self.task_ids[row] for row in order if self.task_train_class[row] >= 0)

    def indices(self, split: str) -> np.ndarray:
        """Sample indices of a split; an integrity fold selects on its training samples."""

        if split not in {"train", "validation"}:
            raise ValueError("archive splits are train and validation")
        if split == "validation" and self.integrity_only:
            split = "train"
        return np.flatnonzero(np.asarray(self.sample_split) == split)

    def task_rows(self, split: str) -> np.ndarray:
        if split == "validation" and self.integrity_only:
            split = "train"
        return np.flatnonzero(np.asarray(self.task_split) == split)

    def tokens(self, sample_indices: Sequence[int]) -> np.ndarray:
        """Materialize ``[batch, tokens, width]`` weight tokens for the given samples."""

        selected = np.asarray(sample_indices, dtype=np.int64)
        out = np.empty((len(selected), self.layout.token_count, self.layout.token_width), np.float32)
        for position, sample in enumerate(selected):
            shard, row = self._locations[sample]
            out[position] = self._shards[shard][row]
        return out

    def task_evidence(self, task_rows: Sequence[int]) -> dict[str, np.ndarray]:
        rows = np.asarray(task_rows, dtype=np.int64)
        return {name: value[rows] for name, value in self.evidence.items()}

    def sample_evidence(self, sample_indices: Sequence[int]) -> dict[str, np.ndarray]:
        return self.task_evidence(self.sample_task_rows[np.asarray(sample_indices, np.int64)])


def standardized_evidence_arrays(
    raw: Mapping[str, np.ndarray], standardization: EvidenceStandardization
) -> dict[str, np.ndarray]:
    """Apply a fitted standardization to padded arrays, re-zeroing padded entries."""

    result = {}
    for name in COMPONENT_NAMES:
        values = standardization.modalities[name].apply(raw[f"evidence_{name}"])
        mask = np.asarray(raw[f"evidence_{name}_mask"], bool)
        result[name] = np.where(mask[..., None], values, 0.0).astype(np.float32)
        result[f"{name}_mask"] = mask
    return result


def open_alignment_archive(
    directory: str | Path, *, verify_hashes: bool = True
) -> AlignmentArchive:
    """Open an archive read-only, re-checking its index, hashes, and leakage rules."""

    root = Path(directory)
    manifest_path = root / ARCHIVE_MANIFEST
    if not manifest_path.is_file():
        if root.suffix == ".npz" or (root.parent / f"{root.name}.npz").is_file():
            raise LegacyArtifactError(
                f"{root} is a version-one single-file archive; rebuild it from a regenerated zoo"
            )
        raise FileNotFoundError(f"alignment archive manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_schema(manifest.get("schema_version"), ALIGNMENT_ARCHIVE_SCHEMA_VERSION, "alignment archive")
    for name, digest in manifest["files"].items():
        path = root / name
        if not path.is_file():
            raise ArchiveIntegrityError(f"archive file is missing: {path}")
        if verify_hashes and sha256_file(path) != digest:
            raise ArchiveIntegrityError(f"archive file hash mismatch: {path}")
    spec = AdapterSpec.from_dict(manifest["adapter_spec"])
    if spec.sha256() != manifest["shared"]["adapter_spec_sha256"]:
        raise ArchiveIntegrityError("archive AdapterSpec differs from its recorded digest")
    with np.load(root / "layout.npz", allow_pickle=False) as arrays:
        layout = TokenLayout.from_arrays(arrays)
    if layout.sha256() != manifest["shared"]["layout_sha256"]:
        raise ArchiveIntegrityError("archive token layout differs from the zoo's layout")
    with np.load(root / "standardization.npz", allow_pickle=False) as arrays:
        standardization = EvidenceStandardization.from_arrays(arrays)
    tasks = sorted(manifest["tasks"], key=lambda item: item["row"])
    held_out_suite = manifest["fold"].get("held_out_suite")
    train_ids = {item["task_id"] for item in tasks if item["split"] == "train"}
    for item in tasks:
        if held_out_suite and split_task_id(item["task_id"])[0] == held_out_suite:
            raise ArchiveIntegrityError(
                f"archive contains held-out suite task {item['task_id']}"
            )
    if set(standardization.fit_task_ids) != train_ids:
        raise ArchiveIntegrityError(
            "evidence standardization was not fitted on exactly the training tasks"
        )
    samples = manifest["samples"]
    task_split = {item["task_id"]: item["split"] for item in tasks}
    for item in samples:
        if task_split.get(item["task_id"]) != item["split"]:
            raise ArchiveIntegrityError(
                f"sample {item['sample_id']} is not in its task's alignment split"
            )
    shard_count = max(item["shard"] for item in samples) + 1
    shards = [np.load(root / f"tokens_{shard:03d}.npy", mmap_mode="r") for shard in range(shard_count)]
    raw = {
        name: np.load(root / f"{name}.npy", allow_pickle=False)
        for name in (
            "evidence_vision",
            "evidence_vision_mask",
            "evidence_language",
            "evidence_language_mask",
            "evidence_action",
            "evidence_action_mask",
        )
    }
    return AlignmentArchive(
        directory=root,
        manifest=manifest,
        spec=spec,
        layout=layout,
        standardization=standardization,
        entry_rms=np.load(root / "entry_rms.npy", allow_pickle=False),
        task_ids=tuple(item["task_id"] for item in tasks),
        task_split=tuple(item["split"] for item in tasks),
        task_train_class=np.asarray([item["train_class"] for item in tasks], dtype=np.int32),
        sample_task_rows=np.asarray([item["task_row"] for item in samples], dtype=np.int64),
        sample_split=tuple(item["split"] for item in samples),
        sample_ids=tuple(item["sample_id"] for item in samples),
        sample_run_ids=tuple(item["run_id"] for item in samples),
        evidence=standardized_evidence_arrays(raw, standardization),
        _shards=shards,
        _locations=np.asarray([(item["shard"], item["row"]) for item in samples], dtype=np.int64),
    )
