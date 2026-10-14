"""Build and persist source-suite-only OOD retrieval baselines."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..contracts import AlignmentCheckpoint
from ..experiments.manifest import split_task_id
from .ood import MODALITIES, OODReferenceBank, ReferenceAdapter


@dataclass(frozen=True)
class OODReferenceBankArtifact:
    bank: OODReferenceBank
    base_sha256: str
    adapter_spec_sha256: str
    alignment_checkpoint_sha256: str
    source_sample_count: int


def build_reference_bank_from_checkpoint(
    checkpoint: AlignmentCheckpoint,
    *,
    held_out_suite: str,
    alignment_checkpoint_sha256: str,
) -> OODReferenceBankArtifact:
    """Reference bank of the fold's *training* tasks, taken from the checkpoint.

    A corrected alignment checkpoint already stores, for every training task,
    its evidence embeddings and its mean latent sequence over seeds and
    snapshots (computed from hash-verified archive samples). The bank reuses
    those; no population count is hard-coded and nothing from the held-out
    suite can enter, because it never entered the archive.
    """

    metadata = checkpoint.metadata
    if metadata.get("held_out_suite") != held_out_suite:
        raise ValueError("alignment checkpoint belongs to a different held-out suite")
    bank = metadata["task_bank"]
    task_ids = [str(value) for value in bank["task_ids"]]
    samples = []
    for position, row in enumerate(bank["train_rows"]):
        identifier = task_ids[int(row)]
        suite = split_task_id(identifier)[0]
        samples.append(
            ReferenceAdapter(
                task_id=identifier,
                suite=suite,
                evidence={name: np.asarray(bank[f"evidence_{name}"][int(row)]) for name in MODALITIES},
                latents={
                    name: np.asarray(bank[f"train_mean_latents_{name}"][position])
                    for name in MODALITIES
                },
            )
        )
    return OODReferenceBankArtifact(
        bank=OODReferenceBank(samples, held_out_suite=held_out_suite),
        base_sha256=str(metadata["shared"]["base_sha256"]),
        adapter_spec_sha256=str(metadata["adapter_spec_sha256"]),
        alignment_checkpoint_sha256=alignment_checkpoint_sha256,
        source_sample_count=int(metadata["split_counts"]["train"]),
    )


def build_reference_bank_from_population(
    checkpoint: AlignmentCheckpoint,
    sample_directories: Sequence[str | Path] | None = None,
    *,
    held_out_suite: str,
    alignment_checkpoint_sha256: str,
) -> OODReferenceBankArtifact:
    """Compatibility wrapper: the population is no longer re-encoded here."""

    del sample_directories
    return build_reference_bank_from_checkpoint(
        checkpoint,
        held_out_suite=held_out_suite,
        alignment_checkpoint_sha256=alignment_checkpoint_sha256,
    )


def save_reference_bank(artifact: OODReferenceBankArtifact, path: str | Path) -> None:
    samples = artifact.bank.samples
    metadata = {
        "schema_version": 2,
        "held_out_suite": artifact.bank.held_out_suite,
        "training_suites": list(artifact.bank.training_suites),
        "task_ids": [sample.task_id for sample in samples],
        "suites": [sample.suite for sample in samples],
        "base_sha256": artifact.base_sha256,
        "adapter_spec_sha256": artifact.adapter_spec_sha256,
        "alignment_checkpoint_sha256": artifact.alignment_checkpoint_sha256,
        "source_sample_count": artifact.source_sample_count,
        "reference_task_count": len(samples),
    }
    arrays: dict[str, Any] = {
        "metadata_json": np.frombuffer(
            json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
        )
    }
    for name in MODALITIES:
        arrays[f"evidence_{name}"] = np.stack(
            [np.asarray(sample.evidence[name]) for sample in samples]
        )
        arrays[f"latents_{name}"] = np.stack(
            [np.asarray(sample.latents[name]) for sample in samples]
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)


def load_reference_bank(path: str | Path) -> OODReferenceBankArtifact:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["metadata_json"]).decode("utf-8"))
        if metadata.get("schema_version") != 2:
            raise ValueError(
                "unsupported OOD reference-bank schema; rebuild it from a corrected "
                "alignment checkpoint"
            )
        task_ids = tuple(str(value) for value in metadata["task_ids"])
        suites = tuple(str(value) for value in metadata["suites"])
        samples = [
            ReferenceAdapter(
                task_id=task_id,
                suite=suite,
                evidence={name: np.array(archive[f"evidence_{name}"][index], copy=True) for name in MODALITIES},
                latents={name: np.array(archive[f"latents_{name}"][index], copy=True) for name in MODALITIES},
            )
            for index, (task_id, suite) in enumerate(zip(task_ids, suites))
        ]
        bank = OODReferenceBank(samples, held_out_suite=str(metadata["held_out_suite"]))
    artifact = OODReferenceBankArtifact(
        bank=bank,
        base_sha256=str(metadata["base_sha256"]),
        adapter_spec_sha256=str(metadata["adapter_spec_sha256"]),
        alignment_checkpoint_sha256=str(metadata["alignment_checkpoint_sha256"]),
        source_sample_count=int(metadata["source_sample_count"]),
    )
    if len(artifact.bank.samples) != int(metadata["reference_task_count"]):
        raise ValueError("OOD reference bank differs from its recorded task count")
    return artifact
