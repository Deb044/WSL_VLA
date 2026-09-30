"""Build and persist source-suite-only OOD retrieval baselines."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..adapters.packing import load_packed_adapter
from ..alignment.checkpoint import encode_evidence_prompts, encode_weight_tokens
from ..alignment.evidence_io import load_task_evidence
from ..contracts import AlignmentCheckpoint
from .ood import MODALITIES, OODReferenceBank, aggregate_reference_adapters


@dataclass(frozen=True)
class OODReferenceBankArtifact:
    bank: OODReferenceBank
    base_sha256: str
    adapter_spec_sha256: str
    alignment_checkpoint_sha256: str
    source_sample_count: int


def build_reference_bank_from_population(
    checkpoint: AlignmentCheckpoint,
    sample_directories: Sequence[str | Path],
    *,
    held_out_suite: str,
    alignment_checkpoint_sha256: str,
) -> OODReferenceBankArtifact:
    """Encode the exact 270 non-held-out zoo samples without loading one giant archive."""

    expected_spec = str(checkpoint.metadata["adapter_spec_sha256"])
    expected_base = str(checkpoint.metadata["base_sha256"])
    if checkpoint.metadata.get("held_out_suite") != held_out_suite:
        raise ValueError("alignment checkpoint belongs to a different held-out suite")
    task_ids: list[str] = []
    suites: list[str] = []
    evidence_values = {name: [] for name in MODALITIES}
    latent_values = {name: [] for name in MODALITIES}
    identities = set()

    for raw_directory in sorted(Path(value) for value in sample_directories):
        metadata_path = raw_directory / "metadata.json"
        adapter_path = raw_directory / "adapter.npz"
        evidence_path = raw_directory / "evidence.npz"
        if not all(path.is_file() for path in (metadata_path, adapter_path, evidence_path)):
            raise FileNotFoundError(f"incomplete model-zoo sample: {raw_directory}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        suite = str(metadata.get("suite", ""))
        if suite == held_out_suite:
            continue
        if metadata.get("schema_version") != 1 or metadata.get("synthetic", True):
            raise ValueError(f"invalid research model-zoo sample: {raw_directory}")
        identity = (
            str(metadata.get("task_id", "")),
            int(metadata.get("seed", -1)),
            float(metadata.get("checkpoint_fraction", -1)),
        )
        if identity in identities:
            raise ValueError(f"duplicate OOD reference sample identity: {identity}")
        identities.add(identity)
        if metadata.get("base_sha256") != expected_base:
            raise ValueError("OOD population and alignment checkpoint use different bases")

        packed = load_packed_adapter(adapter_path)
        import hashlib

        spec_sha256 = hashlib.sha256(
            json.dumps(packed.spec.to_dict(), sort_keys=True).encode()
        ).hexdigest()
        if spec_sha256 != expected_spec:
            raise ValueError("OOD population AdapterSpec differs from alignment checkpoint")
        evidence = load_task_evidence(evidence_path)
        task_id = str(metadata["task_id"])
        if evidence.task_id != task_id or evidence.suite != suite:
            raise ValueError("OOD population evidence identity mismatch")
        if evidence.vision_mask is None:
            raise ValueError("OOD reference evidence requires a visual mask")

        token_count = checkpoint.token_mask.shape[0]
        if packed.tokens.shape[0] > token_count:
            raise ValueError("packed OOD sample exceeds alignment token layout")
        tokens = np.zeros((1, *checkpoint.token_mask.shape), dtype=np.float32)
        mask = np.zeros_like(tokens, dtype=bool)
        component_ids = np.asarray(checkpoint.component_ids, dtype=np.int32)[None]
        layer_ids = np.asarray(checkpoint.layer_ids, dtype=np.int32)[None]
        count = packed.tokens.shape[0]
        tokens[0, :count] = packed.tokens
        mask[0, :count] = packed.mask
        if not np.array_equal(mask[0], checkpoint.token_mask):
            raise ValueError("OOD sample token mask differs from alignment checkpoint")
        if not np.array_equal(packed.component_ids, checkpoint.component_ids[:count]):
            raise ValueError("OOD sample component layout differs from alignment checkpoint")
        if not np.array_equal(packed.layer_ids, checkpoint.layer_ids[:count]):
            raise ValueError("OOD sample layer layout differs from alignment checkpoint")

        encoded = np.asarray(
            encode_weight_tokens(
                checkpoint,
                tokens=tokens,
                token_mask=mask,
                component_ids=component_ids,
                layer_ids=layer_ids,
            )
        )[0]
        prompts = encode_evidence_prompts(
            checkpoint,
            vision_features=evidence.vision_features,
            vision_mask=evidence.vision_mask,
            language_features=evidence.language_features,
            action_features=evidence.action_statistics,
        )
        for name, component_index in zip(MODALITIES, range(3)):
            positions = np.flatnonzero(
                (checkpoint.component_ids == component_index)
                & np.any(checkpoint.token_mask, axis=-1)
            )
            latent_values[name].append(encoded[positions])
            evidence_values[name].append(np.asarray(prompts[name]))
        task_ids.append(task_id)
        suites.append(suite)

    if len(task_ids) != 270:
        raise ValueError(f"OOD reference fold requires 270 zoo samples, found {len(task_ids)}")
    counts = {task_id: task_ids.count(task_id) for task_id in set(task_ids)}
    if len(counts) != 30 or set(counts.values()) != {9}:
        raise ValueError("OOD reference fold requires 30 tasks with 9 checkpoints each")
    bank = aggregate_reference_adapters(
        task_ids=task_ids,
        suites=suites,
        evidence={name: np.stack(values) for name, values in evidence_values.items()},
        latents={name: np.stack(values) for name, values in latent_values.items()},
        held_out_suite=held_out_suite,
    )
    return OODReferenceBankArtifact(
        bank=bank,
        base_sha256=expected_base,
        adapter_spec_sha256=expected_spec,
        alignment_checkpoint_sha256=alignment_checkpoint_sha256,
        source_sample_count=len(task_ids),
    )


def save_reference_bank(artifact: OODReferenceBankArtifact, path: str | Path) -> None:
    samples = artifact.bank.samples
    metadata = {
        "schema_version": 1,
        "held_out_suite": artifact.bank.held_out_suite,
        "training_suites": list(artifact.bank.training_suites),
        "task_ids": [sample.task_id for sample in samples],
        "suites": [sample.suite for sample in samples],
        "base_sha256": artifact.base_sha256,
        "adapter_spec_sha256": artifact.adapter_spec_sha256,
        "alignment_checkpoint_sha256": artifact.alignment_checkpoint_sha256,
        "source_sample_count": artifact.source_sample_count,
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
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported OOD reference-bank schema")
        task_ids = tuple(str(value) for value in metadata["task_ids"])
        suites = tuple(str(value) for value in metadata["suites"])
        bank = aggregate_reference_adapters(
            task_ids=task_ids,
            suites=suites,
            evidence={name: np.asarray(archive[f"evidence_{name}"]) for name in MODALITIES},
            latents={name: np.asarray(archive[f"latents_{name}"]) for name in MODALITIES},
            held_out_suite=str(metadata["held_out_suite"]),
        )
    artifact = OODReferenceBankArtifact(
        bank=bank,
        base_sha256=str(metadata["base_sha256"]),
        adapter_spec_sha256=str(metadata["adapter_spec_sha256"]),
        alignment_checkpoint_sha256=str(metadata["alignment_checkpoint_sha256"]),
        source_sample_count=int(metadata["source_sample_count"]),
    )
    if artifact.source_sample_count != 270 or len(artifact.bank.samples) != 30:
        raise ValueError("OOD reference bank does not match the locked 30-task fold")
    return artifact
