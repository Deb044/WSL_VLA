#!/usr/bin/env python3
"""Evaluate aligned versus reconstruction-only checkpoints on locked validation tasks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsl_vla.alignment.checkpoint import load_alignment_checkpoint
from wsl_vla.alignment.evaluation import (
    compare_alignment_checkpoints,
    modality_alignment_metrics,
)


REQUIRED_ARRAYS = {
    "tokens",
    "token_mask",
    "component_ids",
    "layer_ids",
    "vision_features",
    "vision_mask",
    "language_features",
    "action_features",
    "task_labels",
    "split",
}
COMPONENT_INDEX = {"vision": 0, "language": 1, "action": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_validation_archive(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("alignment archive and its manifest are required")
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS - set(archive.files)
        if missing:
            raise ValueError(f"alignment archive lacks arrays: {sorted(missing)}")
        split = np.asarray(archive["split"])
        if set(np.unique(split)) - {b"train", b"validation"}:
            raise ValueError("alignment archive has an unsupported split")
        indices = np.flatnonzero(split == b"validation")
        if not len(indices):
            raise ValueError("alignment archive contains no validation samples")
        data = {name: np.array(archive[name][indices], copy=True) for name in REQUIRED_ARRAYS - {"split"}}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported alignment archive manifest")
    expected = len(manifest.get("validation_task_indices", [])) * 3 * 3 * 3
    if len(indices) != expected:
        raise ValueError(
            f"validation archive must contain {expected} samples from locked tasks; found {len(indices)}"
        )
    return data, manifest


def validate_checkpoint_pair(aligned, reconstruction, archive_path: Path, manifest: dict) -> None:
    aligned_metadata = aligned.metadata
    reconstruction_metadata = reconstruction.metadata
    identity_fields = (
        "schema_version",
        "archive_sha256",
        "archive_manifest_sha256",
        "base_sha256",
        "adapter_spec_sha256",
        "held_out_suite",
        "validation_task_indices",
    )
    for field in identity_fields:
        if aligned_metadata.get(field) != reconstruction_metadata.get(field):
            raise ValueError(f"alignment checkpoints differ on {field}")
    if aligned_metadata["archive_sha256"] != sha256_file(archive_path):
        raise ValueError("checkpoint archive hash does not match the supplied archive")
    manifest_path = archive_path.with_suffix(".manifest.json")
    if aligned_metadata["archive_manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("checkpoint manifest hash does not match the supplied manifest")
    for field in ("base_sha256", "adapter_spec_sha256", "held_out_suite", "validation_task_indices"):
        if aligned_metadata.get(field) != manifest.get(field):
            raise ValueError(f"checkpoint and archive manifest differ on {field}")
    aligned_weight = float(aligned_metadata.get("model", {}).get("contrastive_weight", 0))
    reconstruction_weight = float(
        reconstruction_metadata.get("model", {}).get("contrastive_weight", -1)
    )
    if aligned_weight <= 0:
        raise ValueError("aligned checkpoint must have positive contrastive weight")
    if reconstruction_weight != 0:
        raise ValueError("reconstruction-only checkpoint must have zero contrastive weight")
    architecture_fields = (
        "token_width",
        "latent_dim",
        "hidden_dim",
        "layers",
        "heads",
        "max_tokens",
        "max_layers",
        "vision_feature_dim",
        "language_feature_dim",
        "action_feature_dim",
        "task_count",
    )
    for field in architecture_fields:
        if aligned_metadata["model"].get(field) != reconstruction_metadata["model"].get(field):
            raise ValueError(f"alignment architectures differ on {field}")
    for name in ("token_mask", "component_ids", "layer_ids"):
        if not np.array_equal(getattr(aligned, name), getattr(reconstruction, name)):
            raise ValueError(f"alignment checkpoint layouts differ on {name}")


def evaluate_checkpoint(checkpoint, data: dict[str, np.ndarray], *, batch_size: int):
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - WSL research environment
        raise RuntimeError("alignment evaluation requires JAX in the WSL research environment") from exc

    token_mask = data["token_mask"]
    component_ids = data["component_ids"]
    layer_ids = data["layer_ids"]
    if not np.all(token_mask == checkpoint.token_mask[None]):
        raise ValueError("validation token masks differ from the checkpoint layout")
    if not np.all(component_ids == checkpoint.component_ids[None]):
        raise ValueError("validation component layout differs from the checkpoint")
    if not np.all(layer_ids == checkpoint.layer_ids[None]):
        raise ValueError("validation layer layout differs from the checkpoint")

    latent_parts = []
    evidence_parts = {name: [] for name in COMPONENT_INDEX}
    for start in range(0, len(data["tokens"]), batch_size):
        selection = slice(start, start + batch_size)
        latents = checkpoint.model.apply(
            {"params": checkpoint.params},
            jnp.asarray(data["tokens"][selection]),
            jnp.asarray(token_mask[selection]),
            jnp.asarray(component_ids[selection]),
            jnp.asarray(layer_ids[selection]),
            train=False,
            method=checkpoint.model.encode_weights,
        )
        evidence = checkpoint.model.apply(
            {"params": checkpoint.params},
            jnp.asarray(data["vision_features"][selection]),
            jnp.asarray(data["vision_mask"][selection]),
            jnp.asarray(data["language_features"][selection]),
            jnp.asarray(data["action_features"][selection]),
            method=checkpoint.model.encode_evidence,
        )
        latent_parts.append(np.asarray(jax.device_get(latents)))
        for name in COMPONENT_INDEX:
            evidence_parts[name].append(np.asarray(jax.device_get(evidence[name])))
    latents = np.concatenate(latent_parts)
    evidence = {name: np.concatenate(parts) for name, parts in evidence_parts.items()}
    token_valid = np.any(token_mask, axis=-1)
    metrics = {}
    for name, component_index in COMPONENT_INDEX.items():
        positions = np.flatnonzero(
            (checkpoint.component_ids == component_index)
            & np.any(checkpoint.token_mask, axis=-1)
        )
        targets = latents[:, positions]
        valid = token_valid[:, positions]
        prediction = checkpoint.mappers[name].predict(evidence[name])
        metrics[name] = modality_alignment_metrics(
            weight_token_embeddings=targets,
            token_valid=valid,
            evidence_embeddings=evidence[name],
            task_labels=data["task_labels"],
            mapper_prediction=prediction,
        )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--aligned-checkpoint", required=True)
    parser.add_argument("--reconstruction-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    archive_path = Path(args.archive)
    data, manifest = load_validation_archive(archive_path)
    aligned = load_alignment_checkpoint(args.aligned_checkpoint)
    reconstruction = load_alignment_checkpoint(args.reconstruction_checkpoint)
    validate_checkpoint_pair(aligned, reconstruction, archive_path, manifest)
    aligned_metrics = evaluate_checkpoint(aligned, data, batch_size=args.batch_size)
    reconstruction_metrics = evaluate_checkpoint(
        reconstruction, data, batch_size=args.batch_size
    )
    comparison = compare_alignment_checkpoints(
        aligned=aligned_metrics,
        reconstruction_only=reconstruction_metrics,
        enforce=False,
    )
    payload = {
        "schema_version": 1,
        "passed": comparison.passed,
        "held_out_suite": manifest["held_out_suite"],
        "validation_task_indices": manifest["validation_task_indices"],
        "sample_count": len(data["tokens"]),
        "task_count": int(np.unique(data["task_labels"]).size),
        "archive_sha256": sha256_file(archive_path),
        "aligned_checkpoint": str(Path(args.aligned_checkpoint).resolve()),
        "reconstruction_checkpoint": str(Path(args.reconstruction_checkpoint).resolve()),
        "aligned": {name: asdict(value) for name, value in comparison.aligned.items()},
        "reconstruction_only": {
            name: asdict(value) for name, value in comparison.reconstruction_only.items()
        },
        "macro": asdict(comparison.gate),
        "decision_rule": "aligned macro retrieval must be strictly higher and macro mapper MSE strictly lower",
    }
    atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if comparison.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
