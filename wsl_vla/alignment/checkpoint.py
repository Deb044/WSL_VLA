"""Save, reload, and apply a complete alignment checkpoint.

A checkpoint directory is self-contained: the fully resolved configuration, the
parameters, the token layout, the training-only RMS scales (inverted before any
decoded update reaches a policy), the fold's evidence standardization, the
training-task bank, the ridge mappers, and the empirical shells.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from ..adapters.latent import decoded_token_task_loss
from ..adapters.packing import TokenLayout, unpack_tokens
from ..adapters.rank import RankProjection, project_updates_to_rank
from ..contracts import (
    ALIGNMENT_CHECKPOINT_SCHEMA_VERSION,
    COMPONENT_NAMES,
    AdapterSpec,
    AlignmentCheckpoint,
    LegacyArtifactError,
    TaskEvidence,
    require_schema,
)
from ..experiments.provenance import sha256_array_tree, sha256_file
from .evidence import EvidenceStandardization
from .mapping import LinearRidgeMapper
from .models import (
    EmpiricalShell,
    assert_task_loss_gradients,
    project_to_empirical_shell,
    refine_latents,
)
from .windows import LayoutBinding


COMPONENT_INDEX = {name: index for index, name in enumerate(COMPONENT_NAMES)}
PARAMS_FILE = "alignment_params.msgpack"
REQUIRED_FILES = (
    PARAMS_FILE,
    "metadata.json",
    "layout.npz",
    "entry_scales.npy",
    "standardization.npz",
    "task_bank.npz",
    "mappers.npz",
    "shells.npz",
)
# Fields that must be identical between the two primary conditions.
PAIRING_FIELDS = (
    "archive_manifest_sha256",
    "experiment_manifest_sha256",
    "fold",
    "seed",
    "adapter_spec_sha256",
    "layout_sha256",
    "entry_scales_sha256",
    "standardization_sha256",
    "initial_params_sha256",
    "batch_schedule_sha256",
    "optimizer_exposure",
    "train_task_ids",
    "validation_task_ids",
)
CONFIG_FIELDS_ALLOWED_TO_DIFFER = ("condition", "effective_contrastive_weight", "sha256")


def _research_imports():
    try:
        import flax.serialization
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - WSL research environment
        raise RuntimeError("alignment checkpoint loading requires JAX/Flax") from exc
    return flax.serialization, jax, jnp


def _array_file_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    return sha256_array_tree({name: np.asarray(value) for name, value in arrays.items()})


def save_alignment_checkpoint(
    directory: str | Path,
    *,
    config: Mapping[str, Any],
    params: Any,
    archive: Any,
    entry_scales: np.ndarray,
    geometry: Mapping[str, Any],
    training: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a reloadable checkpoint directory and return its metadata.

    Files are written to a sibling ``.partial`` directory that is renamed into
    place once complete: ``directory`` holds a whole checkpoint or nothing.
    """

    serialization, jax, _ = _research_imports()
    destination = Path(directory)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite an alignment checkpoint: {destination}")
    root = destination.with_name(f".{destination.name}.partial")
    if root.exists():
        # Left by an interrupted save; nothing can depend on it.
        shutil.rmtree(root)
    root.mkdir(parents=True)
    host_params = jax.device_get(params)
    (root / PARAMS_FILE).write_bytes(serialization.to_bytes(host_params))
    np.savez_compressed(root / "layout.npz", **archive.layout.to_arrays())
    np.save(root / "entry_scales.npy", np.asarray(entry_scales, dtype=np.float32))
    standardization_arrays = archive.standardization.to_arrays()
    np.savez(root / "standardization.npz", **standardization_arrays)

    train_rows = archive.task_rows("train")
    validation_rows = archive.task_rows("validation")
    bank = {
        "task_ids": np.asarray(archive.task_ids, dtype=np.str_),
        "task_split": np.asarray(archive.task_split, dtype=np.str_),
        "train_rows": train_rows.astype(np.int64),
    }
    for name in COMPONENT_NAMES:
        positions = archive.layout.component_positions(name)
        bank[f"evidence_{name}"] = geometry["task_evidence_embeddings"][name].astype(np.float32)
        # Complete latent sequences of the *training* tasks only: the mapper's
        # targets, the nearest-neighbour gallery, and the training-mean baseline.
        bank[f"train_mean_latents_{name}"] = geometry["task_mean_latents"][train_rows][
            :, positions
        ].astype(np.float32)
    np.savez(root / "task_bank.npz", **bank)

    mapper_arrays = {}
    for name in COMPONENT_NAMES:
        mapper: LinearRidgeMapper = geometry["mappers"][name]
        mapper_arrays[f"{name}_projection"] = mapper.projection
        mapper_arrays[f"{name}_x_mean"] = mapper.x_mean
        mapper_arrays[f"{name}_y_mean"] = mapper.y_mean
        mapper_arrays[f"{name}_ridge_alpha"] = np.asarray(mapper.ridge_alpha)
    np.savez(root / "mappers.npz", **mapper_arrays)
    np.savez(
        root / "shells.npz",
        **{
            f"{name}_{field}": geometry["shells"][name][field]
            for name in COMPONENT_NAMES
            for field in ("center", "radius")
        },
    )
    metadata = {
        "schema_version": ALIGNMENT_CHECKPOINT_SCHEMA_VERSION,
        "config": dict(config),
        "condition": config["condition"],
        "seed": int(config["seed"]),
        "fold": archive.manifest["fold"]["name"],
        "held_out_suite": archive.manifest["fold"].get("held_out_suite"),
        "integrity_only": bool(archive.integrity_only),
        "experiment": archive.manifest["experiment"],
        "experiment_manifest_sha256": archive.manifest["experiment_manifest_sha256"],
        # Relative, so a copied experiment tree does not carry a host path.
        "archive": os.path.relpath(archive.directory.resolve(), destination.resolve()),
        "archive_manifest_sha256": archive.manifest_sha256,
        "shared": dict(archive.manifest["shared"]),
        "base_sha256": archive.manifest["shared"]["base_sha256"],
        "adapter_spec_sha256": archive.spec.sha256(),
        "adapter_spec": archive.spec.to_dict(),
        "layout_sha256": archive.layout.sha256(),
        "entry_scales_sha256": sha256_array_tree({"scales": np.asarray(entry_scales, np.float32)}),
        "entry_scale_fit": {"split": "train", "floor": float(config["rms_floor"])},
        "standardization_sha256": _array_file_sha256(standardization_arrays),
        "feature_extractor_sha256": dict(archive.manifest["feature_extractor_sha256"]),
        "split_counts": dict(archive.manifest["split_counts"]),
        "train_task_ids": list(archive.train_task_ids),
        "validation_task_ids": [archive.task_ids[row] for row in validation_rows],
        "params_sha256": sha256_array_tree(host_params),
        "initial_params_sha256": training["initial_params_sha256"],
        "batch_schedule_sha256": training["batch_schedule_sha256"],
        "optimizer_exposure": training["optimizer_exposure"],
        "selected_update": training["selected_update"],
        "best_validation_reconstruction": training["best_validation_reconstruction"],
        "history": training["history"],
        "resources": training["resources"],
        "mapper_selection": geometry["mapper_selection"],
        "shell_fit": geometry["shell_fit"],
        "provenance": dict(provenance),
    }
    metadata["files"] = {
        name: sha256_file(root / name) for name in REQUIRED_FILES if name != "metadata.json"
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if destination.exists():
        destination.rmdir()
    os.replace(root, destination)
    return metadata


def matching_checkpoint_metadata(
    directory: str | Path, *, config: Mapping[str, Any], archive: Any
) -> dict[str, Any] | None:
    """Metadata of a finished checkpoint in ``directory`` for this exact request.

    Returns ``None`` when the directory holds no checkpoint. A checkpoint
    trained with another resolved configuration or archive raises: checkpoints
    are never overwritten.
    """

    path = Path(directory) / "metadata.json"
    if not path.is_file():
        if Path(directory).exists() and any(Path(directory).iterdir()):
            raise FileExistsError(
                f"{directory} is not empty but holds no checkpoint metadata; move it aside first"
            )
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    requested = json.loads(json.dumps(dict(config), sort_keys=True))
    if (
        metadata.get("config") != requested
        or metadata.get("archive_manifest_sha256") != archive.manifest_sha256
    ):
        raise FileExistsError(
            f"{directory} holds a checkpoint trained with a different configuration or "
            "archive; checkpoints are never overwritten. Use a new experiment manifest "
            "name or research root."
        )
    return metadata


def load_alignment_checkpoint(directory: str | Path) -> AlignmentCheckpoint:
    """Reload a corrected (schema 2) checkpoint; version-one checkpoints are rejected."""

    serialization, _, jnp = _research_imports()
    from .training import build_alignment_model, initialize_alignment_params

    root = Path(directory)
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"alignment checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") == 1:
        raise LegacyArtifactError(
            f"{root} is a version-one alignment checkpoint (whole-sequence autoencoder, "
            "token-wise MLP decoder, post-block zoo). It is inspection-only."
        )
    require_schema(
        metadata.get("schema_version"), ALIGNMENT_CHECKPOINT_SCHEMA_VERSION, "alignment checkpoint"
    )
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"alignment checkpoint lacks files: {sorted(missing)}")
    for name, digest in metadata["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"alignment checkpoint file hash mismatch: {root / name}")
    config = metadata["config"]
    spec = AdapterSpec.from_dict(metadata["adapter_spec"])
    if spec.sha256() != metadata["adapter_spec_sha256"]:
        raise ValueError("alignment checkpoint AdapterSpec digest is inconsistent")
    with np.load(root / "layout.npz", allow_pickle=False) as arrays:
        layout = TokenLayout.from_arrays(arrays)
    if layout.sha256() != metadata["layout_sha256"]:
        raise ValueError("alignment checkpoint token layout digest is inconsistent")
    entry_scales = np.load(root / "entry_scales.npy", allow_pickle=False)
    with np.load(root / "standardization.npz", allow_pickle=False) as arrays:
        standardization = EvidenceStandardization.from_arrays(arrays)

    model = build_alignment_model(config)
    template = initialize_alignment_params(model, config)
    params = serialization.from_bytes(template, (root / PARAMS_FILE).read_bytes())
    if sha256_array_tree(params) != metadata["params_sha256"]:
        raise ValueError("alignment parameters differ from their recorded hash")

    with np.load(root / "task_bank.npz", allow_pickle=False) as arrays:
        bank = {name: np.array(arrays[name], copy=True) for name in arrays.files}
    train_rows = bank["train_rows"]
    mappers = {}
    with np.load(root / "mappers.npz", allow_pickle=False) as arrays:
        for name in COMPONENT_NAMES:
            y_mean = np.array(arrays[f"{name}_y_mean"], copy=True)
            mappers[name] = LinearRidgeMapper(
                projection=np.array(arrays[f"{name}_projection"], copy=True),
                x_mean=np.array(arrays[f"{name}_x_mean"], copy=True),
                y_mean=y_mean,
                # Centered training-task targets are rebuilt from the task bank
                # rather than stored twice.
                y_centered=bank[f"train_mean_latents_{name}"]
                - bank[f"train_mean_latents_{name}"].mean(axis=0),
                ridge_alpha=float(arrays[f"{name}_ridge_alpha"]),
            )
            if not np.allclose(
                bank[f"train_mean_latents_{name}"].mean(axis=0), y_mean, atol=1e-4, rtol=1e-4
            ):
                raise ValueError(f"{name} mapper intercept differs from the task bank mean")
    shells = {}
    with np.load(root / "shells.npz", allow_pickle=False) as arrays:
        for name in COMPONENT_NAMES:
            shells[name] = EmpiricalShell(
                center=jnp.asarray(arrays[f"{name}_center"]),
                radius=jnp.asarray(arrays[f"{name}_radius"]),
            )
    metadata = {**metadata, "task_bank": bank, "train_rows": train_rows, "directory": str(root)}
    checkpoint = AlignmentCheckpoint(
        params=params,
        model=model,
        config=config,
        adapter_spec=spec,
        layout=layout,
        entry_scales=entry_scales,
        standardization=standardization,
        mappers=mappers,
        shells=shells,
        metadata=metadata,
    )
    checkpoint.validate()
    return checkpoint


def assert_paired_checkpoints(aligned: AlignmentCheckpoint, control: AlignmentCheckpoint) -> None:
    """Integrity gate: the two conditions may differ only in the contrastive term."""

    if aligned.config["condition"] != "aligned":
        raise ValueError("the first checkpoint must be the aligned condition")
    if control.config["condition"] != "reconstruction_only":
        raise ValueError("the second checkpoint must be the reconstruction-only control")
    if float(aligned.config["effective_contrastive_weight"]) <= 0:
        raise ValueError("aligned checkpoint was trained without a contrastive term")
    if float(control.config["effective_contrastive_weight"]) != 0:
        raise ValueError("reconstruction-only control was trained with a contrastive term")
    for name in PAIRING_FIELDS:
        if aligned.metadata.get(name) != control.metadata.get(name):
            raise ValueError(f"paired alignment checkpoints differ on {name}")
    differing = sorted(
        key
        for key in set(aligned.config) | set(control.config)
        if key not in CONFIG_FIELDS_ALLOWED_TO_DIFFER
        and aligned.config.get(key) != control.config.get(key)
    )
    if differing:
        raise ValueError(
            f"paired alignment checkpoints differ in configuration beyond the contrastive "
            f"term: {differing}"
        )


# --------------------------------------------------------------------- inference
def _binding(checkpoint: AlignmentCheckpoint) -> LayoutBinding:
    cached = getattr(checkpoint, "_binding", None)
    if cached is None:
        cached = LayoutBinding(
            checkpoint.layout,
            checkpoint.entry_scales,
            window_chunk=int(checkpoint.config["window_chunk"]),
        )
        checkpoint._binding = cached
    return cached


def _compiled(checkpoint: AlignmentCheckpoint, name: str):
    _, jax, _ = _research_imports()
    cache = getattr(checkpoint, "_compiled", None)
    if cache is None:
        cache = {}
        checkpoint._compiled = cache
    if name not in cache:
        binding, model = _binding(checkpoint), checkpoint.model
        functions = {
            "encode": lambda params, tokens: binding.encode(model, params, tokens),
            "decode": lambda params, latents: binding.decode(model, params, latents),
            "evidence": lambda params, evidence: model.apply(
                {"params": params}, evidence, method=model.encode_evidence
            ),
        }
        # Parameters are an argument, not a closed-over constant, so compiling
        # does not constant-fold the whole model.
        compiled = jax.jit(functions[name])
        device_params = cache.setdefault("__params__", jax.device_put(checkpoint.params))
        cache[name] = lambda value: compiled(device_params, value)
    return cache[name]


def encode_weight_tokens(checkpoint: AlignmentCheckpoint, *, tokens: Any, **_ignored: Any):
    """Encode ``[batch, tokens, width]`` effective-update tokens to token latents.

    The layout arrays earlier callers passed (``token_mask``, ``component_ids``,
    ``layer_ids``) now live in the checkpoint and are accepted but ignored.
    """

    _, _, jnp = _research_imports()
    values = jnp.asarray(tokens, dtype=jnp.float32)
    if values.ndim != 3 or values.shape[1:] != checkpoint.token_mask.shape:
        raise ValueError("tokens must have shape [batch, tokens, width] of the checkpoint layout")
    return _compiled(checkpoint, "encode")(values)


def standardized_evidence(checkpoint: AlignmentCheckpoint, evidence: TaskEvidence) -> dict:
    """Apply the fold's training-only standardization to one task's raw evidence."""

    evidence.validate_for_research()
    expected = checkpoint.metadata["feature_extractor_sha256"]
    if dict(evidence.feature_extractor_sha256) != dict(expected):
        raise ValueError("evidence was extracted with a different frozen feature extractor")
    for name in ("base_sha256", "preprocessing_sha256", "action_convention_sha256"):
        if getattr(evidence, name) != checkpoint.metadata["shared"][name]:
            raise ValueError(f"evidence and alignment checkpoint differ on {name}")
    return checkpoint.standardization.apply(evidence)


def encode_evidence_prompts(
    checkpoint: AlignmentCheckpoint, evidence: TaskEvidence
) -> dict[str, np.ndarray]:
    """Trained modality evidence embeddings of one task (before the ridge mapper)."""

    _, _, jnp = _research_imports()
    arrays = standardized_evidence(checkpoint, evidence)
    batch = {name: jnp.asarray(value)[None] for name, value in arrays.items()}
    embeddings = _compiled(checkpoint, "evidence")(batch)
    return {name: np.asarray(embeddings[name][0]) for name in COMPONENT_NAMES}


def map_evidence_embeddings(
    checkpoint: AlignmentCheckpoint,
    embeddings: Mapping[str, Any],
    *,
    shell_projection: bool,
) -> dict[str, Any]:
    """Ridge-map evidence embeddings to complete modality latent sequences.

    Both the raw mapped latents and their empirical-shell projection are valid
    initializations; evaluations report both.
    """

    _, _, jnp = _research_imports()
    mapped = {}
    for name in COMPONENT_NAMES:
        value = jnp.asarray(
            checkpoint.mappers[name].predict(np.asarray(embeddings[name])), dtype=jnp.float32
        )
        if shell_projection:
            value = project_to_empirical_shell(value, checkpoint.shells[name])
        mapped[name] = value
    return mapped


def encode_and_map_evidence(
    checkpoint: AlignmentCheckpoint,
    evidence: TaskEvidence,
    *,
    shell_projection: bool = True,
) -> dict[str, Any]:
    """Evidence -> embeddings -> mapped modality latents (optionally on the shell)."""

    return map_evidence_embeddings(
        checkpoint, encode_evidence_prompts(checkpoint, evidence), shell_projection=shell_projection
    )


def compose_latent_sequence(checkpoint: AlignmentCheckpoint, latents: Mapping[str, Any]):
    """Assemble per-modality latents into the complete ``[tokens, latent]`` sequence."""

    _, _, jnp = _research_imports()
    if set(latents) != set(COMPONENT_NAMES):
        raise ValueError("latents must contain vision, language, and action")
    latent_dim = int(checkpoint.config["latent_dim"])
    sequence = jnp.zeros((checkpoint.layout.token_count, latent_dim), dtype=jnp.float32)
    for name in COMPONENT_NAMES:
        positions = checkpoint.layout.component_positions(name)
        if latents[name].shape != (len(positions), latent_dim):
            raise ValueError(f"{name} latent shape differs from checkpoint layout")
        sequence = sequence.at[positions].set(latents[name])
    return sequence


def split_latent_sequence(checkpoint: AlignmentCheckpoint, sequence: Any) -> dict[str, Any]:
    return {
        name: sequence[..., checkpoint.layout.component_positions(name), :]
        for name in COMPONENT_NAMES
    }


def decode_latent_sequence(checkpoint: AlignmentCheckpoint, sequence: Any):
    """``[tokens, latent]`` -> original-unit ``[tokens, width]`` effective-update tokens."""

    _, _, jnp = _research_imports()
    return _compiled(checkpoint, "decode")(jnp.asarray(sequence, dtype=jnp.float32)[None])[0]


def decode_alignment_latents(checkpoint: AlignmentCheckpoint, latents: Mapping[str, Any]):
    """Differentiable decode of per-modality latents (scales already inverted)."""

    binding = _binding(checkpoint)
    sequence = compose_latent_sequence(checkpoint, latents)
    return binding.decode(checkpoint.model, checkpoint.params, sequence[None])[0]


def decoded_updates(checkpoint: AlignmentCheckpoint, tokens: Any) -> dict[str, np.ndarray]:
    """Dense effective updates of decoded tokens (unrestricted rank; diagnostic)."""

    values = np.where(checkpoint.token_mask, np.asarray(tokens, dtype=np.float32), 0.0)
    return {
        path: np.array(update, copy=True)
        for path, update in unpack_tokens(checkpoint.adapter_spec, values).items()
    }


def rank_matched_updates(
    checkpoint: AlignmentCheckpoint, tokens: Any
) -> RankProjection:
    """Project decoded tokens to each entry's maximum rank for deployment."""

    return project_updates_to_rank(checkpoint.adapter_spec, decoded_updates(checkpoint, tokens))


def refine_alignment_latents(
    checkpoint: AlignmentCheckpoint,
    *,
    bundle: Any,
    adapter_spec: AdapterSpec,
    initial: Mapping[str, Any],
    batches: Any,
    gammas: Mapping[str, float],
    steps: int,
    learning_rate: float,
    validation_batches: Any = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
):
    """Refine mapped latents through decoder, Octo, and official diffusion loss."""

    def task_loss(values, batch_and_rng):
        batch, rng = batch_and_rng
        decoded = decode_alignment_latents(checkpoint, values)
        loss_and_metrics = decoded_token_task_loss(
            bundle,
            adapter_spec,
            decoded,
            checkpoint.token_mask,
            batch,
            rng,
            train=True,
        )
        return loss_and_metrics[0]

    def validation_loss(values, batch_and_rng):
        batch, rng = batch_and_rng
        decoded = decode_alignment_latents(checkpoint, values)
        return decoded_token_task_loss(
            bundle,
            adapter_spec,
            decoded,
            checkpoint.token_mask,
            batch,
            rng,
            train=False,
        )[0]

    # Publication runs fail before optimization if any modality is detached
    # from the real decoded-adapter -> Octo -> diffusion-loss computation.
    training_batches = batches
    if callable(batches):
        first_iterator = iter(batches())
        try:
            first_batch = next(first_iterator)
        except StopIteration as exc:
            raise ValueError("batch factory produced no refinement batches") from exc
        first_epoch_pending = True

        def training_batches():
            nonlocal first_epoch_pending
            if first_epoch_pending:
                first_epoch_pending = False
                yield first_batch
                yield from first_iterator
            else:
                yield from batches()
    else:
        first_batch = batches[0]
    assert_task_loss_gradients(
        initial,
        task_loss=lambda values: task_loss(values, first_batch),
    )
    return refine_latents(
        initial,
        task_loss=task_loss,
        batches=training_batches,
        gammas=gammas,
        steps=steps,
        learning_rate=learning_rate,
        validation_loss=validation_loss if validation_batches is not None else None,
        validation_batches=validation_batches,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )
