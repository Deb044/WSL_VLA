#!/usr/bin/env python3
"""Train the joint aligned weight autoencoder on a strict packed-zoo archive.

The input NPZ is deliberately explicit and contains no pickled objects. Required
arrays are documented in ``docs/RESEARCH_PIPELINE.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsl_vla.alignment.mapping import fit_linear_ridge_mapper
from wsl_vla.experiments.protocol import load_yaml, validate_research_config
from wsl_vla.experiments.sampling import task_balanced_index_batches


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


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS - set(archive.files)
        if missing:
            raise ValueError(f"packed zoo archive lacks arrays: {sorted(missing)}")
        data = {name: archive[name] for name in REQUIRED_ARRAYS}
    sample_count = data["tokens"].shape[0]
    if any(value.shape[0] != sample_count for value in data.values()):
        raise ValueError("all packed zoo arrays must share the sample dimension")
    if data["tokens"].shape != data["token_mask"].shape:
        raise ValueError("tokens and token_mask must have identical shapes")
    if set(np.unique(data["split"])) - {b"train", b"validation"}:
        raise ValueError("alignment archive may contain only train and validation samples")
    if not np.any(data["split"] == b"train") or not np.any(data["split"] == b"validation"):
        raise ValueError("both train and validation samples are required")
    return data


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/research/base.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks-per-batch", type=int)
    parser.add_argument("--samples-per-task", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--ridge-alphas", default="0.001,0.01,0.1,1,10")
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=None,
        help="Set to zero for the locked reconstruction-only ablation.",
    )
    parser.add_argument("--classification-weight", type=float)
    args = parser.parse_args()

    try:
        import flax.serialization
        from flax.training import train_state
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError as exc:
        raise RuntimeError("install requirements-research.txt in WSL2 before alignment training") from exc

    from wsl_vla.alignment.models import (
        AlignmentSystem,
        estimate_empirical_shell,
        masked_reconstruction_loss,
        multi_positive_info_nce,
    )

    config = load_yaml(args.config)
    validate_research_config(config)
    alignment_config = config["alignment"]
    tasks_per_batch = args.tasks_per_batch or int(alignment_config["tasks_per_batch"])
    samples_per_task = args.samples_per_task or int(alignment_config["samples_per_task"])
    gradient_accumulation = args.gradient_accumulation or int(
        alignment_config["gradient_accumulation_steps"]
    )
    patience = args.patience or int(alignment_config["early_stopping_patience"])
    contrastive_weight = (
        1.0 if args.contrastive_weight is None else args.contrastive_weight
    )
    classification_weight = (
        float(alignment_config["auxiliary_classification_weight"])
        if args.classification_weight is None
        else args.classification_weight
    )
    if contrastive_weight == 0:
        classification_weight = 0.0
    reconstruction_weight = float(alignment_config["reconstruction_weight"])
    modality_weights = {
        name: float(alignment_config["modality_alignment_weights"][name])
        for name in ("vision", "language", "action")
    }

    data = load_dataset(args.archive)
    archive_path = Path(args.archive)
    archive_manifest_path = archive_path.with_suffix(".manifest.json")
    if not archive_manifest_path.is_file():
        raise FileNotFoundError("alignment archive manifest is required")
    archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
    if archive_manifest.get("schema_version") != 1 or "adapter_spec" not in archive_manifest:
        raise ValueError("alignment archive manifest has no supported AdapterSpec")
    if gradient_accumulation <= 0 or patience <= 0:
        raise ValueError("gradient accumulation and patience must be positive")
    if contrastive_weight < 0 or classification_weight < 0 or reconstruction_weight <= 0:
        raise ValueError("alignment loss weights must be non-negative and reconstruction positive")
    train_indices = np.flatnonzero(data["split"] == b"train")
    validation_indices = np.flatnonzero(data["split"] == b"validation")
    token_width = data["tokens"].shape[-1]
    model = AlignmentSystem(
        token_width=token_width,
        vision_feature_dim=data["vision_features"].shape[-1],
        language_feature_dim=data["language_features"].shape[-1],
        action_feature_dim=data["action_features"].shape[-1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        max_tokens=data["tokens"].shape[1],
        max_layers=int(data["layer_ids"].max()) + 1,
        task_count=int(data["task_labels"].max()) + 1,
        initial_temperature=float(alignment_config["temperature"]),
    )

    def subset(indices):
        return {name: jnp.asarray(value[indices]) for name, value in data.items() if name != "split"}

    initial_indices = task_balanced_index_batches(
        train_indices,
        data["task_labels"],
        tasks_per_batch=tasks_per_batch,
        samples_per_task=samples_per_task,
        seed=args.seed,
        shuffle=False,
    )[0]
    initial_batch = subset(initial_indices)

    def apply_model(params, batch, *, train):
        return model.apply(
            {"params": params},
            batch["tokens"],
            batch["token_mask"],
            batch["component_ids"],
            batch["layer_ids"],
            batch["vision_features"],
            batch["vision_mask"],
            batch["language_features"],
            batch["action_features"],
            train=train,
        )

    def loss_fn(params, batch, *, train):
        reconstruction, latents, evidence, temperatures, task_logits = apply_model(
            params, batch, train=train
        )
        loss_reconstruction = masked_reconstruction_loss(
            reconstruction, batch["tokens"], batch["token_mask"]
        )
        token_valid = jnp.any(batch["token_mask"], axis=-1)
        losses = {}
        for component_index, name in enumerate(("vision", "language", "action")):
            modality_valid = token_valid & (batch["component_ids"] == component_index)
            losses[name] = multi_positive_info_nce(
                latents,
                modality_valid,
                evidence[name],
                batch["task_labels"],
                temperature=temperatures[name],
            )
        classification_losses = {
            name: jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(
                    task_logits[name], batch["task_labels"]
                )
            )
            for name in ("vision", "language", "action")
        }
        weighted_alignment = sum(modality_weights[name] * losses[name] for name in losses)
        total = (
            reconstruction_weight * loss_reconstruction
            + contrastive_weight * weighted_alignment
            # Validation task identities are intentionally unseen. Their
            # classifier rows receive no training signal, so auxiliary class
            # loss is a training regularizer rather than an early-stop metric.
            + (classification_weight if train else 0.0)
            * sum(classification_losses.values())
        )
        return total, {
            "reconstruction": loss_reconstruction,
            "vision": losses["vision"],
            "language": losses["language"],
            "action": losses["action"],
            "classification": sum(classification_losses.values()),
        }

    rng = jax.random.PRNGKey(args.seed)
    variables = model.init(
        rng,
        initial_batch["tokens"],
        initial_batch["token_mask"],
        initial_batch["component_ids"],
        initial_batch["layer_ids"],
        initial_batch["vision_features"],
        initial_batch["vision_mask"],
        initial_batch["language_features"],
        initial_batch["action_features"],
        train=True,
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.MultiSteps(
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(args.learning_rate, weight_decay=1e-4),
            ),
            every_k_schedule=gradient_accumulation,
        ),
    )

    @jax.jit
    def train_step(current, batch):
        (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            current.params, batch, train=True
        )
        return current.apply_gradients(grads=gradients), metrics

    @jax.jit
    def validation_loss(params, batch):
        return loss_fn(params, batch, train=False)

    best_params = state.params
    best_validation = float("inf")
    stale_epochs = 0
    history = []
    generator = np.random.default_rng(args.seed)

    def batches(indices, *, shuffle):
        batch_seed = int(generator.integers(0, 2**31 - 1)) if shuffle else args.seed
        for selected in task_balanced_index_batches(
            indices,
            data["task_labels"],
            tasks_per_batch=tasks_per_batch,
            samples_per_task=samples_per_task,
            seed=batch_seed,
            shuffle=shuffle,
        ):
            yield subset(selected)

    def plain_batches(indices):
        batch_size = tasks_per_batch * samples_per_task
        ordered = np.asarray(indices)
        for start in range(0, len(ordered), batch_size):
            yield subset(ordered[start : start + batch_size])

    def evaluate(params, indices):
        totals = []
        metric_rows = []
        weights = []
        for batch in batches(indices, shuffle=False):
            total, metrics = validation_loss(params, batch)
            weight = int(batch["tokens"].shape[0])
            totals.append(float(total) * weight)
            metric_rows.append({name: float(value) * weight for name, value in metrics.items()})
            weights.append(weight)
        denominator = sum(weights)
        return sum(totals) / denominator, {
            name: sum(row[name] for row in metric_rows) / denominator
            for name in metric_rows[0]
        }

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        train_rows = []
        train_weights = []
        for train_batch in batches(train_indices, shuffle=True):
            state, batch_metrics = train_step(state, train_batch)
            train_rows.append({name: float(value) for name, value in batch_metrics.items()})
            train_weights.append(int(train_batch["tokens"].shape[0]))
        train_metrics = {
            name: sum(row[name] * weight for row, weight in zip(train_rows, train_weights))
            / sum(train_weights)
            for name in train_rows[0]
        }
        validation_value, validation_metrics = evaluate(state.params, validation_indices)
        if validation_value < best_validation:
            best_validation = validation_value
            best_params = state.params
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": {
                    "total": validation_value,
                    **{name: float(value) for name, value in validation_metrics.items()},
                },
            }
        )
        print(
            f"epoch {epoch} train={train_metrics['reconstruction']:.6g} (recon) "
            f"validation={validation_value:.6g} best={best_validation:.6g} "
            f"stale={stale_epochs} seconds={time.monotonic() - epoch_start:.1f}",
            flush=True,
        )
        if stale_epochs >= patience:
            break

    def collect_outputs(indices):
        latent_parts = []
        evidence_parts = {name: [] for name in ("vision", "language", "action")}
        # Preserve a one-to-one, input-order correspondence for mapper targets.
        # Contrastive validation batches may wrap samples to preserve positives.
        for batch in plain_batches(indices):
            _, latents, evidence, _, _ = jax.device_get(
                apply_model(best_params, batch, train=False)
            )
            latent_parts.append(np.asarray(latents))
            for name in evidence_parts:
                evidence_parts[name].append(np.asarray(evidence[name]))
        return np.concatenate(latent_parts), {
            name: np.concatenate(parts) for name, parts in evidence_parts.items()
        }

    train_latents, train_evidence = collect_outputs(train_indices)
    validation_latents, validation_evidence = collect_outputs(validation_indices)
    train_component_ids = data["component_ids"][train_indices]
    validation_component_ids = data["component_ids"][validation_indices]
    token_valid_train = np.any(data["token_mask"][train_indices], axis=-1)

    mapper_payload = {}
    shell_payload = {}
    selected_alphas = {}
    alpha_candidates = [float(value) for value in args.ridge_alphas.split(",")]
    for component_index, name in enumerate(("vision", "language", "action")):
        positions = (train_component_ids[0] == component_index) & token_valid_train[0]
        if not np.all(train_component_ids[:, positions] == component_index):
            raise ValueError("component token layout differs across adapter samples")
        targets = np.asarray(train_latents)[:, positions]
        validation_targets = np.asarray(validation_latents)[:, positions]
        best_mapper = None
        best_mapper_error = float("inf")
        for alpha in alpha_candidates:
            mapper = fit_linear_ridge_mapper(np.asarray(train_evidence[name]), targets, ridge_alpha=alpha)
            prediction = mapper.predict(np.asarray(validation_evidence[name]))
            error = float(np.mean(np.square(prediction - validation_targets)))
            if error < best_mapper_error:
                best_mapper_error = error
                best_mapper = mapper
        selected_alphas[name] = best_mapper.ridge_alpha
        mapper_payload[f"{name}_coefficient"] = best_mapper.coefficient
        mapper_payload[f"{name}_intercept"] = best_mapper.intercept
        mapper_payload[f"{name}_latent_shape"] = np.asarray(best_mapper.latent_shape)
        shell = estimate_empirical_shell(
            jnp.asarray(targets), jnp.ones(targets.shape[:2], dtype=bool)
        )
        shell_payload[f"{name}_center"] = np.asarray(shell.center)
        shell_payload[f"{name}_radius"] = np.asarray(shell.radius)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output / "alignment_params.msgpack", flax.serialization.to_bytes(best_params))
    np.savez_compressed(output / "linear_mappers.npz", **mapper_payload)
    np.savez_compressed(output / "empirical_shells.npz", **shell_payload)
    np.savez_compressed(
        output / "token_layout.npz",
        token_mask=data["token_mask"][0],
        component_ids=data["component_ids"][0],
        layer_ids=data["layer_ids"][0],
    )
    metadata = {
        "schema_version": 1,
        "archive": str(Path(args.archive).resolve()),
        "archive_sha256": __import__("hashlib").sha256(Path(args.archive).read_bytes()).hexdigest(),
        "archive_manifest_sha256": __import__("hashlib").sha256(
            archive_manifest_path.read_bytes()
        ).hexdigest(),
        "base_sha256": archive_manifest["base_sha256"],
        "adapter_spec_sha256": archive_manifest["adapter_spec_sha256"],
        "adapter_spec": archive_manifest["adapter_spec"],
        "held_out_suite": archive_manifest["held_out_suite"],
        "validation_task_indices": archive_manifest["validation_task_indices"],
        "best_validation_loss": best_validation,
        "selected_ridge_alphas": selected_alphas,
        "model": {
            "token_width": token_width,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "heads": args.heads,
            "max_tokens": int(data["tokens"].shape[1]),
            "max_layers": int(data["layer_ids"].max()) + 1,
            "vision_feature_dim": int(data["vision_features"].shape[-1]),
            "language_feature_dim": int(data["language_features"].shape[-1]),
            "action_feature_dim": int(data["action_features"].shape[-1]),
            "task_count": int(data["task_labels"].max()) + 1,
            "initial_temperature": float(alignment_config["temperature"]),
            "tasks_per_batch": tasks_per_batch,
            "samples_per_task": samples_per_task,
            "effective_batch_size": tasks_per_batch * samples_per_task,
            "gradient_accumulation": gradient_accumulation,
            "patience": patience,
            "reconstruction_weight": reconstruction_weight,
            "contrastive_weight": contrastive_weight,
            "classification_weight": classification_weight,
            "modality_alignment_weights": modality_weights,
        },
        "history": history,
    }
    atomic_write(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode())
    print(json.dumps({"ok": True, "output": str(output), "best_validation_loss": best_validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
