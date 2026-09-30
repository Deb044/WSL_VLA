"""
models/contrastive_alignment.py

Modality-Specific InfoNCE Contrastive Aligner and joint AlignmentSystem.
Aligns multi-modal task evidence with adapter weight latents using learnable temperature scales.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from models.weight_autoencoder import (
    PackedWeightAutoencoder,
    require_jax,
)

try:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
except ImportError:
    nn = None
    jax = None
    jnp = None


if nn is not None:

    class EvidenceProjector(nn.Module):
        latent_dim: int = 128

        @nn.compact
        def __call__(self, evidence):
            x = nn.Dense(self.latent_dim * 2)(evidence)
            x = nn.gelu(x)
            x = nn.Dense(self.latent_dim)(x)
            return nn.LayerNorm()(x)

    class DeepSetsEvidenceEncoder(nn.Module):
        output_dim: int = 128
        hidden_dim: int = 256

        @nn.compact
        def __call__(self, features, mask):
            x = nn.Dense(self.hidden_dim)(features)
            x = nn.gelu(x)
            x = nn.Dense(self.hidden_dim)(x)
            mask_f = mask.astype(x.dtype)[..., None]
            pooled = jnp.sum(x * mask_f, axis=1) / jnp.maximum(
                jnp.sum(mask_f, axis=1), 1
            )
            pooled = nn.Dense(self.output_dim)(pooled)
            return nn.LayerNorm()(pooled)

    class AlignmentSystem(nn.Module):
        """Joint weight autoencoder and trainable task-evidence encoders."""
        token_width: int
        vision_feature_dim: int
        language_feature_dim: int
        action_feature_dim: int
        latent_dim: int = 128
        hidden_dim: int = 256
        layers: int = 3
        heads: int = 4
        max_tokens: int = 8192
        max_layers: int = 64

        def setup(self):
            self.weight_autoencoder = PackedWeightAutoencoder(
                token_width=self.token_width,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                layers=self.layers,
                heads=self.heads,
                max_tokens=self.max_tokens,
                max_layers=self.max_layers,
                name="weight_autoencoder",
            )
            self.vision_evidence = DeepSetsEvidenceEncoder(
                output_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                name="vision_evidence",
            )
            self.language_evidence = EvidenceProjector(
                latent_dim=self.latent_dim, name="language_evidence"
            )
            self.action_evidence = EvidenceProjector(
                latent_dim=self.latent_dim, name="action_evidence"
            )
            self.log_temperatures = {
                name: self.param(
                    f"log_temperature_{name}",
                    lambda key: jnp.asarray(np.log(0.07), dtype=jnp.float32),
                )
                for name in ("vision", "language", "action")
            }

        def __call__(
            self,
            tokens,
            token_mask,
            component_ids,
            layer_ids,
            vision_features,
            vision_mask,
            language_features,
            action_features,
            *,
            train: bool,
        ):
            reconstruction, latents = self.weight_autoencoder(
                tokens,
                token_mask,
                component_ids,
                layer_ids,
                train=train,
            )
            evidence = {
                "vision": self.vision_evidence(vision_features, vision_mask),
                "language": self.language_evidence(language_features),
                "action": self.action_evidence(action_features),
            }
            temperatures = {
                name: jnp.clip(jnp.exp(value), 0.01, 1.0)
                for name, value in self.log_temperatures.items()
            }
            return reconstruction, latents, evidence, temperatures

        def decode(self, latents, token_mask, component_ids, layer_ids, *, train: bool = False):
            return self.weight_autoencoder.decoder(
                latents, token_mask, component_ids, layer_ids, train=train
            )

else:
    class _MissingJax:
        def __init__(self, *args, **kwargs):
            require_jax()

    DeepSetsEvidenceEncoder = _MissingJax
    EvidenceProjector = _MissingJax
    AlignmentSystem = _MissingJax


def multi_positive_info_nce(
    token_latents,
    token_valid,
    evidence,
    task_labels,
    *,
    temperature,
):
    """Symmetric multi-positive contrastive InfoNCE loss."""
    require_jax()
    z = token_latents / jnp.maximum(jnp.linalg.norm(token_latents, axis=-1, keepdims=True), 1e-8)
    e = evidence / jnp.maximum(jnp.linalg.norm(evidence, axis=-1, keepdims=True), 1e-8)
    positives = task_labels[:, None] == task_labels[None, :]

    logits = jnp.einsum("btd,kd->btk", z, e) / temperature
    log_denominator = jax.scipy.special.logsumexp(logits, axis=-1)
    positive_logits = jnp.where(positives[:, None, :], logits, -jnp.inf)
    log_numerator = jax.scipy.special.logsumexp(positive_logits, axis=-1)
    z_to_e = -(log_numerator - log_denominator)
    z_to_e = jnp.sum(z_to_e * token_valid) / jnp.maximum(jnp.sum(token_valid), 1)

    mask_f = token_valid.astype(z.dtype)[..., None]
    pooled = jnp.sum(z * mask_f, axis=1) / jnp.maximum(jnp.sum(mask_f, axis=1), 1)
    pooled = pooled / jnp.maximum(jnp.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8)
    reverse_logits = e @ pooled.T / temperature
    reverse_denominator = jax.scipy.special.logsumexp(reverse_logits, axis=-1)
    reverse_positive = jnp.where(positives, reverse_logits, -jnp.inf)
    reverse_numerator = jax.scipy.special.logsumexp(reverse_positive, axis=-1)
    e_to_z = -jnp.mean(reverse_numerator - reverse_denominator)
    return 0.5 * (z_to_e + e_to_z)


from models.alignment_eval import (
    AlignmentGateResult,
    alignment_advantage_gate,
    multi_positive_retrieval_accuracy,
)


__all__ = [
    "AlignmentGateResult",
    "AlignmentSystem",
    "DeepSetsEvidenceEncoder",
    "EvidenceProjector",
    "alignment_advantage_gate",
    "multi_positive_info_nce",
    "multi_positive_retrieval_accuracy",
    "require_jax",
]
