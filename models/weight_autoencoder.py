"""
models/weight_autoencoder.py

Component-Factorized Masked Weight Autoencoder (g_phi, h_psi) for effective adapter updates.
Compacts high-dimensional weight matrices into low-dimensional latent vectors.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
except ImportError as exc:
    nn = None
    jax = None
    jnp = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def require_jax() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "The official research path requires Python 3.10/3.11 with the "
            "'research' extra installed under Linux/WSL2."
        ) from _IMPORT_ERROR


if nn is not None:

    class PackedWeightEncoder(nn.Module):
        """Encodes packed effective update tokens into low-dimensional latent space."""
        token_width: int
        latent_dim: int
        hidden_dim: int
        layers: int
        heads: int
        max_tokens: int
        max_layers: int
        component_count: int

        @nn.compact
        def __call__(self, tokens, token_mask, component_ids, layer_ids, *, train: bool):
            token_valid = jnp.any(token_mask, axis=-1)
            x = nn.Dense(self.hidden_dim, name="input_projection")(tokens)
            position = self.param(
                "position_embedding",
                nn.initializers.normal(stddev=0.02),
                (self.max_tokens, self.hidden_dim),
            )[: tokens.shape[1]]
            component = nn.Embed(
                self.component_count, self.hidden_dim, name="component_embedding"
            )(component_ids)
            layer = nn.Embed(self.max_layers, self.hidden_dim, name="layer_embedding")(
                layer_ids
            )
            x = x + position[None] + component + layer
            attention_mask = nn.make_attention_mask(token_valid, token_valid)
            for index in range(self.layers):
                residual = x
                x = nn.LayerNorm(name=f"encoder_norm_attn_{index}")(x)
                x = nn.SelfAttention(
                    num_heads=self.heads,
                    dropout_rate=0.0,
                    name=f"encoder_attention_{index}",
                )(x, mask=attention_mask, deterministic=not train)
                x = residual + x
                residual = x
                x = nn.LayerNorm(name=f"encoder_norm_mlp_{index}")(x)
                x = nn.Dense(self.hidden_dim * 4, name=f"encoder_mlp_in_{index}")(x)
                x = nn.gelu(x)
                x = nn.Dense(self.hidden_dim, name=f"encoder_mlp_out_{index}")(x)
                x = residual + x
            return nn.Dense(self.latent_dim, name="to_latent")(x) * token_valid[..., None]

    class PackedWeightDecoder(nn.Module):
        """Decodes latent vectors back to effective update tokens."""
        token_width: int
        latent_dim: int
        hidden_dim: int

        @nn.compact
        def __call__(self, latent, token_mask, component_ids, layer_ids, *, train: bool):
            del component_ids, layer_ids, train
            token_valid = jnp.any(token_mask, axis=-1)
            x = nn.Dense(self.hidden_dim, name="from_latent")(latent)
            x = nn.gelu(x)
            x = nn.LayerNorm(name="decoder_norm")(x)
            output = nn.Dense(self.token_width, name="output_projection")(x)
            return output * token_mask * token_valid[..., None]

    class PackedWeightAutoencoder(nn.Module):
        """Masked shared token autoencoder with a separately callable decoder."""
        token_width: int
        latent_dim: int = 128
        hidden_dim: int = 256
        layers: int = 3
        heads: int = 4
        max_tokens: int = 8192
        max_layers: int = 64
        component_count: int = 3

        def setup(self):
            self.encoder = PackedWeightEncoder(
                token_width=self.token_width,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                layers=self.layers,
                heads=self.heads,
                max_tokens=self.max_tokens,
                max_layers=self.max_layers,
                component_count=self.component_count,
                name="encoder",
            )
            self.decoder = PackedWeightDecoder(
                token_width=self.token_width,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                name="decoder",
            )

        def __call__(self, tokens, token_mask, component_ids, layer_ids, *, train: bool):
            latent = self.encoder(
                tokens, token_mask, component_ids, layer_ids, train=train
            )
            reconstruction = self.decoder(
                latent, token_mask, component_ids, layer_ids, train=train
            )
            return reconstruction, latent

else:
    class _MissingJax:
        def __init__(self, *args, **kwargs):
            require_jax()

    PackedWeightAutoencoder = _MissingJax
    PackedWeightEncoder = _MissingJax
    PackedWeightDecoder = _MissingJax


def masked_reconstruction_loss(reconstruction, target, mask):
    """Normalized mean squared error over unmasked effective update entries."""
    require_jax()
    weights = mask.astype(reconstruction.dtype)
    numerator = jnp.sum(jnp.square(reconstruction - target) * weights)
    return numerator / jnp.maximum(jnp.sum(weights), 1)


__all__ = [
    "PackedWeightAutoencoder",
    "PackedWeightEncoder",
    "PackedWeightDecoder",
    "masked_reconstruction_loss",
    "require_jax",
]
