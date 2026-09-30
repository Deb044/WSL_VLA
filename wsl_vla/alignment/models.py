"""Optional JAX/Flax implementation for aligned weight-space learning.

Importing this module is safe without research dependencies. Constructing a
module or calling an algorithm raises a targeted dependency error instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

try:  # pragma: no cover - exercised in the Linux research environment
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - host-dependent
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


if nn is not None:  # pragma: no branch

    class ModalityResidualAdapter(nn.Module):
        """Zero-initialized low-rank residual for one token modality."""

        rank: int
        alpha: float

        @nn.compact
        def __call__(self, inputs):
            down = nn.Dense(
                self.rank,
                use_bias=False,
                kernel_init=nn.initializers.normal(stddev=0.02),
                name="down",
            )(inputs)
            up = nn.Dense(
                inputs.shape[-1],
                use_bias=False,
                kernel_init=nn.initializers.zeros,
                name="up",
            )(down)
            return up * (self.alpha / self.rank)


    class PackedWeightEncoder(nn.Module):
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


    class EvidenceProjector(nn.Module):
        latent_dim: int = 128

        @nn.compact
        def __call__(self, evidence):
            x = nn.Dense(self.latent_dim * 2)(evidence)
            x = nn.gelu(x)
            x = nn.Dense(self.latent_dim)(x)
            return nn.LayerNorm()(x)


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
        task_count: int = 1
        initial_temperature: float = 0.07

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
                    lambda key: jnp.asarray(np.log(self.initial_temperature), dtype=jnp.float32),
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
            task_logits = {
                name: nn.Dense(self.task_count, name=f"{name}_task_classifier")(embedding)
                for name, embedding in evidence.items()
            }
            temperatures = {
                name: jnp.clip(jnp.exp(value), 0.01, 1.0)
                for name, value in self.log_temperatures.items()
            }
            return reconstruction, latents, evidence, temperatures, task_logits

        def decode(self, latents, token_mask, component_ids, layer_ids, *, train: bool = False):
            return self.weight_autoencoder.decoder(
                latents, token_mask, component_ids, layer_ids, train=train
            )

        def encode_weights(
            self, tokens, token_mask, component_ids, layer_ids, *, train: bool = False
        ):
            return self.weight_autoencoder.encoder(
                tokens, token_mask, component_ids, layer_ids, train=train
            )

        def encode_evidence(
            self, vision_features, vision_mask, language_features, action_features
        ):
            return {
                "vision": self.vision_evidence(vision_features, vision_mask),
                "language": self.language_evidence(language_features),
                "action": self.action_evidence(action_features),
            }


else:

    class _MissingJax:
        def __init__(self, *args, **kwargs):
            require_jax()

    ModalityResidualAdapter = _MissingJax
    PackedWeightAutoencoder = _MissingJax
    PackedWeightEncoder = _MissingJax
    PackedWeightDecoder = _MissingJax
    DeepSetsEvidenceEncoder = _MissingJax
    EvidenceProjector = _MissingJax
    AlignmentSystem = _MissingJax


def masked_reconstruction_loss(reconstruction, target, mask):
    require_jax()
    weights = mask.astype(reconstruction.dtype)
    numerator = jnp.sum(jnp.square(reconstruction - target) * weights)
    return numerator / jnp.maximum(jnp.sum(weights), 1)


def multi_positive_info_nce(
    token_latents,
    token_valid,
    evidence,
    task_labels,
    *,
    temperature,
):
    """Symmetric contrastive loss without false negatives across checkpoints."""

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


@dataclass(frozen=True)
class EmpiricalShell:
    center: Any
    radius: Any


def estimate_empirical_shell(latents, valid_mask) -> EmpiricalShell:
    require_jax()
    weights = valid_mask.astype(latents.dtype)[..., None]
    count = jnp.maximum(jnp.sum(weights, axis=0), 1)
    center = jnp.sum(latents * weights, axis=0) / count
    distances = jnp.linalg.norm(latents - center[None], axis=-1)
    radius = jnp.sum(distances * valid_mask, axis=0) / jnp.maximum(
        jnp.sum(valid_mask, axis=0), 1
    )
    return EmpiricalShell(center=center, radius=jnp.maximum(radius, 1e-6))


def project_to_empirical_shell(latents, shell: EmpiricalShell):
    require_jax()
    offset = latents - shell.center
    norm = jnp.linalg.norm(offset, axis=-1, keepdims=True)
    fallback = jnp.zeros_like(offset).at[..., 0].set(1)
    direction = jnp.where(norm > 1e-8, offset / jnp.maximum(norm, 1e-8), fallback)
    return shell.center + shell.radius[..., None] * direction


def differential_local_penalty(
    current: Mapping[str, Any],
    previous: Mapping[str, Any],
    gammas: Mapping[str, float],
):
    require_jax()
    expected = {"vision", "language", "action"}
    if set(current) != expected or set(previous) != expected or set(gammas) != expected:
        raise ValueError("current, previous, and gammas must contain all three modalities")
    return sum(
        gammas[name] * jnp.sum(jnp.square(current[name] - previous[name]))
        for name in sorted(expected)
    )


def assert_task_loss_gradients(
    latents: Mapping[str, Any],
    *,
    task_loss: Callable[[Mapping[str, Any]], Any],
    minimum_norm: float = 1e-12,
):
    """Hard gate: task loss must reach every modality latent."""

    require_jax()
    gradients = jax.grad(task_loss)(latents)
    for name in ("vision", "language", "action"):
        if name not in gradients:
            raise AssertionError(f"task loss has no {name} latent gradient")
        value = gradients[name]
        if not bool(jnp.all(jnp.isfinite(value))):
            raise AssertionError(f"task loss has non-finite {name} gradients")
        if float(jnp.linalg.norm(value)) <= minimum_norm:
            raise AssertionError(f"task loss is detached from {name} latents")
    return gradients


def refine_latents(
    initial: Mapping[str, Any],
    *,
    task_loss: Callable[[Mapping[str, Any], Any], Any],
    batches: Any,
    gammas: Mapping[str, float],
    steps: int,
    learning_rate: float,
):
    """Differentiate task loss through decoder and Octo adapter application."""

    require_jax()
    if steps <= 0:
        raise ValueError("refinement requires positive steps")
    if not callable(batches) and not batches:
        raise ValueError("refinement requires at least one batch")
    current = {name: jnp.asarray(value) for name, value in initial.items()}
    previous = {name: jax.lax.stop_gradient(value) for name, value in current.items()}

    def objective(values, batch):
        task = task_loss(values, batch)
        regularizer = differential_local_penalty(values, previous, gammas)
        return task + regularizer, {"task_loss": task, "regularizer": regularizer}

    def stream():
        if callable(batches):
            while True:
                produced = False
                for batch in batches():
                    produced = True
                    yield batch
                if not produced:
                    raise ValueError("batch factory produced no refinement batches")
        else:
            while True:
                yield from batches

    history = []
    iterator = stream()
    for _ in range(steps):
        (loss, auxiliary), gradients = jax.value_and_grad(objective, has_aux=True)(
            current, next(iterator)
        )
        current = jax.tree_util.tree_map(
            lambda value, gradient: value - learning_rate * gradient,
            current,
            gradients,
        )
        history.append({"loss": loss, **auxiliary})
    return current, history
