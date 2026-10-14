"""JAX/Flax implementation of aligned weight-space learning.

Importing this module is safe without research dependencies. Constructing a
module or calling an algorithm raises a targeted dependency error instead.

Architecture
------------
Weight tokens of one modality form a stream that is cut into fixed sequence
windows (128 tokens by default, trailing padding masked). One encoder and one
decoder transformer - shared by the vision, language, and action streams -
process every window independently; ``wsl_vla.adapters.packing.from_windows``
restores the complete sequence for mapping and decoding. Each token is tagged
with five structural embeddings: component, transformer layer, parameter
entry, row, and column window.

Attention is implemented here from ``nn.Dense`` layers rather than through
``flax.linen`` attention modules, whose signatures and parameter names changed
between the two locked environments; alignment checkpoints therefore load in
both.

No module below has an architecture default: every size is a required field
that comes from the resolved, stored configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from ..contracts import COMPONENT_NAMES

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


STRUCTURAL_EMBEDDINGS = ("component", "layer", "entry", "row", "column_window")
MASKED_LOGIT = -1e9
MIN_TEMPERATURE = 0.01
MAX_TEMPERATURE = 1.0


def require_jax() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "The official research path requires one of the locked research "
            "environments (envs/scripts/setup_research.sh)."
        ) from _IMPORT_ERROR


if nn is not None:  # pragma: no branch

    class MultiHeadSelfAttention(nn.Module):
        """Masked self-attention over the tokens of one sequence window."""

        heads: int

        @nn.compact
        def __call__(self, x, valid):
            width = x.shape[-1]
            if width % self.heads:
                raise ValueError("transformer width must be divisible by the head count")
            head_dim = width // self.heads

            def split(values):
                return values.reshape(*values.shape[:-1], self.heads, head_dim)

            query = split(nn.Dense(width, name="query")(x))
            key = split(nn.Dense(width, name="key")(x))
            value = split(nn.Dense(width, name="value")(x))
            logits = jnp.einsum("...qhd,...khd->...hqk", query, key) / np.sqrt(head_dim)
            # Padding tokens are never attended to; a fully padded window keeps
            # finite (uniform) weights and is discarded by the caller.
            logits = jnp.where(valid[..., None, None, :], logits, MASKED_LOGIT)
            weights = jax.nn.softmax(logits, axis=-1)
            mixed = jnp.einsum("...hqk,...khd->...qhd", weights, value)
            return nn.Dense(width, name="out")(mixed.reshape(*x.shape))


    class WindowTransformer(nn.Module):
        """Pre-norm transformer stack applied independently to each window."""

        layers: int
        heads: int
        mlp_ratio: int

        @nn.compact
        def __call__(self, x, valid):
            width = x.shape[-1]
            for index in range(self.layers):
                attended = MultiHeadSelfAttention(self.heads, name=f"attention_{index}")(
                    nn.LayerNorm(name=f"norm_attention_{index}")(x), valid
                )
                x = x + attended
                hidden = nn.LayerNorm(name=f"norm_mlp_{index}")(x)
                hidden = nn.gelu(nn.Dense(width * self.mlp_ratio, name=f"mlp_in_{index}")(hidden))
                x = x + nn.Dense(width, name=f"mlp_out_{index}")(hidden)
            return nn.LayerNorm(name="norm_out")(x)


    class StructuralEmbedding(nn.Module):
        """Sum of component, layer, parameter-entry, row, and column-window embeddings."""

        width: int
        sizes: tuple[int, int, int, int, int]

        @nn.compact
        def __call__(self, identities):
            total = 0.0
            for name, size in zip(STRUCTURAL_EMBEDDINGS, self.sizes):
                total = total + nn.Embed(size, self.width, name=f"{name}_embedding")(
                    identities[name]
                )
            return total


    class WeightWindowEncoder(nn.Module):
        latent_dim: int
        width: int
        layers: int
        heads: int
        mlp_ratio: int
        embedding_sizes: tuple[int, int, int, int, int]

        @nn.compact
        def __call__(self, tokens, valid, identities):
            x = nn.Dense(self.width, name="input_projection")(tokens)
            x = x + StructuralEmbedding(self.width, self.embedding_sizes, name="structure")(identities)
            x = WindowTransformer(self.layers, self.heads, self.mlp_ratio, name="transformer")(x, valid)
            return nn.Dense(self.latent_dim, name="to_latent")(x) * valid[..., None]


    class WeightWindowDecoder(nn.Module):
        """Transformer decoder from latent tokens back to normalized weight tokens."""

        token_width: int
        width: int
        layers: int
        heads: int
        mlp_ratio: int
        embedding_sizes: tuple[int, int, int, int, int]

        @nn.compact
        def __call__(self, latents, valid, identities):
            x = nn.Dense(self.width, name="from_latent")(latents)
            x = x + StructuralEmbedding(self.width, self.embedding_sizes, name="structure")(identities)
            x = WindowTransformer(self.layers, self.heads, self.mlp_ratio, name="transformer")(x, valid)
            return nn.Dense(self.token_width, name="output_projection")(x) * valid[..., None]


    def _masked_mean(values, mask):
        weights = mask.astype(values.dtype)[..., None]
        return jnp.sum(values * weights, axis=-2) / jnp.maximum(jnp.sum(weights, axis=-2), 1.0)


    class SetEvidenceEncoder(nn.Module):
        """Trainable aggregation of a masked feature set (frames or tokens)."""

        latent_dim: int
        hidden_dim: int

        @nn.compact
        def __call__(self, features, mask):
            x = nn.gelu(nn.Dense(self.hidden_dim, name="element_in")(features))
            x = nn.Dense(self.hidden_dim, name="element_out")(x)
            pooled = _masked_mean(x, mask)
            pooled = nn.gelu(nn.Dense(self.hidden_dim, name="pooled_in")(pooled))
            return nn.LayerNorm(name="norm")(nn.Dense(self.latent_dim, name="pooled_out")(pooled))


    class ChunkEvidenceEncoder(nn.Module):
        """Action evidence: chunks are pooled per episode, then episodes per task."""

        latent_dim: int
        hidden_dim: int

        @nn.compact
        def __call__(self, features, mask):
            x = nn.gelu(nn.Dense(self.hidden_dim, name="chunk_in")(features))
            x = nn.Dense(self.hidden_dim, name="chunk_out")(x)
            episodes = _masked_mean(x, mask)
            episodes = nn.gelu(nn.Dense(self.hidden_dim, name="episode_in")(episodes))
            pooled = _masked_mean(episodes, jnp.any(mask, axis=-1))
            pooled = nn.gelu(nn.Dense(self.hidden_dim, name="pooled_in")(pooled))
            return nn.LayerNorm(name="norm")(nn.Dense(self.latent_dim, name="pooled_out")(pooled))


    class AlignmentSystem(nn.Module):
        """Shared weight autoencoder plus jointly trained evidence encoders."""

        token_width: int
        latent_dim: int
        width: int
        encoder_layers: int
        decoder_layers: int
        heads: int
        mlp_ratio: int
        evidence_hidden_dim: int
        embedding_sizes: tuple[int, int, int, int, int]
        train_task_count: int
        initial_temperature: float

        def setup(self):
            # Submodules are plain attributes: Flax names each one after its
            # attribute, so parameter paths are stable and explicit.
            self.encoder = WeightWindowEncoder(
                self.latent_dim, self.width, self.encoder_layers, self.heads,
                self.mlp_ratio, self.embedding_sizes,
            )
            self.decoder = WeightWindowDecoder(
                self.token_width, self.width, self.decoder_layers, self.heads,
                self.mlp_ratio, self.embedding_sizes,
            )
            self.vision_evidence = SetEvidenceEncoder(self.latent_dim, self.evidence_hidden_dim)
            self.language_evidence = SetEvidenceEncoder(self.latent_dim, self.evidence_hidden_dim)
            self.action_evidence = ChunkEvidenceEncoder(self.latent_dim, self.evidence_hidden_dim)
            # Auxiliary heads cover the fold's *training* task identities only.
            self.vision_task_classifier = nn.Dense(self.train_task_count)
            self.language_task_classifier = nn.Dense(self.train_task_count)
            self.action_task_classifier = nn.Dense(self.train_task_count)
            initial = float(np.log(self.initial_temperature))
            self.log_temperature = self.param(
                "log_temperature",
                lambda key, shape: jnp.full(shape, initial, dtype=jnp.float32),
                (len(COMPONENT_NAMES),),
            )

        def encode(self, tokens, valid, identities):
            return self.encoder(tokens, valid, identities)

        def decode(self, latents, valid, identities):
            return self.decoder(latents, valid, identities)

        def encode_evidence(self, evidence):
            return {
                "vision": self.vision_evidence(evidence["vision"], evidence["vision_mask"]),
                "language": self.language_evidence(evidence["language"], evidence["language_mask"]),
                "action": self.action_evidence(evidence["action"], evidence["action_mask"]),
            }

        def classify(self, embeddings):
            return {
                "vision": self.vision_task_classifier(embeddings["vision"]),
                "language": self.language_task_classifier(embeddings["language"]),
                "action": self.action_task_classifier(embeddings["action"]),
            }

        def temperatures(self):
            values = jnp.clip(jnp.exp(self.log_temperature), MIN_TEMPERATURE, MAX_TEMPERATURE)
            return {name: values[index] for index, name in enumerate(COMPONENT_NAMES)}

        def __call__(self, tokens, valid, identities, evidence):
            latents = self.encode(tokens, valid, identities)
            reconstruction = self.decode(latents, valid, identities)
            embeddings = self.encode_evidence(evidence)
            return reconstruction, latents, embeddings, self.classify(embeddings), self.temperatures()


else:

    class _MissingJax:
        def __init__(self, *args, **kwargs):
            require_jax()

    MultiHeadSelfAttention = _MissingJax
    WindowTransformer = _MissingJax
    StructuralEmbedding = _MissingJax
    WeightWindowEncoder = _MissingJax
    WeightWindowDecoder = _MissingJax
    SetEvidenceEncoder = _MissingJax
    ChunkEvidenceEncoder = _MissingJax
    AlignmentSystem = _MissingJax


def map_windows(function: Callable[..., Any], arrays: tuple, *, chunk: int):
    """Apply ``function`` to windows in rematerialized chunks.

    ``arrays`` share a leading window axis. With ``chunk`` windows per step the
    backward pass keeps one chunk of activations alive and recomputes the rest,
    so a batch of several thousand windows fits on a small accelerator. The
    result equals a single call on all windows.
    """

    require_jax()
    count = arrays[0].shape[0]
    if chunk <= 0 or chunk >= count:
        return function(*arrays)
    padding = (-count) % chunk
    stacked = tuple(
        jnp.pad(value, [(0, padding)] + [(0, 0)] * (value.ndim - 1)).reshape(
            -1, chunk, *value.shape[1:]
        )
        for value in arrays
    )
    output = jax.lax.map(jax.checkpoint(lambda items: function(*items)), stacked)
    return output.reshape(-1, *output.shape[2:])[:count]


def component_balanced_reconstruction_loss(reconstruction, target, token_mask, component_ids):
    """Masked squared error averaged within each component, then across components.

    ``reconstruction`` and ``target`` are ``[batch, tokens, width]`` in
    RMS-normalized units; ``token_mask`` is ``[tokens, width]`` and
    ``component_ids`` is ``[tokens]``. The action stream holds more than twice
    as many tokens as the others; equal component weights keep it from
    dominating the objective.
    """

    require_jax()
    mask = jnp.asarray(token_mask).astype(reconstruction.dtype)
    squared = jnp.square(reconstruction - target) * mask
    per_token = jnp.sum(squared, axis=(0, 2))
    valid_per_token = jnp.sum(mask, axis=-1) * reconstruction.shape[0]
    component_ids = jnp.asarray(component_ids)
    losses = []
    for index in range(len(COMPONENT_NAMES)):
        selected = (component_ids == index).astype(reconstruction.dtype)
        losses.append(
            jnp.sum(per_token * selected) / jnp.maximum(jnp.sum(valid_per_token * selected), 1.0)
        )
    per_component = jnp.stack(losses)
    return jnp.mean(per_component), {
        name: per_component[index] for index, name in enumerate(COMPONENT_NAMES)
    }


def bidirectional_token_contrastive_loss(
    token_latents,
    token_valid,
    evidence,
    task_labels,
    *,
    temperature,
):
    """Token-wise, bidirectional, same-task multi-positive InfoNCE.

    ``token_latents`` is ``[batch, tokens, dim]`` for one modality stream,
    ``token_valid`` is ``[batch, tokens]`` (False for padding), ``evidence`` is
    ``[batch, dim]`` with one evidence embedding per sample, and
    ``task_labels`` is ``[batch]``. With ``s[b, t, k] = cos(z[b, t], e[k]) / tau``
    and ``P(i)`` the samples sharing sample ``i``'s task:

    * weight -> evidence, for every valid token ``(b, t)``::

          -log( sum_{k in P(b)} exp s[b,t,k] / sum_k exp s[b,t,k] )

    * evidence -> weight, for every evidence ``k`` and token position ``t``,
      contrasting the batch's tokens at that same position::

          -log( sum_{b in P(k), valid} exp s[b,t,k] / sum_{b valid} exp s[b,t,k] )

    Padding is excluded as a query, as a candidate, and from both averages. The
    two directional means are averaged with equal weight.
    """

    require_jax()
    z = token_latents / jnp.maximum(jnp.linalg.norm(token_latents, axis=-1, keepdims=True), 1e-8)
    e = evidence / jnp.maximum(jnp.linalg.norm(evidence, axis=-1, keepdims=True), 1e-8)
    valid = jnp.asarray(token_valid, dtype=bool)
    positives = task_labels[:, None] == task_labels[None, :]
    logits = jnp.einsum("btd,kd->btk", z, e) / temperature

    log_all = jax.scipy.special.logsumexp(logits, axis=-1)
    log_positive = jax.scipy.special.logsumexp(
        jnp.where(positives[:, None, :], logits, MASKED_LOGIT), axis=-1
    )
    forward = -(log_positive - log_all)
    forward_weight = valid.astype(logits.dtype)
    weight_to_evidence = jnp.sum(forward * forward_weight) / jnp.maximum(
        jnp.sum(forward_weight), 1.0
    )

    candidate = valid[:, :, None]
    log_candidates = jax.scipy.special.logsumexp(
        jnp.where(candidate, logits, MASKED_LOGIT), axis=0
    )
    positive_candidate = candidate & positives[:, None, :]
    log_positive_candidates = jax.scipy.special.logsumexp(
        jnp.where(positive_candidate, logits, MASKED_LOGIT), axis=0
    )
    # A (position, evidence) pair counts only if a valid positive token exists.
    reverse_valid = jnp.any(positive_candidate, axis=0)
    reverse = -jnp.where(reverse_valid, log_positive_candidates - log_candidates, 0.0)
    reverse_weight = reverse_valid.astype(logits.dtype)
    evidence_to_weight = jnp.sum(reverse * reverse_weight) / jnp.maximum(
        jnp.sum(reverse_weight), 1.0
    )
    return 0.5 * (weight_to_evidence + evidence_to_weight)


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
    validation_loss: Callable[[Mapping[str, Any], Any], Any] | None = None,
    validation_batches: Any = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
):
    """Differentiate task loss through decoder and Octo adapter application."""

    require_jax()
    if steps <= 0:
        raise ValueError("refinement requires positive steps")
    if not callable(batches) and not batches:
        raise ValueError("refinement requires at least one batch")
    validation_enabled = validation_loss is not None
    if validation_enabled != (validation_batches is not None):
        raise ValueError("validation loss and batches must be provided together")
    if validation_enabled and (
        early_stopping_patience is None or early_stopping_patience <= 0
    ):
        raise ValueError("validation refinement requires positive early-stopping patience")
    if not validation_enabled and early_stopping_patience is not None:
        raise ValueError("early-stopping patience requires validation data")
    if early_stopping_min_delta < 0:
        raise ValueError("early-stopping min_delta cannot be negative")
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
    validation_iterator = None
    if validation_enabled:
        def validation_stream():
            if callable(validation_batches):
                while True:
                    produced = False
                    for batch in validation_batches():
                        produced = True
                        yield batch
                    if not produced:
                        raise ValueError("validation batch factory produced no batches")
            else:
                while True:
                    yield from validation_batches

        validation_iterator = validation_stream()
    best = current
    best_validation = float("inf")
    stale_checks = 0
    for _ in range(steps):
        (loss, auxiliary), gradients = jax.value_and_grad(objective, has_aux=True)(
            current, next(iterator)
        )
        current = jax.tree_util.tree_map(
            lambda value, gradient: value - learning_rate * gradient,
            current,
            gradients,
        )
        record = {"loss": loss, **auxiliary}
        if validation_enabled:
            observed_validation = float(
                validation_loss(current, next(validation_iterator))
            )
            record["validation_loss"] = observed_validation
            if observed_validation < best_validation - early_stopping_min_delta:
                best_validation = observed_validation
                best = jax.tree_util.tree_map(lambda value: value, current)
                stale_checks = 0
            else:
                stale_checks += 1
        history.append(record)
        if validation_enabled and stale_checks >= early_stopping_patience:
            break
    return best if validation_enabled else current, history
