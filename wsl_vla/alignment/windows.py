"""Bind a token layout and its RMS scales to the windowed alignment model.

Training, checkpoint inference, and evaluation all encode and decode through
:class:`LayoutBinding`, so the three cannot disagree on normalization, on how
sequence windows are cut, or on how the complete sequence is restored.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..adapters.packing import TokenLayout, from_windows, to_windows
from .models import STRUCTURAL_EMBEDDINGS, map_windows, require_jax


def floored_entry_scales(entry_rms: np.ndarray, *, floor: float) -> np.ndarray:
    """Training-only per-entry RMS with a numerical floor.

    An entry that is (numerically) zero in every training adapter has no scale
    of its own; the floor keeps its normalization invertible instead of
    dividing by zero.
    """

    values = np.asarray(entry_rms, dtype=np.float64)
    if floor <= 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("entry RMS values must be finite, non-negative, with a positive floor")
    return np.maximum(values, floor).astype(np.float32)


class LayoutBinding:
    """Static layout arrays plus normalization for one alignment checkpoint."""

    def __init__(self, layout: TokenLayout, entry_scales: np.ndarray, *, window_chunk: int):
        require_jax()
        import jax.numpy as jnp

        scales = np.asarray(entry_scales, dtype=np.float32)
        if scales.ndim != 1 or scales.shape[0] != int(layout.entry_ids.max()) + 1:
            raise ValueError("one positive scale is required per adapter entry")
        if np.any(scales <= 0) or not np.isfinite(scales).all():
            raise ValueError("entry scales must be finite and positive")
        self.layout = layout
        self.window_chunk = int(window_chunk)
        self.entry_scales = scales
        self.token_scale = jnp.asarray(scales[layout.entry_ids])[:, None]
        self.token_mask = jnp.asarray(layout.token_mask)
        self.component_ids = np.asarray(layout.component_ids)
        self.window_valid = jnp.asarray(layout.window_valid)
        identities = {
            "component": layout.component_ids,
            "layer": layout.layer_ids,
            "entry": layout.entry_ids,
            "row": layout.row_ids,
            "column_window": layout.column_window_ids,
        }
        self.identities = {
            name: jnp.asarray(to_windows(layout, identities[name], feature_dims=0))
            for name in STRUCTURAL_EMBEDDINGS
        }
        self._jnp = jnp

    # ------------------------------------------------------------ normalization
    def normalize(self, tokens: Any) -> Any:
        return tokens / self.token_scale

    def denormalize(self, tokens: Any) -> Any:
        """Invert the RMS normalization and zero every padded entry."""

        return self._jnp.where(self.token_mask, tokens * self.token_scale, 0.0)

    # ------------------------------------------------------------------ windows
    def _flat(self, values: Any, feature_dims: int = 1):
        jnp = self._jnp
        windows = to_windows(self.layout, values, feature_dims=feature_dims, xp=jnp)
        batch = windows.shape[0]
        return windows.reshape(batch * self.layout.window_count, *windows.shape[2:]), batch

    def _tiled(self, batch: int):
        jnp = self._jnp
        valid = jnp.tile(self.window_valid, (batch, 1))
        identities = tuple(
            jnp.tile(self.identities[name], (batch, 1)) for name in STRUCTURAL_EMBEDDINGS
        )
        return valid, identities

    def _restore(self, flat: Any, batch: int):
        jnp = self._jnp
        windows = flat.reshape(
            batch, self.layout.window_count, self.layout.sequence_window, flat.shape[-1]
        )
        return from_windows(self.layout, windows, xp=jnp)

    # ------------------------------------------------------------------- model
    def encode(self, model: Any, params: Any, tokens: Any) -> Any:
        """``[batch, tokens, width]`` original-unit tokens -> ``[batch, tokens, latent]``."""

        flat, batch = self._flat(self.normalize(tokens))
        valid, identities = self._tiled(batch)

        def run(window_tokens, window_valid, *ids):
            return model.apply(
                {"params": params},
                window_tokens,
                window_valid,
                dict(zip(STRUCTURAL_EMBEDDINGS, ids)),
                method=model.encode,
            )

        latents = map_windows(run, (flat, valid, *identities), chunk=self.window_chunk)
        return self._restore(latents, batch)

    def decode_normalized(self, model: Any, params: Any, latents: Any) -> Any:
        """``[batch, tokens, latent]`` -> normalized ``[batch, tokens, width]`` tokens."""

        flat, batch = self._flat(latents)
        valid, identities = self._tiled(batch)

        def run(window_latents, window_valid, *ids):
            return model.apply(
                {"params": params},
                window_latents,
                window_valid,
                dict(zip(STRUCTURAL_EMBEDDINGS, ids)),
                method=model.decode,
            )

        decoded = map_windows(run, (flat, valid, *identities), chunk=self.window_chunk)
        return self._restore(decoded, batch)

    def decode(self, model: Any, params: Any, latents: Any) -> Any:
        """``[batch, tokens, latent]`` -> original-unit tokens with padding zeroed."""

        return self.denormalize(self.decode_normalized(model, params, latents))

    def autoencode(self, model: Any, params: Any, tokens: Any):
        """Return ``(normalized reconstruction, latents, normalized targets)``."""

        jnp = self._jnp
        targets = self.normalize(tokens)
        flat, batch = self._flat(targets)
        valid, identities = self._tiled(batch)

        def run(window_tokens, window_valid, *ids):
            identity = dict(zip(STRUCTURAL_EMBEDDINGS, ids))
            latents = model.apply(
                {"params": params}, window_tokens, window_valid, identity, method=model.encode
            )
            decoded = model.apply(
                {"params": params}, latents, window_valid, identity, method=model.decode
            )
            return jnp.concatenate([latents, decoded], axis=-1)

        combined = self._restore(
            map_windows(run, (flat, valid, *identities), chunk=self.window_chunk), batch
        )
        width = self.layout.token_width
        return combined[..., -width:], combined[..., :-width], targets
