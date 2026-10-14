"""Canonical packing of effective low-rank updates for weight-space learning.

A packed adapter is the dense effective update of every adapter entry, cut into
fixed-width row windows ("weight tokens"). Every token carries five identities:
component, transformer layer, parameter entry, row, and column window. Tokens of
one component form a modality stream that is further cut into fixed-length
*sequence windows*; the alignment transformer processes those windows with
shared weights, and :func:`from_windows` restores the complete sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..contracts import (
    COMPONENT_NAMES,
    PACKED_ADAPTER_SCHEMA_VERSION,
    SITE_HEAD_KERNEL,
    AdapterSpec,
    Component,
    LegacyArtifactError,
    canonical_sha256,
    require_schema,
)


DEFAULT_SEQUENCE_WINDOW = 128
COMPONENT_INDEX = {name: index for index, name in enumerate(COMPONENT_NAMES)}


@dataclass(frozen=True)
class PackedAdapterMetadata:
    """Sample identity that varies while the shared AdapterSpec stays fixed."""

    task_id: str
    suite: str
    seed: int
    optimizer_updates: int
    snapshot_index: int
    sample_exposure: int
    source_dtype: str

    def validate(self) -> None:
        if not self.task_id or not self.suite:
            raise ValueError("packed adapter task and suite identities are required")
        if self.seed < 0 or self.snapshot_index < 0:
            raise ValueError("packed adapter seed and snapshot index must be non-negative")
        if self.optimizer_updates <= 0 or self.sample_exposure < self.optimizer_updates:
            raise ValueError("snapshot update count and sample exposure are inconsistent")
        try:
            np.dtype(self.source_dtype)
        except TypeError as exc:
            raise ValueError(f"unsupported adapter source dtype: {self.source_dtype}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "suite": self.suite,
            "seed": int(self.seed),
            "optimizer_updates": int(self.optimizer_updates),
            "snapshot_index": int(self.snapshot_index),
            "sample_exposure": int(self.sample_exposure),
            "source_dtype": self.source_dtype,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PackedAdapterMetadata":
        return cls(
            task_id=str(payload["task_id"]),
            suite=str(payload["suite"]),
            seed=int(payload["seed"]),
            optimizer_updates=int(payload["optimizer_updates"]),
            snapshot_index=int(payload["snapshot_index"]),
            sample_exposure=int(payload["sample_exposure"]),
            source_dtype=str(payload["source_dtype"]),
        )


@dataclass(frozen=True)
class TokenLayout:
    """Deterministic token and sequence-window identities for one AdapterSpec."""

    token_width: int
    sequence_window: int
    component_ids: np.ndarray
    layer_ids: np.ndarray
    entry_ids: np.ndarray
    row_ids: np.ndarray
    column_window_ids: np.ndarray
    token_mask: np.ndarray
    sequence_window_ids: np.ndarray
    window_position_ids: np.ndarray
    window_token_index: np.ndarray
    window_component_ids: np.ndarray

    @property
    def token_count(self) -> int:
        return int(self.component_ids.shape[0])

    @property
    def window_count(self) -> int:
        return int(self.window_token_index.shape[0])

    @property
    def window_valid(self) -> np.ndarray:
        """Boolean ``[windows, sequence_window]`` mask of non-padding positions."""

        return self.window_token_index >= 0

    @property
    def column_offsets(self) -> np.ndarray:
        return self.column_window_ids * self.token_width

    @property
    def embedding_sizes(self) -> dict[str, int]:
        """Table sizes for the five structural embeddings."""

        return {
            "component": len(COMPONENT_NAMES),
            "layer": int(self.layer_ids.max()) + 1,
            "entry": int(self.entry_ids.max()) + 1,
            "row": int(self.row_ids.max()) + 1,
            "column_window": int(self.column_window_ids.max()) + 1,
        }

    def component_positions(self, component: str | Component) -> np.ndarray:
        """Token indices of one modality stream in canonical order."""

        name = component.value if isinstance(component, Component) else str(component)
        return np.flatnonzero(self.component_ids == COMPONENT_INDEX[name])

    def component_windows(self, component: str | Component) -> np.ndarray:
        name = component.value if isinstance(component, Component) else str(component)
        return np.flatnonzero(self.window_component_ids == COMPONENT_INDEX[name])

    def validate(self) -> None:
        count = self.token_count
        for name in (
            "layer_ids",
            "entry_ids",
            "row_ids",
            "column_window_ids",
            "sequence_window_ids",
            "window_position_ids",
        ):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} length does not match the token count")
        if self.token_mask.shape != (count, self.token_width):
            raise ValueError("token_mask must have shape [tokens, token_width]")
        if not self.token_mask.any(axis=1).all():
            raise ValueError("every weight token must hold at least one real entry")
        index = self.window_token_index
        if index.ndim != 2 or index.shape[1] != self.sequence_window:
            raise ValueError("window_token_index must be [windows, sequence_window]")
        assigned = np.sort(index[index >= 0])
        if not np.array_equal(assigned, np.arange(count)):
            raise ValueError("sequence windows must cover every token exactly once")
        restored = index[self.sequence_window_ids, self.window_position_ids]
        if not np.array_equal(restored, np.arange(count)):
            raise ValueError("sequence-window identities do not invert the window index")
        valid = index >= 0
        streams = self.component_ids[np.where(valid, index, 0)]
        if np.any(valid & (streams != self.window_component_ids[:, None])):
            raise ValueError("a sequence window mixes modality streams")
        if np.any(valid[:, 1:] & ~valid[:, :-1]):
            raise ValueError("sequence-window padding must be trailing")

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "token_width": np.asarray(self.token_width, dtype=np.int64),
            "sequence_window": np.asarray(self.sequence_window, dtype=np.int64),
            "component_ids": self.component_ids,
            "layer_ids": self.layer_ids,
            "entry_ids": self.entry_ids,
            "row_ids": self.row_ids,
            "column_window_ids": self.column_window_ids,
            "token_mask": self.token_mask,
            "sequence_window_ids": self.sequence_window_ids,
            "window_position_ids": self.window_position_ids,
            "window_token_index": self.window_token_index,
            "window_component_ids": self.window_component_ids,
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, Any]) -> "TokenLayout":
        layout = cls(
            token_width=int(arrays["token_width"]),
            sequence_window=int(arrays["sequence_window"]),
            component_ids=np.array(arrays["component_ids"], dtype=np.int32),
            layer_ids=np.array(arrays["layer_ids"], dtype=np.int32),
            entry_ids=np.array(arrays["entry_ids"], dtype=np.int32),
            row_ids=np.array(arrays["row_ids"], dtype=np.int32),
            column_window_ids=np.array(arrays["column_window_ids"], dtype=np.int32),
            token_mask=np.array(arrays["token_mask"], dtype=bool),
            sequence_window_ids=np.array(arrays["sequence_window_ids"], dtype=np.int32),
            window_position_ids=np.array(arrays["window_position_ids"], dtype=np.int32),
            window_token_index=np.array(arrays["window_token_index"], dtype=np.int32),
            window_component_ids=np.array(arrays["window_component_ids"], dtype=np.int32),
        )
        layout.validate()
        return layout

    def sha256(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        for name, value in sorted(self.to_arrays().items()):
            array = np.ascontiguousarray(value)
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(array.tobytes())
        return digest.hexdigest()


def layer_embedding_ids(spec: AdapterSpec) -> dict[str, int]:
    """Embedding identity per entry: block index, or ``blocks + head ordinal``.

    Action-head kernels are not transformer layers, so they are numbered after
    the last transformer block instead of colliding with block indices.
    """

    block_layers = [entry.layer for entry in spec.entries if entry.site != SITE_HEAD_KERNEL]
    block_count = max(block_layers) + 1 if block_layers else 0
    return {
        entry.parameter_path: (
            block_count + entry.layer if entry.site == SITE_HEAD_KERNEL else entry.layer
        )
        for entry in spec.entries
    }


def build_token_layout(
    spec: AdapterSpec, *, sequence_window: int = DEFAULT_SEQUENCE_WINDOW
) -> TokenLayout:
    """Derive every token and window identity from the AdapterSpec alone."""

    if sequence_window <= 0:
        raise ValueError("sequence_window must be positive")
    width = spec.token_width
    layer_ids_by_path = layer_embedding_ids(spec)
    component_ids, layer_ids, entry_ids, row_ids, column_ids, masks = [], [], [], [], [], []
    for entry_index, entry in enumerate(spec.entries):
        windows_per_row = -(-entry.output_dim // width)
        row_mask = np.zeros((windows_per_row, width), dtype=bool)
        row_mask.reshape(-1)[: entry.output_dim] = True
        count = entry.input_dim * windows_per_row
        component_ids.append(np.full(count, COMPONENT_INDEX[entry.component.value], np.int32))
        layer_ids.append(np.full(count, layer_ids_by_path[entry.parameter_path], np.int32))
        entry_ids.append(np.full(count, entry_index, np.int32))
        row_ids.append(np.repeat(np.arange(entry.input_dim, dtype=np.int32), windows_per_row))
        column_ids.append(np.tile(np.arange(windows_per_row, dtype=np.int32), entry.input_dim))
        masks.append(np.tile(row_mask, (entry.input_dim, 1)))
    if not component_ids:
        raise ValueError("AdapterSpec has no entries to pack")
    component = np.concatenate(component_ids)
    token_count = component.shape[0]

    window_rows: list[np.ndarray] = []
    window_components: list[int] = []
    sequence_window_ids = np.zeros(token_count, dtype=np.int32)
    window_position_ids = np.zeros(token_count, dtype=np.int32)
    for component_index in range(len(COMPONENT_NAMES)):
        positions = np.flatnonzero(component == component_index)
        for start in range(0, len(positions), sequence_window):
            chunk = positions[start : start + sequence_window]
            row = np.full(sequence_window, -1, dtype=np.int32)
            row[: len(chunk)] = chunk
            sequence_window_ids[chunk] = len(window_rows)
            window_position_ids[chunk] = np.arange(len(chunk), dtype=np.int32)
            window_rows.append(row)
            window_components.append(component_index)

    layout = TokenLayout(
        token_width=width,
        sequence_window=int(sequence_window),
        component_ids=component,
        layer_ids=np.concatenate(layer_ids),
        entry_ids=np.concatenate(entry_ids),
        row_ids=np.concatenate(row_ids),
        column_window_ids=np.concatenate(column_ids),
        token_mask=np.concatenate(masks),
        sequence_window_ids=sequence_window_ids,
        window_position_ids=window_position_ids,
        window_token_index=np.stack(window_rows),
        window_component_ids=np.asarray(window_components, dtype=np.int32),
    )
    layout.validate()
    return layout


def to_windows(layout: TokenLayout, values: Any, *, feature_dims: int = 1, xp: Any = np) -> Any:
    """Gather ``[..., tokens, *feature]`` values into padded sequence windows.

    ``feature_dims`` is the number of trailing feature axes after the token
    axis (1 for token vectors, 0 for per-token scalars). The result has shape
    ``[..., windows, sequence_window, *feature]`` with zeros at padding
    positions. ``xp`` may be ``numpy`` or ``jax.numpy``.
    """

    token_axis = values.ndim - 1 - feature_dims
    if feature_dims < 0 or token_axis < 0 or values.shape[token_axis] != layout.token_count:
        raise ValueError(
            f"values with shape {values.shape} have no token axis of length "
            f"{layout.token_count} before {feature_dims} feature axes"
        )
    feature_shape = values.shape[token_axis + 1 :]
    padding = xp.zeros((*values.shape[:token_axis], 1, *feature_shape), dtype=values.dtype)
    padded = xp.concatenate([values, padding], axis=token_axis)
    index = np.where(layout.window_token_index >= 0, layout.window_token_index, layout.token_count)
    gathered = xp.take(padded, index.reshape(-1), axis=token_axis)
    return gathered.reshape(
        *values.shape[:token_axis], layout.window_count, layout.sequence_window, *feature_shape
    )


def from_windows(
    layout: TokenLayout, windowed: Any, *, feature_dims: int = 1, xp: Any = np
) -> Any:
    """Restore the complete token sequence from sequence windows (exact inverse)."""

    window_axis = windowed.ndim - 2 - feature_dims
    if (
        feature_dims < 0
        or window_axis < 0
        or windowed.shape[window_axis : window_axis + 2]
        != (layout.window_count, layout.sequence_window)
    ):
        raise ValueError(f"windowed values with shape {windowed.shape} do not match the layout")
    feature_shape = windowed.shape[window_axis + 2 :]
    flat = windowed.reshape(
        *windowed.shape[:window_axis],
        layout.window_count * layout.sequence_window,
        *feature_shape,
    )
    index = (
        layout.sequence_window_ids.astype(np.int64) * layout.sequence_window
        + layout.window_position_ids
    )
    return xp.take(flat, index, axis=window_axis)


@dataclass(frozen=True)
class PackedAdapter:
    """Effective-update tokens, their layout, and the reloadable factors."""

    tokens: np.ndarray
    layout: TokenLayout
    spec: AdapterSpec
    metadata: PackedAdapterMetadata
    factors: Mapping[str, tuple[np.ndarray, np.ndarray]]

    @property
    def mask(self) -> np.ndarray:
        return self.layout.token_mask

    @property
    def component_ids(self) -> np.ndarray:
        return self.layout.component_ids

    @property
    def layer_ids(self) -> np.ndarray:
        return self.layout.layer_ids

    @property
    def entry_ids(self) -> np.ndarray:
        return self.layout.entry_ids

    @property
    def row_ids(self) -> np.ndarray:
        return self.layout.row_ids

    @property
    def column_offsets(self) -> np.ndarray:
        return self.layout.column_offsets

    def validate(self) -> None:
        if self.tokens.shape != (self.layout.token_count, self.spec.token_width):
            raise ValueError("tokens do not match the AdapterSpec token layout")
        if not np.isfinite(self.tokens).all():
            raise ValueError("tokens contain non-finite values")
        if np.any(self.tokens[~self.layout.token_mask] != 0):
            raise ValueError("padded token entries must be exactly zero")
        self.metadata.validate()
        validate_factor_shapes(self.spec, self.factors)


def validate_factor_shapes(
    spec: AdapterSpec, factors: Mapping[str, tuple[Any, Any]]
) -> None:
    expected = {entry.parameter_path for entry in spec.entries}
    if set(factors) != expected:
        raise ValueError(
            f"factor paths differ from AdapterSpec; missing={sorted(expected-set(factors))}, "
            f"extra={sorted(set(factors)-expected)}"
        )
    for entry in spec.entries:
        down, up = factors[entry.parameter_path]
        if np.shape(down) != (entry.input_dim, entry.rank):
            raise ValueError(f"bad down shape for {entry.parameter_path}: {np.shape(down)}")
        if np.shape(up) != (entry.rank, entry.output_dim):
            raise ValueError(f"bad up shape for {entry.parameter_path}: {np.shape(up)}")


def effective_update(
    down: np.ndarray,
    up: np.ndarray,
    *,
    alpha: float,
    rank: int,
) -> np.ndarray:
    """Return the basis-invariant dense update for row-vector Flax kernels.

    Flax applies ``x @ down @ up``. The dense kernel update is therefore
    ``down @ up``; this is the transpose convention of the common ``B @ A``
    notation for column-vector layers.
    """

    down = np.asarray(down)
    up = np.asarray(up)
    if down.ndim != 2 or up.ndim != 2:
        raise ValueError("low-rank factors must be matrices")
    if down.shape[1] != rank or up.shape[0] != rank:
        raise ValueError("factor rank does not match AdapterSpec")
    return (alpha / rank) * (down @ up)


def factors_to_updates(
    spec: AdapterSpec, factors: Mapping[str, tuple[Any, Any]]
) -> dict[str, np.ndarray]:
    """Dense effective update of every entry; raw factors are never coordinates."""

    validate_factor_shapes(spec, factors)
    return {
        entry.parameter_path: effective_update(
            *factors[entry.parameter_path], alpha=spec.alpha, rank=entry.rank
        )
        for entry in spec.entries
    }


def pack_effective_updates(
    spec: AdapterSpec, updates: Mapping[str, Any], *, layout: TokenLayout | None = None
) -> np.ndarray:
    """Cut dense updates into ``[tokens, token_width]`` row windows."""

    expected = {entry.parameter_path for entry in spec.entries}
    if set(updates) != expected:
        raise ValueError("update paths differ from AdapterSpec")
    width = spec.token_width
    pieces = []
    for entry in spec.entries:
        dense = np.asarray(updates[entry.parameter_path])
        if dense.shape != (entry.input_dim, entry.output_dim):
            raise ValueError(f"bad update shape for {entry.parameter_path}: {dense.shape}")
        windows_per_row = -(-entry.output_dim // width)
        padded = np.zeros((entry.input_dim, windows_per_row * width), dtype=np.float32)
        padded[:, : entry.output_dim] = dense
        pieces.append(padded.reshape(entry.input_dim * windows_per_row, width))
    tokens = np.concatenate(pieces)
    if layout is not None and tokens.shape != layout.token_mask.shape:
        raise ValueError("packed tokens do not match the supplied token layout")
    return tokens


def unpack_tokens(spec: AdapterSpec, tokens: Any, *, xp: Any = np) -> dict[str, Any]:
    """Invert :func:`pack_effective_updates`; ``xp`` may be ``jax.numpy``."""

    width = spec.token_width
    if tokens.ndim != 2 or tokens.shape[1] != width:
        raise ValueError("tokens must have shape [tokens, token_width]")
    cursor = 0
    updates = {}
    for entry in spec.entries:
        windows_per_row = -(-entry.output_dim // width)
        count = entry.input_dim * windows_per_row
        if cursor + count > tokens.shape[0]:
            raise ValueError("token count is smaller than the AdapterSpec layout")
        block = tokens[cursor : cursor + count].reshape(entry.input_dim, windows_per_row * width)
        updates[entry.parameter_path] = block[:, : entry.output_dim]
        cursor += count
    if cursor != tokens.shape[0]:
        raise ValueError("token count differs from the AdapterSpec layout")
    return updates


def pack_low_rank_adapter(
    spec: AdapterSpec,
    factors: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    metadata: PackedAdapterMetadata,
    sequence_window: int = DEFAULT_SEQUENCE_WINDOW,
) -> PackedAdapter:
    """Pack all effective updates into fixed-width row windows.

    ``factors[path]`` contains ``(down, up)`` with shapes ``[in, rank]`` and
    ``[rank, out]``. Raw factors are not used as weight-space coordinates.
    """

    layout = build_token_layout(spec, sequence_window=sequence_window)
    tokens = pack_effective_updates(spec, factors_to_updates(spec, factors), layout=layout)
    packed = PackedAdapter(
        tokens=tokens,
        layout=layout,
        spec=spec,
        metadata=metadata,
        factors={
            path: (np.asarray(pair[0]).copy(), np.asarray(pair[1]).copy())
            for path, pair in factors.items()
        },
    )
    packed.validate()
    return packed


def unpack_effective_updates(packed: PackedAdapter) -> dict[str, np.ndarray]:
    """Reconstruct dense effective updates from a packed representation."""

    packed.validate()
    return {
        path: np.array(value, dtype=np.float32, copy=True)
        for path, value in unpack_tokens(packed.spec, packed.tokens).items()
    }


def save_packed_adapter(packed: PackedAdapter, path: str | Path) -> None:
    packed.validate()
    spec_metadata = json.dumps(packed.spec.to_dict(), sort_keys=True).encode("utf-8")
    sample_metadata = json.dumps(
        {
            "schema_version": PACKED_ADAPTER_SCHEMA_VERSION,
            "sequence_window": packed.layout.sequence_window,
            "layout_sha256": packed.layout.sha256(),
            "adapter_spec_sha256": packed.spec.sha256(),
            **packed.metadata.to_dict(),
        },
        sort_keys=True,
    ).encode("utf-8")
    factor_arrays = {}
    for index, entry in enumerate(packed.spec.entries):
        down, up = packed.factors[entry.parameter_path]
        factor_arrays[f"factor_down_{index:04d}"] = np.asarray(down)
        factor_arrays[f"factor_up_{index:04d}"] = np.asarray(up)
    np.savez_compressed(
        path,
        tokens=packed.tokens,
        adapter_spec_json=np.frombuffer(spec_metadata, dtype=np.uint8),
        packed_adapter_metadata_json=np.frombuffer(sample_metadata, dtype=np.uint8),
        **packed.layout.to_arrays(),
        **factor_arrays,
    )


def _read_json_member(archive: Any, name: str) -> dict[str, Any]:
    return json.loads(bytes(archive[name]).decode("utf-8"))


def load_packed_adapter(path: str | Path) -> PackedAdapter:
    """Load a corrected (schema 2) packed adapter and re-derive its layout.

    The stored tokens must equal the effective update of the stored factors, and
    the stored identities must equal the layout derived from the AdapterSpec, so
    a hand-edited or stale file cannot enter an alignment archive.
    """

    with np.load(path, allow_pickle=False) as archive:
        metadata_payload = _read_json_member(archive, "packed_adapter_metadata_json")
        spec_payload = _read_json_member(archive, "adapter_spec_json")
        if "schema_version" not in metadata_payload:
            raise LegacyArtifactError(
                f"{path} is a version-one packed adapter (post-block placement, per-task "
                "action normalization). It is inspection-only; regenerate the zoo."
            )
        require_schema(
            metadata_payload["schema_version"], PACKED_ADAPTER_SCHEMA_VERSION, "packed adapter"
        )
        spec = AdapterSpec.from_dict(spec_payload)
        if metadata_payload.get("adapter_spec_sha256") != spec.sha256():
            raise ValueError(f"packed adapter AdapterSpec digest is inconsistent: {path}")
        metadata = PackedAdapterMetadata.from_dict(metadata_payload)
        layout = build_token_layout(spec, sequence_window=int(metadata_payload["sequence_window"]))
        if layout.sha256() != metadata_payload.get("layout_sha256"):
            raise ValueError(f"packed adapter layout differs from its AdapterSpec: {path}")
        stored_layout = TokenLayout.from_arrays({name: archive[name] for name in layout.to_arrays()})
        if stored_layout.sha256() != layout.sha256():
            raise ValueError(f"stored token identities differ from the derived layout: {path}")
        factors = {
            entry.parameter_path: (
                np.array(archive[f"factor_down_{index:04d}"], copy=True),
                np.array(archive[f"factor_up_{index:04d}"], copy=True),
            )
            for index, entry in enumerate(spec.entries)
        }
        tokens = np.array(archive["tokens"], dtype=np.float32, copy=True)
    packed = PackedAdapter(
        tokens=tokens, layout=layout, spec=spec, metadata=metadata, factors=factors
    )
    packed.validate()
    expected = pack_effective_updates(spec, factors_to_updates(spec, factors), layout=layout)
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    if not np.allclose(tokens, expected, rtol=1e-5, atol=1e-6 * max(scale, 1e-12)):
        raise ValueError(f"stored tokens are not the effective update of the stored factors: {path}")
    return packed


def packed_adapter_identity(path: str | Path) -> dict[str, Any]:
    """Cheap identity read (no token validation) for indexing a population."""

    with np.load(path, allow_pickle=False) as archive:
        metadata = _read_json_member(archive, "packed_adapter_metadata_json")
        spec_payload = _read_json_member(archive, "adapter_spec_json")
    if "schema_version" not in metadata:
        raise LegacyArtifactError(f"{path} is a version-one packed adapter")
    require_schema(metadata["schema_version"], PACKED_ADAPTER_SCHEMA_VERSION, "packed adapter")
    return {**metadata, "adapter_spec_sha256_recomputed": canonical_sha256(spec_payload)}
