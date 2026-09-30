"""Canonical packing of effective low-rank updates for weight-space learning."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .contracts import AdapterSpec, Component


@dataclass(frozen=True)
class PackedAdapter:
    """Padded effective-update windows and their non-padding mask."""

    tokens: np.ndarray
    mask: np.ndarray
    component_ids: np.ndarray
    layer_ids: np.ndarray
    entry_ids: np.ndarray
    row_ids: np.ndarray
    spec: AdapterSpec
    factors: Mapping[str, tuple[np.ndarray, np.ndarray]]

    def validate(self) -> None:
        n, width = self.tokens.shape
        if width != self.spec.token_width:
            raise ValueError("token width does not match AdapterSpec")
        for name, value in (
            ("mask", self.mask),
            ("component_ids", self.component_ids),
            ("layer_ids", self.layer_ids),
            ("entry_ids", self.entry_ids),
            ("row_ids", self.row_ids),
        ):
            if value.shape[0] != n:
                raise ValueError(f"{name} length does not match tokens")
        if self.mask.shape != self.tokens.shape:
            raise ValueError("mask must have the same shape as tokens")
        if not np.isfinite(self.tokens).all():
            raise ValueError("tokens contain non-finite values")
        expected = {entry.parameter_path for entry in self.spec.entries}
        if set(self.factors) != expected:
            raise ValueError("reloadable factor paths differ from AdapterSpec")
        by_path = {entry.parameter_path: entry for entry in self.spec.entries}
        for path, (down, up) in self.factors.items():
            entry = by_path[path]
            if np.asarray(down).shape != (entry.input_dim, entry.rank):
                raise ValueError(f"bad stored down factor shape for {path}")
            if np.asarray(up).shape != (entry.rank, entry.output_dim):
                raise ValueError(f"bad stored up factor shape for {path}")


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


def pack_low_rank_adapter(
    spec: AdapterSpec,
    factors: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> PackedAdapter:
    """Pack all effective updates into fixed-width row windows.

    ``factors[path]`` contains ``(down, up)`` with shapes ``[in, rank]`` and
    ``[rank, out]``. Raw factors are not used as weight-space coordinates.
    """

    tokens: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    component_ids: list[int] = []
    layer_ids: list[int] = []
    entry_ids: list[int] = []
    row_ids: list[int] = []
    component_index = {
        Component.VISION: 0,
        Component.LANGUAGE: 1,
        Component.ACTION: 2,
    }

    expected = {entry.parameter_path for entry in spec.entries}
    if set(factors) != expected:
        raise ValueError(
            f"factor paths differ from AdapterSpec; missing={sorted(expected-set(factors))}, "
            f"extra={sorted(set(factors)-expected)}"
        )

    for entry_index, entry in enumerate(spec.entries):
        down, up = (np.asarray(x) for x in factors[entry.parameter_path])
        if down.shape != (entry.input_dim, entry.rank):
            raise ValueError(f"bad down shape for {entry.parameter_path}: {down.shape}")
        if up.shape != (entry.rank, entry.output_dim):
            raise ValueError(f"bad up shape for {entry.parameter_path}: {up.shape}")
        dense = effective_update(down, up, alpha=spec.alpha, rank=entry.rank)
        for row_index, row in enumerate(dense):
            for offset in range(0, row.size, spec.token_width):
                piece = row[offset : offset + spec.token_width]
                token = np.zeros(spec.token_width, dtype=np.float32)
                mask = np.zeros(spec.token_width, dtype=bool)
                token[: piece.size] = piece.astype(np.float32, copy=False)
                mask[: piece.size] = True
                tokens.append(token)
                masks.append(mask)
                component_ids.append(component_index[entry.component])
                layer_ids.append(entry.layer)
                entry_ids.append(entry_index)
                row_ids.append(row_index)

    packed = PackedAdapter(
        tokens=np.stack(tokens),
        mask=np.stack(masks),
        component_ids=np.asarray(component_ids, dtype=np.int32),
        layer_ids=np.asarray(layer_ids, dtype=np.int32),
        entry_ids=np.asarray(entry_ids, dtype=np.int32),
        row_ids=np.asarray(row_ids, dtype=np.int32),
        spec=spec,
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
    outputs: dict[str, np.ndarray] = {}
    cursor = 0
    for entry_index, entry in enumerate(packed.spec.entries):
        count_per_row = (entry.output_dim + packed.spec.token_width - 1) // packed.spec.token_width
        matrix = np.zeros((entry.input_dim, entry.output_dim), dtype=np.float32)
        for row in range(entry.input_dim):
            chunks = []
            for _ in range(count_per_row):
                if packed.entry_ids[cursor] != entry_index or packed.row_ids[cursor] != row:
                    raise ValueError("packed token metadata is not in canonical order")
                chunks.append(packed.tokens[cursor][packed.mask[cursor]])
                cursor += 1
            matrix[row] = np.concatenate(chunks)[: entry.output_dim]
        outputs[entry.parameter_path] = matrix
    if cursor != len(packed.tokens):
        raise ValueError("unconsumed packed tokens")
    return outputs


def save_packed_adapter(packed: PackedAdapter, path: str | Path) -> None:
    packed.validate()
    metadata = json.dumps(packed.spec.to_dict(), sort_keys=True).encode("utf-8")
    factor_arrays = {}
    for index, entry in enumerate(packed.spec.entries):
        down, up = packed.factors[entry.parameter_path]
        factor_arrays[f"factor_down_{index:04d}"] = np.asarray(down)
        factor_arrays[f"factor_up_{index:04d}"] = np.asarray(up)
    np.savez_compressed(
        path,
        tokens=packed.tokens,
        mask=packed.mask,
        component_ids=packed.component_ids,
        layer_ids=packed.layer_ids,
        entry_ids=packed.entry_ids,
        row_ids=packed.row_ids,
        adapter_spec_json=np.frombuffer(metadata, dtype=np.uint8),
        **factor_arrays,
    )


def load_packed_adapter(path: str | Path) -> PackedAdapter:
    with np.load(path, allow_pickle=False) as archive:
        spec_payload = json.loads(bytes(archive["adapter_spec_json"]).decode("utf-8"))
        spec = AdapterSpec.from_dict(spec_payload)
        factors = {
            entry.parameter_path: (
                archive[f"factor_down_{index:04d}"],
                archive[f"factor_up_{index:04d}"],
            )
            for index, entry in enumerate(spec.entries)
        }
        packed = PackedAdapter(
            tokens=archive["tokens"],
            mask=archive["mask"],
            component_ids=archive["component_ids"],
            layer_ids=archive["layer_ids"],
            entry_ids=archive["entry_ids"],
            row_ids=archive["row_ids"],
            spec=spec,
            factors=factors,
        )
    packed.validate()
    return packed
