"""Contracts and algorithms used by the official Octo/JAX research path."""

from .contracts import AdapterEntry, AdapterSpec, AlignmentCheckpoint, TaskEvidence
from .metrics import continual_learning_metrics
from .packing import (
    PackedAdapter,
    PackedAdapterMetadata,
    pack_low_rank_adapter,
    unpack_effective_updates,
)

__all__ = [
    "AdapterEntry",
    "AdapterSpec",
    "AlignmentCheckpoint",
    "TaskEvidence",
    "PackedAdapter",
    "PackedAdapterMetadata",
    "pack_low_rank_adapter",
    "unpack_effective_updates",
    "continual_learning_metrics",
]
