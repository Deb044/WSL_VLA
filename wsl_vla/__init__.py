"""Research-grade Methodology 1 implementation.

The legacy ``models`` and ``scripts`` packages are retained for CPU smoke tests.
Publication experiments must import from :mod:`wsl_vla.research`.
"""

from .research.contracts import (
    AdapterEntry,
    AdapterSpec,
    AlignmentCheckpointMetadata,
    EvaluationRecord,
    RunManifest,
    TaskEvidence,
)

__all__ = [
    "AdapterEntry",
    "AdapterSpec",
    "AlignmentCheckpointMetadata",
    "EvaluationRecord",
    "RunManifest",
    "TaskEvidence",
]
