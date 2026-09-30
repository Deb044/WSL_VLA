"""Official Octo/JAX implementation of Methodology 1.

The package is organized by responsibility. The former PyTorch surrogate is
isolated under :mod:`wsl_vla.smoke.legacy_torch` and is never a research path.
"""

from .contracts import (
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
