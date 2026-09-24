"""
models/__init__.py

Exposes the VLA registry, factory, and supported architectures.
"""
from models.base_vla import BaseVLA, build_vla_model, register_vla
from models.small_vla import SmallVLA
from models.octo_small import OctoSmallVLA
from models.evidence_extractor import TaskEvidenceExtractor

__all__ = [
    "BaseVLA",
    "build_vla_model",
    "register_vla",
    "SmallVLA",
    "OctoSmallVLA",
    "TaskEvidenceExtractor",
]
