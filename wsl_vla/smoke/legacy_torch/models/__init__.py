"""
models/__init__.py

Exposes the VLA registry, factory, supported architectures, and Weight-Space Alignment modules.
"""
from .base_vla import BaseVLA, build_vla_model, register_vla
from .small_vla import SmallVLA
from .octo_small import OctoSmallVLA
from .evidence_extractor import TaskEvidenceExtractor
from .weight_autoencoder import FactorizedWeightAutoencoder, WeightTokenEncoder, WeightTokenDecoder
from .contrastive_alignment import ModalityContrastiveAligner, PromptProjector
from .differential_regularizer import DifferentialRegularizer, hyperspherical_shell_projection

__all__ = [
    "BaseVLA",
    "build_vla_model",
    "register_vla",
    "SmallVLA",
    "OctoSmallVLA",
    "TaskEvidenceExtractor",
    "FactorizedWeightAutoencoder",
    "WeightTokenEncoder",
    "WeightTokenDecoder",
    "ModalityContrastiveAligner",
    "PromptProjector",
    "DifferentialRegularizer",
    "hyperspherical_shell_projection",
]
