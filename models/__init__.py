"""
models/__init__.py

Exposes the VLA registry, factory, supported architectures, and Weight-Space Alignment modules.
"""
from models.base_vla import BaseVLA, build_vla_model, register_vla
from models.small_vla import SmallVLA
from models.octo_small import OctoSmallVLA
from models.evidence_extractor import TaskEvidenceExtractor
from models.weight_autoencoder import FactorizedWeightAutoencoder, WeightTokenEncoder, WeightTokenDecoder
from models.contrastive_alignment import ModalityContrastiveAligner, PromptProjector
from models.differential_regularizer import DifferentialRegularizer, hyperspherical_shell_projection

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
