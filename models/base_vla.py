"""
models/base_vla.py

Defines the unified abstract interface for all Vision-Language-Action models
and the model registry/factory for easily swapping backbones (SmallVLA, Octo-Small, OpenVLA, etc.).
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type, Optional
import torch
import torch.nn as nn


class BaseVLA(nn.Module, ABC):
    """
    Standard interface that every VLA backbone must implement to plug into the
    Model Zoo training loop and Weight-Space Alignment pipeline.
    """
    def __init__(self, num_layers: int, hidden_dim: int, action_dim: int = 7):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        # Use object.__setattr__ so PyTorch does NOT register _peft_model into self._modules
        # (which prevents circular submodule reference when PeftModel wraps self)
        object.__setattr__(self, "_peft_model", None)
        self._lora_rank: int = 16

    @abstractmethod
    def forward(
        self,
        vis_features: torch.Tensor,
        lang_embed: torch.Tensor,
        act_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass predicting the next robot action.
        Returns: Tensor of shape [Batch, action_dim]
        """
        pass

    @abstractmethod
    def attach_factorized_lora(
        self, rank: int = 16, alpha: int = 32, dropout: float = 0.0
    ) -> nn.Module:
        """
        Injects LoRA adapters into vision, language, and action sub-modules,
        freezing the backbone. Must store the resulting model in self._peft_model.
        """
        pass

    @abstractmethod
    def extract_delta_w(self, peft_model: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Extracts the trained LoRA matrices and formats them into the
        modality-tensorized representation:
            Delta W in R^[L, 3, r, H]
        where:
            dim 0: Layer index (L)
            dim 1: Modality index (0: vis, 1: lang, 2: act)
            dim 2: LoRA rank (r)
            dim 3: Hidden dimension (H)
        """
    @abstractmethod
    def inject_delta_w(self, delta_w: torch.Tensor, peft_model: Optional[nn.Module] = None) -> None:
        """
        Injects the modality-tensorized LoRA weights Delta W in R^[L, 3, r, H]
        back into the PEFT model parameters (lora_A).
        """
        pass

    def get_dims(self) -> Tuple[int, int, int, int]:
        """Returns (L, 3, r, H)."""
        return (self.num_layers, 3, self._lora_rank, self.hidden_dim)


# ==============================================================================
# Model Registry & Factory
# ==============================================================================
_VLA_REGISTRY: Dict[str, Type[BaseVLA]] = {}


def register_vla(name: str):
    """Decorator to register a new VLA model architecture."""
    def decorator(cls: Type[BaseVLA]):
        _VLA_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def build_vla_model(config: dict) -> BaseVLA:
    """
    Factory function to instantiate any registered VLA backbone from config.
    To switch models, simply change `model.name` in configs/vla_config.yaml.
    """
    model_name = config["model"]["name"].lower()
    if model_name not in _VLA_REGISTRY:
        available = list(_VLA_REGISTRY.keys())
        raise ValueError(
            f"Unknown VLA model '{model_name}'. Available registered models: {available}"
        )

    model_cls = _VLA_REGISTRY[model_name]
    return model_cls.from_config(config)
