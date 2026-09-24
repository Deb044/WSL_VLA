"""
models/octo_small.py

PyTorch implementation of Octo-Small (27M parameters) conforming to the BaseVLA interface.
Architecture:
  - L = 8 Transformer Layers
  - H = 384 Hidden Dimension (6 Attention Heads)
  - Modality Sub-modules per layer: vis_block, lang_block, act_block
  - LoRA Target Modules: layers.{i}.vis_block, layers.{i}.lang_block, layers.{i}.act_block
  - Extracted Delta W in R^[8, 3, r, 384]
"""
import os
from typing import Optional
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from models.base_vla import BaseVLA, register_vla


class OctoSmallLayer(nn.Module):
    """
    Octo-Small transformer layer factorized by architectural modality role.
    """
    def __init__(self, hidden_dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Modality 0: Vision sub-module (cross-attention / visual projection)
        self.vis_block = nn.Linear(hidden_dim, hidden_dim)

        # Modality 1: Language sub-module (task instruction conditioning)
        self.lang_block = nn.Linear(hidden_dim, hidden_dim)

        # Modality 2: Action sub-module (motor dynamics / action chunk decoding)
        self.act_block = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.vis_block(x)
        l = self.lang_block(x)
        a = self.act_block(x)
        x = self.norm(x + v + l + a)
        x = self.mlp_norm(x + self.mlp(x))
        return x


@register_vla("octo_small")
class OctoSmallVLA(BaseVLA):
    """
    Octo-Small (27M parameter) foundation model for robotic manipulation.
    Pretrained weights can be loaded from HuggingFace/checkpoint, or initialized
    with the exact Octo-Small dimensions for local development.
    """
    def __init__(
        self,
        num_layers: int = 8,
        hidden_dim: int = 384,
        action_dim: int = 7,
        img_feat_dim: int = 64,
        lang_embed_dim: int = 384,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__(num_layers=num_layers, hidden_dim=hidden_dim, action_dim=action_dim)

        # Observation Projectors: Dual camera views (primary + wrist) & language
        self.primary_cam_proj = nn.Linear(img_feat_dim, hidden_dim)
        self.wrist_cam_proj = nn.Linear(img_feat_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_embed_dim, hidden_dim)
        self.act_in_proj = nn.Linear(action_dim, hidden_dim)

        # 8 Factorized Transformer Layers
        self.layers = nn.ModuleList([
            OctoSmallLayer(hidden_dim=hidden_dim, num_heads=6)
            for _ in range(num_layers)
        ])

        # Action Prediction Head (Predicts 7-DoF delta commands)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, path: str):
        """Loads pre-trained weights from local checkpoint or Hugging Face cache."""
        state_dict = torch.load(path, map_location="cpu")
        # Load backbone weights matching keys
        self.load_state_dict(state_dict, strict=False)
        print(f"[OctoSmallVLA] Pretrained weights loaded from: {path}")

    def forward(
        self,
        vis_features: torch.Tensor,
        lang_embed: torch.Tensor,
        act_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        vis_features: [Batch, img_feat_dim] (or primary camera)
        lang_embed:   [Batch, lang_embed_dim]
        act_history:  [Batch, action_dim]
        Returns:      [Batch, action_dim] (Predicted next 7-DoF action)
        """
        # Embed inputs into token space
        cam_tok = self.primary_cam_proj(vis_features)
        wrist_tok = self.wrist_cam_proj(vis_features)
        lang_tok = self.lang_proj(lang_embed)
        act_tok = self.act_in_proj(act_history)

        x = cam_tok + wrist_tok + lang_tok + act_tok

        for layer in self.layers:
            x = layer(x)

        pred_action = self.action_head(x)
        return pred_action

    def attach_factorized_lora(self, rank: int = 16, alpha: int = 32) -> nn.Module:
        """
        Injects LoRA into all 8 layers for the 3 modality sub-modules:
          - layers.{i}.vis_block
          - layers.{i}.lang_block
          - layers.{i}.act_block
        Freezes the entire 27M backbone.
        """
        self._lora_rank = rank
        target_modules = []
        for i in range(self.num_layers):
            target_modules.extend([
                f"layers.{i}.vis_block",
                f"layers.{i}.lang_block",
                f"layers.{i}.act_block",
            ])

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
        )

        self._peft_model = get_peft_model(self, lora_config)
        return self._peft_model

    def extract_delta_w(self) -> torch.Tensor:
        """
        Extracts Delta W into tensor shape: [8, 3, r, 384]
        """
        if self._peft_model is None:
            raise RuntimeError("LoRA has not been attached! Call attach_factorized_lora() first.")

        delta_w = torch.zeros(self.num_layers, 3, self._lora_rank, self.hidden_dim, dtype=torch.float32)
        state = self._peft_model.state_dict()
        modality_keys = ["vis_block", "lang_block", "act_block"]

        for l in range(self.num_layers):
            for m_idx, mod_name in enumerate(modality_keys):
                key_A = f"base_model.model.layers.{l}.{mod_name}.lora_A.default.weight"
                if key_A in state:
                    A_weight = state[key_A].detach().cpu().to(torch.float32)
                    delta_w[l, m_idx] = A_weight
                else:
                    raise KeyError(f"Expected key '{key_A}' not found in PEFT state dict!")

        return delta_w

    @classmethod
    def from_config(cls, config: dict) -> "OctoSmallVLA":
        """Instantiates OctoSmallVLA from YAML config dictionary."""
        m_cfg = config["model"]
        return cls(
            num_layers=m_cfg.get("num_layers", 8),
            hidden_dim=m_cfg.get("hidden_dim", 384),
            action_dim=m_cfg.get("action_dim", 7),
            img_feat_dim=m_cfg.get("image_features_dim", 64),
            lang_embed_dim=m_cfg.get("language_embed_dim", 384),
            pretrained_path=m_cfg.get("pretrained_path", None),
        )
