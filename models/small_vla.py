"""
models/small_vla.py

Defines the SmallVLA architecture conforming to the unified BaseVLA interface.
Registered as "small_vla" in the model registry.
"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from models.base_vla import BaseVLA, register_vla


class SmallVLALayer(nn.Module):
    """
    A single multi-modal layer with dedicated sub-modules for vision, language, and action.
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.vis_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.lang_attn = nn.Linear(hidden_dim, hidden_dim)
        self.act_dense = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.act_fn(self.vis_mlp(x))
        l = self.act_fn(self.lang_attn(x))
        a = self.act_fn(self.act_dense(x))
        return self.norm(x + v + l + a)


@register_vla("small_vla")
class SmallVLA(BaseVLA):
    """
    Lightweight Vision-Language-Action architecture for fast CPU/dry-run testing.
    """
    def __init__(
        self,
        num_layers: int = 4,
        hidden_dim: int = 256,
        action_dim: int = 7,
        img_feat_dim: int = 64,
        lang_embed_dim: int = 384,
    ):
        super().__init__(num_layers=num_layers, hidden_dim=hidden_dim, action_dim=action_dim)

        # Input projectors
        self.vis_projector = nn.Linear(img_feat_dim, hidden_dim)
        self.lang_projector = nn.Linear(lang_embed_dim, hidden_dim)
        self.act_in_projector = nn.Linear(action_dim, hidden_dim)

        # Multi-modal backbone layers
        self.layers = nn.ModuleList([SmallVLALayer(hidden_dim) for _ in range(num_layers)])

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        vis_features: torch.Tensor,
        lang_embed: torch.Tensor,
        act_history: torch.Tensor,
    ) -> torch.Tensor:
        v_tok = self.vis_projector(vis_features)
        l_tok = self.lang_projector(lang_embed)
        a_tok = self.act_in_projector(act_history)

        x = v_tok + l_tok + a_tok

        for layer in self.layers:
            x = layer(x)

        pred_action = self.action_head(x)
        return pred_action

    def attach_factorized_lora(self, rank: int = 16, alpha: int = 32) -> nn.Module:
        self._lora_rank = rank
        target_modules = []
        for i in range(self.num_layers):
            target_modules.extend([
                f"layers.{i}.vis_mlp",
                f"layers.{i}.lang_attn",
                f"layers.{i}.act_dense",
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
        if self._peft_model is None:
            raise RuntimeError("LoRA has not been attached! Call attach_factorized_lora() first.")

        delta_w = torch.zeros(self.num_layers, 3, self._lora_rank, self.hidden_dim, dtype=torch.float32)
        state = self._peft_model.state_dict()
        modality_keys = ["vis_mlp", "lang_attn", "act_dense"]

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
    def from_config(cls, config: dict) -> "SmallVLA":
        m_cfg = config["model"]
        return cls(
            num_layers=m_cfg.get("num_layers", 4),
            hidden_dim=m_cfg.get("hidden_dim", 256),
            action_dim=m_cfg.get("action_dim", 7),
            img_feat_dim=m_cfg.get("image_features_dim", 64),
            lang_embed_dim=m_cfg.get("language_embed_dim", 384),
        )
