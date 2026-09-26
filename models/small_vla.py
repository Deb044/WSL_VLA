"""
models/small_vla.py

Defines the SmallVLA architecture conforming to the unified BaseVLA interface.
Registered as "small_vla" in the model registry.
"""
from typing import Optional
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

    def attach_factorized_lora(
        self, rank: int = 16, alpha: int = 32, dropout: float = 0.0
    ) -> nn.Module:
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
            lora_dropout=dropout,
            bias="none",
        )

        peft_model = get_peft_model(self, lora_config)

        # Initialize fixed factorized B matrix (WIZARD convention: fixed deterministic projection)
        # so that Delta W (A in R^[L, 3, r, H]) directly modulates the forward pass without
        # being canceled by LayerNorm shift-invariance
        for l in range(self.num_layers):
            for m_idx, mod in enumerate(["vis_mlp", "lang_attn", "act_dense"]):
                layer_mod = getattr(self.layers[l], mod)
                if hasattr(layer_mod, "lora_B") and hasattr(layer_mod.lora_B, "default"):
                    g = torch.Generator().manual_seed(1000 + l * 10 + m_idx)
                    w = torch.randn(layer_mod.lora_B.default.weight.shape, generator=g) / (rank ** 0.5)
                    layer_mod.lora_B.default.weight.data.copy_(w)

        # Use object.__setattr__ to avoid PyTorch circular submodule registration
        object.__setattr__(self, "_peft_model", peft_model)
        return peft_model

    def extract_delta_w(self, peft_model: Optional[nn.Module] = None) -> torch.Tensor:
        target_peft = peft_model if peft_model is not None else getattr(self, "_peft_model", None)
        if target_peft is None:
            raise RuntimeError("LoRA has not been attached! Call attach_factorized_lora() first.")

        delta_w = torch.zeros(self.num_layers, 3, self._lora_rank, self.hidden_dim, dtype=torch.float32)
        params = dict(target_peft.named_parameters())
        modality_keys = ["vis_mlp", "lang_attn", "act_dense"]

        for l in range(self.num_layers):
            for m_idx, mod_name in enumerate(modality_keys):
                key_A = f"base_model.model.layers.{l}.{mod_name}.lora_A.default.weight"
                if key_A in params:
                    A_weight = params[key_A].detach().cpu().to(torch.float32)
                    delta_w[l, m_idx] = A_weight
                else:
                    raise KeyError(f"Expected key '{key_A}' not found in PEFT parameters!")

        return delta_w

    def inject_delta_w(self, delta_w: torch.Tensor, peft_model: Optional[nn.Module] = None) -> None:
        """
        Injects Delta W (shape [L, 3, r, H] or [1, L, 3, r, H]) into PEFT lora_A weights.
        Ensures fixed factorized B projection is preserved.
        """
        target_peft = peft_model if peft_model is not None else getattr(self, "_peft_model", None)
        if target_peft is None:
            raise RuntimeError("LoRA has not been attached! Call attach_factorized_lora() first.")

        if delta_w.dim() == 5:
            delta_w = delta_w.squeeze(0)

        modality_keys = ["vis_mlp", "lang_attn", "act_dense"]
        for l in range(self.num_layers):
            for m_idx, mod_name in enumerate(modality_keys):
                layer_mod = getattr(target_peft.base_model.model.layers[l], mod_name)
                if hasattr(layer_mod, "lora_B") and hasattr(layer_mod.lora_B, "default"):
                    g = torch.Generator().manual_seed(1000 + l * 10 + m_idx)
                    w = torch.randn(layer_mod.lora_B.default.weight.shape, generator=g) / (self._lora_rank ** 0.5)
                    layer_mod.lora_B.default.weight.data.copy_(w.to(layer_mod.lora_B.default.weight.device))
                layer_mod.lora_A.default.weight.data.copy_(delta_w[l, m_idx].to(layer_mod.lora_A.default.weight.device))

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
