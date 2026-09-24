"""
models/weight_autoencoder.py

Implements Section 1.1 of Methodology:
Component-Factorized Tokenization and Shared Weight Autoencoder (g_phi, h_psi).

Transforms modality-factorized LoRA weight slices:
    Delta W_m in R^[L, r * H]
into compressed latent token representations:
    Z_m in R^[L, d_latent]
and reconstructs them back with a Transformer Decoder:
    hat{Delta W}_m = h_psi(Z_m)
"""
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightTokenEncoder(nn.Module):
    """
    g_phi: Transformer Encoder mapping flattened layer weight tokens into latent space.
    Input per layer token: [r * H]
    Output latent token:   [d_latent]
    """
    def __init__(self, input_dim: int, d_model: int = 256, d_latent: int = 128, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 32, d_model) * 0.02) # Max 32 layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, L, r * H]
        Returns: [Batch, L, d_latent]
        """
        B, L, _ = x.shape
        h = self.input_proj(x) + self.pos_embed[:, :L, :]
        h = self.transformer(h)
        z = self.to_latent(h)
        return z


class WeightTokenDecoder(nn.Module):
    """
    h_psi: Transformer Decoder mapping latent token sequences back to weight space.
    Input latent token: [d_latent]
    Output weight slice: [r * H]
    """
    def __init__(self, output_dim: int, d_model: int = 256, d_latent: int = 128, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.from_latent = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [Batch, L, d_latent]
        Returns: [Batch, L, r * H]
        """
        B, L, _ = z.shape
        h = self.from_latent(z) + self.pos_embed[:, :L, :]
        h = self.transformer(h)
        out = self.out_proj(h)
        return out


class FactorizedWeightAutoencoder(nn.Module):
    """
    Unified Component-Factorized Weight Autoencoder.
    Processes Delta W in R^[B, L, 3, r, H] by decomposing along modality dimension:
      - vis  (m=0): Z_vis  = g_phi(Delta W_vis)
      - lang (m=1): Z_lang = g_phi(Delta W_lang)
      - act  (m=2): Z_act  = g_phi(Delta W_act)
    """
    def __init__(self, num_layers: int = 8, rank: int = 16, hidden_dim: int = 384, d_latent: int = 128):
        super().__init__()
        self.num_layers = num_layers
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.weight_dim = rank * hidden_dim
        self.d_latent = d_latent

        # Shared Encoder & Decoder across modalities
        self.encoder = WeightTokenEncoder(input_dim=self.weight_dim, d_latent=d_latent)
        self.decoder = WeightTokenDecoder(output_dim=self.weight_dim, d_latent=d_latent)

    def encode(self, delta_w: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        delta_w: [Batch, L, 3, r, H]
        Returns dict containing:
          - Z_vis:  [Batch, L, d_latent]
          - Z_lang: [Batch, L, d_latent]
          - Z_act:  [Batch, L, d_latent]
          - Z_stacked: [Batch, L, 3, d_latent]
        """
        if delta_w.dim() == 4:
            delta_w = delta_w.unsqueeze(0)
        B, L, M, r, H = delta_w.shape
        assert M == 3, f"Expected 3 modalities (vis, lang, act), got {M}"

        # Flatten (r, H) -> (r * H)
        flat_w = delta_w.view(B, L, 3, self.weight_dim)

        z_vis = self.encoder(flat_w[:, :, 0, :])
        z_lang = self.encoder(flat_w[:, :, 1, :])
        z_act = self.encoder(flat_w[:, :, 2, :])

        z_stacked = torch.stack([z_vis, z_lang, z_act], dim=2) # [B, L, 3, d_latent]

        return {
            "vis": z_vis,
            "lang": z_lang,
            "act": z_act,
            "stacked": z_stacked,
        }

    def decode(self, latents: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        latents: dict with 'vis', 'lang', 'act' [Batch, L, d_latent] or 'stacked' [Batch, L, 3, d_latent]
        Returns: hat{Delta W} in R^[Batch, L, 3, r, H]
        """
        if "stacked" in latents and latents["stacked"] is not None:
            z_stacked = latents["stacked"]
            z_vis, z_lang, z_act = z_stacked[:, :, 0, :], z_stacked[:, :, 1, :], z_stacked[:, :, 2, :]
        else:
            z_vis, z_lang, z_act = latents["vis"], latents["lang"], latents["act"]

        B, L, _ = z_vis.shape
        rec_vis = self.decoder(z_vis).view(B, L, 1, self.rank, self.hidden_dim)
        rec_lang = self.decoder(z_lang).view(B, L, 1, self.rank, self.hidden_dim)
        rec_act = self.decoder(z_act).view(B, L, 1, self.rank, self.hidden_dim)

        rec_delta_w = torch.cat([rec_vis, rec_lang, rec_act], dim=2) # [B, L, 3, r, H]
        return rec_delta_w

    def forward(self, delta_w: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward autoencoding pass.
        Returns: (reconstructed_delta_w, latent_dict)
        """
        latents = self.encode(delta_w)
        rec_delta_w = self.decode(latents)
        return rec_delta_w, latents
