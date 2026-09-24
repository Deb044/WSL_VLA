"""
models/contrastive_alignment.py

Implements Section 1.2 of Methodology:
Modality-Specific Contrastive Alignment.

Aligns component-factorized weight latents Z_m in R^[L, d_latent] against
modality-specific prompt evidence e_m:
  - e_vis:  DeepSets demonstration frames embedding [128]
  - e_lang: Natural language instruction sentence embedding [384]
  - e_act:  Action-chunk distribution summary statistics [28]

Features:
  1. Learnable temperature parameters tau_m per modality.
  2. Independent bidirectional InfoNCE alignment loss per modality across layer tokens.
  3. Auxiliary task classification heads on prompt embeddings for supervision.
"""
from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptProjector(nn.Module):
    """Projects task evidence e_m to the shared latent dimension d_latent."""
    def __init__(self, in_dim: int, d_latent: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_latent * 2),
            nn.LayerNorm(d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent),
            nn.LayerNorm(d_latent),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)


class ModalityContrastiveAligner(nn.Module):
    """
    Manages the 3 modality-specific prompt projectors, learnable temperatures tau_m,
    auxiliary task classifiers, and computes L_align^(m) and L_total.
    """
    def __init__(
        self,
        d_latent: int = 128,
        vis_dim: int = 128,
        lang_dim: int = 384,
        act_dim: int = 28,
        num_classes: int = 40,
        init_tau: float = 0.07,
    ):
        super().__init__()
        self.d_latent = d_latent

        # Dedicated prompt projectors per modality
        self.proj_vis = PromptProjector(vis_dim, d_latent)
        self.proj_lang = PromptProjector(lang_dim, d_latent)
        self.proj_act = PromptProjector(act_dim, d_latent)

        # Learnable log-temperatures (log(tau)) per modality
        self.log_tau_vis = nn.Parameter(torch.tensor(math.log(init_tau)))
        self.log_tau_lang = nn.Parameter(torch.tensor(math.log(init_tau)))
        self.log_tau_act = nn.Parameter(torch.tensor(math.log(init_tau)))

        # Auxiliary task-classification heads on prompt embeddings (WeightCLIP style)
        self.aux_cls_vis = nn.Linear(d_latent, num_classes)
        self.aux_cls_lang = nn.Linear(d_latent, num_classes)
        self.aux_cls_act = nn.Linear(d_latent, num_classes)

    def get_temperature(self, modality: str) -> torch.Tensor:
        if modality == "vis":
            return torch.clamp(self.log_tau_vis.exp(), min=0.01, max=1.0)
        elif modality == "lang":
            return torch.clamp(self.log_tau_lang.exp(), min=0.01, max=1.0)
        elif modality == "act":
            return torch.clamp(self.log_tau_act.exp(), min=0.01, max=1.0)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def project_evidence(self, evidence: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        evidence: dict with 'e_vis' [B, 128], 'e_lang' [B, 384], 'e_act' [B, 28]
        Returns: dict with projected embeddings in R^[B, d_latent]
        """
        return {
            "vis": self.proj_vis(evidence["e_vis"]),
            "lang": self.proj_lang(evidence["e_lang"]),
            "act": self.proj_act(evidence["e_act"]),
        }

    def compute_modality_align_loss(
        self,
        z_tokens: torch.Tensor,
        e_proj: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes bidirectional InfoNCE loss between weight token sequence and prompt embedding:
            z_tokens: [B, L, d_latent]
            e_proj:   [B, d_latent]
            tau:      scalar temperature
        """
        B, L, D = z_tokens.shape

        # Normalize representations
        z_norm = F.normalize(z_tokens, p=2, dim=-1) # [B, L, D]
        e_norm = F.normalize(e_proj, p=2, dim=-1)   # [B, D]

        # 1. Weight-to-Evidence Direction: For each layer token t, match against batch evidence
        # Similarity: [B, L, B_keys] = z_norm @ e_norm.T / tau
        sim_z2e = torch.einsum("bld,kd->blk", z_norm, e_norm) / tau # [B, L, B]

        # Target index is batch index i for each sample i
        targets = torch.arange(B, device=z_tokens.device) # [B]
        targets_expanded = targets.view(B, 1).expand(B, L) # [B, L]

        loss_z2e = F.cross_entropy(sim_z2e.view(B * L, B), targets_expanded.reshape(-1))

        # 2. Evidence-to-Weight Direction (Reverse direction):
        # Average pooled weight representation across layers: [B, D]
        z_pooled = F.normalize(torch.mean(z_norm, dim=1), p=2, dim=-1) # [B, D]
        sim_e2z = torch.matmul(e_norm, z_pooled.T) / tau               # [B, B]
        loss_e2z = F.cross_entropy(sim_e2z, targets)

        loss_bidirectional = (loss_z2e + loss_e2z) / 2.0
        return loss_bidirectional

    def compute_total_loss(
        self,
        rec_delta_w: torch.Tensor,
        target_delta_w: torch.Tensor,
        latents: Dict[str, torch.Tensor],
        evidence: Dict[str, torch.Tensor],
        task_labels: Optional[torch.Tensor] = None,
        lambda_vis: float = 1.0,
        lambda_lang: float = 1.0,
        lambda_act: float = 1.0,
        lambda_aux: float = 0.2,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates L_total = L_recon + lambda_vis L_align^(vis) + lambda_lang L_align^(lang) + lambda_act L_align^(act) + L_aux
        """
        # 1. Reconstruction Loss (Smooth L1 / Huber Loss on Delta W weights)
        loss_recon = F.smooth_l1_loss(rec_delta_w, target_delta_w)

        # 2. Project evidence embeddings
        p_ev = self.project_evidence(evidence)

        # 3. Modality-Specific Alignment Losses
        loss_align_vis = self.compute_modality_align_loss(latents["vis"], p_ev["vis"], self.get_temperature("vis"))
        loss_align_lang = self.compute_modality_align_loss(latents["lang"], p_ev["lang"], self.get_temperature("lang"))
        loss_align_act = self.compute_modality_align_loss(latents["act"], p_ev["act"], self.get_temperature("act"))

        # 4. Auxiliary Task Classification Loss
        loss_aux = torch.tensor(0.0, device=rec_delta_w.device)
        if task_labels is not None:
            logits_v = self.aux_cls_vis(p_ev["vis"])
            logits_l = self.aux_cls_lang(p_ev["lang"])
            logits_a = self.aux_cls_act(p_ev["act"])
            loss_aux = (
                F.cross_entropy(logits_v, task_labels) +
                F.cross_entropy(logits_l, task_labels) +
                F.cross_entropy(logits_a, task_labels)
            ) / 3.0

        # Total combined loss
        total_loss = (
            loss_recon +
            lambda_vis * loss_align_vis +
            lambda_lang * loss_align_lang +
            lambda_act * loss_align_act +
            lambda_aux * loss_aux
        )

        metrics = {
            "loss_total": total_loss.item(),
            "loss_recon": loss_recon.item(),
            "loss_align_vis": loss_align_vis.item(),
            "loss_align_lang": loss_align_lang.item(),
            "loss_align_act": loss_align_act.item(),
            "loss_aux": loss_aux.item(),
            "tau_vis": self.get_temperature("vis").item(),
            "tau_lang": self.get_temperature("lang").item(),
            "tau_act": self.get_temperature("act").item(),
        }

        return total_loss, metrics
