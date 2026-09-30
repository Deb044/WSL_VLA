"""
models/evidence_extractor.py

Implements task evidence extractors for the three modalities:
  1. e_vis:  DeepSets permutation-invariant encoder over demonstration camera frames.
  2. e_lang: Dense sentence embedding over the task's natural language instruction.
  3. e_act:  Set encoder over summary statistics of the task's action trajectory distribution.
"""
from typing import Dict, Optional
import torch
import torch.nn as nn


class DeepSetsVisionEncoder(nn.Module):
    """
    Permutation-invariant encoder over demonstration camera frames.
    Maps an arbitrary set of N demonstration frames to a fixed-size embedding.
    Formula: e_vis = rho( 1/N sum_i phi(frame_i) )
    """
    def __init__(self, in_features: int = 64, embed_dim: int = 128):
        super().__init__()
        # phi: per-frame feature projector
        self.phi = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # rho: set aggregator
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [N_frames, in_features]
        Returns: [embed_dim]
        """
        if frames.dim() == 1:
            frames = frames.unsqueeze(0)
        projected = self.phi(frames)        # [N, embed_dim]
        pooled = torch.mean(projected, dim=0) # [embed_dim]
        return self.rho(pooled)


class TaskEvidenceExtractor:
    """
    Extracts the multi-modal evidence tuple M^(k) = {e_vis, e_lang, e_act}
    used for contrastive weight-space alignment.
    """
    def __init__(self, device: str = "cpu", vis_in_dim: int = 64, vis_embed_dim: int = 128):
        self.device = device
        self.vis_encoder = DeepSetsVisionEncoder(in_features=vis_in_dim, embed_dim=vis_embed_dim).to(device)
        self.vis_encoder.eval()

        self._sentence_model = None

    def _get_sentence_model(self):
        """Lazy loader for sentence-transformers model."""
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._sentence_model = "fallback"
        return self._sentence_model

    def encode_text(self, text: str) -> torch.Tensor:
        """Encodes text to a 384-dimensional embedding."""
        model = self._get_sentence_model()
        if model != "fallback":
            with torch.no_grad():
                emb = model.encode(text, convert_to_numpy=False)
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb, dtype=torch.float32)
                return emb.cpu()
        else:
            # Deterministic hash projection fallback (useful for offline/network-free unit tests)
            g = torch.Generator().manual_seed(abs(hash(text)) % (2**31))
            return torch.randn(384, generator=g, dtype=torch.float32)

    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Computes summary statistics over action trajectories:
          - Mean action vector [action_dim]
          - Action variance/std [action_dim]
          - Action velocity magnitude [action_dim]
          - Action smoothness/jerk [action_dim]
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        mean_act = torch.mean(actions, dim=0)
        std_act = torch.std(actions, dim=0) if actions.size(0) > 1 else torch.zeros_like(mean_act)

        if actions.size(0) > 1:
            diffs = torch.diff(actions, dim=0)
            velocity = torch.mean(torch.abs(diffs), dim=0)
            if diffs.size(0) > 1:
                jerk = torch.mean(torch.abs(torch.diff(diffs, dim=0)), dim=0)
            else:
                jerk = torch.zeros_like(mean_act)
        else:
            velocity = torch.zeros_like(mean_act)
            jerk = torch.zeros_like(mean_act)

        e_act = torch.cat([mean_act, std_act, velocity, jerk], dim=-1) # [action_dim * 4]
        return e_act

    @torch.no_grad()
    def extract_evidence(
        self,
        demo_frames: torch.Tensor,
        task_instruction: str,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the complete evidence tuple for task k.
        """
        e_vis = self.vis_encoder(demo_frames.to(self.device)).cpu()
        e_lang = self.encode_text(task_instruction)
        e_act = self.encode_actions(actions)

        return {
            "e_vis": e_vis,
            "e_lang": e_lang,
            "e_act": e_act,
        }
