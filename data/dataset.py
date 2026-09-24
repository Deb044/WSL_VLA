"""
data/dataset.py

Dataset loader for LIBERO robotic demonstration trajectories.
Supports both real HDF5 datasets and synthetic demonstration generation for local dry-runs.
"""
import os
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset


class LiberoTaskDataset(Dataset):
    """
    Dataset representing demonstration trajectories for a single task.
    Yields:
      - vis_features: [img_feat_dim] (pre-extracted or projected camera observations)
      - lang_embed:   [lang_embed_dim] (sentence embedding of task instruction)
      - act_history:  [action_dim] (robot's previous action state)
      - target_act:   [action_dim] (ground-truth next teleoperated action)
    """
    def __init__(
        self,
        task_id: str,
        task_instruction: str,
        data_path: Optional[str] = None,
        num_synthetic_samples: int = 500,
        img_feat_dim: int = 64,
        lang_embed_dim: int = 384,
        action_dim: int = 7,
    ):
        self.task_id = task_id
        self.task_instruction = task_instruction
        self.img_feat_dim = img_feat_dim
        self.lang_embed_dim = lang_embed_dim
        self.action_dim = action_dim

        if data_path and os.path.exists(data_path):
            self._load_hdf5(data_path)
        else:
            self._generate_synthetic_demos(num_synthetic_samples)

    def _load_hdf5(self, hdf5_path: str):
        """Loads demonstration episodes from a real LIBERO HDF5 dataset."""
        import h5py
        self.samples = []
        with h5py.File(hdf5_path, "r") as f:
            data_grp = f["data"]
            for demo_key in data_grp.keys():
                demo = data_grp[demo_key]
                actions = demo["actions"][:]
                # For demonstration, extract agentview image features or states
                # In full pipeline, images are processed via SigLIP/ResNet
                num_steps = len(actions)
                for t in range(num_steps - 1):
                    # Mock projection of image observation
                    vis_feat = torch.randn(self.img_feat_dim)
                    act_hist = torch.tensor(actions[t], dtype=torch.float32)
                    target = torch.tensor(actions[t + 1], dtype=torch.float32)
                    self.samples.append((vis_feat, act_hist, target))

    def _generate_synthetic_demos(self, num_samples: int):
        """Generates deterministic synthetic demonstration data for on-device dry-runs."""
        g = torch.Generator().manual_seed(abs(hash(self.task_id)) % (2**31))
        
        # Consistent task instruction vector
        self.lang_vector = torch.randn(self.lang_embed_dim, generator=g)

        self.samples = []
        for _ in range(num_samples):
            vis = torch.randn(self.img_feat_dim, generator=g)
            act_hist = torch.randn(self.action_dim, generator=g)
            # Semi-deterministic target action conditioned on visual & historical features
            target = 0.5 * act_hist + 0.1 * vis[:self.action_dim] + 0.05 * torch.randn(self.action_dim, generator=g)
            self.samples.append((vis, act_hist, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vis, act_hist, target = self.samples[idx]
        return vis, self.lang_vector, act_hist, target

    def get_raw_trajectory_for_evidence(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns stacked demonstration frames and actions for task evidence extraction.
        """
        all_frames = torch.stack([s[0] for s in self.samples[:50]], dim=0) # 50 sample frames
        all_actions = torch.stack([s[2] for s in self.samples], dim=0)     # All actions
        return all_frames, all_actions
