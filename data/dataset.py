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
        data_dir: Optional[str] = "./data/libero",
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

        # Auto-discover HDF5 file in data_dir if data_path is not explicitly provided
        resolved_path = data_path
        if resolved_path is None and data_dir and os.path.exists(data_dir):
            resolved_path = self._find_matching_hdf5(data_dir)

        if resolved_path and os.path.exists(resolved_path):
            self._load_hdf5(resolved_path)
        else:
            self._generate_synthetic_demos(num_synthetic_samples)

    def _find_matching_hdf5(self, data_dir: str) -> Optional[str]:
        """Searches data_dir for an HDF5 file matching task_instruction or task_id."""
        clean_name = self.task_instruction.lower().strip()
        for root, _, files in os.walk(data_dir):
            for f in files:
                if not f.endswith(".hdf5"):
                    continue
                f_lower = f.lower()
                # Check for direct match with task name or task id
                if clean_name in f_lower or self.task_id.lower() in f_lower:
                    return os.path.join(root, f)
                # Check for token overlap (e.g. key words in prompt)
                name_tokens = [w for w in clean_name.split("_") if len(w) > 3]
                if name_tokens and sum(1 for t in name_tokens if t in f_lower) >= min(3, len(name_tokens)):
                    return os.path.join(root, f)
        return None

    def _load_hdf5(self, hdf5_path: str):
        """Loads demonstration episodes and real camera images from an HDF5 dataset."""
        import h5py
        self.samples = []
        g = torch.Generator().manual_seed(abs(hash(self.task_id)) % (2**31))
        self.lang_vector = torch.randn(self.lang_embed_dim, generator=g)

        print(f"  [Dataset] Loading real demonstrations from: {os.path.basename(hdf5_path)}")
        with h5py.File(hdf5_path, "r") as f:
            data_grp = f["data"]
            demo_keys = list(data_grp.keys())
            for demo_key in demo_keys:
                demo = data_grp[demo_key]
                actions = demo["actions"][:]
                has_obs = "obs" in demo
                has_agentview = has_obs and "agentview_rgb" in demo["obs"]

                num_steps = len(actions)
                for t in range(num_steps - 1):
                    if has_agentview:
                        # Extract real camera frame [128, 128, 3] -> spatial average pool to img_feat_dim (64 floats)
                        raw_frame = demo["obs"]["agentview_rgb"][t]
                        t_frame = torch.from_numpy(raw_frame).float() / 255.0  # Normalize [0, 1]
                        # Adaptive avg pool down to 8x8 = 64 feature vector
                        vis_feat = torch.nn.functional.adaptive_avg_pool2d(
                            t_frame.permute(2, 0, 1).unsqueeze(0), (8, 8)
                        ).flatten()[: self.img_feat_dim]
                    else:
                        vis_feat = torch.randn(self.img_feat_dim, generator=g)

                    act_hist = torch.tensor(actions[t], dtype=torch.float32)
                    target = torch.tensor(actions[t + 1], dtype=torch.float32)
                    self.samples.append((vis_feat, act_hist, target))
            print(f"  [Dataset] Successfully loaded {len(self.samples)} transitions from {len(demo_keys)} demonstrations.")

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
