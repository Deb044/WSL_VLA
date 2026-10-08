#!/usr/bin/env python3
"""
scripts/train/train_alignment.py

Legacy PyTorch alignment smoke fixture; not a Methodology 1 research command.
Loads the Model Zoo population checkpoints:
    Delta W in R^[N_tasks, L, 3, r, H]
and multi-modal evidence tuples:
    M = {e_vis, e_lang, e_act}
and exercises the proxy autoencoder and contrastive-aligner plumbing.

Objective:
    L_total = L_recon + lambda_vis L_align^(vis) + lambda_lang L_align^(lang) + lambda_act L_align^(act) + lambda_aux L_aux
"""
import os
import sys
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Repo root
def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.smoke.legacy_torch.models import FactorizedWeightAutoencoder, ModalityContrastiveAligner
from wsl_vla.smoke_guard import require_explicit_smoke_test, write_smoke_marker


class ModelZooPopulationDataset(Dataset):
    """Dataset of trained task LoRA weights and multi-modal task evidence."""
    def __init__(self, checkpoint_dir: str):
        self.samples = []
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory '{checkpoint_dir}' does not exist.")

        ckpt_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("task_") and f.endswith(".pt")])
        if not ckpt_files:
            raise RuntimeError(f"No task checkpoints found in '{checkpoint_dir}'. Train the Model Zoo first.")

        print(f"Loading {len(ckpt_files)} Model Zoo task checkpoints for Alignment...")
        for idx, fname in enumerate(ckpt_files):
            fpath = os.path.join(checkpoint_dir, fname)
            data = torch.load(fpath, map_location="cpu")
            self.samples.append({
                "task_id": data["task_id"],
                "task_label": idx,
                "delta_w": data["delta_w"], # [L, 3, r, H]
                "e_vis": data["e_vis"],     # [128]
                "e_lang": data["e_lang"],   # [384]
                "e_act": data["e_act"],     # [28]
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_zoo_batch(batch):
    return {
        "task_id": [b["task_id"] for b in batch],
        "task_label": torch.tensor([b["task_label"] for b in batch], dtype=torch.long),
        "delta_w": torch.stack([b["delta_w"] for b in batch], dim=0),
        "e_vis": torch.stack([b["e_vis"] for b in batch], dim=0),
        "e_lang": torch.stack([b["e_lang"] for b in batch], dim=0),
        "e_act": torch.stack([b["e_act"] for b in batch], dim=0),
    }


def train_alignment(
    checkpoint_dir: str = "./smoke_results/model_zoo",
    output_dir: str = "./smoke_results/weight_alignment",
    num_epochs: int = 150,
    lr: float = 5e-4,
    d_latent: int = 128,
    lambda_vis: float = 1.0,
    lambda_lang: float = 1.0,
    lambda_act: float = 1.0,
    lambda_aux: float = 0.2,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" Training Weight-Space Alignment (Methodology 1.1 & 1.2)")
    print(f" Device: {device} | Output: {output_dir}")
    print("=" * 80)

    dataset = ModelZooPopulationDataset(checkpoint_dir)
    sample_dw = dataset[0]["delta_w"]
    num_layers, _, rank, hidden_dim = sample_dw.shape
    num_tasks = len(dataset)

    # Instantiate Autoencoder & Aligner
    autoencoder = FactorizedWeightAutoencoder(
        num_layers=num_layers,
        rank=rank,
        hidden_dim=hidden_dim,
        d_latent=d_latent,
    ).to(device)

    aligner = ModalityContrastiveAligner(
        d_latent=d_latent,
        vis_dim=dataset[0]["e_vis"].shape[-1],
        lang_dim=dataset[0]["e_lang"].shape[-1],
        act_dim=dataset[0]["e_act"].shape[-1],
        num_classes=num_tasks,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(autoencoder.parameters()) + list(aligner.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    # Batch size (include all available tasks in population for contrastive negatives)
    batch_size = min(len(dataset), 64)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_zoo_batch,
    )

    print(f"Model Configuration -> Layers: {num_layers}, Rank: {rank}, Hidden: {hidden_dim}, Latent D: {d_latent}")
    print(f"Population Size: {num_tasks} tasks | Batch size: {batch_size}")

    for epoch in range(1, num_epochs + 1):
        autoencoder.train()
        aligner.train()
        epoch_metrics = {}

        for batch in dataloader:
            delta_w = batch["delta_w"].to(device)
            evidence = {
                "e_vis": batch["e_vis"].to(device),
                "e_lang": batch["e_lang"].to(device),
                "e_act": batch["e_act"].to(device),
            }
            labels = batch["task_label"].to(device)

            optimizer.zero_grad()

            # 1. Forward autoencoder
            rec_delta_w, latents = autoencoder(delta_w)

            # 2. Compute total contrastive + reconstruction + aux loss
            total_loss, metrics = aligner.compute_total_loss(
                rec_delta_w=rec_delta_w,
                target_delta_w=delta_w,
                latents=latents,
                evidence=evidence,
                task_labels=labels,
                lambda_vis=lambda_vis,
                lambda_lang=lambda_lang,
                lambda_act=lambda_act,
                lambda_aux=lambda_aux,
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(autoencoder.parameters()) + list(aligner.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

        if epoch % 25 == 0 or epoch == num_epochs:
            print(
                f"[Epoch {epoch:03d}/{num_epochs}] Total Loss: {epoch_metrics['loss_total']:.4f} | "
                f"Recon: {epoch_metrics['loss_recon']:.4f} | "
                f"Align(V/L/A): {epoch_metrics['loss_align_vis']:.3f}/{epoch_metrics['loss_align_lang']:.3f}/{epoch_metrics['loss_align_act']:.3f} | "
                f"Tau(V/L/A): {epoch_metrics['tau_vis']:.3f}/{epoch_metrics['tau_lang']:.3f}/{epoch_metrics['tau_act']:.3f}"
            )

    # Save Aligned Autoencoder & Aligner Checkpoint
    save_path = os.path.join(output_dir, "aligned_weight_autoencoder.pt")
    torch.save({
        "autoencoder_state": autoencoder.state_dict(),
        "aligner_state": aligner.state_dict(),
        "config": {
            "num_layers": num_layers,
            "rank": rank,
            "hidden_dim": hidden_dim,
            "d_latent": d_latent,
            "num_tasks": num_tasks,
        },
    }, save_path)
    print(f"\n[SUCCESS] Aligned Weight Autoencoder saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./smoke_results/model_zoo")
    parser.add_argument("--output_dir", type=str, default="./smoke_results/weight_alignment")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--smoke-test", action="store_true", help="Acknowledge this is the legacy PyTorch proxy")
    args = parser.parse_args()
    require_explicit_smoke_test(args.smoke_test, output_directories=(args.output_dir,))
    write_smoke_marker(args.output_dir, command="scripts/train/train_alignment.py")

    train_alignment(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        lr=args.lr,
    )
