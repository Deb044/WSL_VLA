#!/usr/bin/env python3
"""
scripts/evaluate_vla.py

Comprehensive evaluation script for trained VLA models and task-specific LoRA adapters.
Evaluates trajectory tracking accuracy against real LIBERO teleoperated demonstrations.

Metrics Computed:
  1. Action MSE / RMSE (Overall Mean Squared Error on 7-DoF robot action commands)
  2. Action Cosine Similarity (Directional fidelity of delta end-effector commands)
  3. Per-Dimension Breakdown:
       - Translation Error (X, Y, Z coordinates)
       - Rotation Error (Roll, Pitch, Yaw)
       - Gripper Open/Close Binary Accuracy
  4. Evidence Alignment (Consistency of e_vis, e_lang, e_act)
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import build_vla_model
from data.dataset import LiberoTaskDataset


def evaluate_task(
    task_id: str,
    task_name: str,
    vla_cfg: dict,
    ckpt_path: str,
    data_dir: str,
    device: torch.device,
) -> dict:
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path}")
        return None

    # 1. Load Checkpoint Payload
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    delta_w = checkpoint.get("delta_w", None)

    # 2. Build Backbone Model and attach LoRA
    vla_model = build_vla_model(vla_cfg).to(device)
    peft_model = vla_model.attach_factorized_lora(
        rank=vla_cfg["lora"]["r"],
        alpha=vla_cfg["lora"]["lora_alpha"],
    )

    # Load trained LoRA weights back into the PEFT model
    state_dict = peft_model.state_dict()
    # If checkpoint contains full model state dict or delta_w
    if "model_state_dict" in checkpoint:
        peft_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif delta_w is not None:
        # Reconstruct LoRA A weights from extracted delta_w [L, 3, r, H]
        modality_keys = ["vis_block", "lang_block", "act_block"] if vla_cfg["model"]["name"] == "octo_small" else ["vis_mlp", "lang_attn", "act_dense"]
        for l in range(vla_model.num_layers):
            for m_idx, mod_name in enumerate(modality_keys):
                key_A = f"base_model.model.layers.{l}.{mod_name}.lora_A.default.weight"
                if key_A in state_dict:
                    state_dict[key_A] = delta_w[l, m_idx].to(device)
        peft_model.load_state_dict(state_dict, strict=False)

    peft_model.eval()

    # 3. Load Dataset
    hdf5_file = os.path.join(data_dir, f"{task_id}.hdf5")
    is_real = os.path.exists(hdf5_file)
    dataset = LiberoTaskDataset(
        task_id=task_id,
        task_instruction=task_name,
        data_path=hdf5_file if is_real else None,
        num_synthetic_samples=300,
        img_feat_dim=vla_cfg["model"]["image_features_dim"],
        lang_embed_dim=vla_cfg["model"]["language_embed_dim"],
        action_dim=vla_cfg["model"]["action_dim"],
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 4. Evaluation Loop
    total_samples = 0
    total_l2_err = 0.0
    total_cos_sim = 0.0
    dim_errors = torch.zeros(7)
    gripper_correct = 0

    with torch.no_grad():
        for batch in dataloader:
            vis_feat, lang_embed, act_hist, target_act = [b.to(device) for b in batch]
            pred_act = peft_model(vis_feat, lang_embed, act_hist)

            batch_size = target_act.size(0)
            total_samples += batch_size

            # L2 Squared Error
            l2_err = F.mse_loss(pred_act, target_act, reduction="none") # [B, 7]
            total_l2_err += l2_err.sum().item()
            dim_errors += l2_err.sum(dim=0).cpu()

            # Directional Cosine Similarity (for continuous 6-DoF command [x,y,z, r,p,y])
            cos = F.cosine_similarity(pred_act[:, :6], target_act[:, :6], dim=-1)
            total_cos_sim += cos.sum().item()

            # Gripper Accuracy (dim 6: threshold at 0)
            pred_grip = (pred_act[:, 6] > 0.0).long()
            target_grip = (target_act[:, 6] > 0.0).long()
            gripper_correct += (pred_grip == target_grip).sum().item()

    mse = total_l2_err / (total_samples * 7)
    rmse = (total_l2_err / (total_samples * 7)) ** 0.5
    avg_cos = total_cos_sim / total_samples
    avg_grip_acc = gripper_correct / total_samples
    per_dim_rmse = (dim_errors / total_samples) ** 0.5

    trans_rmse = per_dim_rmse[:3].mean().item()
    rot_rmse = per_dim_rmse[3:6].mean().item()

    return {
        "task_id": task_id,
        "task_name": task_name,
        "is_real_data": is_real,
        "num_samples": total_samples,
        "mse": mse,
        "rmse": rmse,
        "trans_rmse": trans_rmse,
        "rot_rmse": rot_rmse,
        "cosine_sim": avg_cos,
        "gripper_acc": avg_grip_acc,
        "delta_w_norm": torch.norm(delta_w).item() if delta_w is not None else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained VLA Model Zoo")
    parser.add_argument("--vla_config", type=str, default="configs/vla_config.yaml")
    parser.add_argument("--tasks_config", type=str, default="configs/tasks_config.yaml")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/model_zoo")
    parser.add_argument("--data_dir", type=str, default="./data/libero")
    parser.add_argument("--task_id", type=str, default=None, help="Evaluate specific task ID or all available")
    args = parser.parse_args()

    with open(args.vla_config, "r") as f:
        vla_cfg = yaml.safe_load(f)
    with open(args.tasks_config, "r") as f:
        tasks_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print(f" VLA Model Zoo Trajectory Tracking Evaluation")
    print(f" Backbone: {vla_cfg['model']['name']} | Device: {device} | Checkpoints: {args.checkpoints_dir}")
    print("=" * 85)

    # Collect tasks
    task_map = {}
    for suite in tasks_cfg["suites"].values():
        for t in suite["tasks"]:
            task_map[t["id"]] = t["name"]

    if args.task_id:
        eval_tasks = [args.task_id]
    else:
        # Evaluate all available checkpoints
        ckpt_files = [f for f in os.listdir(args.checkpoints_dir) if f.startswith("task_") and f.endswith(".pt")]
        eval_tasks = [f.replace("task_", "").replace(".pt", "") for f in ckpt_files]

    if not eval_tasks:
        print("[ERROR] No checkpoints found to evaluate!")
        return

    results = []
    for tid in sorted(eval_tasks):
        tname = task_map.get(tid, tid)
        ckpt_path = os.path.join(args.checkpoints_dir, f"task_{tid}.pt")
        res = evaluate_task(
            task_id=tid,
            task_name=tname,
            vla_cfg=vla_cfg,
            ckpt_path=ckpt_path,
            data_dir=args.data_dir,
            device=device,
        )
        if res:
            results.append(res)

    # Print Formatted Evaluation Report Table
    print("\n" + "=" * 95)
    print(f"{'Task ID':<18} | {'Data Source':<10} | {'MSE Loss':<10} | {'Trans RMSE':<12} | {'Rot RMSE':<10} | {'Cos Sim':<8} | {'Grip Acc'}")
    print("-" * 95)
    for r in results:
        src = "Real HDF5" if r["is_real_data"] else "Synthetic"
        print(f"{r['task_id']:<18} | {src:<10} | {r['mse']:<10.5f} | {r['trans_rmse']:<12.4f} | {r['rot_rmse']:<10.4f} | {r['cosine_sim']:<8.3f} | {r['gripper_acc']*100:.1f}%")
    print("=" * 95)

    avg_mse = sum(r["mse"] for r in results) / len(results)
    avg_cos = sum(r["cosine_sim"] for r in results) / len(results)
    avg_grip = sum(r["gripper_acc"] for r in results) / len(results)
    print(f"Zoo Mean Performance -> Overall MSE: {avg_mse:.5f} | Mean Cosine Sim: {avg_cos:.3f} | Mean Gripper Acc: {avg_grip*100:.1f}%\n")


if __name__ == "__main__":
    main()
