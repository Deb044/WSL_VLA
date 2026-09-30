#!/usr/bin/env python3
"""
scripts/compare_base_vs_adapted.py

Legacy PyTorch proxy head-to-head smoke fixture:
Compares the unadapted Base VLA against the VLA equipped with task-specific
factorized LoRA adapters.
Evaluates Action MSE, Directional Cosine Similarity, and Gripper Precision.

Modes:
  - 'adaptation' (default): Evaluates task adaptation on the task demonstration manifold.
  - 'checkpoint': Directly injects pre-trained checkpoint weights from the Model Zoo.
"""
import os
import sys
import yaml
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wsl_vla.smoke.legacy_torch.data import LiberoTaskDataset
from wsl_vla.smoke.legacy_torch.models import build_vla_model
from wsl_vla.smoke_guard import require_explicit_smoke_test


def evaluate_loader(model, data_loader, device):
    """Evaluates Action MSE, 6-DoF heading cosine similarity, and gripper accuracy."""
    model.eval()
    total_loss, total_cos, total_grip, total_samples = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for vis, lang, act_hist, target in data_loader:
            vis = vis.to(device)
            lang = lang.to(device)
            act_hist = act_hist.to(device)
            target = target.to(device)

            pred = model(vis, lang, act_hist)
            total_loss += F.mse_loss(pred, target, reduction="sum").item()
            total_cos += F.cosine_similarity(pred[:, :6], target[:, :6], dim=-1).sum().item()
            total_grip += ((pred[:, 6] > 0.0) == (target[:, 6] > 0.0)).sum().item()
            total_samples += target.size(0)

    mse = total_loss / (total_samples * 7)
    cos = total_cos / total_samples
    grip = total_grip / total_samples
    return mse, cos, grip


def main():
    parser = argparse.ArgumentParser(description="Direct Head-to-Head Benchmark: Base VLA vs Adapted VLA")
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
        choices=["all", "libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="LIBERO suite to compare or 'all' for all 40 tasks",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="adaptation",
        choices=["adaptation", "checkpoint"],
        help="Evaluation mode: 'adaptation' (fine-tunes adapter on task) or 'checkpoint' (evaluates static checkpoint directly)",
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of optimizer steps for adaptation mode (default: 50)")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of demonstration samples per task")
    parser.add_argument("--smoke-test", action="store_true", help="Acknowledge this is the legacy PyTorch proxy")
    args = parser.parse_args()
    require_explicit_smoke_test(args.smoke_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("configs/vla_config.yaml") as f:
        vla_cfg = yaml.safe_load(f)
    with open("configs/tasks_config.yaml") as f:
        tasks_cfg = yaml.safe_load(f)

    vla = build_vla_model(vla_cfg).to(device)
    peft = vla.attach_factorized_lora(16, 32)

    if args.suite == "all":
        suites_to_run = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    else:
        suites_to_run = [args.suite]

    tasks = []
    for s_name in suites_to_run:
        for t in tasks_cfg["suites"][s_name]["tasks"]:
            tasks.append(t["id"])

    print("=" * 100)
    print(f" DIRECT HEAD-TO-HEAD BENCHMARK: BASE VLA vs. ADAPTED VLA ({len(tasks)} TASKS)")
    print(f" Mode: {args.mode.upper()} | Suites: {', '.join(suites_to_run)}")
    print("=" * 100)
    print(f"{'Task ID':<18} | {'Model Configuration':<28} | {'Action MSE':<10} | {'Cosine Sim':<10} | {'Gripper Acc'}")
    print("-" * 100)

    total_mse_base, total_mse_adapt = 0.0, 0.0
    total_cos_base, total_cos_adapt = 0.0, 0.0
    total_grip_base, total_grip_adapt = 0.0, 0.0

    zero_weights = torch.zeros(vla.num_layers, 3, 16, vla.hidden_dim)

    for idx, tid in enumerate(tasks, 1):
        ckpt_path = f"smoke_results/model_zoo/task_{tid}.pt"
        tname = tid
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            tname = ckpt.get("task_name", tid)

        ds = LiberoTaskDataset(
            task_id=tid,
            task_instruction=tname,
            data_dir="data/libero",
            num_synthetic_samples=args.num_samples,
        )
        dl = DataLoader(ds, batch_size=32, shuffle=False)

        # 1. Base VLA (Zero-Shot without Task LoRA)
        vla.inject_delta_w(zero_weights)
        mse_base, cos_base, grip_base = evaluate_loader(peft, dl, device)

        # 2. Adapted VLA
        if args.mode == "adaptation":
            # Task-specific weights adapted on task demonstration manifold
            peft.train()
            opt = torch.optim.AdamW([p for p in peft.parameters() if p.requires_grad], lr=1e-3)
            train_dl = DataLoader(ds, batch_size=16, shuffle=True)
            t_iter = iter(train_dl)
            for _ in range(args.steps):
                try:
                    b = next(t_iter)
                except StopIteration:
                    t_iter = iter(train_dl)
                    b = next(t_iter)
                v, l, ah, tgt = [x.to(device) for x in b]
                opt.zero_grad()
                loss = F.mse_loss(peft(v, l, ah), tgt)
                loss.backward()
                opt.step()
        else:
            # Static checkpoint weights directly injected
            vla.inject_delta_w(ckpt["delta_w"])

        mse_adapt, cos_adapt, grip_adapt = evaluate_loader(peft, dl, device)

        total_mse_base += mse_base
        total_mse_adapt += mse_adapt
        total_cos_base += cos_base
        total_cos_adapt += cos_adapt
        total_grip_base += grip_base
        total_grip_adapt += grip_adapt

        reduction = ((mse_base - mse_adapt) / mse_base) * 100
        print(f"[{idx:02d}/{len(tasks)}] {tid:<14} | {'1. Base VLA (Zero-Shot)':<28} | {mse_base:<10.4f} | {cos_base:<10.3f} | {grip_base*100:.1f}%")
        print(f"         {'':<14} | {'2. Adapted VLA (Task Weights)':<28} | {mse_adapt:<10.4f} | {cos_adapt:<10.3f} | {grip_adapt*100:.1f}%")
        print(f"         {'':<14} | {'   -> Delta':<28} | -{reduction:<9.1f}% | +{cos_adapt - cos_base:<9.3f} | +{(grip_adapt-grip_base)*100:.1f}%")
        print("-" * 100)

    num_t = len(tasks)
    print("=" * 100)
    print(f" OVERALL BENCHMARK SUMMARY ACROSS {num_t} TASKS")
    print("=" * 100)
    avg_mb = total_mse_base / num_t
    avg_ma = total_mse_adapt / num_t
    avg_cb = total_cos_base / num_t
    avg_ca = total_cos_adapt / num_t
    avg_gb = total_grip_base / num_t
    avg_ga = total_grip_adapt / num_t
    print(f"Base VLA Mean MSE     : {avg_mb:.4f}  |  Cosine Sim: {avg_cb:.3f}  |  Gripper Acc: {avg_gb*100:.1f}%")
    print(f"Adapted VLA Mean MSE  : {avg_ma:.4f}  |  Cosine Sim: {avg_ca:.3f}  |  Gripper Acc: {avg_ga*100:.1f}%")
    print(f"Overall Action Error Reduction: -{((avg_mb - avg_ma) / avg_mb) * 100:.1f}%")
    print(f"Overall Heading Alignment    : +{(avg_ca - avg_cb):.3f}")
    print(f"Overall Gripper Precision    : +{((avg_ga - avg_gb) * 100):.1f}%")
    print("=" * 100)


if __name__ == "__main__":
    main()
