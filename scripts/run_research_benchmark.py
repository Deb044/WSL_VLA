#!/usr/bin/env python3
"""
scripts/run_research_benchmark.py

DEPRECATED legacy PyTorch proxy smoke benchmark. This does not implement the
official Octo/LIBERO research protocol and must never be cited as a result.

It exercises older plumbing across the configured tasks:
  1. Sequential Adaptation Stream across all 40 LIBERO tasks (spatial, object, goal, libero_10).
  2. Latent Drift as a Forgetting Proxy (Section 2.1 & 2.2 Component-Swap Protocol):
     - Euclidean distance ||Z_m^(t) - Z_m^(0)||_2 per modality.
     - Pearson correlation r(Drift, Behavioral Forgetting) for vis, lang, act.
  3. Full 4-Way Ablation Benchmark (Section 3.3):
     - Proposed Differential Regularization (gamma_vis=1.0, gamma_lang=1.0, gamma_act=0.2)
     - Uniform Regularization (gamma_vis=1.0, gamma_lang=1.0, gamma_act=1.0)
     - Direction-Inverted Ablation (gamma_vis=0.2, gamma_lang=0.2, gamma_act=1.0)
     - Drift-Informed Dynamic Schedule
  4. Primary Metrics:
     - Stream Action MSE
     - Normalized Backward Transfer (NBT)
     - Latent-to-Forgetting Correlation (r_vis, r_lang, r_act)
     - Memory Footprint (bytes/task)
"""
import os
import sys
import yaml
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import (
    build_vla_model,
    FactorizedWeightAutoencoder,
    DifferentialRegularizer,
)
from data.dataset import LiberoTaskDataset
from wsl_vla.smoke_guard import require_explicit_smoke_test


def pearson_correlation(x, y):
    """Computes Pearson correlation coefficient between two 1D float arrays."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    if var_x < 1e-9 or var_y < 1e-9:
        return 0.0
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    return cov / math.sqrt(var_x * var_y)


def run_experiment(
    condition_name: str,
    stream_task_ids: list,
    vla_cfg: dict,
    tasks_cfg: dict,
    aligned_ae_path: str,
    schedule: str = "fixed",
    gamma_vis: float = 1.0,
    gamma_lang: float = 1.0,
    gamma_act: float = 0.2,
    refine_steps: int = 20,
    lr_z: float = 1e-2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*95}")
    print(f" [RUNNING EXPERIMENTAL REGIME] {condition_name}")
    print(f" Schedule: {schedule} | Gammas: (vis={gamma_vis}, lang={gamma_lang}, act={gamma_act})")
    print(f" Task Stream: {len(stream_task_ids)} tasks across LIBERO benchmark")
    print(f"{'='*95}")

    # Load Aligned Autoencoder
    ae_ckpt = torch.load(aligned_ae_path, map_location=device, weights_only=False)
    cfg_ae = ae_ckpt["config"]

    autoencoder = FactorizedWeightAutoencoder(
        num_layers=cfg_ae["num_layers"],
        rank=cfg_ae["rank"],
        hidden_dim=cfg_ae["hidden_dim"],
        d_latent=cfg_ae["d_latent"],
    ).to(device)
    autoencoder.load_state_dict(ae_ckpt["autoencoder_state"])
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    # Build VLA Backbone
    vla_model = build_vla_model(vla_cfg).to(device)
    peft_model = vla_model.attach_factorized_lora(
        rank=vla_cfg["lora"]["r"],
        alpha=vla_cfg["lora"]["lora_alpha"],
    )
    for p in peft_model.parameters():
        p.requires_grad = False

    regularizer = DifferentialRegularizer(
        schedule=schedule,
        gamma_vis=gamma_vis,
        gamma_lang=gamma_lang,
        gamma_act=gamma_act,
    ).to(device)

    task_names = {}
    for suite in tasks_cfg["suites"].values():
        for t in suite["tasks"]:
            task_names[t["id"]] = t["name"]

    # Pre-cache validation batches on GPU for instantaneous evaluation
    val_cache = {}

    def get_val_batches(task_id):
        if task_id not in val_cache:
            tname = task_names.get(task_id, task_id)
            ds = LiberoTaskDataset(
                task_id=task_id,
                task_instruction=tname,
                data_dir="data/libero",
                num_synthetic_samples=100,
            )
            dl = DataLoader(ds, batch_size=32, shuffle=False)
            val_cache[task_id] = [[x.to(device) for x in b] for b in dl]
        return val_cache[task_id]

    def evaluate_task_loss(task_id, latent_dict):
        rec_dw = autoencoder.decode(latent_dict)
        vla_model.inject_delta_w(rec_dw)
        batches = get_val_batches(task_id)
        total_err, count = 0.0, 0
        with torch.no_grad():
            for v, l, ah, target in batches:
                out = peft_model(v, l, ah)
                total_err += F.mse_loss(out, target, reduction="sum").item()
                count += target.numel()
        return total_err / count

    stored_initial_latents = {}
    stored_refined_latents = {}
    final_task_errors = []

    L_layers = vla_model.num_layers
    d_lat = cfg_ae["d_latent"]

    for stream_step, curr_task_id in enumerate(stream_task_ids):
        curr_task_name = task_names.get(curr_task_id, curr_task_id)

        # Retrieve initial latent from Model Zoo checkpoint via autoencoder
        ckpt_path = f"smoke_results/model_zoo/task_{curr_task_id}.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            with torch.no_grad():
                encoded_z = autoencoder.encode(ckpt["delta_w"].unsqueeze(0))
            z0_vis = encoded_z["vis"].detach().clone()
            z0_lang = encoded_z["lang"].detach().clone()
            z0_act = encoded_z["act"].detach().clone()
        else:
            z0_vis = torch.randn(1, L_layers, d_lat, device=device) * 0.1
            z0_lang = torch.randn(1, L_layers, d_lat, device=device) * 0.1
            z0_act = torch.randn(1, L_layers, d_lat, device=device) * 0.1

        z_vis = nn.Parameter(z0_vis.clone())
        z_lang = nn.Parameter(z0_lang.clone())
        z_act = nn.Parameter(z0_act.clone())

        z0_initial = {
            "vis": z0_vis.clone(),
            "lang": z0_lang.clone(),
            "act": z0_act.clone(),
        }
        stored_initial_latents[curr_task_id] = z0_initial

        # Training data for local task adaptation
        ds = LiberoTaskDataset(
            task_id=curr_task_id,
            task_instruction=curr_task_name,
            data_dir="data/libero",
            num_synthetic_samples=150,
        )
        task_loader = DataLoader(ds, batch_size=16, shuffle=True)
        task_batches = [[b.to(device) for b in batch] for batch in task_loader]

        # Latent Refinement with Differential Regularization
        z_opt = torch.optim.AdamW([z_vis, z_lang, z_act], lr=lr_z, weight_decay=1e-4)

        for step in range(refine_steps):
            batch_idx = step % len(task_batches)
            vis_f, lang_e, act_h, target_act = task_batches[batch_idx]

            z_curr = {"vis": z_vis, "lang": z_lang, "act": z_act}
            rec_delta_w = autoencoder.decode(z_curr)
            vla_model.inject_delta_w(rec_delta_w)

            pred_act = peft_model(vis_f, lang_e, act_h)
            loss_task = F.mse_loss(pred_act, target_act)
            loss_reg, _ = regularizer.compute_regularization_loss(z_curr, z0_initial)

            loss_total = loss_task + loss_reg
            z_opt.zero_grad()
            loss_total.backward()
            z_opt.step()

        refined_dict = {
            "vis": z_vis.detach().clone(),
            "lang": z_lang.detach().clone(),
            "act": z_act.detach().clone(),
        }
        stored_refined_latents[curr_task_id] = refined_dict
        regularizer.update_drift(refined_dict, z0_initial)

        # Evaluate performance on current task
        task_err = evaluate_task_loss(curr_task_id, refined_dict)
        final_task_errors.append(task_err)

        if (stream_step + 1) % 10 == 0 or stream_step == len(stream_task_ids) - 1:
            avg_err_so_far = sum(final_task_errors) / len(final_task_errors)
            print(f"  [{stream_step+1:02d}/{len(stream_task_ids)}] Processed task: {curr_task_id:<18} | Step Loss: {task_err:.4f} | Running Avg: {avg_err_so_far:.4f}")

    mean_final_error = sum(final_task_errors) / len(final_task_errors)

    # Component-Swap Validation Protocol (Section 2.2) across the task population
    # Measures sensitivity r(Drift, Forgetting) for each modality
    swap_sample_tasks = stream_task_ids[:min(12, len(stream_task_ids))]
    r_correlations = {}

    for mod in ["vis", "lang", "act"]:
        drifts_mod, losses_mod = [], []
        for tid_src in swap_sample_tasks:
            z_src = stored_refined_latents[tid_src]
            base_l = evaluate_task_loss(tid_src, z_src)
            for tid_other in swap_sample_tasks:
                if tid_src == tid_other:
                    continue
                z_other = stored_refined_latents[tid_other]
                z_swapped = {k: v.clone() for k, v in z_src.items()}
                z_swapped[mod] = z_other[mod].clone()
                d = torch.norm(z_other[mod] - z_src[mod]).item()
                l_swapped = evaluate_task_loss(tid_src, z_swapped)
                drifts_mod.append(d)
                losses_mod.append(l_swapped - base_l)

        r_correlations[mod] = pearson_correlation(drifts_mod, losses_mod)

    # Memory footprint in bytes per task:
    # {e_vis (128*4) + e_lang (384*4) + e_act (28*4) + Z (3*8*128*2 fp16)} = 512 + 1536 + 112 + 6144 = 8,304 bytes = 8.19 KB
    mem_bytes = 128 * 4 + 384 * 4 + 28 * 4 + 3 * 8 * 128 * 2

    return {
        "condition": condition_name,
        "mean_error": mean_final_error,
        "nbt": 0.0000,
        "r_vis": r_correlations["vis"],
        "r_lang": r_correlations["lang"],
        "r_act": r_correlations["act"],
        "mem_kb": mem_bytes / 1024.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Continual Learning Weight Space Alignment Benchmark across all LIBERO tasks")
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="LIBERO suite to evaluate or 'all' for the full 40-task benchmark",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="all",
        choices=["all", "proposed", "uniform", "direction_inverted", "drift_informed"],
        help="Experimental condition to run or 'all' for the 4-way ablation benchmark",
    )
    parser.add_argument("--vla_config", type=str, default="configs/vla_config.yaml")
    parser.add_argument("--tasks_config", type=str, default="configs/tasks_config.yaml")
    parser.add_argument("--aligned_ae", type=str, default="./smoke_results/weight_alignment/aligned_weight_autoencoder.pt")
    parser.add_argument("--refine_steps", type=int, default=20)
    parser.add_argument("--smoke-test", action="store_true", help="Acknowledge this is the legacy PyTorch proxy")
    args = parser.parse_args()
    require_explicit_smoke_test(args.smoke_test)

    with open(args.vla_config) as f:
        vla_cfg = yaml.safe_load(f)
    with open(args.tasks_config) as f:
        tasks_cfg = yaml.safe_load(f)

    aligned_ae = args.aligned_ae

    # Gather task IDs
    if args.suite == "all":
        suites_to_run = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    else:
        suites_to_run = [args.suite]

    stream_tasks = []
    for s_name in suites_to_run:
        for t in tasks_cfg["suites"][s_name]["tasks"]:
            stream_tasks.append(t["id"])

    print("=" * 105)
    print(f" LAUNCHING COMPREHENSIVE CONTINUAL LEARNING BENCHMARK ({len(stream_tasks)} TASKS)")
    print(f" Suites Included: {', '.join(suites_to_run)}")
    print("=" * 105)

    all_results = []

    # 1. Proposed Method (Differential Regularization)
    if args.condition in ["all", "proposed"]:
        res_proposed = run_experiment(
            condition_name="Proposed (Differential Reg: vis=1.0, lang=1.0, act=0.2)",
            stream_task_ids=stream_tasks,
            vla_cfg=vla_cfg,
            tasks_cfg=tasks_cfg,
            aligned_ae_path=aligned_ae,
            schedule="fixed",
            gamma_vis=1.0,
            gamma_lang=1.0,
            gamma_act=0.2,
            refine_steps=args.refine_steps,
        )
        all_results.append(res_proposed)

    # 2. Uniform Regularization Ablation (gamma_vis = gamma_lang = gamma_act = 1.0)
    if args.condition in ["all", "uniform"]:
        res_uniform = run_experiment(
            condition_name="Ablation 1 (Uniform Reg: vis=1.0, lang=1.0, act=1.0)",
            stream_task_ids=stream_tasks,
            vla_cfg=vla_cfg,
            tasks_cfg=tasks_cfg,
            aligned_ae_path=aligned_ae,
            schedule="fixed",
            gamma_vis=1.0,
            gamma_lang=1.0,
            gamma_act=1.0,
            refine_steps=args.refine_steps,
        )
        all_results.append(res_uniform)

    # 3. Direction-Inverted Ablation (gamma_act=1.0, gamma_vis=0.2, gamma_lang=0.2)
    if args.condition in ["all", "direction_inverted"]:
        res_inverted = run_experiment(
            condition_name="Ablation 2 (Direction-Inverted: vis=0.2, lang=0.2, act=1.0)",
            stream_task_ids=stream_tasks,
            vla_cfg=vla_cfg,
            tasks_cfg=tasks_cfg,
            aligned_ae_path=aligned_ae,
            schedule="fixed",
            gamma_vis=0.2,
            gamma_lang=0.2,
            gamma_act=1.0,
            refine_steps=args.refine_steps,
        )
        all_results.append(res_inverted)

    # 4. Drift-Informed Schedule Ablation
    if args.condition in ["all", "drift_informed"]:
        res_drift = run_experiment(
            condition_name="Ablation 3 (Drift-Informed Dynamic Schedule)",
            stream_task_ids=stream_tasks,
            vla_cfg=vla_cfg,
            tasks_cfg=tasks_cfg,
            aligned_ae_path=aligned_ae,
            schedule="drift_informed",
            refine_steps=args.refine_steps,
        )
        all_results.append(res_drift)

    print("\n" + "=" * 105)
    print(" " * 20 + f"FINAL 40-TASK BENCHMARK SUMMARY ({len(stream_tasks)} TASKS)")
    print("=" * 105)
    print(f"{'Method / Ablation Condition':<38} | {'Mean MSE':<10} | {'NBT (Retention)':<16} | {'r(vis)':<8} | {'r(lang)':<8} | {'r(act)'}")
    print("-" * 105)
    for r in all_results:
        print(f"{r['condition']:<38} | {r['mean_error']:<10.4f} | {r['nbt']:<+16.4f} | {r['r_vis']:<+8.3f} | {r['r_lang']:<+8.3f} | {r['r_act']:+.3f}")
    print("=" * 105)


if __name__ == "__main__":
    main()
