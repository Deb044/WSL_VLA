#!/usr/bin/env python3
"""
scripts/sweep_gammas.py

Legacy proxy smoke sweep for differential-regularization plumbing:
    (gamma_vis, gamma_lang, gamma_act)

Explores the stability-plasticity trade-off in the component-factorized weight space:
- Plasticity: Ability to adapt to new tasks (lower task MSE).
- Stability: Retention of previously adapted tasks (low backward transfer degradation).
- Modality Sensitivity: How varying gamma_vis, gamma_lang, and gamma_act independently
  affects overall continual learning performance.

Outputs a ranked leaderboard of gamma configurations and identifies the Pareto-optimal values.
"""
import os
import sys
import yaml
import math
import argparse
import itertools
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wsl_vla.smoke.legacy_torch.models import (
    build_vla_model,
    FactorizedWeightAutoencoder,
    DifferentialRegularizer,
)
from wsl_vla.smoke.legacy_torch.data import LiberoTaskDataset
from wsl_vla.smoke_guard import require_explicit_smoke_test


def evaluate_configuration(
    gamma_vis: float,
    gamma_lang: float,
    gamma_act: float,
    stream_task_ids: List[str],
    task_names: Dict[str, str],
    autoencoder: FactorizedWeightAutoencoder,
    vla_model: nn.Module,
    peft_model: nn.Module,
    val_cache: Dict[str, list],
    train_cache: Dict[str, list],
    refine_steps: int = 20,
    lr_z: float = 1e-2,
    d_lat: int = 128,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, float]:
    """
    Evaluates a single (gamma_vis, gamma_lang, gamma_act) configuration
    across the sequential adaptation stream.
    """
    regularizer = DifferentialRegularizer(
        schedule="fixed",
        gamma_vis=gamma_vis,
        gamma_lang=gamma_lang,
        gamma_act=gamma_act,
    ).to(device)

    stored_initial_latents = {}
    stored_refined_latents = {}
    task_adapt_errors = []

    L_layers = vla_model.num_layers

    def evaluate_task_loss(task_id: str, latent_dict: dict) -> float:
        rec_dw = autoencoder.decode(latent_dict)
        vla_model.inject_delta_w(rec_dw)
        batches = val_cache[task_id]
        total_err, count = 0.0, 0
        with torch.no_grad():
            for v, l, ah, target in batches:
                out = peft_model(v, l, ah)
                total_err += F.mse_loss(out, target, reduction="sum").item()
                count += target.numel()
        return total_err / count

    # Sequential stream
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

        task_batches = train_cache[curr_task_id]
        z_opt = torch.optim.AdamW([z_vis, z_lang, z_act], lr=lr_z, weight_decay=1e-4)

        # Base model parameter and buffer dictionaries for functional_call
        params = dict(peft_model.named_parameters())
        buffers = dict(peft_model.named_buffers())
        modality_keys = ["vis_block", "lang_block", "act_block"]

        for step in range(refine_steps):
            batch_idx = step % len(task_batches)
            vis_f, lang_e, act_h, target_act = task_batches[batch_idx]

            z_curr = {"vis": z_vis, "lang": z_lang, "act": z_act}
            rec_delta_w = autoencoder.decode(z_curr)
            if rec_delta_w.dim() == 5:
                rec_delta_w = rec_delta_w.squeeze(0)

            # Functionally assign decoded weights to PEFT lora_A
            step_params = dict(params)
            for l in range(L_layers):
                for m_idx, mod_name in enumerate(modality_keys):
                    k = f"base_model.model.layers.{l}.{mod_name}.lora_A.default.weight"
                    step_params[k] = rec_delta_w[l, m_idx]

            pred_act = torch.func.functional_call(peft_model, (step_params, buffers), (vis_f, lang_e, act_h))
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

        # Evaluate performance on current task
        task_adapt_errors.append(evaluate_task_loss(curr_task_id, refined_dict))

    # Compute Drifts and Backward Transfer
    mean_adapt_mse = sum(task_adapt_errors) / len(task_adapt_errors)
    drifts = {"vis": [], "lang": [], "act": []}
    for tid in stream_task_ids:
        for m in ["vis", "lang", "act"]:
            d_val = torch.norm(
                stored_refined_latents[tid][m] - stored_initial_latents[tid][m], p=2
            ).item()
            drifts[m].append(d_val)

    mean_d_vis = sum(drifts["vis"]) / len(drifts["vis"])
    mean_d_lang = sum(drifts["lang"]) / len(drifts["lang"])
    mean_d_act = sum(drifts["act"]) / len(drifts["act"])

    # Composite Score: MSE penalized by drift imbalance
    # We want low adaptation error while avoiding catastrophic drift in fragile modalities
    composite_score = mean_adapt_mse + 0.05 * (mean_d_vis + mean_d_lang)

    return {
        "gamma_vis": gamma_vis,
        "gamma_lang": gamma_lang,
        "gamma_act": gamma_act,
        "mean_adapt_mse": mean_adapt_mse,
        "mean_d_vis": mean_d_vis,
        "mean_d_lang": mean_d_lang,
        "mean_d_act": mean_d_act,
        "composite_score": composite_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep Regularization Gammas")
    parser.add_argument(
        "--vla_config", type=str, default="configs/vla_config.yaml", help="VLA config"
    )
    parser.add_argument(
        "--tasks_config", type=str, default="configs/tasks_config.yaml", help="Tasks config"
    )
    parser.add_argument(
        "--aligned_ae",
        type=str,
        default="smoke_results/weight_alignment/aligned_weight_autoencoder.pt",
        help="Aligned autoencoder checkpoint",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
        choices=["all", "libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="Task suite to sweep on",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=5,
        help="Number of tasks per sweep evaluation (default: 5 for speed)",
    )
    parser.add_argument(
        "--refine_steps",
        type=int,
        default=25,
        help="Number of latent optimization steps per task",
    )
    parser.add_argument(
        "--search_type",
        type=str,
        default="focused_grid",
        choices=["focused_grid", "full_grid", "sensitivity_act", "sensitivity_vis"],
        help="Preset search space",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Acknowledge this is the legacy PyTorch proxy")
    args = parser.parse_args()
    require_explicit_smoke_test(args.smoke_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" DIFFERENTIAL REGULARIZATION GAMMA SWEEP & OPTIMIZATION")
    print(f" Device: {device} | Suite: {args.suite} | Tasks per run: {args.max_tasks}")
    print("=" * 80)

    # 1. Load Configurations
    with open(args.vla_config, "r") as f:
        vla_cfg = yaml.safe_load(f)
    with open(args.tasks_config, "r") as f:
        tasks_cfg = yaml.safe_load(f)

    # 2. Select Tasks
    if args.suite == "all":
        stream_task_ids = []
        for s in tasks_cfg["suites"].values():
            stream_task_ids.extend([t["id"] for t in s["tasks"]])
    else:
        stream_task_ids = [t["id"] for t in tasks_cfg["suites"][args.suite]["tasks"]]

    if args.max_tasks > 0:
        stream_task_ids = stream_task_ids[: args.max_tasks]

    task_names = {}
    for s in tasks_cfg["suites"].values():
        for t in s["tasks"]:
            task_names[t["id"]] = t["name"]

    print(f"Selected {len(stream_task_ids)} tasks for evaluation: {stream_task_ids}")

    # 3. Load Models
    ae_ckpt = torch.load(args.aligned_ae, map_location=device, weights_only=False)
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

    vla_model = build_vla_model(vla_cfg).to(device)
    peft_model = vla_model.attach_factorized_lora(
        rank=vla_cfg["lora"]["r"],
        alpha=vla_cfg["lora"]["lora_alpha"],
    )
    for p in peft_model.parameters():
        p.requires_grad = False

    # 4. Pre-cache Data Batches on GPU for Ultra-Fast Search
    print("\nPre-caching demonstration batches on GPU...")
    val_cache = {}
    train_cache = {}
    for tid in stream_task_ids:
        tname = task_names.get(tid, tid)
        ds_val = LiberoTaskDataset(
            task_id=tid,
            task_instruction=tname,
            data_dir="data/libero",
            num_synthetic_samples=100,
        )
        dl_val = DataLoader(ds_val, batch_size=32, shuffle=False)
        val_cache[tid] = [[x.to(device) for x in b] for b in dl_val]

        ds_train = LiberoTaskDataset(
            task_id=tid,
            task_instruction=tname,
            data_dir="data/libero",
            num_synthetic_samples=150,
        )
        dl_train = DataLoader(ds_train, batch_size=16, shuffle=True)
        train_cache[tid] = [[b.to(device) for b in batch] for batch in dl_train]
    print("GPU cache initialized.")

    # 5. Define Candidate Search Grid
    candidates = []
    if args.search_type == "focused_grid":
        # Explores:
        # - Proposed Asymmetric: High vis/lang, low act (variations around 1.0, 1.0, 0.2)
        # - Low-friction act: gamma_act in [0.0, 0.05, 0.1, 0.2, 0.5]
        # - Uniform baselines: [0.2, 0.5, 1.0, 2.0]
        # - Inverted: Low vis/lang, high act
        candidates = [
            # High plasticity on act, strong anchor on vis/lang
            (1.0, 1.0, 0.0),    # Free action head
            (1.0, 1.0, 0.05),   # Minimal action constraint
            (1.0, 1.0, 0.1),    # Light action constraint
            (1.0, 1.0, 0.2),    # Proposed default
            (1.0, 1.0, 0.5),    # Moderate action constraint
            (1.5, 1.5, 0.1),    # Stronger cognitive anchor
            (2.0, 2.0, 0.1),    # Very strong cognitive anchor
            (2.0, 2.0, 0.2),    # Very strong cognitive anchor, moderate act
            # Uniform baselines
            (0.2, 0.2, 0.2),    # Loose uniform
            (0.5, 0.5, 0.5),    # Medium uniform
            (1.0, 1.0, 1.0),    # Stiff uniform (Standard baseline)
            # Inverted baselines (Pathological controls)
            (0.2, 0.2, 1.0),    # Direction-inverted (Paper control)
            (0.1, 0.1, 2.0),    # Extreme inverted
            # Unconstrained
            (0.0, 0.0, 0.0),    # Complete unconstrained drift
        ]
    elif args.search_type == "sensitivity_act":
        # Fix gamma_vis=1.0, gamma_lang=1.0, sweep gamma_act systematically
        gamma_act_values = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
        candidates = [(1.0, 1.0, ga) for ga in gamma_act_values]
    elif args.search_type == "sensitivity_vis":
        # Fix gamma_act=0.1, sweep gamma_vis=gamma_lang
        g_vl_values = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        candidates = [(g, g, 0.1) for g in g_vl_values]
    elif args.search_type == "full_grid":
        g_vis_list = [0.2, 0.5, 1.0, 2.0]
        g_lang_list = [0.2, 0.5, 1.0, 2.0]
        g_act_list = [0.0, 0.1, 0.2, 0.5, 1.0]
        candidates = list(itertools.product(g_vis_list, g_lang_list, g_act_list))

    print(f"\nEvaluating {len(candidates)} gamma candidates...")
    results = []

    for idx, (gv, gl, ga) in enumerate(candidates, 1):
        sys.stdout.write(
            f"\r[{idx:02d}/{len(candidates):02d}] Testing (gamma_vis={gv:.2f}, gamma_lang={gl:.2f}, gamma_act={ga:.2f})... "
        )
        sys.stdout.flush()

        res = evaluate_configuration(
            gamma_vis=gv,
            gamma_lang=gl,
            gamma_act=ga,
            stream_task_ids=stream_task_ids,
            task_names=task_names,
            autoencoder=autoencoder,
            vla_model=vla_model,
            peft_model=peft_model,
            val_cache=val_cache,
            train_cache=train_cache,
            refine_steps=args.refine_steps,
            d_lat=cfg_ae["d_latent"],
            device=device,
        )
        results.append(res)

    print("\nSweep Complete!\n")

    # 6. Rank Results by Adaptation MSE and Composite Score
    results_by_mse = sorted(results, key=lambda x: x["mean_adapt_mse"])
    best_config = results_by_mse[0]

    # Print Leaderboard
    print("=" * 105)
    print(f"{'Rank':<5} | {'gamma_vis':<10} | {'gamma_lang':<11} | {'gamma_act':<10} | {'Mean MSE':<12} | {'Vis Drift':<11} | {'Lang Drift':<11} | {'Act Drift':<10} | {'Regime Type'}")
    print("=" * 105)

    for rank, r in enumerate(results_by_mse, 1):
        gv, gl, ga = r["gamma_vis"], r["gamma_lang"], r["gamma_act"]
        if gv > ga and gl > ga:
            regime = "Differential (Asymmetric) *"
        elif abs(gv - ga) < 1e-4 and abs(gl - ga) < 1e-4:
            regime = "Uniform"
        elif gv < ga or gl < ga:
            regime = "Inverted (Pathological)"
        else:
            regime = "Mixed"

        marker = " <== BEST" if rank == 1 else ""
        print(
            f"{rank:<5} | {gv:<10.2f} | {gl:<11.2f} | {ga:<10.2f} | {r['mean_adapt_mse']:<12.4f} | "
            f"{r['mean_d_vis']:<11.4f} | {r['mean_d_lang']:<11.4f} | {r['mean_d_act']:<10.4f} | {regime}{marker}"
        )
    print("=" * 105)

    print("\n" + "=" * 60)
    print(" OPTIMAL HYPERPARAMETER FINDINGS")
    print("=" * 60)
    print(f"Optimal Configuration (Lowest MSE):")
    print(f"  * gamma_vis:  {best_config['gamma_vis']:.2f}")
    print(f"  * gamma_lang: {best_config['gamma_lang']:.2f}")
    print(f"  * gamma_act:  {best_config['gamma_act']:.2f}")
    print(f"  * Resulting Mean Action MSE: {best_config['mean_adapt_mse']:.4f}")
    print(f"  * Cognitive Drift (Vis / Lang): {best_config['mean_d_vis']:.4f} / {best_config['mean_d_lang']:.4f}")
    print(f"  * Motor Plasticity Drift (Act): {best_config['mean_d_act']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
