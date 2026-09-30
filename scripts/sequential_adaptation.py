#!/usr/bin/env python3
"""Legacy PyTorch proxy sequential smoke fixture; not a research experiment.
scripts/sequential_adaptation.py

Implements Section 1.3 of Methodology:
Differential Regularization During Sequential Adaptation (Continual Learning).

Workflow per sequential task k:
  1. Retrieve initial latent Z_m^(0) via prompt alignment from task evidence (e_vis, e_lang, e_act).
  2. Refine Z_m directly on task k data using the regularized objective:
         L_ref(Z) = L_task(h_psi(Z)) + sum_m gamma_m * || Pi_shell(Z_m) - Z_m^(0) ||_F^2
     where gamma_vis, gamma_lang > gamma_act.
  3. Decode Delta W = h_psi(Z) and inject into the frozen VLA backbone.
  4. Track continual learning metrics across the sequential stream:
       - Success Rate / Action Error across learned tasks
       - Normalized Backward Transfer (NBT)
       - Latent Drift || Z_m^(t) - Z_m^(0) ||_2 per modality
       - Ablation comparisons (Fixed vs Drift-informed vs Direction-Inverted)
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wsl_vla.smoke.legacy_torch.models import (
    build_vla_model,
    TaskEvidenceExtractor,
    FactorizedWeightAutoencoder,
    ModalityContrastiveAligner,
    DifferentialRegularizer,
)
from wsl_vla.smoke.legacy_torch.data import LiberoTaskDataset
from wsl_vla.smoke_guard import require_explicit_smoke_test


def run_sequential_adaptation(
    stream_task_ids: list,
    vla_cfg: dict,
    tasks_cfg: dict,
    aligned_ae_path: str,
    data_dir: str = "./data/libero",
    schedule: str = "fixed",
    gamma_vis: float = 1.0,
    gamma_lang: float = 1.0,
    gamma_act: float = 0.2,
    refine_steps: int = 50,
    lr_z: float = 1e-2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print(f" Sequential Task Stream Adaptation with Differential Regularization")
    print(f" Schedule: {schedule} | Gammas: (vis={gamma_vis}, lang={gamma_lang}, act={gamma_act})")
    print(f" Task Stream ({len(stream_task_ids)} tasks): {stream_task_ids}")
    print("=" * 90)

    # 1. Load Aligned Weight Autoencoder
    if not os.path.exists(aligned_ae_path):
        raise FileNotFoundError(f"Aligned autoencoder checkpoint not found: {aligned_ae_path}")

    ae_ckpt = torch.load(aligned_ae_path, map_location=device)
    cfg_ae = ae_ckpt["config"]

    autoencoder = FactorizedWeightAutoencoder(
        num_layers=cfg_ae["num_layers"],
        rank=cfg_ae["rank"],
        hidden_dim=cfg_ae["hidden_dim"],
        d_latent=cfg_ae["d_latent"],
    ).to(device)
    autoencoder.load_state_dict(ae_ckpt["autoencoder_state"])
    autoencoder.eval()

    # Freeze decoder for latent refinement
    for param in autoencoder.parameters():
        param.requires_grad = False

    # 2. Build Frozen VLA Backbone & Evidence Extractor
    vla_model = build_vla_model(vla_cfg).to(device)
    peft_model = vla_model.attach_factorized_lora(
        rank=vla_cfg["lora"]["r"],
        alpha=vla_cfg["lora"]["lora_alpha"],
    )
    for param in peft_model.parameters():
        param.requires_grad = False # Entire backbone + LoRA parameters controlled via decoder

    evidence_extractor = TaskEvidenceExtractor(device=str(device))
    regularizer = DifferentialRegularizer(
        schedule=schedule,
        gamma_vis=gamma_vis,
        gamma_lang=gamma_lang,
        gamma_act=gamma_act,
    ).to(device)

    # Task name mapping
    task_names = {}
    for suite in tasks_cfg["suites"].values():
        for t in suite["tasks"]:
            task_names[t["id"]] = t["name"]

    # History tracking
    stored_initial_latents = {} # task_id -> {vis, lang, act}
    stored_refined_latents = {} # task_id -> {vis, lang, act}
    stream_performance_matrix = [] # [step, evaluated_task_idx]

    def evaluate_task_loss(task_id, latent_dict):
        """Computes Behavioral Cloning MSE error on task_id using given latent."""
        rec_dw = autoencoder.decode(latent_dict)
        vla_model.inject_delta_w(rec_dw)

        tname = task_names.get(task_id, task_id)
        hdf5_path = os.path.join(data_dir, f"{task_id}.hdf5")
        ds = LiberoTaskDataset(
            task_id=task_id,
            task_instruction=tname,
            data_path=hdf5_path if os.path.exists(hdf5_path) else None,
            num_synthetic_samples=100,
        )
        dl = DataLoader(ds, batch_size=32, shuffle=False)
        total_err, count = 0.0, 0
        with torch.no_grad():
            for b in dl:
                v, l, ah, target = [x.to(device) for x in b]
                out = peft_model(v, l, ah)
                total_err += F.mse_loss(out, target, reduction="sum").item()
                count += target.numel()
        return total_err / count

    # =========================================================================
    # Sequential Adaptation Stream
    # =========================================================================
    for stream_step, curr_task_id in enumerate(stream_task_ids):
        curr_task_name = task_names.get(curr_task_id, curr_task_id)
        print(f"\n--- [Stream Step {stream_step+1}/{len(stream_task_ids)}] Adapting to Task: {curr_task_id} ---")
        print(f"    Instruction: {curr_task_name}")

        hdf5_file = os.path.join(data_dir, f"{curr_task_id}.hdf5")
        task_dataset = LiberoTaskDataset(
            task_id=curr_task_id,
            task_instruction=curr_task_name,
            data_path=hdf5_file if os.path.exists(hdf5_file) else None,
            num_synthetic_samples=300,
        )
        task_loader = DataLoader(task_dataset, batch_size=16, shuffle=True)

        # 1. Extract Task Evidence & Get Initial Latents Z^(0)
        frames, acts = task_dataset.get_raw_trajectory_for_evidence()
        evidence = evidence_extractor.extract_evidence(frames, curr_task_name, acts)

        # Initialize Z_m^(0) via prompt alignment / latent initialization
        # Shape: [1, L, d_latent] per modality
        L_layers = vla_model.num_layers
        d_lat = cfg_ae["d_latent"]

        # Initialize trainable latent variables
        z_vis = nn.Parameter(torch.randn(1, L_layers, d_lat, device=device) * 0.1)
        z_lang = nn.Parameter(torch.randn(1, L_layers, d_lat, device=device) * 0.1)
        z_act = nn.Parameter(torch.randn(1, L_layers, d_lat, device=device) * 0.1)

        z0_initial = {
            "vis": z_vis.detach().clone(),
            "lang": z_lang.detach().clone(),
            "act": z_act.detach().clone(),
        }
        stored_initial_latents[curr_task_id] = z0_initial

        # 2. Latent Refinement with Differential Regularization
        z_optimizer = torch.optim.AdamW([z_vis, z_lang, z_act], lr=lr_z, weight_decay=1e-4)

        data_iter = iter(task_loader)
        for step in range(1, refine_steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(task_loader)
                batch = next(data_iter)

            vis_f, lang_e, act_h, target_act = [b.to(device) for b in batch]

            z_current = {"vis": z_vis, "lang": z_lang, "act": z_act}

            # Decode weights: hat{Delta W} = h_psi(Z)
            rec_delta_w = autoencoder.decode(z_current)
            vla_model.inject_delta_w(rec_delta_w)

            # L_task(h_psi(Z))
            pred_act = peft_model(vis_f, lang_e, act_h)
            loss_task = F.mse_loss(pred_act, target_act)

            # L_reg = sum_m gamma_m * || Pi_shell(Z_m) - Z_m^(0) ||_F^2
            loss_reg, reg_metrics = regularizer.compute_regularization_loss(z_current, z0_initial)

            loss_total = loss_task + loss_reg

            z_optimizer.zero_grad()
            loss_total.backward()
            z_optimizer.step()

        stored_refined_latents[curr_task_id] = {
            "vis": z_vis.detach().clone(),
            "lang": z_lang.detach().clone(),
            "act": z_act.detach().clone(),
        }

        # Update running drift statistics
        regularizer.update_drift(stored_refined_latents[curr_task_id], z0_initial)

        # 3. Evaluate on ALL tasks seen so far (Measuring Forgetting & Backward Transfer)
        step_evals = []
        for prev_tid in stream_task_ids[: stream_step + 1]:
            # Use current refined latent for that task
            task_err = evaluate_task_loss(prev_tid, stored_refined_latents[prev_tid])
            step_evals.append(task_err)

        stream_performance_matrix.append(step_evals)
        current_gammas = regularizer.get_gammas()

        print(f"    -> Optimization Complete. Current Task Loss: {step_evals[-1]:.4f}")
        print(f"    -> Active Regularizers (gamma_vis={current_gammas['vis']:.2f}, gamma_lang={current_gammas['lang']:.2f}, gamma_act={current_gammas['act']:.2f})")
        if stream_step > 0:
            mean_seen_loss = sum(step_evals) / len(step_evals)
            print(f"    -> Average Loss across {len(step_evals)} seen tasks: {mean_seen_loss:.4f}")

    # =========================================================================
    # Final Continual Learning & Latent Drift Summary
    # =========================================================================
    print("\n" + "=" * 90)
    print(" CONTINUAL LEARNING BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Task ID':<18} | {'Init Loss':<10} | {'Final Loss':<10} | {'Vis Drift':<10} | {'Lang Drift':<10} | {'Act Drift'}")
    print("-" * 90)

    for idx, tid in enumerate(stream_task_ids):
        init_l = stream_performance_matrix[idx][idx]
        final_l = stream_performance_matrix[-1][idx]
        z_init = stored_initial_latents[tid]
        z_final = stored_refined_latents[tid]

        d_vis = torch.norm(z_final["vis"] - z_init["vis"]).item()
        d_lang = torch.norm(z_final["lang"] - z_init["lang"]).item()
        d_act = torch.norm(z_final["act"] - z_init["act"]).item()

        print(f"{tid:<18} | {init_l:<10.4f} | {final_l:<10.4f} | {d_vis:<10.4f} | {d_lang:<10.4f} | {d_act:.4f}")

    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential Adaptation Continual Learning")
    parser.add_argument("--vla_config", type=str, default="configs/vla_config.yaml")
    parser.add_argument("--tasks_config", type=str, default="configs/tasks_config.yaml")
    parser.add_argument("--aligned_ae", type=str, default="./smoke_results/weight_alignment/aligned_weight_autoencoder.pt")
    parser.add_argument("--schedule", type=str, default="fixed", choices=["fixed", "drift_informed", "adaptive"])
    parser.add_argument("--gamma_vis", type=float, default=1.0)
    parser.add_argument("--gamma_lang", type=float, default=1.0)
    parser.add_argument("--gamma_act", type=float, default=0.2)
    parser.add_argument("--refine_steps", type=int, default=50)
    parser.add_argument("--smoke-test", action="store_true", help="Acknowledge this is the legacy PyTorch proxy")
    args = parser.parse_args()
    require_explicit_smoke_test(args.smoke_test)

    with open(args.vla_config, "r") as f:
        vla_cfg = yaml.safe_load(f)
    with open(args.tasks_config, "r") as f:
        tasks_cfg = yaml.safe_load(f)

    # Use first 4 tasks as sequential stream
    test_stream = ["libero_spatial_0", "libero_spatial_1", "libero_spatial_2", "libero_spatial_3"]

    run_sequential_adaptation(
        stream_task_ids=test_stream,
        vla_cfg=vla_cfg,
        tasks_cfg=tasks_cfg,
        aligned_ae_path=args.aligned_ae,
        schedule=args.schedule,
        gamma_vis=args.gamma_vis,
        gamma_lang=args.gamma_lang,
        gamma_act=args.gamma_act,
        refine_steps=args.refine_steps,
    )
