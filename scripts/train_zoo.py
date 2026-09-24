#!/usr/bin/env python3
"""
scripts/train_zoo.py

Resilient, idempotent multi-task training loop for constructing the Model Zoo.
Uses build_vla_model() to dynamically instantiate any registered VLA backbone
(e.g., Octo-Small, SmallVLA, OpenVLA) directly from configs/vla_config.yaml.
"""
import os
import sys
import gc
import shutil
import logging
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import build_vla_model, TaskEvidenceExtractor
from data.dataset import LiberoTaskDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def check_disk_space(path: str = ".", min_gb: float = 2.0):
    """Halts training gracefully if disk space is dangerously low."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    if free_gb < min_gb:
        raise RuntimeError(f"FATAL: Insufficient disk space! Free: {free_gb:.2f} GB (minimum: {min_gb} GB).")


def is_checkpoint_valid(filepath: str) -> bool:
    """Verifies that the checkpoint exists, is non-empty, and contains all required keys."""
    if not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) < 512:
        return False
    try:
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        required_keys = {"task_id", "task_name", "delta_w", "e_vis", "e_lang", "e_act"}
        return required_keys.issubset(data.keys())
    except Exception:
        return False


def atomic_save(payload: dict, target_filepath: str):
    """
    Saves state dictionary to a temporary file first, then atomically renames it.
    Prevents corrupt/partial checkpoints if interrupted mid-save.
    """
    tmp_filepath = target_filepath + ".tmp"
    torch.save(payload, tmp_filepath)
    os.replace(tmp_filepath, target_filepath)


def train_single_task(
    task_info: dict,
    vla_cfg: dict,
    evidence_extractor: TaskEvidenceExtractor,
    output_dir: str,
    device: torch.device,
    max_steps: int = 500,
    force: bool = False,
):
    task_id = task_info["id"]
    task_name = task_info["name"]
    ckpt_path = os.path.join(output_dir, f"task_{task_id}.pt")

    # 1. Idempotency Check: Skip completed tasks unless forced
    if not force and is_checkpoint_valid(ckpt_path):
        logging.info(f"[SKIP] Task {task_id} is already completed. Skipping.")
        return

    check_disk_space(output_dir, min_gb=2.0)
    logging.info(f"===> Starting training on Task [{task_id}]: {task_name}")

    # 2. Build Dataset & Dataloader
    data_dir = vla_cfg.get("paths", {}).get("data_dir", "./data/libero")
    task_hdf5 = os.path.join(data_dir, f"{task_id}.hdf5")
    data_path = task_hdf5 if os.path.exists(task_hdf5) else None

    if data_path:
        logging.info(f"  -> Using real LIBERO demonstration data: {data_path}")
    else:
        logging.info(f"  -> No HDF5 found at {task_hdf5}. Using synthetic demonstration generator.")

    dataset = LiberoTaskDataset(
        task_id=task_id,
        task_instruction=task_name,
        data_path=data_path,
        data_dir=data_dir,
        num_synthetic_samples=400,
        img_feat_dim=vla_cfg["model"]["image_features_dim"],
        lang_embed_dim=vla_cfg["model"]["language_embed_dim"],
        action_dim=vla_cfg["model"]["action_dim"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=vla_cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # 3. Dynamically Build VLA Model (OctoSmall, SmallVLA, etc.) via Factory
    vla_model = build_vla_model(vla_cfg).to(device)

    # Attach factorized LoRA & get trainable parameters
    peft_model = vla_model.attach_factorized_lora(
        rank=vla_cfg["lora"]["r"],
        alpha=vla_cfg["lora"]["lora_alpha"],
    )

    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(vla_cfg["training"]["learning_rate"]),
        weight_decay=float(vla_cfg["training"]["weight_decay"]),
    )
    loss_fn = nn.MSELoss()

    # 4. Behavioral Cloning Optimization Loop
    peft_model.train()
    step = 0
    running_loss = 0.0

    while step < max_steps:
        for vis_b, lang_b, act_hist_b, target_act_b in dataloader:
            vis_b = vis_b.to(device)
            lang_b = lang_b.to(device)
            act_hist_b = act_hist_b.to(device)
            target_act_b = target_act_b.to(device)

            optimizer.zero_grad()
            pred_action = peft_model(vis_b, lang_b, act_hist_b)
            loss = loss_fn(pred_action, target_act_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % 100 == 0:
                avg_loss = running_loss / 100.0
                logging.info(f"  [Task {task_id}] Step {step}/{max_steps} | BC Loss: {avg_loss:.4f}")
                running_loss = 0.0

            if step >= max_steps:
                break

    # 5. Extract Modality-Factorized Delta W: [L, 3, r, H]
    delta_w = vla_model.extract_delta_w()

    # 6. Extract Multi-Modal Task Evidence
    raw_frames, raw_actions = dataset.get_raw_trajectory_for_evidence()
    evidence = evidence_extractor.extract_evidence(
        demo_frames=raw_frames,
        task_instruction=task_name,
        actions=raw_actions,
    )

    # 7. Atomic Serialization
    payload = {
        "task_id": task_id,
        "task_name": task_name,
        "delta_w": delta_w,        # Tensor [L, 3, r, H]
        "e_vis": evidence["e_vis"],
        "e_lang": evidence["e_lang"],
        "e_act": evidence["e_act"],
        "model_name": vla_cfg["model"]["name"],
        "dimensions": vla_model.get_dims(),
    }
    atomic_save(payload, ckpt_path)
    logging.info(f"[SAVED] Task {task_id} successfully serialized to {ckpt_path} (Delta W: {list(delta_w.shape)})")

    # 8. Memory Sanitation
    del vla_model, peft_model, optimizer, dataloader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Resilient Model Zoo Multi-Task Training")
    parser.add_argument("--vla_config", type=str, default="configs/vla_config.yaml")
    parser.add_argument("--tasks_config", type=str, default="configs/tasks_config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override model name (e.g. octo_small, small_vla)")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--limit_tasks", type=int, default=None, help="Train only first N tasks (for debug)")
    parser.add_argument("--force", action="store_true", help="Force re-training even if checkpoints already exist")
    args = parser.parse_args()

    with open(args.vla_config, "r") as f:
        vla_cfg = yaml.safe_load(f)

    if args.model:
        vla_cfg["model"]["name"] = args.model

    with open(args.tasks_config, "r") as f:
        tasks_cfg = yaml.safe_load(f)

    output_dir = vla_cfg["paths"]["checkpoints_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Device selection: NVIDIA CUDA -> Apple Silicon MPS -> CPU
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        device = torch.device("cuda")
        logging.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)} (TF32 enabled)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple Silicon GPU: Metal Performance Shaders (MPS) active")
    else:
        device = torch.device("cpu")
        logging.info("Using device: CPU")

    model_name = vla_cfg["model"]["name"]
    logging.info(f"Active VLA Backbone: '{model_name}'")

    # Flatten task registry
    all_tasks = []
    for suite_name, suite_data in tasks_cfg["suites"].items():
        for t in suite_data["tasks"]:
            all_tasks.append(t)

    if args.limit_tasks is not None:
        all_tasks = all_tasks[: args.limit_tasks]

    steps_per_task = args.max_steps or vla_cfg["training"]["steps_per_task"]
    logging.info(f"Loaded {len(all_tasks)} tasks to train. Steps per task: {steps_per_task}")

    evidence_extractor = TaskEvidenceExtractor(
        device="cpu",
        vis_in_dim=vla_cfg["model"]["image_features_dim"],
        vis_embed_dim=128,
    )

    # Sequential Training Loop with Crash Isolation
    for task_info in all_tasks:
        try:
            train_single_task(
                task_info=task_info,
                vla_cfg=vla_cfg,
                evidence_extractor=evidence_extractor,
                output_dir=output_dir,
                device=device,
                max_steps=steps_per_task,
                force=args.force,
            )
        except Exception as e:
            logging.error(f"[ERROR] Task {task_info['id']} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            continue

    logging.info("Model Zoo training process completed!")


if __name__ == "__main__":
    main()
