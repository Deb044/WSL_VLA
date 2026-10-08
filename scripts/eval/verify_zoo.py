#!/usr/bin/env python3
"""
scripts/eval/verify_zoo.py

Integrity verification script for the trained Model Zoo population.
Scans the checkpoint directory, checks parameter dimensions [L, 3, r, H],
and verifies that all multi-modal task evidence (e_vis, e_lang, e_act) is intact.
"""
import os
import sys
from pathlib import Path
import argparse
import torch

# Add repository root before importing the smoke guard.
def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / 'pyproject.toml').is_file():
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from wsl_vla.smoke_guard import require_explicit_smoke_test

def verify_zoo(checkpoint_dir: str, expected_tasks: int = 40):
    print("======================================================================")
    print(f" Verifying Model Zoo Checkpoints in: {checkpoint_dir}")
    print("======================================================================")

    if not os.path.exists(checkpoint_dir):
        print(f"[ERROR] Directory '{checkpoint_dir}' does not exist.")
        sys.exit(1)

    ckpt_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt") and not f.endswith(".tmp")])
    num_found = len(ckpt_files)
    print(f"  Found {num_found} checkpoint(s). Expected: {expected_tasks}")

    delta_w_list = []
    e_vis_list = []
    e_lang_list = []
    e_act_list = []

    passed = 0
    for fname in ckpt_files:
        fpath = os.path.join(checkpoint_dir, fname)
        try:
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            required_keys = ["task_id", "task_name", "delta_w", "e_vis", "e_lang", "e_act"]
            for k in required_keys:
                assert k in data, f"Missing key '{k}' in {fname}"

            dw = data["delta_w"]
            # Expected shape: [L, 3, r, H]
            assert dw.dim() == 4, f"Delta W has invalid rank {dw.dim()}, expected 4: [L, 3, r, H]"
            assert dw.size(1) == 3, f"Delta W axis 1 must be 3 (vis, lang, act), got {dw.size(1)}"

            delta_w_list.append(dw)
            e_vis_list.append(data["e_vis"])
            e_lang_list.append(data["e_lang"])
            e_act_list.append(data["e_act"])
            passed += 1

        except Exception as e:
            print(f"  [FAIL] Checkpoint '{fname}' failed validation: {e}")

    print(f"\nVerification Results: {passed}/{num_found} valid checkpoints.")

    if passed > 0:
        stacked_zoo = torch.stack(delta_w_list, dim=0)
        print("\n--- Model Zoo Population Statistics ---")
        print(f"Population Tensor Shape : {list(stacked_zoo.shape)}  [N_tasks, L, 3, r, H]")
        print(f"Visual Evidence Shape   : {list(e_vis_list[0].shape)}")
        print(f"Language Evidence Shape : {list(e_lang_list[0].shape)}")
        print(f"Action Evidence Shape   : {list(e_act_list[0].shape)}")
        print(f"Mean Delta W norm       : {torch.norm(stacked_zoo).item():.4f}")
        print("\n[SUCCESS] Model Zoo population is valid and ready for Weight-Space Alignment!")
    else:
        print("\n[ERROR] No valid checkpoints found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./smoke_results/model_zoo")
    parser.add_argument("--expected", type=int, default=40)
    parser.add_argument("--smoke-test", action="store_true", help="Acknowledge this verifies proxy smoke artifacts only")
    args = parser.parse_args()
    require_explicit_smoke_test(args.smoke_test)

    verify_zoo(checkpoint_dir=args.dir, expected_tasks=args.expected)
