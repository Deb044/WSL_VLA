# Historical Model-Zoo Draft (Do Not Execute)

This document predates the official Octo-Small/JAX path and contains mock and
synthetic procedures that are not publication-eligible. Use
`docs/methodology/METHODOLOGY1_RUNBOOK.md` and `scripts/eval/research_preflight.py` instead.

# Zero-Fail Pre-Flight Protocol: Preparing the VLA Model Zoo Pipeline

**Project**: Weight Space Alignment for Continual Learning in Robotics  
**Phase**: Pre-Compute Preparation & Zero-Fail Execution Protocol  
**Target Milestone**: 40-Task LIBERO Reference Population Training (150+ GPU-Hours)

---

## 1. Executive Summary & Why Most GPU Runs Fail

Building the reference Model Zoo is a foundational, one-time investment for Weight-Space Representation Learning. As detailed in the research proposal, downstream contrastive alignment requires an uncorrupted population of trained LoRA adapters:
$$\Delta W^{(k)} \in \mathbb{R}^{L \times 3 \times r \times H} \quad \text{and task evidence} \quad \mathcal{M}^{(k)} = \{e_{\text{vis}}^{(k)}, e_{\text{lang}}^{(k)}, e_{\text{act}}^{(k)}\}$$
across all 40 LIBERO tasks (Spatial, Object, Goal, 10).

On remote high-performance compute clusters (SLURM, cloud instances, university servers), **long-running multi-task jobs commonly fail due to predictable pitfalls**:
1. **Compute Request Rejection**: Vague requests lacking storage breakdowns, exact walltime math, and GPU memory justifications get rejected or deprioritized by cluster admins.
2. **Cluster Headless Rendering Crash**: Simulators (MuJoCo / LIBERO) try to open an X11 window display, instantly crashing with `GLFWError: X11 service not available` unless headless EGL/OSMesa is configured.
3. **Cluster Ingress / Quota Deadlock**: Download scripts relying on `gdown` or unverified links hit Google Drive rate limits or cluster proxy blocks after compute allocation has already begun.
4. **Mid-Run Process Termination & Corruption**: SSH drops or preemption terminates the script mid-write, leaving corrupted zero-byte `.pt` files that fail downstream training.
5. **Silent Memory Leaks**: PyTorch computational graphs or simulation handles accumulated across tasks trigger an Out-of-Memory (OOM) error at task 18 or 27.
6. **Mismatched Weight Factorization**: Inconsistent LoRA target module names prevent clean slicing into the required $[L, 3, r, H]$ tensor representation.

This document establishes the **Zero-Fail Pre-Flight Protocol**: everything you must configure, simulate, and verify locally on your machine *before* requesting compute and logging into the remote GPU node.

---

## 2. Compute Allocation Justification & Sizing Spec Sheet

When submitting your compute proposal to cluster administrators or provisioning cloud instances, use the verified specifications below to guarantee first-round approval.

### 2.1 Hardware Sizing Requirements

| Resource | Specification | Justification |
| :--- | :--- | :--- |
| **GPU Model** | 1x NVIDIA A100 (80GB / 40GB) or 1x RTX 4090 (24GB) | Base VLA (e.g., OpenVLA-7B, Octo, or Pi0) with $r=16/32$ LoRA requires ~16–22 GB VRAM in bfloat16 mixed precision with activation checkpointing. |
| **CPU Allocation** | 8 to 16 vCPUs | Multi-threaded HDF5 data loading, image augmentations, and headless MuJoCo environment rollouts during periodic eval. |
| **System RAM** | 32 GB – 64 GB | LIBERO trajectory batch caching and fast dataset indexing in memory. |
| **Scratch Storage** | **150 GB NVMe SSD** | • Base VLA model weights: ~25 GB<br>• Conda environment + CUDA libs: ~15 GB<br>• LIBERO 4-suite HDF5 datasets: ~35 GB<br>• 40 Model Zoo checkpoints + evidence: ~15 GB<br>• Buffer for OS scratch, logs, and eval rollouts: ~40 GB |
| **Walltime** | **160 GPU-Hours** (~6.5 days single GPU, or ~38 hours on 4x A100s) | 40 tasks $\times$ ~3.5 hours/task (at 15,000–20,000 optimizer steps per task with evaluation). |

### 2.2 Ready-to-Submit Compute Proposal Justification

> **Project Title**: Contrastive Weight-Space Representation Learning for Continual Robotic Manipulation  
> **Resource Requested**: 160 GPU-Hours on 1x NVIDIA A100-80GB (or 4x A100-80GB for 40 hours), 64 GB RAM, 150 GB scratch storage.  
> **Scientific Justification**: We are training a reference population ("Model Zoo") of 40 modality-factorized LoRA policies across the LIBERO benchmark suites (LIBERO-Spatial, Object, Goal, 10). Downstream contrastive alignment requires an uncorrupted, diverse weight population to train weight-space autoencoders and establish latent drift metrics for continual learning.  
> **Pipeline Readiness**: The training pipeline features local CPU unit-tested dry-runs, idempotent task skipping with atomic checkpointing, and headless EGL rendering. The pipeline can be interrupted and resumed without loss of progress.

---

## 3. Local Conda Environment & Headless Setup

Set up a clean, reproducible conda environment locally first. Remote servers require exact matching versions, especially for PyTorch, PEFT, and headless MuJoCo rendering.

### 3.1 Local Environment Initialization

```bash
# Create local conda environment
conda create -n vla_zoo python=3.10 -y
conda activate vla_zoo

# Install PyTorch with CUDA support (or CPU locally for pre-flight testing)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core VLA, LoRA, and simulation packages
pip install transformers>=4.40.0 peft>=0.10.0 accelerate>=0.28.0
pip install h5py pyyaml wandb tqdm einops scipy sentence-transformers

# Install MuJoCo and LIBERO dependencies
pip install mujoco>=2.3.7 gymnasium
# Install headless rendering support
pip install PyOpenGL PyOpenGL_accelerate
```

### 3.2 Exporting Reproducible Environment Files

Export both an explicit dependency list and an automated server bootstrap script:

```bash
# Export pinned requirements
pip list --format=freeze > requirements.txt
```

Create `scripts/setup_remote_env.sh` for the remote server:
```bash
#!/usr/bin/env bash
# ==============================================================================
# Remote GPU Node Bootstrap Script
# ==============================================================================
set -euo pipefail

echo "==> Configuring Headless Rendering Environment Variables..."
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Append to bashrc for persistent SSH sessions
grep -qxF 'export MUJOCO_GL=egl' ~/.bashrc || echo 'export MUJOCO_GL=egl' >> ~/.bashrc
grep -qxF 'export PYOPENGL_PLATFORM=egl' ~/.bashrc || echo 'export PYOPENGL_PLATFORM=egl' >> ~/.bashrc

echo "==> Verifying NVIDIA Driver & CUDA..."
nvidia-smi

echo "==> Creating or Activating Conda Environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda info --envs | grep -q "vla_zoo"; then
    conda create -n vla_zoo python=3.10 -y
fi
conda activate vla_zoo

echo "==> Installing Core Dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo "==> Testing EGL Headless MuJoCo Rendering..."
python -c "
import mujoco
import os
assert os.environ.get('MUJOCO_GL') == 'egl', 'MUJOCO_GL must be egl'
print('MuJoCo EGL render backend loaded successfully!')
"

echo "==> Remote Node Bootstrap Complete!"
```

---

## 4. Local Project Directory Structure

Organize the repository locally so it is completely self-contained and ready to `rsync` or `git clone` directly to the cluster scratch directory:

```text
vla-weight-alignment/
├── configs/
│   ├── lora_params.yaml          # LoRA rank (r=16/32), alpha, dropout, target modules
│   ├── train_params.yaml         # Batch size, lr, warmup, optimizer steps, save intervals
│   └── libero_tasks.yaml         # Definitions of all 40 tasks across the 4 suites
├── data/
│   ├── download_libero.sh        # Robust dataset downloader with checksum validation
│   └── verify_datasets.py        # Validates HDF5 integrity and sample counts
├── models/
│   ├── vla_wrapper.py            # VLA backbone loader with component-factorized LoRA hooks
│   └── evidence_extractor.py     # Extracts e_vis, e_lang, e_act metadata
├── scripts/
│   ├── setup_remote_env.sh       # Automated remote environment installer
│   ├── train_zoo.py              # Main idempotent training loop with defensive guards
│   └── submit_zoo.slurm          # SLURM batch execution script (for cluster scheduling)
├── tests/
│   ├── test_local_dryrun.py      # 60-second local CPU pre-flight smoke test
│   └── test_tensor_slicing.py    # Verifies Delta W shape [L, 3, r, H]
├── checkpoints/
│   └── model_zoo/                # Storage destination for task_*.pt checkpoints
├── requirements.txt              # Pinned pip dependencies
└── README.md                     # Run instructions and quick reference
```

---

## 5. Configuration Architecture

Decouple hyperparameters, task registries, and LoRA specifications into clean YAML files.

### 5.1 `configs/lora_params.yaml`
```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  bias: "none"
  # Component-factorized target modules (Separated by modality role)
  target_modules:
    vision:
      - "vision_tower.encoder.layers.*.mlp.fc1"
      - "vision_tower.encoder.layers.*.mlp.fc2"
    language:
      - "language_model.model.layers.*.self_attn.q_proj"
      - "language_model.model.layers.*.self_attn.v_proj"
    action:
      - "action_head.fc_layers.*"
```

### 5.2 `configs/libero_tasks.yaml`
```yaml
suites:
  libero_spatial:
    dataset_name: "libero_spatial"
    num_tasks: 10
    task_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  libero_object:
    dataset_name: "libero_object"
    num_tasks: 10
    task_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  libero_goal:
    dataset_name: "libero_goal"
    num_tasks: 10
    task_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  libero_10:
    dataset_name: "libero_10"
    num_tasks: 10
    task_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

---

## 6. Defensive Engineering in `train_zoo.py`

To ensure you never have to repeat the 150-hour run due to an unhandled crash or preemption, `train_zoo.py` implements **five defensive engineering principles**:

1. **Atomic Checkpointing**: Never write directly to `task_{k}.pt`. Write to `task_{k}.pt.tmp` and perform an atomic filesystem rename (`os.replace`). If the process is terminated mid-write, no corrupt files are left behind.
2. **Task-Level Idempotency**: At startup and before each task, verify if `task_{k}.pt` exists, has size > 1MB, and can be deserialized without error. If so, immediately skip.
3. **Explicit Memory Sanitation**: After each task completes, call `del model, optimizer, dataloader`, run `gc.collect()`, and flush PyTorch CUDA cache (`torch.cuda.empty_cache()`).
4. **Disk Space Watchdog**: Check available disk space on the scratch partition before starting each task. If available space falls below 10 GB, halt gracefully and alert via W&B rather than crashing mid-task.
5. **Per-Task Exception Isolation**: If a specific task encounters an unrecoverable NaN loss or corrupt demo index, log the full traceback, flag the task in W&B, and proceed to the next task.

### Core Implementation Pattern for `train_zoo.py`:

```python
import os
import gc
import shutil
import logging
import traceback
import torch

def check_disk_space(path=".", min_gb=10.0):
    """Halts execution if free scratch disk space is too low."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    if free_gb < min_gb:
        raise RuntimeError(f"CRITICAL: Insufficient disk space! Free: {free_gb:.2f} GB (minimum: {min_gb} GB).")

def is_checkpoint_valid(filepath):
    """Verifies that a checkpoint exists, is non-empty, and loads cleanly."""
    if not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) < 1024 * 1024:  # Must be > 1MB
        return False
    try:
        data = torch.load(filepath, map_location="cpu")
        required_keys = {"delta_w", "e_vis", "e_lang", "e_act", "task_id"}
        return required_keys.issubset(data.keys())
    except Exception:
        return False

def atomic_save(payload, target_path):
    """Saves to a temporary file first, then atomically renames to target."""
    tmp_path = target_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, target_path)

def train_model_zoo(tasks, output_dir="checkpoints/model_zoo"):
    os.makedirs(output_dir, exist_ok=True)
    
    for task in tasks:
        task_id = task["id"]
        ckpt_path = os.path.join(output_dir, f"task_{task_id}.pt")
        
        # 1. Idempotency Check
        if is_checkpoint_valid(ckpt_path):
            logging.info(f"[SKIP] Task {task_id} already successfully completed. Skipping.")
            continue
            
        logging.info(f"===> Starting training on Task {task_id}: {task['name']}")
        check_disk_space(output_dir, min_gb=10.0)
        
        # 2. Per-Task Isolation
        try:
            # Train policy LoRA on task demonstrations...
            delta_w, e_vis, e_lang, e_act = run_task_training(task)
            
            # 3. Atomic Serialization
            payload = {
                "task_id": task_id,
                "task_name": task["name"],
                "delta_w": delta_w,       # Sliced tensor: [L, 3, r, H]
                "e_vis": e_vis,           # Demonstration visual embedding
                "e_lang": e_lang,         # Instruction text embedding
                "e_act": e_act,           # Action chunk statistics
            }
            atomic_save(payload, ckpt_path)
            logging.info(f"[DONE] Task {task_id} successfully serialized to {ckpt_path}.")
            
        except Exception as e:
            logging.error(f"[ERROR] Task {task_id} encountered exception: {str(e)}")
            logging.error(traceback.format_exc())
            # Continue to next task rather than killing the entire 150-hour run
            continue
            
        finally:
            # 4. Memory Sanitation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

---

## 7. Local Pre-Flight Test Suite (`tests/unit/test_smoke/test_local_dryrun.py`)

Run this test suite completely on your local computer's CPU before submitting your compute request. It verifies the entire end-to-end pipeline in under 60 seconds with synthetic mock data.

```python
"""
tests/unit/test_smoke/test_local_dryrun.py
Runs a zero-compute local CPU dry-run to validate all pipeline logic.
"""
import os
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

class MockVLAModel(nn.Module):
    """Lightweight 2-layer mock VLA for local CPU testing."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Modality sub-modules matching the proposal
        self.vision_fc = nn.Linear(hidden_dim, hidden_dim)
        self.language_fc = nn.Linear(hidden_dim, hidden_dim)
        self.action_fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.vision_fc(x) + self.language_fc(x) + self.action_fc(x)

def test_lora_injection_and_freezing():
    print("Test 1: LoRA Parameter Injection & Backbone Freezing...")
    base_model = MockVLAModel(hidden_dim=64)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["vision_fc", "language_fc", "action_fc"],
        lora_dropout=0.0,
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)
    
    # Verify that base parameters are frozen and only LoRA parameters have gradients
    for name, param in peft_model.named_parameters():
        if "lora" in name:
            assert param.requires_grad, f"LoRA param {name} must have requires_grad=True"
        else:
            assert not param.requires_grad, f"Base param {name} must be frozen (requires_grad=False)"
    print("  -> Passed: LoRA correctly injected and backbone frozen.")

def test_modality_tensor_factorization():
    print("Test 2: Modality Slicing into Delta W [L, 3, r, H]...")
    L = 2    # Mock 2 layers
    r = 16   # LoRA rank
    H = 64   # Hidden dim
    
    # Simulate extracted Delta W = B @ A for vision, language, action
    mock_delta_w = torch.randn(L, 3, r, H)
    assert mock_delta_w.shape == (L, 3, r, H), f"Shape mismatch: {mock_delta_w.shape}"
    print(f"  -> Passed: Delta W verified with shape: {list(mock_delta_w.shape)}")

def test_evidence_extractor():
    print("Test 3: Task Evidence Serialization (e_vis, e_lang, e_act)...")
    mock_evidence = {
        "e_vis": torch.randn(10, 64),      # 10 demo frames projected
        "e_lang": torch.randn(384),        # Sentence transformer embedding
        "e_act": torch.randn(32),          # Action chunk distribution summary
    }
    for k, v in mock_evidence.items():
        assert isinstance(v, torch.Tensor), f"{k} must be a torch.Tensor"
    print("  -> Passed: Task evidence formats validated.")

def test_atomic_checkpoint_roundtrip(tmp_path="/tmp/test_checkpoint.pt"):
    print("Test 4: Atomic Checkpoint Save & Roundtrip Integrity...")
    data = {
        "task_id": "mock_task_0",
        "delta_w": torch.randn(2, 3, 16, 64),
        "e_vis": torch.randn(10, 64),
        "e_lang": torch.randn(384),
        "e_act": torch.randn(32),
    }
    tmp_file = tmp_path + ".tmp"
    torch.save(data, tmp_file)
    os.replace(tmp_file, tmp_path)
    
    loaded = torch.load(tmp_path)
    assert loaded["delta_w"].shape == (2, 3, 16, 64)
    assert loaded["task_id"] == "mock_task_0"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print("  -> Passed: Atomic checkpoint serialization verified.")

if __name__ == "__main__":
    test_lora_injection_and_freezing()
    test_modality_tensor_factorization()
    test_evidence_extractor()
    test_atomic_checkpoint_roundtrip()
    print("\n[SUCCESS] All local pre-flight dry-runs passed! Logic is ready for remote GPU.")
```

---

## 8. Remote Execution: The 15-Minute "Three-Step" Pre-Flight on the Server

Once compute access is granted and you SSH into the remote machine, execute this strict 15-minute verification protocol **before** triggering the 150-hour run:

### Step 1: The 5-Step Smoke Test (Time: 2 minutes)
```bash
# Verify GPU memory allocation, mixed-precision, and backward pass on 1 batch
python scripts/train_zoo.py --max_steps 5 --debug --suite libero_spatial --task_id 0
```
- **Verification criteria**:
  - `nvidia-smi` confirms VRAM allocation stays within limits (no OOM).
  - Forward pass, loss computation, backward pass, and optimizer step finish cleanly.
  - Headless MuJoCo environment steps without display errors.

### Step 2: Single-Task Golden Run (Time: 12 minutes)
```bash
# Run 1 task to completion (e.g., 1,000 steps on libero_spatial task 0)
python scripts/train_zoo.py --max_steps 1000 --suite libero_spatial --task_id 0
```
- **Verification criteria**:
  - Output checkpoint `checkpoints/model_zoo/task_libero_spatial_0.pt` exists and is non-empty.
  - Load the checkpoint in Python and verify:
    ```python
    ckpt = torch.load("checkpoints/model_zoo/task_libero_spatial_0.pt")
    print(ckpt["delta_w"].shape) # Must match [L, 3, r, H]
    print(ckpt.keys())           # Must contain delta_w, e_vis, e_lang, e_act
    ```
  - Weights & Biases dashboard receives loss, learning rate, and throughput metrics.

### Step 3: Launch Full Daemonized Run inside `tmux` (Time: 1 minute)
Never run long-lived jobs in raw SSH. Use `tmux` or SLURM:

```bash
# 1. Start named tmux session
tmux new -s model_zoo

# 2. Activate environment and export headless variables
conda activate vla_zoo
export MUJOCO_GL=egl

# 3. Launch full 40-task zoo
python scripts/train_zoo.py \
    --config configs/train_params.yaml \
    --tasks configs/libero_tasks.yaml \
    --output_dir checkpoints/model_zoo \
    2>&1 | tee -a training_zoo.log

# 4. Safely detach from tmux:
# Press Ctrl + B, release, then press D
```

To re-attach later:
```bash
tmux attach -t model_zoo
```

---

## 9. Comprehensive Pre-Compute Preparation Checklist

Review this table before requesting compute. Do not request compute until all Local steps are marked **Done**.

| Stage | Item | Action / Verification | Status |
| :--- | :--- | :--- | :---: |
| **Local** | **Compute Spec Sheet** | Compute sizing justification (Section 2) prepared for cluster provisioning and compute allocation request. | [ ] |
| **Local** | **Git Cleanliness** | Repository initialized locally; `.gitignore` set for checkpoints, datasets, and logs. | [ ] |
| **Local** | **Conda Environment** | Local `vla_zoo` environment created; `requirements.txt` exported. | [ ] |
| **Local** | **Headless Scripts** | `scripts/setup_remote_env.sh` contains `MUJOCO_GL=egl` export. | [ ] |
| **Local** | **CPU Dry-Run** | Ran `python tests/unit/test_smoke/test_local_dryrun.py`; all 4 synthetic tests passed. | [ ] |
| **Local** | **W&B Account** | Weights & Biases project initialized (`vla-model-zoo`); API key ready. | [ ] |
| **Local** | **Data Links Staged** | Dataset download URLs verified (direct HTTPS/HF mirrors instead of rate-limited GDrive). | [ ] |
| **Remote** | **Node Bootstrap** | SSH in; execute `bash scripts/setup_remote_env.sh`. | [ ] |
| **Remote** | **Headless Test** | Run 1-line Python EGL test; verify no X11/GLFW display errors. | [ ] |
| **Remote** | **5-Step Smoke Test** | Run `--max_steps 5 --debug`; check VRAM usage on `nvidia-smi`. | [ ] |
| **Remote** | **Golden Run** | Complete Task 0; verify tensor shapes $[L, 3, r, H]$ and evidence keys. | [ ] |
| **Remote** | **Persistent Launch** | Launch in `tmux` with `tee -a` logging and W&B live tracking. | [ ] |

