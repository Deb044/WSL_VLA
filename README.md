# WSL_VLA: Modality-Aligned Weight-Space Learning for Continual Robot Policies

WSL_VLA provides a mechanism-first study of modality-aligned weight-space learning for continual Vision-Language-Action (VLA) robot policies. It interfaces directly with the official pretrained **Octo-Small 1.5** backbone (`rail-berkeley/octo-small-1.5`), introduces zero-initialized modality adapters across observation, language, and action diffusion tokens, aligns effective weight updates with multi-modal demonstration evidence in a shared latent space, and evaluates evolving policies via rollout-based continual learning metrics.

The codebase contains **one unified, publication-grade implementation** organized into standard, intuitive modules: `models/`, `data/`, `core/`, `configs/`, `scripts/`, and `tests/`. There are no legacy tiers, no synthetic data generators, and no toy prototypes.

---

## 1. System Architecture

```text
Official Octo-Small 1.5 + Real LIBERO HDF5 Demonstrations
     │
     ▼
[models/octo_model.py] ── Zero-initialized modality-specific residual adapters
     │                     (vision, language, action-readout tokens + diffusion head)
     ▼
[models/packing.py] ──── Basis-invariant effective updates (ΔW = B · A) packed into
     │                     fixed-width row windows (gauge-invariant)
     ▼
[models/weight_autoencoder.py] ── Component-factorized masked autoencoder (g_phi, h_psi)
     │
     ▼
[models/contrastive_alignment.py] ── Multi-positive InfoNCE contrastive alignment
     │                                 against DeepSets task evidence (e_vis, e_lang, e_act)
     ▼
[models/differential_regularizer.py] ── Latent refinement on Frobenius hyperspherical shell (Π_shell)
     │                                    with asymmetric penalties: γ_vis, γ_lang > γ_act
     ▼
[core/sequential.py] ──── Continual learning across locked task sequences (4 suites × 10 tasks)
     │
     ▼
[core/metrics.py] ────── SR, NBT, normalized NBT, forgetting, forward transfer, recovery ratio
```

---

## 2. Directory Map

| Directory | Purpose | Core Modules |
|---|---|---|
| `models/` | Neural architectures and adaptation algorithms | `octo_model.py`, `flax_adapters.py`, `octo_training.py`, `weight_autoencoder.py`, `packing.py`, `contrastive_alignment.py`, `differential_regularizer.py`, `evidence_extractor.py`, `prompt_mapper.py`, `alignment_eval.py`, `component_analysis.py`, `latent_adapter.py` |
| `data/` | Strict dataset loading and preprocessing | `dataset.py` (StrictLiberoHDF5 fail-closed loader), `batches.py` (Octo batch format & action normalization), `splits.py` (leakage-resistant train/val splits) |
| `core/` | Invariants, contracts, and evaluation | `contracts.py` (AdapterSpec, TaskEvidence, RunManifest), `protocol.py` (suite rules), `provenance.py` (SHA256 & environment capture), `metrics.py` (CL metrics & bootstrap CIs), `records.py` (rollout records), `sequential.py` (sequential runner) |
| `configs/` | Canonical experiment configurations | `base.yaml` (hyperparameters & execution bounds), `folds.yaml` (leave-one-suite-out folds), `reference_tasks.yaml` (locked 40 LIBERO tasks) |
| `scripts/` | Executable commands and CLI pipelines | `preflight.py`, `verify_octo.py`, `train_zoo.py`, `train_alignment.py`, `run_continual.py`, `report_metrics.py`, `build_alignment_archive.py`, `write_run_manifest.py`, `download_libero.py`, `download_octo.py` |
| `tests/` | Unit and integration test suite | 24 comprehensive test suites verifying contracts, packing, splits, metrics, and models |

For a complete retrospective on architectural decisions, engineering rationales, and the elimination of earlier prototypes, see [`Explanation.md`](file:///home/bruhwhy/ICML/WSL_VLA/Explanation.md).  
For the comprehensive audit of implementation, mathematical rigor, and progress left to complete the research proposal, see [`docs/PROPOSAL_IMPLEMENTATION_AND_PROGRESS.md`](file:///home/bruhwhy/ICML/WSL_VLA/docs/PROPOSAL_IMPLEMENTATION_AND_PROGRESS.md).

---

## 3. Getting Started & Environment Setup

### Prerequisites
- Python 3.10 or 3.11
- Recommended: Linux / WSL2 with an NVIDIA GPU (RTX 30xx/40xx or A100/H100)
- Conda package manager ([Miniforge](https://github.com/conda-forge/miniforge) or [Miniconda](https://docs.anaconda.com/miniconda/))

---

### Option A: Automated One-Command Setup (Recommended)

Run the automated installer script for Linux / WSL2:
```bash
./setup_env.sh
```
This script automatically:
1. Detects your Conda installation and creates/updates the `vla_zoo` environment with Python 3.10.
2. Checks for an NVIDIA GPU via `nvidia-smi` and installs JAX with CUDA acceleration (or CPU fallback).
3. Configures headless rendering and memory variables in `~/.bashrc` (`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MUJOCO_GL=egl`).
4. Installs all model dependencies from `requirements.txt` and packages `wsl-vla` in editable mode (`pip install -e .`).
5. Executes the full 22-test suite (`pytest tests/`) to verify everything is working out of the box.

After completion, simply activate:
```bash
conda activate vla_zoo
```

*(For macOS run `./setup_mac.sh`, and for Windows PowerShell run `.\setup_windows.ps1`)*

---

### Option B: Conda Environment Setup (`environment.yml`)

Create the pre-configured `vla_zoo` environment directly from `environment.yml`:
```bash
conda env create -f environment.yml
conda activate vla_zoo
```

---

### Option C: Manual Installation in an Existing Environment

If you already have a Conda environment (such as `vla_zoo`):

```bash
conda activate vla_zoo

# 1. Install JAX:
# For NVIDIA GPUs with CUDA 12:
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Or for CPU-only:
# pip install jax jaxlib

# 2. Install core research dependencies:
pip install -r requirements.txt

# 3. Install the WSL_VLA package in editable mode:
pip install -e .
```

---

### Recommended Environment Variables

Add these to your shell profile (`~/.bashrc` or `~/.zshrc`) to ensure smooth execution:
```bash
# Prevent JAX from hogging 90% of GPU memory upfront
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Enable headless EGL hardware rendering for MuJoCo simulation rollouts
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

---

### Verification & Preflight Check

Verify that all models, contracts, and JAX autodiff components are operational:
```bash
# Run full unit test suite (22 tests)
pytest tests/

# Verify official Octo zero-adapter equivalence
python scripts/verify_octo.py

# Verify system environment and real LIBERO dataset integrity
python scripts/preflight.py --config configs/base.yaml
```

---

## 4. End-to-End Pipeline

### Step 1: Model Zoo Adapter Training
Train task-specific modality adapters on real LIBERO demonstrations for all tasks:
```bash
python scripts/train_zoo.py --suite libero_spatial --task-id libero_spatial_0 --seed 42
```

### Step 2: Assemble Alignment Archive
Pack trained low-rank adapters into a validated, basis-invariant NPZ archive:
```bash
python scripts/build_alignment_archive.py research_results/population \
    --output research_results/alignment/archive.npz \
    --held-out-suite libero_10
```

### Step 3: Train Aligned Weight Autoencoder
Train the joint weight autoencoder and multi-positive InfoNCE aligner:
```bash
python scripts/train_alignment.py research_results/alignment/archive.npz \
    --output research_results/checkpoints/alignment_system.pt \
    --epochs 200
```

### Step 4: Continual Learning & Evaluation
Run sequential continual adaptation under proposed asymmetric differential regularization:
```bash
python scripts/run_continual.py \
    --suite libero_spatial \
    --condition proposed_asymmetric \
    --seed 42 \
    --rollout-count 20 \
    --output-records research_results/records/eval_records.jsonl
```

### Step 5: Report Metrics
Compute publication-grade continual learning metrics (ASR, NBT, normalized NBT, forgetting):
```bash
python scripts/report_metrics.py research_results/records/eval_records.jsonl \
    --suite libero_spatial \
    --seed 42 \
    --condition proposed_asymmetric \
    --output-json research_results/metrics_report.json
```

---

## 5. Strict Research Invariants

1. **Zero Synthetic Data**: All loaders fail closed if real LIBERO HDF5 demonstration files are absent. Synthetic data cannot be written to research outputs.
2. **Canonical Implementation**: A single, publication-grade implementation directly backed by official Octo-Small 1.5 (`rail-berkeley/octo-small-1.5`).
3. **Basis Invariance**: Weight packing operates on effective updates $\Delta W = B \cdot A$ rather than raw low-rank factor matrices to remain invariant to arbitrary internal LoRA gauge transformations.
4. **Reproducible Provenance**: Every experiment run emits an atomic `RunManifest` tracking exact git commit, dirty flag, full Python environment, GPU hardware specifications, dataset SHA256 hashes, and base checkpoint hashes.
