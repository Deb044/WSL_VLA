# VLA Model Zoo & Weight Space Alignment Pipeline

**Project**: Weight Space Alignment for Continual Learning in Robotics  
**Default Model**: **Octo-Small (27M parameters)** with Component-Factorized LoRA  
**Target Output**: Modality-Factorized LoRA Population ($\Delta W^{(k)} \in \mathbb{R}^{L \times 3 \times r \times H}$) with Multi-Modal Task Evidence ($\mathcal{M}^{(k)} = \{e_{\text{vis}}^{(k)}, e_{\text{lang}}^{(k)}, e_{\text{act}}^{(k)}\}$) and Continual Weight Space Alignment ($g_\phi, h_\psi$).

---

## Directory Organization

```text
WSL_VLA/
├── configs/
│   ├── vla_config.yaml               # Active model config (Octo-Small default), LoRA rank, training params
│   └── tasks_config.yaml             # 40-task registry across the 4 LIBERO benchmark suites
├── data/
│   └── dataset.py                    # PyTorch Dataset with auto-discovery of real HDF5 and synthetic fallback
├── docs/
│   ├── METHODOLOGY_1_GUIDE.md        # Complete mathematical formulation and guide for Methodology 1
│   ├── MODEL_ZOO_PREFLIGHT_PLAN.md   # Strategic pre-flight plan and verification checklist
│   └── compute.md                    # GPU, VRAM, and storage sizing derivations
├── models/
│   ├── base_vla.py                   # Abstract BaseVLA interface & build_vla_model factory
│   ├── octo_small.py                 # Octo-Small (27M) with factorized LoRA & [8, 3, 16, 384] extract/inject
│   ├── small_vla.py                  # Lightweight 1.2M model for rapid CPU dry-runs
│   ├── evidence_extractor.py         # DeepSets (e_vis), Sentence-Transformer (e_lang), Action stats (e_act)
│   ├── weight_autoencoder.py         # Component-Factorized Weight Autoencoder (g_phi, h_psi) [Methodology 1.1]
│   ├── contrastive_alignment.py      # Modality-Specific InfoNCE Contrastive Alignment [Methodology 1.2]
│   └── differential_regularizer.py   # Asymmetric Differential Regularizer & Pi_shell [Methodology 1.3]
├── scripts/
│   ├── download_libero.py            # Unified downloader for LIBERO benchmark suites and task subsets
│   ├── download_octo.py              # Downloader for official pre-trained Octo-Small weights
│   ├── train_zoo.py                  # Multi-task training loop with atomic saving and crash recovery
│   ├── orchestrate_zoo.py            # Automated multi-suite Model Zoo pipeline execution
│   ├── verify_zoo.py                 # Population integrity validator & [N, 8, 3, 16, 384] tensor aggregator
│   ├── evaluate_vla.py               # Comprehensive evaluation suite (MSE, RMSE, Cosine Sim, Gripper Acc)
│   ├── train_alignment.py            # Weight-Space Contrastive Alignment training loop [Methodology 1.2]
│   ├── sequential_adaptation.py      # Continual learning stream adaptation with gamma_vis > gamma_act [Methodology 1.3]
│   ├── run_research_benchmark.py     # Full continual learning benchmark: Pearson correlation & 4-way ablations
│   └── compare_base_vs_adapted.py    # Direct head-to-head empirical benchmark: Base VLA vs Adapted VLA
├── tests/
│   ├── test_local_dryrun.py          # Pre-flight unit tests (Octo-Small, factory, LoRA, evidence, roundtrip)
│   └── test_methodology1.py          # Methodology 1 unit tests (Autoencoder, Alignment loss, Pi_shell, Reg)
├── setup_windows.ps1                 # Automated Windows PowerShell environment installer
├── setup_env.sh                      # Automated Linux/WSL installer (CUDA, EGL)
├── setup_mac.sh                      # Automated macOS installer (MPS, CGL)
├── requirements.txt                  # Pinned dependencies
├── environment.yml                   # Conda environment definition
└── .gitignore                        # Excludes large binaries (HDF5 datasets, checkpoints, caches)
```

---

## Environment Setup & Unit Tests

```bash
# 1. Environment installation (Ubuntu/Linux)
bash setup_env.sh
conda activate vla_zoo

# 2. Run unit tests
python tests/test_local_dryrun.py       # Pre-flight architecture and factory tests
python tests/test_methodology1.py       # Weight autoencoder and alignment tests
```

---

## Scripts Usage Guide

### 1. Data & Pre-Trained Weights Staging

#### `scripts/download_libero.py`
Downloads demonstration datasets from Hugging Face (`yifengzhu-hf/LIBERO-datasets`).
```bash
# Download all 40 tasks across all suites
python scripts/download_libero.py --suite all

# Download a specific suite (or limit to N tasks to conserve disk)
python scripts/download_libero.py --suite libero_spatial --max_tasks 4
```
* Key flags: `--suite {all, libero_spatial, libero_object, libero_goal, libero_10}`, `--dest <path>`, `--max_tasks <int>`, `--token <hf_token>`.

#### `scripts/download_octo.py`
Downloads official pre-trained Octo-Small checkpoint weights (~108 MB) from `rail-berkeley/octo-small-1.5`.
```bash
python scripts/download_octo.py --dest ./checkpoints/octo_pretrained
```

---

### 2. Model Zoo Training & Verification

#### `scripts/train_zoo.py`
Trains task-specific factorized LoRA weights $\Delta W \in \mathbb{R}^{L \times 3 \times r \times H}$ and extracts task evidence.
```bash
# Train a specific task
python scripts/train_zoo.py --task libero_spatial_0 --max_steps 500

# Train an entire suite
python scripts/train_zoo.py --suite libero_spatial --max_steps 500
```
* Key flags: `--task <task_id>`, `--suite <suite_name>`, `--max_steps <int>`, `--batch_size <int>`.

#### `scripts/orchestrate_zoo.py`
Automates the full pipeline: sequentially stages datasets and trains checkpoints across all 4 suites.
```bash
python scripts/orchestrate_zoo.py --max_steps 500
```

#### `scripts/verify_zoo.py`
Validates checkpoint integrity, tensor dimensions $[L, 3, r, H]$, and aggregates all 40 checkpoints into a tensor stack.
```bash
python scripts/verify_zoo.py --dir ./checkpoints/model_zoo
```

---

### 3. Weight Space Alignment & Continual Learning

#### `scripts/train_alignment.py`
Trains the Component-Factorized Weight Autoencoder $(g_\phi, h_\psi)$ and Modality-Specific InfoNCE Contrastive Aligner.
```bash
python scripts/train_alignment.py --checkpoint_dir ./checkpoints/model_zoo --epochs 150 --batch_size 40
```
* Key flags: `--checkpoint_dir <dir>`, `--epochs <int>`, `--batch_size <int>`, `--d_latent <int>`.

#### `scripts/sequential_adaptation.py`
Runs sequential task stream adaptation with custom differential regularization ratios ($\gamma_{\text{vis}}, \gamma_{\text{lang}}, \gamma_{\text{act}}$).
```bash
python scripts/sequential_adaptation.py --schedule fixed --gamma_vis 1.0 --gamma_lang 1.0 --gamma_act 0.2
```
* Key flags: `--schedule {fixed, drift_informed, adaptive}`, `--gamma_vis <float>`, `--gamma_lang <float>`, `--gamma_act <float>`.

#### `scripts/run_research_benchmark.py`
Executes the comprehensive continual learning benchmark (4-way ablations, Component-Swap protocol, Pearson correlations, NBT).
```bash
# Run all 4 ablation regimes across all 40 LIBERO tasks
python scripts/run_research_benchmark.py --suite all --condition all

# Run a specific experimental condition or suite
python scripts/run_research_benchmark.py --suite libero_spatial --condition proposed
```
* Key flags:
  * `--suite {all, libero_spatial, libero_object, libero_goal, libero_10}`
  * `--condition {all, proposed, uniform, direction_inverted, drift_informed}`
  * `--refine_steps <int>` (default: 20)

#### `scripts/sweep_gammas.py`
Performs an empirical hyperparameter sweep over $(\gamma_{\text{vis}}, \gamma_{\text{lang}}, \gamma_{\text{act}})$ using end-to-end differentiable latent optimization to identify Pareto-optimal regularization values.
```bash
# Run focused grid search across tasks
python scripts/sweep_gammas.py --suite all --max_tasks 10 --search_type focused_grid

# Run sensitivity sweep on action head or vision/language regularizers
python scripts/sweep_gammas.py --suite all --max_tasks 10 --search_type sensitivity_act
python scripts/sweep_gammas.py --suite all --max_tasks 10 --search_type sensitivity_vis
```
* Key flags:
  * `--search_type {focused_grid, sensitivity_act, sensitivity_vis, full_grid}`
  * `--suite {all, libero_spatial, libero_object, libero_goal, libero_10}`
  * `--max_tasks <int>` (default: 5)
  * `--refine_steps <int>` (default: 25)

---

### 4. Evaluation & Head-to-Head Benchmarks

#### `scripts/compare_base_vs_adapted.py`
Empirical head-to-head comparison of zero-shot unadapted Base VLA against the VLA equipped with task weight adapters.
```bash
# Compare across all 40 LIBERO tasks
python scripts/compare_base_vs_adapted.py --suite all

# Compare on a specific suite
python scripts/compare_base_vs_adapted.py --suite libero_spatial
```

#### `scripts/evaluate_vla.py`
Computes detailed trajectory tracking metrics: Action MSE/RMSE, directional cosine similarity, Cartesian translation/rotation error, and gripper accuracy.
```bash
# Evaluate all tasks in a suite
python scripts/evaluate_vla.py --suite libero_spatial

# Evaluate a specific single task checkpoint
python scripts/evaluate_vla.py --task libero_spatial_0
```
