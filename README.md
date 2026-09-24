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
│   ├── dataset.py                    # PyTorch Dataset with auto-discovery of real HDF5 and synthetic fallback
│   ├── download_libero.py            # Targeted downloader for LIBERO benchmark suites
│   └── download_libero_subset.py     # On-demand downloader for specific tasks/subsets from Hugging Face
├── docs/
│   ├── METHODOLOGY_1_GUIDE.md        # Complete mathematical formulation and guide for Methodology 1
│   ├── MODEL_ZOO_PREFLIGHT_PLAN.md   # Strategic pre-flight plan and verification checklist
│   └── compute.md                    # GPU, VRAM, and storage sizing derivations
├── models/
│   ├── base_vla.py                   # Abstract BaseVLA interface & build_vla_model factory
│   ├── octo_small.py                 # Octo-Small (27M) with factorized LoRA & [8, 3, 16, 384] extraction
│   ├── small_vla.py                  # Lightweight 1.2M model for rapid CPU dry-runs
│   ├── evidence_extractor.py         # DeepSets (e_vis), Sentence-Transformer (e_lang), Action stats (e_act)
│   ├── weight_autoencoder.py         # Component-Factorized Weight Autoencoder (g_phi, h_psi) [Methodology 1.1]
│   ├── contrastive_alignment.py      # Modality-Specific InfoNCE Contrastive Alignment [Methodology 1.2]
│   └── differential_regularizer.py   # Asymmetric Differential Regularizer & Pi_shell [Methodology 1.3]
├── scripts/
│   ├── train_zoo.py                  # Multi-task training loop with atomic saving and crash recovery
│   ├── verify_zoo.py                 # Population integrity validator & [N, 8, 3, 16, 384] tensor aggregator
│   ├── evaluate_vla.py               # Comprehensive evaluation suite (MSE, RMSE, Cosine Sim, Gripper Acc)
│   ├── train_alignment.py            # Weight-Space Contrastive Alignment training loop [Methodology 1.2]
│   └── sequential_adaptation.py      # Continual learning stream adaptation with gamma_vis > gamma_act [Methodology 1.3]
├── tests/
│   ├── test_local_dryrun.py          # Pre-flight unit tests (Octo-Small, factory, LoRA, evidence)
│   └── test_methodology1.py          # Methodology 1 unit tests (Autoencoder, Alignment loss, Pi_shell, Reg)
├── setup_windows.ps1                 # Automated Windows PowerShell environment installer
├── setup_env.sh                      # Automated Linux/WSL installer (CUDA, EGL)
├── setup_mac.sh                      # Automated macOS installer (MPS, CGL)
├── requirements.txt                  # Pinned dependencies
├── environment.yml                   # Conda environment definition
└── .gitignore                        # Excludes large binaries (HDF5 datasets, checkpoints, caches)
```

---

## Quickstart Guide

### 1. Environment Setup

* **On Native Windows (PowerShell)**:
  ```powershell
  .\setup_windows.ps1
  conda activate vla_zoo
  ```

* **On Linux / WSL (Ubuntu)**:
  ```bash
  bash setup_env.sh
  conda activate vla_zoo
  ```

* **On macOS (Apple Silicon M1/M2/M3/M4)**:
  ```bash
  bash setup_mac.sh
  conda activate vla_zoo
  ```

---

### 2. Run Test Suites

* **Local Pre-Flight Dry-Run Suite (All 5 tests)**:
  ```bash
  python tests/test_local_dryrun.py
  ```

* **Methodology 1 Unit Tests (All 4 tests)**:
  ```bash
  python tests/test_methodology1.py
  ```

---

### 3. Stage Datasets

To download real LIBERO demonstration suites from Hugging Face:

* **Download 4 tasks from `libero_spatial` (~2.3 GB)**:
  ```bash
  python scripts/download_libero_subset.py --suite libero_spatial --max_tasks 4
  ```

* **Download tasks from `libero_object`**:
  ```bash
  python scripts/download_libero_subset.py --suite libero_object --max_tasks 2
  ```

---

### 4. Train the Model Zoo

* **Train all available tasks with 500 optimizer steps**:
  ```bash
  python scripts/train_zoo.py --max_steps 500
  ```

* **Verify Population Checkpoints**:
  ```bash
  python scripts/verify_zoo.py --dir ./checkpoints/model_zoo
  ```

* **Evaluate Trajectory Tracking Accuracy**:
  ```bash
  python scripts/evaluate_vla.py
  ```

---

### 5. Methodology 1: Weight Space Alignment & Continual Learning

#### 5.1 Train Weight Space Contrastive Alignment
Compresses and aligns the population weights $\Delta W \in \mathbb{R}^{L \times 3 \times r \times H}$ with multi-modal task prompts $(e_{\text{vis}}, e_{\text{lang}}, e_{\text{act}})$ using modality-specific InfoNCE losses:
```bash
python scripts/train_alignment.py --checkpoint_dir ./checkpoints/model_zoo --epochs 100
```

#### 5.2 Sequential Adaptation (Continual Learning Stream)
Performs continual learning with **Differential Regularization** ($\gamma_{\text{vis}}, \gamma_{\text{lang}} > \gamma_{\text{act}}$), mitigating catastrophic forgetting of vision/language components:
```bash
python scripts/sequential_adaptation.py --schedule fixed --gamma_vis 1.0 --gamma_lang 1.0 --gamma_act 0.2
```

Supported schedules:
- `--schedule fixed`: Fixed ratio regularization.
- `--schedule drift_informed`: Regularization scaled by historical drift sensitivity.
- `--schedule adaptive`: Online live drift tracking in latent space.
