# VLA Model Zoo Construction Pipeline (Octo-Small & Modular VLA)

**Project**: Weight Space Alignment for Continual Learning in Robotics  
**Default Model**: **Octo-Small (27M parameters)** with Component-Factorized LoRA  
**Target Output**: Modality-Factorized LoRA Population ($\Delta W^{(k)} \in \mathbb{R}^{L \times 3 \times r \times H}$) with Multi-Modal Task Evidence ($\mathcal{M}^{(k)} = \{e_{\text{vis}}^{(k)}, e_{\text{lang}}^{(k)}, e_{\text{act}}^{(k)}\}$)

---

## Clean Directory Organization

```text
WSL_VLA/
├── configs/
│   ├── vla_config.yaml         # Active model config (Octo-Small default), LoRA rank, training params
│   └── tasks_config.yaml       # 40-task registry across the 4 LIBERO benchmark suites
├── data/
│   ├── dataset.py              # PyTorch Dataset supporting auto-discovery of real HDF5 and synthetic fallback
│   ├── download_libero.py      # Targeted downloader (downloads single suites or task subsets)
│   └── download_libero.sh      # Shell script to fetch all LIBERO suites (cluster use)
├── docs/
│   ├── compute.md              # Mathematical derivations for GPU, VRAM, and storage sizing
│   └── MODEL_ZOO_PREFLIGHT_PLAN.md # Strategic Zero-Fail pre-flight guide and checklist
├── models/
│   ├── base_vla.py             # Abstract BaseVLA interface & build_vla_model factory
│   ├── octo_small.py           # Octo-Small (27M) with factorized LoRA & [8, 3, 16, 384] extraction
│   ├── small_vla.py            # Lightweight 1.2M model for instant dry-runs
│   ├── download_octo_weights.py# Downloader for official UC Berkeley Octo-Small checkpoint
│   └── evidence_extractor.py   # DeepSets (e_vis), Sentence-Transformer (e_lang), Action stats (e_act)
├── scripts/
│   ├── train_zoo.py            # Resilient multi-task training loop with atomic saving and TF32 acceleration
│   └── verify_zoo.py           # Integrity validator & population tensor aggregator
├── tests/
│   └── test_local_dryrun.py    # Local pre-flight dry-run suite testing Octo-Small, factory & LoRA
├── setup_env.sh                # Automated Linux/WSL environment installer (Conda, PyTorch CUDA, EGL)
├── setup_mac.sh                # Automated macOS installer (Apple Silicon MPS, CGL)
├── requirements.txt            # Pinned Python dependencies
├── environment.yml             # Conda environment definition
├── .gitignore                  # Excludes heavy datasets (data/libero), checkpoints, and cache
└── README.md                   # Project overview & quickstart guide
```

---

## Quickstart Guide

### 1. Environment Setup

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

### 2. Run Local Pre-Flight Dry-Run
Run the self-contained unit test suite (~10 seconds, CPU):
```bash
python tests/test_local_dryrun.py
```

---

### 3. Stage Datasets (Disk Saver Option)
To save storage, download only a single suite or subset:

* **Download 2 tasks of `libero_spatial` (~600 MB)**:
  ```bash
  python data/download_libero.py --suite libero_spatial --max_tasks 2
  ```

* **Download all 10 tasks of `libero_spatial` (~3.2 GB)**:
  ```bash
  python data/download_libero.py --suite libero_spatial
  ```

---

### 4. Train the Model Zoo

* **Train first 2 tasks**:
  ```bash
  python scripts/train_zoo.py --limit_tasks 2 --max_steps 500
  ```

* **Train full 40-task zoo**:
  ```bash
  python scripts/train_zoo.py
  ```

---

### 5. Verify Checkpoints
Verify that all checkpoints extract valid tensors:
```bash
python scripts/verify_zoo.py --dir ./checkpoints/model_zoo --expected 2
```
Expected output:
$$\text{Population Tensor Shape: } [N, 8, 3, 16, 384]$$
$$\text{Evidence: } e_{\text{vis}} [128], \quad e_{\text{lang}} [384], \quad e_{\text{act}} [28]$$
