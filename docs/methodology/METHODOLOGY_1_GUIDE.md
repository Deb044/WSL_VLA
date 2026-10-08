# Historical Methodology 1 Draft (Do Not Execute)

This file describes the superseded PyTorch surrogate and contains stale tensor
shapes and commands. Use `docs/METHODOLOGY1_RUNBOOK.md` for the official
Octo-Small/JAX workflow and `docs/IMPLEMENTATION_STATUS.md` for current status.

# Methodology 1: Weight Space Alignment & Continual Adaptation

This document details the mathematical formulation, implementation, and evaluation workflow for **Methodology 1** of the research proposal: *Weight Space Alignment for Continual Learning in Robotics*.

---

## 1. Mathematical Formulation

### 1.1 Component-Factorized Tokenization
Given task $k$, its factorized LoRA update is represented as:
$$\Delta W^{(k)} \in \mathbb{R}^{L \times 3 \times r \times H}$$
where $L$ is the number of layers (8 for Octo-Small), $r$ is the LoRA rank (16), $H$ is the hidden dimension (384), and the size-3 axis represents the sub-modules:
- **Modality 0 (Vision)**: `vis_block` $\to \Delta W_{\text{vis}}^{(k)} \in \mathbb{R}^{L \times (r \cdot H)}$
- **Modality 1 (Language)**: `lang_block` $\to \Delta W_{\text{lang}}^{(k)} \in \mathbb{R}^{L \times (r \cdot H)}$
- **Modality 2 (Action)**: `act_block` $\to \Delta W_{\text{act}}^{(k)} \in \mathbb{R}^{L \times (r \cdot H)}$

Each stream is compressed by a shared Transformer Autoencoder $(g_\phi, h_\psi)$:
$$Z_m^{(k)} = g_\phi(\Delta W_m^{(k)}) \in \mathbb{R}^{L \times d_{\text{latent}}}, \quad \widehat{\Delta W}_m^{(k)} = h_\psi(Z_m^{(k)})$$

---

### 1.2 Modality-Specific Contrastive Alignment
For each modality $m \in \{\text{vis}, \text{lang}, \text{act}\}$, we extract task-specific evidence:
- $e_{\text{vis}}$: DeepSets permutation-invariant representation of demonstration camera frames ($d=128$).
- $e_{\text{lang}}$: Dense sentence embedding of the natural language instruction ($d=384$).
- $e_{\text{act}}$: Distribution statistics of action trajectories (mean, std, velocity, jerk) ($d=28$).

We define an independent bidirectional InfoNCE alignment loss per modality with learnable temperature $\tau_m$:
$$\mathcal{L}_{\text{align}}^{(m)} = -\frac{1}{BL} \sum_{i=1}^B \sum_{t=1}^L \log \frac{\exp\left(\tau_m^{-1} \frac{z_{m,t}^{(i) \top} e_m^{(i)}}{\|z_{m,t}^{(i)}\| \|e_m^{(i)}\|}\right)}{\sum_{j=1}^B \exp\left(\tau_m^{-1} \frac{z_{m,t}^{(i) \top} e_m^{(j)}}{\|z_{m,t}^{(i)}\| \|e_m^{(j)}\|}\right)} + (\text{reverse direction})$$

Total training objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{vis}} \mathcal{L}_{\text{align}}^{(\text{vis})} + \lambda_{\text{lang}} \mathcal{L}_{\text{align}}^{(\text{lang})} + \lambda_{\text{act}} \mathcal{L}_{\text{align}}^{(\text{act})} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}$$

---

### 1.3 Differential Regularization During Sequential Adaptation
When adapting sequentially to task stream $k \in \{1, \dots, K\}$, the latent representation is refined directly via the decoder:
$$\mathcal{L}_{\text{ref}}(Z) = \mathcal{L}_{\text{task}}(h_\psi(Z)) + \sum_{m \in \{\text{vis}, \text{lang}, \text{act}\}} \gamma_m \|\Pi_{\text{shell}}(Z_m) - Z_m^{(0)}\|_F^2$$
where $\Pi_{\text{shell}}(Z) = R \cdot \frac{Z}{\|Z\|_F}$ is the hyperspherical shell projection, and:
$$\gamma_{\text{vis}}, \gamma_{\text{lang}} > \gamma_{\text{act}}$$
Fragile vision and language backbone components are constrained more tightly to their aligned anchors, while action components have flexibility to adapt.

---

## 2. Code Architecture & Execution

### 2.1 Unit Tests
```bash
python tests/unit/test_smoke/test_methodology1.py
```
Validates autoencoder encoding/decoding, bidirectional InfoNCE gradients, hyperspherical projection radius, and differential regularizer backward passes.

### 2.2 Alignment Training
```bash
python scripts/train/train_alignment.py --checkpoint_dir ./checkpoints/model_zoo --epochs 100
```
Saves the aligned representation to `./checkpoints/weight_alignment/aligned_weight_autoencoder.pt`.

### 2.3 Sequential Adaptation (Continual Learning)
```bash
python scripts/train/sequential_adaptation.py --schedule fixed --gamma_vis 1.0 --gamma_lang 1.0 --gamma_act 0.2
```

Available schedules:
- `--schedule fixed`: Fixed ratio grid search.
- `--schedule drift_informed`: $\gamma_m$ proportional to historical drift sensitivity.
- `--schedule adaptive`: Online live drift tracking in latent space.

### 2.4 Policy Evaluation
```bash
python scripts/eval/evaluate_vla.py --suite libero_spatial
```
Computes Behavioral Cloning MSE, Cartesian translation RMSE, rotation RMSE, directional cosine similarity, and gripper accuracy across all trained tasks.

### 2.5 Head-to-Head Base vs. Adapted Benchmark
```bash
python scripts/eval/compare_base_vs_adapted.py
```
Directly measures the empirical performance gap between the unadapted Base VLA and the VLA injected with task-specific factorized weight adapters, validating trajectory error reduction, heading alignment, and gripper precision.

### 2.6 Research-Level Continual Learning Benchmark Suite
```bash
python scripts/eval/run_research_benchmark.py --condition all
```
Executes the comprehensive continual learning evaluation protocol:
- Measures the Pearson correlation $r(\text{Latent Drift}, \text{Behavioral Forgetting})$ across visual ($r_{\text{vis}}$), linguistic ($r_{\text{lang}}$), and motor ($r_{\text{act}}$) sub-spaces.
- Evaluates the 4-way differential regularization ablation suite (Proposed vs. Uniform vs. Direction-Inverted vs. Drift-Informed).
- Computes Normalized Backward Transfer (NBT) and per-task kilobyte footprint.
