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
pytest tests/
```
Executes the comprehensive 24-test suite validating contract schemas, packing basis invariance, episode splitting, action statistics, JAX latent optimization, and continual learning metrics.

### 2.2 Model Zoo & Alignment Pipeline
1. **Preflight Verification**:
   ```bash
   python scripts/preflight.py
   ```
2. **Zero-Adapter Equivalence Check**:
   ```bash
   python scripts/verify_octo.py
   ```
3. **Train Task Modality Adapters**:
   ```bash
   python scripts/train_zoo.py --suite libero_spatial
   ```
4. **Build Alignment Archive**:
   ```bash
   python scripts/build_alignment_archive.py research_results/population --held-out-suite libero_spatial --output research_results/archive_fold0.npz
   ```
5. **Train Weight Autoencoder & InfoNCE Alignment**:
   ```bash
   python scripts/train_alignment.py research_results/archive_fold0.npz --output research_results/checkpoints/alignment_fold0
   ```

### 2.3 Sequential Continual Adaptation
```bash
python scripts/run_continual.py --suite libero_spatial --condition proposed_asymmetric --seed 42
```
Trains one evolving policy across sequential task stages under locked regularizations (unregularized, uniform, and proposed asymmetric differential regularization), evaluating the policy at each stage on all seen tasks.

### 2.4 Compute Continual Learning Publication Metrics
```bash
python scripts/report_metrics.py research_results/records/eval_records.jsonl --suite libero_spatial --seed 42 --condition proposed_asymmetric
```
Computes Average Success Rate (ASR), Negative Backward Transfer (NBT), Normalized NBT, Average Forgetting, Forward Transfer, and bootstrap confidence intervals.
