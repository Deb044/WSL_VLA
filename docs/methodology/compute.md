# Historical Large-VLA Compute Estimate (Deferred Phase)

This 7B-model estimate is not the Octo-Small Methodology 1 execution spec and
must not be cited as measured resource use. Run manifests record actual wall
time, VRAM, RAM, and storage. Follow `docs/METHODOLOGY1_RUNBOOK.md`.

# Compute Allocation Justification & Sizing Spec Sheet: Mathematical Derivations

**Project**: Weight Space Alignment for Continual Learning in Robotics  
**Target Workload**: Training the 40-Task LIBERO Reference Model Zoo ($\Delta W^{(k)} \in \mathbb{R}^{L \times 3 \times r \times H}$)  
**Reference Architecture**: 7B-parameter Vision-Language-Action (VLA) Model (e.g., OpenVLA-7B / Pi0 / Octo) with Parameter-Efficient LoRA Adapters ($r=16, \alpha=32$)

---

## 1. Executive Summary of Requested Resources

| Resource | Value | Governing Formula / Constraint |
| :--- | :--- | :--- |
| **Total Compute** | **160 GPU-Hours** | $40 \text{ tasks} \times 3.48 \text{ hrs/task} + 15\% \text{ safety buffer}$ |
| **GPU Architecture** | **1x NVIDIA A100 (80GB/40GB)** or **1x RTX 4090 (24GB)** | Peak VRAM consumption: **$21.4 \text{ GB}$** in BF16 with activation checkpointing |
| **System RAM** | **64 GB** | Dataset caching ($32.2\text{ GB}$) + OS/PyTorch DataLoader worker pinned memory ($24\text{ GB}$) |
| **Scratch Disk** | **150 GB NVMe** | Base model ($22\text{ GB}$) + datasets ($33\text{ GB}$) + env ($15\text{ GB}$) + checkpoints ($8\text{ GB}$) + buffer ($45\text{ GB}$) |
| **Recommended Topologies** | **1x A100-80GB** for 6.7 days **OR** **4x A100-80GB** for 40 hours | Tasks are independent; task-level parallelism achieves **$99.4\%$ linear scaling efficiency** |

---

## 2. Compute & Walltime Derivations (FLOPs to GPU-Hours)

### 2.1 Model Dimensions & Sequence Length
For a standard 7B-class VLA (Prismatic / OpenVLA / Llama-2-7B backbone + SigLIP-400M vision encoder):
- **Base parameters ($N$)**: $7.0 \times 10^9$ parameters.
- **Hidden dimension ($H$)**: $4096$.
- **Transformer layers ($L$)**: $32$.
- **LoRA rank ($r$)**: $16$.
- **Sequence length ($S$) per observation**:
  - Image patch tokens (SigLIP $224 \times 224$ / patch 14): $16 \times 16 = 256$ tokens.
  - Task instruction tokens: $\approx 32$ tokens.
  - Action chunk tokens (7 DoF robot actions $\times$ chunk size): $\approx 12$ tokens.
  - Total tokens per step ($S$): $256 + 32 + 12 = 300\text{ tokens}$.

### 2.2 FLOPs per Optimization Step
$$\text{FLOPs}_{\text{sample}} = 4N \times S = 8.40 \times 10^{12} \text{ FLOPs}$$
$$\text{FLOPs}_{\text{step}} = 8 \times 8.40 \times 10^{12} = 67.2 \text{ TFLOPs} \quad (B=8)$$

### 2.3 Hardware Throughput & Realized MFU
On an **NVIDIA A100-SXM4-80GB**:
$$\mathcal{T}_{\text{realized}} = 312 \times 0.38 = 118.56 \text{ TFLOPS}$$
$$t_{\text{step, actual}} \approx 0.65 \text{ seconds/step} \quad (\approx 1.54 \text{ steps/second})$$

### 2.4 Walltime Calculation
$$T_{\text{train}} = 15,000 \times 0.65\text{ s} = 2.71 \text{ hours/task}$$
Including evaluation and recovery buffers:
$$\mathbf{T_{\text{requested}} \approx 160 \text{ GPU-Hours}}$$

---

## 3. Storage Breakdown (150 GB NVMe Requested)
- Base Model Weights: 22.0 GB
- LIBERO 4-Suite Datasets: 32.2 GB
- Conda Environment & CUDA: 14.8 GB
- Model Zoo 40 Checkpoints: 7.8 GB
- Evaluation Rollout Logs & Videos: 12.0 GB
- OS Buffer & Shared Memory: 45.0 GB
- **Total: 133.8 GB $\implies$ 150 GB Requested**
