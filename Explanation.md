# WSL_VLA: Codebase and Research Explanation

## Executive summary

WSL_VLA implements a mechanism-first study of modality-aligned weight-space
learning for continual robot policies. The publication path uses the official
pretrained Octo-Small 1.5 implementation, adds zero-initialized modality adapters
to real language, observation, and action-readout tokens, aligns effective
adapter updates with task evidence, and evaluates one evolving policy with
rollout-based continual-learning metrics.

The repository also retains its original PyTorch policy analogue for fast smoke
tests. That model does not implement official Octo and is explicitly excluded
from research claims. The fastest onboarding path is `README.md`,
`docs/RESEARCH_PIPELINE.md`, `configs/research/base.yaml`, then the modules under
`wsl_vla/research`.

### Scope and evidence

- **Starting revision:** `6ff408713127dc6634b2ac186656de4c0d7ac8e7` on `main`.
- **Analysis and implementation date:** 2026-09-29.
- **External sources consulted:** official Octo code/checkpoint documentation,
  WeightCLIP, the continual-VLA paper, and the supplied prior codebase audit.
- **Executed here:** static inspection and local unit tests for contracts,
  packing, splits, HDF5 validation, mapping, and metrics.
- **Not executed here:** JAX/Flax Octo loading, LIBERO training, simulator
  rollouts, model-zoo generation, or publication experiments; this Windows host
  lacks the pinned Linux research environment and real datasets.

## 1. Codebase overview

The research flow is:

```text
official Octo checkpoint + real LIBERO episodes
    -> one immutable base hash
    -> modality adapters across 40 tasks × 3 seeds × 3 checkpoints
    -> basis-invariant effective-update windows + task evidence
    -> masked autoencoder + multi-positive contrastive alignment
    -> linear prompt-to-latent mapper + empirical shells
    -> OOD task adaptation and one-state continual learning
    -> rollout records -> SR/NBT/forgetting/recovery/drift analyses
```

Important safeguards are enforced in code: missing data fails closed, episode
splits cannot overlap, adapter archives must share one base hash and schema,
synthetic runs cannot write to research results, and metric reports reject
incomplete or duplicate success-matrix cells.

The main frameworks are official Octo, JAX/Flax/Optax, NumPy/SciPy, HDF5, and
LIBERO/MuJoCo. PyTorch/PEFT remains a legacy smoke dependency only.

## 2. Repository map

| Path | Purpose | Research status |
|---|---|---|
| `wsl_vla/research/` | Contracts, packing, strict data, Octo patch, JAX alignment/refinement, mapping, metrics, provenance | Publication path |
| `configs/research/` | Pinned model, population, folds, ablations, rollouts, outputs | Publication path |
| `configs/reference_tasks.yaml` | Exact four ten-task orders from the continual-VLA paper | Publication path |
| `scripts/*research*`, `verify_official_octo.py` | Preflight, archive assembly, alignment training, and reporting | Publication path |
| `models/`, older scripts | Custom PyTorch policy and earlier experiment scaffold | Smoke tests only |
| `data/dataset.py` | Legacy loader; now fails closed unless synthetic data is explicitly allowed | Smoke/compatibility |
| `tests/` | Pure-Python correctness tests plus optional JAX gradient checks | Verification |
| `docs/` | Method description, compute estimates, and executable research contract | Documentation |

Generated datasets, checkpoints, logs, and `research_results/` are ignored.

## 3. Experiments

| Experiment | Question | Required output | Status |
|---|---|---|---|
| Zero-adapter gate | Does the patched model preserve official Octo behavior? | Exact output comparison and base hash | Implemented, not run here |
| Model zoo | Are comparable task updates produced from one base? | 360 validated population samples | Protocol implemented; compute pending |
| Alignment | Does task evidence organize adapter geometry? | reconstruction, retrieval, aligned-vs-unaligned metrics | Trainer implemented; data pending |
| OOD mapping | Can an unseen task prompt generate a useful adapter latent? | mapper/retrieval/mean-latent and matched refinement baselines | Differentiable generation/refinement path implemented; rollout integration pending real data |
| Continual learning | Does asymmetric regularization improve shared-policy retention? | complete rollout matrices for eight locked conditions | Shared-state runner and records implemented; simulator rollouts pending |
| Mechanistic validation | Does modality drift predict consecutive-checkpoint swap loss? | Pearson/Spearman estimates and bootstrap CIs | Analysis utilities implemented; rollouts pending |

## 4. Compute and runtime

The local configuration is designed for one GPU with at least 8 GB VRAM using
micro-batches of two, eight-step gradient accumulation, frozen backbone features,
and disabled JAX memory preallocation. No observed runtime or peak-memory value
is yet available. Documentation estimates from the original repository are
planning numbers for different model scales and must not be reported as measured
Octo-Small compute.

Final experiments require three seeds, four held-out-suite folds, and 50 rollout
initializations per task-stage. Actual GPU-hours, CPU-hours, peak VRAM, RAM, and
storage must be captured in each `RunManifest`.

## 5. Results and benchmarks

No WSL_VLA research result has been reproduced or generated in this checkout.
The repository now provides result schemas and exact calculations, not empirical
claims. Offline MSE remains diagnostic and cannot replace simulator success.

Primary metrics are final average success rate and reference-paper NBT. Secondary
metrics are normalized NBT, average forgetting, forward transfer, recovery-step
ratio, memory per task, and drift/swap correlations with bootstrap confidence
intervals. Result construction fails if a lower-triangular task-by-stage matrix
is incomplete or contains duplicate cells.

## 6. Suggested reading path

1. `README.md` - environment, evidence boundary, and runnable entry points.
2. `docs/RESEARCH_PIPELINE.md` - complete contracts and scaling gates.
3. `docs/METHODOLOGY_1_GUIDE.md` - original mathematical proposal.
4. WeightCLIP - alignment, mapping, empirical shell, and latent refinement.
5. The continual-VLA paper - exact LIBERO order, SR/NBT, component swaps, and
   recovery analysis.
6. Official Octo documentation and source - real model/data/action interfaces.
7. LIBERO - simulator suites and lifelong-learning evaluation protocol.

## 7. Practical next steps

Run the research preflight in WSL2, install the pinned environment, stage all 40
real HDF5 task files, and pass official Octo zero-adapter equivalence. Then
generate a one-task real-data adapter and a two-task shared-state rollout matrix.
Only after those gates pass should the 360-checkpoint population or multi-seed
suite experiments consume significant compute.

The remaining integration work that depends on external assets is the LIBERO
environment rollout worker and the hardware execution layer for all locked
conditions. The episode-to-Octo converter, adapter-only official
Octo loss, differentiable decoded-update path, shared-state sequential runner,
interfaces, and result contracts are implemented, but cannot be integration
tested on this host without the official environment, checkpoint, and datasets.

## Source links

- [Official Octo repository](https://github.com/octo-models/octo)
- [Pinned Octo-Small checkpoint](https://huggingface.co/rail-berkeley/octo-small-1.5)
- [Research pipeline](docs/RESEARCH_PIPELINE.md)
- [Locked experiment configuration](configs/research/base.yaml)
