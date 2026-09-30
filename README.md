# WSL_VLA: Weight-Space Alignment for Continual Robot Learning

This repository now has two deliberately separate paths:

1. **Official research path (`wsl_vla/research`)** - pinned Octo-Small 1.5,
   JAX/Flax, strict real LIBERO data, versioned adapter/evidence contracts,
   rollout-based continual-learning records, and fail-closed provenance checks.
2. **Legacy smoke path (`models`, older scripts)** - a small PyTorch policy analogue
   retained for shape and plumbing checks. It is not official Octo and must not be
   used for papers, benchmark tables, or claims about pretrained VLAs.

## Scientific scope

The local study tests three hypotheses:

- task evidence can organize modality-factorized adapter updates in an aligned
  weight latent space;
- mapped latents improve held-out task initialization and matched-step adaptation;
- stronger visual/language than action regularization improves retention in a
  genuinely shared sequential policy state.

Large-VLA generalization is a later confirmation phase and is not implied by the
Octo-Small results.

## Research environment

The official path targets Ubuntu/WSL2, Python 3.10 or 3.11, CUDA, and an NVIDIA
GPU with at least 8 GB memory. Octo is pinned to commit
`241fb3514b7c40957a86d869fecb7c7fc353f540`; the model configuration pins
`rail-berkeley/octo-small-1.5` and records the resolved checkpoint hash in every
run.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python3.10 -m venv .venv-research
source .venv-research/bin/activate
pip install -r requirements-research.txt
python scripts/research_preflight.py
python scripts/verify_official_octo.py
pytest -m "not integration"
```

The preflight fails when real data, pinned dependencies, task counts, episode
structure, CUDA safeguards, or checkpoint provenance are missing. Use
`--host-inspection-only` only to inspect an incompatible development host.
After the Octo equivalence gate, create each run's manifest with
`scripts/write_run_manifest.py`; it hashes all ten suite datasets and records
the code revision, dirty state, hardware, environment, base hash, seeds, and
task order before training begins.

## Publication protocol

- Exact ten-task orders are in `configs/reference_tasks.yaml`.
- `configs/research/base.yaml` locks three seeds, three late checkpoints per
  task, baselines, rollout counts, and output locations.
- Four leave-one-suite-out folds train alignment/mapping on the other three
  suites; task indices 8 and 9 within those suites are reserved for calibration.
- The expected model zoo contains `40 × 3 × 3 = 360` adapter checkpoints, all
  tied to one immutable base hash.
- Standard research outputs live under ignored `research_results/` and contain
  manifests, checkpoints, JSONL rollout records, and generated tables/figures.

See `docs/RESEARCH_PIPELINE.md` for the data contracts, commands, experiment
gates, and interpretation rules.

## Implemented research utilities

- Official Octo transformer patch with zero-initialized visual, language, and
  action-readout adapters after every transformer block, plus differentiable
  low-rank updates to the official diffusion action head.
- Basis-invariant packing of effective low-rank updates into masked WeightCLIP
  windows.
- Strict HDF5 episode validation and leakage-free splits.
- Shared masked weight autoencoder, trainable evidence encoders, multi-positive
  InfoNCE, empirical shells, differentiable latent refinement, and a linear
  ridge prompt-to-latent mapper.
- Exact SR, NBT, normalized NBT, forgetting, forward-transfer, recovery, and
  bootstrap correlation utilities.
- Immutable rollout record and run-manifest schemas.

## Alignment archive and training

Each population sample directory contains `adapter.npz`, `evidence.npz`, and
`metadata.json`. Assemble and train with:

```bash
python scripts/build_alignment_archive.py research_results/population \
  --output research_results/alignment/fold.npz \
  --held-out-suite libero_spatial

python scripts/train_research_alignment.py research_results/alignment/fold.npz \
  --output research_results/checkpoints/alignment/fold
```

No test-suite samples are accepted in an alignment archive. Multiple checkpoints
from the same task share a positive label in contrastive training.

## Reporting

Rollout evaluators append one `EvaluationRecord` per matrix cell. Metrics are
computed only from the immutable records:

```bash
python scripts/report_research_metrics.py research_results/records/run.jsonl \
  --suite libero_spatial --seed 42 --condition proposed_asymmetric
```

Offline action MSE is diagnostic only. Publication claims require rollout-based
success matrices, three seeds, confidence intervals, complete provenance, and
the locked ablation set.
