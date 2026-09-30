# Official Methodology 1 Research Pipeline

## 1. Evidence boundary

The functional subpackages under `wsl_vla/` are the primary publication path:
`octo`, `adapters`, `alignment`, `data`, `evaluation`, and `experiments`. In particular,
`scripts/run_research_benchmark.py` is a deprecated legacy fixture despite its
historical filename. Every legacy PyTorch command now requires `--smoke-test`,
writes under `smoke_results/`, and marks artifacts as publication-ineligible.
The custom analogue is isolated under `wsl_vla/smoke/legacy_torch`; it is not
Octo-Small.

The official path does not silently generate samples, download an unpinned base,
reuse a task-specific latent during shared-policy evaluation, or report an
offline regression score as robot success.

Each suite data directory must contain `manifest.json` (schema 1) with ten
ordered `{task_index, instruction, file}` entries. The preflight compares every
instruction to `configs/reference_tasks.yaml`; filesystem enumeration is never
used to infer task identity or order.

## 2. Model and adapter design

`wsl_vla.octo.bridge.install_octo_modality_patch` replaces Octo's internal
`BlockTransformer` before constructing the research model. It retains all
official parameter paths and adds three rank-r residual updates after each
transformer block:

- task/language tokens receive the language update;
- observation/image tokens receive the vision update;
- action readout tokens receive the action update.

The action component also includes external low-rank updates for every Dense
kernel under the official diffusion action head. `flax_adapters.py` discovers
those paths from the loaded parameter tree and applies `down @ up` functionally,
so gradients flow from diffusion loss to both the decoded head adapters and the
action latent. Its up factors are zero-initialized for exact base equivalence.

The up-projection is initialized to zero. `verify_official_octo.py` loads the
unmodified and patched models, merges official weights by path and shape, and
requires their transformer and deterministic diffusion-head output trees to
match within absolute tolerance `1e-6` before any experiment can proceed.

`octo_training.py` exposes the official diffusion loss as a function of adapter
state only. Frozen parameters are closed over rather than merely assigned a
zero learning rate, preventing accidental base-model updates.

Raw LoRA factors are not comparable across runs because their rank basis may
rotate. `packing.py` therefore computes the effective dense update and slices it
row-wise into fixed-width windows. Every token carries component, layer,
parameter-entry, and row metadata. Padding is masked in reconstruction and
alignment losses.

For generated adapters, `latent_adapter.py` reverses those windows in JAX and
applies the decoded effective updates through separate zero-initialized dense
injection paths. These paths are frozen during low-rank zoo training and avoid
an unstable SVD or an arbitrary factor basis. Consequently the official
diffusion task loss remains connected to decoded visual, language, and action
latents during refinement. `assert_task_loss_gradients` is a mandatory gate and
deliberately fails for a detached decoder.

## 3. Population sample contract

Each sample directory must contain:

### `adapter.npz`

Created by `save_packed_adapter` and containing only non-pickled numeric arrays:

- `tokens`, `mask`;
- `component_ids`, `layer_ids`, `entry_ids`, `row_ids`, `column_offsets`;
- `factor_down_NNNN`, `factor_up_NNNN` for exact checkpoint reload;
- UTF-8 JSON bytes for the complete `AdapterSpec` and per-sample task, suite,
  seed, checkpoint stage/fraction, and source dtype metadata.

### `evidence.npz`

- `vision_features [samples, feature_dim]`: frozen Octo image-tokenizer features;
- `vision_mask [samples]`;
- `language_features [feature_dim]`: masked official Octo language features;
- `action_statistics [feature_dim]`: normalized mean, standard deviation,
  velocity, and jerk statistics.

The evidence artifact also stores the exact training episode IDs,
preprocessing version, and raw dataset/base references.

### `metadata.json`

Required fields are `schema_version`, `task_id`, `suite`, `seed`,
`checkpoint_fraction`, `base_sha256`, `split`, and `synthetic`. Research assembly
requires schema 1, `synthetic=false`, one shared base hash, one adapter schema,
and a split of either `train` or `validation`.

`extract_research_evidence.py` creates evidence from the frozen official Octo
image and language tokenizers and complete real trajectories.
`train_research_zoo.py` verifies zero-adapter equivalence, updates adapter state
through the official diffusion loss with micro-batch accumulation, and saves
the three locked late checkpoints. Legacy `.pt` zoo commands cannot write
publication artifacts.

## 4. Alignment and mapping

`build_alignment_archive.py` verifies all population samples and produces a
single no-pickle NPZ. It requires an explicit held-out suite, rejects any sample
from that suite, derives the calibration split from locked task indices, and
requires all 270 samples from the other three suites. `train_research_alignment.py`
jointly optimizes:

- masked effective-update reconstruction;
- a shared token autoencoder with component/layer/position embeddings;
- trainable DeepSets visual evidence;
- language and action projectors;
- modality-specific learned temperatures;
- symmetric multi-positive InfoNCE keyed by task identity;
- auxiliary task classification from every evidence encoder.

Every contrastive batch contains at least two task identities and two adapter
samples per identity. Optimizer accumulation is not treated as extra negatives.

The best validation checkpoint is used to fit one linear ridge mapper per
modality. Ridge strength is chosen only on validation tasks. Empirical token
centres and radii are estimated from training latents and saved beside the
mapper. The unseen suite is never accepted into these archives.
The checkpoint includes `token_layout.npz`; `alignment_checkpoint.py` reloads
the complete system, maps evidence, projects each modality to its empirical
per-token shell, decodes the combined sequence, and refines it through official
Octo diffusion loss.
The reconstruction-only control is trained with the identical command and
`--contrastive-weight 0`; held-out retrieval and mapper quality must pass
`alignment_advantage_gate` before the aligned checkpoint advances. Run the
executable gate against the exact archive used by both checkpoints:

```bash
python scripts/run_alignment_advantage_gate.py artifacts/fold/archive.npz \
  --aligned-checkpoint artifacts/fold/aligned \
  --reconstruction-checkpoint artifacts/fold/reconstruction_only \
  --output artifacts/fold/alignment_advantage.json
```

The gate reads only the locked validation tasks, reports retrieval and complete
latent-sequence mapper MSE for every modality, and applies a predeclared macro
decision: aligned retrieval must be strictly higher and aligned mapper MSE must
be strictly lower. A failed comparison is still written to JSON and exits
non-zero so the negative result is retained while later scaling is stopped.

## 5. Sequential evaluation contract

The policy state is one evolving latent adapter. After training stage `k`, the
same current policy is evaluated on all tasks `0..k`; evaluators must not restore
stored task-specific latents. Each rollout cell is written as an
`EvaluationRecord`, and the lower triangular matrix is reconstructed from those
records with duplicate and missing-cell checks.

Required conditions are sequential LoRA, replay 10, replay 100, uniform
regularization, asymmetric regularization, direction-inverted regularization,
reconstruction-only latent space, and independent-adapter oracle. Gamma and
stopping choices are frozen from validation tasks before the held-out suite is
evaluated.

For component swaps, compare checkpoints from consecutive stages only. Report
vision, language, action, combined vision-language, and full swaps. Drift versus
success-rate drop uses both Pearson and Spearman coefficients with bootstrap
95% confidence intervals and the actual sample count.

## 6. Scaling gates

1. Unit tests and static checks.
2. Official Octo zero-adapter equivalence.
3. One-task real-data learning above the frozen-base success rate.
4. Two-task shared-state smoke run with a complete `2×2` success matrix.
5. One ten-task suite, three seeds, 20 rollouts per cell.
6. Four leave-one-suite-out folds, three seeds, 50 rollouts per cell.
7. Large-VLA confirmation after cluster resources become available.

A failed gate stops later compute. Negative scientific outcomes are retained in
the records and reported; they are not filtered out of the final analysis.
