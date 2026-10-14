# Research Pipeline Contracts

This document describes the data and model contracts of the corrected
Octo-Small zoo and alignment pipeline. For the execution order see
`METHODOLOGY1_RUNBOOK.md`; for what has been run where, `IMPLEMENTATION_STATUS.md`.

## 1. Evidence boundary

The functional subpackages under `wsl_vla/` are the research path: `octo`,
`adapters`, `alignment`, `data`, `evaluation`, and `experiments`. The legacy
PyTorch commands require `--smoke-test`, write under `smoke_results/`, and mark
their artifacts as ineligible; the custom analogue under
`wsl_vla/smoke/legacy_torch` is not Octo-Small.

The research path does not generate samples, download an unpinned base, reuse a
task-specific latent during shared-policy evaluation, or report an offline
regression score as robot success.

**The experiment manifest is the sole source of counts.** Task selections,
seeds, optimizer-update budget, snapshot stages, episode split, folds,
alignment seeds, rollout budgets, and the expected sample counts live in
`configs/research/manifests/<profile>.yaml`. `configs/research/base.yaml` holds
only model, optimizer, preprocessing, and simulator settings. No trainer or
validator contains a population size. A manifest's declared `expected` counts
must equal the counts derived from its selections, or it is rejected.

Each suite data directory contains `manifest.json` with ordered
`{task_index, instruction, file}` entries. Instructions are compared with
`configs/reference_tasks.yaml`; filesystem order never defines task identity.

Artifacts are versioned individually (`wsl_vla/contracts.py`). Adapter specs,
packed adapters, task evidence, zoo samples, alignment archives, and alignment
checkpoints are at schema 2. Version-one artifacts raise `LegacyArtifactError`
in every corrected command and are readable only through
`scripts/reporting/inspect_legacy_artifacts.py`.

## 2. Model and adapter design

### Token-conditioned factorization

The proposal factorizes a VLA update into vision, language, and action
components and assumes separate weights per component. Octo-Small has one
shared transformer over a mixed token sequence, so the factorization here is
*token-conditioned*: each transformer block owns three zero-initialized rank-r
residuals, and a token only ever receives the residual of its own modality.

| Token group (octo-small-1.5) | Tokens | Route |
| --- | --- | --- |
| `obs_primary`, `obs_wrist` (image tokenizers) | 256 + 64 per timestep | vision |
| `task_language`, `obs_task_language` (T5 tokens and their per-timestep repeat) | 16 each | language |
| `readout_action` | 1 per timestep | action |
| every 2-D kernel of the diffusion action head | - | action |

The routing table is derived from the checkpoint configuration
(`build_token_routing`). A `LowdimObsTokenizer` group (proprioception) is
routed to `frozen` and receives no residual; a tokenizer of unknown type is an
error, never labelled "vision". octo-small-1.5 has no proprioception tokenizer,
and the pipeline records `proprio_consumed: false`; a checkpoint that does
consume proprioception is rejected because no shared preprocessing exists for
it. `verify_official_octo.py` reads the routing back from the live graph.

### Placement

Residuals are added to the **input** of each block, and all three are computed
from that same block input, so the branches do not depend on evaluation order.
The previous post-block placement left the final-layer vision and language
residuals with no path to the action readout: after the last block only a
token-wise LayerNorm remains, so those adapters received exactly zero gradient
and stayed zero in every version-one zoo sample.

`up` kernels are initialized to zero. The patched model is therefore
functionally identical to the official one before adaptation.
`assert_zero_adapter_equivalence` checks transformer outputs **and** sampled
actions drawn with the identical random key, against references captured
before the patch was installed, in true float32, with tolerance `1e-5`.
`assert_base_parameters_unchanged` re-hashes the pretrained tree after training
and after every adapter reload.

Octo instantiates its block transformer from a module global. The adapter
identity (rank, scale, routing) is written into the research module's
configuration, each bundle re-installs its own patched class before tracing,
and a module traced under a different patch fails loudly.

### Identities recorded independently

`ResearchOctoBundle.identity()` records the checkpoint revision and parameter
hash, the architecture hash (configuration plus parameter and example-batch
structure, no values), the preprocessing hash, the controller-action
convention hash, the token routing, the consumed observations, and the hashes
of the frozen feature extractors.

### Controller actions

One convention, `libero_osc_pose_delta_v1`, is fixed for every zoo task:
identity policy normalization in OSC_POSE controller coordinates, bounds
`[-1, 1]`, gripper `-1` open and `+1` close, the same dimension order in
demonstrations and rollouts. Training targets are validated against the
bounds and never clipped; a demonstration outside the bounds is an error.
The diffusion head's own clamp (`max_action = 5`) is checked to preserve every
valid target. At deployment each sampled command is clipped to the controller
bounds and the clipping frequency is recorded per episode.

Because normalization is identical for all tasks, adapters from different tasks
live in one output space and a generated adapter needs no task statistics.

### Shared preprocessing

`wsl_vla/data/preprocessing.py` builds every model input - camera frames, the
history window, the language task, the action chunk, all padding masks - for
training batches, evidence extraction, and the rollout policy alike. Its
identity (`PreprocessingSpec.sha256()`) covers the cameras, the 128 x 128
source resolution rendered by both demonstrations and rollouts, the resize
method, the orientation handling, the number of executed actions per plan, and
the tokenizer's output on fixed probe instructions.

LIBERO demonstrations record the robosuite image convention they were
generated with (`opengl`: rows stored bottom-up). `image_transform:
flip_vertical` presents upright frames to the checkpoint.
`simulator_alignment: identity` states that the simulator renders in the
demonstrations' storage orientation; the rollout evaluator asserts the
simulator's declared convention, and `verify_camera_orientation.py` measures
it by re-rendering recorded demonstration states.

### Packing

Raw LoRA factors are not comparable across runs because their rank basis can
rotate. `packing.py` computes each effective dense update
`(alpha / rank) * down @ up` and slices it row-wise into tokens of width 384.
The layout carries, per token, the component, layer (head kernels are numbered
after the blocks), parameter entry, row, and column-window identity, and groups
tokens into fixed sequence windows of 128: 20,012 tokens in 157 windows
(36 vision, 36 language, 85 action) for octo-small-1.5. Windows are padded and
masked, and the complete sequence is restored deterministically.

## 3. Zoo sample contract

One independent run trains one task with one seed from the shared base and
writes, under `population/<suite>/<task>/seed_<s>/`:

- `run.json`: completion flag, failures, counters, history, resource use,
  provenance, gate results;
- `update_<n>/adapter.npz`: tokens, layout arrays, the factors for exact
  reload, the `AdapterSpec`, and per-sample metadata, all without pickle;
- `update_<n>/metadata.json`: identities and hashes, optimizer updates, micro
  steps, sample exposure, effective update norms per component, held-out loss,
  the open-loop sampling check, and the validation status;
- `update_<n>/behaviour.json`: written later by `evaluate_zoo_behaviour.py`.

Counters are kept apart: *optimizer updates* (parameter updates), *micro steps*
(gradient evaluations), and *sample exposure* (training windows consumed).
`sample_exposure = micro_steps x micro_batch_size`.

The resume state (optimizer moments, noise counter, dataset cursor) is stored
in `zoo_resume/`, never beside the snapshots, and is accepted only for the
identical run identity.

A sample is *structurally valid* when it reloads and its tokens equal the
effective update of its factors. It is *behaviourally evaluated* only when
paired frozen-base versus adapter rollouts exist. Runs that fail or perform
poorly keep their directory, history, and failure record.

### Evidence

`evidence/<suite>/<task>.npz` is extracted once per task from the training
episodes of the manifest's episode split, with frozen Octo tokenizers:

- `vision_features [frames, 1024]` with the episode index and timestep of each
  frame (per camera, the masked mean of the current timestep's image tokens;
  primary and wrist concatenated);
- `language_features [16, 768]` with the tokenizer attention mask;
- `action_chunk_features [episodes, chunks, 56]` with a mask: per episode,
  non-overlapping action chunks, each summarized by the mean and standard
  deviation of the commands and of their first, second, and third differences.
  These are **command differences**, not physical velocity or jerk. Chunks
  never cross an episode boundary and padded steps are never used.

The file records episode ids, feature-extractor hashes, preprocessing and
action-convention identities, and the base, architecture, and dataset hashes.
Visual aggregation and the evidence projectors are trained during alignment,
not here.

## 4. Alignment and mapping

### Archive

`build_alignment_archive.py --fold <fold>` re-checks every adapter and evidence
hash (not only in an optional validator), requires every manifest sample of
the fold, rejects any task of the held-out suite, and keeps all seeds and
snapshots of a task in one split. It fits evidence standardization and the
per-entry RMS scales on **training tasks only**; constant features keep unit
scale. Tokens are stored as memory-mapped shards indexed by
`archive_manifest.json`; evidence is stored once per task. The archive is built
in a staging directory and renamed into place.

### Model

Separate modality streams are processed by one shared windowed transformer
encoder and one transformer decoder. Tokens carry component, layer, parameter
entry, row, and column-position embeddings. Reconstruction targets are divided
by the training-only per-entry RMS scale (with a floor); the masked squared
error is averaged within each component and then equally across components.
The scales are stored in the checkpoint and inverted before any update reaches
a policy.

The contrastive loss is token-wise, bidirectional, and multi-positive: every
weight token and every evidence token of the same task is a positive, padding
is excluded in both directions, and each modality has a learned temperature.
`tests/unit/test_alignment/test_losses.py` checks it against an explicit small
reference calculation. Evidence encoders are trained jointly with auxiliary
classifiers over **training-task** identities only.

### The paired comparison

Two conditions are trained: `aligned` and `reconstruction_only`. They share
architecture, initialization, batch schedule, optimizer exposure, evidence
classification, and checkpoint selection by validation reconstruction. The
control differs only in that its contrastive weight is zero.
`assert_paired_checkpoints` verifies this from recorded fingerprints before any
comparison is computed; a violation is an integrity failure.

Batches hold six distinct tasks with two adapter samples each where the fold
allows it (two tasks in the integrity run). Gradient accumulation changes
optimizer exposure only; the contrastive negative set is always the tasks of
one micro-batch.

The checkpoint stores the fully resolved configuration. Command-line overrides
take precedence over configuration values, and there are no architecture
defaults in code.

### Mapping, shells, deployment

Ridge mappers go from task evidence to complete latent sequences. Seed and
snapshot targets are averaged within each training task first, so every task
counts once; ridge strength is selected on validation tasks only. Shell
statistics come from training latents only, and mapped latents are evaluated
both before and after shell projection.

Deployment is rank-matched. Each decoded update is projected to its entry's
maximum rank with a truncated SVD and exported as factors that load through
the same low-rank path as a zoo adapter. The projection error is reported.
Unrestricted dense updates remain available as a diagnostic and for latent
refinement, but they are not part of the primary comparison.

### Evaluation

There is no gate that alignment must win. `evaluate_alignment_pair.py` exits
non-zero only on an integrity failure. Every comparison is labelled
`supports_first_condition`, `inconclusive`, or `negative_finding` from the
confidence interval of the task-paired difference.

Reported offline: task-balanced reconstruction error, mapped decoded-update
error, rank-projection error, latent prediction error relative to a
training-mean predictor, and evidence-to-mapped-latent retrieval against
task-balanced galleries with the chance level. A normalized metric with a zero
denominator is undefined. Direct weight-to-evidence cosine retrieval is
reported as a descriptive diagnostic only.

Reported from rollouts, on identical initial states and action-sampling seeds:
the frozen base, the independently trained zoo adapter, the autoencoder
reconstruction and the mapper-generated adapter of both conditions, the
nearest-neighbour adapter, and the training-mean adapter. Episode-level
outcomes, task-balanced success rates, paired differences, and intervals from a
two-level bootstrap over tasks and training runs are stored. Validation and
held-out results are separate; a held-out result freezes the manifest.

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

## 6. Downstream study (not exercised by the pipeline repair)

The sections above are covered by tests and by the integrity run. The
continual-learning, OOD, and probe stages below were re-pointed at the
corrected artifacts and their commands are checked against each script's
argument parser, but they have not been executed on corrected artifacts.

1. Unit tests and the run-profile gates of the runbook.
2. One-task real-data learning with deterministic repeated rollouts.
3. Two-task shared-state run with a complete `2×2` success matrix.
4. One ten-task suite, three seeds, 20 rollouts per cell.
5. Four leave-one-suite-out folds, three seeds, 50 rollouts per cell.
6. Large-VLA confirmation after cluster resources become available.

A failed *integrity* gate stops later compute. Scientific outcomes, including
negative ones, are retained in the records and reported; they never stop the
pipeline and are never filtered out of the final analysis.

After all continual runs (four suites × seeds × eight conditions) finish, the
complete study is validated and summarized. Paths are those planned by
`plan_methodology1_jobs.py` inside the experiment directory:

```bash
STUDY=research_results/experiments/full/methodology1/publication
python scripts/reporting/report_publication_study.py $STUDY/continual/ten_task_study \
  --output-json $STUDY/reports/continual_study.json \
  --output-parquet $STUDY/reports/continual_records.parquet
```

This command rejects missing cells, mixed run identities, inconsistent rollout
budgets, or absent conditions. Confidence intervals resample suite folds and
training seeds as crossed factors. Undefined normalized NBT caused by all-zero
initial task success is stored as null with an explicit excluded-run count.

Recovery efficiency follows the reference paper rather than the OOD protocol.
For each consecutive checkpoint pair, the stage `k+1` shared adapter is
fine-tuned again on task `k` until it regains the success rate achieved when
task `k` was first learned, or until the original training budget is exhausted:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/eval/run_recovery_probe.py \
  $STUDY/continual/ten_task_study/libero_spatial/seed_17/sequential_no_regularization
```

The fixed curve grid is dense below 10% of the original budget and uses the
same initialization indices and rollout seeds as the source run. Step zero must
exactly reproduce the stored post-forgetting evaluation. Unrecovered tasks are
reported with a null ratio instead of being silently assigned 1.0, and tasks
whose original peak was zero are marked ineligible. The probe reuses the source
run's learning rate, batch size, and optimizer accumulation and currently
accepts LoRA-space continual runs only; latent-space recovery must not be
misreported as the same optimization protocol.

Gate 2 is executable as:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/eval/run_one_task_learning_gate.py \
  --suite libero_spatial --task-index 0 --seed 17
```

It evaluates the frozen base twice and the trained adapter twice with identical
official LIBERO initialization indices and rollout seeds. It persists all
episode outcomes and the reloadable trained adapter. The gate passes when both
repetitions are deterministic; whether the adapter beat the frozen base is
recorded as a finding (`improved`) and does not decide the exit status.

Gate 3 reuses the production continual runner but is explicitly separated from
the ten-task study output:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/eval/run_official_continual.py \
  --suite libero_spatial --condition sequential_no_regularization \
  --seed 17 --task-count 2
```

The runner trains one evolving adapter and writes all three valid cells of the
two-stage lower-triangular success matrix. The future-task cell at stage zero is
explicitly stored as null; it is not fabricated from a separately trained
adapter. Gamma-selection input is required only for latent-space conditions.
