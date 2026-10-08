# Publishable Methodology 1 Implementation Plan

## Summary

Rebuild the research path around the official pretrained Octo-Small model, using JAX/Flax end to end under WSL2/Linux. The existing PyTorch “Octo” surrogate remains only as a clearly labelled smoke-test fixture; it must not produce reported research results.

The implementation will address the confirmed blockers identified in :codex-file-citation{path="C:\Users\Lenovo\Downloads\Explanation.md" purpose="source"}, while following the alignment, mapping, empirical-shell, and refinement design in :codex-file-citation{path="C:\Debarghya\IIT Guwahati\Third Year\Sem 5\Ayon Sir\27651_WeightCLIP_Aligning_Data.pdf" purpose="source"} and the sequential LIBERO evaluation protocol in :codex-file-citation{path="C:\Debarghya\IIT Guwahati\Third Year\Sem 5\Ayon Sir\VLA Paper.pdf" purpose="source"}.

The immediate target is a mechanism-first Octo-Small study on an ≤8 GB consumer GPU. The code and experiment manifests will also define a gated large-VLA confirmation phase for later cluster access.

## Key Implementation Changes

### 1. Establish a valid pretrained policy and dataset foundation

- Integrate [`rail-berkeley/octo-small-1.5`](https://huggingface.co/rail-berkeley/octo-small-1.5) through the official Flax/JAX loading and fine-tuning interface demonstrated by the [Octo repository](https://github.com/octo-models/octo/blob/main/examples/02_finetune_new_observation_action.py).
- Pin the Octo repository revision, checkpoint revision, checkpoint hash, LIBERO version, preprocessing configuration, action normalization statistics, and task definitions in every run manifest.
- Remove silent synthetic fallback from every research command. Synthetic data remains available only through an explicit `--smoke-test` mode and its outputs cannot enter research result directories.
- Load real multi-view images, proprioception, language instructions, action chunks, padding masks, and episode boundaries. Split by complete demonstration episode, never by transition.
- Provide a separate reference task configuration matching the paper’s exact four ten-task sequences. Custom task lists remain available but must be labelled non-reference.
- Make the official Octo base immutable and assert that every adapter, evaluator, and rollout uses the same base-checkpoint hash.

### 2. Replace the invalid adapter representation

- Insert zero-initialized, modality-specific residual adapters into every frozen Octo transformer block:
  - vision adapters operate on observation/image-token positions;
  - language adapters operate on language/task-token positions;
  - action adapters operate on action-readout tokens and the diffusion action head.
- Preserve the official Octo output when adapters are zero. Train only adapters unless an experiment explicitly selects a fine-tuning baseline.
- Store both reloadable low-rank factors and the effective update `ΔW = scale × B @ A`. Weight-space learning uses effective updates so results are invariant to arbitrary LoRA factor rotations.
- Replace the fixed `[L,3,r,H]` A-matrix abstraction with a versioned packed-adapter schema containing component, transformer layer, parameter path, tensor shape, token-window offsets, dtype, base hash, seed, and checkpoint stage.
- Tokenize effective updates row-wise into fixed-width windows as in WeightCLIP, with masks for padding and unequal modality sequence lengths. Use shared autoencoder weights plus component, layer, and parameter-position embeddings.
- Generate the model zoo from one shared base using three training seeds and three late-stage checkpoints per task. The publication dataset is therefore `40 tasks × 3 seeds × 3 checkpoints = 360` adapter samples.

### 3. Correct evidence extraction, alignment, and mapping

- Build visual evidence from frozen Octo image-tokenizer features sampled across demonstrations, followed by a trainable DeepSets aggregator.
- Build language evidence from the official Octo language tokenizer with masked pooling and a trainable projection.
- Build action evidence from normalized trajectory statistics—mean, standard deviation, velocity, and jerk—with masks and a trainable projection.
- Train evidence encoders, prompt projectors, and the weight autoencoder jointly. Do not serialize random untrained visual embeddings as permanent evidence.
- Use masked reconstruction loss and modality-specific bidirectional contrastive losses. Treat adapters from the same task as multiple positives so repeated seeds/checkpoints are not false negatives.
- Fit the WeightCLIP linear ridge mapper from each modality prompt to its complete latent token sequence as the required first mapper. Add the token-wise memory-bank mapper only after the linear mapper passes validation.
- Estimate shell center and radius per modality and token position from training-zoo latents. Initialize mapped representations on this empirical shell; remove the current fixed origin-centred unit sphere.

### 4. Implement two scientifically distinct evaluation paths

- **OOD generation/adaptation:** map evidence from an unseen task to a latent adapter, decode it, and compare zero-shot performance, matched-step latent refinement, weight-space fine-tuning, nearest-neighbour retrieval, and mean-latent initialization.
- **Continual learning:** maintain one evolving shared adapter state through each ten-task suite. Stage `k` begins from stage `k−1`, trains on the new task, and evaluates the current policy on every task seen so far. Stored per-task adapters must not be substituted during evaluation.
- Implement latent refinement entirely in JAX using `jax.value_and_grad` through the decoder, adapter application, Octo transformer, and diffusion loss. Add a hard test that task loss alone produces non-zero gradients for visual, language, and action latents.
- Apply the empirical-shell projection at initialization and use component-wise local penalties against the pre-stage latent:
  `Σm γm ||Zm − Zprevious,m||²`.
- Compare:
  - sequential LoRA without regularization;
  - small experience replay with 10 and 100 transitions per previous task;
  - uniform latent regularization;
  - proposed `γvis, γlang > γact`;
  - direction-inverted regularization;
  - aligned versus reconstruction-only latent spaces;
  - independent task adapters as an upper-bound oracle.
- Select gamma values and all early-stopping settings only on validation folds. Test suites remain untouched until configurations are frozen.

### 5. Add publication-grade evaluation and provenance

- Use four leave-one-suite-out folds: alignment and mapper training use three LIBERO suites; the fourth suite is the unseen ten-task continual stream. Repeat for all four held-out suites.
- Run three independent training seeds. Use 20 rollouts per task-stage during development and 50 fixed evaluation initializations for final reporting.
- Save the complete task-by-training-stage success matrix and calculate:
  - average Success Rate;
  - the reference paper’s NBT;
  - normalized NBT with zero-baseline handling documented;
  - final average performance, forgetting, and forward transfer;
  - recovery-step ratio;
  - adapter/evidence memory per task;
  - wall time, peak VRAM, RAM, and storage.
- Reproduce component swapping only between consecutive sequential checkpoints: swap vision, language, action, combined vision-language, and full adapters, then measure the resulting success-rate drop.
- Relate latent drift to behavioural degradation using Pearson and Spearman correlations with bootstrap 95% confidence intervals. Report sample count, uncertainty, and scatter plots; do not report correlation from arbitrary cross-task swaps as forgetting.
- Persist JSON/Parquet results, configuration snapshots, environment lock, Git revision, checkpoint hashes, data provenance, seeds, runtime, and failures. Generate tables and figures exclusively from these stored records.

## Public Interfaces and Artifacts

- `AdapterSpec`: immutable component/layer/path/shape/tokenization description tied to a base-model hash.
- `PackedAdapter`: reloadable factors, effective-update token windows, masks, and metadata.
- `TaskEvidence`: raw feature references and projected visual/language/action evidence with preprocessing version.
- `AlignmentCheckpoint`: autoencoder, evidence encoders, projectors, mapper, empirical-shell statistics, split identifiers, and training manifest.
- `RunManifest`: code revision, environment, dataset/checkpoint hashes, hardware, seed, task order, configuration, and runtime.
- `EvaluationRecord`: task-stage rollout counts, successes, SR, losses, drift, component-swap condition, and checkpoint identity.
- Research commands will fail closed on missing data, incompatible checkpoint hashes, unknown schema versions, or synthetic inputs.

## Test and Acceptance Plan

- **Model integrity:** zero adapters reproduce official Octo outputs; the base hash is identical across every zoo checkpoint; save/load and packed-token round trips preserve predictions.
- **Gradient correctness:** task loss produces finite, non-zero gradients through decoded adapters into every latent modality; the same test must fail when the decoder is intentionally detached.
- **Data integrity:** episode splits have no shared trajectories; reference task orders match the paper; action normalization and camera mappings round-trip correctly.
- **Metric correctness:** golden task matrices verify SR, NBT, normalized NBT, forgetting, forward transfer, and recovery calculations, including missing and zero-success cases.
- **Alignment validity:** aligned representations must beat reconstruction-only representations on held-out retrieval and linear-mapper validation without using test-suite adapters.
- **Rollout validity:** a one-task Octo fine-tuning smoke run must learn above its frozen-base SR and produce deterministic evaluation results for fixed seeds.
- **Scaling gates:** progress through 2-task smoke, one-suite pilot, four-fold three-seed Octo study, and only then the later cluster-backed large-VLA confirmation.
- A result is publication-ready only if all primary comparisons include three seeds, rollout-based SR, confidence intervals, complete provenance, and both positive and negative findings. Offline MSE alone cannot support the paper’s continual-learning claims.

## Assumptions and Defaults

- WSL2/Linux with CUDA is available; JAX GPU preallocation will be limited for the ≤8 GB device, using micro-batches, gradient accumulation, cached frozen features, and sequential rollout workers.
- Octo-Small is the sole research backbone for the local phase. The current custom PyTorch proxy is retained only for fast unit tests.
- The primary paper claim is mechanism-level: modality-aligned adapter geometry and asymmetric latent regularization improve OOD adaptation and continual retention on a real pretrained small VLA.
- Large-VLA generalization is explicitly deferred to a later cluster phase and will not be implied by the Octo-Small results.
- Exact gamma values, mapper regularization, and stopping points are selected on validation folds rather than fixed from the current unvalidated defaults.
