# Methodology 1 Rigor Audit

Audit date: 2026-09-30  
Repository: `Deb044/WSL_VLA`  
Audited commit: `de50968e68706d2783b5f441c9046750c049df0b` (`main`, synchronized with `origin/main`)  
Scope: implementation and evidentiary readiness for Methodology 1 in the supplied Weight Space Alignment proposal, interpreted through the supplied WeightCLIP and continual-VLA papers and the user's Octo-Small implementation plan.

## Executive verdict

The repository is **not yet a working implementation of Methodology 1 and is not ready to support research claims**. It is a useful static scaffold: it introduces a pinned official-Octo integration attempt, versioned adapter packing, strict dataset and split utilities, fold/task protocol definitions, several metric utilities, and fail-closed archive validation. But the official research path is not executable end to end, the most important Octo integrity check is at high risk of being invalid, evidence generation and model-zoo training are absent, latent refinement is not connected to official Octo, rollout evaluation is absent, and there are no experimental artifacts.

The current evidence supports the narrower statement: **“A partially tested research infrastructure prototype for Methodology 1 exists.”** It does not support claims about alignment quality, OOD adaptation, continual retention, drift as a forgetting proxy, memory efficiency, or publishable performance.

## Severity-ranked findings

### P0 — No official end-to-end Methodology 1 experiment exists

Confirmed.

- The official package exposes disconnected primitives: `decoded_token_task_loss` (`wsl_vla/research/latent_adapter.py:79`), `refine_latents` (`wsl_vla/research/jax_modules.py:389`), `run_sequential_protocol` (`wsl_vla/research/sequential.py:26`), and `OODEvaluationRecord` (`wsl_vla/research/contracts.py:204`). There is no production command that assembles them into an Octo/LIBERO training and rollout run.
- The sequential runner is callback orchestration only; it does not implement Octo training, replay, any gamma condition, evidence-to-latent mapping, shell initialization, component swaps, or simulator rollouts (`wsl_vla/research/sequential.py:22-75`).
- No `research_results`, `checkpoints`, or `results` directory exists in the audited checkout. Consequently there are no checkpoints, success matrices, rollouts, confidence intervals, or figures to audit.

Impact: every empirical Methodology 1 claim remains unsupported.

### P0 — The zero-adapter Octo equivalence gate is likely invalid before it is run

Strong static inference; must be resolved by an integration test in the pinned WSL/JAX environment.

- `install_octo_modality_patch` globally replaces `octo.model.octo_module.BlockTransformer` (`wsl_vla/research/octo_bridge.py:151`).
- `load_research_octo` performs that mutation before the equivalence comparison (`wsl_vla/research/octo_bridge.py:197-205`).
- Only afterward does `assert_zero_adapter_equivalence` call `bundle.pretrained_model.run_transformer` (`wsl_vla/research/octo_bridge.py:228-244`).
- In the pinned upstream Octo source, `OctoTransformer.__call__` resolves the module-global `BlockTransformer` when it applies the transformer (`octo/model/octo_module.py:277-284`). Thus the supposedly “original” model is liable to execute the patched class while carrying an unpatched parameter tree. At best the test compares two patched graphs; at worst it fails on missing adapter parameters. It does not robustly preserve a pre-patch reference output.
- No automated test exercises `octo_bridge.py`; the only check is the external `scripts/verify_official_octo.py` command.

Required correction: compute and retain reference outputs before patching, or construct an isolated unmodified module namespace/class; then compare transformer outputs and action-head loss/predictions after merge. Make this an integration test that must pass before any zoo training.

### P0 — Invalid legacy programs remain discoverable as research programs

Confirmed.

- `docs/RESEARCH_PIPELINE.md:5-7` says scripts whose names contain `research` belong to the publication path.
- `scripts/run_research_benchmark.py` imports and runs the PyTorch surrogate, reports offline action MSE, initializes missing latents randomly (`:151-163`), and performs arbitrary cross-task swaps (`:224-245`). This directly conflicts with the required official-Octo, rollout-SR, mapped-initialization, and consecutive-checkpoint protocols.
- `scripts/orchestrate_zoo.py:9-15` and `:76-80` claim to train “Octo-Small,” but dispatch `scripts/train_zoo.py`; that script builds `models/octo_small.py`, explicitly a custom PyTorch implementation (`models/octo_small.py:1-10`), and optimizes MSE (`scripts/train_zoo.py:109-127`).
- The dataset class now defaults to `allow_synthetic=False`, so these scripts no longer silently fall back to synthetic data. Nevertheless, their naming and printed claims can still cause surrogate results to be mistaken for research results.

Impact: the repository can generate scientifically invalid artifacts under authoritative-sounding command names.

Required correction: move all proxy programs under a clearly isolated `smoke/` namespace, add an irreversible `artifact_kind=synthetic_or_proxy` marker, prohibit writes below `research_results`, and make every official command import only `wsl_vla.research`.

### P1 — The model-zoo and evidence-production stages are missing

Confirmed.

- `scripts/build_alignment_archive.py` consumes `adapter.npz`, `evidence.npz`, and metadata, but no official-Octo command produces those sample directories.
- `wsl_vla/research/evidence.py` implements action statistics and feature padding only (`:8-80`). It does not extract frozen Octo visual-tokenizer features or official masked language features.
- `TaskEvidence` is defined (`wsl_vla/research/contracts.py:98-142`) but is not used by the archive builder. The builder reads raw arrays directly and therefore does not enforce evidence episode IDs, preprocessing version, raw feature references, or train-only evidence provenance (`scripts/build_alignment_archive.py:89-98`).
- The legacy orchestrator produces one proxy checkpoint per task, not the required official population of 40 tasks × 3 seeds × 3 late checkpoints.

Impact: alignment training cannot be supplied with valid inputs, and evidence leakage/preprocessing consistency is not auditable.

### P1 — Alignment training materially diverges from the proposal

Confirmed.

- The proposal and WeightCLIP require auxiliary task/dataset-classification supervision on prompt encoders. `AlignmentSystem` contains no classification heads (`wsl_vla/research/jax_modules.py:190-263`) and the trainer contains no classification loss (`scripts/train_research_alignment.py:156-178`).
- `configs/research/base.yaml:40-45` defines reconstruction and modality-specific weights, but the trainer never reads this config. It applies one CLI scalar uniformly to the sum of all three losses (`scripts/train_research_alignment.py:156-178`).
- The stated architecture is a shared transformer encoder/decoder. The encoder is transformer-like, but the decoder is only a per-token Dense–GELU–LayerNorm–Dense MLP and discards component/layer identifiers (`wsl_vla/research/jax_modules.py:105-118`). This is a material architecture change that needs justification or correction.
- Default contrastive batches contain two samples (`scripts/train_research_alignment.py:81`). Gradient accumulation occurs after each independently computed loss (`:193-210`) and therefore does not enlarge the negative set. Random sample shuffling (`:220-227`) rarely guarantees repeated-task positives. With only one negative and usually only the paired positive, the implemented multi-positive objective is poorly identified despite the proposal explicitly requiring population diversity.
- The alignment advantage gate exists (`wsl_vla/research/alignment_eval.py:45-74`) but is not called by any non-test code. A reconstruction-only comparison is therefore not enforced before downstream use.

Impact: even if an archive were supplied, the trained representation would not yet implement or validate the proposed alignment objective.

### P1 — Mapping and differential refinement are only partial primitives

Confirmed.

- Linear ridge mapping and per-token empirical-shell estimation are implemented in the alignment trainer (`scripts/train_research_alignment.py:295-322`). This is a valid first mapper.
- The memory-bank mapper and nearest-neighbor retrieval initialization are absent.
- `project_to_empirical_shell` and `refine_latents` exist separately (`wsl_vla/research/jax_modules.py:344-350`, `:389-422`), but refinement never invokes shell projection and no production path maps evidence, projects it, decodes it, and applies official Octo loss.
- `decoded_token_task_loss` returns the Octo action-head loss object directly (`wsl_vla/research/latent_adapter.py:79-91`), while refinement expects a scalar task loss. No integrated adapter resolves the action head's auxiliary metrics or validates gradients through the actual decoder/Octo graph.
- Fixed, drift-informed, and online/adaptive gamma schedules are not implemented in the official path. Gamma grids and eight conditions exist only as YAML declarations (`configs/research/base.yaml:49-70`).
- Replay buffers, independent-adapter oracle evaluation, matched-step weight-space fine-tuning, nearest-neighbor retrieval, and mean-latent initialization are absent.

Impact: the central intervention—component-asymmetric latent regularization during real sequential adaptation—has not been implemented.

### P1 — Evaluation and reporting cannot substantiate the planned claims

Confirmed.

- There is no LIBERO simulator rollout implementation. The generic sequential runner accepts a rollout callback but no official caller supplies one.
- The report command computes one condition/seed matrix and basic metrics only (`scripts/report_research_metrics.py:21-54`). It does not aggregate three seeds, calculate confidence intervals for primary comparisons, produce tables/figures, measure memory, or analyze drift-to-behavior correlations.
- Correlation and component-swap utilities are isolated and unused by production code. The safe swap helper correctly enforces consecutive stages (`wsl_vla/research/component_analysis.py:12-41`), but the misleading legacy research benchmark performs arbitrary cross-task swaps.
- `EvaluationRecord` lacks a swap condition, paired checkpoint identity, resource measurements, evaluation initialization identity, and failure status (`wsl_vla/research/contracts.py:178-200`).
- `forward_transfer` is defined as diagonal performance minus an independent-task baseline (`wsl_vla/research/metrics.py:76-81`). That may be a defensible “benefit at acquisition” measure, but it is not documented and is not conventional pre-training forward transfer. The record layer rejects upper-triangular evaluations (`wsl_vla/research/records.py:58-64`), so conventional forward transfer cannot be recovered.
- NBT explicitly assigns the final task a zero contribution and divides by all K tasks (`wsl_vla/research/metrics.py:31-44`). This resolves an ambiguity in the paper's printed formula, but the convention must be stated in the paper and checked against the authors' evaluation code before comparison.

Impact: even successful training would not currently yield the required publication evidence.

### P1 — The acceptance tests do not exercise the scientific path

Confirmed.

- Local result: **21 passed, 3 skipped** using an explicit writable pytest base directory.
- The skipped JAX test uses a synthetic arithmetic loss over three arrays (`tests/test_jax_research.py:13-49`); it does not traverse the trained decoder, Octo transformer, or diffusion head. Therefore it does not satisfy the required gradient-integrity gate.
- No test loads the pinned Octo checkpoint, proves zero-adapter equivalence, performs a packed-token prediction round trip, demonstrates one-task learning over frozen-base SR, or runs a two-task rollout matrix.
- Host preflight correctly reports the current environment as invalid: Windows, Python 3.13.5, JAX/Flax/Optax/Octo absent, no GPU visible, and no LIBERO manifests. This is good fail-closed behavior, but it means the official path is untested here.

Impact: green unit tests currently validate utilities, not the research mechanism.

### P2 — Reproducibility and provenance are incomplete

Confirmed.

- Model and Octo Git revisions are pinned, and adapter/data hashes are designed into several schemas.
- `requirements-research.txt` pins several direct packages but is not a complete transitive lock with artifact hashes. `environment.yml` describes the legacy PyTorch environment instead of the official research environment.
- The run-manifest writer creates only an initial manifest. No official runner updates `finished_at`, failures, wall time, peak VRAM, peak RAM, or storage.
- Archive protocol values for seeds and checkpoint fractions are duplicated as literals (`scripts/build_alignment_archive.py:106-120`) instead of being derived from and hashing the locked config.

Impact: a future run would still be difficult to reproduce exactly and to audit after failure.

## Claim-to-evidence matrix

| Methodology claim | Status | Repository evidence | What is still required |
|---|---|---|---|
| Official immutable Octo-Small base | Partial | Pinned IDs/revisions and base tree hash exist | Valid zero-equivalence test and real checkpoint execution |
| Modality-specific adapters in all blocks and diffusion head | Partial, unverified | Static Flax patch and external diffusion factors exist | Inspect real parameter paths, run forward/loss/gradient tests |
| Rotation-invariant effective-update tokenization | Supported at utility level | Versioned `AdapterSpec`, effective `B@A`, masks, round-trip unit tests | Prediction-preserving real-Octo round trip |
| Real episode-safe LIBERO ingestion | Supported at utility level | Strict HDF5 validation and episode splitting | Real manifests/data and end-to-end train-only normalization proof |
| 360-sample official model zoo | Unsupported | Config/preflight count only | Official trainer, three seeds, three late checkpoints, artifacts |
| Visual/language/action evidence | Action partial; visual/language unsupported | Action statistics and array schema | Official tokenizer feature extractor, provenance, no-leakage checks |
| Joint aligned autoencoder with auxiliary supervision | Partial/deviating | Reconstruction plus three contrastive losses | Classification heads, configured weights, task-balanced batches, validated architecture |
| Linear mapper and empirical shell | Partial | Ridge mapper and per-position shell statistics | End-to-end mapped initialization and held-out gate |
| Memory-bank mapper | Unsupported | None | Implement only after linear gate passes |
| OOD generation/adaptation comparison | Unsupported | Record dataclass only | All five baselines, matched data/steps, rollouts |
| Shared-state continual learning | Scaffold only | Generic state-carrying callback runner | Official Octo stage trainer and all eight conditions |
| Differential regularization improves retention | Unsupported | Penalty function only | Validation-selected gammas, sequential rollouts, uncertainty |
| Latent drift predicts behavioral forgetting | Unsupported | Drift/correlation helpers only | Consecutive swaps, SR drops, causal stress test, CIs, plots |
| Memory advantage/privacy | Unsupported | No measured artifacts or privacy test | Measured bytes/task at matched budget; qualify/remove non-reconstructibility claim |
| Publishable result | Unsupported | No result artifacts | Four folds × three seeds × final fixed rollouts and complete provenance |

## Confirmed strengths

- The 40 LIBERO task strings and their order match Appendix B.2 of the supplied continual-VLA paper.
- The four leave-one-suite-out folds are explicitly represented.
- The archive builder rejects synthetic samples, mixed base hashes, incompatible adapter specifications, wrong splits, duplicate population identities, and incomplete 270-sample fold populations.
- Effective updates rather than arbitrary LoRA factor bases are tokenized.
- Episode-level splitting, padding masks, action normalization utilities, and action/velocity/jerk statistics are present.
- The shared-state sequential runner prevents substituting per-task adapters during evaluation by construction.
- The component-swap utility rejects nonconsecutive checkpoint swaps.
- The preflight fails closed on unsupported environments and missing data.

These are worthwhile foundations, but none independently validates the research hypothesis.

## Paper/proposal alignment notes

1. The original proposal names Pi0 and GR00T N1.5 as backbones. The user's implementation plan deliberately scopes the local study to Octo-Small and defers large-VLA confirmation. This is an explicit research-scope change, not a hidden implementation bug, but resulting claims must be limited to a small pretrained VLA and must not imply direct reproduction of the continual-VLA backbone results.
2. The proposal explicitly includes auxiliary task classification for every prompt encoder; the implementation omits it.
3. WeightCLIP includes both a linear mapper and a token-wise memory-bank mapper and validates shell-constrained refinement. Only the linear/shell estimation primitives exist here.
4. The continual-VLA paper evaluates complete lower-triangular rollout success matrices and consecutive checkpoint component swaps. The new utility design follows this, while the legacy benchmark contradicts it.
5. The proposal's statement that stored embeddings are “not reconstructible into identifiable demonstration data” is not established by code or experiment. Treat this as an untested privacy hypothesis, not a result.

## Required implementation order

1. **Make invalid execution impossible.** Quarantine/rename all proxy scripts, add artifact-type guards, and remove contradictory documentation.
2. **Repair and pass the Octo integrity gate.** Preserve a true pre-patch reference, verify exact parameter coverage and diffusion paths, test repeated-language token classification, and prove zero equivalence plus nonzero real gradients.
3. **Build one official task end to end.** Real episode split → Octo batches → train-only normalization → adapter-only training → late checkpoints → evidence extraction → packed sample → deterministic rollouts. Require adapted SR above frozen SR before scaling.
4. **Build the valid population producer.** Generate 40 × 3 × 3 samples with `TaskEvidence` provenance, immutable base hash, checkpoint-stage identities, and restart-safe manifests.
5. **Correct alignment training.** Add classification heads/loss, honor per-modality config weights, use task-balanced/global contrastive batches, justify or replace the MLP decoder, and automatically run the aligned-vs-reconstruction gate.
6. **Integrate mapped refinement.** Evidence → ridge sequence → empirical-shell projection → decoder → official Octo diffusion loss → component penalties. Make the real graph gradient test mandatory. Add memory-bank mapping only after the ridge gate passes.
7. **Implement both experiment paths.** OOD baselines and the shared-state continual conditions, including replay 10/100, oracle, uniform/proposed/inverted gamma, and reconstruction-only latent space.
8. **Implement simulator evaluation and analysis.** Fixed initialization IDs, complete matrices, consecutive swaps, correlations with bootstrap CIs, three-seed aggregation, memory/runtime/resource measurements, and stored-record-only figures.
9. **Freeze the environment and protocol.** Generate a transitive lock, hash it and every config, update manifests on completion/failure, and block publication runs unless every scaling gate passes.

## Verification performed

- Confirmed audited commit equals `origin/main`; working tree was clean at audit start.
- Read the complete proposal and both supplied reference papers.
- Compared the configured task order against Appendix B.2; all four sequences match.
- Inspected the official Octo source pinned alongside the workspace where relevant to the monkeypatch behavior.
- Ran the local unit suite with a writable base directory: 21 passed, 3 skipped.
- Ran research preflight in inspection mode: correctly failed readiness for the current Windows/Python 3.13/non-JAX/no-data host.
- Searched for generated research outputs: none were present.

## Bottom line

Do not start the 360-adapter model zoo or report any “research benchmark” output from the current repository. The next legitimate milestone is much smaller: one real LIBERO task, one pinned official Octo-Small checkpoint, a proven zero-equivalent adapter insertion, adapter-only learning above frozen-base rollout SR, and an evidence/packing/reload round trip with complete provenance. Only after that gate passes should alignment and sequential claims be attempted.
