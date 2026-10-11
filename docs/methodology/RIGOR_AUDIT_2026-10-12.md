# Methodology 1 Rigor Audit (2026-10-12)

Audited commit: `9908871` (clean working tree)
Reference: `docs/papers/Weight_Space_Alignment_Proposal.pdf`
Supersedes: `RIGOR_AUDIT.md` (2026-09-30, commit `de50968`), whose main findings are now out of date (see "Status of the previous audit").

## Verdict

The engineering scaffold is now real: the official Octo path runs end to end, the zero-adapter gate passes on the Blackwell cluster, and the protocol, provenance and leakage controls are careful. The earlier audit's "nothing is executable" verdict no longer holds.

The scientific implementation is not yet a faithful test of the proposal. Three of the proposal's central claims are not tested by the code as written:

1. **The aligned space is barely used during continual learning.** Evidence-to-latent mapping runs once, for the first task. Tasks 2 to 10 never touch the prompt encoders, the mapper or the shell.
2. **The latent conditions and the LoRA baselines are not trained on comparable budgets.** Latent refinement gets at most one eighth of the data, a different optimiser, and an early-stopping rule that is likely to end it after a few dozen steps.
3. **The drift-as-forgetting-proxy study (§2) measures weight-space drift and mixes new-task loss with old-task forgetting.**

Until these are fixed, a publication-tier run would produce numbers, but they would not support the proposal's claims about alignment, differential regularisation or latent drift.

## Scope and method

- Read in full: `wsl_vla/alignment`, `wsl_vla/adapters`, `wsl_vla/octo`, `wsl_vla/data`, `wsl_vla/evaluation` (rollout, sequential, metrics, components, component reporting, publication, recovery, gates), `wsl_vla/experiments` (conditions, gamma selection, sampling, job plan), `wsl_vla/contracts.py`, both research configs, and the zoo, evidence, archive, alignment, continual, gamma-validation, component-swap and OOD scripts.
- Read in part: `run_recovery_probe.py`, `run_alignment_advantage_gate.py`, `study_audit.py`, the OOD bank and reporting modules.
- Not read: provenance, protocol validation, zoo validation, record I/O, the legacy PyTorch smoke package, and the tests.
- Checked directly: one local LIBERO file (`data/libero/libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin...hdf5`) for image resolution and episode count.
- Evidence from the cluster: job 17739 (JAX 0.7.1 on GPU, 110 tests passed, zero-adapter equivalence `ok` at 2.86e-6).
- Not done: no training, alignment or rollout was run for this audit, and no result artifacts were inspected. Statements about what a run "will" do are inferences from the code and are marked as such.

Severity: **P0** invalidates a central claim; **P1** biases or weakens a reported comparison; **P2** is a deviation or risk to state or fix.

## P0 findings

### P0-1. Alignment is used only to initialise the first task

- `scripts/eval/run_official_continual.py:509-517`: `initial_latents = previous.latents`, and `encode_and_map_evidence` is called only when that is `None`, which is stage 0.
- For stages 1 to 9 the task's evidence is loaded and provenance-checked, then discarded. Its size is still added to `evidence_bytes`.
- The anchor `Z⁽⁰⁾` in the penalty is therefore the previous stage's refined latent, not a mapped latent for task k (`wsl_vla/alignment/models.py:491`).
- There is no nearest-neighbour retrieval step in the continual path. Nearest-neighbour exists only as an OOD baseline.
- `scripts/eval/run_gamma_validation.py` has the same structure, so γ is also selected under it.

Proposal §1.3 says the initial latent for each task k comes from retrieval-augmented mapping of that task's prompt, and that components are "pulled back toward their aligned position after each task".

Impact: the only things that differ between `proposed_asymmetric` and `reconstruction_only_latent` are the decoder weights and the starting point of task 1. The "contrastive vs direct regression" ablation therefore cannot isolate whether alignment helps continual learning. The method as coded is "L2-to-previous-latent regularisation through a learned decoder".

Fix: decide and document the intended algorithm. Either map each new task's evidence to `Z_k⁽⁰⁾` and anchor to it (as proposed), or keep the shared-latent design and restate the claim. The first option needs a rule for combining the mapped latent with the running shared latent.

### P0-2. Latent and LoRA conditions are not budget-matched

Per task, in `run_official_continual.py`:

| | LoRA conditions (`:467-478`) | Latent conditions (`:548-560`) |
|---|---|---|
| Micro-batches | 2000 updates × 8 accumulation = 16,000 | at most 2000 |
| Optimiser | AdamW, global-norm clip 1.0 | plain gradient descent (`models.py:537`) |
| Learning rate | 3e-4 | 3e-4 (same number, different optimiser) |
| Early stopping | none | patience 10, 20 or 40 steps |
| Free parameters | about 0.25 M (rank-8 factors) | about 2.6 M (20k tokens × 128) |

The early-stopping rule (`models.py:541-552`) evaluates one validation micro-batch of two samples after every step, each a fresh batch with fresh diffusion noise. The running minimum of a noisy single-batch loss is rarely beaten, so the patience counter is likely to expire within tens of steps. This is an inference; the recorded `training_update_steps` in any existing latent run will confirm or refute it.

Impact: if latent stages stop early, the latent conditions will show low forgetting because they learn little, and the comparison against LoRA baselines is confounded in both directions. The OOD script already matches budgets (`run_official_ood.py:430` sets accumulation to 1); the continual script does not.

Fix: use the same optimiser and the same number of micro-batches for both spaces, and evaluate validation loss on a fixed multi-batch set with a fixed noise key, every N steps.

### P0-3. The drift study does not measure the proposal's quantity

Proposal §2.1 defines drift as the Euclidean distance task k's latent coordinate has moved from its originally aligned position after training through task k+t. §2.2 correlates it with `swap_drop(m, k, t)`.

- **Weight space, not latent space.** `wsl_vla/evaluation/components.py:55-74` computes the root-sum-square difference of dense effective updates between consecutive checkpoints. The field is named `latent_drift` but the autoencoder is not involved. LoRA conditions do the same (`run_official_continual.py:166-182`).
- **No per-task coordinate.** There is one shared latent, so "task k's latent" does not exist after stage k. Latent conditions record only the consecutive-stage distance of the shared latent (`wsl_vla/octo/continual.py:213-221`).
- **Consecutive transitions only.** t is always 1.
- **New-task and old-task effects are averaged together.** `run_component_swaps.py:261` evaluates all seen tasks including the one just learned, and `component_reporting.py:82-90` averages `success_rate_drop` over all of them. Swapping the previous component back in removes new-task skill (positive drop) and restores old-task skill (negative drop). The mean of the two is not a forgetting measure.
- **The causal stress test is absent.** Nothing sweeps γ to check that drift and swap-drop move together.
- **The stage-level drift-vs-forgetting correlation is not in the plan.** `drift_degradation_correlations` exists (`metrics.py:278`) but only `report_research_metrics.py` calls it, and `job_plan.py` never schedules that script.

Impact: §2 of the proposal (drift as a rollout-free forgetting proxy, and §2.3 monitoring) is not tested.

Fix: store each task's encoded latent at the stage it was learned; at later stages re-encode the current policy's update and measure the per-modality distance; restrict swap-drop to previously learned tasks and fix its sign convention; add the γ-sweep.

## P1 findings

### P1-1. Shell projection is not in the regulariser

`differential_local_penalty` (`models.py:423-436`) computes `γ·‖Z − Z⁽⁰⁾‖²`. The proposal's term is `γ·‖Π_shell(Z) − Z⁽⁰⁾‖²`. `project_to_empirical_shell` is applied once, to the mapped initial latent (`checkpoint.py:139`), and never during refinement.

### P1-2. Only the fixed γ schedule exists

Drift-informed and online schedules appear only in the deprecated `run_research_benchmark.py`. Ablation 3 (fixed vs drift-informed) cannot be run.

### P1-3. The inverted and reconstruction-only conditions use untuned γ

`conditions.py:125-137`: `direction_inverted` is derived mechanically from the selected proposed values, and `reconstruction_only_latent` reuses γ tuned on the aligned checkpoint. Only `proposed` and `uniform` are searched (`gamma_selection.py`). The proposed condition gets a validation-tuned setting and its two key ablations do not, which favours the proposed condition.

### P1-4. Memory accounting does not match §3.1 and omits the replay buffer

- Latent conditions report `adapter_bytes = tree_nbytes(decoded adapter state)` (`continual.py:233`), which is the dense decoded kernels plus factors, not `Z`.
- `evidence_bytes` is the raw evidence features, not the embeddings `e_m`.
- Replay conditions record `evidence_bytes = 0` and nothing for the buffer (`run_official_continual.py:466`), so replay's stored transitions are not counted at all.
- By my estimate `Z` is about 20k tokens × 128 float32 ≈ 10 MB, against about 1 MB for the rank-8 factors it replaces. The token count is from the docstring at `models.py:41`; the byte figures are arithmetic, not measured.
- Replay sizes are fixed at 10 and 100 transitions; nothing matches them to the latent method's footprint ("matched memory-footprint budgets", §3.3).

Impact: the NBT-vs-memory Pareto cannot be drawn correctly, and the "orders of magnitude smaller" claim is doubtful for this adapter design.

### P1-5. Train and evaluation images are processed differently

- The local demonstration file stores both cameras at 128×128 (checked on one `libero_spatial` file).
- Training upsamples the primary camera to Octo's 256×256 with bilinear resize (`octo_batches.py:48-63`).
- Rollouts render at 256×256 (`configs/research/base.yaml:15-16`, `libero_rollout.py:183`), so the primary image is native-resolution and sharp, and the wrist image is downsampled from 256 instead of being native 128.

Impact: a systematic train/eval distribution shift in every rollout number. Fix: render at the dataset resolution and apply the identical resize.

### P1-6. Contrastive batches have one negative task

`configs/research/base.yaml:49-50` sets 2 tasks × 2 samples; `jobs/04_alignment.sh` does not override it. Gradient accumulation does not add negatives. §3.2 identifies batch diversity as the binding constraint. The chunked attention work was done to fit a 96 GB slice, so this may be a memory limit, but it should be stated and, if possible, raised (for example by computing the contrastive loss over cached embeddings).

### P1-7. The alignment-advantage gate is not a fair comparison

`wsl_vla/alignment/evaluation.py:39-43` requires higher retrieval and lower mapper MSE than the reconstruction-only checkpoint.

- With `--contrastive-weight 0`, the classification weight is also zeroed (`train_research_alignment.py:139`), so the reconstruction-only prompt encoders are never trained. Its retrieval is at chance by construction.
- Mapper MSE is computed in each checkpoint's own latent space. The two spaces have different scales, so the MSE values are not comparable.
- No gate checks behaviour: nothing decodes an encoded zoo adapter and confirms that rollout success is preserved.

### P1-8. Recovery efficiency cannot be reported for the proposed method

`run_recovery_probe.py:166` rejects any non-LoRA run, and `job_plan.py:220` schedules it only for `sequential_no_regularization`. Recovery efficiency is a primary metric in §3.3.

### P1-9. Uncertainty estimates rest on very small samples

- `bootstrap_mean` over three seeds (`metrics.py:176`) has ten distinct resamples.
- The crossed bootstrap uses 4 suites × 3 seeds (`publication.py:14`).
- No paired difference between conditions is computed anywhere, although rollouts are paired by design (same initial states and seeds across conditions).
- Alignment is trained with a single seed per fold (`train_research_alignment.py:91`), so alignment variance is not represented.

Fix: report paired per-(suite, seed, task) differences between conditions and state the limits of the interval estimates.

## P2 findings

- **Backbone.** Octo-Small, not Pi0 or GR00T N1.5. Presumably a compute decision; claims must be scoped accordingly.
- **Adapter form.** Not LoRA on existing weights. It is a zero-initialised low-rank residual after each transformer block, applied only to tokens of one modality (`bridge.py:94-118`), plus low-rank updates to the diffusion head counted as "action". This is a reasonable reading of "component-factorised" for a single-transformer model, but it is an interpretation. The language adapter acts on the T5 output tokens inside Octo, not on the language encoder.
- **Decoded policies are full rank.** The decoder emits dense 384×384 updates (`bridge.py:111-118`), so latent conditions are not constrained to rank 8.
- **Decoder architecture.** A per-token MLP that ignores component and layer ids (`models.py:149-162`); the proposal specifies a transformer encoder/decoder pair.
- **Reverse contrastive direction** uses mean-pooled latents (`models.py:385-392`), not per-token terms.
- **Action evidence encoder** is an MLP over summary statistics, not a set encoder.
- **Auxiliary classifier carries little signal.** Evidence is identical for all nine samples of a task, so the head only has to memorise 24 distinct inputs.
- **Mapper sample size.** Each fold fits the ridge mapper on 216 samples that contain only 24 distinct evidence vectors (`train_research_alignment.py:398`).
- **Per-task action normalisation.** Each task has its own mean and std (`train_research_zoo.py:212`, `run_official_continual.py:600`), and rollouts use the evaluated task's statistics. The shared policy therefore receives task-specific output scaling. Prior continual-VLA work normalises per dataset.
- **Data split.** 70/15/15 by episode leaves 35 of 50 demonstrations for training; published LIBERO numbers use all 50.
- **Action chunking.** The policy re-plans every step and executes only the first action of each chunk (`libero_rollout.py:378`).
- **Image orientation.** I did not verify whether the stored frames match Octo's pretraining orientation. Train and eval use the same convention, so this affects transfer, not correctness.
- **Component swaps** are scheduled only for `proposed_asymmetric` (`job_plan.py:212`), so there is no baseline swap profile to compare against.
- **Refinement is not JIT-compiled** (`models.py:533`); a performance cost only.

## Claim-to-evidence matrix

| Proposal claim | Status | Blocking finding |
|---|---|---|
| §1.1 Component-factorised tokenisation with a shared autoencoder | Implemented, with interpretation | P2 (adapter form, decoder) |
| §1.2 Three per-modality contrastive losses, learnable τ, λ weights | Implemented | P1-6 |
| §1.2 Jointly trained prompt encoders with classification heads | Implemented | P2 (weak signal) |
| §1.3 Retrieval-augmented initial latent per task | Stage 0 only, no retrieval | P0-1 |
| §1.3 Refinement through the decoder with per-modality γ | Implemented | P0-2, P1-1 |
| §1.3 γ_vis, γ_lang > γ_act, fixed schedule by held-out grid search | Implemented | P1-3 |
| §1.3 Drift-informed and online schedules | Absent | P1-2 |
| §2.1 Latent drift of task k | Not implemented as defined | P0-3 |
| §2.2 Drift vs swap-drop correlation per component | Runs, but on a different quantity | P0-3 |
| §2.2 Causal stress test | Absent | P0-3 |
| §3.1 Memory = {e, Z}, bytes/task, Pareto vs NBT | Mis-measured | P1-4 |
| §3.2 Model zoo, ~40 tasks, multi-seed | Implemented (40 × 3 seeds × 3 checkpoints) | none |
| §3.3 SR, NBT | Implemented | P0-2, P1-5, P1-9 |
| §3.3 Recovery efficiency | LoRA baseline only | P1-8 |
| §3.3 Ablation 1, contrastive vs regression | Runs, cannot isolate alignment | P0-1, P1-3 |
| §3.3 Ablation 2, component-wise vs uniform | Implemented | P0-2 |
| §3.3 Ablation 3, fixed vs drift-informed | Absent | P1-2 |
| §3.3 Ablation 4, direction inverted | Implemented, untuned | P1-3 |
| §3.3 Pi0 / GR00T backbones, matched memory budgets | Not followed | P2, P1-4 |

## What is sound

- Zero-adapter equivalence is computed against a reference captured before the module patch, and passes on the target hardware.
- Effective updates, not raw factors, are tokenised, so the representation is invariant to the factor basis.
- Leave-one-suite-out folds are enforced in the archive builder, the gamma selection loader and the OOD bank; held-out-suite adapters cannot enter alignment training.
- Evidence comes from the frozen pretrained tokenisers and training episodes only, and its dataset, base and episode identities are checked wherever it is consumed.
- Rollouts across conditions share initial states and seeds; every record carries them.
- The shared-state runner offers no per-task checkpoint lookup, so continual conditions cannot silently use task-specific adapters; the oracle is a separate function.
- γ and patience are selected on validation tasks of the source suites only, and the selection file is bound to its fold.
- A gradient-integrity check runs before every latent refinement.
- The multi-positive InfoNCE correctly treats same-task checkpoints as positives.

## Status of the previous audit (2026-09-30)

| Earlier finding | Now |
|---|---|
| No end-to-end official experiment | Resolved: continual, OOD, swap and recovery commands exist |
| Zero-adapter gate likely invalid | Resolved: reference captured pre-patch; passes on cluster |
| Legacy programs discoverable as research | Partly resolved: quarantined under `wsl_vla/smoke`, but `scripts/eval/run_research_benchmark.py` keeps its name |
| Zoo and evidence stages missing | Resolved |
| No classification heads; config weights ignored | Resolved |
| MLP decoder; two-sample contrastive batches | Still open (P2, P1-6) |
| Advantage gate not called | Called by `jobs/04_alignment.sh`, but see P1-7 |
| Refinement never invokes shell projection | Still open (P1-1) |
| Only fixed γ; no memory-bank mapper | Still open (P1-2) |
| No rollouts, no multi-seed reporting | Resolved, with P1-9 |
| Acceptance tests do not exercise the science | Not re-assessed; the 110-test suite runs in 19 s, so it cannot include real training or rollouts |

## Recommended order

1. **P0-2 first.** It is the cheapest to fix and, if the early-stopping inference is right, every existing latent-condition number is uninformative. Check `training_update_steps` in any completed latent run before doing anything else.
2. **P1-5** (image processing), because it affects every rollout, including baselines.
3. **P0-1.** Decide the per-task mapping design; this determines what the paper claims.
4. **P0-3** together with **P1-4**, since both need per-task latents stored at learning time.
5. **P1-1, P1-3, P1-7, P1-8**, which are local changes.
6. **P1-2** (drift-informed schedule) only after P0-3, because it depends on a valid drift measure.
7. Re-run gamma validation after 1 to 3; the current selection was made under the confounded budget.
