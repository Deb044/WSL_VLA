# Implementation Status

Three levels are kept apart in this table:

- **Tested**: automated tests cover it (most on a toy model built from Octo's
  own classes, or on synthetic populations).
- **Run on real inputs**: executed with the pinned checkpoint, real LIBERO
  demonstrations, and, where stated, the real simulator, on an RTX 4050 Laptop
  GPU (6 GB) in the `legacy-gpu` environment, at integrity scale.
- **Not run**: needs collaborator hardware.

Nothing below is evidence that alignment improves anything. The integrity
profile has two tasks and 24 optimizer updates per run.

## Zoo and alignment pipeline (the October 2026 repair)

| Plan area | Tested | Run on real inputs | Implementation |
|---|---|---|---|
| Pre-block, token-conditioned adapters; explicit routing; frozen or rejected unsupported groups | yes | yes | `wsl_vla/octo/bridge.py` |
| Zero-adapter equivalence (transformer outputs and sampled actions); base immutability | yes | yes (difference 0.0) | `wsl_vla/octo/bridge.py` |
| Per-branch gradient reachability with zero-init semantics; a branch one random draw leaves at zero is redrawn (needed under JAX 0.7) | yes | yes (47 branches) | `wsl_vla/octo/training.py` |
| Fixed controller-action convention; target validation; deployment clipping report | yes | yes | `wsl_vla/data/actions.py`, `wsl_vla/contracts.py` |
| Shared train/evidence/rollout preprocessing; tokenizer fingerprint | yes | yes | `wsl_vla/data/preprocessing.py` |
| Camera orientation measured against demonstration generation | - | yes (simulator) | `scripts/setup/verify_camera_orientation.py` |
| Independent zoo runs; explicit updates, micro steps, exposure; snapshots | yes | yes | `wsl_vla/octo/zoo.py`, `scripts/train/train_research_zoo.py` |
| Exact resume, separate from snapshots, refused on any identity mismatch | yes (bit-exact on CPU) | yes: killed at update 50, resumed, bit-identical at 75 and 100 (with deterministic GPU ops) | `wsl_vla/octo/zoo.py` |
| Bit-reproducible GPU zoo training across processes | yes (runner sets it per job) | yes with `--xla_gpu_deterministic_ops=true`, at no measurable cost; without it two identical runs differed by float rounding. Not used for alignment training (about 36x slower there) | `wsl_vla/experiments/zoo_jobs.py`, `runtime.py` |
| A zoo run learns | - | yes: 400 updates, held-out loss 3.27 to 0.55, open-loop action error 0.68 to 0.24 | `scripts/train/train_research_zoo.py` |
| Held-out loss, per-component norms, open-loop sampling check | yes | yes | `wsl_vla/octo/zoo.py` |
| Paired frozen-base versus adapter rollouts; structural versus behavioural validity | yes (scripted simulator) | yes (simulator) | `scripts/eval/evaluate_zoo_behaviour.py` |
| Per-artifact schema versions; version-one artifacts inspectable and rejected | yes | yes | `wsl_vla/contracts.py`, `wsl_vla/experiments/legacy.py` |
| Layout with parameter, row, column-window, and sequence-window identities | yes | yes | `wsl_vla/adapters/packing.py` |
| Frozen-feature evidence with episode ids; per-episode action-chunk command differences | yes | yes | `wsl_vla/alignment/evidence.py`, `octo_evidence.py` |
| Training-only evidence standardization and RMS scales | yes | yes | `wsl_vla/alignment/archive.py` |
| Experiment manifest as the sole source of counts, splits, folds | yes | yes | `wsl_vla/experiments/manifest.py` |
| Leakage-checked, hash-rechecked, memory-mapped fold archives | yes | yes | `wsl_vla/alignment/archive.py` |
| Fully resolved alignment configuration with explicit override precedence | yes | yes | `wsl_vla/alignment/training.py` |
| Shared windowed transformer encoder/decoder; structural embeddings | yes | yes | `wsl_vla/alignment/models.py`, `windows.py` |
| Component-balanced reconstruction on RMS-normalized targets | yes | yes | `wsl_vla/alignment/models.py` |
| Token-wise bidirectional multi-positive contrastive loss (reference-checked) | yes | yes | `wsl_vla/alignment/models.py` |
| Paired aligned / reconstruction-only conditions with fingerprint checks | yes | yes | `wsl_vla/alignment/checkpoint.py` |
| Task-balanced ridge mappers; training-only shells; before/after projection | yes | yes | `wsl_vla/alignment/mapping.py`, `training.py` |
| Rank-matched deployment with reported projection error | yes | yes | `wsl_vla/adapters/rank.py` |
| Offline diagnostics separating integrity, inconclusive, negative | yes | yes (all inconclusive, as two tasks must be) | `wsl_vla/alignment/diagnostics.py` |
| Generated-policy rollouts, ten paired conditions, clustered intervals | yes (scripted simulator) | yes (simulator) | `scripts/eval/run_generated_policy_rollouts.py` |
| Resumable job manifests with dependencies and failure summaries | yes | yes | `wsl_vla/experiments/zoo_jobs.py` |
| Several runners on one job manifest (one per GPU slice): atomic claims, waiting on other runners, takeover of a killed runner's jobs, interruption on SIGTERM; several jobs per runner with a separate limit for GPU training | yes (threads and separate processes on one host) | - (not yet on Slurm or across nodes) | `wsl_vla/experiments/zoo_jobs.py`, `jobs/submit_workers.sh` |
| Atomic, idempotent stage outputs; resumable rollouts; preserved failed attempts | yes | yes | `wsl_vla/evaluation/rollout_runs.py` and stage scripts |
| Every planned command accepted by its script | yes | - | `tests/unit/test_experiments/test_script_cli.py` |
| Real Octo/LIBERO integration test with explicit prerequisite reporting | - | yes | `tests/integration/test_real_pipeline.py` |
| `legacy-gpu` environment lock | - | yes | `envs/requirements/research-legacy-gpu.txt` |

## Not run, or not verified

| Item | State |
|---|---|
| `blackwell` environment with the corrected pipeline | Not run on Blackwell hardware. A local CPU replica with its exact package versions (Python 3.11, JAX 0.7.1, Flax 0.12, TensorFlow 2.20, Transformers 4.57.6, patched Octo `a4cc964`; CUDA wheels and simulator left out) passes the unit suite: 344 passed, 2 legacy PyTorch smoke modules skipped. Run profiles 1 and 2 on the cluster first; the architecture hash must equal the one in the runbook |
| From-scratch install of the `legacy-gpu` lock | Not executed (the verified environment was built incrementally); every pin was checked to exist |
| Pilot profile, budget inspection, budget freeze | Not run |
| Full profile | Not run |
| Whether 5,000 updates is a sufficient zoo budget | Unknown. At 24 updates the denoiser is no better than predicting zero noise and rollouts saturate the controller. At 400 updates (1 percent of the planned exposure) the run clearly learns but has no rollout success yet |
| Deterministic GPU ops on Blackwell | JAX 0.7.1 accepts the flag (an unknown XLA flag aborts at startup; this one does not). Its cost and its effect on reproducibility on Blackwell are not measured |
| Several runners on Slurm | Runners coordinate through the state directory (claims made with `mkdir`, heartbeats judged by the filesystem's clock), which must be on scratch shared by all nodes. Tested with separate processes on one host; `jobs/submit_workers.sh` and `jobs/03_model_zoo.sh` / `04_alignment.sh` were not submitted |

## Downstream study

Gamma validation, continual learning, OOD adaptation, component swaps, and
recovery consume alignment checkpoints. Their loaders were updated to the new
contracts, their job plan (`scripts/reporting/plan_methodology1_jobs.py`) now
extends the corrected plan and waits for diagnostics *integrity* rather than an
alignment win, and every planned flag is checked against each script's parser.
They have **not been executed** on corrected artifacts.

| Area | Implementation | State |
|---|---|---|
| Validation-frozen gamma/patience selection | `scripts/eval/run_gamma_validation.py`, `select_validation_gammas.py` | re-pointed, not run |
| Shared-state continual protocol, eight conditions | `scripts/eval/run_official_continual.py` | re-pointed, not run |
| OOD five-method protocol and reference bank | `scripts/eval/run_official_ood.py`, `scripts/data/build_ood_reference_bank.py` | re-pointed, not run |
| Component swaps, recovery curves | `scripts/eval/run_component_swaps.py`, `run_recovery_probe.py` | re-pointed, not run |
| Reports and completion audit | `scripts/reporting/report_*.py`, `audit_methodology1_study.py` | audit rewritten for the corrected layout; reports not run |
| Latent-refinement conditions | `wsl_vla/alignment/checkpoint.py` | still deploy dense decoded updates; outside the primary rank-matched comparison |
| Large-VLA confirmation | - | deferred |

Earlier design and audit notes (`IMPLEMENTATION_REVIEW_2026-10-02.md`,
`RIGOR_AUDIT*.md`, `MODEL_ZOO_PREFLIGHT_PLAN.md`) describe the pipeline before
the repair and are kept for history.
