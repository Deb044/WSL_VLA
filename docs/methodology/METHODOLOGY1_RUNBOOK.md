# Zoo and Alignment Runbook

This is the execution order for the corrected Octo-Small model-zoo and
weight-space alignment experiments. It replaces the earlier Methodology 1
runbook. Every command below exists in this repository and was exercised on the
integrity profile unless a section says otherwise.

What "done" means here: the infrastructure runs, its integrity checks pass, and
results are recorded honestly. It does **not** mean that alignment improves
anything. Section 9 explains how outcomes, including negative ones, are
reported.

## 0. What was verified, and where

| Item | Status |
| --- | --- |
| `legacy-gpu` environment (Python 3.10, JAX 0.4.20, CUDA 11) | Verified on an RTX 4050 Laptop GPU (6 GB), WSL2 |
| Unit and integration tests | Pass in the `legacy-gpu` environment |
| Integrity profile, non-simulator stages (zoo, evidence, archive, paired alignment, diagnostics) | Run end to end on the real checkpoint and real LIBERO demonstrations |
| Integrity profile, simulator stages (camera check, paired rollouts) | Run in real LIBERO with software rendering (Mesa llvmpipe EGL) |
| A zoo run learns (400 updates, one task) | Held-out loss 3.27 to 0.55; open-loop action error 0.68 to 0.24 |
| Reproducibility on GPU | Zoo training bit-identical across processes and across kill-and-resume **with** `--xla_gpu_deterministic_ops=true` (no measurable cost); without it identical runs differed by float rounding. The flag slows alignment training about 36x, so it is not used there |
| `blackwell` environment (Python 3.11, JAX 0.7.1, CUDA 12) | **Not run on Blackwell hardware.** A CPU replica with its exact package versions passes the unit suite (344 passed; the legacy PyTorch smoke tests are not installed there), and JAX 0.7.1 accepts the deterministic-ops flag. GPU kernels, MIG, and CUDA 12.8 remain untested |
| Several runners sharing one job manifest (one per GPU slice) | Tested with threads and with separate processes on one host: each job runs once, runners wait for each other, a killed runner's jobs are taken over, SIGTERM interrupts cleanly. Not yet run on Slurm or across nodes |
| Pilot and full profiles | **Not run.** They need collaborator hardware |
| Downstream study (`m1:*` jobs: gamma validation, continual learning, OOD, probes) | Commands were re-pointed and are checked against each script's argument parser, but have **not been executed** on corrected artifacts |

Run section 2 and section 3 on your own hardware before anything else. A
profile is verified on a machine only after the integrity profile passes there.

## 1. Environments

There are two separately locked research environments and one legacy smoke
environment. They are never mixed inside one experiment: every artifact records
the profile that produced it.

| Profile | Python | JAX | Octo | Install |
| --- | --- | --- | --- | --- |
| `blackwell` | 3.11 | 0.7.1 via `jax[cuda12]` (NVIDIA CUDA 12 pip wheels, CUDA >= 12.8 for sm_120) | upstream `241fb35` + `third_party/octo` patch = commit `a4cc964` | `envs/scripts/setup_research.sh --profile blackwell <venv>` |
| `legacy-gpu` | 3.10 | 0.4.20 with CUDA 11 pip wheels (the path JAX supported for that release and that Octo documents) | unmodified upstream `241fb35` | `envs/scripts/setup_research.sh --profile legacy-gpu <venv>` |
| `legacy-smoke` | 3.10 | none (PyTorch) | none | `envs/scripts/setup_legacy_smoke.sh` |

```bash
cd WSL_VLA
PYTHON=python3.11 envs/scripts/setup_research.sh --profile blackwell /scratch/$USER/envs/vla
source /scratch/$USER/envs/vla/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false     # required by every research command
```

GPU determinism is set per stage. Without
`XLA_FLAGS=--xla_gpu_deterministic_ops=true`, XLA may choose different GPU
kernels in different processes: two identical zoo runs on the RTX 4050 differed
by float rounding (held-out loss 0.5533 versus 0.5531 after 400 updates), and a
resumed run was not bit-identical to an uninterrupted one. With the flag, two
runs and a killed-and-resumed run were bit-identical at no measurable cost for
zoo training. For alignment training the same flag made each update about 36
times slower (61 s instead of 1.7 s), because its backward pass is
scatter-heavy. `run_job_manifest.py` therefore adds the flag to every job
except alignment training; do not export it globally. When you run a zoo
command by hand, export it for that command only. Every environment record
states whether the flag was set. JAX 0.7.1 (`blackwell`) accepts the flag
(checked with its CPU build: an unknown XLA flag aborts at startup, this one
does not); what it costs and whether it makes runs bit-identical on Blackwell
is not measured. If it ever has to go, run the job runner with
`--allow-nondeterministic-gpu` and record that.

Notes:

- `requirements/research-blackwell.txt` holds pinned direct dependencies and is
  resolved by pip. `requirements/research-legacy-gpu.txt` is the complete
  `pip freeze` of the verified environment and is installed with `--no-deps`
  (resolving it would fail: TensorFlow 2.15 declares `ml-dtypes~=0.2`, the
  verified environment runs 0.5.0). The legacy-gpu environment was built
  incrementally; a from-scratch reinstall from the lock has not been executed.
  All 192 pins were checked to exist on their indexes.
- `wsl_vla/octo/compat.py` is the documented compatibility layer. The Blackwell
  port only replaces removed JAX aliases. Both profiles must report the same
  `architecture_sha256` for the pinned checkpoint
  (`e0572281170c112805ec1c0b52471eb8e4f91a62084469ec9010b8edd35a8e0d`); a
  different value on Blackwell means the port changed the architecture and is a
  blocking defect.
- After the integrity profile passes in an environment, record its complete
  lock: `envs/scripts/freeze_environment.sh <profile>` writes
  `envs/locks/research-<profile>.freeze.txt`.
- If you already have an environment with JAX, Flax, Octo, and torch but no
  simulator, `envs/scripts/add_simulator_overlay.sh <python> <dir>` adds the
  pinned LIBERO stack in a `--system-site-packages` virtual environment without
  touching the base environment. This is how the simulator stages were run on
  the laptop.
- The legacy-smoke profile runs the old PyTorch smoke scripts only. Nothing it
  produces is used by these experiments.

### Rendering

Rollouts need a MuJoCo OpenGL backend. `MUJOCO_GL=egl` with Mesa's software
renderer works without a display (this is what was verified). On a cluster,
`jobs/01_render_check.sh` reports which of `egl`, `mesa-egl`, `osmesa` works;
pass the result as `RENDER=...` to the job scripts. MIG GPU slices have no
graphics API, so they need `mesa-egl`.

### On the cluster: GPU slices and parallel runners

The target cluster gives a user one RTX PRO 6000 Blackwell (96 GB) at a time,
split into MIG slices: four `1g.24gb` (`gpu_small`: 6 CPUs and 24 GB of host
memory each), two `2g.48gb`, or one `4g.96gb` (`gpu_large`: 24 CPUs, 96 GB).
The total is the same; how much of it the pipeline can use is not. Use four
small slices with one runner on each:

```bash
RENDER=mesa-egl jobs/submit_workers.sh small 4 jobs/03_model_zoo.sh
```

Why, from measurements on the laptop:

- No job needs more than a small slice. Zoo training at micro-batch 2 peaked
  at 2.5 GB of GPU memory and 4.8 GB of host memory (micro-batch 8, below,
  needs about 8.5 GB of GPU memory). Alignment training peaked at 1.7 GB of GPU
  memory with 4 adapter samples per batch and 1.9 GB with 6, so about 2.3 GB
  at the pilot's 12, and 2 GB of host memory.
- The work is many independent jobs: 24 zoo runs in the pilot and 120 in the
  full profile, 8 and 24 alignment runs, rollouts per fold and alignment seed.
  MIG gives each runner dedicated compute; processes sharing one large slice
  take turns on it.
- Rollouts dominate the cost, and on MIG they are CPU-bound. MIG slices have no
  graphics API, so MuJoCo renders with Mesa's software renderer. One simulator
  step took 0.20 s, 0.18 s of it rendering the two 128x128 cameras; the policy
  took 0.025 s on the laptop GPU. All three layouts have 24 CPUs, and using them
  takes several rollout processes per slice. A runner therefore executes
  `WORKERS=3` jobs at once (the default), at most `TRAINING_SLOTS=1` of them GPU
  training, so two rollouts run beside a training job. A rollout process
  peaked at 4.8 GB of host memory and 1.1 GB of GPU memory, like a training
  job's host memory, and keeps about two CPUs busy; three jobs fit a slice's
  24 GB and 6 CPUs.
  `jobs/render_env.sh` gives each simulator two rendering threads
  (`LP_NUM_THREADS=2`): two were 1.3 times faster than one, more gave nothing.
- `gpu_large` allows one running job per user, and a whole card is harder to
  schedule than a quarter of one.

How runners share the work: all runners of an experiment read the same job
manifest and state directory (`$RESEARCH_ROOT` on scratch, visible from every
node). A runner claims a job by creating `jobs/claims/<job>/` (an atomic
`mkdir`), writes the outcome to `jobs/records/<job>.json`, waits for
dependencies that another runner is executing, and takes over the job of a
runner that has sent no heartbeat for 15 minutes. At the time limit Slurm's
SIGTERM makes a runner terminate its jobs and record them as interrupted; zoo
training and rollouts resume from their saved state. Resubmit the same
`submit_workers.sh` command until `jobs/failure_summary.json` shows nothing
pending, and give every runner of one experiment the same `MANIFEST`,
`STAGES`, and `RESEARCH_ROOT`. A job that fails is retried by runners started
later, never by those that saw it fail.

Before training the pilot zoo, choose the micro-batch for the slices you will
use; it is part of the zoo identity, so decide it once. Zoo training needs about
0.5 GB plus 1 GB of GPU memory per sample of micro-batch (measured: 2.5 GB at
micro-batch 2, 4.4 GB at 4; 8 did not fit in 6 GB), so about 8.5 GB at 8 and
16 GB at 16. The pilot and full manifests therefore use `micro_batch_size: 8`
with `gradient_accumulation_steps: 2`, which fits a 24 GB slice beside two
rollout processes; the effective batch is 16 either way. Larger micro-batches are faster per sample (21 against 17 samples per
second at 4 against 2 on the laptop).

Measured in the integrity run on one 1g.24gb slice (2026-10-15, 17 of 17 jobs,
44 minutes): zoo training 21.4 to 21.9 micro-steps per second at micro-batch 2
(44 samples per second, about 2.5 times the laptop), 17 to 38 s compile,
2.4 GB GPU memory, 5.8 GB host memory; generated-policy rollouts 0.35 s per
simulator step including setup (4,800 steps in 1,694 s).

Rough pilot timeline on four small slices, assuming every episode runs to the
600-step horizon (failed episodes do) at 0.35 s per step: zoo training and
paired behaviour rollouts about 10 hours, alignment about 2, validation
rollouts about 35, held-out rollouts about 85. The rollout phases are bound by
the number of rollout jobs (one per fold and alignment seed, four in the
pilot), not by the slices; splitting them by task would let all twelve worker
slots take part and cut those two phases about threefold.

Rollout jobs outlast a 24-hour walltime, and resubmitting resumes them only
from the last *completed task* (all conditions and episodes of a task are
recorded together). A validation task is at most about 12 hours (10 conditions
x 20 episodes x 600 steps x 0.35 s), so it always completes. A held-out task at
the final tier is up to about 29 hours (50 episodes), longer than one 24-hour
job: request a longer walltime for the held-out stage, or make the rollout
record finer-grained before running it.

Jobs run with the hub switched off, and compute nodes may have no internet.
On a login node, first fill the cache the jobs use and fetch the data:

```bash
export HF_HOME=/scratch/$USER/hf_cache
python scripts/setup/prefetch_research_cache.py    # pinned checkpoint + t5-base tokenizer/config
python scripts/setup/download_libero.py --manifest configs/research/manifests/pilot.yaml \
    --dest /scratch/$USER/data/libero
```

Order on the cluster: `sbatch jobs/00_gpu_check.sh` (JAX sees the slice;
zero-adapter equivalence), `sbatch jobs/01_render_check.sh` (which `RENDER`
works; expect `mesa-egl`), then the integrity profile with one runner:

```bash
STAGES=gates,evidence,zoo,zoo_behaviour,zoo_verification,archive,alignment,diagnostics_validation,rollouts_validation,report \
  sbatch --export=ALL,RENDER=mesa-egl,MANIFEST=configs/research/manifests/integrity.yaml jobs/03_model_zoo.sh
```

`STAGES` goes in the environment rather than in `--export`, because sbatch
splits `--export` at commas and would keep only `gates`. The job log's second
line prints the stages it received.

### Offline operation

After the first successful load, run with `HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1` (or `run_job_manifest.py --offline`). The checkpoint is
pinned by revision and hash, but Octo fetches the T5 tokenizer files by model
name. Offline mode stops a hub outage from stalling jobs; independently, the
tokenizer's output on fixed probe instructions is part of the preprocessing
hash, so a changed tokenizer cannot go unnoticed.

## 2. Run profile 1: environment, checkpoint, and gates

```bash
pytest -m "not integration" -q
python scripts/setup/download_libero.py --manifest configs/research/manifests/integrity.yaml
python scripts/eval/research_preflight.py \
    --manifest configs/research/manifests/integrity.yaml \
    --output-dir research_results/experiments/integrity/gates/preflight
python scripts/eval/verify_official_octo.py \
    --output research_results/experiments/integrity/gates/octo_gates.json
python scripts/setup/verify_camera_orientation.py --suite libero_spatial --task-index 0 \
    --output-dir research_results/experiments/integrity/gates/camera_orientation/libero_spatial
```

Expected:

- `pytest`: no failures. Tests marked `research` need a research environment;
  a skipped test is not a passed test. On a machine without the research stack
  the summary line lists what was skipped.
- `preflight.json`: `"ok": true`. `problems` block the run; `limitations` are
  recorded facts (for example, a profile that is unverified on this hardware).
  It also lists any version-one directories it found; see section 8.
- `octo_gates.json`: zero-adapter equivalence for transformer outputs **and**
  sampled actions (identical random key), per-branch gradient reachability, the
  token routing read back from the live graph, and the unchanged base hash.
  On the RTX 4050 the equivalence difference is exactly 0; a few-ulp residue is
  acceptable on other hardware, the tolerance is `1e-5`.
- `camera_orientation.json`: `"status": "verified"`. The script restores
  recorded demonstration states in the simulator, re-renders them, and checks
  that the configured `simulator_alignment` is the best match by a clear margin
  (measured: `identity`, error 15 to 63 times lower than any other transform).
  Without a simulator it exits 3 and reports `simulator_unavailable`; that is
  not a pass. It also saves a three-panel image per camera (stored frame,
  simulator frame, frame presented to the checkpoint) for visual inspection.

Gradient gate semantics: adapter `up` factors start at zero. Every `up` factor
must have a non-zero gradient immediately. Every `down` gradient is exactly
zero at initialization (it is multiplied by the zero `up` factor) and must
become non-zero after one update of the `up` factors.

## 3. Run profile 2: two-task integrity run

The integrity manifest trains two tasks for 24 optimizer updates, saves three
snapshots, builds a fold that reuses its training tasks for selection, trains
both alignment conditions for 40 updates, and rolls out every policy condition
with two episodes of 120 steps. It proves that the pipeline executes. It is not
evidence about anything: every artifact it writes is marked `integrity_only`.

```bash
python scripts/reporting/plan_zoo_alignment_jobs.py \
    --manifest configs/research/manifests/integrity.yaml
python scripts/train/run_job_manifest.py \
    research_results/experiments/integrity/jobs/jobs.jsonl --offline
```

The runner prints counts of `completed`, `failed`, `blocked`, and `skipped`
jobs and writes `jobs/failure_summary.json` with the tail of every failed log.
Rerun the same command to retry: completed jobs are skipped.

On a machine without the simulator add `--skip-simulator`. The camera gate and
all rollout jobs are then reported as skipped, and nothing that depends on them
is marked complete.

Measured on the RTX 4050 (6 GB VRAM, 7.8 GB RAM):

| Stage | Measurement |
| --- | --- |
| Zoo training | 8.5 to 9.2 micro-steps/s at micro-batch 2 in steady state, plus about 30 s of JIT compilation per process; peak RAM 4.4 to 5.2 GB; peak VRAM 2.5 to 2.7 GB |
| One adapter snapshot | 26 MB (`adapter.npz`) |
| Evidence per task | 0.6 MB |
| Alignment training | 1.7 to 1.8 s per update with 2 tasks x 2 samples, `window_chunk: 64` |
| Fold archive | 30.7 MB of float32 tokens per adapter sample |
| Paired rollouts | about 0.45 s per environment step and policy with software rendering |

`run.json` of every zoo run records compilation time separately from
`steady_micro_steps_per_second`, plus held-out evaluation time, snapshot and
resume I/O, and peak host/accelerator memory. Size longer runs from the steady
rate: in a 24-update run compilation dominates the wall time. On the RTX 4050
a 5,000-update run at effective batch 16 (40,000 micro-steps) is about 75
minutes of gradient steps.

What to look at in the integrity results:

- `population/<suite>/<task>/seed_<s>/run.json`: `completed: true`, held-out
  loss falling (measured 3.27 to 1.08 and 4.67 to 1.24 in 24 updates).
- `population/.../update_<n>/metadata.json`: `held_out_sampling` reports, for
  the adapter and the frozen base, the error of open-loop sampled actions on
  held-out demonstration windows and how often samples leave the controller
  bounds or hit the diffusion head's clamp.
- `population/.../update_<n>/behaviour.json`: paired frozen-base versus adapter
  rollouts on identical initial states and action-sampling seeds, with
  controller clipping frequency.

**Expect near-total clipping in the integrity run.** The diffusion loss is 1.0
for a denoiser that predicts zero noise. After 24 updates the loss is about
1.1, so the sampler does not remove noise and samples drift to the head's
clamp of 5: measured 72 to 100 percent of rollout steps were clipped and no
episode succeeded, for the adapters and for the frozen base (which has never
seen LIBERO's action scale). That is the expected behaviour of an untrained
policy and exactly what the clipping report is for.

A longer local check (one task, 400 updates at effective batch 2, i.e. 800
training windows, 1 percent of the pilot's exposure) shows that a run does
learn: held-out loss 3.27 (frozen base) to 0.64, 0.79, and 0.55 at 100, 200,
and 400 updates; open-loop action error on held-out demonstration windows 0.68
to 0.67, 0.30, and 0.24; rollout steps clipped 93, 75, and 64 percent; still
no success in two 120-step episodes. Whether the 5,000-update budget is enough
is what profile 4 decides.

## 4. Run profiles 3 and 4: pilot zoo and budget validation

The pilot manifest names task indices 0, 1, 8 of each suite and seeds 17 and
42: 24 independent runs, 72 adapter samples. Its starting budget is 5,000
optimizer updates at effective batch 16 (micro-batch 8 x accumulation 2; use 2 x 8 on a
6 GB GPU, see section 1), i.e.
80,000 training windows per run, with snapshots at 4,000 / 4,500 / 5,000.

```bash
python scripts/setup/download_libero.py --manifest configs/research/manifests/pilot.yaml
python scripts/reporting/plan_zoo_alignment_jobs.py --manifest configs/research/manifests/pilot.yaml
python scripts/train/run_job_manifest.py research_results/experiments/pilot/jobs/jobs.jsonl \
    --offline --stages gates,evidence,budget_inspection,zoo,zoo_behaviour,zoo_verification
```

On Slurm: `RENDER=mesa-egl jobs/submit_workers.sh small 4 jobs/03_model_zoo.sh`
(four runners, one per small slice; see section 1).

Budget inspection trains four source tasks (`libero_object` and `libero_goal`,
indices 0 and 1) to 10,000 updates, snapshots at 2,000 / 5,000 / 10,000, and
rolls each snapshot out against the frozen base. Inspect, per stage:

- held-out loss in `run.json` (well below 1.0 is necessary, not sufficient);
- `held_out_sampling` in each snapshot's `metadata.json`;
- success rates and `controller_clipping` in each snapshot's `behaviour.json`.

Then freeze the budget. This writes a new manifest file and records why:

```bash
python scripts/reporting/freeze_zoo_budget.py \
    --manifest configs/research/manifests/pilot.yaml \
    --optimizer-updates 5000 --snapshot-updates 4000,4500,5000 \
    --rationale "<what the inspection showed>" \
    --output configs/research/manifests/pilot_frozen.yaml
```

Held-out diagnostics and held-out rollouts refuse to run on an unfrozen
manifest. Once an experiment holds held-out results its manifest can no longer
change; a changed configuration needs a new experiment name. Changing the
budget changes `zoo_sha256`, so adapters trained under the old budget are not
accepted by the new manifest.

Zoo training is resumable. Optimizer moments, the noise counter, and the
dataset cursor live in `zoo_resume/`, apart from the adapter snapshots. A
resume is refused unless the run identity (manifest, budget, dataset hash,
episode split, base, architecture, preprocessing, action convention, adapter
layout, environment profile) matches. With deterministic GPU ops a
killed-and-resumed run was bit-identical to an uninterrupted one on the RTX
4050 (and on CPU in the unit tests). At a snapshot stage the resume state is
written before the snapshot, so a restart never has to recompute a snapshot:
a missing one is written from the exactly restored state, an existing one is
adopted after an exact check, and nothing is overwritten.

`zoo_verification` writes `population/verification.json`. A sample is
*structurally valid* once it reloads and round-trips; it is *behaviourally
evaluated* only once `behaviour.json` exists. Failed and poorly performing runs
stay on disk and are listed, never deleted.

## 5. Run profile 5: paired alignment, diagnostics, rollouts

```bash
python scripts/train/run_job_manifest.py research_results/experiments/<name>/jobs/jobs.jsonl \
    --offline --stages archive,alignment,diagnostics_validation,rollouts_validation,report
```

On Slurm: `RENDER=mesa-egl jobs/submit_workers.sh small 4 jobs/04_alignment.sh`.

Per fold this runs:

1. `build_alignment_archive.py --fold <fold>`: re-checks every adapter and
   evidence hash, rejects held-out-suite tasks, keeps all seeds and snapshots
   of a task in one split, fits evidence standardization and per-entry RMS
   scales on training tasks only, and writes memory-mapped token shards with
   task evidence stored once per task.
2. `train_research_alignment.py --fold <fold> --seed <s> --condition aligned`
   and `--condition reconstruction_only`. The two runs share initialization,
   batch schedule, optimizer exposure, evidence classifiers, and
   reconstruction-based checkpoint selection. Only the contrastive loss
   differs. `metadata.json` stores the fully resolved configuration;
   command-line overrides take precedence over `configs/research/base.yaml`.
3. `evaluate_alignment_pair.py --fold <fold> --seed <s> --scope validation`:
   offline diagnostics on the fold's validation tasks.
4. `run_generated_policy_rollouts.py --fold <fold> --seed <s> --scope
   validation --tier development`: 20 rollouts per task for the frozen base,
   the zoo adapter, the reconstruction and mapper-generated adapters of both
   conditions (before and after shell projection), the nearest-neighbour
   adapter, and the training-mean adapter. Every generated update is projected
   to the zoo's rank before deployment.

After the budget is frozen, re-plan and add
`diagnostics_held_out,rollouts_held_out` (50 rollouts, `--tier final`).

Rollout runs resume per task. Rerunning the same command after an interruption
continues after the last completed task; a finished run is left untouched; an
interrupted run with a different configuration is moved to
`failed_attempts/`, never deleted.

## 6. Run profile 6: full protocol

`configs/research/manifests/full.yaml`: 40 tasks x 3 seeds = 120 runs, 360
adapter samples, four leave-one-suite-out folds, three alignment seeds, source
indices 8 and 9 as validation tasks. Copy the frozen pilot budget into it and
set `frozen: true` (use `freeze_zoo_budget.py`) before launching.

Approximate storage for the full profile: 9.4 GB of adapter snapshots and
8.3 GB of token shards per fold (270 samples), about 33 GB for four folds.

The downstream study (gamma validation, continual learning, OOD adaptation,
component swaps, recovery) is planned on top of the same experiment:

```bash
python scripts/reporting/plan_methodology1_jobs.py --manifest <frozen full manifest>
python scripts/train/run_job_manifest.py \
    research_results/experiments/full/jobs/methodology1.jsonl --offline
python scripts/reporting/audit_methodology1_study.py --manifest <frozen full manifest> \
    --allow-incomplete
```

Its jobs (`m1:*`) depend on the *integrity* of the paired alignment
diagnostics, not on alignment winning. These downstream commands have not been
executed on corrected artifacts; treat the first run of each stage as an
integration check and expect to fix defects there.

## 7. Artifact layout

Everything a corrected experiment writes lives under
`research_results/experiments/<manifest name>/`:

```text
manifest.lock.json                         manifest hash this directory is bound to
gates/preflight/preflight.json             environment, data index, legacy directories
gates/octo_gates.json                      equivalence, gradients, routing, base hash
gates/camera_orientation/<suite>/          measured rollout camera orientation
evidence/<suite>/<task>.npz (+ .manifest.json)
population/<suite>/<task>/seed_<s>/run.json
population/<suite>/<task>/seed_<s>/update_<n>/{adapter.npz,metadata.json,behaviour.json}
population/verification.json
zoo_resume/                                optimizer state, noise counter, data cursor
budget_inspection/                         inspection runs, outside the population
alignment/<fold>/archive/                  token shards, evidence, scales, archive_manifest.json
alignment/<fold>/seed_<s>/<condition>/     alignment checkpoint (8 files)
diagnostics/<scope>/<fold>/seed_<s>/alignment_pair.json
rollouts/<scope>/<fold>/seed_<s>/{manifest.json,episodes.jsonl,policies.json,summary.json}
failed_attempts/                           interrupted runs that were replaced
reports/generated_policy_rollouts.json
jobs/{jobs.jsonl,job_state.json,failure_summary.json,logs/}
HELD_OUT_RESULTS.json                      written with the first held-out result
```

Every stage is safe to rerun. A finished output produced by the identical
request is reported as already complete; an output produced by a different
request is a conflict and is never overwritten.

## 8. Legacy (version-one) artifacts

`research_results/{population,evidence,alignment,preflight}` hold artifacts of
the previous pipeline: post-block adapters whose final-layer vision and
language branches received exactly zero gradient, per-task action
normalization, a single action-summary vector, and a whole-sequence
autoencoder. They are kept as diagnostics.

```bash
python scripts/reporting/inspect_legacy_artifacts.py research_results
```

lists them and states why each is not usable. Corrected commands reject them
with `LegacyArtifactError`. There is no migration: the zoo must be regenerated.
Corrected experiments never write into those directories.

## 9. Reading results

Three outcomes are kept apart:

- **Integrity failure.** A check on the experiment itself failed: unpaired
  checkpoints, a leak, a hash mismatch, a non-finite metric, a changed base.
  The command exits non-zero. Nothing can be concluded.
- **Inconclusive.** Integrity passed and the confidence interval of the paired
  difference includes zero, or fewer than six tasks were evaluated.
- **Supports the first condition / negative finding.** Integrity passed and the
  interval excludes zero in one direction or the other.

A negative finding is a result. It is reported with the same prominence as a
positive one, and no gate anywhere requires alignment to win.

Further rules:

- Metrics are task-balanced: averaged within a task first, then across tasks.
  Uncertainty is a two-level bootstrap over tasks and training runs. Snapshots
  of one run are never counted as independent seeds.
- A normalized metric with a zero denominator is reported as undefined
  (`null`) and counted, not scored.
- Direct weight-to-evidence cosine retrieval is descriptive only: the control
  never trains those coordinates to match, so it cannot be a fair advantage
  test. The comparison that counts for retrieval is evidence-to-mapped-latent
  retrieval, computed for both conditions against task-balanced galleries with
  the chance level stated.
- Rank-matched updates are the primary deployed objects. Unrestricted decoded
  updates are diagnostics; the projection error is reported.
- Validation diagnostics and held-out results are separate files in separate
  directories. A held-out result freezes the experiment's configuration.
- Policy conditions are paired episode by episode: identical initial states and
  identical action-sampling seeds.

## 10. Handing results over

- Run from a clean Git commit. Every artifact records the commit and whether
  the tree was dirty; the downstream audit rejects dirty publication runs.
- Transfer `research_results/experiments/<name>/` whole. Hashes are re-checked
  when archives are built and when checkpoints are loaded.
- Send `gates/preflight/preflight.json`, `gates/octo_gates.json`, the
  environment freeze (`envs/locks/`), `jobs/failure_summary.json`, and the
  reports with any result.
- Do not combine artifacts from different manifests, base hashes, architecture
  hashes, preprocessing hashes, or environment profiles. The archive builder
  and the checkpoint loaders refuse such mixtures.
- State which profile ran on which hardware, and whether that profile had
  passed the integrity run there first.
