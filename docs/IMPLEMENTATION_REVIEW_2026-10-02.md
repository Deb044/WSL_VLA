# Implementation Review (2026-10-02): current code vs PLAN.md

## Context
This is a critical read of the codebase against PLAN.md. `docs/IMPLEMENTATION_STATUS.md` marks almost every plan item "Implemented", but that only means code and host tests exist. Nothing has run on CUDA yet. This review traces the real code paths to find where the implementation will fail or will not support the paper's claims, even if every unit test passes. It then sets out a remediation plan sized for the compute we actually have: the IIT Guwahati GPU cluster (see **Target compute** below).

## Overall verdict
The engineering around the method is strong: provenance, fail-closed contracts, leakage checks, the job graph, metrics and audits. The core path is official Octo, then adapters, then packing, then alignment, then the mapper, then refinement. It has blockers in compute feasibility and design that will stop it before any publishable number exists. The biggest risk is not missing features. It is that the weight-space representation is far larger than the plan assumed, and everything downstream inherits that.

The cluster's 24 GB GPU slice relaxes the plan's ≤8 GB assumption for Octo training, but **it does not remove blockers 1–3**. It also adds new constraints: one GPU slice per user, SLURM scheduling, tight `/home` quota, and likely no GPU rendering on MIG.

## Target compute: IIT Guwahati GPU cluster

| Resource | What we have | Implication |
|---|---|---|
| GPU (`gpu_small`) | One `1g.24gb` MIG slice, 6 CPUs, 24 GB RAM | A quarter of the card's compute. Good for checks, evidence extraction and small jobs. |
| GPU (`gpu_large`) | One `4g.96gb` MIG (the whole card), 24 CPUs, 96 GB RAM; 48 h max walltime; at most 1 running and 2 submitted jobs per user | Job arrays and long dependency chains exceed the submit limit. Use one resumable worker job per stage and resubmit it. |
| Hardware | Both profiles match an RTX PRO 6000 Blackwell (96 GB) | Needs CUDA ≥ 12.8. This is why the stack was ported (below). |
| Scheduler | SLURM (`sbatch`, `squeue`) | Every stage must be a resumable batch job. Interactive notebook-style runs are not viable. |
| `/home/<user>` | 20 GB quota | Code, configs and small results only. The conda env (~15 GB) would nearly fill it, so it goes in `/scratch`. |
| `/scratch/<user>` | 250 GB quota | Datasets, env, caches, zoo, archives, checkpoints and rollout records. |
| Access | `ssh <user>@gpu.iitg.ac.in`; reset password via https://gpu.iitg.ac.in "Forgot Password" before first login | Never commit credentials to this repo or put them in job scripts. |

**Still to confirm with the cluster admins:**
- Maximum walltime per job.
- Maximum concurrent or queued jobs per user.
- CPU cores and RAM per job.
- Whether `/scratch` is purged or backed up.
- Whether outbound internet is allowed from compute nodes, which matters for HF and LIBERO downloads.
- Whether MIG nodes support EGL rendering. NVIDIA's MIG documentation states that graphics APIs are not supported on MIG instances, so expect LIBERO/MuJoCo rendering to need `MUJOCO_GL=osmesa` (CPU) rather than `egl`. `setup_env.sh` currently hard-codes `egl`.

### Software stack: Blackwell port (branch `blackwell-stack`)

The original stack (JAX 0.4.20 on CUDA 11) cannot run on Blackwell. It has been ported and validated on CPU (macOS arm64). It still has to be confirmed on the cluster GPU.

- **New stack:** Python 3.11, JAX 0.7.1 (`jax[cuda12]`, with CUDA ≥ 12.8 libraries from pip), Flax 0.12.0, optax 0.2.8, orbax-checkpoint 0.12.6, TensorFlow 2.20 (file I/O only), Transformers 4.57.6 (< 5, because Octo's T5 encoder needs Flax). The exact pins are in `requirements-research.txt`.
- **Octo port:** a 38-line mechanical patch in `third_party/octo/`:
  - replaces removed `jax.tree_map` / `jax.tree_leaves` / `jax.random.KeyArray`;
  - fixes `process_allgather`, which on new JAX adds a process axis and corrupted the example batch;
  - drops `distrax`.
  
  `scripts/build_patched_octo.sh` rebuilds it as the deterministic commit `a4cc964b…` (verified identical across rebuilds), so preflight still checks an exact installed revision.
- **Validated on the new stack:**
  - The pinned `octo-small-1.5` checkpoint loads; the new orbax reads the old format unchanged.
  - **Zero-adapter equivalence passes** at atol 1e-6 (72 adapter tensors, 11 diffusion kernels). The base hash is unchanged (`6bc3c654…`).
  - A jitted adapter gradient step gives finite, non-zero gradients for every modality.
  - All host tests pass.
  - A real LIBERO environment builds, renders and steps.
- **Install** only through `scripts/setup_research_env.sh`, which handles the ordered `--no-deps` steps.
- **Do not mix environments within a comparison.** No results exist yet, so the whole study moves to this stack. The environment lock in every manifest records it.

## Blockers (will fail or cannot run at the planned scale, including on 24 GB)

1. **Token count makes the weight autoencoder infeasible.**
   - Source: `bridge.py:91-115` and `packing.py:151-164`.
   - Every Octo-Small block gets three 384×384 residual updates, and each matrix row becomes one 384-wide token. That is 3 × 12 × 384 = 13,824 tokens before the diffusion-head kernels (about 5–6k more).
   - `PackedWeightEncoder` (`models.py:86-101`) runs full self-attention over all ~20k tokens. One attention map at batch 1 with 4 heads is about 6.4 GB per layer in fp32. Three layers plus their backward pass will not fit in 24 GB, and batch 4 would need about 25 GB per layer.
   - The archive is about 360 × 20k × 384 float32 ≈ 11 GB plus masks. `train_research_alignment.py:45` loads it all into RAM.
2. **The latent space has no bottleneck.**
   - The decoder (`models.py:105-118`) is a pointwise MLP that ignores component and layer IDs. The latent is one 128-d vector per row-token, so the "latent adapter" is about 20k × 128 numbers.
   - That is roughly 2.5M dims, larger than the rank-8 factors it replaces (about 220k). It is not a compact task code.
   - Effects: the ridge mapper outputs millions of values, `γ‖Z−Z_prev‖²` is a sum over thousands of tokens, and "latent drift" is not comparable across modalities.
3. **Latent refinement is not jitted and is Python-unrolled.**
   - `refine_latents` (`models.py:484`) calls `jax.value_and_grad` eagerly every step.
   - `unpack_effective_tokens_jax` (`latent.py:68-72`) builds about 20k per-row concatenations in a Python loop.
   - At `train_steps_per_task: 2000`, a single stage would likely exceed the cluster's per-job walltime. A bigger GPU does not fix this, because the bottleneck is Python dispatch.
4. **The mapper is fit on only 30 distinct inputs.**
   - Evidence is per task, so all 9 samples per task (3 seeds × 3 checkpoints) share the same input. The ridge fit in `mapping.py` / `train_research_alignment.py:374-397` has 30 unique rows mapping to about 2.5M outputs.
   - Predictions for unseen tasks are therefore just a linear mix of 30 training latents. The nearest-neighbour and mean-latent baselines may be very hard to beat, and that needs to be expected and reported.

## Scientific-validity issues (results would be contestable)

5. **Unfair comparison between LoRA and latent conditions.**
   - LoRA stages run `steps` *update* steps × 8 accumulation with AdamW (`continual.py:111-154`).
   - Latent stages run `steps` *micro* steps with plain SGD and no accumulation (`models.py:484-492`).
   - So LoRA gets 8× more data and a better optimizer. This breaks the plan's "matched-step" requirement and biases results against the proposed method.
6. **Gamma scales are not comparable.**
   - The penalty sums over all tokens per modality, and action has the most tokens because it includes the diffusion head.
   - So "uniform" γ is not uniform per modality, and γ_act < γ_vis may simply cancel out the token-count imbalance. The penalty should be normalized per token (mean, not sum), or the asymmetry claim is confounded.
7. **Layer-ID collision in action tokens.**
   - Diffusion-head kernels use `layer=head_index` (`bridge.py:380-383`), which reuses IDs 0..k that transformer action layers already use.
   - The encoder's layer embedding cannot tell them apart. Use a separate offset or an entry/parameter-position embedding (PLAN.md asks for "parameter-position embeddings"; `entry_ids` exist but are not used).
8. **The adapter type differs from what the paper describes.**
   - The transformer "adapters" are post-block linear residual bottlenecks on the residual stream (`x += mask_m · 2·(x·down)·up`), not LoRA on attention or MLP weights. Only the diffusion head uses true LoRA.
   - That is a valid choice and preserves zero-equivalence, but the paper and docs should describe it accurately.
   - Decoded policies also inject full-rank 384×384 kernels, so they are not rank-8. Say so, or project them to rank r.
9. **Contrastive alignment is token-level.**
   - `multi_positive_info_nce` asks every one of thousands of row-tokens to identify the task. That is a strong and odd inductive bias that is likely to conflict with reconstruction.
   - WeightCLIP-style alignment usually pools per modality before contrasting. The current reverse direction already pools, so consider pooling in both directions.
10. **Validation tasks are reused.**
    - Task indices 8 and 9 of the source suites select the mapper ridge α, alignment early stopping, *and* gamma/patience.
    - This is legal (the test suite stays untouched), but the same 6 tasks drive every selection, so validation estimates will be optimistic and noisy.

## Data and pipeline risks (from the latest commit, 77d47b4)

11. **Proprio alias mismatch.**
    - `PROPRIO_ALIASES` maps `robot0_eef_quat` to `ee_ori`/`ee_states`. These are Euler (3-d) or pos+ori (6-d) values, not a quaternion (4-d), so data shapes can change silently.
    - Config now uses the HDF5 names (`ee_pos`, `ee_ori`, `gripper_states`). Live LIBERO env observations use `robot0_*`, so `libero_rollout.py:284-291` will raise if the Octo example batch ever contains `proprio`.
    - Octo-Small-1.5 has no proprio tokenizer, so proprio is effectively unused. Either drop it or add the alias handling on the rollout side as well.
12. **Image orientation.** LIBERO HDF5 and env frames are 180°-flipped, and neither path flips them. Training and rollout are consistent with each other, but both are off-distribution for Octo's pretrained tokenizer and for the visual evidence features. Most Octo/OpenVLA LIBERO fine-tunes flip them.
13. **`allow_missing=True` in `train_research_zoo.py`** weakens the fail-closed manifest. It is fine for partial downloads, but the zoo verifier must still reject incomplete populations.
14. **Will Octo-Small learn at all?** Rank-8 adapters only, 2000 updates at effective batch 16, frozen tokenizers. Single-task LIBERO success may stay near zero. The one-task learning gate is exactly the right first experiment; run it before investing in anything else.

## Findings from the Blackwell port

20. **The last layer's vision and language adapters are dead parameters.** Adapters run after each block. After block 11, vision and language tokens feed nothing downstream; only readout tokens reach the action head. So `adapter_vision_11` and `adapter_language_11` always receive exactly zero gradient (measured). They add 2 × 384 always-zero rows to every packed sample. This is structural, not caused by the port. Either drop them from training and packing, or apply adapters before each block instead.
21. **LIBERO was never installable from `requirements-research.txt`.** LIBERO's `setup.py` declares no dependencies (robosuite, bddl, mujoco and torch were never installed). Its top-level package also lacks `__init__.py`, so `pip install libero @ git+…` installs nothing importable. Editable installs then drop the commit from pip metadata, so the preflight revision check could not have passed either. Fixed: the setup script installs the simulator dependencies and an editable checkout, and the provenance check now reads the commit of a clean editable git checkout.
22. **Proprio key mismatch confirmed (#11).** The live LIBERO environment exposes `robot0_eef_pos`, `robot0_eef_quat` and `robot0_gripper_qpos`, not the `ee_pos` / `ee_ori` / `gripper_states` HDF5 names now in `configs/research/base.yaml`.
23. **Speed reference (CPU, not representative of the GPU):** one jitted adapter step at batch 2 takes about 0.6 s after about 15 s of compilation. LIBERO steps at about 0.2 s per step with rendering on a laptop.

## Cluster-specific gaps in the current code

15. **No SLURM integration.** `scripts/plan_methodology1_jobs.py` builds a dependency graph but nothing emits or submits `sbatch` scripts, and nothing resumes after a walltime kill. Long scripts (zoo training, continual runs) must checkpoint per stage and skip completed stages on restart. `orchestrate_research_zoo.py` is the closest existing pattern.
16. **The ≤8 GB profile is hard-coded.** Several scripts refuse to run unless `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and they keep micro-batch 2 × accumulation 8. That still works on 24 GB, but it wastes the slice. Add a `24gb` profile with a larger micro-batch (to be measured; roughly 8–16) and less accumulation.
17. **Rendering backend.** `setup_env.sh` and the runbook assume EGL. If MIG rejects EGL, rollouts must fall back to OSMesa on CPU. That makes rollouts CPU-bound, so per-job CPU count matters and the CPU request must be sized accordingly.
18. **Rollout budget will dominate the GPU-hours.**
    - Each continual run evaluates 1+2+…+10 = 55 task-stages.
    - At 50 rollouts per task-stage, that is 2,750 episodes per run. The full grid is 4 folds × 3 seeds × 8 conditions = 96 runs, about 264k episodes.
    - At an *illustrative* 15 s per episode, that is about 1,100 hours of rollouts alone. The real figure must be measured in the one-task gate.
    - With one 24 GB slice per account, this needs more than one account in parallel, and 50 rollouts should be reserved for the final frozen configurations only (see step 6).
19. **Storage layout is unspecified.** Nothing directs envs, HF and JAX caches, datasets or outputs to `/scratch`. With the current packed format, per-fold alignment archives (~11–14 GB each × 4 folds) plus the zoo and datasets (~35 GB) fit in 250 GB but leave little headroom. After the token reduction in step 3, storage stops being a concern.

## What's good (keep)
- Pinned Octo/LIBERO revisions, base-hash checks, and the zero-adapter equivalence test against pre-patch reference outputs.
- Packing by basis-invariant effective update, with full reloadable factors.
- Clean shared-state continual semantics (`train_stage(previous, …)`), with replay sampled from the training split.
- Leave-one-suite-out folds, metric golden tests, audit and job-graph tooling, and the explicitly quarantined smoke path.

## Remediation plan (ordered for the IITG cluster)

1. **Cluster bring-up (no research code changes).**
   - Create the conda env under `/scratch/<user>/envs`.
   - Set `HF_HOME`, `XDG_CACHE_HOME` and `JAX_COMPILATION_CACHE_DIR` under `/scratch`.
   - Download LIBERO and the pinned Octo checkpoint to `/scratch/<user>/data`. If compute nodes have no internet, do this from the login node.
   - Keep the git checkout in `/home` and symlink `data/` and `research_results/` to `/scratch`.
   - Run a one-line MuJoCo render test under both `MUJOCO_GL=egl` and `MUJOCO_GL=osmesa` inside an `sbatch` job, and record which one works.
   - Build the environment on the login node with `scripts/setup_research_env.sh` (Blackwell stack, branch `blackwell-stack`).
   - In a `gpu_small` job, confirm `jax.default_backend() == "gpu"` and that a matmul runs. Then run preflight and `verify_official_octo.py` on the GPU.
2. **SLURM job layer (#15–#17).**
   - Add `scripts/slurm/` templates (`--gres` for the MIG slice, `--time`, `--cpus-per-task`, `--mem`) and a submitter that walks the existing job graph from `plan_methodology1_jobs.py` using `--dependency=afterok`.
   - Make every long script resumable at stage granularity.
   - Add a `24gb` resource profile alongside the ≤8 GB one, and choose the rendering backend from an environment variable.
3. **Run the gates first, on the cluster:** preflight, zero-equivalence, then the one-task learning gate (`scripts/run_one_task_learning_gate.py`).
   - If Octo-Small plus these adapters cannot beat the frozen base, nothing downstream matters (#14).
   - Fix image orientation (#12) before this.
   - **Measure seconds per training step and seconds per rollout episode here.** These two numbers set the real GPU-hour budget for #18.
4. **Shrink the weight representation (#1, #2).**
   - Tokenize the low-rank structure instead of dense rows, e.g. canonicalize ΔW by SVD into r singular-vector tokens per matrix. That gives about 3 × 12 × 8 + head ≈ 300–400 tokens.
   - Alternatively, chunk rows coarsely and add a per-modality pooled bottleneck (K learned query tokens) so Z is small.
   - Update `packing.py`, `latent.py`, `models.py` and the archive builder together, and keep effective-update invariance.
   - At that size the alignment model trains comfortably in one 24 GB slice.
5. **Make refinement fast and fair (#3, #5, #6, #7).**
   - Vectorize unpacking with a precomputed gather/reshape index instead of Python loops, and `jax.jit` the refinement step (mirror `continual.py:117`).
   - Use the same optimizer family, accumulation and number of task samples seen for the LoRA and latent conditions, and record both in `EvaluationRecord`.
   - Normalize the penalty per token and per modality, and fix the layer/entry embeddings.
6. **Budget the rollouts (#18).**
   - Use 20 rollouts per task-stage for the pilot and gamma validation.
   - Use 50 only for the frozen final conditions.
   - Run the one-suite pilot (1 fold × 3 seeds × 8 conditions) before committing to all 4 folds.
   - Split the 96 continual runs across team accounts by fold.
7. **Revisit the alignment objective (#9)** with pooled per-modality contrastive loss, and re-check the alignment-advantage gate.
8. **Documentation and data fixes (#8, #4, #11, #13).**
   - Describe the adapters as residual bottlenecks.
   - Pre-register that NN/mean-latent baselines may dominate given 30 training tasks.
   - Fix proprio handling and the partial-download path.
   - Replace the stale 7B estimates in `docs/compute.md` with numbers measured in step 3.

## Verification
- **On the cluster, via `sbatch`:**
  - Render test.
  - `scripts/research_preflight.py`, then `assert_zero_adapter_equivalence`, then the one-task gate, then the 2-task smoke run.
  - Check that the manifests record measured VRAM, wall time and episode time.
- **Resumability:** kill a continual job mid-run with `scancel`, resubmit it, and confirm that completed stages are skipped and the final records match an uninterrupted run.
- **After step 4:** assert the packed token count is under a budget (e.g. ≤ 1024) in `tests/test_research_contracts.py`, and keep round-trip tests in `tests/test_alignment_checkpoint_contract.py`.
- **After step 5:**
  - Add a test that times one jitted refinement step, and keep the "detached decoder must fail" gradient test.
  - Add a test asserting that LoRA and latent stages see an equal number of samples.
- **Host suite:** `pytest tests/`.
