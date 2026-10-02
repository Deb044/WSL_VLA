# Implementation Review (2026-10-02): current code vs PLAN.md

## Context
The user asked for a critical read of the codebase against PLAN.md. `docs/IMPLEMENTATION_STATUS.md` marks almost every plan item "Implemented", but that only means code and host tests exist. Nothing has run on WSL/CUDA. This review traces the real code paths to find where the implementation will fail or will not support the paper's claims, even if every unit test passes.

## Overall verdict
The engineering around the method is strong: provenance, fail-closed contracts, leakage checks, the job graph, metrics and audits. The core path is official Octo, then adapters, then packing, then alignment, then the mapper, then refinement. It has blockers in compute feasibility and design that will stop it before any publishable number exists. The biggest risk is not missing features. It is that the weight-space representation is far larger than the plan assumed, and everything downstream inherits that.

## Blockers (will fail or cannot run at the planned scale)

1. **Token count makes the weight autoencoder infeasible.**
   - Source: `bridge.py:91-115` and `packing.py:151-164`.
   - Every Octo-Small block gets three 384×384 residual updates, and each matrix row becomes one 384-wide token. That is 3 × 12 × 384 = 13,824 tokens before the diffusion-head kernels (about 5–6k more).
   - `PackedWeightEncoder` (`models.py:86-101`) runs full self-attention over all ~20k tokens. With batch 4 and 4 heads, one attention map is about 25 GB per layer, far over the ≤8 GB target.
   - The archive is about 360 × 20k × 384 float32 ≈ 11 GB plus masks. `train_research_alignment.py:45` loads it all into RAM.
2. **The latent space has no bottleneck.**
   - The decoder (`models.py:105-118`) is a pointwise MLP that ignores component and layer IDs. The latent is one 128-d vector per row-token, so the "latent adapter" is about 20k × 128 numbers.
   - That is roughly 2.5M dims, larger than the rank-8 factors it replaces (about 220k). It is not a compact task code.
   - Effects: the ridge mapper outputs millions of values, `γ‖Z−Z_prev‖²` is a sum over thousands of tokens, and "latent drift" is not comparable across modalities.
3. **Latent refinement is not jitted and is Python-unrolled.**
   - `refine_latents` (`models.py:484`) calls `jax.value_and_grad` eagerly every step.
   - `unpack_effective_tokens_jax` (`latent.py:68-72`) builds about 20k per-row concatenations in a Python loop.
   - At `train_steps_per_task: 2000`, this will take hours or days per stage, across 10 stages × 8 conditions × 3 seeds × 4 folds.
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
   - The encoder's layer embedding cannot tell them apart. Use a separate offset or an entry/parameter-position embedding (the plan asks for "parameter-position embeddings"; `entry_ids` exist but are not used).
8. **The adapter type differs from what the paper describes.**
   - The "adapters" are post-block residual bottlenecks on the residual stream, not LoRA on attention or MLP weights. That is a valid choice and preserves zero-equivalence, but the paper and docs should describe it accurately.
   - The decoded path also injects full-rank 384×384 kernels, so decoded policies are not rank-8. Say so, or project them to rank r.
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

## What's good (keep)
- Pinned Octo/LIBERO revisions, base-hash checks, and the zero-adapter equivalence test against pre-patch reference outputs.
- Packing by basis-invariant effective update, with full reloadable factors.
- Clean shared-state continual semantics (`train_stage(previous, …)`), with replay sampled from the training split.
- Leave-one-suite-out folds, metric golden tests, audit and job-graph tooling, and the explicitly quarantined smoke path.

## Recommended remediation order
1. **Run the gates first, on real hardware:** preflight, zero-equivalence, then the one-task learning gate (`scripts/run_one_task_learning_gate.py`). If Octo-Small plus these adapters cannot beat the frozen base, nothing downstream matters (#14). Fix image orientation (#12) before this.
2. **Shrink the weight representation (#1, #2):**
   - Tokenize the low-rank structure instead of dense rows, e.g. canonicalize ΔW by SVD into r singular-vector tokens per matrix. That gives about 3 × 12 × 8 + head ≈ 300–400 tokens.
   - Alternatively, chunk rows coarsely and add a per-modality pooled bottleneck (K learned query tokens) so Z is small.
   - Update `packing.py`, `latent.py`, `models.py` and the archive builder together, and keep effective-update invariance.
3. **Make refinement fast (#3):** vectorize unpacking with a precomputed gather/reshape index instead of Python loops, and `jax.jit` the refinement step (mirror `continual.py:117`).
4. **Equalize optimization budgets (#5):** same optimizer family, same accumulation, and the same number of task samples seen for the LoRA and latent conditions. Record both in `EvaluationRecord`.
5. **Normalize the penalty per token and per modality (#6), and fix the layer/entry embeddings (#7).**
6. **Revisit the alignment objective (#9)** with pooled per-modality contrastive loss, and re-check the alignment-advantage gate.
7. **Documentation honesty (#8, #4):** describe the adapters as residual bottlenecks, and pre-register that NN/mean-latent baselines may dominate given 30 training tasks.
8. **Data fixes (#11, #13).**

## Verification
- On WSL/CUDA: `scripts/research_preflight.py`, then `assert_zero_adapter_equivalence`, then the one-task gate, then the 2-task smoke run.
- After step 2: assert the packed token count is under a budget (e.g. ≤ 1024) in `tests/test_research_contracts.py`, and keep round-trip tests in `tests/test_alignment_checkpoint_contract.py`.
- After step 3: add a test that times one jitted refinement step, and keep the "detached decoder must fail" gradient test.
- After step 4: add a test asserting that LoRA and latent stages see an equal number of samples.
- Run the existing host suite with `pytest tests/`.
