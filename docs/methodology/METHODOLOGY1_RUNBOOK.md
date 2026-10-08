# Methodology 1 Reproduction Runbook

This is the authoritative execution order for the official Octo-Small study.
The old PyTorch scripts are smoke fixtures and are not publication-eligible.

## 1. Environment and immutable inputs

Use Linux, Python 3.11, a CUDA 12.8+ capable GPU (Blackwell included), and
the pinned Octo port and LIBERO revisions. Install only through
`scripts/setup_research_env.sh`, which performs the ordered `--no-deps` steps
for dlimp, the patched Octo commit, and the editable LIBERO checkout. Freeze a
clean Git commit before publication jobs. Every condition in a comparison must
come from the same environment lock.

```bash
git checkout <frozen-study-commit>
PYTHON=python3.11 envs/scripts/setup_linux.sh .venv-research
source .venv-research/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export M1_ROOT="$PWD/research_results"
export M1_DEV="$M1_ROOT/development"
export M1_PUB="$M1_ROOT/publication"
mkdir -p "$M1_DEV" "$M1_PUB"
```

Each `data/libero/<suite>/manifest.json` must list ten ordered
`{task_index, instruction, file}` items matching `configs/reference_tasks.yaml`.
Never derive task identity from filesystem order.

After the data manifests exist, generate the complete dependency graph used by
collaborators or a cluster scheduler:

```bash
python scripts/reporting/plan_methodology1_jobs.py \
  --research-root "$M1_ROOT" --data-root data/libero \
  --output "$M1_ROOT/methodology1_jobs.jsonl"
```

The JSONL plan contains stable job IDs, argument arrays, dependencies, expected
outputs, and GPU requirements. It is the machine-readable counterpart of the
steps below; generating it does not execute experiments.

## 2. Mandatory gates

```bash
pytest -m "not integration"
python scripts/eval/research_preflight.py --output-dir "$M1_ROOT/preflight"
python scripts/eval/verify_official_octo.py \
  --output "$M1_ROOT/preflight/octo_equivalence.json"
python scripts/eval/run_one_task_learning_gate.py \
  --suite libero_spatial --task-index 0 --seed 17 \
  --output-root "$M1_ROOT/gates/one_task"
python scripts/eval/run_official_continual.py \
  --suite libero_spatial --condition sequential_no_regularization \
  --seed 17 --task-count 2 --output-root "$M1_ROOT/gates/continual"
```

Stop on any non-zero exit. The one-task gate requires deterministic repeated
rollouts and strict improvement over the frozen base. The two-task gate writes
the three observed lower-triangle cells; the future-task cell remains null.
Preflight also writes `environment_lock.json`, hashes the research requirements,
configuration, task order, gamma space, and every indexed dataset, and records
the pinned Octo/LIBERO revisions. Keep this lock with all transferred results.

## 3. Evidence and model zoo

For all 40 `(suite, task-index)` pairs, use the exact data file named by the
suite manifest:

```bash
python scripts/data/extract_research_evidence.py \
  --data-file data/libero/<suite>/<manifest-file> \
  --suite <suite> --task-index <0..9> \
  --output "$M1_ROOT/evidence/<suite>/<suite>_<task-index>.npz"
```

For every task and seed in `{17,42,73}`:

```bash
python scripts/train/train_research_zoo.py \
  --suite <suite> --task-index <0..9> --seed <seed> \
  --evidence "$M1_ROOT/evidence/<suite>/<suite>_<task-index>.npz" \
  --output-root "$M1_ROOT/population"
```

The 120 independent jobs save three late checkpoints each:

```bash
test "$(find "$M1_ROOT/population" -name adapter.npz | wc -l)" -eq 360
python scripts/eval/verify_research_zoo.py "$M1_ROOT/population" \
  --output "$M1_ROOT/population/verification.json"
```

`verify_zoo.py` is intentionally excluded here: it only validates the legacy
PyTorch smoke fixture. The research verifier checks all 360 identities, file
hashes, evidence metadata, frozen-base identity, split labels, and adapter spec.

## 4. Four leave-one-suite-out folds

Repeat for `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`:

```bash
export HELD_OUT=libero_spatial
export FOLD="$M1_ROOT/alignment/$HELD_OUT"
mkdir -p "$FOLD"
python scripts/data/build_alignment_archive.py "$M1_ROOT/population" \
  --held-out-suite "$HELD_OUT" --validation-task-indices 8,9 \
  --output "$FOLD/archive.npz"
python scripts/train/train_research_alignment.py "$FOLD/archive.npz" \
  --output "$FOLD/aligned"
python scripts/train/train_research_alignment.py "$FOLD/archive.npz" \
  --output "$FOLD/reconstruction_only" --contrastive-weight 0
python scripts/eval/run_alignment_advantage_gate.py "$FOLD/archive.npz" \
  --aligned-checkpoint "$FOLD/aligned" \
  --reconstruction-checkpoint "$FOLD/reconstruction_only" \
  --output "$FOLD/alignment_advantage.json"
python scripts/data/build_ood_reference_bank.py "$M1_ROOT/population" "$FOLD/aligned" \
  --held-out-suite "$HELD_OUT" --output "$FOLD/ood_reference_bank.npz"
```

Each archive must contain 270 source-suite samples. The aligned checkpoint must
strictly beat reconstruction-only validation retrieval and mapper MSE.

## 5. Validation-only hyperparameter selection

`configs/research/gamma_candidates.yaml` is the frozen search space. For each
fold, run every candidate × patience × source suite × seed. One job is:

```bash
python scripts/eval/run_gamma_validation.py \
  --held-out-suite <held-out> --source-suite <non-held-out> --seed <17|42|73> \
  --family <proposed|uniform> --gamma-vision <v> --gamma-language <l> \
  --gamma-action <a> --patience <p> --alignment-checkpoint "$FOLD/aligned" \
  --evidence-root "$M1_ROOT/evidence" \
  --output-root "$M1_ROOT/gamma_validation"
```

Freeze the result only after all nine source-suite/seed cells exist for every
candidate:

```bash
python scripts/eval/select_validation_gammas.py \
  "$M1_ROOT/gamma_validation/$HELD_OUT" --held-out-suite "$HELD_OUT" \
  --seeds 17,42,73 --output "$FOLD/gamma_selection.json" \
  --report "$FOLD/gamma_selection.report.json"
```

## 6. Pilot and final continual study

First run one full suite, three seeds, and all conditions with
`--tier development --output-root "$M1_DEV/continual"`. Then use a fresh root
for final 50-rollout jobs. Conditions are:

`sequential_no_regularization`, `replay_10`, `replay_100`,
`uniform_regularization`, `proposed_asymmetric`, `direction_inverted`,
`reconstruction_only_latent`, and `independent_adapter_oracle`.

For every fold × seed × condition:

```bash
python scripts/eval/run_official_continual.py \
  --suite "$HELD_OUT" --seed <17|42|73> --condition <condition> \
  --gamma-selection "$FOLD/gamma_selection.json" \
  --alignment-checkpoint "$FOLD/aligned" \
  --reconstruction-checkpoint "$FOLD/reconstruction_only" \
  --evidence-root "$M1_ROOT/evidence" --tier publication \
  --output-root "$M1_PUB/continual"
```

Expected final total: 96 runs, 55 cells per run, 5,280 evaluation records.

## 7. OOD and mechanistic probes

For every fold and seed:

```bash
python scripts/eval/run_official_ood.py \
  --suite "$HELD_OUT" --seed <17|42|73> \
  --alignment-checkpoint "$FOLD/aligned" \
  --reference-bank "$FOLD/ood_reference_bank.npz" \
  --evidence-root "$M1_ROOT/evidence" --tier publication \
  --output-root "$M1_PUB/ood"

python scripts/eval/run_component_swaps.py \
  "$M1_PUB/continual/ten_task_study/$HELD_OUT/seed_<seed>/proposed_asymmetric" \
  --tier publication --output-root "$M1_PUB/component_swaps"

python scripts/eval/run_recovery_probe.py \
  "$M1_PUB/continual/ten_task_study/$HELD_OUT/seed_<seed>/sequential_no_regularization" \
  --output-root "$M1_PUB/recovery"
```

Recovery is currently defined for shared LoRA-space runs. Unrecovered and
zero-original-peak tasks receive null ratios, never favorable fabricated values.

## 8. Final report

```bash
python scripts/reporting/report_publication_study.py \
  "$M1_PUB/continual/ten_task_study" \
  --output-json "$M1_PUB/reports/continual_study.json" \
  --output-parquet "$M1_PUB/reports/continual_records.parquet"

python scripts/reporting/report_ood_study.py "$M1_PUB/ood" \
  --output-json "$M1_PUB/reports/ood_study.json" \
  --output-parquet "$M1_PUB/reports/ood_records.parquet"

python scripts/reporting/report_component_swaps.py "$M1_PUB/component_swaps" \
  --output-json "$M1_PUB/reports/component_swaps.json" \
  --output-parquet "$M1_PUB/reports/component_swaps.parquet" \
  --output-figure "$M1_PUB/reports/component_drift_scatter.png"

python scripts/reporting/audit_methodology1_study.py "$M1_ROOT" \
  --output "$M1_PUB/reports/completion_audit.json"
```

The reporter rejects missing cells, mixed run identities, inconsistent rollout
budgets, and absent conditions. It uses crossed suite/seed bootstrap intervals.
Offline losses are diagnostic only.
The final audit exits non-zero unless every mandatory gate, fold, run, probe,
record count, report, and provenance-consistency check is present. During an
active cluster run, add `--allow-incomplete` to save a progress snapshot.

## 9. Scheduler, resume, and transfer rules

- Parallelize only independent task/seed/fold jobs. Each GPU worker writes to a
  unique leaf directory.
- Never overwrite a run directory. Preserve failures, then requeue into a new
  root so negative evidence and diagnostics remain auditable.
- Completion requires `manifest.json` with `finished_at`, an empty `failures`
  list, expected record counts, and matching SHA-256 identities.
- Do not combine artifacts from different Git commits, base hashes, dataset
  hashes, adapter specs, or alignment checkpoints.
- Transfer immutable artifacts and verify recorded hashes on the destination.
- A scheduler may dispatch only jobs whose `dependencies` in
  `methodology1_jobs.jsonl` completed successfully. Preserve the JSONL and its
  summary alongside the final results.
