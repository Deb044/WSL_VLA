# WSL_VLA: Weight-Space Alignment for Continual Robot Learning

Research code for building a zoo of Octo-Small adapters on LIBERO, aligning
their weights with task evidence in a shared latent space, and evaluating
reconstructed and evidence-generated policies in simulation.

The pipeline was repaired in October 2026 (see `../PLAN.md`). The zoo must be
regenerated: adapters produced before the repair used a placement in which the
final-layer vision and language branches could not learn, and they are kept
only as legacy diagnostics.

```text
wsl_vla/
├── octo/          # pinned Octo loading, pre-block modality adapters, zoo trainer
├── adapters/      # adapter schema, token packing, rank-matched deployment
├── alignment/     # evidence, archives, windowed autoencoder, mapping, diagnostics
├── data/          # LIBERO loading, shared preprocessing, controller actions
├── evaluation/    # paired rollouts, generated policies, clustered statistics
├── experiments/   # experiment manifests, job plans, provenance
└── smoke/         # legacy PyTorch fixtures, never used for results
```

## Status in one paragraph

The infrastructure runs end to end on the real checkpoint, real LIBERO
demonstrations, and the real simulator at the smallest (integrity) scale, on a
6 GB laptop GPU with the `legacy-gpu` environment. The pilot and full
experiments have not been run, the `blackwell` environment has not run on
Blackwell hardware (its exact package versions pass the unit suite on CPU),
and nothing here shows that alignment helps.
`docs/methodology/IMPLEMENTATION_STATUS.md` has the details.

## Scientific scope

The study asks three questions:

- can task evidence organize modality-factorized adapter updates in an aligned
  weight latent space;
- do mapped latents improve held-out task initialization and matched-step
  adaptation;
- does stronger visual/language than action regularization improve retention
  in a shared sequential policy state.

Each has an honest negative answer available. No gate requires a method to
win; results are recorded as supporting, inconclusive, or negative.
Large-VLA generalization is a later phase and is not implied by Octo-Small
results.

## Environments

| Profile | Stack | Status |
| --- | --- | --- |
| `blackwell` | Python 3.11, JAX 0.7.1 (`jax[cuda12]`), Flax 0.12, patched Octo | unit suite passes with these versions on CPU; not run on Blackwell hardware |
| `legacy-gpu` | Python 3.10, JAX 0.4.20 (CUDA 11 wheels), Flax 0.7.5, upstream Octo | verified on an RTX 4050 |
| `legacy-smoke` | PyTorch smoke scripts | not a research environment |

```bash
envs/scripts/setup_research.sh --profile <blackwell|legacy-gpu> <venv-directory>
source <venv-directory>/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

The job runner adds XLA's deterministic GPU ops per stage (zoo training is then
bit-reproducible; alignment training runs without them, see the runbook).

The pinned inputs are `rail-berkeley/octo-small-1.5` at revision
`dc9aa3019f764726c770814b27e4ab0fc6e32a58`, Octo source
`241fb3514b7c40957a86d869fecb7c7fc353f540` (plus `third_party/octo` on
Blackwell), and LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`.

## Running

Everything is driven by an experiment manifest and a resumable job runner:

```bash
pytest -m "not integration" -q
python scripts/setup/download_libero.py --manifest configs/research/manifests/integrity.yaml
python scripts/reporting/plan_zoo_alignment_jobs.py --manifest configs/research/manifests/integrity.yaml
python scripts/train/run_job_manifest.py research_results/experiments/integrity/jobs/jobs.jsonl --offline
```

Several runners may execute one job manifest at once, sharing its state
directory. On the cluster submit one per GPU slice, e.g.
`RENDER=mesa-egl jobs/submit_workers.sh small 4 jobs/03_model_zoo.sh`; the
runbook explains the choice of slices.

Manifests ship for three profiles:

| Manifest | Tasks x seeds x snapshots | Purpose |
| --- | --- | --- |
| `integrity.yaml` | 2 x 1 x 3 = 6 samples | proves the pipeline executes; not evidence |
| `pilot.yaml` | 12 x 2 x 3 = 72 samples | engineering pilot and budget validation; not evidence |
| `full.yaml` | 40 x 3 x 3 = 360 samples | four leave-one-suite-out folds, three alignment seeds |

`docs/methodology/METHODOLOGY1_RUNBOOK.md` gives the full ordered procedure,
expected artifacts, measured resource use, and how to read results.
`docs/methodology/RESEARCH_PIPELINE.md` documents the model and data contracts.

## Tests

```bash
pytest -m "not integration" -q                                   # unit and contract tests
pytest tests/integration/test_real_pipeline.py -m integration    # real Octo + LIBERO, ~40 min on a laptop GPU
```

The second command drives the actual command-line pipeline on a reduced
integrity manifest: it trains, snapshots, reloads, reconstructs, maps, and
rolls out policies in the simulator. It needs a GPU, the cached checkpoint, the
integrity demonstrations, and the simulator. Missing prerequisites are named
and the test is skipped; set `WSL_VLA_REQUIRE_INTEGRATION=1` to make that a
failure. A skipped test is not a passed test, and the session summary says so.

Tests that need Octo run on a toy model assembled from Octo's own classes
(`tests/octo_tiny.py`), so adapter placement, routing, gradients, equivalence,
snapshots, and exact resume are checked without the 547 MB checkpoint.

## What changed in the repair

- Adapters act on the **input** of every transformer block and are routed by an
  explicit token table; unknown token groups are rejected.
- One fixed controller-action convention with identity normalization; targets
  are validated, deployment clipping is measured.
- One preprocessing path for training, evidence, and rollouts, with measured
  camera orientation.
- Zoo training with explicit update counts, exact resume, held-out losses,
  per-component norms, an open-loop sampling check, and paired rollouts.
- Per-episode action-chunk evidence, training-only standardization, manifest-
  driven counts and folds, hash re-checks, memory-mapped archives.
- A windowed transformer autoencoder with per-entry RMS scaling, a
  bidirectional multi-positive contrastive loss, and a control that differs
  only in that loss.
- Task-balanced ridge mappers, training-only shells, rank-matched deployment.
- Evaluation that separates integrity failures from inconclusive and negative
  findings, with task-balanced metrics and clustered uncertainty.

## Legacy artifacts

`research_results/{population,evidence,alignment,preflight}` hold version-one
artifacts. Corrected commands reject them; inspect them with
`python scripts/reporting/inspect_legacy_artifacts.py research_results`.
Corrected experiments write only under `research_results/experiments/<name>/`.
