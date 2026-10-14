#!/bin/bash
# Model zoo of one experiment manifest: gates, evidence, budget inspection, zoo
# training, paired base-vs-adapter behaviour rollouts, and zoo verification.
#
# Tasks, seeds, optimizer-update budget, and snapshot stages come from the
# manifest (default: the pilot profile). Resumable: resubmit after a walltime
# kill; finished jobs are skipped and interrupted zoo runs continue from their
# saved optimizer state. Data and the Hugging Face cache must already be on
# scratch (download on the login node first; jobs run with HF_HUB_OFFLINE=1).
#
# One submission is one runner on one GPU slice. Several runners share the
# experiment's job manifest through $RESEARCH_ROOT on scratch, so submit one
# per slice you may hold at once (jobs/submit_workers.sh does this). Each
# runner executes WORKERS jobs at once (default 3), at most TRAINING_SLOTS
# (default 1) of them GPU training; the others are mostly rollouts, which are
# CPU-bound because MIG slices render in software.
#
# Submit from the repository root, e.g.
#   RENDER=mesa-egl jobs/submit_workers.sh small 4 jobs/03_model_zoo.sh   # four 1g.24gb runners
#   sbatch --export=ALL,RENDER=mesa-egl jobs/03_model_zoo.sh              # one runner
#   sbatch --export=ALL,RENDER=mesa-egl,MANIFEST=configs/research/manifests/full.yaml jobs/03_model_zoo.sh
#   sbatch --export=ALL,STAGES=gates,evidence,zoo jobs/03_model_zoo.sh     # no simulator
#SBATCH --job-name=m1_model_zoo
#SBATCH --account=research
#SBATCH --partition=gpu_small
#SBATCH --qos=gpu_small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

STAGES=${STAGES:-gates,evidence,budget_inspection,zoo,zoo_behaviour,zoo_verification}

set -euo pipefail
S=/scratch/$USER
source $S/envs/vla/bin/activate
export HF_HOME=$S/hf_cache
export JAX_COMPILATION_CACHE_DIR=$S/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# run_job_manifest.py adds XLA's deterministic GPU ops per job (not for
# alignment training, where they cost ~36x). Do not export them globally.
# Rollout stages need a MuJoCo backend; pick one that 01_render_check reports as working.
if [ -n "${RENDER:-}" ]; then source jobs/render_env.sh; fi

MANIFEST=${MANIFEST:-configs/research/manifests/pilot.yaml}
R=${RESEARCH_ROOT:-$S/research_results}
DATA=${DATA_ROOT:-$S/data/libero}
EXTRA=()
if [ -z "${RENDER:-}" ]; then EXTRA+=(--skip-simulator); fi

echo "== job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "manifest=$MANIFEST research_root=$R stages=$STAGES render=${RENDER:-none (simulator jobs skipped)}"
echo "workers=${WORKERS:-3} training_slots=${TRAINING_SLOTS:-1}"
nvidia-smi -L
df -h $S | tail -1

# The job manifest is regenerated from the experiment manifest every time; job
# ids are stable, so completed work is recognised and skipped.
JOBS=$(python scripts/reporting/plan_zoo_alignment_jobs.py --manifest "$MANIFEST" \
    --research-root "$R" --data-root "$DATA" \
  | python -c "import json,sys; print(json.load(sys.stdin)['output'])")
echo "job manifest: $JOBS"

set +e
python scripts/train/run_job_manifest.py "$JOBS" --offline --stages "$STAGES" \
    --workers "${WORKERS:-3}" --training-slots "${TRAINING_SLOTS:-1}" "${EXTRA[@]}"
STATUS=$?
set -e
echo "== runner exit status $STATUS (0 = nothing left for this runner, 1 = a job failed, 3 = stopped by"
echo "   the time limit: resubmit; see $(dirname "$JOBS")/failure_summary.json)"
echo "== done at $(date)"
exit $STATUS
