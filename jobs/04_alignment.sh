#!/bin/bash
# Weight-space alignment of one experiment manifest: per fold, build the
# leakage-checked archive, train the paired aligned and reconstruction-only
# models, and evaluate them with offline diagnostics and generated-policy
# rollouts on the validation tasks.
#
# Held-out stages (diagnostics_held_out, rollouts_held_out) are planned only
# once the zoo budget is frozen (scripts/reporting/freeze_zoo_budget.py); add
# them to STAGES then. There is no pass/fail "advantage gate": the diagnostics
# job fails only on an integrity violation, and every comparison is recorded as
# supports / inconclusive / negative.
#
# Needs the verified zoo of the same manifest (jobs/03_model_zoo.sh). Like
# that script, one submission is one runner on one GPU slice; runners share
# the job manifest, so submit one per slice (jobs/submit_workers.sh).
# Submit from the repository root, e.g.
#   RENDER=mesa-egl jobs/submit_workers.sh small 4 jobs/04_alignment.sh
#   sbatch --export=ALL,RENDER=mesa-egl jobs/04_alignment.sh
#   STAGES=archive,alignment,diagnostics_validation sbatch jobs/04_alignment.sh
# Set STAGES in the environment, not in --export: sbatch splits --export at commas.
#SBATCH --job-name=m1_alignment
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

STAGES=${STAGES:-archive,alignment,diagnostics_validation,rollouts_validation,report}

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
