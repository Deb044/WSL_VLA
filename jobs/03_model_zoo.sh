#!/bin/bash
# Methodology 1 model zoo: evidence extraction + 3 seeds x 3 late adapters per task.
# Resumable: finished evidence and seeds are skipped; resubmit after a walltime kill.
# Data must already be on scratch (download on the login node first).
# Submit from the repository root, e.g.
#   sbatch --export=ALL,SUITES=libero_spatial,TASKS=1,SEEDS=17,STEPS=50,TAG=smoke jobs/03_model_zoo.sh
#   sbatch jobs/03_model_zoo.sh
#SBATCH --job-name=m1_model_zoo
#SBATCH --account=research
#SBATCH --partition=gpu_large
#SBATCH --qos=gpu_large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4g.96gb:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-gpu=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
S=/scratch/$USER
source $S/envs/vla/bin/activate
export HF_HOME=$S/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export JAX_COMPILATION_CACHE_DIR=$S/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SUITES=${SUITES:-libero_spatial libero_object libero_goal libero_10}
SUITES=${SUITES//,/ }
SEEDS=${SEEDS:-17 42 73}
SEEDS=${SEEDS//,/ }
TASKS=${TASKS:-10}
TAG=${TAG:-population}
EXTRA=()
if [ -n "${STEPS:-}" ]; then EXTRA+=(--steps "$STEPS"); fi
R=$S/research_results

echo "== job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "suites=[$SUITES] seeds=[$SEEDS] tasks=$TASKS steps=${STEPS:-config} out=$R/$TAG"
nvidia-smi -L
df -h $S | tail -1

set +e
python scripts/orchestrate_research_zoo.py \
  --suites $SUITES --seeds $SEEDS --tasks-per-suite "$TASKS" \
  --data-root "$S/data/libero" --evidence-root "$R/evidence" \
  --output-root "$R/$TAG" --skip-download --keep-raw "${EXTRA[@]}"
STATUS=$?
set -e
echo "== orchestrator exit status $STATUS (0 = all runs done, 1 = see FAILED RUNS above)"
echo "== done at $(date)"
exit $STATUS
