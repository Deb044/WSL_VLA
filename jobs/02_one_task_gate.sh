#!/bin/bash
# One-task learning gate: frozen Octo vs. rank-8 adapters on one LIBERO task.
# Submit from the repository root after 01_render_check picks a backend, e.g.
#   sbatch --export=ALL,RENDER=osmesa,STEPS=200,ROLLOUTS=5 jobs/02_one_task_gate.sh
#   sbatch --export=ALL,RENDER=osmesa jobs/02_one_task_gate.sh
#SBATCH --job-name=m1_one_task_gate
#SBATCH --account=research
#SBATCH --partition=gpu_small
#SBATCH --qos=gpu_small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
S=/scratch/$USER
source $S/envs/vla/bin/activate
export HF_HOME=$S/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export JAX_COMPILATION_CACHE_DIR=$S/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=${RENDER:?set RENDER=egl or RENDER=osmesa from 01_render_check}
export PYOPENGL_PLATFORM=$MUJOCO_GL

SUITE=${SUITE:-libero_spatial}
TASK=${TASK:-0}
SEED=${SEED:-17}
EXTRA=()
TAG=full
if [ -n "${STEPS:-}" ]; then EXTRA+=(--steps "$STEPS"); TAG=s${STEPS}; fi
if [ -n "${ROLLOUTS:-}" ]; then EXTRA+=(--rollouts "$ROLLOUTS"); TAG=${TAG}_r${ROLLOUTS}; fi
OUT=$S/research_results/gates/one_task/${TAG}_job${SLURM_JOB_ID}

echo "== job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "MUJOCO_GL=$MUJOCO_GL suite=$SUITE task=$TASK seed=$SEED ${EXTRA[*]:-} out=$OUT"
nvidia-smi -L

set +e
python scripts/eval/run_one_task_learning_gate.py \
  --suite "$SUITE" --task-index "$TASK" --seed "$SEED" \
  --data-root "$S/data/libero" --output-root "$OUT" "${EXTRA[@]}"
STATUS=$?
set -e
echo "== gate exit status $STATUS (0 = passed, 1 = ran but did not pass, other = error)"
echo "== done at $(date)"
exit $STATUS
