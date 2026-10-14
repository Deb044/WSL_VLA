#!/bin/bash
# Submit several runners of one stage script that share the experiment's job
# manifest, one per GPU slice.
#
#   jobs/submit_workers.sh <small|medium|large> <count> <stage script>
#
#   RENDER=mesa-egl jobs/submit_workers.sh small 4 jobs/03_model_zoo.sh    # recommended
#   RENDER=mesa-egl jobs/submit_workers.sh medium 2 jobs/03_model_zoo.sh
#   RENDER=mesa-egl jobs/submit_workers.sh large 1 jobs/04_alignment.sh
#
# Variables in the calling shell (RENDER, MANIFEST, STAGES, RESEARCH_ROOT,
# DATA_ROOT, WORKERS, TRAINING_SLOTS) reach every runner. All runners of one
# experiment must use the same MANIFEST, STAGES, and RESEARCH_ROOT. TIME
# (e.g. TIME=12:00:00) replaces the stage script's time limit; it must not
# exceed the partition's (`sinfo -p gpu_small -o "%P %l"`).
#
# Slice sizes are the MIG profiles of one RTX PRO 6000 Blackwell. The small and
# large settings match the cluster's gpu_small and gpu_large partitions; the
# medium ones (partition gpu_medium, 12 CPUs, 48 GB) are assumed by proportion:
# check them with `sinfo -o "%P %l %c %m %G"` before use. Per runner, WORKERS
# jobs run at once, TRAINING_SLOTS of them GPU training (each training job
# needs about 5 GB of host memory, like a rollout job).
#
# Resubmit the same command after a time limit: finished jobs are skipped,
# interrupted ones resume, and a killed runner's jobs are taken over by the
# others after 15 minutes without its heartbeat.
set -euo pipefail

SLICE=${1:?usage: submit_workers.sh <small|medium|large> <count> <stage script>}
COUNT=${2:?usage: submit_workers.sh <small|medium|large> <count> <stage script>}
SCRIPT=${3:?usage: submit_workers.sh <small|medium|large> <count> <stage script>}
case "$SLICE" in
  small)  RESOURCES=(--partition=gpu_small --qos=gpu_small --gres=gpu:1g.24gb:1 --cpus-per-task=6 --mem=24G)
          DEFAULT_WORKERS=3; DEFAULT_TRAINING_SLOTS=1 ;;
  medium) RESOURCES=(--partition=gpu_medium --qos=gpu_medium --gres=gpu:2g.48gb:1 --cpus-per-task=12 --mem=48G)
          DEFAULT_WORKERS=6; DEFAULT_TRAINING_SLOTS=2 ;;
  large)  RESOURCES=(--partition=gpu_large --qos=gpu_large --gres=gpu:4g.96gb:1 --cpus-per-task=24 --mem=96G)
          DEFAULT_WORKERS=12; DEFAULT_TRAINING_SLOTS=4 ;;
  *) echo "unknown slice '$SLICE' (expected small, medium, or large)" >&2; exit 2 ;;
esac
[ -f "$SCRIPT" ] || { echo "no such job script: $SCRIPT" >&2; exit 2; }

if [ -n "${TIME:-}" ]; then RESOURCES+=(--time="$TIME"); fi
mkdir -p logs  # the scripts' --output directory must exist at submission
for _ in $(seq 1 "$COUNT"); do
  sbatch "${RESOURCES[@]}" \
    --export="ALL,WORKERS=${WORKERS:-$DEFAULT_WORKERS},TRAINING_SLOTS=${TRAINING_SLOTS:-$DEFAULT_TRAINING_SLOTS}" \
    "$SCRIPT"
done
