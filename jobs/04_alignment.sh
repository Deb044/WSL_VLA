#!/bin/bash
# Methodology 1 weight-space alignment: verify zoo, then per held-out fold build the
# archive, train aligned + reconstruction-only autoencoders, and run the advantage gate.
# Resumable: finished archives, checkpoints and gates are skipped; resubmit after a kill.
# Needs the complete 360-adapter population in $R/population.
# Submit from the repository root, e.g.
#   sbatch --export=ALL,FOLDS=libero_10,EPOCHS=2,OUT=alignment_smoke jobs/04_alignment.sh
#   sbatch jobs/04_alignment.sh
#SBATCH --job-name=m1_alignment
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

FOLDS=${FOLDS:-libero_spatial libero_object libero_goal libero_10}
FOLDS=${FOLDS//,/ }
OUT=${OUT:-alignment}
R=$S/research_results
P=$R/population
EXTRA=()
if [ -n "${EPOCHS:-}" ]; then EXTRA+=(--epochs "$EPOCHS"); fi

echo "== job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "folds=[$FOLDS] epochs=${EPOCHS:-200} population=$P out=$R/$OUT"
nvidia-smi -L
df -h $S | tail -1

echo "== verify population $(date)"
python scripts/verify_research_zoo.py "$P" --output "$P/verification.json" > /dev/null
echo "population verified"

FAILED=()
for FOLD in $FOLDS; do
  D=$R/$OUT/$FOLD
  echo "== fold $FOLD $(date)"
  if [ ! -f "$D/archive.manifest.json" ]; then
    python scripts/build_alignment_archive.py "$P" --held-out-suite "$FOLD" \
      --validation-task-indices 8,9 --output "$D/archive.npz" | tail -3
  fi
  for VARIANT in aligned reconstruction_only; do
    if [ -f "$D/$VARIANT/metadata.json" ]; then echo "skip $FOLD/$VARIANT (done)"; continue; fi
    WEIGHT=()
    if [ "$VARIANT" = reconstruction_only ]; then WEIGHT=(--contrastive-weight 0); fi
    echo "-- train $FOLD/$VARIANT $(date)"
    if ! python scripts/train_research_alignment.py "$D/archive.npz" \
        --output "$D/$VARIANT" ${WEIGHT[@]+"${WEIGHT[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}; then
      FAILED+=("$FOLD/$VARIANT")
    fi
  done
  if [ -f "$D/alignment_advantage.json" ]; then echo "skip $FOLD gate (done)"; continue; fi
  if [ -f "$D/aligned/metadata.json" ] && [ -f "$D/reconstruction_only/metadata.json" ]; then
    echo "-- gate $FOLD $(date)"
    set +e
    python scripts/run_alignment_advantage_gate.py "$D/archive.npz" \
      --aligned-checkpoint "$D/aligned" --reconstruction-checkpoint "$D/reconstruction_only" \
      --output "$D/alignment_advantage.json" > "$D/gate.log" 2>&1
    GATE=$?
    set -e
    case $GATE in
      0) echo "GATE $FOLD: PASSED" ;;
      1) echo "GATE $FOLD: NOT PASSED (aligned did not beat reconstruction-only)" ;;
      *) echo "GATE $FOLD: ERROR, see $D/gate.log"; FAILED+=("$FOLD/gate") ;;
    esac
  fi
done

echo "== summary $(date)"
for FOLD in $FOLDS; do
  F=$R/$OUT/$FOLD/alignment_advantage.json
  if [ -f "$F" ]; then
    python -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], 'passed' if d['passed'] else 'NOT passed', json.dumps(d['macro']))" "$F" "$FOLD"
  fi
done
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED RUNS: ${FAILED[*]}"; exit 1; fi
echo "== all folds done"
