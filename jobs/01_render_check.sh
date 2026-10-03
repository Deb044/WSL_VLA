#!/bin/bash
# Find a MuJoCo rendering backend that works for LIBERO on a MIG slice.
# Submit from the repository root: sbatch jobs/01_render_check.sh
#SBATCH --job-name=m1_render_check
#SBATCH --account=research
#SBATCH --partition=gpu_small
#SBATCH --qos=gpu_small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
S=/scratch/$USER
source $S/envs/vla/bin/activate
export HF_HOME=$S/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=$S/research_results/preflight/render
mkdir -p "$OUT"

echo "== job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
echo "== system GL libraries"
ldconfig -p | grep -E -i 'libEGL|libOSMesa|libGL\.so' || echo "(none found by ldconfig)"
ls /usr/share/glvnd/egl_vendor.d/ 2>/dev/null || echo "(no glvnd EGL vendor files)"
python -c "import OpenGL; print('PyOpenGL', OpenGL.__version__)"

echo "== render probes"
python scripts/check_libero_render.py --output-dir "$OUT"

echo "== done at $(date)"
