#!/bin/bash
# GPU bring-up check: JAX sees the GPU, then Octo zero-adapter equivalence.
# Submit from the repository root: sbatch jobs/00_gpu_check.sh
#SBATCH --job-name=m1_gpu_check
#SBATCH --account=research
#SBATCH --partition=gpu_small
#SBATCH --qos=gpu_small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
S=/scratch/$USER
source $S/envs/vla/bin/activate
export HF_HOME=$S/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export JAX_COMPILATION_CACHE_DIR=$S/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=$S/research_results/preflight
mkdir -p "$OUT"

echo "== job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi

echo "== JAX GPU check"
python - <<'PY'
import time, jax, jax.numpy as jnp
print("jax", jax.__version__, "backend", jax.default_backend())
for d in jax.devices(): print(" device:", d, getattr(d, "device_kind", ""))
assert jax.default_backend() == "gpu", "JAX did not find the GPU"
x = jnp.ones((4096, 4096), jnp.float32)
for precision in ("default", "highest"):
    f = jax.jit(lambda a: jnp.matmul(a, a, precision=precision))
    f(x).block_until_ready()
    t = time.time(); [f(x).block_until_ready() for _ in range(10)]
    print("matmul precision=%s: %.1f TFLOP/s" % (precision, 10 * 2 * 4096**3 / (time.time() - t) / 1e12))
PY

echo "== Octo zero-adapter equivalence (true float32)"
python scripts/eval/verify_official_octo.py --output "$OUT/octo_equivalence.json" \
  | grep -v '"octo_transformer/'

echo "== done at $(date)"
