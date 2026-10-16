#!/bin/bash
# Submit the render check (jobs/01_render_check.sh) on GPU slices that have
# graphics, to find out whether MuJoCo can render on the GPU (RENDER=egl)
# instead of in software on the CPU (RENDER=mesa-egl, ~0.3 s per step).
#
# Plain MIG slices (1g.24gb, 2g.48gb, 4g.96gb) are compute-only. Graphics-capable
# MIG profiles, where the cluster defines them, carry "gfx" in their name (for
# example 1g.24gb+gfx); this script looks for such GPU types in Slurm's
# configuration and submits one check per (partition, type).
#
# Run on a login node from the repository root:
#   jobs/submit_gfx_render_check.sh                 # discover and submit
#   jobs/submit_gfx_render_check.sh --list          # only show the GPU types Slurm offers
#   GRES=gpu:<type>:1 PARTITION=<partition> jobs/submit_gfx_render_check.sh   # a type the admins named
#
# Each job's log (logs/m1_render_check_<job>.out) ends with "WORKING BACKENDS:";
# "egl" in that list means GPU rendering works on that slice type.
set -euo pipefail

# Every GPU type per partition, e.g. "gpu_small gpu:1g.24gb:4(S:0-1)" -> "gpu_small 1g.24gb".
gpu_types() {
  sinfo -h -o "%P %G" | while read -r partition gres; do
    partition=${partition%\*}
    tr ',' '\n' <<<"$gres" | sed -E 's/\(.*\)//' | awk -F: -v p="$partition" '$1 == "gpu" && NF >= 3 {print p, $2}'
  done | sort -u
}

if [ "${1:-}" = "--list" ]; then
  echo "GPU types per partition (from sinfo):"
  gpu_types
  exit 0
fi

mkdir -p logs
submit() {  # partition gres
  # QOS names follow the partition names on this cluster (gpu_small, gpu_large).
  echo "== partition=$1 gres=$2"
  sbatch --partition="$1" --qos="$1" --gres="$2" --job-name=m1_render_check_gfx jobs/01_render_check.sh
}

if [ -n "${GRES:-}" ]; then
  submit "${PARTITION:?set PARTITION together with GRES}" "$GRES"
  exit 0
fi

found=0
while read -r partition type; do
  [ -n "$type" ] || continue
  case "$type" in
    *gfx*|*GFX*) submit "$partition" "gpu:$type:1"; found=1 ;;
  esac
done < <(gpu_types)

if [ "$found" = 0 ]; then
  echo "No graphics-capable GPU type (name containing 'gfx') is configured in Slurm."
  echo "GPU types this cluster offers:"
  gpu_types | sed 's/^/  /'
  echo "Ask the administrators for a non-MIG GPU or a graphics-enabled MIG profile;"
  echo "if they name one, submit it with GRES=gpu:<type>:1 PARTITION=<partition> $0"
  exit 1
fi
