#!/usr/bin/env bash
# Create one of the two locked research environments.
#
# Usage:
#   envs/scripts/setup_research.sh --profile <blackwell|legacy-gpu> <venv-directory>
#
#   blackwell   Python 3.11, JAX 0.7.1 from `jax[cuda12]` (NVIDIA CUDA 12 pip
#               wheels, CUDA >= 12.8 for sm_120), Flax 0.12, Octo upstream
#               241fb35 + third_party/octo patches.
#               UNVERIFIED for the corrected pipeline until the integrity run
#               profile passes on Blackwell hardware.
#   legacy-gpu  Python 3.10, JAX 0.4.20 with CUDA 11 pip wheels (the path JAX
#               supported for that release and that Octo documents), Flax
#               0.7.5, unmodified Octo 241fb35. Verified on an RTX 4050 (sm_89).
#
# Choose the interpreter with PYTHON=/path/to/python3.x (it must match the
# profile's Python version). On a cluster with a small home quota, put the
# environment on scratch and run this on a node with network access:
#   conda create -y -p /scratch/$USER/envs/py311 python=3.11
#   PYTHON=/scratch/$USER/envs/py311/bin/python \
#     envs/scripts/setup_research.sh --profile blackwell /scratch/$USER/envs/vla
#
# The script ends by asking the repository which locked profile the new
# environment matches; anything else is an error.
set -euo pipefail

usage() { sed -n '2,6p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; }
PROFILE=""
VENV=""
while [ $# -gt 0 ]; do
    case "$1" in
        --profile) PROFILE="${2:-}"; shift 2 ;;
        --profile=*) PROFILE="${1#--profile=}"; shift ;;
        -h|--help) usage ;;
        *) [ -z "$VENV" ] || usage; VENV="$1"; shift ;;
    esac
done
[ -n "$PROFILE" ] && [ -n "$VENV" ] || usage

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQUIREMENTS="$REPO_ROOT/envs/requirements"
DLIMP_REVISION="5edaa4691567873d495633f2708982b42edf1972"
LIBERO_REVISION="8f1084e3132a39270c3a13ebe37270a43ece2a01"
OCTO_PATCHED_REVISION="a4cc964b7e77f8d8b19f533a0dfa95d653501ab7"

case "$PROFILE" in
    blackwell)  PYTHON_VERSION="3.11"; PYTHON="${PYTHON:-python3.11}" ;;
    legacy-gpu) PYTHON_VERSION="3.10"; PYTHON="${PYTHON:-python3.10}" ;;
    *) echo "unknown profile '$PROFILE' (expected blackwell or legacy-gpu)" >&2; exit 2 ;;
esac
if [ "$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" != "$PYTHON_VERSION" ]; then
    echo "profile $PROFILE requires Python $PYTHON_VERSION (got $("$PYTHON" --version))" >&2
    exit 1
fi

cd "$REPO_ROOT"
"$PYTHON" -m venv "$VENV"
VENV="$(cd "$VENV" && pwd)"
PIP=("$VENV/bin/python" -m pip --default-timeout 180 --retries 10)
"${PIP[@]}" install --quiet --upgrade pip

if [ "$PROFILE" = "blackwell" ]; then
    echo "== Pinned packages (resolved; envs/requirements/research-blackwell.txt)"
    "${PIP[@]}" install -r "$REQUIREMENTS/research-blackwell.txt"

    echo "== dlimp (its metadata pins TensorFlow 2.15; the code runs on 2.20)"
    "${PIP[@]}" install --no-deps "dlimp @ git+https://github.com/kvablack/dlimp@$DLIMP_REVISION"

    echo "== Octo: upstream 241fb35 + third_party/octo patches, deterministic commit"
    "$REPO_ROOT/scripts/setup/build_patched_octo.sh" "$VENV/src/octo" >/dev/null
    "${PIP[@]}" install --no-deps "octo @ git+file://$VENV/src/octo@$OCTO_PATCHED_REVISION"
else
    echo "== Exact lock, no dependency resolution (envs/requirements/research-legacy-gpu.txt)"
    "${PIP[@]}" install --no-deps -r "$REQUIREMENTS/research-legacy-gpu.txt"

    echo "== This repository (editable)"
    "${PIP[@]}" install --no-deps -e "$REPO_ROOT"
fi

echo "== LIBERO: editable checkout at $LIBERO_REVISION (its top-level package has no __init__.py)"
"${PIP[@]}" install --no-deps --src "$VENV/src" --config-settings editable_mode=compat \
    -e "git+https://github.com/Lifelong-Robot-Learning/LIBERO.git@$LIBERO_REVISION#egg=libero"

echo "== LIBERO config inside the environment (first import prompts on stdin)"
mkdir -p "$VENV/libero_config"
if ! grep -q "LIBERO_CONFIG_PATH" "$VENV/bin/activate"; then
    {
        echo ""
        echo "# Added by envs/scripts/setup_research.sh"
        echo "export LIBERO_CONFIG_PATH=\"$VENV/libero_config\""
        echo "export MUJOCO_GL=\"\${MUJOCO_GL:-egl}\""
        echo "export XLA_PYTHON_CLIENT_PREALLOCATE=\"\${XLA_PYTHON_CLIENT_PREALLOCATE:-false}\""
    } >> "$VENV/bin/activate"
fi
printf 'N\n' | LIBERO_CONFIG_PATH="$VENV/libero_config" MUJOCO_GL="${MUJOCO_GL:-egl}" \
    "$VENV/bin/python" -c "import libero.libero" >/dev/null

echo "== Import and profile check"
LIBERO_CONFIG_PATH="$VENV/libero_config" MUJOCO_GL="${MUJOCO_GL:-egl}" \
    EXPECTED_PROFILE="$PROFILE" "$VENV/bin/python" - <<'PY'
import os
import jax, flax, optax, tensorflow, transformers, robosuite, mujoco, torch
import octo.model.octo_model
from libero.libero.envs import OffScreenRenderEnv  # the class rollouts instantiate
from wsl_vla.experiments.protocol import LIBERO_GIT_REVISION
from wsl_vla.experiments.provenance import assert_installed_vcs_revision
from wsl_vla.octo.compat import ENVIRONMENT_PROFILES, require_environment_profile

profile = require_environment_profile()
assert profile == os.environ["EXPECTED_PROFILE"], (
    f"installed environment matches profile {profile!r}, not {os.environ['EXPECTED_PROFILE']!r}"
)
assert_installed_vcs_revision("libero", LIBERO_GIT_REVISION)
print("profile:", profile, "|", ENVIRONMENT_PROFILES[profile]["hardware_status"])
print("jax", jax.__version__, "backend", jax.default_backend(), jax.devices())
PY
echo "Environment ready: source $VENV/bin/activate"
echo "Next: python scripts/eval/research_preflight.py --manifest configs/research/manifests/integrity.yaml --output-dir <dir>"
