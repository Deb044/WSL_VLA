#!/usr/bin/env bash
# Create the Blackwell-compatible research environment.
#
# Usage:
#   PYTHON=/path/to/python3.11 envs/scripts/setup_linux.sh <venv-directory>
#
# On the IITG cluster, keep the environment on /scratch (home is 20 GB):
#   conda create -y -p /scratch/$USER/envs/py311 python=3.11
#   PYTHON=/scratch/$USER/envs/py311/bin/python \
#     envs/scripts/setup_linux.sh /scratch/$USER/envs/vla
# Run it on the login node: it downloads packages and source checkouts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${1:?usage: setup_linux.sh <venv-directory>}"
PYTHON="${PYTHON:-python3.11}"
DLIMP_REVISION="5edaa4691567873d495633f2708982b42edf1972"
LIBERO_REVISION="8f1084e3132a39270c3a13ebe37270a43ece2a01"
OCTO_REVISION="a4cc964b7e77f8d8b19f533a0dfa95d653501ab7"

if [ "$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" != "3.11" ]; then
    echo "Python 3.11 is required (got $("$PYTHON" --version))" >&2
    exit 1
fi
cd "$REPO_ROOT"
"$PYTHON" -m venv "$VENV"
PIP=("$VENV/bin/python" -m pip)
"${PIP[@]}" install --quiet --upgrade pip

echo "== Pinned packages"
"${PIP[@]}" install -r "$REPO_ROOT/envs/requirements/research.txt"

echo "== dlimp (its metadata pins TensorFlow 2.15; the code runs on 2.20)"
"${PIP[@]}" install --no-deps "dlimp @ git+https://github.com/kvablack/dlimp@$DLIMP_REVISION"

echo "== Octo: upstream 241fb35 + third_party/octo patches, deterministic commit"
"$REPO_ROOT/scripts/setup/build_patched_octo.sh" "$VENV/src/octo" >/dev/null
"${PIP[@]}" install --no-deps "octo @ git+file://$VENV/src/octo@$OCTO_REVISION"

echo "== LIBERO: editable checkout (its top-level package has no __init__.py)"
"${PIP[@]}" install --no-deps --src "$VENV/src" --config-settings editable_mode=compat \
    -e "git+https://github.com/Lifelong-Robot-Learning/LIBERO.git@$LIBERO_REVISION#egg=libero"

echo "== LIBERO config (first import prompts on stdin, which batch jobs cannot answer)"
printf 'N\n' | "$VENV/bin/python" -c "import libero.libero" >/dev/null

echo "== Import check"
"$VENV/bin/python" - <<'EOF'
import jax, flax, optax, orbax.checkpoint, tensorflow, transformers, robosuite, mujoco, torch
import octo.model.octo_model, libero
from wsl_vla.experiments.provenance import assert_installed_vcs_revision
from wsl_vla.experiments.protocol import LIBERO_GIT_REVISION
from wsl_vla.octo.bridge import OCTO_GIT_REVISION
assert_installed_vcs_revision("octo", OCTO_GIT_REVISION)
assert_installed_vcs_revision("libero", LIBERO_GIT_REVISION)
print("jax", jax.__version__, "backend", jax.default_backend(), jax.devices())
EOF
echo "Environment ready: source $VENV/bin/activate"
