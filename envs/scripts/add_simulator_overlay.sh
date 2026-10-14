#!/usr/bin/env bash
# Add the pinned LIBERO simulator stack on top of an existing JAX/Octo
# environment, without reinstalling that environment.
#
# Usage:
#   envs/scripts/add_simulator_overlay.sh <python-of-existing-env> <overlay-venv-dir>
#
# Example (legacy-gpu profile already installed in a conda env):
#   envs/scripts/add_simulator_overlay.sh ~/miniforge3/envs/vla_zoo/bin/python .venv-sim-overlay
#   source .venv-sim-overlay/bin/activate
#
# The overlay is a virtual environment created with --system-site-packages, so
# it sees every package of the base environment (JAX, Flax, Octo, ...) and adds
# only robosuite, MuJoCo 2.3.7, bddl, and the pinned LIBERO checkout. Activating
# it also exports LIBERO_CONFIG_PATH, so the user's ~/.libero is never touched.
set -euo pipefail

BASE_PYTHON="${1:?usage: add_simulator_overlay.sh <python-of-existing-env> <overlay-venv-dir>}"
VENV="${2:?usage: add_simulator_overlay.sh <python-of-existing-env> <overlay-venv-dir>}"
LIBERO_REVISION="8f1084e3132a39270c3a13ebe37270a43ece2a01"
RETRIES="${PIP_ATTEMPTS:-8}"

"$BASE_PYTHON" -c 'import jax, flax, octo' >/dev/null 2>&1 || {
    echo "the base interpreter must already provide jax, flax, and octo" >&2
    exit 1
}
"$BASE_PYTHON" -c 'import torch' >/dev/null 2>&1 || {
    echo "the base interpreter must provide torch (LIBERO loads its initial states with it)" >&2
    exit 1
}
mkdir -p "$VENV"
VENV="$(cd "$VENV" && pwd)"
[ -x "$VENV/bin/python" ] || "$BASE_PYTHON" -m venv --system-site-packages "$VENV"
PIP=("$VENV/bin/python" -m pip --default-timeout 180 --retries 10)

# Keep the base environment's numerical stack: the overlay must not shadow it.
"$BASE_PYTHON" - > "$VENV/constraints.txt" <<'EOF'
import importlib.metadata as metadata
for name in ("numpy", "scipy", "pillow", "h5py"):
    print(f"{name}=={metadata.version(name)}")
print("opencv-python==4.11.0.86")
print("mujoco==2.3.7")
EOF

pip_install() {
    local attempt
    for attempt in $(seq 1 "$RETRIES"); do
        if "${PIP[@]}" install --quiet -c "$VENV/constraints.txt" "$@"; then
            return 0
        fi
        echo "pip attempt $attempt failed; retrying" >&2
    done
    return 1
}

echo "== Simulator packages (pinned; see envs/requirements/simulator.txt)"
pip_install --no-deps "robosuite==1.4.0"
pip_install -r "$(cd "$(dirname "${BASH_SOURCE[0]}")/../requirements" && pwd)/simulator.txt"

echo "== LIBERO: editable checkout at $LIBERO_REVISION"
mkdir -p "$VENV/src"
if [ ! -d "$VENV/src/LIBERO/.git" ]; then
    git clone --quiet https://github.com/Lifelong-Robot-Learning/LIBERO.git "$VENV/src/LIBERO"
fi
git -C "$VENV/src/LIBERO" checkout --quiet "$LIBERO_REVISION"
"${PIP[@]}" install --quiet --no-deps --config-settings editable_mode=compat -e "$VENV/src/LIBERO"

echo "== LIBERO config (kept inside the overlay; first import prompts on stdin)"
mkdir -p "$VENV/libero_config"
if ! grep -q "LIBERO_CONFIG_PATH" "$VENV/bin/activate"; then
    {
        echo ""
        echo "# Added by envs/scripts/add_simulator_overlay.sh"
        echo "export LIBERO_CONFIG_PATH=\"$VENV/libero_config\""
        echo "export MUJOCO_GL=\"\${MUJOCO_GL:-egl}\""
        echo "export XLA_PYTHON_CLIENT_PREALLOCATE=\"\${XLA_PYTHON_CLIENT_PREALLOCATE:-false}\""
    } >> "$VENV/bin/activate"
fi
printf 'N\n' | LIBERO_CONFIG_PATH="$VENV/libero_config" MUJOCO_GL="${MUJOCO_GL:-egl}" \
    "$VENV/bin/python" -c "import libero.libero" >/dev/null

echo "== Import check"
LIBERO_CONFIG_PATH="$VENV/libero_config" MUJOCO_GL="${MUJOCO_GL:-egl}" "$VENV/bin/python" - <<'EOF'
import importlib.metadata as metadata
import jax, flax, octo.model.octo_model, robosuite, mujoco, bddl
import robosuite.macros as macros
from libero.libero import benchmark, get_libero_path
# The class rollouts instantiate; importing it pulls in everything LIBERO needs.
from libero.libero.envs import OffScreenRenderEnv
print("jax", jax.__version__, "| robosuite", robosuite.__version__, "| mujoco", mujoco.__version__)
print("robosuite image convention:", macros.IMAGE_CONVENTION)
print("bddl files:", get_libero_path("bddl_files"))
print("suites:", sorted(benchmark.get_benchmark_dict())[:6])
EOF
echo "Overlay ready: source $VENV/bin/activate"
