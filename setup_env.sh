#!/usr/bin/env bash
# ==============================================================================
# WSL_VLA: Automated Environment Installation & Setup Script (Linux / WSL2)
# ==============================================================================
set -euo pipefail

ENV_NAME="vla_zoo"

echo "======================================================================"
echo " Step 1: Configuring Environment & Headless Rendering Variables"
echo "======================================================================"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Persist environment variables to ~/.bashrc if not already present
for VAR in "export MUJOCO_GL=egl" "export PYOPENGL_PLATFORM=egl" "export XLA_PYTHON_CLIENT_PREALLOCATE=false"; do
    if ! grep -Fxq "$VAR" ~/.bashrc 2>/dev/null; then
        echo "$VAR" >> ~/.bashrc
        echo "  -> Appended to ~/.bashrc: $VAR"
    fi
done

echo "======================================================================"
echo " Step 2: Locating Conda & Setting up Environment (${ENV_NAME})"
echo "======================================================================"
# Locate conda initialization script
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "Error: conda not found. Please install Miniforge or Miniconda first:"
    echo "  https://github.com/conda-forge/miniforge"
    exit 1
fi

if conda info --envs | grep -E "^${ENV_NAME}\s" > /dev/null; then
    echo "  -> Conda environment '${ENV_NAME}' already exists. Activating..."
else
    echo "  -> Creating new conda environment '${ENV_NAME}' with Python 3.10..."
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda activate "${ENV_NAME}"

echo "======================================================================"
echo " Step 3: Installing JAX, Flax & Hardware Acceleration"
echo "======================================================================"
pip install --upgrade pip setuptools wheel

if command -v nvidia-smi &> /dev/null; then
    echo "  -> NVIDIA GPU detected via nvidia-smi!"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "  -> Installing JAX with CUDA 12 support..."
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || {
        echo "  -> CUDA 12 pip fallback to standard JAX..."
        pip install jax jaxlib
    }
else
    echo "  -> No NVIDIA GPU detected. Installing standard CPU JAX..."
    pip install jax jaxlib
fi

echo "======================================================================"
echo " Step 4: Installing Core Model, Simulation & Pipeline Packages"
echo "======================================================================"
pip install -r requirements.txt
pip install -e .

echo "======================================================================"
echo " Step 5: Running System Verification & Test Suite"
echo "======================================================================"
python -c "
import jax
import flax
import optax
import h5py
import numpy as np
import core
import data
import models

print('--- Environment Verification ---')
print('Python Version  :', np.__version__)
print('JAX Version     :', jax.__version__)
print('JAX Default Dev :', jax.default_backend())
print('JAX Devices     :', jax.devices())
print('Flax Version    :', flax.__version__)
print('Optax Version   :', optax.__version__)
print('HDF5 Version    :', h5py.__version__)
print('WSL_VLA Package : OK')
"

echo "Running pytest test suite..."
pytest tests/

echo "======================================================================"
echo " Setup Completed Successfully!"
echo " Activate your environment with: conda activate ${ENV_NAME}"
echo "======================================================================"
