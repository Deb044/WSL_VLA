#!/usr/bin/env bash
# ==============================================================================
# Environment Installation & Setup Script for VLA Model Zoo
# ==============================================================================
set -euo pipefail

ENV_NAME="vla_zoo"

echo "======================================================================"
echo " Step 1: Configuring Headless Rendering Variables"
echo "======================================================================"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Persist to ~/.bashrc if not already present
if ! grep -q "export MUJOCO_GL=egl" ~/.bashrc 2>/dev/null; then
    echo "export MUJOCO_GL=egl" >> ~/.bashrc
    echo "export PYOPENGL_PLATFORM=egl" >> ~/.bashrc
    echo "  -> Appended MUJOCO_GL=egl to ~/.bashrc"
fi

echo "======================================================================"
echo " Step 2: Creating Conda Environment (${ENV_NAME})"
echo "======================================================================"
# Locate conda initialization script
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
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
echo " Step 3: Installing PyTorch & CUDA Dependencies"
echo "======================================================================"
pip install --upgrade pip setuptools wheel

if command -v nvidia-smi &> /dev/null; then
    echo "  -> NVIDIA GPU detected! Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "  -> No NVIDIA GPU detected. Installing PyTorch CPU version..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "======================================================================"
echo " Step 4: Installing Core Model, Simulation & Evidence Packages"
echo "======================================================================"
pip install -r "$REPO_ROOT/envs/requirements/base.txt"

echo "======================================================================"
echo " Step 5: Verification Check"
echo "======================================================================"
python -c "
import torch
import peft
import h5py
import mujoco
import sentence_transformers

print('--- Verification Summary ---')
print('PyTorch Version :', torch.__version__)
print('CUDA Available  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device      :', torch.cuda.get_device_name(0))
print('PEFT Version    :', peft.__version__)
print('MuJoCo Version  :', mujoco.__version__)
print('Setup completed successfully!')
"

echo "======================================================================"
echo " Done! Activate the environment with: conda activate ${ENV_NAME}"
echo "======================================================================"
