#!/usr/bin/env bash
# ==============================================================================
# macOS Setup Script for VLA Model Zoo (Apple Silicon M1/M2/M3/M4 & Intel Mac)
# ==============================================================================
set -euo pipefail

ENV_NAME="vla_zoo"

echo "======================================================================"
echo " Setting up VLA Model Zoo on macOS"
echo "======================================================================"

# 1. Check for Homebrew / Conda
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
    echo "Conda not found. Please install Miniconda or Miniforge first:"
    echo "  brew install --cask miniconda"
    exit 1
fi

# 2. Create or activate conda environment
if conda info --envs | grep -E "^${ENV_NAME}\s" > /dev/null; then
    echo "  -> Conda environment '${ENV_NAME}' already exists. Activating..."
else
    echo "  -> Creating Conda environment '${ENV_NAME}' (Python 3.10)..."
    conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"

# 3. macOS Rendering Configuration
# On macOS, MuJoCo uses CGL/GLFW rendering rather than Linux EGL
export MUJOCO_GL=cgl

# 4. Install Dependencies
echo "  -> Upgrading pip & installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install torch torchvision
pip install -r requirements.txt

# 5. Verification Check (Testing Apple Silicon MPS backend)
echo "======================================================================"
echo " Verifying Environment & Apple Silicon Metal (MPS) Acceleration"
echo "======================================================================"
python -c "
import torch
import peft
import mujoco

print('--- macOS Verification Summary ---')
print('PyTorch Version :', torch.__version__)
mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
print('Apple MPS Available (Metal GPU):', mps_available)
if mps_available:
    print('Hardware Acceleration : Apple Silicon GPU (Metal) ACTIVE')
else:
    print('Hardware Acceleration : CPU')
print('PEFT Version    :', peft.__version__)
print('MuJoCo Version  :', mujoco.__version__)
print('Setup completed successfully on macOS!')
"

echo "======================================================================"
echo " Ready! Run: conda activate ${ENV_NAME}"
echo "======================================================================"
