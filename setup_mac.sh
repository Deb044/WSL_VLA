#!/usr/bin/env bash
# ==============================================================================
# WSL_VLA: macOS Setup Script (Apple Silicon & Intel)
# ==============================================================================
set -euo pipefail

ENV_NAME="vla_zoo"

echo "======================================================================"
echo " Setting up WSL_VLA on macOS"
echo "======================================================================"

# 1. Locate Conda
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "Conda not found. Please install Miniforge or Miniconda first:"
    echo "  brew install --cask miniforge"
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
export MUJOCO_GL=cgl

# 4. Install Dependencies
echo "  -> Upgrading pip & installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install jax jaxlib
pip install -r requirements.txt
pip install -e .

# 5. Verification Check
echo "======================================================================"
echo " Running Verification Tests"
echo "======================================================================"
pytest tests/

echo "======================================================================"
echo " Setup Completed! Run: conda activate ${ENV_NAME}"
echo "======================================================================"
