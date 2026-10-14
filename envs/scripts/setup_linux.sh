#!/usr/bin/env bash
# Kept for existing references: creates the `blackwell` research environment.
#
# Usage:
#   PYTHON=/path/to/python3.11 envs/scripts/setup_linux.sh <venv-directory>
#
# Equivalent to
#   envs/scripts/setup_research.sh --profile blackwell <venv-directory>
# Older GPUs (CUDA 11, JAX 0.4.20) use `--profile legacy-gpu` instead.
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/setup_research.sh" --profile blackwell "$@"
