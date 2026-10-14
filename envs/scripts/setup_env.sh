#!/usr/bin/env bash
# Kept for existing references: creates the `legacy-smoke` (PyTorch) profile.
#
# The research experiments do not use that environment. For Octo-Small zoo and
# alignment work run instead:
#   envs/scripts/setup_research.sh --profile <blackwell|legacy-gpu> <venv-directory>
set -euo pipefail
echo "setup_env.sh installs the legacy PyTorch smoke profile, not a research environment." >&2
echo "Research: envs/scripts/setup_research.sh --profile <blackwell|legacy-gpu> <venv-directory>" >&2
exec "$(dirname "${BASH_SOURCE[0]}")/setup_legacy_smoke.sh" "$@"
