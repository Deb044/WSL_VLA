#!/usr/bin/env bash
# Record the complete transitive lock of the ACTIVE research environment.
#
# Usage (inside an activated research environment, from the repository root):
#   envs/scripts/freeze_environment.sh <blackwell|legacy-gpu>
#
# Run this only after the integrity run profile has passed in the environment:
# a lock records what was verified, it does not make anything verified. The
# lock is written to envs/locks/research-<profile>.freeze.txt together with the
# interpreter, platform, GPU, and the profile the repository detects.
set -euo pipefail

PROFILE="${1:?usage: freeze_environment.sh <blackwell|legacy-gpu>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="$REPO_ROOT/envs/locks/research-$PROFILE.freeze.txt"
mkdir -p "$(dirname "$OUTPUT")"

DETECTED="$(cd "$REPO_ROOT" && python -c 'from wsl_vla.octo.compat import require_environment_profile; print(require_environment_profile())')"
if [ "$DETECTED" != "$PROFILE" ]; then
    echo "active environment matches profile '$DETECTED', not '$PROFILE'" >&2
    exit 1
fi
{
    echo "# pip freeze of the '$PROFILE' research environment"
    echo "# generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# python: $(python -c 'import sys; print(sys.version.split()[0])')  platform: $(uname -srm)"
    echo "# gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unavailable)"
    echo "# Editable checkouts (this repository, LIBERO) are listed as VCS references."
    python -m pip freeze | grep -v ' @ file://'
} > "$OUTPUT"
echo "wrote $OUTPUT ($(grep -c '==' "$OUTPUT") pinned distributions)"
