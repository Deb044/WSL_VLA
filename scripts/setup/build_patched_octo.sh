#!/usr/bin/env bash
# Rebuild the JAX 0.7 port of Octo as a byte-identical git commit.
#
# Usage: scripts/setup/build_patched_octo.sh <destination-directory>
# Prints the destination path. Install with:
#   pip install --no-deps "octo @ git+file://<destination>@${OCTO_GIT_REVISION}"
# The commit hash is deterministic because author, committer, and dates are
# fixed, so preflight can verify the installed revision exactly.
set -euo pipefail

UPSTREAM_URL="https://github.com/octo-models/octo.git"
UPSTREAM_REVISION="241fb3514b7c40957a86d869fecb7c7fc353f540"
PATCHED_REVISION="a4cc964b7e77f8d8b19f533a0dfa95d653501ab7"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${1:?usage: build_patched_octo.sh <destination-directory>}"

if [ -d "$DEST/.git" ] && [ "$(git -C "$DEST" rev-parse HEAD)" = "$PATCHED_REVISION" ]; then
    echo "$DEST"
    exit 0
fi
rm -rf "$DEST"
git clone --quiet "$UPSTREAM_URL" "$DEST"
git -C "$DEST" checkout --quiet "$UPSTREAM_REVISION"

export GIT_AUTHOR_NAME="WSL-VLA Blackwell Port"
export GIT_AUTHOR_EMAIL="wsl-vla-port@users.noreply.github.com"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
export GIT_COMMITTER_DATE="2026-10-03T00:00:00+00:00"
git -C "$DEST" -c commit.gpgsign=false am --quiet "$REPO_ROOT"/third_party/octo/*.patch

ACTUAL="$(git -C "$DEST" rev-parse HEAD)"
if [ "$ACTUAL" != "$PATCHED_REVISION" ]; then
    echo "patched Octo revision mismatch: got $ACTUAL, expected $PATCHED_REVISION" >&2
    exit 1
fi
echo "$DEST"
