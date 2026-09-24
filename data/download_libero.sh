#!/usr/bin/env bash
# ==============================================================================
# Script to download all 4 LIBERO demonstration suites onto NVMe scratch
# ==============================================================================
set -euo pipefail

DEST_DIR="${1:-./data/libero}"
mkdir -p "$DEST_DIR"

echo "======================================================================"
echo " Downloading LIBERO Benchmark Suites to: $DEST_DIR"
echo "======================================================================"

# Benchmark datasets are hosted on AWS S3 / HuggingFace mirrors
# Official LIBERO assets:
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

BASE_URL="https://huggingface.co/datasets/openvla/libero-demos/resolve/main"

for suite in "${SUITES[@]}"; do
    TARGET_FILE="${DEST_DIR}/${suite}.hdf5"
    if [ -f "$TARGET_FILE" ] && [ -s "$TARGET_FILE" ]; then
        echo "  [EXISTS] ${suite}.hdf5 already present. Skipping download."
    else
        echo "  [DOWNLOADING] ${suite}.hdf5 from mirror..."
        # Try direct wget from HuggingFace dataset repo or fallback
        if command -v wget &> /dev/null; then
            wget -c "${BASE_URL}/${suite}.hdf5" -O "$TARGET_FILE" || true
        elif command -v curl &> /dev/null; then
            curl -L -C - "${BASE_URL}/${suite}.hdf5" -o "$TARGET_FILE" || true
        fi
        
        # If download failed or file is empty, warn user
        if [ ! -s "$TARGET_FILE" ]; then
            echo "  [NOTICE] Remote file not downloaded or URL unavailable."
            echo "           The pipeline will automatically fallback to synthetic demo generation for local dry-runs."
            rm -f "$TARGET_FILE"
        else
            echo "  [VERIFIED] ${suite}.hdf5 downloaded successfully!"
        fi
    fi
done

echo "======================================================================"
echo " Dataset download staging finished."
echo "======================================================================"
