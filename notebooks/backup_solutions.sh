#!/usr/bin/env bash
#
# Backup the 'solutions' directory into OneDrive as a timestamped tar.gz.
# Usage: ./backup_solutions.sh

set -euo pipefail

# --- Step 1: Check that ONEDRIVE environment variable is set ---
if [[ -z "${ONEDRIVE:-}" ]]; then
    echo "❌ Environment variable \$ONEDRIVE is not set."
    echo "Please set it to your OneDrive directory, e.g.:"
    echo "    export ONEDRIVE='/Users/janisou1/OneDrive/OneDrive - KamIT 365/__OWNDRIVE'"
    echo "Then re-run this script."
    exit 1
fi

# --- Step 2: Check that solutions directory exists ---
if [[ ! -d "nb/solutions" ]]; then
    echo "❌ No 'nb/solutions' directory found in the current folder: $(pwd)"
    echo "Please run this script from the root of your course repository."
    exit 1
fi

# --- Step 3: Define backup location and filename ---
BACKUP_DIR="${ONEDRIVE}/__SOLUTIONS_BACKUPS/syvaoppiminen"
TIMESTAMP=$(date +"%Y-%m-%d")
FILENAME="${TIMESTAMP}-solutions.tar.gz"
DEST_PATH="${BACKUP_DIR}/${FILENAME}"

# --- Step 4: Create backup directory if needed ---
mkdir -p "${BACKUP_DIR}"

# --- Step 5: Create the tar.gz archive ---
echo "📦 Creating backup of 'notebooks/solutions/' ..."
tar -czf "${DEST_PATH}" notebooks/solutions

echo "✅ Backup created successfully:"
echo "    ${DEST_PATH}"