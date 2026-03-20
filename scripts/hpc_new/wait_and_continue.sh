#!/bin/bash
# Wait for a running process (e.g. transfuser phase 2) to finish,
# then automatically run preprocess -> build_cache -> anchors -> stats
#
# Usage:
#   bash scripts/hpc_new/wait_and_continue.sh <PID>
#
# Example:
#   bash scripts/hpc_new/wait_and_continue.sh 3502000

set -euo pipefail

CODE_DIR=/workspace1/z_project/code/motdp_z
LOG_DIR=${CODE_DIR}/logs

mkdir -p "${LOG_DIR}"

WAIT_PID=${1:-""}
if [ -z "${WAIT_PID}" ]; then
    echo "Usage: bash scripts/hpc_new/wait_and_continue.sh <PID>"
    exit 1
fi

echo "[wait_and_continue] Waiting for PID ${WAIT_PID} to finish..."
tail --pid="${WAIT_PID}" -f /dev/null
echo "[wait_and_continue] PID ${WAIT_PID} finished. Starting pipeline continuation."

cd "${CODE_DIR}"

run_step() {
    local step=$1
    echo ""
    echo "========================================"
    echo "  Starting: ${step}"
    echo "  $(date)"
    echo "========================================"
    bash scripts/hpc_new/deploy_pipeline.sh "${step}"
    echo "[wait_and_continue] ${step} done at $(date)"
}

run_step preprocess
run_step build_cache
run_step anchors
run_step stats

echo ""
echo "========================================"
echo "  All steps done at $(date)"
echo "========================================"
