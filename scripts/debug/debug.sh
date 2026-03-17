#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/debug.sh [gpu_id] [config_path] [resume_ckpt] [debug_dir] [val_only]
# Examples:
#   bash scripts/debug.sh
#   bash scripts/debug.sh 0 config/pdm_local.yaml
#   bash scripts/debug.sh 0 config/pdm_local.yaml /path/to/ckpt.pt /tmp/grid_debug 1

GPU_ID=${1:-0}
CONFIG_PATH=${2:-"config/pdm_local.yaml"}
RESUME_PATH=${3:-""}
DEBUG_DIR=${4:-"debug_grid_bev"}
VAL_ONLY=${5:-0}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export DEBUG_GRID_BEV=1
export DEBUG_GRID_BEV_DIR=${DEBUG_DIR}

mkdir -p "${DEBUG_GRID_BEV_DIR}"

echo "========================================="
echo "Grid BEV Debug Run"
echo "========================================="
echo "GPU_ID          : ${GPU_ID}"
echo "CONFIG_PATH     : ${CONFIG_PATH}"
echo "RESUME_PATH     : ${RESUME_PATH:-none}"
echo "DEBUG_GRID_BEV  : ${DEBUG_GRID_BEV}"
echo "DEBUG_GRID_BEV_DIR: ${DEBUG_GRID_BEV_DIR}"
echo "VAL_ONLY        : ${VAL_ONLY}"
echo "========================================="

CMD=(python training/train_carla_bev.py --config_path "${CONFIG_PATH}")

if [[ -n "${RESUME_PATH}" ]]; then
  CMD+=(--resume "${RESUME_PATH}")
fi

if [[ "${VAL_ONLY}" == "1" ]]; then
  CMD+=(--val_only)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
