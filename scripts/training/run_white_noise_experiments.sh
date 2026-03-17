#!/bin/bash
# Run two white noise experiments sequentially:
# 1. white_noise_delta: predict delta -> cumsum -> abs traj
# 2. white_noise_abs: predict abs traj directly (no delta/cumsum)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_ROOT}/training/train_carla_bev.py"

echo "========================================"
echo "Experiment 1: white_noise_delta"
echo "  Full diffusion + delta mode + cumsum"
echo "========================================"
python "${TRAIN_SCRIPT}" \
    --config_path "${PROJECT_ROOT}/config/pdm_local_white_noise_delta.yaml"

# echo ""
# echo "========================================"
# echo "Experiment 2: white_noise_abs"
# echo "  Full diffusion + abs traj mode"
# echo "========================================"
# python "${TRAIN_SCRIPT}" \
#     --config_path "${PROJECT_ROOT}/config/pdm_local_white_noise_abs.yaml"

# echo ""
# echo "========================================"
# echo "Both experiments completed!"
# echo "Check wandb for results."
# echo "========================================"
