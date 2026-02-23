#!/bin/bash
# Single GPU Training Script for CARLA BEV Policy
# Usage: bash training/train_carla_bev_single_gpu.sh [gpu_id] [config_path] [resume_ckpt]
# Kill:  pkill -9 -f "train_carla_bev"
#
# Examples:
#   bash training/train_carla_bev_single_gpu.sh
#   bash training/train_carla_bev_single_gpu.sh 1
#   bash training/train_carla_bev_single_gpu.sh 0 /path/to/config.yaml
#   bash training/train_carla_bev_single_gpu.sh 0 /path/to/config.yaml /path/to/checkpoint.pth

GPU_ID=${1:-0}
CONFIG_PATH=${2:-"/media/z/data/mzq/others/MoT-DP/config/pdm_local.yaml"}
RESUME_PATH=${3:-""}

export CUDA_VISIBLE_DEVICES=$GPU_ID
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-"offline"}

echo "========================================="
echo "Single GPU Training"
echo "========================================="
echo "GPU ID       : $GPU_ID"
echo "Config Path  : $CONFIG_PATH"
echo "Resume       : ${RESUME_PATH:-none}"
echo "WANDB_MODE   : $WANDB_MODE"
echo "========================================="

CMD="python training/train_carla_bev.py --config_path \"$CONFIG_PATH\""

if [ -n "$RESUME_PATH" ]; then
    CMD="$CMD --resume \"$RESUME_PATH\""
fi

eval $CMD
