#!/bin/bash
# Multi-GPU Distributed Training Script for CARLA BEV Policy
# Usage: bash training/train_carla_bev_multi_gpu.sh [gpu_ids] [config_path] [resume_ckpt]
# Kill:  pkill -9 -f "train_carla_bev"
#
# Examples:
#   bash training/train_carla_bev_multi_gpu.sh
#   bash training/train_carla_bev_multi_gpu.sh 0,1,2,3
#   bash training/train_carla_bev_multi_gpu.sh 0,1,2,3 /path/to/config.yaml
#   bash training/train_carla_bev_multi_gpu.sh 0,1,2,3 /path/to/config.yaml /path/to/checkpoint.pth

GPU_IDS=${1:-"0,1"}
CONFIG_PATH=${2:-"/media/z/data/mzq/others/MoT-DP/config/pdm_local.yaml"}
RESUME_PATH=${3:-""}

export CUDA_VISIBLE_DEVICES=$GPU_IDS
# Auto-derive NUM_GPUS from GPU_IDS so they are always consistent
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-"online"}
# Use a random port to avoid "address already in use" when re-launching quickly
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

echo "========================================="
echo "Multi-GPU Distributed Training"
echo "========================================="
echo "GPU IDs      : $GPU_IDS"
echo "Num GPUs     : $NUM_GPUS"
echo "Config Path  : $CONFIG_PATH"
echo "Resume       : ${RESUME_PATH:-none}"
echo "Master Port  : $MASTER_PORT"
echo "WANDB_MODE   : $WANDB_MODE"
echo "========================================="

CMD="torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --max_restarts=0 \
    --rdzv_id=$$ \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$MASTER_PORT \
    training/train_carla_bev.py \
    --config_path \"$CONFIG_PATH\""

if [ -n "$RESUME_PATH" ]; then
    CMD="$CMD --resume \"$RESUME_PATH\""
fi

eval $CMD
