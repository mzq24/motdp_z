#!/bin/bash
# Local single/multi-GPU training script
# Usage: bash training/train_local.sh [gpu_ids] [resume_ckpt] [debug]
# Kill:  pkill -9 -f "train_carla_bev"
#
# Examples:
#   bash training/train_local.sh
#   bash training/train_local.sh 0
#   bash training/train_local.sh 0,1
#   bash training/train_local.sh 0 /path/to/checkpoint.pth
#   bash training/train_local.sh 0 "" 1
#   bash training/train_local.sh 0 /path/to/checkpoint.pth true

GPU_IDS=${1:-"0"}
RESUME_PATH=${2:-""}
DEBUG_FLAG_RAW=${3:-"0"}
DEBUG_FLAG=$(echo "$DEBUG_FLAG_RAW" | tr '[:upper:]' '[:lower:]')
CONFIG_PATH="/media/z/data/mzq/others/MoT-DP/config/pdm_local.yaml"
DEBUG_PORT=${DEBUG_PORT:-5678}

export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-"online"}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

DEBUG_ARGS=""
DEBUG_MODE="off"
if [ "$DEBUG_FLAG" = "1" ] || [ "$DEBUG_FLAG" = "true" ] || [ "$DEBUG_FLAG" = "yes" ]; then
    DEBUG_ARGS="-m debugpy --listen 0.0.0.0:$DEBUG_PORT --wait-for-client"
    DEBUG_MODE="on (port: $DEBUG_PORT)"
fi

echo "========================================="
echo "Local Training"
echo "========================================="
echo "GPU IDs      : $GPU_IDS"
echo "Num GPUs     : $NUM_GPUS"
echo "Config Path  : $CONFIG_PATH"
echo "Resume       : ${RESUME_PATH:-none}"
echo "Debug Mode   : $DEBUG_MODE"
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
    $DEBUG_ARGS \
    training/train_carla_bev.py \
    --config_path \"$CONFIG_PATH\""

if [ -n "$RESUME_PATH" ]; then
    CMD="$CMD --resume \"$RESUME_PATH\""
fi

eval $CMD
