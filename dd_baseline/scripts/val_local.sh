#!/bin/bash
# Local single/multi-GPU evaluation for DD Baseline
# Usage: bash dd_baseline/scripts/val_local.sh [gpu_ids] /path/to/checkpoint.pt

GPU_IDS=${1:-"0"}
RESUME_PATH=$2
CONFIG_PATH="dd_baseline/dd_config.yaml"

if [ -z "$RESUME_PATH" ]; then
    echo "Error: Need to provide a checkpoint path."
    echo "Usage: bash dd_baseline/scripts/val_local.sh [gpu_ids] /path/to/checkpoint.pt"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

echo "========================================="
echo "Evaluating DD Baseline"
echo "========================================="
echo "GPU IDs      : $GPU_IDS"
echo "Num GPUs     : $NUM_GPUS"
echo "Checkpoint   : $RESUME_PATH"
echo "========================================="

if [ "${DEBUG:-0}" = "1" ]; then
    # Single-GPU debugpy mode: attach with VSCode on port 5678
    export MASTER_ADDR=localhost
    export MASTER_PORT=$MASTER_PORT
    export WORLD_SIZE=1
    export RANK=0
    export LOCAL_RANK=0
    echo "Waiting for debugger on 0.0.0.0:5678 ..."
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
        dd_baseline/train.py \
        --config_path "$CONFIG_PATH" \
        --resume "$RESUME_PATH" \
        --val_only
else
    torchrun \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --max_restarts=0 \
        --rdzv_id=$$ \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:$MASTER_PORT \
        dd_baseline/train.py \
        --config_path "$CONFIG_PATH" \
        --resume "$RESUME_PATH" \
        --val_only
fi