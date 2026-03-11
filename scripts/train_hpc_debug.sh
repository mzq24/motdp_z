#!/bin/bash
# HPC interactive debug training (single GPU, short run)
# Usage: bash scripts/train_hpc_debug.sh [gpu_id]
# Kill:  pkill -9 -f "train_carla_bev"

GPU_ID=${1:-0}
SCRATCH=/home/users/ntu/wh.huang/scratch
CODE_DIR=${SCRATCH}/z_projects/code/motdp_z
CONFIG_PATH=${CODE_DIR}/config/pdm_hpc.yaml

cd ${CODE_DIR}

echo "========================================="
echo "HPC Debug Training (single GPU)"
echo "========================================="
echo "GPU ID       : $GPU_ID"
echo "Config Path  : $CONFIG_PATH"
echo "========================================="

export CUDA_VISIBLE_DEVICES=$GPU_ID
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline

torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --max_restarts=0 \
    --rdzv_id=$$ \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$((29500 + RANDOM % 1000)) \
    training/train_carla_bev.py \
    --config_path "${CONFIG_PATH}"
