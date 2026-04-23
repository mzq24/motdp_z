#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/workspace1/z_project/code/motdp_z
CONDA_ENV=z_dpauto
CONFIG=config/tmp/pdm_hpc_route_b_lidar_bev_stage1_fulltrain_val.yaml
GPUS=${GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29517}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}

cd "${CODE_DIR}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_PATH="logs/route_b_lidar_bev_stage1_fulltrain_val_${STAMP}.log"

echo "========================================"
echo "  Route B LiDAR BEV Stage1 Full Train + Val"
echo "  Config: ${CONFIG}"
echo "  GPUs:   ${GPUS}"
echo "  CUDA:   ${CUDA_VISIBLE_DEVICES}"
echo "  Log:    ${LOG_PATH}"
echo "  Time:   $(date)"
echo "========================================"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun \
  --standalone \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  training/train_carla_bev.py \
  --config_path "${CONFIG}" 2>&1 | tee "${LOG_PATH}"
