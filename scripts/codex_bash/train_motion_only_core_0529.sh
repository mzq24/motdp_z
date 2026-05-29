#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:-/data/z_project/code/motdp_z_semantic_state_strict_ablation_v1}
CONFIG=${CONFIG:-config/pdm_hpc_route_b_lidar_bev_motion_only_core_0529.yaml}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
GPUS=${GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29591}
RESUME=${RESUME:-}

cd "${CODE_DIR}"
ulimit -n 65535 || true

echo "CODE_DIR=${CODE_DIR}"
echo "CONFIG=${CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "GPUS=${GPUS}"
echo "MASTER_PORT=${MASTER_PORT}"
if [[ -n "${RESUME}" ]]; then
  echo "RESUME=${RESUME}"
fi

export CUDA_VISIBLE_DEVICES

CMD=(torchrun --standalone --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" \
  training/train_carla_bev.py --config_path "${CONFIG}")
if [[ -n "${RESUME}" ]]; then
  CMD+=(--resume "${RESUME}")
fi

"${CMD[@]}"
