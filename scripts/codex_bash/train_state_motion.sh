#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:-$(pwd)}
CONDA_ENV=${CONDA_ENV:-z_dpauto}
CONFIG=${CONFIG:-config/pdm_hpc_route_b_state_motion_motion_only_0520.yaml}
GPUS=${GPUS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29531}
RESUME=${RESUME:-}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-}
STAMP=$(date +%Y%m%d_%H%M%S)

cd "${CODE_DIR}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

mkdir -p logs
LOG_PATH="logs/state_motion_$(basename "${CONFIG}" .yaml)_${STAMP}.log"

echo "========================================"
echo "  State-Motion Decoupled Alignment Train"
echo "  CODE_DIR: ${CODE_DIR}"
echo "  CONFIG:   ${CONFIG}"
echo "  GPUS:     ${GPUS}"
echo "  CUDA:     ${CUDA_VISIBLE_DEVICES}"
echo "  LOG:      ${LOG_PATH}"
echo "  RESUME:   ${RESUME:-<none>}"
echo "  INIT:     ${INIT_CHECKPOINT:-<none>}"
echo "  TIME:     $(date)"
echo "========================================"

EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi
if [[ -n "${INIT_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--init_checkpoint "${INIT_CHECKPOINT}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun \
  --standalone \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  training/train_carla_bev.py \
  --config_path "${CONFIG}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_PATH}"
