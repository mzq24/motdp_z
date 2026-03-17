#!/usr/bin/env bash
set -euo pipefail

# 直接在这里配置，不需要命令行传参
CKPT_PATH="server_folder/MoT-DP/checkpoints/transfuser_dp/dit_policy_best.pt"
CONFIG_PATH="config/pdm_local.yaml"
NUM_GPUS="1"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}"
  exit 1
fi

# 如需联网同步可改为 online
export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  echo "[INFO] Running distributed validation with ${NUM_GPUS} GPUs"
  torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    training/train_carla_bev.py \
    --config_path "${CONFIG_PATH}" \
    --resume "${CKPT_PATH}" \
    --val_only
else
  echo "[INFO] Running single-GPU validation"
  python -m debugpy --listen 0.0.0.0:5678 --wait-for-client training/train_carla_bev.py \
    --config_path "${CONFIG_PATH}" \
    --resume "${CKPT_PATH}" \
    --val_only
fi
