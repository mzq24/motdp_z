#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:-/data/z_project/code/motdp_z_semantic_state_strict_ablation_v1}
CONFIG=${CONFIG:-config/paper_motion_only_core_oldddim_0531.yaml}
GPUS=${GPUS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
MASTER_PORT=${MASTER_PORT:-29597}
RESUME=${RESUME:-}
VAL_ONLY=${VAL_ONLY:-0}

export CUDA_VISIBLE_DEVICES
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd "${CODE_DIR}"
ulimit -n 65535 || true

ARGS=(--config_path "${CONFIG}")
if [[ -n "${RESUME}" ]]; then
  ARGS+=(--resume "${RESUME}")
fi
if [[ "${VAL_ONLY}" == "1" ]]; then
  ARGS+=(--val_only)
fi

python -m torch.distributed.run   --standalone   --nproc_per_node="${GPUS}"   --master_port="${MASTER_PORT}"   training/train_motion_only_clean.py "${ARGS[@]}"
