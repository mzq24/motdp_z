#!/usr/bin/env bash
set -euo pipefail

STAGE=${STAGE:?Set STAGE to r0, r1, r2, r3-natural, r3-common, r3t-direct, or r4}
CODE_DIR=${CODE_DIR:-/data/z_project/code/motdp_z_semantic_state_strict_ablation_v1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
GPUS=${GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29620}

case "${STAGE}" in
  r0)
    CONFIG=config/recovery/legacy_e60_r0_clean_cond4_0614.yaml
    TRAINER=training/train_motion_only_clean.py
    ;;
  r1)
    CONFIG=config/recovery/legacy_e60_r1_clean_cond6_0614.yaml
    TRAINER=training/train_motion_only_clean.py
    ;;
  r2)
    CONFIG=config/recovery/legacy_e60_r2_legacy_wrapper_motion_only_0614.yaml
    TRAINER=training/train_carla_bev.py
    ;;
  r3-natural)
    CONFIG=config/recovery/legacy_e60_r3_full_state_frozen_natural_0614.yaml
    TRAINER=training/train_carla_bev.py
    ;;
  r3-common)
    CONFIG=config/recovery/legacy_e60_r3_full_state_frozen_common_init_0614.yaml
    TRAINER=training/train_carla_bev.py
    INIT_CKPT=${CODE_DIR}/checkpoints/legacy_e60_recovery_r2_legacy_wrapper_motion_only_0614/initial_model.pt
    if [[ ! -f "${INIT_CKPT}" ]]; then
      echo "[ERROR] R3 common-init requires ${INIT_CKPT}" >&2
      echo "Run R2 first so its deterministic epoch-0 checkpoint is materialized." >&2
      exit 1
    fi
    ;;
  r3t-direct)
    CONFIG=config/recovery/legacy_e60_r3t_direct_transformer_0625.yaml
    TRAINER=training/train_carla_bev.py
    ;;
  r4)
    CONFIG=config/recovery/legacy_e60_r4_legacy_exact_0614.yaml
    TRAINER=training/train_carla_bev.py
    ;;
  *)
    echo "[ERROR] Unknown STAGE=${STAGE}" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd "${CODE_DIR}"
ulimit -n 65535 || true

echo "stage=${STAGE} config=${CONFIG} trainer=${TRAINER} gpus=${CUDA_VISIBLE_DEVICES}"
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${TRAINER}" --config_path "${CONFIG}"
