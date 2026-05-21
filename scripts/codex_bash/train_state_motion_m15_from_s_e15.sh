#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:-$(pwd)}
CONDA_ENV=${CONDA_ENV:-z_dpauto}
CONFIG=${CONFIG:-config/pdm_hpc_route_b_state_motion_adapter_m15_0521.yaml}
GPUS=${GPUS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5}
MASTER_PORT=${MASTER_PORT:-29543}
S_E15_CKPT=${S_E15_CKPT:-/workspace1/z_project/code/motdp_z_state_motion_decoupled_alignment_v1/checkpoints/route_b_state_motion_s_semantic_detached_0521/dit_policy_epoch15.pt}

cd "${CODE_DIR}"

if [[ ! -f "${S_E15_CKPT}" ]]; then
  echo "S e15 checkpoint not found: ${S_E15_CKPT}" >&2
  echo "Set S_E15_CKPT=/path/to/dit_policy_epoch15.pt if the checkpoint lives elsewhere." >&2
  exit 3
fi

echo "========================================"
echo "  Stage M1.5: condition-cleanup adapter from S e15"
echo "  CODE_DIR:    ${CODE_DIR}"
echo "  CONFIG:      ${CONFIG}"
echo "  INIT S e15:  ${S_E15_CKPT}"
echo "  GPUS:        ${GPUS}"
echo "  CUDA:        ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

CONFIG="${CONFIG}" \
INIT_CHECKPOINT="${S_E15_CKPT}" \
GPUS="${GPUS}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
MASTER_PORT="${MASTER_PORT}" \
CONDA_ENV="${CONDA_ENV}" \
bash scripts/codex_bash/train_state_motion.sh
