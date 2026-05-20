#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:-$(pwd)}
CONDA_ENV=${CONDA_ENV:-z_dpauto}
GPUS=${GPUS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29531}
M1_MASTER_PORT=${M1_MASTER_PORT:-$((MASTER_PORT + 1))}

S_CONFIG=${S_CONFIG:-config/pdm_hpc_route_b_state_motion_semantic_detached_0521.yaml}
M1_CONFIG=${M1_CONFIG:-config/pdm_hpc_route_b_state_motion_adapter_m1_0521.yaml}
NOSTATE_CKPT=${NOSTATE_CKPT:-}
S_CKPT=${S_CKPT:-/workspace1/z_project/code/motdp_z_state_motion_decoupled_alignment_v1/checkpoints/route_b_state_motion_s_semantic_detached_0521/dit_policy_best.pt}

SKIP_S=${SKIP_S:-0}
SKIP_M1=${SKIP_M1:-0}

cd "${CODE_DIR}"

if [[ "${SKIP_S}" == "1" && "${SKIP_M1}" == "1" ]]; then
  echo "Both SKIP_S=1 and SKIP_M1=1 are set; nothing to do." >&2
  exit 1
fi

if [[ "${SKIP_S}" != "1" && -z "${NOSTATE_CKPT}" ]]; then
  echo "NOSTATE_CKPT is required for Stage S unless SKIP_S=1" >&2
  echo "Example: NOSTATE_CKPT=/path/to/nostate/dit_policy_best.pt bash scripts/codex_bash/train_state_motion_s_to_m1.sh" >&2
  exit 2
fi

if [[ "${SKIP_S}" != "1" ]]; then
  echo "========================================"
  echo "  Stage S: detached semantic state"
  echo "  CODE_DIR:      ${CODE_DIR}"
  echo "  CONFIG:        ${S_CONFIG}"
  echo "  INIT G CKPT:   ${NOSTATE_CKPT}"
  echo "  EXPECT S CKPT: ${S_CKPT}"
  echo "========================================"
  CONFIG="${S_CONFIG}" \
  INIT_CHECKPOINT="${NOSTATE_CKPT}" \
  GPUS="${GPUS}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  MASTER_PORT="${MASTER_PORT}" \
  CONDA_ENV="${CONDA_ENV}" \
  bash scripts/codex_bash/train_state_motion.sh
fi

if [[ "${SKIP_M1}" == "1" ]]; then
  echo "SKIP_M1=1, stopping after Stage S."
  exit 0
fi

if [[ ! -f "${S_CKPT}" ]]; then
  echo "S checkpoint not found: ${S_CKPT}" >&2
  echo "Set S_CKPT=/path/to/dit_policy_best.pt if Stage S wrote a different checkpoint." >&2
  exit 3
fi

echo "========================================"
echo "  Stage M1: zero-init state-to-motion adapter"
echo "  CODE_DIR:    ${CODE_DIR}"
echo "  CONFIG:      ${M1_CONFIG}"
echo "  INIT S CKPT: ${S_CKPT}"
echo "========================================"
CONFIG="${M1_CONFIG}" \
INIT_CHECKPOINT="${S_CKPT}" \
GPUS="${GPUS}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
MASTER_PORT="${M1_MASTER_PORT}" \
CONDA_ENV="${CONDA_ENV}" \
bash scripts/codex_bash/train_state_motion.sh
