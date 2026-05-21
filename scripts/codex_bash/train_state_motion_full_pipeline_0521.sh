#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:-/workspace1/z_project/code/motdp_z_state_motion_decoupled_alignment_v1}
CONDA_ENV=${CONDA_ENV:-z_dpauto}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5}
GPUS=${GPUS:-4}
S_MASTER_PORT=${S_MASTER_PORT:-29551}
M_MASTER_PORT=${M_MASTER_PORT:-29552}
A2_MASTER_PORT=${A2_MASTER_PORT:-29553}
S_PICK_EPOCH=${S_PICK_EPOCH:-15}

NOSTATE_CKPT=${NOSTATE_CKPT:-/workspace1/z_project/code/motdp_z_semantic_state_strict_ablation_v1/checkpoints/route_b_simple_diffusion_nostate_0519/dit_policy_epoch60.pt}
S_CONFIG=${S_CONFIG:-config/pdm_hpc_route_b_state_motion_s_structured_chain_gt_route_0521.yaml}
M_CONFIG=${M_CONFIG:-config/pdm_hpc_route_b_state_motion_state_conditioned_motion_0521.yaml}
A2_CONFIG=${A2_CONFIG:-config/pdm_hpc_route_b_state_motion_alignment_critic_a2_0521.yaml}

S_DIR=${S_DIR:-${CODE_DIR}/checkpoints/route_b_state_motion_s_structured_chain_0521}
M_DIR=${M_DIR:-${CODE_DIR}/checkpoints/route_b_state_motion_state_conditioned_motion_0521}
A2_DIR=${A2_DIR:-${CODE_DIR}/checkpoints/route_b_state_motion_alignment_critic_a2_0521}
S_CKPT=${S_CKPT:-${S_DIR}/dit_policy_epoch${S_PICK_EPOCH}.pt}
M_CKPT=${M_CKPT:-${M_DIR}/dit_policy_best.pt}

cd "${CODE_DIR}"

if [[ ! -f "${NOSTATE_CKPT}" ]]; then
  echo "Missing nostate init checkpoint: ${NOSTATE_CKPT}" >&2
  exit 1
fi

echo "========== Stage S: structured semantic chain =========="
echo "INIT=${NOSTATE_CKPT}"
CONFIG="${S_CONFIG}" \
INIT_CHECKPOINT="${NOSTATE_CKPT}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
GPUS="${GPUS}" \
MASTER_PORT="${S_MASTER_PORT}" \
CONDA_ENV="${CONDA_ENV}" \
bash scripts/codex_bash/train_state_motion.sh

if [[ ! -f "${S_CKPT}" ]]; then
  echo "Missing S checkpoint for next stage: ${S_CKPT}" >&2
  echo "Set S_PICK_EPOCH or S_CKPT if you want a different S checkpoint." >&2
  exit 1
fi

echo "========== Stage M: state-conditioned motion =========="
echo "INIT=${S_CKPT}"
echo "TEACHER=${S_CKPT}"
CONFIG="${M_CONFIG}" \
INIT_CHECKPOINT="${S_CKPT}" \
SEMANTIC_TEACHER_CHECKPOINT="${S_CKPT}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
GPUS="${GPUS}" \
MASTER_PORT="${M_MASTER_PORT}" \
CONDA_ENV="${CONDA_ENV}" \
bash scripts/codex_bash/train_state_motion.sh

if [[ ! -f "${M_CKPT}" ]]; then
  echo "Missing M checkpoint for A2 stage: ${M_CKPT}" >&2
  echo "Set M_CKPT if the selected M checkpoint is elsewhere." >&2
  exit 1
fi

echo "========== Stage A2: counterfactual compatibility critic =========="
echo "INIT=${M_CKPT}"
echo "TEACHER=${S_CKPT}"
CONFIG="${A2_CONFIG}" \
INIT_CHECKPOINT="${M_CKPT}" \
SEMANTIC_TEACHER_CHECKPOINT="${S_CKPT}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
GPUS="${GPUS}" \
MASTER_PORT="${A2_MASTER_PORT}" \
CONDA_ENV="${CONDA_ENV}" \
bash scripts/codex_bash/train_state_motion.sh

echo "========== Pipeline complete =========="
echo "S_DIR=${S_DIR}"
echo "M_DIR=${M_DIR}"
echo "A2_DIR=${A2_DIR}"
