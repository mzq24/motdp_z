#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:-/workspace1/z_project/code/motdp_z_state_motion_decoupled_alignment_v1}
CONDA_ENV=${CONDA_ENV:-z_dpauto}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5}
GPUS=${GPUS:-4}
DIRECT_S_MASTER_PORT=${DIRECT_S_MASTER_PORT:-29561}
DIRECT_M_MASTER_PORT=${DIRECT_M_MASTER_PORT:-29562}
DIRECT_A2_MASTER_PORT=${DIRECT_A2_MASTER_PORT:-29565}
DIRECT_M2_MASTER_PORT=${DIRECT_M2_MASTER_PORT:-29566}
CHAIN_S_MASTER_PORT=${CHAIN_S_MASTER_PORT:-29563}
CHAIN_M_MASTER_PORT=${CHAIN_M_MASTER_PORT:-29564}
CHAIN_A2_MASTER_PORT=${CHAIN_A2_MASTER_PORT:-29567}
CHAIN_M2_MASTER_PORT=${CHAIN_M2_MASTER_PORT:-29568}
S_PICK_EPOCH=${S_PICK_EPOCH:-15}
RUN_DIRECT=${RUN_DIRECT:-1}
RUN_CHAIN=${RUN_CHAIN:-1}
RUN_A2=${RUN_A2:-1}
RUN_M2=${RUN_M2:-0}

NOSTATE_CKPT=${NOSTATE_CKPT:-/workspace1/z_project/code/motdp_z_semantic_state_strict_ablation_v1/checkpoints/route_b_simple_diffusion_nostate_0519/dit_policy_epoch60.pt}

S_DIRECT_CONFIG=${S_DIRECT_CONFIG:-config/pdm_hpc_route_b_state_motion_s_direct_gt_route_0522.yaml}
M_DIRECT_CONFIG=${M_DIRECT_CONFIG:-config/pdm_hpc_route_b_state_motion_state_conditioned_motion_direct_0522.yaml}
A2_DIRECT_CONFIG=${A2_DIRECT_CONFIG:-config/pdm_hpc_route_b_state_motion_alignment_critic_a2_direct_0522.yaml}
M2_DIRECT_CONFIG=${M2_DIRECT_CONFIG:-config/pdm_hpc_route_b_state_motion_m2_a2_rerank_direct_0522.yaml}
S_CHAIN_CONFIG=${S_CHAIN_CONFIG:-config/pdm_hpc_route_b_state_motion_s_chain_hist4_gt_route_0522.yaml}
M_CHAIN_CONFIG=${M_CHAIN_CONFIG:-config/pdm_hpc_route_b_state_motion_state_conditioned_motion_chain_hist4_0522.yaml}
A2_CHAIN_CONFIG=${A2_CHAIN_CONFIG:-config/pdm_hpc_route_b_state_motion_alignment_critic_a2_chain_hist4_0522.yaml}
M2_CHAIN_CONFIG=${M2_CHAIN_CONFIG:-config/pdm_hpc_route_b_state_motion_m2_a2_rerank_chain_hist4_0522.yaml}

S_DIRECT_DIR=${S_DIRECT_DIR:-${CODE_DIR}/checkpoints/route_b_state_motion_s_direct_0522}
M_DIRECT_DIR=${M_DIRECT_DIR:-${CODE_DIR}/checkpoints/route_b_state_motion_state_conditioned_motion_direct_0522}
A2_DIRECT_DIR=${A2_DIRECT_DIR:-${CODE_DIR}/checkpoints/route_b_state_motion_alignment_critic_a2_direct_0522}
M2_DIRECT_DIR=${M2_DIRECT_DIR:-${CODE_DIR}/checkpoints/route_b_state_motion_m2_a2_rerank_direct_0522}
S_CHAIN_DIR=${S_CHAIN_DIR:-${CODE_DIR}/checkpoints/route_b_state_motion_s_chain_hist4_0522}
M_CHAIN_DIR=${M_CHAIN_DIR:-${CODE_DIR}/checkpoints/route_b_state_motion_state_conditioned_motion_chain_hist4_0522}
A2_CHAIN_DIR=${A2_CHAIN_DIR:-${CODE_DIR}/checkpoints/route_b_state_motion_alignment_critic_a2_chain_hist4_0522}
M2_CHAIN_DIR=${M2_CHAIN_DIR:-${CODE_DIR}/checkpoints/route_b_state_motion_m2_a2_rerank_chain_hist4_0522}

S_DIRECT_CKPT=${S_DIRECT_CKPT:-${S_DIRECT_DIR}/dit_policy_epoch${S_PICK_EPOCH}.pt}
M_DIRECT_CKPT=${M_DIRECT_CKPT:-${M_DIRECT_DIR}/dit_policy_best.pt}
A2_DIRECT_CKPT=${A2_DIRECT_CKPT:-${A2_DIRECT_DIR}/dit_policy_best.pt}
S_CHAIN_CKPT=${S_CHAIN_CKPT:-${S_CHAIN_DIR}/dit_policy_epoch${S_PICK_EPOCH}.pt}
M_CHAIN_CKPT=${M_CHAIN_CKPT:-${M_CHAIN_DIR}/dit_policy_best.pt}
A2_CHAIN_CKPT=${A2_CHAIN_CKPT:-${A2_CHAIN_DIR}/dit_policy_best.pt}

cd "${CODE_DIR}"

if [[ ! -f "${NOSTATE_CKPT}" ]]; then
  echo "Missing nostate init checkpoint: ${NOSTATE_CKPT}" >&2
  exit 1
fi

run_stage() {
  local name="$1"
  local config="$2"
  local init_ckpt="$3"
  local master_port="$4"
  local teacher_ckpt="${5:-}"
  local critic_ckpt="${6:-}"
  echo "========== ${name} =========="
  echo "CONFIG=${config}"
  echo "INIT=${init_ckpt}"
  [[ -n "${teacher_ckpt}" ]] && echo "TEACHER=${teacher_ckpt}"
  [[ -n "${critic_ckpt}" ]] && echo "CRITIC=${critic_ckpt}"
  CONFIG="${config}"   INIT_CHECKPOINT="${init_ckpt}"   SEMANTIC_TEACHER_CHECKPOINT="${teacher_ckpt}"   ALIGNMENT_CRITIC_CHECKPOINT_OVERRIDE="${critic_ckpt}"   CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"   GPUS="${GPUS}"   MASTER_PORT="${master_port}"   CONDA_ENV="${CONDA_ENV}"   bash scripts/codex_bash/train_state_motion.sh
}

if [[ "${RUN_DIRECT}" == "1" ]]; then
  run_stage "Direct Stage S" "${S_DIRECT_CONFIG}" "${NOSTATE_CKPT}" "${DIRECT_S_MASTER_PORT}"
  [[ -f "${S_DIRECT_CKPT}" ]] || { echo "Missing direct S checkpoint: ${S_DIRECT_CKPT}" >&2; exit 1; }
  run_stage "Direct Stage M" "${M_DIRECT_CONFIG}" "${S_DIRECT_CKPT}" "${DIRECT_M_MASTER_PORT}" "${S_DIRECT_CKPT}"
  if [[ "${RUN_A2}" == "1" ]]; then
    [[ -f "${M_DIRECT_CKPT}" ]] || { echo "Missing direct M checkpoint: ${M_DIRECT_CKPT}" >&2; exit 1; }
    run_stage "Direct Stage A2" "${A2_DIRECT_CONFIG}" "${M_DIRECT_CKPT}" "${DIRECT_A2_MASTER_PORT}" "${S_DIRECT_CKPT}"
  fi
  if [[ "${RUN_M2}" == "1" ]]; then
    [[ -f "${M_DIRECT_CKPT}" ]] || { echo "Missing direct M checkpoint: ${M_DIRECT_CKPT}" >&2; exit 1; }
    [[ -f "${A2_DIRECT_CKPT}" ]] || { echo "Missing direct A2 checkpoint: ${A2_DIRECT_CKPT}" >&2; exit 1; }
    run_stage "Direct Stage M2" "${M2_DIRECT_CONFIG}" "${M_DIRECT_CKPT}" "${DIRECT_M2_MASTER_PORT}" "${S_DIRECT_CKPT}" "${A2_DIRECT_CKPT}"
  fi
fi

if [[ "${RUN_CHAIN}" == "1" ]]; then
  run_stage "Chain Stage S" "${S_CHAIN_CONFIG}" "${NOSTATE_CKPT}" "${CHAIN_S_MASTER_PORT}"
  [[ -f "${S_CHAIN_CKPT}" ]] || { echo "Missing chain S checkpoint: ${S_CHAIN_CKPT}" >&2; exit 1; }
  run_stage "Chain Stage M" "${M_CHAIN_CONFIG}" "${S_CHAIN_CKPT}" "${CHAIN_M_MASTER_PORT}" "${S_CHAIN_CKPT}"
  if [[ "${RUN_A2}" == "1" ]]; then
    [[ -f "${M_CHAIN_CKPT}" ]] || { echo "Missing chain M checkpoint: ${M_CHAIN_CKPT}" >&2; exit 1; }
    run_stage "Chain Stage A2" "${A2_CHAIN_CONFIG}" "${M_CHAIN_CKPT}" "${CHAIN_A2_MASTER_PORT}" "${S_CHAIN_CKPT}"
  fi
  if [[ "${RUN_M2}" == "1" ]]; then
    [[ -f "${M_CHAIN_CKPT}" ]] || { echo "Missing chain M checkpoint: ${M_CHAIN_CKPT}" >&2; exit 1; }
    [[ -f "${A2_CHAIN_CKPT}" ]] || { echo "Missing chain A2 checkpoint: ${A2_CHAIN_CKPT}" >&2; exit 1; }
    run_stage "Chain Stage M2" "${M2_CHAIN_CONFIG}" "${M_CHAIN_CKPT}" "${CHAIN_M2_MASTER_PORT}" "${S_CHAIN_CKPT}" "${A2_CHAIN_CKPT}"
  fi
fi

echo "========== Direct vs chain pipeline complete =========="
echo "S_DIRECT_DIR=${S_DIRECT_DIR}"
echo "M_DIRECT_DIR=${M_DIRECT_DIR}"
echo "A2_DIRECT_DIR=${A2_DIRECT_DIR}"
echo "M2_DIRECT_DIR=${M2_DIRECT_DIR}"
echo "S_CHAIN_DIR=${S_CHAIN_DIR}"
echo "M_CHAIN_DIR=${M_CHAIN_DIR}"
echo "A2_CHAIN_DIR=${A2_CHAIN_DIR}"
echo "M2_CHAIN_DIR=${M2_CHAIN_DIR}"
