#!/usr/bin/env bash
set -euo pipefail

STAGE=${STAGE:?Set STAGE to r0, r1, r2, r3-natural, r3-common, r3t-direct, or r4}
DATA_ROOT=${DATA_ROOT:-/data/z_project}
CODE_DIR=${CODE_DIR:-${DATA_ROOT}/code/motdp_z_semantic_state_strict_ablation_v1}
B2D_ROOT=${B2D_ROOT:-${DATA_ROOT}/code/Bench2Drive}
CARLA_ROOT=${CARLA_ROOT:-${DATA_ROOT}/carla}
CONDA_SH=${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-z_dpauto}

case "${STAGE}" in
  r0)
    CONFIG_REL=config/recovery/legacy_e60_r0_clean_cond4_0614.yaml
    CKPT_DIR=checkpoints/legacy_e60_recovery_r0_clean_cond4_0614
    DEFAULT_EPOCH=45
    ;;
  r1)
    CONFIG_REL=config/recovery/legacy_e60_r1_clean_cond6_0614.yaml
    CKPT_DIR=checkpoints/legacy_e60_recovery_r1_clean_cond6_0614
    DEFAULT_EPOCH=60
    ;;
  r2)
    CONFIG_REL=config/recovery/legacy_e60_r2_legacy_wrapper_motion_only_0614.yaml
    CKPT_DIR=checkpoints/legacy_e60_recovery_r2_legacy_wrapper_motion_only_0614
    DEFAULT_EPOCH=55
    ;;
  r3-natural)
    CONFIG_REL=config/recovery/legacy_e60_r3_full_state_frozen_natural_0614.yaml
    CKPT_DIR=checkpoints/legacy_e60_recovery_r3_full_state_frozen_natural_0614
    DEFAULT_EPOCH=55
    ;;
  r3-common)
    CONFIG_REL=config/recovery/legacy_e60_r3_full_state_frozen_common_init_0614.yaml
    CKPT_DIR=checkpoints/legacy_e60_recovery_r3_full_state_frozen_common_init_0614
    DEFAULT_EPOCH=55
    ;;
  r3t-direct)
    CONFIG_REL=config/recovery/legacy_e60_r3t_direct_transformer_0625.yaml
    CKPT_DIR=checkpoints/legacy_e60_recovery_r3t_direct_transformer_0625
    DEFAULT_EPOCH=60
    ;;
  r4)
    CONFIG_REL=config/recovery/legacy_e60_r4_legacy_exact_0614.yaml
    CKPT_DIR=checkpoints/legacy_e60_recovery_r4_legacy_exact_0614
    DEFAULT_EPOCH=55
    ;;
  *)
    echo "[ERROR] Unknown STAGE=${STAGE}" >&2
    exit 2
    ;;
esac

CKPT_EPOCH=${CKPT_EPOCH:-${DEFAULT_EPOCH}}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-dit_policy_epoch${CKPT_EPOCH}.pt}
TEAM_CONFIG=${TEAM_CONFIG:-${CODE_DIR}/${CONFIG_REL}}
TEAM_AGENT=${TEAM_AGENT:-${CODE_DIR}/team_code/route_b_b2d_agent.py}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${CODE_DIR}/${CKPT_DIR}/${CHECKPOINT_NAME}}
CHECKPOINT_PATH_OVERRIDE=${CHECKPOINT_PATH_OVERRIDE:-${CHECKPOINT_PATH}}

BASE_ROUTES=${BASE_ROUTES:-${B2D_ROOT}/leaderboard/data/bench2drive220_skip_23695_24071}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
GPU_RANK_LIST=${GPU_RANK_LIST:-"0 0 1 1 2 2 3 3"}
TASK_NUM=${TASK_NUM:-8}
TASK_LIST=${TASK_LIST:-"0 1 2 3 4 5 6 7"}
NUM_INFERENCE_STEPS_OVERRIDE=${NUM_INFERENCE_STEPS_OVERRIDE:-10}

STAGE_TAG=${STAGE//-/_}
RUN_TAG=${RUN_TAG:-recovery_${STAGE_TAG}_e${CKPT_EPOCH}_skip23695_24071}
DATE=${DATE:-0615_${RUN_TAG}}
PLANNER_TYPE=${PLANNER_TYPE:-${RUN_TAG}}
RESULT_DIR=${RESULT_DIR:-${B2D_ROOT}/eval/results/route_b_b2d_${DATE}}
SAVE_PATH=${SAVE_PATH:-${B2D_ROOT}/eval/debug/eval_M_route_b_${DATE}}
LOG_DIR=${LOG_DIR:-${B2D_ROOT}/bash_commands/logs}

SPEED_SOURCE=${SPEED_SOURCE:-speed_head}
SOFT_SPEED_LIMIT_MS=${SOFT_SPEED_LIMIT_MS:-10}
HARD_SPEED_LIMIT_MS=${HARD_SPEED_LIMIT_MS:-0}
STAGE1_ENERGY_SPEED_CAP_ENABLE=${STAGE1_ENERGY_SPEED_CAP_ENABLE:-0}
WINDOW_SOFT_SPEED_CAP_ENABLE=${WINDOW_SOFT_SPEED_CAP_ENABLE:-0}
USE_CHASE_FRONT_FOLLOWING_STATE=${USE_CHASE_FRONT_FOLLOWING_STATE:-0}
CHASE_SPEED_CAP_ENABLE=${CHASE_SPEED_CAP_ENABLE:-0}
FRONT_ROUTE_RISK_SPEED_CAP_ENABLE=${FRONT_ROUTE_RISK_SPEED_CAP_ENABLE:-0}

for path in "${TEAM_CONFIG}" "${TEAM_AGENT}" "${CHECKPOINT_PATH_OVERRIDE}" "${BASE_ROUTES}.xml"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] required path missing: ${path}" >&2
    exit 1
  fi
done

export CODE_DIR B2D_ROOT CARLA_ROOT CONDA_SH CONDA_ENV
export TEAM_CONFIG TEAM_AGENT CHECKPOINT_PATH CHECKPOINT_PATH_OVERRIDE
export BASE_ROUTES CUDA_VISIBLE_DEVICES GPU_RANK_LIST TASK_NUM TASK_LIST
export NUM_INFERENCE_STEPS_OVERRIDE DATE PLANNER_TYPE RESULT_DIR SAVE_PATH LOG_DIR
export SPEED_SOURCE SOFT_SPEED_LIMIT_MS HARD_SPEED_LIMIT_MS
export STAGE1_ENERGY_SPEED_CAP_ENABLE WINDOW_SOFT_SPEED_CAP_ENABLE
export USE_CHASE_FRONT_FOLLOWING_STATE CHASE_SPEED_CAP_ENABLE
export FRONT_ROUTE_RISK_SPEED_CAP_ENABLE

echo "========================================"
echo " Legacy e60 recovery close-loop"
echo " STAGE:       ${STAGE}"
echo " CONFIG:      ${TEAM_CONFIG}"
echo " CHECKPOINT:  ${CHECKPOINT_PATH_OVERRIDE}"
echo " ROUTES:      ${BASE_ROUTES}"
echo " RESULT_DIR:  ${RESULT_DIR}"
echo " GPUS:        ${CUDA_VISIBLE_DEVICES}"
echo " TASKS:       ${TASK_LIST}"
echo "========================================"

bash "${CODE_DIR}/scripts/hpc_new/run_route_b_lidar_stage1_next05_closeloop.sh"
