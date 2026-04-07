#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"

B2D_ROOT="${B2D_ROOT:-/media/z/data/mzq/others/Bench2Drive}"
DPAUTO_BIN="${DPAUTO_BIN:-/home/z/anaconda3/envs/dpauto/bin}"
TEAM_CONFIG="${TEAM_CONFIG:-${PROJECT_ROOT}/config/tmp/pdm_local_route_b_lidar_bev_stage1_epoch145.yaml}"

BASE_PORT="${BASE_PORT:-30000}"
BASE_TM_PORT="${BASE_TM_PORT:-50000}"
IS_BENCH2DRIVE=True
BASE_ROUTES="${BASE_ROUTES:-${B2D_ROOT}/leaderboard/data/bench2drive220}"
FULL_ROUTES_XML="${FULL_ROUTES_XML:-${BASE_ROUTES}.xml}"
TEAM_AGENT="${TEAM_AGENT:-${PROJECT_ROOT}/team_code/route_b_b2d_agent.py}"
BASE_CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT:-eval}"
PLANNER_TYPE="${PLANNER_TYPE:-epoch145_stage1}"
ALGO="${ALGO:-route_b}"
DATE="${DATE:-0407_epoch145}"
SAVE_PATH="${SAVE_PATH:-./eval_M_${ALGO}_${DATE}}"

TARGET_POSE_SOURCE=${TARGET_POSE_SOURCE:-filtered}
LOCALIZER_STRATEGY=${LOCALIZER_STRATEGY:-complementary}
LOCALIZER_ALPHA=${LOCALIZER_ALPHA:-0.5}
LIDAR_POSE_SOURCE=${LIDAR_POSE_SOURCE:-ukf}
STEER_SIGN_SCALE=${STEER_SIGN_SCALE:-1}
TARGET_YAW_SIGN=${TARGET_YAW_SIGN:-1}
TARGET_GEOM_YAW_SIGN=${TARGET_GEOM_YAW_SIGN:-1}
SOFT_SPEED_LIMIT_MS=${SOFT_SPEED_LIMIT_MS:-10}
HARD_SPEED_LIMIT_MS=${HARD_SPEED_LIMIT_MS:-0}
NUM_INFERENCE_STEPS_OVERRIDE=${NUM_INFERENCE_STEPS_OVERRIDE:-}
SPEED_SOURCE=${SPEED_SOURCE:-}

GPU_RANK_LIST=(${GPU_RANK_LIST:-0 0})
TASK_LIST=(${TASK_LIST:-0 1})
ROUTES_SUBSET_LIST="${ROUTES_SUBSET_LIST:-}"
ROUTES_SUBSET_ARR=()
if [[ -n "${ROUTES_SUBSET_LIST}" ]]; then
  read -r -a ROUTES_SUBSET_ARR <<< "${ROUTES_SUBSET_LIST}"
fi

if [[ ! -d "${B2D_ROOT}" ]]; then
  echo "[ERROR] Bench2Drive root not found: ${B2D_ROOT}"
  exit 1
fi
if [[ ! -f "${TEAM_CONFIG}" ]]; then
  echo "[ERROR] TEAM_CONFIG not found: ${TEAM_CONFIG}"
  exit 1
fi
if [[ ! -f "${TEAM_AGENT}" ]]; then
  echo "[ERROR] TEAM_AGENT not found: ${TEAM_AGENT}"
  exit 1
fi
if [[ ! -x "${DPAUTO_BIN}/python" ]]; then
  echo "[ERROR] dpauto python not found: ${DPAUTO_BIN}/python"
  exit 1
fi

cd "${B2D_ROOT}"
export PATH="${DPAUTO_BIN}:$PATH"
mkdir -p "${ALGO}_b2d_${DATE}"

if [[ ${#ROUTES_SUBSET_ARR[@]} -gt 0 ]]; then
  echo "[INFO] ROUTES_SUBSET_LIST detected: ${ROUTES_SUBSET_ARR[*]}"
  echo "[INFO] Skipping split_xml and using full routes file: ${FULL_ROUTES_XML}"
else
  SPLIT_FLAG="${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"
  if [[ ! -f "${SPLIT_FLAG}" ]]; then
    echo "[INFO] Splitting routes into 2 tasks..."
    python tools/split_xml.py "${BASE_ROUTES}" 2 "${ALGO}" "${PLANNER_TYPE}"
    touch "${SPLIT_FLAG}"
  fi
fi

echo "========================================"
echo "  Route-B epoch145 local evaluation"
echo "  PROJECT_ROOT:   ${PROJECT_ROOT}"
echo "  B2D_ROOT:       ${B2D_ROOT}"
echo "  TEAM_AGENT:     ${TEAM_AGENT}"
echo "  TEAM_CONFIG:    ${TEAM_CONFIG}"
echo "  SAVE_PATH:      ${SAVE_PATH}"
echo "  PLANNER_TYPE:   ${PLANNER_TYPE}"
echo "  TARGET_POSE:    ${TARGET_POSE_SOURCE}"
echo "  LOC/LIDAR:      ${LOCALIZER_STRATEGY} a=${LOCALIZER_ALPHA} lidar=${LIDAR_POSE_SOURCE}"
echo "  SPEED_SOURCE:   ${SPEED_SOURCE}"
echo "  SOFT CAP:       ${SOFT_SPEED_LIMIT_MS}"
echo "  HARD CAP:       ${HARD_SPEED_LIMIT_MS}"
echo "  INF STEPS:      ${NUM_INFERENCE_STEPS_OVERRIDE}"
echo "  GPU_RANK_LIST:  ${GPU_RANK_LIST[*]}"
echo "  TASK_LIST:      ${TASK_LIST[*]}"
if [[ ${#ROUTES_SUBSET_ARR[@]} -gt 0 ]]; then
  echo "  ROUTE IDS:      ${ROUTES_SUBSET_ARR[*]}"
fi
echo "========================================"

length=${#GPU_RANK_LIST[@]}
for ((i=0; i<length; i++)); do
  PORT=$((BASE_PORT + i * 150))
  TM_PORT=$((BASE_TM_PORT + i * 150))
  ROUTES_SUBSET=""
  if [[ ${#ROUTES_SUBSET_ARR[@]} -gt 0 ]]; then
    ROUTES="${FULL_ROUTES_XML}"
    ROUTES_SUBSET="${ROUTES_SUBSET_ARR[$i]:-}"
    ROUTE_TAG="${ROUTES_SUBSET:-task${TASK_LIST[$i]}}"
  else
    ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.xml"
    ROUTE_TAG="${TASK_LIST[$i]}"
  fi
  CHECKPOINT_ENDPOINT="${ALGO}_b2d_${DATE}/${BASE_CHECKPOINT_ENDPOINT}_${ROUTE_TAG}.json"
  GPU_RANK="${GPU_RANK_LIST[$i]}"

  echo "[TASK ${i}] GPU=${GPU_RANK} PORT=${PORT} TM_PORT=${TM_PORT}"
  echo "[TASK ${i}] ROUTES=${ROUTES}"
  if [[ -n "${ROUTES_SUBSET}" ]]; then
    echo "[TASK ${i}] ROUTES_SUBSET=${ROUTES_SUBSET}"
  fi
  echo "[TASK ${i}] CHECKPOINT_ENDPOINT=${CHECKPOINT_ENDPOINT}"

  TARGET_POSE_SOURCE="${TARGET_POSE_SOURCE}" \
  LOCALIZER_STRATEGY="${LOCALIZER_STRATEGY}" \
  LOCALIZER_ALPHA="${LOCALIZER_ALPHA}" \
  LIDAR_POSE_SOURCE="${LIDAR_POSE_SOURCE}" \
  STEER_SIGN_SCALE="${STEER_SIGN_SCALE}" \
  TARGET_YAW_SIGN="${TARGET_YAW_SIGN}" \
  TARGET_GEOM_YAW_SIGN="${TARGET_GEOM_YAW_SIGN}" \
  SOFT_SPEED_LIMIT_MS="${SOFT_SPEED_LIMIT_MS}" \
  HARD_SPEED_LIMIT_MS="${HARD_SPEED_LIMIT_MS}" \
  NUM_INFERENCE_STEPS_OVERRIDE="${NUM_INFERENCE_STEPS_OVERRIDE}" \
  SPEED_SOURCE="${SPEED_SOURCE}" \
  bash -e leaderboard/scripts/run_evaluation.sh \
    "${PORT}" "${TM_PORT}" "${IS_BENCH2DRIVE}" "${ROUTES}" \
    "${TEAM_AGENT}" "${TEAM_CONFIG}" \
    "${CHECKPOINT_ENDPOINT}" "${SAVE_PATH}" \
    "${PLANNER_TYPE}" "${GPU_RANK}" "${ROUTES_SUBSET}" \
    > "${BASE_ROUTES}_${ROUTE_TAG}_${ALGO}_${PLANNER_TYPE}.log" 2>&1 &

  sleep 10
done

wait
