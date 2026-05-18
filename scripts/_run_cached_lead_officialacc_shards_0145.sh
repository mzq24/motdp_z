#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/workspace1/z_project/code/motdp_z_navsim_motdp}
SHARD_DIR=${SHARD_DIR:-/workspace2/z_project/tmp/navtest_log_shards_ddim_eta0}
LOG_DIR=${LOG_DIR:-/workspace2/z_project/motdp_logs}
EXP_PREFIX=${EXP_PREFIX:-motdp_cached_lead_officialacc_full}
GPUS_STR=${GPUS_STR:-"0 1 4 5"}
WORKERS=${WORKERS:-1}
CACHE_DIR=${CACHE_DIR:-/workspace2/z_project/motdp_bev_cache_navtest_official4cam_officialacc_npy}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/workspace1/z_project/models/navsim_backbones/tfv6_navsim/model_0060.pth}
METRIC_CACHE_PATH=${METRIC_CACHE_PATH:-/workspace2/z_project/navsim_exp_lead_official/metric_cache_navtest_v1_1}
NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/workspace2/z_project/navsim_exp_motdp}

mkdir -p "${LOG_DIR}"
read -r -a GPUS <<< "${GPUS_STR}"
NUM_SHARDS=${#GPUS[@]}
if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "No GPUs specified" >&2
  exit 1
fi

pids=()
for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}
  shard_file="${SHARD_DIR}/shard${i}.json"
  if [[ ! -f "${shard_file}" ]]; then
    echo "Missing shard file: ${shard_file}" >&2
    exit 1
  fi
  logs=$(cat "${shard_file}")
  exp_name="${EXP_PREFIX}_shard${i}_of${NUM_SHARDS}_gpu${gpu}"
  out_log="${LOG_DIR}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
  echo "[launch] shard=${i}/${NUM_SHARDS} gpu=${gpu} log=${out_log}"
  (
    cd "${REPO}"
    GPU_ID="${gpu}"       CACHE_DIR="${CACHE_DIR}"       CHECKPOINT_PATH="${CHECKPOINT_PATH}"       METRIC_CACHE_PATH="${METRIC_CACHE_PATH}"       NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT}"       EXPERIMENT_NAME="${exp_name}"       MAX_SCENES=       scripts/_run_cached_lead_officialacc_smoke.sh       "train_test_split.scene_filter.log_names=${logs}"
  ) > "${out_log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "[done] status=${status}"
exit "${status}"
