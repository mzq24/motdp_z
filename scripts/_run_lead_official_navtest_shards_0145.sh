#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/workspace1/z_project/code/motdp_z_navsim_motdp}
SHARD_DIR=${SHARD_DIR:-/workspace2/z_project/tmp/navtest_log_shards_ddim_eta0}
LOG_DIR=${LOG_DIR:-/workspace2/z_project/navsim_exp_lead_official/logs}
EXP_PREFIX=${EXP_PREFIX:-lead_official_navtest_full_model0060_0145}
GPUS_STR=${GPUS_STR:-"0 1 4 5"}
WORKERS=${WORKERS:-1}

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
  runtime_ckpt_dir="/workspace2/z_project/navsim_exp_lead_official/checkpoints/tfv6_navsim_model0060_config_ordered_${exp_name}"

  echo "[launch] shard=${i}/${NUM_SHARDS} gpu=${gpu} log=${out_log}"
  (
    cd "${REPO}"
    GPU_ID="${gpu}"       MODE=full       WORKERS="${WORKERS}"       MAX_SCENES=       EXPERIMENT_NAME="${exp_name}"       LOG_NAMES_JSON="${logs}"       RUNTIME_CHECKPOINT_DIR="${runtime_ckpt_dir}"       scripts/_run_lead_official_navtest.sh
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
