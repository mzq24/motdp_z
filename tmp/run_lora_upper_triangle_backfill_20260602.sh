#!/usr/bin/env bash
set -euo pipefail

REPO=/data/z_project/code/nuplan_whitenoise_diffusion_v1
PY=/workspace1/miniconda/envs/z_navsim_motdp/bin/python
OUT_ROOT=/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/backfill_lora_cl7_upper_triangle_20260602
SCRIPT=${REPO}/tmp/backfill_cl7_upper_triangle.py

NORMAL_CONFIG=${REPO}/tmp/nuplan_diffusion_target_cl7_monotonic_norm_lora_r16_gpu0123_bs256_20260601.yaml
PEGP_CONFIG=${REPO}/tmp/nuplan_diffusion_target_cl7_monotonic_pegp_lora_r16_gpu4567_bs256_20260601.yaml
NORMAL_EXP=/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/target_cl7_monotonic_norm_lora_r16_gpu0123_bs256_20260601
PEGP_EXP=/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/target_cl7_monotonic_pegp_lora_r16_gpu4567_bs256_20260601

TASKS=(
  starting_straight_traffic_light_intersection_traversal
  following_lane_with_lead
  high_lateral_acceleration
  near_multiple_vehicles
  waiting_for_pedestrian_to_cross
  traversing_pickup_dropoff
  stationary_in_traffic
)

mkdir -p "${OUT_ROOT}/normal/parts" "${OUT_ROOT}/pegp/parts" "${OUT_ROOT}/logs"
: > "${OUT_ROOT}/logs/runner.log"
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

run_worker() {
  local label=$1
  local config=$2
  local exp_dir=$3
  local parts_dir=$4
  local gpu=$5
  shift 5
  local idx task ckpt out log
  for idx in "$@"; do
    task=${TASKS[$((idx - 1))]}
    ckpt="${exp_dir}/checkpoints/task_$(printf "%02d" "${idx}")_${task}_epoch_0059.pth"
    out="${parts_dir}/task_$(printf "%02d" "${idx}").json"
    log="${OUT_ROOT}/logs/${label}_task_$(printf "%02d" "${idx}").log"
    if [[ ! -f "${ckpt}" ]]; then
      echo "missing checkpoint: ${ckpt}" >&2
      exit 1
    fi
    echo "[$(date +%F_%T)] ${label} GPU${gpu} eval after task ${idx}: ${task}" | tee -a "${OUT_ROOT}/logs/runner.log"
    CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${SCRIPT}" eval-one \
      --config "${config}" \
      --checkpoint "${ckpt}" \
      --completed-task-index "${idx}" \
      --current-task "${task}" \
      --run-label "${label}" \
      --output-json "${out}" \
      --max-batches 10 \
      --val-batch-size 64 \
      --num-workers 4 >"${log}" 2>&1
  done
}

run_worker normal_lora_r16 "${NORMAL_CONFIG}" "${NORMAL_EXP}" "${OUT_ROOT}/normal/parts" 0 1 5 &
run_worker normal_lora_r16 "${NORMAL_CONFIG}" "${NORMAL_EXP}" "${OUT_ROOT}/normal/parts" 1 2 6 &
run_worker normal_lora_r16 "${NORMAL_CONFIG}" "${NORMAL_EXP}" "${OUT_ROOT}/normal/parts" 2 3 7 &
run_worker normal_lora_r16 "${NORMAL_CONFIG}" "${NORMAL_EXP}" "${OUT_ROOT}/normal/parts" 3 4 &
run_worker pegp_lora_r16 "${PEGP_CONFIG}" "${PEGP_EXP}" "${OUT_ROOT}/pegp/parts" 4 1 5 &
run_worker pegp_lora_r16 "${PEGP_CONFIG}" "${PEGP_EXP}" "${OUT_ROOT}/pegp/parts" 5 2 6 &
run_worker pegp_lora_r16 "${PEGP_CONFIG}" "${PEGP_EXP}" "${OUT_ROOT}/pegp/parts" 6 3 7 &
run_worker pegp_lora_r16 "${PEGP_CONFIG}" "${PEGP_EXP}" "${OUT_ROOT}/pegp/parts" 7 4 &
wait

"${PY}" "${SCRIPT}" merge \
  --config "${NORMAL_CONFIG}" \
  --parts-dir "${OUT_ROOT}/normal/parts" \
  --run-label normal_lora_r16 \
  --output-json "${OUT_ROOT}/normal/upper_triangle_validation.json" \
  --output-md "${OUT_ROOT}/normal/upper_triangle_validation.md"

"${PY}" "${SCRIPT}" merge \
  --config "${PEGP_CONFIG}" \
  --parts-dir "${OUT_ROOT}/pegp/parts" \
  --run-label pegp_lora_r16 \
  --output-json "${OUT_ROOT}/pegp/upper_triangle_validation.json" \
  --output-md "${OUT_ROOT}/pegp/upper_triangle_validation.md"

"${PY}" "${SCRIPT}" compare \
  --normal-json "${OUT_ROOT}/normal/upper_triangle_validation.json" \
  --pegp-json "${OUT_ROOT}/pegp/upper_triangle_validation.json" \
  --output-md "${OUT_ROOT}/lora_upper_triangle_comparison.md"

echo "[$(date +%F_%T)] done: ${OUT_ROOT}"
