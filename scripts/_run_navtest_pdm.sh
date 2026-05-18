#!/bin/bash
set -euo pipefail

if [[ -z "${WORKTREE_ROOT:-}" ]]; then
  SOURCE_PATH="${BASH_SOURCE[0]:-$0}"
  SCRIPT_DIR="$(cd "$(dirname "${SOURCE_PATH}")" && pwd)"
  WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp

export TMPDIR=/workspace2/z_project/tmp
mkdir -p "$TMPDIR"
export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps
export NAVSIM_EXP_ROOT=/workspace2/z_project/navsim_exp_motdp
export HYDRA_FULL_ERROR=1

SCORER_FLAVOR=${SCORER_FLAVOR:-v1_1}
LEAD_PROJECT_ROOT=${LEAD_PROJECT_ROOT:-/workspace1/z_project/code/lead}

if [[ "$SCORER_FLAVOR" == "v1_1" ]]; then
  export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1}
  export PYTHONPATH=${WORKTREE_ROOT}:${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}
  SCORE_SCRIPT=${SCORE_SCRIPT:-${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score.py}
  DEFAULT_METRIC_CACHE_PATH=/workspace2/z_project/navsim_exp_lead_official/metric_cache_navtest_v1_1
else
  export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-/home/z/code/navsim}
  export PYTHONPATH=${WORKTREE_ROOT}:${PYTHONPATH:-}
  SCORE_SCRIPT=${SCORE_SCRIPT:-${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage.py}
  DEFAULT_METRIC_CACHE_PATH=/workspace2/data/navsim/processed_data/metric_cache_navtest
fi

GPU_ID=${GPU_ID:-2}
AGENT_DEVICE=${AGENT_DEVICE:-cuda}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-motdp_joint_route_speed_navtest_officialacc_e90}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/workspace2/z_project/motdp_logs/navsim_joint_route_speed_officialacc_e90_gpus0145_b128_pergpu_20260517_185212/best_model.pt}
CACHE_DIR=${CACHE_DIR:-/workspace2/z_project/motdp_bev_cache_navtest_official4cam_officialacc_npy}
METRIC_CACHE_PATH=${METRIC_CACHE_PATH:-$DEFAULT_METRIC_CACHE_PATH}
MAX_SCENES=${MAX_SCENES:-}
WORKERS=${WORKERS:-1}
SAMPLER_STOCHASTIC=${SAMPLER_STOCHASTIC:-false}
SCENE_FILTER_LOG_NAMES_JSON=${SCENE_FILTER_LOG_NAMES_JSON:-}
SCENE_FILTER_LOG_NAMES_FILE=${SCENE_FILTER_LOG_NAMES_FILE:-}
SCENE_FILTER_TOKENS_JSON=${SCENE_FILTER_TOKENS_JSON:-}
SCENE_FILTER_TOKENS_FILE=${SCENE_FILTER_TOKENS_FILE:-}

if [[ -n "$SCENE_FILTER_LOG_NAMES_FILE" ]]; then
  SCENE_FILTER_LOG_NAMES_JSON="$(tr -d '\n' < "$SCENE_FILTER_LOG_NAMES_FILE")"
fi

if [[ -n "$SCENE_FILTER_TOKENS_FILE" ]]; then
  SCENE_FILTER_TOKENS_JSON="$(tr -d '\n' < "$SCENE_FILTER_TOKENS_FILE")"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

if [[ ! -f "$SCORE_SCRIPT" ]]; then
  echo "Missing score script: $SCORE_SCRIPT" >&2
  exit 2
fi

if [[ ! -d "$METRIC_CACHE_PATH" ]]; then
  echo "Missing metric cache: $METRIC_CACHE_PATH" >&2
  exit 2
fi

CMD=(
  python "$SCORE_SCRIPT"
  train_test_split=navtest
  experiment_name="$EXPERIMENT_NAME"
  agent._target_=navsim_motdp.agents.cached_diffusion_agent.CachedDiffusionAgent
  +agent.checkpoint_path="$CHECKPOINT_PATH"
  +agent.cache_dir="$CACHE_DIR"
  +agent.device="$AGENT_DEVICE"
  +agent.fallback=raise
  +agent.sampler_stochastic="$SAMPLER_STOCHASTIC"
  metric_cache_path="$METRIC_CACHE_PATH"
  worker=single_machine_thread_pool
  worker.max_workers="$WORKERS"
)

if [[ -n "$MAX_SCENES" ]]; then
  CMD+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
fi

if [[ -n "$SCENE_FILTER_LOG_NAMES_JSON" ]]; then
  CMD+=("train_test_split.scene_filter.log_names=${SCENE_FILTER_LOG_NAMES_JSON}")
fi

if [[ -n "$SCENE_FILTER_TOKENS_JSON" ]]; then
  CMD+=("train_test_split.scene_filter.tokens=${SCENE_FILTER_TOKENS_JSON}")
fi

echo "WORKTREE_ROOT=$WORKTREE_ROOT"
echo "SCORER_FLAVOR=$SCORER_FLAVOR"
echo "NAVSIM_DEVKIT_ROOT=$NAVSIM_DEVKIT_ROOT"
echo "SCORE_SCRIPT=$SCORE_SCRIPT"
echo "METRIC_CACHE_PATH=$METRIC_CACHE_PATH"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

"${CMD[@]}"

python - "$NAVSIM_EXP_ROOT" "$EXPERIMENT_NAME" <<'PY'
import csv
import glob
import os
import sys

root, experiment = sys.argv[1], sys.argv[2]
csv_paths = glob.glob(os.path.join(root, experiment, "*", "*.csv"))
if not csv_paths:
  sys.exit(0)

latest_csv = max(csv_paths, key=os.path.getmtime)
with open(latest_csv, newline="", encoding="utf-8") as f:
  rows = list(csv.DictReader(f))

average_row = None
for row in reversed(rows):
  if row.get("token") in {"average", "average_all_frames"}:
    average_row = row
    break

print(f"LATEST_CSV={latest_csv}")
if average_row is None:
  sys.exit(0)

for key in [
    "score",
    "comfort",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "lane_keeping",
    "history_comfort",
    "two_frame_extended_comfort",
]:
  if key in average_row:
    print(f"{key}={average_row[key]}")
PY

echo DONE
