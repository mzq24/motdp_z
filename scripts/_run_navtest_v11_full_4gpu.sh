#!/bin/bash
set -euo pipefail

if [[ -z "${WORKTREE_ROOT:-}" ]]; then
	SOURCE_PATH="${BASH_SOURCE[0]:-$0}"
	SCRIPT_DIR="$(cd "$(dirname "${SOURCE_PATH}")" && pwd)"
	WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp

export TMPDIR=${TMPDIR:-/workspace2/z_project/tmp}
mkdir -p "$TMPDIR"
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-/workspace2/data/navsim}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/workspace2/data/navsim/maps}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/workspace2/z_project/navsim_exp_motdp}
export LEAD_PROJECT_ROOT=${LEAD_PROJECT_ROOT:-/workspace1/z_project/code/lead}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1}
export HYDRA_FULL_ERROR=1

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
MODE=${MODE:-both}
NUM_SHARDS=${NUM_SHARDS:-4}
ONLINE_GPU_LIST=${ONLINE_GPU_LIST:-4 5 6 7}
CACHED_GPU_LIST=${CACHED_GPU_LIST:-4 5 6 7}
ONLINE_EXPERIMENT_PREFIX=${ONLINE_EXPERIMENT_PREFIX:-motdp_online_official4cam_e90_v11_full_${STAMP}}
CACHED_EXPERIMENT_PREFIX=${CACHED_EXPERIMENT_PREFIX:-motdp_cached_official4cam_e90_v11_full_${STAMP}}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/workspace2/z_project/motdp_logs/navsim_official4cam_e90_gpus0145_b128_pergpu_20260517_074858/best_model.pt}
CACHE_DIR=${CACHE_DIR:-/workspace2/z_project/motdp_bev_cache_navtest_official4cam_npy}
METRIC_CACHE_PATH=${METRIC_CACHE_PATH:-/workspace2/z_project/navsim_exp_lead_official/metric_cache_navtest_v1_1}
LOG_DIR=${LOG_DIR:-${NAVSIM_EXP_ROOT}/logs}
SHARD_SPEC_DIR=${SHARD_SPEC_DIR:-${TMPDIR}/navtest_v11_shards_${STAMP}}
MAX_SCENES=${MAX_SCENES:-}
WORKERS=${WORKERS:-1}

mkdir -p "$LOG_DIR" "$SHARD_SPEC_DIR"

read -r -a ONLINE_GPUS <<< "$ONLINE_GPU_LIST"
read -r -a CACHED_GPUS <<< "$CACHED_GPU_LIST"

if [[ "$MODE" == "online" || "$MODE" == "both" ]]; then
	if [[ ${#ONLINE_GPUS[@]} -ne $NUM_SHARDS ]]; then
		echo "ONLINE_GPU_LIST count ${#ONLINE_GPUS[@]} != NUM_SHARDS $NUM_SHARDS" >&2
		exit 2
	fi
fi

if [[ "$MODE" == "cached" || "$MODE" == "both" ]]; then
	if [[ ${#CACHED_GPUS[@]} -ne $NUM_SHARDS ]]; then
		echo "CACHED_GPU_LIST count ${#CACHED_GPUS[@]} != NUM_SHARDS $NUM_SHARDS" >&2
		exit 2
	fi
fi

python - "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml" "$NUM_SHARDS" "$SHARD_SPEC_DIR" <<'PY'
import json
import os
import sys

import yaml

yaml_path, num_shards_str, out_dir = sys.argv[1:4]
num_shards = int(num_shards_str)

with open(yaml_path, "r", encoding="utf-8") as f:
    scene_filter = yaml.safe_load(f)

log_names = list(scene_filter["log_names"])
for shard_idx in range(num_shards):
    shard_logs = log_names[shard_idx::num_shards]
    out_path = os.path.join(out_dir, f"shard{shard_idx}.log_names.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(shard_logs, f, separators=(",", ":"))
    print(f"shard={shard_idx} logs={len(shard_logs)} path={out_path}")
PY

launch_job() {
	local role="$1"
	local gpu="$2"
	local shard_idx="$3"
	local experiment_name="$4"
	local shard_file="$5"
	local script_path="$6"
	local logfile="${LOG_DIR}/${experiment_name}.log"
	local -a env_args=(
		WORKTREE_ROOT="$WORKTREE_ROOT"
		LEAD_PROJECT_ROOT="$LEAD_PROJECT_ROOT"
		NAVSIM_DEVKIT_ROOT="$NAVSIM_DEVKIT_ROOT"
		GPU_ID="$gpu"
		AGENT_DEVICE="cuda"
		CHECKPOINT_PATH="$CHECKPOINT_PATH"
		METRIC_CACHE_PATH="$METRIC_CACHE_PATH"
		EXPERIMENT_NAME="$experiment_name"
		SCENE_FILTER_LOG_NAMES_FILE="$shard_file"
		WORKERS="$WORKERS"
	)

	if [[ -n "$MAX_SCENES" ]]; then
		env_args+=(MAX_SCENES="$MAX_SCENES")
	fi

	if [[ "$role" == "cached" ]]; then
		env_args+=(SCORER_FLAVOR="v1_1" CACHE_DIR="$CACHE_DIR")
	fi

	nohup env "${env_args[@]}" bash "$script_path" > "$logfile" 2>&1 < /dev/null &
	local pid=$!
	echo "$role shard=$shard_idx gpu=$gpu pid=$pid log=$logfile exp=$experiment_name"
}

if [[ "$MODE" == "online" || "$MODE" == "both" ]]; then
	for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
		launch_job \
			online \
			"${ONLINE_GPUS[$shard_idx]}" \
			"$shard_idx" \
			"${ONLINE_EXPERIMENT_PREFIX}_shard${shard_idx}_of${NUM_SHARDS}_gpu${ONLINE_GPUS[$shard_idx]}" \
			"${SHARD_SPEC_DIR}/shard${shard_idx}.log_names.json" \
			"${WORKTREE_ROOT}/scripts/_run_lead_navtest_full.sh"
	done
fi

if [[ "$MODE" == "cached" || "$MODE" == "both" ]]; then
	for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
		launch_job \
			cached \
			"${CACHED_GPUS[$shard_idx]}" \
			"$shard_idx" \
			"${CACHED_EXPERIMENT_PREFIX}_shard${shard_idx}_of${NUM_SHARDS}_gpu${CACHED_GPUS[$shard_idx]}" \
			"${SHARD_SPEC_DIR}/shard${shard_idx}.log_names.json" \
			"${WORKTREE_ROOT}/scripts/_run_navtest_pdm.sh"
	done
fi

echo "STAMP=$STAMP"
echo "WORKTREE_ROOT=$WORKTREE_ROOT"
echo "SHARD_SPEC_DIR=$SHARD_SPEC_DIR"
echo "CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "CACHE_DIR=$CACHE_DIR"
echo "METRIC_CACHE_PATH=$METRIC_CACHE_PATH"