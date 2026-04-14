#!/usr/bin/env bash
set -euo pipefail

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_dpauto

cd /home/z/code/motdp_z

PYTHON_BIN="${PYTHON_BIN:-python}"
IMAGE_DATA_ROOT="${IMAGE_DATA_ROOT:-/workspace1/z_project/dataset/pdm_lite}"
SHARD_ROOT="${SHARD_ROOT:-/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_stage1_shards_16}"
CHECKPOINT_EVERY_MINUTES="${CHECKPOINT_EVERY_MINUTES:-10}"

mapfile -t SHARD_DIRS < <(find "$SHARD_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'shard_*' | sort)

if [[ "${#SHARD_DIRS[@]}" -eq 0 ]]; then
  echo "No shard directories found under $SHARD_ROOT" >&2
  exit 1
fi

pids=()
for shard_dir in "${SHARD_DIRS[@]}"; do
  log_path="$shard_dir/stage1_relabel.log"
  echo "Launching $shard_dir -> $log_path"
  "$PYTHON_BIN" scripts/data_tools/precompute_semantic_labels.py \
    --dataset_path "$shard_dir" \
    --image_data_root "$IMAGE_DATA_ROOT" \
    --stage1_only \
    --force \
    --checkpoint_every_minutes "$CHECKPOINT_EVERY_MINUTES" \
    >"$log_path" 2>&1 &
  pids+=("$!")
done

failed=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  shard_dir="${SHARD_DIRS[$idx]}"
  if ! wait "$pid"; then
    echo "Shard failed: $shard_dir" >&2
    failed=1
  fi
done

exit "$failed"