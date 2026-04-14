#!/usr/bin/env bash
set -euo pipefail

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_dpauto

cd /home/z/code/motdp_z

BASE="${BASE:-/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.pkl}"
SHARD_ROOT="${SHARD_ROOT:-/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_stage1_shards_16}"
OUTPUT="${OUTPUT:-/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_merged.pkl}"

python scripts/data_tools/merge_stage1_relabel_shards.py \
  --base "$BASE" \
  --overlay_root "$SHARD_ROOT" \
  --output "$OUTPUT"