#!/usr/bin/env bash
set -euo pipefail

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_dpauto

cd /home/z/code/motdp_z

NUM_SHARDS="${NUM_SHARDS:-16}"
SOURCE="${SOURCE:-/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.pkl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_stage1_shards_${NUM_SHARDS}}"

python scripts/data_tools/split_stage1_relabel_shards.py \
  --source "$SOURCE" \
  --output_root "$OUTPUT_ROOT" \
  --num_shards "$NUM_SHARDS"