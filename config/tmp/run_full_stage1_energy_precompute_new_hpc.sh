#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/z/anaconda3/envs/dpauto/bin/python}"
SPLIT_ROOT="${SPLIT_ROOT:-/workspace1/z_project/dataset/pdm_lite/tmp_data}"
IMAGE_DATA_ROOT="${IMAGE_DATA_ROOT:-/workspace1/z_project/dataset/pdm_lite}"
ANCHOR_PATH="${ANCHOR_PATH:-$REPO_ROOT/dd_baseline/anchors/carla_kmeans_32.npy}"

if [[ ! -d "$SPLIT_ROOT/train" || ! -d "$SPLIT_ROOT/val" ]]; then
  echo "Expected processed full-dataset splits under: $SPLIT_ROOT/{train,val}" >&2
  exit 1
fi

if [[ ! -d "$IMAGE_DATA_ROOT" ]]; then
  echo "Missing raw image data root: $IMAGE_DATA_ROOT" >&2
  exit 1
fi

if [[ ! -f "$ANCHOR_PATH" ]]; then
  echo "Missing anchor file: $ANCHOR_PATH" >&2
  echo "Override with: ANCHOR_PATH=/abs/path/to/carla_kmeans_32.npy bash $0" >&2
  exit 1
fi

cd "$REPO_ROOT"

for split in train val; do
  echo "=== Precomputing stage1 energy labels for $split ==="
  "$PYTHON_BIN" scripts/data_tools/precompute_semantic_labels.py \
    --dataset_path "$SPLIT_ROOT/$split" \
    --image_data_root "$IMAGE_DATA_ROOT" \
    --anchor_path "$ANCHOR_PATH" \
    --force
done
