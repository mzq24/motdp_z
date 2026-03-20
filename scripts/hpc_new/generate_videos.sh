#!/bin/bash
###############################################################################
# HPC: Generate scenario videos for all scenario types
#
# Runs generate_scenario_videos.py on the full pdm_lite dataset.
# Output goes to /workspace1/z_project/scenario_videos/
#
# Usage:
#   nohup bash scripts/hpc_new/generate_videos.sh > logs/generate_videos.log 2>&1 &
###############################################################################

set -euo pipefail

DATA_RAW=/workspace1/z_project/dataset/pdm_lite
CODE_DIR=/workspace1/z_project/code/motdp_z
CONDA_ENV=z_dpauto
OUTPUT_DIR=/workspace1/z_project/scenario_videos

cd "${CODE_DIR}"

echo "========================================"
echo "  Generate Scenario Videos"
echo "  Dataset: ${DATA_RAW}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  $(date)"
echo "========================================"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

python tools/generate_scenario_videos.py \
    --dataset_root "${DATA_RAW}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_per_type 3 \
    --fps 10 \
    --max_frames 200

echo ""
echo "========================================"
echo "  Done at $(date)"
echo "  Videos: ${OUTPUT_DIR}"
echo "  Summary: ${OUTPUT_DIR}/summary.txt"
echo "========================================"
