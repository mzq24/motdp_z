#!/bin/bash
###############################################################################
# MoT-DP New HPC Full Deployment Pipeline
#
# Prerequisites:
#   - Raw PDM Lite dataset downloaded to DATA_RAW
#   - Conda env z_dpauto available
#   - TransFuser pretrained weights at MODEL_DIR
#
# Usage:
#   bash scripts/hpc_new/deploy_pipeline.sh [step]
#
# Steps (must run in order):
#   all           - run everything (default)
#   transfuser    - Step 1: extract BEV features (GPU, must run FIRST)
#   preprocess    - Step 2: raw data -> sample pkl (depends on transfuser_feature/)
#   build_cache   - Step 3: pack route_features.pt -> memmap .bin
#   anchors       - Step 4: generate anchor files
#   stats         - Step 5: compute norm statistics
#   dryrun        - Step 6: verify everything
###############################################################################

set -euo pipefail

# ============ Path Configuration ============
DATA_RAW=/workspace1/z_project/dataset/pdm_lite
CODE_DIR=/workspace1/z_project/code/motdp_z
MODEL_DIR=/workspace1/z_project/models/pretrain   # <- update to actual path when ready
CONDA_ENV=z_dpauto

# Derived paths
# preprocess output: DATA_RAW/tmp_data/train/*.pkl, DATA_RAW/tmp_data/val/*.pkl
PROCESSED_DIR=${DATA_RAW}/tmp_data
CACHE_DIR=${DATA_RAW}/tmp_data    # memmap .bin lives at DATA_RAW/tmp_data/
# ============================================

STEP=${1:-all}

cd "${CODE_DIR}"
echo "========================================"
echo "  MoT-DP New HPC Deployment Pipeline"
echo "  Step: ${STEP}"
echo "  Code: ${CODE_DIR}"
echo "  Data: ${DATA_RAW}"
echo "========================================"

# ---- Step 2: Preprocess raw data -> sample-level pkl ----
# NOTE: Must run AFTER transfuser, because preprocess reads transfuser_feature/ paths
run_preprocess() {
    echo ""
    echo "==== Step 2: Preprocess raw data -> sample pkl ===="
    python dataset/preprocess_pdm_lite.py \
        --data-root "${DATA_RAW}" \
        --out-dir "${PROCESSED_DIR}" \
        --tmp-dir data \
        --obs-horizon 4 \
        --action-horizon 6 \
        --sample-interval 1 \
        --hz-interval 2 \
        --workers 4 \
        --save-mode frame
    echo "Step 2 done. Output: ${PROCESSED_DIR}/train/ and ${PROCESSED_DIR}/val/"
    echo "Checking counts:"
    echo "  train: $(ls ${PROCESSED_DIR}/train/*.pkl 2>/dev/null | wc -l) pkl files"
    echo "  val:   $(ls ${PROCESSED_DIR}/val/*.pkl 2>/dev/null | wc -l) pkl files"
}

# ---- Step 1: TransFuser BEV feature extraction (GPU) ----
# NOTE: Must run FIRST — preprocess depends on transfuser_feature/ existing
run_transfuser() {
    echo ""
    echo "==== Step 1: TransFuser BEV feature extraction ===="
    if [ ! -d "${MODEL_DIR}" ]; then
        echo "ERROR: MODEL_DIR not found: ${MODEL_DIR}"
        echo "Please update MODEL_DIR to the TransFuser pretrained weights directory."
        exit 1
    fi
    # Find model file
    MODEL_FILE=$(find "${MODEL_DIR}" -name "model_0030_1.pth" -o -name "*.pth" | head -1)
    if [ -z "${MODEL_FILE}" ]; then
        echo "ERROR: No .pth file found in ${MODEL_DIR}"
        exit 1
    fi
    echo "Using model: ${MODEL_FILE}"
    echo "Using config: ${MODEL_DIR}"

    # pack_and_extract: Phase 1 packs source (CPU), Phase 2 extracts features (GPU)
    python model/transfuser_extractor/preprocess_dataset.py \
        --dataset_path "${DATA_RAW}" \
        --config_path "${MODEL_DIR}" \
        --model_path "${MODEL_FILE}" \
        --batch_size 128 \
        --device cuda:0 \
        --mode pack_and_extract
    echo "Step 1 done. route_features.pt generated per route."
}

# ---- Step 3: Build memmap cache (.bin) ----
run_build_cache() {
    echo ""
    echo "==== Step 3: Build memmap feature cache ===="
    python scripts/data_tools/build_feature_cache_fp16.py \
        --dataset_root "${DATA_RAW}"
    echo "Step 3 done. Output:"
    for f in feature_index.pkl bev_features_fp16.bin bev_upsamples_fp16.bin; do
        fpath="${DATA_RAW}/tmp_data/$f"
        if [ -f "$fpath" ]; then
            echo "  $f: $(du -h "$fpath" | cut -f1)"
        else
            echo "  $f: MISSING"
        fi
    done
}

# ---- Step 4: Generate anchor files ----
run_anchors() {
    echo ""
    echo "==== Step 4: Generate anchor files ===="

    # Ensure samples_packed.pkl exists for --fast mode
    PACKED="${PROCESSED_DIR}/train/samples_packed.pkl"
    if [ ! -f "${PACKED}" ]; then
        echo "samples_packed.pkl not found, generating..."
        # Trigger auto-generation by importing dataset
        python -c "
import sys; sys.path.insert(0, '.')
from dataset.unified_carla_dataset import CARLAImageDataset
ds = CARLAImageDataset('${PROCESSED_DIR}/train', '${DATA_RAW}', mode='train', skip_memmap=True)
print(f'Loaded {len(ds)} train samples, samples_packed.pkl should now exist.')
"
    fi

    # Also for val
    PACKED_VAL="${PROCESSED_DIR}/val/samples_packed.pkl"
    if [ ! -f "${PACKED_VAL}" ]; then
        echo "Generating val samples_packed.pkl..."
        python -c "
import sys; sys.path.insert(0, '.')
from dataset.unified_carla_dataset import CARLAImageDataset
ds = CARLAImageDataset('${PROCESSED_DIR}/val', '${DATA_RAW}', mode='val', skip_memmap=True)
print(f'Loaded {len(ds)} val samples.')
"
    fi

    # 4a: Bridge baseline - route anchor (20 modes, 10 waypoints)
    echo "Generating bridge route anchors..."
    mkdir -p bridge_baseline/anchors
    python bridge_baseline/scripts/generate_anchors.py \
        --dataset_path "${PROCESSED_DIR}/train" \
        --key route \
        --num_poses 10 \
        --num_modes 20 \
        --fast \
        --output bridge_baseline/anchors/carla_kmeans_20_route10.npy

    # 4b: DD baseline - traj anchor (20 modes, 6 waypoints)
    echo "Generating dd traj anchors (20 modes)..."
    mkdir -p dd_baseline/anchors
    python bridge_baseline/scripts/generate_anchors.py \
        --dataset_path "${PROCESSED_DIR}/train" \
        --key agent_pos \
        --num_poses 6 \
        --num_modes 20 \
        --fast \
        --output dd_baseline/anchors/carla_kmeans_20.npy

    # 4c: DD baseline / Route B - traj anchor (32 modes, 6 waypoints)
    echo "Generating dd traj anchors (32 modes)..."
    python bridge_baseline/scripts/generate_anchors.py \
        --dataset_path "${PROCESSED_DIR}/train" \
        --key agent_pos \
        --num_poses 6 \
        --num_modes 32 \
        --fast \
        --output dd_baseline/anchors/carla_kmeans_32.npy

    echo "Step 4 done. Anchors:"
    ls -lh bridge_baseline/anchors/*.npy dd_baseline/anchors/*.npy 2>/dev/null
}

# ---- Step 5: Compute norm statistics ----
run_stats() {
    echo ""
    echo "==== Step 5: Compute norm statistics ===="

    # 5a: Bridge baseline per-waypoint stats
    echo "Computing bridge norm stats..."
    python bridge_baseline/scripts/compute_route_stats.py \
        --dataset_path "${PROCESSED_DIR}/train" \
        --key all --fast \
        --output_yaml bridge_baseline/bd_config.yaml

    # 5b: Main config action stats
    echo "Computing action stats for main config..."
    python dataset/compute_action_stats.py \
        --dataset_path "${PROCESSED_DIR}" \
        --image_data_root "${DATA_RAW}" \
        --config_path config/pdm_local.yaml

    echo "Step 5 done."
    echo "IMPORTANT: Copy the norm stats from bd_config.yaml -> bd_config_hpc_new.yaml"
    echo "IMPORTANT: Copy action_stats from pdm_local.yaml -> pdm_hpc_new.yaml / pdm_local_route_b.yaml"
}

# ---- Step 6: Dry-run verification ----
run_dryrun() {
    echo ""
    echo "==== Step 6: Dry-run verification ===="
    bash scripts/hpc_new/dryrun.sh
}

# ---- Dispatch ----
case "${STEP}" in
    all)
        run_transfuser
        run_preprocess
        run_build_cache
        run_anchors
        run_stats
        run_dryrun
        ;;
    preprocess)   run_preprocess ;;
    transfuser)   run_transfuser ;;
    build_cache)  run_build_cache ;;
    anchors)      run_anchors ;;
    stats)        run_stats ;;
    dryrun)       run_dryrun ;;
    *)
        echo "Unknown step: ${STEP}"
        echo "Usage: $0 {all|preprocess|transfuser|build_cache|anchors|stats|dryrun}"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "  Pipeline step '${STEP}' complete!"
echo "========================================"
