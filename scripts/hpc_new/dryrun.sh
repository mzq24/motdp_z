#!/bin/bash
###############################################################################
# New HPC Dry-Run Check
# Verifies all paths, files, and dependencies before training
###############################################################################

set -e

# ---- Config (keep in sync with deploy_pipeline.sh) ----
DATA_RAW=/workspace1/z_project/dataset/pdm_lite
CODE_DIR=/workspace1/z_project/code/motdp_z
CONDA_ENV=z_dpauto
PROCESSED_DIR=${DATA_RAW}/tmp_data
CACHE_DIR=${DATA_RAW}/tmp_data
# --------------------------------------------------------

cd "${CODE_DIR}"

echo "=========================================="
echo "  New HPC — Dry Run Check"
echo "=========================================="

FAIL=0

# 1. Key directories
echo ""
echo "[1] Directories..."
for d in "${CODE_DIR}" "${DATA_RAW}" "${PROCESSED_DIR}/train" "${PROCESSED_DIR}/val"; do
    if [ -d "$d" ]; then
        echo "  OK: $d"
    else
        echo "  MISSING: $d"
        FAIL=1
    fi
done

# 2. Memmap cache
echo ""
echo "[2] Memmap cache (${CACHE_DIR})..."
for f in feature_index.pkl bev_features_fp16.bin bev_upsamples_fp16.bin; do
    fpath="${CACHE_DIR}/$f"
    if [ -f "$fpath" ]; then
        echo "  OK: $f ($(du -h "$fpath" | cut -f1))"
    else
        echo "  MISSING: $fpath"
        FAIL=1
    fi
done

# 3. samples_packed.pkl
echo ""
echo "[3] samples_packed.pkl..."
for split in train val; do
    fpath="${PROCESSED_DIR}/${split}/samples_packed.pkl"
    if [ -f "$fpath" ]; then
        echo "  OK: ${split}/samples_packed.pkl ($(du -h "$fpath" | cut -f1))"
    else
        echo "  NOT YET: $fpath (will be auto-generated on first run)"
    fi
done

# 4. Anchor files
echo ""
echo "[4] Anchor files..."
for anchor in bridge_baseline/anchors/carla_kmeans_20_route10.npy \
              dd_baseline/anchors/carla_kmeans_20.npy \
              dd_baseline/anchors/carla_kmeans_32.npy; do
    if [ -f "$anchor" ]; then
        echo "  OK: $anchor"
    else
        echo "  MISSING: $anchor"
        FAIL=1
    fi
done

# 5. Config files
echo ""
echo "[5] HPC config files..."
for cfg in bridge_baseline/bd_config_hpc_new.yaml config/pdm_hpc_new.yaml config/pdm_local_route_b.yaml; do
    if [ -f "$cfg" ]; then
        echo "  OK: $cfg"
        # Show key paths
        grep -E 'dataset_path|image_data_root|cache_dir|anchor_path|plan_anchor_path|traj_anchor_path' "$cfg" 2>/dev/null | sed 's/^/    /'
    else
        echo "  MISSING: $cfg"
    fi
done

# 6. Norm stats populated?
echo ""
echo "[6] Norm stats (checking bd_config_hpc_new.yaml)..."
if [ -f bridge_baseline/bd_config_hpc_new.yaml ]; then
    has_default=$(grep 'norm_x_mean:.*\[0\.0' bridge_baseline/bd_config_hpc_new.yaml | head -1)
    if [ -n "$has_default" ]; then
        echo "  WARNING: norm stats appear to be defaults (all zeros). Run compute_route_stats.py!"
    else
        echo "  OK: norm stats appear populated"
    fi
fi

# 7. Conda
echo ""
echo "[7] Conda..."
if command -v conda &>/dev/null; then
    if conda info --envs | grep -q "${CONDA_ENV}"; then
        echo "  OK: env ${CONDA_ENV} exists"
    else
        echo "  MISSING: env ${CONDA_ENV}"
        FAIL=1
    fi
else
    echo "  conda not found in PATH"
fi

# 8. GPU
echo ""
echo "[8] GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
else
    echo "  nvidia-smi not found"
fi

# 9. Python imports
echo ""
echo "[9] Python imports..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
from dataset.unified_carla_dataset import CARLAImageDataset
print('  CARLAImageDataset: OK')
from bridge_baseline.policy import BDBaselinePolicyV2
print('  BDBaselinePolicyV2: OK')
" 2>&1 || echo "  IMPORT FAILED"

# 10. Memory
echo ""
echo "[10] Memory..."
free -h | head -2

# 11. Disk space
echo ""
echo "[11] Disk (${DATA_RAW})..."
df -h "$(dirname ${DATA_RAW})" | tail -1

echo ""
echo "=========================================="
if [ ${FAIL} -eq 0 ]; then
    echo "  All checks passed!"
else
    echo "  Some checks FAILED. Fix MISSING items before training."
fi
echo "=========================================="
