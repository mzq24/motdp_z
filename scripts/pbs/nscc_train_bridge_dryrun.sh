#!/bin/bash

### Bridge Baseline — dry-run version (no PBS, no training)
### 用法: bash scripts/pbs/nscc_train_bridge_dryrun.sh
### 只检查路径、文件是否存在、conda 环境、GPU 状态，不实际训练

set -e

# ---- User config (和 pbs 保持一致) ----
SCRATCH=/home/users/ntu/wh.huang/scratch
CODE_DIR=${SCRATCH}/z_projects/code/motdp_z
DATA_ROOT=${SCRATCH}/z_projects/dataset/pdm_lite
CONFIG_PATH=${CODE_DIR}/bridge_baseline/bd_config_hpc.yaml
CONDA_ENV=dpautomotive
USE_TMPFS=true
# ---------------------

# Derived paths (match HPC dataset structure)
DATASET_PATH=${DATA_ROOT}/tmp_data          # train/ val/ under here
CACHE_DIR=${DATA_ROOT}/tmp_data             # feature_index.pkl + .bin here

echo "=========================================="
echo "  Bridge Baseline — Dry Run Check"
echo "=========================================="

# 1. 检查关键目录
echo ""
echo "[1] 检查目录..."
for d in "${CODE_DIR}" "${DATA_ROOT}" "${DATASET_PATH}/train" "${DATASET_PATH}/val"; do
    if [ -d "$d" ]; then
        echo "  OK: $d"
    else
        echo "  MISSING: $d"
    fi
done

# 2. 检查 config 文件
echo ""
echo "[2] 检查 config..."
if [ -f "${CONFIG_PATH}" ]; then
    echo "  OK: ${CONFIG_PATH}"
    echo "  --- key settings ---"
    grep -E 'dataset_path|image_data_root|cache_dir|use_per_frame|predict_traj|plan_anchor|traj_anchor' "${CONFIG_PATH}" | sed 's/^/  /'
else
    echo "  MISSING: ${CONFIG_PATH}"
fi

# 3. 检查 memmap cache
echo ""
echo "[3] 检查 memmap cache (${CACHE_DIR})..."
for f in feature_index.pkl bev_features_fp16.bin bev_upsamples_fp16.bin; do
    fpath="${CACHE_DIR}/$f"
    if [ -f "$fpath" ]; then
        echo "  OK: $f ($(du -h "$fpath" | cut -f1))"
    else
        echo "  MISSING: $fpath"
    fi
done

# 4. 检查 samples_packed.pkl
echo ""
echo "[4] 检查 samples_packed.pkl..."
for split in train val; do
    fpath="${DATASET_PATH}/${split}/samples_packed.pkl"
    if [ -f "$fpath" ]; then
        echo "  OK: ${split}/samples_packed.pkl ($(du -h "$fpath" | cut -f1))"
    else
        echo "  NOT YET: $fpath (will be auto-generated on first run)"
    fi
done

# 5. 检查 anchor 文件 (相对路径，基于 CODE_DIR)
echo ""
echo "[5] 检查 anchor 文件..."
cd ${CODE_DIR}
for anchor_key in plan_anchor_path traj_anchor_path; do
    anchor_path=$(grep "${anchor_key}" "${CONFIG_PATH}" 2>/dev/null | head -1 | awk '{print $2}')
    if [ -z "$anchor_path" ]; then
        echo "  NOT SET: ${anchor_key}"
    elif [ -f "$anchor_path" ]; then
        echo "  OK: ${anchor_key} -> $anchor_path"
    else
        echo "  MISSING: ${anchor_key} -> $anchor_path (cwd: $(pwd))"
    fi
done

# 6. 检查 /tmp (tmpfs)
echo ""
echo "[6] 检查 /tmp (tmpfs)..."
if [ "${USE_TMPFS}" = true ]; then
    tmp_total=$(df -h /tmp | tail -1 | awk '{print $2}')
    tmp_avail=$(df -h /tmp | tail -1 | awk '{print $4}')
    echo "  tmpfs total: ${tmp_total}, available: ${tmp_avail}"
    cache_size=$(du -sh "${CACHE_DIR}" 2>/dev/null | cut -f1 || echo "unknown")
    echo "  BEV cache size: ${cache_size}"
    echo "  USE_TMPFS=true -> will copy to /tmp/tmp_data"
else
    echo "  USE_TMPFS=false -> skip"
fi

# 7. 检查 conda 环境
echo ""
echo "[7] 检查 conda..."
if command -v conda &>/dev/null; then
    current_env=$(conda info --envs | grep '\*' | awk '{print $1}')
    echo "  current env: ${current_env}"
    if conda info --envs | grep -q "${CONDA_ENV}"; then
        echo "  OK: ${CONDA_ENV} exists"
    else
        echo "  MISSING: env ${CONDA_ENV} not found"
    fi
else
    echo "  conda not found in PATH"
fi

# 8. 检查 GPU
echo ""
echo "[8] 检查 GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo "  nvidia-smi not found"
fi

# 9. 检查 python import
echo ""
echo "[9] 检查 python import..."
cd ${CODE_DIR}
python -c "
import torch
print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
from dataset.unified_carla_dataset import CARLAImageDataset
print('  CARLAImageDataset: OK')
from bridge_baseline.policy import BDBaselinePolicyV2
print('  BDBaselinePolicyV2: OK')
" 2>&1 || echo "  IMPORT FAILED"

# 10. 内存
echo ""
echo "[10] 内存状态..."
free -h | head -2

echo ""
echo "=========================================="
echo "  Dry run complete. Fix any MISSING items before submitting PBS."
echo "=========================================="
