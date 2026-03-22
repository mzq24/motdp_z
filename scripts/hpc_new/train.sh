#!/bin/bash
###############################################################################
# Multi-GPU Training Script for New HPC (8x A100-80GB)
#
# Usage:
#   bash scripts/hpc_new/train.sh route_b              # Route B (annealed energy guidance)
#   bash scripts/hpc_new/train.sh bridge                # Bridge baseline
#   bash scripts/hpc_new/train.sh dd                    # DD baseline
#
#   GPUS=4 bash scripts/hpc_new/train.sh route_b        # Use 4 GPUs
#   CUDA_VISIBLE_DEVICES=0,1 GPUS=2 bash scripts/hpc_new/train.sh bridge
###############################################################################

set -euo pipefail

CODE_DIR=/workspace1/z_project/code/motdp_z
CONDA_ENV=z_dpauto
GPUS=${GPUS:-8}

cd "${CODE_DIR}"

MODEL=${1:-""}

case "${MODEL}" in
    route_b)
        CONFIG=config/pdm_hpc_route_b.yaml
        SCRIPT=training/train_carla_bev.py
        ;;
    bridge)
        CONFIG=bridge_baseline/bd_config_hpc_new.yaml
        SCRIPT=bridge_baseline/train.py
        ;;
    dd)
        CONFIG=dd_baseline/dd_config.yaml
        SCRIPT=training/train_carla_bev.py
        ;;
    *)
        echo "Usage: bash scripts/hpc_new/train.sh {route_b|bridge|dd}"
        echo ""
        echo "Options:"
        echo "  GPUS=N       Number of GPUs (default: 8)"
        echo "  CUDA_VISIBLE_DEVICES=0,1,2,3  Select specific GPUs"
        exit 1
        ;;
esac

echo "========================================"
echo "  Training: ${MODEL}"
echo "  Config:   ${CONFIG}"
echo "  GPUs:     ${GPUS}"
echo "  Script:   ${SCRIPT}"
echo "========================================"

if [ "${GPUS}" -eq 1 ]; then
    python ${SCRIPT} --config ${CONFIG}
else
    torchrun \
        --nproc_per_node=${GPUS} \
        --master_port=$(( RANDOM % 10000 + 20000 )) \
        ${SCRIPT} --config ${CONFIG}
fi
