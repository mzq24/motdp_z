#!/bin/bash
# Train Bridge Baseline v2 (BridgeDrive DDBM reproduction)
# Usage: bash scripts/train_bridge_baseline.sh [--resume <ckpt_path>]

set -e
cd "$(dirname "$0")/.."

CONFIG=bridge_baseline/bd_config.yaml

# Parse optional --resume argument
RESUME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume) RESUME="--resume $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

if [ "$NUM_GPUS" -le 1 ]; then
    echo "[bridge_baseline] Single-GPU training"
    python bridge_baseline/train.py \
        --config_path $CONFIG \
        $RESUME
else
    echo "[bridge_baseline] Multi-GPU training (${NUM_GPUS} GPUs)"
    torchrun --nproc_per_node=$NUM_GPUS \
        bridge_baseline/train.py \
        --config_path $CONFIG \
        $RESUME
fi
