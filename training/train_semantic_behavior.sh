#!/bin/bash
# =============================================================================
# Training Script: Semantic Behavior-Conditioned Diffusion Policy
# =============================================================================
# Architecture: BehaviorDecoder + MultiSourceAttentionBlock + Contrastive Learning
#
# Losses:
#   - reg_loss:         L1 trajectory regression (best mode)
#   - cls_loss:         Focal loss for mode selection
#   - route_loss:       L1 route prediction
#   - behavior_loss:    CE for 11-class behavior classification (from BehaviorDecoder)
#   - allowed_loss:     BCE for allowed/forbidden per-anchor (from BehaviorDecoder)
#   - contrastive_loss: InfoNCE (expert traj vs anchor embeddings)
#
# Usage:
#   Single GPU:  bash training/train_semantic_behavior.sh
#   Resume:      bash training/train_semantic_behavior.sh --resume /path/to/checkpoint.pth
# =============================================================================

set -e

# ===== Paths =====
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/config/carla.yaml"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/semantic_behavior"

# ===== GPU Config =====
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1

# ===== WandB =====
# Set to "online" to sync to cloud, "offline" for local-only, "disabled" to skip
export WANDB_MODE="${WANDB_MODE:-offline}"

# ===== Parse arguments =====
RESUME_ARG=""
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_ARG="--resume $2"
            shift 2
            ;;
        --val_only)
            EXTRA_ARGS="${EXTRA_ARGS} --val_only"
            shift
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ===== Create checkpoint dir =====
mkdir -p "${CHECKPOINT_DIR}"

# ===== Print info =====
echo "========================================="
echo " Semantic Behavior Training"
echo "========================================="
echo " Config:      ${CONFIG_PATH}"
echo " Checkpoint:  ${CHECKPOINT_DIR}"
echo " GPU:         ${CUDA_VISIBLE_DEVICES}"
echo " WandB:       ${WANDB_MODE}"
echo " Resume:      ${RESUME_ARG:-none}"
echo "========================================="
echo ""
echo " Key settings from config:"
echo "   - BehaviorDecoder: 2-layer transformer -> behavior_tokens"
echo "   - Trajectory Decoder: MultiSourceAttentionBlock (3-way attn)"
echo "   - Contrastive: InfoNCE (tau=0.07, dim=128, weight=0.3)"
echo "   - Behavior loss weight: 0.1, Allowed loss weight: 0.5"
echo "   - Truncated diffusion: 8 steps train, 2 steps inference"
echo "========================================="
echo ""

# ===== Launch training =====
cd "${PROJECT_ROOT}"

python training/train_carla_bev.py \
    --config_path "${CONFIG_PATH}" \
    ${RESUME_ARG} \
    ${EXTRA_ARGS}
