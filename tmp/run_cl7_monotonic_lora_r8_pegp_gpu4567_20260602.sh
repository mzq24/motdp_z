#!/usr/bin/env bash
set -euo pipefail

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp

REPO_ROOT=${REPO_ROOT:-/data/z_project/code/nuplan_whitenoise_diffusion_v1_git}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}

IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE=${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}
MASTER_PORT=${MASTER_PORT:-29682}
INIT_CKPT=${INIT_CKPT:-/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/source_overlap_3x100k_norm_20260517/checkpoints/epoch_0119.pth}
CONFIG_PATH=${CONFIG_PATH:-${REPO_ROOT}/tmp/nuplan_diffusion_target_cl7_monotonic_pegp_lora_r8_gpu4567_bs256_20260602.yaml}
LOG_ROOT=${LOG_ROOT:-/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/logs/lora_cl7_monotonic_r8_20260602}
LOG_PATH="$LOG_ROOT/pegp_lora_rank8_gpu4567_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_ROOT"
echo "===== START cl7_monotonic_pegp_lora_rank8 ====="
echo "config=${CONFIG_PATH}"
echo "gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} port=${MASTER_PORT}"
echo "init_ckpt=${INIT_CKPT}"
echo "log=${LOG_PATH}"

/workspace1/miniconda/envs/z_navsim_motdp/bin/torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  ${REPO_ROOT}/training/train_nuplan.py \
  --config "$CONFIG_PATH" \
  --init_ckpt "$INIT_CKPT" \
  2>&1 | tee "$LOG_PATH"
