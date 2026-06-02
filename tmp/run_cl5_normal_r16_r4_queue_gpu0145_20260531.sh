#!/usr/bin/env bash
set -euo pipefail

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp

cd /home/z/code/nuplan_whitenoise_diffusion_v1
export PYTHONPATH=/home/z/code/nuplan_whitenoise_diffusion_v1:${PYTHONPATH:-}

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,4,5}

IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE=${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}
INIT_CKPT=${INIT_CKPT:-/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/source_overlap_3x100k_norm_20260517/checkpoints/epoch_0119.pth}
LOG_ROOT=${LOG_ROOT:-/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/logs/queued_cl5_normal_r16_r4_20260531}
MIN_VAL_CACHE_KB=${MIN_VAL_CACHE_KB:-340000000}
WAIT_SECONDS=${WAIT_SECONDS:-600}

mkdir -p "$LOG_ROOT"

wait_for_cache() {
  while true; do
    if [[ ! -f "$INIT_CKPT" ]]; then
      echo "Waiting for init checkpoint: $INIT_CKPT"
      sleep "$WAIT_SECONDS"
      continue
    fi

    local val_kb
    val_kb=$(du -s /workspace2/z_project/exp/nuplan/cache_plantf_val 2>/dev/null | awk '{print $1}' || echo 0)
    if [[ "$val_kb" =~ ^[0-9]+$ ]] && (( val_kb >= MIN_VAL_CACHE_KB )); then
      echo "Cache gate passed: cache_plantf_val=${val_kb}KB >= ${MIN_VAL_CACHE_KB}KB"
      return
    fi

    echo "Waiting for cache_plantf_val: ${val_kb}KB < ${MIN_VAL_CACHE_KB}KB"
    sleep "$WAIT_SECONDS"
  done
}

run_one() {
  local label=$1
  local config_path=$2
  local master_port=$3
  local log_path="$LOG_ROOT/${label}_$(date +%Y%m%d_%H%M%S).log"

  echo "===== START ${label} ====="
  echo "config=${config_path}"
  echo "gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} port=${master_port}"
  echo "log=${log_path}"

  /workspace1/miniconda/envs/z_navsim_motdp/bin/torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$master_port" \
    /home/z/code/nuplan_whitenoise_diffusion_v1/training/train_nuplan.py \
    --config "$config_path" \
    --init_ckpt "$INIT_CKPT" \
    2>&1 | tee "$log_path"

  echo "===== DONE ${label} ====="
}

wait_for_cache

run_one \
  "normal_adapter_rank16" \
  "/home/z/code/nuplan_whitenoise_diffusion_v1/tmp/nuplan_diffusion_target_cl5_norm_adapter_r16_finetune_gpu0145_bs256_20260531.yaml" \
  "${MASTER_PORT_R16:-29631}"

run_one \
  "normal_adapter_rank4" \
  "/home/z/code/nuplan_whitenoise_diffusion_v1/tmp/nuplan_diffusion_target_cl5_norm_adapter_r4_finetune_gpu0145_bs256_20260531.yaml" \
  "${MASTER_PORT_R4:-29632}"
