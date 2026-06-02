#!/usr/bin/env bash
set -euo pipefail

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp

cd /home/z/code/nuplan_whitenoise_diffusion_v1
export PYTHONPATH=/home/z/code/nuplan_whitenoise_diffusion_v1:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}

CONFIG_PATH=${CONFIG_PATH:-/home/z/code/nuplan_whitenoise_diffusion_v1/tmp/nuplan_diffusion_target_cl5_norm_adapter_pegp_finetune_gpu23.yaml}
MASTER_PORT=${MASTER_PORT:-29532}
INIT_CKPT=${INIT_CKPT:-}
RESUME_CKPT=${RESUME_CKPT:-}

if [[ -n "$INIT_CKPT" && -n "$RESUME_CKPT" ]]; then
  echo "Set only one of INIT_CKPT or RESUME_CKPT" >&2
  exit 1
fi

if [[ -z "$INIT_CKPT" && -z "$RESUME_CKPT" ]]; then
  echo "Set INIT_CKPT for a warm start or RESUME_CKPT for stage resume" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "$RESUME_CKPT" ]]; then
  EXTRA_ARGS+=(--resume "$RESUME_CKPT")
else
  EXTRA_ARGS+=(--init_ckpt "$INIT_CKPT")
fi

/workspace1/miniconda/envs/z_navsim_motdp/bin/torchrun \
  --nproc_per_node=2 \
  --master_port="$MASTER_PORT" \
  /home/z/code/nuplan_whitenoise_diffusion_v1/training/train_nuplan.py \
  --config "$CONFIG_PATH" \
  "${EXTRA_ARGS[@]}"