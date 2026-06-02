#!/usr/bin/env bash
set -euo pipefail

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp

cd /home/z/code/nuplan_whitenoise_diffusion_v1
export PYTHONPATH=/home/z/code/nuplan_whitenoise_diffusion_v1:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1

INIT_CKPT=${INIT_CKPT:-/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/source_overlap_3x100k_norm_20260517/checkpoints/epoch_0119.pth}

/workspace1/miniconda/envs/z_navsim_motdp/bin/torchrun \
  --nproc_per_node=2 \
  --master_port=29527 \
  /home/z/code/nuplan_whitenoise_diffusion_v1/training/train_nuplan.py \
  --config /home/z/code/nuplan_whitenoise_diffusion_v1/tmp/nuplan_diffusion_target_overlap_3x100k_norm_finetune_gpu01.yaml \
  --init_ckpt "$INIT_CKPT"