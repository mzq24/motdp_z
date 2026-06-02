#!/usr/bin/env bash
set -euo pipefail

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp

cd /home/z/code/nuplan_whitenoise_diffusion_v1
export PYTHONPATH=/home/z/code/nuplan_whitenoise_diffusion_v1:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5,6,7

/workspace1/miniconda/envs/z_navsim_motdp/bin/torchrun \
  --nproc_per_node=4 \
  --master_port=29523 \
  /home/z/code/nuplan_whitenoise_diffusion_v1/training/train_nuplan.py \
  --config /home/z/code/nuplan_whitenoise_diffusion_v1/configs/nuplan_diffusion_source_overlap_3x100k.yaml
