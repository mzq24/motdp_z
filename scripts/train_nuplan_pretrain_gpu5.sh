#!/bin/bash
set -euo pipefail

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp

cd /home/z/code/nuplan_whitenoise_diffusion_v1

export PYTHONPATH=/home/z/code/nuplan_whitenoise_diffusion_v1
export CUDA_VISIBLE_DEVICES=5

# Stage 1: pretrain the diffusion backbone on the current full PlanTF-cache mix.
# Keep this single-GPU launcher simple and stable; finetune should be launched
# separately after the pretrain checkpoint is ready.
/workspace1/miniconda/envs/z_navsim_motdp/bin/python   /home/z/code/nuplan_whitenoise_diffusion_v1/training/train_nuplan.py   --config /home/z/code/nuplan_whitenoise_diffusion_v1/configs/nuplan_diffusion.yaml
