#!/bin/bash
# Training script for NuPlan White-Noise Diffusion
# Run on newhpc with DDP

set -e

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp

cd /home/z/code/nuplan_whitenoise_diffusion_v1

# Compute normalization stats if not already done
if [ ! -f /workspace2/z_project/exp/nuplan/cache_motdp_v1/norm_stats.json ]; then
    echo "Computing normalization stats..."
    python scripts/compute_norm_stats.py \
        --data_dir /workspace2/z_project/exp/nuplan/cache_motdp_v1/
fi

# Launch DDP training
echo "Starting training..."
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    training/train_nuplan.py \
    --config configs/nuplan_diffusion.yaml
