#!/bin/bash
source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp
export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps

SCRIPT=/home/z/code/motdp_z_navsim_motdp/scripts/precompute_bev_cache.py
GPUS=(0 1 4 5)

for s in 0 1 2 3; do
    gpu=${GPUS[$s]}
    echo "[tmux] Shard $s on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python $SCRIPT --shard $s --num_shards 4         > /workspace2/z_project/motdp_bev_cache/shard${s}.log 2>&1 &
done
wait
echo "[tmux] ALL DONE"
