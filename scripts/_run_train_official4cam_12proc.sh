#!/bin/bash
source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp
export TMPDIR=/workspace2/z_project/tmp && mkdir -p $TMPDIR
export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps

SCRIPT=/home/z/code/motdp_z_navsim_motdp/scripts/precompute_train_official4cam.py
CACHE=/workspace2/z_project/motdp_bev_cache_train_official4cam
GPUS=(0 1 4 5 6 7)

for chunk in 0 1 2 3 4 5 6 7 8 9 10 11; do
    gpu_idx=$((chunk % 6))
    GPU=${GPUS[$gpu_idx]}
    echo "Chunk $chunk -> GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT \
        --num_shards 1 --shard 0 \
        --cache_dir $CACHE \
        --token_list $CACHE/tokens_c$(printf '%02d' $chunk).txt \
        --output_suffix _c$(printf '%02d' $chunk) \
        > $CACHE/train_c$(printf '%02d' $chunk).log 2>&1 &
done
wait
echo ALL DONE
