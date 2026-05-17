#!/bin/bash
source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp
export TMPDIR=/workspace2/z_project/tmp && mkdir -p $TMPDIR
export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps

SCRIPT=/home/z/code/motdp_z_navsim_motdp/scripts/precompute_train_official4cam.py
CACHE=/workspace2/z_project/motdp_bev_cache_train_official4cam
GPUS=(0 1 4 5 6 7)

for i in 0 1 2 3 4 5; do
    GPU=${GPUS[$i]}
    echo "GPU $GPU (token list $i)"
    CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT \
        --num_shards 1 --shard 0 \
        --cache_dir $CACHE \
        --token_list $CACHE/tokens_gpu$i.txt \
        --output_suffix _gpu$i \
        > $CACHE/train_gpu${i}.log 2>&1 &
done
wait
echo ALL DONE
