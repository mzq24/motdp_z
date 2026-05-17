#!/bin/bash
set -euo pipefail

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp

export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps

SCRIPT=/workspace1/z_project/code/motdp_z_navsim_motdp/scripts/precompute_bev_cache.py
CACHE=/workspace2/z_project/motdp_bev_cache
TOKEN_LIST=$CACHE/missing_official_tokens.txt
GPUS=(2 3 2 3)
NUM_SHARDS=${#GPUS[@]}

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu=${GPUS[$shard]}
    echo "[missing official] shard=$shard gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python $SCRIPT \
        --shard $shard \
        --num_shards $NUM_SHARDS \
        --cache_dir $CACHE \
        --output_suffix _missing \
        --token_list $TOKEN_LIST \
        > $CACHE/precompute_missing_shard${shard}.log 2>&1 &
done

wait
echo "[missing official] all done"
