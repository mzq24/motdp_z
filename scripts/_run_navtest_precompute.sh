#!/bin/bash
source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp
export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps
export TMPDIR=/workspace2/z_project/tmp
mkdir -p $TMPDIR
SCRIPT=/home/z/code/motdp_z_navsim_motdp/scripts/precompute_navtest.py
CACHE=/workspace2/z_project/motdp_bev_cache_navtest
mkdir -p $CACHE

GPUS=(4 5 6 7)
for s in 0 1 2 3; do
    gpu=${GPUS[$s]}
    echo "shard $s GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python $SCRIPT --num_shards 4 --shard $s --cache_dir $CACHE > $CACHE/shard${s}.log 2>&1 &
done
wait
echo DONE
