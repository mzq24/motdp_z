#!/bin/bash
# Launch 4-GPU parallel BEV precomputation (15w navtrain sliding window)
SCRIPT=/home/z/code/motdp_z_navsim_motdp/scripts/precompute_bev_cache.py
NUM_SHARDS=4
LOG_DIR=/workspace2/z_project/motdp_bev_cache

mkdir -p $LOG_DIR

export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD nohup \
        conda run -n z_navsim_motdp python $SCRIPT \
        --num_shards $NUM_SHARDS --shard $SHARD \
        > $LOG_DIR/precompute_shard${SHARD}.log 2>&1 &
    echo "Launched shard $SHARD on GPU $SHARD (PID $!)"
done

echo "Monitor: tail -f $LOG_DIR/precompute_shard*.log"
