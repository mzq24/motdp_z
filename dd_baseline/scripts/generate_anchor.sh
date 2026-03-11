#!/bin/bash
cd /media/z/data/mzq/others/MoT-DP

python dd_baseline/generate_anchors.py \
    --dataset_path /media/z/data/dataset/pdm_lite_mini/train \
    --num_modes 20 \
    --num_poses 6
