# 本地测试
python tools/generate_scenario_videos.py \
    --dataset_root /media/z/data/dataset/pdm_lite_mini \
    --output_dir /tmp/scenario_videos --n_per_type 3

# HPC 全量（所有场景类型，每种3条route）
# nohup bash scripts/hpc_new/generate_videos.sh > logs/generate_videos.log 2>&1 &

# # 只看某一种场景
# python tools/generate_scenario_videos.py \
#     --dataset_root /path/to/dataset \
#     --scenario ConstructionObstacle --n_per_type 5
