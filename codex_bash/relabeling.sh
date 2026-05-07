# new_hpc repo roots:
#   /workspace1/z_project/code/motdp_z
#   /home/z/code/motdp_z
# These resolve to the same path on new_hpc.


# padding from training to full


# merge train val


# full split to 16 shards


# relabeling on shards
# SHARD_ROOT=/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_stage1_shards_16 \
# bash codex_bash/run_stage1_full_shards.sh


# merge shards to padded relabel
# BASE=/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.pkl \
# SHARD_ROOT=/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_stage1_shards_16 \
# OUTPUT=/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.relabel.pkl \
# bash codex_bash/merge_stage1_full.sh


# temporary occupancy postprocess on padded relabel
# python scripts/data_tools/postprocess_stage1_temporary_occupancy_cover.py \
#   --input_path /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.relabel.pkl \
#   --output_path /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.relabel.tempocc_v2.pkl \
#   --overwrite_existing


# chase/front-following postprocess
# python scripts/data_tools/postprocess_stage1_chase_front_following.py \
#   --input_path /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.relabel.tempocc_v2.pkl \
#   --output_path /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.relabel.tempocc_v2.chase.pkl \
#   --overwrite_existing


# phase-object binding postprocess
# python scripts/data_tools/postprocess_stage1_phase_object_binding.py \
#   --input_path /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.relabel.tempocc_v2.chase.pkl \
#   --output_path /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.relabel.tempocc_v2.chase.phaseobj.pkl \
#   --overwrite_existing


# merge follow-through vbmin postprocess
# python scripts/data_tools/postprocess_stage1_merge_follow_through_vbmin.py \
#   --input_path /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.relabel.tempocc_v2.chase.phaseobj.pkl \
#   --output_path /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.relabel.tempocc_v2.chase.phaseobj.vbmin.pkl \
#   --overwrite_existing


# boundary speed consistency audit on non-collision routes
# python scripts/data_tools/postprocess_stage1_boundary_speed_consistency.py \
#   --input_path /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.relabel.tempocc_v2.chase.phaseobj.vbmin.pkl \
#   --output_path /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.relabel.tempocc_v2.chase.phaseobj.vbmin.consistency.pkl \
#   --image_data_root /workspace1/z_project/dataset/pdm_lite \
#   --issue_csv /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/boundary_speed_consistency_issues.csv \
#   --overwrite_existing


# project back to training index
# python scripts/data_tools/project_stage1_fields_from_padded.py \
#   --base /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.pkl \
#   --padded_relabel /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.relabel.tempocc_v2.chase.phaseobj.vbmin.consistency.pkl \
#   --output /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_merged.tempocc_v2.chase.phaseobj.vbmin.consistency.pkl \
#   --overwrite


# split train val again for holdout
# python scripts/data_tools/build_scene_holdout_split.py \
#   --packed-path /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_merged.tempocc_v2.chase.phaseobj.vbmin.consistency.pkl \
#   --out-root /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_scene_split_95_5 \
#   --val-scene-ratio 0.05 \
#   --seed 3407 \
#   --overwrite


# shard-level debug rerun example (local / remote style)
# DATA_ROOT=/media/z/data/dataset/pdm_lite \
# DEBUG_ROOT=/media/z/data/dataset/pdm_lite/tmp_data/debug \
# SRC_SHARD=/media/z/data/dataset/pdm_lite/tmp_data/debug/full_scene_refresh_stage1_shard08_debug_16_merged.pkl \
# SUB_ROOT=/media/z/data/dataset/pdm_lite/tmp_data/debug/full_scene_refresh_stage1_shard08_debug_16_local_parts \
# OUT_PKL=/media/z/data/dataset/pdm_lite/tmp_data/debug/full_scene_refresh_stage1_shard08_debug_16_merged.local.pkl \
# OUT_JSON=/media/z/data/dataset/pdm_lite/tmp_data/debug/full_scene_refresh_stage1_shard08_debug_16_conflict_area_stats.local.json \
# IMAGE_DATA_ROOT=/media/z/data/dataset/pdm_lite \
# PYTHON_BIN=/home/z/anaconda3/envs/dpauto/bin/python \
# bash codex_bash/tmp.sh > logs/stage1_shard08_rerun.log 2>&1


# shard-level tempocc postprocess example
# /home/z/anaconda3/envs/dpauto/bin/python scripts/data_tools/postprocess_stage1_temporary_occupancy_cover.py \
#   --input_path /media/z/data/dataset/pdm_lite/tmp_data/debug/full_scene_refresh_stage1_shard01_debug_16_merged.local.pkl \
#   --output_path /media/z/data/dataset/pdm_lite/tmp_data/debug/full_scene_refresh_stage1_shard01_debug_16_merged.local.tempocc.pkl \
#   --overwrite_existing


# video examples
# /home/z/anaconda3/envs/dpauto/bin/python tools/generate_stage1_label_video_lite.py \
#   --packed_path /media/z/data/dataset/pdm_lite/tmp_data/debug/full_scene_refresh_stage1_shard01_debug_16_merged.local.tempocc.pkl \
#   --image_data_root /media/z/data/dataset/pdm_lite \
#   --route_list_txt /media/z/data/mzq/others/MoT-DP/codex_bash/shard_01_of_16.merge_debug_routes.txt \
#   --output_dir /media/z/data/mzq/others/MoT-DP/visualizations/stage1_label_videos_merge/shard_01_of_16 \
#   --fps 8
