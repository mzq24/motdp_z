#!/bin/bash
source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp
export TMPDIR=/workspace2/z_project/tmp
mkdir -p $TMPDIR
export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps
export NAVSIM_EXP_ROOT=/workspace2/z_project/navsim_exp_motdp
export NAVSIM_DEVKIT_ROOT=/home/z/code/navsim

CUDA_VISIBLE_DEVICES=4 python /home/z/code/navsim/navsim/planning/script/run_pdm_score_one_stage.py     train_test_split=navtrain     'train_test_split.scene_filter.max_scenes=16'     experiment_name=motdp_online_lead_smoke16     agent._target_=navsim_motdp.agents.online_lead_agent.OnlineLeadDiffusionAgent     +agent.checkpoint_path=/workspace2/z_project/motdp_logs/navsim_simple_official_lr1e4_e60_b64/best_model.pt     +agent.device=cuda     metric_cache_path=/workspace2/z_project/motdp_metric_cache_navtrain_smoke_64     worker=single_machine_thread_pool     worker.max_workers=1
echo DONE
