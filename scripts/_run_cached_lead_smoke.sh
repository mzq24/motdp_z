#!/bin/bash
source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp
export TMPDIR=/workspace2/z_project/tmp && mkdir -p $TMPDIR
export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps
export NAVSIM_EXP_ROOT=/workspace2/z_project/navsim_exp_motdp
export NAVSIM_DEVKIT_ROOT=/home/z/code/navsim
export PYTHONPATH=/home/z/code/motdp_z_navsim_motdp:$PYTHONPATH

python /home/z/code/navsim/navsim/planning/script/run_pdm_score_one_stage.py     train_test_split=navtrain     'train_test_split.scene_filter.max_scenes=8'     experiment_name=motdp_cached_lead_smoke8     agent._target_=navsim_motdp.agents.cached_lead_agent.CachedLeadAgent     +agent.checkpoint_path=/workspace1/z_project/models/navsim_backbones/tfv6_navsim/model_0060.pth     +agent.cache_dir=/workspace2/z_project/motdp_bev_cache_navtest_npy     +agent.device=cuda     metric_cache_path=/workspace2/z_project/motdp_metric_cache_navtrain_smoke_64     worker=single_machine_thread_pool     worker.max_workers=1
echo DONE
