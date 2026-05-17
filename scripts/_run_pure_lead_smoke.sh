#!/bin/bash
source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp
export TMPDIR=/workspace2/z_project/tmp && mkdir -p $TMPDIR
export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps
export NAVSIM_EXP_ROOT=/workspace2/z_project/navsim_exp_motdp
export NAVSIM_DEVKIT_ROOT=/home/z/code/navsim
export PYTHONPATH=/home/z/code/motdp_z_navsim_motdp:$PYTHONPATH

python /home/z/code/navsim/navsim/planning/script/run_pdm_score_one_stage.py     train_test_split=navtest     'train_test_split.scene_filter.max_scenes=8'     experiment_name=motdp_pure_lead_smoke8     agent._target_=navsim_motdp.agents.pure_lead_agent.PureLeadAgent     +agent.device=cuda:0     metric_cache_path=/workspace2/data/navsim/processed_data/metric_cache_navtest     worker=single_machine_thread_pool     worker.max_workers=1
echo DONE
