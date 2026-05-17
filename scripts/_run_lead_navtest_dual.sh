#!/bin/bash
source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp
export TMPDIR=/workspace2/z_project/tmp && mkdir -p $TMPDIR
export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps
export NAVSIM_EXP_ROOT=/workspace2/z_project/navsim_exp_motdp
export NAVSIM_DEVKIT_ROOT=/home/z/code/navsim
export PYTHONPATH=/home/z/code/motdp_z_navsim_motdp:$PYTHONPATH

run_on_gpu() {
    local gpu=$1 exp=$2
    CUDA_VISIBLE_DEVICES=$gpu python /home/z/code/navsim/navsim/planning/script/run_pdm_score_one_stage.py         train_test_split=navtest         experiment_name=$exp         agent._target_=navsim_motdp.agents.online_lead_agent.OnlineLeadDiffusionAgent         +agent.checkpoint_path=/workspace2/z_project/motdp_logs/navsim_simple_official_lr1e4_e60_b64/best_model.pt         +agent.device=cuda         metric_cache_path=/workspace2/data/navsim/processed_data/metric_cache_navtest         worker=single_machine_thread_pool         worker.max_workers=1 &
}

run_on_gpu 0 motdp_lead_navtest_full_gpu0
run_on_gpu 1 motdp_lead_navtest_full_gpu1
wait
echo DONE
