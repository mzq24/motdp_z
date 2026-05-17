#!/bin/bash
source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp
export TMPDIR=/workspace2/z_project/tmp && mkdir -p $TMPDIR
export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps
export NAVSIM_EXP_ROOT=/workspace2/z_project/navsim_exp_motdp
export PYTHONPATH=/home/z/code/lead:/home/z/code/lead/3rd_party/navsim_workspace/navsimv2.2:/home/z/code/motdp_z_navsim_motdp/scripts:$PYTHONPATH

python /home/z/code/motdp_z_navsim_motdp/scripts/run_carla_tf_pdm.py     train_test_split=navtest     train_test_split.scene_filter.max_scenes=4     experiment_name=motdp_carla_tf_smoke4     agent=carla_transfuser_agent     +agent.checkpoint_path=/workspace1/z_project/models/navsim_backbones/tfv6_navsim/model_0060.pth     metric_cache_path=/workspace2/data/navsim/processed_data/metric_cache_navtest     worker=single_machine_thread_pool     worker.max_workers=1
echo DONE
