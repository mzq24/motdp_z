#!/bin/bash
set -euo pipefail

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp

export PYTHONPATH=/workspace1/z_project/code/motdp_z_navsim_motdp:${PYTHONPATH:-}
export TMPDIR=/workspace2/z_project/tmp
mkdir -p "$TMPDIR"
export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps
export NAVSIM_EXP_ROOT=/workspace2/z_project/navsim_exp_motdp
export NAVSIM_DEVKIT_ROOT=/home/z/code/navsim

GPU_ID=${GPU_ID:-2}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-motdp_cached_navtest_npy_full_e60}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/workspace2/z_project/motdp_logs/navsim_simple_official_lr1e4_e60_b64/best_model.pt}
CACHE_DIR=${CACHE_DIR:-/workspace2/z_project/motdp_bev_cache_navtest_npy}
METRIC_CACHE_PATH=${METRIC_CACHE_PATH:-/workspace2/data/navsim/processed_data/metric_cache_navtest}
MAX_SCENES=${MAX_SCENES:-}

CMD=(
  python /home/z/code/navsim/navsim/planning/script/run_pdm_score_one_stage.py
  train_test_split=navtest
  experiment_name="$EXPERIMENT_NAME"
  agent._target_=navsim_motdp.agents.cached_diffusion_agent.CachedDiffusionAgent
  +agent.checkpoint_path="$CHECKPOINT_PATH"
  +agent.cache_dir="$CACHE_DIR"
  +agent.device="cuda:${GPU_ID}"
  +agent.fallback=raise
  metric_cache_path="$METRIC_CACHE_PATH"
  worker=single_machine_thread_pool
  worker.max_workers=1
)

if [[ -n "$MAX_SCENES" ]]; then
  CMD+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
fi

"${CMD[@]}"
echo DONE
