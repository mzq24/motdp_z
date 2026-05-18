#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${GPU_ID:-4}
REPO=${REPO:-/workspace1/z_project/code/motdp_z_navsim_motdp}
CONDA_ENV=${CONDA_ENV:-z_navsim_motdp}
LEAD_PROJECT_ROOT=${LEAD_PROJECT_ROOT:-/workspace1/z_project/code/lead}
NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1}
OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-/workspace2/data/navsim}
NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/workspace2/data/navsim/maps}
NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/workspace2/z_project/navsim_exp_motdp}
METRIC_CACHE_PATH=${METRIC_CACHE_PATH:-/workspace2/z_project/navsim_exp_lead_official/metric_cache_navtest_v1_1}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/workspace1/z_project/models/navsim_backbones/tfv6_navsim/model_0060.pth}
CACHE_DIR=${CACHE_DIR:-/workspace2/z_project/motdp_bev_cache_navtest_official4cam_officialacc_npy}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-motdp_cached_lead_officialacc_smoke16}
MAX_SCENES=${MAX_SCENES-16}
WORKERS=${WORKERS:-1}
TMPDIR=${TMPDIR:-/workspace2/z_project/tmp}

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export TMPDIR OPENSCENE_DATA_ROOT NUPLAN_MAPS_ROOT NAVSIM_EXP_ROOT HYDRA_FULL_ERROR=1
export PYTHONPATH=${REPO}:${LEAD_PROJECT_ROOT}:${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}
mkdir -p "${TMPDIR}" "${NAVSIM_EXP_ROOT}"
cd "${REPO}"

OVERRIDES=(
  "train_test_split=navtest"
  "experiment_name=${EXPERIMENT_NAME}"
  "agent._target_=navsim_motdp.agents.cached_lead_agent.CachedLeadAgent"
  "+agent.checkpoint_path=${CHECKPOINT_PATH}"
  "+agent.cache_dir=${CACHE_DIR}"
  "+agent.device=cuda:0"
  "+agent.fallback=raise"
  "metric_cache_path=${METRIC_CACHE_PATH}"
  "worker=single_machine_thread_pool"
  "worker.max_workers=${WORKERS}"
)

if [[ -n "${MAX_SCENES}" ]]; then
  OVERRIDES+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
fi

OVERRIDES+=("$@")

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score.py" "${OVERRIDES[@]}"
