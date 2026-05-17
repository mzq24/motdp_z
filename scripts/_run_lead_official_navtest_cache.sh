#!/usr/bin/env bash
set -euo pipefail

# Build a NAVSIM v1.1 metric cache compatible with LEADs vendored devkit.
# Default mode builds a smoke cache over 16 navtest scenarios.
# Use MODE=full to cache all navtest scenarios.

MODE="${MODE:-smoke}"
CONDA_ENV="${CONDA_ENV:-z_navsim_motdp}"
WORKERS="${WORKERS:-1}"

LEAD_PROJECT_ROOT="${LEAD_PROJECT_ROOT:-/workspace1/z_project/code/lead}"
NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/workspace2/data/navsim}"
NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/workspace2/data/navsim/maps}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/workspace2/z_project/navsim_exp_lead_official}"
CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache_navtest_v1_1}"
TMPDIR="${TMPDIR:-/workspace2/z_project/tmp}"

if [[ "${MODE}" == "full" ]]; then
  MAX_SCENES="${MAX_SCENES:-}"
else
  MAX_SCENES="${MAX_SCENES:-16}"
fi

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export TMPDIR
export LEAD_PROJECT_ROOT
export NAVSIM_DEVKIT_ROOT
export OPENSCENE_DATA_ROOT
export NUPLAN_MAPS_ROOT
export NAVSIM_EXP_ROOT
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${LEAD_PROJECT_ROOT}:${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${TMPDIR}" "${NAVSIM_EXP_ROOT}" "${CACHE_PATH}"
cd "${LEAD_PROJECT_ROOT}"

declare -a OVERRIDES=(
  "train_test_split=navtest"
  "cache.cache_path=${CACHE_PATH}"
  "worker=single_machine_thread_pool"
  "worker.max_workers=${WORKERS}"
)

if [[ -n "${MAX_SCENES}" ]]; then
  OVERRIDES+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
fi

echo "LEAD_PROJECT_ROOT=${LEAD_PROJECT_ROOT}"
echo "NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT}"
echo "CACHE_PATH=${CACHE_PATH}"
echo "MODE=${MODE}"
echo "MAX_SCENES=${MAX_SCENES:-FULL}"
echo "CONDA_ENV=${CONDA_ENV}"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" "${OVERRIDES[@]}"
