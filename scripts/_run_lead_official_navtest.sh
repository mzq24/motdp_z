#!/usr/bin/env bash
set -euo pipefail

# Official LEAD NAVSIM navtest evaluation.
# Default mode is a smoke run over 16 scenarios.
# Use MODE=full to evaluate all navtest scenarios.

MODE="${MODE:-smoke}"
GPU_ID="${GPU_ID:-2}"
CONDA_ENV="${CONDA_ENV:-z_navsim_motdp}"
WORKERS="${WORKERS:-1}"

LEAD_PROJECT_ROOT="${LEAD_PROJECT_ROOT:-/workspace1/z_project/code/lead}"
NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/workspace2/data/navsim}"
NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/workspace2/data/navsim/maps}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/workspace2/z_project/navsim_exp_lead_official}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache_navtest_v1_1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/workspace1/z_project/models/navsim_backbones/tfv6_navsim/model_0060.pth}"
CHECKPOINT_FILE="$(basename "${CHECKPOINT_PATH}")"
CHECKPOINT_DIR="$(dirname "${CHECKPOINT_PATH}")"
RUNTIME_CHECKPOINT_DIR="${RUNTIME_CHECKPOINT_DIR:-${NAVSIM_EXP_ROOT}/checkpoints/tfv6_navsim_model0060_config_ordered}"
TMPDIR="${TMPDIR:-/workspace2/z_project/tmp}"

if [[ "${MODE}" == "full" ]]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-lead_official_navtest_full_model0060}"
  MAX_SCENES="${MAX_SCENES:-}"
else
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-lead_official_navtest_smoke16_model0060}"
  MAX_SCENES="${MAX_SCENES:-16}"
fi

source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TMPDIR
export LEAD_PROJECT_ROOT
export NAVSIM_DEVKIT_ROOT
export OPENSCENE_DATA_ROOT
export NUPLAN_MAPS_ROOT
export NAVSIM_EXP_ROOT
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${LEAD_PROJECT_ROOT}:${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${TMPDIR}" "${NAVSIM_EXP_ROOT}" "${RUNTIME_CHECKPOINT_DIR}"
cd "${LEAD_PROJECT_ROOT}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

if [[ ! -f "${CHECKPOINT_DIR}/config.json" ]]; then
  echo "Missing checkpoint config: ${CHECKPOINT_DIR}/config.json" >&2
  exit 2
fi

if [[ ! -d "${METRIC_CACHE_PATH}" ]]; then
  echo "Missing metric cache: ${METRIC_CACHE_PATH}" >&2
  echo "Build it first with scripts/_run_lead_official_navtest_cache.sh" >&2
  exit 2
fi

# LEADs config loader evaluates some properties while iterating through the
# loaded JSON. Put the NAVSIM switches first so the official config initializes
# as NAVSIM before dependent properties such as target_dataset/epochs are read.
ln -sf "${CHECKPOINT_PATH}" "${RUNTIME_CHECKPOINT_DIR}/${CHECKPOINT_FILE}"
python - "${CHECKPOINT_DIR}/config.json" "${RUNTIME_CHECKPOINT_DIR}/config.json" <<PYCFG
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

ordered = {}
for key, default in [
    ("use_navsim_data", True),
    ("use_carla_data", False),
    ("use_waymo_e2e_data", False),
    ("LTF", True),
    ("use_planning_decoder", True),
]:
    ordered[key] = cfg.get(key, default)

for key, value in cfg.items():
    if key not in ordered:
        ordered[key] = value

with open(dst, "w", encoding="utf-8") as f:
    json.dump(ordered, f, indent=2)
    f.write("\n")
PYCFG

CHECKPOINT_PATH="${RUNTIME_CHECKPOINT_DIR}/${CHECKPOINT_FILE}"

declare -a OVERRIDES=(
  "train_test_split=navtest"
  "experiment_name=${EXPERIMENT_NAME}"
  "agent=carla_transfuser_agent"
  "agent.checkpoint_path=${CHECKPOINT_PATH}"
  "metric_cache_path=${METRIC_CACHE_PATH}"
  "worker=single_machine_thread_pool"
  "worker.max_workers=${WORKERS}"
)

if [[ -n "${MAX_SCENES}" ]]; then
  OVERRIDES+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
fi

if [[ -n "${LOG_NAMES_JSON:-}" ]]; then
  OVERRIDES+=("train_test_split.scene_filter.log_names=${LOG_NAMES_JSON}")
fi

echo "LEAD_PROJECT_ROOT=${LEAD_PROJECT_ROOT}"
echo "NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT}"
echo "NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "RUNTIME_CHECKPOINT_DIR=${RUNTIME_CHECKPOINT_DIR}"
echo "METRIC_CACHE_PATH=${METRIC_CACHE_PATH}"
echo "MODE=${MODE}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "MAX_SCENES=${MAX_SCENES:-FULL}"
echo "LOG_NAMES_JSON=${LOG_NAMES_JSON:-FULL}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "CONDA_ENV=${CONDA_ENV}"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score.py" "${OVERRIDES[@]}"
