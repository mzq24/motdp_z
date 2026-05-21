#!/usr/bin/env bash
set -euo pipefail

# Run after the standard stage1 relabel chain has produced:
#   samples_packed.stage1_padded.relabel.tempocc_v2.chase.phaseobj.vbmin.graph.consistency.pkl
#
# This script appends the state-motion alignment label patch, projects the
# patched padded labels back to train-index samples, creates a scene holdout
# split, then writes offline prev-semantic-state fields independently for
# train/val so links never cross the split.

PYTHON_BIN="${PYTHON_BIN:-python}"

FULL_ROOT="${FULL_ROOT:-/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh}"
BASE_PACKED="${BASE_PACKED:-${FULL_ROOT}/samples_packed.pkl}"

PADDED_CONSISTENCY="${PADDED_CONSISTENCY:-${FULL_ROOT}/samples_packed.stage1_padded.relabel.tempocc_v2.chase.phaseobj.vbmin.graph.consistency.pkl}"
PADDED_ALIGNMENT="${PADDED_ALIGNMENT:-${FULL_ROOT}/samples_packed.stage1_padded.relabel.tempocc_v2.chase.phaseobj.vbmin.graph.consistency.align.pkl}"
MERGED_ALIGNMENT="${MERGED_ALIGNMENT:-${FULL_ROOT}/samples_packed.stage1_merged.tempocc_v2.chase.phaseobj.vbmin.graph.consistency.align.pkl}"
ALIGNMENT_SUMMARY="${ALIGNMENT_SUMMARY:-${FULL_ROOT}/state_motion_alignment_label_summary.json}"

SPLIT_ROOT="${SPLIT_ROOT:-/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_scene_split_95_5_alignment}"
VAL_SCENE_RATIO="${VAL_SCENE_RATIO:-0.05}"
SPLIT_SEED="${SPLIT_SEED:-3407}"
PREV_MAX_FRAME_GAP="${PREV_MAX_FRAME_GAP:-20}"
SEMANTIC_HISTORY_DEPTH="${SEMANTIC_HISTORY_DEPTH:-4}"

if [[ ! -f "${PADDED_CONSISTENCY}" ]]; then
  echo "Missing upstream padded consistency labels: ${PADDED_CONSISTENCY}" >&2
  echo "Run the existing codex_bash/relabeling.sh chain through boundary-speed consistency first." >&2
  exit 1
fi

echo "[1/4] state-motion alignment postprocess"
"${PYTHON_BIN}" scripts/data_tools/postprocess_stage1_state_motion_alignment.py \
  --input_path "${PADDED_CONSISTENCY}" \
  --output_path "${PADDED_ALIGNMENT}" \
  --summary_json "${ALIGNMENT_SUMMARY}" \
  --overwrite_existing

echo "[2/4] project alignment labels back to training index"
"${PYTHON_BIN}" scripts/data_tools/project_stage1_fields_from_padded.py \
  --base "${BASE_PACKED}" \
  --padded_relabel "${PADDED_ALIGNMENT}" \
  --output "${MERGED_ALIGNMENT}" \
  --overwrite

echo "[3/4] scene holdout split"
"${PYTHON_BIN}" scripts/data_tools/build_scene_holdout_split.py \
  --packed-path "${MERGED_ALIGNMENT}" \
  --out-root "${SPLIT_ROOT}" \
  --val-scene-ratio "${VAL_SCENE_RATIO}" \
  --seed "${SPLIT_SEED}" \
  --overwrite

echo "[4/4] offline prev semantic state for train/val"
"${PYTHON_BIN}" scripts/data_tools/postprocess_semantic_prev_state.py \
  --input "${SPLIT_ROOT}/train/samples_packed.pkl" \
  --output "${SPLIT_ROOT}/train/samples_packed.pkl" \
  --summary-json "${SPLIT_ROOT}/train/prev_state_summary.json" \
  --max-frame-gap "${PREV_MAX_FRAME_GAP}" \
  --history-depth "${SEMANTIC_HISTORY_DEPTH}" \
  --overwrite

"${PYTHON_BIN}" scripts/data_tools/postprocess_semantic_prev_state.py \
  --input "${SPLIT_ROOT}/val/samples_packed.pkl" \
  --output "${SPLIT_ROOT}/val/samples_packed.pkl" \
  --summary-json "${SPLIT_ROOT}/val/prev_state_summary.json" \
  --max-frame-gap "${PREV_MAX_FRAME_GAP}" \
  --history-depth "${SEMANTIC_HISTORY_DEPTH}" \
  --overwrite

echo "Done. Training dataset_path should be:"
echo "  ${SPLIT_ROOT}"
echo "Alignment summary:"
echo "  ${ALIGNMENT_SUMMARY}"
