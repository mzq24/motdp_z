#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME=${SESSION_NAME:-clt}
UNIT_NAME=${UNIT_NAME:-nuplan_gpu67_idlewatch}
TRAIN_PATTERN=${TRAIN_PATTERN:-/home/z/code/nuplan_whitenoise_diffusion_v1/configs/nuplan_diffusion_source_overlap_3x100k_norm.yaml}
LOG_DIR=${LOG_DIR:-/workspace2/z_project/exp/nuplan/nuplan_whitenoise_diffusion_v1/source_overlap_3x100k_norm_20260517/logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/train_gpu67.log}
LAUNCHER=${LAUNCHER:-/home/z/code/nuplan_whitenoise_diffusion_v1/scripts/train_nuplan_source_overlap_3x100k_norm_gpu67.sh}

if pgrep -f "$TRAIN_PATTERN" >/dev/null 2>&1; then
  echo "training_already_running"
  systemctl --user stop "${UNIT_NAME}.timer" >/dev/null 2>&1 || true
  exit 0
fi

GPU67_PIDS=$(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+$' || true)
if [[ -n "$GPU67_PIDS" ]]; then
  echo "gpu67_busy"
  echo "$GPU67_PIDS"
  exit 0
fi

mkdir -p "$LOG_DIR"
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux new-session -d -s "$SESSION_NAME"
fi

tmux send-keys -t "$SESSION_NAME" "bash $LAUNCHER 2>&1 | tee $LOG_FILE" C-m
echo "launched_in_tmux:$SESSION_NAME"
systemctl --user stop "${UNIT_NAME}.timer" >/dev/null 2>&1 || true