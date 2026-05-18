#!/usr/bin/env bash
set -euo pipefail

# Suggested use:
#   tmux new-session -d -s nt 'cd /workspace1/z_project/code/motdp_z_navsim_motdp && bash scripts/train_navsim_joint_route_speed_ddp_b128_template.sh'

REPO=${REPO:-/workspace1/z_project/code/motdp_z_navsim_motdp}
CACHE_DIR=${CACHE_DIR:-/workspace2/z_project/motdp_bev_cache_train_official4cam_officialacc_npy}
LABEL_DIR=${LABEL_DIR:-/workspace2/z_project/motdp_navsim_labels/navtrain_official_h8_r50_sparsemask}
STATS_DIR=${STATS_DIR:-/workspace2/z_project/motdp_navsim_norm_stats/navtrain_official_h8_r50_20260518_sparsemask}
LOG_ROOT=${LOG_ROOT:-/workspace2/z_project/motdp_logs}

TRAIN_GPUS=${TRAIN_GPUS:-0,1,4,5}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
PER_GPU_BATCH=${PER_GPU_BATCH:-128}
EPOCHS=${EPOCHS:-90}
LR=${LR:-1e-4}
LR_FINAL=${LR_FINAL:-1e-6}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
VAL_RATIO=${VAL_RATIO:-0.05}
VAL_EVERY_EPOCHS=${VAL_EVERY_EPOCHS:-5}
SAVE_EVERY_EPOCHS=${SAVE_EVERY_EPOCHS:-5}
MAX_KEEP_CKPTS=${MAX_KEEP_CKPTS:-10}
NUM_WORKERS=${NUM_WORKERS:-4}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
AMP_DTYPE=${AMP_DTYPE:-bf16}
ROUTE_LOSS_WEIGHT=${ROUTE_LOSS_WEIGHT:-1.0}
SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-0.5}
RUN_NAME=${RUN_NAME:-navsim_joint_route_speed_officialacc_e${EPOCHS}_gpus${TRAIN_GPUS//,/}_b${PER_GPU_BATCH}_pergpu_$(date +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-${LOG_ROOT}/${RUN_NAME}}

TRAJ_STATS_PATH=${TRAJ_STATS_PATH:-${STATS_DIR}/navtrain_official_h8_r50_sparsemask_abs_stats.npz}
ROUTE_STATS_PATH=${ROUTE_STATS_PATH:-${STATS_DIR}/navtrain_official_h8_r50_sparsemask_route_abs_stats.npz}
SPEED_STATS_PATH=${SPEED_STATS_PATH:-${STATS_DIR}/navtrain_official_h8_r50_sparsemask_speed_profile_stats.npz}

cd "$REPO"
source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp

export TMPDIR=${TMPDIR:-/workspace2/z_project/tmp}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-/workspace2/data/navsim}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/workspace2/data/navsim/maps}
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$LOG_DIR"

echo "[train-joint] repo=$REPO"
echo "[train-joint] cache=$CACHE_DIR"
echo "[train-joint] labels=$LABEL_DIR"
echo "[train-joint] stats=$STATS_DIR"
echo "[train-joint] log_dir=$LOG_DIR"
echo "[train-joint] gpus=$TRAIN_GPUS nproc=$NPROC_PER_NODE per_gpu_batch=$PER_GPU_BATCH global_batch=$((PER_GPU_BATCH * NPROC_PER_NODE))"
echo "[train-joint] epochs=$EPOCHS lr=$LR lr_final=$LR_FINAL warmup=$WARMUP_EPOCHS weight_decay=$WEIGHT_DECAY"

/workspace1/miniconda/envs/z_navsim_motdp/bin/torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    training/train_navsim_joint_route_speed_ddp.py \
    --cache-dir "$CACHE_DIR" \
    --label-dir "$LABEL_DIR" \
    --log-dir "$LOG_DIR" \
    --load-mode memmap \
    --epochs "$EPOCHS" \
    --batch-size "$PER_GPU_BATCH" \
    --lr "$LR" \
    --lr-final "$LR_FINAL" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --weight-decay "$WEIGHT_DECAY" \
    --val-ratio "$VAL_RATIO" \
    --val-every-epochs "$VAL_EVERY_EPOCHS" \
    --save-every-epochs "$SAVE_EVERY_EPOCHS" \
    --max-keep-ckpts "$MAX_KEEP_CKPTS" \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --traj-stats-path "$TRAJ_STATS_PATH" \
    --route-stats-path "$ROUTE_STATS_PATH" \
    --speed-stats-path "$SPEED_STATS_PATH" \
    --route-loss-weight "$ROUTE_LOSS_WEIGHT" \
    --speed-loss-weight "$SPEED_LOSS_WEIGHT" \
    --use-raw-bev-feature \
    --raw-bev-dim 512 \
    --amp-dtype "$AMP_DTYPE" \
    --use-amp
