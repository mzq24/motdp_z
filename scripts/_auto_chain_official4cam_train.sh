#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace1/z_project/code/motdp_z_navsim_motdp
CACHE_DIR=/workspace2/z_project/motdp_bev_cache_train_official4cam
NPY_DIR=/workspace2/z_project/motdp_bev_cache_train_official4cam_npy
LOG_ROOT=/workspace2/z_project/motdp_logs
CHAIN_NAME=official4cam_auto_chain_$(date +%Y%m%d_%H%M%S)
CHAIN_LOG=${LOG_ROOT}/${CHAIN_NAME}.log
TRAIN_LOG_DIR=${LOG_ROOT}/navsim_official4cam_e90_gpus0145_gb64_$(date +%Y%m%d_%H%M%S)
EXPECTED_SHARDS=16
EXPECTED_MIN_TOKENS=100000
PRECOMPUTE_PATTERN='python .*/scripts/precompute_train_official4cam.py'
TRAIN_GPUS=0,1,4,5
PER_GPU_BATCH=16
WORLD_SIZE=4

mkdir -p "$LOG_ROOT" "$CACHE_DIR" "$NPY_DIR"
exec > >(tee -a "$CHAIN_LOG") 2>&1

cd "$REPO"
source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp
export TMPDIR=/workspace2/z_project/tmp
export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$TMPDIR"

echo "[chain] start $(date)"
echo "[chain] repo=$REPO"
echo "[chain] cache=$CACHE_DIR"
echo "[chain] npy=$NPY_DIR"
echo "[chain] train_log_dir=$TRAIN_LOG_DIR"
echo "[chain] train_gpus=$TRAIN_GPUS per_gpu_batch=$PER_GPU_BATCH global_batch=$((PER_GPU_BATCH * WORLD_SIZE)) epochs=90"

while pgrep -f "$PRECOMPUTE_PATTERN" >/dev/null; do
    running=$(pgrep -f "$PRECOMPUTE_PATTERN" | wc -l)
    final_count=$(find "$CACHE_DIR" -maxdepth 1 -name 'bev_cache_shard*.npz' ! -name '*_tmp.npz' 2>/dev/null | wc -l)
    tmp_count=$(find "$CACHE_DIR" -maxdepth 1 -name '*_tmp.npz' 2>/dev/null | wc -l)
    size=$(du -sh "$CACHE_DIR" 2>/dev/null | awk '{print $1}')
    echo "[chain] $(date) waiting precompute: running=$running final=$final_count tmp=$tmp_count cache_size=${size:-NA}"
    sleep 300
done

echo "[chain] precompute processes exited at $(date); waiting 30s for filesystem flush"
sleep 30

mapfile -t final_shards < <(find "$CACHE_DIR" -maxdepth 1 -name 'bev_cache_shard*.npz' ! -name '*_tmp.npz' -type f | sort)
echo "[chain] final shards: ${#final_shards[@]}"
printf '  %s\n' "${final_shards[@]}"

if [[ ${#final_shards[@]} -ne $EXPECTED_SHARDS ]]; then
    echo "[chain][ERROR] expected $EXPECTED_SHARDS final shards, got ${#final_shards[@]}"
    exit 2
fi

for shard in "${final_shards[@]}"; do
    base=$(basename "$shard")
    if [[ ! "$base" =~ ^bev_cache_shard000_c[0-9][0-9]\.npz$ ]]; then
        echo "[chain][ERROR] unexpected final shard name: $base"
        exit 3
    fi
    size_bytes=$(stat -c%s "$shard")
    if [[ "$size_bytes" -le 0 ]]; then
        echo "[chain][ERROR] empty final shard: $base"
        exit 4
    fi
done

echo "[chain] dry-run NPY conversion"
DRY_LOG=${CACHE_DIR}/convert_dryrun_$(date +%Y%m%d_%H%M%S).log
/workspace1/miniconda/envs/z_navsim_motdp/bin/python scripts/convert_navsim_cache_to_npy.py \
    --input-dir "$CACHE_DIR" \
    --output-dir "$NPY_DIR" \
    --dedupe-tokens \
    --overwrite \
    --dry-run | tee "$DRY_LOG"

kept=$(awk '/Kept entries:/ {print $3}' "$DRY_LOG" | tail -1)
if [[ -z "${kept:-}" ]]; then
    echo "[chain][ERROR] failed to parse kept token count from $DRY_LOG"
    exit 5
fi
if [[ "$kept" -lt $EXPECTED_MIN_TOKENS ]]; then
    echo "[chain][ERROR] kept token count too low: $kept < $EXPECTED_MIN_TOKENS"
    exit 6
fi

echo "[chain] converting NPZ -> NPY at $(date)"
/workspace1/miniconda/envs/z_navsim_motdp/bin/python scripts/convert_navsim_cache_to_npy.py \
    --input-dir "$CACHE_DIR" \
    --output-dir "$NPY_DIR" \
    --dedupe-tokens \
    --overwrite
echo "[chain] converted metadata:"
cat "$NPY_DIR/metadata.json"

/workspace1/miniconda/envs/z_navsim_motdp/bin/python - <<PY
import json, sys
from pathlib import Path
meta = json.loads(Path('$NPY_DIR/metadata.json').read_text())
count = int(meta['count'])
print('[chain] npy count', count)
if count < $EXPECTED_MIN_TOKENS:
    raise SystemExit(f'npy token count too low: {count}')
for name in ['cache_index.npz', 'bev_grid.npy', 'bev_feature.npy', 'ego_status.npy', 'trajectory.npy']:
    path = Path('$NPY_DIR') / name
    print('[chain]', name, path.stat().st_size if path.exists() else 'MISSING')
    if not path.exists():
        raise SystemExit(f'missing {path}')
PY

echo "[chain] starting 4-GPU DDP training at $(date)"
echo "[chain] logs: $TRAIN_LOG_DIR"
mkdir -p "$TRAIN_LOG_DIR"
export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
export OMP_NUM_THREADS=4
PYTHONUNBUFFERED=1 /workspace1/miniconda/envs/z_navsim_motdp/bin/torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$WORLD_SIZE \
    training/train_navsim_diffusion_ddp.py \
    --cache-dir "$NPY_DIR" \
    --log-dir "$TRAIN_LOG_DIR" \
    --load-mode memmap \
    --epochs 90 \
    --batch-size "$PER_GPU_BATCH" \
    --lr 1e-4 \
    --lr-final 1e-6 \
    --warmup-epochs 3 \
    --weight-decay 1e-4 \
    --val-ratio 0.05 \
    --val-every-epochs 5 \
    --save-every-epochs 5 \
    --max-keep-ckpts 10 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --amp-dtype bf16 \
    --use-amp

echo "[chain] finished $(date)"
echo "[chain] training log dir: $TRAIN_LOG_DIR"
