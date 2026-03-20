#!/bin/bash
# =============================================================================
# Route B+ 本地测试脚本（单卡 4090）
# 用法:
#   bash scripts/training/train_route_b_local.sh          # 从头训练
#   bash scripts/training/train_route_b_local.sh --resume /path/to/ckpt.pth  # 断点续训
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ===== 可调参数 =====
CONFIG="config/pdm_local_route_b.yaml"
ANCHOR_NPY="dd_baseline/anchors/carla_kmeans_32.npy"   # (32, 6, 2) K-means anchor
GPU_ID=0

# 小 batch 快速验证（改完确认没问题后可以去 yaml 里调回 128）
OVERRIDE_BATCH_SIZE=128
OVERRIDE_EPOCHS=300

# ===== 检查依赖 =====
echo "========================================="
echo "  Route B+ Local Training"
echo "========================================="
echo "Project root: $PROJECT_ROOT"
echo "Config:       $CONFIG"
echo "Anchor:       $ANCHOR_NPY"
echo ""

# 检查 config
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

# 检查 anchor npy
if [ ! -f "$ANCHOR_NPY" ]; then
    echo "ERROR: Anchor file not found: $ANCHOR_NPY"
    echo "  Generate with: python bridge_baseline/scripts/generate_anchors.py \\"
    echo "    --dataset_path /media/z/data/dataset/pdm_lite_mini/train \\"
    echo "    --key agent_pos --num_modes 32 --num_poses 6 --fast \\"
    echo "    --output $ANCHOR_NPY"
    exit 1
fi

# 打印 anchor 信息
python -c "import numpy as np; a=np.load('$ANCHOR_NPY'); print(f'  Anchor shape: {a.shape}, dtype: {a.dtype}')"

# 检查 dataset
DATASET_PATH=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('training',{}).get('dataset_path',''))")
if [ -n "$DATASET_PATH" ] && [ ! -d "$DATASET_PATH" ]; then
    echo "WARNING: Dataset path not found: $DATASET_PATH"
    echo "  Training will fail at data loading."
fi

# 检查 GPU
echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  No GPU detected!"
echo ""

# ===== 生成临时 config（覆盖 batch_size, epochs, anchor_path）=====
TMP_CONFIG="/tmp/route_b_local_$(date +%s).yaml"
python - "$CONFIG" "$TMP_CONFIG" "$ANCHOR_NPY" "$OVERRIDE_BATCH_SIZE" "$OVERRIDE_EPOCHS" <<'PYEOF'
import sys, yaml, os

config_path, out_path, anchor_npy, bs, epochs = sys.argv[1:]

with open(config_path) as f:
    cfg = yaml.safe_load(f)

# Override for local testing
cfg['anchor_path'] = os.path.abspath(anchor_npy)
cfg['dataloader']['batch_size'] = int(bs)
cfg['training']['num_epochs'] = int(epochs)
cfg['device']['gpu_ids'] = [0]

# 本地测试: wandb online
cfg['logging']['use_wandb'] = True
cfg.setdefault('logging', {})['run_name'] = "route_b_local_test"

# 确保 checkpoint 目录存在
ckpt_dir = cfg.get('training', {}).get('checkpoint_dir', 'checkpoints/route_b_local_test/')
os.makedirs(ckpt_dir, exist_ok=True)

with open(out_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

print(f"Temp config written to: {out_path}")
print(f"  batch_size: {bs}")
print(f"  epochs: {epochs}")
print(f"  anchor_path: {os.path.abspath(anchor_npy)}")
print(f"  checkpoint_dir: {ckpt_dir}")
PYEOF

echo ""

# ===== 启动训练 =====
EXTRA_ARGS=""
# 透传所有参数（如 --resume）
for arg in "$@"; do
    EXTRA_ARGS="$EXTRA_ARGS $arg"
done

echo "Starting training..."
echo "  CUDA_VISIBLE_DEVICES=$GPU_ID"
echo "  python training/train_carla_bev.py --config_path $TMP_CONFIG $EXTRA_ARGS"
echo "========================================="
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID python training/train_carla_bev.py \
    --config_path "$TMP_CONFIG" \
    $EXTRA_ARGS

# 清理临时 config
rm -f "$TMP_CONFIG"
echo ""
echo "Done."
