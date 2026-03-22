# HPC Dataset Guide

## 数据集文件结构

```
${DATASET_ROOT}/
├── train/
│   ├── samples_packed.pkl          # 所有 sample 的 pkl 聚合（route, agent_pos, speed 等小数据）
│   │                                 首次训练由 rank 0 自动生成，后续直接读
│   ├── tmp_data/                   # memmap cache（由 build_feature_cache_fp16.py 一次性生成）
│   │   ├── feature_index.pkl       # 索引：route → (offset, n_frames, frame_num_to_idx)
│   │   ├── bev_features_fp16.bin   # 全部 frame 的 BEV 特征 (total_frames, 1512, 8, 8) float16
│   │   └── bev_upsamples_fp16.bin  # 全部 frame 的 BEV upsample (total_frames, 64, 32, 32) float16
│   │                                 存储时 2x 空间降采样，读取时 F.interpolate 回 64×64
│   ├── <event>/<route>/
│   │   └── transfuser_feature/
│   │       └── route_features.pt   # 单 route 的 packed 特征（LRU fallback 用）
│   └── *.pkl                       # 单 sample pkl（首次聚合后不再直接读）
└── val/
    ├── samples_packed.pkl
    └── ...（同 train 结构）
```

## 数据加载层级

`CARLAImageDataset.__getitem__` 中 BEV 特征有 4 条加载路径，按优先级：

| 优先级 | 路径 | 条件 | 说明 |
|--------|------|------|------|
| 1 | per_frame | `use_per_frame=True` | 本地 SSD，逐帧读 `*_feature.pt` |
| 2 | RAM | `inject_ram_features()` | val 专用，预加载到内存 |
| 3 | memmap | `feature_index.pkl` 存在 | **HPC 主要路径**，np.memmap → OS page cache |
| 4 | LRU | 以上均不满足 | 加载整个 `route_features.pt`，慢 |

小数据（route, agent_pos, speed, command 等）来自 `samples_packed.pkl`，启动时一次性读入内存，`__getitem__` 中直接从 `self._sample_cache[idx]` 取，零 IO。

## 大小参考

| 文件 | 大小 | 说明 |
|------|------|------|
| `bev_features_fp16.bin` | ~130GB | (1512×8×8) × 2 bytes × total_frames |
| `bev_upsamples_fp16.bin` | ~70GB | (64×32×32) × 2 bytes × total_frames |
| `feature_index.pkl` | ~几 MB | 索引，读入内存 |
| `samples_packed.pkl` | ~几百 MB | 所有小数据，读入内存 |

## tmpfs 加速（/tmp）

HPC 节点通常有 tmpfs 挂载的 /tmp（与内存共用），独占节点时可将 memmap .bin 复制到 /tmp，避免 Lustre 网络 IO 抖动。

**原理**：memmap 在 Lustre 上依赖 OS page cache，内存紧张时页面会被 evict，重新 fault-in 要走网络 IO，导致训练 step 时间波动。tmpfs 上的文件常驻内存，不会被 evict（swap=0 时）。

**前提**：
- 节点是独占的（`who` 或 `nvidia-smi` 确认无其他用户）
- /tmp 空间 > BEV .bin 总大小（~200GB）
- 总 RAM 减去 BEV 大小后仍足够训练（模型+梯度+activations 约 30-50GB）

**PBS 脚本**：
```bash
DATASET_ROOT=/path/to/pdm_lite

# 复制 memmap 到 tmpfs
echo "Copying BEV cache to tmpfs..."
mkdir -p /tmp/tmp_data
cp ${DATASET_ROOT}/train/tmp_data/feature_index.pkl /tmp/tmp_data/
cp ${DATASET_ROOT}/train/tmp_data/bev_features_fp16.bin /tmp/tmp_data/
cp ${DATASET_ROOT}/train/tmp_data/bev_upsamples_fp16.bin /tmp/tmp_data/
echo "Done. $(du -sh /tmp/tmp_data/)"

# 训练
python bridge_baseline/train.py --config bridge_baseline/bd_config_hpc.yaml
```

**Config**：
```yaml
dataset:
  use_per_frame: false
  cache_dir: /tmp/tmp_data    # 覆盖默认的 {image_data_root}/tmp_data
```

`cache_dir` 只影响 memmap .bin 的查找路径，`image_data_root` 仍指向 Lustre（pkl 索引、路径拼接用）。

## 统计脚本

`bridge_baseline/scripts/compute_route_stats.py` 计算 per-waypoint 归一化统计量。

```bash
# 快速模式：直接读 samples_packed.pkl，跳过 CARLAImageDataset（推荐）
python bridge_baseline/scripts/compute_route_stats.py \
    --dataset_path /path/to/pdm_lite/train \
    --key all --fast \
    --output_yaml bridge_baseline/bd_config.yaml

# 慢速模式：走 dataset.__getitem__，会触发 BEV 加载
python bridge_baseline/scripts/compute_route_stats.py \
    --dataset_path /path/to/pdm_lite/train \
    --key all \
    --output_yaml bridge_baseline/bd_config.yaml
```

`--key all` 一次遍历同时统计 `route[:10]` 和 `agent_pos[:6]`，分别写入 `norm_*` 和 `traj_norm_*`。

## build_feature_cache_fp16.py

一次性脚本，将所有 `route_features.pt` 合并为 flat binary + 索引。

```bash
python scripts/build_feature_cache_fp16.py --dataset_root /path/to/pdm_lite/train
```

输出到 `{dataset_root}/tmp_data/`，只需跑一次。
