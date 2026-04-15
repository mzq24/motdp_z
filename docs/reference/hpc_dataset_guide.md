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

## Stage1 Full Relabeling 产物路径

`new_hpc` 上这一轮 stage1 full relabeling 相关文件，后续排查时优先看这些
小 `json/meta/log`，不要默认反复扫大 `samples_packed.pkl`。

### Full merged 输出

- full relabeled packed：
  - `/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_merged.pkl`
- full dataset episode 统计：
  - `/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/stage1_episode_stats.json`

### 16-way shard 输出

- shard root：
  - `/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_stage1_shards_16`
- 全局 split 摘要：
  - `/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_stage1_shards_16/split_summary.json`
- 单 shard 目录示例：
  - `/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_stage1_shards_16/shard_11_of_16`
- 单 shard 常用文件：
  - `samples_packed.pkl`
  - `split_meta.json`
  - `stage1_relabel.log`

### Scene-level train / val split

- scene split root：
  - `/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_scene_split_95_5`
- train packed：
  - `/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_scene_split_95_5/train/samples_packed.pkl`
- val packed：
  - `/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_scene_split_95_5/val/samples_packed.pkl`
- 如果单独导出过 train / val episode 统计，默认也放在：
  - `.../train/stage1_episode_stats.json`
  - `.../val/stage1_episode_stats.json`

### 当前排查约定

- 想确认 shard 切分是否正确：先看 `split_summary.json` 和单 shard 的 `split_meta.json`
- 想确认 full relabeling 是否已经 merge 回总包：先看 `samples_packed.stage1_merged.pkl`
  的时间戳，再用少量 sample 对照 shard
- 想看 merge / borrow / junction 的总量：先看 `stage1_episode_stats.json`
- 想看 train / val 是否继承了 full relabeling：优先对比 `train/val` 的
  `stage1_episode_stats.json`

## samples_packed 的头尾裁边规则

`samples_packed.pkl` 不是 raw route 的逐帧完整拷贝。当前打包逻辑会系统性裁掉：

- 头部若干帧
- 尾部若干帧

原因在：

- `dataset/preprocess_pdm_lite.py`

当前公式是：

- `scen_start_frame_offset = max((obs_horizon - 1) * hz_interval, 4)`
- `last_valid_future_frame_offset = action_horizon * hz_interval`
- `last_frame_idx = num_seq - last_valid_future_frame_offset - 1`
- 中心帧循环：
  - `for ii in range(scen_start_frame_offset, last_frame_idx, sample_interval):`

这意味着：

- 头部至少会丢前 `max((obs_horizon - 1) * hz_interval, 4)` 帧
- 尾部会丢最后 `action_horizon * hz_interval + 1` 附近的一段中心帧

所以对于 stage1 relabeling / corridor-end 检查，必须明确：

- raw route 的总帧数
- packed 中实际保留的 frame_id 范围

一个已确认的例子：

- `ConstructionObstacleTwoWays/Town12_Rep0_1490_0_route0_11_08_09_11_32`
  - raw route: `0..125`（126 帧）
  - packed: `6..112`（107 帧）
  - 丢失尾帧：`113..125`

这会直接影响依赖尾部可见性的标签，例如：

- `borrow_end_distance_m <= 0.5`

如果 end 发生在 packed 看不到的尾帧里，当前 full relabeling 就会把这类
scene 误读成“没有 end / 没有 active”，即使 raw route / video 中其实还能继续看到。

## Stage1 头尾补全 relabel 流程

如果我们希望：

- 不改 `precompute_semantic_labels.py` 的核心 episode 逻辑
- 但又让 stage1 relabeling 能看到 raw route 的头尾帧

当前推荐流程是：

1. 先基于 raw-vs-packed coverage index，生成一份 **仅给 stage1 使用** 的
   临时 padded packed
2. 在这份 padded packed 上跑 `--stage1_only --force`
3. 再把 relabel 后的 stage1 字段投影回原始 trimmed packed

这样：

- training / `getitem` 继续用原始 trimmed packed
- stage1 labeling 能看到 head/tail 时间轴
- 不需要为了补头尾去改 merge / borrow / junction 的判定逻辑

### 相关脚本

- 构建临时 padded packed：
  - `scripts/data_tools/build_stage1_padded_packed.py`
- 把 padded relabel 的 stage1 字段投影回原始 packed：
  - `scripts/data_tools/project_stage1_fields_from_padded.py`

### 典型调用链

```bash
# 1. 基于 raw-vs-packed coverage index 生成 padded packed
python scripts/data_tools/build_stage1_padded_packed.py \
  --packed_path /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.pkl \
  --coverage_json /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/route_frame_coverage_index.json \
  --image_root /workspace1/z_project/dataset/pdm_lite \
  --output_path /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.pkl \
  --overwrite

# 2. 用现有 shard 流程在 padded packed 上跑 stage1 relabel
SOURCE=/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.pkl \
NUM_SHARDS=16 \
bash codex_bash/split_stage1_full.sh

SHARD_ROOT=/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_stage1_shards_16 \
bash codex_bash/run_stage1_full_shards.sh

BASE=/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.pkl \
SHARD_ROOT=/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_stage1_shards_16 \
OUTPUT=/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.relabel.pkl \
bash codex_bash/merge_stage1_full.sh

# 3. 只把 stage1 字段投影回原始 trimmed packed
python scripts/data_tools/project_stage1_fields_from_padded.py \
  --base /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.pkl \
  --padded_relabel /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_padded.relabel.pkl \
  --output /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh/samples_packed.stage1_merged.pkl \
  --overwrite
```
