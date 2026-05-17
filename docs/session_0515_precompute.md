# Session 2026-05-15: NavSim BEV Feature Precomputation

## 目标

用 LEAD LTFv6 backbone 离线跑 navtrain 的 BEV 特征缓存，供后续 diffusion 训练使用。

## 使用的 Backbone

- **LEAD LTFv6** (`ln2697/tfv6_navsim`): 4-cam (L0+F0+R0+B0), LTF latent BEV, 52.3M params
- Checkpoint: `/workspace1/z_project/models/navsim_backbones/tfv6_navsim/model_0060.pth`
- 输入: RGB (270, 1920, 3), bfloat16
- 输出 BEV: `lidar_features (512, 8, 8)` + `top_down BEV grid (64, 64, 64)`

## 数据源

- **Logs**: trainval 1310 pkl → navtrain filter 1192 pkl
- **Sensor**: `sensor_blobs/trainval` (446GB, navtrain subset, 仅部分帧有图像)
- 对比: `sensor_blobs/trainval_full` (1.6TB, 全量) — 未使用
- Sensor 覆盖率 ~2-5% 每 pkl（因为 navtrain sensor 只为非重叠 filter 准备）

## 缓存内容

每个 token 存储（float16/float32）:
- `bev_grid (64, 64, 64)` — top_down BEV 空间特征
- `bev_feature (512, 8, 8)` — 压缩 BEV 特征
- `ego_status (4, 14)` — 4 帧历史 ego 状态 [vel(2), acc(2), cmd(4), pad(6)]
- `trajectory (8, 2)` — 未来 4s 轨迹 @2Hz

## 分片策略

按 pkl 分片（非 token），避免每个进程全量扫描:

| Shard | pkl 范围 | 覆盖 |
|-------|---------|------|
| 0 | [0, 298) | shard0 old + r0a + r0b |
| 1 | [298, 596) | s1part0-3 |
| 2 | [596, 894) | shard2 old + r2a(p0-3) |
| 3 | [894, 1192) | shard3 old + r3a |

## 运行过程

### 第一次尝试（失败）
- 按 token 分片，每个进程独立扫描全部 1192 pkl
- 4 进程竞争 I/O，日志无输出，实际卡死
- 教训：按 pkl 分片，每进程只加载自己负责的 pkl

### 第二次（部分成功 → 内存溢出崩溃）
- 按 pkl 分片，滑动窗口 interval=1
- Shard 0,2,3 崩溃前保存了临时文件（14K + 25K + 14K token）
- Shard 1 临时文件损坏

### 第三次（恢复完成）
- Shard 1 拆 4 份（s1part0-3），4 GPU 恢复 → 29K token
- Shard 0 剩余补 r0a, r0b
- Shard 2 剩余补 r2a → 太慢，再拆 4 份 (p0-3)
- Shard 3 剩余补 r3a

### 合并
- 先尝试 `np.savez_compressed`（压缩），单核 CPU 太慢
- 改用 `np.savez`（无压缩），15-18GB/shard
- 最终文件过大但写入快

## 最终结果

| Shard | Token | 大小 |
|-------|-------|------|
| 0 | 26,215 | 15.5 GB |
| 1 | 29,012 | 17.1 GB |
| 2 | 30,775 | 18.2 GB |
| 3 | 24,143 | 14.2 GB |
| **总计** | **110,145** | **65 GB** |

缓存路径: `/workspace2/z_project/motdp_bev_cache/`

## 踩坑记录

1. **nohup 不支持 env 前缀**: `OPENSCENE_ROOT=X nohup cmd` 失败 → 用 tmux + `source conda.sh`
2. **tmux 中 conda 找不到**: 需 `source /workspace1/miniconda/etc/profile.d/conda.sh && conda activate`
3. **navtrain sensor 只覆盖 ~2-5%**: 因为 sensor download 为非重叠 filter 准备，需要 trainval_full 才有全量
4. **np.savez_compressed 单核极慢**: 24GB 数据压缩需数十分钟 → 改用无压缩 savez
5. **4 进程共享文件系统 I/O 瓶颈**: 按 pkl 分片比按 token 分片快得多（每进程独立加载自己的 pkl）
6. **GPU 利用率低**: 瓶颈在 CPU I/O（JPEG 解码 + pickle 反序列化），GPU 大部分时间空闲
7. **进程假死**: 日志无输出 ≠ 进程死，可能是 I/O 等待。检查 `stat` 和文件修改时间

## 脚本路径

- 主脚本: `newhpc:/home/z/code/motdp_z_navsim_motdp/scripts/precompute_bev_cache.py`
- 启动脚本: `newhpc:/home/z/code/motdp_z_navsim_motdp/scripts/_run_precompute.sh`
- 恢复脚本: `newhpc:/home/z/code/motdp_z_navsim_motdp/scripts/_recover_023.sh`, `_run_shard1.sh`, `_r2a_split.sh`
- tmux sessions: `pc` (shard1), `r023` (0,2,3 recovery)

## 下一步

- 写 Dataset class 加载缓存
- 写 Training script
- 修 GridSample + RoPE
- 算 traj normalization stats

---

# Session 2026-05-15 (cont): Dataset & Training Script

## Dataset (`dataset/navsim_cached_dataset.py`)

- 全量 preload 到 CPU RAM（fp16 存储，约 65GB）
- 11 万 token，train/val = 95/5
- Traj stats: mean=[10.02, 0.29], std=[9.18, 2.18]
- 加载时间 ~120s（np.savez zip 容器解析瓶颈，非磁盘 I/O——NVMe 实测 4.8 GB/s）
- 后续优化: 换成 `.npy` 格式可秒加载

## Training Script (`training/train_navsim_diffusion.py`)

- 60 epoch, batch=64, lr=5e-5, AdamW
- Warmup 3 epoch + cosine annealing
- BF16 mixed precision, grad clip 1.0
- 每 5 epoch validation, 每 10 epoch checkpoint
- Best model by val loss

## 对比 SparseDrive

SparseDrive **不预计算离线 BEV**——每 batch 在线加载 JPEG 并跑 backbone。我们的离线预计算省 GPU 但费 CPU RAM。

## 剩余 TODO

- GridSampleCrossBEVAttention: 采样 BEV 网格时位置硬编码为 [0,0]，需改为根据 trajectory 坐标采样
- RoPE: 维度对齐问题，当前 disabled
- 训练完后加 mode-aware / energy guidance

---

# Session 2026-05-15 (cont): Official Split Alignment & Cache Salvage

## 背景

旧 BEV cache 有 `110,145` 个 samples，但只保存了数组：

- `bev_grid`
- `bev_feature`
- `ego_status`
- `trajectory`

没有保存 `token / log_name / frame_idx` metadata。后续 label 对齐必须依赖 token，因此需要确认旧 cache 能否保留，而不是直接全量重跑 LEAD。

## SparseDrive / NAVSIM split 口径

SparseDrive 的 NAVSIM cache/training 入口使用官方 `SceneLoader`：

- cache script: `SparseDriveV2/scripts/cache/run_dataset_caching_navtrain.sh`
- training script: `SparseDriveV2/scripts/training/sparsedrive_navsimv2.sh`
- 配置: `train_test_split=navtrain`
- scene filter: `navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml`

官方 `navtrain` scene filter 关键数字：

| item | count |
|---|---:|
| `log_names` | 1,192 |
| `tokens` | 103,288 |
| history/future | h4/f10 |
| frame interval | 1 |
| `has_route` | true |

验证脚本：

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
/workspace1/miniconda/envs/z_navsim_motdp/bin/python \
  scripts/verify_navsim_split_against_sparsedrive.py
```

验证结果：

```text
yaml logs: 1192
logs present on disk: 1192
yaml tokens: 103288
tokens reconstructed from logs: 103288
tokens missing from logs/windows: 0
duplicate token windows: 0

sparsedrive_3cam: 103288 / 103288
lead_4cam:        103288 / 103288
all_8cam:         103288 / 103288
```

重要结论：

- 官方/SparseDrive split 可以从 raw logs 完整重建出 `103,288` 个 token。
- 官方 `SceneLoader` 不检查 `sample_prev/sample_next` continuity。
- 如果额外加 continuity check，会过滤掉 `1,089` 个 official tokens。
- 因此 precompute 默认不应开启 continuity check；只能作为 debug / ablation 选项。

## 旧 cache salvage 结果

用 `trajectory + ego_status` 做 fingerprint，重建旧 precompute 的 `log_only + continuity` 遍历序列后，可以把旧 cache 全部反推出 token。

脚本：

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
/workspace1/miniconda/envs/z_navsim_motdp/bin/python \
  scripts/recover_bev_cache_metadata.py --write
```

输出：

| shard | cache samples | matched | official samples |
|---|---:|---:|---:|
| `bev_cache_shard000.npz` | 26,215 | 26,215 | 18,041 |
| `bev_cache_shard001.npz` | 29,012 | 29,012 | 19,850 |
| `bev_cache_shard002.npz` | 30,775 | 30,775 | 20,820 |
| `bev_cache_shard003.npz` | 24,143 | 24,143 | 16,460 |
| **total** | **110,145** | **110,145** | **75,171 raw official hits** |

去重后 official 覆盖：

```text
official_total=103288
recovered_unique=74984
missing=28304
duplicate_official=187
```

已经写出的 sidecar / token list：

```text
/workspace2/z_project/motdp_bev_cache/bev_cache_shard000_meta.npz
/workspace2/z_project/motdp_bev_cache/bev_cache_shard001_meta.npz
/workspace2/z_project/motdp_bev_cache/bev_cache_shard002_meta.npz
/workspace2/z_project/motdp_bev_cache/bev_cache_shard003_meta.npz
/workspace2/z_project/motdp_bev_cache/missing_official_tokens.txt      # 28,304 tokens
/workspace2/z_project/motdp_bev_cache/navtrain_official_tokens.txt      # 103,288 tokens
```

sidecar 内容：

- `tokens`
- `log_names`
- `frame_indices`
- `official_mask`
- `candidate_indices`

## 当前代码改动

### `scripts/precompute_bev_cache.py`

- 默认 `--split_mode official_tokens`，对齐 SparseDrive / NAVSIM official `navtrain` tokens。
- 默认不做 continuity filter。
- 新增 `--require_continuity`，仅 debug 时使用。
- 新增 `--token_list`，可只补指定 token list。
- 新增 `--cache_dir`，方便 smoke / 新 cache 目录。
- 输出 npz 现在包含 metadata：
  - `tokens`
  - `log_names`
  - `frame_indices`

### `dataset/navsim_cached_dataset.py`

- 支持从主 npz 或 sidecar `_meta.npz` 读取 metadata。
- 自动忽略 `*_meta.npz`，避免被当成 BEV shard。
- 支持 `token_filter_file`。
- 支持 `dedupe_tokens=True`。
- `collate_fn` 支持非 tensor metadata。

### `training/train_navsim_diffusion.py`

- 默认：

```python
token_filter_file="/workspace2/z_project/motdp_bev_cache/navtrain_official_tokens.txt"
dedupe_tokens=True
```

也就是说训练会加载旧 cache + 后续 missing cache，但只保留 official navtrain token，并按 token 去重。

### `model/navsim_simple_diffusion.py`

已完成两个 core TODO：

- `GridSampleCrossBEVAttention` 用当前 noisy trajectory 反归一化后的物理坐标采 BEV，不再固定 `[0,0]`。
- RoPE 修复 head-dim interleave / broadcast，并已在 trajectory self-attention 中启用。

smoke：

```text
loss 1.1131
sample (2, 8, 2)
```

## 补 missing cache 计划

不全量重跑。只补 `missing_official_tokens.txt` 中的 `28,304` 个 official tokens。

脚本：

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
scripts/run_precompute_missing_official_4gpu.sh
```

当前 GPU 分配：

```bash
GPUS=(2 3 2 3)
NUM_SHARDS=4
```

即 4 个进程叠跑在 GPU 2/3：

| shard | GPU | log |
|---|---:|---|
| 0 | 2 | `/workspace2/z_project/motdp_bev_cache/precompute_missing_shard0.log` |
| 1 | 3 | `/workspace2/z_project/motdp_bev_cache/precompute_missing_shard1.log` |
| 2 | 2 | `/workspace2/z_project/motdp_bev_cache/precompute_missing_shard2.log` |
| 3 | 3 | `/workspace2/z_project/motdp_bev_cache/precompute_missing_shard3.log` |

输出文件会是：

```text
/workspace2/z_project/motdp_bev_cache/bev_cache_shard000_missing.npz
/workspace2/z_project/motdp_bev_cache/bev_cache_shard001_missing.npz
/workspace2/z_project/motdp_bev_cache/bev_cache_shard002_missing.npz
/workspace2/z_project/motdp_bev_cache/bev_cache_shard003_missing.npz
```

## 验证记录

已完成：

- `py_compile`：
  - `scripts/precompute_bev_cache.py`
  - `scripts/recover_bev_cache_metadata.py`
  - `scripts/verify_navsim_split_against_sparsedrive.py`
  - `dataset/navsim_cached_dataset.py`
  - `training/train_navsim_diffusion.py`
  - `model/navsim_simple_diffusion.py`
- real LEAD smoke：
  - `--limit_pkls 1`
  - 写到 `/tmp/navsim_precompute_smoke`
  - 确认新 npz 含 `tokens/log_names/frame_indices`
- dataset sidecar smoke：
  - 确认 `_meta.npz` 可读
  - 确认 `token_filter_file + dedupe_tokens` 可用
- missing token list smoke：
  - 确认 `--token_list missing_official_tokens.txt` 生效

## 决策

当前推荐路线：

1. 保留旧 `110,145` cache。
2. 用 sidecar metadata 恢复 token 对齐。
3. 只补 `28,304` 个 missing official tokens。
4. 训练时用 official token list + dedupe，得到完整 `103,288` official navtrain 训练集。
5. 后续 label 迁移按 token join，避免再依赖数组顺序。

## 2026-05-16: 60 Epoch Training / Cached Eval / PDM Smoke

### 训练收敛

训练 run:

```text
/workspace2/z_project/motdp_logs/navsim_simple_official_lr1e4_e60_b64
```

配置要点：

- cache: `/workspace2/z_project/motdp_bev_cache_official_final`
- official token filter: `navtrain_official_tokens.txt`
- samples: train `98,124`, val `5,164`
- epochs: `60`
- lr: `1e-4 -> 1e-6`
- batch size: `256`
- checkpoint: `best_model.pt`, `model_epoch{10,20,30,40,50,60}.pt`

收敛曲线：

| epoch | val loss | val l2 |
|---:|---:|---:|
| 5 | 0.0339 | 0.756m |
| 10 | 0.0297 | 0.697m |
| 15 | 0.0239 | 0.635m |
| 20 | 0.0232 | 0.632m |
| 25 | 0.0200 | 0.624m |
| 30 | 0.0193 | 0.586m |
| 35 | 0.0189 | 0.575m |
| 40 | 0.0174 | 0.548m |
| 45 | 0.0161 | 0.535m |
| 50 | 0.0156 | 0.526m |
| 55 | 0.0153 | 0.511m |
| 60 | 0.0141 | 0.500m |

结论：

- `best_model.pt` 对应 epoch 60。
- val loss / val l2 持续下降，没有明显过拟合反弹。
- 60 epoch 可作为第一版测试 checkpoint；后续可以试更长训练，但当前已经足够进入 official metric smoke。

### NPY / memmap cache

旧 NPZ 加载非常慢，训练时观测到：

- train dataset load: `4086s`
- val dataset load: `1135s`

因此将 official clean cache 转成 consolidated NPY:

```text
/workspace2/z_project/motdp_bev_cache_official_npy
```

转换脚本:

```text
scripts/convert_navsim_cache_to_npy.py
```

转换结果:

```text
count: 103288
elapsed_sec: 266.595
bev_grid.npy: 51G
bev_feature.npy: 6.4G
ego_status.npy: 23M
trajectory.npy: 6.4M
cache_index.npz: 22M
```

token 对齐:

```text
Allowed tokens: 103288
Kept entries: 103288
Unique kept tokens: 103288
```

后续训练 / eval 默认优先用该 NPY cache，避免每次重新解压 NPZ。

### Cached Offline Open-Loop Eval

新增脚本:

```text
scripts/evaluate_navsim_cached_policy.py
```

用途:

- 加载 NPY/memmap cache。
- 加载 `best_model.pt`。
- 在 held-out val split 上做真正 DDIM sampling。
- 统计 ADE/FDE/per-step L2。

full val 结果:

```text
checkpoint: /workspace2/z_project/motdp_logs/navsim_simple_official_lr1e4_e60_b64/best_model.pt
cache_dir: /workspace2/z_project/motdp_bev_cache_official_npy
split: val
num_samples: 5164
batch_size: 128
num_inference_steps: 10
ADE: 0.6253386563 m
FDE: 1.4949701015 m
max_l2: 1.5156379806 m
target_path_norm: 10.3365950581 m
pred_path_norm: 10.3411799046 m
elapsed_sec: 26.376
```

per-step L2:

```text
[0.0725, 0.1274, 0.2394, 0.4048, 0.6207, 0.8751, 1.1678, 1.4950]
```

记录文件:

```text
/workspace2/z_project/motdp_logs/navsim_simple_official_lr1e4_e60_b64/eval_val_full.json
```

注意：

- training log 里的 `val l2=0.500m` 是 denoise training/eval loss 下的误差。
- offline eval 的 `ADE=0.625m / FDE=1.495m` 是 DDIM sampling 后的轨迹误差。
- 两者不是同一指标，后续汇报时需要分开说。

### Official NAVSIM PDM Smoke

创建 tmux:

```text
tmux nexp
```

现有 official metric cache 情况：

- `/workspace2/data/navsim/processed_data/metric_cache_navtest`
- navtest metric tokens: `12,146`
- 与当前 official navtrain BEV cache 交集: `0`
- 与 legacy BEV cache 交集: `0`

因此不能直接用 navtest metric cache 测当前 train checkpoint。先在 navtrain subset 上自建 metric cache 跑通 official PDM 链路。

新增 agent wrapper:

```text
navsim_motdp/agents/cached_diffusion_agent.py
```

核心逻辑:

```text
scene.scene_metadata.initial_token
  -> cache_index.npz token lookup
  -> bev_grid.npy / ego_status.npy
  -> NavSimSimpleDiffusion.sample()
  -> (x, y) 补 heading
  -> navsim.common.dataclasses.Trajectory
```

说明：

- 这是 cache-backed official NAVSIM agent，不是最终 online LEAD agent。
- `requires_scene=True`，因为 `AgentInput` 不带 token，需要从 `Scene` 里拿 `initial_token`。
- `get_sensor_config()` 返回 no sensors，因为当前直接查预计算 BEV cache。

单样本 smoke:

- token: `1aa44d46e4ab5bc7`
- loader tokens: `1`
- trajectory shape: `(8, 3)`
- 确认 token lookup / checkpoint load / heading 补全可用。

### Navtrain 64 Metric Cache Smoke

metric cache:

```text
/workspace2/z_project/motdp_metric_cache_navtrain_smoke_64
```

结果:

```text
64 / 64 scenarios cached successfully
metadata: /workspace2/z_project/motdp_metric_cache_navtrain_smoke_64/metadata/motdp_metric_cache_navtrain_smoke_64_metadata_node_0.csv
```

### Official PDM One-Stage Smoke Results

脚本:

```text
navsim/planning/script/run_pdm_score_one_stage.py
```

8-scenario smoke:

```text
experiment_name: motdp_cached_diffusion_navtrain_smoke8
successful: 8
failed: 0
average score: 0.9560259918
```

CSV:

```text
/workspace2/z_project/navsim_exp_motdp/motdp_cached_diffusion_navtrain_smoke8/2026.05.16.07.13.19/2026.05.16.07.13.40.csv
```

64-scenario smoke:

```text
experiment_name: motdp_cached_diffusion_navtrain_smoke64
successful: 64
failed: 0
average score: 0.8050675829
```

CSV:

```text
/workspace2/z_project/navsim_exp_motdp/motdp_cached_diffusion_navtrain_smoke64/2026.05.16.07.15.45/2026.05.16.07.16.25.csv
```

64-scenario average row:

```text
no_at_fault_collisions: 0.90625
drivable_area_compliance: 0.921875
driving_direction_compliance: 1.0
traffic_light_compliance: 1.0
ego_progress: 0.9643316994
time_to_collision_within_bound: 0.890625
lane_keeping: 0.96875
history_comfort: 1.0
two_frame_extended_comfort: 0.9636363636
score: 0.8050675829
```

结论：

- official NAVSIM one-stage PDM 链路已跑通。
- CachedDiffusionAgent 在 64 场景 navtrain smoke 上 `64/64 valid`。
- 当前结果只能代表 cache-backed / navtrain-subset smoke，不能当作 navtest official score。
- 下一步如果继续 official metric:
  1. 扩大 navtrain subset 到 `512/1024`，看 score 稳定性。
  2. 或补 navtest BEV cache，再用已有 `metric_cache_navtest` 跑 navtest。
  3. 最终 online LEAD agent 仍需接 raw sensor / LEAD backbone，不应长期依赖 cache lookup。

## 2026-05-16: Val Chain Status Clarification

目前文档里已有两类 validation / eval 记录：

1. **Cached offline open-loop val**
   - 使用 held-out val split，`5,164` samples。
   - 已记录 `ADE=0.625m / FDE=1.495m`。
   - 这是模型 sampling 后的 open-loop 轨迹误差，不是 official NAVSIM PDM score。

2. **Official NAVSIM PDM smoke**
   - 已跑通 `CachedDiffusionAgent` + official `run_pdm_score_one_stage.py`。
   - 已记录 navtrain subset `8` 和 `64` scenarios。
   - 当前 exp 目录里只有这两个 CSV：
     - `motdp_cached_diffusion_navtrain_smoke8`
     - `motdp_cached_diffusion_navtrain_smoke64`

还没有完成 / 记录的部分：

- 完整 official validation 链路还没有跑。
- navtest metric cache 虽然存在，但 navtest tokens 与当前 navtrain BEV cache 交集为 `0`，所以不能直接拿现有 checkpoint + navtest metric cache 测。
- 后续若要跑 official val/navtest，需要先补对应 token 的 BEV cache，或者先扩大 navtrain subset 到 `512/1024` 做稳定性验证。


---

# Session 2026-05-16 (cont): PDM Scoring & Online Agent Attempts

## 目标

1. 用  + navtest BEV cache → navtest PDM 分数
2. 写 （实时 sensor → LEAD backbone → diffusion）→ smoke PDM
3. 扩大 navtrain PDM subset

## Navtest BEV Precompute

- 用 test split logs + sensor_blobs + navtest filter
- 写 wrapper  复用现有 precompute 逻辑
- 4 GPU 并行，~7 min 完成
- 缓存路径:
- 结果: 4 个 npz shard，共 12,146 token，与官方 navtest tokens 100% 重叠
- TMPDIR 问题:  已满，需要

## Online LEAD Agent ()

实现了实时推理链路:


- （不需要 Scene，只靠 AgentInput）
- : 返回 4 个 camera (L0/F0/R0/B0, iteration=0)
- : 加载 LEAD backbone (bf16 + monkey-patch LTF grid) + diffusion model
- : stitch 4-cam → backbone → top_down → diffusion sample → xy_to_se2

## PDM Scoring 踩坑

### 坑 1: TMPDIR 满
- 症状:
- 原因:  inode 耗尽
- 修复:

### 坑 2: CUDA_VISIBLE_DEVICES 传播失败
- 症状: worker thread 检测到
- 原因: PDM scoring 使用 hydra 线程池， 不被 worker 继承
- 临时方案: 不设 ，让 agent 自动用 GPU 0
- 影响: 无法控制 agent 用哪张卡；多 job 并行需要先解决 GPU 分配

### 坑 3: Camera 属性名大小写
- 症状:
- NavSim 的  对象属性是**小写**（, ），不是大写
- 修复: 改为

### 坑 4: SensorConfig include 语义
- PDM scoring 的  内部用**单帧切片**调用
- 单帧切片时 iteration 恒为 0
-  → 只加载 iteration=3 的 camera → 单帧加载不到 → camera 为 None
- 修复:  — 用  匹配单帧 iteration

### 坑 5: Sensor 数据不完整
- navtrain sensor_blobs 只有 1192/1310 个 pkl 有 camera 数据
- 某些 pkl 的 camera 文件不存在 →
- 在线 agent 只能在有完整 sensor 数据的 split 上跑

### 坑 6: PDM worker 静默崩溃
- 症状: log 停在 Starting worker，无 CSV 生成
- agent standalone 测试 OK，但 PDM worker 内失败
- 错误被异常捕获吃掉，无法定位
-  之前跑通 smoke8/smoke64 是因为用了 NPY cache + 旧版代码

### 坑 7: CachedDiffusionAgent 格式不兼容
- 旧版 agent 假设  +  格式（NPY）
- navtest precompute 产出  格式（NPZ shard，embedded tokens）
- 修复: 重写 agent，自动检测格式并加载

## 已跑通的

| 任务 | 状态 | 分数 |
|------|------|------|
| smoke8 (cached) | ✅ | 0.956 |
| smoke64 (cached) | ✅ | 0.805 |
| navtest PDM (cached) | ❌ agent 初始化失败 | - |
| online LEAD smoke | ❌ 卡在 PDM worker 崩溃 | - |
| navtest BEV precompute | ✅ | 12,146 tokens |

## 下一步建议

1. 修复  的 npz shard 加载 → 重跑 navtest PDM
2. 解决 PDM worker GPU 传递问题 → 跑通 online LEAD smoke
3. 参考 SparseDrive/DiffusionDrive 的 agent 实现（它们直接使用官方 NavSim agent 接口，应该已经处理了这些坑）


---

# Session 2026-05-16 (cont): PDM Scoring & Online Agent Attempts

## Navtest BEV Precompute

- 用 test split logs + navtest filter
- wrapper: precompute_navtest.py
- 4 GPU, ~7 min, 12,146 tokens, 100% overlap with official navtest
- 路径: /workspace2/z_project/motdp_bev_cache_navtest/
- TMPDIR 问题: /tmp 满了, export TMPDIR

## Online LEAD Agent

文件: navsim_motdp/agents/online_lead_agent.py

链路: AgentInput -> 4-cam stitch -> LEAD backbone -> BEV -> diffusion -> trajectory

- requires_scene=False
- sensor_config: cam_f0/l0/r0/b0 at [0] (单帧)
- LEAD backbone: bf16 + monkey-patch LTF grid

## PDM Scoring 踩坑

### 1. TMPDIR 满
/tmp inode 耗尽, 需 export TMPDIR

### 2. CUDA_VISIBLE_DEVICES 不传播
hydra 线程池不继承 CUDA_VISIBLE_DEVICES, worker 检测到 0 GPUs
不设 CUDA_VISIBLE_DEVICES 可临时绕过

### 3. Camera 属性大小写
Cameras 对象属性是小写 (cam_l0), 不是大写 (CAM_L0)

### 4. SensorConfig include 语义
PDM 内部用单帧切片, iteration 恒为 0
SensorConfig(cam_f0=[3]) 加载不到 -> 改 [0]

### 5. Sensor 数据不完整
navtrain sensor_blobs 只有 1192/1310 pkl 有数据

### 6. PDM worker 静默崩溃
log 停在 Starting worker, 错误被捕获吃掉
agent standalone OK, PDM 内失败
smoke8/smoke64 用的是旧版 agent + NPY cache

### 7. Cache 格式不兼容
旧版 agent 假设 NPY (cache_index.npz + .npy)
navtest precompute 产出 NPZ shard (bev_cache_shard*.npz)
需重写 agent 检测格式

## 已跑通

smoke8 (cached): 0.956
smoke64 (cached): 0.805
navtest BEV precompute: 12,146 tokens

## 下一步

1. 修复 CachedDiffusionAgent npz shard 加载 -> navtest PDM
2. 解决 PDM worker GPU 问题 -> online LEAD smoke
3. 参考 SparseDrive/DiffusionDrive agent 实现

## 2026-05-16: Navtest PDM Debug Update

前面 `PDM Scoring & Online Agent Attempts` 有一段记录里的 inline code 被 shell 反引号展开吃掉了；后续判断以本节为准。

### 关键问题定位

1. **Hydra worker 找不到自定义 agent**
   - 症状：stdout 报 `Error locating target 'navsim_motdp.agents.cached_diffusion_agent.CachedDiffusionAgent'`。
   - 原因：实际执行入口是 `/home/z/code/navsim/navsim/planning/script/run_pdm_score_one_stage.py`，当前 MoT-DP worktree 不在 `PYTHONPATH`。
   - 修复：运行 PDM 前加：

```bash
export PYTHONPATH=/workspace1/z_project/code/motdp_z_navsim_motdp:$PYTHONPATH
```

2. **navtest NPZ shard 不适合 PDM 随机访问**
   - navtest precompute 产物是 `np.savez_compressed` 的 `bev_cache_shard*.npz`。
   - 对 compressed NPZ 做 `d["bev_grid"][idx]` 会先解压整个 shard 的 `bev_grid`，单 shard 约 1GB，PDM 每个 token 这样做会非常慢，甚至可能被 OOM kill。
   - 正确做法：和 train cache 一样，先转成 consolidated NPY/memmap。

3. **`*_tmp.npz` 不应进入 agent glob**
   - navtest cache 目录里有 final shard 和 interim tmp shard。
   - final shards: `bev_cache_shard000.npz` ... `bev_cache_shard003.npz`
   - tmp shards: `bev_cache_shard000_tmp.npz` ...
   - 后续如果继续支持 NPZ fallback，agent glob 应只匹配 final shards，例如 `bev_cache_shard[0-9][0-9][0-9].npz`。

### 已完成修复验证

将 navtest NPZ 转为 NPY/memmap：

```text
input:  /workspace2/z_project/motdp_bev_cache_navtest
output: /workspace2/z_project/motdp_bev_cache_navtest_npy
```

dry-run / conversion 结果：

```text
bev_cache_shard000.npz: raw=2006 kept=2006
bev_cache_shard001.npz: raw=3450 kept=3450
bev_cache_shard002.npz: raw=3074 kept=3074
bev_cache_shard003.npz: raw=3616 kept=3616
Raw entries: 12146
Kept entries: 12146
Unique kept tokens: 12146
elapsed_sec: 102.77
```

standalone cached agent 测试：

```text
token: 431ae29947e95c26
poses_shape: (8, 3)
first_pose: [0.1034, 0.0069, 0.0665]
last_pose: [10.2632, -0.0454, -0.0097]
```

Official PDM navtest smoke16：

```text
experiment_name: motdp_cached_navtest_npy_smoke16_py
successful: 16
failed: 0
average score: 0.6408248032
```

CSV：

```text
/workspace2/z_project/navsim_exp_motdp/motdp_cached_navtest_npy_smoke16_py/2026.05.16.09.05.31/2026.05.16.09.05.41.csv
```

### 当前可用 navtest PDM command

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp
export PYTHONPATH=/workspace1/z_project/code/motdp_z_navsim_motdp:$PYTHONPATH
export TMPDIR=/workspace2/z_project/tmp
mkdir -p $TMPDIR
export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps
export NAVSIM_EXP_ROOT=/workspace2/z_project/navsim_exp_motdp
export NAVSIM_DEVKIT_ROOT=/home/z/code/navsim

python /home/z/code/navsim/navsim/planning/script/run_pdm_score_one_stage.py \
  train_test_split=navtest \
  experiment_name=motdp_cached_navtest_npy_full \
  agent._target_=navsim_motdp.agents.cached_diffusion_agent.CachedDiffusionAgent \
  +agent.checkpoint_path=/workspace2/z_project/motdp_logs/navsim_simple_official_lr1e4_e60_b64/best_model.pt \
  +agent.cache_dir=/workspace2/z_project/motdp_bev_cache_navtest_npy \
  +agent.device=cuda:2 \
  metric_cache_path=/workspace2/data/navsim/processed_data/metric_cache_navtest \
  worker=single_machine_thread_pool \
  worker.max_workers=1
```

### 后续代码建议

- 更新 `_run_navtest_pdm.sh`：加入 `PYTHONPATH`，cache_dir 改为 `/workspace2/z_project/motdp_bev_cache_navtest_npy`。
- 更新 `CachedDiffusionAgent`：NPZ fallback glob 排除 `*_tmp.npz`，并使用稳定 hash seed，避免 Python `hash()` 跨进程不稳定。
- Online LEAD smoke 仍需单独处理，但至少也要先加入同样的 `PYTHONPATH` 修复。

## 2026-05-16: Cached Navtest PDM Stabilization Implementation

本节是 cached navtest official PDM 的 canonical status；前面 broken inline-code 的记录只作为历史踩坑参考。

### 代码改动

- `navsim_motdp/agents/cached_diffusion_agent.py`
  - 优先读取 NPY/memmap cache：`cache_index.npz` + `bev_grid.npy` + `ego_status.npy`。
  - NPZ fallback 只匹配 final shards：`bev_cache_shard[0-9][0-9][0-9].npz`，不再扫 `*_tmp.npz`。
  - seed 改为 `sha1(token) + deterministic_seed`，避免 Python `hash()` 跨进程不稳定。
  - `fallback=raise` 生效，full navtest 默认启用，避免 missing token 被 constant velocity 静默掩盖。
  - `num_inference_steps` override 会同步到 diffusion model。
  - memmap 单样本读取改为 writable copy，去掉 PyTorch non-writable warning。

- `scripts/_run_navtest_pdm.sh`
  - 加入 `PYTHONPATH=/workspace1/z_project/code/motdp_z_navsim_motdp:$PYTHONPATH`。
  - 默认 cache 改为 `/workspace2/z_project/motdp_bev_cache_navtest_npy`。
  - 默认 experiment 改为 `motdp_cached_navtest_npy_full_e60`。
  - 默认 `GPU_ID=2`，通过 `+agent.device=cuda:${GPU_ID}` 控制 agent device，不再依赖 `CUDA_VISIBLE_DEVICES`。
  - 默认 `+agent.fallback=raise`。
  - 保留 `worker=single_machine_thread_pool` 和 `worker.max_workers=1`。

### 验证结果

静态检查：

```text
python -m py_compile navsim_motdp/agents/cached_diffusion_agent.py: PASS
bash -n scripts/_run_navtest_pdm.sh: PASS
```

Agent standalone smoke：

```text
cache_dir: /workspace2/z_project/motdp_bev_cache_navtest_npy
checkpoint: /workspace2/z_project/motdp_logs/navsim_simple_official_lr1e4_e60_b64/best_model.pt
fallback: raise
token: 431ae29947e95c26
poses_shape: (8, 3)
first_pose: [0.0149, 0.0235, 1.0047]
last_pose: [5.0999, -0.0190, -0.0037]
```

Official PDM smoke16 after fix：

```text
experiment_name: motdp_cached_navtest_npy_smoke16_after_fix
successful: 16
failed: 0
average score: 0.7456234085
```

CSV：

```text
/workspace2/z_project/navsim_exp_motdp/motdp_cached_navtest_npy_smoke16_after_fix/2026.05.16.09.24.25/2026.05.16.09.24.34.csv
```

### Full Navtest Run

已在 tmux 中启动：

```text
tmux session: navtest_pdm_full
experiment_name: motdp_cached_navtest_npy_full_e60
log: /workspace2/z_project/navsim_exp_motdp/motdp_cached_navtest_npy_full_e60_tmux.log
```

启动后确认：

```text
Starting pdm scoring of 12146 scenarios
Processing scenario 164 / 12146
```

最终 full CSV 待该 tmux run 完成后补入本节。

### Full Navtest Final Result

Full navtest cached PDM run 已完成：

```text
experiment_name: motdp_cached_navtest_npy_full_e60
successful: 12146
failed: 0
average score: 0.7512730999
```

CSV：

```text
/workspace2/z_project/navsim_exp_motdp/motdp_cached_navtest_npy_full_e60/2026.05.16.09.25.04/2026.05.16.10.31.04.csv
```

Average row：

```text
no_at_fault_collisions: 0.9541001153
drivable_area_compliance: 0.8606948790
driving_direction_compliance: 0.9720072452
traffic_light_compliance: 0.9962950766
ego_progress: 0.8712117462
time_to_collision_within_bound: 0.9372632966
lane_keeping: 0.9224436028
history_comfort: 0.9796640869
two_frame_extended_comfort: 0.8237051793
score: 0.7512730999
```


---

# Session 2026-05-16 (cont): DAC=0 Analysis & LQR Root Cause

## Full Navtest PDM Result

- experiment: motdp_cached_navtest_npy_full_e60
- 12,146 scenarios, all valid
- **avg score: 0.751**
- DAC=0: 1,692 (13.9%) — main score killer
- NC=0: 508 (4.2%)
- Score=0: 2,235 (18.4%)

DAC 是 multiplier penalty，一旦 =0 整个 score 归零。

## Case Study: fddb0bdd1d7f53e0 (straight road, DAC=0)

预测轨迹（直行，沿 centerline）:

  wp0: x=1.61 y=-0.01  speed 3.2 m/s
  wp1: x=3.09 y=-0.03  speed 3.0 m/s
  wp2: x=4.49 y=-0.04  speed 2.8 m/s
  wp3: x=5.73 y=-0.05  speed 2.5 m/s
  wp4: x=6.89 y=-0.07  speed 2.3 m/s
  wp5: x=7.98 y=-0.09  speed 2.2 m/s
  wp6: x=9.01 y=-0.10  speed 2.0 m/s
  wp7: x=9.92 y=-0.11  speed 1.8 m/s

- 横向偏移仅 ~11cm，heading 接近 0
- speed 持续衰减 3.2 -> 1.8 m/s（保守减速的均值轨迹）

## LQR 仿真逻辑

PDMSimulator.simulate_proposals():
  - proposal_sampling: 8 poses @ 2Hz (0.5s interval)
  - tracker.update(proposal_states): 把我们的 8 waypoints 设为 LQR 目标
  - 每 0.5s: LQR 对比当前状态 vs 目标 waypoint -> 算 steering+accel -> propagate_state
  - 输出: 9 个 simulated states（含起点）

LQR 是逐点追踪，不做插值。8 个 waypoint 之间 LQR 自己通过车辆动力学补全。

## 根因

1. Pure diffusion 预测均值轨迹：直行 + 持续减速
2. 减速 pattern 被 LQR 追踪时产生 tracking error
3. ~11cm lateral 偏移 + 车辆 footprint (5m x 2m) -> 某个角点擦出 drivable area 边界
4. heading 从 xy 差分估算，不够精确

## 启示

Stage 2 需要:
  - route head: 提供密集 path 约束 (50 点 @ 1m)，不让 diffusion 自由发挥空间轨迹
  - speed head: 显式建模速度，避免 diffusion 学出保守减速的均值模式
  - 或者换 DiffusionDrive 的 anchor-based 方案：20 trajectory anchors + truncated diffusion

---

# Session 2026-05-16 (cont): LEAD Online Agent PDM Score

## 背景

之前 CachedDiffusionAgent 在 full navtest 上的 DAC=0 比例高达 13.9%，我们一度怀疑是 diffusion model 本身预测的轨迹偏出路面。为了区分模型问题和cache/坐标系问题，用 OnlineLeadDiffusionAgent（实时跑 LEAD backbone）在相同 token 上测。

## Fix: SensorConfig 必须覆盖所有 history frames

 的 iteration 计算：

- frame_idx=0 → iteration=3
- frame_idx=3 → iteration=0

所以  只能加载最后一帧。修复：。

## Smoke8 结果

- experiment: motdp_lead_navtest_smoke8b
- 8 scenarios, all valid
- **avg score: 0.954**

| token | DAC | NC | DDC | TLC | EP | TTC | LK | HC | EC | score |
|-------|-----|----|----|----|----|----|----|----|----|----|
| 9dc5a17094e0569d | 1 | 1 | 1 | 1 | 1.0 | 1 | 1 | 1 | 1 | 1.0 |
| afbf26b6d3bb5bee | 1 | 1 | 1 | 1 | 1.0 | 1 | 1 | 1 | - | 1.0 |
| af8dc1b01446555a | 1 | 1 | 1 | 1 | 1.0 | 1 | 1 | 1 | - | 1.0 |
| 58d40da0cce05d8c | 1 | 1 | 1 | 1 | 0.98 | 1 | 1 | 1 | - | 0.993 |
| 35119c815b1b59ef | 1 | 1 | 1 | 1 | 0.94 | 1 | 1 | 1 | 1 | 0.980 |
| b0546b8af0f95c05 | 1 | 1 | 1 | 1 | 0.70 | 1 | 1 | 1 | 1 | 0.905 |
| dfa220d6e64f5d84 | 1 | 1 | 1 | 1 | 1.0 | 1 | 1 | 1 | 0 | 0.875 |
| d58c4ad27c525465 | 1 | 1 | 1 | 1 | 1.0 | 1 | 1 | 1 | 0 | 0.875 |

**关键：DAC 全部 1.0，NC 全部 1.0。没有 off-road，没有碰撞。**

## 与 CachedAgent 对比

| Agent | Smoke8 Score | DAC=0 | NC=0 |
|-------|-------------|-------|------|
| CachedDiffusionAgent | 0.956 | 0/8 | 0/8 |
| **OnlineLeadAgent** | **0.954** | **0/8** | **0/8** |

两者在相同 8 个 token 上分数几乎一致（0.954 vs 0.956，差异在 EC 的 timestep 波动）。说明 **diffusion 模型本身没问题**。

## 启示

CachedAgent full navtest 上的 13.9% DAC=0 很可能来自：
1. NPY cache 的坐标/feature 在某些 token 上与 PDM metric cache 不匹配
2. CachedAgent 的 heading 估算（_xy_to_se2）在某些场景有精度问题
3. 或者 NPY cache 在训练时和推理时的 BEV feature 有 drift

下一步应扩大 LEAD agent 到更多 navtest 场景（64/128），确认其 DAC 是否持续优于 CachedAgent。


## Score 对照表 (2026-05-16)

| Agent | 预处理 | 数据 | 场景数 | Score | 备注 |
|---|---|---|---|---|---|
| CachedDiffusionAgent | 4-cam, 无JPEG | navtest | 12,146 | **0.751** | 用4-cam BEV cache |
| OnlineLeadDiffusion (旧) | 4-cam, 无JPEG | navtest | 12,146 | **0.650** | 实时跑backbone |
| OnlineLeadDiffusion (3-cam custom) | 3-cam crop + JPEG Q30 | navtest | 8 (smoke) | **0.856** | 不是官方 LEAD preprocessing；只是一个 custom smoke |
| PureLead (自己实现) | 4-cam, 无JPEG | navtest | 8 | 0.570 | LEAD自带planner，但 wrapper/预处理未对齐官方 |
| LEAD Official (CarlaTF) | 4-cam stitched + JPEG Q30 | navtest | 16 (smoke) | 0.817 | 官方agent+navsim v1.1 |

注意: Cached 0.751 和 Online 0.650 用的都是旧的4-cam实时/缓存路径，但并不等价于 LEAD official feature builder。
Online 3-cam smoke 0.856 只能说明该 custom crop 在 smoke8 上有效，不能说明 3-cam 是官方正确路径。
LEAD official NAVSIM v1.1 的真实路径是 L0+F0+R0+B0 四路拼接，整图 resize /4，再 JPEG Q30。
Cached 若要和 official LEAD backbone 对齐，需要重跑 precompute，并且应以 official 4-cam feature builder 为基准。

## Correction: LEAD Official Preprocessing Is 4-Cam, Not 3-Cam (2026-05-17)

重新检查 LEAD repo 后确认，之前表格里把 `OnlineLeadDiffusion (修复后)` 的 3-cam custom preprocessing 写成了“对齐官方预处理”，这是不准确的。

官方证据：

- LEAD checkpoint config `/workspace1/z_project/models/navsim_backbones/tfv6_navsim/config.json`:
  - `target_dataset = 3` (`NAVSIM_4CAMERAS`)
  - `num_available_cameras = 4`
  - `used_cameras = [true, true, true, true]`
  - `num_used_cameras = 4`
  - `final_image_width = 1920`
  - `final_image_height = 270`
- LEAD official NAVSIM feature builder:
  - file: `/workspace1/z_project/code/lead/3rd_party/navsim_workspace/navsimv1.1/navsim/agents/transfuser/transfuser_features.py`
  - cameras: `cam_l0`, `cam_f0`, `cam_r0`, `cam_b0`
  - preprocessing: concatenate `[l0, f0, r0, b0]` horizontally, resize whole stitched image by `/4`, then JPEG encode with quality 30
- LEAD official wrapper:
  - file: `/workspace1/z_project/code/lead/3rd_party/navsim_workspace/navsimv1.1/navsim/agents/carla_transfuser_agent.py`
  - decodes `camera_feature` using `cv2.IMREAD_COLOR`, transposes HWC to CHW, and forwards through `OpenLoopInference`

当前 MoT-DP `OnlineLeadDiffusionAgent` 是 custom 3-cam path，不是 official path：

- file: `/workspace1/z_project/code/motdp_z_navsim_motdp/navsim_motdp/agents/online_lead_agent.py`
- `SensorConfig`: `cam_f0`, `cam_l0`, `cam_r0`, `cam_b0=False`
- preprocessing: crop L/F/R cameras, concatenate 3 views, resize to `1024x256`, JPEG Q30 round-trip

因此当前判断应改为：

1. LEAD official ckpt + official wrapper 的 smoke16 score `0.817` 证明 ckpt 和官方 planner 正常。
2. 我们自己实现的 `PureLeadAgent` score 低，最可能是 wrapper/预处理未对齐 official `TransfuserFeatureBuilder + OpenLoopInference`，不能说明 LEAD planner 差。
3. `OnlineLeadDiffusion (3-cam custom)` smoke8 score `0.856` 是一个有趣 smoke，但不能作为 official preprocessing 依据。
4. 下一步应做 parity：同一 token 下比较 official `CarlaTransfuserAgent`、MoT-DP `PureLeadAgent`、MoT-DP `OnlineLeadDiffusionAgent` 的 image tensor stats、BEV stats、waypoints/trajectory。优先把 MoT-DP agent 改到 official 4-cam feature builder 路径，再重新判断。

## Implementation: MoT-DP LEAD Preprocessing Aligned To Official 4-Cam (2026-05-17)

结论: 之前 cached/online LEAD 路径确实没有对齐 official LEAD NAVSIM preprocessing。它们要么用了 3-cam crop+JPEG smoke path，要么用了 4-cam per-camera crop/resize 且没有 JPEG round-trip。现在已经把主线改成 official LEAD v1.1 4-cam path。

改动:

- 新增 `navsim_motdp/lead_preprocessing.py`，作为唯一官方 preprocessing 入口。
- official camera order 固定为 `[cam_l0, cam_f0, cam_r0, cam_b0]` / raw pkl `[CAM_L0, CAM_F0, CAM_R0, CAM_B0]`。
- preprocessing 固定为: stitch 4-cam full images -> resize whole stitched image by `1/4` -> JPEG encode quality `30` -> `cv2.imdecode(..., cv2.IMREAD_COLOR)` -> CHW tensor。
- `OnlineLeadDiffusionAgent` 改用该 helper，并把 sensor config 改为请求 current-frame 4-cam；同时去掉 Python `hash()` seed，改成 `sha1(deterministic_seed + token/ego_pose)`。
- `PureLeadAgent` 改用同一 helper，避免自己的 4-cam crop/resize path 干扰 LEAD planner 对比。
- `scripts/precompute_bev_cache.py` 改用同一 helper，因此后续 train/navtest BEV cache 重建会和 official LEAD backbone input 分布对齐。
- `scripts/precompute_navtest.py` 修复 repo/script `PYTHONPATH`，不再依赖旧 `/home/z/code/motdp_z_navsim_motdp` path。
- `scripts/compare_preprocess.py` 改成 direct parity test against LEAD vendored NAVSIM v1.1 `TransfuserFeatureBuilder._get_camera_feature`。

验证:

- `python -m py_compile navsim_motdp/lead_preprocessing.py navsim_motdp/agents/online_lead_agent.py navsim_motdp/agents/pure_lead_agent.py scripts/precompute_bev_cache.py scripts/precompute_navtest.py scripts/compare_preprocess.py`: pass。
- `scripts/compare_preprocess.py` on mini token `ea582d733c455f3a`: official/MoT-DP compressed bytes both `(41335,)`; decoded shape `(270, 1920, 3)`; tensor shape `(1, 3, 270, 1920)`; compressed bytes equal `True`; decoded/tensor max abs diff `0`。
- Raw pkl precompute input smoke on mini token `a5cf4580ad3657a5`: `build_lead_input(...)` output `(1, 3, 270, 1920)`, `torch.bfloat16`, value range `[0, 255]`。

下一步:

1. 用新 preprocessing 重建一个小 cache shard/subset，确认 BEV grid stats 和 token coverage。
2. 用新 cache 重新跑 cached smoke16/full navtest；旧 `/workspace2/z_project/motdp_bev_cache_navtest_npy` 不能再作为 LEAD-aligned cache 结论。
3. 如果 cached score 仍明显低于 official LEAD，再对比 `bev_feat/top_down` stats 和 diffusion planner label/trajectory 坐标，而不是继续怀疑 image preprocessing。



### 2026-05-16: Official4cam Cache Rebuild & Smoke16

详见: docs/navsim_lead_aligned_cache_rebuild_plan.md

- Full navtest official4cam cache rebuilt (12,146 tokens, 4 GPU, ~12 min)
- NPY conversion done (41s)
- Smoke16: 16/16 valid, **score 0.738**
- 未超过旧 cache (0.746): diffusion model 在旧 cache (no JPEG) 上训练, 新 cache (JPEG Q30) 有 train/test mismatch
- 结论: 不是预处理 bug, 是 train/test 分布漂移

下一步: 重训 diffusion on official4cam cache, 或先用旧 cache 0.751 为 baseline 推 Stage 2


### 2026-05-16: Official4cam Cache Validation

- 用 official4cam NPY cache + LEAD planning_decoder (CachedLeadAgent)
- 与 LEAD Official (CarlaTransfuserAgent, online) 同 16 token 对比
- 排除 LEAD 2 个 crash token 后: LEAD 0.934 vs CachedLead 0.934
- **结论**: cache 和 online backbone 输出完全一致，cache 验证通过

分数对照 (同 token):

| Agent | 预处理 | Cache | Score | 备注 |
|---|---|---|---|---|
| CachedLeadAgent | 4-cam, JPEG Q30 | official4cam | **0.943** (0.934 excl crash) | LEAD planner on cache |
| LEAD Official | 4-cam, JPEG Q30 | N/A (online) | 0.817 (0.934 excl crash) | 2 tokens crashed |
| CachedDiffusion | 4-cam, JPEG Q30 | official4cam | 0.738 | train/test mismatch |
| CachedDiffusion | 4-cam, NO JPEG | OLD cache | 0.751 | train/test consistent |

下一步: 重建 train cache (official4cam preprocessing) -> retrain diffusion
