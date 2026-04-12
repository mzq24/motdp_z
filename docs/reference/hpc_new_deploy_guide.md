# 新 HPC 部署指南 (workspace1)

> 与旧 HPC（NTU）的主要区别：
> - **不需要 PBS 调度**，直接 bash 运行
> - **1TB RAM**，IO 不是瓶颈（旧 HPC 用 Lustre 网络存储，IO 抖动严重）
> - **NVMe SSD 本地存储**，不需要 tmpfs 加速技巧
> - Conda 环境名：`z_dpauto`（旧 HPC 是 `dpautomotive`）

## 路径规划

| 项目 | 路径 |
|------|------|
| Raw dataset | `/workspace1/z_project/dataset/pdm_lite` |
| Code | `/workspace1/z_project/code/motdp_z` |
| Code alias | `/home/z/code/motdp_z`（link 到上面） |
| TransFuser 权重 | `/workspace1/z_project/models/.../pretrain/` |
| Processed data | `/workspace1/z_project/dataset/pdm_lite/tmp_data/{train,val}/` |
| Memmap cache | `pdm_lite/tmp_data/{train,val}/tmp_data/` 下的 `.bin` |
| Checkpoints | `code/motdp_z/checkpoints/<version>/` |
| Conda env | `z_dpauto` |

## 完整部署流程

### 前置条件

- Raw dataset 已下载（pdm_lite 下各 event 目录）
- Conda env `z_dpauto` 已创建
- **需要额外安装 `laszip`**：LiDAR `.laz` 文件解压依赖

```bash
conda install -c conda-forge laszip
pip install laszip  # Python binding
# 验证
python -c "import laszip; print('laszip OK')"
```

### Step 1: TransFuser BEV 特征提取（GPU）

```bash
bash scripts/hpc_new/deploy_pipeline.sh transfuser
```

- **Phase 1 (pack_source)**：读 `.laz` + `.jpg`，打包成 `route_source.pt`
  - 瓶颈是 **laszip CPU 解压**，不是磁盘 IO（`iostat` 确认 `%iowait ≈ 0`）
  - 每个 scene 前半慢（10-20 FPS，冷缓存），后半快（几百 FPS，OS page cache 命中）
  - 整体约 **2-3 小时**
- **Phase 2 (extract)**：GPU batch forward（batch_size=128），约 **1-2 小时**
- 输出：每个 route 下 `transfuser_feature/route_features.pt`
- 支持 `--skip_existing`（默认），中断重跑会跳过已完成的 route

### Step 2: Preprocess 生成 sample-level pkl

```bash
bash scripts/hpc_new/deploy_pipeline.sh preprocess
```

- 读取 raw data + `transfuser_feature/` 路径，生成 per-sample `.pkl`
- 自动 split train/val
- 较快（不涉及 `.laz` 解压）

### Step 3: 构建 memmap cache（.bin）

```bash
bash scripts/hpc_new/deploy_pipeline.sh build_cache
```

- 将所有 `route_features.pt` 聚合为 flat binary + 索引
- 输出：`bev_features_fp16.bin`（~130GB）+ `bev_upsamples_fp16.bin`（~70GB）+ `feature_index.pkl`
- 采用 memmap 方式加载，1TB RAM 下 OS page cache 基本全命中，效果等同全部放 RAM

### Step 4: 生成 anchor 文件

```bash
bash scripts/hpc_new/deploy_pipeline.sh anchors
```

生成 3 种 anchor：
- bridge baseline: route anchor (20 modes × 10 waypoints)
- dd baseline: traj anchor (20 modes × 6 waypoints)
- dd baseline / Route B: traj anchor (32 modes × 6 waypoints)

### Step 5: 计算 norm 统计量

```bash
bash scripts/hpc_new/deploy_pipeline.sh stats
```

- bridge baseline: per-waypoint route + traj 归一化统计（写入 `bd_config.yaml`）
- main config: action stats（写入 `pdm_local.yaml`）
- **注意**：跑完后需手动复制 norm 值到 `bd_config_hpc_new.yaml` / `pdm_hpc_new.yaml`

### Step 6: Dry-run 验证

```bash
bash scripts/hpc_new/deploy_pipeline.sh dryrun
```

检查所有路径、cache、anchor、GPU 是否就绪。

## 串联执行（睡前挂后台）

如果某一步正在跑，想等它完成后自动跑后续步骤：

```bash
# 找到正在跑的 python PID
pgrep -u $USER -a python | grep transfuser

# 挂后台等待 + 自动续跑
mkdir -p logs
nohup bash scripts/hpc_new/wait_and_continue.sh <PID> > logs/auto_pipeline.log 2>&1 &
```

脚本 `wait_and_continue.sh` 会等 PID 退出后依次跑 preprocess → build_cache → anchors → stats。

## 关于 memmap vs 全部放 RAM

新 HPC 有 1TB RAM。memmap 在这种条件下效果等同于全部放 RAM：
- 首次访问触发 page fault，从 SSD 读入 page cache
- 之后全部从 page cache 读，速度等于 RAM
- 不需要改代码，OS 会自动管理

只有当多个用户同时占用大量内存、page cache 被挤占时，memmap 才会退化。独占使用时无需担心。

## 与旧 HPC 的关键差异

| 对比项 | 旧 HPC (NTU) | 新 HPC (workspace1) |
|--------|-------------|-------------------|
| 存储 | Lustre 网络存储，IO 抖动 | NVMe SSD，稳定 |
| RAM | ~256GB，page cache 不够 | 1TB，page cache 充裕 |
| 调度 | PBS 脚本 (`qsub`) | 直接 bash 运行 |
| tmpfs 加速 | 需要拷贝 .bin 到 /tmp | 不需要，SSD + page cache 够用 |
| Conda env | `dpautomotive` | `z_dpauto` |
| .laz 依赖 | 已装 | 需手动 `conda install laszip` + `pip install laszip` |
