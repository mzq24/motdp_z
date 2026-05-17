# NavSim Backbone Verification Record

验证日期: 2026-05-12

## 0. 统一 Conda 环境

- **名称**: `z_navsim_motdp`（在 newhpc）
- **Python**: 3.10.20
- **关键包**:
  - `torch==2.5.1+cu124` (CUDA 12.4, GPU OK)
  - `numpy==1.26.4`（兼容 sklearn 1.2.2）
  - `nuplan-devkit==1.2.0` (git @nuplan-devkit-v1.2)
  - `navsim` (editable, `/home/z/code/navsim`, --no-deps)
  - `pytorch-lightning==2.2.1`
  - `timm==1.0.27`, `diffusers==0.38.0`, `transformers==5.8.0`
  - `opencv-python==4.9.0.80`
- **LEAD**: editable install from `/home/z/code/lead`
- **已验证 import**: torch, navsim, nuplan, pytorch_lightning, timm, diffusers, transformers, cv2, einops, jaxtyping, beartype, TransfuserBackbone, TransfuserAgent, TransfuserModel, TransfuserConfig
- **数据集**: `/workspace2/data/navsim/` (2.4TB)
- **踩坑**:
  - `pkg_resources` missing → setuptools 降级到 69.5.1
  - numpy 2.x 与 sklearn 1.2.2 二进制不兼容 → numpy 1.26.4
  - NavSim 用 `--no-deps` 安装避免 torch/numpy 版本冲突

## 1. 下载的 Checkpoint

| 名称 | 来源 (HuggingFace) | 大小 | LiDAR | Camera | 格式 |
|------|-------------------|------|-------|--------|------|
| LTFv6 NavSim | `ln2697/tfv6_navsim` | 238MB | ❌ LTF latent | 4-cam 270×1920 | `.pth` |
| NavSim TransFuser | `autonomousvision/navsim_baselines/transfuser/` | 642MB | ✅ 真实 LiDAR | 3-cam 256×1024 | `.ckpt` (Lightning) |
| NavSim LTF | `autonomousvision/navsim_baselines/ltf/` | 643MB | ❌ latent | 3-cam 256×1024 | `.ckpt` (Lightning) |

存放路径: `newhpc:/workspace1/z_project/models/navsim_backbones/`

## 2. Smoke Test 结果

### 2.1 NavSim TransFuser（带真实 LiDAR）

**测试脚本**: `smoke_test_navsim_transfuser.py` on newhpc

```
Input:
  camera_feature:  [B, 3, 256, 1024]   ← CAM_L0 + CAM_F0 + CAM_R0 拼接
  lidar_feature:   [B, 1, 256, 256]    ← LiDAR 点云 BEV histogram
  status_feature:  [B, 8]              ← [command(4), velocity(2), acceleration(2)]

Backbone 输出:
  BEV top-down:    [B, 64, 64, 64]     ← ✅ 匹配 MoT-DP transfuser_bev_feature_upsample
  BEV global:      [B, 512, 8, 8]     ← 需适配至 MoT-DP 的 1512 通道

Full model 输出:
  trajectory:       [B, 8, 3]          ← (x, y, heading)
  agent_states:     [B, 30, 5]
  bev_semantic_map: [B, 7, 128, 256]
```

### 2.2 NavSim LTF（latent，无真实 LiDAR）

```
Input: 同上（LiDAR 被忽略，内部用可学习 latent 替代）
Backbone: 同 TransFuser 结构，但 LiDAR branch 输入被替换
输出格式相同
```

### 2.3 LEAD LTFv6（4-camera，无 LiDAR）

**测试脚本**: `smoke_test_lead_backbone.py` on newhpc

```
Input:
  rgb:             [B, 3, 270, 1920]   ← 4-cam (L0+F0+R0+B0) 拼接
  command:         [B, 4]
  speed:           [B]
  acceleration:    [B]

Backbone 输出:
  lidar_features:  [B, 512, 8, 8]
  image_features:  [B, 512, 9, 60]
  top_down BEV:    [B, 64, 64, 64]    ← ✅ 与 MoT-DP 空间维度对齐

Full model 输出:
  waypoints:       [B, 8, 2]          ← CARLA left-handed (需转 ISO 8855)
  headings:        [B, 8]
```

注意: LEAD LTFv6 使用 bfloat16 mixed precision，需要在 backbone.forward 中 fix LTF grid 的 dtype（monkey-patch 解决）。

## 3. 关键结论

### 3.1 BEV 空间网格对齐

**所有 backbone 的 `top_down` BEV 网格都是 `(B, 64, 64, 64)`**，和 MoT-DP 的 `transfuser_bev_feature_upsample` 完全匹配。MoT-DP 的 `GridSampleCrossBEVAttention` 可以直接采样。

### 3.2 BEV 特征通道需适配

| Backbone | BEV global channels | MoT-DP 期望 |
|----------|-------------------|-------------|
| NavSim TransFuser | 512 | 1512 |
| NavSim LTF | 512 | 1512 |
| LEAD LTFv6 | 512 (lidar_features) | 1512 |

需要在 Step 5 中添加 `nn.Linear(512, 1512)` adapter 或修改 MoT-DP 的 `bev_feature_proj` 输入维度。

### 3.3 坐标系

- NavSim/nuPlan 使用 ISO 8855（right-handed，y-left）
- LEAD LTFv6 输出 CARLA left-handed（y-right），需转换 `y *= -1`
- NavSim TransFuser 直接输出 ISO 8855 坐标

### 3.4 预测 horizon

所有 backbone 都预测 **8 waypoints, 2Hz (4s)**，MoT-DP 默认是 **6 waypoints, 5Hz**。Step 5 需要统一 horizon。

## 4. CKPT 加载方式

### NavSim TransFuser / LTF (Lightning .ckpt)
```python
ckpt = torch.load("transfuser_seed_0.ckpt")
state_dict = {k.removeprefix("agent."): v for k, v in ckpt["state_dict"].items()}
model = TransfuserModel(trajectory_sampling, config)
model.load_state_dict(state_dict, strict=False)
```

### LEAD LTFv6 (standalone .pth)
```python
model = load_tf("model_0060.pth", device)  # defined in ltfv6.py
```

## 5. 测试脚本位置

| 脚本 | 用途 |
|------|------|
| `newhpc:/home/z/code/smoke_test_lead_backbone.py` | LEAD LTFv6 合成数据测试 |
| `newhpc:/home/z/code/smoke_test_navsim_transfuser.py` | NavSim 真实数据测试 |

## 6. Label 迁移阶段决策（2026-05-13）

### 6.1 NAVSIM annotation 与 inference 可用性

- NAVSIM/OpenScene raw log 里有 HD map / route roadblock / traffic light / object track 等 privileged annotation：
  - `roadblock_ids`
  - `traffic_lights`
  - `anns.gt_boxes`
  - `anns.gt_names`
  - `anns.gt_velocity_3d`
  - `anns.instance_tokens`
  - `anns.track_tokens`
  - `ego2global` / future ego poses
- 这些 annotation 对训练 label / auxiliary supervision 有价值，但 leaderboard/test inference 输入不能直接依赖它们。
- 因此当前迁移阶段不把 HD map / metric cache 改造成 model inference 输入，避免大改现有 labeling 逻辑。

### 6.2 与 B2D PDM-Lite label 的差异

- NAVSIM 更像真实 nuPlan/OpenScene log，不像 B2D PDM-Lite 那样有明确的 CARLA scenario event name。
- 原 B2D 中 `ConstructionObstacleTwoWays / AccidentTwoWays / ParkedObstacleTwoWays` 这类强 scene prior 是 `borrow` 逻辑的重要来源。
- NAVSIM mini 视频里可以看到 lateral offset / lane-change-like motion，但这不等价于 MoT-DP 现有定义里的 two-way borrowed-lane corridor。
- 因此 `borrow` 不作为第一阶段主监督迁移目标。

### 6.3 当前 agreed label 迁移顺序

1. `path -> route`
   - 使用 SparseDriveV2 NAVSIM target builder 风格的 future ego path。
   - 该 path 是按距离约 1m 采样的未来 ego 轨迹，与 MoT-DP/LEAD route label 语义最接近。
2. `trajectory / speed / object labels`
   - trajectory 来自 NAVSIM future ego poses。
   - speed 可由 future trajectory 相邻点距离 / 0.5s 派生。
   - object label 以 NAVSIM `anns` 为主：box、class、velocity、instance/track token。
3. `junction-like`
   - 先基于 map intersection / lane connector / traffic light / cross traffic 做统计与视频验证。
   - 不急着完全复刻 B2D 的 junction label。
4. `generic agent-conflict / merge-like`
   - 先从 route/path corridor 与 other-agent motion 里构造候选。
   - 重点看 same-direction / converging conflict 是否能稳定出现。
   - 先做统计和视频确认，再决定是否进入训练主监督。
5. `borrow`
   - 第一阶段只做候选扫描和视频抽查。
   - 训练主监督里先 mask 掉 / valid=0。
   - 只有在 full trainval 里确认有足够稳定的 two-way borrow-like 样本后，再单独设计迁移逻辑。

### 6.4 暂不使用的 annotation 来源

- HD map / route roadblock：
  - 当前只用于 label 构建或验证，不作为 inference 输入。
  - 暂不为了 HD map 重写现有 labeling 主逻辑。
- official metric cache：
  - 主要服务 NAVSIM PDM / EPDMS 评测预处理。
  - 可作为后续 debug / score-aligned auxiliary 的参考，但不是当前 label 迁移主来源。

## 7. NAVSIM Label Migration Mini Prototype Plan（2026-05-13）

### 7.1 样本数口径

- `log` 是 NAVSIM raw log 的一个 `.pkl` 连续片段文件。
- `frame` 是 log 内的单帧。
- `sample/window` 是从连续帧里切出的 `history + future` 训练窗口。
- 后续迁移主线倾向使用 sliding window，而不是官方 non-overlap 切片。

| split / 口径 | logs | frames | samples / windows |
|---|---:|---:|---:|
| `mini` raw sliding `h4/f8` | 64 | 51,867 | 51,163 |
| `mini` raw sliding `h4/f8 + route` | 64 | 51,867 | 50,287 |
| `trainval` raw sliding `h4/f8` | 1,310 | 723,019 | 708,609 |
| `trainval` raw sliding `h4/f8 + route` | 1,310 | 723,019 | 665,218 |
| `test` raw sliding `h4/f8 + route` | 147 | 75,122 | 69,977 |
| `trainval` official `all_scenes` non-overlap `h4/f10` | 1,310 | 723,019 | 47,950 |
| official `navtrain` curated tokens | 1,192 listed logs | - | 103,288 |
| official `navmini` curated tokens | 62 listed logs | - | 396 |
| official `navtest` curated tokens | 136 listed logs | - | 12,146 |

说明：

- `raw sliding h4/f8` 是我们自己按 stride=1 切出的窗口：4 帧 history，8 帧 future。
- `+ route` 表示 current frame 有 `roadblock_ids`，与 NAVSIM `has_route=true` 过滤逻辑一致。
- official `navtrain` 的 103,288 不是 non-overlap；它是官方 YAML 里列出的 curated token sample 列表。
- `all_scenes non-overlap h4/f10` 是官方默认全场景缓存风格，仅作为参考口径，不作为后续主线。
- `trainval h4/f8 + route` 的 665,218 个窗口可视为我们后续 sliding label 构建的 full-scale 上限，但信息密度可能低于官方 curated split。

### 7.2 本轮目标

- 先做一个 mini 级别 label prototype，不接训练主链。
- 输入 raw NAVSIM logs，不依赖当前 partial training cache。
- 在 `mini` 上跑 sliding-window 统计，验证 label 可行性和分布。
- 本轮不处理 `borrow` 主监督；`borrow` 只保留后续候选扫描方向。

### 7.3 Prototype label 内容

1. `trajectory`
   - current frame 后 8 帧 ego future poses。
   - 坐标保持 NAVSIM / nuPlan y-left。
   - 输出形态：`(8, 3)`，即 `(x, y, heading)`。

2. `speed`
   - 由 current/future ego pose 相邻点距离除以 `0.5s` 派生。
   - 输出形态：`(8,)`。

3. `path -> route`
   - 从 current frame 往后最多 80 帧 ego poses 转到当前 ego 坐标。
   - 按 arc length 每 1m 采样得到 `path50` 和 mask。
   - `route20 = path50[:20, :2]` 作为 MoT-DP / LEAD route label 原型。

4. `object tracks`
   - 当前帧读取：
     - `anns.gt_boxes`
     - `anns.gt_names`
     - `anns.gt_velocity_3d`
     - `anns.instance_tokens`
     - `anns.track_tokens`
   - 固定统计 7 类：
     - `generic_object`
     - `vehicle`
     - `pedestrian`
     - `traffic_cone`
     - `barrier`
     - `bicycle`
     - `czone_sign`
   - 未来 8 帧按 `track_token` 匹配。
   - 统计每类实例数、ROI 内实例数、future track coverage：`>=1`, `>=4`, `==8`。

5. `junction-like`
   - 第一版仅做候选统计，不复刻 B2D junction label。
   - 候选条件：
     - `command in {left, right}`，或
     - 4s future yaw delta `abs(yaw_delta) >= 0.45rad`
   - 额外统计 `signalized_junction_like = junction_like && traffic_lights 非空`。

6. `merge-like`
   - 第一版仅做候选统计，不直接作为最终监督。
   - 只看 tracked `vehicle`。
   - actor 与 `route20` corridor 同向：heading diff `<=45deg`。
   - actor 从侧方接近 route corridor：
     - 当前 lateral distance `>2.0m`
     - 未来 8 帧内 min lateral distance `<=2.0m`
   - longitudinal overlap 范围 `[0, 40]m`。
   - future coverage 至少 4 帧。
   - `front_chase_like` 单独统计，不混入 merge-like。

### 7.4 实现计划

- 新增 tracked prototype 脚本：
  - `scripts/data_tools/navsim_label_stats.py`
- 默认参数：
  - `--history 4`
  - `--future 8`
  - `--stride 1`
  - `--require-route`
- 输出：
  - summary JSON：label 分布和关键统计。
  - examples JSONL：每类候选的 `log_name + frame_idx + token`，用于后续视频抽查。
- 推荐 newhpc 输出路径：
  - `/workspace1/z_project/outputs/navsim_label_stats/mini_h4f8_stats.json`
  - `/workspace1/z_project/outputs/navsim_label_stats/mini_h4f8_examples.jsonl`

### 7.5 验收标准

- 本地：
  - `python -m py_compile scripts/data_tools/navsim_label_stats.py`
- newhpc smoke：
  - `z_navsim_motdp` 环境下跑 `--max-logs 1 --max-windows 200`。
- newhpc mini full：
  - route-valid window 数应接近已统计的 `50,287`。
  - JSON 中包含 trajectory/path/object/junction-like/merge-like 分布。
  - examples manifest 每类至少保留若干样本，便于下一步生成视频验证。

### 7.6 当前默认假设

- 本轮不加载 camera，不生成 feature cache，只读 raw `.pkl` annotation。
- 本轮不接 HD map / metric cache，避免改变现有 labeling 主逻辑。
- 坐标不做 LEAD/CARLA y flip，全部保持 NAVSIM / nuPlan y-left。
- `borrow` 不进入主监督；等 full trainval 候选扫描确认存在稳定 two-way borrow-like 样本后再单独设计。
