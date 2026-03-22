# Bridge Baseline v2 — BridgeDrive 正确复现方案

## 背景与问题

`bridge_baseline/` v1 的实现存在结构偏差：
- BEV 特征：1512-ch 小 BEV `(1512, 8, 8)` 经 Conv1×1 压缩到 256-ch，信息损失大
- ego_status：简单 14-dim 向量（速度+命令+waypoints 拼接），没有独立的 velocity/command/target_point token
- TransformerDecoder：只有 3 层，1 个 query

v2 目标：用 BridgeDrive 的 PlanningContextEncoder 结构，将 velocity/command/target_point 作为独立 token。

---

## 关键决定：不用 lead ckpt，使用预计算 BEV

**原因**：lead 官方 ckpt（HuggingFace tfv6_resnet34 等）是用 `carla_leaderboard_mode=True` 训练的（3摄像头, lidar 384×320），而我们的数据集是单摄像头、lidar 256×256。两者 GPT pos_emb 大小不同（480 vs 448），无法直接加载。

**决定**：直接使用数据集里已有的预计算特征 `transfuser_bev_feature_upsample (64, 64, 64)`，不运行任何 backbone，不需要 ckpt。

此特征由 MoT-DP 自己的 TransFuser extractor 产生，与 lead top_down() 输出格式完全一致（64 channels BEV）。

---

## 实现架构

```
transfuser_bev_feature_upsample (B, 64, 64, 64)  ← 直接来自 CARLAImageDataset
        │
        ├─ bev_proj  Conv1×1(64→256)  → (B, 256, 64, 64)   [供 GridSample BEV attention]
        │
        └─ PlanningContextEncoder:
             BEV tokens:    Conv1×1(64→256) + sine PE → flatten → (B, 4096, 256)
             velocity token: Linear(1→256),  speed / 25.0
             command token:  Linear(6→256),  one-hot
             tp token:       Linear(2→256),  target_point / [200, 50]
             tp_next token:  Linear(2→256),  (共享 tp_encoder)
             status_pos_embedding (learnable) + concat
             → context_tokens: (B, 4100, 256)
        │
        6-layer TransformerDecoder  (nhead=8, d_ffn=1024)
             1 个 learnable ego_query: (B, 1, 256)
             → ego_query: (B, 1, 256)
        │
        TrajectoryHead  [不变，直接复用 v1]
             20-mode DDBM Brownian Bridge diffusion
             → trajectory: (B, 10, 2)
```

---

## 文件变动

### 新建
| 文件 | 说明 |
|------|------|
| `bridge_baseline/planning_encoder.py` | PlanningContextEncoder + PositionEmbeddingSine |
| `bridge_baseline/model_v2.py` | BDModelV2 顶层模型 |

### 修改
| 文件 | 修改内容 |
|------|----------|
| `bridge_baseline/policy.py` | 新增 BDBaselinePolicyV2（v1 保留） |
| `bridge_baseline/config.py` | 新增 max_speed, tp_norm 字段 |
| `bridge_baseline/bd_config.yaml` | policy_version: v2；max_speed；tp_norm |
| `bridge_baseline/train.py` | 按 policy_version 选 v1/v2 |

### 不变（直接复用）
| 文件 | 原因 |
|------|------|
| `bridge_baseline/trajectory_head.py` | DDBM head 已正确验证 ✓ |
| `bridge_baseline/modules/` | blocks, loss, refinement, ddbm_scheduler 均验证 ✓ |
| `dataset/unified_carla_dataset.py` | 已提供所有需要的 batch keys |

---

## Batch keys（来自 CARLAImageDataset）

| Key | Shape | 说明 |
|-----|-------|------|
| `transfuser_bev_feature_upsample` | (B, 64, 64, 64) float16 | 主要 BEV 输入 |
| `speed` | (B, 4) | speed_hist，取最后一步 |
| `command_hist` | (B, 4, 6) | one-hot command |
| `target_point_hist` | (B, 4, 2) | 导航目标点 |
| `target_point_next_hist` | (B, 4, 2) | 下一个目标点 |
| `route` | (B, 20, 2) | GT 轨迹，head 取 [:, :10] |

---

## Lead/BridgeDrive TransFuser backbone 调研结论

（以防之后想切换到 lead backbone）

| 配置 | MoT-DP 数据 | lead leaderboard ckpt |
|------|-------------|----------------------|
| 摄像头 | 单摄像头 1024×512 | 3摄像头 480×270 |
| Lidar 范围 | [-32,32]×[-32,32] | [-32,64]×[-40,40] |
| Lidar 分辨率 | 256×256 | 384×320 |
| img_vert_anchors | 12 | 8 |
| img_horz_anchors | 32 | 45 |
| GPT pos_emb 大小 | 12×32+8×8=448 | 8×45+10×12=480 |
| top_down() 输出 | (64, 64, 64) | (64, 80, 96) |

**结论**：架构代码相同，但 pos_emb 大小不同 → ckpt 不兼容。若要使用 lead ckpt，需要从 HuggingFace 下载新的 carla_leaderboard2 数据集并重新处理（3摄像头格式）。

Lead ckpt 路径：`/media/z/data/models/garage2/tf6/`（tfv6_resnet34, noradar_resnet34 等）

---

## 归一化统计量（已修正，v1 也适用）

```yaml
norm_x_mean: [2.491, 3.481, 4.464, 5.441, 6.415, 7.388, 8.359, 9.327, 10.290, 11.248]
norm_x_std:  [0.089, 0.113, 0.144, 0.193, 0.253, 0.315, 0.381, 0.456, 0.545, 0.646]
norm_y_mean: [-0.013, -0.024, -0.047, -0.075, -0.103, -0.127, -0.146, -0.162, -0.178, -0.194]
norm_y_std:  [0.194, 0.299, 0.434, 0.602, 0.788, 0.979, 1.168, 1.362, 1.567, 1.786]
```

来源：BridgeDrive 官方 model_diffusion_head_ddbm.py（pdm_lite CARLA 训练集统计）。

---

## Speed 预测（两种方式）

闭环评测时横纵向控制分离，需要速度输出。已实现两种方式，对应 BridgeDrive 原版：

### Method 1：`predict_target_speed`（默认开启）

- 独立 speed query token → 6-layer TransformerDecoder → MLP → 8-class two-hot 分布
- 8 个速度 bin：`[0.0, 4.0, 8.0, 10.0, 13.89, 16.0, 17.78, 20.0]` m/s
- GT = `speed[:, -1]`（当前帧专家速度），two-hot 插值编码，CE loss
- Inference 输出 `target_speed` (B,) m/s（加权平均解码）

### Method 2：`diffusion_speed`（默认关闭）

- speed 作为第 11 个 "waypoint" `(speed, 0)` 拼入 DDBM route，联合扩散
- Init 时自动扩展 10-pt anchor → 11-pt anchor（`*_speed11.npy`）
- 切换方式：`predict_target_speed: false` + `diffusion_speed: true`

---

## Route-to-Traj 工具（备用）

**文件**：`bridge_baseline/route_utils.py`，函数 `route_speed_to_traj(route, target_speed)`

**用途**：将模型输出的 route（空间路径）+ target_speed → 时序 traj（6步 × 0.5s = 3s），供 open-loop 评估与 GT agent_pos 对比。

**原理**：
- Pure pursuit 几何法求转向角（不通过 PID，直接用几何关系）：`steer = arctan(2L·sin(α)/d_aim)`
- aim 距离来自 BridgeDrive agent：`clip(0.975*v + 1.915, 2.4, 10.5)` m
- 自行车运动学模型积分：`x += v·cos(yaw)·dt`，`yaw += v·tan(steer)/L·dt`
- 速度假设恒定（等于 target_speed），不模拟加速度

**为什么暂时不用**：
- Open-loop 难以区分「自行车模型误差」和「模型本身预测误差」
- 建议等闭环评测能跑通后再引入，作为辅助分析工具

**接口**：
```python
from bridge_baseline.route_utils import route_speed_to_traj
traj = route_speed_to_traj(route_np, target_speed_scalar)  # → (6, 2) np.float32
```

---

## TrajectoryHead 内部架构解析

### 数据流总结

```
traj_feature  (M 个 mode 的轨迹坐标 → 正弦 pos embed → MLP)
    │
    ├─ Branch 1: GridSampleCrossBEVAttention @ noisy_traj_points
    │     "在当前噪声轨迹的空间位置采样 BEV" → traj_f
    │
    └─ Branch 2: GridSampleCrossBEVAttention @ plan_anchor，queries = traj_f
          "用噪声轨迹的语义查询 anchor 位置的 BEV 场景" → traj_f_T

concat([traj_f, traj_f_T])  (bs, M, d_model*2)
    │
    FiLM(time_embed)  ← timestep 注入（缩放+偏移）
    │
    DiffMotionPlanningRefinementModule
        ├─ plan_cls:   (bs, M)      ← 哪个 mode 最合适的 logit
        └─ plan_reg:   (bs, M, T, 2) ← 轨迹偏移量（残差）
    │
poses_reg = plan_reg + noisy_traj_points  ← 残差加回噪声轨迹，得到预测的去噪轨迹
```

### 关键设计问题 Q&A

**Q：为什么不直接把 context_tokens 传给 TrajectoryHead，而是先用 ego_query 做一次 attention？**

TrajectoryHead 内部只有一个 `ego_query` slot，无法直接消费 4100 个 token。6 层 TransformerDecoder 的作用是把 BEV token + 导航信号（velocity / command / target_point）压缩蒸馏成一个紧凑的规划意图向量 `ego_query`。TrajectoryHead 内部的 GridSample BEV attention 另外负责轨迹-空间感知，两者职责分离：
- `ego_query`：规划意图（"我要往哪走"）
- `bev_feature`：原始空间感知（"周围是什么"）

**Q：traj_feature 是什么？**

`traj_feature = _encode_traj(noisy_abs)` — 对**轨迹坐标**做正弦位置编码再经 MLP 压缩，shape `(bs, M, d_model)`。它编码的是轨迹的几何形状，**不是**从 BEV 采样出的特征。

**Q：Branch 2 为什么用 Branch 1 的输出 traj_f 而不是原始 traj_feature 作为 queries？**

刻意设计：Branch 2 的 queries（traj_f）已包含噪声轨迹位置的 BEV 上下文，再去 anchor 位置采样，相当于在问："我知道当前噪声轨迹在哪，anchor 处的场景又是什么样？" concat 后网络同时看到两个位置的场景信息，更好地预测去噪方向。

**Q：Branch 2 的 queries 语义来自 noisy traj，采样位置却是 anchor——位置不对吗？**

不是问题，是标准用法。`GridSampleCrossBEVAttention` 中：
- `queries` 决定"**怎么看**"（attention 权重由 queries 预测）
- `traj_points` 决定"**在哪看**"（grid_sample 的采样位置）
- 两者本来就可以不对应；`return out + queries` 的残差连接保留了 queries 的原始信息

Branch 2 = 用 noisy traj 的语义去关注 anchor 的空间位置，是对比两个端点的场景差异。

**Q：FiLM（ModulationLayer）做了什么？**

```python
scale, shift = MLP(time_embed).chunk(2)
out = feature * (1 + scale) + shift
```

对 concat 特征做**通道级缩放+偏移**，条件是扩散时间步 t。高 t（噪声大）→ 偏向大幅修正；低 t（接近干净）→ 偏向精细调整。标准扩散模型的时间步注入机制（类似 U-Net 里的 AdaGN）。

**Q：DiffMotionPlanningRefinementModule 做了什么？**

预测头，同时输出：
- `plan_cls (bs, M)`：哪个 mode 最优的 logit（focal loss 监督）
- `plan_reg (bs, M, T, 2)`：从当前噪声轨迹的**偏移量**（不是绝对坐标），残差加回 noisy_traj_points 得到预测的去噪轨迹

---

## 验证状态

- [x] 模型 import 无误
- [x] Forward pass (train)：loss ~9（含 speed_loss ≈ 2.27，符合 ln(8)≈2.08 预期）
- [x] Inference：`action` shape (B, 10, 2)，`target_speed` shape (B,) m/s
- [x] 参数量：~18.7M（+speed head ≈ 0.1M）
- [ ] 实际训练：待运行，期望 cls_loss < 2.0 in first 100 steps

## 已知问题

1. **target_point 坐标系**：`target_point_hist` 可能是世界坐标而非 ego-relative，需要调查是否影响导航 token 质量
2. **BEV token 数量大**：4096 tokens，训练速度可能慢。可在 model_v2 中加 avg_pool 降到 1024（32×32）
3. **数据集 norm stats**：当前使用 BridgeDrive 官方统计，建议用 `scripts/compute_route_stats.py` 在自己数据集上重新计算
4. **diffusion_speed anchor speed**：当前所有 20 个 mode 共用 8.0 m/s 作为 speed anchor，理想情况应按 cluster 计算各自的平均速度
