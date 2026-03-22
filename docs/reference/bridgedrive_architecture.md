# BridgeDrive 网络结构与训练流程

> 根目录：`BridgeDrive/BridgeDrive_adaptation_LEAD/lead/`（下文路径均相对此目录）

---

## 1. 整体网络结构

**主模型类 `TFv6`**：[tfv6/tfv6_bridgedrive.py:25](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/tfv6_bridgedrive.py#L25)

```
输入：RGB (多相机) + LiDAR (光栅化) + Radar + 导航信息
          ↓
    TransfuserBackbone                          tfv6_bridgedrive.py:37 (__init__), :126 (forward)
          ├─→ bev_features  (B, num_lidar_features, H, W)
          └─→ image_features (多尺度相机特征)
          ↓
    bev_feature_grid = backbone.top_down()      tfv6_bridgedrive.py:164

    ┌──── 并行 Decoder 头 ─────────────────────────────────────────────────────────┐
    │  [RadarDetector]        → radar_features, radar_pred   :95 (__init__), :131  │
    │  [PlanningDecoderDDBM]  → route, waypoints, speed      :109 (__init__), :147 │  ← 核心规划
    │  [PerspectiveDecoder]   → pred_semantic                :40 (__init__), :157  │  ← 辅助监督
    │  [PerspectiveDecoder]   → pred_depth                   :51 (__init__), :161  │  ← 辅助监督
    │  [BEVDecoder]           → pred_bev_semantic            :63 (__init__), :178  │  ← 辅助监督
    │  [CenterNetDecoder]     → pred_bounding_box            :80 (__init__), :167  │  ← 辅助监督 + stop sign
    └─────────────────────────────────────────────────────────────────────────────┘
```

**输出 `Prediction` dataclass**：[tfv6_bridgedrive.py:270-301](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/tfv6_bridgedrive.py#L270-L301)

**注意**：除 Planning + BoundingBox 以外的 decoder 均为 training-time 辅助监督，closed-loop 推理不使用其输出。

---

## 2. PlanningDecoderDDBM 内部结构

**类定义**：[tfv6/planning_decoder_bridgedrive.py](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py)

### 2.1 Context Encoder（条件信息编码）

**`PlanningContextEncoder` 类**：[planning_decoder_bridgedrive.py:472](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L472)

将所有条件信息编码为统一的 token 序列：

| Token | 来源 | 编码方式 | 代码位置 |
|---|---|---|---|
| BEV context tokens | bev_features → Conv2d → flatten | (B, H×W, token_dim) | [L549](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L549) `dimension_adapter` |
| velocity token | 当前车速标量 | MLP(1 → token_dim) | [L485](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L485) `velocity_encoder` |
| acceleration token | 加速度标量 | MLP(1 → token_dim) | [L492](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L492) `acceleration_encoder` |
| command token | 导航命令（6维 one-hot） | MLP(6 → token_dim) | [L499](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L499) `command_encoder` |
| target_point token | 当前/前/后航点 (×3) | MLP(2 → token_dim) | [L508](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L508) `tp_encoder` |
| radar tokens (可选) | RadarDetector 输出 | Linear + 正弦位置编码(x,y) → 20 tokens | [L535](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L535) `radar_encoder` |

**forward 中 token 拼接流程**：
- `status_tokens = []` 初始化：[L594](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L594)
- `status_tokens.append(velocity_token)`：[L603](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L603)
- `status_tokens.append(radar_token)`：[L680](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L680)
- `status_pos_embedding` 可学习位置编码：[L545](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L545)
- `cat([context_tokens, status_tokens])`：[L710-712](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L710-L712)

```
all_tokens = cat([bev_context_tokens, status_tokens], dim=1)
# shape: (B, H*W + num_status_tokens, 64)
```

### 2.2 Transformer Decoder

**实例化**：[planning_decoder_bridgedrive.py:62](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L62)

- **类型**：标准 PyTorch `TransformerDecoder`
- **层数**：6（`transfuser_num_bev_cross_attention_layers`，[config_training.py:676](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py#L676)）
- **heads**：8（[config_training.py:678](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py#L678)）
- **token dim**：64（[config_training.py:680](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py#L680)）
- **激活函数**：GELU
- **可学习 query**：[planning_decoder_bridgedrive.py:54](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L54)，shape `(1, num_queries, 64)`
- **输出**：ego_query `(B, num_queries, 64)`

query 的组成（num_queries = 10+8+1 = 19）：
- 10 个 route checkpoint 的 query
- 8 个 future waypoint 的 query
- 1 个 target speed 的 query

**输出 decoder 头**：
- `route_decoder` Linear：[L84](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L84)
- `wp_decoder` Linear：[L86](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L86)
- `target_speed_decoder` MLP：[L90](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L90)

### 2.3 TrajectoryHead（DDBM Diffusion Head）

**实例化**：[planning_decoder_bridgedrive.py:74](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L74)
**类定义**：[tfv6/diffusion_modules/model_diffusion_head_ddbm.py:125](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py#L125)

接收 Transformer Decoder 输出的 ego_query，进行 anchor-based 扩散推理。

**Plan Anchor**：预先用 K-Means 聚类得到 60 个轨迹 mode，shape `(60, 10, 2)`，作为扩散起点 x_T。加载位置：[model_diffusion_head_ddbm.py:178](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py#L178)

**内部结构**：
- Plan anchor encoder：正弦位置编码 → MLP（[L178-183](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py#L178-L183)）
- Time MLP：SinusoidalPosEmb(t) → Linear → SiLU → Linear（[L191](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py#L191)）
- 2 层自定义 Transformer decoder：
  - **Regression branch**：轨迹回归（BEV cross-attention + ego cross-attention）
  - **Classification branch**：mode 分类（选 60 个 anchor 中的最优）

**`forward()` 入口**：[planning_decoder_bridgedrive.py:128](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L128)
**`compute_loss()`**：[planning_decoder_bridgedrive.py:227](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L227)

---

## 3. DDBM 扩散流程

**`DDBMScheduler` 类**：[model_diffusion_head_ddbm.py:44](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py#L44)

### 3.1 噪声调度（VP Schedule）

使用 Variance-Preserving (VP) 噪声调度，核心公式：

```
vp_logs(t) = -0.25 * t² * beta_d - 0.5 * t * beta_min    # L52
           (beta_d=2.0, beta_min=0.1)

x_t = a_t * x_T + b_t * x_0 + c_t * noise                # L75 add_noise()
```

其中 x_T = plan_anchor（60 个聚类轨迹之一），x_0 = GT 轨迹。

关键函数：
- `vp_logs(t)`：[L52](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py#L52)
- `get_abc(t)`：[L62](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py#L62)
- `add_noise()`：[L75](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py#L75)
- `sample_step()`（reverse step）：[L95](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py#L95)

### 3.2 Training Forward

**`forward_train()`**：[model_diffusion_head_ddbm.py:250](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py#L250)

```
1. 加载 60 个 plan_anchor，展开到 batch: (B, 60, 10, 2)       # L178
2. GT 轨迹归一化（预计算 mean/std per pose index）
3. 随机采样 timestep t ~ U(1, 1000)                           # L267
4. 加噪：x_t = a_t * x_T + b_t * x_0 + c_t * noise           # L75 add_noise()
5. 将 noisy trajectory + anchor 送入 2 层 decoder
6. 输出 poses_reg_list（回归）+ poses_cls_list（分类）
7. 通过 argmax(final cls scores) 选出 best mode
```

### 3.3 Inference（DDIM 采样，20 步）

**`forward_test()`**：[model_diffusion_head_ddbm.py:318](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py#L318)
**DDIM 采样循环**：[model_diffusion_head_ddbm.py:350](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py#L350)
**推理步数配置**：[config_training.py:265](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py#L265) `step_num=20`

```
1. 初始化：x_t = plan_anchor（t=1000）
2. for t in linspace(1000 → 0, 20 steps):          # L350
   a. 模型预测 x_0（denoised trajectory）
   b. DDBM sample_step() 计算 x_{t-1}              # L95
   c. x_t ← x_{t-1}
3. 最终 x_0 = 精炼后的轨迹
4. 分类网络选出最优 mode → 输出 route checkpoints
```

---

## 4. 损失函数

**Loss 计算入口**：[multimodal_loss_ddbm.py:114](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/multimodal_loss_ddbm.py#L114)

| 损失项 | 类型 | 权重 | 代码位置 |
|---|---|---|---|
| `loss_ddbm_reg` | L1 | 1.0 | [multimodal_loss_ddbm.py:149](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/multimodal_loss_ddbm.py#L149) |
| `loss_ddbm_cls` | Focal Loss (γ=2, α=0.25) | 1.0 | [multimodal_loss_ddbm.py:138](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/multimodal_loss_ddbm.py#L138) |
| `loss_spatio_temporal_waypoints` | L1 | 1.0 | [planning_decoder_bridgedrive.py:227](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L227) |
| `loss_target_speed` | Cross-Entropy (Two-Hot) | 1.0 | [planning_decoder_bridgedrive.py:227](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/planning_decoder_bridgedrive.py#L227) |
| `loss_bev_semantic` | Cross-Entropy | 1.0 | [tfv6_bridgedrive.py:224](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/tfv6_bridgedrive.py#L224) |
| `loss_semantic` | Cross-Entropy | 1.0 | [tfv6_bridgedrive.py:212](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/tfv6_bridgedrive.py#L212) |
| `loss_depth` | L1 | 0.00001 | [tfv6_bridgedrive.py:218](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/tfv6_bridgedrive.py#L218) |
| `radar_loss` (cls/reg) | BCE + L1 | 1.0 / 5.0 | [tfv6_bridgedrive.py:252](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/tfv6_bridgedrive.py#L252) |
| CenterNet losses | Focal + L1 | 各 1.0 | [tfv6_bridgedrive.py:235](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/tfv6_bridgedrive.py#L235) |
| loss weight 归一化 | — | — | [config_training.py:953](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py#L953) |

---

## 5. 训练配置

**训练入口**：[training/train_bridgedrive.py](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/train_bridgedrive.py)
**配置类**：[training/config_training.py](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py)

| 参数 | 值 | 代码位置 |
|---|---|---|
| Optimizer | AdamW, amsgrad=True | [train_bridgedrive.py:64](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/train_bridgedrive.py#L64) |
| Learning Rate | 3e-4 | [config_training.py:346](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py#L346) |
| Weight Decay | 0.01 | [config_training.py:346](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py#L346) |
| LR Schedule | CosineAnnealingWarmRestarts | [train_bridgedrive.py:64](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/train_bridgedrive.py#L64) |
| Batch Size | 64 per GPU | [config_training.py:289](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py#L289) |
| Mixed Precision | bfloat16 (L40s) / float32 | [train_bridgedrive.py:158](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/train_bridgedrive.py#L158) |
| Gradient Scaler | 动态，init=1024 | [train_bridgedrive.py:175](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/train_bridgedrive.py#L175) |
| Epochs | 31 (CARLA Leaderboard) | [config_training.py:273](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py#L273) |
| BEV 分辨率 | 256×256（覆盖 ±32m） | — |
| Token Dim | 64 | [config_training.py:680](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py#L680) |
| Cross-Attention Layers | 6 | [config_training.py:676](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py#L676) |
| Attention Heads | 8 | [config_training.py:678](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py#L678) |
| Diffusion Steps（训练） | t ~ U(1, 1000) | [model_diffusion_head_ddbm.py:267](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py#L267) |
| Diffusion Steps（推理） | 20 步 DDIM | [config_training.py:265](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py#L265) |
| Plan Anchor Modes | 60（K-Means 聚类） | — |
| Route Checkpoints | 10 个 | — |
| Future Waypoints | 8 个 | — |
| Radar Queries | 20 | [config_training.py:385](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/config_training.py#L385) |
| Speed Classes | 8 类 [0,4,8,10,13.9,16,17.8,20] m/s | — |

**训练循环关键步骤**：
- forward pass：[train_bridgedrive.py:158](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/train_bridgedrive.py#L158)
- loss 计算：[train_bridgedrive.py:159](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/train_bridgedrive.py#L159)
- backward：[train_bridgedrive.py:175](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/train_bridgedrive.py#L175)
- optimizer.step：[train_bridgedrive.py:179](BridgeDrive/BridgeDrive_adaptation_LEAD/lead/training/train_bridgedrive.py#L179)

---

## 6. 与 MoT-DP 的关系

| BridgeDrive 组件 | MoT-DP 对应 | 说明 |
|---|---|---|
| `TransfuserBackbone` | `model/transfuser_extractor/` | 直接复用，加载预训练权重 |
| `PlanningDecoderDDBM`（含 DDBM head） | `model/transformer_for_diffusion_multi_head.py` + `policy/` | 替换为 DiT + DDIM，Route A/B 各有变体 |
| Radar / Perception Decoders | **不使用** | 辅助监督，不在 MoT-DP 中 |
| CenterNet → StopSign 后处理 | **不使用** | MoT-DP 暂无此后处理 |
| DDBM anchor-based 扩散 | Route A: anchor residual；Route B: anchor-free | Route B 完全抛弃 anchor，从 N(0,I) 出发 |
