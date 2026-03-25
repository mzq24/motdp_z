# MOA 改造计划

## Context

当前 MultiSourceAttentionBlock 存在以下问题：
1. **BEV 特征被稀释**：Self KV 和 BEV KV 合并 softmax，self-attention 强时 BEV 权重被挤压
2. **Traj token 粒度不足**：每条轨迹压缩为 1 token，无法与 route 的 20 个 waypoint 做 fine-grained attention
3. **Target point 无降噪能力**：GPS 噪声直接影响 route 和 traj，无法在推理中逐步修正
4. **Route RoPE 依赖 M**：M=1 和 M=34 时 route 的 RoPE 位置不同（已知问题，split forward 已部分解决）

目标：将 MOA 改为混合式 attention（BEV 独立 + self 独立），扩展 traj 到 per-waypoint tokens，加入 TG denoising query。

---

## 设计决策总结

| 决策 | 结论 |
|------|------|
| Attention 结构 | **混合模式**：BEV cross-attn 独立 softmax + self-attn 独立 softmax |
| Traj token | **展开为 6 tokens**（每个 waypoint 一个），仅 ego forward |
| Anchor token | **保持 1 token/轨迹**，不展开，energy forward 中 anchor isolation 不变 |
| Energy / Guidance 路径 | **不引入 learned pooling**；ego 6-token 先解码出轨迹，再走现有 single-token energy eval 路径 |
| Target point 输入 | **保留在 conditioning 中**（包括 route conditioning），依靠 GPS 数据增广保鲁棒性 |
| GPS 鲁棒性 | **训练端先加 GPS noise**（baseline，必须做），TG denoising head 是 Phase 2 增强，不替代 train-time noise aug |
| TG query | **加入 1 个 learnable query**，接收 noisy TG embedding，通过 BEV attention 预测 clean TG |
| TG 可见性 | **只看 BEV**，self-attention 中 isolated（不看 route、traj） |
| Route conditioning | **保留 TG**，不去掉 |
| Anchor 在 ego forward | **不参与**，split forward 中 ego/energy 分离 |
| Output head | **per-token MLP**：每个 waypoint token 直接预测对应的 (x, y) |

---

## 新 Token 组成

### Ego Forward (M=1 diffusion denoising)

```
Sequence: [traj_wp(6) | route(20) | tg(1)] = 27 tokens
BEV KV:   64 tokens (8×8 grid)

Self-attention mask:
           traj(6)  route(20)  tg(1)
traj(6)      ✓        ✓         ✗
route(20)    ✗        ✓         ✗
tg(1)        ✗        ✗         ✓
```

### Energy Forward (M=33 anchor evaluation)

```
Sequence: [anchor(33) | route(20)] = 53 tokens  (不变，anchor 保持 1 token)
BEV KV:   64 tokens

Self-attention mask: 不变（anchor diagonal isolation + route block）
```

### Alignment Forward (M=1, t=0)

```
不直接复用 6-token traj 输出做 energy。
流程：
  1. Ego forward (6-token traj) → pred_x0 / tg_pred
  2. 将 pred_x0 重新走 single-token trajectory embedding
  3. 用现有 energy 路径评估（必要时构造 M=33 layout 与 energy training 对齐）
```

---

## 架构变更详细

### 1. MultiSourceAttentionBlock → HybridAttentionBlock

**当前**（merged softmax）:
```python
scores = cat([Q @ K_self^T, Q @ K_bev^T], dim=-1)
weights = softmax(scores)
output = weights @ cat([V_self, V_bev])
```

**新**（hybrid，两步独立 softmax）:
```python
# Step 1: BEV Cross-Attention (独立)
bev_out = softmax(Q_bev @ K_bev^T / sqrt(d)) @ V_bev
x = x + gate_bev * o_proj_bev(bev_out)          # segment-specific gate

# Step 2: Self-Attention (独立 + mask)
self_out = softmax(Q_self @ K_self^T / sqrt(d) + mask) @ V_self
x = x + gate_self * o_proj_self(self_out)        # segment-specific gate

# Step 3: FFN (不变)
```

**新增参数**（per block）:
- `o_proj_bev`: Linear(d_model, d_model) — BEV attention 输出投影
- `gate_bev_traj`, `gate_bev_route`, `gate_bev_tg`: 每个 segment 独立的 BEV gate
- 原有 `o_proj` 改为 self-attention 专用

**保留的参数**:
- Q adapter (traj/route 各自的 BEV Q 适配器)
- route-specific AdaLN, temperatures, biases
- BEV residual path
- RoPE

**新增 TG segment 处理**:
- `tg_adaLN_modulation`: TG 专用 AdaLN（类似 route 的做法）
- `tg_q_adapter_bev`: TG 专用 BEV Q 适配器
- `gate_bev_tg`: TG 的 BEV gate

### 2. Traj 展开为 6 tokens

**当前**: `(B, M, T, 2)` → sinusoidal → flatten → MLP → `(B, M, n_emb)` (1 token/轨迹)

**新** (ego forward only):
```python
# (B, 1, 6, 2) → per-waypoint embedding
wp_pos_embed = gen_sineembed_for_position(x_t_abs)  # (B, 1, 6, 64)
wp_embed = self.wp_emb(wp_pos_embed)                 # (B, 1, 6, n_emb)  Linear(64, n_emb)
wp_embed = wp_embed.squeeze(1)                        # (B, 6, n_emb)

# 加入 waypoint positional encoding + segment embedding
traj_emb = wp_embed + wp_pos_encoding + traj_segment_emb + conditioning.unsqueeze(1)
# + diff_mode_query broadcast 到 6 个 token
```

**Energy forward 保持不变**: anchor 仍然用 flatten → MLP → 1 token

**新增参数**:
- `wp_emb`: Linear(64, n_emb) — per-waypoint embedding (替代 anchor_emb for ego)
- `wp_pos_encoding`: (1, 6, n_emb) — 6 个 waypoint 的位置编码

### 2.5 Energy / Alignment 不做 Learned Pooling

不把 6 个 traj tokens 直接 pool 成 1 个 token 再接 energy head，原因：
- 新增 pooling 层本身需要训练，额外引入不确定性
- energy training 链路（anchor + GT）当前全部是 **1 token / trajectory**，若只有 ego 路径额外加 pooling，会制造新的 train/inference gap

采用更保守的方案：

```python
# Pass 1: ego forward（6-token traj）
pred_x0, route_pred, tg_pred = forward_ego(...)

# Pass 2: energy/alignment forward（single-token traj）
# 用 pred_x0_abs 重新做现有的 sineembed -> flatten -> anchor_emb
energy_scores = forward_energy_eval(pred_x0, pred_x0_abs, ...)
```

推荐默认布局：
- **训练 energy**：保持 `[anchor(32) | gt(1)]`
- **推理 / alignment eval**：优先使用 `[dummy_anchor(32) | pred_x0(1)]`
- 这样 route token 的位置与 energy training 保持一致，避免再引入新的 RoPE 位置偏移问题

### 3. TG Query 与 Prediction Head

**TG token 构造**:
```python
tg_pos_embed = gen_sineembed_for_position(noisy_tg)  # (B, 1, 64)  noisy_tg: (B, 1, 2)
tg_embed = self.tg_emb(tg_pos_embed)                  # (B, 1, n_emb)
tg_emb = tg_embed + self.tg_query + tg_conditioning   # learnable query + conditioning
```

**Prediction Head**:
```python
self.tg_prediction_head = nn.Sequential(
    nn.LayerNorm(n_emb),
    nn.Linear(n_emb, n_emb // 2), nn.SiLU(),
    nn.Linear(n_emb // 2, 2),  # predict (x, y)
)
```

**TG conditioning**: `tg_conditioning = time_emb + self.tg_status_proj(noisy_tg_flat)`

**Training loss**: `tg_loss = F.l1_loss(tg_pred, gt_target_point)`

**重要：TG head 不是 GPS noise 的替代品**
- 闭环 train/test gap 的主因是 `ego_pose -> world_to_ego -> target_point_ego` 链路中的定位噪声
- 先做 dataset 侧 GPS noise augmentation，让训练分布对齐闭环 test
- TG denoising head 的角色是“进一步利用 BEV 把 noisy TG 拉回到 lane / junction 合理位置”

**Training / Inference 策略（分阶段）**:

Phase A（推荐先做）:
- dataset 开启 GPS noise augmentation
- trajectory / route 仍使用 noisy TG conditioning
- TG head 只做 auxiliary supervision：`L1(pred_tg, gt_tg_clean)`
- **不**把 `pred_tg` 回灌给 trajectory decoder

Phase B（TG head 稳定后再做）:
- 推理 two-pass：
  - pass1: noisy TG → `pred_tg_clean`
  - pass2: 用 `pred_tg_clean.detach()` 替换当前帧 TG conditioning，再跑 ego forward
- 训练可选做 scheduled replacement：
  - 小概率用 `pred_tg.detach()` 或 teacher-forced clean TG 替换 noisy TG
  - 避免训练永远只见 noisy TG、推理却突然全换成 predicted TG

**不建议**:
- 直接依赖 diffusion/滤波去“解决 GPS”而不做 train-time noise aug
- 从一开始就让 `pred_tg` 强耦合回 trajectory 路径

### 4. RoPE 位置设计

Ego forward 的 RoPE 位置固定：
```
traj waypoints:  pos 0-5    (6 tokens)
route waypoints: pos 6-25   (20 tokens)
tg query:        pos 26     (1 token)
```

Energy forward RoPE 位置：
```
anchors:         pos 0-32   (33 tokens, 不变)
route waypoints: pos 33-52  (20 tokens, 不变)
```

两个 forward 的 route RoPE 位置不同（6-25 vs 33-52），因此：
- ego 训练/推理必须始终走自己的固定布局
- energy 训练/评估也必须始终走自己的固定布局
- alignment / guidance 若要调用 energy heads，优先复用 energy 布局，而不是把 6-token traj 直接池化后硬接过去

### 5. Mask 与 Segment 处理

**Ego forward mask** (27×27):
```python
def _create_ego_mask(T_wp=6, T_route=20, T_tg=1):
    T = T_wp + T_route + T_tg
    mask = torch.zeros(T, T)
    # route → traj: blocked
    mask[T_wp:T_wp+T_route, :T_wp] = -inf
    # tg → everything except self: blocked
    mask[T_wp+T_route:, :T_wp+T_route] = -inf
    # traj → tg: blocked
    mask[:T_wp, T_wp+T_route:] = -inf
    # route → tg: blocked
    mask[T_wp:T_wp+T_route, T_wp+T_route:] = -inf
    return mask
```

**Segment-specific 处理**: 3 segments (traj/route/tg)
- 各自有独立的 AdaLN modulation
- 各自有独立的 BEV gate
- 各自有独立的 BEV Q adapter

### 6. Output Head

**Per-token MLP**，每个 waypoint token 直接预测对应的 (x, y)：
```python
self.wp_output_head = nn.Sequential(
    nn.LayerNorm(n_emb),
    nn.Linear(n_emb, n_emb // 2), nn.SiLU(),
    nn.Linear(n_emb // 2, 2),
)
# input: (B, 6, n_emb), output: (B, 6, 2)
```

---

## 文件改动

| 文件 | 改动 |
|------|------|
| `model/transformer_for_diffusion_multi_head.py` | MultiSourceAttentionBlock → HybridAttentionBlock; 新增 wp_emb/tg_emb/tg_query/tg_prediction_head; traj 展开逻辑; 新增 wp_output_head; 更新 mask |
| `policy/annealed_energy_guidance_policy.py` | forward_ego 传入 noisy_tg; TG loss; inference TG refinement; traj 6-token 适配 |
| `training/train_carla_bev.py` | TG loss 日志; TG loss weight config |
| `config/pdm_local_route_b.yaml` | `use_predicted_tg`, `tg_loss_weight` 等新参数 |
| `config/pdm_hpc_route_b.yaml` | 同上 |
| `team_code/route_b_b2d_agent.py` | TG prediction 输出; traj 6-token 适配 |

---

## 实施步骤

### Phase 0: Split Forward 基建（前置条件）
1. 恢复/重建 `forward_ego`、`forward_energy`、`forward_alignment(or forward_energy_eval)`
2. 明确三条路径的 token layout、RoPE 位置、mask 各自固定
3. Policy 侧改为显式 dispatch，不再依赖单个 unified forward 硬兼容所有场景

### Phase 1: HybridAttentionBlock（核心 attention 拆分）
1. 复制 MultiSourceAttentionBlock → HybridAttentionBlock
2. 拆分 merged softmax 为 BEV cross-attn + self-attn 两步
3. 新增 o_proj_bev、per-segment BEV gate
4. 单元测试：输出 shape 一致，梯度流正常

### Phase 2: Traj 展开为 6 tokens（仅 ego path）
1. 新增 wp_emb (per-waypoint embedding)
2. 修改 forward_ego 构造 6-token traj_emb
3. 新增 wp_output_head（per-token → waypoint prediction）
4. 更新 ego forward mask
5. Energy / alignment 路径保持 single-token traj，不引入 pooling

### Phase 3: GPS Noise Baseline
1. 确认 `augmentation.gps_noise` 为训练默认配置
2. 保持 target point / next target point / waypoints history 的噪声注入逻辑
3. 先验证“只加噪、不加 TG head”时闭环鲁棒性收益

### Phase 4: TG Query（Auxiliary）
1. 新增 tg_emb、tg_query、tg_prediction_head
2. TG token 拼入 ego forward 序列
3. TG AdaLN + BEV Q adapter
4. TG prediction loss
5. 先不回灌 trajectory 路径，只做监控和可视化

### Phase 5: Two-Pass TG Refinement（可选增强）
1. pass1: noisy TG → `pred_tg_clean`
2. pass2: 用 `pred_tg_clean.detach()` 替换当前帧 TG conditioning
3. 训练端按概率做 scheduled replacement，减小 train/inference mismatch

### Phase 6: 集成与验证
1. Policy 适配（loss、inference）
2. Config 参数
3. Training script 适配
4. Agent 适配
5. 本地训练验证

---

## 验证

1. `python training/train_carla_bev.py --config config/pdm_local_route_b.yaml`
2. 确认 shape 正确：traj_out (B, 6, n_emb), route_out (B, 20, n_emb), tg_out (B, 1, n_emb)
3. Energy loss 6 heads 正常：single-token energy path 的指标不回退
4. TG prediction loss 非零且下降
5. 仅开 GPS noise aug 时，traj L2 与闭环 target_point 鲁棒性不恶化
6. 开启 two-pass TG refinement 后，L2 / closed-loop 成绩进一步提升
7. 推理：conditional_sample 无 NaN，TG prediction 合理
