# Main Project Model Structure Analysis

## 总体流程

```
DiffusionDiTCarlaPolicy.forward()
  -> compute_loss()
    -> _compute_multimodal_loss()
      -> self.model(anchors, anchors_abs, timestep, bev_features, ego_status)
         即 TransformerForDiffusion.forward()
```

## TransformerForDiffusion 整体结构

```
输入:
  anchors:          (B, M, T, 2)  -- M=num_modes(32), T=horizon(6)
  anchors_abs:      (B, M, T, 2)  -- 物理空间坐标 (用于BEV grid_sample)
  timestep:         (B,)
  bev_feature:      (B, 1512, 8, 8)
  bev_upsample:     (B, 64, 64, 64)
  ego_status:       (B, T_obs, status_dim)

输出:
  poses_reg:  (B, M, horizon, 2)   -- 每个mode的轨迹预测
  poses_cls:  (B, M)               -- mode分类logits
  route_pred: (B, num_waypoints, 2) -- 路线预测
```

### Forward 流程

1. **Conditioning 生成** (三部分相加):
   - `time_emb`: SinusoidalPosEmb(timestep) -> (B, n_emb)
   - `status_emb`: Linear(ego_status[:, -1, :]) -> (B, n_emb) (只用最后一帧)
   - `hist_global_emb`: HistoryEncoder(ego_status) -> (B, n_emb) (GRU+Attention编码全部history)
   - **conditioning = time_emb + status_emb + hist_global_emb**

2. **Anchor Embedding**:
   - `gen_sineembed_for_position(anchors_abs)` -> (B, M, T, 64) 正弦位置编码
   - flatten -> (B, M, T*64) -> MLP -> (B, M, n_emb)
   - **mode_emb = anchor_emb + mode_queries + conditioning**
   - 这里 `gen_sineembed_for_position` 确实就是给2D坐标加正弦位置编码

3. **Route Conditioning**: `route_status_proj(current_status)` -> (B, n_emb)

4. **UnifiedDecoderOnlyTransformer**: 核心decoder

5. **Output Heads**:
   - `trajectory_head`: MLP + route guidance cross-attn -> (B, M, horizon*2)
   - `cls_head`: MLP -> (B, M)
   - `route_head`: MLP + AdaLN -> (B, num_waypoints, 2)

---

## UnifiedDecoderOnlyTransformer 详解

### Query 构建

```
traj_emb: (B, M, n_emb)           -- 来自anchor embedding
route_emb: (B, num_waypoints, n_emb)  -- learnable route_queries

两者拼接:
x = [traj_emb | route_emb]  ->  (B, M + num_waypoints, n_emb)
```

位置编码: 共享的 sinusoidal + learnable scale + segment embedding 区分 traj/route

### `_create_block_diagonal_mask` 的作用

**对，就是把traj和route分开，但更精细:**

```
Attention Mask 结构:
              | Traj (M个anchor) | Route (20个waypoint) |
---------------------------------------------------------
Traj queries  |   diagonal only  | 可选 (traj_can_attend_route) |
Route queries |     blocked      |       full attention         |
```

- **Traj-to-Traj**: 只允许对角线 (每个anchor只能attend自己，不能看其他anchor)
  - 这是 DiffusionDrive 的关键设计: anchor之间要独立预测
- **Traj-to-Route**: 由 `traj_can_attend_route` 控制 (默认True，允许)
- **Route-to-Traj**: 完全blocked (route不依赖具体anchor)
- **Route-to-Route**: 完全可见 (route waypoints之间可以互相attend)

### GridSampleCrossBEVAttention

**在decoder的开头，只执行一次 (不在loop内)**

```python
# 在 UnifiedDecoderOnlyTransformer.forward() 中:
if traj_points is not None:
    x_traj = x[:, :T_traj, :]  # 只取traj部分
    x_traj = self.bev_spatial_attn(x_traj, traj_points, bev_feature_upsample)
    x = torch.cat([x_traj, x[:, T_traj:, :]], dim=1)  # 重新拼回route
```

#### 输入输出

| 参数 | 维度 | 含义 |
|------|------|------|
| `queries` (x_traj) | (B, M, d_model) | 每个anchor mode的embedding |
| `traj_points` | (B, M, T, 2) | anchor的空间坐标，即 `bev_traj_points` |
| `bev_feature` | (B, 64, 64, 64) | TransFuser FPN p3 输出，**不是上采样的** |
| **输出** | (B, M, d_model) | 增强后的query features (含residual) |

#### 内部计算流程

```
1. value_proj: Conv2d(64 → d_model, k=3, p=1)
   (B, 64, 64, 64) → (B, d_model, 64, 64)
   作用: 类似attention中的V projection，投影通道到d_model维
   空间尺寸不变 (64×64)，语义不变（车还是车，人还是人）

2. 坐标归一化: traj_points / lidar_max → [-1,1]，交换x,y适配grid_sample

3. F.grid_sample(value, grid, mode='bilinear')
   在每个waypoint的(x,y)位置做双线性插值采样BEV特征
   grid: (B, M, T, 2)
   sampled_features: (B, d_model, M, T)
   直观理解: M个mode × T个point，每个point采到一个d_model维特征向量

4. attention_weights = softmax(Linear(queries))  → (B, 1, M, T)
   每个mode对自己T个采样点的注意力权重（基于query内容，不是坐标）

5. 加权聚合: (weights * sampled_features).sum(dim=-1)
   T个point加权合并 → 每个mode得到1个d_model维特征
   (B, d_model, M, T) → (B, d_model, M) → permute → (B, M, d_model)

6. output_proj + dropout + residual → (B, M, d_model)
```

#### BEV 特征图分辨率与采样精度

- 特征图: 64×64 覆盖 64m×64m → **1m/pixel**
- Truncated diffusion 噪声很小 (trunc_timesteps 只有几步)
- 噪声带来的位置偏移可能 **< 1m，不到1个像素**
- 因此 noisy_traj_points vs clean_anchor 采到的 sampled_features **几乎一样**

**结论**: BEV 动态采样（noisy位置 vs 固定anchor位置）在 truncated diffusion 下差异极小，
可能**不是**多步DDIM中abs和delta表现差异的主要原因。需要通过实验4a消融验证。

#### traj_points 来源（policy层控制）

| 场景 | traj_points 的值 | 说明 |
|------|-----------------|------|
| 训练 (默认) | `noisy_anchors_abs` | 加噪后的轨迹（绝对坐标） |
| 推理 DDIM (默认) | `x_abs` (当前去噪状态) | 每步DDIM变化 → 动态采样 |
| `fix_bev_at_anchor=True` | `all_anchors` (clean) | 消融实验：固定在anchor位置 |

注意: `traj_points` 不是固定的，在推理的多步DDIM中每步都会变化（因为policy层每步传入新的 `x_abs`）。
但由于truncated diffusion噪声小，实际位移很小，采样差异可忽略。

#### 与标准Cross-Attention的对比

| | 标准 Cross-Attention | GridSampleCrossBEVAttention |
|---|---|---|
| 采样方式 | attend to 所有空间位置 | 只在轨迹坐标位置采样 |
| KV来源 | 全局 flatten 的 BEV tokens | F.grid_sample 在指定坐标插值 |
| 计算量 | O(Q × H × W) | O(Q × T_points) |
| 信息来源 | 全局 BEV | 轨迹路径附近的局部 BEV |

---

## MultiSourceAttentionBlock 详解

### 核心结构: 拼接式Multi-Source Attention

**当前只有 2 组 KV** (并行拼接到同一个softmax中):

| KV Source | Key | Value | Q Projection | Tokens数 |
|-----------|-----|-------|-------------|----------|
| **Self** | `k_self = Linear(x_norm)` | `v_self = Linear(x_norm)` | `q_base = Linear(x_norm)` | T (=M+20) |
| **BEV** | `k_bev = Linear(bev_tokens)` | `v_bev = Linear(bev_tokens)` | `q_bev = q_base + q_adapter_bev` | T_bev (=64) |

### Attention 计算流程

```
1. Pre-LayerNorm + AdaLN modulation (traj和route分别用不同的AdaLN参数)

2. Q projections:
   q_base = Linear(x_norm)                              # 共享base Q
   q_bev  = q_base + q_adapter(x_norm)                  # BEV专用Q (加了低秩adapter)
   (traj和route有各自的q_adapter: q_adapter_bev vs route_q_adapter_bev)

3. K, V projections:
   k_self, v_self = Linear(x_norm), Linear(x_norm)      # self-attention
   k_bev,  v_bev  = Linear(bev_tokens), Linear(bev_tokens)  # BEV cross-attention

4. Multi-head reshape: (B, L, d_model) -> (B, nhead, L, head_dim)

5. QK Norm: RMSNorm(q), RMSNorm(k)

6. RoPE:
   - Self-attention: Q和K都加RoPE (用主序列的位置)
   - BEV cross-attention: Q加主序列RoPE, K加BEV位置的RoPE

7. Attention scores (带per-source温度和bias):
   attn_self = (q_base @ k_self^T) * temp_self + bias_self
   attn_bev  = (q_bev  @ k_bev^T)  * temp_bev  + bias_bev
   (route部分用独立的 route_temp_bev, route_bias_bev)

8. 拼接并softmax:
   attn_scores = cat([attn_self, attn_bev], dim=-1) / sqrt(head_dim)
   attn_weights = softmax(attn_scores)

9. Value聚合:
   v_combined = cat([v_self, v_bev], dim=2)
   output = attn_weights @ v_combined

10. BEV Residual Path (额外):
    bev_pooled = mean_pool(bev_tokens) -> MLP -> gated residual
    output += sigmoid(gate) * bev_residual

11. Residual + gated output (traj和route分别用gate_attn/route_gate_attn)

12. FFN + AdaLN (同样traj和route分开modulate)
```

### Conditioning 注入方式

两种conditioning **都通过AdaLN**注入，不参与attention计算:
- **conditioning** (time+status+history): `adaLN_modulation` -> 6个参数 (shift_pre, scale_pre, gate_attn, shift_ffn, scale_ffn, gate_ffn) -> 控制traj部分的norm
- **route_conditioning** (status): `route_adaLN_modulation` -> 同样6个参数 -> 控制route部分的norm

**所以确认: 主要的信息注入来源是BEV tokens (64个，通过cross-attention)。Conditioning只做AdaLN调制。**

### 当前 KV 总结 (用于扩展参考)

```
一个 MultiSourceAttentionBlock 中的 KV pairs:

1. Self KV:  (B, nhead, T, head_dim)     -- T = num_modes + num_waypoints
2. BEV KV:   (B, nhead, 64, head_dim)    -- 来自 bev_feature (1512,8,8) flatten+project

Total KV tokens per query: T + 64
```

---

## 如果要添加新的 semantic source

参考当前BEV的并行KV模式，添加新source需要:

1. **在 `MultiSourceAttentionBlock.__init__` 中添加**:
   - `k_new`, `v_new`: 新source的KV projection
   - `q_adapter_new`, `q_adapter_new_out`: 新source的Q adapter (可选)
   - `temp_new`, `bias_new`: 新source的温度和bias
   - route版本: `route_q_adapter_new`, `route_temp_new`, `route_bias_new` (如果route也需要attend)

2. **在 `forward` 中**:
   - 计算 `k_new = self.k_new(new_tokens)`, `v_new = self.v_new(new_tokens)`
   - 计算 `q_new = q_base + q_adapter_new(x_norm)` (或复用q_base)
   - 计算 `attn_new = (q_new @ k_new^T) * temp_new + bias_new`
   - 拼接: `attn_scores = cat([attn_self, attn_bev, attn_new], dim=-1)`
   - 拼接: `v_combined = cat([v_self, v_bev, v_new], dim=2)`

3. **在 `UnifiedDecoderOnlyTransformer` 中**:
   - 添加新source的input projection (如果维度不匹配d_model)
   - forward中传入新tokens到每个layer

4. **在 `TransformerForDiffusion` 中**:
   - 接收新source features参数
   - 传给decoder

这种拼接式设计的好处: 所有KV source共享同一个softmax，模型自动学习attention权重分配。

---

## 参数规模参考

默认配置 (n_emb=512, n_head=8, n_layer=4):
- 每个 MultiSourceAttentionBlock: ~5.5M params (估算)
  - Q/K/V projections, adapters, temperature, FFN 等
- 总模型约 ~30-50M params (取决于layer数)
