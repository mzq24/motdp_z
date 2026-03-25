# Split Forward 改动总结

## 背景

**问题 1：Route RoPE 依赖 M**
- `MultiSourceAttentionBlock` 对 `x = cat([traj_tokens, route_tokens])` 整体算 RoPE
- M=34 训练时 route 在 pos 34-53，M=1 推理时 route 在 pos 1-20
- 不同 RoPE → 不同 route_out → 通过 `trajectory_head.route_guidance_attn` 影响 traj 预测
- 实测：M=1 Route_L2=0.33, Traj L2_1s=0.58；M=34 Route_L2=0.07, Traj L2_1s=0.17

**问题 2：Alignment loss 用 noisy mode_out**
- 旧实现直接用 unified forward 中 noisy x_t 对应的 mode_out 做 energy 评估
- 应该用 clean pred_x0 重新过 decoder（t=0）获取 mode_out

**解决方案：拆分 forward**
- Ego 训练/推理都是 M=1 → route RoPE 位置一致
- Energy 独立 M=33 forward → 不影响 ego 的 RoPE
- Alignment 用 clean pred_x0 单独过 decoder → 有意义的 energy 评估
- BEV 特征 `compute_bev_proj` 一次计算，三次 forward 复用

---

## 改动文件一览

| 文件 | 改动类型 |
|------|----------|
| `model/transformer_for_diffusion_multi_head.py` | 新增 6 个方法 |
| `policy/annealed_energy_guidance_policy.py` | 新增 `compute_split_loss`，重写 `conditional_sample`，修改 `forward` dispatch |
| `training/train_carla_bev.py` | 训练/验证 phase 切换 |
| `config/pdm_local_route_b.yaml` | 新增 `use_split_forward: true` |

---

## 1. `model/transformer_for_diffusion_multi_head.py`

### 1a. `UnifiedDecoderOnlyTransformer.compute_bev_proj` (line ~1136)

BEV 特征预计算，供三次 forward 复用。

```python
def compute_bev_proj(self, transfuser_bev_feature):
    # (B, 1512, 8, 8) → flatten → project → add pos_emb → (B, 64, d_model)
    bev_feat = transfuser_bev_feature.flatten(2).permute(0, 2, 1)
    bev_proj = self.bev_feature_proj(bev_feat)
    bev_proj = bev_proj + self.combined_pos_emb[:, :bev_proj.shape[1], :]
    return bev_proj
```

逻辑来源：从 `UnifiedDecoderOnlyTransformer.forward()` 中提取 BEV 处理部分。

### 1b. `UnifiedDecoderOnlyTransformer.forward` 新增 `bev_proj_cached` 参数 (line ~1158)

```python
def forward(self, ..., bev_proj_cached=None):
    if bev_proj_cached is not None:
        bev_proj = bev_proj_cached   # 跳过重复计算
    else:
        bev_proj = self.compute_bev_proj(transfuser_bev_feature)  # 原逻辑
```

### 1c. `TransformerForDiffusion._compute_conditioning` (line ~1516)

从 `forward()` line 1554-1573 提取的辅助方法，三个 split forward 共用。

```python
def _compute_conditioning(self, timestep, ego_status):
    """Returns: (conditioning, current_status, route_conditioning)"""
    time_emb = self.time_emb(timesteps)
    status_emb = self.ego_status_proj(current_status)
    hist_global_emb = self.history_encoder(ego_status)
    conditioning = time_emb + status_emb + hist_global_emb
    route_conditioning = self.route_status_proj(current_status)
    return conditioning, current_status, route_conditioning
```

### 1d. `TransformerForDiffusion._embed_trajectory` (line ~1549)

从 `forward()` 提取的 trajectory 几何编码。

```python
def _embed_trajectory(self, x_t_abs):
    """(B, M, T, 2) → sinusoidal pos embed → flatten → anchor_emb MLP → (B, M, n_emb)"""
    anchor_pos_embed = gen_sineembed_for_position(x_t_abs, hidden_dim=self.anchor_pos_hidden_dim)
    anchor_pos_embed = anchor_pos_embed.flatten(-2)
    return self.anchor_emb(anchor_pos_embed)
```

### 1e. `TransformerForDiffusion.forward_ego` (line ~1566)

M=1 ego 去噪 forward。

```
输入：x_t (B,1,T,2), timestep, ego_status, bev_proj_cached, bev_upsample, x_t_abs
输出：poses_reg (B,1,T,2), route_pred (B,20,2), mode_out (B,1,n_emb), conditioning (B,n_emb)
```

关键逻辑：
- 使用 `diff_mode_query`（不是 mode_queries）
- Decoder M=1 → route RoPE 在 pos 1-20（与推理一致）
- 经过 `trajectory_head` 和 `route_head` 得到输出

```python
anchor_emb = self._embed_trajectory(bev_traj_points)  # (B, 1, n_emb)
mode_emb = anchor_emb + self.diff_mode_query + conditioning.unsqueeze(1)
mode_out, route_out = self.decoder(traj_emb=mode_emb, bev_proj_cached=bev_proj, ...)
poses_reg = self.trajectory_head(mode_out, conditioning, route_features=route_out)
route_pred = self.route_head(route_out, conditioning, current_status)
```

### 1f. `TransformerForDiffusion.forward_energy` (line ~1614)

M=33 energy 评估 forward（32 anchors + 1 GT）。

```
输入：x_t (B,33,T,2), timestep=0, ego_status, bev_proj_cached, bev_upsample,
      x_t_abs, traj_for_energy (B,33,T,2), behavior_labels, allowed_flags
输出：energy_scores (dict of (B,33)), mode_out (B,33,n_emb)
```

关键逻辑：
- 使用 `cat([mode_queries(32), gt_mode_query(1)])` — 不用 diff_mode_query
- 附加 behavior_emb + allowed_emb 到 mode_emb
- energy_input = `cat([traj_for_energy.flatten(-2), mode_out], dim=-1)` → 三个 energy head
- **Decoder 传梯度**（不是 no_grad），energy loss 也训练 decoder 共享权重

```python
mode_queries = torch.cat([self.mode_queries, self.gt_mode_query], dim=1)[:, :M, :]
mode_emb = anchor_emb + mode_queries + conditioning.unsqueeze(1)
# + behavior_emb + allowed_emb (if available)
mode_out, _ = self.decoder(traj_emb=mode_emb, bev_proj_cached=bev_proj, ...)
energy_input = torch.cat([eval_traj_flat, mode_out], dim=-1)
energy_scores = {k: head(energy_input).squeeze(-1) for k, head in ...}
```

### 1g. `TransformerForDiffusion.forward_alignment` (line ~1680)

M=1 alignment forward，用 clean pred_x0 在 t=0 过 decoder。

```
输入：pred_x0 (B,1,T,2) normalized, ego_status, bev_proj_cached, bev_upsample, pred_x0_abs
输出：mode_out (B,1,n_emb)
```

关键逻辑：
- timestep 固定为 0（clean trajectory）
- 使用 `diff_mode_query`（同 forward_ego）
- 只返回 mode_out，不走 trajectory_head / route_head
- **pred_x0 保留梯度**，alignment loss 可反传到 ego decoder 路径

```python
timestep = torch.zeros(B, dtype=torch.long, device=device)
conditioning, _, route_conditioning = self._compute_conditioning(timestep, ego_status)
anchor_emb = self._embed_trajectory(bev_traj_points)
mode_emb = anchor_emb + self.diff_mode_query + conditioning.unsqueeze(1)
mode_out, _ = self.decoder(traj_emb=mode_emb, bev_proj_cached=bev_proj, ...)
return mode_out
```

### 1h. 原有 `forward()` 保持不动 (line ~1723)

向后兼容，旧 checkpoint / 旧 config 仍可用 unified forward。

---

## 2. `policy/annealed_energy_guidance_policy.py`

### 2a. `forward()` dispatch 新增 `'split'` (line ~366)

```python
def forward(self, batch, return_loss_dict=False, phase='unified'):
    if phase == 'split':
        return self.compute_split_loss(batch)    # ← 新增
    elif phase == 'unified':
        return self.compute_unified_loss(batch)
    elif phase == 'energy':
        return self.compute_energy_loss(batch)
    else:
        return self.compute_diffusion_loss(batch)
```

### 2b. `compute_split_loss` (line ~627)

三次独立 forward 的训练 loss 计算。

**数据准备**（同 compute_unified_loss）：
- trajectory, ego_status, transfuser_bev_feature, route_gt
- behavior_labels, allowed_flags
- anchor_abs (32 anchors), gt_abs (1 GT)
- noisy_traj (加噪 GT → x_t)

**Forward 1 — Ego (M=1)**：
```python
bev_proj = self.model.decoder.compute_bev_proj(transfuser_bev_feature)
poses_reg, route_pred, _, _ = self.model.forward_ego(
    x_t=noisy_traj, timestep=diff_timesteps, ego_status=ego_status,
    bev_proj_cached=bev_proj, ..., x_t_abs=noisy_traj_abs)

loss_reg = F.l1_loss(norm_to_abs(poses_reg), trajectory.unsqueeze(1))
route_loss = F.l1_loss(route_pred, route_gt)
```

**Forward 2 — Energy (M=33)**：
```python
energy_normed = cat([anchor_normed(32), gt_normed(1)], dim=1)  # (B, 33, T, 2)
energy_abs = cat([anchor_abs(32), gt_abs(1)], dim=1)

energy_scores, _ = self.model.forward_energy(
    x_t=energy_normed, timestep=zeros, ego_status=ego_status,
    bev_proj_cached=bev_proj, ..., traj_for_energy=energy_abs,
    behavior_labels=behavior_labels, allowed_flags=allowed_flags)

# 同 compute_unified_loss 的 energy loss 计算
energy_loss = smooth_l1(energy_scores, targets)
```

**Forward 3 — Alignment (M=1, clean pred_x0)**：
```python
# 仅在 alignment_warmup_epochs 之后激活
pred_x0_abs = norm_to_abs(poses_reg)  # 保留 grad → 反传到 ego decoder

mode_out_clean = self.model.forward_alignment(
    pred_x0=poses_reg, ego_status=ego_status,
    bev_proj_cached=bev_proj, ..., pred_x0_abs=pred_x0_abs)

# 用 detached energy heads 评估（只训练 decoder，不训练 energy heads）
align_col = self._eval_energy_head_detached(energy_collision_head, ...)
alignment_loss = w_col * sigmoid(align_col).mean() + w_off * ... + w_tgt * ...
```

**Total loss**：
```python
total_loss = (energy_loss_weight * energy_loss
            + reg_loss_weight * loss_reg
            + route_loss_weight * route_loss
            + alignment_loss_weight * alignment_loss)
```

返回 dict keys: `total_loss, energy_loss, energy_col/off/tgt_loss, reg_loss, cls_loss(=0), route_loss, alignment_loss`

### 2c. `conditional_sample` 重写 (line ~1095)

推理改用 split forward，不再需要 M=34 padding workaround。

**改动要点**：
- 使用 `self.model.decoder.compute_bev_proj` 预计算 BEV
- `x_t = randn(B, 1, T, 2)` — 直接 M=1，不构造 M=34
- 每个 DDIM step：
  - Pass 1: `forward_ego`（M=1）→ pred_x0
  - Pass 2: `forward_alignment` → mode_out → energy heads → 梯度 → guidance correction
- 移除了 `use_m34_inference`, `m34_slot_order`, `use_zero_context` 等 M=34 workaround 属性

**推理流程**：
```
for each DDIM step:
    ┌─ no_grad ──────────────────────────────────────┐
    │  forward_ego(x_t, t) → poses_reg (pred_x0)    │
    └────────────────────────────────────────────────┘
    ┌─ enable_grad ──────────────────────────────────┐
    │  pred_x0_for_grad = pred_x0.clone().requires_grad_()
    │  forward_alignment(pred_x0_for_grad, t=0)      │
    │  → mode_out → energy_heads → total_energy      │
    │  → autograd.grad → grad clipping → correction  │
    └────────────────────────────────────────────────┘
    pred_x0_corrected = pred_x0 - guidance_scale * grad
    DDIM step: x_t → x_{t-1}
```

---

## 3. `training/train_carla_bev.py`

### 3a. 新增 `route_b_cfg` 提取 (line ~556)

```python
route_b_cfg = config.get('route_b', {})
```

### 3b. 训练 loop phase 切换 (line ~835)

```python
route_b_phase = 'split' if route_b_cfg.get('use_split_forward', False) else 'unified'
loss_dict = policy(batch, return_loss_dict=True, phase=route_b_phase)
```

### 3c. 验证函数 `validate_model` 新增 `route_b_cfg` 参数 (line ~107)

```python
def validate_model(..., route_b_cfg=None):
    ...
    route_b_phase = 'split' if (route_b_cfg or {}).get('use_split_forward', False) else 'unified'
    loss_dict = model_for_inference(batch, return_loss_dict=True, phase=route_b_phase)
```

两个调用处（val-only 和 epoch-end）均传入 `route_b_cfg=route_b_cfg`。

---

## 4. `config/pdm_local_route_b.yaml`

```yaml
route_b:
  ...
  energy_grad_clip_norm: 1.0
  use_split_forward: true      # ← 新增
```

设为 `false` 或删除此行则回退到旧 unified forward。

---

## 计算量对比

| | 旧 (M=34 unified) | 新 (split) |
|---|---|---|
| 训练 decoder tokens | 54 (34 traj + 20 route) × 1 pass | 21 + 53 + 21 = 95, 3 passes |
| 推理 decoder tokens | 54 (M=34 padding) | 21 (M=1) — **2.6x 加速** |
| BEV 计算 | 每次 forward 都算 | 1 次，缓存复用 |
| Alignment 质量 | noisy mode_out → 无意义 | clean pred_x0 at t=0 → 有意义 |

训练 ~1.76x decoder 计算，推理 2.6x 加速。

---

## 梯度流向

```
              ┌──────────────┐
              │  BEV Encoder │  ← 三条路径都反传
              └──────┬───────┘
                     │ bev_proj (cached)
        ┌────────────┼────────────┐
        ▼            ▼            ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Ego (M=1)│ │Energy(33)│ │Align(M=1)│
  │ decoder  │ │ decoder  │ │ decoder  │
  └────┬─────┘ └────┬─────┘ └────┬─────┘
       │            │            │
  reg_loss     energy_loss  alignment_loss
  route_loss   (训练 heads   (heads detached,
               + decoder)    只训练 decoder)
```

- **Ego forward**: reg_loss + route_loss → 训练 decoder + trajectory_head + route_head
- **Energy forward**: energy_loss → 训练 energy heads + decoder（共享权重）
- **Alignment forward**: alignment_loss → 只训练 decoder（energy heads detached）

三条路径共享同一个 decoder 权重，各自的 loss 通过各自的 backward 更新。
