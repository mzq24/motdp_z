# Route B+: Compositional Energy-Guided Diffusion Policy — Implementation Summary

## Context

Route B 的 energy heads 存在两个问题：
1. **训练输入-标签不匹配**：GT 复制 32 份作输入，但 behavior_labels 来自 32 个不同 anchor → 能量头学不到判别能力
2. **缺少 planb_plus 设计的核心机制**：alignment loss、连续能量输出、梯度裁剪、LLM 权重接口

**解决方案**：双优化器（GAN-style D/G 隔离）+ GT 增广 + anchor 负样本

**梯度策略**：TransFuser 是外部模块（特征作为 batch 数据传入），DiT model 内的所有参数（decoder + energy heads）都参与梯度更新。

参考文档：`docs/planb_plus.md`

---

## 修改文件总览

| 文件 | 改动量 | 内容 |
|------|--------|------|
| `model/transformer_for_diffusion_multi_head.py` | 小 | 去 Sigmoid，连续能量输出 |
| `policy/annealed_energy_guidance_policy.py` | 大（全重写） | 双阶段 loss + GT 增广 + 推理改进 + 丰富输出 |
| `training/train_carla_bev.py` | 中 | 双优化器 + 两阶段训练循环 + anchor 注入 + checkpoint |
| `config/pdm_local_route_b.yaml` | 小 | 新配置项 |

---

## 核心架构：双优化器训练（GAN-style）

```
for batch in dataloader:
    # ===== Phase 1: 训练 energy heads（考官学打分）=====
    energy_loss = policy(batch, phase='energy')    # compute_energy_loss
    optimizer_energy.zero_grad()
    energy_loss.backward()
    optimizer_energy.step()

    # ===== Phase 2: 训练 diffusion decoder（学先验 + 向考官对齐）=====
    diff_loss = policy(batch, phase='diffusion')   # compute_diffusion_loss
    optimizer_diff.zero_grad()
    diff_loss.backward()
    optimizer_diff.step()
```

**为什么双优化器**：alignment loss 鼓励 decoder 骗过 energy heads（生成低能量轨迹），如果共享优化器，energy heads 会同时被拉向"输出零"，失去判别能力。分离后各自独立更新，类似 GAN 的 D/G 隔离。

---

## Step 1: 模型层 — 连续能量输出 ✅

**文件**: `model/transformer_for_diffusion_multi_head.py` (~line 1378-1391)

去掉 collision 和 offroad head 的 `nn.Sigmoid()`：
- `nn.Linear(n_emb // 2, 1), nn.Sigmoid()` → `nn.Linear(n_emb // 2, 1),`
- `energy_target_head` 本来就是连续输出，不变

**理由**（planb_plus §2.2）：Sigmoid 压缩梯度，连续输出保证轨迹空间梯度平滑可优化。

---

## Step 2: 策略层 — Phase 1 Energy Head Training ✅

**文件**: `policy/annealed_energy_guidance_policy.py` → `compute_energy_loss()`

### 输入构建（M=32 slots）：
- **前 K 个 slot**（默认 K=4）：GT 速度缩放增广（0.8-1.0x），标记为 safe
- **后 M-K 个 slot**：anchor 轨迹（带 behavior_labels）

### 正负样本策略：
- GT 增广 = 可靠正样本（safe，能量应低）
- Forbidden anchors = 负样本（collision/offroad，能量应高）
- Safe anchors = 可配置（`use_safe_anchors: false` 时跳过，default off）

### 损失函数：
- SmoothL1Loss（连续能量，替代 BCE）
- Masked：`use_safe_anchors=false` 时只在 GT增广 + forbidden 上算 loss

### 可选 noisy training：
- `energy_noisy_training: false`（默认）：t=0 clean trajectory
- `energy_noisy_training: true`：随机 t + add_noise，提升 energy heads 鲁棒性

---

## Step 3: 策略层 — Phase 2 Diffusion + Alignment ✅

**文件**: `policy/annealed_energy_guidance_policy.py` → `compute_diffusion_loss()`

```
L_total = L_reg * w_reg + L_cls * w_cls + L_route * w_route + β * L_alignment
```

- L_reg, L_cls, L_route: 标准 diffusion 损失（不变）
- L_alignment = Σ w_i * E_i(x̂_0).mean() — 鼓励 decoder 生成低能量轨迹
- β = `alignment_loss_weight`（默认 0.1）

---

## Step 4: 推理改进 ✅

### 梯度裁剪（planb_plus §4.3）
```python
grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
grad = grad * torch.clamp(max_norm / grad_norm, max=1.0)
```

### 动态权重接口（planb_plus §4.4）
`conditional_sample(energy_weights={"collision": 2.0, ...})` — LLM Router 可运行时覆盖

### 丰富输出
返回 dict 包含：`best_trajectory`, `route_pred`, `all_trajectories` (B,M,T,2), `energy_scores`, `poses_cls`, `safe_logits`, `best_idx`

---

## Step 5: 训练脚本 ✅

**文件**: `training/train_carla_bev.py`

- **Anchor 注入**：DDP 前 load `.npy`（np.load）或 `.pkl` → `policy.register_anchor_centers()`
  - 使用 `dd_baseline/anchors/carla_kmeans_32.npy`（K-means 聚类，shape `(32, 6, 2)`）
- **双优化器**：`optimizer_energy`（energy heads）+ `optimizer`（decoder）— 参数通过 `id()` 分离
- **双 scheduler**：各自 warmup + cosine annealing
- **Checkpoint**：保存/加载 `optimizer_energy_state_dict` + `scheduler_energy_state_dict`
- **Wandb**：额外 log `energy_loss`, `energy_col/off/tgt_loss`, `alignment_loss`, `lr_energy`

---

## Step 6: 配置 ✅

**文件**: `config/pdm_local_route_b.yaml`

```yaml
anchor_path: dd_baseline/anchors/carla_kmeans_32.npy  # K-means anchor (32, 6, 2)

route_b:
  alignment_loss_weight: 0.1        # β — alignment loss weight
  num_gt_augmentations: 4           # GT augmentation count (reliable positive samples)
  use_safe_anchors: false           # Trust safe anchor labels? (default off)
  energy_noisy_training: false      # Train energy heads on noisy trajectories?
  energy_grad_clip_norm: 1.0        # Per-element gradient clipping during inference
```

---

## 后续规划（本次不做）

1. **MOA 集成**：energy heads 作为 MultiSourceAttentionBlock 的 KV 源
2. **Compositional Conditional Denoising**：逐步叠加条件 ε(x,c1) → ε(x,c1,c2)
3. **Rule-based 正样本**：BEV road structure 生成合理轨迹
4. **连续 risk score 标签**：anchor_semantic_labeler 输出距离/比例
5. **能量头拆分**：forward_collision + pedestrian_collision
6. **Safe anchor 质量提升**：改进 labeling 后开启 `use_safe_anchors: true`
7. **num_samples 调优**：32 条趋同可降到 8 或 4

---

## 验证清单

- [ ] energy_loss 有意义：GT 增广 → 低能量，forbidden anchor → 高能量
- [ ] alignment_loss 下降：decoder 逐渐生成低能量轨迹
- [ ] 双优化器独立：energy_loss.backward() 不更新 decoder
- [ ] 正负样本比例：打印 active_mask 统计
- [ ] 推理稳定：梯度裁剪生效，无 NaN/Inf
- [ ] switch 可用：`energy_noisy_training` / `use_safe_anchors` 切换正常
