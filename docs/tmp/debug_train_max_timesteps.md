# Debug: train_max_timesteps 训练-推理不匹配问题

> 日期：2026-03-22  
> 状态：已修复  
> 影响：训练 loss 持续下降，但 val L2 不降反升

---

## 1. 异常现象

新一轮训练（lr=2e-4）的数据：

```
Epoch  1: train_loss=14.41
Epoch  5: train_loss= 5.05  → val_L2_avg=4.88   ← 最好的时候
Epoch 10: train_loss= 3.61  → val_L2_avg=5.54   ← 开始变差
Epoch 15: train_loss= 2.41  → val_L2_avg=6.45   ← 继续恶化
Epoch 50: train_loss= 1.35  → val_L2_avg=6.40
Epoch 95: train_loss= 1.18  → val_L2_avg=6.03   ← 训练越好，推理越差
```

**核心矛盾**：training loss 稳定下降（14.4 → 1.18），但 validation L2 先升后平（4.88 → 6.45 → 6.0）。  
这不是「收敛慢」的问题，而是**训练和推理在做不同的事**。

---

## 2. 排查思路

### 2.1 先确认不是归一化/反归一化 bug

之前怀疑过 delta/abs 转换流水线有问题，逐一验证了：

- ✅ 原始数据：`ego_waypoints`shape=(7,2)，step[0]永远是(0,0)
- ✅ Dataset: `agent_pos = ego_waypoints[1:]` → 6步，去掉原点
- ✅ Delta 计算：`delta[0] = pos1`（第一个 waypoint），`delta[1:] = step displacement`
- ✅ `compute_action_stats.py` 和 dataset 用一样的逻辑
- ✅ `z_norm` / `z_denorm` / `norm_to_abs` 数学上正确

**结论：归一化/反归一化流水线没问题。**

### 2.2 检查推理流水线

完整追溯了推理调用链：

```
validate_model()
  → model.predict_action(obs_dict)
    → self.conditional_sample(...)
      → x_t = torch.randn(B, M, T, 2)             # 从 N(0,I) 采样
      → for step in DDIM_schedule:
          model.forward(x_t, timestep=t_cur)        # 去噪
          → DDIM step
      → norm_to_abs(pred_x0_corrected)              # 反归一化到绝对坐标
    → return absolute trajectory
  → compute_driving_metrics(predicted, target)       # L2 对比
```

关键发现：`x_t` 从纯高斯噪声 `N(0,I)` 出发。

### 2.3 检查训练流水线

```
compute_unified_loss()
  → traj_normed = abs_to_norm(trajectory)            # GT 转 z-score delta
  → diff_timesteps = randint(0, train_max_timesteps)  # 随机 timestep
  → noisy_traj = scheduler.add_noise(traj_normed, noise, diff_timesteps)
  → model.forward(noisy_traj, timestep=diff_timesteps)
  → loss = L1(norm_to_abs(pred), GT)
```

关键参数：`train_max_timesteps = 100`。

### 2.4 挖出真正的 bug

DDIM scheduler 是 `num_train_timesteps=1000`。但 `train_max_timesteps=100` 意味着训练时 timestep 只在 `[0, 100)` 范围内采样。

那么 t=99（训练中最大噪声）对应的噪声有多大？

```python
from diffusers import DDIMScheduler
scheduler = DDIMScheduler(num_train_timesteps=1000, ...)
alphas_cumprod = scheduler.alphas_cumprod

# t=99: alpha_bar = 0.9717
# signal_weight = sqrt(0.9717) = 0.9857  (98.6% 信号保留!)
# noise_weight  = sqrt(1-0.9717) = 0.1682 (仅 16.8% 噪声)
```

也就是说模型训练时最多见过的噪声是 `0.168 * eps`，输入里 98.6% 都是干净信号。

但推理从 `N(0,I)` 出发 — **100% 纯噪声，0% 信号**。

```
训练见过的最大噪声: std = 0.168
推理起始噪声:       std = 1.000
比值: 6.5x — 远超训练分布!
```

---

## 3. 为什么 training loss 下降但 L2 上升

这其实是 **overfitting to low-noise regime** 的经典表现：

1. 模型越训练，越擅长在 t∈[0,99]（几乎干净的输入）上预测 x0
2. 但它从未学过如何处理 t=900（几乎纯噪声）的输入
3. 推理时第一步就收到了它从未见过的噪声级别的输入
4. 模型在这种 OOD 输入上产生垃圾输出
5. 后续 DDIM step 基于垃圾 pred_x0 继续，无法挽回

为什么 epoch 5 的 L2 (4.88) 反而比 epoch 95 (6.0) 好？  
因为 epoch 5 时模型还没有学会任何东西，输出接近随机初始化 — 某种程度上，随机猜测反而比一个专门学"低噪声去噪"的模型在面对"高噪声"时表现更好。

---

## 4. 为什么 Route A 用 100 没问题

Route A（anchor-based）的推理**不是从 N(0,I) 出发的**。它从 anchor trajectory 出发，加少量噪声后去噪。所以 truncated diffusion（只用前 100 步）是合理的 — 起点已经很接近 clean trajectory。

Route B（anchor-free）是 **pure diffusion from N(0,I)**，必须用完整的 1000 步训练范围。

```
Route A: anchor + small noise → denoise → trajectory
         ↓ truncated OK (noise small)
         train_max_timesteps = 100 ✅

Route B: N(0,I) → denoise → trajectory  
         ↓ must cover full range
         train_max_timesteps = 1000 ✅ (was 100 ✗)
```

---

## 5. 完整 Forward 数据流追踪

为了定位 bug，完整追踪了 training 和 inference 两条 forward 路径中每个 tensor 的空间（z-normed delta vs absolute）和形状变化。

### 5.1 Training Forward: `compute_unified_loss`

文件：`policy/annealed_energy_guidance_policy.py`

```
输入:
  batch['agent_pos']  →  trajectory (B, T=6, 2)  [绝对坐标, 单位: 米]

Step 1: 归一化 GT
  abs_to_delta(trajectory)  →  delta (B, 6, 2)     delta[0]=pos1, delta[1:]=step disp
  z_norm(delta)             →  traj_normed (B, 6, 2)  z-scored delta

Step 2: 加噪
  diff_timesteps = randint(0, train_max_timesteps=100)   ← BUG: 应该是 1000
  add_noise(traj_normed, noise, diff_timesteps)
    → noisy_traj (B, 1, 6, 2)  z-normed delta + noise
    公式: x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε

Step 3: 构建 34-slot 统一输入
  anchor_normed = abs_to_norm(anchor_abs)           (B, 32, 6, 2)  z-normed delta
  gt_normed     = abs_to_norm(gt_abs)               (B, 1, 6, 2)   z-normed delta
  x_t_unified   = cat([anchor_normed, gt_normed, noisy_traj])  (B, 34, 6, 2)  z-normed delta
  x_t_abs_unif  = cat([anchor_abs, gt_abs, noisy_abs])         (B, 34, 6, 2)  绝对坐标
  energy_traj   = cat([anchor_abs, gt_abs, gt_abs_placeholder]) (B, 34, 6, 2)  绝对坐标

Step 4: Model forward
  model(x_t=x_t_unified,           # z-normed delta → 用于 anchor embedding
        x_t_abs=x_t_abs_unified,    # 绝对坐标 → 用于 BEV grid_sample
        timestep=diff_timesteps,
        traj_for_energy=energy_traj) # 绝对坐标 → 用于 energy head 评估
  →  poses_reg (B, 34, 6, 2)   模型输出 [绝对坐标, anchor_free=True 无残差]
     mode_out  (B, 34, n_emb)  decoder 输出 embedding
     energy_scores {col, off, tgt} (B, 33)  前33个slot的能量分数

Step 5: Loss 计算
  poses_reg_diff = poses_reg[:, -1:]              (B, 1, 6, 2)  最后slot的预测
  poses_reg_diff_abs = norm_to_abs(poses_reg_diff) (B, 1, 6, 2)  转绝对坐标
  loss_reg = L1(poses_reg_diff_abs, gt_abs)        扩散损失

  energy_loss = smooth_l1(energy_scores, behavior_labels)  能量损失
  alignment_loss = sigmoid(detached_energy_eval(poses_reg_diff_abs, mode_out)).mean()
```

### 5.2 Model Forward: `TransformerForDiffusion.forward()`

文件：`model/transformer_for_diffusion_multi_head.py`

```
输入:
  x_t       (B, M, T, 2)      z-normed delta
  x_t_abs   (B, M, T, 2)      绝对坐标 (用于 BEV grid_sample)
  timestep  (B,)               扩散时间步
  traj_for_energy (B, M, T, 2) 绝对坐标 (用于 energy head)

bev_traj_points = x_t_abs     # BEV 采样用绝对坐标

=== Conditioning 构建 ===
  time_emb     = SinusoidalPosEmb(timestep)       (B, n_emb)
  status_emb   = Linear(ego_status[:, -1])         (B, n_emb)
  hist_emb     = GRU_encoder(ego_status)           (B, n_emb)
  conditioning = time_emb + status_emb + hist_emb  (B, n_emb)  → 用于 AdaLN

=== Anchor Embedding ===
  gen_sineembed(bev_traj_points, hidden=64)  (B, M, T, 64)  对绝对坐标做正弦位置编码
  flatten(-2)                                (B, M, T*64)
  anchor_emb = MLP(flatten)                  (B, M, n_emb)   Linear→SiLU→Linear

=== Mode Query 选择 ===
  训练 M=34: cat([mode_queries(32), gt_mode_query(1), diff_mode_query(1)])
  推理 M=1:  diff_mode_query

=== 组合 ===
  mode_emb = anchor_emb + mode_queries + conditioning  (B, M, n_emb)
  + behavior_emb (可选)
  → LayerNorm → Dropout

=== UnifiedDecoderOnlyTransformer ===
  输入:
    traj_emb (B, 34, n_emb) + route_queries (B, 20, n_emb) = x (B, 54, n_emb)
    self_attn_mask (54, 54):  trajectory 对角隔离 + route 互相可见
    
  处理流程 (每层):
    1. GridSampleCrossBEVAttention: traj_points 归一化到[-1,1] → grid_sample BEV upsample
       → 在轨迹点位置采样 BEV 特征, softmax 加权聚合
    2. Self-attention (with isolation mask): 每个 trajectory query 只能自注意
    3. Cross-attention to BEV tokens: 所有 query 关注 BEV grid features
    4. FFN + AdaLN(conditioning) + Residual
    
  输出:
    mode_out  (B, 34, n_emb)
    route_out (B, 20, n_emb)

=== Output Heads ===
  trajectory_head(mode_out, conditioning, route_out) → poses_reg (B, 34, T, 2)
    anchor_free=True → 无残差, 直接输出绝对坐标 (若 anchor_free=False 则 += x_t_abs)
  cls_head(mode_out) → poses_cls (B, 34)
  route_head(route_out, conditioning, current_status) → route_pred (B, 20, 2)
  
  energy_head 输入: cat([traj_for_energy.flatten(), mode_out])  (B, M, T*2+n_emb)
    → collision_head(input) → (B, M)
    → offroad_head(input)  → (B, M)  
    → target_head(input)   → (B, M)
```

### 5.3 Inference Forward: `conditional_sample`

文件：`policy/annealed_energy_guidance_policy.py`

```
x_t = randn(B, 1, T, 2)   ← 纯高斯噪声, z-normed delta space

DDIM schedule (修复前): step_ratio = 100/10 = 10
  timesteps = [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]  ← 最大 t=90

DDIM schedule (修复后): step_ratio = 1000/10 = 100
  timesteps = [900, 800, 700, 600, 500, 400, 300, 200, 100, 0]  ← 最大 t=900 ✅

for t_cur in timesteps:
  ┌─ Pass 1: 去噪 (no_grad) ─────────────────────────┐
  │  x_input = x_t                    (B, 1, T, 2)    │
  │  x_t_abs = norm_to_abs(x_input)   (B, 1, T, 2)    │
  │    = delta_to_abs(z_denorm(x_input))               │
  │    = cumsum(x_input * std + mean)                  │
  │                                                    │
  │  model(x_t=x_input, x_t_abs=x_t_abs, t=t_cur)    │
  │  → pred_x0 = poses_reg  (B, 1, T, 2) 绝对坐标     │
  └────────────────────────────────────────────────────┘

  ┌─ Pass 2: 能量引导 (with_grad) ────────────────────┐
  │  pred_x0_for_grad = pred_x0.clone().requires_grad_│
  │  pred_x0_abs = norm_to_abs(pred_x0_for_grad)      │  ← 可微分!
  │                                                    │
  │  model(x_t=pred_x0_for_grad,                       │
  │        x_t_abs=pred_x0_abs,                        │
  │        t=0,                                        │
  │        traj_for_energy=pred_x0_abs)                │
  │  → energy_scores {col, off, tgt}                   │
  │                                                    │
  │  total_energy = w_col*col + w_off*off + w_tgt*tgt  │
  │  grad = autograd.grad(total_energy, pred_x0_for_grad) │
  │  grad = clip_norm(grad, max=1.0)                   │
  │                                                    │
  │  pred_x0_corrected = pred_x0 - guidance_scale * grad │
  └────────────────────────────────────────────────────┘

  ┌─ DDIM Step ───────────────────────────────────────┐
  │  α_t = alphas_cumprod[t_cur]                       │
  │  α_next = alphas_cumprod[t_next]                   │
  │                                                    │
  │  pred_eps = (x_t - √α_t * pred_x0_corr) / √(1-α_t) │
  │  x_t = √α_next * pred_x0_corr + √(1-α_next) * pred_eps │
  └────────────────────────────────────────────────────┘

最终输出:
  final_abs = norm_to_abs(pred_x0_corrected)  (B, T, 2)  绝对坐标
```

### 5.4 关键发现：为什么 t_max=100 是灾难

在追踪过程中，清楚看到两个路径的 **timestep 使用位置**：

```
训练:  diff_timesteps = randint(0, train_max_timesteps)  → t ∈ [0, 100)
       → 最大噪声: sqrt(1-α_99) = 0.168

推理:  step_ratio = train_max_timesteps / num_inference_steps = 100/10 = 10
       timesteps = [90, 80, ..., 0]
       → 第一步 t=90, 期望的噪声: sqrt(1-α_90) = 0.155
       → 但实际输入 x_t ~ N(0,1), 噪声 std = 1.000

       就像让一个只学过「轻微模糊图片→清晰图片」的模型去处理「纯白噪声→清晰图片」
```

---

## 6. 修复

**config/pdm_local_route_b.yaml:**

```yaml
# 修改前 (WRONG)
route_b:
  train_max_timesteps: 100    # 只训练 [0,100) → noise_std 最大 0.168

truncated_diffusion:
  trunc_timesteps: 100
  train_trunc_timesteps: 100

# 修改后 (CORRECT)
route_b:
  train_max_timesteps: 1000   # 全范围 [0,1000) → noise_std 覆盖到 0.9996

truncated_diffusion:
  trunc_timesteps: 1000
  train_trunc_timesteps: 1000
```

修复后的 DDIM 推理 schedule：

```
timesteps: [900, 800, 700, 600, 500, 400, 300, 200, 100, 0]

t=900: noise_std=0.9977  ← 第一步，几乎匹配 N(0,I) 的 1.0 ✅
t=800: noise_std=0.9898
t=700: noise_std=0.9661
...
t=100: noise_std=0.1698
t=  0: noise_std=0.0100  ← 最后一步，几乎干净
```

---

## 7. 教训总结

1. **训练 loss 下降 ≠ 模型在变好**。当训练和推理的输入分布不匹配时，训练 loss 是在优化错误的东西。

2. **从 Route A 直接继承配置到 Route B 时要重新审视每个假设**。`train_max_timesteps=100` 对 Route A 是合理的，对 Route B 是灾难性的。

3. **关键诊断信号**："early epoch L2 比 late epoch L2 好" — 这意味着不是收敛问题，而是模型在往错误方向专精化。

4. **Diffusion 模型的核心约束：训练噪声范围必须覆盖推理噪声范围。** 如果推理从 N(0,I) 开始，训练必须包含 t 接近 T_max 的 timestep。

---

## 8. 数值验证

| 指标 | 修复前 (t_max=100) | 修复后 (t_max=1000) |
|------|-------------------|-------------------|
| 训练 noise range | [0, 0.168] | [0, 0.9996] |
| 推理起点 noise | 1.000 | 1.000 |
| 推理起点 vs 训练最大 | 6.5x 超出 | 1.00x 匹配 |
| DDIM 首步 timestep | t=90 | t=900 |
| 首步 expected noise | 0.155 | 0.998 |
