# Route B+ 归一化数据流（全过程）

> **当前方案: Per-Timestep Abs Z-Score**
> `z = (abs - mean[t]) / std[t]`，每步独立归一化，无 delta/cumsum
> 逆变换: `abs = z * std[t] + mean[t]`，每步独立，无误差累积

## Slots 0-32: 33 条 anchor + GT（训练 energy head）

```
输入: anchor_abs / gt_abs                          (B, M, T, 2) abs 坐标
  │
  ├─── abs_to_norm ──→ anchor_normed / gt_normed   (B, M, T, 2) z-normed abs
  │      = (abs - mean[t]) / std[t]                 每步独立 z-score
  │
  │    拼接 → x_t_unified[0:33]                     z-normed abs   ← 传给 model 的 x_t
  │
  ├─── 直接用 ──→ x_t_abs_unified[0:33]             abs 坐标      ← 传给 model 的 x_t_abs
  │    model 里:
  │      bev_traj_points = x_t_abs                   abs
  │        → gen_sineembed_for_position(abs)          anchor_pos_embed
  │        → grid_sample(BEV, abs)                    采沿途 BEV 特征
  │      → decoder → mode_out                         (B, 33, n_emb)
  │      → trajectory_head(mode_out) → poses_reg      z-normed abs（不用这 33 个的 poses_reg）
  │
  ├─── anchor_abs / gt_abs ──→ traj_for_energy       abs 坐标     ← energy head 评估用
  │    model 里:
  │      energy_input = concat(traj_for_energy.flatten, mode_out)
  │      → energy_scores                               (B, 33, 3)
  │      collision/offroad/target 都是空间属性，abs 空间直接可判断
  │
  └─── label 来自: behavior_labels + allowed_flags（跟 anchor_abs 对应）
```

## Slot 33: x_t noisy GT（训练 diffusion decoder）

```
输入: trajectory                                    (B, T, 2) GT abs 坐标
  │
  ├─── abs_to_norm ──→ traj_normed                   (B, T, 2) z-normed abs
  │      = (abs - mean[t]) / std[t]
  │
  ├─── + noise ──→ noisy_traj                        (B, 1, T, 2) z-normed abs + noise
  │      scheduler.add_noise(traj_normed, noise, t)
  │      噪声在 z-normed abs 空间加
  │
  │    → x_t_unified[33]                              z-normed abs ← model 的 x_t
  │
  ├─── norm_to_abs(noisy_traj) ──→ noisy_traj_abs    (B, 1, T, 2) abs
  │      = z * std[t] + mean[t]                       每步独立反归一化，无 cumsum
  │
  │    → x_t_abs_unified[33]                          abs          ← model 的 x_t_abs
  │    model 里:
  │      bev_traj_points[33] = noisy_traj_abs          abs
  │        → sine embed + grid_sample BEV              沿噪声轨迹路径采 BEV
  │      → decoder → mode_out[33]
  │      → trajectory_head → poses_reg[33]             z-normed abs（这是 pred_x0）
  │
  ├─── norm_to_abs(poses_reg[33]) ──→ poses_reg_abs   abs
  │      = z * std[t] + mean[t]                        无 cumsum
  │    L1_loss(poses_reg_abs, trajectory)              abs vs abs ✓
  │
  └─── energy_scores[33] 来自 traj_for_energy 的 zero padding，不参与 energy loss
```

## 推理 DDIM Loop

```
x_t = randn(B, 1, T, 2)                             z-normed abs 空间的纯噪声

每步 DDIM:
  │
  ├─── Pass 1: 去噪
  │    x_t_abs = norm_to_abs(x_t)                     z * std + mean → abs
  │    model(x_t, x_t_abs, t) → pred_x0               z-normed abs
  │
  ├─── Pass 2: energy 评估（用 pred_x0 重新走一遍 model）
  │    pred_x0_for_grad = pred_x0.requires_grad_(True)  z-normed abs
  │    pred_x0_abs = norm_to_abs(pred_x0_for_grad)      abs（可微：z * std + mean）
  │    model(x_t=pred_x0_for_grad,                      z-normed abs
  │          x_t_abs=pred_x0_abs,                        abs（sine embed + grid_sample 沿 pred_x0 路径）
  │          t=0,                                        clean
  │          traj_for_energy=pred_x0_abs)                abs（energy head 输入）
  │    → energy_scores
  │    grad = autograd.grad(energy, pred_x0_for_grad)   梯度通过 abs → (z*std+mean) 回传
  │                                                      = grad_abs * std（线性变换，梯度简单缩放）
  │
  ├─── Correct
  │    pred_x0_corrected = pred_x0 - scale * grad      z-normed abs
  │
  └─── DDIM step
       pred_eps = (x_t - √α * pred_x0_corrected) / √(1-α)
       x_t = √α_next * pred_x0_corrected + √(1-α_next) * pred_eps   z-normed abs（下一步输入）

最终:
  norm_to_abs(pred_x0_corrected) → abs 轨迹输出
```

## 关键原则

1. **扩散过程** 全程在 z-normed abs 空间（加噪、去噪、DDIM step）
2. **BEV grid_sample** 必须用 abs 坐标（`x_t_abs`）
3. **Energy head 评估** 用 abs 坐标（`traj_for_energy`），因为 collision/offroad/target 是空间属性
4. **Energy 梯度** 通过可微的 `norm_to_abs`（`z * std + mean`，线性变换）回传，梯度 = `grad_abs * std`
5. **Loss 计算** 在 abs 空间比较（`norm_to_abs(poses_reg)` vs GT abs）
6. **每步独立** 无 cumsum，无误差累积，逆变换每个 timestep 独立

## 对比旧方案（Delta Z-Score）

| | Abs Z-Score (当前) | Delta Z-Score (旧) |
|---|---|---|
| 归一化 | `(abs - mean[t]) / std[t]` | `abs → delta → (delta - mean[t]) / std[t]` |
| 反归一化 | `z * std[t] + mean[t]` | `z * std + mean → cumsum` |
| 误差累积 | 无，每步独立 | 有，cumsum 导致前面步的误差传播到后面 |
| 梯度 | 线性缩放 `grad * std` | 通过 cumsum 回传，梯度耦合 |
