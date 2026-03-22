# Route B 基础架构改进

## 1. 命名规范

避免 anchor/residual/delta 混淆：

| 术语 | 含义 |
|------|------|
| `x_t` | 当前去噪轨迹（normalized delta z-score 空间） |
| `x_t_abs` | 当前去噪轨迹（绝对坐标，用于 BEV grid_sample） |
| `pred_x0` | model 预测的 clean 轨迹（normalized delta 空间） |
| `anchor_centers` | 32 个 k-means 聚类中心（abs 坐标），energy 训练用 |
| `delta` | 时间增量：`[p0, p1-p0, p2-p1, ...]`，p0 是相对 ego 的位移 |
| `traj_for_energy` | 传给 energy head 评估的轨迹（训练用 anchor 原始坐标，推理用 pred_x0） |

Model forward 参数重命名：`anchors` → `x_t`，`anchors_abs` → `x_t_abs`

---

## 2. 归一化：Delta Z-Score

### 2.1 数据流

```
训练:
  GT abs (B, T, 2) → abs_to_delta → z_norm → add noise → x_t
  传给 model: x_t (z-normed delta), x_t_abs (denorm → cumsum → abs)
  model 输出 pred_x0 → denorm → cumsum → abs → L1 loss with GT

推理:
  x_T ~ N(0, I) → DDIM denoise → pred_x0 → denorm → cumsum → abs
```

### 2.2 per-step z-score

每个 timestep 独立统计 mean/std：
```
delta_mean: (T, 2)  # 每步 delta 的均值
delta_std:  (T, 2)  # 每步 delta 的标准差
```

归一化：`z = (delta - mean[t]) / std[t]`
反归一化：`delta = z * std[t] + mean[t]`

### 2.3 abs ↔ delta 转换

```python
def abs_to_delta(abs_traj):
    # abs_traj: (B, T, 2), p0 已经是相对 ego
    delta = abs_traj.clone()
    delta[:, 1:] = abs_traj[:, 1:] - abs_traj[:, :-1]
    return delta  # [p0, p1-p0, p2-p1, ...]

def delta_to_abs(delta):
    return delta.cumsum(dim=-2)  # [p0, p0+(p1-p0), ...]
```

---

## 3. Energy Head 输入与训练/推理一致性

### 3.1 问题：energy head 应该评估什么轨迹

旧版 energy head 输入 `concat(poses_reg, mode_out)`，其中 `poses_reg` 是 model 对 GT 的去噪预测。
但 energy label（collision/offroad/target）是基于 **原始 anchor 坐标** 标注的，不是基于 model 输出。
这导致 label 和输入不匹配。

### 3.2 解决：traj_for_energy 参数

Model forward 新增 `traj_for_energy` 参数：
- **训练时**：传入原始 anchor 坐标，energy head 评估 `concat(anchor_flat, mode_out)`
- **推理时**：传入 `pred_x0`（需要 autograd），energy head 评估 `concat(pred_x0_flat, mode_out)`
- 不传时：fallback 到 `poses_reg`（向后兼容）

### 3.3 推理两次 forward

`mode_out` 依赖输入轨迹（通过 anchor_emb 和 BEV grid_sample），
所以评估 pred_x0 需要用 pred_x0 作为输入拿到正确的 mode_out：

```
Pass 1: model(x_t) → pred_x0          # 去噪
Pass 2: model(pred_x0, traj_for_energy=pred_x0) → energy_scores  # 沿 pred_x0 路径采 BEV
grad = autograd.grad(total_energy, pred_x0)
pred_x0_corrected = pred_x0 - guidance_scale * grad
x_{t-1} = DDIM_step(x_t, pred_x0_corrected)
```

---

## 4. Energy Gradient 作用点

### 问题

如果将 energy gradient 注入到 DDIM step 后的 x_t_next，会造成 diffusion model OOD——
model 训练时只见过 `pred_x0 + scheduled noise` 形式的 x_t，`x_t_next + energy_grad` 不在这个分布里。

### 解决

energy gradient 作用在 **pred_x0**（clean 预测）上，然后用修正后的 pred_x0 做 DDIM step。
corrected pred_x0 只是微调了 clean 轨迹，x_{t-1} 仍是 "clean + scheduled noise" 结构，model 不 OOD。
Training 时不做 correction（alignment loss 软引导即可），correction 仅 inference 时使用。

---

## 5. 统一 Forward Pass（34 modes）

### 旧版：两次 forward

Phase 1: 32 anchor → model → energy loss（单独 forward）
Phase 2: 1 x_t → model → diffusion loss（单独 forward）

### 新版：一次 forward

将所有 mode concat 成统一输入：

```
slots [0, 32):  anchor trajectories（带 behavior labels）
slot  [32]:     GT trajectory（clean，标记为 safe）
slot  [33]:     x_t（noisy GT，用于 diffusion 去噪）
```

- Mode queries: 前 33 个用 `mode_queries`，最后 1 个用 `diff_mode_query`
- Decoder self-attention 有 anchor isolation（block diagonal mask），mode 之间无信息泄漏
- Energy loss 只算前 33 个 slot，diffusion loss 只算最后 1 个 slot
- 梯度自然隔离：energy loss → `mode_queries` + energy heads，diffusion loss → `diff_mode_query` + decoder

---

## 6. 涉及文件

| 文件 | 改动 |
|------|------|
| `model/transformer_for_diffusion_multi_head.py` | 参数重命名 + `traj_for_energy` + 34-mode mode_query 路由 |
| `policy/annealed_energy_guidance_policy.py` | unified forward + 两次推理 forward |
| `training/train_carla_bev.py` | 统一训练循环（单次 forward + 双 optimizer step） |
| `dataset/compute_action_stats.py` | 统计 per-step delta mean/std |
| `config/pdm_local_route_b.yaml` | 移除旧 norm 参数 |
| `config/pdm_hpc_route_b.yaml` | 同上 |

---

## 7. 状态

- [x] 重命名 anchors → x_t, anchors_abs → x_t_abs
- [x] norm/denorm 改为 delta z-score
- [x] energy gradient 作用在 pred_x0
- [x] energy head 输入修正：traj_for_energy 评估原始 anchor
- [x] 统一 34-mode forward pass
- [x] 推理两次 forward（pass1 去噪 + pass2 energy 评估）
- [x] 训练脚本适配统一 forward
- [x] config 移除旧 norm 参数
- [x] stats 脚本统计 per-step delta mean/std
