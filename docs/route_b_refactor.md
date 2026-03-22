# Route B 基础架构改进

## 1. 命名规范

避免 anchor/residual/delta 混淆：

| 术语 | 含义 |
|------|------|
| `x_t` | 当前去噪轨迹（normalized delta z-score 空间） |
| `x_t_abs` | 当前去噪轨迹（绝对坐标，用于 BEV grid_sample） |
| `pred_x0` | model 预测的 clean 轨迹（normalized delta 空间） |
| `anchor_centers` | 32 个 k-means 聚类中心（abs 坐标），仅 energy Phase 1 训练用 |
| `delta` | 时间增量：`[p0, p1-p0, p2-p1, ...]`，p0 是相对 ego 的位移 |
| `residual` | 模型输出相对 anchor 的偏移（Route A 概念，Route B 不使用） |

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

## 3. Energy Gradient 作用点

### 问题

如果将 energy gradient 注入到 DDIM step 后的 x_t_next，会造成 diffusion model OOD——
model 训练时只见过 `pred_x0 + scheduled noise` 形式的 x_t，`x_t_next + energy_grad` 不在这个分布里。

### 解决

energy gradient 作用在 **pred_x0**（clean 预测）上，然后用修正后的 pred_x0 做 DDIM step：

```
model(x_t) → pred_x0, energy_scores
grad = autograd.grad(total_energy, pred_x0)
pred_x0_corrected = pred_x0 - guidance_scale * grad
x_{t-1} = DDIM_step(x_t, pred_x0_corrected)
```

这样：
- x_t_next 仍然是"某个 clean trajectory + scheduled noise"的形式，diffusion model 不 OOD
- energy head 评估的是 clean 预测，与其训练分布（clean anchor）一致

---

## 4. 涉及文件

| 文件 | 改动 |
|------|------|
| `model/transformer_for_diffusion_multi_head.py` | 参数重命名 anchors→x_t |
| `policy/annealed_energy_guidance_policy.py` | delta z-score norm + energy gradient 修正 |
| `dataset/compute_action_stats.py` | 改为统计 per-step delta mean/std |
| `config/pdm_local_route_b.yaml` | norm 参数格式更新 |
| `config/pdm_hpc_route_b.yaml` | 同上 |

---

## 5. 状态

- [ ] 重命名 anchors → x_t, anchors_abs → x_t_abs
- [ ] norm/denorm 改为 delta z-score
- [ ] energy gradient 作用在 pred_x0
- [ ] stats 脚本统计 per-step delta mean/std
- [ ] config 更新 norm 参数格式
