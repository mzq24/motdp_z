# Session Findings 2026-03-23

## 1. Normalization Ablation Results

历史最佳 Route B: L2_1s=0.293 (offline-run-20260320, 用 norm_odo)
历史最佳 Route A: L2_1s=0.240 (run-20260306, 用 norm_odo)

### Delta Z-Score (offline-run-20260322_202158)
- abs -> delta差分 -> per-step z-score
- 训练 validation 报的 L2_1s 卡在 ~0.65, 越训越高 (overfitting on mini dataset)
- 问题: cumsum 逆变换导致误差累积

### Per-Step Abs Z-Score (offline-run-20260323_022011)
- abs -> per-step z-score (每步独立 mean/std)
- 训练 validation 报的 L2_1s 同样卡在 ~0.65
- 实际 eval L2_1s = 0.40 (见下方 "validation 指标偏高" 发现)
- 问题: per-step 独立归一化抹掉了轨迹时序结构 (step 0 和 step 5 都归到 ~N(0,1))

### Global Abs Z-Score (待测)
- abs -> global z-score (所有 timestep 共享同一个 mean/std, x/y 分别)
- global_abs_stats: mean=(9.92, 0.19), std=(12.63, 2.58)
- 代码已实现, 未跑完训练

---

## 2. 重大发现: 训练 Validation 指标偏高 (不可信)

### 现象
训练 validation 报 val_L2_1s=0.65, 但独立 eval 脚本测出 L2_1s=0.40

### 根因
`validate_model()` 调用 `policy.compute_loss(batch)` -> `compute_diffusion_loss()`,
这是 legacy M=1 forward path (只用 diff_mode_query).

但训练用的是 `compute_unified_loss()` = M=34 unified forward
(32 anchor mode_queries + 1 gt_mode_query + 1 diff_mode_query).

模型以 M=34 方式训练, decoder self-attention 有 anchor isolation (block diagonal mask).
用 M=1 评估时 self-attention pattern 完全不同, 导致指标偏高.

### 影响
- 训练 validation 的 reg_loss 和 L2 指标都不可信
- 之前所有关于 "L2_1s 卡在 0.65" 的判断需要修正
- 实际 per-step abs z-score 的真实 L2_1s = 0.40, 不是 0.65

### 修复方向
修改 `validate_model()` 或 `compute_loss()`, 让 validation 走 unified forward path (或至少 M=1 但用 diff_mode_query 的正确方式)

---

## 3. Train vs Val 对比 (per-step abs z-score, best checkpoint)

| | Train | Val |
|---|---|---|
| L2_1s | 0.3951 | 0.4027 |
| L2_2s | 0.8036 | 0.8322 |
| L2_3s | 1.2580 | 1.3264 |
| L2_avg | 0.8189 | 0.8538 |

结论: 完全没有过拟合, train ~= val

---

## 4. Checkpoint Loading 陷阱

### register_buffer(name, None) 不进 state_dict
PyTorch 的 `register_buffer('name', None)` 注册的 buffer 不会出现在 `state_dict()` 中.
因此 `load_state_dict()` 时, checkpoint 中的对应 tensor 会变成 unexpected key 被跳过.

训练时的 register_buffer -> 非 None -> 正常保存到 checkpoint.
但加载时如果新 policy init 把 buffer 注册为 None, load_state_dict 不会覆盖.

修复: 加载 checkpoint 后手动恢复 buffer:
```python
sd = ckpt['model_state_dict']
for buf_name in ['abs_mean', 'abs_std', 'delta_mean', 'delta_std', 'anchor_centers_abs']:
    if buf_name in sd and sd[buf_name] is not None:
        policy.register_buffer(buf_name, sd[buf_name].to(device))
```

### EMA state_dict 格式不是标准 PyTorch state_dict
当前 EMA 用 diffusers 格式, `ema_state_dict` 包含 `shadow_params` (flat list),
不能直接用 `policy.load_state_dict()` 加载. 需要用 EMAModel 的 API.

---

## 5. 待做

- [ ] 修复 validation 函数, 让指标可信
- [ ] 测试 global abs z-score 训练效果
- [ ] 考虑恢复 norm_odo 做对照 (历史 Route B best = 0.293 用的就是 norm_odo)
- [ ] 修复 checkpoint loading (buffer + EMA)
- [ ] 用修正后的 eval 重新比较各 normalization 方案

---

## 6. 归一化方案对比 (代码已实现, 通过 config 切换)

| 方案 | config key | dispatch 优先级 | 状态 |
|------|-----------|----------------|------|
| Global abs z-score | `global_abs_stats_path` | 最高 | 代码+stats 就绪 |
| Per-step abs z-score | `abs_stats_path` | 中 | 已训练, L2_1s=0.40 |
| Delta z-score | `delta_stats_path` | 最低 (fallback) | 已训练, 效果差 |

dispatch 逻辑在 `norm_to_abs()` / `abs_to_norm()`:
global_abs_mean != None -> global abs z-score
abs_mean != None -> per-step abs z-score
else -> delta z-score (legacy)
