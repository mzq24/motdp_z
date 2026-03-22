# Route B+: Compositional Energy-Guided Diffusion Policy

## 1. 核心思想

将条件扩散策略升级为**可组合、可控的轨迹生成框架**。

- **基座**：纯 Diffusion，从 N(0,I) 白噪声出发，10-step DDIM 去噪
- **能量引导**：Multi-head Energy Decomposition，显式附加语义独立的能量头
- **单 mode 去噪**：diffusion 训练/推理 M=1，energy 训练用 M=32 anchor 提供正负样本多样性

### 组合能量公式

$$E(\tau) = \sum_{i=1}^{N} w_i E_i(\tau)$$

权重 $w_i$ 代表约束优先级，是接入 LLM Router 的核心接口。

---

## 2. 能量头定义

| 能量头 | 含义 | 标签来源 | 输出类型 |
|--------|------|----------|----------|
| `E_collision` | 碰撞风险 | behavior_labels 1-4 | 连续（无 Sigmoid） |
| `E_offroad` | 越界风险 | behavior_labels 5-6 | 连续（无 Sigmoid） |
| `E_target` | 导航偏离度 | 1 - allowed_flags | 连续 |

去掉 Sigmoid 保证梯度在轨迹空间平滑可优化。

---

## 3. 时间尺度退火调度

```
t=100→70 (大噪声期): 导航能量场 → 宏观方向（如右转）
t=70→30  (中噪声期): 碰撞能量场 → 纵向速度压缩（减速右转）
t=30→0   (小噪声期): 越界能量场 + 平滑先验 → 车道级打磨
```

```python
def get_energy_weights(t, T=100):
    progress = 1.0 - t / T
    w_nav = 1.0                              # 全程
    w_col = max(0, (progress - 0.3) / 0.4)   # 30% 后
    w_off = max(0, (progress - 0.7) / 0.3)   # 70% 后
    return w_nav, w_col, w_off
```

---

## 4. 架构：双 mode_query 设计

模型内有两套独立的 mode query 参数：
- `mode_queries (1, 32, n_emb)` — Phase 1 energy 训练，32 个 anchor slot
- `diff_mode_query (1, 1, n_emb)` — Phase 2 diffusion 训练 + 推理，单 mode

forward 根据输入 M 自动选择：M=1 且 anchor_free 时用 `diff_mode_query`，否则用 `mode_queries[:, :M]`。

---

## 5. 训练：双优化器（GAN-style D/G 隔离）

```
for batch in dataloader:
    # Phase 1: 训练 energy heads（考官学打分）
    energy_loss = policy(batch, phase='energy')     # M=32, anchor 输入
    optimizer_energy.zero_grad(); energy_loss.backward(); optimizer_energy.step()

    # Phase 2: 训练 diffusion decoder（M=1 单 mode 去噪）
    diff_loss = policy(batch, phase='diffusion')    # M=1, GT 加噪
    optimizer_diff.zero_grad(); diff_loss.backward(); optimizer_diff.step()
```

### Phase 1: Energy Head Training (`compute_energy_loss`)

- 输入：32 个 anchor 轨迹（带 behavior_labels）
- GT 增广：暂时关闭（`num_gt_augmentations: 0`），待设计更好的增广方案
- Loss：SmoothL1Loss（连续能量）
- Mask：默认只在 forbidden anchor 上算 loss（`use_safe_anchors: false`）

### Phase 2: Diffusion Training (`compute_diffusion_loss`)

- 输入：GT 轨迹 M=1，加随机噪声
- Loss：L_reg (L1) + L_route (L1) + β * L_alignment
- L_alignment = Σ w_i * E_i(x̂_0).mean() — 鼓励生成低能量轨迹
- alignment_warmup_epochs 控制延迟开启

**双优化器原因**：alignment loss 让 decoder 骗过 energy heads（生成低能量轨迹），共享优化器会让 energy heads 也被拉向"输出零"，失去判别能力。

---

## 6. 推理：10-step DDIM + 能量梯度引导

```
x_0 ~ N(0,I)，shape (B, 1, T, 2)
for each DDIM step t:
    w_nav, w_col, w_off = get_energy_weights(t)
    开梯度 forward → energy_scores
    grad = autograd.grad(total_energy, x_t)
    grad = clip(grad, max_norm)
    x_{t-1} = DDIM_step(x_t) - guidance_scale * grad
output = denorm(x_0)  # 单条轨迹
```

M=1 不需要 energy shielding 选择，纯靠梯度引导生成最优轨迹。

动态权重接口：`conditional_sample(energy_weights={"collision": 2.0})` — LLM Router 可运行时覆盖。

---

## 7. 涉及文件

| 文件 | 角色 |
|------|------|
| `model/transformer_for_diffusion_multi_head.py` | DiT 模型 + energy heads + 双 mode_query |
| `policy/annealed_energy_guidance_policy.py` | 双阶段 loss + DDIM 推理 + 梯度引导 |
| `training/train_carla_bev.py` | 双优化器 + anchor 注入 + checkpoint |
| `config/pdm_local_route_b.yaml` | 本地配置 |
| `config/pdm_hpc_route_b.yaml` | HPC 配置 |

---

## 8. 后续规划

### 8.1 MOA 改造（下一步）
将 MultiSourceAttentionBlock 从"合并 softmax"改为"分离式 cross-attention"：
- 各 KV 源独立 softmax、独立输出
- Self KV → traj head，BEV KV → scene understanding head
- Route 从拼接序列改为独立 KV 源

### 8.2 Text-Conditioned BEV Query（远期）
文本指令作为"语义透镜"query BEV grid，渐进注入任务特定空间条件：
- 固定 6 指令词表（3 正 + 3 负）
- TextConditionedBEVQuery 模块 cross-attend BEV
- 对比学习（Phase 3 独立训练）
- 渐进指令调度与 energy annealing 三段式对齐

### 8.3 GT 增广改进（待设计）
等间距上采样方案：轨迹上采样到高密度点后取不同子集，比速度缩放更物理合理。
需考虑并线窗口期等 edge case。

### 8.4 其他
- 连续 risk score 标签（替代二元 behavior label）
- 能量头拆分：forward_collision + pedestrian_collision
- Safe anchor 质量提升后开启 `use_safe_anchors: true`

---

## 9. 验证清单

- [ ] energy_loss 有意义：forbidden anchor → 高能量
- [ ] alignment_loss 下降：decoder 逐渐生成低能量轨迹
- [ ] 双优化器独立：energy_loss.backward() 不更新 decoder
- [ ] M=1 diffusion 训练收敛正常
- [ ] 推理稳定：梯度裁剪生效，无 NaN/Inf
