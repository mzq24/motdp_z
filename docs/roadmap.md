# MoT-DP 开发路线图

## 现状总结

| 项目 | 当前值 |
|------|--------|
| 推理底座 | 32 Anchor + Truncated Diffusion (trunc=100) |
| 推理步数 | 2-step DDIM |
| 模型输出 | `poses_reg (B,32,6,2)` + `poses_cls (B,32)` + `route_pred (B,20,2)` |
| 轨迹选择 | `argmax(cls_logits)` (黑盒 focal loss) |
| 语义标签 | 已有 11 类 behavior label + allowed/forbidden flag (dataset 已支持) |
| 已验证 | 从白噪声 N(0,I) 去噪效果更好，收敛更快；缺点是需要 10 步推理 (5x 于 2-step) |

---

## 路线 A：极速闭环王者 (1-Step + Energy Evaluator)

### 状态：当前执行

### 核心思路
不改变高效 1-Step 推理底座，将"语义"降级为物理与几何约束，重构分类头。

### 保留的基座
- 1-Step Denoise + Normalized Forward + 33 Anchors (含 VLM 先验)
- 利用微弱噪声做数据增强 → 极度平滑的轨迹回归

### 白盒能量评估 (White-box Energy Evaluator)

抛弃单一黑盒 `poses_cls`，利用 `semantic_behavior_labeling` 自动生成的 Label，训练多任务能量头：

| 能量项 | 含义 | 数据来源 |
|--------|------|----------|
| `E_collision` | 碰撞概率 | Box 膨胀 + 未来帧检测 (behavior label 1-4) |
| `E_offroad` | 越界概率 | Distance Transform + BEV 语义图 (behavior label 5-6) |
| `E_target` | 导航服从度 | 路线点距离 |

### 需要修改的代码

#### 1. 模型层 (`model/transformer_for_diffusion_multi_head.py`)

在现有 `cls_head` 旁边添加能量头：

```python
# 现有 (约 line 1360)
self.cls_head = nn.Sequential(
    nn.Linear(n_emb, n_emb // 2), nn.SiLU(),
    nn.Linear(n_emb // 2, 1)
)

# 新增
self.energy_collision_head = nn.Sequential(
    nn.Linear(n_emb, n_emb // 2), nn.SiLU(),
    nn.Linear(n_emb // 2, 1), nn.Sigmoid()
)
self.energy_offroad_head = nn.Sequential(
    nn.Linear(n_emb, n_emb // 2), nn.SiLU(),
    nn.Linear(n_emb // 2, 1), nn.Sigmoid()
)
self.energy_target_head = nn.Sequential(
    nn.Linear(n_emb, n_emb // 2), nn.SiLU(),
    nn.Linear(n_emb // 2, 1)
)
```

#### 2. 策略层 (`policy/diffusion_dit_carla_policy.py`)

训练时添加能量损失：

```python
# 从 semantic_behavior_labels 构造监督信号
collision_target = (behavior_labels >= 1) & (behavior_labels <= 4)  # bool (B, M)
offroad_target = (behavior_labels >= 5) & (behavior_labels <= 6)    # bool (B, M)
target_dist = compute_route_distance(poses_reg, route)              # float (B, M)

energy_loss = (
    bce_loss(E_collision, collision_target) +
    bce_loss(E_offroad, offroad_target) +
    l1_loss(E_target, target_dist)
)
```

推理时进行 Energy Shielding：

```python
# 推理阶段
safe_logits = cls_logits - w_col * E_collision - w_off * E_offroad - w_tgt * E_target
best_idx = safe_logits.argmax(dim=-1)
```

#### 3. 训练脚本 (`training/train_carla_bev.py`)

添加能量损失的权重配置和 loss_dict 追踪。

### 优点
- 推理速度不变 (1-2 step)
- 可解释：知道为什么拒绝某条轨迹
- 闭环 Debug 友好

### 缺点
- 受限于 Anchor 覆盖范围，如果 33 条都不安全则无解
- 能量评估仅在离散候选上做后处理，无法生成新轨迹

---

## 路线 B：终极浪漫 (Annealed Energy Guidance)

### 状态：代码框架已完成 ✅（分支 `annealed-energy-guidance`）

### 已知问题
- 推理需要 **10 步 DDIM**，是当前 2-step 的 **5 倍推理时间**

### 核心思路
回归纯正 Diffusion，利用 Classifier Guidance / CFG，实现自上而下的模态坍缩。

### 砸碎 Anchor
- 起点不再是 32 个先验，而是纯高斯白噪声 $N(0, I)$
- 可以生成任意数量的候选轨迹 (不受 Anchor 数量限制)

### 时间尺度上的模态坍缩

```
t=100 → 70 (大噪声期)
  导航能量场发挥拉力
  轨迹从叠加态坍缩至 "宏观大方向 (如右转)"

t=70 → 30 (中噪声期)
  防碰撞能量场施加排斥力
  纵向速度被压缩
  坍缩至 "减速右转"

t=30 → 0 (小噪声期)
  越界能量场与平滑先验发力
  完成车道级几何打磨
```

### 已完成的代码（分支 `annealed-energy-guidance`）

#### 1. 模型层 (`model/transformer_for_diffusion_multi_head.py`)

- ✅ `anchor_free=True`：跳过 anchor residual，模型直接预测绝对轨迹
- ✅ `energy_heads=True`：添加 E_collision / E_offroad / E_target 能量头
- ✅ 向后兼容：`energy_heads=False` 时返回 4 值，`True` 时返回 5 值

#### 2. 策略层 (`policy/annealed_energy_guidance_policy.py`) — 新文件

- ✅ 训练：GT 轨迹加噪 → 模型预测 clean x_0 + 能量头用 behavior label 监督
- ✅ 推理：10-step DDIM 从 N(0,I) + 每步能量梯度引导
- ✅ 时间尺度权重调度 `get_energy_weights(t, T)`
- ✅ 推理最终选择：energy shielding + cls_logits 联合排序
- ✅ 采用方案一：复用主网络 decoder 输出 + 能量头（无需独立评估网络）

#### 3. 训练脚本 (`training/train_carla_bev.py`)

- ✅ `policy_type: anchor_free` 配置切换 Route B policy
- ✅ wandb logging 兼容 energy_loss

#### 4. 配置 (`config/pdm_local_route_b.yaml`)

- ✅ Route B 专用配置（`policy_type: anchor_free`）

### 时间尺度权重调度（已实现）

```python
def get_energy_weights(t, T=100):
    progress = 1 - t / T  # 0→1 随去噪推进
    w_nav = 1.0                              # 全程生效
    w_col = max(0, (progress - 0.3) / 0.4)   # 30% 进度后生效
    w_off = max(0, (progress - 0.7) / 0.3)   # 70% 进度后生效
    return w_nav, w_col, w_off
```

### 优点
- 不受 Anchor 限制，理论上能找到全局最优
- 物理直觉清晰：从粗到细、从宏观到微观
- 可以生成任意数量候选轨迹

### 缺点
- 10 步推理 = 5x 时间成本 (实测)
- 需要训练额外的 Energy Evaluator
- 梯度引导的超参数调优困难 (权重调度)

---

## 实施优先级建议

### 第一步：路线 A (当前)
1. 在现有模型上添加 3 个能量头
2. 用已有的 semantic_behavior_label 训练
3. 推理时做 Energy Shielding
4. **改动最小，风险最低**

### 第二步：路线 B (进阶)
1. 先训练好 Energy Evaluator (可从路线 A 的能量头复用)
2. 切换到白噪声起点 + 10-step DDIM
3. 添加梯度引导逻辑
4. 调优时间尺度权重

### 共享组件
路线 A 的能量头可以直接复用为路线 B 的 Energy Evaluator，两条路线互不冲突。

---

## 关键发现记录

- [x] 白噪声去噪验证通过：效果更好，收敛更快
- [x] 代价：10-step 推理 (vs 2-step)，5 倍时间
- [x] 路线 B 代码框架完成（分支 `annealed-energy-guidance`）
- [ ] 能量头训练效果待验证
- [ ] Energy Shielding 超参数待调优（路线 A）
- [ ] 梯度引导稳定性待验证（路线 B）
- [ ] `guidance_scale` 超参数待调优（路线 B）
