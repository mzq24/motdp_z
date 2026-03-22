# MoT-DP 两条技术路线规划

## 路线 A：极速闭环王者 (1-Step + Energy Evaluator)

**分支：`energy-evaluator`** | 状态：当前执行阶段

### 核心思想
在不改变现有高效 1-Step 推理底座的前提下，将"语义"降级为物理与几何约束，重构分类头。

### 保留的强基座（不变）
- 1-Step Denoise + Normalized Forward + 33 Anchors（含 VLM 先验）
- 利用微弱噪声做数据增强，回归极度平滑的轨迹

### 新增：白盒化能量评估 (White-box Energy Evaluator)
抛弃单一的黑盒 `poses_cls`，利用 `semantic_behavior_labeling` 自动生成的 Label，训练多任务能量头：

| 能量头 | 含义 | 标签来源 |
|--------|------|----------|
| `E_collision` | 碰撞概率 | 基于 Box 膨胀 |
| `E_offroad` | 越界概率 | 基于 Distance Transform |
| `E_target` | 导航服从度 | 目标点距离 |

### 推理：能量屏蔽 (Energy Shielding)
1. 模型输出 33 条打磨好的轨迹 + 原始 `cls_logits`
2. 计算安全打分：
   ```
   safe_logits = cls_logits - w_col * E_collision - w_off * E_offroad - w_nav * E_target
   ```
3. 选 `safe_logits` 最高的轨迹
4. 效果：模型主动拒绝 VLM 提出的高碰撞风险轨迹，实现"知其然，知其所以然"的可解释闭环

### 涉及修改的文件
- `model/transformer_for_diffusion_multi_head.py` — 加能量头
- `dataset/unified_carla_dataset.py` — 加载 semantic label
- `training/train_carla_bev.py` — 加能量头的 loss
- `policy/diffusion_dit_carla_policy.py` — 推理时 energy shielding
- config yaml — 加能量头相关参数

---

## 路线 B：终极浪漫 (Annealed Energy Guidance)

**分支：`annealed-energy-guidance`** | 状态：代码框架已完成 ✅

### 核心思想
回归纯正 Diffusion，砸碎 Anchor，利用分类器引导 (Classifier Guidance) 实现自上而下的模态坍缩。

### 无 Anchor 起点
- 起点不再是 33 个先验，而是纯高斯白噪声 N(0, I)
- 配置：`anchor_free=True`

### 时间尺度上的模态坍缩
| 阶段 | 时间步 | 主导能量场 | 效果 |
|------|--------|-----------|------|
| 大噪声期 | t=100 → 70 | 导航能量场 (E_target) | 轨迹从叠加态坍缩至"宏观大方向（如右转）" |
| 中噪声期 | t=70 → 30 | 防碰撞能量场 (E_collision) | 纵向速度被压缩，坍缩至"减速右转" |
| 小噪声期 | t=30 → 0 | 越界能量场 (E_offroad) + 平滑先验 | 完成车道级几何打磨 |

### 执行方式
在每一步 DDIM 去噪时，注入评估网络的梯度：
```
x_{t-1} = DDIM_Step(x_t) + w1 * ∇_{x_t} E_collision + w2 * ∇_{x_t} E_offroad
```

### 实验发现
- 从白噪声开始去噪，收敛更快、性能更好
- 10-step vs 1-step 的 L2 error 几乎一致（说明 multi-step 的作用更像 noise robustness 而非 iterative refinement）：
  ```
  val_L2_avg (10-step): 0.8456
  val_L2_avg (1-step):  0.8420
  ```
- 缺点：需要 10-step DDIM 推理，推理时间是 2-step 的 5 倍

### 已完成的文件
- ✅ `model/transformer_for_diffusion_multi_head.py` — `anchor_free=True` 跳过 residual + `energy_heads=True` 添加 3 能量头
- ✅ `policy/annealed_energy_guidance_policy.py` — 10-step DDIM 从 N(0,I) + 能量梯度引导 + 时间尺度权重调度
- ✅ `config/pdm_local_route_b.yaml` — Route B 专用配置（`policy_type: anchor_free`）
- ✅ `training/train_carla_bev.py` — `policy_type` 配置驱动策略选择 + energy_loss logging

---

## 两条路线对比

| 维度 | 路线 A | 路线 B |
|------|--------|--------|
| 推理步数 | 1-2 step | 10 step |
| 推理速度 | 快（实时） | 慢（5x） |
| 起点 | 33 Anchors | N(0, I) 白噪声 |
| 能量头作用 | 后处理筛选 | 每步梯度引导 |
| 改动量 | 小（加 head + shielding） | 大（重构推理流程） |
| 可解释性 | 高（能量分数直接可读） | 中（梯度引导隐式） |
| 当前状态 | 待实现 | 代码框架已完成 ✅ |
