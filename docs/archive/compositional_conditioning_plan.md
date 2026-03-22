# Compositional Conditional Denoising: Text-Conditioned BEV Query

## 核心思想

在去噪过程中，用语义文本指令逐步 query BEV grid，渐进注入任务特定的空间条件。

文本指令作为"语义透镜"：
- "follow lane" → attend 到车道结构区域
- "avoid collision" → attend 到前方车辆/障碍物区域
- "follow route" → attend 到导航目标方向区域

不同指令在不同去噪阶段激活，从粗到细渐进叠加条件。

---

## 与 Energy Guidance 的关系

**完全解耦，可独立开关**，分别对应两篇文章：

| 维度 | Energy Guidance (Paper 1) | Text-Conditioned BEV (Paper 2) |
|------|--------------------------|-------------------------------|
| 模型参数 | 3 个 MLP energy heads | TextConditionedBEVQuery + decoder text_bev K/V |
| 训练 loss | energy supervision + alignment | contrastive loss (独立 phase) |
| 推理机制 | 后处理梯度注入 (model forward 之后) | 前处理条件注入 (model forward 之内) |
| 配置开关 | `energy_heads: true/false` | `text_conditioning.enabled: true/false` |

四种组合都合法：Energy only / Text only / Both / Neither。

---

## 架构设计

### 1. 固定词表（6 个指令，3 正 + 3 负）

| ID | 指令 | 语义 | 对比配对 |
|----|------|------|----------|
| 0 | `follow_lane` | 车道跟随（正） | ↔ 3 |
| 1 | `avoid_collision` | 避碰（正） | ↔ 4 |
| 2 | `follow_route` | 跟随导航（正） | ↔ 5 |
| 3 | `leave_lane` | 偏离车道（负） | ↔ 0 |
| 4 | `collision` | 碰撞（负） | ↔ 1 |
| 5 | `ignore_route` | 忽略导航（负） | ↔ 2 |

每个指令是一个可学习的 embedding 向量 (d_model 维)。

### 2. TextConditionedBEVQuery 模块

```
text_emb (K, d_model) → cross-attention(Q=text, K=V=bev_tokens) → text_bev_features (B, K, d_model)
```

- 输入：指令 ID 列表 + BEV tokens (B, 64, d_model)
- cross-attention + residual + FFN
- 输出：每个指令的 BEV 特征 + attention weights（可视化）
- 附带 contrastive 投影头（用于对比学习）

### 3. Decoder 注入方式

text_bev_features 作为 **第三 KV 源** 注入 MultiSourceAttentionBlock：

```
attn_scores = cat([attn_self, attn_bev, attn_text_bev], dim=-1)
v_combined  = cat([v_self,    v_bev,    v_text_bev],    dim=2)
```

- 每层有独立的 K/V 投影 + Q adapter + temperature + bias
- 门控残差 (gate 初始化为 ~0 → 初始不影响现有行为)
- `has_text_bev=False` 时完全跳过 → 向后兼容

### 4. 渐进指令调度

与 energy annealing 三段式对齐：

```
t=100→70 (大噪声期): [follow_lane]                    — 车道结构
t=70→30  (中噪声期): [follow_lane, avoid_collision]    — + 障碍物
t=30→0   (小噪声期): [follow_lane, avoid_collision, follow_route] — + 导航
```

推理时只传正向指令 (0/1/2)。训练 Phase 2 也按 timestep 渐进。
训练 Phase 3（contrastive）传全部 6 个指令（正+负）。

---

## 训练机制

### 三阶段独立训练

| Phase | 内容 | 条件 | Optimizer |
|-------|------|------|-----------|
| Phase 1 | Energy heads supervision | `energy_heads=True` | `optimizer_energy` |
| Phase 2 | Diffusion + alignment | 始终 | `optimizer_diff` |
| Phase 3 | Contrastive learning | `text_conditioning=True` | `optimizer_text` |

### 对比学习 (Contrastive Loss)

- 正对：(正向指令 embedding, 安全轨迹 embedding) → 高相似度
- 负对1：(负向指令 embedding, 安全轨迹 embedding) → 低相似度
- 负对2：(正向指令 embedding, 危险轨迹 embedding) → 低相似度
- Loss: InfoNCE with learnable temperature
- 正负判断：用 `allowed_flags` 区分安全/危险

### 对比学习为什么能让 text attention 关注正确区域？

1. **contrastive gradient**：loss 回传到 cross-attention 的 Q (text embedding)，鼓励它选择能区分安全/危险轨迹的 BEV 区域
2. **energy head co-training**：text-conditioned features 流入 decoder → energy heads 评估 → 间接教 text attention 关注安全相关区域
3. **渐进注入**：不同噪声级别激活不同指令，每个指令在最相关的尺度上学习

---

## 数据流

### 训练

```
Phase 1 (energy):     anchor → model(ids=None)      → energy_loss
Phase 2 (diffusion):  noisy GT → model(ids=[0,1,2])  → diff_loss + alignment
Phase 3 (contrastive): anchor → model(ids=[0..5])    → contrastive_loss
```

### 推理

```
DDIM step at t:
  ids = get_active_instructions(t)
  model(x_t, ids) → decoder 看到渐进增加的语义条件
  + energy gradient guidance (如果 energy_heads 开启)
```

---

## 修改文件

| 文件 | 改动 |
|------|------|
| `model/transformer_for_diffusion_multi_head.py` | TextConditionedBEVQuery 模块 + MultiSourceAttentionBlock text_bev 扩展 + 穿透到 decoder/model |
| `policy/annealed_energy_guidance_policy.py` | contrastive loss + get_active_instructions + 推理/训练 instruction 传递 |
| `training/train_carla_bev.py` | optimizer_text + Phase 3 循环 + logging |
| `config/pdm_local_route_b.yaml` | text_conditioning 配置节 |

---

## 配置

```yaml
text_conditioning:
  enabled: true                   # 总开关
  num_instructions: 6             # 3 正 + 3 负
  contrastive_loss_weight: 0.5
  instruction_schedule:
    follow_lane: 0.0              # progress=0% 即激活
    avoid_collision: 0.3          # progress=30% 激活
    follow_route: 0.7             # progress=70% 激活
```

---

## 后续

1. **LLM Router 接入**：LLM 动态选择/加权指令，替代固定时间表
2. **MOA 集成**：text_bev_features 作为第四 KV 源
3. **开放词表**：固定 embedding → CLIP/text encoder
4. **注意力正则化**：鼓励不同指令关注 BEV 不同区域 (diversity loss)
5. **连续 risk score**：contrastive 从二元升级为连续距离
