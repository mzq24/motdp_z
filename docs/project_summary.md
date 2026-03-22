# MoT-DP 项目工作总结

自动驾驶轨迹规划项目。TransFuser backbone 提取 BEV 特征，DiT 架构做 diffusion-based 轨迹预测。本文档为各模块的 high-level 分类总结，详细实现见各子文档。

---

## 1. 基座系统

- **Backbone**: TransFuser (双编码器 image+lidar → 4 层 GPT Transformer 融合 → FPN)，来自 carla_garage
- **Diffusion Model**: DiT 架构，32 个 kmeans anchor，每个 anchor 预测 6-step (3s) 轨迹
- **输出**: `poses_reg (B,32,6,2)` + `poses_cls (B,32)` + `route_pred (B,20,2)`
- **推理**: Truncated Diffusion + DDIM (默认 2-step)

详见: [project_notes.md](project_notes.md) (backbone 对比), [model_structure_analysis.md](model_structure_analysis.md) (模型前向流程)

---

## 2. DD Baseline（DiffusionDrive 复现 + 消融）

DiffusionDrive 的独立复现，用于控制变量做消融实验。代码在 `dd_baseline/`。

### 消融实验汇总

| 实验 | 配置 | 多步 L2_1s | 1-step L2_1s | 结论 |
|------|------|-----------|-------------|------|
| Abs 默认 | 绝对坐标 + 动态 BEV | **0.240** | 0.295 | 基线 |
| Abs + Normalized Forward | [-1,1] 空间前向 | — | 0.290 | BEV 空间信息非关键（线性缩放不改变形状） |
| Delta (per-step) | Z-score delta + 固定 BEV | 0.564 | **0.264** | 1-step 更优，多步严重退化 |
| Delta + 动态 BEV | delta + cumsum→abs BEV | 0.344 | 0.268 | 恢复 55% gap，BEV 动态采样是多步稳定性关键之一 |

### 关键结论

1. **BEV 动态采样**对多步 DDIM 稳定性至关重要（贡献约 55%）
2. Per-step delta 表征本身更好（1-step: 0.264 vs 0.295），但多步 DDIM 下缺少 residual 连接导致发散
3. Abs 模式的 `output = offset + noisy_input` residual 连接在多步中起稳定作用

详见: [experiment_ood_validation.md](experiment_ood_validation.md) (实验 2/2b/2c/4b 完整记录)

---

## 3. Main Project 改进

在基座系统上叠加的功能增强：

| 功能 | 说明 | 配置开关 |
|------|------|---------|
| **VQA Anchor** | VLM 预测轨迹作为第 33 个 mode，接近 GT 的 anchor 缓解 OOD | `use_vqa_anchor: true` |
| **Normalized Forward** | 模型在 [-1,1] 空间前向，BEV grid_sample 不再提供物理空间信息 | `use_normalized_forward: true` |
| **Semantic Behavior Label** | 11 类行为标签 (碰撞/越界/正常等)，用于能量头监督和 mode 过滤 | `semantic_behavior_cfg` |
| **White Noise + Delta 模式** | 从 energy-evaluator 提取：full diffusion (白噪声训练) + predict_delta + dynamic_bev | 见 `*_energy_eval.py` 文件 |

White noise / delta 相关文件（从 `energy-evaluator` 分支提取，未合并到主文件，供参考对比）：
- `policy/diffusion_dit_carla_policy_energy_eval.py` — 含 full_diffusion + delta + dynamic_bev 的完整 policy
- `model/transformer_for_diffusion_multi_head_dynamic_bev.py` — 动态 BEV 模型变体
- `config/pdm_local_energy_eval.yaml` — energy-evaluator 的 config
- `config/pdm_local_white_noise_abs.yaml` / `config/pdm_local_white_noise_delta.yaml` — 白噪声实验 configs
- `training/train_carla_bev_energy_eval.py` — energy-evaluator 的训练脚本

详见: [experiment_ood_validation.md](experiment_ood_validation.md) (实验 1/3), [semantic_behavior_labeling.md](semantic_behavior_labeling.md)

---

## 4. Route A：1-Step + White-box Energy Evaluator

**思路**: 保留高效 1-step 推理底座，添加可解释的能量头做后处理 shielding。

- 在现有 cls_head 旁添加 `E_collision`, `E_offroad`, `E_target` 三个能量头
- 用 semantic behavior label 监督训练
- 推理时: `safe_logits = cls_logits - w_col * E_col - w_off * E_off - w_tgt * E_tgt`
- 优点: 推理速度不变，可解释; 缺点: 受限于 anchor 覆盖范围

**状态**: 代码框架在 `energy-evaluator` 分支，能量头已加入模型

详见: [roadmap.md](roadmap.md) (路线 A 完整设计)

---

## 5. Route B：Annealed Energy Guidance

**思路**: 抛弃 anchor，从纯 N(0,I) 噪声出发，10-step DDIM + 分阶段能量梯度引导。

- `anchor_free=True`: 模型直接预测绝对轨迹（非 anchor residual）
- `energy_heads=True`: 同时输出能量评估
- 时间尺度调度: 导航能量(全程) → 防碰撞(30%后) → 防越界(70%后)
- 优点: 不受 anchor 限制，物理直觉清晰; 缺点: 10-step 推理 = 5x 时间

**状态**: 代码完成，在当前分支 (`annealed-energy-guidance`)

关键文件: `policy/annealed_energy_guidance_policy.py`, `config/pdm_local_route_b.yaml`

详见: [roadmap.md](roadmap.md) (路线 B 完整设计), [route_a_b_plan.md](route_a_b_plan.md)

---

## 6. BridgeDrive 参考分析

分析了 BridgeDrive 的 DDBM (Denoising Diffusion Bridge Model) 方案如何从根本上解决 OOD 问题：

- **核心**: 用 Brownian Bridge 替代 truncated diffusion，恢复 forward/reverse 对称性
- **双分支 BEV**: 同时在 noisy 位置（动态）和 anchor 位置（固定）做 grid_sample 后 concat
- **局限**: 噪声被两端钉住，探索能力弱；anchor 选错则 bridge 精准送到错误终点

详见: [experiment_ood_validation.md](experiment_ood_validation.md) (BridgeDrive 部分), [bridgedrive_architecture.md](bridgedrive_architecture.md), [bridgedrive_engineering.md](bridgedrive_engineering.md)

---

## 7. Branch 对应关系

| Branch | 内容 | 状态 |
|--------|------|------|
| `vla_adapter` | 基础版本：TransFuser + DiT + 32 anchor truncated diffusion | 基线，无 dd_baseline |
| `energy-evaluator` | Route A：dd_baseline + delta/full_diffusion + 能量头 + dynamic_bev 模型 + **main project white noise 训练** | 有独立的 delta policy 和 white noise configs，已提取到 `*_energy_eval.py` |
| `annealed-energy-guidance` | **最完整**：Route B + VQA anchor + semantic behavior + dd_baseline 消融 | 当前主要开发分支 |

**注意**: `energy-evaluator` 上有部分文件不在当前分支（如 `transformer_for_diffusion_multi_head_dynamic_bev.py`, `diffusion_dit_carla_policy_delta.py`, white noise configs），暂未合并。
