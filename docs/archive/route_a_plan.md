# Plan: Route A — Energy Evaluator + Energy Shielding

## Context

Route A 的目标：在现有 1-Step + 33 Anchors 基座上，添加白盒能量头（E_collision, E_offroad, E_target），训练时多任务监督，推理时做 Energy Shielding 重排序轨迹。

**当前状态：**
- `energy-evaluator` 分支与 `vla_adapter` 相同（无新改动）
- 模型里已有 energy heads 结构（`energy_heads=True` 触发），但未在 Route A policy 中启用
- Dataset 已能生成 `behavior_labels` 和 `allowed_flags`
- Route B policy (`annealed_energy_guidance_policy.py`) 已有能量 loss 参考实现（代码框架已完成）
- 当前 policy 有 `allowed_pred_head` + `behavior_pred_head`，但这是黑盒分类，不是白盒能量

## 修改文件

### 1. `policy/diffusion_dit_carla_policy.py`（主要改动）

**a) `__init__` — 启用 energy heads（~line 65-111）**
- 读取 config 中 `energy_shielding` 配置段
- 传 `energy_heads=True` 给模型构造函数
- 读取推理权重 `w_collision`, `w_offroad`, `w_target` 和 `energy_loss_weight`

**b) `compute_loss` — 添加能量 loss（~line 675, 718-737）**
- 模型 forward 解包 5 个返回值（加 `energy_scores`）
- 从 `behavior_labels` 派生能量监督目标（参考 Route B 实现）：
  ```python
  collision_target = ((behavior_labels >= 1) & (behavior_labels <= 4)).float()
  offroad_target = ((behavior_labels >= 5) & (behavior_labels <= 6)).float()
  target_proxy = 1.0 - allowed_flags
  ```
- 计算能量 loss：BCE(collision) + BCE(offroad) + L1(target)
- 加入 total_loss：`total_loss += energy_loss_weight * energy_loss`
- 保留现有 allowed/behavior loss（兼容，可通过 config 设为 0 关闭）

**c) `conditional_sample` — Energy Shielding 推理（~line 948-960）**
- 模型 forward 解包 5 个返回值
- 新增 energy shielding 模式选择：
  ```python
  safe_logits = poses_cls - w_col * E_collision - w_off * E_offroad - w_nav * E_target
  best_mode_idx = argmax(safe_logits)
  ```
- 优先级：energy_shielding > semantic_behavior filtering > plain argmax

### 2. `config/pdm_local_route_a.yaml`（新建，从 pdm_local.yaml 复制）

添加配置段：
```yaml
energy_shielding:
  enabled: true
  energy_loss_weight: 1.0
  w_collision: 5.0
  w_offroad: 3.0
  w_target: 1.0

semantic_behavior:
  enabled: true   # 必须开启，提供 behavior_labels 给 energy heads
```

### 3. `training/train_carla_bev.py`（小改动）

- checkpoint 加载加 `strict=False`（新增 energy heads 参数时兼容旧 ckpt）
- wandb logging 加 energy loss 子项

## 不需要改的文件

- `model/transformer_for_diffusion_multi_head.py` — energy heads 已实现，只需传 `energy_heads=True`
- `dataset/unified_carla_dataset.py` — 已能生成 behavior_labels/allowed_flags
- `tools/anchor_semantic_labeler.py` — 标签生成逻辑不变

## 验证方式

1. 小数据集训练，确认 energy loss 下降
2. 检查推理时 energy scores 分布是否合理（collision/offroad 应在 [0,1]）
3. 对比有无 energy shielding 的轨迹选择差异
4. L2 error 指标不应退化
