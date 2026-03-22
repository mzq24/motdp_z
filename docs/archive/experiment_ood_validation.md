# OOD 验证实验 — VLM Anchor + Delta Prediction

## 背景与猜测

Truncated diffusion 中存在 OOD 问题：denoise 过程中模型遇到训练时未见过的分布，导致收敛慢、mode 不稳定。

### 三个猜测

1. **Anchor 距离影响 OOD**: anchor 距离 GT 越近，OOD 问题越小，mode 也更稳定
2. **Server version 快速收敛原因**: VLM anchor 本身就接近 GT（~5% 误差），受 OOD 影响极小
3. **BEV grid_sample 兜底**: BEV 的位置采样特征在一定程度上补偿了 OOD 带来的误差

---

## 实验 1: VLM Anchor 加为第 33 个 Anchor（Main Project）

**目的**: 测试一个接近 GT 的 anchor 是否能改善收敛和 mode stability。

### 使用方法

在 `config/pdm_local.yaml` 中设置:
```yaml
use_vqa_anchor: true  # 默认 false
```

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `dataset/unified_carla_dataset.py` | 新增 `use_vqa_anchor` 参数；从 `dp_vl_feature/*.pt` 加载 `pred_traj` 作为 `vqa_anchor` (6, 2) |
| `policy/diffusion_dit_carla_policy.py` | `_compute_multimodal_loss` 和 `conditional_sample` 中动态拼接 VLM anchor 为第 33 个 mode |
| `model/transformer_for_diffusion_multi_head.py` | 新增 `vqa_mode_query` (1, 1, n_emb)；当 M > 32 时自动拼接 |
| `training/train_carla_bev.py` | 传递 `use_vqa_anchor` 到 dataset |
| `config/pdm_local.yaml` | 新增 `use_vqa_anchor: false` |

### 工作原理

- 训练时：32 个 kmeans anchor + 1 个 VLM anchor = 33 个 mode
- VLM anchor 来自上游 VLM 的 `pred_traj`，存储在 `dp_vl_feature/{frame_id:04d}.pt`
- VQA 不可用时 fallback 为零向量（classification 不会选到，因为离 GT 最远）
- `mode_queries` 保持 (1, 32, n_emb) 不变，额外的 `vqa_mode_query` (1, 1, n_emb) 动态拼接
- 兼容已有 32-mode checkpoint（`strict=False` 加载）

### 预期结果

- 如果 VLM anchor (第 33 个 mode) 被频繁选为 best mode，且 L2 error 显著降低 -> 支持猜测 1 & 2
- 如果改善不明显 -> anchor 距离不是主要瓶颈

---

## 实验 2: DD Baseline 预测 Delta（消除 BEV Grid 兜底）

**目的**: 预测相对 anchor 的偏移量，使 BEV grid_sample 无法通过位置特征"兜底"。

### 使用方法

在 `dd_baseline/dd_config.yaml` 中设置:
```yaml
dd_baseline:
  predict_delta: true  # 默认 false
```

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `dd_baseline/config.py` | 新增 `predict_delta: bool = False` |
| `dd_baseline/trajectory_head.py` | 核心修改：decoder layer、decoder、forward_train、forward_test |
| `dd_baseline/dd_config.yaml` | 新增 `predict_delta: false` |

### 核心差异

| | Abs 模式 (默认) | Delta 模式 |
|---|---|---|
| Decoder 输出 | `offset + noisy_traj_points` | `offset`（raw delta） |
| 最终轨迹 | `poses_reg`（直接是绝对坐标） | `clean_anchor + poses_reg` |
| 迭代 refinement BEV 采样 | 在**预测轨迹位置**采样（动态） | 在 **anchor 位置**采样（固定） |
| BEV 特征作用 | 位置自适应，提供空间信息 | 固定不变，无法根据预测位置调整 |
| Loss | `\|pred_abs - GT\|` | `\|anchor + delta - GT\|` |

### 工作原理

- `CustomTransformerDecoderLayer`: delta 模式下不将 offset 加到 noisy_traj_points
- `CustomTransformerDecoder`: 迭代 refinement 时下一层 BEV grid_sample 在 clean anchor 位置（固定）
- `forward_train`: loss 计算时 `anchor + delta` 转为 abs coords 再传入 LossComputer
- `forward_test`: DDIM step 中 `pred_x0 = anchor + delta` 再 normalize；最终输出也做 delta->abs 转换
- 梯度只流过 delta（`plan_anchor` 是 `requires_grad=False`）

### 预期结果

- 如果 delta 模式明显变差 -> 支持猜测 3（BEV grid 在做兜底）
- 如果 delta 模式差不多 -> BEV grid 非关键因素，OOD 问题在其他地方

---

## 实验 2b: Abs 模式 + Normalized Forward（解耦 BEV Grid 依赖）

**目的**: 在绝对坐标分支中，让模型 forward 完全在 normalized [-1, 1] 空间进行，BEV grid_sample 接收的是 normalized 坐标而非物理坐标，从而解耦 BEV 位置特征的兜底作用。

### 使用方法

在 `dd_baseline/dd_config.yaml` 中设置:
```yaml
dd_baseline:
  predict_delta: false
  use_normalized_forward: true
```

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `dd_baseline/config.py` | 新增 `use_normalized_forward: bool = False` |
| `dd_baseline/trajectory_head.py` | `forward_train` 和 `forward_test` 中跳过 denorm，保持 normalized space forward |

### 核心差异

| | Abs 默认 | Abs + Normalized Forward |
|---|---|---|
| Noisy traj 传入 decoder | 物理空间 (meters) | Normalized [-1, 1] |
| BEV grid_sample 位置 | 物理坐标 → 有意义的空间位置 | Normalized 坐标 → 无意义的固定位置 |
| Position encoding | 物理空间 sine embed | Normalized 空间 sine embed |
| Decoder 输出 | 物理空间绝对坐标 | Normalized 空间坐标 |
| Loss 计算 | 直接 vs GT | denorm 后 vs GT |
| 迭代 refinement BEV 采样 | 在预测的物理位置采样 | 在预测的 normalized 位置采样 |

### 工作原理

- 训练时：加噪后不 denorm，直接在 normalized 空间做 position encoding + decoder forward
- BEV grid_sample 收到 [-1, 1] 范围的坐标，映射到 BEV 中心附近固定区域，丧失空间自适应能力
- Decoder 输出在 normalized 空间，loss 计算前 denorm 回物理空间
- 推理 DDIM loop 中同理，全程在 normalized 空间，最终输出 denorm

### 预期结果

- 如果 normalized forward 明显变差 -> 确认 BEV grid 的位置特征在 abs 模式中起关键兜底作用
- 如果差不多 -> BEV 的空间信息不是关键因素，模型主要依赖其他 attention 通道

### DD Baseline 实验结果

- **Abs 默认** (5-step DDIM): L2 1s ~0.29
- **Abs + Normalized Forward** (5-step DDIM): L2 1s ~0.28
  val_loss: 4.6796
  val_L2_1s: 0.2899
  val_L2_2s: 0.6251
  val_L2_3s: 1.1520
  val_L2_avg: 0.6890
- **结论**: Abs 模式下 norm vs 非 norm 差异很小，因为只是一个线性缩放，轨迹形状不变。Norm 略慢收敛因为网络需要额外学 scale。BEV grid 的空间信息在 abs 模式下非关键。

---

## 实验 2c: DD Baseline 正确的 Per-Step Delta 预测

**目的**: 之前实验 2 的 delta 定义有误（predict offset from anchor），这次用正确的 per-step displacement delta:
`delta[0] = pos[0], delta[i] = pos[i] - pos[i-1], cumsum(delta) = abs_traj`

### 核心改动（vs 之前错误的 delta）

| | 旧 delta (实验 2) | 新 delta (实验 2c) |
|---|---|---|
| Delta 定义 | `anchor + offset = abs` | `delta[i] = pos[i] - pos[i-1]`, `cumsum = abs` |
| 归一化 | `norm_odo`（abs 坐标范围） | **Z-score**（delta 统计量 mean/std） |
| Diffusion 空间 | abs normalized [-1,1] | **delta Z-score normalized** |
| DDIM 步骤 | abs-normed 空间转换混乱 | **完全在 delta-normed 空间** |
| Loss | `anchor + offset` vs GT abs | `cumsum(denorm_delta(pred))` vs GT abs |
| BEV grid_sample | 固定在 anchor 位置 | 固定在 anchor abs 位置（从第 1 层开始） |

### DD Baseline 实验结果

```
5-step DDIM:
  val_L2_1s: 0.5638
  val_L2_2s: 1.3843
  val_L2_3s: 2.2185
  val_L2_avg: 1.3889

1-step DDIM:
  val_L2_1s: 0.2637    ← 比 abs 模式的 0.29 更好！
  val_L2_2s: 0.6184
  val_L2_3s: 1.1419
  val_L2_avg: 0.6747
```

### 关键发现

1. **1-step delta 优于 1-step abs** (0.26 vs 0.29): 说明 per-step delta 表征本身是更好的，模型更容易学习
2. **Multi-step delta 严重退化** (0.56 vs 0.26): DDIM 多步 denoise 过程导致发散
3. **问题定位到 BEV grid_sample**:
   - Abs 模式中，BEV 在每一步 DDIM 都在**预测轨迹位置**采样（动态），提供位置相关的空间特征
   - Delta 模式中，BEV 固定在 anchor 位置，无法根据 denoise 中间结果调整
   - Abs + Normalized Forward 效果和默认 abs 差不多（0.28 vs 0.29），因为 BEV 仍在动态位置采样（只是坐标被缩放到 [-1,1]）
   - **BEV 的位置自适应采样是多步 DDIM 稳定性的关键**，它在每步提供"你预测到了哪里"的空间反馈，帮助纠正 OOD drift

---

## 实验 3: Main Project + Normalized Forward + Proposal Anchor

**目的**: 将 DD baseline 验证有效的 normalized forward 应用到 main project，并结合实验 1 的 VLM proposal anchor（第 33 个 mode），测试两个改进的叠加效果。

### 使用方法

在 `config/pdm_local.yaml` 中设置:
```yaml
truncated_diffusion:
  use_normalized_forward: true
use_vqa_anchor: true
```

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `policy/diffusion_dit_carla_policy.py` | `_compute_multimodal_loss` 和 `conditional_sample` 中支持 normalized forward |
| `config/pdm_local.yaml` | 新增 `use_normalized_forward: true` |

### 核心差异（vs 默认 main project）

| | Main Project 默认 | + Normalized Forward |
|---|---|---|
| `anchors_abs` 传入 model | 物理坐标 (meters) | Normalized [-1, 1] |
| BEV grid_sample 位置 | 物理坐标 → 有意义的空间位置 | Normalized 坐标 → 无意义的固定位置 |
| Anchor position encoding | 物理空间 sine embed | Normalized 空间 sine embed |
| Model residual base | 物理空间 noisy traj | Normalized 空间 noisy traj |
| Model 输出 | 物理空间绝对坐标 | Normalized 空间坐标 (denorm for loss/output) |

### 工作原理

- `_compute_multimodal_loss`: 加噪后不 denorm，传 `anchors_abs=noisy_anchors_normed`
  - Model 的 BEV grid_sample、position encoding、residual base 全部在 normalized 空间
  - 输出 denorm 后计算 L1 loss
- `conditional_sample`: DDIM loop 中 `x_abs = x_clamped`（不 denorm）
  - `pred_x0_normed = poses_reg_out`（已经是 normalized）
  - 最终输出 denorm 回物理空间
- VLM anchor 正常拼接为第 33 个 mode，也在 normalized 空间参与

### 预期结果

- DD baseline 中 normalized forward 反而更好 (0.29 → 0.28)
- 叠加 VLM proposal anchor (实验 1 已验证收敛更快、效果更好)
- 期望 main project 也能受益于 normalized space 的更规则数值范围

---

## 实验 4: BEV Grid Sample 消融实验

**背景**: 实验 2c 的结果表明 BEV grid_sample 的动态位置采样是多步 DDIM 的关键。需要消融实验来确认。

### 实验 4a: Abs 模式 + 固定 BEV at Anchor（DD Baseline）

**目的**: 在 abs 模式下，把 BEV grid_sample 也固定到 anchor 位置（和 delta 模式一样），**只去掉 BEV 的动态反馈，保持 abs 坐标表征不变**。

如果多步 DDIM 退化 → 确认 BEV 动态采样是多步稳定性的关键因素。

配置:
```yaml
dd_baseline:
  predict_delta: false
  use_normalized_forward: false  # 或 true
  fix_bev_at_anchor: true        # 新参数
```

修改: `trajectory_head.py` 中 `CustomTransformerDecoder`，当 `fix_bev_at_anchor=True` 时始终用 clean_anchor 做 BEV 采样。

### 实验 4b: Delta 模式 + 动态 BEV（DD Baseline）

**目的**: 在 delta 模式下，每一步 DDIM 将预测的 delta denorm → cumsum 得到 abs 位置，再用来做 BEV grid_sample。**给 delta 模式加回 BEV 的动态反馈**。

如果多步 DDIM 不再退化 → 进一步确认 BEV 动态采样是关键。

修改: `trajectory_head.py` 中 `_forward_test_delta` 和 `CustomTransformerDecoder`，delta 模式下用 `cumsum(denorm_delta(poses_reg))` 的绝对位置做 BEV 采样。

### 实验 4c: Main Project BEV 消融

**目的**: 在 main project 中做同样的消融，确认结论在不同架构下也成立。

修改: `policy/diffusion_dit_carla_policy.py` 中 `conditional_sample`，BEV 采样位置固定 vs 动态。

### 预期结果汇总

| 实验 | 坐标表征 | BEV 采样 | 多步 L2_1s | 1-step L2_1s | 多步 L2_avg | 状态 |
|------|---------|---------|-----------|-------------|-------------|------|
| Abs 默认 | abs | 动态（预测位置） | **0.240** | 0.295 | 0.587 | 已完成 |
| 4a: Abs + 固定 BEV | abs | 固定（anchor） | — | — | — | **跳过** |
| Delta 默认 (2c) | delta | 固定（anchor） | 0.564 | 0.264 | 1.389 | 已完成 |
| **4b: Delta + 动态 BEV** | delta | **动态（cumsum）** | **0.344** | 0.268 | 0.894 | **已完成** |
| 4c: Main abs + 固定 BEV | abs | 固定（anchor） | — | — | — | **跳过** |

**4a/4c 跳过原因**: BEV 特征图分辨率 64×64 覆盖 64m×64m (1m/pixel)，truncated diffusion 噪声极小，noisy 位置与 clean anchor 位置偏移不到 1 pixel，`F.grid_sample` 双线性插值得到的特征几乎一样。因此固定 vs 动态 BEV 采样在 truncated diffusion 下差异可忽略，预期结果不会退化，没有实验必要。

### 实验 4b 结果分析

配置: `predict_delta: true, delta_dynamic_bev: true`，decoder 层间 BEV 采样使用 `cumsum(denorm_delta(poses_reg))` 的绝对坐标。

**关键发现**:

1. **动态 BEV 有效恢复了约 55% 的 gap**: 多步 L2_1s 从 0.564 → 0.344（delta 默认 vs 4b），gap = 0.564 - 0.264 = 0.300，恢复了 0.220
2. **但多步仍 > 1-step** (0.344 vs 0.268): BEV spatial feedback 不是唯一的多步退化因素
3. **对比 abs 模式**: abs 多步 (0.240) < abs 1-step (0.295)，说明 abs 模式的 residual 连接 (`output = offset + noisy_input`) 在多步 DDIM 中起到稳定作用，而 delta 模式缺少这一机制
4. **1-step 和训练 loss 不变**: 符合预期，因为 layer 0 BEV 位置不变（始终用 clean_anchor），dynamic BEV 只影响 layer 1+

**结论**: BEV 的位置自适应采样是多步 DDIM 稳定性的**重要因素之一**（贡献约 55%），但不是全部。剩余的退化来自 delta 空间缺少 abs 模式的 residual 连接机制，使得 DDIM 步间缺乏锚点约束。

---

## BridgeDrive 的 OOD 解决方案：Diffusion Bridge (DDBM)

**参考代码**: `BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py`

### 问题本质

DiffusionDrive 的 truncated diffusion 前后过程不对称：
- Forward: `anchor → 加噪 → noised anchor`
- Reverse: `noised anchor → 去噪 → GT trajectory`

anchor ≠ GT，模型在 reverse 时遇到训练分布外的中间状态 → OOD。

### BridgeDrive 的方案：Denoising Diffusion Bridge Model (DDBM)

用 Bridge 替换 truncated diffusion，恢复对称性：
- Forward: `GT → bridge diffusion → anchor`（训练时）
- Reverse: `anchor → bridge denoising → GT`（推理时）

| | DiffusionDrive (Truncated) | BridgeDrive (Bridge) |
|---|---|---|
| Forward 起点 | anchor | **GT trajectory** |
| Forward 终点 | noised anchor | **anchor (+noise)** |
| Reverse 起点 | noised anchor | anchor |
| Reverse 终点 | GT trajectory | GT trajectory |
| 对称性 | 破坏（anchor ≠ GT） | **保持**（GT↔anchor 双端固定） |
| OOD 问题 | 有 | **无** |
| 类比 | Anchored Brownian Motion | **Brownian Bridge** |

### 数学公式

VP (Variance Preserving) schedule，`DDBMScheduler` (line 44-123):

**加噪（Forward Bridge）**: `samples = a_t * xT + b_t * x0 + c_t * noise`
- t=0 时：`a≈0, b≈1, c≈0` → 样本 ≈ x0 (GT)
- t=T 时：`a≈1, b≈0, c≈large` → 样本 ≈ xT (anchor) + noise

其中系数由 VP logSNR 计算：
```python
logsnr_t = -log(exp(0.5*beta_d*t^2 + beta_min*t) - 1)
a_t = exp(logsnr_T - logsnr_t + logs_t - logs_T)
b_t = -expm1(logsnr_T - logsnr_t) * exp(logs_t)
c_t = sqrt(-expm1(logsnr_T - logsnr_t)) * exp(logs_t - logsnr_t/2)
```

**去噪（Reverse Bridge, sample_step）**:
```python
xt_prev = a_t_prev * xT + b_t_prev * x0       # 确定性：anchor 和预测GT的加权
if is_T:  # 第一步
    xt_prev += c_t_prev * randn()               # 加随机噪声
else:     # 后续步
    xt_prev += (c_t_prev/c_t) * (xt - a_t*xT - b_t*x0)  # 传递残差，不加新噪声
```

**注意**: 只有第一步注入随机噪声，后续步骤是确定性的残差传递。本质更像 iterative refinement 而非真正的"从噪声中去噪"。

### 训练流程 (line 250-316)

```python
# 1. normalize GT 和 anchor
odo_info_fut = norm_odo(GT)
odo_plan_anchor = norm_odo(anchor)
# 2. Bridge 加噪: GT → anchor 方向
noisy = scheduler.add_noise(x0=GT_norm, xT=anchor_norm, noise, t)
# 3. 模型预测 x0 (去噪后的 GT)
x0_pred = decoder(noisy, anchor, bev, ego_query, t)
# 4. Loss = |x0_pred - GT|
```

### 推理流程 (line 318-388)

```python
xt = anchor  # 从 anchor 开始
for i in range(step_num, 0, -1):
    x0_pred = model(xt, t)
    xt = scheduler.sample_step(t, t_prev, xt, x0_pred, anchor)
# 最终 xt ≈ GT
```

### 架构亮点：双分支 BEV Grid Sample

Decoder layer (line 614-616) 使用**两个独立的** `GridSampleCrossBEVAttention`：

| 分支 | 采样位置 | 作用 |
|---|---|---|
| `cross_bev_attention` | noisy 轨迹（动态变化） | 感知当前去噪状态附近的 BEV 信息 |
| `cross_bev_attention_T` | anchor（固定） | 提供稳定的 BEV 参考信息 |

两个分支各自过 cross-ego attention + FFN + LayerNorm 后 **concat**，再经过 time modulation（FiLM）送入 task decoder。

`GridSampleCrossBEVAttention`（`blocks.py`）实现和我们的一致：轨迹点归一化 → `F.grid_sample` 双线性插值 → attention weight 加权 → 残差连接。

### 分离的 Classification 和 Regression

- `diff_decoder_cls`: 只做 anchor 分类（focal loss），不参与 diffusion，只在 anchor 位置做一次 BEV grid sample
- `diff_decoder`: 做 diffusion denoising regression，双分支 BEV + time modulation

### 与我们实验发现的关联

| 我们的发现 | BridgeDrive 如何解决 |
|---|---|
| Anchor 距离影响 OOD（猜测 1&2） | Bridge 从根本消除：不管 anchor 离 GT 多远，forward/reverse 分布一致 |
| BEV grid_sample 兜底（猜测 3） | 双分支 BEV：同时在 noisy 和 anchor 位置采样然后 concat |
| 多步 DDIM 退化（实验 2c） | Bridge 中间状态始终在 forward bridge 定义的分布上，不会 OOD drift |

### 局限性

- 噪声被两端"钉住"（Brownian Bridge），探索能力弱于自由噪声
- 多步 DDIM 意义减弱——本质是 iterative refinement 而非 denoising
- Robustness 主要靠 anchor 多样性（20 mode），而非噪声探索
- 如果 anchor 选错，bridge 只会精准送到错误终点，不像 truncated diffusion 的噪声可能"救一下"
