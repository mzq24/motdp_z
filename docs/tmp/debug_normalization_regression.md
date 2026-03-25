# Debug: Delta Z-Score 归一化导致 L2 回归

> 日期：2026-03-23  
> 状态：已定位根因  
> 影响：Route B L2_1s 从 0.29 退化到 0.71

---

## 0. 一句话结论

**未提交的代码改动**把归一化方法从 `norm_odo`（绝对坐标线性映射到 [-1,1]）  
换成了 `abs_to_norm`（绝对坐标 → 差分 → z-score），导致 L2_1s 从 0.29 退化到 0.71。

---

## 1. 异常现象

300 epoch 训练完成后，指标如下：

| 指标 | 当前运行 (0322) | 历史最佳 (0320) |
|------|----------------|----------------|
| L2_1s | **0.71** | **0.29** |
| L2_2s | 1.65 | 0.58 |
| L2_3s | 2.68 | 0.96 |
| L2_avg | 1.54 | 0.61 |
| 最佳 epoch | 24 (!) | 185 |

当前运行在 epoch 24 就达到最佳，之后一直退化 —— 从 epoch 141 到 300，L2_1s 始终卡在 0.70 附近。

---

## 2. 排查过程

### 第一步：对比所有 wandb 历史运行

扫描 `wandb/*/files/wandb-summary.json` 中所有 `val/L2_1s` 指标，按升序排列：

```
L2_1s   运行 ID                                   类型
0.240   run-20260306_195526-qkb4rawj              Route A (pdm_local.yaml)
0.274   run-20260311_190811-rmom9tpt              Route A
0.274   run-20260312_002536-8pyea292              Route A
0.290   run-20260306_221940-rdgcapm8              Route A
0.293   offline-run-20260320_032916-no7ugwgh      ← Route B!
0.307   run-20260309_194108-v4crhld0              Route A
...
0.727   run-20260304_221905-bu7acygz              Route A (旧 config)
```

**关键发现**：`offline-run-20260320` 是一个 Route B 运行（有 `alignment_loss`、`energy_col/off/tgt_loss`），L2_1s = 0.29。同样的架构，同样的数据集，为什么当前运行只有 0.71？

### 第二步：确认好运行是 Route B

检查 `wandb-summary.json`，确认该运行有 Route B 特有的指标：

```json
{
  "train/alignment_loss": 1.648,
  "train/energy_col_loss": 0.095,
  "train/energy_off_loss": 0.046,
  "train/energy_tgt_loss": 0.056,
  "train/energy_loss": 0.198
}
```

output.log 也确认:
```
Policy: AnnealedEnergyGuidancePolicy (Route B+ - anchor-free)
✓ Dual optimizers: decoder (274 param groups) + energy (12 param groups)
```

→ 确认是 Route B+，双优化器，和当前代码架构一致。

### 第三步：定位代码版本差异

好运行的 config 显示它使用了临时配置文件：
```
args: ['--config_path', '/tmp/route_b_local_1773948553.yaml']
```
该文件已不存在。运行开始时间：2026-03-20 03:29。

查看 git 提交历史：
```
fa57c2f  2026-03-17  add HPC config...
b061468  2026-03-20 15:51  Route B+ energy guidance: dual optimizer...
a600acd  2026-03-20 16:52  add alignment loss warmup...
489e5b9  2026-03-22 15:12  single-mode diffusion (M=1)...
3e7b7ba  2026-03-22 17:28  split route_b_refactor.md (HEAD)
```

好运行在 03:29 开始，b061468 在 15:51 才提交。  
但 output.log 里出现了 "Dual optimizers" —— 说明代码在工作树中已改但尚未提交。

→ 好运行使用的代码 = HEAD(489e5b9) 之前的某个工作树状态。

### 第四步：对比已提交代码 vs 当前工作树

```bash
git diff HEAD -- policy/annealed_energy_guidance_policy.py | wc -l
# 611 行未提交改动！
```

在这 611 行 diff 中，最关键的变化是**归一化方法的替换**：

**已提交代码 (HEAD = 489e5b9)** —— 好运行用的：
```python
def norm_odo(self, odo):
    x = 2 * (x + self.norm_x_offset) / self.norm_x_range - 1
    y = 2 * (y + self.norm_y_offset) / self.norm_y_range - 1
    return torch.cat([x, y], dim=-1)

def denorm_odo(self, odo):
    x = (x + 1) / 2 * self.norm_x_range - self.norm_x_offset
    y = (y + 1) / 2 * self.norm_y_range - self.norm_y_offset
    return torch.cat([x, y], dim=-1)
```

**未提交改动（当前工作树）** —— 当前运行用的：
```python
def abs_to_norm(self, abs_traj):
    return self.z_norm(self.abs_to_delta(abs_traj))
    # abs → delta差分 → z-score标准化

def norm_to_abs(self, z):
    return self.delta_to_abs(self.z_denorm(z))
    # z反标准化 → delta → cumsum累加回abs
```

### 第五步：验证因果关系

| 检查项 | 好运行 (0.29) | 坏运行 (0.71) |
|--------|-------------|-------------|
| 归一化 | `norm_odo` (绝对坐标 → [-1,1]) | `abs_to_norm` (delta z-score) |
| batch_size | 128 | 128 |
| lr | 5e-5 | 5e-5 |
| train_max_timesteps | 1000 | 1000 |
| 双优化器 | ✓ | ✓ |
| alignment_loss | ✓ | ✓ |
| 34-mode unified | ✓ | ✓ |
| 数据集 | pdm_lite_mini | pdm_lite_mini |

**唯一显著差异就是归一化方法。**

---

## 3. 为什么 Delta Z-Score 归一化更差

### 3.1 误差累积

`norm_to_abs` 的逆过程是 `z_denorm → cumsum`：

```
z = [z₁, z₂, z₃, z₄, z₅, z₆]      ← 模型预测的 z-score delta
δ = z * std + mean                    ← 反标准化得到 delta
p = cumsum(δ)                          ← 累加得到绝对坐标
  = [δ₁, δ₁+δ₂, δ₁+δ₂+δ₃, ...]
```

如果 z₁ 预测有误差 ε，则：
- p₁ 误差 = ε·std₁
- p₂ 误差 = ε·std₁ + ...
- p₆ 误差 = 所有前面误差之和

**而 `norm_odo` 方法**，每个时间步的坐标是独立映射到 [-1,1] 的，不存在累积效应。

### 3.2 非均匀目标分布

Z-score 标准化后，每个时间步有不同的 mean 和 std：
- 第 1 步 delta 很小（刚起步）
- 最后一步 delta 可能很大（高速行驶）

这导致扩散目标分布不均匀，模型需要学习更复杂的条件分布。

### 3.3 与 N(0,I) 噪声的不匹配

Route B 从纯高斯噪声 N(0,I) 开始去噪：
- `norm_odo` 把坐标映射到 [-1,1]，自然地约束了信号范围，和单位高斯噪声尺度匹配
- delta z-score 后的值是无界的，且分布不是标准正态，和 N(0,I) 噪声的尺度可能不匹配

---

## 4. 两种归一化方法的对比图

```
方法 A: norm_odo（绝对坐标归一化）— Route A 和好的 Route B 都用这个
┌────────────────────────────────────────────────────────┐
│  abs coords               normed coords                │
│  (0,0)──(5,0)──(12,1)    (-0.875,-0.286)──...──...    │
│                                                        │
│  映射: x_norm = 2*(x + offset) / range - 1            │
│  逆映射: x = (x_norm + 1) / 2 * range - offset        │
│                                                        │
│  ✓ 每个时间步独立映射                                    │
│  ✓ 值域 [-1, 1]，和高斯噪声匹配                         │
│  ✓ 逆映射无误差累积                                      │
└────────────────────────────────────────────────────────┘

方法 B: abs_to_norm（delta z-score）— 当前代码
┌────────────────────────────────────────────────────────┐
│  abs coords → delta → z-score                          │
│  (0,0)──(5,0)──(12,1)                                 │
│    ↓ abs_to_delta                                      │
│  (0,0)──(5,0)──(7,1)     ← 相邻差分                    │
│    ↓ z_norm                                            │
│  (z₁)──(z₂)──(z₃)        ← 标准化: (δ-μ)/σ           │
│                                                        │
│  逆: z_denorm → cumsum                                  │
│  ✗ cumsum 导致误差级联放大                               │
│  ✗ 值域无界，和 N(0,I) 不严格匹配                       │
└────────────────────────────────────────────────────────┘
```

---

## 5. 修复方案

**把归一化方法从 delta z-score 改回 `norm_odo`/`denorm_odo`。**

需要改动的地方：
1. `policy/annealed_energy_guidance_policy.py`：恢复 `norm_odo`/`denorm_odo`，删除 delta z-score 相关代码
2. `config/pdm_local_route_b.yaml`：移除 `delta_stats_path`，添加 `norm_x_offset/range`、`norm_y_offset/range`
3. 确保训练和推理都使用 `norm_odo`/`denorm_odo`

---

## 6. 调查时间线

```
1. 观察到 L2_1s = 0.71，用户期望 0.3
2. 扫描所有 wandb 运行，发现有 Route B 运行达到 0.29
3. 对比两次运行的配置 → batch_size, lr, 架构完全一致
4. 检查 git 历史，发现好运行用的代码版本
5. 发现 611 行未提交改动 → 归一化方法被替换
6. 排除其他差异 → 唯一显著变化是归一化
7. 分析 delta z-score 为什么更差 → 误差累积 + 非均匀分布
```
