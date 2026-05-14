# Mode-aware Semantic Planning：Paper 行文包装与实验修改计划

> 目标：把当前方法从“加了一些 state head / 跑了两个版本”整理成一条能成文的主线，同时把未来 3 周内可以做的修改和这条主线对齐。

---

## 0. 当前主线的收缩版

当前不要再讲成：

> imitation learning 不好，所以我们不做 imitation。

更稳的说法是：

> trajectory-only imitation 在 long-tail closed-loop 场景中不够稳定；我们仍然使用 imitation learning，但把 imitation 从 trajectory level 提升到 semantic mode level。

推荐主线：

```text
From trajectory-only imitation
→ to mode-aware imitation
→ to semantic-mode-conditioned diffusion planning.
```

也就是：

```text
scene
  → semantic mode distribution
  → selected / sampled semantic mode
  → mode-conditioned trajectory / route / speed
```

当前最稳的 claim：

```text
Explicit semantic mode helps diffusion planners learn better behavioral mode selection in long-tail closed-loop driving.
```

暂时不要 overclaim：

```text
We solve temporal consistency.
We solve trajectory-mode consistency.
We solve OOD generalization.
```

更稳的写法：

```text
The current method improves semantic mode prediction.
Temporal consistency and trajectory-mode consistency are natural next extensions.
```

---

# Part I. Paper 行文包装需要注意的点

## 1. 不要否定 imitation learning，而是改成 mode-aware imitation

### 风险说法

```text
Imitation learning has an upper bound.
We should not imitate expert trajectories.
```

这个容易被审稿人攻击，因为我们自己也是通过 expert label / expert trajectory 做监督。

### 推荐说法

```text
Direct trajectory imitation treats expert behavior as a single trajectory realization.
For long-tail closed-loop cases, this supervision is insufficient to expose the semantic mode behind expert behavior.
```

中文表述：

```text
直接模仿 expert trajectory 仍然有用，但它只给了模型一个单条轨迹 realization。
我们希望在 semantic mode 层面进行更结构化的 imitation，再生成与 mode 对齐的 trajectory。
```

可以总结为：

```text
We do not replace imitation learning; we make imitation learning mode-aware.
```

---

## 2. 多模态故事：从 trajectory-space multimodality 到 semantic-mode-space multimodality

现有很多方法的思路是：

```text
scene → sample trajectory candidates → score / rank → selected trajectory
```

我们的区别不应该只讲成“先打分 vs 后打分”，而应该强调 representation shift：

```text
Existing methods hide modes inside trajectory candidates.
We expose modes explicitly in semantic state space.
```

推荐表述：

```text
We move multimodality from trajectory space to semantic mode space.
```

中文：

```text
我们不是把多模态隐藏在多条候选轨迹里，而是把多模态显式建模到 semantic mode space 中。
```

当前实现可以是 MAP / greedy mode：

```text
semantic mode distribution → most likely mode → mode-conditioned trajectory
```

未来可以扩展为 sampling：

```text
semantic mode distribution → sampled semantic modes → mode-aligned multi-modal trajectories
```

关键解释：

```text
Multimodal modeling does not require executing all modes.
It requires preserving mode uncertainty before the final executable decision.
```

---

## 3. 把 temporary occupancy 包装成“机会模态 / 可通行分布”

现在如果说：

```text
window → conflict area → temporary occupancy → phase
```

可能会显得 window 和 phase 是 mode，但 occupancy 不像 mode。可以改成三层 semantic mode chain：

```text
interaction mode → opportunity mode → action mode
```

对应关系：

```text
interaction mode: window / interaction type
spatial grounding: conflict area
opportunity mode: passability / go-probability distribution around the conflict area
action mode: phase, e.g., yield / go
```

推荐说法：

```text
The middle state is not merely occupancy. It is an opportunity distribution that describes whether and when the conflict area is passable.
```

中文：

```text
中间的 temporary occupancy 可以包装成可通行机会模态，也就是围绕 conflict area 的 passability / go-probability distribution。
它描述的是这个空间冲突区在未来时间内是否可通行。
```

更统一的结构：

```text
window mode
  → conflict-area grounding
  → opportunity / passability distribution
  → action phase mode
  → mode-conditioned trajectory
```

---

## 4. 少用“decoupling”，多用“structured semantic factorization / decomposition”

“decouple” 容易被理解成严格物理解耦，例如 lateral / longitudinal、speed / route 等。我们现在做的更像是：

```text
把 expert trajectory 背后的 task-relevant semantic factors 结构化暴露出来。
```

建议少说：

```text
decouple dominant factors
```

改成：

```text
structure dominant semantic factors
semantic factorization
structured semantic mode chain
task-aligned semantic decomposition
```

推荐论文表述：

```text
We do not claim strict disentanglement.
Instead, we use a structured semantic factorization to expose task-relevant planning modes behind expert trajectories.
```

---

## 5. 少用 OOD，改用 long-tail

“OOD → ID” 的故事有吸引力，但风险较高。审稿人可能会要求证明 trajectory space 是 OOD，而 semantic mode space 是 in-distribution。

建议改成：

```text
long-tail trajectory patterns can correspond to familiar semantic modes.
```

中文：

```text
轨迹层面看起来是 long-tail 的场景，在 semantic mode 层面可能对应的是模型熟悉的交互模式，例如 gap available → go，gap occupied → yield。
```

稳妥表述：

```text
Semantic mode space is more compact and reusable than raw trajectory space.
```

---

## 6. Coarse-to-fine / structured reasoning 的包装

第二个主要创新点可以包装为：

```text
Coarse-to-fine semantic mode reasoning
```

或者：

```text
Structured semantic mode chain
```

推荐英文：

```text
We structure closed-loop planning as a coarse-to-fine semantic reasoning process:
from interaction mode, to spatially grounded opportunity, to action phase, and finally to mode-conditioned trajectory generation.
```

注意：不要说严格因果链。

推荐补充：

```text
This is not a strict causal graph, but a task-aligned factorization of correlated planning modes.
```

---

## 7. Multi-step denoising 的创新点：iterative semantic context refinement

Joint state 不应该只讲成“把 state 和 trajectory 一起预测”。更有价值的包装是：

```text
multi-step denoising provides iterative semantic context refinement.
```

对比：

```text
Independent state:
semantic context is predicted once and reused across DDIM steps.

Joint state:
semantic context and motion hypothesis can be refined together across DDIM steps.
```

推荐说法：

```text
Independent state implements one-shot semantic conditioning.
Joint semantic-state diffusion aims at iterative semantic context refinement through multi-step denoising.
```

边界：当前 joint 结果还没有 independent 好，所以不要说已经实现了完整能力。更稳：

```text
Joint state remains a promising direction, but current continuous joint formulation may not yet fully exploit this iterative refinement.
```

---

## 8. Mode–trajectory alignment 的说法

这种结构天然有一个 inductive bias：

```text
semantic mode → trajectory
```

所以 trajectory 不再是自由生成，而是 selected / sampled semantic mode 的 realization。

推荐说法：

```text
The semantic-to-trajectory pipeline naturally encourages mode–trajectory alignment.
```

但不要说已经严格保证：

```text
The current structure encourages alignment, but explicit consistency losses are still future work.
```

---

## 9. 当前 contribution slide 推荐写法

```text
Key Contributions

1. Mode-aware imitation for closed-loop planning
   We reformulate diffusion planning from trajectory-only imitation
   to semantic-mode-aware imitation.

2. Coarse-to-fine semantic mode chain
   We expose task-relevant planning modes through:
   interaction mode → opportunity mode → action mode → trajectory.
   The opportunity mode is grounded by conflict area and represented as a passability / go-probability distribution.

3. Two semantic-conditioned diffusion formulations
   Independent semantic conditioning performs one-shot mode-conditioned generation.
   Joint semantic-state diffusion aims at iterative semantic context refinement through multi-step denoising.

4. Evidence
   Current results show that independent conditioning learns window / phase more reliably
   and achieves stronger closed-loop performance.
```

如果 PPT 要压缩，可以保留前三条，把 evidence 放到 results 页里讲。

---

# Part II. 实际实验中可以怎么改

下面按照“是否和 paper story 对齐”以及“三周内可完成性”排序。

---

## A. 必做：统一命名、输出和指标，先让 story 对齐

### A1. 命名改造

建议把原有变量包装成：

```text
window → interaction mode
conflict area → spatial grounding / conflict-area grounding
temporary occupancy / go prob → opportunity distribution / passability distribution
phase → action mode
```

代码里可以先不大改变量名，但 log、metric、PPT、paper draft 中统一使用这些概念。

### A2. 指标组织

Results 不只放 closed-loop score，还要放：

```text
interaction mode recall / accuracy
opportunity prediction metric, e.g., go-prob / occupancy error / AUC / BCE
action mode recall / phase recall
closed-loop score
```

目标：证明 improvement 不是偶然轨迹采样，而是 semantic mode prediction 更好。

### A3. 增加 mode–trajectory alignment 的分析指标

可以先不加 loss，只做 metric：

```text
if predicted phase = go:
    trajectory progress through conflict area should be high

if predicted phase = yield:
    progress before conflict area should be limited
```

这个可以作为后续 consistency loss 的依据。

---

## B. 低风险实验：让训练目标更概率化 / 多模态化

你们现在 label 和 semantic mode 具有 soft / probabilistic 的形式，但 loss 可能仍然偏 deterministic。为了让故事闭环，可以做以下低风险修改。

### B1. 对 window / phase 使用 soft CE 或 KL loss

如果 transition label 不是 hard one-hot，而是有 0→1 的连续过渡，建议不要硬化成 one-hot。

形式：

```text
L_mode = KL(q(s|x) || p_theta(s|x))
```

其中：

```text
q(s|x): soft label distribution
p_theta(s|x): predicted semantic mode distribution
```

对应故事：

```text
semantic modes are learned as distributions, not hard-coded labels.
```

### B2. Opportunity / go-prob 使用概率 loss

如果 opportunity 是 0~1 的 go probability，可以继续用 soft BCE；如果是连续覆盖率，也可以尝试 Gaussian NLL。

低风险首选：

```text
soft BCE / weighted BCE
```

中风险：

```text
Gaussian NLL over opportunity values
```

### B3. Trajectory / speed 的 Gaussian NLL

师兄提到的 NLL 可以先做成一个可选实验。

单 Gaussian NLL：

```text
model predicts mean μ and log-variance log σ^2
L = 0.5 * (y - μ)^2 / σ^2 + 0.5 * log σ^2
```

注意：单 Gaussian NLL 更像 uncertainty-aware，不一定是真正 multi-modal。

更贴合故事的表述：

```text
from point-wise imitation loss to likelihood-based mode-aware imitation.
```

### B4. Mode-conditioned likelihood，更贴合最终故事

更完整但稍复杂：

```text
p(τ|x) = Σ_s p(s|x) p(τ|x,s)
L = -log Σ_s p_theta(s|x) p_theta(τ_gt|x,s)
```

三周内可以先不做完整 mixture，只做：

```text
L_traj = -log p_theta(τ_gt | x, s_gt)
```

或者：

```text
根据 predicted / GT semantic mode condition 轨迹 decoder，并用 NLL / L1 训练。
```

---

## C. Inference：从 greedy semantic mode 到 semantic mode sampling

当前如果直接 threshold / greedy，会显得最终还是 deterministic。为了让 multi-modal story 更完整，可以做一个轻量 inference ablation。

### C1. 当前版本：MAP semantic mode

```text
s* = argmax p(s|x)
trajectory = decoder(x, s*)
```

这是稳定主结果。

### C2. Sampling 版本：semantic mode sampling

```text
s_k ~ p(s|x)
trajectory_k = decoder(x, s_k)
```

然后可以有两种方式：

```text
1. 执行最高 mode probability 的 trajectory
2. 对 sampled trajectories 做简单 rule score / safety score
```

关键 storytelling：

```text
Each sampled trajectory has an explicit semantic mode explanation.
```

三周内建议：

```text
只做小规模 ablation，不作为主结果依赖。
```

---

## D. Consistency：从普通正则到 semantic transition learning

当前还没有显式 consistency。可以分三层做。

### D1. Frame-to-frame mode consistency

目标：抑制连续帧 yield/go 振荡。

形式：

```text
KL(p_phase^t || p_phase^{t+1})
```

但必须加 mask：

```text
only apply when GT phase is unchanged
or when opportunity distribution does not change significantly
```

不然会阻止合理 transition：

```text
yield → yield → go
```

### D2. Opportunity time-shift consistency

这个最贴合 temporary occupancy / opportunity distribution。

如果：

```text
O_t = [o_t^0, o_t^1, ..., o_t^K]
```

那么下一帧应该近似满足：

```text
O_{t+1}^k ≈ O_t^{k+1}
```

推荐作为后续最有特色的 consistency：

```text
time-shift consistency for opportunity distribution
```

### D3. State–trajectory consistency

目标：让轨迹符合 phase / opportunity。

例子：

```text
phase = go:
    encourage progress through conflict area

phase = yield:
    discourage early crossing of conflict area
```

可以先用 energy / auxiliary loss：

```text
L_align = p_go * ReLU(min_progress - progress_to_conflict_area)
        + p_yield * ReLU(progress_to_conflict_area - max_yield_progress)
```

这个和当前 story 很对齐：

```text
mode–trajectory alignment
```

---

## E. AR next-token prediction：后续增强 structured reasoning 与 temporal consistency

AR 可以分成两个方向。

### E1. Intra-frame AR：模态内部结构推理

目标：加强 single-frame 内部的 structured reasoning chain。

形式：

```text
p(s_t | x_t)
= p(w_t | x_t)
  p(o_t | x_t, w_t)
  p(φ_t | x_t, w_t, o_t)
```

如果包含 conflict area：

```text
p(s_t | x_t)
= p(w_t | x_t)
  p(A_t | x_t, w_t)
  p(O_t | x_t, w_t, A_t)
  p(φ_t | x_t, w_t, A_t, O_t)
```

可以包装为：

```text
structured semantic next-token prediction
```

或者：

```text
CoT-like structured semantic reasoning
```

注意不要说成 LLM free-form CoT，而是 task-designed semantic chain。

### E2. Inter-frame AR：帧之间时序相关性

目标：学习 semantic state 的合理时间演化，不只是平滑。

形式：

```text
p(s_{t+1} | s_{≤t}, x_{≤t})
```

作用：

```text
occupancy should evolve continuously
phase transition should be meaningful
yield → go is critical, but yield → go → yield is often unstable
```

推荐说法：

```text
Consistency is not only a regularization term;
it can be formulated as next-token prediction over semantic states.
```

### E3. 三周内可行性

如果 dataloader 已经能输出连续帧 pair，可以先做 inter-frame AR 的小版本：

```text
previous semantic feature / logits + current scene feature → next semantic logits
```

否则先把 AR 放到 paper discussion / next step，不要影响当前主线。

---

## F. Joint state 的后续修改

当前 joint state 的问题可能不是方向错，而是形式还不适合。

### F1. mixed continuous-discrete modeling

问题：

```text
trajectory / speed: continuous
window / phase: categorical or semi-discrete
opportunity: probability / continuous
```

不要把所有东西都简单放进 continuous Gaussian diffusion。

后续方向：

```text
continuous diffusion for motion
discrete / categorical diffusion for semantic mode
shared transformer with different noise protocols
```

这个三周内风险高，适合写 future / next step。

### F2. Two-stage DDIM denoising

更符合 reasoning：

```text
early DDIM steps: refine semantic mode / opportunity
late DDIM steps: generate trajectory conditioned on refined semantic mode
```

这个和 joint state 的“multi-step semantic context refinement”非常对齐。

三周内如果要做，建议只做最简单 ablation：

```text
前 K/3 步降低 trajectory 更新权重，强化 state prediction
后 2K/3 步正常联合 denoise
```

### F3. State 与 trajectory 的独立噪声调度

适合低到中风险实验：

```text
state tokens use a smoother schedule
trajectory tokens keep current schedule
```

动机：state token 不应该在高噪声阶段完全失去判别信息，否则无法指导轨迹。

---

# Part III. 三周优先级建议

## Week 1：先对齐故事和现有结果

必须完成：

```text
1. PPT / paper wording 改成 mode-aware imitation，而不是 anti-imitation。
2. 把 temporary occupancy 包装成 opportunity / passability distribution。
3. results 增加 semantic metrics：window recall、phase recall、opportunity metric。
4. 增加 mode–trajectory alignment 的分析 metric，不一定加 loss。
5. 整理 independent vs joint 的结构图和说法。
```

低风险代码：

```text
soft CE / KL for soft mode labels
soft BCE / weighted BCE for opportunity
```

## Week 2：做最能支撑 story 的实验

优先做：

```text
1. phase / opportunity → trajectory consistency energy or loss
2. semantic mode sampling inference ablation
3. opportunity time-shift consistency if dataloader supports consecutive frames
4. Gaussian NLL for speed / opportunity or trajectory head as optional ablation
```

目标：让 story 从“mode prediction 更好”推进到：

```text
mode prediction can better guide trajectory generation.
```

## Week 3：稳定结果 + 写作

```text
1. 固定 independent state 主结果。
2. joint state 作为 promising but currently unstable / harder formulation。
3. AR next-token prediction、mixed discrete-continuous diffusion 可以写成 future extension 或 discussion。
4. 如果 sampling / NLL / consistency 没有稳定提升，不作为主方法，只作为 analysis / ablation。
```

---

# Part IV. 最终推荐实验优先级表

| 优先级 | 修改 | 对应 story | 风险 | 建议 |
|---|---|---|---|---|
| ★★★ | 统一术语：mode-aware imitation / opportunity distribution | 行文主线 | 低 | 必做 |
| ★★★ | semantic metrics + alignment metric | 证明 mode reasoning | 低 | 必做 |
| ★★★ | soft CE / KL for mode labels | label/loss 概率化 | 低 | 优先 |
| ★★★ | phase-opportunity → trajectory consistency energy | mode–trajectory alignment | 中低 | 强烈建议 |
| ★★☆ | semantic mode sampling inference | 完整多模态 story | 中 | 小规模 ablation |
| ★★☆ | opportunity time-shift consistency | temporal semantic consistency | 中 | 取决于 dataloader |
| ★★☆ | Gaussian NLL / uncertainty prediction | likelihood-based imitation | 中 | 可选 ablation |
| ★★☆ | AR inter-frame next-token prediction | learnable consistency | 中高 | 有时间再做 |
| ★☆☆ | AR intra-frame structured prediction | structured CoT-like reasoning | 中高 | 更适合下一版 |
| ★☆☆ | mixed discrete-continuous diffusion | joint state 根本改进 | 高 | future / discussion |
| ★☆☆ | two-stage DDIM joint denoising | iterative semantic refinement | 高 | 视时间尝试 |

---

# Part V. 当前最稳的 paper story

```text
1. Direct trajectory imitation is useful but treats expert behavior as a single trajectory realization.

2. Long-tail closed-loop planning requires reasoning over semantic modes behind expert behavior.

3. We move multimodality from trajectory candidate space to semantic mode space.

4. We propose a structured semantic mode chain:
   interaction mode → opportunity mode → action mode.

5. The opportunity mode is grounded by conflict area and represented as a passability / go-probability distribution.

6. We propose two semantic-conditioned diffusion formulations:
   independent semantic conditioning and joint semantic-state diffusion.

7. Current results show that independent conditioning learns semantic modes more reliably and improves closed-loop performance.

8. Future extensions include semantic mode sampling, likelihood-based training, temporal AR consistency, and mixed discrete-continuous joint diffusion.
```

---

## 一句话总结

```text
We do not abandon imitation learning;
we make it mode-aware by moving multimodality from trajectory space to semantic mode space,
and use semantic-conditioned diffusion to generate mode-aligned trajectories.
```
