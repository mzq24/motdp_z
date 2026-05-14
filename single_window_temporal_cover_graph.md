# Single-Window Temporal Cover Graph 方案记录

## 1. 背景与问题

我们当前希望在 diffusion planner 之前显式建模语义 state，尤其是与 **window、current cover、future cover、ego、phase** 相关的状态。核心目标不是构建一个完整的 traffic scene graph，也不是像 LLM/VLM 方法那样把 scene graph serialize 成 prompt，而是设计一个更轻量、更贴近 planning 的 **single-window temporal cover graph**。

GraphPilot 这类工作主要属于 LLM/VLM driving 范式：将 traffic scene graph 以 Text / JSON / YAML 的形式放入 prompt 中，帮助语言模型学习 relational grounding。我们的方向不同：我们是 non-language diffusion planner，graph 不是语言 prompt，而是 planner-side structured state / relational bottleneck。我们的 graph 只服务于一个关键问题：**当前唯一 window 在 current/future cover 下，ego 应该何时 go / yield。**

因此，我们不需要 full scene graph，也不需要多 window 的复杂连接。我们当前只有一个 window，graph 的设计可以非常简洁。

---

## 2. 符号定义

因为只有一个 window，所以不需要额外的 window index。只保留时间维度：

```text
t = 当前帧 / 当前时刻
k = future cover 的第 k 个未来时间 bin
```

当前时刻的语义 state 可以写成：

```text
S_t = {
  ego_t,
  window_t,
  current_cover_t,
  future_cover_{t,1:K},
  phase_t
}
```

其中：

```text
ego_t:
  当前 ego 状态，例如速度、heading、到 window 的相对位置、预计到达时间等。

window_t:
  当前唯一候选 window 的属性，例如 window type、几何位置、可通过区域等。

current_cover_t:
  当前时刻这个 window 是否被占据 / 阻塞。

future_cover_{t,k}:
  当前 t 帧预测：这个 window 在未来第 k 个时间 bin 是否会被占据。

phase_t:
  当前时刻针对这个 window 的 go / yield phase。
```

如果 `K=3`，可以理解为：

```text
future_cover_{t,1}: 未来 0.5s 是否 cover
future_cover_{t,2}: 未来 1.0s 是否 cover
future_cover_{t,3}: 未来 1.5s 是否 cover
```

---

## 3. 两种 next-token / state prediction 思路

我们目前实际上只有两种 state prediction 分支，而不是三种。因为无论是 temporal next-token 还是 graph / structural next-token，它们都会使用当前 BEV observation。因此，current BEV observation 的作用已经被写入到了这两个分支里。

### 3.1 Temporal next-token branch

输入：

```text
BEV_t + S_{t-1}
```

输出：

```text
h_temp 或 logits_temp(S_t)
```

作用：学习 state 在帧之间的时间连续性、shift 和 persistence。尤其是 occupancy / cover 这种 state，不应该在相邻帧之间突然跳远、消失或者无规律变化。

核心关系是：

```text
future_cover_{t-1,k+1} -> future_cover_{t,k}
future_cover_{t-1,1}   -> current_cover_t
current_cover_{t-1}    -> current_cover_t
phase_{t-1}            -> phase_t
```

如果 `K=3`，就是：

```text
future_cover_{t-1,3} ≈ future_cover_{t,2}
future_cover_{t-1,2} ≈ future_cover_{t,1}
future_cover_{t-1,1} ≈ current_cover_t
```

直观理解：上一帧预测的“更远未来 occupancy”，到了当前帧之后，应该变成“更近未来 occupancy”。这比普通的 `S_{t-1} -> S_t` 更有结构，因为它显式利用了 future cover 的时间 bin shift。

注意，如果 state 是 ego-centric BEV 坐标，做 temporal consistency 时最好考虑 ego-motion warp。否则上一帧的 cover 在当前帧坐标系里会因为 ego 自身运动而发生坐标变化。

---

### 3.2 Graph / structural next-token branch

输入：

```text
BEV_t + single-window cover graph structure
```

输出：

```text
h_graph 或 logits_graph(S_t)
```

作用：学习当前帧内部 state 之间的结构化依赖，也就是从 ego、current cover、future cover 推理 window availability 和 phase。

概念结构可以写成：

```text
ego_t
current_cover_t
future_cover_{t,1:K}
        ↓
window_available_t
        ↓
phase_t
```

或者更直接：

```text
ego_t + current_cover_t + future_cover_{t,1:K}
        ↓
phase_t
```

这一支学习的是：

```text
如果 current cover 正在阻塞 window，phase 更可能是 yield。
如果 future cover 和 ego arrival time 有 temporal overlap，phase 更可能是 yield。
如果 cover 即将 clear，且 ego arrival 不与 future cover overlap，phase 更可能是 go。
```

这部分更像结构化 reasoning / lightweight CoT，但不一定要强行做成线性 token chain。因为 state 之间不是严格线性关系，而是图结构关系：current cover、future cover、ego arrival 都会共同影响 phase。

---

## 4. Single-window temporal cover graph

由于只有一个 window，graph 可以非常小：

```text
             future_cover_{t,3}
                    ↓
             future_cover_{t,2}
                    ↓
             future_cover_{t,1}
                    ↓
ego_t ---> current_cover_t ---> phase_t
```

这个图只是概念图，不代表实现时必须硬性链式生成。更推荐在 embedding / logit 层做 soft prediction 和 fusion，避免 early hard decision 带来的 error accumulation。

核心节点：

```text
Ego node:
  ego_t

Window node:
  window_t

Current cover node:
  current_cover_t

Future cover nodes:
  future_cover_{t,1}, ..., future_cover_{t,K}

Decision node:
  phase_t
```

核心边：

```text
Temporal shift edges:
  future_cover_{t-1,k+1} -> future_cover_{t,k}
  future_cover_{t-1,1}   -> current_cover_t
  current_cover_{t-1}    -> current_cover_t

Structural reasoning edges:
  ego_t -> current_cover_t
  ego_t -> future_cover_{t,k}
  current_cover_t -> phase_t
  future_cover_{t,k} -> phase_t
  ego_t -> phase_t
```

这里的 graph 不是 general dynamic scene graph generation，而是 **planning-state graph completion / refinement**。它只关心 planning 需要的 intermediate semantic state，而不是描述整个交通场景。

---

## 5. 两个分支如何融合

最终要融合的是两个分支得到的 state representation / state logits：

```text
Temporal branch:
  关注 S_{t-1} 到 S_t 的连续性、shift、persistence。

Graph branch:
  关注 S_t 内部 state 之间的结构化依赖，例如 cover -> phase。
```

不推荐做 hard state 融合：

```text
argmax temporal state
argmax graph state
再投票 / 平均
```

因为这样会把早期错误固化。更推荐在 embedding 或 logits 层做 soft fusion。

### 5.1 最简单版本：global gate

```text
h_final = α · h_temp + (1 - α) · h_graph
```

其中 `α` 是一个 learnable scalar。

优点：实现简单。  
缺点：表达力弱，因为不同 state 对 temporal 和 graph 的依赖不同。

---

### 5.2 推荐版本：state-wise / token-wise gate

```text
g_i = sigmoid(MLP([h_i^temp, h_i^graph]))

h_i = g_i · h_i^temp + (1 - g_i) · h_i^graph
```

其中 `i` 表示不同 state token，例如：

```text
current_cover
future_cover_1
future_cover_2
...
phase
```

这样模型可以自动学习：

```text
current_cover / future_cover:
  更相信 temporal branch，因为 occupancy 有连续性。

phase:
  更相信 graph branch，因为 go / yield 是由 ego + cover 关系推理出来的。
```

这是当前最推荐的融合方式，因为它既不复杂，又能很好地解释两种 next-token 分支的互补性。

---

### 5.3 更强版本：uncertainty-aware fusion

两个分支不仅输出 state logits，也输出 uncertainty：

```text
logits_final = w_temp · logits_temp + w_graph · logits_graph

w_temp  ∝ 1 / σ_temp
w_graph ∝ 1 / σ_graph
```

直观理解：哪个分支更确定，就更相信哪个分支。

例如：

```text
Temporal branch 发现上一帧 cover 很稳定：
  对 current/future cover 给更高权重。

Graph branch 发现 ego-window-cover 关系很清楚：
  对 phase 给更高权重。
```

这个版本更优雅，但工程上可以晚一点再做。

---

## 6. 推荐整体模型结构

推荐的主线结构：

```text
BEV_t + S_{t-1}
        ↓
Temporal State Predictor
        ↓
h_temp

BEV_t + single-window cover graph
        ↓
Graph State Predictor
        ↓
h_graph

h_temp, h_graph
        ↓
State-wise Gate Fusion
        ↓
h_state_final
        ↓
State prediction + diffusion conditioning
        ↓
Diffusion planner
```

这里的 graph branch 不一定要复杂 GNN。第一版可以用 relation token / graph attention / small transformer 实现。重点不是 graph 网络本身，而是把 state 组织成一个结构化 bottleneck。

---

## 7. Loss 设计

基础 loss 可以写成：

```text
L = L_state
  + λ_temporal · L_temporal
  + λ_graph · L_graph
  + L_diffusion
```

其中：

```text
L_state:
  final fused state 的监督 loss。

L_temporal:
  temporal branch 的 next-state / shift consistency loss。
  例如 future_cover_{t-1,k+1} 与 future_cover_{t,k} 的一致性。

L_graph:
  graph branch 的结构化推理 loss。
  例如用 ego + current/future cover 预测 phase / availability。

L_diffusion:
  原本 diffusion planner 的 trajectory denoising / planning loss。
```

如果短期工程时间有限，可以先做：

```text
L = L_state + λ_temporal L_temporal + λ_graph L_graph + L_diffusion
```

并且只把 fused state 作为 diffusion condition，不强行改变 diffusion 主体结构。

---

## 8. 实验优先级

### 第一阶段：最稳实现

保留原来的 state heads，只增加 temporal branch 和 graph branch 的辅助监督：

```text
Temporal branch:
  BEV_t + S_{t-1} -> S_t

Graph branch:
  BEV_t + cover graph -> phase / availability

Fusion:
  state-wise gate
```

这个版本不需要大改 diffusion，只是让 state 更稳定、更结构化。

---

### 第二阶段：作为主方法包装

将方法命名为：

```text
Single-window Temporal Cover Graph
Temporal Cover Graph Completion
Cover-centric State Graph Refinement
```

推荐使用：

```text
Temporal Cover Graph Completion
```

因为它强调了两点：

```text
Temporal:
  学习 cover 在帧间的 shift 和 persistence。

Cover Graph Completion:
  补全 ego、cover、window、phase 之间的结构化 state。
```

---

### 第三阶段：进一步增强

可以加入 uncertainty-aware fusion 或者更强的 graph transformer，但这不是第一优先级。

---

## 9. 和 GraphPilot / scene graph 工作的区别

GraphPilot：

```text
通用 traffic scene graph
LLM / VLM driving
Text / JSON / YAML prompt-level conditioning
不改 architecture，不加 graph encoder，不改 loss
目标是让语言模型 internalize relational priors
```

我们：

```text
single-window temporal cover graph
non-language diffusion planning
graph token / state representation / relational bottleneck
目标是建模 ego、current cover、future cover 和 phase 的时序关系
服务于 go / yield timing 和 closed-loop planning
```

一句话区别：

```text
GraphPilot 用 graph 描述交通场景，让 VLM 更懂 relational context；
我们用 graph 描述唯一 window 上的 current/future cover 时序关系，让 diffusion planner 更准地判断什么时候 yield / go。
```

---

## 10. 论文表述草稿

英文版本：

```text
We formulate semantic state prediction as a single-window temporal cover graph completion problem. Instead of constructing a general traffic scene graph, our graph focuses on the decision-critical relations among ego, current cover, future cover, and the go/yield phase over a candidate driving window. Temporal shift edges model the persistence and progression of occupancy states across frames, while intra-frame relational edges capture the structural dependency from cover states to phase decisions. A learnable state-wise gate adaptively fuses the temporal and graph-based predictions before conditioning the diffusion planner.
```

中文理解：

```text
我们把语义 state prediction 建模为 single-window temporal cover graph completion。这个 graph 不描述完整交通场景，而是聚焦唯一候选 window 上 ego、current cover、future cover 和 go/yield phase 之间的决策关键关系。帧间 temporal shift edge 学习 occupancy 的连续性和时间推进，帧内 relational edge 学习 cover 到 phase 的结构化依赖。最后通过一个可学习的 state-wise gate 融合 temporal prediction 和 graph prediction，再作为 diffusion planner 的条件。
```

---

## 11. 当前建议结论

当前最推荐的实现路线是：

```text
不要做复杂 full graph。
不要做多 window graph。
不要做 hard token-by-token argmax。

做一个 single-window temporal cover graph：
  temporal branch 学 cover shift / persistence；
  graph branch 学 cover -> phase 的结构化推理；
  state-wise gate 融合两者；
  fused state condition diffusion planner。
```

这个方案和我们已有的 story 高度一致：

```text
phase 的时机和 decisiveness 很关键；
current/future cover 是 phase 判断的核心依据；
cover 在时间上应该连续、平滑、有 shift；
state 之间应该结构自洽，而不是各个 head 独立预测。
```

因此，graph 的作用不是增加复杂度，而是把已有 state 组织成一个更稳定、更可解释、更适合 planning 的结构化中间表示。

---

## 12. 固定空间、搜索时间：从 Object Detection 到 Temporal Occupancy

一个重要表述：

```text
Instead of detecting which object will matter in space,
we predict when the fixed conflict area will be occupied in time.
```

中文理解：

```text
我们不在当前帧的空间里搜索 future cover 是哪辆车；
我们固定 conflict area，在时间维度上预测它什么时候会被占用。
```

这可以理解为把 future-cover reasoning 从 object detection 转成 temporal
occupancy prediction：

```text
object detection:
  fixed time, search space

temporary occupancy:
  fixed space, search time
```

这条思路的价值：

- 避免显式 detection / tracking。
- 不需要知道 future cover 到底是哪辆车。
- 只需要预测唯一 conflict area 在未来 bins 上是否被占用。
- 与现有 `temporary_occupancy_cover_bins` label 完全对齐。
- 与 perception-limited semantic mode learning 的故事一致。

因此，`future_cover_{t,k}` 不应该理解成：

```text
the k-th future object
```

而应该理解成：

```text
occupancy of the fixed conflict area at future time bin k
```

也就是说：

```text
future_cover_{t,k}
  = O(conflict_area_t, future_bin_k)
```

这也解释了为什么 single-window temporal cover graph 不需要 object
identity：

```text
window / conflict area gives spatial anchor
temporary occupancy gives temporal search
go_opportunity summarizes passability
phase chooses action
```

### Model Implication

后续实现可以显式做一个 temporal cover decoder：

```text
area_token = encode(conflict_area_route_mask / route_out / area_status)
time_bin_tokens = learnable time tokens + time embedding

time_bin_tokens attend to:
  BEV feature
  route_out
  area_token
  optional prev tempocc bins

outputs:
  temporary_occupancy_cover_bins
```

这个结构比 pooled feature 直接吐 13 个 logits 更贴合任务：

```text
given fixed area, predict temporal occupancy sequence
```

### Limitation

这种方法不是完整 object reasoning。

它可能丢失：

- precise future actor identity
- actor-specific speed / distance
- long-range occluded approach information

但这正是当前 V1 的边界：

```text
When object identity is unavailable,
use area-temporal occupancy as a compact surrogate.
```

这个点应该在之后写 plan / paper story 时保留。
