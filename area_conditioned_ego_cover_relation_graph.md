# Area-conditioned Ego-Cover Relation Graph 设计记录

## 1. 核心问题与设计动机

我们当前希望解决的问题不是构建一个通用的 traffic scene graph，也不是单纯做 object detection，而是希望让 planner 显式理解：**ego 在当前 route/window/area 条件下，current cover 和 future cover 会如何影响 go/yield phase**。

之前的想法中，我们曾经把 `window`、`current cover`、`future cover`、`phase` 都放成 graph node。但这样存在一个明显问题：这些元素并不在同一语义层级上。

- `window / area / route` 更像 scene-level 或 task-level context。
- `ego` 是当前规划主体。
- `current cover / future cover` 更像 object-level 或 latent cover slot。
- `phase` 是 ego-level decision，不应该被当作场景实体。
- `temporary occupancy / overlap / boundary` 更像 ego-area 与 cover 之间的 relation，而不是某个 node 自身的属性。

因此我们将 graph 重新定义为一个更紧凑、更任务对齐的 **Area-conditioned Ego-Cover Relation Graph**。

该 graph 的核心思想是：

> 节点尽量保持在同一抽象层级，主要表示 ego 与 cover slots；window/area/route 作为 context 或 query 条件；temporary occupancy、future occupancy bins、boundary、temporal overlap 等 planning-relevant 状态作为 edge attributes；phase 则由这些 relation edges 推理得到，并作为 diffusion planner 的 condition。

换句话说，我们不是把所有 state 都粗暴放进 graph，而是做清楚分层：

```text
Context = scene/window/area/route/dir information
Node    = object-level or slot-level evidence container
Edge    = ego-area 与 cover 之间的时空关系
Decision= phase / go-yield condition
```

---

## 2. 与通用 scene graph / 4D occupancy 的区别

### 2.1 不是通用 traffic scene graph

类似 GraphPilot 这类方法通常构建的是通用交通 scene graph，节点包括 lane、road、junction、ego、vehicle、pedestrian、traffic light 等，边包括 `near`、`left of`、`is in`、`controls traffic of` 等。这种图的目标是帮助 VLM/LLM planner 理解通用 scene relation。

我们的目标不同。我们不是要完整描述交通场景，而是只关注一个 planner-critical relation：

> 在 ego 当前 route/window/area 条件下，current cover 和 future cover 是否、何时、如何影响 ego 的通过决策。

因此我们的 graph 更小、更聚焦、更偏 decision-centric。

### 2.2 不是 full 4D occupancy forecasting

4D occupancy / occupancy world model 确实已经在预测未来时空占据。但它们通常预测的是全局 dense occupancy field，目标是 scene-centric forecasting。

我们的目标不是重建完整未来世界，而是预测一个紧凑的、window-conditioned 的 temporal cover state：

```text
Does the candidate area/window become occupied at future bin k?
Does this occupancy overlap with ego's intended passing time?
Should ego go or yield?
```

所以区别不是“别人没有时序，我们有时序”，而是：

```text
Dense 4D occupancy: scene-centric future reconstruction
Ours: decision-centric window-level temporal relation prediction
```

对于全局 occupancy 指标来说，conflict window 附近很小的 timing error 可能影响不大；但对 go/yield phase 来说，这个 timing error 可能直接决定能不能安全通过。因此我们希望将监督集中到与 phase decision 最相关的 temporal relation 上。

---

## 3. 为什么不是 object-level future cover detection

一个直接想法是显式检测 future cover object：找出哪一个 object 未来会进入 area/window，并影响 ego。这个思路的问题是：

1. `future cover object` 不是普通当前帧 object detection target。它依赖未来运动、ego route、window geometry、ego arrival time，以及是否与 ego 发生 temporal overlap。
2. 对 phase-aware planning 来说，具体是哪一个 object 并不是必要变量。planner 更关心的是：ego 尝试通过 window 时，这个 window 是否会被占据。
3. object-level detection / tracking / matching 会引入额外的中间瓶颈和误差传播。

因此我们不显式监督 future cover object identity，而是做 object identity 的边缘化：

```text
P(C_k = 1 | BEV, area, direction)
= P(there exists some object that occupies the area at future bin k | BEV, area, direction)
```

也就是说，我们不是放弃 spatial evidence，而是避免把 object identity 作为必须显式预测的中间目标。

更准确的表述是：

> 我们仍然利用 BEV 中的 spatial evidence，但中间监督从 object-level future-cover identity 转为 window-level temporal occupancy relation。

这可以避免 reviewer 质疑“连 detection 都做不到，为什么要做更难的未来预测”。我们的回答应该是：

> We do not replace spatial perception with pure temporal guessing. Instead, we use the candidate area and route direction as explicit queries to aggregate spatial evidence from BEV, and supervise the temporal relation that is directly required by go/yield decision making.

---

## 4. 总体输入与输出

### 4.1 输入

Graph module 的输入不是一个现成的 graph，而是：

```text
Input:
  - BEV feature F_t
  - route-based area heatmap A_t
  - direction / route direction D_t
  - ego state E_t
  - optional previous relation state R_{t-1}
```

其中：

- `BEV feature F_t` 提供 spatial evidence。
- `route-based area heatmap A_t` 是显式 area/window query，它告诉模型 ego 当前真正关心的 candidate area。
- `direction D_t` 提供 route direction / approach direction，有助于判断哪些 BEV evidence 与未来 cover 有关。
- `ego state E_t` 可以包含 speed、heading、距离 area 的距离、估计 arrival time 等。
- `previous relation state R_{t-1}` 用于 temporal shift / persistence consistency。

### 4.2 输出

Graph module 输出的是 planner-relevant relation state，而不是 object boxes：

```text
Output:
  - current-cover relation
      current occupancy / blocking / confidence

  - future-cover temporal relation
      future occupancy bins over K future steps
      temporal overlap with ego arrival
      confidence / uncertainty

  - go/yield boundary or related signal
      boundary score / safe gap / decision margin

  - phase condition
      go / yield logits or embedding condition
```

最终 phase condition 会被送入 diffusion planner，作为结构化 condition 帮助 trajectory generation。

---

## 5. Graph 的层级设计

### 5.1 Context：不作为 node

以下信息不作为 graph node，而是作为 context / query condition：

```text
C_t = {
  BEV feature,
  route-based area heatmap,
  route direction,
  window geometry,
  scene understanding,
  ego kinematic state
}
```

特别是 `window / area` 不再被当作 node。它们是 ego 当前关注的空间区域，是 query 条件，而不是与 ego/current cover/future cover 同层级的实体。

### 5.2 Nodes：object-level / slot-level evidence containers

Graph 中只保留少量同层级 node：

```text
V_t = {
  v_ego_area,
  v_current_cover,
  v_future_cover
}
```

#### v_ego_area

`v_ego_area` 不是裸 ego，而是 **area-conditioned ego query**。它融合了：

```text
ego state + route-based area heatmap + direction + window/area context + BEV evidence
```

可以理解为：ego 在当前 route/window/area 条件下的 planning query。

#### v_current_cover

`v_current_cover` 表示当前可能正在占据 / 阻塞 area 的 cover evidence。它可以通过 current-cover latent query 从 BEV 中检索得到。

#### v_future_cover

`v_future_cover` 表示未来可能影响 area 的 latent cover evidence。它不一定对应某个明确检测出的 object，也不要求有 object id 或 box。它是一个 **area-conditioned latent future-cover slot**。

这点很重要，因为我们目前不希望强行做 future cover object detection。future cover node 的作用是：从 BEV 中聚合那些可能在未来进入 area/window、影响 ego phase 的 evidence。

---

## 6. Area + Direction Query

我们不建议使用纯 learnable query 来表示 area，因为我们已经有 route-based area heatmap，这是非常明确的 spatial prior。

因此 query 应该尽量显式：

```text
q_area_dir = Encode(area_heatmap, direction, ego_state)
```

可选实现方式包括：

### 6.1 Weighted pooling

使用 area heatmap 对 BEV feature 做加权池化：

```text
f_area = sum_i A_t(i) * F_t(i)
```

再结合 direction / ego state：

```text
q_area_dir = MLP([f_area, dir_emb, ego_state])
```

优点：简单稳定。

缺点：可能只关注 area 内部，不容易看到 area 外即将进入的 actor。

### 6.2 Grid BEV sampling + cross-attention

在 area / boundary / route direction / incoming corridor 附近采样一组 BEV tokens：

```text
T_area = SampleBEV(F_t, area_heatmap, direction, margin/corridor)
```

然后用 `q_area_dir` 对采样 tokens 做 cross attention：

```text
h_area = CrossAttention(query=q_area_dir, key=T_area, value=T_area)
```

优点：

- 保留显式空间约束。
- 不让 attention 在全 BEV 中乱找。
- 能同时看到 area 内部和将要进入 area 的附近 evidence。

这是目前更推荐的实现。

### 6.3 Deformable cross-attention

用 area+dir 生成若干 reference points / offsets，在 BEV feature 上做 deformable attention。

优点：更灵活。

缺点：实现复杂度更高，当前 deadline 下不一定优先。

---

## 7. Cover Node 的表示方式

### 7.1 不使用纯 learnable token

`current cover node` 和 `future cover node` 可以用 learnable token 初始化，但不应该是纯 learnable token 直接预测。更合理的是：

```text
cover node = learnable slot query + area/window context + BEV evidence aggregation
```

也就是说，cover node 类似 DETR object query / latent slot，但必须通过 BEV cross-attention 聚合空间证据。

### 7.2 Current cover node

```text
q_cur = learnable_current_cover_token + q_area_dir
v_current_cover = CrossAttention(q_cur, sampled BEV tokens)
```

含义：

> 以 ego-area 为条件，在 BEV 中聚合当前正在占据或阻塞 area 的 evidence。

### 7.3 Future cover node

```text
q_fut = learnable_future_cover_token + q_area_dir
v_future_cover = CrossAttention(q_fut, sampled BEV tokens)
```

含义：

> 以 ego-area 为条件，在 BEV 中聚合未来可能进入 area、影响 ego phase 的 object / motion / lane evidence。

这里 `future cover node` 不要求对应一个明确的 object。它可以是 latent slot，用于承载 future-cover evidence。

---

## 8. Edge 的定义：edge attribute 是核心

我们当前的 graph topology 很小，基本只有两条核心 relation edges：

```text
v_ego_area  <---- e_cur ---->  v_current_cover
v_ego_area  <---- e_fut ---->  v_future_cover
```

这里 graph 的价值不在于发现复杂 topology，而在于 **预测 edge attributes**。

如果 edge 只是“存在/不存在”，信息量太少，因为这两条边基本是固定存在的。真正有用的是 edge 上的属性：

```text
current occupancy
future occupancy bins
temporal overlap
blocking relation
safe gap / boundary
confidence / uncertainty
```

因此这里应该是 edge-centric prediction：

```text
node = evidence container
edge = planner-relevant relation
phase = decision from relations
```

### 8.1 Current edge

```text
e_cur = EdgeHead_cur(v_ego_area, v_current_cover, context)
```

输出：

```text
e_cur.attr = {
  current_occ,
  current_blocking,
  current_confidence,
  optional current_boundary_signal
}
```

含义：当前 cover slot 是否正在占据/阻塞 ego-area。

### 8.2 Future edge

严格来说，不需要 K 条 future edges。应该是一条 `e_fut`，其 attribute 是一个 temporal vector：

```text
e_fut = EdgeHead_fut(v_ego_area, v_future_cover, context)
```

输出：

```text
e_fut.attr = {
  future_occ_bins[1:K],
  temporal_overlap_bins[1:K],
  future_confidence,
  optional boundary_bins[1:K]
}
```

其中：

```text
future_occ_bins[k]
```

表示在当前 t 帧预测，候选 area/window 在未来第 k 个时间 bin 是否会被占据。

注意：`k` 是同一条 edge attribute 中的 temporal bin index，不是第 k 条 edge，也不是第 k 个 node。

---

## 9. Phase Head

`phase` 不作为 node，而是 ego-level decision head。

它由 ego-area context 和 edge attributes 推理得到：

```text
phase_logits = PhaseHead(
  v_ego_area,
  e_cur.attr,
  e_fut.attr,
  optional boundary/context
)
```

输出：

```text
phase ∈ {go, yield}
```

或者输出 phase embedding / confidence，用作 diffusion planner condition。

直观上：

```text
ego 根据当前 area/window、route direction、current cover relation、future cover temporal bins、boundary/safe-gap signal，判断现在应该 go 还是 yield。
```

---

## 10. Temporal next-token / shift consistency 如何落到 graph 上

之前讨论过两种 next-token：

1. 帧间 next-token：学习 state 在时间上的 shift / persistence。
2. state 内部 next-token：学习 state 之间的结构化推理。

在这个新的 graph 版本中，它们可以重新解释为：

### 10.1 帧间 next-token = edge attribute temporal shift

时序连续性主要约束的是 edge 上的 temporal occupancy bins，而不是 node identity。

核心关系：

```text
future_occ_{t-1,k+1}  ≈  future_occ_{t,k}
future_occ_{t-1,1}    ≈  current_occ_t
```

含义：上一帧预测的更远未来 occupancy，到了当前帧之后应该变成更近未来 occupancy；上一帧预测的最近未来 occupancy，当前帧应该接近 current occupancy。

这比直接让模型预测 abstract state 更清楚，因为 occupancy 本身就是 edge relation：它描述的是 cover slot 相对于 ego-area 的时序占据关系。

### 10.2 帧内 graph reasoning = edge completion + phase decision

帧内结构化推理不再是简单的 node chain，而是：

```text
v_ego_area + v_current_cover -> e_cur.attr
v_ego_area + v_future_cover  -> e_fut.attr
[e_cur.attr, e_fut.attr, boundary/context] -> phase
```

也就是说，graph 的作用是让模型显式补全 ego-area 与 cover slots 之间的 relation attributes，并基于这些 attributes 做 phase decision。

---

## 11. 推荐模型结构

一个可实现的结构如下：

```text
BEV feature F_t
    ↓
Area + direction query encoder
    ↓
v_ego_area

q_cur = learnable_current_cover_token + q_area_dir
    ↓ cross-attention over sampled BEV tokens
v_current_cover

q_fut = learnable_future_cover_token + q_area_dir
    ↓ cross-attention over sampled BEV tokens
v_future_cover

(v_ego_area, v_current_cover)
    ↓ Current Edge Head
e_cur.attr: current_occ / blocking / confidence

(v_ego_area, v_future_cover)
    ↓ Future Edge Head
e_fut.attr: future_occ_bins / overlap_bins / confidence / boundary

[v_ego_area, e_cur.attr, e_fut.attr]
    ↓ Phase Head
phase logits / phase embedding

phase + relation state
    ↓
condition diffusion planner
```

其中 sampled BEV tokens 可以来自：

```text
area mask + margin
route corridor
incoming direction corridor
boundary points
nearby BEV grid around candidate area
```

这样既利用了 BEV spatial evidence，又避免了显式 future-cover object detection。

---

## 12. 训练目标

可以使用如下 loss：

```text
L = L_cur
  + λ_fut L_fut
  + λ_phase L_phase
  + λ_shift L_shift
  + optional λ_boundary L_boundary
  + L_diffusion
```

### 12.1 Current cover loss

```text
L_cur = BCE(current_occ_pred, current_occ_gt)
```

或者多分类：

```text
current relation ∈ {free, occupied, blocking, uncertain}
```

### 12.2 Future occupancy bins loss

```text
L_fut = Σ_k BCE(future_occ_pred[k], future_occ_gt[k])
```

其中 label 可以通过未来帧判断：未来第 k 个时间 bin，是否有任意 actor 与候选 area/window overlap。

### 12.3 Phase loss

```text
L_phase = CE(phase_pred, phase_gt)
```

phase 是最终 go/yield decision，建议作为较强监督，因为它直接影响 closed-loop performance。

### 12.4 Temporal shift consistency loss

```text
L_shift = Σ_k || future_occ_pred_{t-1,k+1} - future_occ_pred_{t,k} ||
        + || future_occ_pred_{t-1,1} - current_occ_pred_t ||
```

如果 BEV/state 是 ego-centric，需要考虑 ego-motion warp 或者在 label 构造时统一到 area/window coordinate，避免因为 ego 自身运动导致 shift 关系不成立。

### 12.5 Boundary / overlap loss

如果有 boundary、safe gap、ego-arrival overlap label，可以加：

```text
L_boundary = BCE/CE/MSE(boundary_pred, boundary_gt)
```

但第一版可以先不做太复杂，先跑通 current/future occupancy + phase + shift。

---

## 13. Edge attribute 是否必要

在这个 graph 中，edge attribute 是必要的。

原因：graph topology 非常小且基本固定，只有：

```text
ego_area -- current_cover
ego_area -- future_cover
```

如果 edge 只表示“连接存在”，几乎没有额外信息。此时模型会退化成：

```text
concat(v_ego_area, v_current_cover, v_future_cover) -> phase
```

这样 graph 的设计意义会变弱。

真正的贡献应该是显式建模 relation：

```text
这个 cover 是否当前 block ego-area？
这个 future cover 在什么时候进入 area？
这个 occupancy 是否与 ego arrival overlap？
当前应该 go 还是 yield？
```

所以我们应该强调：

> The graph is edge-centric: nodes store evidence, while edge attributes store planner-relevant temporal relations.

---

## 14. 代码适配思路

### 14.1 最小实现版本

第一版不需要真正写复杂 graph library，可以用普通 PyTorch module 实现：

```text
AreaQueryEncoder
CoverSlotCrossAttention
CurrentEdgeHead
FutureEdgeHead
PhaseHead
```

其中：

```text
v_ego_area = AreaQueryEncoder(BEV, area_heatmap, dir, ego)
v_cur      = CoverSlotCrossAttention(q_cur, sampled_BEV)
v_fut      = CoverSlotCrossAttention(q_fut, sampled_BEV)
e_cur      = CurrentEdgeHead(v_ego_area, v_cur)
e_fut      = FutureEdgeHead(v_ego_area, v_fut)
phase      = PhaseHead(v_ego_area, e_cur, e_fut)
```

这在实现上不一定需要显式 graph data structure，但在论文中可以解释为 relation graph。

### 14.2 Cross-attention vs grid sampling + cross-attention

推荐优先实现：

```text
grid BEV sampling + cross attention
```

原因：

- area heatmap 和 direction 是强 prior。
- 采样能限制搜索空间，减少 attention 学 shortcut 或乱找。
- cross attention 保留了灵活的信息聚合能力。

### 14.3 Future cover slot 的数量

当前设计中可以先用一个 future cover slot：

```text
v_future_cover
```

它表示对所有可能 future-cover evidence 的聚合。

如果后续发现多个 actor 同时影响 area，可以扩展成多个 future cover slots：

```text
v_future_cover^1, ..., v_future_cover^M
```

然后通过 pooling / max / attention 聚合它们的 edge attributes。但第一版为了稳，不建议做太复杂。

---

## 15. 可能的 reviewer 质疑与回应

### 15.1 质疑：为什么不直接检测 future cover object？

回应：

future-cover object identity 不是 phase-aware planning 的必要变量。真正需要的是候选 area 在 ego 通过时是否会被占据。我们仍然从 BEV 中聚合 object-level spatial evidence，但不显式监督 object identity，而是监督更贴近 planning 的 ego-area-cover temporal relation。

### 15.2 质疑：这是不是比 detection 更难？

回应：

我们并不是用纯 temporal guessing 替代 spatial detection。相反，我们使用 route-based area heatmap 和 direction 作为显式 query，从 BEV 中检索相关 spatial evidence。预测目标从 object-level identity 转为低维的 window-level temporal occupancy，这减少了不必要的 tracking/matching bottleneck，并直接对齐 go/yield decision。

### 15.3 质疑：这和 4D occupancy 有什么不同？

回应：

4D occupancy 预测完整 dense future scene，目标是 scene-centric forecasting。我们预测的是 single-window / area-conditioned temporal relation，目标是 decision-centric phase reasoning。对 phase 来说，conflict window 附近的 occupancy timing 比全局 occupancy reconstruction 更关键。

### 15.4 质疑：为什么需要 graph？直接 concat features 预测 phase 不行吗？

回应：

直接 concat 难以显式监督 ego-area 与 cover 之间的 temporal relation。我们的 graph 是 edge-centric 的：nodes 聚合 evidence，edges 显式承载 current occupancy、future occupancy bins、temporal overlap 和 boundary。这样可以让模型学习中间 planner state，而不仅仅是端到端地从 BEV shortcut 到 phase。

---

## 16. 可放进论文的方法表述

### 英文版本

```text
We formulate the intermediate state prediction as an area-conditioned ego-cover relation graph. Instead of constructing a generic traffic scene graph with heterogeneous nodes from different semantic levels, our graph keeps nodes at the object/slot level, including an ego-area query, a current-cover slot, and a future-cover slot. The candidate window and route direction are treated as contextual queries rather than graph nodes. Given BEV features, the ego-area query and cover slots retrieve spatial evidence through area-conditioned cross-attention. The key planning variables, such as current occupancy, future occupancy bins, temporal overlap, and go/yield boundary, are modeled as edge attributes between the ego-area query and cover slots. The final go/yield phase is predicted from these relation edges and used as structured conditioning for diffusion planning.
```

### 中文版本

```text
我们将中间状态预测建模为一个 area-conditioned ego-cover relation graph。与构建包含不同语义层级节点的通用 traffic scene graph 不同，我们的 graph 将节点保持在 object/slot 层级，包括 ego-area query、current-cover slot 和 future-cover slot。候选 window 和 route direction 不作为图节点，而是作为 context/query 条件。给定 BEV 特征后，ego-area query 和 cover slots 通过 area-conditioned cross-attention 从 BEV 中检索相关空间证据。current occupancy、future occupancy bins、temporal overlap、go/yield boundary 等关键规划变量被建模为 ego-area query 与 cover slots 之间的 edge attributes。最终 go/yield phase 由这些 relation edges 推理得到，并作为 diffusion planner 的结构化条件。
```

---

## 17. 一句话总结

最终版本可以总结为：

> 我们用 route-based area heatmap 和 direction 构造显式 ego-area query，从 BEV 中聚合候选 window 相关的 spatial evidence；current/future cover 用 latent cover slots 从 BEV 中检索 evidence；graph 只包含 ego-area 与 cover slots 之间的少量 relation edges。真正被监督的不是 node identity，而是 edge attributes，包括 current occupancy、future occupancy bins、temporal overlap 和 go/yield boundary。最终 phase 由这些 edge attributes 推理得到，并作为 diffusion planner 的 condition。

更短的版本：

> BEV is the spatial evidence; area+direction is the query; cover slots retrieve relevant evidence; edge attributes model temporal cover relations; phase is the decision condition.

---

## 18. Boundary Naming And V1 Scope Update

During discussion, we decided to avoid the term `boundary_bins` in V1 because it is easy to confuse with the existing speed-boundary labels.

Important distinction:

```text
speed_boundary:
  existing yld/go/chase speed constraints
  e.g. yld_max, go_min, chase_speed_max

temporal passability / overlap:
  whether entering the conflict area at a future time bin is safe / feasible
  this is not a speed boundary
```

Therefore, if we later need a time-bin signal, prefer names such as:

```text
temporal_overlap_bins
passability_bins
gap_margin_bins
```

But for V1, do not add these extra bin heads yet. The first implementation should stay compact:

```text
current edge:
  current_occ / current_blocking

future edge:
  future_occ_bins

phase / opportunity:
  go_opportunity_prob / yld_pressure_prob
  decision_phase / control_phase

speed boundary:
  yld_max / go_min / chase_speed_max
```

Rationale:

- `go_opportunity_prob` is already a compressed passability signal.
- Adding passability/gap-margin bins now would blur the scope and create another label burden.
- The immediate goal is to solve phase-boundary mismatch by separating cover-event relation from speed-boundary supervision, not by adding more boundary-like heads.

