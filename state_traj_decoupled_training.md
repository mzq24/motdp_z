# State Branch 与 Trajectory / Route Branch 的分阶段训练策略

## 1. 问题背景

当前实验中观察到一个明显的收敛速度不匹配问题：

- **state / phase / cover 相关监督** 通常在 **15–25 epoch** 左右已经收敛；
- **route / trajectory / diffusion planner** 往往需要到 **35 epoch 以后** 才开始稳定提升；
- 如果继续 joint training，state branch 可能出现过拟合、漂移，甚至被 trajectory loss 牵引到不再表达原本的语义状态。

这说明 state branch 和 trajectory branch 不应该完全共享同一个训练节奏。更合理的做法是：

> **state 先收敛并稳定下来，随后作为结构化 condition 服务于 diffusion planner；trajectory / route branch 则继续训练。**

换句话说，state branch 负责学习场景理解和 cover/phase reasoning，diffusion branch 负责在这个结构化 condition 下学习轨迹生成。

---

## 2. 为什么不能简单继续一起训

如果所有 loss 一直联合优化：

```text
L = L_state + L_route + L_traj + L_diffusion
```

那么在 state 已经收敛之后，后续训练主要由 route / traj / diffusion loss 主导。这会带来几个风险：

1. **state label 过拟合**  
   state head 已经学到足够好的语义状态后，继续使用较大权重训练可能让模型过度拟合 label noise，尤其是 cover、phase、temporary occupancy 这类标签本身可能并非完全精确。

2. **trajectory loss 扭曲 state representation**  
   如果 state branch 和 diffusion planner 共享 BEV backbone 或中间 adapter，后期 trajectory loss 会继续改变 shared feature。即使 state loss 关闭，state head 的输入分布也可能变化，导致 state 输出漂移。

3. **semantic state 不再是稳定 condition**  
   你的方法主线是让 state 成为 diffusion planner 的 structured condition。如果这个 condition 在后期一直被 trajectory loss 拉动，它就不再是一个稳定语义瓶颈，而可能退化为 trajectory regression 的隐式 feature。

4. **收敛节奏不匹配导致互相干扰**  
   state branch 早收敛，diffusion branch 晚收敛。继续强行 joint training，可能既不能进一步提升 state，又会让 trajectory branch 在变化的 condition 上学习，增加优化难度。

---

## 3. 总体思路

推荐将训练拆成两个或三个阶段：

```text
Stage 1: joint warmup / state learning
  训练 state branch + route/traj/diffusion branch
  目标是让 state / cover / phase 先学稳定

Stage 2: freeze state branch / train planner
  冻结 state adapter、state head、graph/edge head
  state condition detach
  继续训练 route / traj / diffusion

Optional Stage 3: light joint fine-tune
  用很小 lr 解冻部分 state branch
  做短暂联合微调
```

核心原则：

> **state branch 收敛后，不再让 trajectory loss 反向扭曲 state representation。**

---

## 4. 最简单方案：state loss 后期降权

可以先用最小改动的方案：对 state loss 做 annealing。

```python
loss = (
    lambda_state * loss_state
    + lambda_route * loss_route
    + lambda_traj * loss_traj
    + lambda_diffusion * loss_diffusion
)
```

一个可行 schedule：

```text
epoch 0–15:
  lambda_state = 1.0

epoch 15–25:
  lambda_state 从 1.0 线性衰减到 0.1

epoch 25 以后:
  lambda_state = 0.0 或非常小
```

优点：

- 实现简单；
- 不需要改模型结构；
- 可以快速验证 state 后期过拟合是否影响 closed-loop。

缺点：

- 如果 state branch 和 traj branch 共享 backbone，仅仅关闭 state loss 不一定能防止 state 输出漂移；
- trajectory loss 仍然会更新共享 feature，从而改变 state head 的输入分布。

因此，这个方案适合作为 baseline，但不是最稳的最终方案。

---

## 5. 推荐方案：freeze state branch + detach state condition

更推荐的做法是：state 收敛后冻结 state branch，并且把 state condition detach 掉。

### Stage 1: semantic / state warmup

```text
epoch 0–20/25:
  train backbone
  train state adapter
  train state head / graph edge head
  train route/traj/diffusion
```

这一阶段目标是让模型学会：

- current cover；
- future cover / future occupancy bins；
- edge attributes / temporal occupancy；
- phase / go-yield；
- route / trajectory 的基本生成能力。

### Stage 2: freeze state branch, train planner

```text
epoch 25–50:
  freeze state adapter
  freeze state head
  freeze graph / edge head
  detach state condition
  continue training route / traj / diffusion
```

伪代码：

```python
# freeze state branch
for p in state_adapter.parameters():
    p.requires_grad = False
for p in state_head.parameters():
    p.requires_grad = False
for p in graph_edge_head.parameters():
    p.requires_grad = False

# forward
state_embed = state_branch(bev_feature)
state_condition = state_embed.detach()
traj_pred = diffusion_planner(bev_feature, state_condition)

loss = loss_route + loss_traj + loss_diffusion
loss.backward()
```

这个方案的直觉是：

```text
state branch 负责学“场景应该如何理解”；
diffusion planner 后期负责学“在给定 state condition 下如何生成轨迹”。
```

这也更符合论文叙事：state 是一个 structured semantic bottleneck，而不是被 trajectory loss 随意改写的 hidden feature。

---

## 6. 只 freeze head 可能不够

如果模型结构是：

```text
BEV backbone -> state head
BEV backbone -> diffusion planner
```

那么只 freeze `state_head` 可能不够，因为后期 trajectory loss 仍然会更新 BEV backbone，导致 state head 的输入 feature 变化。

更稳的结构是让 state branch 有独立 adapter：

```text
BEV feature
   ├── state adapter -> state / graph / edge attributes
   └── traj adapter / diffusion planner -> trajectory
```

后期冻结：

```text
state adapter
state head
graph / edge head
```

但允许：

```text
BEV backbone
traj adapter
diffusion planner
route head
trajectory head
```

继续更新。

如果当前代码里 state 和 trajectory 共享太深，至少应该引入下面的 teacher consistency 来防止 state drift。

---

## 7. 防止 state drift：state teacher consistency

在 state validation 最好的 epoch，例如 epoch 20 或 25，保存一个 state teacher：

```text
teacher_state_model = checkpoint at best state validation
```

后期继续训练 trajectory branch 时，引入轻量 consistency loss：

```text
L_state_keep = KL(state_current || state_teacher)
```

最终 loss：

```text
L = L_route + L_traj + L_diffusion
  + lambda_keep * L_state_keep
```

其中 `lambda_keep` 不需要大，可以从：

```text
0.01 ~ 0.1
```

开始尝试。

这个 loss 的目的不是继续让 state label 变得更强，而是：

> **防止后期 trajectory training 把已经学好的 state representation 带偏。**

对于 logits，可以使用 KL：

```python
loss_keep = KLDiv(
    log_softmax(state_logits_current / T),
    softmax(state_logits_teacher / T)
)
```

对于 embedding，可以使用 L2 / cosine：

```python
loss_keep = mse_loss(state_embed_current, state_embed_teacher.detach())
```

---

## 8. optimizer 也应该分组

不建议所有模块用完全相同 learning rate。可以按模块分组：

```python
optimizer = AdamW([
    {"params": backbone_params, "lr": 1e-4},
    {"params": state_adapter_params, "lr": 5e-5},
    {"params": state_head_params, "lr": 5e-5},
    {"params": graph_edge_params, "lr": 5e-5},
    {"params": diffusion_params, "lr": 1e-4},
    {"params": route_traj_params, "lr": 1e-4},
])
```

后期可以选择：

```python
for p in state_params:
    p.requires_grad = False
```

或者不完全 freeze，但将 state lr 降到很小：

```text
state lr = 1e-6
traj / diffusion lr = 1e-4
```

不过从稳定性角度，**freeze + detach** 更干净。

---

## 9. 推荐训练 recipe

### 版本 A：最小改动 baseline

```text
Epoch 0–15:
  lambda_state = 1.0
  train all

Epoch 15–25:
  lambda_state 从 1.0 decay 到 0.1
  train all

Epoch 25–50:
  lambda_state = 0.0 或 0.05
  train route / traj / diffusion 为主
```

适合快速验证，但不一定能避免 state drift。

### 版本 B：推荐主线

```text
Epoch 0–25:
  train all
  保存 best state checkpoint

Epoch 25–50:
  freeze state adapter + state head + graph/edge head
  state condition detach
  lambda_state = 0
  train route / traj / diffusion
```

这是最推荐先跑的方案。

### 版本 C：更稳版本

```text
Epoch 0–25:
  train all
  保存 best state checkpoint as teacher

Epoch 25–50:
  freeze or very-low-lr state branch
  state condition detach
  train route / traj / diffusion
  add small state teacher consistency loss

Optional final 3–5 epochs:
  very small lr unfreeze partial state branch
  short joint fine-tune
```

---

## 10. 和当前 graph/state 方法的关系

对于 area-conditioned ego-cover relation graph，state branch 包含：

```text
area + dir query
current cover slot
future cover slot
current edge attributes
future edge attributes / temporal bins
phase head
```

这些内容往往比 diffusion trajectory 更早收敛。因此可以把它们视为早期学习到的 structured condition。

后期训练 diffusion planner 时：

```text
state / edge / phase condition should be stable
```

而不是继续被 trajectory loss 大幅修改。

因此后期建议：

```text
freeze:
  area-query encoder if it mainly serves state
  current/future cover slot encoder
  edge attribute heads
  phase head

detach:
  state condition / graph condition before feeding diffusion

continue training:
  route head
  trajectory head
  diffusion denoiser
  planner adapter
```

如果 area-query encoder 同时服务 trajectory branch，就不能完全 freeze，可以只 freeze state-specific heads，并用 teacher consistency 保护输出。

---

## 11. 论文/汇报中的表述

可以这样包装：

> Because semantic state supervision converges significantly earlier than trajectory generation, we decouple their training schedules. The state branch is first trained to provide stable cover/phase reasoning, and then frozen as a structured condition while the diffusion planner continues to optimize trajectory generation. This avoids overfitting the semantic state labels and prevents trajectory losses from distorting the learned state representation.

中文：

> 由于语义 state 的监督明显早于轨迹生成收敛，我们将二者的训练节奏解耦。首先训练 state branch 学习稳定的 cover/phase reasoning，随后将其冻结作为结构化 condition，让 diffusion planner 继续优化轨迹生成。这样可以避免 state label 过拟合，也防止后期 trajectory loss 扭曲已经学好的 state representation。

---

## 12. 关键结论

一句话总结：

> **state 可以也应该和 route/traj/diffusion 分开训练；关键不是简单关闭 state loss，而是防止后期 trajectory loss 改坏已经收敛的 state representation。**

优先尝试：

```text
0–25 epoch: train all
25 epoch: save best state checkpoint
25–50 epoch: freeze state branch + detach state condition
continue training route/traj/diffusion
```

然后再加：

```text
state loss annealing
state teacher consistency
final short joint fine-tune
```
