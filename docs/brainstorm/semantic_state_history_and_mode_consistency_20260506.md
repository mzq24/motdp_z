# Semantic State History / Mode Consistency Notes

## 1. Phase 与 go opportunity 的方向修正

重要结论：

```text
decision_phase / control_phase 是 expert 在 good route 中已经成功执行的既定事实。
go_opportunity / yld_pressure 是对这个事实背后 temporal passability 的解释性估计。
```

因此训练侧不应该做：

```text
go_opportunity low -> suppress phase_go
```

原因：

- 当前训练集使用 good routes。
- 如果 expert phase 是 `go` 且最终正确通过，就说明这个 `go` 行为本身是成立的。
- `go_opportunity` 估计偏低时，应该校准 opportunity，而不是反过来否定 expert phase。

正确方向：

```text
phase_go + successful passage
  -> go_opportunity should not be too low
  -> calibrate go_opportunity to a conservative lower bound, e.g. 0.5
```

也就是说：

```text
phase -> opportunity calibration
```

而不是：

```text
opportunity -> phase target modification
```

实现提醒：

- `temporary_occupancy_phase_alpha` 应保持 `0.0`，不要让 `go_opportunity_logits` additive modulate `decision_phase_logits`。
- 如果做 consistency，也应只约束 opportunity 的合理下界，不应修改 phase CE target。

## 2. Route 与 state 的依赖方向

更合理的结构方向：

```text
scene + target
  -> high-level interaction mode
  -> route geometry
  -> conflict area / area status / timing
  -> opportunity / passability
  -> action phase / boundary / chase
  -> traj / speed
```

因此能直接影响 route 的 state 应该是 high-level scene mode：

- `window / interaction mode`
- `dir`

不建议第一版直接让 route 吃：

- `conflict_area`
- `dist_to_entry / dist_to_exit / time_to_entry`
- `temporary_occupancy`
- `go_opportunity`
- `boundary`
- `chase`

原因：

- conflict area / timing 很多时候是 route-conditioned 的结果。
- 如果让 route 依赖 area，而 area 又由 route 定义，会形成不清晰的循环。

推荐第一版：

```text
window + dir -> route token / route query adapter
area/timing/tempocc/opportunity/phase/boundary/chase -> traj/speed adapter
```

## 3. Dir 的 active 语义与时空一致性

`dir` 不应在 active window 内随意变成 `none`。

推荐语义：

```text
window = none:
    dir = none is allowed

window != none:
    dir should be one of same / opposite / cross
```

动机：

- `dir` 是 merge / borrow / junction 区分的关键语义。
- borrow 与 merge 形状接近时，主要依赖 `same / opposite` 区分。
- 没车或感知弱时，单帧 `dir` 容易乱跳。
- 但当 ego pose 和 route 没有明显变化时，`dir` 应有很强时间/空间一致性。

典型风险：

```text
borrow running midway:
  window drops to none
  dir drops to none
  route loses "return to own lane" semantic anchor
  ego may continue driving in opposite lane
```

## 4. 是否给 state 输入 history

担心：

```text
If history state is directly fed to current state heads,
model may shortcut from history and ignore scene understanding.
```

但完全不用 history 会伤害：

- borrow near-end
- dir persistence
- window active continuation
- temporary occupancy / go opportunity temporal stability

推荐拆分职责：

```text
base state head:
    current scene + route understanding

transition state head:
    prev semantic state + current scene + route -> current semantic state

inference/postprocess:
    causal latch / sticky state for safety-critical persistence
```

训练防 shortcut 建议：

- history state 只进 transition branch，不直接进 base current-state head。
- transition loss 权重小一些，例如 `0.1 ~ 0.3`。
- 对 prev state 做 dropout / corruption。
- route/traj 使用 state condition 时默认 detach。
- 后期再考虑 scheduled sampling，不要一开始完全喂 predicted history。

## 5. 同一帧 semantic chain consistency 是什么

这里指 intra-frame consistency，也就是同一帧内部不同 semantic factors 之间的结构关系约束。

它不是 frame-to-frame smoothing。

例子：

```text
window = none
  -> conflict_area should be empty or status none
  -> dir should be none

window != none
  -> dir should be same / opposite / cross, not none

area_status = inside
  -> action should not be stop/yield forever
  -> inside-area go prior can be weakly encouraged

temporary occupancy shows strong near-future blockage
  -> go_opportunity should not be high

phase_go + successful passage
  -> go_opportunity should not be too low
```

注意：

- 这些是 semantic consistency，不一定都要做成 hard loss。
- 有些适合 label postprocess。
- 有些适合 training auxiliary loss。
- 有些适合 debug metric。

## 6. Opportunity bins time-shift consistency 是什么

`temporary_occupancy_cover_bins` / opportunity bins 是一个未来时间序列。

如果数据是连续帧，理论上下一帧的 bins 应该近似等于上一帧向前平移：

```text
O_t = [o_t[0], o_t[1], ..., o_t[K]]
O_{t+1}[k] ≈ O_t[k + delta]
```

在 2Hz 情况下，`delta` 取决于 bins 的时间分辨率。

直观例子：

```text
上一帧：未来 1s 有车占用 conflict area
下一帧：这个占用应更接近当前，而不是突然消失
```

这个约束主要解决：

- temporary occupancy bins 抖动
- go opportunity 抖动
- phase 在 yld/go 间来回跳

## 7. 当前代码是否已经实现 time-shift consistency

当前代码里已有两个相关机制，但不是显式 time-shift consistency。

### 已有 1：同 sample 双 noise-view state consistency

代码中已有：

- `state_consistency_tempocc_loss`
- `state_consistency_opportunity_loss`

作用：

```text
same frame, different noise/timestep views
  -> predicted state should be consistent
```

它约束的是同一帧不同 denoise/noise view 的稳定性，不涉及相邻帧 bins 的时间平移。

### 已有 2：semantic transition branch

当前 prev-state 机制已有：

- `prev_temporary_occupancy_cover_bins`
- `prev_go_opportunity_prob`
- `prev_yld_pressure_prob`
- `semantic_transition_*`

作用：

```text
prev semantic state + current obs -> current semantic state
```

它可以隐式学到 temporal evolution，但没有显式写：

```text
O_{t+1}[k] ≈ O_t[k+1]
```

### 尚未显式实现

还没有明确的 loss：

```text
temporary_occupancy_time_shift_loss
```

或者：

```text
opportunity_bins_shift_consistency_loss
```

如果后续要做，建议作为独立 ablation，而不是混进当前 baseline。

## 8. 后续可选实现

V1 可以先不做。

如果要加，建议：

```text
only apply when:
  prev_semantic_state_valid = 1
  same route / same scene
  conflict_area stable enough
  window family unchanged or compatible
```

loss 形式：

```text
L_shift = BCE / SmoothL1(
    pred_tempocc_t[:, :-delta],
    stopgrad(prev_tempocc[:, delta:])
)
```

或者用于 label sanity：

```text
current_tempocc_target[:, :-delta]
should be close to
prev_tempocc_target[:, delta:]
```

注意不要强行约束真实突变：

- 新车刚进入 conflict area
- ego route/area 改变
- window family 改变
- conflict area definition 发生跳变

## 9. Inference Semantic Condition Update Schedule

当前 infer 的 branch condition 逻辑是 per-denoise-step two-pass：

```text
for each DDIM step:
    pass1 forward_ego -> read semantic state
    compose branch condition
    pass2 forward_ego(branch_condition) -> DDIM update
```

如果 `num_inference_steps = 10`：

```text
branch off: about 10 main forward_ego
branch on:  about 20 main forward_ego
```

最后还有一次 `t=0` stage1 readout/debug forward，但它不参与 DDIM update，可以先不作为核心计算链路讨论。

### 观察

State 主要由：

```text
scene feature + explicit route / route_out + speed_out
```

得到。

而 route 通常较容易学，很多场景中 1-step / early-step route hypothesis 已经比较准确。

因此 state 在每个 denoise step 都重算，不一定总是带来强收益。尤其：

```text
window / dir / coarse phase / coarse opportunity
```

理论上不应该每步大幅变化。

更可能需要逐步 refinement 的是：

```text
conflict area
area timing
temporary occupancy / opportunity
```

因为它们更依赖当前 route hypothesis。

### 当前 per-step two-pass 的解释

当前逻辑可以理解成一个轻量 EM-style iteration：

```text
E-like:
    current route/traj hypothesis -> estimate semantic state

M-like:
    semantic state -> refine traj/speed/route hypothesis
```

重复多步后，state 与 motion 可以相互修正。

但如果 route 对 state 的影响较小，这个每步更新可能带来：

- 额外计算开销
- state 抖动
- phase / go opportunity 在 denoise steps 间不稳定

当前讨论的判断：

- semantic state 主要从 scene feature 和显式 route 中读出来。
- route 对很多样本来说比较容易，early denoise / 甚至一次 denoise 后已经足够接近。
- 因此 `route -> state -> route/traj` 的多步互相修正可能不是主要收益来源。
- 更值得验证的是：state 是否可以先早期确定，然后后续复用或 EMA 平滑，减少 phase / opportunity / window 的 step-to-step 抖动。
- 这不否认 area/timing/tempocc 可能需要 route refinement，只是建议先用 `every_step / first_k / once` 做 ablation，而不是默认假设每步 state 重算都必要。

### 建议 ablation

新增 infer-side config：

```yaml
route_b:
  semantic_condition_update_mode: every_step  # every_step / first_k / once
  semantic_condition_update_steps: 2
  semantic_condition_ema: true
  semantic_condition_ema_alpha: 0.7
```

语义：

```text
every_step:
    当前逻辑。每个 DDIM step 都重新预测 semantic condition。

first_k:
    前 K 个 denoise step 更新 semantic condition，后续复用最后一次 condition。

once:
    只在第一个 denoise step 更新 semantic condition，后续全程复用。

ema:
    新预测 condition 与历史 condition 做 EMA，降低 step-to-step 抖动。
```

计算量示例：

```text
current every_step, K=10:
    10 * (pass1 + pass2) = about 20 main forward_ego

first_k=2:
    2 * (pass1 + pass2) + 8 * pass2 = about 12 main forward_ego
```

### 注意

第一版建议只做 infer ablation，不改 training。

原因：

- 不改 backbone。
- 不改 loss。
- 能直接观察 close-loop 速度与稳定性。
- 如果 `first_k/once + EMA` 分数不差，说明 state 确实不需要每 step 重算。

保留风险：

- area/timing/opportunity 可能确实需要 later-step route refinement。
- borrow near-end 可能需要 state 与 route 互相修正。
- 所以不要直接删 every-step，先用 config ablation。

## 10. 下一版计划候选关键词

后续根据 label 更新后写 plan 时，可以围绕：

```text
Semantic-State V2:
  prev coarse route memory
  3-level intra-frame semantic next-token
  inter-frame semantic transition strengthening
  infer-side semantic condition update schedule
```

其中 3-level intra-frame semantic chain 暂定为：

```text
Level 1:
    window / interaction mode

Level 2:
    spatial interaction token
    outputs: dir, conflict_area, area_status, timing

Level 3:
    temporal/action token
    outputs: tempocc, go_opportunity, phase, boundary, chase
```

Route memory 暂定只使用 coarse previous state：

```text
prev_window
prev_dir
optional prev_borrow_time / latch state
```

不使用：

```text
prev_area_status
prev_area_mask
prev_timing
prev_tempocc
prev_phase
```

原因：

```text
current route should not depend on current state interpretation,
but can use previous coarse interaction intent as temporal memory.
```

## 11. Infer-Time Semantic Transition 与 Direct/Transition Fuse

当前状态：

- Training 已经有 frame-to-frame semantic transition branch。
- 它使用 offline `prev_*` GT semantic state：

```text
prev semantic state + current obs/route context -> current semantic state
```

- 但 infer 里目前 branch condition 仍主要来自 direct state head：

```text
current obs/route context -> current semantic state
```

也就是说：

```text
training has transition auxiliary
infer does not yet use transition as causal state memory
```

原因：

- Training 的 `prev_*` 来自 offline packed dataset。
- Close-loop infer 没有 GT `prev_*`。
- 要在 infer 使用 transition，需要 policy / agent 维护 causal predicted semantic state cache。

推荐下一版 infer 结构：

```text
episode start:
    prev_state_valid = 0
    prev_state = neutral

each frame:
    direct_state = DirectHead(obs_t, route_t)
    transition_state = TransitionHead(prev_pred_state, obs_t, route_t)
    fused_state = Fuse(direct_state, transition_state)

    branch_condition = ComposeAllStateCondition(fused_state)
    traj/speed = MotionDecoder(branch_condition)

    cache fused_state as prev_pred_state
```

### Route Memory 与 Semantic Transition 分工

Route 可以读 previous memory，但只读 coarse previous state：

```text
prev_window / prev_family
prev_dir
optional prev_borrow_time / borrow_latch
```

Route 不读：

```text
prev_area_status
prev_area_mask
prev_timing
prev_tempocc
prev_opportunity
prev_phase
prev_boundary
prev_chase
```

原因：

- area / timing / tempocc 多数是 route-conditioned 的结果。
- 如果 route 读这些细粒度 previous state，容易形成循环论证或 shortcut。
- route 需要的是 temporal intent memory，不是 previous route-conditioned interpretation。

Semantic transition branch 可以读 richer previous state：

```text
prev_window / prev_dir
prev_area_status
prev_area_route_mask
prev_dist_to_entry / prev_dist_to_exit / prev_time_to_entry
prev_tempocc_bins
prev_go_opportunity / prev_yld_pressure
prev_decision_phase / prev_control_phase
prev_chase_has_lead / prev_chase_speed_max
prev_boundary
```

原因：

- semantic transition 的目标就是学习 semantic state evolution。
- `tempocc -> go_opportunity -> phase` 的时序 shift 很重要。
- 特别是 `yld -> go` 的切换，应该由 previous rich state 和 current scene/route 一起预测。

### Direct / Transition Fuse 的作用

Direct state 与 transition state 应该互补：

```text
direct_state:
    更依赖当前 scene / route evidence
    不容易被 history shortcut

transition_state:
    更稳定
    更容易保持 borrow / dir / tempocc / phase shift 的时间连续性
```

推荐 fuse 方式先做轻量版：

```text
fused_logits = w * transition_logits + (1 - w) * direct_logits
```

其中 `w` 可以来自：

```text
prev_state_valid
per-head learnable scalar
or small gate MLP([direct_conf, transition_conf, prev_valid])
```

第一版建议：

```text
prev_valid = 0:
    use direct_state

prev_valid = 1:
    fuse direct + transition
```

并保留 debug：

```text
direct_state_probs
transition_state_probs
fused_state_probs
transition_gate
```

### Motion 侧仍使用完整 fused state

Motion 不应该只吃一个 compressed semantic token。

更合理的是继续沿用当前 scheduled branch condition 思路：

```text
all fused semantic state
  -> scheduled branch condition
  -> traj/speed decoder
```

Motion condition 应包含：

```text
window
dir
area/status/timing
temporary occupancy / opportunity
decision_phase / control_phase
boundary
chase
borrow_time
```

并保持：

```text
traj_branch_condition_detach = true by default
step-dependent / scheduled injection
optional first_k / once / EMA update schedule
```

这样可以避免：

```text
single semantic token wrong -> traj/speed all wrong
```

也保留了当前 diffusion-step scheduled injection 的鲁棒性。

## 12. Semantic Decoder V2：替代 Current Shared Neck 的 Ablation

当前 shared semantic neck 是从 main path 输出上读：

```text
traj_out / route_out / speed_out / route_geom / conditioning
  -> pooled shared semantic feature
  -> stage1 semantic heads
```

这个结构稳定、简单，但有一个隐患：

```text
all semantic heads share one pooled neck,
model may learn scene/detail shortcuts instead of explicit mode relations.
```

因此可以新增一个 `semantic_decoder_v2`，用 config 与当前 shared neck 对比。

### 目标

不是重写整个 Route-B decoder。

第一版只替换 state prediction 的 readout 结构：

```text
current shared neck
  vs
semantic decoder tokens
```

主 `traj / route / speed` path 先不大改。

更重要的是，它不是单纯“更强的 head”，而是一个 shortcut barrier：

```text
Semantic decoder V2 is not just a stronger head.
It is a shortcut barrier between motion-detail features and semantic-mode prediction.
```

原因：

- 当前 shared neck 直接读 `traj_out / route_out / speed_out`。
- 这些 feature 已经被 expert trajectory / route / speed 强监督。
- state head 很容易从 motion details 里 shortcut 到 label。
- 这会让 state 变成 motion-detail classifier，而不是 semantic-mode reasoner。

我们真正希望 semantic state 学：

```text
window
  -> spatial area / dir / timing
  -> temporary occupancy / opportunity
  -> phase / boundary / chase
```

而不是简单学：

```text
expert traj length / speed shape / local route pattern
  -> phase / window / opportunity
```

### 输入

Semantic decoder 读：

```text
scene / conditioning memory
traj_out
route_out
speed_out
route_geom
optional prev semantic memory
```

它不直接修改 route/traj 主链，只负责输出 semantic state。

为了验证 shortcut 问题，第一版输入源应可配置：

```yaml
route_b:
  semantic_decoder_use_traj_context: false
  semantic_decoder_use_speed_context: false
  semantic_decoder_use_route_context: true
  semantic_decoder_use_route_geom: true
```

推荐 ablation：

```text
route + route_geom + scene only:
    更接近 semantic reasoning，shortcut 少

route + route_geom + scene + traj/speed:
    可能单帧 metric 更高，但更容易学 motion-detail shortcut
```

第一版更建议默认弱化或关闭 `traj_out / speed_out` context。

### Token 设计

推荐先用少量结构化 token：

```text
window token
spatial token
temporal/action token
boundary/chase token
```

其中：

```text
window token:
    outputs window / high-level interaction mode

spatial token:
    outputs dir, conflict_area, area_status, timing

temporal/action token:
    outputs tempocc, go_opportunity, decision_phase, control_phase

boundary/chase token:
    outputs boundary speeds, chase_has_lead, chase_speed_max
```

也可以第一版把 `boundary/chase` 并入 temporal/action token，减少 token 数。

### Attention / Conditioning 第一版

先不做 hard directed mask。

第一版使用：

```text
semantic tokens self-attend
semantic tokens cross/read main context
```

其中 main context 包括：

```text
route_out tokens
traj_out tokens
speed_out token
route_geom tokens
semantic_feature / conditioning
```

如果要轻量体现层次，可以用 residual conditioning：

```text
spatial token gets window token embedding
temporal/action token gets window + spatial embedding
```

但不要一开始把 mask 设计得太死。

### Config 开关

建议新增：

```yaml
route_b:
  use_semantic_decoder_v2: false
  semantic_decoder_num_layers: 2
  semantic_decoder_num_heads: 8
  semantic_decoder_hidden_dim: 512
  semantic_decoder_use_prev_memory: false
```

行为：

```text
false:
    使用当前 shared semantic neck

true:
    使用 semantic decoder tokens 输出 stage1 semantic state
```

注意：

- direct semantic state 可以切到 semantic decoder。
- transition branch 可以后续再切，第一版可仍用当前 transition encoder。
- branch condition / losses / debug key 尽量保持不变，降低迁移风险。

### 与 Direct / Transition Fuse 的关系

推荐组合方式：

```text
direct_state:
    semantic_decoder_v2(obs_t, route_t)

transition_state:
    transition_encoder(prev_state, obs_t, route_t)

fused_state:
    fuse(direct_state, transition_state)
```

所以 semantic decoder v2 主要改善：

```text
current-frame semantic understanding
```

transition branch 主要改善：

```text
frame-to-frame semantic evolution
```

两者不是互斥关系。

### 验收重点

对比 current shared neck：

- window / dir 是否不变差
- area/status/timing 是否更稳定
- tempocc / go opportunity 是否减少跳变
- phase yld/go 是否更少摇摆
- close-loop 中 borrow / merge dir 是否更稳定
- inference latency 增量是否可接受

第一版不追求大幅改分，只要 semantic 更稳、debug 更可解释，就值得继续。
