# Semantic State Direct/Transition Fusion V1 Implementation

---

## Code Review Notes (2026-05-10)

审查时发现以下问题：

### 1. [严重] `graph_values` 末尾 4 维 train/inference 不匹配

**Status**: fixed in `policy/annealed_energy_guidance_policy.py`.

Fix summary:

- Training/offline prev-state layout remains unchanged.
- Predicted-cache layout now keeps the last 4 slots as speed-valid flags.
- Because V1 has no separate predicted `*_speed_valid` heads, inference cache uses scalar-head availability (`1` if the corresponding speed scalar exists, else `0`).
- It no longer reuses `current_edge_valid`, `future_edge_valid`, or `chase_has_lead` in those slots.

**问题**：文档描述 `graph_values` (20D) 的 layout 为：

```
current_edge_valid: 1
current_edge_mode_probs: 5
future_edge_valid: 1
future_edge_mode_probs: 5
edge_speed_values_norm: 4   (current_upper, future_lower, front_follow_upper, merge_flow_lower)
edge_speed_valids: 4        (current_upper_valid, future_lower_valid, front_follow_upper_valid, merge_flow_lower_valid)
```

**训练侧**（`_get_semantic_transition_prev_state`）最后 4 维正确：
```python
prev_current_cover_upper_speed_valid,   # 1
prev_future_cover_lower_speed_valid,    # 1
prev_front_follow_upper_speed_valid,    # 1
prev_merge_flow_lower_speed_valid,      # 1
```

**推理侧 predicted-cache**（`_stage1_raw_scores_to_prev_state`，line 2516–2530）最后 4 维**错误**：
```python
current_edge_valid,      # 重复位置1 ← 应为 current_upper_valid
future_edge_valid,       # 重复位置3 ← 应为 future_lower_valid
chase_has_lead,          # 不相关     ← 应为 front_follow_upper_valid
future_edge_valid,       # 再次重复   ← 应为 merge_flow_lower_valid
```

这是 train/inference distribution mismatch。模型在位置 17–20 学到的是 speed valid 信号，predicted-cache 推理时填入的是 edge_valid (重复) 和 chase_has_lead，导致 transition 使用自身 predicted cache 时这 4 个 slot 含义错误。

**代码位置**：`policy/annealed_energy_guidance_policy.py` line 2516–2530。

**修复方向**：在 `_stage1_raw_scores_to_prev_state` 中用模型输出的 speed head + valid 信号（或退化为全 1 valid 代替），而不是重用 edge_valid 和 chase_has_lead。

---

### 2. [文档不准确] `transition_only` 模式描述中 "skips direct loss" 措辞有误

**Status**: clarified in code/logging.

Transition-only now logs:

- `stage1_semantic_next_token_loss`: the current-frame semantic supervision computed from transition outputs.
- `stage1_semantic_direct_aux_loss = 0`: direct aux is not active in transition-only.

The full semantic supervision still exists; only the direct head forward is skipped.

**文档原文**：
> Training skips the direct semantic/graph head forward and direct loss.

**实际行为**：
- direct 头 forward 确实被跳过（`if not transition_only_state:` 分支，line 3058）✓
- 但 `direct_stage1_base_loss` 并没有被"跳过"：在 `transition_only` 模式下，`raw_stage1_scores = transition_stage1_scores`（重命名），所有语义 loss（`direct_stage1_base_loss`）是用 transition 头的输出以**满权重 1.0** 计算的，并非 aux 降权（`semantic_direct_aux_loss_weight=0.25`），也不是 zero。

**代码位置**：`policy/annealed_energy_guidance_policy.py` line 3936–3946（loss 汇总处）：
```python
else:  # transition_stage1_scores is None (= transition_only mode)
    stage1_loss = (
        direct_stage1_base_loss          # 实际是 transition 输出，满权重
        + self.state_consistency_loss_weight * loss_state_consistency
    )
```

**建议措辞**：
> Training skips the direct head forward. Semantic supervision (`direct_stage1_base_loss`) is computed from transition outputs at full weight (not downweighted by `semantic_direct_aux_loss_weight`).

---

### 3. [文档冗余] `transition_only` 模式下推荐 config 中 `use_semantic_state_fusion` 是多余的

**Status**: cleaned up in current configs.

For `semantic_state_predictor_mode: transition_only`, configs now set:

```yaml
use_semantic_state_fusion: false
semantic_state_fusion_update_cache: true
```

Cache update remains enabled because the transition predictor still needs the previous predicted state.

**文档推荐**：
```yaml
semantic_state_predictor_mode: transition_only
use_semantic_state_fusion: true
semantic_state_fusion_alpha: 1.0
```

**实际代码**（line 4469–4472）：
```python
use_infer_semantic_transition = (
    (self.use_semantic_state_fusion or transition_only_state)
    and ...
)
```

`transition_only_state=True` 时，`use_semantic_state_fusion` 的值对 `use_infer_semantic_transition` 没有影响。且在 `transition_only` 推理路径中 `_fuse_stage1_raw_scores` 不会被调用，fusion gate 直接设 1.0（line 4556–4558）。

**建议**：推荐 config 去掉 `use_semantic_state_fusion` 和 `semantic_state_fusion_alpha` 的冗余设置，或加注释说明它们在 `transition_only` 下不起作用。

---

### 4. [已验证] 其余文档描述与代码一致

- config 开关（`use_semantic_state_fusion`, `semantic_state_fusion_alpha`, `semantic_state_fusion_update_cache`, `semantic_state_predictor_mode`）均存在，默认值一致 ✓
- helper 函数全部存在（`reset_semantic_state_cache`, `_build_neutral_semantic_prev_state`, `_semantic_state_cache_to_device`, `_get_inference_semantic_prev_state`, `_stage1_raw_scores_to_prev_state`, `_fuse_stage1_raw_scores`）✓
- `semantic_state_reset > 0.5` 触发 cache 清空 ✓
- gate 公式 `gate = alpha * prev_valid` ✓
- `transition_only` 时 direct 头 forward 跳过，state_consistency 关闭 ✓
- `graph_token` 注入 `timing_token + boundary_token + chase_token`，model 中 `semantic_transition_graph_proj(20 → n_emb)` ✓
- 中性 prev state 中 `graph_values = zeros(B, 20)` ✓
- 第 1 帧使用中性 prev state（cache 为 None 时 `_get_inference_semantic_prev_state` 返回 neutral）✓
- May 10 验证数字已记录在文档中（window F1 / merge recall / go opportunity CE 等）✓

---

## Context

This note records the first implementation of inference-side fusion between:

- direct graph state prediction from the current frame
- semantic transition / next-token state prediction from previous fused state

The goal is to let the transition branch actually affect closed-loop inference instead of only acting as a training auxiliary loss.

## Naming Clarification

The code still uses historical names such as:

- `shared_stage1`
- `raw_stage1_scores`
- `compute_shared_stage1_from_ego_outputs`

In the current semantic-state branch, these names should be interpreted as:

- `shared_stage1` = direct semantic/graph state heads attached to main Route-B features
- `raw_stage1_scores` = raw state head outputs, e.g. logits and normalized scalar predictions
- `transition_stage1_scores` = next-token semantic state outputs from previous semantic state plus current context

Here, "scores" does not mean reward or RL score. It is just a dict of raw model outputs.

## Implemented Behavior

New config keys:

```yaml
route_b:
  use_semantic_state_fusion: false
  semantic_state_fusion_alpha: 0.35
  semantic_state_fusion_update_cache: true
```

Default is off to preserve current baseline behavior.

When enabled during inference:

1. At episode/cache start, previous semantic state is neutral and `prev_valid=0`.
2. Direct graph state is predicted from current main-path features.
3. Transition state is predicted from cached previous fused state plus current context.
4. The raw state outputs are fused with a fixed gate:

```text
gate = semantic_state_fusion_alpha * prev_valid
fused = (1 - gate) * direct + gate * transition
```

5. Fused state is used to build traj/speed branch condition.
6. After final denoise, fused state is cached as the previous semantic state for the next frame.

First frame behavior:

```text
prev_valid = 0
gate = 0
fused = direct
```

Later frame behavior:

```text
prev_valid = 1
gate = semantic_state_fusion_alpha
fused = direct/transition blend
```

## Code Touch Points

Policy file:

```text
policy/annealed_energy_guidance_policy.py
```

Added helpers:

```text
reset_semantic_state_cache()
_build_neutral_semantic_prev_state(...)
_semantic_state_cache_to_device(...)
_get_inference_semantic_prev_state(...)
_stage1_raw_scores_to_prev_state(...)
_fuse_stage1_raw_scores(...)
```

Inference changes:

- `conditional_sample(...)` now optionally builds a previous semantic state from cache.
- During each branch-condition pass, direct graph scores are optionally fused with transition scores.
- Final `stage1_scores` are also optionally fused.
- The final fused raw state is converted back into a `prev_state` cache for the next frame.

Debug outputs:

```text
semantic_state_fusion_enabled
semantic_state_fusion_gate
```

Reset hook:

- `predict_action(..., reset_semantic_state_cache=True)` clears the cache.
- If obs contains `semantic_state_reset > 0.5`, cache is also cleared.

## What This Does Not Change

Training loss is unchanged:

- direct graph state still has current-frame supervision
- transition state still has current-frame supervision
- direct/transition consistency still exists

This implementation does not add:

- learnable fusion gates
- train-time fused branch condition
- state cache persistence outside the policy object
- object grounding
- RL / GRPO state optimization

## Prev-State Token Coverage Principle

The May 10 transition-head validation clarified an important modeling rule:

```text
next-token semantic transition requires explicit previous tokens for every
state dimension whose frame-to-frame shift we want the model to learn.
```

In other words, the transition branch is not magic memory. It can only learn
autoregressive state evolution for fields that are represented in
`S_{t-1}` or are recoverable from strongly correlated previous fields plus
current context.

Current transition formulation:

```text
S_{t-1}^{compressed} + X_t -> S_t
```

where `X_t` is current BEV / route / ego / trajectory-speed context, and
`S_{t-1}^{compressed}` currently contains a hand-defined semantic token set:

- previous window/family
- previous direction
- previous decision/control phase
- previous area status and route mask
- previous temporary occupancy bins
- previous go/yield opportunity probabilities
- previous timing values
- previous boundary speeds
- previous chase state

This is already a valid next-token / AR formulation. The missing piece is that
it is not yet a full structured semantic token: previous graph relation fields
are not explicitly included.

### Fields That Should Be Explicit Previous Tokens

If we expect the model to learn a temporal shift, the field should enter the
previous-token state explicitly.

High-priority examples:

- `window/family`: merge / junction / borrow / none has strong temporal
  continuity and entry/exit dynamics.
- `phase`: yield/go/control phase is inherently a state transition.
- `go/yield probability`: opportunity often ramps up or down before the hard
  phase label changes.
- `temporary occupancy`: occupancy bins are a temporal cover process.
- `area status/timing`: approach / inside / past and time-to-entry/exit are
  continuous temporal variables.
- `boundary speeds`: yld/go speed bounds should move smoothly over time.
- `chase state`: front-follow state and cap should persist across adjacent
  frames.
- `graph relation state`: current/future edge valid, edge mode, and graph speed
  bounds should be explicit if we want graph relation transition rather than
  indirect reconstruction from coarse semantic fields.

### Fields That Can Stay Frame-Internal

Some fields are primarily intra-frame relational structure. They can still be
predicted by the transition branch, but they do not necessarily need a separate
previous-token input unless we want temporal smoothing / AR evolution for them.

Examples:

- current route-query residual features
- current BEV evidence
- current relation-graph decoding from timing / boundary / chase tokens
- one-frame geometric compatibility features

These are better viewed as `X_t` or current-frame graph decoding context, not
as previous semantic memory.

### Graph Is Not Mutually Exclusive With Next Token

The right decomposition is:

```text
next-token = temporal prediction rule
graph = structured relation fields inside the semantic state
```

So the state is:

```text
S_t = {
  flat semantic fields,
  relation graph fields,
}
```

and transition predicts:

```text
S_{t-1} + X_t -> S_t
```

The graph fields are part of `S_t`. They are not a competing state family.
Direct and transition can both output graph fields, but the May 10 validation
shows that the transition version is much stronger when given clean previous
state.

### Current Missing Previous Graph Token

Current transition V1 predicts graph outputs but does not explicitly read these
previous graph fields:

- `prev_current_cover_edge_valid`
- `prev_current_cover_edge_mode`
- `prev_future_cover_edge_valid`
- `prev_future_cover_edge_mode`
- `prev_current_cover_upper_speed`
- `prev_future_cover_lower_speed`
- `prev_front_follow_upper_speed`
- `prev_merge_flow_lower_speed`

This explains the current design:

```text
prev coarse semantic state + current context -> graph_t
```

It is still next-token prediction, but graph transition is learned indirectly
through previous window/phase/area/timing/boundary/chase fields.

The stronger next version should be:

```text
prev coarse semantic state
+ prev graph relation state
+ current context
    -> current structured semantic state
```

Implementation sketch:

```text
window_token
dir_token
phase_token
area_token
tempocc_token
timing_token
boundary_token
chase_token
graph_token   # new: previous graph valid/mode/speed-boundary memory
```

### Design Implication

For future state additions, first classify the field:

```text
Does it need frame-to-frame shift?
    yes -> add it to prev_state token input and supervise current output
    no  -> keep it as current-frame context/head/graph decoding only
```

This avoids a half-AR design where some temporal fields are forced to be
reconstructed indirectly. It also gives a cleaner mental model:

```text
compressed semantic next-token prediction
    -> current implementation

full structured semantic next-token prediction
    -> desired next version with explicit prev graph token
```

## May 10 Transition-Head Validation Note

A full validation on `dit_policy_epoch30.pt` showed that the transition head is
substantially stronger than the direct head when using offline previous semantic
state:

```text
window active F1: direct 0.338, transition 0.630
merge recall:     direct 0.375, transition 0.960
borrow recall:    direct 0.452, transition 1.000
decision go R:    direct 0.856, transition 0.986
go opportunity CE: direct 0.546, transition 0.139
graph current valid F1: direct 0.269, transition 0.488
graph future valid F1:  direct 0.388, transition 0.482
```

This supports the hypothesis that semantic state evolution should primarily be
modeled as next-token prediction. Direct state should remain useful as a
cold-start / reset / drift-anchor path, but the main closed-loop state should
move toward transition output after predicted-cache validation.

## Direct vs Transition Structural Decision

The direct and transition branches predict the same semantic state schema, but
they represent different modeling assumptions.

Direct branch:

```text
X_t -> S_t
```

- Uses only current-frame Route-B context.
- Predicts window / phase / go opportunity / timing / chase / graph fields.
- Has no explicit temporal state memory.
- In practice it behaves like a one-frame classifier and tends to become
  conservative on window state, especially by over-predicting `none`.

Transition branch:

```text
S_{t-1} + X_t -> S_t
```

- Uses previous semantic state plus current-frame context.
- Predicts the same output fields as direct, including relation-graph fields.
- Matches the actual problem structure: window, phase, opportunity, occupancy,
  timing, boundary, chase, and graph relations all have strong frame-to-frame
  continuity.
- May 10 validation shows it is much stronger than direct when given clean
  previous semantic state.

Important interpretation:

```text
direct vs transition = predictor choice
graph = structured fields inside S_t
```

So the comparison is not "direct state vs graph state". Both branches can output
graph fields. The decision is that the state predictor should be transition /
next-token first.

### Decision

Going forward, direct should no longer be treated as the main semantic-state
path or as a route worth optimizing for closed-loop behavior.

The intended hierarchy is:

```text
primary state predictor:
    transition / next-token semantic state

secondary fallback:
    direct state only for cold start, reset, invalid prev_state, and debugging
```

Consequences:

- Best-checkpoint selection should prioritize transition metrics and
  predicted-cache transition rollout metrics, not direct metrics.
- Validation should report `transition_*` and `pred_cache_transition_*` as the
  main state quality numbers.
- `direct_*` metrics can remain for diagnosis, but a bad direct window recall is
  no longer a blocker if transition/pred-cache transition is stable.
- Fusion should be treated as a temporary debugging bridge only, not the target
  architecture.
- Once predicted-cache transition is validated, closed-loop condition should use
  transition output directly.

Recommended near-term inference setting:

```yaml
use_semantic_state_fusion: true
semantic_state_fusion_alpha: 1.0
semantic_state_fusion_update_cache: true
```

With `alpha=1.0`, fusion degenerates into transition-only whenever
`prev_valid=1`; direct only handles the first frame / reset case where no
previous semantic token exists.

The operational model should be:

```text
prev transition state cache + current context -> current transition state
```

with direct used only when `prev_valid=0` or when an explicit reset happens.

## Recommended Ablation

The main ablation is no longer a direct/transition fusion-alpha sweep. It should
be a transition-cache robustness check:

- `offline_prev_transition`: transition with GT/offline previous state.
- `pred_cache_transition`: transition with its own previous predicted state.
- `direct_first_frame_only`: direct only for `prev_valid=0` / reset.

Expected benefit:

- more stable window/dir/phase over adjacent closed-loop frames
- less borrow/merge identity flicker
- transition branch becomes useful at inference, not only as auxiliary training regularizer

Main risk:

- stale cache across episode boundaries if the agent does not reset policy state
- predicted-cache drift if transition repeatedly consumes its own imperfect
  previous state
- direct first-frame output still needs to be reasonable enough to bootstrap the
  transition cache

Mitigation:

- call `reset_semantic_state_cache()` or pass `reset_semantic_state_cache=True` at route/episode start
- add predicted-cache transition validation before relying on transition-only closed-loop behavior

## May 10 Follow-Up Implementation

After reviewing validation results, the implementation was updated from
"direct/transition fusion" to a transition-first path.

### Transition-Only Predictor Mode

New config:

```yaml
route_b:
  semantic_state_predictor_mode: transition_only
  use_semantic_state_fusion: true
  semantic_state_fusion_alpha: 1.0
  semantic_state_fusion_update_cache: true
```

Behavior:

- Training skips the direct semantic/graph head forward and direct loss.
- Training uses transition / next-token outputs as the supervised semantic
  state prediction.
- Direct/transition consistency is disabled in transition-only mode.
- Inference skips direct cold-start.
- First frame uses a neutral previous semantic token:

```text
neutral S_{t-1} + X_t -> S_t
```

- Later frames use the predicted semantic cache:

```text
cached predicted S_{t-1} + X_t -> S_t
```

This removes direct from the main state path. Direct remains in code only for
`direct_transition` ablation/debug mode.

### Explicit Previous Graph Memory

Transition input now includes a compact previous graph memory vector:

```text
graph_values: 20 dims
```

Layout:

```text
current_edge_valid: 1
current_edge_mode_probs: 5
future_edge_valid: 1
future_edge_mode_probs: 5
edge_speed_values_norm: 4
edge_speed_valids: 4
```

Offline previous labels are read when available:

```text
prev_current_cover_edge_valid
prev_current_cover_edge_mode
prev_future_cover_edge_valid
prev_future_cover_edge_mode
prev_current_cover_upper_speed_mps
prev_future_cover_lower_speed_mps
prev_front_follow_upper_speed_mps
prev_merge_flow_lower_speed_mps
prev_current_cover_upper_speed_valid
prev_future_cover_lower_speed_valid
prev_front_follow_upper_speed_valid
prev_merge_flow_lower_speed_valid
```

Predicted-cache inference builds the same `graph_values` from previous predicted
state.

To preserve checkpoint compatibility, the model does not resize
`semantic_transition_slot_embed`. Instead it projects the 20D graph vector and
injects it into relation-bearing transition slots:

```text
timing_token   += graph_token
boundary_token += graph_token
chase_token    += graph_token
```

This gives the next-token branch explicit previous relation-graph state without
breaking old checkpoints via slot shape mismatch.

### Expected Runtime Effect

Compared with direct/transition mode, transition-only should save some semantic
head compute:

- no direct graph-state head forward in stage1 training
- no direct semantic consistency view in transition-only mode
- no direct graph-state inference path

It does not remove the main Route-B diffusion forward, so the speedup is
expected to be moderate rather than dramatic. The bigger benefit is structural:
the model now optimizes and deploys the stronger next-token state predictor
instead of carrying a weaker direct predictor in the main path.
