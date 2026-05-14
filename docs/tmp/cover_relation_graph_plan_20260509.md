# Area-Cover Relation Graph + Compact Semantic Conditioning Plan

Date: 2026-05-09

Worktree:

`/media/z/data/mzq/others/MoT-DP-worktrees/semantic_state_next_token_rl_v1`

## Summary

Upgrade the current shared semantic neck into an **area-conditioned ego-cover relation graph decoder**, and narrow motion/route semantic conditioning to more causally relevant state subsets.

Core decisions:

- Do not implement a separate traditional `Semantic Decoder V2`; the graph decoder is the new semantic decoder.
- Motion should only receive compact causal state, not the full auxiliary semantic state.
- Route should only receive previous coarse semantic memory: previous `window/dir/borrow`-like memory.
- Speed consistency should use edge-aware labels, not the old global `phase=go -> go_min` rule.

## Motivation

The previous independent-state design had many useful semantic heads, but too many states were injected directly into traj/speed. This risks making motion sensitive to auxiliary predictions such as `dir`, `area`, `timing`, and `temporary occupancy` even when those heads are mainly there to help semantic reasoning.

The phase/boundary mismatch problem also suggested that a single global boundary target is not expressive enough. Current cover and future cover impose different constraints:

- Current cover mainly says ego cannot pass before it; ego generally has to pass after current cover.
- Future cover creates the actual opportunity/yield choice: go before future cover or yield after it.
- Front-follow and merge-flow constraints are separate speed constraints and should not be collapsed into `phase`.

Therefore the graph view explicitly separates cover edges and their speed constraints.

## Key Changes

### 1. Dataset / Label Interface

The model consumes new edge-aware fields after relabel/project/split:

```text
current_cover_edge_valid
current_cover_edge_occupied
current_cover_edge_mode
current_cover_edge_mode_valid
current_cover_upper_speed_mps
current_cover_upper_speed_valid
current_cover_upper_speed_source

future_cover_edge_valid
future_cover_edge_mode
future_cover_edge_mode_valid
future_cover_lower_speed_mps
future_cover_lower_speed_valid
future_cover_lower_speed_source

front_follow_upper_speed_mps
front_follow_upper_speed_valid
merge_flow_lower_speed_mps
merge_flow_lower_speed_valid
```

If transition is enabled, previous semantic state is still read from the existing `prev_*` fields. If relabel does not yet provide `prev_*` edge fields, transition V1 should still run by using the existing previous semantic state and supervising current graph heads.

### 2. Graph Semantic Decoder

Add config:

```yaml
route_b:
  use_cover_relation_graph_decoder: true
```

Direct graph decoder default input:

```text
route_out
route_geom
conditioning
ego_status / current speed
```

Default behavior should avoid reading motion details:

```yaml
cover_graph_use_traj_context: false
cover_graph_use_speed_context: false
```

The graph decoder outputs the existing semantic heads:

```text
window / dir / decision_phase / control_phase
conflict_area_logits
conflict_area_status / timing
temporary_occupancy / go_opportunity
family yld/go raw boundary
chase
```

It also outputs new graph heads:

```text
current_cover_edge_valid_logit
current_cover_edge_mode_logits
future_cover_edge_valid_logit
future_cover_edge_mode_logits
current_cover_upper_speed
future_cover_lower_speed
front_follow_upper_speed
merge_flow_lower_speed
```

V1 does not add raw BEV grid sampling or deformable attention. It uses the existing Route-B decoder context first.

### 3. Transition / Next-Token State

Extend the existing semantic transition branch so it predicts the same graph heads.

Transition input remains:

```text
prev semantic state + current route/context
```

Current planned previous state support:

```text
prev_window / prev_dir
prev_area/status/timing
prev_tempocc/opportunity
prev_phase
prev_boundary/chase
optional prev_current/future_edge fields
```

Loss policy:

- Direct graph heads supervise current labels.
- Transition graph heads supervise current labels.
- Direct/transition soft consistency is preserved.
- Hard temp-occ shift loss is not added in this pass, but can remain as a future config placeholder.

### 4. Compact Motion Condition

Replace current full-state branch condition with a compact graph mode:

```yaml
route_b:
  semantic_motion_condition_mode: compact_graph
```

Compact condition includes:

```text
window_probs
decision_phase_probs
control_phase_probs
go_opportunity / yld_pressure

current_edge_valid + current_edge_mode_probs
future_edge_valid + future_edge_mode_probs

edge speed margins:
  current_upper_margin
  future_lower_margin
  front_follow_upper_margin
  merge_flow_lower_margin

edge speed valid flags
borrow_time
```

Do not directly inject:

```text
dir
conflict_area_logits
area_status
timing
temporary_occupancy_bins
raw family yld/go boundary
```

These remain supervised semantic states for graph reasoning, but they should not directly perturb traj/speed.

Margin convention:

```text
upper_margin = (upper_speed - current_speed) / scale
lower_margin = (current_speed - lower_speed) / scale
invalid margin = 0
all margins clamp to [-1, 1]
```

### 5. Route Previous Coarse Memory

Add config:

```yaml
route_b:
  use_route_prev_coarse_memory: true
```

Route decoder only reads:

```text
prev_semantic_state_valid
prev_window / prev_family
prev_dir
optional prev_borrow_latch or prev_borrow_time summary
```

Route decoder should not read:

```text
prev_area
prev_tempocc
prev_phase
prev_boundary
prev_chase
```

Implementation idea:

- Encode prev coarse memory with an MLP.
- Add it to `route_conditioning` or route tokens.
- Only affect route tokens, not traj tokens directly.

This should help borrow/merge route continuity without current-state circular reasoning.

### 6. Inference Fuse / Cache [V2 / Not in V1]

> **Status**: Not implemented in V1. Current inference uses direct graph state only for branch condition. Fuse/cache planned after direct graph training is stable.

Add inference semantic state cache:

```text
episode start:
  prev_state_valid = 0
  prev_state = neutral

each frame:
  direct_state = graph_decoder(current context)
  transition_state = transition_decoder(prev_state, current context)
  fused_state = gate(direct_state, transition_state)
  compact_condition = compose(fused_state)
  cache fused_state as prev_state
```

Default fuse:

```text
prev_valid=0 -> direct only
prev_valid=1 -> learnable per-head gate between direct and transition
```

Debug outputs:

```text
direct_*_probs
transition_*_probs
fused_*_probs
transition_gate
compact_motion_condition
```

This cache/fuse can be implemented after direct graph training is stable if necessary.

## Losses

New graph losses:

```text
current_edge_valid_loss: BCE
current_edge_mode_loss: CE masked by current_cover_edge_mode_valid
future_edge_valid_loss: BCE
future_edge_mode_loss: CE masked by future_cover_edge_mode_valid

current_cover_upper_loss: SmoothL1 masked by current_cover_upper_speed_valid
future_cover_lower_loss: SmoothL1 masked by future_cover_lower_speed_valid
front_follow_upper_loss: SmoothL1 masked by front_follow_upper_speed_valid
merge_flow_lower_loss: SmoothL1 masked by merge_flow_lower_speed_valid
```

Optional edge-aware motion consistency:

```yaml
route_b:
  use_edge_speed_consistency_loss: false
```

If enabled:

```text
front_follow_upper_valid -> predicted_speed <= front_follow_upper
merge_flow_lower_valid -> predicted_speed >= merge_flow_lower
current_cover_upper_valid -> predicted_speed <= current_cover_upper
future_cover_lower_valid + selected/predicted mode go_before_future
  -> predicted_speed >= future_cover_lower
```

Do not use old global rules:

```text
phase=go -> speed >= family_go_min
phase=yld -> speed <= family_yld_max
```

## Test Plan

### Dataset Smoke

- Relabeled batch contains all edge fields.
- Shapes and dtypes are correct.
- Missing graph fields fail fast when graph decoder is enabled.
- Missing prev edge fields should not block V1 unless a future config explicitly requires them.

### Train Smoke

- `use_cover_relation_graph_decoder=false` keeps old shared neck path runnable.
- `use_cover_relation_graph_decoder=true` single batch forward/backward is finite.
- Graph losses, transition losses, and compact condition are finite.
- Invalid speed edges produce zero scalar loss.

### Infer Smoke

- Neutral semantic cache start is safe.
- Direct/transition/fused debug can be emitted once fuse is implemented.
- Compact condition dimension matches model projection.
- Route prev coarse memory does not change traj branch condition dim.

### Behavior Check

- Removing direct `dir/area/tempocc` motion injection should reduce traj/speed jitter.
- Borrow/merge route continuity should improve with previous coarse route memory.
- Edge modes should distinguish (5-way enum, both current and future edge):
  - `0: none` — no cover edge applicable
  - `1: pass_after_current` — current cover is present, ego must pass after it clears
  - `2: go_before_future` — ego can go before future cover arrives
  - `3: yield_after_future` — ego must yield until future cover passes
  - `4: ambiguous` — conflicting or unclear timing
- Edge speed constraints should avoid phase-boundary single-ref mismatch.

## Assumptions

- Worktree: `/media/z/data/mzq/others/MoT-DP-worktrees/semantic_state_next_token_rl_v1`
- V1 does not implement object grounding, bbox, actor tracking, BEV grid sampling, confidence, uncertainty, passability bins, or boundary bins.
- Motion condition defaults to compact graph mode for the HPC graph experiment.
- Traditional `Semantic Decoder V2` is not implemented as a separate module; graph decoder replaces it.
- Route memory only uses previous coarse semantic state.
