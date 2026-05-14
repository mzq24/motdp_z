# State Next-Token Collapse Diagnosis

Date: 2026-05-11

## Context

Model/result under discussion:

```text
route_b_b2d_0510_state_next_token_graph_e40_spd_98fuselimit_diff
```

This run uses the state next-token formulation.  Closed-loop debugging showed a
strong route-level collapse in semantic state tokens.

## Main Finding

Many next-token-dependent semantic states appear to be decided by the first few
frames of the route.  Once the early frames fall into one mode, the mode is
copied forward for most or all of the route.

Affected states include:

- `window`: `none / merge / junction / borrow`
- `dir`: `none / same / opposite / cross`
- `area status`: `before / inside / after`
- area occupancy / temporary occupancy bins
- area route mask / area occupancy
- `go/yld` and phase-like states

This is not just a single-route label issue.  It looks like next-token
conditioning has become too strong, so the model overuses previous semantic
tokens and underuses current observation/context.

## Evidence From 0510 Eval

Full-route meta scan output:

```text
/home/z/code/Bench2Drive/codex_analysis/0510_all_routes_window_dir_distribution.csv
```

Global argmax frame counts:

```text
window:
  merge    44529
  junction 12799
  borrow    1249
  none        62

dir:
  opposite 53597
  cross     4004
  none       605
  same       433
```

Route-level main class counts:

```text
main window:
  merge    64 routes
  junction 22 routes
  borrow    2 routes
  none      0 routes

main dir:
  opposite 79 routes
  cross     8 routes
  same      1 route
  none      0 routes
```

So the problem is broader than "borrow is bad":

- `none` almost disappears.
- `dir` collapses heavily to `opposite`.
- many true merge scenes are predicted as `opposite`.
- some non-junction scenes become long stable `junction`.
- once a route enters a token basin, it rarely recovers.

Example route `3410`:

```text
AccidentTwoWays_1
score: 19.77
window_arg: merge for 3968 / 3969 frames
dir_arg: opposite for 3969 / 3969 frames
phase_arg: go for 3969 / 3969 frames
borrow_latched: never
borrow_time: always 0
```

But the scene is borrow-like / opposite-lane obstacle behavior, so this is a
semantic transition failure rather than a low-level control-only failure.

## Hypothesis

The current next-token setup likely trains with too clean / too reliable
previous semantic state.  In closed loop, early prediction errors become the
next input token, and the model learns a copy-prior:

```text
prev_z_t dominates obs_t
```

Then route start becomes dangerous.  If the first 1-2 predicted frames are
`merge + opposite`, later frames keep copying that mode even when current
observation should be `none`, `borrow`, or `junction`.

This creates train/test mismatch:

```text
training: mostly clean GT previous state
closed loop: predicted previous state, with early errors
```

## Required Training Change

The fix should target previous semantic state dependence, not just add ordinary
network dropout.

### 1. Heavy Prev-Token Dropout

Apply dropout to previous semantic tokens during training.

Recommended first ablation:

```text
prev_state_token_dropout = 0.8
prev_window_token_dropout = 0.9
prev_dir_token_dropout = 0.9
prev_area_status_token_dropout = 0.8
prev_area_mask_dropout = 0.8
prev_tempocc_bin_dropout = 0.8
prev_phase_token_dropout = 0.7
```

The exact values are intentionally high.  The current failure suggests small
dropout will not break the copy-prior.

### 2. Prev-Token Noise / Replacement

Do not only mask.  Also corrupt previous tokens:

```text
prev_token_random_replace_prob = 0.2
prev_token_replace_to_none_prob = 0.5
```

For `window`, replacing to `none` is especially important because route starts
and most normal driving should not immediately inherit non-none interaction.

### 3. Prefix Dropout

Route/session prefix is where the collapse starts.  Add special corruption for
the first few linked frames:

```text
first_k_prev_state_dropout_frames = 5 to 10
first_k_prev_state_dropout_prob = 1.0
```

This prevents the first two predicted tokens from deciding the entire route.

Implementation caveat found on new_hpc:

```text
repo: /home/z/code/motdp_z_semantic_state_next_token_rl_v1
config: config/pdm_hpc_route_b_lidar_bev_stage1_tempocc_0429.yaml
dataset: /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_scene_split_95_5_prevstate
```

The config has prefix dropout enabled:

```yaml
semantic_prev_prefix_dropout_frames: 8
semantic_prev_prefix_dropout_prob: 1.0
```

But the policy currently searches only these batch keys:

```text
semantic_route_frame_index
semantic_prev_route_frame_index
route_local_frame_index
route_frame_index
frame_in_route
sample_route_index
```

The packed samples inspected on new_hpc expose:

```text
frame_id
```

and do **not** expose the keys above.  Therefore prefix dropout is likely not
active yet, even though the YAML knob is enabled.  Fix needed:

```text
add frame_id to the prefix-index key candidates
```

or add an explicit `route_frame_index` / `semantic_route_frame_index` field in
the dataset output.

This is important because the observed collapse starts from the first 1-2
frames of a route.

### 4. Scheduled Sampling / Predicted Prev State

If feasible, mix predicted/noisy previous state during training:

```text
prev_state_source:
  GT prev state
  noisy GT prev state
  detached predicted prev state
```

Start simple with noisy GT prev state.  Add predicted prev state later if the
training loop can support it cleanly.

### 5. Observation-Only Anchor

Add or keep an auxiliary current-state prediction path that does not consume
previous semantic state.  This can be used as an anchor loss:

```text
obs_t -> current semantic state
```

The next-token state should improve temporal dynamics, but it must not replace
the current observation as the main evidence.

### 6. Diagnostic Metrics

Add validation diagnostics for collapse:

```text
per-route argmax fraction
per-route state entropy
state flip rate
prev-token copy rate
none recall at route start
window confusion by scenario family
dir confusion by scenario family
```

Useful red flags:

```text
main_window_frac > 0.95 for many routes
main_dir_frac > 0.95 for many routes
none window recall near zero
dir opposite dominating true merge routes
```

## SMART Reference And What We Can Actually Borrow

We checked the local SMART repository:

```text
/media/z/data/mzq/others/SMART
```

SMART is also a next-token model, but its setup is materially different from
ours.  It predicts motion-primitive tokens for all agents, and in its benchmark
setting it can use GT / known agent boxes and their future/past tokenized
geometry during training.  Its next-token rollout is grounded by:

- agent-to-agent graph from known agent positions
- agent-to-map graph from map tokens
- temporal graph over tokenized agent motion
- predicted motion token converted back into geometry before the next step

This is not directly transferable to our closed-loop agent because we do not
have GT boxes for all surrounding agents at inference time.  We should not copy
SMART's full agent-agent construction as if it were available in Bench2Drive
closed loop.

What SMART does still teach us:

1. **Do not trust previous tokens too much.**  SMART masks/corrupts historical
   token context during training.  We should do heavy prev-semantic-token
   dropout/noise.
2. **Prediction must be grounded in current observation.**  SMART re-computes
   graph context from geometry at every rollout step.  For us, the closest
   feasible substitute is an observation-only current-state anchor using BEV /
   route / lidar / available detections, not GT boxes.
3. **Token sequences should not be perfectly deterministic in training.**
   SMART's token matching includes top-k/noisy token choices.  We should add
   semantic prev-token corruption / replacement so the model does not learn a
   clean copy-prior.
4. **Closed-loop rollout should feed back physical state when possible.**
   SMART converts selected token to position/heading before the next step.  For
   us, the semantic token should be checked against current ego pose, route
   progress, traffic-light/stop state, lidar evidence, and conflict geometry.

What we cannot assume:

- GT boxes for all agents in close loop
- GT future agent trajectories
- dense agent-agent graph with oracle positions
- SMART-style multi-agent token rollout as a direct module replacement

Practical adaptation for our setting:

```text
prev semantic token dropout/noise
+ observation-only semantic head
+ lidar/BEV/route-grounded state consistency losses
+ route/area geometry features
+ optional detected-object graph only when detections are available
```

The main lesson is therefore not "build SMART's graph exactly", but:

```text
break the prev-token copy shortcut and force current observation/geometry to
explain semantic state.
```

## Closed-Loop Safety Until Fixed

Until a retrained checkpoint passes the diagnostics, do not trust next-token
semantic state as a hard postprocess driver.

Safer options:

- disable next-token state conditioning for fragile heads in close loop
- or use it only as a soft auxiliary display/debug signal
- keep speed/trajectory heads from being hard-gated by collapsed
  `window/dir/area/phase`

In particular, avoid hard rules that assume:

```text
window=merge + dir=opposite
```

is semantically correct for the whole route.

## Next Debug Set

Use a small route panel before full eval:

```text
3410  AccidentTwoWays_1
25845 AccidentTwoWays_1
2127  OppositeVehicleTakingPriority_1
2143  OppositeVehicleTakingPriority_1
2283  MergerIntoSlowTraffic_1
3048  MergerIntoSlowTraffic_1
3936  SignalizedJunctionLeftTurn_1
4683  SignalizedJunctionLeftTurn_1
```

For each route, inspect the first 20 frames and verify:

- start frames can be `none`
- true merge routes can be `same`
- borrow-like routes can become `borrow`
- junction routes do not force `opposite` everywhere
- area status can move `before -> inside -> after`
- temp-occ bins are not constant over the route

## Exit / Ramp-Like Route Under-Sampling

Follow-up close-loop result:

```text
route_b_b2d_0511_state_next_token_graph_modulation_e20_spd_99fuselimit_diff
```

Several blocked / unfinished routes appear to be exit- or ramp-like geometry
rather than ordinary merge / junction semantic failures.  Example evidence:

- `RouteScenario_23901` (`InterurbanActorFlow_1`) blocked after a layout
  collision with a static pole near the highway shoulder / exit geometry.
- `RouteScenario_24041` (`HighwayExit_1`) produced debug/meta output but did
  not enter the merged records; it ended near almost the same shoulder / pole
  location with speed near zero and throttle high.

Working hypothesis:

```text
main-road driving is over-represented,
exit / ramp entry data is under-represented,
so the model can miss the early ramp/exit geometry and drift toward road edge,
curb, guardrail, pole, or shoulder.
```

This should be treated as a sampling / data coverage issue, not only a
`window/yld/go` tuning problem.

### Candidate Exit-Like Mining Signals

Do not rely only on scenario names.  Use a two-layer miner:

1. Name-based seed routes:

```text
HighwayExit
InterurbanActorFlow
EnterActorFlow
MergerIntoSlowTraffic
SequentialLaneChange
```

2. Geometry / GT based exit-like score:

- future target / route points move laterally from main lane toward ramp or
  road shoulder
- route heading changes strongly over the next 15-40 m
- future route road/lane id changes, if lane topology is available
- GT ego trajectory leaves the main-road lane group and enters a side branch
- current target is still main-road-like but later route points enter a branch,
  meaning the decision must be recognized early
- GT passes close to road boundary / lane boundary / shoulder infrastructure

Sketch:

```text
exit_like =
    scenario_name_match
    OR route_heading_delta_30m > threshold
    OR future_tg_lateral_shift_right_large
    OR route_lane_or_road_id_changes
    OR gt_lateral_shift_right_large_and_sustained
```

For training, oversample the pre-exit decision window rather than the whole
route:

```text
exit entry - 2s to exit entry + 5s
```

or, more generally, the frames where future route geometry already indicates a
ramp/branch but the ego is still on the main road.  This is the rare and
important part; once the vehicle is already on the ramp, the behavior becomes
closer to normal lane following.

### Future Action Items

- Build a route/frame-level `exit_like_score` script using route/tg geometry
  and, where available, GT lane/road id.
- Compare mined exit-like frames against name-based seeds to estimate recall.
- Add sampler weight for pre-exit decision windows.
- Re-run a small route panel containing `23901`, `24041`, and several mined
  false-negative exit-like routes before another full diff eval.

## Route Intent Strength / BridgeDrive Training Reference

Follow-up discussion after inspecting `24041` suggests that the HighwayExit
failure is not only a route-loss-weight issue.  The current route path may be
under-conditioning on target-point intent.

Original concern:

```text
If target point is injected too strongly, route may over-follow planner intent
and lose the ability to handle merge / borrow / junction scene evidence.
```

Current observation:

```text
The route path already receives many other conditions, including semantic state
and previous coarse state.  In particular, prev_route_coarse_memory is projected
and added directly into route tokens, while target point currently enters mostly
through shared ego_status / conditioning.
```

So the actual risk has shifted:

```text
tg / next_tg may be too weak for exit / ramp bifurcations.
```

This is especially plausible for HighwayExit / ramp-like scenes.  These require
the model to commit to a branch early while the ego is still on the main road.
If target-point intent is only a small part of a mixed conditioning vector, the
route branch can drift toward the more common main-road continuation and then
cut into the curb / shoulder / guardrail.

BridgeDrive-style training gives a useful reference:

- The supervised target is directly the geometric route, e.g. `batch['route']`.
- `speed`, `command`, `target_point`, and `target_point_next` are encoded as
  separate planning tokens.
- A query attends to BEV tokens plus these planning tokens to produce an
  `ego_query`.
- The route / planning head then predicts route through an anchor / DDBM-style
  prior.

The important lesson is not to make target point a hard rule.  The lesson is to
make target point an explicit planning token / route-branch input rather than
letting it be diluted inside a shared conditioning vector.

Recommended route-path change:

```text
keep existing conditioning
+ add explicit route planning tokens:
    command
    target_point
    target_point_next
+ let route tokens cross-attend to these tokens, or add a small gated residual
  into route_emb / route_out
```

Training recommendation:

- Do not apply target-point group dropout to the route path, or at least make
  it much weaker than for traj / speed.
- Initialize any new target-point route gate small, e.g. `0.1`, so target point
  becomes audible without becoming a hard geometric override.
- Combine this with exit-like / ramp-like pre-entry oversampling.  Stronger
  target intent alone may still not help if the rare branch frames are not
  sampled enough.

This keeps the intended balance:

```text
tg should anchor route intent,
BEV / semantic state should still decide local feasibility and behavior.
```
