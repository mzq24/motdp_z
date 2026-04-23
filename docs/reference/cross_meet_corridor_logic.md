# Stage1 Speed Cross / Merge Logic

Last updated: 2026-04-13

This note records the current stage1 speed interaction logic used by:

- [tools/generate_front_route_label_video.py](/media/z/data/mzq/others/MoT-DP/tools/generate_front_route_label_video.py)
- [scripts/data_tools/precompute_semantic_labels.py](/media/z/data/mzq/others/MoT-DP/scripts/data_tools/precompute_semantic_labels.py)

As of `2026-04-13`, the `video` path and the `precompute` path are aligned on
the same mainline logic.

The goal is to avoid mixing up four different things:

1. generic route-cover detection
2. scene-level two-way corridor localization
3. generic merge-episode parsing
4. historical wait/release-time corridor checking

## 0. Boundary that must stay explicit

`junction_left_cross_meet` and `borrow_cross_meet` are similar in timing logic,
but they are not the same geometry system.

- `junction_left_cross_meet`
  - uses generic cross-conflict geometry from route cover detection
  - does not use the scene-level two-way borrowed-lane corridor
- `borrow_cross_meet`
  - uses the scene-level two-way corridor built from blocker + context-frame
    route
- they do share:
  - current cover / future cover semantics
  - `cross_safe_gap_s`
  - cross-wait timing ideas
- they do not share:
  - corridor localization
  - blocker-route geometry
  - two-way start/end estimation

So the two-way corridor should never change the geometry of junction cross
meet; it only provides extra structure for `borrow_cross_meet`.

## 1. Which scenes are treated as borrow-cross

`borrow_cross_meet` can appear for:

- `ConstructionObstacleTwoWays`
- `AccidentTwoWays`
- `ParkedObstacleTwoWays`

Special scene-level corridor localization is only implemented for:

- `ConstructionObstacleTwoWays`
- `AccidentTwoWays`

That split matters:

- `borrow_cross_meet` is the interaction subtype
- scene-level corridor localization is only the geometry helper used for the
  two-way borrowed-lane corridor

## 2. High-level cross-meet flow

The current pipeline is:

1. run generic front-route cover detection on the route corridor
2. classify current/future cover interaction type
3. if the interaction is cross-direction and the event name belongs to the
   borrow-cross scene set, convert that meet subtype to `borrow_cross_meet`
4. for `borrow_cross_meet`, attach a borrowed-lane corridor and compute
   borrow-specific speed logic such as `borrow_yld` / `borrow_go`

So:

- `borrow_cross_meet` does not come from the corridor locator directly
- it comes from generic cover detection + event-name routing
- the corridor locator only tells us where the borrowed lane starts and ends

## 3. Route used for two-way corridor work

For two-way scenes, the code uses the extended route:

- `_route_corridor_input_local`

This is usually:

- the front route in local coordinates
- plus the scene polyline extension
- effectively the `20 + 12` point route

If scene polyline extension is missing for a two-way corridor scene, the code is
expected to fail instead of silently falling back.

### Global route-extension rule

For stage1 route-based geometry, the default rule should be:

- use the extended route, not the raw planner front route
- in practice this means `front route + extra 12 points`
- effectively the `20 + 12` point route

This should apply whenever the code is doing route-geometry work such as:

- corridor start / end localization
- conflict-area world-geometry localization
- route-based direction / heading estimation
- route-progress based collision / cover localization

If a module reads the raw `sample['route']` 20-point route directly for one of
the geometry tasks above, that should usually be treated as a bug unless the
code path is explicitly marked as an exception.

Current documented exception:

- `HazardAtSideLane` local video validation
- for that debug-only view, do **not** append the extra `12` route points
- use the raw 20-point route only, because route reshaping / rerouting near the
  event window can make the extension misleading for qualitative inspection
- this exception is for local video validation only, not the main label logic

### Debug note: `borrow_cover_shape_mismatch`

`borrow_cover_shape_mismatch` is currently a debug-only issue, not a mainline
borrow-window gate.

Observed failure mode:

- many `borrow_cover_shape_mismatch` routes are early-borrow routes
- in those scenes, the local route used by the route-shape checker does not
  extend far enough to include the "return to lane" part of the corridor
- in practice this usually means the available local route only shows the
  "merge out / borrow enter" heading change, but not the later "return / merge
  back" heading change
- the shape checker then sees something more monotonic / junction-like and
  records `borrow_cover_shape_mismatch`

Interpretation:

- this issue often means the local route-shape debug view is truncated
- it does **not** necessarily mean the scene-level two-way corridor or the
  borrow conflict window is wrong
- it should be treated as a route-shape visibility limitation first, especially
  for early-borrow samples

Impact on current labels:

- this issue is attached after the borrow window start / end has already been
  resolved
- it does **not** change `borrow` family selection
- it does **not** change `conflict_area_active`
- it does **not** move borrow window start / end by itself
- it mainly shows up as debug `missing_reason` / `issue_count`

So a long pre-borrow corridor or an incomplete local route-shape view may make
the debug issue fire, but by itself should not confuse the active borrow window.
The main risk is qualitative debugging noise, not label-family drift.

## 4. Preferred corridor localization: scene-level blocker-route context

This is the logic in `_build_event_two_way_borrow_context(...)`.

### 4.1 Build route candidates

Loop over all frame records and keep frames that satisfy:

- command is `LANE_FOLLOW`
- the route shows a usable borrow pattern

Borrow-pattern detection is done from the route itself with
`_estimate_borrow_points_from_signed_route_shift(...)`.

Current thresholds:

- `borrow_enter_lateral_thresh = 1.25 m`
- `return_lateral_thresh = 0.8 m`
- `min_enter_progress_m = 4.0 m`
- `min_return_progress_m = 6.0 m`

The signed-shift detector works like this:

1. prepend route origin
2. estimate baseline lane center from the first few route points
3. compute signed lateral offset along the route
4. find the first point where lateral shift exceeds the borrow-enter threshold
5. find a later point where the route comes back near the lane center

At this stage, the route is only being used to say:

- this frame contains a likely borrow maneuver
- this is the shift direction

It is not yet the final corridor.

### 4.2 Pick the blocker actor

For the earliest / best route candidate, scan current boxes and keep only
blockers that are valid for the event:

- `ConstructionObstacleTwoWays`: only `static`
- `AccidentTwoWays`: only stopped vehicles
  - allowed classes: `car`, `truck`, `bus`, `van`, `vehicle`
  - speed must be `<= 0.25 m/s`

Additional blocker filters:

- must be ahead of ego: `local_x > 0`
- must still be near the original lane center:
  `abs(local_y) <= blocker_pre_shift_lateral_thresh`
- current threshold:
  `blocker_pre_shift_lateral_thresh = 1.75 m`

Among those boxes, choose the closest valid one in front.

This becomes the `seed_actor`.

### 4.3 Pick the context frame

Now go back over all borrow-route candidates and look for the same actor id.

For frames where the same blocker actor is still valid:

- compute score `(local_x, abs(local_y), -frame_id)`

Meaning:

- prefer the nearest blocker in front
- then prefer a blocker closest to the route center
- then prefer the later frame if the first two are tied

The winning frame is the `context frame`.

Important:

- this `context frame` is the frame whose route and blocker pose are used to
  build the scene-level corridor
- `route_head` in current code means the route head of this `context frame`,
  not the route head of every displayed frame
- because the chosen frame prefers:
  - the same blocker actor
  - small positive blocker `x`
  - small blocker `|y|`
  it often lands near the moment when ego is just about to enter the corridor
- in that sense, `context_frame_id` is often close to the geometric `go` / entry
  point for the borrowed lane

### 4.4 Compute corridor start

Use the `context frame` route:

- `route_local_context`
- then prepend origin
- then compute arclength `arc_context`

The code currently supports two start modes.

#### `route_head`

Current default in code:

- `route_start_s = arc_context[route_head_idx]`

Meaning:

- corridor start is the first actual point of the context-frame route
- this is not the prepended origin

#### `obstacle_align`

Alternative mode:

1. find route points near original lane center:
   `abs(y) <= TWOWAY_START_LATERAL_THRESH_M`
2. among them, choose the point whose `x` is closest to the blocker `x`

Current threshold:

- `TWOWAY_START_LATERAL_THRESH_M = 0.5 m`

Meaning:

- start is anchored to where the blocker sits longitudinally, while still
  staying on the unshifted part of the route

### 4.5 Compute corridor end

The current end logic is:

1. compute lateral-shift statistics on the full extended route
   - front route + extra `12` points
   - effectively the `32`-point route
2. find the borrow peak first
3. from that peak onward, find the recovery-side extremum
   - for the current left-borrow two-way scenes, this is effectively the
     right-most route point after the borrow peak
4. use that extremum as `end`
5. if this statistic is missing or invalid:
   - fall back to the extra-route tail
   - use the 10th point inside the extra `12` route points

Current fallback setting:

- `TWOWAY_RETURN_FALLBACK_EXTRA_POINT_INDEX = 10`

Meaning:

- the preferred `end` is still route-shift based
- the extra-tail fallback is only a deterministic backup when the threshold
  return is not found

### 4.6 Build the final scene-level corridor

Sample the route from:

- `route_start_s`
- to `route_return_s`

with `route_step_m`, then transform that segment to world coordinates.

This produces a fixed scene-level corridor with fields such as:

- `borrow_start_world_xy`
- `borrow_end_world_xy`
- `borrow_distance_m`
- `context_frame_id`
- `anchor_actor_id`
- `blocking_actor_id`

This is the cleanest corridor definition for two-way scenes.

### 4.7 Expected current-follow-chase start gate cases

`merge_start_blocked_by_current_follow_chase` should not automatically be read
as a merge failure.

A checked group on `2026-04-19`:

- `AccidentTwoWays/Town12_Rep0_26_0_route0_11_08_18_12_42`
- `AccidentTwoWays/Town12_Rep0_866_0_route0_11_08_23_43_01`
- `AccidentTwoWays/Town13_Rep0_1157_1_route0_11_08_23_47_31`

showed the same pattern:

- the route had already finished the borrow interaction
- ego then encountered a current `follow_chase` state near the junction
- a future merge hint existed in the background
- the future merge start was blocked by the current-chase gate

For this pattern, the gate firing is correct:

- borrow had already ended
- current chase semantics should dominate
- blocking the future merge start avoids opening an unnecessary merge window

So these cases should currently be treated as:

- expected gate behavior
- not a merge regression

## 5. Historical logic: wait/release-time corridor

This is different from the scene-level corridor above.

It is built in `_annotate_release_ready(...)`.

This path is only active on frames where:

- `wait_state == 1`

For each wait frame:

1. find the next future release frame using ego speed thresholds
2. take the current wait frame's route
3. run `_estimate_borrow_points_from_wait_route(...)`
4. get a borrow segment from that current wait frame
5. transform that segment into world coordinates
6. match its start/end to nearest future expert ego frames
7. over the release horizon, transform future actor boxes back into the current
   ego frame
8. if any dynamic actor overlaps the borrow segment:
   - `source = blocked_dynamic_corridor`
   - `ready = 0`
9. otherwise:
   - `source = clear_dynamic_corridor`
   - `ready = 1`

This path records:

- `release_frame_id`
- `enter_frame_id`
- `return_frame_id`
- `borrow_duration_s`
- `release_to_return_s`
- `blocked_frame_id`
- `blocking_actor_id`

But this path is not the same as the scene-level blocker-route corridor.

It is a per-wait-frame corridor used for release checking.

Important:

- for two-way scene corridor visualization, we do not want to mix this older
  wait/release corridor with the scene-level blocker-route corridor
- this path may still exist as an older debug/control helper
- but it is not the source of the current displayed two-way corridor
- and it is not the source of the current mainline `borrow_cross_meet`
  speed-curve geometry

## 6. Current rule for two-way scenes

For `ConstructionObstacleTwoWays` and `AccidentTwoWays`, the displayed
corridor should come only from:

- `_build_event_two_way_borrow_context(...)`

That means:

- no reuse of `release_ready` geometry
- no wait-route geometry fallback
- no cover-centered fallback corridor
- if scene-level corridor construction fails, the video generation should fail

For these two-way scenes, the displayed corridor is therefore:

- a fixed scene-level corridor
- built from blocker selection + context-frame route
- not a per-wait-frame temporary corridor

The same scene-level corridor is also the mainline geometry source for stage1
borrow-speed computation in precompute.

## 7. Current borrow-cross speed logic

Once a cover is classified as `borrow_cross_meet`, the speed logic is
decomposed into:

- `borrow_yld`
- `borrow_go`

### 7.1 Current cover

If a blocking actor already covers the borrowed corridor:

- we compute the time until that actor fully clears the corridor start
- `v_yield_max_mps` is the largest ego speed that still yields safely behind it
- `v_go_min_mps` is treated as effectively impossible at this stage
  - current implementation records this as `inf`
- therefore:
  - `borrow_yld` is the meaningful term
  - `borrow_go` stays high

### 7.2 Future cover

If the blocking actor has not entered the corridor yet:

- compute `d_bg_to_end_m`
  - bbox distance from the actor to corridor end
- compute `t_bg_to_end_s`
  - time until the actor reaches corridor end
- compute `t_bg_exit_to_start`
  - time until the actor fully exits the corridor at the start side
- then derive:
  - `v_yield_max_mps`
    - ego must stay below this to safely wait until the actor clears
  - `v_go_min_mps`
    - ego must exceed this to enter, traverse, and fully clear the corridor
      before the actor reaches corridor end, with `cross_safe_gap_s`

Interpretation:

- `borrow_yld` asks:
  - can ego remain behind / outside the corridor safely?
- `borrow_go` asks:
  - can ego clear the whole corridor early enough?
- `meet_risks` for `borrow_cross_meet` is the min of those two decomposed risks
- `total_risks` is still built from the usual max over chase / meet / ped

## 8. Cross-active and cross-wait timing

For cross-wait timing, the intended logic is episode-based:

1. detect the first frame where raw cross meet becomes active
2. latch `cross_active = 1`
3. keep that active state through temporary current/future-cover dropouts
4. stop the episode only when `go` happens

This means:

- no grace window is needed for the cross-active bit itself
- `cross_wait_time_s` should not restart just because one cover drops out for a
  frame or two
- the episode is "first cross-active to go", not "every local cover fragment"

In the current implementation:

- the wait episode starts only when ego is near the cross start and speed is
  near zero
- after the first `go`, the timer resets
- for two-way borrow scenes, the cross start comes from the scene corridor
- for junction cross scenes, it comes from generic cover geometry instead

### 8.1 2026-04-14 agreed direction for `borrow_cross_meet` window

For the two-way borrowed-lane case, the geometry stack is now considered
stable enough to separate:

- corridor localization
- `yld / go` speed logic
- borrow-cross episode window

The intended layering is:

1. localize blocker / obstacle from event-driven scene logic
2. choose the blocker-consistent `context_frame_id`
3. use that context-frame `route + 12` extension to build corridor `start/end`
4. detect corridor `current_cover / future_cover`
5. compute:
   - `v_yield_max_mps`
   - `v_go_min_mps`
6. build `borrow_cross_active` as an explicit start/end window

Important interpretation:

- `yld / go`
  - speed-phase semantics
- `borrow_cross_active`
  - episode window semantics
- they are related, but not the same label

### 8.2 Agreed `borrow_cross_meet` speed semantics

Once corridor `start/end` is fixed:

- if an oncoming actor already covers the corridor:
  - ego can only `yld`
- if the corridor is not currently covered:
  - `go` means ego can enter, traverse, and fully clear the corridor before the
    next oncoming actor reaches it

So the key question is not classification anymore; it is the timing logic:

- `yld`
  - oncoming traffic already occupies / reaches the corridor first
- `go`
  - ego fully clears the corridor before that oncoming occupation happens

With correct `v_yield_max_mps / v_go_min_mps`, the later speed-curve energy is
treated as a deterministic mapping and should not be the fragile part.

### 8.3 Agreed `borrow_cross_meet` window start/end

Current agreed direction:

- the window should be route/progress based
- `start` and `end` must be an explicit pair

#### `go` phase start

At the geometric `context frame`, ego is usually near the borrowed-lane entry.

If ego speed is already greater than `0.5 m/s` there:

- treat that frame as the start of `go`

Otherwise:

- keep searching forward near corridor start
- the first frame where speed rises above `0.5 m/s`
  - becomes the `go` start

#### window end

Current intended definition:

- the borrow-cross window ends when ego reaches corridor end

This is intentionally simpler than a full-vehicle clear condition, because the
current corridor end already includes a conservative margin.

#### active timer

Because the oncoming background vehicle can move outside the perception range
while the borrowed-lane interaction is still semantically active, we also keep
an explicit active timer:

- `borrow_cross_active_time_s`
- it starts at `0.0` on the active-window start frame
- it increases by `0.25s` each frame while `borrow_cross_episode_active == 1`
- it resets to `0.0` outside the active window

This timer is intended as an extra conditioning signal for models when the
causal background actor is no longer directly visible.

#### window start

Current intended definition:

- use route-based distance to corridor start
- inspect the band:
  - `[corridor_start_s - 10 m, corridor_start_s]`
- if a stable slowdown onset is present there:
  - use that slowdown point as the window start
- if no clear slowdown onset is present:
  - fall back to the `5 m` point before corridor start

So the practical fallback is:

- preferred: slowdown onset in the `10 m -> 0 m` band
- fallback: fixed `5 m` pre-start point

### 8.4 Stability note

The main stability concern is not blocker localization anymore.

## 9. Unified Conflict Framework Scaffold

- `precompute_semantic_labels.py` 已经开始接统一 `conflict_area` 骨架层，先不改 energy。
- 当前统一层直接从 route-level area/window 生成 unified conflict window，不再直接继承旧
  `borrow_cross_episode / merge_episode / junction_cross_episode` 的 active。
- 旧 family episode 仍然保留，但当前角色是：
  - baseline 对照
  - 回归检查
  - debug 比较
- 当前统一层只先回答：
  - 当前帧是否处于 conflict area window
  - 当前主 family 是谁
  - 当前粗方向是什么
- 当前粗方向的临时映射是：
  - `borrow -> opposite`
  - `merge -> same`
  - `junction -> none`
- 如果某个 family 已经 active，但缺少 area/source 元数据：
  - 当前先记录为 `conflict_area` issue
  - 不直接在这层 hard crash
- 当前统一层已经按 family 分三条 area 驱动路径：
  - `borrow`
    - 基于 `event_name + scene_borrow_context + corridor`
    - 再从 borrow cover 在 corridor progress 上提纯 `conflict area`
  - `merge`
    - 基于 future conflict points -> merge area
  - `junction`
    - 基于 conflict history cluster -> junction local area
- 当前统一层的 family 选择优先级暂定：
  - `borrow > merge > junction`
- 这是框架层，不是最终语义层：
  - `borrow conflict area` 后面还要从 corridor 中提纯
  - `merge / junction direction` 后面还要改成 area-based approach direction

## 8.5 Borrow decoupling and conflict-area-based direction

Recent review suggests that the current interaction-family split is mixing two
different questions:

- what conflict region ego is really approaching
- what the actor's current local yaw / motion looks like before reaching that
  region

This is especially fragile for:

- pre-turn ego states
- opposite-turn junction conflicts
- borrow scenes where the current local heading is not the right proxy for the
  eventual corridor competition

### Current issue

`_interaction_signal_from_candidate(...)` still uses a local heading-angle test:

- `same_direction` if angle `<= 45 deg`
- `cross_direction` if angle `>= 70 deg`
- the middle band still falls into merge-like handling

As a result:

- `merge` is not purely "same direction"
- `borrow_cross_meet` still depends on first being classified as a
  cross-direction candidate
- the family split can be unstable before the true conflict region

### Agreed direction

#### Borrow should not conceptually depend on generic cross-family routing

For:

- `ConstructionObstacleTwoWays`
- `AccidentTwoWays`

the stable geometry is already scene-level:

- `event_name`
- `scene_borrow_context`
- corridor `start/end`
- `context_frame_id`

So long-borrow labeling should conceptually be treated as a scene-level
corridor problem, not as a generic cross-family subtype that first relies on a
local angle test.

#### Direction should be defined relative to the conflict area

Instead of using:

- current actor yaw / motion vs current local ego route heading

the more stable definition is:

- ego approach direction **into the conflict area**
- background actor approach direction **into the same conflict area**

using the conflict-region geometry itself:

- `borrow`: corridor
- `merge`: merge area
- `junction`: conflict circle / local junction area

This gives a more stable auxiliary direction notion for later conflict labels,
for example:

- `same`
- `opposite`
- `cross` / `none`

depending on the final auxiliary-label design.

The key idea is:

- local heading is only a weak proxy
- conflict-area approach direction is the actual quantity we care about

The main remaining sensitivity is:

- how we define "slowdown starts"

Current agreed fallback means this is no longer a hard blocker:

- use slowdown if it is clearly visible
- otherwise use the fixed `5 m` pre-start fallback

### 8.5 Packed-frame truncation caveat

For two-way borrow scenes, an important failure mode is not geometry, but packed
sample coverage.

The raw route can contain more frames than the corresponding
`samples_packed.pkl` route slice:

- some head frames may be missing
- some tail frames may also be missing

So a scene can simultaneously have:

- valid `scene_borrow_context`
- valid `borrow_motion`
- many `borrow_cross_meet` frames
- a reasonable `go` candidate

but still fail to produce `borrow_cross_episode_active`, simply because the
packed route slice ends before ego ever reaches:

- `borrow_end_distance_m <= 0.5`

One concrete confirmed example during relabeling review was:

- `ConstructionObstacleTwoWays/Town12_Rep0_1490_0_route0_11_08_09_11_32`
  - raw route frames: `0..125`
  - packed route frames: `6..112`
  - missing tail frames: `113..125`

In that case:

- corridor localization was valid
- `context_frame_id` and `go` were reasonable
- but the packed slice never observed the final corridor-end clearance

So for `ConstructionObstacleTwoWays` scenes that show:

- valid context
- no `borrow_cross_episode_active`
- no `borrow_end_distance_m <= 0.5`

we should first suspect packed-frame truncation before concluding that borrow
window logic itself is wrong.

The current preferred mitigation is **not** to weaken borrow end logic with a
fallback. Instead:

- build a temporary stage1-only padded packed with raw head/tail frames restored
- run stage1 relabeling on that padded packed
- then project the relabeled stage1 fields back onto the original trimmed packed

This keeps:

- the geometry / episode logic unchanged
- the training dataset shape unchanged
- while letting stage1 labeling see the full route head/tail timeline

### 8.6 Current-chase start gate for future merge / junction

For hard `merge` / `junction` episode start, a future interaction candidate
should not open too early when the current main route blocker is still a very
near, slow same-direction lead vehicle.

Current agreed gate:

- only applied to **episode start**
- does **not** cancel an already-started episode
- does **not** change soft risk computation

Gate condition:

- `current_cover.subtype == follow_chase`
- `current_cover.other_speed <= 0.5 m/s`
- `current_cover.route_distance_m <= 15 m`

When all three hold:

- do not start a future-driven hard `merge` episode yet
- do not start a future-driven hard `junction` episode yet

This was introduced to suppress cases where:

- current interaction is still a near lead blocker
- but a farther `future_cover` candidate would otherwise open a hard episode
  too early

## 9. Junction-left cross split: `yld` / `go`

`junction_left_cross_meet` should be treated as a decomposed cross-decision
label, not only as one folded `meet_risk`.

The intended interpretation is parallel to `borrow_cross_meet`, but the
geometry source stays different:

- `junction_left_cross_meet`
  - geometry comes from generic route-cover cross conflict
  - no two-way blocker corridor is involved
- `borrow_cross_meet`
  - geometry comes from the scene-level borrowed-lane corridor

### 9.1 What already existed

Even before the split, the junction-cross logic already had internal timing
quantities such as:

- `v_go_min_mps`
- `v_yield_max_mps`
- `t_bg_s`
- `t_bg_exit_s`
- `t_ego_exit_s`

Those values were mainly stored inside `meet_debug`, but the saved supervision
was still mostly folded into:

- `speed_risk_meet_values`

### 9.2 Current `junction_cross_active` direction

`junction_left_cross_meet` now also carries an explicit episode window:

- `junction_cross_episode_id`
- `junction_cross_episode_active`
- `junction_cross_episode_start_frame`
- `junction_cross_episode_end_frame`

The intended logic is:

1. treat single-frame cover only as conflict-point evidence
2. accumulate multi-frame conflict-point history
3. cluster those conflict points into a stable local conflict area
4. choose the in-area frame with minimum ego route progress as `start`
5. keep the episode active until ego leaves that conflict area

Important:

- `cover` is evidence, not the hard window start by itself
- the area is built from collision/conflict history, not one raw cover frame
- `end` is defined by ego leaving the conflict area

In practice this means:

- the false early current-only junction-cross cases are less likely to open a
  window by themselves
- `junction_cross_active` is now parallel in spirit to:
  - route-based `merge_active`
  - corridor-based `borrow_cross_active`
  while still using its own local cross-conflict geometry

### 9.3 Junction should also become conflict-area-based

For junction, the remaining direction logic should follow the same overall
principle as merge:

- cover / conflict-point detection remains the evidence source
- conflict area gets explicit `start/end`
- direction should be derived from approach into / through that conflict area
  instead of current local heading alone

Working interpretation:

- `junction_active` should be built from the conflict area itself
- `junction + left` from raw labels / command is useful as a validation signal
  or scene prior
- but it should not be treated as the only hard condition for a junction
  conflict window

Reason:

- raw `junction` and `command == LEFT` signals can lag slightly in time
- the meaningful event is whether ego and another actor compete around the
  conflict area, not whether the left-turn metadata has already fired exactly
  on the same frame

Important note:

- `junction + left` can still behave merge-like in some cases
- a representative example is ego left turn vs opposite right-turn actor
  competing for the same downstream lane

So future junction labeling should not assume:

- `junction_left_cross` and `merge` are always separate

Instead, the cleaner split is:

- family / area geometry
- conflict direction
- whether the downstream lane competition is merge-like

## 10. Borrow corridor vs borrow conflict area

For long two-way borrow scenes, the current scene-level corridor is already
quite stable, but it should be treated as the **larger maneuver region**, not
as the final conflict region itself.

### 10.1 Corridor is the large maneuver frame

Current scene-level borrow corridor is still useful as the main maneuver
geometry because it gives stable:

- `borrow_start_world_xy`
- `borrow_end_world_xy`
- `borrow_distance_m`
- `dStart / dEnd`
- borrow active / go timing

So the corridor is not the problem. It remains the right outer frame for the
borrow maneuver.

### 10.2 Current end is slightly too conservative

The current corridor-end logic tends to choose a late recovery point:

- first find the borrow peak
- then search later recovery-side points
- then fall back to a relatively late extra-route point if needed

Working hypothesis:

- for the maneuver corridor end, a better definition may be:
  - after the borrow peak, take the **first** point whose lateral shift has
    recovered sufficiently close to the original lane

This should place the corridor end closer to the real re-entry region, instead
of drifting too far into the extra tail.

### 10.3 Conflict area should be a subset of the corridor

Important distinction:

- `corridor`
  - the full borrowed-lane maneuver region
- `conflict area`
  - only the subsegment where the opposite-direction actor actually occupies or
    competes for the borrowed lane

Proposed borrow conflict-area definition:

- project bbox cover onto the borrow corridor progress axis
- `conflict_start = min(progress of cover overlap)`
- `conflict_end = max(progress of cover overlap)`

Interpretation:

- the full corridor includes:
  - entering the borrowed lane
  - traversing the conflict region
  - returning to the original lane
- the conflict area should include only:
  - the pure opposite-lane competition zone

This is the cleaner geometry source for:

- `conflict_area_active`
- conflict-direction labels
- more precise `yld/go` timing

In short:

- borrow corridor = large maneuver frame
- borrow conflict area = tighter causal subregion inside that corridor

## 11. Merge direction should be defined after the merge area

For merge, the current geometry stack is already mostly acceptable:

- cover is usable
- current same-direction blocker becoming `follow_chase` is expected
- future conflict points are the right source for merge-area construction
- merge-area `start` is already good
- merge-area `end + 3 m` is not the main problem

The remaining weak point is the direction definition.

### 11.1 Current issue

Current direction logic still relies too much on local heading / local angle
near the current cover or future candidate.

This is fragile because:

- the route may still be in the slanted merge-in segment
- local heading there is not the same as the true downstream merged-lane
  direction
- using current local heading can therefore make merge-vs-cross distinction
  less stable than it should be

### 11.2 Agreed direction

The preferred direction for merge should be derived from the downstream route
**after** the merge area, not inside the slanted approach region.

Recommended first version:

- keep merge-area construction as it is:
  - future conflict points
  - clustered into merge area
  - area end still uses the existing `+3 m` post margin
- define merge downstream heading from route tangent samples taken after
  `merge_area_end`

Suggested window:

- preferred: `merge_area_end -> merge_area_end + 5 m`
- fallback: `merge_area_end -> merge_area_end + 3 m` if route is too short

### 11.3 Practical interpretation

This means:

- merge area still comes from cover + future conflict points
- but merge direction should come from the lane ego is actually merging into
- not from the slanted segment while ego is still entering that lane

The expected benefit is:

- cleaner merge direction
- less contamination from slanted approach headings
- more stable future auxiliary labels such as conflict direction

## 12. What to compare after the unified conflict-area rerun

Before designing new `yld / go / energy`, the first job is to check whether the
new unified conflict-area windows are geometrically and statistically sane.

The preferred evaluation order is:

1. family coverage
2. family purity by event
3. issue distribution
4. old-vs-new delta
5. small-scene qualitative checks

### 12.1 Family coverage

At the full-dataset or shard-summary level, check:

- `active_scenes_by_family`
- `active_samples_by_family`
- average / median window length by family
- start/end frame count by family

Goal:

- `borrow`
  - should stay concentrated in long two-way scenes
- `merge`
  - can be broad, but should not drift upward mainly because of clearly wrong
    event families
- `junction`
  - should stay concentrated in route-turn / junction-related scenes

### 12.2 Family purity by event

For each family, compare:

- active scenes by event
- active samples by event
- event-level coverage ratio

Important watchpoints:

- `borrow`
  - should remain dominated by:
    - `AccidentTwoWays`
    - `ConstructionObstacleTwoWays`
- `merge`
  - should keep reasonable coverage in:
    - `HighwayExit`
    - `ParkingExit`
    - `MergerIntoSlowTraffic*`
    - `EnterActorFlow`
    - `noScenarios`
  - but should not remain heavily polluted by:
    - `HazardAtSideLaneTwoWays`
    - `VehicleOpensDoorTwoWays`
- `junction`
  - should remain strongest in:
    - `NonSignalizedJunctionLeftTurn`
    - `SignalizedJunctionLeftTurn`
    - `VehicleTurningRoute`
    - `Interurban*`

### 12.3 Issue distribution

The new framework intentionally records issues instead of silently repairing
them, so issue summaries are part of the main acceptance criteria.

Check:

- `issue_scenes_by_family`
- `issue_frames`
- `missing_reason_counts`

Important interpretation:

- issue counts are not only "bad news"
- they also tell us whether the new framework is exposing missing geometry
  honestly instead of hiding it behind fallback behavior

### 12.4 Old-vs-new delta

After rerun, compare old and new labels at least at these levels:

- active-scene delta by family
- active-sample delta by family
- event-level active-scene delta
- per-scene start/end shifts for representative scenes
- family disagreement table:
  - old active -> new inactive
  - old inactive -> new active
  - old family -> new family

Goal:

- new framework should not be judged only by total counts
- it should be judged by whether the changed scenes are the scenes we wanted to
  change

### 12.5 Small-scene qualitative validation

Keep a tiny representative set for direct debug / video checks:

- `AccidentTwoWays/Town12_Rep0_26_0_route0_11_08_18_12_42`
- `Accident/Town12_Rep0_10_0_route0_11_08_23_53_07`
- `NonSignalizedJunctionLeftTurn/Town12_Rep0_1105_0_route0_11_08_02_43_48`
- `SignalizedJunctionLeftTurn/Town03_Rep0_Town03_Scenario7_32_route0_11_08_20_16_15`

These are not enough for acceptance by themselves, but they are the fastest
way to explain any large statistical delta.

## 13. Positioning of `yld / go / energy`

The current agreed design order is:

1. build stable `conflict_area` windows first
2. decide what `yld / go` is supposed to supervise
3. only then choose numeric form / normalization for energy

### 13.1 What `yld / go` should mean

`yld / go` should not answer:

- what family this scene belongs to
- whether a conflict area exists

Those belong to:

- `family`
- `dir`
- `active`
- `start/end`

Instead, `yld / go` should answer:

- inside an already-valid local conflict area window, what speed-phase decision
  is currently preferred

In other words:

- `window` solves geometry and timing of *where / when the local conflict is*
- `yld / go` solves *how ego should negotiate that local conflict*

### 13.2 What problem `yld / go / energy` is expected to solve

The label is meant to reduce failure modes like:

- entering a valid conflict area too fast
- hesitating in a window where ego should already commit
- treating every conflict as a binary class instead of a speed-dependent choice
- forcing traj prediction alone to encode local negotiation preference

So the intended role is:

- family / active
  - localize the relevant conflict
- `yld / go`
  - express branch preference inside that conflict
- energy
  - make that preference speed-sensitive and smooth enough for closed-loop use

### 13.3 Family-specific interpretation

#### Borrow

Inside a valid borrow conflict area:

- `yld`
  - oncoming occupancy or arrival dominates
- `go`
  - ego can clear the borrow conflict area before that occupancy matters

#### Merge

Inside a valid merge conflict area:

- `yld`
  - ego should pass behind the key competing actor
- `go`
  - ego should pass before / claim the downstream lane first

#### Junction

Inside a valid junction conflict area:

- `yld`
  - ego should let the crossing / turning actor clear first
- `go`
  - ego should commit through the conflict area first

### 13.4 Design principle for the next step

Before picking numeric targets, first agree on:

- what object `yld / go` conditions on
  - local conflict area
- what it is trying to change in behavior
  - speed-phase choice
- what it is not trying to do
  - re-derive family / area geometry

Only after that should we decide:

- raw energy vs relative energy
- pairwise normalization
- probability targets vs energy targets
- temporal smoothing

So older sessions may remember that "junction cross had go/yld ideas already".
That memory is correct at the debug / formula level, but not at the stable
top-level label-field level.

### 9.2 Split we want other sessions to use

For model / dataset design, junction-left cross should now be thought of as two
separate speed-affordance branches:

- `junction_cross_yld`
  - ego waits and lets the background actor clear the conflict region
- `junction_cross_go`
  - ego passes before the background actor reaches the conflict region, while
    respecting `cross_safe_gap_s`

The intended saved top-level fields are:

- `speed_risk_junction_cross_yld_values`
- `speed_risk_junction_cross_go_values`

And the intended relationship is:

- `speed_risk_meet_values`
  - remains the combined cross-meet risk used by older consumers
- `speed_risk_junction_cross_yld_values`
  - explicit yield branch for junction-left cross
- `speed_risk_junction_cross_go_values`
  - explicit go branch for junction-left cross

### 9.3 Training-side reading

When another session modifies the model:

- `merge_meet`
  - already has decomposed `merge_yld` / `merge_go`
- `borrow_cross_meet`
  - already has decomposed `borrow_yld` / `borrow_go`
- `junction_left_cross_meet`
  - should now be treated with the same decomposed idea:
    `junction_cross_yld` / `junction_cross_go`

This keeps the label design conceptually consistent:

- same-direction merge -> `merge_yld/go`
- two-way borrowed-lane cross -> `borrow_yld/go`
- junction-left cross -> `junction_cross_yld/go`

### 9.4 One boundary that should remain explicit

The split does **not** mean that junction cross and two-way borrow share one
common corridor implementation.

Only the decision decomposition is analogous.

- `junction_left_cross_meet`
  - generic cross-conflict geometry
- `borrow_cross_meet`
  - scene-level borrow corridor geometry

So:

- similar timing semantics
- separate geometry sources

## 10. Current merge speed logic

`merge_meet` is separate from `cross meet`.

For the speed curve, the main future-merge quantities are:

- `t_bg_s`
  - background actor arrival time
- `t_bg_exit_s`
  - background actor corridor-occupancy exit time
- `t_bg_clear_s`
  - extra clearance time used for conservative yield-behind logic
- `v_equal_mps`
  - same-arrival speed at the merge point
- `v_go_min_mps`
  - minimum speed to get clearly in front before `merge_tau_s`
- `v_behind_min_mps`
  - rear-car floor / keep-up floor
- `v_go_need_mps`
  - actual go threshold after combining front-arrive and rear-floor constraints
- `v_yield_max_mps`
  - maximum speed that still yields behind safely

Interpretation:

- `merge_yld_risks`
  - safe-behind branch
- `merge_go_risks`
  - safe-front / committed-merge branch
- `meet_risks`
  - the combined merge timing risk

Near the future-to-current transition:

- raw `v_equal_mps` / `v_go_min_mps` can become numerically huge
- the code keeps the risk curve
- but hides those raw debug values when they stop being interpretable

## 11. Current merge-episode labels

Stage1 also tracks a merge episode separate from the instantaneous speed curve.

This logic is built in `_annotate_route_stage1_merge_decisions(...)`.

High-level flow:

1. seed an episode from consecutive `future_cover == merge_meet`
2. allow short future-cover dropouts via `future_grace`
3. allow red-light waiting to extend the episode without falsely ending it
4. merge nearby raw episodes if they are likely the same actor stream
5. collect actor tracks over future-merge, current-merge, and chase states
6. choose a resolution actor
7. mark `go_frame`
  - usually when the resolution actor first appears as current cover
8. continue the episode slightly after `go`
  - to cover chase / continuity / lane-settle transition
9. end `go` when lane-settle is confirmed
  - current code uses steer + heading alignment

## 12. 2026-04-14 merge relabeling direction

The current packed labels should not be treated as authoritative for merge
analysis.

We already found at least one route where:

- packed `current_cover / future_cover` reported raw merge cover
- but the same route, when recomputed with the current video-time cover logic,
  had no raw merge cover at all

So the current working rule is:

- do not use old packed merge-cover debug to decide final merge logic
- for the next full relabeling pass, use the current code path and regenerate
  stage1 from scratch

### 12.1 Agreed merge layering

The intended interpretation stays:

- `cover`
  - geometric base fact
- `v_yield_max / v_go_min / v_go_need`
  - threshold layer derived from cover and timing
- `merge_yld_risks / merge_go_risks / meet_risks`
  - deterministic energy / risk mapping from the threshold layer

But this interpretation is only trustworthy when the underlying merge cover is
freshly recomputed.

### 12.2 Agreed start / end principles

The merge window should be built as an explicit start/end pair.

- if there is a `merge start`, there must be a matching `merge end`
- if there is a `merge end`, there must be a matching `merge start`

Current agreed direction:

- `merge_start`
  - anchor from the first valid `future_cover == merge_meet`
- `merge_active`
  - should span the full merge window from `start` to `end`
- do not rely on ad-hoc active grace to define the window itself
  - short cover dropouts may exist inside the window, but the window should be
    defined by start/end, not by local grace bookkeeping

### 12.3 Agreed hold semantics

`hold / wait` is not on the same layer as `yld / go`.

The intended decomposition is:

- `merge_active`
  - episode window
- `merge_phase`
  - `yld` or `go`
- `merge_hold`
  - separate state inside the active window
  - for example red-light hold

So:

- `yld / go` describes the merge decision phase
- `hold` describes why ego is not moving / is still paused
- `hold` should not replace the active window

### 12.4 Agreed finite-threshold policy

For merge debug / saved threshold values:

- avoid emitting `NaN` just because a denominator is very small
- instead, clamp very large speeds to a large finite cap
  - current discussion example: `1000 m/s`

This applies especially to:

- `v_equal_mps`
- `v_go_min_mps`
- `v_go_need_mps`
- `v_yield_max_mps`

The branch split between "front-arrive timing" and "rear catch-up regime" can
remain, but the exported values should be finite and clipped rather than hidden
behind `NaN`.

### 12.5 Agreed route-based merge-area direction

For the next merge relabeling pass, the preferred end logic is route-based.

Current agreed direction:

1. when merge first appears, record the future-merge conflict / collision point
2. keep recording later merge conflict points as the same merge evolves
3. convert those points into a route-aligned merge area
4. end the merge window when ego has progressed past that merge area

Why route-based:

- it keeps the active window aligned with ego progress
- it is cheaper than purely world-space geometric fitting
- it naturally supports a single start/end window

### 12.6 Geometry confidence ideas that are still open

The following idea is considered promising, but is not yet fixed:

- use recorded merge conflict points to build a confidence signal for the merge
  area

Two candidate confidence formulations were discussed:

1. fit a circle to the conflict points
  - smaller fitted radius -> higher confidence
2. use a fixed-radius circle, for example radius `5 m`
  - center it at the first merge conflict point
  - more later conflict points inside the circle -> higher confidence

Current status:

- the merge area itself should still be route-based
- the circle-based logic is only a candidate confidence / quality score, not
  yet the primary definition of `merge_end`

### 12.7 Small merge questions intentionally left open

These are the items still worth discussing before the final relabeling pass:

- the exact finite cap used for very large threshold speeds
- how the route-aligned merge area is best projected / widened
- whether we want to save an explicit `merge_hold` top-level field in stage1
- whether circle-based confidence is useful enough to save as debug

### 12.8 2026-04-14 implementation status in code

The current in-repo mainline has now moved part of this direction into
`precompute_semantic_labels.py`.

Current implementation status:

- merge cover summaries now retain route/progress geometry:
  - `route_distance_m`
  - `ego_route_front_s_m`
  - `route_point_local_xy`
  - `scene_route_conflict_s_m`
  - `scene_route_conflict_world_xy`
- merge motion debug now retains route-progress state:
  - `scene_route_center_s_m`
  - `scene_route_front_s_m`
  - `scene_route_rear_s_m`
  - `ego_half_length_m`
- merge threshold debug now uses finite clipped values rather than hiding near
  singular cases with `NaN`
  - current cap in code: `STAGE1_MERGE_SPEED_CAP_MPS = 1000.0`
- merge episode parsing is no longer driven by a resolution actor as the main
  source of truth
  - `merge_start`
    - first future merge seed with valid route-progress conflict geometry
  - `merge_area`
    - built from min/max recorded merge conflict progress on the scene route
    - no longer adds a front pre-margin before the first future conflict point
  - `go`
    - triggered either by entering the merge area or by speed crossing the go
      threshold
  - `end`
    - triggered when ego rear progress passes the merge-area end
  - `hold`
    - kept separate and currently tied to red-light wait

What this means:

- `merge_active` is now intended to be a route-window label
- `yld / go` is a phase label inside that window
- actor ids remain as debug only and should not be treated as the main merge
  decision source

Per-frame episode output contains:

- `phase`
  - `none`, `yld`, `go`
- `episode_id`
- `active`
- `no_go`
- `start_frame`
- `end_frame`
- `go_frame`
- `resolution_actor_id`
- `end_state`
- `actor_ids`
- `actor_switch_frames`

Important:

- merge episode parsing is about temporal decision structure
- merge speed thresholds are about instantaneous speed affordance
- they are related, but they are not the same label

## 12. How to read a saved JSON quickly

If a frame JSON contains:

- `context_frame_id`
- `anchor_actor_id`

then the displayed two-way corridor is coming from the scene-level
blocker-route locator.

For the cleaned-up two-way video path, that is the expected case.

## 13. Current ambiguity we should remember

There are two different meanings of "route head":

1. current frame route head
2. context-frame route head

Current `route_head` mode in code means:

- the route head of the selected context frame

It does not mean:

- "for every displayed frame, use that frame's own first route point"

That distinction must stay explicit in future discussions.

## 14. Recommended terminology for future debugging

To avoid confusion, we should use these names consistently:

- `scene corridor`
  - the fixed two-way corridor built from blocker + context-frame route
- `release corridor`
  - the per-wait-frame corridor used by `release_ready`
- `displayed corridor`
  - the corridor actually drawn in the video

When debugging a video, always ask:

1. which one of these three corridors am I looking at?
2. is `context_frame_id` present?
3. is the displayed corridor allowed to differ from the scene corridor for this scene type?

## 14. Settled notes as of 2026-04-13

- current two-way scene corridor start:
  - context-frame `route_head`
- current two-way scene corridor end:
  - 32-point route-shift recovery statistic
  - fallback to extra-12 tail point 10 only if needed
- current mainline `borrow_cross_meet` geometry:
  - scene corridor, not wait/release corridor
- current `cross_active` behavior:
  - sticky from first activation to `go`
- current `video` / `precompute` state:
  - aligned on the same mainline cross / merge speed logic
