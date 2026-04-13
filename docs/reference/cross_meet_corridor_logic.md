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
