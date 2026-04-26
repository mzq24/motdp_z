# Temporary Occupancy for Conflict-Aware Phase Timing

Date: 2026-04-25

## Summary

This note records the current discussion around using **temporary occupancy** to improve `yld/go` phase timing.

The main idea is:

- We do not need dense future 4D occupancy over the whole BEV.
- We already have interaction windows and conflict areas.
- The missing piece is whether the conflict area is temporarily occupied or available when ego wants to pass.

In short:

```text
current conflict_area: where the conflict is
temporary occupancy: when that area is occupied / available
phase decision: whether ego should yld or go at this time
```

This is mainly for:

- junction
- merge
- borrow

These cases often fail because the model predicts the right rough interaction family, but the `yld/go` switch is too early, too late, or unstable.

## Why This Is Needed

Current model signals include:

- `window`: none / merge / junction / borrow
- `dir`: none / same / opposite / cross
- `conflict_area_logits`: route-point heatmap, shape `(B, 20)`
- `decision_phase`: yld / go
- `control_phase`: coast_yld / slow_yld / stop_yld / go
- boundary values:
  - `yld_max_speed`
  - `go_min_speed`
  - `chase_max_speed`

Current limitation:

- `conflict_area_logits` tells the model which route segment is conflict-related.
- It does not explicitly tell the model when that area is occupied.
- `go_min_speed` / `yld_max_speed` are useful boundary summaries, but they do not by themselves force the model to understand the temporal cause of the decision.

Observed failure:

- `go_min_speed` may be near the cap, e.g. `27-30 m/s`, which means go is effectively infeasible.
- The phase head can still predict `go`.
- Post-processing can fix this, but the training objective should also teach the model not to produce this contradiction.

Important correction:

- `yld` and `go` should be treated as two modes, not as one single feasible speed interval.
- Boundary values should not be interpreted as:
  - `go_min <= speed <= yld_max`
- Instead:
  - `go_min_speed` describes the lower speed requirement for the go mode.
  - `yld_max_speed` describes the upper speed compatible with the yield mode.
  - `chase_max_speed` is a separate front / lead-vehicle safety cap.

## Temporary Occupancy Definition

Temporary occupancy is sparse and conflict-area centered.

Instead of predicting:

```text
O(x, y, t)
```

we only care about:

```text
O(C, t)
```

where `C` is the current conflict area.

The target question is:

```text
Will the conflict area be occupied during the future time interval when ego would enter / pass through it?
```

Useful temporal variables:

- whether the conflict area is occupied now
- whether it will be occupied in future bins
- when it becomes free
- when it becomes occupied again
- whether ego can clear it before the gap closes

Suggested occupancy bins:

- `T_occ = 8` or `10`
- bin size around `0.5s`
- output shape:
  - `conflict_occupancy_future_bins`: `(B, T_occ)`

Possible gap state:

- `blocked`
- `opening`
- `closing`
- `clear`

This gap state should directly support `yld/go` timing.

## Current Label Decision (2026-04-26)

We currently prefer to treat **future occupancy bins** as the primary label signal.

Meaning:

- `conflict_occupancy_now`
- `conflict_occupancy_future_bins`
- `conflict_dist_to_entry_m`
- `conflict_dist_to_exit_m`
- `conflict_span_m`

are the main supervision signals.

By contrast:

- `gap_state = blocked / opening / closing / clear`

should be treated as a **debug-side derived summary**, not the main training target.

Reason:

- `gap_state` is a useful human-readable compression.
- But it discretizes temporal structure too early.
- The occupancy bins preserve more direct timing information and let the model learn its own temporal abstraction.

Another important clarification:

- **current-box leaving**
- **future-box occupation**

should be thought of as related but distinct signals.

In other words:

- "current blocker clears the area" is one event
- "future actor occupies the area" is another event

These should be inspected separately in debug/video, even if they are later summarized into one coarse `gap_state`.

Practical implication:

- keep `first_future_free_bin` as the earliest leave/clear cue
- keep `first_future_occupied_bin` as the earliest occupy/arrival cue
- demo them separately in video/debug

## Ego Distance / Timing To Conflict

Temporary occupancy alone is not enough. The model also needs to know how close ego is to the conflict area.

Recommended label-side quantities:

- `conflict_route_entry_idx`
- `conflict_route_exit_idx`
- `conflict_dist_to_entry_m`
- `conflict_dist_to_exit_m`
- optional `conflict_span_m`

Preferred definition:

- Use the current route polyline and conflict-area route span.
- Compute distance along the route.
- `entry` is the first conflict route point / segment.
- `exit` is the last conflict route point / segment.

Important caution:

- `conflict_time_to_exit` should not be a primary GT label in V1.
- It depends on a speed assumption.
- If computed with current speed, it becomes unstable when current speed is near zero.

Preferred treatment:

- Label distance explicitly.
- Compute time-to-conflict as a model-side deterministic summary later:
  - route
  - conflict area probabilities
  - current speed or predicted speed
  - explicit speed floor
  - reachable-speed logic

This keeps label supervision cleaner.

## How This Connects To Phase

The final target is not temporary occupancy itself.

The target is better phase timing:

- less hesitation
- fewer oscillations
- sharper `yld -> go` transitions
- fewer unsafe early `go` decisions
- fewer missed narrow go gaps

Recommended conceptual chain:

```text
conflict area geometry
-> temporary occupancy / availability
-> ego distance / reachable timing
-> gap feasibility
-> decision_phase and control_phase
-> speed / traj behavior
```

This is more causal than predicting `phase` only from global scene features.

## Latest Direction (2026-04-26)

Recent discussion suggests simplifying the intermediate representation.

### 1. Do not over-engineer explicit `entry` / `area_depth_ratio` labels in V1

Current preference:

- the existing conflict-area prediction is already reasonably accurate
- the model should be able to infer "still before area" vs "already entering / inside area" from area probabilities
- for example:
  - low-but-rising area probability means ego is still before the area
  - sustained high area probability across nearby route points means ego is already entering / inside the area

Therefore:

- exact hand-crafted `entry` and `area_depth_ratio` supervision is not the immediate priority
- future tuning should first focus on improving `merge` / `junction` area start placement through video inspection and empirical adjustment

### 2. Temporary-occupancy bins should not be permanently limited to `future6`

Current implementation may start from a short 7-slot view for convenience.

But conceptually:

- the occupancy sequence should not be tied forever to the current `future6` cover layout
- later versions may use a longer horizon if that improves gap-timing supervision
- this should be treated as an implementation detail, not a semantic constraint

### 3. Focus the next label on `go_opportunity_prob`

The next useful intermediate state is not another hard phase label.

Instead, define a soft pair:

- `go_opportunity_prob`
- `yld_pressure_prob = 1 - go_opportunity_prob`

## Current Label Spec (V2, 2026-04-26)

The current implementation has moved away from the older all-actor occupancy idea.

### 1. Postprocess-only temporary occupancy

Current temporary occupancy is produced in a postprocess step, not in the main stage1 precompute.

Current top-level fields:

- `temporary_occupancy_cover_bins`: shape `(13,)`
- `temporary_occupancy_cover_valid`: shape `(13,)`
- `go_opportunity_prob`
- `yld_pressure_prob`
- `go_opportunity_valid`

The bins are always written route-wise inside the same active conflict window.

### 2. Family-specific bin semantics

The bins are still family-aware.

- `borrow`
  - use family-matched `current_cover`
  - current rule: borrow bin = 1 only when current cover is `borrow_cross_meet`

- `junction`
  - use family-matched `current_cover`
  - current rule: junction bin = 1 only when current cover is a cross-type `meet`

- `merge`
  - still only use `current_cover`
  - but do **not** require current-cover subtype to remain `merge_meet`
  - current rule:
    - `current_cover.exists == 1`
    - and current-cover conflict point is still inside merge area
  - primary check:
    - `current_cover.scene_route_conflict_s_m`
    - must lie within `[area_start_s_m - 1m, area_end_s_m]`
  - fallback:
    - project `current_cover.scene_route_conflict_world_xyz` onto `conflict_area.area_segment_world_xyz`

This matches the practical merge behavior better:

- early in the merge window, `future_cover` may be merge-type
- after it becomes `current_cover`, the subtype may already turn into `chase`
- but as long as that current cover is still physically overlapping the merge area, the bin should stay occupied

### 3. Reference quantities

Current debug/reference quantities are:

- `reference_run_start_expert`
- `reference_run_start_geom`
- `reference_run_start_final`
- `reference_run_len`
- `raw_reference_pass_time_bins`

Current meaning:

- `reference_run_start_expert`
  - expert-side run-start proxy from `go_frame`
- `reference_run_start_geom`
  - geometric run-start from `go_frame` to the frame where ego reaches or exceeds `area_start`
- `reference_run_start_final`
  - fused run-start used by the rule
- `reference_run_len`
  - how many bins are needed to pass through the area
- `raw_reference_pass_time_bins`
  - raw expert pass duration starting at `go_frame`

### 4. Distance adjustment

Current distance adjustment no longer modifies pass time directly.

Instead:

- distance is measured to `area_start`
- extra bins are added to `reference_run_start_final`
- this produces `adjusted_run_start_bins`

Current debug fields:

- `distance_to_area_start_m`
- `distance_adjustment_bins`
- `adjusted_run_start_bins`

If area-start distance cannot be computed from route progress:

- fallback uses older conflict-distance proxy
- an explicit issue string is written

### 5. Current-run quantities

Current-frame quantities are:

- `current_run_start`
- `current_run_len`
- `current_remaining_run_len`

Meaning:

- `current_run_start`
  - the first qualifying `0-run` start
- `current_run_len`
  - the qualifying `0-run` length
- `current_remaining_run_len`
  - remaining free length after ego would reach `area_start`

Current `goable` rule is:

- `current_run_start <= adjusted_run_start_bins`
- and `current_remaining_run_len >= reference_run_len`

This means:

- a current gap can be a candidate opening
- ego can still wait until the adjusted run-start budget
- and when ego really arrives at `area_start`, the remaining free run is still long enough

### 6. Go probability behavior

Current `go_opportunity_prob` behavior:

- before `go_frame`
  - derived from cycle logic
- exactly at `go_frame`
  - keep the soft probability
  - do **not** force it to `1`
- after `go_frame`
  - force:
    - `go_opportunity_prob = 1`
    - `yld_pressure_prob = 0`
    - `go_opportunity_valid = 1`

This keeps dangerous expert-launch cases visible:

- `go_frame` itself can still have a moderate or low `go_prob`
- from the next frame onward, the route is treated as committed go

### 7. Merge-specific training caveat

There is an important caveat for merge:

- once ego actually merges in, following traffic may react to ego and yield
- this can make future occupancy bins look much more open than they would have been if ego had not merged

Typical artifact:

- after merge-in, bins may look like `111000000...`
- but counterfactually, if ego had not gone, the traffic pattern might have looked more like `11100001111...`

Implication:

- current merge bins are still useful as expert-conditioned supervision
- but at training time we may want to:
  - use only a shorter prefix of merge bins
  - or otherwise downweight later merge bins after commitment

This is currently a training-side consideration, not a label-side fix.

### 8. Merge counterfactual contamination

Another way to describe the same merge issue is:

- before ego merges in, the visible opening can be informative
- after ego actually merges in, trailing traffic may react to ego and yield
- this can make the later bins look much more open than the counterfactual "ego did not merge" case

Typical example:

- factual bins after commitment may look like:
  - `1110000000000`
- but a counterfactual non-merge rollout might have looked more like:
  - `1110000111111`

Important conclusion:

- we should **not** try to heuristically fill back later `1`s
- we do not have reliable counterfactual supervision from a single expert rollout
- symbolic backfilling would likely introduce more label noise than the current factual bins

Preferred handling:

- keep the factual merge bins in the label file
- treat late merge bins as less reliable for training
- for merge occupancy supervision, prefer a prefix-only training mask

One practical rule is:

- define a merge supervision horizon
  - `H = adjusted_run_start_bins + reference_run_len`
- use bins up to `H`
- mask bins after `H`

Reason:

- bins before `H` are most relevant to launch timing
- bins after `H` are more likely to be contaminated by ego's own committed merge behavior

Current recommendation:

- `borrow / junction`
  - can continue to use the full available bins
- `merge`
  - should likely use only a prefix during training
  - while still preserving full factual bins in debug / saved labels

### 4. Use expert `go` moment as conservative opening reference

Current preference:

- keep doing threshold-style work
- but do **not** hand-define a universal geometric "gap"
- use the expert's actual `go` moment as the conservative accepted opening reference

Intuition:

- expert only moves forward, never backward
- therefore expert `go` frame is already near the latest safe / acceptable launch point in that realized opening
- this makes the expert `go` frame a practical per-window reference

Related idea:

- use the full expert `go` phase duration as a first `passing_time` proxy
- use this proxy to define a latest acceptable launch threshold
- then shape a soft pre-entry launch signal between:
  - earliest accepted launch reference: expert `go` frame
  - latest acceptable launch reference: expert `go` frame plus passing-time proxy

Important simplification:

- occupancy `bins` should be given directly to the model
- avoid over-handcrafting a separate explicit `gap` definition too early
- the same physical opportunity may look different in bins depending on ego distance to the area start
- this is similar to the earlier `yld_max / go_min` lesson: over-compressing the boundary too early makes learning harder

Intuition:

- `temporary_occupancy` describes whether the area is opening or closing
- conflict-area probability describes how close ego is to entering / being inside the area
- together they should teach the model when a go opportunity is strongest, fading, or already gone

Important behavior note:

- once ego is already sufficiently deep in the area, it no longer makes sense to supervise a strong `yld`
- in that regime the probability mass should move toward `go`

So the next design target is:

```text
temporary occupancy bins
+ conflict-area probability / proximity pattern
-> go_opportunity_prob
-> yld_pressure_prob
```

## Model-Side Direction

Current joint-state diffusion already has:

- `conflict_area_logits` as a route-aligned noisy state
- global noisy state tokens for:
  - window
  - dir
  - decision phase
  - control phase
  - boundary values
  - speed logits

Current `conflict_area_logits` feedback:

```text
conflict_area_logits_t
-> route token embedding
-> full attention decoder
-> predicted conflict_area_logits
-> DDIM update
-> conflict_area_logits_{t-1}
```

Potential next state additions:

- `conflict_occupancy_logits`: `(B, T_occ)`
- `gap_state_logits`: `(B, 4)`
- optional deterministic `conflict_timing_summary`:
  - distance to entry
  - distance to exit
  - soft area span
  - time-to-entry with speed floor

The temporal representation should not remain isolated as an auxiliary head.

Preferred architecture:

```text
base phase logits = PhaseHead(global state / route / scene)
temporal phase prior = TempPhaseHead(temporary occupancy / gap state / distance summary)
final phase logits = base phase logits + alpha * temporal phase prior
```

Reason:

- If temporary occupancy is only an auxiliary loss, the final phase head may ignore it.
- Direct modulation makes the temporal signal responsible for changing `yld/go` timing.

## Training Signals

Useful training targets for a label-management session:

- `conflict_occupancy_future_bins`
- `gap_state`
- `conflict_route_entry_idx`
- `conflict_route_exit_idx`
- `conflict_dist_to_entry_m`
- `conflict_dist_to_exit_m`
- optional `gap_open_frame`
- optional `gap_close_frame`

Useful training losses:

- occupancy BCE over future bins
- gap-state CE
- phase CE
- optional phase temporal margin in stable regions
- optional phase smoothness only inside stable same-phase segments

Important:

- Do not force temporal smoothness near true phase transition frames.
- The model must be allowed to switch quickly when the correct go gap opens.

## Negative Samples

Negative samples are important for phase timing.

Recommended forms:

- same scene / route / conflict area, but modified current speed or speed margin
- bad-route short pre-failure windows
- windows where `go` was predicted even though `go_min_speed` was effectively unreachable
- windows where ego went too fast and violated `chase_max_speed`

These negatives should mainly supervise:

- `decision_phase`
- `control_phase`
- boundary feasibility
- gap state

They should not require a fabricated trajectory GT.

Trajectory-side use, if needed later:

- ranking / inequality constraints
- not synthetic L1 regression

Example:

- Under a `go` counterfactual, first-step progress should be greater than under a matched `yld` counterfactual.
- Under a `yld` counterfactual, the first 1-2 predicted points should be more conservative.

## Open Questions

- How to generate `conflict_occupancy_future_bins` without relying on online detection?
- Should temporary occupancy be total occupancy or direction-specific occupancy?
- Should `gap_state` be manually labeled or derived from occupancy + ego timing?
- Should `conflict_timing_summary` be a supervised label, a deterministic computation, or both?
- How much temporal phase modulation should be allowed through `alpha`?

Current preference:

- Start with sparse, label-derived temporary occupancy if available.
- Add distance-to-entry / exit labels first.
- Compute time-to-conflict with a speed floor or predicted speed inside the model.
- Add temporal phase prior only after the labels are stable.

## Current Label Spec (V2, 2026-04-26)

This section records the **current implemented label version** after switching away from all-actor geometry occupancy.

### 1. Temporary occupancy bins

Current implementation:

- `temporary_occupancy_cover_bins`: shape `(13,)`
- `temporary_occupancy_cover_valid`: shape `(13,)`

Semantics:

- all three families currently use the same `13` raw `4Hz` slots:
  - `borrow`
  - `merge`
  - `junction`
- bins are **current-cover driven only**
- no all-actor bbox occupancy
- no `gap_state`
- no occupancy actor tracking

Meaning of one slot:

- `1`: that future route frame still has a family-matched `current_cover`
- `0`: it does not

Family-matched current-cover means:

- `merge`: `merge_meet`
- `borrow`: `borrow_cross_meet`
- `junction`: `junction cross meet`

So this label is now explicitly:

```text
temporary occupancy = rollout of family-matched current_cover over future route frames
```

### 2. Go / yield soft labels

Current implementation also writes:

- `go_opportunity_prob`
- `yld_pressure_prob`
- `go_opportunity_valid`

with:

```text
yld_pressure_prob = 1 - go_opportunity_prob
```

Current usage:

- these labels are written inside active conflict windows
- pre-entry frames use the `go-able cycle` logic
- from `go_frame` until window end:
  - `go_opportunity_prob = 1`
  - `yld_pressure_prob = 0`
  - `go_opportunity_valid = 1`

Reason:

- once expert has already committed to `go`
- and the route is still inside the same active window
- we still want the label to remain strongly on the `go` side

### 3. Reference frame and reference pass time

Each active window uses:

- `reference_frame = expert go frame`

Reference timing:

- `raw_reference_pass_time_bins`
  - expert `go` phase duration measured in raw `4Hz` route bins

This is used as the conservative passing-time proxy for later frames.

### 4. `lead_blocked_len` and `passable_len`

The current rule no longer treats `lead_blocked_len` as "prefix count of consecutive ones".

Instead:

- scan the bins from left to right
- find the **first zero-run that is long enough** for the required pass-time

Then define:

- `lead_blocked_len`
  - the start index of that qualifying zero-run
- `lead_open_len`
  - the zero-run length itself
- `passable_len`
  - total usable length from now until that zero-run ends
  - i.e.:

```text
passable_len = lead_blocked_len + lead_open_len
```

Examples:

- `0111000000`, required pass-time `= 4`
  - `lead_blocked_len = 4`
  - `lead_open_len = 6`
  - `passable_len = 10`

- `0110000000`, required pass-time `= 4`
  - `lead_blocked_len = 3`
  - `lead_open_len = 7`
  - `passable_len = 10`

Important intuition:

- the initial `1`s are not treated as "impossible forever"
- if ego can still wait through those blocked bins and then enter the opening in time,
  that whole prefix-plus-opening span is part of the passable opportunity

### 5. Current `go-able` rule

For a given frame, after distance-adjusted pass-time is computed:

- `lead_blocked_len <= reference_lead_blocked_len`
- `passable_len >= adjusted_ref_pass_time_bins`

If both hold, that frame is treated as `go-able`.

Compared with the older version:

- we do **not** compare only zero-run length anymore
- we compare the full passable span

This is the current intended behavior for patterns like:

- `0111...`
- `0110...`
- delayed opening after a short blocked prefix

### 6. Distance adjustment for pass time

The pass-time adjustment is no longer anchored to `area_start`.

Current primary distance reference:

- take the `front_s` at the expert `go_frame`
- for the current frame, compute route-progress distance to that `go_frame` position

So the adjustment now answers:

```text
how much farther is ego from the expert launch position than it was at expert go time?
```

This is intentionally decoupled from later tuning of:

- `merge area start`
- `junction area start`
- `junction area end`

Fallback behavior:

- if `go_frame front_s` is missing, fallback may still use conflict-distance
- but this now records an explicit issue

### 7. Issue recording

Current postprocess does **not** silently fallback anymore.

If route-progress reference is missing, debug issue strings are written, e.g.:

- `missing_go_frame_front_s_fallback_conflict_distance`
- `missing_go_frame_front_s_no_fallback`
- `missing_front_s_fallback_conflict_distance`
- `missing_front_s_no_fallback`

Other issues may still be merged in, for example:

- `reference_pass_time_clipped_to_passable_len`
- `reference_has_no_opening`

### 8. Video demo fields

Current `generate_stage1_label_video_lite.py` demos the new label with:

- `tempocc bins=...`
- `tempocc valid=...`
- `tempocc go=... yld=... valid=... cyc=... acc=...`
- `tempocc ref blk=... pass=...`
- `tempocc cur blk=... pass=... goable=...`
- `tempocc refpass raw=... adj=...`
- `tempocc issue=...`

This should be treated as the current debugging contract for validating the label.
