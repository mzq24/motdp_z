# Temporary Occupancy for Conflict-Aware Phase Timing

Date: 2026-04-25

## Summary

This note records the current discussion around using **temporary occupancy** to improve `yld/go` phase timing.

## Current Label Docs

For the current stage1 label line, the two main documents to keep in sync are:

- `docs/motdp.md`
  - the higher-level project / stage1 label context
  - stable conclusions around:
    - `merge / junction / borrow`
    - stage1 speed labels
    - conflict-area route-mask expectations

- `docs/tmp/temporary_occupancy_phase_reasoning_20260425.md`
  - the detailed working note for:
    - temporary occupancy
    - `go_opportunity_prob`
    - conflict-area auxiliary labels
    - current label geometry / area conventions

Practical rule:

- use `docs/motdp.md` for high-level stable context
- use this note for the latest detailed label semantics

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

### 0. Current conflict-area geometry (2026-04-27)

Current area geometry is:

- `borrow`
  - keep the existing corridor-subset logic
  - `area start / end` come from borrow conflict progress on the corridor
  - no major geometry change is currently needed

- `merge`
  - `area start = first_conflict_s_m`
  - `area end = last_conflict_s_m + 8m`
  - interpretation:
    - merge start is already aligned to the conflict cluster start
    - merge end keeps a modest post-margin for ego box length and small extra room

- `junction`
  - `window start = cluster center - 15m`
  - `area start = first_conflict_s_m`
  - `area end = cluster center + 7m`
  - this is stricter than the older `center - 7m` start
  - current preference is to align junction area start more directly with conflict geometry

Current expectation for `conflict_area_route_mask`:

- `borrow`
  - many `17/20`, `18/20`, `19/20`, `20/20` masks are expected because the corridor can be long

- `merge / junction`
  - should **not** be dominated by `20/20`
  - local shard reruns on `shard01` and `shard08` are consistent with this expectation

### 1. Postprocess-only temporary occupancy

Current temporary occupancy is produced in a postprocess step, not in the main stage1 precompute.

Current top-level fields:

- `temporary_occupancy_cover_bins`: shape `(13,)`
- `temporary_occupancy_cover_valid`: shape `(13,)`
- `go_opportunity_prob`
- `yld_pressure_prob`
- `go_opportunity_valid`

The bins are always written route-wise inside the same active conflict window.

### 1.5. New conflict-area auxiliary labels (2026-04-27)

We now also want a small set of conflict-area-centered auxiliary labels directly from stage1 precompute.

Current top-level fields:

- `conflict_dist_to_entry_m`
- `conflict_dist_to_exit_m`
- `conflict_time_to_entry_s`
- `conflict_area_status`

Meaning:

- `conflict_dist_to_entry_m`
  - route distance to conflict-area entry
  - positive means ego is still before the area
  - `<= 0` means ego is at or beyond entry

- `conflict_dist_to_exit_m`
  - route distance to conflict-area exit
  - useful for separating:
    - just before entry
    - already inside
    - already passed

- `conflict_time_to_entry_s`
  - helper quantity only
  - computed from:
    - `max(conflict_dist_to_entry_m, 0) / max(|v_curr|, speed_floor)`
  - current implementation:
    - speed floor `= 0.5 m/s`
    - cap `= 10s`
  - this should not replace the distance labels as the main supervision

- `conflict_area_status`
  - 4-code enum:
    - `0 = none`
    - `1 = before`
    - `2 = inside`
    - `3 = after`
  - practically, the useful semantic states are:
    - before / inside / after

Current implementation notes:

- prefer scene-route progress when available:
  - `area_start_s_m`
  - `area_end_s_m`
  - `front_s`
- fallback to the existing local-interval logic when scene-route progress is not available
- this makes the auxiliary labels family-agnostic enough to cover:
  - `borrow`
  - `merge`
  - `junction`
- in practice, `conflict_dist_to_entry_m` is intended to align semantically with
  the tempocc debug quantity `distance_to_area_start_m` (`distA` in video)

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

### 6.5. Boundary speed naming cleanup debt

Current phase-boundary semantics should be interpreted as phase-specific speed
intervals:

- `yld phase`
  - lower bound: `0`
  - upper bound: family `yld_max`

- `go phase`
  - lower bound: family `go_min`
  - upper bound:
    - `merge`: `chase_speed_max` / front-following cap
    - `junction`: use `junction_yld_max_speed` as a `go_max` when the phase
      reference is `current_area_actor` (`role=3`)
    - `borrow`: no dedicated go upper bound yet

Important naming caveat:

- `junction_yld_max_speed` is not purely a yld-only concept anymore.
- In `junction phase=go + role=3`, the same value is currently the most
  reliable upper-speed cap for the current area actor.
- Video / consistency debug should therefore use neutral wording like `upper`
  rather than `chase` or `yld` when showing the effective cap.

Cleanup TODO:

- After label bugs settle, consider introducing neutral derived names such as:
  - `phase_speed_lower_mps`
  - `phase_speed_upper_mps`
  - `phase_speed_lower_valid`
  - `phase_speed_upper_valid`
  - `phase_speed_upper_source`
- Keep existing family fields as raw/source labels until the rename is planned.

### 6.6. Phase-object binding is post-hoc interpretation

Current `conflict_decision_phase` is still primarily an ego-state / area-state
label. It is not originally bound to an object.

`phase_ref` in `postprocess_stage1_phase_object_binding.py` is therefore a
post-hoc relation interpretation:

```text
phase + current_cover / future_cover / opening evidence
-> phase_ref role / actor
```

Important implications:

- `phase_ref=current_area_actor` means the postprocess found a current cover
  that still belongs to the active conflict area.
- It does **not** mean the original phase label was generated with that actor
  as an explicit anchor.
- For `merge`, current cover must still overlap the merge area before it can be
  treated as `current_area_actor`; otherwise it may just be a front/chase cover
  after the actor already left the merge conflict area.

`open_unbounded` boundary cases:

- usually mean there is no usable future/current actor boundary
- raw scalar fallback is often the full range:
  - `yld_max = 30`
  - `go_min = 0`
- this fallback is useful for debug but should not supervise scalar boundary
  loss

Current rule:

- if boundary/object is missing or open-unbounded:
  - `conflict_phase_boundary_scalar_loss_valid = 0`
- if boundary is actor-conditioned but does not match the post-hoc phase ref:
  - `conflict_phase_boundary_actor_match = 0`
  - `conflict_phase_boundary_scalar_loss_valid = 0`

Therefore many `phase_ref != boundary_ref` mismatches are expected and harmless,
as long as scalar loss is masked. The cases worth inspecting are the ones where
an actor-conditioned boundary would otherwise be used as supervision for the
wrong phase object.

Expected mismatch family: object-free / open-unbounded boundary

```text
merge + phase=yld
phase_ref=none
boundary_ref=open_unbounded
current_candidate.gate_reason=merge_current_cover_outside_area
future_candidate.gate_reason=no_future_cover
```

and similarly:

```text
junction + phase=yld
phase_ref=none
boundary_ref=open_unbounded
current_candidate.gate_reason=missing_current_cover
future_candidate.gate_reason=no_future_cover
```

Interpretation:

- there is no object that can be used as an actor-conditioned boundary target
- for merge, a current cover may exist but has already left the merge conflict area
- for junction, there may simply be no current or future cover in the visible /
  gated evidence
- there is no usable future cover yet
- therefore no actor-conditioned `yld/go` scalar boundary exists
- raw fallback boundary is the full range:
  - `go_min = 0`
  - `yld_max = 30`
- this is a normal mismatch and should not supervise scalar boundary loss

Speed interpretation for this case:

- `yld/go` boundary should be ignored
- if speed reasoning is needed, use independent speed constraints:
  - `vchase` / `chase_speed_max` where valid
  - `merge_follow_through_vbmin` where valid
- do not use junction `vchase`; junction cross timing should use its boundary
  labels instead

Why this can happen:

- ego has entered / passed the merge area, so future cover may no longer be
  computed as an actor-conditioned boundary
- or the next rear actor has not reached the future-cover gate yet
- in junction, the active/yld window can continue even when no current/future
  actor is visible or gateable in the current frame

Important phase caveat:

- current `phase=yld` can still be caused by ego-state / speed behavior
  rather than yielding to a next actor
- e.g. ego may be decelerating because of `vchase`, not because it is yielding
  to a future merge actor
- this is another sign that phase cause attribution should be handled later as
  a separate consistency/cause analysis step

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

Important semantic distinction:

- `go_opportunity_prob` means "a go opportunity is available / selectable now"
- `decision_phase = go` means "the expert / policy actually chooses to go now"

So these two labels should not be collapsed into one another.

Useful interpretation:

- `go_opportunity_prob` is an affordance / temporal opportunity prior
- `decision_phase` is the chosen behavior mode
- `go_opportunity_prob` should influence `decision_phase`, but should not hard-define it

Valid combinations:

- high `go_opportunity_prob` + `decision_phase=go`
  - opportunity exists and expert chooses to take it
- low `go_opportunity_prob` + `decision_phase=yld`
  - no usable opportunity yet; standard wait/yield case
- high `go_opportunity_prob` + `decision_phase=yld`
  - opportunity may exist, but expert/policy remains conservative
  - possible reasons: speed/reachability, chase cap, preparation state, style difference, or label noise
- low `go_opportunity_prob` + `decision_phase=go`
  - risky case to inspect
  - possible reasons: already inside area, occupancy miss, aggressive expert, OOD, or label bug

Training implication:

- do not enforce:

```text
decision_phase=go  <=>  go_opportunity_prob high
decision_phase=yld <=>  go_opportunity_prob low
```

- prefer:

```text
temporary occupancy
+ distance / entry status
+ speed reachability
+ boundary / chase
-> go_opportunity_prob

go_opportunity_prob
+ scene / route / state
+ expert behavior style
-> decision_phase
```

Model-side usage:

- keep the current prior-style modulation:

```text
final_decision_phase_logits =
    base_decision_phase_logits
  + alpha * [yld_pressure_logit, go_opportunity_logit]
```

- keep `alpha` moderate, e.g. `0.3 ~ 0.5`
- treat `go_opportunity_prob` as a prior, not as a replacement for `decision_phase`

Consistency loss should be weak and asymmetric:

- `decision_phase=go` while `go_opportunity_prob` is very low can receive a small penalty
- `decision_phase=yld` while `go_opportunity_prob` is high should not be strongly penalized
- the latter can be a legitimate conservative choice rather than a contradiction

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

## Phase Object Binding Postprocess (2026-05-05)

We added a postprocess-only label that connects area/ego phase semantics back to
the object or opening that the phase refers to.

Reason:

- `conflict_decision_phase` is area / ego-state based.
- speed boundaries are object-conditioned through `current_cover` /
  `future_cover`.
- front-view perception often cannot observe the next / rear vehicle until it is
  close, so an opening may be real but object-unbounded.

Role codes:

- `0 none`
- `1 yld_target_actor`
- `2 go_before_next_actor`
- `3 current_area_actor`
- `4 open_unbounded`

These roles are relation labels between ego's current phase and an
object/opening, not intrinsic actor classes. In GNN terms, this is closer to an
edge/relation attribute than a node label.

Implemented fields:

- `conflict_phase_ref_role`
- `conflict_phase_ref_actor_id`
- `conflict_phase_ref_actor_valid`
- `conflict_phase_open_unbounded`
- `conflict_phase_boundary_ref_role`
- `conflict_phase_boundary_mode`
- `conflict_phase_boundary_state_valid`
- `conflict_phase_boundary_object_missing`
- `conflict_phase_boundary_scalar_loss_valid`
- `conflict_phase_boundary_ref_actor_id`
- `conflict_phase_boundary_ref_valid`
- `conflict_phase_boundary_actor_match`

Current semantics:

- `current_cover` is usually treated as an actor that currently covers the route
  / area, so inside an active conflict window it maps to role `3`.
- Merge is stricter: current cover must still overlap the merge area. Once that
  actor has left the area and only remains a front/chase cover, phase-object
  binding does not use it as `current_area_actor`.
- Merge mixed state rule:
  - if current cover is still inside the merge area and gated future cover is
    already available, bind phase reference to the future actor instead of the
    current area actor
  - reason: merge yld/go scalar boundaries are future-actor conditioned in this
    state, while the current area actor is mainly a clearing/chase constraint
  - `phase=yld` uses role `1 yld_target_actor`
  - `phase=go` uses role `2 go_before_next_actor`
- `future_cover` becomes a next-actor reference only when it passes the default
  gate:
  - `frame_index <= 6`
  - `d_bg <= 20m`; missing `d_bg` is an issue, not a fallback to `distance`
- `phase=go` with no current cover and no close future cover becomes
  `open_unbounded`.
- Boundary relation modes:
  - `future_actor_boundary`: old future-cover yld/go speed boundary.
  - `current_clear_transition`: once the actor becomes `current_cover`, the
    old go-before boundary is no longer meaningful, but the phase/boundary
    relation still binds to the current actor as role `3`.
  - `open_unbounded`: no close future cover; scalar speed thresholds may be
    unconstrained (`yld=30`, `go_min=0`) and `actor_id=-1`, but the boundary
    state is still valid as role `4`.
- `conflict_phase_boundary_state_valid` means the boundary state/relation is
  meaningful. `conflict_phase_boundary_ref_valid` only means an actor id exists.
  Therefore `open_unbounded` has `state_valid=1` and `actor_valid=0`.
- `conflict_phase_boundary_object_missing=1` means the phase is still active
  but no current/future bbox reference was available to construct an
  actor-conditioned boundary. This is different from a truly unconstrained
  opening.
- `conflict_phase_boundary_scalar_loss_valid=0` should mask scalar boundary
  losses for missing-object or open-unbounded cases. Existing scalar values may
  still contain full-range fallbacks such as `yld=30` / `go_min=0`, but these
  should not be treated as hard supervision.
- Boundary reference keeps the original threshold scalar source/provenance.
  The postprocess no longer rewrites boundary relation to hide mismatches.
- If actor-conditioned scalar boundary and post-hoc phase reference disagree,
  `conflict_phase_boundary_actor_match=0` and
  `conflict_phase_boundary_scalar_loss_valid=0`.
- This label does not rewrite phase or boundary labels; it exposes their
  object/reference alignment and masks unsafe scalar supervision for training
  and debugging.

Shard08 consistency audit note:

- `yld_over_max_no_decel` examples such as
  `Town12_Rep0_1038_0_route0_11_08_17_08_36` are better interpreted as
  `candidate_chase_limited_go`: the label says yld because ego is slowing, but
  the effective cause is traffic-flow / chase speed, not yielding to the next
  future actor.
- Single-frame `go_under_min_no_accel` / `junction_go_over_max_no_decel`
  examples such as `Town12_Rep0_1105_0_route0_11_08_02_43_48 frame 59` are
  dangerous-looking frames, but not label bugs. Treat them as debug-only unless
  they form a consecutive segment of at least two frames.
- Current route-level consistency issue filtering should therefore:
  - skip collision routes
  - ignore `candidate_chase_limited_go` when evaluating yld/go scalar bugs
  - ignore one-frame isolated issues
  - inspect `chase_over_max_no_decel` separately because chase is intentionally
    strict and often useful.

Shard01 consistency audit note:

- Borrow appears heavily in shard01, so a second benign issue family appears.
- `candidate_yield_after_actor_no_decel_needed`:
  - examples: `Town12_Rep0_679_0_route0_11_08_05_44_31 frame 50-51`,
    `Town13_Rep0_1073_2_route0_11_08_11_43_24 frame 36-37`
  - surface issue: `phase=go` with `go_under_min_no_accel`
  - interpretation: ego is probably passing behind the actor, not trying to
    beat it in front
  - cue: `go_min` can be very large (`>20m/s`), while `yld_max` is close to
    the actual traffic/ego speed
  - this means the go-before relation is not the right scalar supervision for
    the current behavior; do not count this as a label bug unless it persists
    in a qualitatively unsafe way.
- Borrow `yld_over_max_no_decel` local blips:
  - examples: `Town12_Rep0_3687_0_route0_11_09_07_06_03 frame 47-48`,
    `Town13_Rep0_1756_0_route0_11_07_23_52_29 frame 44-45`
  - these are inside a longer slowdown-to-stop trend
  - a local two-frame weak decel / small accel does not mean the yld label is
    wrong; inspect the surrounding frames before escalating.
- For future full-dataset / new_hpc issue review, classify in this order:
  - collision route: skip consistency bug review
  - `candidate_chase_limited_go`: phase-cause issue, not yld-boundary bug
  - `candidate_yield_after_actor_no_decel_needed`: relation/cause issue, not
    go-boundary bug by itself
  - isolated one-frame issue: debug only
  - borrow yld blip inside longer slowdown-to-stop: debug only
  - `chase_over_max_no_decel`: separate strict-chase review bucket
  - remaining non-chase consecutive issues: inspect video first.

### 8.5. Merge follow-through `vbmin` postprocess

Current issue:

- merge future-cover boundary has a `vbmin` component:
  - `go_min = max(pass_before_speed, vbmin)`
  - `vbmin` is approximately the background traffic speed
- after the future actor becomes `current_cover`, the old go-before boundary is
  no longer meaningful and `merge_go_min_speed` can become unconstrained
- however, from a control/training perspective, ego still should not slow down
  abruptly after entering the merge area

V1 postprocess:

- script:
  - `scripts/data_tools/postprocess_stage1_merge_follow_through_vbmin.py`
- merge-only fields:
  - `merge_follow_through_vbmin`
  - `merge_follow_through_vbmin_valid`
  - `merge_follow_through_vbmin_actor_id`
  - `merge_follow_through_vbmin_actor_valid`
- valid region:
  - `family == merge`
  - active conflict window
  - ego is inside the conflict area, using `conflict_area_status == inside`
    or distance fallback `dist_to_entry <= 0 < dist_to_exit`
- source:
  - carry the strongest observed merge traffic speed from recent
    `merge_thresholds.bg_speed_mps`, `future_cover.other_speed`, or
    `current_cover.other_speed`
  - do not lower the carried floor when the background/current actor slows,
    because that slowdown can be ego-induced after insertion/collision
- semantics:
  - this is not the old go-before boundary
  - it is a follow-through speed floor after ego has committed into the merge
    area, lasting until ego exits the area

### 9. Conflict-area route mask sync note

As of `2026-04-27`, conflict-area localization has an offline route-token mask:

- `conflict_area_route_mask`: shape `(20,)`
- `conflict_area_route_mask_valid`: shape `(20,)`

Purpose:

- separate the long active `window` from the short route segment that should be treated as the actual conflict area
- avoid the old online target where `start_frame:end_frame` could make almost all 20 route tokens positive
- give the model a direct route-aligned supervision signal for ego distance/proximity to the conflict area

Implementation contract:

- generated in `scripts/data_tools/precompute_semantic_labels.py`
- passed through by `dataset/unified_carla_dataset.py`
- consumed by policy before falling back to the old frame-window target
- synchronized between:
  - main worktree: `/media/z/data/mzq/others/MoT-DP`
  - joint worktree: `/media/z/data/mzq/others/MoT-DP-joint-state-speed`

Semantics:

- the mask is route-token aligned, not a time-window label
- route tokens are mapped to local route arclength
- conflict area uses a local interval plus a small tolerance margin
- `merge`, `junction`, and `borrow` all use this offline route-mask path
- borrow should not be hardcoded to all 20 positive tokens

Backward compatibility:

- if old packed data has no offline mask, dataset fills:
  - `conflict_area_route_mask = zeros(20)`
  - `conflict_area_route_mask_valid = -1`
- policy treats valid `< 0` as unknown and falls back to the legacy online frame-window target

Labeling-session checks:

- inspect positive-token count by family after relabeling
- `merge` / `junction` should not be dominated by `20/20` positives
- `borrow` may be longer, but should still come from its area/corridor interval rather than a hardcoded all-positive mask
- compare `window` length vs `conflict_area_route_mask` span separately in debug videos

### 10. Joint-state model / training integration note

As of `2026-04-27`, the joint-state worktree has started consuming the new
conflict timing labels directly.

Worktree:

- `/media/z/data/mzq/others/MoT-DP-joint-state-speed`

Dataset interface:

- `dataset/unified_carla_dataset.py` now passes through:
  - `conflict_dist_to_entry_m`
  - `conflict_dist_to_exit_m`
  - `conflict_time_to_entry_s`
  - `conflict_area_status`

Joint-state payload additions:

- `conflict_area_status_logits`: 4 classes
  - `none`
  - `before`
  - `inside`
  - `after`
- `conflict_timing_values`: 3 normalized scalars
  - `dist_to_entry / conflict_timing_dist_norm_scale`
  - `dist_to_exit / conflict_timing_dist_norm_scale`
  - `time_to_entry / conflict_timing_time_norm_scale`

Model changes:

- `model/transformer_for_diffusion_multi_head.py`
  - adds an `area_status` state token
  - adds a `conflict_timing` state token
  - decodes:
    - `conflict_area_status_logits`
    - `conflict_timing_values`

Policy / loss changes:

- `policy/annealed_energy_guidance_policy.py`
  - config gate: `route_b.use_conflict_timing_state`
  - direct supervised losses:
    - `conflict_area_status_loss`
    - `conflict_timing_loss`
  - state diffusion reconstruction losses:
    - `state_conflict_area_status_recon_loss`
    - `state_conflict_timing_recon_loss`
  - inference/debug outputs:
    - `conflict_area_status_probs`
    - `conflict_timing_values`
    - `conflict_dist_to_entry_m`
    - `conflict_dist_to_exit_m`
    - `conflict_time_to_entry_s`

Training config changes:

- `config/tmp/pdm_hpc_route_b_lidar_bev_stage1_joint_state_fulltrain_val.yaml`
  - `use_temporary_occupancy_phase: true`
  - `use_conflict_timing_state: true`
  - `use_speed_profile_head: false`
  - `speed_profile_loss_weight: 0.0`
  - `conflict_area_status_loss_weight: 0.25`
  - `conflict_timing_loss_weight: 0.25`
  - `conflict_timing_dist_norm_scale: 30.0`
  - `conflict_timing_time_norm_scale: 10.0`

Design intent:

- `conflict_area_route_mask` tells the model where the conflict area lies on the route
- conflict timing/status tells the model where ego is relative to that area
- temporary occupancy tells the model whether the area is opening / blocked over future bins
- `go_opportunity_prob` is still an affordance prior
- `decision_phase` remains the actual chosen behavior mode
