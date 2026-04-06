# Route B Joint Route Diffusion Notes

Date: 2026-03-30

## Summary

This note records the joint route diffusion change for the Route B ego path, the new route normalization requirement, local/HPC setup notes, and the current explanation for slower epoch time after the change.

## Claude Session Reference

For the current Route B front-risk / route+LiDAR discussion, the Claude VSCode conversation archive is:

- session id: `623b8650-ca93-487e-b125-8bc241c2fd69`
- archive file:
  - `/home/z/.claude/projects/-media-z-data-mzq-others-MoT-DP/623b8650-ca93-487e-b125-8bc241c2fd69.jsonl`

Current handoff point from that conversation:

- the user wants the new front-route-risk path to move toward an independent structure first
- old legacy energy compatibility is not the short-term priority
- input design still needs follow-up discussion, especially what should remain in the new path beyond route and LiDAR

Operational note:

- if we need to reload the Claude context later, open the JSONL above directly instead of searching local tmux sessions

## What Changed

### 1. Joint ego diffusion state

The ego denoising path now uses a joint 26-point diffusion state:

- first 6 points: trajectory
- last 20 points: route

The route branch is no longer a pure learnable query in `forward_ego`. It is built from noisy route waypoint coordinates, so route is now a true diffusion branch in the ego path.

Files:

- `model/transformer_for_diffusion_multi_head.py`
- `policy/annealed_energy_guidance_policy.py`

### 2. Route-specific normalization

Route diffusion now depends on a dedicated per-waypoint absolute normalization file:

- key: `route_abs_stats_path`
- file format: NPZ
- tensors:
  - `route_abs_mean`: `(20, 2)`
  - `route_abs_std`: `(20, 2)`

This is separate from trajectory abs/global-abs normalization.

Files:

- `policy/annealed_energy_guidance_policy.py`
- `training/train_carla_bev.py`
- `dataset/compute_action_stats.py`
- `config/pdm_local_route_b.yaml`
- `config/pdm_hpc_route_b.yaml`

### 3. Fine-BEV sampling structure

Before the shared decoder:

- trajectory tokens always do center sampling from `bev_feature_upsample`
- route tokens now also do center sampling from `bev_feature_upsample`

At low timesteps, only trajectory tokens receive additional detail features:

- traj-local detail from a fixed rear-biased stencil
- route-based detail using all 20 route waypoints as look-ahead anchors

The ego self-attention mask remains one-way:

- `traj -> route`: allowed
- `route -> traj`: blocked

Reason:

- route already receives its own fine-BEV center sample
- trajectory can already consume route information through decoder interaction and `route_features`

### 4. Energy guidance scope

Energy guidance remains trajectory-only in v1.

During inference:

- trajectory and route denoise jointly in the ego path
- energy correction is applied only to the trajectory slice
- route is carried along as a jointly denoised but non-energy-corrected branch

## Local And HPC Notes

### Local

Formal local route stats file:

- `/media/z/data/dataset/pdm_lite_mini/route_abs_stats.npz`

The local Route B config already points to this path.

### HPC

Formal HPC route stats file:

- `/workspace1/z_project/dataset/pdm_lite/tmp_data/route_abs_stats.npz`

The HPC Route B config points to this path:

- `config/pdm_hpc_route_b.yaml`

### Packed-sample route stats generation

`dataset/compute_action_stats.py` was updated so `--mode route_abs` now prefers:

- `train/samples_packed.pkl`
- `val/samples_packed.pkl`

and only falls back to scanning individual `*.pkl` files if packed samples are absent.

This matters on HPC because scanning hundreds of thousands of individual pkl files is much slower than reading the packed files.

## LiDAR BEV Compatibility Mode

After adding side/rear LiDAR-BEV detail sampling, Route B now has an explicit compatibility switch:

- key: `route_b.use_lidar_bev_detail`
- baseline Route B configs: `false`
- LiDAR experiment configs: `true`

The purpose is to keep one code line while treating:

- non-LiDAR Route B
- LiDAR-enhanced Route B

as two different model variants.

### What the switch does

When `use_lidar_bev_detail: false`:

- dataset does not try to load `transfuser_lidar_bev/*.npy`
- policy does not pass LiDAR BEV into the model
- the decoder does not instantiate LiDAR-specific parameters:
  - `lidar_bev_encoder`
  - `traj_lidar_detail_attn`

So the non-LiDAR path stays behaviorally close to the pre-LiDAR version.

When `use_lidar_bev_detail: true`:

- dataset loads and inverts TransFuser LiDAR BEV histograms
- policy forwards `transfuser_lidar_bev`
- the decoder adds LiDAR obstacle detail into the traj detail fusion path

### Why this matters for checkpoint compatibility

The main point is not just convenience. It is to avoid silent mixed-model warm-starting.

With this switch:

- old non-LiDAR checkpoints should be resumed only with `use_lidar_bev_detail: false`
- LiDAR configs are treated as a separate model variant
- old non-LiDAR checkpoints should not be silently reused for the LiDAR variant

This is stricter than the previous behavior where new branches could be randomly initialized and silently joined into training.

### Dataset behavior

Even when LiDAR loading is disabled, the dataset still returns:

- `transfuser_lidar_bev = zeros(2, 256, 256)`

This keeps batch schema and collate behavior stable, while avoiding LiDAR file IO on the non-LiDAR path.

### LiDAR tensor format

The current Route B LiDAR-detail path uses a TransFuser-style histogram tensor with shape:

- `(2, 256, 256)`

Per-frame storage:

- file location: `transfuser_lidar_bev/<frame_id>.npy`
- dtype: `float16`

Raw channel meaning from `scripts/data_tools/generate_transfuser_lidar_bev.py`:

- channel 0: below-split histogram
  - points with `z <= 0.2m`
- channel 1: above-split histogram
  - points with `z > 0.2m`

Each pixel is a clipped per-cell point count normalized to `[0, 1]`.

### Current inversion convention

For the current Route B code path, the model is fed an inverted LiDAR histogram convention:

- memmap path:
  - `scripts/data_tools/build_lidar_bev_cache.py`
  - default behavior packs `1.0 - raw` into `tmp_data/lidar_bev_fp16.bin`
- per-frame fallback path:
  - `dataset/unified_carla_dataset.py`
  - loads raw `.npy`, then applies `1.0 - raw`

So the batch tensor seen by policy / model is currently:

- `transfuser_lidar_bev: (B, 2, 256, 256)`
- dtype typically `float16`
- already in the same inverted convention on both memmap and per-frame paths

Important operational note:

- other sessions should preserve this convention when updating old branches
- do not mix raw `.npy` histograms with already-inverted memmap tensors
- the exact semantic interpretation matters less than staying consistent across:
  - generator
  - packer
  - dataset
  - model

## Strict Checkpoint Resume

Checkpoint resume was changed back to strict behavior in `training/train_carla_bev.py`.

Current behavior:

- no shape-mismatch filtering
- no `strict=False` warm-start
- resume now requires exact checkpoint/model agreement

Before `load_state_dict(..., strict=True)`, training now checks and prints grouped mismatch information:

- missing keys
- unexpected keys
- shape mismatches in the form:
  - `checkpoint_shape -> model_shape`

Then it raises immediately.

### Practical consequence

This means:

- matching non-LiDAR checkpoint + non-LiDAR config:
  - should load cleanly
- non-LiDAR checkpoint + LiDAR config:
  - should fail loudly because LiDAR branch parameters are absent from the checkpoint

This is the intended behavior. We now prefer fail-fast over silent partial initialization.

### Important caveat

Strict resume does not only test the LiDAR switch. It tests full architectural equality.

In practice, some older Route B checkpoints may still fail even with:

- `use_lidar_bev_detail: false`

if they come from an earlier Route B architecture that also differs in other ways, for example:

- earlier route-detail branch structure
- different decoder depth / width
- older normalization buffers
- pre-joint-route-diffusion layouts

So:

- `use_lidar_bev_detail: false` is necessary for old non-LiDAR continuation
- but it is not sufficient if the checkpoint is older than other major Route B refactors

The switch isolates the LiDAR difference cleanly, but strict resume will still reject any broader architecture drift.

## Why Epoch Time Increased

The current slowdown is expected and mainly comes from the new fine-BEV compute in the ego path.

### Important: the decoder token length is not the main reason

The shared decoder already operated on:

- 6 trajectory tokens
- 20 route tokens/queries

So the sequence length for decoder self-attention is still effectively 26. The main extra cost is not a large jump in decoder token count.

### Main cost increase: more fine-BEV branches

Previously, the ego path mainly used one center point BEV sampling branch for trajectory tokens.

Now the ego path uses four BEV sampling branches:

1. trajectory center sampling
2. trajectory local-detail sampling
3. trajectory route-detail sampling
4. route center sampling

Rough sample-point count per sample changed from:

- old: `6 * 1 = 6`

to approximately:

- traj center: `6 * 1 = 6`
- traj local detail: `6 * 7 = 42`
- traj route detail: `6 * 20 = 120`
- route center: `20 * 1 = 20`

Total:

- new: about `188`

So the BEV sampling workload increased a lot.

### Main cost increase: repeated full-BEV value projection

Each `GridSampleCrossBEVAttention` call contains its own:

- `value_proj = Conv2d(...)`

and each forward call applies that projection to the entire `bev_feature_upsample`.

## Energy Notes

Date: 2026-04-02

This section records the transition from the old Route B energy setup to the newly proposed front-route risk energy.

### 1. Previous Energy Design

The original Route B energy path is anchor-centric:

- energy is trained on anchor trajectories plus GT
- the standard training layout is:
  - anchor slots
  - one GT slot
- supervision comes from:
  - `behavior_labels`
  - `allowed_flags`
  - `energy_targets`
  - `energy_active_mask`

The original meaning of the heads is:

- `front`
- `left`
- `right`
- `pedestrian`
- `offroad`
- `route`

Important property of the old design:

- it depends on negative / forbidden anchors
- it is naturally suited to "which anchor is unsafe" style training
- it is not a clean match to the new scene-level front blocking / hazard idea

### 2. New Front-Route Risk Idea

The new idea is not anchor-negative driven.

Goal:

- detect when the current route is blocked or will soon be blocked
- use that risk mainly as a safety signal for speed reduction / safety cap
- not rely on the old multi-anchor negative-sample formulation

The new label family is route-constrained and scene-level:

- `front_route_distance`
- `front_route_ttc`
- `front_route_risk`
- `front_route_block_risk`
- `front_route_case`
- `front_route_has_lead`
- `front_route_actor_class`
- `front_route_actor_weight`
- `front_route_block_bin`
- `front_route_ttc_bin`
- `front_route_hazard_bin`

Interpretation:

- `case=1`: a current actor already covers the route corridor
  - pursuit / following style TTC
  - blocking risk is also active here
- `case=2`: no current cover, but a future actor will intersect the current route
  - meeting / conflict style TTC
- `hazard_bin` is the discrete hazard summary
- `actor_weight` increases the penalty for bicycle / pedestrian style hazards

### 3. What Is Implemented Right Now

The training code now supports a new switch:

- `route_b.use_front_route_risk_energy`

When enabled:

- the main split Route B path no longer reuses the old `energy_front_head`
- instead, it uses a dedicated `front_route_risk_head`
- supervision is applied on the GT trajectory only, not broadcast to all anchors

Current target priority:

1. `front_route_hazard_bin / 4.0`
2. fallback to `max(front_route_risk, front_route_block_risk)` if the bin field is absent

Current weighting:

- no-hazard samples keep weight `1.0`
- vehicle hazard samples keep weight `1.0`
- bicycle hazard samples use higher weight
- pedestrian hazard samples use the highest weight

Current split-path execution model:

- training:
  - old `left/right/pedestrian/offroad` heads still use the anchor-based energy path
  - new front-route-risk supervision runs through its own GT-only forward path
- inference guidance:
  - old non-front heads can still contribute if enabled
  - front-route-risk guidance is evaluated through the dedicated front-risk head on `pred_x0`

In addition, the split Route B path now feeds the new energy path with the missing context that had previously only existed in the ego denoising branch:

- route waypoints are now passed into `forward_energy(...)`
- LiDAR BEV can now be passed into `forward_energy(...)`

Concretely, for the split path:

- energy queries still start from trajectory-level anchor / GT tokens
- but the decoder now also receives:
  - route center context
  - route detail context
  - route far look-ahead context
  - LiDAR spatial context

This means the new front-risk supervision is no longer limited to the old "trajectory + BEV-only anchor token" input.

LiDAR processing in the model is now:

1. input tensor:
   - `transfuser_lidar_bev` with shape `(B, 2, 256, 256)`
2. lightweight encoder:
   - `Conv2d(2 -> 32, stride=2)` → `(B, 32, 128, 128)`
   - `GroupNorm + GELU`
   - `Conv2d(32 -> 64, stride=2)` → `(B, 64, 64, 64)`
   - `GroupNorm + GELU`
3. ego denoising path:
   - LiDAR detail is sampled with the same 14-point ego detail stencil used by the BEV detail branch
4. split energy path:
   - LiDAR is now also sampled along anchor / GT trajectory points
   - this uses a dedicated LiDAR spatial attention branch on the encoded `(64, 64, 64)` LiDAR feature map

So the new split energy path is no longer "BEV-only":

- route geometry is now part of the energy context
- LiDAR BEV is now part of the energy context

### 4. Current Mixed State

Right now the system is intentionally hybrid:

- `front` in the main split path can use the new independent front-route-risk head
- `left/right/pedestrian/offroad/route` still use the older anchor-based energy supervision
- the split Route B energy path now has access to:
  - route conditioning / route geometry context
  - TransFuser BEV
  - LiDAR BEV
- unified / legacy energy paths have not yet been fully brought to the same input structure

More specifically:

- `forward_ego(...)` already used LiDAR detail before this update
- `forward_energy(...)` now also accepts:
  - `route_points`
  - `transfuser_lidar_bev`
- the main split training / guidance path now passes those tensors through
- unified / legacy routes still need a dedicated cleanup pass if they are to match the split path exactly

So the code is in a transition state:

- old energy path is still present
- new front-risk energy has been split out on the main split path
- the main split path now has the intended route+LiDAR context
- the full future direction is still to reduce dependence on the old anchor-negative setup

### 5. Energy Mode Count

The old code used to assume full-anchor energy training.

It now supports slicing to the first `num_energy_modes` anchors consistently in:

- `compute_split_loss`
- `compute_unified_loss`
- `compute_energy_loss`

This means:

- `num_energy_modes=8` is now a valid lighter-weight option
- `32` still works

Operational recommendation from this session:

- `8` is the safest reduced setting
- `1` is experimental
- `0` is not recommended without a dedicated cleanup pass

Reason:

- `front` can still learn from the GT slot
- but the remaining old anchor-based heads become weak or fragile when the anchor count gets too small

### 6. Local Validation Status

What was locally validated:

- `policy/annealed_energy_guidance_policy.py` compiles
- `model/transformer_for_diffusion_multi_head.py` compiles
- a local smoke test with manually injected
  - `front_route_hazard_bin`
  - `front_route_actor_weight`
  into a mini batch produced a non-zero `energy_front_loss`
- after the route/LiDAR energy-input update, both
  - `policy/annealed_energy_guidance_policy.py`
  - `model/transformer_for_diffusion_multi_head.py`
  pass `python -m py_compile`

What was not yet fully done at the time of writing:

- local mini `samples_packed.pkl` had not yet been fully refreshed with the new `front_route_*` fields
- so the end-to-end local run with real precomputed front-route labels was not yet re-run

What still remains to be validated after the latest change:

- a runtime smoke test on the split Route B path with real `route + transfuser_lidar_bev`
- a larger end-to-end run confirming the new energy path behaves as expected with the heavier route / LiDAR context

### 7. Val Packed Sample Fix

Another issue found during this work:

- `precompute_semantic_labels.py` previously required `samples_packed.pkl` to already exist
- on full HPC data, `val/samples_packed.pkl` was missing

This has now been fixed.

Current behavior of `scripts/data_tools/precompute_semantic_labels.py`:

- if `samples_packed.pkl` exists, load it
- otherwise automatically pack the per-sample `*.pkl` files in that split
- then continue computing labels

Practical result:

- `train` and `val` splits can now both be labeled directly
- no manual "run training once just to create packed samples" step is required for label precompute

### 8. Recommended Near-Term Usage

For the current codebase:

- keep LiDAR detail enabled for Route B LiDAR experiments
- use `use_front_route_risk_energy: true` when the new front-route labels are available
- keep validation enabled in full runs once `val/samples_packed.pkl` and labels are ready

For overnight / lightweight experimentation:

- `validation.enabled: false` is fine
- but then there will be no best-checkpoint selection

### 9. Long-Term Direction

The intended long-term direction is:

- move away from anchor-negative energy as the main formulation
- keep the new front-route risk / block idea as a direct safety signal
- likely use it first for speed limiting / safety cap
- only later revisit whether trajectory-gradient guidance is still needed for this signal

Near-term implementation direction after this update:

- keep the split Route B path as the main experimental path
- treat the split path as the reference implementation for new front-risk energy
- only later decide whether unified / legacy branches should be upgraded to the same route+LiDAR energy input structure

### 10. Multi-Risk Direction (0401 Note Follow-up)

The Bench2Drive note

- `/media/z/data/mzq/others/Bench2Drive/docs/route_b_agent_notes_0401.md`

adds an important clarification:

- a single scalar front-risk score is still too compressed for Route B control

Reason:

- one scalar mixes together at least two different questions:
  - is the route corridor currently occupied / threatened?
  - is it already safe to proceed / release the cap?

This matches the issue observed in the current MoT-DP path:

- label precompute already stores multiple route-constrained signals
  - `front_route_block_risk`
  - `front_route_ttc`
  - `front_route_case`
  - `front_route_has_lead`
  - `front_route_block_bin`
  - `front_route_ttc_bin`
  - `front_route_hazard_bin`
- but the current independent front-risk head still compresses them into one supervision target
  - mainly `front_route_hazard_bin / 4.0`
  - fallback to `max(front_route_risk, front_route_block_risk)`

So the current scalar front-risk path should be treated as:

- a transitional implementation
- useful for bootstrapping the route+LiDAR path
- not yet the desired final Route B safety representation

Preferred next decomposition:

1. hazard / occupancy style signal

- question:
  - is the forward route corridor occupied or threatened?
- candidate names:
  - `front_route_hazard_present`
  - `front_route_corridor_occupied`

2. release / proceed style signal

- question:
  - is it currently safe to continue / merge / enter the corridor?
- candidate names:
  - `front_route_safe_to_proceed`
  - `front_route_gap_available`

Possible richer extension after that:

- replace one scalar with a short occupancy vector along the route corridor
- for example, several distance bins over the first 30-40m of route arc length
- each bin can represent soft occupancy / threat mass rather than only one collapsed score

Why occupancy is attractive:

- it preserves where the blockage sits along the route, not just how risky the full scene is
- it separates "near blocking lead" from "future merge conflict farther ahead"
- it matches the role of LiDAR BEV better than a single scene scalar
- it gives speed control a more direct structure for capping and release

Practical recommendation:

- do not treat the current scalar front-risk head as the final design target
- use it only as the shortest path to validate route+LiDAR conditioning
- the next label/model revision should move toward either:
  - two heads: `hazard_present` + `safe_to_proceed`
  - or a small route-occupancy vector plus an optional release head

Local-first occupancy visualization convention:

- before committing to a new occupancy head, use a provisional occupancy-style
  score only for local video inspection
- current provisional rule:
  - `case=1 current_cover` -> `occ_risk = front_route_block_risk`
  - `case=2 future_cover` -> `occ_risk = front_route_risk`
  - `case=0 none` -> `occ_risk = 0`
- this is intentionally not the final control target
- its purpose is:
  - verify bbox-aware current/future cover geometry
  - verify route-conditioned occupancy semantics locally
  - separate "corridor occupied / threatened" from "safe to proceed"

Local tool support:

- `tools/generate_front_route_label_video.py` now supports:
  - `--frame_min`
  - `--frame_max`
- the video overlay now also shows:
  - `occ_present`
  - `occ_risk`
  - `occ_case`
  - `occ_source`

Local v2 refinement for the current video-only inspection pass:

- do not collapse `current_cover / future_cover` into `chase / meet`
- instead treat them as two different axes:
  - `cover_case`:
    - `current_cover`
    - `future_cover`
  - `interaction_mode`:
    - `chase`
    - `meet`
- for the current local visualization pass:
  - exclude static objects from both occupancy and proceed logic
  - keep the scene route clipped to about `32m` ahead, matching the current
    LiDAR / detail-attention forward range
  - derive `interaction_mode` heuristically from actor-vs-route direction:
    - aligned with route -> `chase`
    - otherwise -> `meet`
- provisional local-only scoring:
  - `occ_risk`
    - `chase` -> `max(block_risk, ttc_risk)`
    - `meet` -> `ttc_risk`
  - `proceed_risk`
    - `chase` -> `max(gap_risk, ttc_risk, block_risk)`
    - `meet` -> `ttc_risk`
- this is still a local video/debug convention only
- it is not yet the committed full-dataset label definition

Local v3 refinement: freeze the objective wait/release state first

- before deciding how `occ`, slowdown, and release confidence should interact,
  first keep only the objective state pieces:
  - `interaction_active`
  - `ego_speed`
  - `wait_state`
  - `release_pulse`
- for the current local-only pass:
  - do not use route progress
  - do not use route remaining
  - do not yet enforce any subjective coupling such as:
    - "slow down before wait"
    - "release requires lower occ than wait"
    - "wait should depend on a calibrated occ score"
- use a simple hysteresis state machine instead:
  - `interaction_active = 1` iff the current frame has a dynamic
    route-interaction actor (`interaction_mode != none`)
  - `wait_candidate = interaction_active and ego_speed <= wait_speed_thresh`
  - if `prev_wait_state == 0`:
    - `wait_state = wait_candidate`
  - if `prev_wait_state == 1`:
    - `wait_state = interaction_active and ego_speed < release_speed_thresh`
  - `release_pulse = 1` iff `prev_wait_state == 1` and `wait_state == 0`
    and `ego_speed >= release_speed_thresh`
- current local defaults:
  - `wait_speed_thresh = 0.5 m/s`
  - `release_speed_thresh = 1.0 m/s`
- this keeps the phase logic clean and inspectable first; the later mapping from
  objective state to a more subjective control policy can be designed afterward

Local v4 debug fix: repair route after release and exclude parked zero-speed actors

- a concrete local bug showed up around the `wait -> release` transition:
  - before wait, the route overlay looked correct
  - after release, the stitched route could bend or jump incorrectly
  - this came from the earlier "scene ego path + tail route" proxy stitch,
    which was too coarse and drifted after the stop-go transition
- local video generation now uses a route-segment stitch built directly from the
  per-frame measurement routes:
  - collect each frame's local route
  - transform each route segment to world coordinates with that frame's
    `ego_matrix`
  - stitch overlapping route segments in world coordinates
  - for each frame, transform the stitched forward route back into the current
    ego frame and clip it to about `32m`
- the local interaction pass now also filters to dynamic actors only:
  - exclude `class == static`
  - exclude parked / stationary zero-speed actors that show neither sufficient
    future speed nor sufficient future displacement
  - keep actors whose current speed may be near zero if they still show clear
    future motion; these can still be valid dynamic interaction actors
- this fix is local-video/debug only for now
- the formal full-dataset `precompute` path has not yet been updated to this
  exact local-v4 convention

### Front-route cheat sheet

The current local/front-route stack mixes several concepts that are easy to
forget while debugging. This section records the exact current meaning.

Sampling frequency:

- raw `measurements/*.json.gz` are treated as `4Hz`
- `future_frames_data[k]` in `front_route` logic means:
  - frame `current + (k + 1)`
  - therefore about `0.25s` per step
- this is different from `ego_waypoints`, which are downsampled to `2Hz`
  in preprocessing

What each axis means:

- `cover_case`
  - `current_cover`: an actor already covers the current route corridor
  - `future_cover`: no actor covers it now, but a future actor box will
    cover it when transformed back to the current ego frame
- `interaction_mode`
  - `chase`: actor direction is roughly aligned with the route direction
  - `meet`: actor direction is not aligned with the route direction; think
    crossing / merging / converging toward the route

How the meet / conflict point is found:

- the meet point is not solved from speed
- instead:
  - take a future actor box
  - transform that future box into the current ego frame
  - find the first point where that transformed box covers the current route
    corridor
- that first covered route point is the current implementation's
  `conflict_pt` / meet point

How current TTC is computed:

- `current_cover / chase`
  - `gap_distance = actor_cover_route_s - ego_route_front_s`
  - `closing_speed = max(ego_speed - lead_speed, 0.1)`
  - `ttc = gap_distance / closing_speed`
- `future_cover / meet`
  - `d_ego = conflict_route_s - ego_route_front_s`
  - `d_bg = distance from actor bbox to conflict point`
  - `meet_dist = d_ego + d_bg`
  - `meet_speed = max(ego_speed + bg_speed, 0.1)`
  - `ttc = meet_dist / meet_speed`

Important consequence when `ego_speed == 0`:

- the meet point itself does not change
- only the TTC changes
- with the current formula:
  - `meet_speed = max(bg_speed, 0.1)`
  - so wait-state frames effectively use only the other actor speed for the
    time term

Why this may be insufficient:

- for pure occupancy semantics this is acceptable:
  - ego is stopped, so a geometry-only conflict point plus actor-only motion
    still describes corridor threat
- for release / proceed semantics it may be too weak:
  - as soon as ego releases, ego will no longer remain at zero speed
  - so a TTC computed with `ego_speed = 0` can overestimate available time

Recorded candidate alternatives for the wait / release case:

1. symmetric proxy

- use `v_ego_eff = bg_speed`
- then `meet_speed = v_ego_eff + bg_speed = 2 * bg_speed`
- this is simple and intentionally pessimistic
- downside:
  - it assumes ego release speed is roughly comparable to the actor speed,
    which is heuristic rather than grounded in current ego state

2. release-aware proxy

- during wait-state frames, replace `ego_speed` with an expected release speed
- examples:
  - `release_speed_thresh`
  - a fixed minimum launch speed
  - the first post-wait speed from local state inspection
- then compute:
  - `meet_speed = v_ego_eff + bg_speed`
- this matches the semantics of:
  - "if ego releases now, how soon would the conflict happen?"

Current recommendation:

- keep the current `ego_speed == 0` formulation for occupancy-style local
  inspection
- when we formalize `wait / release / proceed`, revisit the TTC term for
  `meet` under wait-state frames
- likely first comparison to try locally:
  - current formula
  - `2 * bg_speed`
  - `max(ego_speed, release_speed_thresh) + bg_speed`

Local `release_ready_v1` prototype:

- this is a local-video/debug prototype only
- it is not yet part of the formal packed-dataset labels
- motivation:
  - when `ego_speed == 0`, `proceed_risk` is forced to `0`
  - release should therefore be decided by a separate scene/window signal,
    not by a stopped-state proceed scalar

Current objective definition:

- only evaluate `release_ready_v1` on frames with `wait_state == 1`
- find the next future speed-based release:
  - previous frame speed `<= wait_speed_thresh`
  - current frame speed `>= release_speed_thresh`
- use the current wait frame's extended route (`20 + 12`) to estimate the
  borrow geometry directly:
  - detect `borrow_start` where lateral shift first exceeds the borrow threshold
  - detect `borrow_end` where lateral shift falls back near the own-lane center
  - transform these two route points into world coordinates
  - match them to the nearest future expert ego frames
- this yields:
  - `release_frame_id`
  - `enter_frame_id`
  - `borrow_duration_s`
  - `release_to_return_s`
  - `borrow_distance_m`
  - `return_frame_id`

Current occupancy check used for release:

- take only the current wait frame's borrow segment, i.e. the route slice
  between `borrow_start` and `borrow_end`
- over the next `release_to_return_s` window from the current wait frame:
  - transform each future dynamic actor box back into the current ego frame
  - if any dynamic actor box overlaps that borrowed-lane corridor:
    - `release_ready_v1 = 0`
  - otherwise:
    - `release_ready_v1 = 1`

Current video overlay fields:

- `release_ready`
  - `NA` if not in a valid wait-frame evaluation regime
  - `0/1` otherwise
- `rel_src`
  - current source / failure reason
- `rel_frame`
  - future speed-based release frame
- `enter_frame`
  - first future expert frame nearest to the borrowed-lane entry point
- `ret_frame`
  - future return-to-own-lane frame
- `borrow_t`
  - effective borrow duration for release control, currently
    `release_to_return_s`
- `rel_t`
  - debug field for the same release-to-return duration
- `enter_t`
  - entry-to-return duration; kept as an auxiliary geometric/debug value
- `block_frame`
  - first future frame in the current wait-window that blocks release

### 11. Simple OOD Speed-Control Direction

Before moving to the richer multi-risk / occupancy design, the current practical
goal is simpler:

- use the new safety signal mainly to slow ego down in OOD situations
- push the closed-loop state back toward the in-distribution region seen in training
- do not treat the first version as a full behavior-planning replacement

Why this is the right first step:

- the current dataset is dominated by correct / safe driving samples
- very-close unsafe configurations are rare or absent
- so the model often does fine inside the training distribution
- but when test-time rollout drifts into an unseen close-range situation, it may
  fail to brake because that regime was not present in the data

Important consequence:

- the first safety head should be interpreted mainly as an OOD recovery / safety-cap signal
- not as a precise scalar that fully explains all hazard structure

Representative failure patterns:

1. left lane change with side-by-side vehicle

- training data rarely contains a valid lane change while ego is already abreast
  of a close left vehicle
- so the model may still continue the merge instead of slowing / aborting

2. cut-in with extremely small gap

- moderate-distance cut-in cases may still trigger a reasonable stop reaction
- but when the cut-in vehicle starts from an extremely small gap, the model may
  fail because that regime is outside the training distribution

So the simple near-term objective is:

- when testing enters an OOD close-range state, reduce ego speed
- let the rollout fall back toward a safer, more familiar state distribution
- only after that revisit richer multi-head or occupancy-style formulations

### 12. Raw Measurement Label Audit

As of the current local debug pass, the following raw measurement keys are
available directly from the saved autopilot output:

- `vehicle_hazard`
- `vehicle_affecting_id`
- `light_hazard`
- `walker_hazard`
- `walker_affecting_id`
- `stop_sign_hazard`
- `stop_sign_close`
- `walker_close`
- `walker_close_id`
- `speed_reduced_by_obj_type`
- `speed_reduced_by_obj_id`
- `speed_reduced_by_obj_distance`
- `changed_route`
- `junction`
- plus low-level control fields such as `throttle`, `brake`, `control_brake`

Current meaning from the autopilot code:

- `vehicle_hazard`
  - expert autopilot decided there is a blocking vehicle hazard this frame
- `vehicle_affecting_id`
  - the specific actor id that triggered `vehicle_hazard`
- `speed_reduced_by_obj_*`
  - the object that most reduced the expert target speed in that frame
  - this may be a vehicle, bicycle, traffic light, etc.

Current local release-debug update:

- `release` in the video tool is no longer speed-only
- it now fires from either:
  - the original speed crossing
  - or a throttle-based launch intent:
    - `throttle >= 0.3`
    - `brake == 0`
    - `control_brake == 0`
    - `target_speed > 0.5`

This was added because in `Accident/Town12_Rep0_2208_0_route0_11_08_18_08_45`
the first launch already starts at full throttle while speed is still below the
old `release_speed_thresh`.

Current mini-dataset scan summary (`pdm_lite_mini`):

- total frames scanned: `24612`
- `vehicle_hazard`: `5309` frames across `181` scenes
- `light_hazard`: `1929` frames across `31` scenes
- `stop_sign_hazard`: `1022` frames across `25` scenes
- `stop_sign_close`: `1022` frames across `25` scenes
- `changed_route`: `2706` frames across `181` scenes
- `junction`: `509` frames across `57` scenes
- `walker_hazard`: `0` frames in current mini
- `walker_close`: `0` frames in current mini

Current co-occurrence stats for `vehicle_affecting_id` vs `speed_reduced_by_obj_id`:

- `vehicle_affecting_id` only: `255` frames
- `speed_reduced_by_obj_id` only: `14290` frames
- both present: `5054` frames
- same actor id: `3374` frames
- different actor ids: `1680` frames

Interpretation:

- `vehicle_affecting_id` is a relatively sharp blocker label
- `speed_reduced_by_obj_id` is broader and often exists even without
  `vehicle_hazard`
- when both ids match, the same object is both:
  - the direct hazard
  - and the main speed-reduction source
- when they differ, the scene contains a more interesting conflict:
  - one object is the immediate blocker
  - another object is the dominant speed reducer

Current local debug videos:

- `Accident 2208` rear-collision check with affecting/speed-reduced overlay:
  - `visualizations/front_route_videos/accident2208_affecting_debug_release2_0044-0075.mp4`
- example where `vehicle_affecting_id != speed_reduced_by_obj_id`:
  - `visualizations/front_route_videos/rawlabel_aff_vs_speedred_1506_0060-0075.mp4`
- example traffic-light / `light_hazard` case:
  - `visualizations/front_route_videos/rawlabel_light_hazard_539_0000-0020.mp4`

Current recommendation:

- `vehicle_hazard` / `vehicle_affecting_id` look promising as direct teacher
  signals for blocker attribution
- `speed_reduced_by_obj_*` also looks useful, but should be treated as a
  broader speed-control attribution signal rather than a pure collision label
- `light_hazard` and `stop_sign_hazard` are likely reusable if their local
  videos continue to look clean

### 13. Speed-Conditioned Risk Curve Draft

Current direction:

- do not keep expanding a single scalar `proceed_risk`
- instead, for each frame, predict a small risk curve over candidate speeds
- this keeps the problem supervised and speed-only
- it is intended to help in OOD recovery, not to replace the whole planner

Why this is the cleaner direction:

- `occ_risk == proceed_risk` over-penalizes normal following
- the real question is:
  - "which speeds are still safe in this scene?"
- that naturally becomes a curve, not one scalar

#### 13.1 Candidate Speed Sampling

For each frame:

- let `v_exp` be the expert / recorded ego speed
- only sample speeds around `v_exp`
- clip to a feasible global range:
  - lower bound: `0 m/s`
  - upper bound: `20 m/s`

Current practical proposal:

- use offsets:
  - `[-5, -3, -1, 0, +1, +3, +5] m/s`
- candidate speeds are:
  - `V = clip(v_exp + offsets, 0, 20)`
- remove duplicates after clipping
- keep the index of the exact `v_exp` sample if it survives clipping

Notes:

- this is intentionally a rough local neighborhood, not a strict 1-step
  dynamics model
- a quick mini-dataset scan suggests single-step speed changes are usually
  around `1-2 m/s` at `4Hz`, but we do not need a very strict bound here
- `±5 m/s` is treated as a coarse local counterfactual band around the expert speed

#### 13.2 What Gets Predicted

Per frame, the main target becomes:

- `speed_risk_values: (K,) float32`
  - one risk value for each sampled candidate speed
- optional:
  - `speed_sample_values: (K,) float32`
  - `speed_risk_valid_mask: (K,) bool`

The auxiliary labels we already have can stay:

- `cover_case`
- `interaction_mode`
- `occ_present / occ_risk`
- `wait_state`
- `borrow_t`

These auxiliary quantities help interpret the frame, but the main supervised
object is now the risk curve over speed.

#### 13.3 Semantics of the Curve

The intended semantics are:

- `risk(v) = 0`
  - driving at speed `v` is judged safe enough for this frame
- `risk(v) = 1`
  - driving at speed `v` is clearly unsafe for this frame

This means:

- the label is not "what expert did"
- the label is "what would happen if ego tried this nearby speed instead"

Example:

- current expert speed is `10 m/s`
- for a following scene:
  - `risk(5) = low`
  - `risk(10) = low`
  - `risk(20) = high`

This is exactly the kind of supervision that gives the energy a useful local
gradient in speed-space.

#### 13.4 Wait Frames

Wait frames should still participate in the same speed-curve formulation.

Important rule:

- if `v = 0`, risk should stay low / zero
- positive sampled speeds may still become high-risk
  - for example if the current wait state exists because entering the borrowed
    lane or conflict zone would collide

This is useful because:

- `wait_state` still explains why `v=0` is acceptable
- sampled positive speeds provide the counterfactual supervision
  - e.g. `v=5` at the same wait frame may already be unsafe

So the same frame can provide:

- a safe point at `v=0`
- unsafe points at `v>0`

#### 13.5 V1 Risk Computation

Current recommendation for a first clean version:

- keep `occ` as an auxiliary scene/wait signal
- do not directly copy `occ_risk` into every sampled speed
- compute the speed curve mainly from geometric interaction quantities

##### Chase / Following

For `interaction_mode == chase`:

- use bbox-aware route gap from the current frame
- use the relevant actor speed `v_actor`
- for each sampled ego speed `v`:
  - `closing(v) = max(v - v_actor, 0)`
  - if `closing(v) <= eps`:
    - `ttc(v) = +inf`
  - else:
    - `ttc(v) = gap / closing(v)`
- then:
  - `risk(v) = clip((ttc_safe - ttc(v)) / ttc_safe, 0, 1)`

Rationale:

- this keeps nominal following from becoming high-risk by default
- only faster sampled speeds become dangerous when they shorten TTC too much

##### Meet / Crossing / Borrowed-Lane Conflict

For `interaction_mode == meet`:

- use bbox-aware conflict geometry:
  - `d_ego`: ego-front to conflict point
  - `d_bg`: other actor to conflict point
  - `v_bg`: other actor speed
- for each sampled ego speed `v`:
  - if `v <= eps`:
    - `t_ego(v) = +inf`
  - else:
    - `t_ego(v) = d_ego / v`
  - `t_bg = d_bg / max(v_bg, eps)`
  - `delta_t(v) = abs(t_ego(v) - t_bg)`
- then:
  - `risk(v) = clip((delta_safe - delta_t(v)) / delta_safe, 0, 1)`

Rationale:

- at `v=0`, risk naturally stays low
- as sampled speed increases, the arrival time to the conflict point moves
  closer to the background actor timing
- this is exactly the desired wait/release counterfactual behavior

##### No Active Interaction

For `interaction_mode == none`:

- `risk(v) = 0` for all sampled speeds in v1

This keeps general frames from being inflated by heuristic occupancy terms.

#### 13.6 What Not To Do in V1

To keep the first version clean:

- do not reintroduce `occ_risk` as the moving-speed risk directly
- do not add `v_min / v_max` control heuristics to the target
- do not treat all normal following as negative by assumption
- do not build a full reward-shaped release function here

Instead:

- use `wait / occ` to explain stopped states
- use speed-conditioned TTC / time-gap geometry to label nearby speeds

#### 13.7 Relationship to Scene Family

Scene family is still useful, but not as the main target in this stage.

Current plan:

- first split frames conceptually into:
  - `general`
  - `non-general`
- then use scene family as an auxiliary task:
  - `AccidentTwoWays`
  - `Accident`
  - `Merge / HighwayExit`
  - `Intersection`
  - `CutIn`

Possible later use:

- if the scene-family head becomes reliable enough, it can be added as a
  condition in a future stage
- this is explicitly deferred to a later stage, not part of the first speed-risk implementation

Operational recommendation for the next implementation step:

- prioritize speed control / speed cap over strong trajectory pushing
- treat the safety signal as "slow down now" rather than "solve the whole maneuver"
- keep the complex occupancy / multi-risk idea recorded, but do not block the
  simple OOD-recovery version on it

#### 13.8 Concrete Per-Frame Label Spec

To avoid drifting back into a vague scalar-risk setup, the first concrete target
should be written as a small fixed-size tensor package per frame.

Recommended v1 fields:

- `speed_sample_values: (K,) float32`
  - sampled candidate speeds in `m/s`
  - built from `v_exp + [-5, -3, -1, 0, +1, +3, +5]`
  - clipped to `[0, 20]`
- `speed_risk_values: (K,) float32`
  - risk value for each sampled speed
  - semantic range is `[0, 1]`
- `speed_risk_valid_mask: (K,) bool`
  - `1` where the corresponding speed sample is valid
  - useful if clipping or deduplication leaves fewer than `K` unique samples
- `speed_sample_exp_index: () int64`
  - the index of the exact expert-speed sample after clipping / dedup
  - `-1` only if a bookkeeping failure happens

Recommended supporting labels that stay alongside the curve:

- `cover_case: () int64`
  - `0=none, 1=current_cover, 2=future_cover`
- `interaction_mode: () int64`
  - `0=none, 1=chase, 2=meet`
- `occ_present: () float32`
  - scene/wait auxiliary only
- `occ_risk: () float32`
  - scene/wait auxiliary only
- `wait_state: () float32`
  - whether the expert is in a stopped waiting phase
- `borrow_t: () float32`
  - current best estimate of release-to-return duration for borrow-like scenes

Practical note:

- keep `K=7` fixed in storage
- if clipping creates duplicates, deduplicate first, then pad the tail with:
  - `speed_sample_values = 0`
  - `speed_risk_values = 0`
  - `speed_risk_valid_mask = 0`
- this keeps the packed sample shape stable and easy to batch

#### 13.9 Conversion Rule by Frame Phase

The most important conversion rule is:

- do not turn raw geometry directly into one scalar "risk"
- first decide which phase the frame is in
- then compute the speed-conditioned curve for that phase

Current v1 phase split should be understood as a `wait/move × chase/meet`
grid, plus a `general` fallback:

Important caveat for merge-like scenes:

- `chase` and `meet` are not always mutually exclusive
- in a merge into traffic flow, ego may need to satisfy both:
  - front-car `chase` constraint in the target lane
  - rear-car `meet` / timing-gap constraint in the same target lane
- so the 2x2 grid is the conceptual decomposition, not a claim that every frame
  has exactly one active interaction
- this is especially relevant for `Accident` and `Merge / HighwayExit`

Current local v1 interaction assignment rule is intentionally simple:

- same-direction `current_cover` -> `chase`
- same-direction `future_cover` -> `meet`
- non-same-direction cover -> `meet`
- if the same actor already appears as `current_cover`, do not also keep it as a
  `future_cover` candidate in the same frame

This is a better fit for merge-like cases where:

- a front target-lane vehicle is often a `chase` constraint
- a rear target-lane vehicle, even if moving in the same direction, is still a
  `future_cover` timing constraint and should be treated as `meet`

1. `wait + chase`

- ego is still stopped or nearly stopped
- the relevant interaction is longitudinal / merge-into-flow style
- typical example:
  - waiting to merge into traffic, where the important question is whether ego
    can accelerate into a same-direction gap
- `risk(0)` should stay low
- positive sampled speeds should rise according to chase-style TTC / gap closure

2. `wait + meet`

- ego is stopped because entering the conflict region now would collide
- typical example:
  - borrowed oncoming lane
  - crossing / intersection wait
- `risk(0)` should stay low
- positive sampled speeds should rise from conflict-time geometry
- `occ_risk` is especially useful here as a scene/wait auxiliary

3. `move + chase`

- usually normal following / approaching / merge-follow behavior
- do not copy `occ_risk` into `speed_risk_values`
- use TTC over sampled speeds
- this is the key step that prevents normal following from becoming universally high-risk

4. `move + meet`

- ego is already moving, but conflict timing with another actor still matters
- use sampled-speed arrival-time conflict
- this is the moving counterpart of wait-release geometry

5. `general / no active interaction`

- keep the whole curve low in v1
- this is the default negative regime we need to preserve

In short:

- `occ_risk` mainly explains waiting frames, especially `wait + meet`
- `speed_risk_values` explain which nearby speeds are safe or unsafe within the
  active quadrant
- the two should not be forcibly collapsed into one scalar

#### 13.10 Review Order Across Scene Families

The next local verification pass should be organized by scene family rather than
by isolated routes.

Current family plan:

1. `AccidentTwoWays`

- primary focus:
  - `wait`
  - `release`
  - `meet`
- expected behavior:
  - `risk(0)` low during valid waiting
  - positive sampled speeds rise when the borrowed lane / conflict timing is unsafe

2. `Accident`

- primary focus:
  - `wait + chase`
  - `wait + meet`
  - `move + chase`
- expected behavior:
  - both same-direction merge-into-flow and borrowed-lane style interactions may matter
  - this is the first family where the 2x2 quadrant view really matters
  - some frames may simultaneously have:
    - a front same-lane `chase` constraint
    - and a rear target-lane `meet` constraint

3. `Merge / HighwayExit`

- primary focus:
  - `chase`
  - `gap acceptance`
  - moving-speed risk shape
- expected behavior:
  - low risk near expert speed when the merge is normal
  - higher risk only for faster unsafe counterfactual samples
  - front and rear target-lane vehicles should be treated as different
    constraints, not collapsed too early

4. `Intersection`

- primary focus:
  - `meet`
  - release timing
- expected behavior:
  - the curve should be mainly driven by conflict-point timing rather than following distance

#### 13.11 Current Stage1 Implementation Notes

Current implementation is now slightly more specific than the original v1
spec above.

1. Stage1 speed-energy heads

- the current model predicts three heads:
  - `E_chase`
  - `E_meet`
  - `E_pedestrian`
- total energy is interpreted at inference as:
  - `E_total(v) = max(E_chase(v), E_meet(v), E_pedestrian(v))`

2. Current direct inputs to stage1 speed energy

- explicit geometry:
  - full `route_points`
- implicit scene context:
  - `mode_out`
- speed condition:
  - `speed query`

This means stage1 speed energy no longer uses a short `route prefix` as a fake
trajectory proxy.

3. Speed query instead of speed concatenation

- each candidate speed is embedded as its own query
- these speed queries attend to a small scene memory built from:
  - `mode_out`
  - a projected full-route geometry token
- this is more natural than the previous "concat one speed embedding at the
  end" design because the model directly answers:
  - "what is the energy of this queried speed in the current scene?"

Operationally:

- training still uses a fixed `K=7` sampled speed set
- inference can later evaluate:
  - one queried speed, such as `current_speed`
  - or many queried speeds for a full speed-energy curve

4. Train/infer route convention

- training:
  - stage1 speed energy uses `GT route`
- inference:
  - route is predicted first by `forward_ego`
  - then stage1 speed energy is evaluated on `pred route`

So stage1 speed energy is currently a post-route evaluation branch, not part of
the inner diffusion guidance loop.

5. Speed token and trajectory/route ordering

- in the ego decoder, token order is:
  - `[speed | traj | route]`
- however, this does **not** change how trajectory and route points are sliced
  from `joint_points`
- the code still first extracts:
  - `traj_points = joint_points[:, :self.horizon, :]`
  - `route_points = joint_points[:, self.horizon:self.ego_joint_horizon, :]`
- only after that are the decoder tokens concatenated as:
  - `[speed | traj | route]`

So adding the speed token does not shift the trajectory/route split itself.

6. History LiDAR status

- history LiDAR is no longer current-frame only
- it is loaded as multiple frames and passed through a temporal position
  embedding before BEV encoding
- route does not have a separate direct LiDAR branch
- instead, route benefits indirectly through the shared decoder context, which
  is currently intentional

Current local special cases for validated intersection-like families:

- `junction + RIGHT`
  - future non-pedestrian cover is forced to `merge_meet`
  - this avoids overusing instantaneous heading during the turn and better matches
    "merge into downstream flow" behavior
- `junction + LEFT`
  - future same-direction cover stays `merge_meet`
  - future non-same-direction cover is treated as `junction_left_cross_meet`
  - for this subtype, risk is not computed with a long borrow corridor
  - instead, a short local conflict corridor is used:
    - corridor length `~= 1.5 x ego_length`
    - ego/background distances are clipped to that corridor before computing the
      arrival-time gap
- `pedestrian`
  - only pedestrian handling widens the route from a centerline into a corridor
  - this keeps the earlier vehicle families stable while making walk-across
    scenes much more conservative

5. `CutIn`

- primary focus:
  - future `chase`
  - sudden gap collapse
- expected behavior:
  - moderate expert-speed samples can stay safe
  - faster samples should rise sharply when TTC collapses

Special local-debug note for `HazardAtSideLane`:

- unlike the other families above, this scene can show route reshaping / rerouting
  around the event window
- for local video validation, do **not** append the extra `12` route points here
- use the raw planner route only (`20` points), otherwise the extension can mix
  in route instability that is not part of the intended label check

For all five families, also keep a small `general` sanity bucket:

- ordinary lane-follow / following / non-event windows
- this bucket is not a new family target
- it is only there to make sure v1 does not inflate risk everywhere

#### 13.11 Bad-Route Correction Policy

Current decision:

- first finish the base `stage1` energy labels on normal / validated routes
- then use `bad routes` only as a post-hoc correction source for energy labels
- do **not** let bad routes redefine the whole label system from scratch

Planned usage:

- only use `vehicle-collision` bad routes
- ignore non-vehicle failures such as:
  - offroad
  - lane invasion
  - traffic-light / stop-sign violations
- only use a short pre-crash window
- use the crash actor / violation attribution to correct the existing:
  - `E_chase`
  - `E_meet`

Adoption rule:

- after the base energy labels are generated, test the bad-route correction on top
- if the correction improves the local checks, keep it
- if it does not help, skip it for today and continue with the base label set

#### 13.12 Current Stage1 Implementation Status

As of the current `stage1` implementation, the codebase has already been wired
for a three-head speed-energy setup:

- `E_chase`
- `E_meet`
- `E_pedestrian`

The packed per-frame supervision written by
[`scripts/data_tools/precompute_semantic_labels.py`](/media/z/data/mzq/others/MoT-DP/scripts/data_tools/precompute_semantic_labels.py)
is now:

- `speed_sample_values`
- `speed_sample_valid_mask`
- `speed_sample_exp_index`
- `speed_risk_chase_values`
- `speed_risk_meet_values`
- `speed_risk_ped_values`

Important precompute/runtime notes:

- long full-dataset precompute now supports periodic atomic checkpointing via:
  - `--checkpoint_every_minutes`
- the script periodically overwrites `samples_packed.pkl` safely
- it also writes a sidecar progress file:
  - `samples_packed.pkl.progress.json`
- if the job crashes after several hours, rerunning the same command should skip
  samples that already have the new stage1 speed fields

Dataset/model/policy wiring that is already implemented:

- dataset LiDAR is no longer limited to the current frame only
- [`dataset/unified_carla_dataset.py`](/media/z/data/mzq/others/MoT-DP/dataset/unified_carla_dataset.py)
  now supports `lidar_history_frames`
- the returned LiDAR tensor is history-shaped:
  - `(H_hist, 2, 256, 256)`
- model-side LiDAR encoding is updated to consume the stacked history channels
- [`model/transformer_for_diffusion_multi_head.py`](/media/z/data/mzq/others/MoT-DP/model/transformer_for_diffusion_multi_head.py)
  exposes:
  - `forward_speed_energy`
  - `forward_speed_energy_eval`
- the model now contains three explicit speed-energy heads:
  - `speed_energy_chase_head`
  - `speed_energy_meet_head`
  - `speed_energy_pedestrian_head`

Current train/infer contract:

- training uses GT trajectory + GT route for the stage1 speed-energy loss
- inference uses predicted trajectory + predicted route when evaluating the
  speed-energy heads
- the current policy-side wiring lives in
  [`policy/annealed_energy_guidance_policy.py`](/media/z/data/mzq/others/MoT-DP/policy/annealed_energy_guidance_policy.py)

Current scope decision:

- keep the main backbone/shared context simple for now
- do **not** add extra chase/meet-specific BEV or LiDAR attention samplers in
  this stage
- do **not** add scene-family conditioning yet
- first get the three speed-energy heads trained and sanity-checked

In short:

- semantic/stage1 energy labels are now defined and implemented
- dataset/model/policy plumbing for the three-head speed-energy path is already
  in place
- next major step is running the full-dataset precompute and then training the
  stage1 speed-energy heads

#### 13.13 Stage1 Full-Dataset Training Progress

Stage1 three-head energy training has now started on the full train split with
full validation.

Current HPC launcher:

- [`scripts/hpc_new/train_route_b_lidar_stage1_fulltrain_val.sh`](/media/z/data/mzq/others/MoT-DP/scripts/hpc_new/train_route_b_lidar_stage1_fulltrain_val.sh)

Current validation numbers:

- `val_energy_loss: 0.0049`
- `val_alignment_loss: 0.0000`
- `val_energy_front_loss: 0.0015`
- `val_energy_chase_loss: 0.0015`
- `val_energy_left_loss: 0.0033`
- `val_energy_meet_loss: 0.0033`
- `val_energy_ped_loss: 0.0001`
- `val_energy_pedestrian_loss: 0.0001`
- `val_energy_right_loss: 0.0000`
- `val_energy_off_loss: 0.0000`
- `val_energy_route_loss: 0.0000`

How to read these numbers:

- `front == chase`
  - the logging still exposes the old Route-B slot names
  - in the current stage1 setup, `front` is the same supervised quantity as
    `chase`
- `left == meet`
  - same reason: `left` is the old log alias for the stage1 `meet` head
- `ped == pedestrian`
  - same quantity, just both old/new names are still visible in logging
- `alignment_loss = 0`
  - expected for the current stage1 path
  - the current stage1 speed-energy setup does not use the old alignment term
- `right/off/route = 0`
  - also expected
  - the current stage1 run only supervises:
    - `E_chase`
    - `E_meet`
    - `E_pedestrian`

Current interpretation:

- the three active heads are training stably
- `E_chase` validation loss is already quite low
- `E_meet` is the hardest of the three, which matches intuition because
  merge / cross / borrow timing is more diverse than pure chase
- `E_pedestrian` is extremely low, which suggests the current pedestrian labels
  are comparatively easy for the model to fit

Practical conclusion:

- these losses are good enough to say the model is fitting the stage1 labels
  rather than obviously collapsing or diverging
- however, low validation loss alone does **not** prove closed-loop benefit yet
- the real next checks are still:
  - expert-speed calibration
  - good-route `E_max` sanity checks
  - infer-time speed-energy parameter tuning on successful routes

#### 13.14 Expert-Speed Calibration Notes

We also ran a simple sanity check on the completed `train` split after the new
stage1 energy labels were written.

Statistic definition:

- use the expert-speed sample:
  - `speed_sample_values[speed_sample_exp_index]`
- evaluate the three expert-speed heads:
  - `E_chase`
  - `E_meet`
  - `E_pedestrian`
- define:
  - `E_total = max(E_chase, E_meet, E_pedestrian)`

Recommended calibration subset:

- prefer **successful routes** for infer-time calibration
- current success definition:
  - `status == Completed`
  - and no:
    - `collisions_vehicle`
    - `collisions_pedestrian`
    - `collisions_layout`

Why this matters:

- the full train split mixes successful and failed routes
- failed routes are useful as hard negatives
- but infer-time speed-energy thresholds should be calibrated primarily from the
  successful-route distribution

Meaning of the reported fields:

- `speed_mean`
  - mean expert speed for that event family
- `speed_p50`
  - median expert speed for that event family
- `speed_p90`
  - 90th percentile expert speed for that event family
- `Ec_mean`
  - mean expert-speed `E_chase`
- `Em_mean`
  - mean expert-speed `E_meet`
- `Ep_mean`
  - mean expert-speed `E_pedestrian`
- `Et_mean`
  - mean expert-speed `E_total`
- `Et_p90`
  - 90th percentile expert-speed `E_total`

Interpretation:

- `speed_p50` / `speed_p90` describe what expert speed typically looks like in
  that family
- `Ec_mean` / `Em_mean` / `Ep_mean` tell which head is usually active at expert speed
- `Et_mean` is the average "expert-speed risk floor"
- `Et_p90` is a conservative upper reference for infer-time threshold tuning

Current qualitative conclusion from the successful-route subset:

- wait-heavy families have small expert-speed medians:
  - `AccidentTwoWays`
  - `Accident`
  - `PedestrianCrossing`
  - `ParkingCrossingPedestrian`
  - `NonSignalizedJunctionLeftTurn`
  - `NonSignalizedJunctionRightTurn`
- merge / highway families keep much larger expert speeds:
  - `HighwayExit`
  - `MergerIntoSlowTraffic`
  - `HazardAtSideLane`
- in successful routes, expert-speed risk is usually moderate rather than near-1
- when failed routes are mixed back in, `Et_mean` and especially `Et_p90` rise
  noticeably, which is exactly why bad routes remain useful as hard negatives

Representative successful-route references:

- `AccidentTwoWays`
  - `speed_mean ~= 4.64`
  - `speed_p50 ~= 0.42`
  - `Et_mean ~= 0.156`
  - `Em_mean ~= 0.114`
- `Accident`
  - `speed_mean ~= 4.41`
  - `speed_p50 ~= 0.30`
  - `Et_mean ~= 0.076`
- `PedestrianCrossing`
  - `speed_mean ~= 2.08`
  - `speed_p50 ~= 0`
  - `Et_mean ~= 0.057`
  - `Ep_mean ~= 0.030`
- `ParkingCrossingPedestrian`
  - `speed_mean ~= 2.28`
  - `speed_p50 ~= 0`
  - `Et_mean ~= 0.112`
  - `Ep_mean ~= 0.039`
- `NonSignalizedJunctionLeftTurn`
  - `speed_mean ~= 2.99`
  - `speed_p50 ~= 0.01`
  - `Et_mean ~= 0.157`
  - `Em_mean ~= 0.137`
- `NonSignalizedJunctionRightTurn`
  - `speed_mean ~= 0.58`
  - `speed_p50 ~= 0`
  - `Et_mean ~= 0.231`
- `HighwayExit`
  - `speed_mean ~= 9.49`
  - `speed_p50 ~= 9.60`
  - `Et_mean ~= 0.152`
  - `Ec_mean ~= 0.147`
- `MergerIntoSlowTraffic`
  - `speed_mean ~= 7.20`
  - `speed_p50 ~= 7.30`
  - `Et_mean ~= 0.091`

Current recommendation for infer-time tuning:

- start from the successful-route distribution, not the mixed all-route distribution
- use family-aware priors if needed later, but in the first pass:
  - treat `Et_mean` as a soft typical reference
  - treat `Et_p90` as a conservative upper reference
- if the deployed inference policy looks over-conservative, compare its chosen
  speed-energy range against these successful-route expert references first

Additional note after checking expert-speed maxima on the successful-route subset:

- `Et_max` is usually **too sensitive** to use directly as an infer-time threshold
- even in successful routes, many families still contain isolated hard frames
  whose expert-speed `E_total` reaches or nearly reaches `1.0`
- this means:
  - `Et_max` is useful as a sanity/debug ceiling
  - but not as the main deployment calibration target
- for practical infer-time tuning, prefer:
  - `Et_mean` as a soft central reference
  - `Et_p90` or `Et_p95` as the conservative family-specific reference

Examples from the successful-route subset:

- `AccidentTwoWays`
  - `Et_max ~= 0.985`
  - `Et_p95 ~= 0.754`
- `HazardAtSideLaneTwoWays`
  - `Et_max ~= 0.984`
  - `Et_p95 ~= 0.379`
- `MergerIntoSlowTraffic`
  - `Et_max ~= 0.774`
  - `Et_p95 ~= 0.414`
- `StaticCutIn`
  - `Et_max ~= 0.797`
  - `Et_p95 ~= 0.446`
- `HighwayExit`
  - `Et_max = 1.0`
  - `Et_p95 ~= 0.328`
- `PedestrianCrossing`
  - `Et_max = 1.0`
  - `Et_p95 ~= 0.510`
- `NonSignalizedJunctionLeftTurn`
  - `Et_max = 1.0`
  - `Et_p95 ~= 0.863`

Interpretation:

- successful routes can still contain brief frames where the expert is right at
  the boundary of a high-risk maneuver
- therefore:
  - seeing `Et_max = 1.0` in successful routes is **not** by itself a labeling bug
  - it mainly means the label system is capable of saturating on rare but valid
    expert frames
- infer-time calibration should therefore avoid using raw maxima as thresholds

So the current ego path does not just sample more points. It also runs multiple full-BEV projections per batch:

- old ego path: roughly one `value_proj(bev_upsample)`
- new ego path: one each for
  - `bev_point_attn`
  - `traj_detail_attn`
  - `route_detail_attn`
  - `route_point_attn`

This is a major source of the extra epoch time.

### Current implementation detail: the timestep gate does not save compute

The low-timestep gate currently only gates the fused output, not the actual computation.

That means:

- `traj_local_detail`
- `traj_route_detail`

are still computed even when `t > 400`.

So although the design intent is "detail only for low timestep", the current implementation still pays the cost for these branches on almost every batch.

This is likely the single easiest optimization opportunity.

### Validation epochs can become much slower

Training still uses one noisy joint forward per batch.

However, validation uses `conditional_sample`, which runs multi-step denoising. Because the ego path now denoises a joint traj+route state with the heavier fine-BEV logic, validation epochs can become noticeably more expensive than before.

## Most Likely First Optimizations

If epoch time becomes a problem, the first two optimizations to try are:

1. Actually skip detail computation when `t > 400`

- do not compute `traj_local_detail`
- do not compute `traj_route_detail`
- return only the center-sampled path for those samples

2. Share projected BEV values across detail branches

- compute one projected `value` tensor from `bev_feature_upsample`
- reuse it across center/detail sampling branches instead of running 3-4 separate `value_proj` convolutions

These two changes should recover much of the runtime without changing the overall modeling direction.

## Current Status

The following are in place:

- local Route B formal route stats file exists
- HPC Route B formal route stats file exists
- HPC Route B config points to the correct route stats path
- local commit created for the joint route diffusion change:
  - `5fe56f5` `Add joint route diffusion to Route B ego path`

## Suggested Training Command On HPC

```bash
cd /workspace1/z_project/code/motdp_z

CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace1/z_project/env_conda/z_dpauto/bin/torchrun --standalone --nproc_per_node=4 \
  training/train_carla_bev.py \
  --config_path config/pdm_hpc_route_b.yaml
```
