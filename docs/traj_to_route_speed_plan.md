# Traj To Route Speed Migration Plan

Date: 2026-05-17

Status: planning only. Do not implement in this note.

## 1. Goal

Current MoT-DP still treats direct trajectory prediction as the main ego output.

The next migration target is to move toward predicting route and speed first, and to
consider acceleration as an explicit auxiliary target because half-closed-loop quality
is highly sensitive to smoothness and temporal consistency.

This note is only a planning record. It is not an implementation task list for the
current session.

## 2. Main Direction

The intended long-term direction is:

1. build a route label first
2. train the model to predict route
3. train the model to predict speed
4. add acceleration-oriented supervision because smoothness matters in half-closed-loop
5. study how route plus speed should be converted into trajectory and then into control
6. keep some previous useful tricks, including modulation-like conditioning, extra drop,
   and point-wise normalization assumptions

The migration should be staged. We should not remove the current trajectory path until
the new route plus speed path is validated.

## 3. Key Observation From Current Codebase

The current codebase already contains part of the needed structure.

- `policy/annealed_energy_guidance_policy.py` already has:
  - `route_loss`
  - `speed_loss`
  - `speed_profile_loss`
  - speed target construction
  - speed profile target construction
  - inference outputs including `route_pred`, `target_speed`, and `target_speed_profile`
- `model/transformer_for_diffusion_multi_head.py` already has:
  - speed token ordering `[speed | traj | route]`
  - route head
  - speed head
  - speed profile head
  - current ego-decoder mask semantics for traj and route
- `dataset/unified_carla_dataset.py` already has:
  - route-related sample assembly
  - next-speed-related fields
  - several speed-risk and speed-sample fields

So this should be treated as an extension of the current path, not as a separate planner
rebuilt from scratch.

## 4. SparseDriveV2 Findings To Borrow

Local reference repo:

- `/media/z/data/mzq/others/SparseDriveV2/`

Confirmed reference entrypoints:

- training entry: `scripts/training/sparsedrive_navsimv2.sh`
- cache entry: `scripts/cache/run_dataset_caching_navtrain.sh`

Confirmed target-builder behavior:

- `navsim/agents/sparsedrive/sparsedrive_features.py`
- `SparseDriveTargetBuilder.compute_targets(...)` emits:
  - `trajectory`
  - `path`
  - `path_mask`
  - `velocity`

Important details:

- SparseDrive builds future path from future ego poses.
- The path is sampled by arc length.
- Velocity is derived from future trajectory displacement divided by fixed dt.

Confirmed decoder behavior:

- `navsim/agents/sparsedrive/custom_decoder.py`
- SparseDrive does not directly replace trajectory with route plus speed as the final
  driving output.
- Instead, it first scores path anchors and velocity anchors, filters them, and then
  selects a final trajectory candidate.

This is important for our migration design:

- SparseDrive is a strong reference for label construction and intermediate
  representation design.
- SparseDrive is not evidence that we must immediately drop final trajectory output.
- A safer first migration is to let route and speed become first-class supervised
  branches while keeping trajectory as the stable control-facing fallback.

## 5. Planned Label Contract

### 5.1 Route Label

Route label should be built from future ego path in ego frame.

Recommended shape:

1. generate a denser path from future ego poses using arc-length sampling
2. keep an explicit validity mask
3. expose a public route slice from the dense path, for example `route20`

Recommended reason:

- this matches SparseDrive's path-building logic
- it keeps control decoding easier than predicting only a very short route
- it preserves room for later route-density ablations

### 5.2 Speed Label

Speed label should have two layers:

1. next-step scalar speed target
2. short-horizon speed profile target

Priority:

- prefer exact next-step speed if available from raw measurements
- otherwise derive it from the first trajectory displacement divided by dt

### 5.3 Acceleration Label

Acceleration label should be derived from the speed profile by finite difference on the
same dt used by the trajectory and speed-profile path.

Recommended first use:

- auxiliary supervision for smoothness and consistency
- not the only primary control output in the first pass

Reason:

- half-closed-loop quality is sensitive to jerk, oscillation, and local inconsistency
- acceleration supervision can regularize route plus speed prediction without forcing
  an immediate controller redesign

### 5.4 Normalization Rule

Normalization should follow the point-wise prediction semantics.

Working assumption to preserve in the first design:

- one prediction point uses one normalization slot
- do not collapse all points into a single shared normalization if the model output is
  still point-wise

This should be recorded as a first-class design rule, with any non-point-wise norm only
tested later as an ablation.

## 6. Planned Model And Training Stages

### Phase 0: Freeze The Reference Design

Before implementation, record exactly what we want to borrow from SparseDriveV2 and what
we will keep from MoT-DP.

The most important frozen assumptions are:

- route label comes from future ego path, not from ad hoc route heuristics
- speed target has both scalar and profile forms
- acceleration is derived from speed-profile time differences
- current trajectory path stays alive during the migration

### Phase 1: Dataset And Label Work

Extend the current data pipeline so that one sample can consistently provide:

- trajectory
- route or dense path
- route mask
- next-step speed
- speed profile
- acceleration profile

This phase must also add validation tooling:

- route geometry visualization
- route mask checks
- speed distribution checks
- acceleration range checks
- finite-difference consistency checks between speed profile and acceleration

### Phase 2: Multi-Task Model Extension

Keep the current trajectory head as the baseline path.

Add or refine:

- route supervision
- scalar speed supervision
- speed-profile supervision
- acceleration auxiliary supervision

Recommended rule:

- do not hard-switch to route plus speed only at this stage
- first make the new branches stable beside the existing trajectory branch

### Phase 3: Route Plus Speed To Control Adapter

We need a dedicated design pass for how predicted route plus speed should become a
half-closed-loop control signal.

Recommended first adapter:

1. convert predicted route plus speed into a denser checkpoint sequence
2. reuse current steering and throttle logic instead of inventing a new controller
3. compare against the current trajectory-based control path on the same subset

This is safer than introducing a new direct controller before the planner outputs are
stable.

### Phase 4: Acceleration Use In Control

After route plus speed is stable, decide whether acceleration should:

- only regularize the predicted speed profile
- smooth the control adapter
- or become an explicit control-facing quantity

This decision should be made only after half-closed-loop behavior is inspected.

## 7. Planned Reference Surfaces For Control

Current repo control references:

- `team_code/team_code_transfuser/model.py`
  - `control_pid(...)`
  - `control_pid_direct(...)`
- `team_code/simlingo/nav_planner.py`
  - `LateralPIDController`
  - `get_throttle(...)`

These files should be treated as the first control bridge for route plus speed.

The immediate question is not "how to write a new controller from scratch".
The immediate question is:

- how to convert route plus speed into the checkpoint and target-speed structure already
  expected by the current controller path

## 8. Planned Trick And Ablation Bucket

The following items should stay in the plan as explicit ablations rather than being mixed
into the first implementation blindly.

### 8.1 Modulation Or Demodulation Style Conditioning

We should inspect and reuse the current modulation-style paths first.

Relevant current references include:

- AdaLN-style conditioning in `model/transformer_for_diffusion_multi_head.py`
- AdaLN-style conditioning in `model/navsim_simple_diffusion.py`
- FiLM-like references in `model/action_heads.py`

Plan rule:

- first reuse current conditioning patterns
- only then test a stronger modulation or demodulation variant if it is clearly isolated

### 8.2 Drop Additions

Extra drop should be added in a branch-specific and measurable way.

Plan rule:

- do not change global dropout everywhere at once
- test route branch, speed branch, and acceleration branch drop separately when possible

### 8.3 Point-Wise Norm

Because the model predicts one point at a time, point-wise normalization should remain
the default assumption for trajectory or route outputs unless an ablation shows a better
alternative.

## 9. Decision Gates Before Any Hard Switch

We should not change the default planner behavior until all of the following are true.

1. route labels are visually correct and numerically stable
2. speed and acceleration labels are self-consistent under finite differences
3. small-scale training shows route and speed losses decreasing without destabilizing the
   current trajectory regression path
4. route plus speed control is at least as smooth and as consistent as the current traj
   baseline on the agreed half-closed-loop subset
5. acceleration supervision improves smoothness rather than causing oscillation or drift

Only after these gates are passed should we consider reducing reliance on the direct
trajectory prediction path.

## 10. Scope Boundaries

Included in this plan:

- SparseDrive-style route-label study
- speed and acceleration label design
- current decoder extension
- route plus speed control adapter design
- modulation, drop, and normalization ablations

Explicitly not included in this plan:

- immediate removal of the current trajectory head
- immediate full reproduction of the entire SparseDrive stack
- immediate implementation in this note

## 11. Recommended First Implementation Order Later

When implementation starts, the recommended order is:

1. freeze label schema and masks
2. add offline visualization and stats
3. extend dataset outputs
4. wire losses into the current decoder path
5. add acceleration auxiliary supervision
6. build route plus speed to checkpoint adapter
7. compare half-closed-loop behavior against the current trajectory baseline

This order minimizes the chance of breaking the current system while still moving toward
the route plus speed direction.

## 12. Minimum Trainable Milestone

This plan is intended to go all the way to the first point where we can start a real
training run.

The intended first milestone is not:

- route plus speed already replacing the existing closed-loop trajectory path
- route plus speed already being the default half-closed-loop controller input
- the full controller redesign being finished

The intended first milestone is:

- one training sample can emit all required supervision targets
- the current model path can consume those targets without schema mismatch
- the training loop can log the new losses and run a smoke training job

So the direct answer is:

- yes, this plan should go until the project is directly trainable
- no, it does not require the control handoff to be finished before the first training

For this note, a setup counts as directly trainable when all of the following are true:

1. dataset outputs are stable and batched correctly
2. route or path target, mask, next-step speed, speed profile, and acceleration target
  all exist with fixed shapes and units
3. model forward can compute route-related and speed-related losses without branch or
  dtype mismatches
4. training loop can log the new losses and checkpoint normally
5. a small smoke run can complete at least one validation cycle without schema or shape
  errors

## 13. First Training Slice

The first implementation slice should be the smallest slice that can produce a valid
training run.

Recommended first training slice:

### 13.1 Data Side Minimum

The sample contract should provide at least:

- `trajectory`
- `route` or dense `path`
- `route_mask` or `path_mask`
- `next_speed_target_mps`
- `speed_profile_target_mps`
- `acceleration_profile_target_mps2`

The first trainable version may derive some of these online from existing targets, as
long as the derivation is deterministic and the unit convention is fixed.

Recommended minimum rule:

- if exact next-step speed already exists, use it
- otherwise derive it from the first trajectory step and current dt
- derive acceleration from speed profile finite differences instead of inventing a
  separate incompatible label source in the first pass

### 13.2 Model Side Minimum

The first trainable version does not need a full new planner.

It only needs:

- the existing trajectory path to stay alive
- the existing route branch to remain supervised
- the existing scalar speed head to remain supervised
- the existing speed profile path to remain supervised if enabled
- one new acceleration auxiliary branch or acceleration loss hook

Recommended minimum rule:

- acceleration should first be auxiliary only
- trajectory regression must remain the baseline stabilizer in the first training slice

### 13.3 Training Side Minimum

The first trainable version should update:

- sample assembly
- model loss wiring
- training logging
- config flags

But it does not need to finish:

- route plus speed to control adapter
- half-closed-loop controller replacement
- removal of the trajectory head

### 13.4 Recommended File Touch Set

The first trainable slice should stay close to these files:

- `dataset/unified_carla_dataset.py`
- `policy/annealed_energy_guidance_policy.py`
- `model/transformer_for_diffusion_multi_head.py`
- `training/train_carla_bev.py`
- the current Route B config family under `config/`

This is the preferred minimum file surface because these files already own the current
route, speed, loss, and training wiring.

## 14. What Can Wait Until After First Training

The following items are important, but they are not blockers for the first trainable
milestone.

1. route plus speed to checkpoint adapter
2. route plus speed to direct control adapter
3. half-closed-loop comparison runs
4. stronger modulation or demodulation redesign
5. broader dropout experiments
6. replacing current normalization globally
7. removing the direct trajectory output as the main fallback

These items should remain after the first training milestone because they depend on the
quality of the learned route and speed branches.

## 15. Exit Criteria For "Ready To Train"

This plan should be treated as complete enough to start training when the following are
implemented later.

1. label schema is frozen and documented
2. the dataset returns the new targets with consistent units and masks
3. the model computes route, speed, and acceleration-related loss terms without shape
  errors
4. the training loop can report those losses and save checkpoints
5. a small local or HPC smoke training run finishes without crashing

Only after that point do we move from planning into the first real train experiment.

## 16. 2026-05-18 Shape Contract Update

The current trainable contract is now fixed to:

- trajectory horizon: 8 waypoint tokens
- speed profile horizon: 8 speed tokens
- route/path horizon: 50 route tokens
- compact ego status: 8 dims

For NAVSIM cached-BEV diffusion, the 8-dim status is:

```text
[ego_velocity(2), ego_acceleration(2), driving_command(4)]
```

The old 14-dim NAVSIM status was this same 8-dim payload plus 6 zero padding dims.
New precompute writes 8 dims directly, while dataset and agents still slice/pad old caches
or checkpoints when needed.

For Route-B migration smoke, `CARLAImageDataset` now fits old samples to the new shape:

- `agent_pos` is truncated or linearly tail-extended to 8 points
- `route` and `path` are truncated or linearly tail-extended to 50 points
- `ego_status` is sliced or padded to `dataset.ego_status_dim`

This keeps old PDMLite smoke samples usable, but full training should recompute
normalization stats with `traj_horizon=8` and `route_points=50` instead of relying on old
6/20 stats. The policy has a compatibility fit for old stats only to keep smoke tests from
crashing.

Verified on newhpc:

```text
CARLAImageDataset smoke sample:
route=(50,2), path=(50,2), agent_pos=(8,2), ego_status=(4,8)
speed_profile=(8,), acceleration_profile=(7,)

NavSimCachedDataset old officialacc cache:
raw ego_status.npy=(103288,4,14), returned ego_status=(4,8)

stats smoke:
valid traj=64, valid route=64, valid speed=64
abs_stats=(8,2), route_abs_stats=(50,2), speed_profile=(8,)

train smoke:
/workspace2/z_project/motdp_logs/traj_to_route_speed_smoke_8_50
1 epoch completed, validation completed, val_route_L2 and speed/profile/acc losses logged
```

## 17. 2026-05-18 NAVSIM Route50 Norm Stats

Clarification after rechecking the migration path: NAVSIM route construction is
indeed borrowed from SparseDrive-style target construction. The label is not an
HD-map route polyline. It is reconstructed from future ego poses in the current
ego frame, then sampled by arc length every 1m to form `path50/route50`.

Short paths follow SparseDrive's fixed-length plus mask contract: `route/path`
always has 50 points, the tail is held at the final reachable pose by
interpolation, and `route_mask/path_mask` marks which points are supervised.

The NAVSIM-specific stats script is:

```text
scripts/data_tools/compute_navsim_traj_route_speed_norm_stats.py
```

It uses:

- official NAVSIM train cache tokens from
  `/workspace2/z_project/motdp_bev_cache_train_official4cam_officialacc_npy/cache_index.npz`
- raw NAVSIM trainval logs from `/workspace2/data/navsim/navsim_logs/trainval`
- contiguous sample chains checked by `sample_prev/sample_next` and timestamp gap
- trajectory horizon 8
- route/path horizon 50
- speed horizon 8

SparseDrive-mask smoke run:

```text
/workspace2/z_project/motdp_navsim_norm_stats/navtrain_official_h8_r50_20260518_sparsemask_smoke
allowed official cache tokens: 103288
tokens_matched: 512
valid_traj: 512
valid_speed: 512
valid_route_any: 512
valid_route_full: 445
```

SparseDrive-mask full run:

```text
/workspace2/z_project/motdp_navsim_norm_stats/navtrain_official_h8_r50_20260518_sparsemask
allowed official cache tokens: 103288
logs: 1310
frames: 723019
segments: 1985
tokens_seen_in_logs: 103288
tokens_matched: 102608
tokens_skipped_future_short: 680
tokens_skipped_no_route: 0
valid_traj: 102608
valid_speed: 102608
valid_route_any: 102153
valid_route_full: 82800
failed: 0
```

Generated files:

```text
navtrain_official_h8_r50_sparsemask_navsim_traj_route_speed_norm_stats.npz
navtrain_official_h8_r50_sparsemask_abs_stats.npz
navtrain_official_h8_r50_sparsemask_route_abs_stats.npz
navtrain_official_h8_r50_sparsemask_speed_profile_stats.npz
navtrain_official_h8_r50_sparsemask_metadata.json
```

Important: an accidental PDMLite stats recomputation was discarded. These stats
are NAVSIM-only and should be the reference for the NAVSIM route/speed/traj
contract.
