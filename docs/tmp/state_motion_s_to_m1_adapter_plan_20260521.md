# S -> M1 State-Motion Adapter Plan

Date: 2026-05-21

Worktree:

```text
/media/z/data/mzq/others/MoT-DP-worktrees/state_motion_decoupled_alignment_v1
```

## Summary

The code should be implemented with the full staged path available behind config switches, but tonight's experiment only runs through S -> M1 and does not train A yet:

```text
S: initialize from nostate G, train detached semantic state
M1: initialize from S checkpoint, freeze G/S, train zero-init state-to-motion adapter only
```

Goal:

- Preserve the route/speed convergence ability of the `nostate` G checkpoint.
- Let semantic state act only as a lightweight interaction residual.
- Avoid the old joint-training conflict where route needs longer training but state overfits if training continues.

This plan records implementation intent only. Implementation should add the later A/M2 paths now as disabled-by-default config options so we do not need another structural rewrite after M1.

## Motivation From 0512 vs Nostate Comparison

The comparison in `/tmp/white_noise_vs_0512.txt` and the new-hpc results show:

- `nostate` is a strong motion prior, especially for route geometry, route following, two-way obstacle handling, and general speed/route convergence.
- The state model is useful for some interaction-heavy cases, especially junction timing, merge gap, chase/front-follow, and vehicle-collision-sensitive cases.
- The previous joint route+state training has a schedule conflict: if training continues, route/speed converges better but state can overfit; if training stops early to preserve state, route/speed is undertrained.

Therefore M1 should not retrain G from scratch. It should add a small zero-init residual path on top of an already-good nostate G.

## Key Changes

### Training Stages

- `state_motion_training_mode: semantic_detached`
  - Existing S stage.
  - Semantic losses train S/state heads.
  - Semantic losses do not update the motion backbone.
  - Initialize from the nostate G checkpoint with partial/non-strict init.

- `state_motion_training_mode: adapter_m1`
  - New M1 stage.
  - Freeze G backbone, route head, speed/traj heads, and S heads.
  - Train only the state-to-motion adapter parameters.
  - Route is not modified by the adapter.

- `state_motion_training_mode: alignment_critic`
  - Future A stage.
  - Freeze G and S.
  - Train only A from positive / negative state-motion pairs.
  - Not run tonight.

- `state_motion_training_mode: adapter_m2`
  - Future M2 stage.
  - Freeze G, S, and preferably A.
  - Train adapter with motion loss plus A compatibility energy.
  - Not run tonight.


### Full Code Path To Implement Now

Even though tonight only runs S -> M1, implement the full staged skeleton now:

```text
G  : nostate motion generator / strong motion prior
S  : detached semantic state predictor
M1 : zero-init adapter trained by GT motion loss only
A  : decomposed compatibility-probability critic
M2 : adapter trained with GT motion loss + A compatibility loss
```

Config controls:

```yaml
route_b:
  state_motion_training_mode: semantic_detached   # semantic_detached | adapter_m1 | alignment_critic | adapter_m2
  use_state_motion_adapter: false
  use_alignment_critic: false
  use_alignment_critic_for_adapter: false
```

Default behavior should be safe:

- `semantic_detached`: no adapter, no A.
- `adapter_m1`: adapter on, A off.
- `alignment_critic`: A on, adapter off.
- `adapter_m2`: adapter on, A on as an auxiliary compatibility loss.

### A Critic Skeleton

A is not trained tonight, but code should expose it for the next experiment.

A outputs logits and logs probabilities:

```text
alignment_total_prob
alignment_phase_speed_prob
alignment_edge_speed_prob
alignment_chase_speed_prob
alignment_route_window_prob
alignment_energy = -log(alignment_total_prob + eps)
```

A training uses decomposed soft BCE:

```text
positive: GT state + expert motion
negative: GT state + corrupted speed/traj or corrupted state + expert motion
risky_passable: medium target, e.g. 0.55-0.7
```

A should not be required by M1. M1 config keeps:

```yaml
use_alignment_critic: false
use_alignment_critic_for_adapter: false
```

### M2 Adapter + A Energy Skeleton

M2 is M1 plus A guidance:

```text
loss = motion_loss
     + adapter_norm_loss
     + adapter_gate_sparsity_loss
     + alignment_energy_weight * A_energy_loss
```

Recommended future config:

```yaml
route_b:
  state_motion_training_mode: adapter_m2
  use_state_motion_adapter: true
  use_alignment_critic: true
  use_alignment_critic_for_adapter: true
  alignment_energy_loss_weight: 0.02
  freeze_alignment_critic_for_adapter: true
```

M2 should load from M1 plus A checkpoint if available. If only M1 checkpoint is provided, A is randomly initialized and `use_alignment_critic_for_adapter` should remain false.

### Zero-Init State-to-Motion Adapter

Insert the adapter in `forward_ego` after decoder tokens are produced and before motion heads decode:

```text
traj_out, route_out, speed_out = decoder(...)
state = frozen S(...)
adapter_delta_traj, adapter_delta_speed = Adapter(state, ego_speed)
traj_out'  = traj_out  + gate * adapter_delta_traj
speed_out' = speed_out + gate * adapter_delta_speed
route_out' = route_out
```

Zero-init requirements:

- `adapter_global_gate = 0` at initialization.
- Adapter final projection initialized to zero.
- With an untrained adapter, predictions should be equivalent to the nostate/S checkpoint.

### Adapter Inputs

Use compact interaction state only:

```text
window probs
decision/control phase probs
go_opportunity / yld_pressure
current/future edge valid + mode probs
edge speed margins + valid flags
chase_has_lead + chase_speed_margin
current ego speed
```

Do not directly input:

```text
dir
raw conflict_area route mask
raw temporary occupancy bins
raw timing scalars
legacy family boundary
```

### M1 Loss

M1 trains adapter with:

```text
motion_loss = traj L1 + speed CE
adapter_norm_loss = ||adapter_delta||^2
adapter_gate_sparsity_loss = mean(abs(gate))
```

Route loss should be logged for debug only and should not drive M1, because route is not modified.

Recommended config defaults:

```yaml
route_b:
  state_motion_training_mode: adapter_m1
  use_state_motion_adapter: true
  state_motion_adapter_zero_init: true
  state_motion_adapter_state_source: pred_detached
  state_motion_adapter_modify_route: false
  state_motion_adapter_loss_weight: 1.0
  state_motion_adapter_norm_weight: 0.01
  state_motion_adapter_gate_sparsity_weight: 0.001
  freeze_motion_backbone_for_adapter: true
  freeze_semantic_state_for_adapter: true
  use_alignment_critic: false
```

## Training Flow

### Stage S

```bash
INIT_CHECKPOINT=<nostate_ckpt> CONFIG=config/pdm_hpc_route_b_state_motion_semantic_detached_0521.yaml bash scripts/codex_bash/train_state_motion.sh
```

Output:

```text
S_ckpt
```

### Stage M1

```bash
INIT_CHECKPOINT=<S_ckpt> CONFIG=config/pdm_hpc_route_b_state_motion_adapter_m1_0521.yaml bash scripts/codex_bash/train_state_motion.sh
```

Output:

```text
M1_ckpt
```


### Future Stage A

```bash
INIT_CHECKPOINT=<S_ckpt> CONFIG=config/pdm_hpc_route_b_state_motion_alignment_critic_0521.yaml bash scripts/codex_bash/train_state_motion.sh
```

Output:

```text
A_ckpt
```

### Future Stage M2

```bash
INIT_CHECKPOINT=<M1_or_combined_S_A_ckpt> CONFIG=config/pdm_hpc_route_b_state_motion_adapter_m2_0521.yaml bash scripts/codex_bash/train_state_motion.sh
```

Output:

```text
M2_ckpt
```

## Expected Behavior

- At initialization, M1 should match the S/nostate behavior because the adapter is zero-init.
- After training, route should remain close to nostate.
- Speed/traj should improve mainly in interaction-active cases:
  - junction timing
  - merge gap
  - chase/front-follow
  - vehicle-collision-sensitive cases
- If adapter learns nothing useful, norm/gate regularization should make it safely collapse back toward nostate.

## Test Plan

### Static

- `python -m py_compile` for model, policy, training.
- YAML parse for S and M1 configs.
- Confirm `INIT_CHECKPOINT` partial init can load nostate/S checkpoint while leaving adapter keys randomly zero-initialized.

### Equivalence Smoke

- With adapter enabled but untrained, compare outputs against adapter disabled on the same batch/checkpoint.
- `route_pred` should be unchanged or differ only by numerical noise.
- `speed_pred` and `traj_pred` should initially be equivalent before adapter training.

### Train Smoke

- S stage:
  - semantic losses finite.
  - motion backbone gradients are none/zero.

- M1 stage:
  - only adapter parameters receive gradients.
  - G/S/route parameters are frozen.
  - `adapter_norm_loss` and `adapter_gate_sparsity_loss` are finite.

### Behavior Check

Compare M1 against nostate and the prior state model:

- Preserve nostate strengths:
  - route geometry
  - two-way obstacle handling
  - outside-route-lane stability
  - layout collision stability

- Recover state-model strengths:
  - junction timing
  - merge gap
  - chase/front-follow speed
  - vehicle collision reduction

## Assumptions

- Tonight's run only executes `S -> M1`; A/M2 code paths may be implemented now but must remain disabled unless their config explicitly enables them.
- M1 uses frozen S predicted state, not GT state, to match close-loop inference.
- Adapter V1 modifies only speed/traj tokens, not route tokens.
- Existing nostate checkpoint is the G baseline and should not be retrained from scratch.
- Label dataset uses the existing alignment split:

```text
/workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_scene_split_95_5_alignment
```


## Implementation Notes Added 2026-05-21

Implemented V1 code path should expose the whole staged skeleton through config:

```text
semantic_detached  -> Stage S, train detached state heads from nostate init
adapter_m1         -> Stage M1, freeze G/S and train zero-init state-to-motion adapter
alignment_critic   -> Stage A, train decomposed compatibility critic
adapter_m2         -> Stage M2, adapter + optional frozen A energy
```

Tonight's automatic run uses:

```bash
NOSTATE_CKPT=/path/to/nostate/dit_policy_best.pt \
CODE_DIR=/workspace1/z_project/code/motdp_z_state_motion_decoupled_alignment_v1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 \
bash scripts/codex_bash/train_state_motion_s_to_m1.sh
```

Useful overrides:

```bash
SKIP_S=1 S_CKPT=/path/to/S/dit_policy_best.pt bash scripts/codex_bash/train_state_motion_s_to_m1.sh
SKIP_M1=1 NOSTATE_CKPT=/path/to/nostate.pt bash scripts/codex_bash/train_state_motion_s_to_m1.sh
```

One implementation detail differs from the first sketch: the adapter uses a zero-initialized final projection to preserve exact initial equivalence, while the global gate defaults to `1.0`. Setting both final projection and gate to zero would make the residual path gradient-dead.
