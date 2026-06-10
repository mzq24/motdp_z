# Paper Ablation Ladder Tracking

Date: 2026-06-08

Worktree:

```text
/media/z/data/mzq/others/MoT-DP-worktrees/semantic_state_strict_ablation_v1
```

40G HPC mirror:

```text
/data/z_project/code/motdp_z_semantic_state_strict_ablation_v1
```

## Purpose

This document tracks the clean paper ablation ladder. The goal is not to stop at
nostate baselines, but to rebuild the final strong system step by step and
attribute which components matter.

The first ladder levels are motion-only baselines:

```text
N0 = legacy_routeb_nostate
N1 = paper_simple_oldddim_nostate
N2 = paper_unified_no_detail_nostate
N3 = paper_unified_route_intent_nostate
```

Future levels should add one major capability at a time, for example detail
attention, route intent / TG token, semantic state heads, semantic-to-motion
conditioning, and training tricks.

## Naming Rules

- `N*` means a nostate / motion-only ladder level.
- Later semantic levels can use a new prefix if useful, but should still refer
  back to the nearest `N*` base.
- Each entry should record structure, config, ckpt, open-loop metrics, close-loop
  result path, and current interpretation.
- Do not mix checkpoint identity and experiment result identity. Keep both paths.

## N0: legacy_routeb_nostate

Short name:

```text
legacy_routeb_nostate
legacy_routeb_nostate_0519_e60
```

Role:

```text
Old Route-B nostate reference. This is the strongest legacy motion-only style
baseline and is used as the practical reference for whether clean paper code
has recovered the known route/speed capability.
```

Structure:

- Uses the legacy Route-B large model / policy path.
- No semantic supervision is used for this nostate run.
- The checkpoint may still contain many unused semantic/stage1 modules because
  it comes from the larger research code path.
- Uses the legacy decoder stack and old sampling behavior.
- Important: this is "nostate" in the sense that semantic/stage1 supervision
  and semantic-to-motion conditioning are disabled. It is not the same as the
  later minimal paper motion-only core.
- The recovered e60 checkpoint has `use_route_intent_token=true`, so it includes
  the route-intent / TG token path. This may be one of the important differences
  from N1/N2.

Original important version:

```text
run name:
  route_b_simple_diffusion_nostate_0519

config recovered from checkpoint:
  config/tmp/pdm_hpc_route_b_lidar_bev_simple_diffusion_nostate.yaml

checkpoint:
  checkpoints/route_b_simple_diffusion_nostate_0519/dit_policy_epoch60.pt

local artifact:
  /media/z/data/mzq/others/MoT-DP-worktrees/semantic_state_strict_ablation_v1/
    checkpoints/route_b_simple_diffusion_nostate_0519/dit_policy_epoch60.pt
```

Recovered e60 config highlights:

```text
policy_type: anchor_free
dataset_path:
  /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_scene_split_95_5_prevstate
feature_suffix: ensemble
use_lidar_bev_detail: false

train_stage1: false
use_stage1_state: false
use_semantic_state_transition: false
use_traj_branch_condition: false
use_cover_relation_graph_decoder: false
use_route_prev_coarse_memory: false
use_route_intent_token: true

num_inference_steps: 10
train_max_timesteps: 1000
reg_loss_weight: 3.0
route_loss_weight: 5.0
route_final_loss_weight: 2.0
speed_loss_weight: 0.2
optimizer.lr: 5e-5
lr_final: 1e-7
warmup_epochs: 3
num_epochs: 60
batch_size: 128
use_window_weighted_sampler: false
use_semantic_shift_sampler: false

gps_noise:
  enabled: true
  sigma: 0.05
  probability: 1
  apply_to_route: false
```

Open-loop e60 metrics from checkpoint:

```text
val_L2_avg:      not stored in compact key, legacy val_loss=2.7574
val_reg_loss:    0.4558
val_route_L2:    0.0841
val_route_final: 0.1714
val_route_loss:  0.2637
speed_MAE:
  speed_head:    0.8516
  traj_1s:       0.5406
```

Original close-loop result:

```text
/data/z_project/code/Bench2Drive/eval/results/route_b_b2d_0519_white_noise_none_e60
```

Original close-loop score via `Bench2Drive/cal_score.py`:

```text
routes:        220 / 220
valid:         220
success:       173
Driving Score: 91.270
Success Rate:  78.64%
```

Later rerun / confusing name:

```text
launcher:
  /data/z_project/code/Bench2Drive/bash_commands/0519_white_noise_none_e60.sh

later default checkpoint in that launcher:
  checkpoints/route_b_simple_diffusion_nostate_0519/dit_policy_epoch55.pt

later result:
  /data/z_project/code/Bench2Drive/eval/results/route_b_b2d_0531_nostate_0519_epoch55_skip23695_24071
```

Notes:

- Do not infer the original e60 checkpoint from the later `0519_white_noise_none_e60.sh`
  file alone. That launcher was later edited/defaulted to `epoch55`.
- The original e60 artifact is preserved locally and should be treated as an
  important strong no-state reference.
- This is not paper-clean, but it is a key sanity reference and likely includes
  useful tricks: legacy decoder stack, route-intent token, GPS noise, and no
  semantic loss interference.

## N1: paper_simple_oldddim_nostate

Short name:

```text
paper_simple_oldddim_nostate
```

Role:

```text
Clean paper motion-only model with the simple decoder core, but using the old
pred_x0 DDIM sampling logic. This tests whether a minimal clean motion-only
model can match the old nostate behavior.
```

Structure:

- Uses `PaperMotionPolicy`.
- Uses `PaperMotionDiffusionCore`.
- Keeps only BEV feature projection, ego/status/history conditioning, traj
  diffusion, route diffusion, and speed head.
- Does not include semantic/stage1/graph/tempocc/chase/alignment/route-intent.
- Uses old pred_x0 DDIM sampling after the sampling alignment fix.

Known config / checkpoint:

```text
config:
  config/paper_motion_only_core_oldddim_0531.yaml

training script:
  scripts/codex_bash/train_paper_motion_only_oldddim_0531.sh

checkpoint:
  checkpoints/paper_motion_only_core_oldddim_0531/dit_policy_best.pt
```

Open-loop reference:

```text
best epoch:
  e45

metrics:
  val_L2_avg:      1.1054
  val_route_L2:    0.1154
  val_route_final: 0.2319
  speed_MAE:       0.4331

e65:
  val_L2_avg:      1.1329
  val_route_L2:    0.1104
  val_route_final: 0.2259
  speed_MAE:       0.4303
```

Close-loop result:

```text
/data/z_project/code/Bench2Drive/eval/results/route_b_b2d_0531_paper_oldddim_best_skip23695_24071
```

Close-loop score:

```text
Driving Score: 88.718
routes:        218 / 218
missing:       0
zero-score:    0
```

Interpretation:

- The sampling alignment fixed the catastrophic clean-paper route metric issue.
- N1 is readable and clean, but route metrics still lag the legacy Route-B style
  skeleton.

## N2: paper_unified_no_detail_nostate

Short name:

```text
paper_unified_no_detail_nostate
```

Role:

```text
Clean paper policy/training path, but with the legacy unified decoder skeleton
reintroduced without detail attention. This isolates whether the old
UnifiedDecoderOnlyTransformer / MultiSourceAttentionBlock / base grid BEV
skeleton is important.
```

Structure:

- Uses `PaperMotionPolicy`.
- Uses `PaperMotionUnifiedNoDetailCore`.
- Wraps the legacy motion skeleton through `TransformerForDiffusion` with
  `motion_only_model=true`.
- Reintroduces the old unified decoder / multi-source attention / base grid BEV
  style.
- Explicitly keeps detail attention disabled.
- Does not include semantic/stage1/graph/tempocc/chase/alignment/route-intent.
- Uses old pred_x0 DDIM sampling.

Known config / checkpoint:

```text
config:
  config/paper_motion_unified_no_detail_0531.yaml

training script:
  scripts/codex_bash/train_paper_motion_unified_no_detail_0531.sh

checkpoint:
  checkpoints/paper_motion_unified_no_detail_0531/dit_policy_best.pt
```

Open-loop reference:

```text
best by L2:
  e35 / dit_policy_best.pt

e35 metrics:
  val_L2_avg:      1.0168
  val_route_L2:    0.0997
  val_route_final: 0.1922
  speed_MAE:       0.4095

e65 metrics:
  val_L2_avg:      1.0827
  val_route_L2:    0.0937
  val_route_final: 0.1809
  speed_MAE:       0.4008
```

Close-loop result:

```text
10-step result:
  /data/z_project/code/Bench2Drive/eval/results/route_b_b2d_0608_n2_unified_no_detail_best_skip23695_24071

1-step result:
  /data/z_project/code/Bench2Drive/eval/results/route_b_b2d_0609_n2_unified_no_detail_best_1step_skip23695_24071

ckpt:
  checkpoints/paper_motion_unified_no_detail_0531/dit_policy_best.pt

cal_score.py result:
  10-step:
    routes:        218 / 218
    valid:         218
    success:       152
    Driving Score: 88.650
    Success Rate:  69.72%

  1-step:
    routes:        200 / 218 partial run
    valid:         199 after excluding one zero-score route
    success:       140
    Driving Score: 88.826
    Success Rate:  70.35%
```

Interpretation:

- N2 clearly improves over N1 in open-loop.
- The unified decoder / base grid BEV skeleton appears important for route
  learning.
- N2 10-step and 1-step close-loop are very close under the current scoring
  convention.
- The 1-step number is partial because only 200 routes were merged. Treat it as
  evidence for step-count insensitivity, not as the final full-set score.

## Current Comparison

Open-loop:

```text
model                         ckpt      L2_avg   route_L2   route_final   speed_MAE
legacy_routeb_nostate          e60       -        0.0841     0.1714        0.8516(head) / 0.5406(traj_1s)
paper_simple_oldddim_nostate   e45/best  1.1054   0.1154     0.2319        0.4331
paper_unified_no_detail        e35/best  1.0168   0.0997     0.1922        0.4095
paper_unified_no_detail        e65       1.0827   0.0937     0.1809        0.4008
```

Close-loop:

```text
model                         ckpt       steps  routes      valid  Driving Score  Success Rate
legacy_routeb_nostate          e60        10     220/220     220    91.270         78.64%
paper_simple_oldddim_nostate   best/e45   10     218/218     218    88.718         -
paper_unified_no_detail        best/e35   10     218/218     218    88.650         69.72%
paper_unified_no_detail        best/e35   1      200/218*    199    88.826         70.35%
legacy_routeb_nostate          e55        10     later rerun; keep separate from original e60
```

`*` The 1-step N2 result is a partial close-loop run. `cal_score.py` excluded one
zero-score route, so the averaged score is over 199 valid routes.

## Diffusion Interpretation Note

Current working hypothesis:

```text
In this Route-B motion-only setting, DDIM behaves more like a
scene-conditioned denoising/refinement decoder than a strong multimodal
trajectory generator.
```

The model has three conceptually different inputs:

```text
scene/context tokens:
  BEV, ego status, command / target information.

traj/route token identity:
  learnable query/type/segment/position embeddings that tell the decoder which
  slots are trajectory waypoints and which slots are route waypoints.

x_t noisy waypoint tokens:
  noisy trajectory/route coordinates. During training these are noisy GT at a
  random timestep; during inference they start from white noise and are updated
  through DDIM.
```

Important clarification:

```text
The traj/route learnable queries are not random latent tokens at inference.
They are fixed learned type/task priors. The random part is x_t.
```

Behavioral view:

```text
one-shot decoder:
  scene -> motion

current DDIM denoising policy:
  scene + noisy motion proposal + timestep -> clean motion estimate
  repeated for several steps
```

Because the dataset is mostly single-expert imitation, the conditional decoder
can learn to map different noisy proposals back to the same dominant expert
solution. In that case DDIM stochasticity is weak, and the noise mainly acts as:

```text
denoising regularization
iterative refinement
test-time proposal perturbation / implicit ensemble
```

This means we should be careful not to oversell the current method as a
multimodal generator. A more honest paper framing is:

```text
scene-conditioned denoising policy
```

or:

```text
conditional diffusion as robust iterative motion refinement
```

DDPM may preserve more stochasticity because it injects noise at each reverse
step, but with one expert trajectory per scene it may still collapse to the same
expert mode unless we add explicit multimodal labels, mode tokens, diversity
losses, critic/rerank, or RL-style exploration.

## Proposed Noise-Sensitivity Experiment

Purpose:

```text
Measure whether DDIM initial noise actually produces diverse traj/route outputs,
or whether the model collapses to a nearly deterministic scene-to-motion mapper.
```

Protocol:

```text
For each selected validation / close-loop frame:
  fix scene features, ego status, command, and checkpoint
  sample K different initial x_T noises, e.g. K=16 or K=32
  run the same DDIM sampler for each seed
  measure final traj/route diversity
```

Suggested metrics:

```text
route_endpoint_std
route_mean_pairwise_L2
traj_endpoint_std
traj_mean_pairwise_L2
speed_class_entropy
speed_mps_std
```

Useful splits:

```text
easy straight / lane follow
junction
merge
borrow / obstacle
high route curvature
window-active semantic scenes
```

Expected result:

```text
Route diversity is probably very small.
Trajectory diversity may be slightly larger near speed / interaction-sensitive
scenes, but likely remains much smaller than a true multimodal policy.
```

Interpretation:

```text
If diversity is tiny, DDIM should be described as robust denoising/refinement.
If diversity is meaningful in interaction scenes, then noise may still provide a
limited multimodal proposal mechanism.
```

## Noise-Sensitivity Eval Result

Experiment:

```text
model:
  N2 paper_unified_no_detail_nostate

checkpoint:
  checkpoints/paper_motion_unified_no_detail_0531/dit_policy_best.pt

config:
  config/paper_motion_unified_no_detail_0531.yaml

sampling:
  old_pred_x0_ddim

dataset:
  validation split, first 128 samples after bad-route filtering

seeds:
  32 initial x_T noise seeds per fixed scene
```

Output files:

```text
10-step:
  outputs/noise_sensitivity/n2_best_10step_k32_n128_20260610.json
  outputs/noise_sensitivity/n2_best_10step_k32_n128_20260610.csv

1-step:
  outputs/noise_sensitivity/n2_best_1step_k32_n128_20260610.json
  outputs/noise_sensitivity/n2_best_1step_k32_n128_20260610.csv
```

Aggregate diversity:

```text
metric                         10-step mean   10-step median   10-step p95    1-step mean   1-step median   1-step p95
route_point_std                0.00625        0.000025         0.000163       0.00959       0.00573         0.00849
route_endpoint_std             0.00644        0.000010         0.000033       0.00882       0.00189         0.00561
route_pairwise_l2_per_point    0.00770        0.00176          0.00403        0.01342       0.00858         0.01831
traj_point_std                 0.12475        0.00322          0.74222        0.03995       0.01190         0.13790
traj_endpoint_std              0.29013        0.00387          1.51576        0.09329       0.01704         0.30605
traj_pairwise_l2_per_point     0.17473        0.00394          1.06469        0.06269       0.01906         0.20969
speed_mps_std                  0.03164        0.00101          0.16509        0.01444       0.00325         0.05680
speed_entropy                  0.30901        0.09130          0.75273        0.31491       0.09503         0.76866
```

GT error under the same noise-sensitivity protocol:

```text
metric                         10-step mean   10-step median   10-step p95    1-step mean   1-step median   1-step p95
traj_ade_mean                  0.62993        0.31861          2.41510        0.59341       0.25582         2.20063
traj_ade_best_of_32            0.45322        0.21925          1.93622        0.53610       0.21528         2.01206
traj_ade_worst_of_32           0.84224        0.35591          3.15761        0.66016       0.28314         2.45598
traj_fde_mean                  1.47905        0.46243          6.14399        1.37107       0.34937         5.55761
traj_fde_best_of_32            1.06914        0.26339          5.08519        1.20716       0.21032         5.00793
traj_fde_worst_of_32           1.96806        0.58955          7.59154        1.56401       0.44415         6.38878
```

The GT-error result answers the natural follow-up question:

```text
For a normal single sampled output, 1-step is slightly closer to GT than
10-step on this 128-sample subset.

For oracle best-of-32, 10-step can sometimes contain a better candidate, but it
also produces worse candidates and a larger long tail.
```

Interpretation:

```text
10-step increases proposal spread more than it consistently improves accuracy.
This is useful evidence that iterative DDIM is not giving reliable refinement in
the current route/speed-split setup. It mostly adds small route perturbation and
occasionally larger trajectory variation.
```

Interpretation:

```text
Route is almost deterministic for most scenes under different initial noise.
The 10-step median route endpoint std is only about 1e-5 m, and even the p95 is
only about 3e-5 m. This strongly supports the view that route diffusion is
mostly scene-conditioned spatial denoising / refinement rather than meaningful
multimodal route sampling.
```

The 1-step route has slightly larger small jitter:

```text
route_pairwise_l2_per_point:
  10-step median: 0.00176 m
  1-step median:  0.00858 m
```

but this is still tiny relative to the 1m-ish route waypoint spacing.

Trajectory has a different shape:

```text
Most traj samples are also stable, but there is a long tail.
10-step traj diversity can be larger than 1-step in outlier scenes.
```

This means repeated denoising does not create broad multimodality, but it can
amplify proposal differences for a small subset of interaction-sensitive
frames. For paper framing:

```text
Route diffusion:
  robust spatial route refiner.

Trajectory output:
  mostly deterministic, with limited noise sensitivity in long-tail scenes.

Current policy as a whole:
  not a strong multimodal generator; better described as conditional denoising
  / robust iterative motion refinement.
```

## 1-Step vs 10-Step Close-Loop Observation

Observation:

```text
N2 1-step close-loop is close to the 10-step close-loop result.
```

This supports the "DDIM as scene-conditioned denoising/refinement" hypothesis,
but it should not be attributed to DDIM collapse alone. There is an important
task-structure reason:

```text
Our current motion output is effectively split into:

route:
  diffusion output, spatial waypoints at roughly 1m intervals.

speed:
  direct speed head output, not generated through the diffusion loop.

trajectory / control:
  downstream behavior is strongly influenced by route + direct speed.
```

The route prediction task is mostly spatial:

```text
route ~= road structure + scene geometry + target point / command
```

It is much less tied to exact timing and vehicle interaction than a full
time-indexed trajectory. Therefore, 1-step route denoising and 10-step route
denoising can be very similar:

```text
route has low temporal/interactor ambiguity
route waypoints are dense spatial samples
route is strongly constrained by BEV/road/target-point context
```

At the same time, speed is direct:

```text
speed does not benefit from, or degrade with, the number of DDIM route steps.
```

So the close-loop result can stay nearly unchanged even if the diffusion loop is
reduced from 10 steps to 1 step:

```text
1-step route ~= 10-step route
speed unchanged
controller sees nearly the same route + speed
=> close-loop score nearly unchanged
```

Important implication:

```text
The 1-step result does not necessarily mean diffusion is useless for all motion
prediction. It may mean that route diffusion is an easy spatial denoising task.
A time-indexed trajectory diffusion head could show a larger 1-step vs 10-step
gap because timing, speed, and interactions are entangled in trajectory.
```

For paper framing, this suggests:

```text
Our current diffusion route head is best interpreted as robust spatial route
denoising. The direct speed head carries most timing/speed responsibility.
```

## Next Ladder Candidates

Candidate next steps:

```text
N3 = paper_unified_route_intent_nostate
  Add route-intent / TG token on top of N2, still nostate.

N4 = paper_unified_base_detail_nostate
  Add selected detail attention back, still nostate.

S0 = add semantic state heads, no semantic-to-motion condition
  Test whether semantic supervision alone changes the shared latent.

S1 = compact semantic-to-motion condition
  Add the smallest useful state-to-motion path.

S2 = final SOTA-style semantic mode chain
  Add the components required by the paper story, one at a time.
```

Guideline:

```text
Only add one major mechanism per ladder step. If a step changes both model
structure and training trick, split it.
```

State ladder implementation plan:

```text
docs/tmp/paper_state_ladder_s0_s2_plan_20260611.md
```

## N3: paper_unified_route_intent_nostate

Short name:

```text
paper_unified_route_intent_nostate
```

Role:

```text
Test whether the strong legacy nostate e60 result is mainly explained by the
route-intent / TG token rather than detail attention, lidar history, or semantic
state.
```

Structure:

- Starts from N2 (`paper_unified_no_detail_nostate`).
- Keeps the clean `PaperMotionPolicy` path.
- Keeps the legacy unified decoder skeleton through `PaperMotionUnifiedNoDetailCore`.
- Keeps semantic/stage1/graph/tempocc/chase/alignment/lidar/detail paths disabled.
- Adds only route intent:

```text
route_intent = command(6) + target_point(2) + target_point_next(2)
```

This is projected to the route-token embedding with a learnable gate initialized
to `0.1`, matching the recovered legacy e60 recipe.

Config / script:

```text
config:
  config/paper_motion_unified_route_intent_0610.yaml

training script:
  scripts/codex_bash/train_paper_motion_unified_route_intent_0610.sh

checkpoint dir:
  checkpoints/paper_motion_unified_route_intent_0610
```

Expected comparison:

```text
N2:
  unified decoder, no detail, no route intent

N3:
  unified decoder, no detail, route intent enabled

legacy e60:
  legacy large route-b path, route intent enabled, not paper-clean
```

If N3 closes much of the gap to legacy e60, route intent should be treated as a
core nostate component. If it does not, the remaining gap is likely from older
decoder/path details rather than TG token alone.
