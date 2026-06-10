# Paper State Ladder S0-S2 Plan

Date: 2026-06-11

Worktree:

```text
/media/z/data/mzq/others/MoT-DP-worktrees/semantic_state_strict_ablation_v1
```

40G mirror:

```text
/data/z_project/code/motdp_z_semantic_state_strict_ablation_v1
```

## Purpose

After N3, start adding semantic state back in a controlled ladder. The goal is
not to immediately reproduce the previous SOTA kitchen-sink setup, but to
separate three effects:

```text
S0:
  semantic supervision effect on the shared latent/backbone.

S1:
  minimal semantic-to-motion conditioning effect.

S2:
  paper-facing structured semantic mode chain effect.
```

Base assumption:

```text
N3 = paper_unified_route_intent_nostate
```

Use N3 as the motion baseline unless N3 clearly fails. If N3 ends weaker than
legacy e60, still keep this ladder because the attribution is cleaner than the
old mixed branch.

## Shared Principles

- Keep the clean paper training entry when possible.
- Do not reintroduce legacy energy/alignment/rerank in S0-S2.
- Do not enable lidar history or detail attention in this ladder.
- Keep route intent enabled because it is part of the N3 nostate base.
- Keep GPS noise unchanged: `sigma=0.05`, `probability=1`, `apply_to_route=false`.
- Keep motion losses and schedule aligned with N3 unless explicitly noted.
- Add one semantic mechanism at a time.

## Code Structure

### Current N3 stack

```text
PaperMotionPolicy
  -> PaperMotionUnifiedNoDetailCore
      -> TransformerForDiffusion(motion_only_model=True)
```

N3 enables:

```text
use_route_intent_token: true
```

N3 disables:

```text
stage1 / semantic heads
semantic transition
graph decoder
traj branch semantic condition
route prev coarse memory
lidar detail
detail attention
alignment critic
```

### Proposed S stack

Add a new paper policy/core pair rather than stuffing more switches into
`PaperMotionPolicy`:

```text
policy/paper_semantic_motion_policy.py
model/paper_semantic_motion_core.py
```

The new core can wrap `TransformerForDiffusion` with `motion_only_model=false`
but restrict enabled modules through explicit paper-ladder config:

```text
paper_semantic_ladder_stage: s0 | s1 | s2
```

This keeps the old research policy intact and gives paper code a readable
semantic path.

## S0: Semantic Heads Only

Short name:

```text
paper_s0_semantic_heads_no_condition
```

Question:

```text
Does training semantic state heads hurt/help the shared representation when the
motion branch cannot consume semantic predictions?
```

Structure:

```text
N3 motion stack
+ semantic direct heads
- semantic-to-motion condition
- semantic transition / history chain
- graph decoder unless required by selected labels
```

Recommended supervision profile:

```yaml
semantic_state_supervision_profile: window_decision_control
```

Rationale:

```text
Start with robust, coarse paper-facing states:
  window
  decision_phase
  control_phase

Do not start with area/tempocc/boundary/chase because they are heterogeneous and
can recreate the old loss tug-of-war.
```

Config sketch:

```yaml
policy_type: paper_semantic_motion
route_b:
  paper_semantic_ladder_stage: s0
  paper_motion_core: unified_no_detail
  use_route_intent_token: true
  train_stage1: true
  use_stage1_state: false
  use_traj_branch_condition: false
  use_semantic_state_transition: false
  use_cover_relation_graph_decoder: false
  semantic_state_supervision_profile: window_decision_control
  semantic_motion_condition_profile: window_decision_control
  stage1_loss_weight: 1.0
```

Expected outputs:

```text
motion metrics:
  compare to N3 open-loop and close-loop

state metrics:
  window / decision / control phase recall and CE
```

Interpretation:

```text
If S0 hurts motion, semantic losses are still fighting the motion backbone.
If S0 keeps motion stable, we can safely test semantic-to-motion conditioning.
```

## S1: Minimal Semantic-to-Motion Condition

Short name:

```text
paper_s1_window_phase_condition
```

Question:

```text
Does the smallest causal semantic condition improve motion once semantic heads
exist?
```

Structure:

```text
S0
+ semantic-to-motion condition for window/phase only
```

Condition groups:

```text
window_probs
decision_phase_probs
control_phase_probs
```

Do not inject:

```text
dir
area mask
timing scalars
temporary occupancy bins
boundary speeds
chase
cover graph edge modes
```

Config sketch:

```yaml
policy_type: paper_semantic_motion
route_b:
  paper_semantic_ladder_stage: s1
  use_route_intent_token: true
  train_stage1: true
  use_stage1_state: true
  use_traj_branch_condition: true
  semantic_motion_condition_profile: window_decision_control
  semantic_state_supervision_profile: window_decision_control
  semantic_condition_detach: true
  traj_branch_condition_scale: 1.0
```

Important:

```text
Use detached predicted semantic probabilities for motion condition by default.
This avoids motion loss directly reshaping semantic heads in the first S ladder.
```

Expected interpretation:

```text
S1 > S0:
  coarse semantic mode condition helps motion.

S1 ~= S0:
  route intent / scene features already provide the same information.

S1 < S0:
  even coarse state injection is noisy or misaligned.
```

## S2: Structured Semantic Mode Chain

Short name:

```text
paper_s2_structured_semantic_chain
```

Question:

```text
Does a paper-facing semantic chain improve state quality and motion behavior
without reviving the old next-token collapse or loss tug-of-war?
```

Structured order:

```text
window
  -> spatial_area
  -> temporal_opportunity
  -> phase/control
```

V1 implementation:

```text
direct_group = f(current obs + route/context)
chain_delta = g(direct_group, previous group tokens, weak history)
final_group = direct_group + gate * chain_delta
```

History usage:

```yaml
use_semantic_chain_history: true
semantic_chain_history_prefix: hist4
semantic_chain_history_scale: 0.2
semantic_chain_history_dropout_prob: 0.5
```

Direct ability requirement:

```text
Every group keeps a direct head.
The chain only adds residual refinement.
If history is missing/dropped, prediction should still work from current frame.
```

Recommended supervision profile:

```yaml
semantic_state_supervision_profile: window_phase_opportunity
```

Potential S2 condition groups:

```text
window
decision/control phase
go_opportunity/yld_pressure
```

Still do not inject in first S2:

```text
raw area mask
raw tempocc bins
boundary speeds
chase
graph edge mode
```

Config sketch:

```yaml
policy_type: paper_semantic_motion
route_b:
  paper_semantic_ladder_stage: s2
  use_route_intent_token: true
  train_stage1: true
  use_stage1_state: true
  use_traj_branch_condition: true
  use_semantic_chain: true
  use_semantic_chain_history: true
  semantic_chain_history_prefix: hist4
  semantic_state_supervision_profile: window_phase_opportunity
  semantic_motion_condition_profile: window_phase_opportunity
  semantic_condition_detach: true
  traj_branch_condition_scale: 1.0
```

## Proposed Files

Configs:

```text
config/paper_s0_semantic_heads_no_condition_0611.yaml
config/paper_s1_window_phase_condition_0611.yaml
config/paper_s2_structured_semantic_chain_0611.yaml
```

Scripts:

```text
scripts/codex_bash/train_paper_s0_semantic_heads_no_condition_0611.sh
scripts/codex_bash/train_paper_s1_window_phase_condition_0611.sh
scripts/codex_bash/train_paper_s2_structured_semantic_chain_0611.sh
scripts/codex_bash/train_paper_state_ladder_s0_s2_0611.sh
```

Policy/model:

```text
policy/paper_semantic_motion_policy.py
model/paper_semantic_motion_core.py
```

Optional utilities:

```text
scripts/eval_paper_state_metrics.py
```

## Training Commands

### S0

```bash
cd /data/z_project/code/motdp_z_semantic_state_strict_ablation_v1

NCCL_P2P_DISABLE=1 \
NCCL_CUMEM_ENABLE=0 \
GLOO_SOCKET_IFNAME=bond1 \
NCCL_SOCKET_IFNAME=bond1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
GPUS=4 \
MASTER_PORT=29611 \
bash scripts/codex_bash/train_paper_s0_semantic_heads_no_condition_0611.sh
```

### S1

```bash
cd /data/z_project/code/motdp_z_semantic_state_strict_ablation_v1

NCCL_P2P_DISABLE=1 \
NCCL_CUMEM_ENABLE=0 \
GLOO_SOCKET_IFNAME=bond1 \
NCCL_SOCKET_IFNAME=bond1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
GPUS=4 \
MASTER_PORT=29612 \
bash scripts/codex_bash/train_paper_s1_window_phase_condition_0611.sh
```

### S2

```bash
cd /data/z_project/code/motdp_z_semantic_state_strict_ablation_v1

NCCL_P2P_DISABLE=1 \
NCCL_CUMEM_ENABLE=0 \
GLOO_SOCKET_IFNAME=bond1 \
NCCL_SOCKET_IFNAME=bond1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
GPUS=4 \
MASTER_PORT=29613 \
bash scripts/codex_bash/train_paper_s2_structured_semantic_chain_0611.sh
```

### Sequential Runner

```bash
cd /data/z_project/code/motdp_z_semantic_state_strict_ablation_v1

CUDA_VISIBLE_DEVICES=4,5,6,7 \
GPUS=4 \
bash scripts/codex_bash/train_paper_state_ladder_s0_s2_0611.sh
```

## Dependency Choice

Two possible training flows:

```text
independent:
  S0, S1, S2 all train from scratch with the same N3 architecture base.

sequential:
  S1 initializes from S0 best.
  S2 initializes from S1 best.
```

Recommendation:

```text
Start independent for attribution if time allows.
Use sequential only if training cost becomes the bottleneck.
```

## Evaluation Plan

Open-loop:

```text
val_L2_avg
val_route_L2
val_route_final
val_speed_mae_speed_head
state CE / recall for enabled groups
```

Close-loop:

```text
Run N3, S0, S1, S2 best checkpoints on the same route set.
Use cal_score.py and record both route count and valid count.
```

Key comparisons:

```text
S0 vs N3:
  semantic loss effect.

S1 vs S0:
  semantic condition effect.

S2 vs S1:
  structured chain effect.
```

## Acceptance / Stop Criteria

```text
If S0 already hurts route metrics badly:
  stop and reduce semantic supervision profile to window_only or window_decision.

If S1 hurts close-loop:
  keep semantic heads as auxiliary only; do not condition motion directly.

If S2 improves state metrics but not close-loop:
  keep it as analysis/interpretability, not as final driving model.
```

## Notes

The previous strong 0511/e35 and legacy e60 results mixed many mechanisms. This
ladder deliberately trades peak performance speed for attribution clarity. The
paper story should only claim a component is useful if it survives this ladder.
