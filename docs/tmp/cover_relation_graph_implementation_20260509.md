# Cover Relation Graph Implementation Note 20260509

This records the implementation currently landed in:

`/media/z/data/mzq/others/MoT-DP-worktrees/semantic_state_next_token_rl_v1`

---

## Code Review Notes (2026-05-09)

审查时发现以下 4 个问题，均已在文档中修正：

### 1. Plan doc Section 6（Inference Fuse/Cache）未标注 V2

**问题**：plan doc 将 inference fuse/cache 写在主体设计中，未注明 V1 未实现，新 session 读者可能误以为已完工。

**修正**：plan doc Section 6 标题改为 `[V2 / Not in V1]`，并加了 Status 说明。

**代码实情**：实现文档 Known V1 Simplifications 已有说明，但 plan doc 未同步。

---

### 2. Plan doc Behavior Check 中 edge mode 仅列 3 类，实际为 5 类

**问题**：plan doc Test Plan → Behavior Check 写的是：
```
pass_after_current / go_before_future / yield_after_future
```
但代码实际有 5-way 枚举：`none(0)`, `pass_after_current(1)`, `go_before_future(2)`, `yield_after_future(3)`, `ambiguous(4)`。

**修正**：plan doc 对应位置补全 5 类枚举。implementation doc graph head 清单后新增枚举表。

**代码验证位置**：`policy/annealed_energy_guidance_policy.py` `traj_edge_condition_names`；`model/transformer_for_diffusion_multi_head.py` `shared_stage1_current_edge_mode_head(out_dim=5)`。

---

### 3. Transition branch 中 speed head 的 token 来源与 direct branch 不同，文档未记录

**问题**：direct branch 4 个 speed head 都读 `semantic_feature`（shared neck 输出）；但 transition branch 中：
- `front_follow_upper_speed` → `chase_token`
- `merge_flow_lower_speed` → `boundary_token`

这是一个隐含的设计决策，两份文档原本均未提及，将来调试时容易忽略。

**修正**：implementation doc graph head 清单后新增该 token 选择说明。

**代码位置**：`model/transformer_for_diffusion_multi_head.py` line 2508–2509。

---

### 4. Implementation doc 缺少各 graph loss 的默认权重

**问题**：implementation doc 仅列出了 loss 名称，未记录默认权重，而这些权重是实验关键超参。

**修正**：loss 列表改为表格，含各项默认权重：

| loss | 默认权重 |
|---|---|
| current / future edge valid/mode | 0.10 |
| current_cover_upper / future_cover_lower | 0.15 |
| front_follow_upper / merge_flow_lower | 0.10 |
| edge_speed_consistency（默认关闭） | 0.05 |

**代码位置**：`policy/annealed_energy_guidance_policy.py` line 199–227。

---

## Scope

Implemented the first code version of **Area-Cover Relation Graph + Compact Semantic Conditioning**.

This is a training/model implementation note, not a label-generation note. It assumes the new graph labels have been generated, projected into packed samples, and split into train/val before training.

## Dataset Interface

`dataset/unified_carla_dataset.py` now passes through the new graph fields:

- `current_cover_edge_valid`
- `current_cover_edge_occupied`
- `current_cover_edge_mode`
- `current_cover_edge_mode_valid`
- `current_cover_upper_speed_mps`
- `current_cover_upper_speed_valid`
- `current_cover_upper_speed_source`
- `future_cover_edge_valid`
- `future_cover_edge_mode`
- `future_cover_edge_mode_valid`
- `future_cover_lower_speed_mps`
- `future_cover_lower_speed_valid`
- `future_cover_lower_speed_source`
- `front_follow_upper_speed_mps`
- `front_follow_upper_speed_valid`
- `merge_flow_lower_speed_mps`
- `merge_flow_lower_speed_valid`

Integer fields are converted to `long`; scalar/probability/speed/valid fields are converted to float tensors.

## Model Changes

`model/transformer_for_diffusion_multi_head.py` now supports:

- `use_cover_relation_graph_decoder`
- `cover_graph_use_traj_context`
- `cover_graph_use_speed_context`
- `semantic_motion_condition_mode`
- `use_route_prev_coarse_memory`

### Route-Aware Graph Heads Update

Follow-up calibration showed that `current_cover_edge_valid` had useful probability separation but weak recall at the default threshold. The root cause is that cover-edge valid/mode was implemented as a global semantic scalar head, while the label is route-local: whether the ego route/corridor currently or soon has a cover relation.

The graph valid/mode heads now use a route-query residual path in both direct and semantic-transition branches:

- direct graph input: `[route_out, route_geom_tokens, semantic_feature]`
- transition graph input: `[route_out, route_geom_tokens, timing_token]`
- per-route logits are pooled with smooth logsumexp pooling and added as a residual to the original global logits

This mirrors the older `conflict_area_logits` design, which already used explicit route tokens. `route_out` has already gone through route point BEV sampling inside the ego decoder, so this gives graph valid/mode direct access to route-local BEV evidence instead of relying only on a pooled semantic feature.

When `use_cover_relation_graph_decoder=true`, the shared semantic neck can avoid shortcutting through motion detail:

- `cover_graph_use_traj_context=false` zeros `traj_summary`.
- `cover_graph_use_speed_context=false` zeros `speed_summary`.
- Route summary, route geometry, and conditioning remain available.

New direct graph heads:

- `current_cover_edge_valid_logit`
- `current_cover_edge_mode_logits` with 5 classes
- `future_cover_edge_valid_logit`
- `future_cover_edge_mode_logits` with 5 classes
- `current_cover_upper_speed`
- `future_cover_lower_speed`
- `front_follow_upper_speed`
- `merge_flow_lower_speed`

Edge mode enum (applies to both current and future, 5 classes):

| value | name | meaning |
|---|---|---|
| 0 | `none` | no cover edge applicable |
| 1 | `pass_after_current` | ego must pass after current cover clears |
| 2 | `go_before_future` | ego can go before future cover arrives |
| 3 | `yield_after_future` | ego must yield until future cover passes |
| 4 | `ambiguous` | conflicting or unclear timing |

The same graph heads were also added to the semantic transition branch.

Note on transition branch token choice for speed heads:

- Direct branch: all 4 speed heads read from `semantic_feature` (shared neck output).
- Transition branch: `front_follow_upper_speed` reads from `chase_token`; `merge_flow_lower_speed` reads from `boundary_token`. This matches the semantic grouping of those constraints within the transition slot structure.

## Compact Motion Condition

`semantic_motion_condition_mode=compact_graph` changes the branch condition from the older full-state vector to a compact graph vector.

Compact condition includes:

- `window_probs`
- `decision_phase_probs`
- `control_phase_probs`
- `[yld_pressure, go_opportunity]`
- current edge valid + 5-way current edge mode probs
- future edge valid + 5-way future edge mode probs
- 4 speed margins:
  - current cover upper margin
  - future cover lower margin
  - front-follow upper margin
  - merge-flow lower margin
- 4 speed valid flags
- `borrow_time`

Compact condition intentionally does **not** directly inject:

- `dir`
- `conflict_area_logits`
- `area_status`
- `timing`
- `temporary_occupancy_bins`
- raw family `yld/go` boundaries

This keeps motion conditioning closer to causal motion constraints instead of feeding all auxiliary state into traj/speed.

Current compact condition dim is `33`.

## Route Previous Coarse Memory

Added `use_route_prev_coarse_memory`.

When enabled, route tokens receive only previous coarse semantic memory:

- `prev_semantic_state_valid`
- previous window/family as 4-way one-hot
- previous dir as 4-way one-hot

This is added through an MLP to route tokens only. It does not directly condition traj tokens.

The intent is to help route continuity, especially borrow/merge near-end cases, without letting current detailed state create route-state circular reasoning.

## Policy / Loss Changes

`policy/annealed_energy_guidance_policy.py` now:

- Validates graph fields when `use_cover_relation_graph_decoder=true`.
- Builds graph targets from batch.
- Builds compact graph branch condition for GT training and inference raw predictions.
- Adds graph losses for direct semantic heads.
- Adds graph losses for transition semantic heads.
- Adds graph terms to state consistency and direct/transition consistency.

New graph losses:

| loss | default weight |
|---|---|
| `current_edge_valid_loss` | 0.10 |
| `current_edge_mode_loss` | 0.10 |
| `future_edge_valid_loss` | 0.10 |
| `future_edge_mode_loss` | 0.10 |
| `current_cover_upper_loss` | 0.15 |
| `future_cover_lower_loss` | 0.15 |
| `front_follow_upper_loss` | 0.10 |
| `merge_flow_lower_loss` | 0.10 |
| `edge_speed_consistency_loss` | 0.05 (disabled by default) |

`use_edge_speed_consistency_loss` exists but defaults to `false`.

If enabled, it uses only edge-aware constraints:

- current cover upper speed
- future cover lower speed only when `future_edge_mode == go_before_future`
- front-follow upper speed
- merge-flow lower speed

It does **not** use the old global rule `phase=go -> family_go_min`.

## Debug Outputs

Inference/debug now exposes graph-related outputs where available:

- `traj_current_edge_condition_probs`
- `traj_future_edge_condition_probs`
- `traj_edge_speed_margin_condition`
- `traj_edge_speed_valid_condition`
- `stage1_current_cover_edge_valid_prob`
- `stage1_current_cover_edge_mode_probs`
- `stage1_future_cover_edge_valid_prob`
- `stage1_future_cover_edge_mode_probs`
- `stage1_current_cover_upper_speed_mps`
- `stage1_future_cover_lower_speed_mps`
- `stage1_front_follow_upper_speed_mps`
- `stage1_merge_flow_lower_speed_mps`

Backward-compatible `speed_energy_*` aliases are still emitted.

## Config

Updated:

- `config/pdm_hpc_route_b_lidar_bev_stage1_tempocc_0429.yaml`
- `config/pdm_local_route_b_lidar_bev_stage1_tempocc_0429.yaml`

HPC config currently enables:

```yaml
semantic_motion_condition_mode: compact_graph
use_cover_relation_graph_decoder: true
cover_graph_use_traj_context: false
cover_graph_use_speed_context: false
use_route_prev_coarse_memory: true
```

Local config documents the new switches but keeps graph disabled for compatibility with older/local mini data.

HPC run/checkpoint name was changed to:

`route_b_stage1_tempocc_covergraph_0509`

## Verification Done

Static checks passed:

- `python -m py_compile policy/annealed_energy_guidance_policy.py`
- `python -m py_compile model/transformer_for_diffusion_multi_head.py`
- `python -m py_compile dataset/unified_carla_dataset.py`
- `git diff --check`

Config/model init checks passed:

- HPC compact graph condition dimension: `33`
- compact condition name count: `33`
- local config fallback remains full mode: `28`

Small GT branch-condition smoke passed:

- synthetic graph-label batch builds compact condition with shape `(B, 33)`

## Important Deployment Note

The new HPC config requires graph fields in the train/val packed dataset.

At the time of implementation, new_hpc had generated the graph relabel pkl, but project/split had not yet run. Training should start only after graph fields are projected into the packed train/val split.

If training starts on the old split, it should fail fast with a graph-field missing error. That is intentional.

## Known V1 Simplifications

- No object grounding / bbox / actor tracking.
- No BEV grid sampling / deformable BEV lookup for graph heads.
- No confidence/uncertainty modeling.
- Transition branch outputs graph heads, but V1 does not require previous edge labels as transition inputs.
- Inference-side direct/transition semantic cache fusion is not fully implemented in this pass; current inference still primarily uses direct graph state for branch condition.
- `borrow_time` behavior was not changed in this patch.
