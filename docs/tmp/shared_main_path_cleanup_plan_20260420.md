# Shared Main Path Cleanup Plan

## Goal

在已经完成 shared stage1 path 接线之后，清理旧的 `mode_out -> forward_speed_energy*` 这条
legacy stage1 path，把 stage1 / semantic 相关逻辑统一到 shared main path 上。

这次 cleanup 的目标是：

- 删除旧的 stage1 speed-energy fallback
- 保留 shared main path semantic neck
- 保留 guidance / alignment 仍然需要的 trajectory energy path
- 让 train / infer / agent debug 全部只认 shared path

## Important Scope Boundary

这次 cleanup 只删除：

- legacy stage1 speed-energy path
- old stage1 train/infer fallback branches
- 为 fallback 存在的 config switch

这次 cleanup 不删除：

- `forward_energy(...)`
- `forward_energy_eval(...)`
- `forward_front_route_risk(...)`
- `forward_front_route_risk_eval(...)`
- `_forward_traj_energy_context(...)`

原因：

- 上面这些仍然服务于 guidance / alignment / front-route-risk
- 它们虽然也是 energy path，但不是这次要清的 old stage1 path

## Current Legacy Stage1 Pieces

### Model side

当前仍然属于 legacy stage1 path 的主要内容：

- legacy heads:
  - `speed_energy_chase_head`
  - `speed_energy_merge_yld_head`
  - `speed_energy_merge_go_head`
  - `speed_energy_junction_yld_head`
  - `speed_energy_junction_go_head`
  - `speed_energy_borrow_yld_head`
  - `speed_energy_borrow_go_head`
  - `speed_energy_pedestrian_head`
  - `speed_energy_merge_active_head`
  - `speed_energy_junction_active_head`
  - `speed_energy_borrow_active_head`
  - `speed_energy_lane_dir_relation_head`
- legacy query blocks:
  - `speed_energy_speed_query_proj`
  - `speed_energy_query_token`
  - `speed_energy_query_attn`
  - `speed_energy_query_norm`
- legacy score builder:
  - `_compute_speed_energy_scores(...)`
- legacy public entry:
  - `forward_speed_energy(...)`
  - `forward_speed_energy_eval(...)`

仍应保留但可能要重命名的内容：

- `speed_energy_route_proj`

原因：

- shared path 现在也在复用它做 `route_geom`
- cleanup 时可以先保留实现，再改名为 shared 语义更清楚的名字，比如
  `shared_stage1_route_geom_proj`

### Policy side

当前仍然属于 legacy fallback 的主要内容：

- `self.use_shared_stage1_path`
- `if self.use_shared_stage1_path: ... else: ...` 的双分支
- `_compute_stage1_speed_energy_loss(...)` 这套 old path stage1 training
- `_infer_traj_branch_condition(...)` 中直接调用 `model.forward_speed_energy_eval(...)` 的 fallback
- inference 末尾 `speed_energy_scores_raw / speed_energy_ref_scores_raw` 的 legacy fallback

### Config side

cleanup 后应移除或收敛的配置：

- `route_b.use_shared_stage1_path`

可保留但建议后续改名的配置：

- `route_b.shared_stage1_training_source`

原因：

- fallback 删除后，它不再是 “shared vs old path” 的选择开关
- 如果还保留 `clean / noisy` 两种训练源，后面更适合改名成
  `stage1_training_source`

## Cleanup Steps

### Step 1. Model cleanup

目标：

- 删除 legacy stage1 speed-energy modules
- 保留 shared semantic neck
- 不影响 guidance / alignment energy path

具体动作：

- 删除 legacy stage1 heads 和 query blocks
- 删除 `_compute_speed_energy_scores(...)`
- 删除 `forward_speed_energy(...)`
- 删除 `forward_speed_energy_eval(...)`
- 将 shared path 使用的 `speed_energy_route_proj` 改名到 shared 语义下
- 确认 shared path 不再依赖任何 `mode_out` stage1 模块

### Step 2. Policy cleanup

目标：

- train / infer 全部只走 shared stage1 path

具体动作：

- 删除 `self.use_shared_stage1_path`
- 删除所有 shared/legacy 双分支
- 删除 `_compute_stage1_speed_energy_loss(...)`
- 让 stage1 training 只保留 `_compute_shared_stage1_speed_energy_loss(...)`
- 让 `_infer_traj_branch_condition(...)` 不再回退到 `forward_speed_energy_eval(...)`
- 在 `conditional_sample(...)` 中强制：
  - pass1 先走 `forward_ego(..., return_intermediates=True)`
  - 再从 shared outputs 构造 stage1 raw scores
  - 再构造 branch condition
- 末尾 debug/query 输出统一来自 shared path

### Step 3. Config cleanup

目标：

- 去掉 fallback 概念，避免实验配置继续二义性

具体动作：

- 删除 `use_shared_stage1_path`
- 默认 shared path 成为唯一 stage1 path
- 视代码简化情况，决定是否把
  `shared_stage1_training_source`
  重命名成
  `stage1_training_source`

### Step 4. Verification

最少需要完成：

- `py_compile`
- shared-path synthetic smoke
- 一次带真实 stage1 label 的 train batch smoke
- 一次 infer smoke，确认：
  - `traj_window_condition_probs`
  - `traj_phase_condition_probs`
  - `traj_phase_energy_summary`
  - `traj_conflict_state_probs`
  - `lane_dir_relation_probs`
  都还能正常输出

## Recommended Execution Order

建议顺序：

1. 先删 policy fallback 分支，但先保留 model legacy API 壳子
2. 再删 model legacy stage1 score builder 与 old heads
3. 最后删 config 开关和无用 debug 兼容

这样做的好处是：

- 出错时更容易定位
- 不会一上来同时打断 train 和 infer

## Expected End State

cleanup 完成后，stage1/semantic 相关结构应当变成：

- main decoder path:
  - `forward_ego(...) -> traj_out / route_out / speed_out`
- shared semantic neck:
  - `traj_summary + route_summary + speed_summary + route_geom + conditioning`
- shared stage1 heads:
  - `window`
  - `phase`
  - `conflict_state`
  - `*_active`
  - old 7-point curve heads
- policy:
  - pass1 shared main path
  - semantic bottleneck build
  - pass2 shared main path

而不再存在：

- old `mode_out`-only stage1 path
- `forward_speed_energy_eval(...)` fallback
- `use_shared_stage1_path` 这种双路切换逻辑

