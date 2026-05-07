# MoT-DP Context

## Scope

`MoT-DP` 这条线主要负责：

- 模型结构修改
- policy / decoder / route-b 逻辑修改
- dataset 与 labeling 管线
- 本地 smoke test
- HPC training / resume / full training
- training 后调参、复盘和结论沉淀

## 当前技术主线

- 基座仍是 `TransFuser backbone + DiT planner`。
- 近期主开发线以 `Route B / annealed energy guidance` 为核心。
- 训练和推理都高度依赖轨迹归一化统计、route/global stats、anchor 文件与 config 对齐。
- `MoT-DP` 里的稳定结论优先沉淀在这里，不再散落在 session 对话里。

命名上需要注意：

- `Route B` 更适合指当前已经实现的主线 planner
- `Stage1+` 更适合指 route-progress consistency / stronger semantic control
  这类下一步扩展
- `Stage2` 更适合指更后面的 semantic refinement / relation decoupling

所以：

- 不是所有 branch / progress / multi-step 设想都应该直接叫成 `Route B`
- 当前已经落地的是 hierarchical branch-conditioned Route B
- `traj_from_route_progress` 仍然属于 Stage1+ 方向，不是当前 Route B 训练主链

## 常用基础设施

- `new_hpc` 常用 repo 路径：`/workspace1/z_project/code/motdp_z`
- `new_hpc` 常用 repo alias：`/home/z/code/motdp_z`（link 到同一份代码）
- `new_hpc` 常用环境名：`z_dpauto`
- 如果后续要核对 `new_hpc` 上的 labeling / video / training 行为，默认先按这两个值找代码和环境。
- 如果后续要核对 stage1 full relabeling 产物、shard summary、scene split 输出：
  - 先看 `./reference/hpc_dataset_guide.md`
  - 不要默认先扫大 `samples_packed.pkl`

## 当前重点方向

### 0. 2026-04-29 Direction Turn: Independent 0423 Is The Close-Loop Baseline

Latest training analysis produced an important direction change:

- `independent 0423 best` was trained only to epoch 25, but currently appears
  to be the strongest close-loop candidate.
- Later joint/state/temp-occ models can improve selected offline metrics,
  especially 10-step `L2_1s` and junction recall.
- They still fail to match the independent model's `decision_go_recall` and
  `control_go_recall`, which seem more important for closed-loop decisiveness.
- Current conclusion: do not assume semantic state should be jointly diffused
  with trajectory. Treat independent state / phase prediction as a strong
  baseline and likely main direction.

Detailed metrics and reasoning are recorded in:

- `./tmp/independent_0423_close_loop_turning_point_20260429.md`

### 0.1 2026-05-01 Area / Window Decoupling Handoff

Window and global area/status heads should both be treated as always-defined
global classifications. `valid` masks should remain only for route-bin heatmaps,
timing/distance scalars, temporary occupancy bins, go-opportunity targets, and
boundary-speed labels whose supervision can be genuinely undefined.

Detailed model-session handoff:

- `./tmp/area_window_decoupled_valid_handoff_20260501.md`

### 1. Route B 主线

- 关键思路：
  - 用 `x_t / pred_x0` 统一命名
  - 采用 delta z-score 归一化
  - energy gradient 作用在 `pred_x0`，而不是直接打到 `x_{t-1}`
  - 训练侧用 unified 34-mode forward
- 关键文档：
  - `./route_b_refactor.md`
  - `./project_summary.md`
- 关键代码区域：
  - `../model/transformer_for_diffusion_multi_head.py`
  - `../policy/annealed_energy_guidance_policy.py`
  - `../training/train_carla_bev.py`

### 2. Semantic Behavior Labeling

- 目标：
  - 给 anchor 轨迹打可解释的行为/允许性标签
  - 支持训练辅助监督、推理过滤、anchor 质量分析
- 当前状态：
  - labeler 与 dataset on-the-fly 接入已完成
  - behavior embedding / loss / inference filtering 仍未完全成为主线稳定能力
- 关键文档：
  - `./semantic_behavior_labeling.md`
- 关键代码区域：
  - `../tools/anchor_semantic_labeler.py`
  - `../dataset/unified_carla_dataset.py`
  - `../training/train_carla_bev.py`

### 3. LiDAR BEV 输入升级

- 当前建议：
  - 先把 `transfuser_lidar_bev` 作为输入/细节增强支路接入
  - 先做 `BEV only` vs `BEV + LiDAR BEV` 的清晰消融
  - 暂时不要和 semantic-condition 分支混在同一轮里
- 关键文档：
  - `./detail_sampling_upgrade.md`
  - `./session_0401_lidar_bev_followup.md`

### 4. 训练效率与调参

- `train_energy=false` 是重要的快速训练开关，适合先跑 diffusion-only 版本。
- 调参时需要特别注意：
  - `feature_suffix`
  - `n_emb`
  - `speed_loss_weight`
  - `train_energy`
  - `route_abs_stats_path`
  - `global_abs_stats_path` / `abs_stats_path`
- 本地 mini 验证与 HPC full-dataset 指标不能直接混用比较。

### 5. Stage1 Speed Merge Labels

- `stage1 speed curve energy` 当前已经显式计算 merge 相关速度阈值：
  - `v_go_need_mps`：ego 想在当前 merge actor 前面通过冲突点时所需的最小速度
  - `v_yield_max_mps` / `yld`：ego 想让在当前 merge actor 后面时允许的最大速度
- 这里的 `yld` 不是“下一整个 merge window”的完整定义，更准确地说，它是“针对当前关键 merge actor 的 yield-behind 速度上界”。
- 当前状态：
  - 这些量已存在于 `scripts/data_tools/precompute_semantic_labels.py` 的 stage1 speed labeling / debug 中
  - 但还没有正式进入 dataset / training 主链作为显式 supervision 或 condition
- 当前建议：
  - 若后续要主线化，先把 `merge_valid / v_yield_max / v_go_need / v_behind_min` 做成稳定 label
  - 再优先接到 `predict speed`
  - 最后再考虑把预测到的 merge-affordance 反喂给 `predict traj`
- 2026-04-14 讨论后的更具体建议：
  - 当前阶段先把 stage1 speed-energy 主线化为：
    - `chase`
    - `merge_yld / merge_go + merge_active`
    - `junction_yld / junction_go + junction_active`
    - `borrow_yld / borrow_go + borrow_active`
    - `pedestrian`
  - 这里的 `merge / junction / borrow` 更适合作为 window-level speed-affordance /
    branch-energy 监督
  - 当前更倾向直接做 condition predict，但 condition 形式改为 hierarchical
    soft condition，而不是硬 token：
    - `window = [none, merge, junction, borrow]`
    - `phase = [yld, go]`
    - `borrow_cross_active_time_s`
  - 训练上先用 GT soft condition，后面再逐步混入 model infer 的 condition
  - predictor 侧先直接把 soft condition 强注入 `traj queries / mode embedding`
  - `borrow_cross_active_time_s` 主要用于补偿 borrow 过程中 causal BG
    vehicle 超出感知范围时的条件信息
  - 如果后续仍然存在明显的 speed/traj 不一致，再考虑额外引入
    route-progress consistency 作为 stage1+ 扩展
  - 详细讨论记录在：
    - `./brainstorm/route_b_stage1plus_stage2_brainstorm_20260406.md`
- `junction_left_cross_meet` 这条线也要按 decomposition 来理解：
  - 不要只把它当一个 folded `meet_risk`
  - 应该拆成：
    - `speed_risk_junction_cross_yld_values`
    - `speed_risk_junction_cross_go_values`
  - 但要注意：
    - 它和 `borrow_cross_meet` 只是时序 / 决策语义相似
    - 几何来源并不相同
    - `junction_left_cross_meet` 不使用 two-way corridor
- 现行实现说明：
  - `cross / borrow corridor / junction-cross split / merge episode / merge speed curve` 的当前主线逻辑，统一记录在
    `./reference/cross_meet_corridor_logic.md`
  - `2026-04-14` 之后，merge 这条线还额外记录了“下一次 full relabeling 的 agreed direction”：
    - 旧 packed merge cover 不能直接信
    - merge window 要按明确 start/end 定义
    - `hold` 与 `yld/go` 分层
    - merge area 更偏向 route-based 定义
  - 当前代码主线也已经开始按这个方向实现：
    - merge cover 会额外保留 scene-route conflict progress
    - merge motion 会保留 ego 的 scene-route center/front/rear progress
    - merge parser 正在转向 route-based `start -> merge_area -> go -> end`
    - merge area 的起点现在直接等于第一批 future conflict point，不再往前加 pre-margin
    - actor id 更偏向 debug，不再作为 merge-go 的唯一锚点
  - `borrow_cross_meet` 这条线目前也有了更明确的分层方向：
    - blocker/context-frame/corridor 定位已经相对稳定
    - `v_yield_max / v_go_min` 主要是时序计算问题
    - 后续要重点主线化的是 route-based borrow window `start/end`
    - `yld/go` 与 `borrow_cross_active` 需要分层定义
    - 现在还新增了 `borrow_cross_active_time_s`，从 active start 开始累计，用来补偿对向车超出感知范围时的条件信息
  - `junction_left_cross_meet` 现在也开始有显式 episode：
    - 不再只看单帧 cover
    - 会累计 multi-frame collision/conflict point history
    - 用局部 conflict area 定义 `junction_cross_active`
    - `start` 取 area 内 ego route progress 最小的那一帧
    - `end` 取 ego 离开 conflict area 的那一帧
  - `merge/junction` 的 hard episode start 现在还有一个很窄的
    current-chase gate：
    - `current_cover.subtype == follow_chase`
    - `other_speed <= 0.5`
    - `route_distance_m <= 15`
    - 只影响 future-driven episode 的 **start**
    - 不影响已经开始的 episode 延续
  - `2026-04-19` 的一组 spot check 也说明这个 gate 里至少有一类是
    明确的“预期行为”，不该当成异常：
    - `merge_start_blocked_by_current_follow_chase`
      在
      `Town12_Rep0_26_0_route0_11_08_18_12_42`
      `Town12_Rep0_866_0_route0_11_08_23_43_01`
      `Town13_Rep0_1157_1_route0_11_08_23_47_31`
      这 3 条里都出现在 borrow 已经结束之后
    - 此时 ego 前方已经进入 junction / chase 语义
    - 所以 future-merge start 被 current follow-chase gate 挡住是合理的
    - 这类 case 当前应记为“gate 正常工作”，不是 merge regression
  - 另外，最近对 `merge / junction / borrow` 的 candidate split 也有了新的
    认知：
    - 当前 `_interaction_signal_from_candidate(...)` 仍然 heavily 依赖
      local heading-angle 分类：
      - `<= 45 deg` 视为 same-direction
      - `>= 70 deg` 视为 cross-direction
      - 中间角度仍会落到 merge-like 处理
    - 这意味着当前 `merge` 不是纯粹的 same-direction family
    - `borrow_cross_meet` 的 candidate subtype 也仍然会先经过 cross-direction
      路由
    - 更合理的长期方向是：
      - long borrow 直接基于
        `event_name + scene_borrow_context + corridor`
      - auxiliary `direction` 不再用当前局部 heading 去定义
      - 而改成基于 conflict-area 的 approach direction：
    - `borrow`: corridor
    - `merge`: merge area
    - `junction`: conflict circle / local conflict area
      - 也就是比较 ego 与 bg actor **接近同一 conflict area 的方向**
        ，而不是比较它们在当前帧局部的 yaw
    - 对 `merge` 本身，当前也进一步收敛出一个更稳的 direction 定义：
      - merge area 继续由 future conflict points 聚出来
      - area `start` / `end(+3m)` 目前整体可接受
      - 真正该改的是 direction 的取法
      - merge direction 更适合取自 `merge_area_end` 之后的 downstream
        route heading，而不是 merge 斜线段里的局部 heading
      - 当前更倾向：
        - 默认看 `area_end -> area_end + 5m`
        - route 不够长时 fallback 到 `+3m`
    - 对 `junction` 也有类似的收敛：
      - cover / conflict point 仍然是 evidence
      - 但最终应该落到显式 conflict area `start/end`
      - junction 的 direction 也应基于 conflict area 来定义
      - `junction + left` 更适合作为 scene prior / validation signal
        ，不该当成唯一的 hard 条件
      - 因为 raw `junction` 和 `command == LEFT` 在时间上可能有轻微滞后
      - 同时，`junction + left` 也可能出现 merge-like downstream lane
        competition，例如 ego 左转 vs 对向右转并入同一出口 lane
  - 当前代码里也开始接一个统一 `conflict_area` 骨架层：
    - 直接从新的 route-level area/window 生成 unified conflict window
    - 不再直接继承旧 `borrow / merge / junction episode` 的 active
    - 旧 episode 先保留作 baseline / 回归 / 对照
    - 先提供统一的：
      - `family`
      - `dir`
      - `active`
      - `start/end`
    - 其中当前的粗 direction 临时映射为：
      - `borrow -> opposite`
      - `merge -> same`
      - `junction -> none`
    - 如果某个 family 已经 active，但缺 area/source 元数据：
      - 当前先记录 issue
      - 不在这层直接 hard crash
    - 这层是骨架，不是最终 conflict-area 语义：
    - `borrow conflict area` 后面还要从 corridor 里再提纯
    - `merge / junction direction` 后面还要改成 area-based approach direction
  - `2026-04-27` 的 conflict-area route-mask 同步记录：
    - `scripts/data_tools/precompute_semantic_labels.py` 现在会离线写出
      `conflict_area_route_mask` / `conflict_area_route_mask_valid`
    - mask 是 route-token aligned，默认 20 点，对应当前 stage1 route tokens
    - 这不是 frame-window 里的 `start_frame:end_frame` 全段置 1
    - 它由当前 family 的 conflict-area local interval 投影到 route token
      arclength 后得到
    - `borrow / merge / junction` 都走同一套 route-mask 写法
    - 所以 borrow 也不再默认 20 个点全写成 area
    - `dataset/unified_carla_dataset.py` 已透传这两个字段
    - 老 packed 如果没有该字段，dataset 会给
      `conflict_area_route_mask_valid = -1`
    - policy 看到 valid 为未知时才 fallback 到旧 frame-window target
    - main worktree 与 `MoT-DP-joint-state-speed` worktree 的
      `precompute_semantic_labels.py` / `unified_carla_dataset.py`
      已同步
    - 后续 labeling session 若重跑 label，应优先检查：
      - merge / junction 的 route-mask positive count 不应大面积 20/20
      - borrow 也应由 corridor/area interval 决定 mask span
      - video/debug 里应能区分 `window` 长度和真正 `conflict_area_route_mask`
  - `2026-05-05` 新增 phase-object binding 后处理：
    - `scripts/data_tools/postprocess_stage1_phase_object_binding.py`
    - 目标是把 area/ego-based `conflict_decision_phase` 对齐到当前 phase
      对应的 object / opening
    - `current_cover` 在 active conflict window 内通常视作
      `current_area_actor`；但 merge 会额外检查 current cover 是否仍在
      merge area 内，已经离开 area 的 front/chase cover 不再绑定 role3
    - `future_cover` 只有在 `frame_index <= 6` 且距离 gate `<=20m`
      时才绑定成 next actor；否则 `phase=go` 可写成
      `open_unbounded`
    - 这里的 `role` 是 ego phase 和 object/opening 的 relation 属性，
      不是 actor 的静态类别；更接近 GNN edge attribute
    - boundary binding 也写 relation role/mode：
      - `future_actor_boundary`: old future-cover yld/go speed boundary
      - `current_clear_transition`: future actor 已变 current cover，旧
        go-before boundary 不再成立，但 relation 仍绑定 current actor
      - `open_unbounded`: 无 future cover，speed boundary 无约束且无 actor
    - boundary relation 保留旧 threshold scalar 的真实来源，不再为了
      match 而强行改写成 `phase_ref`
    - 若 actor-conditioned scalar boundary 和 post-hoc `phase_ref` 不一致，
      `actor_match=0` 且 `scalar_loss_valid=0`
    - 主要字段：
      `conflict_phase_ref_role`,
      `conflict_phase_ref_actor_id`,
      `conflict_phase_boundary_ref_role`,
      `conflict_phase_boundary_mode`,
      `conflict_phase_boundary_state_valid`,
      `conflict_phase_boundary_object_missing`,
      `conflict_phase_boundary_scalar_loss_valid`,
      `conflict_phase_boundary_ref_actor_id`,
      `conflict_phase_boundary_actor_match`
    - `object_missing=1` 表示 phase 仍 active，但没有 bbox reference 可构建
      actor-conditioned boundary；这种样本不要用 fallback 的
      `yld=30/go_min=0` 当 hard scalar supervision
    - `scalar_loss_valid=0` 用于 mask 掉 missing-object/open-unbounded 的
      boundary scalar loss
    - 重要语义边界：
      - 原始 `conflict_decision_phase` 仍然主要是 ego-state / area-state
        label，不是原生 object-anchored phase
      - `phase_ref` 是后处理根据 current/future cover 对 phase 做出的
        relation 解释
      - 因此 `phase_ref=current_area_actor` 不代表原始 phase 生成时已经
        绑定该 actor
      - `open_unbounded` / unconstrained boundary 常对应 raw fallback
        `yld=30/go_min=0`，只能 debug，不能监督 scalar loss
      - 真正危险的是 actor-conditioned scalar boundary 与 post-hoc
        phase_ref 不一致；这类现在通过
        `actor_match=0` 和 `scalar_loss_valid=0` mask 掉
      - merge mixed state 的例外：
        当 current cover 仍在 merge area 内、且 gated future cover 已出现，
        phase-object binding 优先绑定到 future cover，因为 yld/go scalar
        boundary 本来就是 future actor-conditioned
      - 该 mixed state 中，`phase=yld` 绑定 role1
        `yld_target_actor`，`phase=go` 绑定 role2
        `go_before_next_actor`
      - 一类预期正常 mismatch：
        `merge + phase=yld + phase_ref=none + boundary_ref=open_unbounded`
        且 `current_cover_outside_area + no_future_cover`
      - 同类还有：
        `junction + phase=yld + phase_ref=none + boundary_ref=open_unbounded`
        且 `missing_current_cover + no_future_cover`
      - 这表示 current cover 已离开 merge area，next future cover 尚不可用，
        或 junction 当前帧没有可见/可 gate 的 current/future actor
      - 这两类本质都是 object-free / open-unbounded boundary：
        没有 actor-conditioned `yld/go` boundary，raw fallback 是全范围
        `go_min=0/yld_max=30`
      - 此时 speed 解释应回到独立 speed constraints：
        `vchase` 和 `merge_follow_through_vbmin`
      - 这种情况下的 `phase=yld` 可能只是 ego-state 减速标签，
        原因可能是 `vchase`，不一定是 yield-next-actor
      - consistency audit 里，`Town12_Rep0_1105_0... frame 59` 这类
        单帧 `go_under_min_no_accel`，以及类似的单帧
        `junction_go_over_max_no_decel`，当前归为危险但不算 label bug
      - route-level issue 需要连续至少 2 帧；单帧只保留 debug
      - `yld_over_max_no_decel` 若满足 merge/future actor 对齐且速度主要
        贴近 `vchase`，归为 `candidate_chase_limited_go`，不是
        yld-boundary 错误
      - shard01 的 borrow/merge consistency 复查增加两类 benign cause：
        `candidate_yield_after_actor_no_decel_needed` 和
        borrow `yld_over_max_no_decel` local blip
      - `candidate_yield_after_actor_no_decel_needed` 代表：
        `phase=go` 但 `go_min > 20m/s` 等明显说明不应该抢前，
        ego 更可能是在 actor 后方通过；此时 `yld_max` 往往更贴近当前
        车流速度
      - borrow `yld_over_max_no_decel` 若处于长期减速到 0 的趋势中，
        连续 2 帧局部减速幅度不足也先归为 debug-only，不当 label bug
      - full-dataset / new_hpc issue review 的顺序：
        先 skip collision，再识别 phase-cause/relation-cause，再过滤单帧
        与 borrow slowdown blip，最后只看 remaining non-chase consecutive
        issue；`chase_over_max_no_decel` 单独成桶看严格度
  - `2026-05-06` 新增 merge follow-through vbmin 后处理：
    - `scripts/data_tools/postprocess_stage1_merge_follow_through_vbmin.py`
    - 只用于 `merge`
    - 不改旧 `merge_go_min_speed`；旧字段仍表示 entry 前
      future-cover go-before boundary
    - 新字段在 ego 进入 conflict area 后、出 area 前持续写入最近一次稳定
      merge traffic `vbmin`，避免 entry 后速度下界突然消失
    - 当前字段：
      `merge_follow_through_vbmin`,
      `merge_follow_through_vbmin_valid`,
      `merge_follow_through_vbmin_actor_id`,
      `merge_follow_through_vbmin_actor_valid`
  - `2026-05-07` boundary speed naming cleanup debt:
    - 当前有效语义应按 phase-specific speed interval 理解：
      - `yld phase`: lower `0`, upper `yld_max`
      - `go phase`: lower `go_min`
      - `go phase` 的 upper 依 family 而定
    - `merge go upper` 当前来自 `chase_speed_max` / front-following cap
    - `junction go upper` 在 `phase=go + role=3/current_area_actor` 时使用
      `junction_yld_max_speed`
    - 因此 `junction_yld_max_speed` 已经不是纯 yld-only 名字，
      更准确是当前 conflict/current-area actor 的 safe upper speed
    - 这属于 naming cleanup，不急于改 raw 字段名
    - 后续 label 稳定后，可以新增中性 derived fields：
      `phase_speed_lower_mps`, `phase_speed_upper_mps`,
      `phase_speed_lower_valid`, `phase_speed_upper_valid`,
      `phase_speed_upper_source`
    - 在此之前，video / consistency debug 优先用 `upper` 这种中性显示，
      避免 `go phase` 下仍显示 `yld_max` 造成语义混淆
  - 当前更推荐的下一步顺序不是马上设计 energy 数值，而是：
    1. 先 rerun 新的 unified `conflict_area` 框架
    2. 先看：
       - `active_scenes_by_family`
       - `active_samples_by_family`
       - `issue_scenes_by_family`
       - `missing_reason_counts`
       - event-level purity / coverage
    3. 再看 old-vs-new delta：
       - 哪些 scene 变了
       - 为什么变
       - 是否正好是我们想改的 scene
    4. 最后再设计 `yld / go / energy`
  - 这里的设计边界也要保持清楚：
    - `family / dir / active / start / end`
      - 解决 conflict localization
    - `yld / go`
      - 解决 local conflict-area 内的 speed-phase choice
    - `energy`
      - 解决这个 speed-phase choice 的数值化表达
  - 所以当前共识是：
    - 先把 window/family/dir 框架跑稳
    - 再决定 `yld/go` 到底要监督什么
    - 最后再讨论数值形式、normalization 和 closed-loop smoothing

## 推荐工作流

1. 在 `MoT-DP` 内完成模型或 labeling 修改。
2. 先做本地 smoke test，确认 forward / loss / config / checkpoint 加载没有明显问题。
3. 再同步到 HPC 跑正式 training 或 resume。
4. 训练结束后，把真正稳定的结论回写到这里，而不是只留在聊天记录里。

## 新 session 读到这里后默认要知道的事

- 如果本次任务是“改模型、改标签、跑训练、看 loss、调 config”，默认归 `MoT-DP` 线。
- 如果本次任务同时涉及 close-loop，不要把 Bench2Drive 运行细节也塞进这里，转去看 `./bench2drive.md`。
- `semantic` 线与 `lidar BEV` 线要分开做实验，避免归因混乱。

## 继续深入时优先看的文档

- `./project_summary.md`
- `./route_b_refactor.md`
- `./semantic_behavior_labeling.md`
- `./detail_sampling_upgrade.md`
- `./session_0401_lidar_bev_followup.md`
