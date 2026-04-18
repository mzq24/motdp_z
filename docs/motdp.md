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
