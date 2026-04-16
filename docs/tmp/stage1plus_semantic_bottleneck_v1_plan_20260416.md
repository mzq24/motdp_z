# Stage1+ V1 Plan: Semantic Bottleneck + Step-Dependent Hierarchical Conditioning

Date: 2026-04-16

## 1. 定位

这是一个 **Stage1+ 主体方向**计划，用来优先验证两件事：

- 现在这套 semantic label 是否真的有信息量
- model 是否真的能学到并利用这些 semantic 信息

这份计划的目标不是一开始就把所有细节做精，而是先做一个能明显看出方向是否成立的版本。

当前 Route B 已实现主线仍然是：

- hierarchical branch-conditioned planner
- `window = [none, merge, junction, borrow]`
- `phase = [yld, go]`
- `borrow_cross_active_time_s`
- `traj_direct + route_pred + speed_profile_head`

这份计划优先解决的问题是：

- 当前 static branch condition 还不够体现 multi-step semantic refinement
- 还没有显式的 planner bottleneck 去强制 model 学习 compact semantic summary

## 2. 核心思想

先做一个 **显式 semantic bottleneck**，再按 diffusion step 分层注入。

这个 bottleneck 是 planner 真正使用的 compact control state，而不是让模型直接依赖整坨 scene feature。

第一版的 **learned semantic bottleneck** 由这些量组成：

- `window = [none, merge, junction, borrow]`
- `phase = [yld, go]`

另外保留一个 **observable side input**：

- `borrow_cross_active_time_s`

注意：

- `borrow_cross_active_time_s` 不是可预测语义
- 它是 agent / inference 时可直接维护的观测量
- 所以不应被当成“模型需要通过 bottleneck 学会预测”的对象

这里仍然沿用当前 label / model 已有的语义体系，不额外引入新的事件家族。

## 3. V1 范围

V1 一次性实现这几个强耦合模块：

### 3.1 显式 semantic bottleneck

把当前的 hierarchical condition 从“一个普通 condition 向量”升级成“planner 必经的 compact semantic summary”。

要求：

- 单独建 `window` projector
- 单独建 `phase` projector
- 单独建 `borrow_time` side-input projector
- `window + phase` 组成 learned semantic bottleneck
- `borrow_time` 只作为外部观测侧输入参与后期调制，不算 learned bottleneck 的一部分

### 3.2 Step-dependent hierarchical conditioning

按照 diffusion step 对不同层级语义做 coarse-to-fine 注入。

建议使用 **soft schedule**，不要用 hard if-else。

按归一化 timestep `tau` 来控制：

- 早期高噪声 step：
  - `window` 最强
- 中期：
  - `phase` 逐渐增强
- 后期低噪声 step：
  - `phase branch-energy summary` 才明显增强
  - `borrow_time` 只作为 borrow-window 下的 observable late-step side input

第一版不引入新的 `progress` 语义输入，只先把现有 hierarchical label 吃透。

这里的第三阶段不再理解成“学习 borrow_time”，而是：

- 学习一个更细的 `phase energy / branch-energy` 语义
- 例如当前参考速度附近的 `yld/go` affinity
- 用来在 late denoise step 中细化 `go -> yld` 这类模式选择
- 第一版优先保留双通道 summary，而不是压成单个 margin

### 3.3 强条件注入

不要只做弱 modulation。

第一版要求：

- semantic bottleneck embedding 直接注入 `traj queries`
- 同时注入 `mode embedding`

目的是让 `go/yld` 条件变化时，traj 能明显变，而不是只产生很小扰动。

### 3.4 两段式条件使用

训练：

- 先用 GT hierarchical condition
- 第一版先不混 infer condition，先看 GT semantic signal 能不能学进去
- 训练时仍然是当前的一次随机 `t` forward，不做每个 denoise step 的显式双 pass

推理：

- 继续使用两段式，但这是 **两个完整的轨迹预测 pass**
- 不是“每一个 denoise step 都先跑一个 base traj 再跑一个 refine traj”

推荐推理方式：

1. **pass1**
   - 用当前较弱/空 semantic condition 跑完整 denoise 链
   - 得到初始 `traj + route + stage1 semantic outputs`
2. 从 pass1 结果中提取：
   - `window probs`
   - `phase probs`
   - `phase branch-energy summary`
   - `borrow_time` 由 agent 侧直接提供，不由模型预测
3. **pass2**
   - 再跑一遍完整 denoise 链
   - 但这次每个 denoise step 都按 soft schedule 注入不同层级的 semantic info

所以：

- 是 **two-pass inference**
- 不是 **per-step two-pass**
- semantic schedule 发生在 pass2 内部的每个 denoise step 上

## 4. 第一版明确不做

为了先验证主体方向，以下内容暂时不进这个 v1：

- `traj_from_route_progress`
- `progress_profile_head`
- route-progress consistency
- branch-energy consistency
- `go/yld` counterfactual separation loss
- previous-step semantic memory / explicit state feedback

这些都保留给下一阶段。

其中：

- `traj_from_route_progress` 的细节优化计划，单独保留在：
  - `docs/tmp/stage1plus_progress_v1_plan_20260416.md`

## 5. 实现设计

### 5.1 Condition 表达

训练和推理都统一使用 hierarchical 表达：

- `window_probs`: `(B, 4)`
- `phase_probs`: `(B, 2)`
- `phase_energy_summary`: small semantic summary derived from branch energies
- `borrow_time`: `(B,)`

其中：

- train 侧来自 GT label
- infer 侧来自 stage1 head 的 first-pass 结果
- `borrow_time` 始终来自外部可观测状态，不作为预测目标

重要约束：

- `phase` 和 `phase_energy_summary` 只有在对应 `window` active 时才有语义
- `window inactive` 时，允许 `E_yld / E_go` 在网络前向里仍然有数值
- 但这些数值不能直接进入有效 semantic bottleneck，也不能直接驱动 pass2 的强条件
- 也就是说，`window` 必须先决定“这类 phase 语义有没有资格被使用”
- `phase / phase-energy` 是 `window` 下面的 refinement，不是独立 family

`phase_energy_summary` 的推荐第一版定义：

- 基于当前参考速度 `v_ref`
- 从 `yld/go` branch curve 提取一个小 summary
- 例如：
  - `aff_yld = 1 - E_yld(v_ref)`
  - `aff_go = 1 - E_go(v_ref)`
- 第一版优先直接保留两维 `aff_yld / aff_go`
- 不优先使用单个 margin，避免把“两边都高”和“两边都低”压成同一种语义

它的作用是：

- 让第三阶段的语义细化不只是“知道当前是 yld/go”
- 还能知道当前 branch 的强弱和置信度
- 也尽量保留 speed curve 的双模态结构

这里明确不采用的做法：

- 不把 `E_merge_yld / E_merge_go`、`E_junction_yld / E_junction_go`、`E_borrow_yld / E_borrow_go` 先用 `min` 折成单个 family summary 再进 bottleneck
- 原因是 `yld` 和 `go` 对 planner 来说是两种模态，不是一个连续 family 值的细小扰动
- `min` 可以继续作为 speed selection 或 family-level coarse energy 的聚合方式
- 但不适合作为 semantic bottleneck 的主体表示

### 5.2 Step-dependent schedule

给 `window / phase / phase_energy / borrow_time` 四部分各自定义 step gate。

建议按 `tau = timestep / train_max_timesteps` 实现。

期望趋势：

- `gate_window(tau)`：
  - 随 `tau` 增大而增强
  - 早期 strongest，后期仍保留弱影响
- `gate_phase(tau)`：
  - 中后期增强
  - 早期弱
- `gate_phase_energy(tau)`：
  - 主要在后期增强
  - 用于做 finer phase refinement
- `gate_borrow_time(tau)`：
  - 主要在后期生效
  - 只在 borrow window 下使用

实现上可以用简单的连续函数：

- linear ramp
- cosine ramp
- sigmoid ramp

第一版优先简单稳定，不追求太复杂。

### 5.3 注入位置

semantic bottleneck 经过投影后：

- 加到 `traj queries`
- 加到 `mode embedding`

不要只走一个弱小的 global conditioning 支路。

### 5.4 日志与调试输出

为了快速判断 label 是否有信息，第一版必须增加可观测量：

- `traj_window_condition_probs`
- `traj_phase_condition_probs`
- `traj_phase_energy_summary`
- `traj_borrow_time_condition`
- 每种 gate 的当前数值

最好再导出：

- pass1 traj
- pass2 traj

方便做同场景对比。

### 5.5 基于 closed-loop 样例的当前观察

下面这些结论来自最近查看的 closed-loop case，用来指导 v1 的实现重点：

- `merge` 的 energy 形状整体上是有语义的
- 但 `merge_yld` 和 `merge_go` 不适合先用 `min` 折成单个 family 表示再喂给 bottleneck
- 原因是这两条对 planner 来说对应两种模态，而不是一个连续值的小扰动

- `junction_active` 目前学得还可以
- 但 `junction_yld / junction_go` 的 raw energy 往往都偏高
- 这说明第三阶段更适合吃相对模态信息或归一化 summary，而不是直接吃 raw energy 绝对值

- `borrow_active` 目前能学到一部分
- 但更像只在对向车还在感知范围内时，模型才容易判断为 `borrow`
- 当关键对向车已经超出感知范围时，模型容易把对向车道看成“空 lane”，从而把 `borrow` 误判成 `none`

- `borrow_time` 的问题更像是 **推理计时初始化** 问题，不是 training active 标注问题
- training 侧的 `borrow active` 本来就是按整段 scene 遍历得到的，语义上已经是完整 episode
- 真正需要回溯修正的是 inference 时 `borrow_time` 的起始点，而不是 training label

这组观察的设计含义是：

- `borrow_time` 只能作为 borrow 已经激活后的补偿 side input
- 它不能负责启动 `borrow` 假设，也不能解决 borrow 的 cold start
- `borrow_active` 的学习很可能需要比当前更长的时间历史，或者更强的时序 summary / memory
- 因此后续如果 `borrow` 仍然是短板，优先考虑的是增强 `window` 级别的 long-term temporal reasoning，而不是继续堆叠 phase 细节

- `merge` 和 `borrow` 的 route 形状经常比较接近
- 两者更关键的区别在于：目标并入车道是 **同向** 还是 **对向**
- 因此适合加一个只在 `merge/borrow` 场景下启用的辅助语义：
  - `lane_dir_relation = {same_direction, opposite_direction}`
- 对这个辅助语义来说：
  - `merge -> same_direction`
  - `borrow -> opposite_direction`
- 这个辅助语义既可以帮助区分 `merge vs borrow`
- 也可以作为后续 `window` 判定的更紧凑中间变量

### 5.6 Borrow time 回填与 merge/borrow 方向辅助

#### 5.6.1 Borrow time 的因果回填

这里的“回溯”只指 inference 里的 timer 初始化，不指 training label。

推荐做法：

- online 维护一个 `borrow_candidate_start_time`
- 当 ego 第一次进入 borrow-corridor 候选区域，或者第一次达到关键对向道 position 时，先记录这个时间
- 如果后面若干帧确认 `borrow_active` 成立
- 则把当前 `borrow_time` 初始化为：
  - `now - borrow_candidate_start_time`
- 而不是从确认帧开始记 `0`

这样做的效果是：

- 不需要修改 training active label
- 但可以减少 inference 时 `borrow_time` 首帧偏小的问题
- 依然保持 causal，不需要真的在线回改过去已经输出过的帧

#### 5.6.2 Merge/Borrow 方向辅助头

增加一个只在 `merge` 或 `borrow` 相关场景下训练的辅助头：

- `lane_dir_relation = {same_direction, opposite_direction}`

监督规则：

- `merge` 样本 -> `same_direction`
- `borrow` 样本 -> `opposite_direction`
- 其他 window 下可直接 mask，不参与这个辅助 loss

这个辅助头的目的不是替代 `window`：

- 而是帮助把“几何上看起来都像换道”的 `merge/borrow` 分开
- 因为二者 route 形状可能接近，但并入车道方向相反

关于上一帧预测的使用：

- 上一帧的 `same/opposite` 预测可以作为下一帧 `merge/borrow active` 判定的 causal side state
- 但第一版更建议把它放在 inference / postprocess 逻辑里使用
- 暂时不要直接作为训练时网络输入

原因：

- 如果直接把上一帧预测当成训练输入，会很快引入 train/infer mismatch
- 第一版先把它作为一个 agent 侧或后处理侧的时序平滑信号更稳
- 等主体方向验证成立后，再考虑是否把这个 side state 显式喂回模型

## 6. 验证目标

这版最重要的不是最后闭环最好，而是能快速回答下面几个问题：

### 6.1 label 是否有信号

看：

- `merge / junction / borrow` 的 condition 是否稳定可解释
- `phase = yld/go` 是否和场景语义对得上
- `window inactive` 时，phase-energy 是否被正确 gate 掉，而不是直接污染 semantic control
- `borrow` 失败样例是否显著集中在 causal oncoming vehicle 已超出感知范围的 case

### 6.2 model 是否真的在用 semantic bottleneck

看：

- 同场景下切换 `phase`
- traj / speed trend 是否明显变化
- `merge / junction / borrow` 是否真的先决定模态家族，再由 `yld/go` 做 family 内 refinement

如果几乎没变化，说明 bottleneck 太弱或被大特征绕开了。

### 6.3 multi-step denoise 是否体现 coarse-to-fine

看：

- 早期 step 是否更依赖 `window`
- 中后期是否开始体现 `phase`
- 后期是否开始体现 `phase_energy`
- `borrow_time` 是否只在 borrow window 且后期影响明显
- `borrow window` 的建立是否仍然受限于短时感知，而没有真正利用更长时间历史

## 7. 测试

训练链路：

- 打开新 semantic bottleneck 后，forward/backward/AMP/DDP 应正常
- 关闭相关 config 时，行为应与当前版本一致

推理链路：

- 两段式推理仍能正常运行
- `predict_action` 输出结构不变

诊断链路：

- 同 scene / 同 noise 下，强制 `phase` 从 `go` 切到 `yld`
- 观察 pass2 的 traj 是否明显更保守

## 8. 下一步边界

如果这版方向成立，下一步再接：

- `traj_from_route_progress`
- branch-energy consistency
- `go/yld` counterfactual controllability
- previous-step semantic summary

也就是说：

- 这份计划负责先把 **主体方向** 立住
- progress 那份计划负责后续 **细节优化和结构稳固**
