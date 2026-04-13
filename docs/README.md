# Context Index

这套文档用于替代零散 session 会话，目标是让新 session 只看少量固定入口就能接上上下文。

## Canonical Entry Points

- `../.claude/codex_memory.md`
  - 放最精简的长期上下文、任务边界、阅读顺序
- `./motdp.md`
  - 放 `MoT-DP` 主线：模型、labeling、本地 smoke、HPC training、调参
- `./bench2drive.md`
  - 放 `Bench2Drive` 主线：close-loop、本地/HPC 测试、worktree、结果分析

## 如何选择文档

- 任务是模型、训练、标签、数据、策略修改：
  - 先看 `./motdp.md`
- 任务是 closed-loop、agent、CARLA、leaderboard、worktree、测试结果复盘：
  - 先看 `./bench2drive.md`

## 深入参考文档

- 项目总览：
  - `./project_summary.md`
- Route B 架构与推理/训练约定：
  - `./route_b_refactor.md`
- Semantic behavior labeling：
  - `./semantic_behavior_labeling.md`
- Stage1 speed cross / merge / corridor logic：
  - `./reference/cross_meet_corridor_logic.md`
  - 其中也记录了 `junction_left_cross_meet -> junction_cross_yld/go` 的拆分约定
- Detail sampling / LiDAR BEV 相关：
  - `./detail_sampling_upgrade.md`
  - `./session_0401_lidar_bev_followup.md`
- Close-loop / HPC 操作参考：
  - `./closedloop_agent_plan.md`
  - `./reference/hpc_new_deploy_guide.md`
- 历史会话摘要：
  - `./session_0330_summary.md`
  - `./session_0331_summary.md`

## 文档落点规则

- 稳定共识和长期上下文：写进 `motdp.md` / `bench2drive.md`
- 临时 debug 和一次性排查：写进 `./tmp/`
- 脑暴、未定设计：写进 `./brainstorm/`
- 过时但仍可能有参考价值的内容：写进 `./archive/`

## 给未来 session 的一句话

如果要快速接上上下文，先读 `../.claude/codex_memory.md`，再读对应主题文档；不要先从旧 `session_*` 文档开始。
