# LEAD Planner Cached Dataset Validation Plan

## Summary

目标是在 official LEAD full 结果出来后，用 **LEAD planner + 我们的 cached BEV dataset** 做验证，判断 `/workspace2/z_project/motdp_bev_cache_navtest_official4cam*_npy` 是否足够对齐 official LEAD online pipeline。

计划落盘路径：`docs/lead_cached_dataset_validation_plan.md`。

本阶段只记录计划，不执行评测、不修改 agent、不启动新的 tmux job。

## Key Changes

- 先以正在跑的 official LEAD full 作为基准：
  - agent: LEAD repo 自己的 `carla_transfuser_agent`
  - scorer: LEAD vendored NAVSIM v1.1 `run_pdm_score.py`
  - tmux: `lead0145`
  - split: navtest full `12,146` scenarios
- 后续新增/修正一个 cached LEAD planner agent：
  - 输入读取 `cache_index.npz + bev_feature.npy + ego_status.npy`
  - planner 使用 LEAD checkpoint 的 planning decoder
  - 输出坐标转换与 official `CarlaTransfuserAgent` 保持一致：waypoint y flip、heading sign flip
  - 默认 `fallback=raise`，navtest full 不允许 missing token 静默走 constant velocity
- canonical cache 优先使用：
  - `/workspace2/z_project/motdp_bev_cache_navtest_official4cam_officialacc_npy`
  - 原因：official LEAD/DiffusionDrive/SparseDrive 都使用 NAVSIM dataclass 的 `ego_acceleration`
- 保留对照 cache：
  - `/workspace2/z_project/motdp_bev_cache_navtest_official4cam_npy`
  - 用于判断 finite-diff acceleration 与 official acceleration 是否影响 LEAD planner score/comfort

## Evaluation Plan

- Stage 1: 等 official LEAD full 完成，记录：
  - score
  - DAC / NC / TTC / EP / lane keeping
  - `history_comfort`
  - `two_frame_extended_comfort`
  - CSV 路径和 failed scenario 数
- Stage 2: cached LEAD smoke16：
  - cache 指向 `officialacc_npy`
  - checkpoint 使用 `model_0060.pth`
  - 成功标准：16/16 valid、无 missing token、CSV 生成
- Stage 3: cached LEAD full navtest：
  - 使用 0145 四卡 log shard
  - total scenarios 应为 `12,146`
  - 合并四个 CSV，按 scenario/token 去重后统计 full average
- Stage 4: 对照分析：
  - official LEAD online vs cached LEAD officialacc
  - cached LEAD officialacc vs cached LEAD finite-diff
  - 如果 cached LEAD 接近 official LEAD，说明 BEV cache 基本可信
  - 如果 cached LEAD 明显低于 official LEAD，优先排查 `bev_feature` 构建、planning decoder 调用、status feature、dtype/bfloat16、坐标转换

## Test Plan

- Static:
  - `python -m py_compile navsim_motdp/agents/cached_lead_agent.py`
  - `bash -n` cached LEAD smoke/full scripts
- Cache sanity:
  - 读取 `cache_index.npz`
  - 确认 `bev_feature.npy` shape/dtype 与 LEAD planner decoder 输入一致
  - 随机抽 token，确认 cache token 覆盖 navtest shard token
- Smoke:
  - navtest smoke16 officialacc cache
  - 期望 16 valid、0 missing、CSV 存在
- Full:
  - 0145 四卡 shard 跑完
  - 合并 CSV 后 total unique scenario/token = `12,146`
  - 记录 comfort 与 official LEAD full 的差距

## Assumptions

- 不用 MoT-DP diffusion head 测 cache 正确性；先固定 LEAD planner，只替换 online BEV 为 cached BEV。
- `officialacc_npy` 是主 cache；finite-diff cache 只做诊断对照。
- 当前不改训练、不重训模型。
- 计划先写入 `docs/lead_cached_dataset_validation_plan.md`，实施阶段再改脚本/agent 并启动评测。


## Implementation Log: 2026-05-17

Prepared for concurrent cached LEAD officialacc validation while official online LEAD full is still running.

Planned scripts:

- `scripts/_run_cached_lead_officialacc_smoke.sh` for navtest smoke16.
- `scripts/_run_cached_lead_officialacc_shards_0145.sh` for four-way navtest full over GPUs `0 1 4 5`.

Canonical cache:

`/workspace2/z_project/motdp_bev_cache_navtest_official4cam_officialacc_npy`

Metric cache:

`/workspace2/z_project/navsim_exp_lead_official/metric_cache_navtest_v1_1`

## Run Log: 2026-05-17 Cached LEAD OfficialAcc

### Static And Cache Sanity

- `python -m py_compile navsim_motdp/agents/cached_lead_agent.py`: pass.
- `bash -n scripts/_run_cached_lead_officialacc_smoke.sh scripts/_run_cached_lead_officialacc_shards_0145.sh`: pass.
- OfficialAcc navtest cache sanity:
  - tokens: `(12146,)`
  - `bev_feature.npy`: `(12146, 512, 8, 8) float16`
  - `bev_grid.npy`: `(12146, 64, 64, 64) float16`
  - `ego_status.npy`: `(12146, 4, 14) float32`
  - `trajectory.npy`: `(12146, 8, 2) float32`

### Smoke16

Command:

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
GPU_ID=4 \
EXPERIMENT_NAME=motdp_cached_lead_officialacc_smoke16_20260517_114419 \
scripts/_run_cached_lead_officialacc_smoke.sh
```

Result:

- successful scenarios: 16
- failed scenarios: 0
- final average score: `0.8172213556736327`
- CSV: `/workspace2/z_project/navsim_exp_motdp/motdp_cached_lead_officialacc_smoke16_20260517_114419/2026.05.17.11.44.31/2026.05.17.11.44.42.csv`

This is very close to the official LEAD smoke score `0.8174167`, so the cached LEAD planner path is valid on smoke16.

### Partial Launch Mistake

A first 0145 launch accidentally inherited `MAX_SCENES=16` from the smoke helper, so it ran only 16 scenarios per shard. It finished successfully but must not be treated as the full result.

Driver log:

`/workspace2/z_project/motdp_logs/motdp_cached_lead_officialacc_full_0145_20260517_driver.log`

### FullAll 0145 Launch

Fixed `MAX_SCENES` handling and relaunched with a new prefix.

Command:

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
EXP_PREFIX=motdp_cached_lead_officialacc_fullall_0145_20260517 \
GPUS_STR="0 1 4 5" \
scripts/_run_cached_lead_officialacc_shards_0145.sh
```

Tmux session:

`cachedleadfull0145`

Driver log:

`/workspace2/z_project/motdp_logs/motdp_cached_lead_officialacc_fullall_0145_20260517_driver.log`

Shard logs:

- `/workspace2/z_project/motdp_logs/motdp_cached_lead_officialacc_fullall_0145_20260517_shard0_of4_gpu0_20260517_114609.log`
- `/workspace2/z_project/motdp_logs/motdp_cached_lead_officialacc_fullall_0145_20260517_shard1_of4_gpu1_20260517_114609.log`
- `/workspace2/z_project/motdp_logs/motdp_cached_lead_officialacc_fullall_0145_20260517_shard2_of4_gpu4_20260517_114609.log`
- `/workspace2/z_project/motdp_logs/motdp_cached_lead_officialacc_fullall_0145_20260517_shard3_of4_gpu5_20260517_114609.log`

Initial shard counts:

- shard0: 3034 scenarios
- shard1: 3041 scenarios
- shard2: 3037 scenarios
- shard3: 3034 scenarios
- total: 12146 scenarios
