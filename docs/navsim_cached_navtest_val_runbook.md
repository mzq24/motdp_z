# NAVSIM Cached Navtest Val / PDM Runbook

本文记录 2026-05-16 这次 **cache-backed NAVSIM navtest official PDM val** 的完整跑法、路径、结果和踩坑。这里的 “val” 指 NAVSIM official PDM scoring on `navtest`。

## 目标

用已经训练好的 MoT-DP simple diffusion checkpoint，在 NAVSIM `navtest` 上跑 official one-stage PDM score。

核心结果：

```text
experiment_name: motdp_cached_navtest_npy_full_e60
successful: 12146
failed: 0
average score: 0.7512730999
```

最终 CSV：

```text
/workspace2/z_project/navsim_exp_motdp/motdp_cached_navtest_npy_full_e60/2026.05.16.09.25.04/2026.05.16.10.31.04.csv
```

## 固定环境和路径

代码 worktree：

```text
/workspace1/z_project/code/motdp_z_navsim_motdp
```

NAVSIM devkit：

```text
/home/z/code/navsim
```

Conda env：

```text
z_navsim_motdp
/workspace1/miniconda/envs/z_navsim_motdp/bin/python
```

数据 / cache / checkpoint：

```text
OPENSCENE_DATA_ROOT=/workspace2/data/navsim
NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps
NAVSIM_DEVKIT_ROOT=/home/z/code/navsim
NAVSIM_EXP_ROOT=/workspace2/z_project/navsim_exp_motdp
TMPDIR=/workspace2/z_project/tmp

checkpoint=/workspace2/z_project/motdp_logs/navsim_simple_official_lr1e4_e60_b64/best_model.pt
navtest_metric_cache=/workspace2/data/navsim/processed_data/metric_cache_navtest
navtest_bev_cache_npy=/workspace2/z_project/motdp_bev_cache_navtest_npy
```

非常重要：运行 official PDM 时入口脚本在 NAVSIM repo 下，所以必须把 MoT-DP worktree 加进 `PYTHONPATH`：

```bash
export PYTHONPATH=/workspace1/z_project/code/motdp_z_navsim_motdp:$PYTHONPATH
```

否则 Hydra worker 会报：

```text
Error locating target 'navsim_motdp.agents.cached_diffusion_agent.CachedDiffusionAgent'
```

## Navtest BEV Cache

navtest BEV 预计算的原始输出是 compressed NPZ shards：

```text
/workspace2/z_project/motdp_bev_cache_navtest/bev_cache_shard000.npz
/workspace2/z_project/motdp_bev_cache_navtest/bev_cache_shard001.npz
/workspace2/z_project/motdp_bev_cache_navtest/bev_cache_shard002.npz
/workspace2/z_project/motdp_bev_cache_navtest/bev_cache_shard003.npz
```

这 4 个 final shards 覆盖 navtest official tokens：

```text
bev_cache_shard000.npz: raw=2006 kept=2006
bev_cache_shard001.npz: raw=3450 kept=3450
bev_cache_shard002.npz: raw=3074 kept=3074
bev_cache_shard003.npz: raw=3616 kept=3616
Raw entries: 12146
Kept entries: 12146
Unique kept tokens: 12146
```

不要直接用 compressed NPZ 跑 PDM。原因是 PDM 按 token 随机访问，`np.savez_compressed` 里的 `d["bev_grid"][idx]` 会反复解压整个 shard，容易极慢或者 OOM。正式 val 使用 consolidated NPY/memmap cache：

```text
/workspace2/z_project/motdp_bev_cache_navtest_npy
```

转换命令：

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
/workspace1/miniconda/envs/z_navsim_motdp/bin/python scripts/convert_navsim_cache_to_npy.py \
  --input-dir /workspace2/z_project/motdp_bev_cache_navtest \
  --output-dir /workspace2/z_project/motdp_bev_cache_navtest_npy \
  --overwrite
```

转换结果：

```text
count: 12146
elapsed_sec: 102.77
bev_grid.npy: 6.0G
bev_feature.npy: 760M
ego_status.npy: 2.6M
trajectory.npy: 760K
cache_index.npz: 2.6M
```

## Cached Agent 关键设置

使用 agent：

```text
navsim_motdp.agents.cached_diffusion_agent.CachedDiffusionAgent
```

关键修复：

- 优先读取 NPY/memmap cache：`cache_index.npz` + `bev_grid.npy` + `ego_status.npy`。
- NPZ fallback 只匹配 final shards：`bev_cache_shard[0-9][0-9][0-9].npz`，排除 `*_tmp.npz`。
- seed 使用 `sha1(token) + deterministic_seed`，不再用 Python `hash(token)`。
- full navtest 默认 `+agent.fallback=raise`，避免 missing token 被 constant velocity 静默掩盖。
- `num_inference_steps` override 会同步到 diffusion model。
- memmap 单样本读取会 copy 成 writable numpy array，避免 PyTorch non-writable warning。

## 固化脚本

正式脚本：

```text
scripts/_run_navtest_pdm.sh
```

脚本支持的主要环境变量：

```bash
GPU_ID=${GPU_ID:-2}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-motdp_cached_navtest_npy_full_e60}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/workspace2/z_project/motdp_logs/navsim_simple_official_lr1e4_e60_b64/best_model.pt}
CACHE_DIR=${CACHE_DIR:-/workspace2/z_project/motdp_bev_cache_navtest_npy}
METRIC_CACHE_PATH=${METRIC_CACHE_PATH:-/workspace2/data/navsim/processed_data/metric_cache_navtest}
MAX_SCENES=${MAX_SCENES:-}
```

实际 PDM command 由脚本调用：

```bash
python /home/z/code/navsim/navsim/planning/script/run_pdm_score_one_stage.py \
  train_test_split=navtest \
  experiment_name="$EXPERIMENT_NAME" \
  agent._target_=navsim_motdp.agents.cached_diffusion_agent.CachedDiffusionAgent \
  +agent.checkpoint_path="$CHECKPOINT_PATH" \
  +agent.cache_dir="$CACHE_DIR" \
  +agent.device="cuda:${GPU_ID}" \
  +agent.fallback=raise \
  metric_cache_path="$METRIC_CACHE_PATH" \
  worker=single_machine_thread_pool \
  worker.max_workers=1
```

如果设置了 `MAX_SCENES`，脚本会额外追加：

```bash
train_test_split.scene_filter.max_scenes=${MAX_SCENES}
```

## 静态检查

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
PYTHONPATH=/workspace1/z_project/code/motdp_z_navsim_motdp:$PYTHONPATH \
  /workspace1/miniconda/envs/z_navsim_motdp/bin/python -m py_compile \
  navsim_motdp/agents/cached_diffusion_agent.py

bash -n scripts/_run_navtest_pdm.sh
```

结果：

```text
py_compile: PASS
bash -n: PASS
```

## Agent Standalone Smoke

用于验证 agent 可以从 navtest NPY cache 中按 token 取 BEV，并输出 `(8,3)` trajectory。

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
PYTHONPATH=/workspace1/z_project/code/motdp_z_navsim_motdp:$PYTHONPATH \
OPENSCENE_DATA_ROOT=/workspace2/data/navsim \
NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps \
/workspace1/miniconda/envs/z_navsim_motdp/bin/python - <<'PY'
import numpy as np
from types import SimpleNamespace
from navsim_motdp.agents.cached_diffusion_agent import CachedDiffusionAgent

cache_dir = '/workspace2/z_project/motdp_bev_cache_navtest_npy'
ckpt = '/workspace2/z_project/motdp_logs/navsim_simple_official_lr1e4_e60_b64/best_model.pt'
index = np.load(cache_dir + '/cache_index.npz', allow_pickle=False)
token = str(index['tokens'][0])
agent = CachedDiffusionAgent(checkpoint_path=ckpt, cache_dir=cache_dir, device='cuda:2', fallback='raise')
agent.initialize()
scene = SimpleNamespace(scene_metadata=SimpleNamespace(initial_token=token, num_history_frames=4), frames=[])
traj = agent.compute_trajectory(SimpleNamespace(ego_statuses=[]), scene)
print('token', token)
print('poses_shape', traj.poses.shape)
print('first_pose', traj.poses[0].tolist())
print('last_pose', traj.poses[-1].tolist())
PY
```

结果：

```text
token 431ae29947e95c26
poses_shape (8, 3)
first_pose [0.014909744262695312, 0.02346116304397583, 1.0046766996383667]
last_pose [5.099947452545166, -0.018973827362060547, -0.0036609689705073833]
```

## Smoke16 Official PDM

先跑 16 个场景确认 official PDM 链路完整：

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
GPU_ID=2 \
MAX_SCENES=16 \
EXPERIMENT_NAME=motdp_cached_navtest_npy_smoke16_after_fix \
scripts/_run_navtest_pdm.sh
```

结果：

```text
successful: 16
failed: 0
average score: 0.7456234085
```

CSV：

```text
/workspace2/z_project/navsim_exp_motdp/motdp_cached_navtest_npy_smoke16_after_fix/2026.05.16.09.24.25/2026.05.16.09.24.34.csv
```

注意：这个 score 和更早 smoke16 的 `0.6408` 不同是正常的，因为 agent seed 从 Python `hash(token)` 改成了跨进程稳定的 `sha1(token)`。

## Full Navtest Official PDM

full run 挂在 tmux：

```bash
ssh new_hpc
cd /workspace1/z_project/code/motdp_z_navsim_motdp

tmux new-session -d -s navtest_pdm_full \
  'cd /workspace1/z_project/code/motdp_z_navsim_motdp && \
   GPU_ID=2 \
   EXPERIMENT_NAME=motdp_cached_navtest_npy_full_e60 \
   scripts/_run_navtest_pdm.sh 2>&1 | \
   tee /workspace2/z_project/navsim_exp_motdp/motdp_cached_navtest_npy_full_e60_tmux.log'
```

监控命令：

```bash
tmux capture-pane -t navtest_pdm_full -p -S -80

tail -f /workspace2/z_project/navsim_exp_motdp/motdp_cached_navtest_npy_full_e60_tmux.log

grep -o 'Processing scenario [0-9]* / 12146' \
  /workspace2/z_project/navsim_exp_motdp/motdp_cached_navtest_npy_full_e60_tmux.log | tail -n 1

find /workspace2/z_project/navsim_exp_motdp/motdp_cached_navtest_npy_full_e60 \
  -maxdepth 3 -type f -name '*.csv' \
  -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail
```

启动后确认：

```text
Starting pdm scoring of 12146 scenarios
Processing scenario 12146 / 12146
```

最终结果：

```text
successful: 12146
failed: 0
average score: 0.7512730999274955
```

CSV：

```text
/workspace2/z_project/navsim_exp_motdp/motdp_cached_navtest_npy_full_e60/2026.05.16.09.25.04/2026.05.16.10.31.04.csv
```

Average row：

```text
no_at_fault_collisions: 0.9541001153
drivable_area_compliance: 0.8606948790
driving_direction_compliance: 0.9720072452
traffic_light_compliance: 0.9962950766
ego_progress: 0.8712117462
time_to_collision_within_bound: 0.9372632966
lane_keeping: 0.9224436028
history_comfort: 0.9796640869
two_frame_extended_comfort: 0.8237051793
score: 0.7512730999
```

## DAC=0 Bad Case 可视化

full CSV 中：

```text
drivable_area_compliance == 0: 1692 / 12146
```

这些 case 因为 DAC 是 multiplicative metric，单场景 score 全部为 `0`。

可视化脚本：

```text
scripts/visualize_navtest_dac0_cases.py
```

smoke3：

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
PYTHONPATH=/workspace1/z_project/code/motdp_z_navsim_motdp:$PYTHONPATH \
OPENSCENE_DATA_ROOT=/workspace2/data/navsim \
NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps \
/workspace1/miniconda/envs/z_navsim_motdp/bin/python \
  scripts/visualize_navtest_dac0_cases.py \
  --max-cases 3 \
  --output-dir /workspace2/z_project/navsim_debug/dac0_bev_smoke3_20260516 \
  --device cuda:2
```

smoke3 输出：

```text
/workspace2/z_project/navsim_debug/dac0_bev_smoke3_20260516/png
/workspace2/z_project/navsim_debug/dac0_bev_smoke3_20260516/dac0_visualization_summary.csv
/workspace2/z_project/navsim_debug/dac0_bev_smoke3_20260516/contact_sheet_first_cases.png
```

全量 DAC=0 可视化挂 tmux：

```bash
tmux new-session -d -s navtest_dac0_viz \
  'cd /workspace1/z_project/code/motdp_z_navsim_motdp && \
   export PYTHONPATH=/workspace1/z_project/code/motdp_z_navsim_motdp:$PYTHONPATH && \
   export OPENSCENE_DATA_ROOT=/workspace2/data/navsim && \
   export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps && \
   /workspace1/miniconda/envs/z_navsim_motdp/bin/python \
     scripts/visualize_navtest_dac0_cases.py \
     --output-dir /workspace2/z_project/navsim_debug/dac0_bev_full_e60_20260516 \
     --device cuda:2 2>&1 | \
   tee /workspace2/z_project/navsim_debug/dac0_bev_full_e60_20260516.log'
```

输出：

```text
/workspace2/z_project/navsim_debug/dac0_bev_full_e60_20260516/png
/workspace2/z_project/navsim_debug/dac0_bev_full_e60_20260516/dac0_visualization_summary.csv
/workspace2/z_project/navsim_debug/dac0_bev_full_e60_20260516/contact_sheet_first_cases.png
/workspace2/z_project/navsim_debug/dac0_bev_full_e60_20260516.log
```

图中叠加：

```text
drivable polygons
route / centerline
human trajectory
model raw trajectory
model y-flip hypothesis
simulated trajectory
off-road footprint steps
```

可视化目的是区分：

1. 多数 bad case 是否整体 y 翻转 / 偏移，若是则优先排坐标系。
2. 多数 bad case 是否转弯切角，若是则优先考虑 smoothing / route following。
3. 多数 bad case 是否 junction lane connector 附近擦边，若是则说明 NAVSIM metric 对 connector/footprint 边界非常敏感，需要更强 route/centerline constraint。

## 常见坑

1. **缺 `PYTHONPATH`**
   - Hydra worker 找不到自定义 agent。
   - 必须 export MoT-DP worktree 到 `PYTHONPATH`。

2. **不要直接用 compressed NPZ cache 跑 full PDM**
   - NPZ 随机访问会反复解压整个 shard。
   - full val 必须走 NPY/memmap cache。

3. **不要依赖 `CUDA_VISIBLE_DEVICES` 控制 worker GPU**
   - 当前官方 worker pool log 仍显示 `Number of GPUs per node: 0`，但 agent 自己用 `+agent.device=cuda:${GPU_ID}` 可以正常工作。
   - 单 worker full PDM 已验证通过。

4. **full navtest 用 `fallback=raise`**
   - 任何 cache/token 不匹配应直接暴露。
   - 本次 full run `failed=0`，说明 navtest cache/token 对齐无问题。

5. **seed 要稳定**
   - Python `hash(token)` 跨进程不稳定。
   - 当前 agent 使用 `sha1(token)`，所以 repeated run 可复现。
