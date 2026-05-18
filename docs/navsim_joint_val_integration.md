# NAVSIM Joint Route/Speed Val Integration

Date: 2026-05-18

## Summary

The joint NAVSIM model is now connected to cached PDM validation.

The trained model uses NAVSIM official acceleration in ego status, so validation
must use official-acceleration status as well:

- cached val uses `/workspace2/z_project/motdp_bev_cache_navtest_official4cam_officialacc_npy`
- online LEAD val should read `agent_input.ego_statuses[-1].ego_acceleration`
- do not use finite-difference acceleration for this model

## Checkpoint

Current canonical joint checkpoint:

```text
/workspace2/z_project/motdp_logs/navsim_joint_route_speed_officialacc_e90_gpus0145_b128_pergpu_20260517_185212/best_model.pt
```

Training summary:

```text
epochs: 90
global_batch_size: 512
model_type: navsim_joint_route_speed_diffusion
traj tokens: 8
route tokens: 50
speed tokens: 8
state tokens: 0
best val loss: 0.0307
best val l2: 0.385m
```

## Code Changes

Implemented val-facing changes:

- `model/navsim_joint_route_speed_diffusion.py`
  - added `sample()` for joint trajectory/route/speed denoising
  - returns trajectory by default for NAVSIM PDM
- `navsim_motdp/agents/cached_diffusion_agent.py`
  - auto-detects `model_type=navsim_joint_route_speed_diffusion`
  - loads trajectory/route/speed normalization buffers from checkpoint
  - keeps legacy `NavSimSimpleDiffusion` support
  - trims 14-dim officialacc cache status down to 8 dims when the model expects 8
- `navsim_motdp/agents/online_lead_agent.py`
  - auto-detects joint checkpoint
  - keeps official LEAD preprocessing
  - builds ego status from NAVSIM dataclass `ego_velocity`, `ego_acceleration`, and `driving_command`
- `scripts/_run_navtest_pdm.sh`
  - default checkpoint is the joint e90 checkpoint
  - default cache is official4cam officialacc navtest cache
  - exposes `SAMPLER_STOCHASTIC=true/false`

## Verified

Static checks on newhpc:

```text
python -m py_compile model/navsim_joint_route_speed_diffusion.py navsim_motdp/agents/cached_diffusion_agent.py navsim_motdp/agents/online_lead_agent.py
bash -n scripts/_run_navtest_pdm.sh
```

Agent smoke on newhpc:

```text
model: NavSimJointRouteSpeedDiffusion
sample shape: (1, 8, 2)
ego_input_dim: 8
```

Cached PDM smoke16:

```text
experiment: motdp_joint_route_speed_navtest_officialacc_e90_smoke16
successful scenarios: 16
failed scenarios: 0
score: 0.7286961304965118
comfort: 1.0
no_at_fault_collisions: 0.875
drivable_area_compliance: 0.875
ego_progress: 0.6738707131916283
time_to_collision_within_bound: 0.8125
driving_direction_compliance: 1.0
csv: /workspace2/z_project/navsim_exp_motdp/motdp_joint_route_speed_navtest_officialacc_e90_smoke16/2026.05.18.04.16.02/2026.05.18.04.16.08.csv
```

## Commands

Cached PDM smoke:

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
MAX_SCENES=16 \
GPU_ID=0 \
EXPERIMENT_NAME=motdp_joint_route_speed_navtest_officialacc_e90_smoke16 \
bash scripts/_run_navtest_pdm.sh
```

Cached PDM full:

```bash
tmux new-session -d -s nval 'cd /workspace1/z_project/code/motdp_z_navsim_motdp && \
GPU_ID=0 \
EXPERIMENT_NAME=motdp_joint_route_speed_navtest_officialacc_e90_full \
bash scripts/_run_navtest_pdm.sh'
```

Deterministic sampler comparison:

```bash
tmux new-session -d -s nval_det 'cd /workspace1/z_project/code/motdp_z_navsim_motdp && \
GPU_ID=1 \
SAMPLER_STOCHASTIC=false \
EXPERIMENT_NAME=motdp_joint_route_speed_navtest_officialacc_e90_full_det \
bash scripts/_run_navtest_pdm.sh'
```

## Notes

- Cached val does not need route/speed labels; route/speed are latent generated tokens.
- Full PDM should use `fallback=raise`, already set in `_run_navtest_pdm.sh`.
- Online LEAD joint val is structurally wired, but full online score still needs a separate smoke/full run because it depends on live image preprocessing and LEAD backbone execution.

## Full Validation Results 2026-05-18

Both cached and online full navtest were run as 4 log-name shards on GPUs `0,1,2,3`.

### Cached Officialacc

```text
experiment_prefix: motdp_cached_joint_route_speed_officialacc_e90_full
rows: 12146
unique_tokens: 12146
failed_scenarios: 0
score: 0.7629066265629388
comfort: 0.9998353367363741
no_at_fault_collisions: 0.9552939239255722
drivable_area_compliance: 0.8698336901037379
ego_progress: 0.7246484699513206
time_to_collision_within_bound: 0.8844063889346286
driving_direction_compliance: 0.9575580438004281
```

Shard CSVs:

```text
/workspace2/z_project/navsim_exp_motdp/motdp_cached_joint_route_speed_officialacc_e90_full_shard0_of4_gpu0/2026.05.18.04.20.55/2026.05.18.04.31.27.csv
/workspace2/z_project/navsim_exp_motdp/motdp_cached_joint_route_speed_officialacc_e90_full_shard1_of4_gpu1/2026.05.18.04.20.55/2026.05.18.04.28.28.csv
/workspace2/z_project/navsim_exp_motdp/motdp_cached_joint_route_speed_officialacc_e90_full_shard2_of4_gpu2/2026.05.18.04.20.54/2026.05.18.04.30.32.csv
/workspace2/z_project/navsim_exp_motdp/motdp_cached_joint_route_speed_officialacc_e90_full_shard3_of4_gpu3/2026.05.18.04.20.55/2026.05.18.04.29.17.csv
```

### Online Officialacc

```text
experiment_prefix: motdp_online_joint_route_speed_officialacc_e90_full
rows: 12146
unique_tokens: 12146
failed_scenarios: 0
score: 0.7711030603819962
comfort: 0.9998353367363741
no_at_fault_collisions: 0.9600691585707228
drivable_area_compliance: 0.8735386135353203
ego_progress: 0.7334650063962411
time_to_collision_within_bound: 0.8893462868434052
driving_direction_compliance: 0.9614687963115429
```

Shard CSVs:

```text
/workspace2/z_project/navsim_exp_motdp/motdp_online_joint_route_speed_officialacc_e90_full_shard0_of4_gpu0/2026.05.18.04.20.55/2026.05.18.04.35.45.csv
/workspace2/z_project/navsim_exp_motdp/motdp_online_joint_route_speed_officialacc_e90_full_shard1_of4_gpu1/2026.05.18.04.20.55/2026.05.18.04.31.49.csv
/workspace2/z_project/navsim_exp_motdp/motdp_online_joint_route_speed_officialacc_e90_full_shard2_of4_gpu2/2026.05.18.04.20.56/2026.05.18.04.34.54.csv
/workspace2/z_project/navsim_exp_motdp/motdp_online_joint_route_speed_officialacc_e90_full_shard3_of4_gpu3/2026.05.18.04.20.55/2026.05.18.04.32.33.csv
```

### Interpretation

Online is slightly better than cached:

```text
score delta: +0.0081964338190574
DAC delta: +0.0037049234315824
NC delta: +0.0047752346451506
ego_progress delta: +0.0088165364449205
TTC delta: +0.0049398979087766
```

Comfort is no longer the bottleneck in this joint run; both cached and online are `0.999835`.
