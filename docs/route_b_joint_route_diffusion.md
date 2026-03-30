# Route B Joint Route Diffusion Notes

Date: 2026-03-30

## Summary

This note records the joint route diffusion change for the Route B ego path, the new route normalization requirement, local/HPC setup notes, and the current explanation for slower epoch time after the change.

## What Changed

### 1. Joint ego diffusion state

The ego denoising path now uses a joint 26-point diffusion state:

- first 6 points: trajectory
- last 20 points: route

The route branch is no longer a pure learnable query in `forward_ego`. It is built from noisy route waypoint coordinates, so route is now a true diffusion branch in the ego path.

Files:

- `model/transformer_for_diffusion_multi_head.py`
- `policy/annealed_energy_guidance_policy.py`

### 2. Route-specific normalization

Route diffusion now depends on a dedicated per-waypoint absolute normalization file:

- key: `route_abs_stats_path`
- file format: NPZ
- tensors:
  - `route_abs_mean`: `(20, 2)`
  - `route_abs_std`: `(20, 2)`

This is separate from trajectory abs/global-abs normalization.

Files:

- `policy/annealed_energy_guidance_policy.py`
- `training/train_carla_bev.py`
- `dataset/compute_action_stats.py`
- `config/pdm_local_route_b.yaml`
- `config/pdm_hpc_route_b.yaml`

### 3. Fine-BEV sampling structure

Before the shared decoder:

- trajectory tokens always do center sampling from `bev_feature_upsample`
- route tokens now also do center sampling from `bev_feature_upsample`

At low timesteps, only trajectory tokens receive additional detail features:

- traj-local detail from a fixed rear-biased stencil
- route-based detail using all 20 route waypoints as look-ahead anchors

The ego self-attention mask remains one-way:

- `traj -> route`: allowed
- `route -> traj`: blocked

Reason:

- route already receives its own fine-BEV center sample
- trajectory can already consume route information through decoder interaction and `route_features`

### 4. Energy guidance scope

Energy guidance remains trajectory-only in v1.

During inference:

- trajectory and route denoise jointly in the ego path
- energy correction is applied only to the trajectory slice
- route is carried along as a jointly denoised but non-energy-corrected branch

## Local And HPC Notes

### Local

Formal local route stats file:

- `/media/z/data/dataset/pdm_lite_mini/route_abs_stats.npz`

The local Route B config already points to this path.

### HPC

Formal HPC route stats file:

- `/workspace1/z_project/dataset/pdm_lite/tmp_data/route_abs_stats.npz`

The HPC Route B config points to this path:

- `config/pdm_hpc_route_b.yaml`

### Packed-sample route stats generation

`dataset/compute_action_stats.py` was updated so `--mode route_abs` now prefers:

- `train/samples_packed.pkl`
- `val/samples_packed.pkl`

and only falls back to scanning individual `*.pkl` files if packed samples are absent.

This matters on HPC because scanning hundreds of thousands of individual pkl files is much slower than reading the packed files.

## Why Epoch Time Increased

The current slowdown is expected and mainly comes from the new fine-BEV compute in the ego path.

### Important: the decoder token length is not the main reason

The shared decoder already operated on:

- 6 trajectory tokens
- 20 route tokens/queries

So the sequence length for decoder self-attention is still effectively 26. The main extra cost is not a large jump in decoder token count.

### Main cost increase: more fine-BEV branches

Previously, the ego path mainly used one center point BEV sampling branch for trajectory tokens.

Now the ego path uses four BEV sampling branches:

1. trajectory center sampling
2. trajectory local-detail sampling
3. trajectory route-detail sampling
4. route center sampling

Rough sample-point count per sample changed from:

- old: `6 * 1 = 6`

to approximately:

- traj center: `6 * 1 = 6`
- traj local detail: `6 * 7 = 42`
- traj route detail: `6 * 20 = 120`
- route center: `20 * 1 = 20`

Total:

- new: about `188`

So the BEV sampling workload increased a lot.

### Main cost increase: repeated full-BEV value projection

Each `GridSampleCrossBEVAttention` call contains its own:

- `value_proj = Conv2d(...)`

and each forward call applies that projection to the entire `bev_feature_upsample`.

So the current ego path does not just sample more points. It also runs multiple full-BEV projections per batch:

- old ego path: roughly one `value_proj(bev_upsample)`
- new ego path: one each for
  - `bev_point_attn`
  - `traj_detail_attn`
  - `route_detail_attn`
  - `route_point_attn`

This is a major source of the extra epoch time.

### Current implementation detail: the timestep gate does not save compute

The low-timestep gate currently only gates the fused output, not the actual computation.

That means:

- `traj_local_detail`
- `traj_route_detail`

are still computed even when `t > 400`.

So although the design intent is "detail only for low timestep", the current implementation still pays the cost for these branches on almost every batch.

This is likely the single easiest optimization opportunity.

### Validation epochs can become much slower

Training still uses one noisy joint forward per batch.

However, validation uses `conditional_sample`, which runs multi-step denoising. Because the ego path now denoises a joint traj+route state with the heavier fine-BEV logic, validation epochs can become noticeably more expensive than before.

## Most Likely First Optimizations

If epoch time becomes a problem, the first two optimizations to try are:

1. Actually skip detail computation when `t > 400`

- do not compute `traj_local_detail`
- do not compute `traj_route_detail`
- return only the center-sampled path for those samples

2. Share projected BEV values across detail branches

- compute one projected `value` tensor from `bev_feature_upsample`
- reuse it across center/detail sampling branches instead of running 3-4 separate `value_proj` convolutions

These two changes should recover much of the runtime without changing the overall modeling direction.

## Current Status

The following are in place:

- local Route B formal route stats file exists
- HPC Route B formal route stats file exists
- HPC Route B config points to the correct route stats path
- local commit created for the joint route diffusion change:
  - `5fe56f5` `Add joint route diffusion to Route B ego path`

## Suggested Training Command On HPC

```bash
cd /workspace1/z_project/code/motdp_z

CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace1/z_project/env_conda/z_dpauto/bin/torchrun --standalone --nproc_per_node=4 \
  training/train_carla_bev.py \
  --config_path config/pdm_hpc_route_b.yaml
```
