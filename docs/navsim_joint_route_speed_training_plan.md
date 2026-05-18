# NAVSIM Joint Route/Speed/Trajectory Training Plan

Date: 2026-05-18

## Summary

Next NAVSIM training should not start from the current trajectory-only cache/model.
The current cache has BEV, ego status, and trajectory labels, but it does not yet
contain per-sample route, route mask, or speed-profile labels.

The next stage is to build a NAVSIM-specific joint training path:

```text
traj tokens: 8
route tokens: 50
speed tokens: 8
state tokens: 0
```

This means we keep white-noise diffusion, add route and speed targets, and clean
out old semantic/state-token assumptions from the NAVSIM path. This is not the
old PDMLite / Route-B state-token setup.

## Current State

Canonical train cache:

```text
/workspace2/z_project/motdp_bev_cache_train_official4cam_officialacc_npy
```

Current cache contents:

```text
bev_feature.npy
bev_grid.npy
cache_index.npz
ego_status.npy
trajectory.npy
```

Current cache does not contain:

```text
route.npy
route_mask.npy
path.npy
path_mask.npy
speed_profile.npy
```

Canonical NAVSIM route/speed/traj stats:

```text
/workspace2/z_project/motdp_navsim_norm_stats/navtrain_official_h8_r50_20260518_sparsemask
```

The route/path construction is already aligned with SparseDrive:

- future ego poses in current ego frame
- arc-length sampling every 1m
- fixed 50 points
- tail held at final reachable pose
- `route_mask/path_mask` marks supervised points

Full stats summary:

```text
tokens_seen_in_logs: 103288
tokens_matched: 102608
tokens_skipped_future_short: 680
valid_traj: 102608
valid_speed: 102608
valid_route_any: 102153
valid_route_full: 82800
failed: 0
```

Current `NavSimSimpleDiffusion` is still trajectory-only:

- no semantic state branch
- no branch conditioning
- no route tokens
- no speed tokens
- no route/speed losses

## Target Label Sidecar

Create a NAVSIM label sidecar builder that reads:

- `cache_index.npz` from the canonical train cache
- raw NAVSIM trainval logs from `/workspace2/data/navsim/navsim_logs/trainval`

Write labels to a new sidecar directory:

```text
/workspace2/z_project/motdp_navsim_labels/navtrain_official_h8_r50_sparsemask
```

Required output files:

```text
label_index.npz
trajectory.npy
route.npy
route_mask.npy
path.npy
path_mask.npy
speed_profile.npy
metadata.json
```

Expected shapes:

```text
trajectory:    (N, 8, 2)
route:         (N, 50, 2)
route_mask:    (N, 50)
path:          (N, 50, 2)
path_mask:     (N, 50)
speed_profile: (N, 8)
```

Label filtering policy:

- keep samples with full `trajectory(8)` and `speed_profile(8)`
- require `route_mask.any()` for route supervision
- do not require full `route50`
- drop only tokens that do not have enough future for 8-step traj/speed
- write `label_index.tokens` in the same order as the output arrays

Alignment policy:

- `label_index.tokens` must be a subset of `cache_index.tokens`
- dataset maps labels to cache rows by token, not by assuming identical array length
- duplicated tokens are deduped consistently with the training dataset setting
- missing labels for selected training tokens raise by default

## Dataset Changes

Extend `NavSimCachedDataset` with an optional `label_dir` argument.

When `label_dir` is provided, dataset returns:

```text
bev_grid
bev_feature
ego_status
trajectory
route
route_mask
path
path_mask
speed_profile
token
log_name
frame_idx
```

Required behavior:

- build a token-to-label index during init
- filter cache samples to tokens available in `label_dir`
- expose matched count and missing count in init logs
- raise if selected split has zero labeled samples
- preserve trajectory-only behavior when `label_dir` is not provided

## Model Changes

Create a NAVSIM joint white-noise decoder from the current simple model.

Token contract:

```text
joint tokens = [traj8, route50, speed8]
state tokens = none
```

Keep:

- LEAD BEV conditioning through cached `bev_grid`
- ego-status history conditioning
- timestep conditioning
- self-attention across joint tokens
- BEV cross-attention for spatial tokens

Remove or avoid migrating:

- semantic state tokens
- previous state cache
- branch-condition tokens
- stage1 state consistency losses
- energy guidance heads
- borrow/junction state heads

Output heads:

```text
traj_head  -> (B, 8, 2)
route_head -> (B, 50, 2)
speed_head -> (B, 8)
```

Losses:

```text
traj_loss:  MSE or L1 on normalized trajectory
route_loss: masked L1 on normalized route using route_mask
speed_loss: L1 or MSE on normalized speed_profile
total_loss: traj_loss + route_loss_weight * route_loss + speed_loss_weight * speed_loss
```

Default loss weights:

```text
route_loss_weight = 1.0
speed_loss_weight = 0.5
```

Normalization:

- use sparsemask `abs_stats.npz` for trajectory normalization
- use sparsemask `route_abs_stats.npz` for route normalization
- use sparsemask `speed_profile_stats.npz` for speed normalization
- store these stats in checkpoint config/buffers

## Training Changes

Add NAVSIM-specific training args:

```text
--label-dir /workspace2/z_project/motdp_navsim_labels/navtrain_official_h8_r50_sparsemask
--route-points 50
--speed-horizon 8
--route-loss-weight 1.0
--speed-loss-weight 0.5
--traj-stats-path .../navtrain_official_h8_r50_sparsemask_abs_stats.npz
--route-stats-path .../navtrain_official_h8_r50_sparsemask_route_abs_stats.npz
--speed-stats-path .../navtrain_official_h8_r50_sparsemask_speed_profile_stats.npz
```

Recommended training defaults after smoke passes:

```text
cache_dir=/workspace2/z_project/motdp_bev_cache_train_official4cam_officialacc_npy
label_dir=/workspace2/z_project/motdp_navsim_labels/navtrain_official_h8_r50_sparsemask
epochs=90
per_gpu_batch=128
gpus=0,1,4,5
lr=1e-4
weight_decay=1e-4
val_every_epochs=5
save_every_epochs=5
```

## Test Plan

Label builder smoke:

```text
--max-tokens 512
```

Expected:

```text
route=(512,50,2)
route_mask=(512,50)
speed_profile=(512,8)
metadata reports zero failures
```

Dataset smoke:

- load train split with `label_dir`
- inspect one batch
- verify all tensor shapes
- verify token-label mapping by checking a few random tokens

Model smoke:

- one forward pass on GPU
- one loss backward pass
- outputs include `traj_loss`, `route_loss`, `speed_loss`
- no NaN when route masks are partial

Training smoke:

```text
max_train_samples=1024
max_val_samples=256
epochs=1
```

Expected logs:

```text
train_loss
traj_loss
route_loss
speed_loss
val_l2
```

Full train starts only after all smoke checks pass.

## Implementation Roadmap To Reach NewHPC Training

The implementation should proceed in this order. Do not start full training until each earlier gate is green.

### 1. Label sidecar builder

Add a NAVSIM-only script:

```text
scripts/data_tools/build_navsim_joint_labels.py
```

Required arguments:

```text
--log-root /workspace2/data/navsim/navsim_logs/trainval
--cache-dir /workspace2/z_project/motdp_bev_cache_train_official4cam_officialacc_npy
--output-dir /workspace2/z_project/motdp_navsim_labels/navtrain_official_h8_r50_sparsemask
--traj-horizon 8
--route-points 50
--path-lookahead-frames 80
--route-step-m 1.0
--max-tokens optional smoke limit
```

Implementation source of truth:

- reuse `trajectory_from_cache`, `speed_from_cache`, `relative_future_points`, and SparseDrive-mask `sample_path_by_distance` from `scripts/data_tools/compute_navsim_traj_route_speed_norm_stats.py`
- preserve token/log/frame metadata from `cache_index.npz`
- output labels in label-index order, not raw log scan order
- write `metadata.json` with matched, skipped-short-future, skipped-empty-route, and failed counts

Expected full output count should be close to the current stats count:

```text
N = 102608
```

### 2. Dataset label loading

Extend `dataset/navsim_cached_dataset.py` rather than introducing a second dataset class. Add these args:

```text
label_dir: Optional[str] = None
require_labels: bool = True
```

When `label_dir` is set:

- load `label_index.npz`, `route.npy`, `route_mask.npy`, `path.npy`, `path_mask.npy`, and `speed_profile.npy`
- create a token-to-label-row map
- filter `base_indices` to cache tokens with labels before train/val split
- return sidecar `trajectory` instead of cache `trajectory` if sidecar exists, so trajectory/route/speed are all generated by the same label code
- keep old trajectory-only behavior when `label_dir` is unset

### 3. Joint model without state tokens

Create a new model file to avoid breaking existing trajectory-only checkpoints:

```text
model/navsim_joint_route_speed_diffusion.py
```

The new class should be separate from `NavSimSimpleDiffusion`:

```text
NavSimJointRouteSpeedDiffusion
```

Do not import or depend on:

- `policy/annealed_energy_guidance_policy.py`
- `model/transformer_for_diffusion_multi_head.py`
- Route-B configs
- semantic/state branch-condition helpers

Token layout is fixed:

```text
traj slice  = [0:8]
route slice = [8:58]
speed slice = [58:66]
state slice = none
```

Speed token representation for v1:

- diffuse scalar speed values as `(B, 8, 1)` tokens
- embed with a speed-token MLP
- output `(B, 8)` speed profile

Route token representation for v1:

- diffuse normalized route points as `(B, 50, 2)` tokens
- apply masked route loss with `route_mask`

### 4. Joint DDP training entrypoint

Create a new training script to avoid ambiguity with old trajectory-only runs:

```text
training/train_navsim_joint_route_speed_ddp.py
```

It should mirror `training/train_navsim_diffusion_ddp.py` for DDP, AMP, validation, and checkpointing, but add:

- `--label-dir`
- `--traj-stats-path`
- `--route-stats-path`
- `--speed-stats-path`
- `--route-loss-weight`
- `--speed-loss-weight`
- per-epoch logging for `traj_loss`, `route_loss`, and `speed_loss`

Checkpoint config must include:

```text
model_type=navsim_joint_route_speed_diffusion
traj_horizon=8
route_points=50
speed_horizon=8
state_tokens=0
cache_dir
label_dir
stats paths
loss weights
```

### 5. NewHPC bash wrapper

Add a dedicated wrapper:

```text
scripts/train_navsim_joint_route_speed_ddp_b128_template.sh
```

It should follow the existing official4cam DDP template but use officialacc cache and label sidecar by default:

```text
REPO=/workspace1/z_project/code/motdp_z_navsim_motdp
CACHE_DIR=/workspace2/z_project/motdp_bev_cache_train_official4cam_officialacc_npy
LABEL_DIR=/workspace2/z_project/motdp_navsim_labels/navtrain_official_h8_r50_sparsemask
STATS_DIR=/workspace2/z_project/motdp_navsim_norm_stats/navtrain_official_h8_r50_20260518_sparsemask
LOG_ROOT=/workspace2/z_project/motdp_logs
TRAIN_GPUS=0,1,4,5
NPROC_PER_NODE=4
PER_GPU_BATCH=128
EPOCHS=90
LR=1e-4
LR_FINAL=1e-6
```

The wrapper must export:

```text
PYTHONPATH=$REPO:$PYTHONPATH
OPENSCENE_DATA_ROOT=/workspace2/data/navsim
NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps
TMPDIR=/workspace2/z_project/tmp
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
PYTHONUNBUFFERED=1
```

## NewHPC Execution Commands

All commands below run from:

```text
/workspace1/z_project/code/motdp_z_navsim_motdp
```

Environment header:

```bash
source /workspace1/miniconda/etc/profile.d/conda.sh
conda activate z_navsim_motdp
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export OPENSCENE_DATA_ROOT=/workspace2/data/navsim
export NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps
export TMPDIR=/workspace2/z_project/tmp
mkdir -p $TMPDIR
```

Label smoke:

```bash
/workspace1/miniconda/envs/z_navsim_motdp/bin/python \
  scripts/data_tools/build_navsim_joint_labels.py \
  --log-root /workspace2/data/navsim/navsim_logs/trainval \
  --cache-dir /workspace2/z_project/motdp_bev_cache_train_official4cam_officialacc_npy \
  --output-dir /workspace2/z_project/motdp_navsim_labels/navtrain_official_h8_r50_sparsemask_smoke \
  --traj-horizon 8 \
  --route-points 50 \
  --max-tokens 512
```

Full label build:

```bash
tmux new-session -d -s nlabel 'cd /workspace1/z_project/code/motdp_z_navsim_motdp && \
source /workspace1/miniconda/etc/profile.d/conda.sh && conda activate z_navsim_motdp && \
export PYTHONPATH=$PWD:${PYTHONPATH:-} PYTHONUNBUFFERED=1 && \
/workspace1/miniconda/envs/z_navsim_motdp/bin/python \
  scripts/data_tools/build_navsim_joint_labels.py \
  --log-root /workspace2/data/navsim/navsim_logs/trainval \
  --cache-dir /workspace2/z_project/motdp_bev_cache_train_official4cam_officialacc_npy \
  --output-dir /workspace2/z_project/motdp_navsim_labels/navtrain_official_h8_r50_sparsemask \
  --traj-horizon 8 \
  --route-points 50 \
  2>&1 | tee /workspace2/z_project/motdp_navsim_labels/navtrain_official_h8_r50_sparsemask/build.log'
```

Dataset/model smoke:

```bash
/workspace1/miniconda/envs/z_navsim_motdp/bin/python scripts/smoke_navsim_joint_dataset.py \
  --cache-dir /workspace2/z_project/motdp_bev_cache_train_official4cam_officialacc_npy \
  --label-dir /workspace2/z_project/motdp_navsim_labels/navtrain_official_h8_r50_sparsemask \
  --batch-size 4

/workspace1/miniconda/envs/z_navsim_motdp/bin/python scripts/smoke_navsim_joint_model.py \
  --cache-dir /workspace2/z_project/motdp_bev_cache_train_official4cam_officialacc_npy \
  --label-dir /workspace2/z_project/motdp_navsim_labels/navtrain_official_h8_r50_sparsemask \
  --batch-size 2 \
  --device cuda:0
```

One-epoch training smoke:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace1/miniconda/envs/z_navsim_motdp/bin/torchrun \
  --standalone --nnodes=1 --nproc_per_node=1 \
  training/train_navsim_joint_route_speed_ddp.py \
  --cache-dir /workspace2/z_project/motdp_bev_cache_train_official4cam_officialacc_npy \
  --label-dir /workspace2/z_project/motdp_navsim_labels/navtrain_official_h8_r50_sparsemask \
  --log-dir /workspace2/z_project/motdp_logs/navsim_joint_route_speed_smoke \
  --load-mode memmap \
  --epochs 1 \
  --batch-size 16 \
  --max-train-samples 1024 \
  --max-val-samples 256 \
  --traj-stats-path /workspace2/z_project/motdp_navsim_norm_stats/navtrain_official_h8_r50_20260518_sparsemask/navtrain_official_h8_r50_sparsemask_abs_stats.npz \
  --route-stats-path /workspace2/z_project/motdp_navsim_norm_stats/navtrain_official_h8_r50_20260518_sparsemask/navtrain_official_h8_r50_sparsemask_route_abs_stats.npz \
  --speed-stats-path /workspace2/z_project/motdp_navsim_norm_stats/navtrain_official_h8_r50_20260518_sparsemask/navtrain_official_h8_r50_sparsemask_speed_profile_stats.npz \
  --route-loss-weight 1.0 \
  --speed-loss-weight 0.5 \
  --amp-dtype bf16 \
  --use-amp
```

Full training after smoke:

```bash
tmux new-session -d -s nt 'cd /workspace1/z_project/code/motdp_z_navsim_motdp && \
bash scripts/train_navsim_joint_route_speed_ddp_b128_template.sh'
```

## Hard Gates Before Full Training

Full training can start only when all are true:

- `metadata.json` reports `failed=0`
- label count is close to `102608`
- dataset smoke shows matched labels greater than `100000`
- one model backward pass succeeds without NaN
- one-epoch smoke writes `metrics.jsonl`, `best_model.pt`, and `model_epoch1.pt`
- smoke logs contain nonzero finite `traj_loss`, `route_loss`, and `speed_loss`
- no code path imports semantic/state branch modules for the NAVSIM joint model

## Failure Triage

If label count is too low:

- check `cache_index.npz` token count
- check raw log root path
- inspect `tokens_skipped_future_short`, `tokens_skipped_empty_route`, and failed examples

If dataset smoke fails:

- inspect token string dtype in `cache_index.npz` and `label_index.npz`
- verify dedupe happens after label matching
- verify selected split is not empty after filtering

If route loss is NaN:

- check `route_mask.sum(dim=-1).clamp(min=1.0)` is used
- check masked samples with no valid route are filtered out or route loss is zeroed

If speed loss dominates:

- inspect speed normalization stats
- log unnormalized speed prediction mean/std during smoke

If training starts but validation is unstable:

- first reduce `PER_GPU_BATCH` to 64
- then reduce LR to `5e-5`
- keep model architecture unchanged until the data path is verified

## Notes

- This plan is NAVSIM-only.
- Do not update PDMLite / Route-B configs with these stats.
- Do not use old state-token branches in this NAVSIM model. Cleanup means the new NAVSIM joint model has zero state tokens; it does not mean deleting old Route-B files from this branch.
- The first version does not convert route+speed deterministically into traj.
  It trains traj, route, and speed jointly, allowing trajectory tokens to attend
  to route and speed tokens.

## Implementation Status 2026-05-18

Implemented files:

```text
dataset/navsim_cached_dataset.py
scripts/data_tools/build_navsim_joint_labels.py
model/navsim_joint_route_speed_diffusion.py
training/train_navsim_joint_route_speed_ddp.py
scripts/smoke_navsim_joint_dataset.py
scripts/smoke_navsim_joint_model.py
scripts/train_navsim_joint_route_speed_ddp_b128_template.sh
```

Verified on newhpc with `z_navsim_motdp`:

```text
python -m py_compile ...: OK
bash -n scripts/train_navsim_joint_route_speed_ddp_b128_template.sh: OK
synthetic joint model forward/backward: OK
```

Real-data smoke label output:

```text
/workspace2/z_project/motdp_navsim_labels/navtrain_official_h8_r50_sparsemask_smoke16
```

Smoke label metadata:

```text
cache_unique_tokens: 64
logs: 18
tokens_seen_in_logs: 16
tokens_matched: 16
tokens_skipped_future_short: 0
tokens_skipped_empty_route: 0
failed: 0
labels_written: 16
```

Dataset smoke succeeded with real officialacc train cache:

```text
dataset_len=16
bev_grid:      (4, 64, 64, 64), float16
bev_feature:   (4, 512, 8, 8), float16
ego_status:    (4, 4, 8), float32
trajectory:    (4, 8, 2), float32
route:         (4, 50, 2), float32
route_mask:    (4, 50), float32
path:          (4, 50, 2), float32
path_mask:     (4, 50), float32
speed_profile: (4, 8), float32
```

Model smoke succeeded on real cache + smoke labels:

```text
loss=2.425865
finite_grads=True
metrics: l2_err_m, traj_loss, route_loss, speed_loss, route_l2_m, speed_mae_mps
```

One-epoch training smoke succeeded:

```text
log_dir=/workspace2/z_project/motdp_logs/navsim_joint_route_speed_smoke16_dev
epochs=1
batch_size=2
max_train_samples=8
max_val_samples=4
d_model=64
n_layer=1
saved best_model.pt
saved model_epoch1.pt
```

Observed smoke metrics:

```text
train_loss=7.3599
train_l2=6.433m
train_traj_loss=1.3434
train_route_loss=5.8796
train_speed_loss=0.2737
val_loss=1.8613
val_l2=8.953m
```

Next full-training sequence:

1. Build the full label sidecar with `scripts/data_tools/build_navsim_joint_labels.py`.
2. Confirm full `metadata.json` has `failed=0` and label count close to `102608`.
3. Run dataset/model smoke against full label sidecar.
4. Start full training:

```bash
tmux new-session -d -s nt 'cd /workspace1/z_project/code/motdp_z_navsim_motdp && \
bash scripts/train_navsim_joint_route_speed_ddp_b128_template.sh'
```
