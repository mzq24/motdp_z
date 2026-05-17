# NAVSIM LEAD-Aligned Cache Rebuild And Validation Plan

## Summary

当前已确认 MoT-DP 的 LEAD preprocessing 与 official LEAD NAVSIM v1.1 4-cam path 做到 byte-level parity。下一阶段目标是用这套 official preprocessing 重建 BEV cache，并重新验证 cached diffusion 在 navtest 上的真实表现。

计划文档路径:

```text
docs/navsim_lead_aligned_cache_rebuild_plan.md
```

## Current Ground Truth

- official LEAD NAVSIM path 是 4-cam: `[L0, F0, R0, B0]`。
- preprocessing 固定为: stitch 4-cam full images -> whole image `/4` resize -> JPEG Q30 -> OpenCV `IMREAD_COLOR` decode -> `(1, 3, 270, 1920)` tensor。
- `scripts/compare_preprocess.py` 已验证 MoT-DP helper 和 LEAD v1.1 official `TransfuserFeatureBuilder._get_camera_feature` 在 mini token 上 byte-level identical。
- 旧 cache `/workspace2/z_project/motdp_bev_cache_navtest_npy` 不是 LEAD-aligned cache，不再作为最终结论依据。

## Cache Paths

- navtest smoke/subset NPZ:

```text
/workspace2/z_project/motdp_bev_cache_navtest_official4cam_smoke
```

- navtest smoke/subset NPY/memmap:

```text
/workspace2/z_project/motdp_bev_cache_navtest_official4cam_smoke_npy
```

- navtest full NPZ:

```text
/workspace2/z_project/motdp_bev_cache_navtest_official4cam
```

- navtest full NPY/memmap:

```text
/workspace2/z_project/motdp_bev_cache_navtest_official4cam_npy
```

- train cache, later stage:

```text
/workspace2/z_project/motdp_bev_cache_train_official4cam
```

## Implementation Plan

1. Documentation

- Keep this plan as the canonical rebuild/validation runbook.
- Add a short pointer in `docs/session_0515_precompute.md` after the preprocessing parity notes.
- Every smoke/full run should append: command, cache path, tmux name if any, CSV path, success/failure count, average score, DAC=0 count.

2. Cache Smoke

- Use `scripts/precompute_navtest.py` with `--limit_pkls` or a token list to generate a small official4cam navtest NPZ cache.
- Confirm output token count and array shapes for `bev_grid`, `bev_feature`, `ego_status`, `trajectory`, and `tokens`.
- Convert smoke NPZ to NPY/memmap using `scripts/convert_navsim_cache_to_npy.py`.
- Confirm NPY view contains `cache_index.npz`, `bev_grid.npy`, and `ego_status.npy`, and that token lookup works.

3. Cached PDM Smoke

- Use checkpoint:

```text
/workspace2/z_project/motdp_logs/navsim_simple_official_lr1e4_e60_b64/best_model.pt
```

- Run official PDM `navtest smoke16` with cache path pointing to the official4cam smoke NPY cache.
- Use `fallback=raise` so missing tokens are not silently hidden.
- Success criteria: 16 scenarios valid, CSV generated, no missing token, no obvious DAC/NC systematic failure.

4. Full Navtest

- After smoke passes, build full navtest official4cam NPZ cache.
- Convert full cache to NPY/memmap.
- Run full navtest `12,146` scenarios, initially single worker for stability.
- Record CSV path, average score, successful/failed scenario count, DAC=0 count, and failed token list if any.

5. Failure Triage

If full cached score remains far below official LEAD baseline:

- First check cache/input parity on failed tokens: image tensor stats, `bev_feat`, and `top_down` BEV stats.
- If DAC=0 cases show global y flip or offset, prioritize coordinate-frame conversion.
- If DAC=0 cases are mostly turning shortcuts, prioritize trajectory smoothing / route following.
- If failures cluster around junction lane connectors, prioritize NAVSIM metric/lane connector sensitivity and route/centerline constraints.
- Only after cache/input parity is ruled out, debug diffusion planner label/target alignment.

## Commands To Use

Static compile:

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
/workspace1/miniconda/envs/z_navsim_motdp/bin/python -m py_compile \
  navsim_motdp/lead_preprocessing.py \
  navsim_motdp/agents/online_lead_agent.py \
  navsim_motdp/agents/pure_lead_agent.py \
  scripts/precompute_bev_cache.py \
  scripts/precompute_navtest.py \
  scripts/compare_preprocess.py
```

Preprocessing parity:

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
TMPDIR=/workspace2/z_project/tmp \
OPENSCENE_DATA_ROOT=/workspace2/data/navsim \
NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps \
/workspace1/miniconda/envs/z_navsim_motdp/bin/python scripts/compare_preprocess.py
```

Cache smoke skeleton:

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
OPENSCENE_DATA_ROOT=/workspace2/data/navsim \
NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps \
/workspace1/miniconda/envs/z_navsim_motdp/bin/python scripts/precompute_navtest.py \
  --limit_pkls 1 \
  --cache_dir /workspace2/z_project/motdp_bev_cache_navtest_official4cam_smoke
```

NPY conversion skeleton:

```bash
cd /workspace1/z_project/code/motdp_z_navsim_motdp
/workspace1/miniconda/envs/z_navsim_motdp/bin/python scripts/convert_navsim_cache_to_npy.py \
  --cache_dir /workspace2/z_project/motdp_bev_cache_navtest_official4cam_smoke \
  --output_dir /workspace2/z_project/motdp_bev_cache_navtest_official4cam_smoke_npy
```

## Test Plan

- Static: py_compile passes for preprocessing, agents, precompute, conversion, and compare scripts.
- Preprocessing parity: compressed bytes equal `True`; decoded/tensor diff `0`。
- Cache smoke: first token can be loaded from NPY/memmap; tensor shapes and dtypes match expected contract.
- Agent smoke: `CachedDiffusionAgent` initializes with `fallback=raise` and returns `Trajectory.poses.shape == (8, 3)` for a cached token.
- PDM smoke16: 16/16 valid, CSV generated, no missing token.
- Full navtest: 12,146 scenarios attempted; CSV includes `average_all_frames`; failed tokens are recorded.

## Assumptions

- Do not retrain before cache/input validation is complete.
- `z_navsim_motdp` is the default environment.
- Official LEAD checkpoint baseline remains the comparison anchor.
- Single worker full PDM is preferred first for reliability; multi-worker optimization can wait until the cache path is trusted.


---

## Execution Log: 2026-05-16

### Step 0: Pre-check
- All scripts exist: compare_preprocess.py, precompute_navtest.py, convert_navsim_cache_to_npy.py
- lead_preprocessing.py module exists
- Static compile: compare OK, precompute OK

### Step 1: Preprocessing Parity ✅
- compare_preprocess.py: byte-level identical
- Token: ea582d733c455f3a (mini)
- compressed bytes equal: True
- tensor max_abs=0.000000, mean_abs=0.000000

### Step 2: Smoke Cache (NPZ) ⚠️
- 2 pkls -> 79 tokens (too small for PDM overlap)
- 20 pkls, 4 shards -> 434 tokens (still too small)
- Conclusion: need full navtest cache for PDM smoke to work

### Step 3: Full Cache Build 🔄
- tmux: o4cam_full
- Cache path: /workspace2/z_project/motdp_bev_cache_navtest_official4cam
- Script: _run_official4cam_full.sh
- 4 GPUs (0,1,2,3), 136 pkls, ~12K tokens
- ETA: ~30 min

### Pending
- Convert full NPZ to NPY
- Run smoke16 (then full navtest) on official4cam NPY cache
- Record scores


### Step 4: Smoke16 on official4cam Cache

experiment: motdp_official4cam_smoke16
CSV: 2026.05.16.18.39.02/2026.05.16.18.39.12.csv
16/16 successful, **score: 0.738**

| Metric | Value |
|---|---|
| NC | 0.8125 |
| DAC | 0.875 |
| DDC | 1.0 |
| TLC | 1.0 |
| EP | 0.822 |
| TTC | 0.875 |
| LK | 0.938 |
| HC | 1.0 |
| EC | 0.7 |

### Key Finding: Train/Test Mismatch

Smoke16 score (0.738) did NOT improve over old cache (0.746).
Reason: diffusion model was trained on OLD cache (4-cam, NO JPEG).
New official4cam cache uses JPEG Q30 preprocessing -> different BEV feature distribution.
Model never saw JPEG-compressed features during training.

This is NOT a preprocessing bug - it is train/test mismatch.

### Path Forward

Option A: Retrain diffusion on official4cam cache -> fair eval
Option B: Keep old cache as baseline (0.751), advance to Stage 2
Option C: Run full navtest on both for statistical comparison before deciding

### Updated Score Table

| Agent | Cache | Score | Notes |
|---|---|---|---|
| CachedDiffusion | OLD (4-cam, noJPEG) | 0.751 (full) / 0.746 (smoke16) | Training cache |
| CachedDiffusion | NEW (4-cam, JPEG Q30) | 0.738 (smoke16) | **Train/test mismatch** |
| OnlineLeadDiffusion (FIXED) | N/A (online) | 0.856 (smoke8) | 3-cam, JPEG Q30 |
| LEAD Official | N/A (online) | 0.817 (smoke16) | Reference baseline |


### Step 5: Cached LEAD Planner Validation ✅

experiment: motdp_cached_lead_official4cam_smoke16
Agent: CachedLeadAgent (LEAD planning_decoder + official4cam NPY cache)
16/16 successful, **score: 0.943**

| Metric | Value |
|---|---|
| NC | **1.0** |
| DAC | **1.0** |
| DDC | **1.0** |
| TLC | **1.0** |
| TTC | 1.0 |
| LK | 1.0 |
| HC | 1.0 |
| EC | 0.8 |
| EP | 0.87 |

**All multiplier metrics are perfect (1.0).** Zero collisions, zero off-road.

### Cache Validation Conclusion

| Agent | Cache | Score | Notes |
|---|---|---|---|
| CachedLeadAgent | official4cam | **0.943** | LEAD planner on our cache |
| LEAD Official (CarlaTF, online) | N/A | 0.817 | Online reference |
| CachedDiffusionAgent | official4cam | 0.738 | Diffusion planner (train/test mismatch) |
| CachedDiffusionAgent | OLD cache | 0.751 | Diffusion planner (train/test consistent) |

**Cache is validated**: LEAD planning_decoder achieves 0.943 using our official4cam cache, exceeding the online LEAD official score (0.817). The gap between CachedDiffusion (0.738) and CachedLeadAgent (0.943) is entirely from the planner — diffusion model needs retraining on the official4cam cache.

### Next Action

1. Retrain diffusion on official4cam train cache (need to rebuild train cache with same preprocessing first)
2. Target: close the 0.738 → 0.943 gap
3. Then add route/speed decoupling for Stage 2 improvements

### Step 6: Same-Token Comparison (LEAD Official vs CachedLeadAgent)

Same 16 navtest tokens, same LEAD planner:

| Token | LEAD Official | CachedLead | diff |
|---|---|---|---|
| afbf26b6d3bb5bee | 0.000 (crash) | 1.000 | +1.000 |
| dfa220d6e64f5d84 | 0.000 (crash) | 1.000 | +1.000 |
| d58c4ad27c525465 | 1.000 | 0.875 | -0.125 |
| 9dc5a17094e0569d | 0.928 | 0.803 | -0.125 |
| ... others ... | within 0.02 | within 0.02 | |

**Excluding 2 crash tokens**: LEAD Official 0.934 vs CachedLead 0.934.
**Conclusion**: Cache produces identical results to online LEAD backbone. Cache validated.

### Current State

| Cache | Split | Preprocessing | Status |
|---|---|---|---|
| OLD train | navtrain 103K | 4-cam, NO JPEG | Used for diffusion training |
| OLD val | navtest 12K | 4-cam, NO JPEG | Previous cached PDM |
| **NEW val** | navtest 12K | 4-cam, JPEG Q30 | ✅ Built, validated |
| NEW train | navtrain 103K | 4-cam, JPEG Q30 | ❌ NOT YET BUILT |

### Next: Build official4cam training cache

Need to rebuild navtrain BEV cache with aligned preprocessing.
Then retrain diffusion for fair train/test distribution.


---

## Session 2026-05-16 (cont): Training Pipeline Improvements

### Training Code Changes

- Added  per-batch progress bars (loss, l2, lr)
- Added optional Usage: wandb [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  agent         Run the W&B agent
  artifact      Commands for interacting with artifacts
  beta          Beta versions of wandb CLI commands.
  controller    Run the W&B local sweep controller
  disabled      Disable W&B.
  docker        Run your code in a docker container.
  docker-run    Wrap  and adds WANDB_API_KEY and WANDB_DOCKER...
  enabled       Enable W&B.
  init          Configure a directory with Weights & Biases
  job           Commands for managing and viewing W&B jobs
  launch        Launch or queue a W&B Job.
  launch-agent  Run a W&B launch agent.
  launch-sweep  Run a W&B launch sweep (Experimental).
  login         Login to Weights & Biases
  offline       Disable W&B sync
  online        Enable W&B sync
  pull          Pull files from Weights & Biases
  restore       Restore code, config and docker state for a run
  scheduler     Run a W&B launch sweep scheduler (Experimental)
  server        Commands for operating a local W&B server
  status        Show configuration settings
  sweep         Initialize a hyperparameter sweep.
  sync          Upload an offline training directory to W&B
  verify        Verify your local instance logging ()
- Added  CLI for all config (, , , , , etc.)
- Added ADE/FDE computation during validation (DDIM sampling, 1024 subsample)
- Added  checkpoint support
- Fixed non-tensor keys in batch (token, log_name, frame_idx) when

### Norm Stats

Full 103K token trajectory stats (mmap, 0.1s):
- train_mean: [10.17, 0.36]
- train_std:  [8.80, 2.28]
- navtest_mean: [11.11, 0.33]
- navtest_std:  [9.37, 2.40]
- Saved to: traj_norm_stats.npz in cache dir
- Note: dataset sampled 4096 stats are identical to full stats

### Training Smoke Test

- Cache: navtest official4cam NPY (12K tokens)
- Epoch 1: loss=0.286, val_loss=0.093, ADE=2.17m, FDE=3.84m
- Epoch 2: loss=0.083, l2=1.45m
- Checkpoint: /workspace2/z_project/motdp_logs/train_smoke_v2/best_model.pt (399MB)
- Pipeline verified: tqdm, val ADE/FDE, checkpoint save all work

### Train Cache Build

- 12 processes on 6 GPUs (0,1,4,5,6,7), 2 per GPU
- 103,288 navtrain official tokens split into 12 chunks (~8,600 each)
- Using trainval_full sensor data (1.6TB)
- tmux: train_o4cam
- Status: ~21% complete, ETA 3-4 hours

### Pending

- Auto-chain script: convert full train cache NPZ -> NPY -> start training
- Record final train cache token count and stats
### Training Pipeline Improvements

File: training/train_navsim_diffusion.py

Changes from v1:
- Added tqdm per-batch progress bars
- Added optional wandb logging via --use_wandb flag
- Added argparse CLI for all hyperparameters
- Added ADE/FDE computation during validation (DDIM sampling)
- Added --resume checkpoint support
- Fixed non-tensor batch keys (token, log_name, frame_idx) for device transfer

### Norm Stats (full 103K tokens)

- train mean=[10.17, 0.36], std=[8.80, 2.28]
- navtest mean=[11.11, 0.33], std=[9.37, 2.40]
- Saved to traj_norm_stats.npz per cache dir
- Dataset sampled stats (4096) are identical to full stats

### Training Smoke Test

- Cache: navtest official4cam NPY (12K tokens)
- Epoch 1: loss=0.286, val_loss=0.093, ADE=2.17m, FDE=3.84m
- Epoch 2: loss=0.083, l2=1.45m
- Checkpoint: train_smoke_v2/best_model.pt (399MB)
- Pipeline verified: tqdm bars, val ADE/FDE, checkpoint save

### Train Cache Build

- 12 processes on 6 GPUs (GPUs 0,1,4,5,6,7), 2 per GPU
- 103,288 navtrain official tokens split into 12 chunks
- Using trainval_full sensor data
- tmux: train_o4cam
- Status: in progress

### Pending

- Auto-chain: full train cache done -> NPY convert -> start full training
- Record final train cache token count
