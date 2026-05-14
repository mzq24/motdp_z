# NavSim Migration Plan

## Target

Migrate MoT-DP planner (from `semantic_state_next_token_rl_v1` worktree) from Bench2Drive to NavSim for testing.

## Architecture

```
NavSim Agent Interface
        │
        ▼
┌───────────────────────────────┐
│  LEAD TransfuserBackbone       │  ← perception module (pretrained, frozen)
│  (camera + LiDAR → BEV feat)  │
└───────────────┬───────────────┘
                │ BEV features
                ▼
┌───────────────────────────────┐
│  MoT-DP DiT Planner            │  ← our planner
│  (BEV feat → trajectory)      │
└───────────────────────────────┘
```

- LEAD perception: extracts BEV features from NavSim sensor data (camera + LiDAR)
- MoT-DP planner: DiT diffusion model that takes BEV features and outputs ego trajectory
- NavSim labels are compatible (PDM-lite-like structure after processing)

## Key Repos

| Repo | Local | HPC (newhpc) |
|------|-------|--------------|
| MoT-DP (main) | `/media/z/data/mzq/others/MoT-DP/` | `/home/z/code/motdp_z` |
| MoT-DP (worktree) | `/media/z/data/mzq/others/MoT-DP-worktrees/semantic_state_next_token_rl_v1/` | `/home/z/code/motdp_z_semantic_state_next_token_rl_v1/` |
| LEAD | `/media/z/data/mzq/others/lead/` | `/home/z/code/lead` |
| NavSim | `/media/z/data/mzq/others/navsim/` | `/home/z/code/navsim` |
| NavSim Dataset | - | `/workspace2/data/navsim/` |

## Steps

### Step 1: Unified conda env (Python 3.10) ← DONE

- **Env name**: `z_navsim_motdp` (on newhpc)
- **Python**: 3.10.20
- **Key packages**:
  - `torch==2.5.1+cu124` (CUDA 12.4, GPU available)
  - `numpy==1.26.4` (downgraded from 2.x for sklearn 1.2.2 binary compat)
  - `nuplan-devkit==1.2.0` (from git @nuplan-devkit-v1.2)
  - `navsim` (editable install from `/home/z/code/navsim`, --no-deps)
  - `pytorch-lightning==2.2.1`
  - `timm==1.0.27`
  - `diffusers==0.38.0`
  - `transformers==5.8.0`
  - `opencv-python==4.9.0.80`
  - `scikit-learn==1.2.2`
- **LEAD**: editable install from `/home/z/code/lead`
- **Resolved issues**:
  - `pkg_resources` missing → setuptools downgraded to 69.5.1
  - numpy 2.x sklearn binary incompat → numpy 1.26.4
- **Verified imports**: torch, navsim, nuplan, pytorch_lightning, timm, diffusers, transformers, cv2, einops, jaxtyping, beartype, TransfuserBackbone, TransfuserAgent, TransfuserModel, TransfuserConfig
- **Dataset**: `/workspace2/data/navsim/` (2.4TB, accessible)

### Step 2: Get LEAD running on NavSim data ← DONE (smoke test)

- **Checkpoint**: `ln2697/tfv6_navsim` (HF) → downloaded to `/workspace1/z_project/models/navsim_backbones/tfv6_navsim/`
  - `model_0060.pth` (238MB), `config.json`, standalone `ltfv6.py`
- **Architecture**: LTFv6 = Latent TransFuser v6, camera-only (4 cams: f0/l0/r0/b0), no LiDAR
  - Config: `LTF=True`, `resnet34`, `num_fusion_stages=4`, `bf16 mixed precision`
- **Smoke test verified shapes** (synthetic input, forward pass OK):
  - Input `rgb`: `(B, 3, 270, 1920)` — 4 cameras stitched
  - Backbone `lidar_features`: `(B, 512, 8, 8)` — flattened BEV tokens
  - Backbone `image_features`: `(B, 512, 9, 60)` — perspective features
  - **`top_down()` BEV grid**: `(B, 64, 64, 64)` ✅ **matches MoT-DP's `transfuser_bev_feature_upsample`**
  - Full model output: `(B, 8, 2)` waypoints + `(B, 8)` headings
- **Issues resolved**:
  - bf16 mixed precision: monkey-patched backbone.forward to fix LTF grid dtype
  - Channel mismatch (512 vs 1512): deferred to Step 5 (add linear adapter or modify MoT-DP)
- **Script**: `smoke_test_lead_backbone.py` on newhpc
- **Next**: Step 2b — run with real NavSim sensor data (AgentInput → camera stitching → backbone)

### Step 3: Extract LEAD perception module

- Isolate `TransfuserBackbone` from LEAD
- Create adapter: NavSim AgentInput → LEAD backbone input format (4-camera stitch at 270×1920)
- Verify BEV output dimensions match what MoT-DP planner expects

### Step 4: Process NavSim labels

- Convert NavSim labels to PDM-lite-compatible structure
- Verify trajectory format, route format, speed hist etc.
- Ensure training dataset can produce same outputs as Bench2Drive dataset

### Step 5: Code adaptation

- Implement NavSim agent wrapping LEAD perception + MoT-DP planner
- Adapt training loop (NavSim uses PyTorch Lightning)
- Adapt dataloader for NavSim data
- Write config for NavSim paths and settings
