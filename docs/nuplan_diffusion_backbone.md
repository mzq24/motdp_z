# NuPlan White-Noise Diffusion Backbone

**Branch**: `nuplan_whitenoise_diffusion_v1` (from `semantic-state-next-token-rl-v1`)  
**Date**: 2026-05-14  
**Environment**: `z_navsim_motdp` (`/workspace1/miniconda/envs/z_navsim_motdp/`)  
**HPC path**: `/home/z/code/nuplan_whitenoise_diffusion_v1/`

## Summary

从 `semantic-state-next-token-rl-v1` 分支删减而来，构建了一个纯向量式 white-noise diffusion backbone，用于 nuPlan 数据集。借鉴 Diffusion-Planner 的 ego-centric 数据处理和 DiT 架构，去掉了所有 CARLA/BEV/camera/LiDAR 代码。

## Architecture

```
输入 (ego-centric 向量) → SceneEncoder → (B, 67, 192) scene tokens
                                          ↓
noisy traj (B, 11, 324) → DiT (adaLN-Zero + Cross-Attn) → 预测 x0 (B, 11, 81, 4)
                                          ↑
            route_enc + t_emb → global conditioning y
```

- **SceneEncoder**: Agent(MLP-Mixer) + Static(MLP) + Lane(MLP-Mixer) → Fusion Self-Attn
- **DiT**: 3 blocks, adaLN-Zero self-attn + cross-attn to scene tokens, 192-dim, 6 heads
- **Output**: Joint ego + 10 neighbors, 80 future steps, (x,y,cos,sin) per step
- **Loss**: L1 over predicted x_start (DDIM), split ego + neighbor terms
- **Params**: 5.5M

## What was removed vs original branch

| Removed | Kept |
|---------|------|
| TransFuser BEV / Camera / LiDAR | DiT decoder skeleton |
| ~30 semantic state heads | DDIM scheduler |
| Energy guidance | Normalization framework |
| Anchor multi-mode | DDP / EMA / AMP infra |
| CARLA dataset | HistoryEncoder |
| GridSampleCrossBEVAttention | TrajectoryMLPHead |

## What was added (from Diffusion-Planner)

| Component | Source |
|-----------|--------|
| Vector SceneEncoder (Agent/Lane/Static/Fusion) | DP's `encoder.py` |
| DiTBlock (adaLN-Zero + cross-attn) | DP's `dit.py` |
| RouteEncoder (global conditioning) | DP's `decoder.py` |
| Ego-centric coordinate transforms | DP's `data_process.py` |
| Joint ego+neighbor prediction | DP's loss/dataset |

## Data pipeline

直接读取 PlanTF cache（`cache_plantf_*`），无需 DB 预处理。
- PlanTF cache 已是 ego-centric，直接转换 tensor 格式
- 4 个 cache 目录：singapore/boston/pittsburgh（train）+ val
- 每场景 ~76KB（内存中），无需磁盘额外开销

## Key files

```
model/
  scene_encoder.py             Vector scene encoder (Agent/Lane/Static/Fusion)
  dit.py                        DiT decoder blocks + RouteEncoder
  nuplan_diffusion_model.py     Combined model (SceneEncoder + DiT)

policy/
  nuplan_diffusion_policy.py    DDIM scheduler, L1 loss, inference, AdamW

dataset/
  nuplan_dataset.py             Direct PlanTF cache loader
  nuplan_data_process.py        (unused) PlanTF→.npz converter

training/
  train_nuplan.py               DDP training (EMA/AMP/cosine schedule)

configs/
  nuplan_diffusion.yaml         配置
```

## Training

```bash
ssh new_hpc
cd /home/z/code/nuplan_whitenoise_diffusion_v1
bash scripts/train_nuplan.sh
```

Config: 8x GPU, batch_size=64/GPU, lr=5e-4, AdamW, 500 epochs, DDIM 1000 train / 10 inference steps.

## Design decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Scheduler | DDIM (x0 prediction) | Reuses existing `diffusers` infra |
| Prediction | Joint ego + N neighbors | Scene-consistent predictions |
| Normalization | None (identity) | PlanTF data already normalized |
| Scene encoder | DP-style MLP-Mixer + SA | Proven effective |
| Data loading | Direct from PlanTF cache | No disk overhead, no preprocessing needed |
