# NuPlan White-Noise Diffusion Backbone

## 动机

从 `semantic-state-next-token-rl-v1` branch 删减 CARLA 专用代码，构建简洁的 white-noise diffusion backbone 用于 nuPlan（纯向量、ego-centric）。借鉴 Diffusion-Planner（ICLR 2025 Oral）的架构设计。

## 分支与环境

- **分支**: `nuplan_whitenoise_diffusion_v1` (base: `semantic-state-next-token-rl-v1`)
- **本地 worktree**: `MoT-DP-worktrees/nuplan_whitenoise_diffusion_v1/`
- **HPC 代码**: `/home/z/code/nuplan_whitenoise_diffusion_v1/`
- **HPC 环境**: `z_navsim_motdp` (`/workspace1/miniconda/envs/z_navsim_motdp/`)
- **数据**: 从 PlanTF cache 转换 (`/workspace2/z_project/exp/nuplan/cache_plantf_*`)
- **输出 cache**: `/workspace2/z_project/exp/nuplan/cache_motdp_v1/`

## 架构

```
输入 (ego-centric 向量) → SceneEncoder → (B, 67, 192) scene tokens
                                         ↓
noisy traj (B, 11, 324) → DiT (adaLN-Zero + Cross-Attn) → x0 (B, 11, 81, 4)
                                         ↑
               route_enc + t_emb → global conditioning y
```

### 核心简化

| 移除 | 保留/新增 |
|------|----------|
| BEV/camera/LiDAR (TransFuser) | Vector-based SceneEncoder（借鉴 DP） |
| ~30 semantic heads, energy guidance | 纯 diffusion L1 loss (ego + neighbor) |
| Multi-mode anchors (32 modes) | Single-mode from N(0,I) |
| CARLA dataset | nuPlan data via PlanTF cache 转换 |
| Route prediction head | Route conditioning（融入 DiT adaLN） |
| Branch conditioning, phase-go smoothing | DDIM scheduler (predict x0, 1000→10 steps) |

### 模型参数: 5.5M

## 文件结构

```
model/
  scene_encoder.py            Agent/Lane/Static/Fusion encoders (MLP-Mixer + SA)
  dit.py                      DiTBlock (adaLN-Zero + CrossAttn), RouteEncoder, FinalLayer
  nuplan_diffusion_model.py   SceneEncoder + DiT 组装

policy/
  nuplan_diffusion_policy.py  DDIM scheduler, L1 loss, inference, normalization

dataset/
  nuplan_data_process.py      PlanTF cache → .npz 转换
  nuplan_dataset.py           PyTorch Dataset

training/
  train_nuplan.py             DDP training (EMA/AMP/AdamW/cosine)

configs/
  nuplan_diffusion.yaml       完整配置

scripts/
  compute_norm_stats.py       计算 z-score 统计量
  train_nuplan.sh             一键启动训练
```

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Scheduler | DDIM (x0 prediction) | 复用 diffusers infra，比 DP 的 VPSDE+DPM-solver 简单 |
| 预测模式 | joint ego + N neighbors | 与 DP 一致，P=11 (1 ego + 10 neighbors) |
| 归一化 | global z-score | 简洁，shared across agents |
| Loss | L1 | 比 MSE 鲁棒 |
| Scene encoder | DP 式 MLP-Mixer + SA fusion | 已验证有效 |
| DiT | adaLN-Zero + CrossAttn | 与 DP 架构一致 |
| 数据来源 | PlanTF cache 转换 | 避免重新读 DB，~76KB/场景 |

## 训练配置

```yaml
hidden_dim: 192, num_heads: 6, encoder_depth: 2, decoder_depth: 3
future_len: 80 (8s@10Hz), agent_num: 32, predicted_neighbor_num: 10
num_train_timesteps: 1000, num_inference_steps: 10
batch_size: 64, lr: 5e-4, epochs: 500, optimizer: AdamW + cosine
```

## 使用流程

```bash
# 1. 数据转换（PlanTF cache → .npz）
python dataset/nuplan_data_process.py --all_splits \
    --save_path /workspace2/z_project/exp/nuplan/cache_motdp_v1

# 2. 计算归一化统计量
python scripts/compute_norm_stats.py \
    --data_dir /workspace2/z_project/exp/nuplan/cache_motdp_v1

# 3. 训练
bash scripts/train_nuplan.sh
```

## 参考

- Diffusion-Planner: `/home/z/code/Diffusion-Planner/` (newhpc)
- 源分支: `semantic-state-next-token-rl-v1`
- nuPlan 数据: `/workspace2/z_project/dataset/nuplan/nuplan-v1.1/trainval/`
- z_navsim_motdp: `/workspace1/miniconda/envs/z_navsim_motdp/`
