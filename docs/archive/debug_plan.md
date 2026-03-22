# 三版本代码参考文档

## 目标
在 mini dataset 上对比三个版本，找出 main project 收敛慢的原因。
- Main project: L2 1s ~0.3
- DD Baseline: L2 1s ~0.29
- Server version: L2 1s <0.2（full dataset）

## 文件结构

### Main Project
| 文件 | 路径 |
|------|------|
| 训练 | `training/train_carla_bev.py` |
| 策略 | `policy/diffusion_dit_carla_policy.py` |
| 模型 | `model/transformer_for_diffusion_multi_head.py` |
| 数据 | `dataset/unified_carla_dataset.py` |
| 配置 | `config/pdm_local.yaml` |
| 启动 | `training/train_local.sh` |

### DD Baseline（DiffusionDrive 复刻，结构不同，作为性能参考）
| 文件 | 路径 |
|------|------|
| 训练 | `dd_baseline/train.py` |
| 策略 | `dd_baseline/policy.py` |
| 模型 | `dd_baseline/model.py` + `dd_baseline/trajectory_head.py` |
| 辅助模块 | `dd_baseline/modules/` (loss, blocks, modulation, refinement) |
| 数据 | 复用 main 的 `dataset/unified_carla_dataset.py` |
| 配置 | `dd_baseline/dd_config.yaml` |
| Anchor | `dd_baseline/anchors/carla_kmeans_20.npy` |

### Server Version (zd_dp)（已收敛，单模态，anchor 从上游 VLM 传入）
| 文件 | 路径 |
|------|------|
| 训练 | `server_folder/zd_dp/training/train_carla_bev.py` |
| 策略 | `server_folder/zd_dp/policy/diffusion_dit_carla_policy.py` |
| 模型 | `server_folder/zd_dp/model/transformer_for_diffusion_multi_head.py` |
| 数据 | `server_folder/zd_dp/dataset/unified_carla_dataset.py` |
| 原始配置 | `server_folder/zd_dp/config/pdm_server.yaml` |
| Preprocess | `server_folder/zd_dp/dataset/preprocess_pdm_lite.py` |

注意: `server_folder/MoT-DP/` 是之前在 server 上的 **delta 实验版本**，不是正确的 server version。

## 关键架构差异

| 特性 | dd_baseline | main project | server (zd_dp) |
|------|------------|--------------|----------------|
| **模态** | 多模态 (20 modes) | 多模态 (32 modes) | **单模态** |
| **Anchor来源** | kmeans_20.npy 文件 | kmeans_32.npy 文件 | **batch 数据 (VLM pred_traj)** |
| **Anchor fallback** | 无 | 无 | GT trajectory |
| **分类 loss** | focal loss (cls=10) | focal loss (cls=10) | **无 (单模态无需分类)** |
| **回归 loss** | L1 (reg=8) | L1 (reg=8) | L1 |
| **Diffusion** | 加性噪声 | 加性噪声 | **乘性噪声** |

## 关键参数对比

| 参数 | dd_baseline | main (local) | server (zd_dp 原始) |
|------|------------|--------------|---------------------|
| 坐标空间 | abs | abs | abs |
| LR | 1e-4 | 1e-4 | 1e-5 |
| Weight Decay | 1e-4 | 1e-4 | 1e-5 |
| Decoder Layers | 2 (traj) + 3 (tf) | 4 | 16 |
| Heads | 8 | 8 | 16 |
| Hidden Dim | 256 | 512 | 1024 |
| trunc_timesteps | 50 | 20 | 8 (infer) / 50 (train) |
| eta | - | 0.0 | 1.0 |
| route_loss | 无 | 0.6 | 0.5 |
| Norm X (off,range) | (5,80)* | (5,80) | (16,92) |
| Norm Y (off,range) | (25,55)* | (25,55) | (45,88) |
| Batch Size | 128 | 256 | 128 |
| Epochs | 200 | 500 | 1000 |
| Attention | Standard MHA | RoPE+RMSNorm+QKNorm | RoPE+RMSNorm+QKNorm |
| Route Prediction | 无 | 有 | 有 |

*dd_baseline norm 已对齐到 main

## VQA Anchor 数据
- Mini dataset pkl 中 `vqa` 字段已 patch，指向 `dp_vl_feature/XXXX.pt`
- 每个 .pt 包含: `pred_traj (1,6,2)`, `reasoning_feat (8,2560)`, `dp_vit_feat (128,2560)`
- train: 18936 samples, val: 2103 samples

## Mini Dataset 验证策略

1. **DD Baseline**: 用 `dd_config.yaml`（norm 已对齐 main），多模态 kmeans anchor
2. **Main Project**: 已跑过，wandb 历史数据
3. **Server Version (zd_dp)**: 需要创建 mini config，单模态 + VLM anchor

**注意**: zd_dp 是单模态架构，和 main/dd_baseline 的多模态架构不同，不能直接对比收敛速度。但可以作为"VLM anchor + 单模态"的性能参考上界。

## 待排查的代码差异

1. main vs dd_baseline 的 policy forward pass（两者都是多模态，更具可比性）
2. model transformer 实现差异
3. loss 计算逻辑
4. anchor matching 策略
