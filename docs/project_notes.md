# MoT-DP 项目笔记

## 项目概览
- 自动驾驶轨迹规划项目
- 使用 TransFuser backbone 提取 BEV 特征
- 基于 Diffusion (DiT 架构) 的轨迹预测
- 参考项目: BridgeDrive, DiffusionDrive, LEAD, carla_garage

## 重要文件路径
| 路径 | 说明 |
|------|------|
| `model/transfuser_extractor/` | TransFuser backbone 提取器 (来自 carla_garage) |
| `model/transformer_for_diffusion_multi_head.py` | DiT 模型 |
| `policy/diffusion_dit_carla_policy.py` | Diffusion policy |
| `training/train_carla_bev.py` | 训练脚本 |
| `dataset/unified_carla_dataset.py` | 数据集 |
| `BridgeDrive/` | BridgeDrive 参考代码 (仅 LEAD 适配) |
| `DiffusionDrive/` | DiffusionDrive 参考代码 |

---

## TransFuser Backbone 对比

### 结论: 同一架构，不同封装

我们的 `model/transfuser_extractor/transfuser.py` 和 BridgeDrive/LEAD 的 `TransfuserBackbone` 都来自 carla_garage。

### 相同的核心结构:
- **双编码器**: `image_encoder` (timm) + `lidar_encoder` (timm/VideoResNet/SwinTransformer3D)
- **4层 GPT Transformer 融合**: avgpool 降采样 → channel 对齐 → Transformer attention → 插值回原尺寸 → 残差相加
- **FPN top_down**: c5_conv → up_conv5 + upsample → up_conv4 + upsample2 → 输出 p3
- **输出**: x4 (融合后的原始 lidar 特征), p3 (FPN 上采样后), fused_features (全局池化拼接)

### 封装差异:
- **我们的** (`backbone_extractor.py`): `_forward_with_intermediate()` 在一次调用中同时返回 x4 和 p3
- **BridgeDrive/LEAD** (`tfv6_bridgedrive.py`): `backbone(data)` 返回 (bev_features, image_features)，然后外层再调 `backbone.top_down(bev_features)` 得到 p3

### BridgeDrive 中的 BEV 语义分割:
- 使用 `BEVDecoder` 对 `bev_feature_grid` (即 top_down 输出的 p3) 进行 BEV 语义分割预测
- 推理时用的是**模型预测的 BEV**，不是 GT label
- 训练可视化时 GT 和预测的 BEV 都可用

---

## BEV 可视化流程 (BridgeDrive / LEAD 框架)

文件: `BridgeDrive/BridgeDrive_adaptation_LEAD/lead/visualization/visualizer_bridgedrive.py`

### Pipeline:
1. **底图**: LiDAR 光栅化 → 归一化 → 颜色插值 (白色到 LIDAR_COLOR) → 上采样
2. **BEV 语义叠加** (`_bev_semantic`):
   - GT 模式: `data["bev_semantic"]`
   - 预测模式: `predictions.pred_bev_semantic.argmax(dim=1)` (来自 BEVDecoder，输入是 TransFuser 特征)
   - 颜色映射: `CARLA_TRANSFUSER_BEV_SEMANTIC_COLOR_CONVERTER`
   - Alpha blending: 背景=0, 道路=0.15, 其他=0.33
3. **标注绘制**: 路线、目标点 (带编号的圆圈)、航点、自车边框、其他车辆边框、雷达点
4. **旋转 90°** (`np.rot90(k=1)`)
5. **拼接**: 与相机视角 (RGB/语义纵向排列) + meta 信息面板

### 入口函数:
| 函数 | BEV 来源 | 使用场景 |
|------|---------|---------|
| `visualize_training_labels()` | GT BEV semantic | 训练时查看标签 |
| `visualize_training_prediction()` | 模型预测 BEV semantic | 训练时查看预测 |
| `visualize_inference_prediction()` | 模型预测 BEV semantic | 推理时 (sensor_agent_bridgedrive.py:688) |
