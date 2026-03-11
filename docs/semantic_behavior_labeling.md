# Semantic Behavior Labeling System

## Overview

为每个 anchor trajectory (32条) 生成语义行为标签，描述该轨迹在当前场景下的行为类别和是否被允许。这些标签可以用于：
- 训练时作为 auxiliary loss 监督 mode selection
- 推理时过滤 forbidden modes
- 分析 anchor 分布的合理性和场景覆盖情况

## 核心文件

| 文件 | 作用 |
|------|------|
| `tools/anchor_semantic_labeler.py` | 标签生成核心逻辑 (950行) |
| `dataset/unified_carla_dataset.py` L501-605 | Dataset 中 on-the-fly 调用 labeler |
| `training/train_carla_bev.py` | 加载 anchor_centers_abs 并传入 dataset |

## 数据来源 (三层信息融合)

标签生成采用三层信息源，按优先级排列：

### 1. Future Frame Boxes (最高优先级, simlingo-style)
- 来源: `boxes/{frame_id}.json.gz` + `measurements/{frame_id}.json.gz`
- 原理: 对 anchor 的第 k 个 waypoint，检查 t+k 帧时该位置是否有车辆/行人
- 需要 `ego_matrix` 做坐标变换 (future ego frame -> current ego frame)
- 优势: 对移动物体最准确，能判断 "那辆车到时候还在不在那"

### 2. BEV Semantic Map (静态环境)
- 来源: `bev_semantics/{frame_id}.png` (256x256, uint8)
- 坐标映射: `col = 128 + x_forward * 2.0`, `row = 128 + y_lateral * 2.0`, 范围 ±64m
- BEV class 定义:

| ID | Class | 说明 |
|----|-------|------|
| 0 | background | 不可行驶区域 |
| 1 | road | 可行驶道路 |
| 2 | sidewalk | 人行道 |
| 3 | solid_line | 实线 (不可跨越) |
| 4 | dashed_line | 虚线 (可跨越) |
| 5 | stop_sign | 停车标志 |
| 6 | green_light | 绿灯 |
| 7 | yellow_light | 黄灯 |
| 8 | red_light | 红灯 |
| 9 | vehicle | 车辆 |
| 10 | pedestrian | 行人 |

- 如果有 boxes 数据，会先过滤掉同向行驶的车辆 (cos(yaw)>0 且 speed>1m/s)，避免误判

### 3. Measurements (CARLA Simulator Flags)
- 来源: `measurements/{frame_id}.json.gz`
- 使用字段: `light_hazard` (红灯), `vehicle_hazard`, `walker_hazard`, `junction`, `speed`, `brake` 等
- 优势: Simulator ground truth，对红灯判断比 BEV 像素更可靠

## 输出标签

### Behavior Labels (per-anchor, 11类)

| ID | Name | 判定条件 | Allowed? |
|----|------|---------|----------|
| 0 | `follow_road` | 在道路上，无碰撞，无违规 | Yes |
| 1 | `collision_front` | 碰撞车辆且 \|lateral_disp\| < 2m | No |
| 2 | `collision_left` | 碰撞车辆且向左偏 (y<0) | No |
| 3 | `collision_right` | 碰撞车辆且向右偏 (y>0) | No |
| 4 | `collision_pedestrian` | 碰撞行人 | No |
| 5 | `off_road` | 超过 30% 轨迹点在 background 上 | No |
| 6 | `on_sidewalk` | 超过 20% 轨迹点在 sidewalk 上 | No |
| 7 | `run_red_light` | 经过红灯区域，或 light_hazard 且位移>5m | No |
| 8 | `lane_change_left` | 跨越车道线且向左 | 仅虚线 |
| 9 | `lane_change_right` | 跨越车道线且向右 | 仅虚线 |
| 10 | `stop` | 总位移 < 2m | Yes |

判定按上表优先级从高到低，collision > off_road > sidewalk > red_light > lane_change > stop > follow_road。

### Allowed Flags (per-anchor, binary)
- `1` = 该轨迹在当前场景下是安全/合法的
- `0` = 该轨迹会导致碰撞/违规

### Scene Buckets (per-sample, 14维 binary vector)
场景级别的分类标签，用于识别 long-tail 场景：

| ID | Bucket | 来源 |
|----|--------|------|
| 0 | vehicle_hazard | measurements |
| 1 | walker_hazard | measurements |
| 2 | light_hazard | measurements |
| 3 | stop_sign_hazard | measurements |
| 4 | junction | measurements |
| 5 | red_light | boxes (traffic_light state) |
| 6 | green_light | boxes (traffic_light state) |
| 7 | vehicle_front | boxes + measurements |
| 8 | vehicle_side | boxes + measurements |
| 9 | brake | measurements |
| 10 | start_from_stop | speed < 0.5 且 target_speed > 0.8 |
| 11 | high_lateral | waypoints lateral mean > 2m |
| 12 | high_decel | speed > 2 且 target < speed*0.5 |
| 13 | low_speed | target_speed < 2 m/s |

## GT Safety Correction

为减少 false positive (把安全轨迹标为 collision):
- 计算每个 anchor 和 GT expert trajectory 的平均距离
- 距离 < `gt_safe_dist` (默认 3m) 的 anchor 视为安全，强制清除碰撞标签
- 原理: expert 轨迹是实际安全驾驶过的，和它接近的 anchor 不可能碰撞

## Labeling Pipeline 流程

```
Dataset.__getitem__()
│
├── 读取 bev_semantics/{frame}.png
├── 读取 boxes/{frame}.json.gz (当前帧)
├── 读取 measurements/{frame}.json.gz (当前帧)
├── 读取未来 K 帧的 boxes + measurements (动态碰撞检测)
│
└── label_anchors_semantic()
    ├── GT safety mask (距离GT近的anchor标为safe)
    ├── Dynamic collision via future frames (优先)
    │   └── 每个waypoint k: anchor位置 vs t+k帧物体位置
    ├── BEV filtering (移除同向车辆)
    ├── 对每个 anchor (32次循环):
    │   ├── 轨迹插值 (0.5m步长)
    │   ├── ego坐标 -> BEV像素
    │   ├── 采样 BEV semantic class
    │   ├── 计算各class占比
    │   ├── 检测车道线穿越
    │   └── 按优先级判定 behavior + allowed
    └── 输出: behavior_labels(32,), allowed_flags(32,), semantic_features(32,11)
```

## 启用方式

Config 中添加:
```yaml
semantic_behavior:
  enabled: true
  num_behaviors: 11
  bev_ppm: 2.0
  bev_size: 256
  allowed_loss_weight: 0.5    # (training用，暂未接入当前branch)
  behavior_loss_weight: 0.1   # (training用，暂未接入当前branch)

anchor_path: wp_tokens.pkl
```

Training script 会从 `wp_tokens.pkl` 加载 anchor centers 传入 dataset。

## 可视化

```bash
python tools/anchor_semantic_labeler.py \
    --dataset /path/to/dataset \
    --anchors wp_tokens.pkl \
    --n_samples 5
```

输出到 `/tmp/anchor_labels_sample_{idx}.png`，包含:
- BEV 语义地图 (彩色)
- 32条 anchor 轨迹 (按 behavior 类别着色)
- GT expert 轨迹 (白色)
- 最近GT的 anchor (黄色高亮)
- 图例 + 统计信息

## 当前状态 (vla_adapter branch)

- [x] `tools/anchor_semantic_labeler.py` - 完整实现
- [x] `dataset/unified_carla_dataset.py` - on-the-fly labeling 已集成
- [x] `training/train_carla_bev.py` - anchor 加载 + dataset 传参已接入
- [ ] `model/` - behavior embedding 未接入 (semantic_behavior_training branch 有实现，但训练范式待定)
- [ ] `policy/` - behavior loss + inference filtering 未接入 (同上)

## 待确定事项

1. **训练范式**: behavior_labels 和 allowed_flags 如何参与训练？
   - 方案A: Auxiliary loss (semantic_behavior_training 的做法: embedding + cross-entropy + BCE)
   - 方案B: 仅作为 mode selection 的 mask (训练时只对 allowed anchors 计算 reg loss)
   - 方案C: 作为 reward signal 用于 RL-style training
   - 方案D: 先用 label 分析 anchor 质量，指导 anchor 重新聚类

2. **Label 质量**: 需要跑可视化确认
   - BEV semantic map 的质量如何？
   - 动态碰撞检测的准确率？
   - GT safety correction 的阈值是否合理？
   - 各 behavior 类别的分布是否均衡？
