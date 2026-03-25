# TransFuser BEV Detection Capability Analysis

> 2026-03-24 session. Demo script: `model/transfuser_extractor/demo_bev.py`

---

## 1. TransFuser 训练时接的 Detection Heads

TransFuser backbone 输出 3 组特征，训练时接了以下 head 进行联合监督：

### BEV Feature Grid (FPN p3, 64ch) — 我们用的 feature

| Head | 输出 | Loss |
|------|------|------|
| **BEV 语义分割** (11类) | (B, 11, 256, 256) | CrossEntropy |
| **CenterNet 目标检测** (7子head) | heatmap + bbox + yaw + velocity | Focal + L1 + CE |

### Image Feature Grid (图像分支)

| Head | 输出 | Loss |
|------|------|------|
| 透视语义分割 (7类) | (B, 7, H, W) | CrossEntropy |
| 深度预测 | (B, 1, H, W) | L1 |

### Fused Features (全局融合)

| Head | 输出 | Loss |
|------|------|------|
| Checkpoint Path GRU | (B, n_checkpoints, 2) | L1 |
| Target Speed 分类 (8档) | (B, 8) | CrossEntropy |

**核心结论**: 我们用的 BEV feature 在训练时被目标检测和 BEV 语义分割联合监督，feature 里编码了物体位置/类别/朝向以及道路语义信息。

---

## 2. BEV 语义分割 — 11 类定义

| Class ID | 名称 | 闭环后处理价值 |
|----------|------|----------------|
| 0 | unlabeled | - |
| 1 | road | 道路区域 |
| 2 | sidewalk | 越界检测 |
| 3 | lane_solid | 不可跨越车道线 |
| 4 | lane_broken | 可跨越车道线 |
| **5** | **stop_sign** | **需要状态机后处理** |
| **6** | **light_green** | **可通行** |
| **7** | **light_yellow** | **注意减速** |
| **8** | **light_red** | **必须停车** |
| 9 | vehicle | 障碍物 |
| 10 | walker | 行人 |

---

## 3. CenterNet 检测 Head — 子 head 详情

| 子 Head | 输出通道 | 说明 |
|---------|---------|------|
| heatmap_head | 5 (vehicle/pedestrian/cyclist/motorcycle/emergency) | Gaussian 中心热力图 |
| wh_head | 2 | bbox 宽高 |
| offset_head | 2 | 亚像素中心偏移 |
| yaw_class_head | 12 | 离散朝向角分类 (12 bins) |
| yaw_res_head | 1 | 朝向角残差 |
| velocity_head | 1 | **仅 lidar_seq_len > 1 时有** |
| brake_head | 2 | **仅 temporal 时有** |

**注意**: 我们用的 ckpt (`model_0030_1.pth`) 是 `lidar_seq_len=1, seq_len=1`，**没有 velocity/brake head**。无法直接从 CenterNet 获取车辆速度，需要连续帧位置差分估算。

---

## 4. BEV 坐标系

### 原始 BEV 图像坐标 (lidar histogram `.T` 之后)

```
         col 0 (x=-32m, behind)  ───→  col 255 (x=+32m, ahead)
row 0    (y=-32m, left)
  │
  ↓
row 255  (y=+32m, right)

Ego 在中心 (128,128)，朝右
```

### 显示用旋转 (demo_bev.py 中 `rotate_bev_ego_up`)

90 CCW 旋转后 ego 朝上，符合直觉：

```
         UP = ahead (ego forward)
         DOWN = behind
         LEFT = left
         RIGHT = right
```

---

## 5. Detection 能力评估

### 可用 (已验证)

- **Traffic light 三色检测**: BEV 语义 class 6/7/8，pixel 级别，位置和大小与 GT 吻合 (红灯场景 pred=385 pixels vs GT=392 pixels)
- **Stop sign 检测**: BEV 语义 class 5，同等训练，可信任
- **Vehicle/Pedestrian 检测**: CenterNet heatmap + bbox，score > 0.8 的检测可靠
- **Vehicle 朝向**: yaw_class (12 bins) + yaw_res，可以知道车辆行驶方向

### 不可用 / 局限

- **Traffic light 不区分朝向**: 只知道某位置有红灯，不知道是给哪个方向的信号灯
- **Lane direction 不可用**: 只有 solid/broken 分类，**没有车道流向信息**，不能判断：
  - 单行道 vs 双行道
  - 左侧同向 vs 逆向车道
- **Vehicle 速度不可用**: `lidar_seq_len=1` 的 ckpt 没有 velocity head
- **Instance 不可区分**: pixel 级分割，多个同类目标的像素混在一起
- **BEV 语义 GT 覆盖范围有限**: 数据采集以 `pixels_per_meter_collection=2.0` 存储，GT 只覆盖相机可见的窄带，但 pred 可以覆盖完整 64m x 64m 范围

### 需要 HD Map 才能做的事

- 车道级路径规划 (lane-level routing)
- 同向/逆向车道判断
- 交叉口拓扑 (哪条车道连接哪条)
- 信号灯与车道的关联 (哪个灯控制哪个方向)

---

## 6. 闭环后处理思路

### Stop Sign 状态机 (解决走走停停问题)

```
状态: NONE → APPROACHING → STOPPED → CLEARED

- APPROACHING: BEV semantic 中检测到 stop_sign pixels 在 ego 前方
- STOPPED: ego speed < 0.1 m/s 且持续 > 1s
- CLEARED: 从 STOPPED 转出，后续即使仍检测到 stop_sign 也不再停车
- NONE: stop_sign pixels 消失 (距离超出 BEV 范围) → 重置状态机
```

### Traffic Light 后处理

```
- 每帧检查 ego 前方扇形区域内的 class 6/7/8 像素数量
- 红灯 (class 8) 像素多 → brake = 1, target_speed = 0
- 绿灯 (class 6) → 正常行驶
- 黄灯 (class 7) → 根据距离决定减速或通过
```

### 显式 BEV Attention 操作 (进阶)

CenterNet 给出 vehicle 的 BEV 位置，可以构造 spatial attention mask 注入 DiT 的 `GridSampleCrossBEVAttention`：
- Pick up 高速/近距离车辆区域，加大 attention weight
- 在 latent BEV feature 上直接操作（不需要解码再编码）

---

## 7. 关键文件路径

| 文件 | 说明 |
|------|------|
| `model/transfuser_extractor/demo_bev.py` | BEV demo 脚本 (GT vs Pred 可视化) |
| `model/transfuser_extractor/backbone_extractor.py` | TransFuser backbone 特征提取器 (只提 feature) |
| `/media/z/data/models/garage2/pretrained_models/all_towns/` | TransFuser ckpt 目录 |
| `/media/z/data/mzq/others/carla_garage/team_code/model.py` | carla_garage 完整模型 (head 接线) |
| `/media/z/data/mzq/others/carla_garage/team_code/center_net.py` | CenterNet head 实现 |
| `/media/z/data/mzq/others/carla_garage/team_code/config.py` | BEV 类别定义、颜色表 |
