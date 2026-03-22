# BridgeDrive 工程实现总结（Close Loop Agent）

> 参考代码: `BridgeDrive/BridgeDrive_adaptation_LEAD/lead/inference/`

---

## 整体架构

```
SensorAgent (sensor_agent_bridgedrive.py)
  ├── tick()              ← 传感器预处理（每帧）
  ├── run_step()          ← 主循环（每帧）
  │     ├── ClosedLoopInference.forward()   ← 模型推理
  │     ├── StopSignPostProcessor.adjust()  ← 停车标志启发式
  │     └── ForceMovePostProcessor.adjust() ← 卡死处理启发式
  └── destroy()           ← 清理 + 视频压缩

ClosedLoopInference (closed_loop_inference_bridgedrive.py)
  ├── 继承 OpenLoopInference
  ├── execute_route_and_target_speed()  ← 路线+目标速度 → PID 控制
  └── execute_waypoints()               ← 航点 → PID 控制

OpenLoopInference (open_loop_inference_bridgedrive.py)
  ├── 模型加载（支持 ensemble 多模型）
  ├── forward()           ← 调用所有模型，ensemble 输出
  └── ensemble_*()        ← BEV语义/深度/BB/规划各自 ensemble
```

---

## 1. 传感器预处理（tick）

每帧在 `tick()` 中做以下工作，**目的是减少 train-test 分布偏移**：

| 处理 | 方法 | 原因 |
|---|---|---|
| RGB JPEG 压缩 | `cv2.imencode/imdecode` (quality=90) | 训练数据也经过 JPEG 压缩 |
| 水平 FOV 裁剪 | crop + resize 回原尺寸 | 匹配训练时的 FOV 设置 |
| 相机选择 | 按 `used_cameras` 切片拼接 | 只使用训练时用的相机 |
| LiDAR 点精度量化 | `round(x / precision) * precision` | 匹配 laspy 存储时的量化精度 |
| LiDAR 光栅化 | `rasterize_lidar()` → 伪图像 | 和训练 pipeline 一致 |
| LiDAR 压缩/解压 | `compress/decompress_float_image` | 匹配训练时的压缩格式 |
| Radar 预处理 | `preprocess_radar_input()` | 和训练一致 |

### 路线规划（set_target_points）

- 使用 `RoutePlanner` 从全局路线中取 `target_point_previous / current / next` 三个航点
- **自适应 pop_distance**: 当相邻目标点过近（< 10m）时自动切换到更密集的 4m pop_distance
- **过远跳过**: `target_point_next` 超过 50m 时直接用 `target_point` 代替，防止过早换道

---

## 2. 模型推理（ClosedLoopInference）

### 模型加载（OpenLoopInference）

- 自动扫描 `model_path` 下所有 `model*.pth` 文件，**支持 ensemble 多个模型**
- `forward()` 对所有模型跑推理，再 ensemble 输出

### Ensemble 策略

| 输出 | Ensemble 方法 |
|---|---|
| 目标速度 logits | 平均后 softmax decode |
| 航点 | 平均 |
| 路线 checkpoints | 平均 |
| BEV 语义 (ch0=背景) | ch0 取 min，其余取 max |
| 图像语义分割 | 同上 |
| 深度图 | 平均 |
| Bounding Boxes | NMS 合并 |

### Trajectory → 控制量转换（execute_route_and_target_speed）

模型输出是轨迹点，需转换为 steer/throttle/brake：

**方式一：route + target_speed（默认）**
- `steer`: `LateralPIDController`，根据 route checkpoints 做 lateral pure pursuit
  - aim distance = `clip(0.975*speed + 1.915, 24, 105) / 10`（速度自适应）
  - `sensor_agent_steer_correction`: 低速时的转向校正
- `throttle/brake`: `get_throttle()` 根据 pred_target_speed vs 当前速度
  - `brake = True` 当 `target_speed < 0.01` 或 `speed/target_speed > 1.1`

**方式二：waypoint（备选）**
- desired_speed 从 `|waypoint[half_second] - waypoint[one_second]| * 2` 估算
- steer 从 aim 点的 `arctan2(y, x)` 计算，再过 PID

**控制模态可配置（config_closed_loop.py）**:
```python
steer_modality = "route"        # or "waypoint"
throttle_modality = "target_speed"  # or "waypoint"
brake_modality = "target_speed"     # or "waypoint"
```

---

## 3. 后处理启发式（Post-Processors）

### StopSignPostProcessor（停车标志）

- 从模型预测的 BoundingBox 中检测 `STOP_SIGN` 类别
- 当 stop sign 在距离阈值（1m）内且车速 > 0.01 → 强制 `brake=True, throttle=0`
- 刹停后有 cool down（120 帧），避免在同一个停车标志反复刹车
- stop sign box 跨帧追踪：用自车位姿变化更新 box 位置（`bb.update(x, y, yaw, ...)`）

### ForceMovePostProcessor（卡死检测）

- 若 `speed < 0.1` 连续超过 `stuck_threshold`（1100 帧 ≈ 55s）帧 → 触发 force move
- Force move 时先做 **LiDAR safety check**：前方安全框内有点云则放弃，否则施加 `throttle=0.4`
- force move 持续 20 帧后自动停止

---

## 4. 可视化与录制

### VideoRecorder

- 同步录制多路视频：debug（BEV + 规划可视化）、demo（第三视角相机）、input（原始 RGB）、grid（demo + input 叠加）
- `viz_downsample_factor=2`：每 2 帧出一次 debug 图，减少渲染开销
- 图像缩小 2x（`image[::2, ::2, :]`）再写视频
- 结束时用 ffmpeg 压缩最终视频（`cleanup_and_compress()`）

### 违规追踪（check_infractions）

- 每帧从 `scenario.get_criteria()` 读取违规事件
- **离散违规**（碰撞）：用 frame 作为 unique key，每次碰撞只记录一次
- **连续违规**（偏离车道）：用 criterion name 作为 key，only log once until cleared
- 违规信息写入 `infractions.json`，含步骤、类型、行驶里程

---

## 5. 配置系统（ClosedLoopConfig）

配置从**环境变量** `LEAD_CLOSED_LOOP_CONFIG` 加载（JSON），支持 override，关键参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `step_num` | 20 | DDIM 步数（BridgeDrive 专属） |
| `diffusion_speed` | False | 是否用 diffusion 预测速度 |
| `jpeg_quality` | 90 | 推理时 JPEG 压缩质量 |
| `steer/throttle/brake_modality` | route/target_speed/target_speed | 控制模态选择 |
| `sensor_agent_stuck_threshold` | 1100 帧 | 卡死检测阈值 |
| `slower_for_stop_sign` | False | 停车标志启发式开关 |
| `is_bench2drive` | 从环境变量读 | 是否用 Bench2Drive 评测模式 |

---

## 6. 评测集成（Bench2Drive）

- `leaderboard_evaluator_v2.py` 对接 Bench2Drive leaderboard
- `eval_bench2drive_bridgedrive.sh` 提供评测脚本
- `is_bench2drive=True` 时每帧写 `metric_info.json`（DS、route completion 等）
- 支持 `set_scenario(scenario)` 接口让 leaderboard 注入场景引用用于违规追踪

---

## 与我们项目的差距

| 能力 | BridgeDrive | 我们的项目 |
|---|---|---|
| 传感器预处理（train-test 对齐） | 完整（JPEG/LiDAR量化/压缩） | 未做 |
| 路线规划 | RoutePlanner + 自适应 pop_distance | 需要实现 |
| PID 控制器 | Lateral PID + Speed PID（双路可选） | 需要实现 |
| 卡死/停车标志 启发式 | 完整实现 | 需要实现 |
| 多模型 Ensemble | 支持 | 未做 |
| 违规追踪 | 完整 JSON 记录 | 未做 |
| 视频录制 | 多路（debug/demo/input/grid） | 未做 |
| Bench2Drive 评测 | 完整集成 | 未做 |
