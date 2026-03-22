# BridgeDrive Agent 闭环推理架构

> 参考代码: `BridgeDrive/BridgeDrive_adaptation_LEAD/lead/inference/`

---

## 文件位置

| 文件 | 说明 |
|------|------|
| `lead/inference/sensor_agent_bridgedrive.py` | 主 Agent，每帧 tick/run_step |
| `lead/inference/closed_loop_inference_bridgedrive.py` | 闭环推理（轨迹→PID控制） |
| `lead/inference/open_loop_inference_bridgedrive.py` | 开环推理（模型forward + ensemble） |
| `lead/inference/config_closed_loop.py` | 闭环配置 |
| `lead/tfv6/tfv6_bridgedrive.py` | TFv6 模型（含 Prediction dataclass） |
| `lead/tfv6/planning_decoder_bridgedrive.py` | 扩散规划头（DDBM） |
| `scripts/eval_bench2drive_bridgedrive.sh` | Bench2Drive 评测脚本 |
| `3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator_v2.py` | leaderboard 接入 |

---

## OpenLoopInference 与原版 LEAD 的差异

**几乎完全相同**，唯一区别是 import 的模型：

```python
# 原版 LEAD:
from lead.tfv6.tfv6 import Prediction, TFv6

# BridgeDrive:
from lead.tfv6.tfv6_bridgedrive import Prediction, TFv6
```

其余代码逐行相同。

---

## Prediction Dataclass（tfv6_bridgedrive.py）

```python
@dataclass
class Prediction:
    # 规划输出（来自 PlanningDecoderDDBM）
    pred_route:                     (bs, n_checkpoints, 2)   # 空间路径 checkpoints
    pred_future_waypoints:          (bs, n_waypoints, 2)     # 时序轨迹点
    pred_target_speed_distribution: (bs, num_speed_classes)  # 速度 logits（8类）
    pred_target_speed_scalar:       (bs,)                    # 解码后速度标量 m/s

    # 感知输出（carla_leaderboard_mode 才用，否则为 None）
    pred_semantic, pred_depth, pred_bev_semantic, pred_bounding_box, ...
```

---

## Ensemble 策略（ensemble_planning_decoder）

| 输出 | 方法 |
|------|------|
| route / waypoints | 直接平均坐标 |
| target_speed | 先平均 logits → softmax → decode_two_hot |
| BEV/图像语义 | ch0（背景）取 min，其余取 max |
| BBox | NMS 合并 |
| 深度 | 直接平均 |

> BBox、BEV semantic、深度都被 `carla_leaderboard_mode` 守着。我们的单摄像头数据集不是 leaderboard 模式，这些头均为 None，真正有用的只有 `ensemble_planning_decoder`。

---

## 数据流

```
TransfuserBackbone
    → bev_features
    → PlanningDecoderDDBM（扩散规划头）
    → Prediction(pred_route, pred_future_waypoints, pred_target_speed_scalar)
    → ensemble_planning_decoder()（多模型合并）
    → OpenLoopPrediction
    → ClosedLoopInference.execute_route_and_target_speed()
    → PID 控制 → steer / throttle / brake
```

---

## 移植到我们项目的思路

| 组件 | 处理方式 |
|------|----------|
| `OpenLoopInference` 框架 | 可直接复用，替换 import 的模型类 |
| `ClosedLoopInference` PID 控制 | 可直接复用（route→steer，target_speed→throttle/brake） |
| `sensor_agent_bridgedrive.py` | 需要适配：传感器预处理部分要对齐我们的数据格式（单摄像头 1024×512，LiDAR 256×256） |
| `TFv6 + PlanningDecoderDDBM` | 需要替换为我们的模型（BDModelV2 / anchor_free policy）并输出相同的 Prediction 结构 |
| Bench2Drive leaderboard 接入 | 可直接复用 `leaderboard_evaluator_v2.py` |
| `StopSignPostProcessor` / `ForceMovePostProcessor` | 可直接复用 |
