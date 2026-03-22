# 闭环 Agent 改造计划

> 目标：把 `BDModelV2`（bridge_baseline v2）接入 Bench2Drive 闭环评测

---

## 结论：需要做什么

| 组件 | 处理方式 |
|------|----------|
| Bench2Drive leaderboard + tick/run_step | 直接复用，不改 |
| RoutePlanner | 直接复用（bench2drive agent 框架已有） |
| `ClosedLoopInference` PID 控制 | 直接复用原版 LEAD（`lead/lead/inference/closed_loop_inference.py`） |
| `OpenLoopInference` ensemble 框架 | 直接复用，只改 model 的 import |
| **模型 forward** | 替换：用 `BDModelV2` 替代 `TFv6` |
| **Prediction dataclass** | 替换：自己定义，只保留规划输出 |
| `StopSignPostProcessor` | **跳过**（我们没有 BBox 检测头） |
| `ForceMovePostProcessor` | 可以加（只看当前速度，不依赖模型） |

---

## 核心：我们的模型输出 vs LEAD 的 Prediction

LEAD 的 `ClosedLoopInference.ensemble()` 需要 `OpenLoopPrediction`，其中控车用的字段：

```python
open_loop_prediction.pred_route           # (1, n_checkpoints, 2) → steer 用
open_loop_prediction.pred_target_speed_scalar  # (1, 1) m/s → throttle/brake 用
open_loop_prediction.pred_future_waypoints     # (1, n_waypoints, 2) → 备选，可为 None
```

我们的 `BDModelV2`（`bridge_baseline/policy.py` BDBaselinePolicyV2）推理输出：
```python
action          # (B, 10, 2)  ← 这就是 pred_route / pred_future_waypoints
target_speed    # (B,)        ← 这就是 pred_target_speed_scalar
```

**所以接口对齐很简单**，只需要 reshape 一下。

---

## 具体改造步骤

### Step 1：写一个轻量的 `BDClosedLoopInference`

新建文件（比如 `bridge_baseline/closed_loop_agent.py`），继承或直接参考 LEAD 的 `ClosedLoopInference`：

```python
class BDClosedLoopInference:
    def __init__(self, config, model_path, device):
        # 加载 BDModelV2
        self.model = BDModelV2(config).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # PID 控制器（直接复用 LEAD 的）
        self.lateral_route_controller = LateralPIDController(config)
        # get_throttle 也直接用 LEAD 的

    def forward(self, data: dict) -> ClosedLoopPrediction:
        # 1. 模型推理
        with torch.no_grad():
            action, target_speed = self.model.inference(data)

        # 2. 转控制量
        pred_route = action.unsqueeze(0)          # (1, 10, 2)
        pred_target_speed = target_speed.reshape(1, 1)  # (1, 1)

        steer, throttle, brake = execute_route_and_target_speed(
            pred_route, pred_target_speed, data["speed"]
        )

        return ClosedLoopPrediction(steer, throttle, brake, ...)
```

### Step 2：适配 data dict 格式

sensor_agent 的 `run_step` 会把传感器数据打包成 `data` dict 传给模型。
我们的模型（`BDBaselinePolicyV2`）需要的 key：

| Key | Shape | 来源 |
|-----|-------|------|
| `transfuser_bev_feature_upsample` | (B, 64, 64, 64) | **问题：闭环时没有预计算特征！** |
| `speed` | (B,) or (B, 4) | sensor_agent 提供 |
| `command_hist` | (B, 4, 6) | sensor_agent 提供 |
| `target_point` | (B, 2) | RoutePlanner 提供 |
| `target_point_next` | (B, 2) | RoutePlanner 提供 |

**关键问题**：`transfuser_bev_feature_upsample` 在离线训练时是预计算好存在数据集里的，闭环时需要**实时计算**——即在线跑 TransFuser backbone。

### Step 3：解决 BEV 特征在线提取问题

**选项 A（推荐）**：加载我们自己训练的 TransFuser extractor，在线提取 BEV 特征
- TransFuser extractor 代码在 `model/transfuser_extractor/`
- ckpt：参考 `docs/transfuser_comparison.md`，需要确认我们自己训练的 ckpt 路径
- 输入：RGB 图像 + LiDAR 点云（sensor_agent 的 tick 已经处理好了）

**选项 B**：先用 dummy BEV（全零），验证控制流程能跑通，再换真实特征

---

## 控制流程（完整版）

```
sensor_agent.tick()
    ├── RGB JPEG压缩
    ├── LiDAR 光栅化
    └── RoutePlanner → target_point, command

sensor_agent.run_step()
    ├── data dict 打包
    ├── TransFuser extractor → bev_feature  ← Step 3 要做的
    ├── BDModelV2.forward(data) → action(10,2), target_speed
    ├── execute_route_and_target_speed()
    │       ├── LateralPIDController(route) → steer
    │       └── get_throttle(target_speed, speed) → throttle, brake
    └── ForceMovePostProcessor（可选）→ 最终 steer/throttle/brake
```

---

## 需要的 LEAD 工具（直接 import，不用重写）

```python
from lead.common.pid_controller import LateralPIDController, PIDController, get_throttle
from lead.inference.config_closed_loop import ClosedLoopConfig
# ClosedLoopConfig 里的 PID 参数：
#   turn_kp/ki/kd/n, speed_kp/ki/kd/n
#   brake_ratio, sensor_agent_steer_correction
#   steer_modality = "route", throttle_modality = "target_speed"
```

## BridgeDrive 的 diffusion_speed

BridgeDrive 的 ClosedLoopInference 唯一新东西：`diffusion_speed` 分支
- 把速度作为第11个"waypoint"拼进 route，推理后再拆回来
- 我们的 `bridge_baseline_v2.md` 的 Method 2 有详细说明
- **第一版先不用，用默认的 `predict_target_speed` 方式**

---

## 待确认事项

1. bench2drive 的 agent 框架（`team_code/mot_b2d_agent.py`）现在接的是哪个模型，tick/run_step 签名是什么
2. TransFuser extractor 在线推理的 ckpt 路径
3. `target_point_hist`（训练时4帧历史）在闭环时是否需要维护帧历史 buffer，还是只用当前帧
