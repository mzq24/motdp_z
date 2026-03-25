# Route vs Traj 关系分析

> 分析日期: 2026-03-23
> 数据: pdm_lite_mini (Accident, AccidentTwoWays 场景)

## 结论: Traj follow Route

Ego traj 是 route 的"时间域实现"——车辆按 route 指示的方向行驶，但受速度和车辆动力学影响，轨迹在空间上不会完全贴合 route。

## 两者定义

| | Route | Ego Traj (ego_waypoints) |
|---|---|---|
| 含义 | 道路中心线/规划路径的几何形状 | 车辆实际行驶的时序轨迹 |
| 点数 | 20 waypoints | 6+1 steps (含 origin) |
| 间距 | 固定 1m 等间距 | 时间等间距 0.5s (hz_interval=2) |
| 总长 | 固定 ~20m | 取决于速度 (speed * 3s) |
| 起点 | ego 前方 ~2-3m | (0, 0) ego 当前位置 |
| 坐标系 | ego frame (x=forward, y=lateral) | ego frame (同上) |

## 关键观察

1. **方向一致**: traj 始终沿 route 方向行驶，直行时几乎重合，转弯时跟随弯道
2. **空间尺度不同**: 高速 (~10m/s) 时 traj ~30m 超过 route 20m；低速/停车时 traj 很短
3. **起点偏移**: route[0] 从 ego 前方 ~2-3m 开始，traj 从 (0,0) 开始
4. **转弯偏差**: 高速转弯时 traj 会"切弯"(走内侧)，与 route 有 1-3m 偏差，属正常物理行为

## 数据来源

- **Route**: 直接来自 CARLA measurements `current_anno['route']`，已在 ego frame 中，20 个点，pad 或截断到 20
- **Ego traj**: 由 `get_waypoints()` 计算，取未来帧的 `ego_matrix` 位置转换到当前 ego frame
  - `preprocess_pdm_lite.py:261` 中 `get_waypoints()`
  - action_horizon=6, hz_interval=2 -> 每隔 2 帧取一个点 -> 0.5s 间隔 -> 3s 总时长

## 具体数值示例

### 直行 (Frame 20, speed=9.3m/s)
- Route 间距: 均匀 ~1.0m, 方向几乎纯 x 轴
- Ego traj 步长: ~5m/step, 方向与 route cos=1.0000
- Traj 总长 ~30m, 超出 route 20m 覆盖范围

### 转弯 (Frame 60, speed=7.7m/s, cmd=4)
- Route: 从直行逐渐右转，末端 lateral ~8m
- Ego traj: 同样右转，方向 cos=0.94，但因速度更快所以走得更远
- 空间偏差 ~2-3m，主要来自 traj 切弯

### 停车 (Frame 50, speed=0.0m/s)
- Route: 正常 20m 延伸
- Ego traj: 前几步几乎不动，后面才开始加速
- Traj 总长仅 ~9m

## 可视化

见 `/tmp/route_vs_traj.png` 和 `/tmp/route_vs_traj_turns.png`
