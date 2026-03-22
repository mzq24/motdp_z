# SimLingo VQA 问题集

## Labeling Pipeline

来源：`simlingo/dataset_generation/language_labels/`

### 数据流

```
data_agent.py 采集
  boxes/{frame}.json.gz       ← 结构化标注（车辆/行人/红绿灯/停止标志/ego_info/weather）
  measurements/{frame}.json.gz ← CARLA simulator flags（speed_limit, hazard flags, command…）
        │
        ▼
carla_vqa_generator.py        → vqa/{frame}.json.gz   (DriveLM/NuScenes 格式)
carla_commentary_generator.py → commentary/{frame}/    (自然语言 commentary)
```

### VQA 生成（`drivelm/carla_vqa_generator.py`，3144行）

`QAsGenerator.create_qa_pairs()` 对每个 boxes 帧调用 `generate_perception_questions()`，依次调用6个子生成器：

| 子生成器 | 覆盖问题类别 | 主要数据来源 |
|---|---|---|
| `generate_vehicle_information()` | 车辆位置、行进方向、运动状态、与 ego 路径是否交叉 | `cars` + `static_car` boxes |
| `analyze_road_layout()` | 路口状态、车道数、车道标线、变道许可、ego 当前车道 | `ego_info` boxes + measurements |
| `process_stop_signs()` | 是否受停止标志影响、ego 应如何响应 | `stop_sign_vqa` boxes |
| `process_traffic_lights()` | 是否受红绿灯影响、灯色、ego 应如何响应 | `traffic_light_vqa` boxes |
| `process_pedestrians()` | 行人数量 | `walker` boxes |
| `generate_ego_vehicle_actions()` | 是否需要制动及原因、速度限制、障碍物绕行 | 综合以上所有 + measurements |

额外加一个总结问题：`"What are the important objects in the scene?"`（合并所有 important_objects 列表）

过滤条件：
- 跳过驾驶评分 < 98 的路线（expert 驾驶质量保证）
- 跳过 `InterurbanAdvancedActorFlow`、`MergerIntoSlowTraffic` 场景（标注不准确）
- 可选跳过含行人场景
- `num_points < 50` 的 bbox 视为不可见，跳过

输出格式：`vqa/{frame:04d}.json.gz`，DriveLM/NuScenes Graph-QA 格式，按 chain/layer 组织 Q-A 依赖关系，每个 Q-A 对有 `type`（perception/planning/prediction）字段。

### Commentary 生成（`commentary/carla_commentary_generator.py`，937行）

`COMsGenerator.create_commentary()` 基于模板生成自然语言驾驶 commentary：
- 模板文件：`data/augmented_templates/commentary_augmented.json`（20k行，大量增广变体）
- 使用历史5帧 + 未来10帧的时序信息（`HISTORY_LEN=5`，`FUTURE_LEN=10`）
- 输出：`commentary/{frame}/` 目录

### 增广模板文件

| 文件 | 用途 |
|---|---|
| `augmented_templates/commentary_augmented.json` | Commentary 增广变体（20k行）|
| `augmented_templates/commentary_subsentence.json` | 子句级别模板 |
| `augmented_templates/feedback.json` | 反馈模板 |
| `augmented_templates/dreamer.json` | Dreamer 模板 |
| `augmented_templates/lmdrive.json` | LMDrive 命令增广（推理时 `LMDRIVE_AUGM=True` 使用）|
| `augmented_templates/drivelm_train_augmented_v2/` | DriveLM 训练增广模板 |

### evalset_vqa.json

`simlingo/data/evalset_vqa.json` 是评测集索引，不是生成模板。
结构：`{问题模板: {答案模板: [匹配该答案的 val set 样本路径列表]}}`
供评测时按问题/答案类别统计性能分布。

---

## 问题集详情

来源：`simlingo/data/evalset_vqa.json`

数据结构：每个问题对应一个 dict，key 是答案模板字符串，value 是 val set 中匹配该答案的样本路径列表（用于评测）。
`<OBJECT>`、`<LOCATION>`、`<DISTANCE>` 是运行时根据 boxes 标注填充的占位符。

其他相关文件：
- `simlingo/data/augmented_templates/commentary_augmented.json` — commentary 增强模板（20k 行）
- `simlingo/data/augmented_templates/commentary_subsentence.json` — commentary 子句模板
- `simlingo/data/augmented_templates/feedback.json` — feedback 模板
- `simlingo/data/augmented_templates/dreamer.json` — dreamer 模板
- `simlingo/data/augmented_templates/lmdrive.json` — LMDrive 命令增强模板（agent 推理时使用）
- `simlingo/data/augmented_templates/drivelm_train_augmented_v2/` — DriveLM 增强模板

---

## 场景感知

**Where on the road is the `<OBJECT>` that is `<LOCATION>` located?**
- The `<OBJECT>` is on the same road driving on the lane of the ego vehicle.
- The `<OBJECT>` is on the same road standing on the lane of the ego vehicle.
- The `<OBJECT>` is on the same road driving in the same direction. It is one/two/three lanes to the left/right of the ego vehicle.
- The `<OBJECT>` is on the same road standing in the same direction. It is one/two/... lanes to the left/right.
- The `<OBJECT>` is on the same road driving in the opposite direction. It is N lanes to the left.
- The `<OBJECT>` is on the same road standing in the opposite direction. It is N lanes to the left.
- The `<OBJECT>` is inside the upcoming junction and is pointing leftwards/rightwards/towards the ego vehicle/in the same direction as the ego vehicle.
- The `<OBJECT>` is after the junction on the road the ego vehicle will enter. It is pointing towards/leftwards/rightwards/in the same direction as the ego vehicle.
- The `<OBJECT>` is on the left/right/opposite side of the junction and is pointing towards/away from the junction/in an unknown direction.
- The `<OBJECT>` is on the highway / driving on the highway.
- The `<OBJECT>` is on the exit lane of the highway.
- The `<OBJECT>` is on the highway near the exit/merging area.
- The `<OBJECT>` is on the highway close to the merging/exit area.
- The `<OBJECT>` is in the merging/exit area of the highway in front of the ego vehicle.
- The `<OBJECT>` is driving on the leftmost/second/third/fourth lane from the left on the highway.
- The `<OBJECT>` is in the same lane leading to the highway as the ego vehicle.
- The `<OBJECT>` is on the acceleration lane of the highway to the right of the ego vehicle.
- The `<OBJECT>` is close to the merging area but on the leftmost/second lane from the left on the highway.
- The `<OBJECT>` is on the lane that leads to the highway.

**Is the ego vehicle at a junction?**
- No, the ego vehicle is not at a junction.
- The ego vehicle is in a junction.
- The ego vehicle is right before a junction.
- The ego vehicle is on an exit lane and about to exit the highway.
- The ego vehicle is on an acceleration lane and about to enter the highway.
- The ego vehicle is on the highway potentially close to a junction.
- The ego vehicle is on a turning lane approaching/close to a junction.
- The ego vehicle is on an interurban road close to a point where a new turning lane emerges.
- The ego vehicle is on the highway close to the entry/exit lane.

**How many lanes are there in the same direction as the ego car?**
- There is one lane in the same direction.
- There are two/three/four lanes in the same direction.
- It is not possible to tell since the ego vehicle is in a junction.

**How many lanes are there in the opposite direction to the ego car?**
- There are no lanes in the opposite direction.
- There is one lane in the opposite direction.
- There are two/three/four lanes in the opposite direction.
- It is not possible to tell since the ego vehicle is in a junction.

**On which lane is the ego vehicle (left most lane = 0)?**
- The ego vehicle is on lane 0/1/2/3.
- The ego vehicle is on lane 1/2 since it overtakes an obstruction.
- The ego vehicle is on lane 1 which is the parking lane.
- It is not possible to tell since the ego vehicle is in a junction.

**What lane marking is on the left side of the ego car?**
- There is no lane marking on the left side.
- White solid / white broken / yellow solid / yellow broken / yellow double solid lane.
- It is not possible to tell since the ego vehicle overtakes an obstruction.

**What lane marking is on the right side of the ego car?**
- There is no lane marking on the right side.
- White solid / white broken / yellow solid / yellow broken lane.
- The lane marking on the right side is a curb.

**How many pedestrians are there?**
- There are no pedestrians.
- There is 1 / 2 / 3 / 4 pedestrians.

**What is the current speed limit?**
- 50 / 80 / 100 / 120 km/h.

---

## 交通信号

**Is the ego vehicle affected by a traffic light?**
- Yes / No.

**Is the ego vehicle affected by a stop sign?**
- No, the ego vehicle is not affected by a stop sign.
- Yes, the ego vehicle is affected by a stop sign, which has not been cleared yet.
- Yes, the ego vehicle was affected by a stop sign, which has already been cleared.

**What is the state of the traffic light?**
- There is no traffic light affecting the ego vehicle.
- The traffic light is green / red / yellow.

**What should the ego vehicle do based on the traffic light?**
- There is no traffic light affecting the ego vehicle.
- The traffic light is too `<LOCATION>` away to affect the ego vehicle.
- The ego vehicle can maintain/accelerate its speed and continue driving because the traffic light is green.
- The ego vehicle should slow down and prepare to stop at the traffic light.
- The ego vehicle should slow down and stop at the traffic light.
- The ego vehicle should slow down and stop and stay behind other vehicles at the red light.
- The ego vehicle should remain stopped (and stay behind other vehicles at the red light).
- Based on the green traffic light the ego vehicle can maintain/accelerate its speed but should pay attention to the vehicle in front.
- The ego vehicle should follow the traffic light.

**What should the ego vehicle do based on the stop sign?**
- There is no stop sign affecting the ego vehicle.
- The ego vehicle was affected by a stop sign, which has already been cleared.
- The ego vehicle should slow down and stop at the stop sign.
- The ego vehicle should slow down and stop at the stop sign and stay behind other vehicles.
- The ego vehicle should remain stopped (and stay behind other vehicles at the stop sign).

---

## 障碍物与制动

**Is there an obstacle on the current road?**
- No, there is no obstacle on the current route.
- Yes, there is a parked vehicle on the current road.
- Yes, there is a `<OBJECT>` on the current road.
- Yes, there is a vehicle with the opened door on the current road.
- Yes, there is an accident on the current road.
- Yes, there might be invading vehicles from the opposite lane on the current road.
- Yes, there is a bicycle / two bicycles on the current road.

**Does the ego vehicle need to change lanes or deviate from the lane center due to an upcoming obstruction?**
- No, the ego vehicle can stay on its current lane.
- The ego vehicle must change to the left/opposite lane to circumvent the parked vehicle/accident/bicycle/`<OBJECT>`.
- The ego vehicle must change to the opposite lane to circumvent the vehicle with the opened door.
- The ego vehicle must shift slightly to the right side to avoid invading vehicles on the opposite lane.
- The ego vehicle has already shifted to the side / changed to another lane to circumvent [obstacle].
- The ego vehicle is changing to another lane to circumvent [obstacle].
- The ego vehicle must change back to the original lane after passing the obstruction.
- The ego vehicle must change to the left to exit the parking lot.

**Does the ego vehicle need to brake? Why?**
（答案数量最多，覆盖各种颜色车辆 × 位置的组合）
- There is no reason for the ego vehicle to brake.
- The ego vehicle should adjust its speed to the speed of the `<OBJECT/color car>` that is to the front/front left/front right.
- The ego vehicle should brake because of the `<OBJECT/color car>` that is to the front/front left/front right/left side of junction/right side of junction/opposite side of junction.
- The ego vehicle should stop because of the `<OBJECT/color car>` that is to the front/front left/front right.
- The ego vehicle should stop because of the traffic light that is red.
- The ego vehicle should slow down and stop at the stop sign.
- The ego vehicle should stop because of the stop sign.
- The ego vehicle should stop/brake because it must invade the opposite lane, which is occupied, in order to bypass [obstacle].
- The ego vehicle should brake/stop because it must change the lane to bypass [obstacle].
- The ego vehicle should stop because of the pedestrian/pedestrians that is crossing the road.
- The ego vehicle should slow down because of the pedestrian/pedestrians that is crossing the road.
- The ego vehicle should brake because it is too fast.
- The ego vehicle should slow down because of the `<OBJECT>` that is blocking the intersection.
- The ego vehicle should stop because of the `<OBJECT>` that is on the oncoming lane and is crossing paths with the ego vehicle.

---

## 车道变换

**In which direction is the ego car allowed to change lanes?**
- The ego vehicle can not change lanes since it is on a one lane road.
- The ego vehicle can not change lanes since it is on a one lane road. But it could change to the parking lane on the right.
- The ego vehicle is allowed to change lanes to the left / right / left and right.
- The ego vehicle is not allowed to change lanes to another driving lane.
- The ego vehicle is allowed to change lanes to the left to enter the highway.
- The ego vehicle is on a parking lane and is allowed to merge into the driving lane.
- The ego vehicle overtakes an obstruction. It is not expected to change lanes.
- It is not possible to tell since the ego vehicle is in a junction.

**From which side are other vehicles allowed to change lanes into the ego lane?**
- There are no lane changes possible since the ego vehicle is on a one lane road.
- Vehicles are allowed to change lanes from the left / right / both sides.
- There are no lane changes allowed from another driving lane into the ego lane.
- Vehicles could potentially change from the left but it is very unlikely since the ego vehicle is on an acceleration lane.
- The ego vehicle is on a parking lane and vehicles only enter the lane to park.
- It is not possible to tell since the ego vehicle is in a junction.

**The ego vehicle wants to do a lane change to the right/left. Which lanes are important to watch out for?**
- Pay attention to traffic in the right/left-hand lane and wait for a gap to change lanes.
- Pay attention to traffic on the leftmost/rightmost lane of the highway, adjust speed, and position itself.

**The ego vehicle wants to do a lane change to the right/left soon. Which lanes are important to watch out for?**
- Pay attention to traffic in the right/left-hand lane and position itself so that no vehicle is at the same height.
- (Various intersection-specific answers)

**The ego vehicle does a lane change to the right/left in `<DISTANCE>`. Is the `<OBJECT>` that is `<LOCATION>` potentially crossing the path?**
- No, the `<OBJECT>` is not crossing paths.
- Yes, the `<OBJECT>` is crossing paths because the ego vehicle does a lane change onto the lane of the `<OBJECT>`.
- Yes, the `<OBJECT>` will cross paths if the ego vehicle does a lane change in `<DISTANCE>`.
- Yes, the `<OBJECT>` might cross paths depending on which way the vehicle is going to turn.
- (Other standard crossing answers)

---

## 路口意图（按行驶方向）

**The ego vehicle wants to follow the road. Which lanes are important to watch out for?**
- No other driving lanes (one lane road), optionally watch out for parking lane.
- Pay attention to vehicles in the junction.
- Pay attention to traffic changing lanes from neighboring lanes.
- Pay attention to oncoming lane and optionally parking lane.
- Pay attention to traffic changing lanes from neighboring lanes and oncoming traffic.
- Pay attention to traffic on the highway close to the acceleration lane.
- Pay attention to the traffic in the lane the ego vehicle wants to enter from the parking space.
- Keep driving regardless of other vehicles since it overtakes an obstruction.

**The ego vehicle wants to go straight at the next intersection. Which lanes are important?**
- Pay attention to traffic from the left (going straight or turning left), from the right (going straight or turning right), and oncoming traffic turning left.
- No lane changes needed.
- (Highway-specific answers)

**The ego vehicle wants to go right at the next intersection. Which lanes are important?**
- Pay attention to traffic coming straight ahead from the left and to oncoming traffic turning left.

**The ego vehicle wants to go left at the next intersection. Which lanes are important?**
- Pay attention to traffic from left (straight/turning left), from right (straight/turning left), and oncoming traffic.
- Pay attention to oncoming traffic the ego vehicle needs to cross in order to turn left.

**The ego vehicle wants to exit the highway. Which lanes are important?**
- Pay attention to the traffic on the exit lane, since they might slow down.
- The ego vehicle is still `<LOCATION>` away from the exit lane, so pay attention to highway traffic.

**路口穿越判断（Is the `<OBJECT>` potentially crossing the path?）**

针对每种行驶意图（follow road / go straight / continue straight / drive straight / turn right / continue turning right / turn left / continue turning left）均有一组答案：
- No, not crossing paths.
- Yes, the `<OBJECT>` is right `<LOCATION>`, pay attention to not crash into it.
- Yes, the `<OBJECT>` is inside the upcoming junction on the same road.
- Yes, the `<OBJECT>` is behind the intersection on the road the ego vehicle will enter.
- Yes, the `<OBJECT>` will cross paths if the ego vehicle [specific maneuver].
- Yes, the `<OBJECT>` is crossing the path of the ego vehicle.
- Yes, might cross depending on which way the vehicle is going to turn.
- Routes might cross as the `<OBJECT>` is on the highway / acceleration lane about to enter the highway.

---

## 目标物动态

**Where is the `<OBJECT>` that is `<LOCATION>` going?**
- The `<OBJECT>` is going straight.
- The `<OBJECT>` is turning slightly right/left.
- The `<OBJECT>` is turning right/left.

**What is the moving status of the `<OBJECT>` that is `<LOCATION>`?**
- The `<OBJECT>` is not moving.
- The `<OBJECT>` is driving slowly / moving slowly.
- The `<OBJECT>` is driving.
