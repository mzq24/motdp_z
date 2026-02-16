"""
Anchor Semantic Labeler

Labels each of the 32 trajectory anchors with a behavior category and allowed/forbidden flag
using a hybrid approach:
  1. BEV semantic map — static environment (road, sidewalk, lane lines)
  2. Future frame boxes — dynamic collision with vehicles/pedestrians at future positions
  3. Measurements — reliable hazard flags from CARLA simulator (light_hazard, etc.)

BEV Semantic Classes:
    0: background/unlabeled
    1: road (drivable area)
    2: sidewalk
    3: solid lane line (cannot cross)
    4: dashed lane line (can cross)
    5: stop sign
    6: traffic light green
    7: traffic light yellow
    8: traffic light red
    9: vehicle
    10: pedestrian

Behavior Categories:
    0: follow_road          - stays on road, no collision               -> allowed
    1: collision_front      - hits vehicle ahead (|lateral_disp|<2m)    -> forbidden
    2: collision_left       - hits vehicle when going left (y<0)        -> forbidden
    3: collision_right      - hits vehicle when going right (y>0)       -> forbidden
    4: collision_pedestrian - passes through pedestrian                 -> forbidden
    5: off_road             - goes off drivable area (background)       -> forbidden
    6: on_sidewalk          - goes onto sidewalk                        -> forbidden
    7: run_red_light        - passes through red light area             -> forbidden
    8: lane_change_left     - crosses lane line going left              -> depends
    9: lane_change_right    - crosses lane line going right             -> depends
    10: stop                - very short displacement (<2m)             -> allowed

BEV Coordinate Mapping (verified empirically):
    col = 128 + x_forward * 2.0
    row = 128 + y_lateral * 2.0
    ppm = 2.0, ego at (128, 128), BEV 256x256, range +/-64m
"""

import numpy as np
from typing import Tuple, Optional, List, Dict

# Behavior category constants
FOLLOW_ROAD = 0
COLLISION_FRONT = 1
COLLISION_LEFT = 2
COLLISION_RIGHT = 3
COLLISION_PEDESTRIAN = 4
OFF_ROAD = 5
ON_SIDEWALK = 6
RUN_RED_LIGHT = 7
LANE_CHANGE_LEFT = 8
LANE_CHANGE_RIGHT = 9
STOP = 10

NUM_BEHAVIORS = 11

BEHAVIOR_NAMES = [
    'follow_road',
    'collision_front', 'collision_left', 'collision_right',
    'collision_pedestrian',
    'off_road', 'on_sidewalk', 'run_red_light',
    'lane_change_left', 'lane_change_right',
    'stop',
]

# Lateral displacement threshold for direction classification (meters)
LATERAL_DIR_THRESH = 2.0

# BEV semantic class IDs
BEV_BACKGROUND = 0
BEV_ROAD = 1
BEV_SIDEWALK = 2
BEV_SOLID_LINE = 3
BEV_DASHED_LINE = 4
BEV_STOP_SIGN = 5
BEV_GREEN_LIGHT = 6
BEV_YELLOW_LIGHT = 7
BEV_RED_LIGHT = 8
BEV_VEHICLE = 9
BEV_PEDESTRIAN = 10

# Classes considered as "drivable" (trajectory can be on these)
DRIVABLE_CLASSES = {BEV_ROAD, BEV_SOLID_LINE, BEV_DASHED_LINE,
                    BEV_GREEN_LIGHT, BEV_YELLOW_LIGHT, BEV_STOP_SIGN}


def _interpolate_trajectory(waypoints: np.ndarray, step_m: float = 0.5) -> np.ndarray:
    """
    Interpolate between waypoints to get evenly-spaced sample points.

    Args:
        waypoints: (N, 2) trajectory waypoints [x_forward, y_lateral]
        step_m: spacing in meters between interpolated points

    Returns:
        (M, 2) interpolated points
    """
    if len(waypoints) < 2:
        return waypoints

    points = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        seg_len = np.linalg.norm(p1 - p0)
        if seg_len < 1e-6:
            continue
        n_steps = max(int(np.ceil(seg_len / step_m)), 1)
        for j in range(1, n_steps + 1):
            t = j / n_steps
            points.append(p0 + t * (p1 - p0))

    return np.array(points)


def _ego_to_bev_pixels(points: np.ndarray, ppm: float = 2.0,
                       bev_size: int = 256) -> np.ndarray:
    """
    Convert ego-centric coordinates to BEV pixel coordinates.

    Args:
        points: (N, 2) [x_forward, y_lateral] in meters
        ppm: pixels per meter
        bev_size: BEV image size (assumed square)

    Returns:
        (N, 2) [row, col] pixel coordinates (may be out of bounds)
    """
    center = bev_size / 2.0
    cols = center + points[:, 0] * ppm  # x_forward -> col
    rows = center + points[:, 1] * ppm  # y_lateral -> row
    return np.stack([rows, cols], axis=1)


def _sample_bev_classes(bev_semantic: np.ndarray,
                        pixel_coords: np.ndarray) -> np.ndarray:
    """
    Look up BEV semantic class at each pixel coordinate.

    Args:
        bev_semantic: (H, W) uint8 semantic map
        pixel_coords: (N, 2) [row, col] pixel coordinates

    Returns:
        (N,) semantic class IDs, out-of-bounds pixels get BEV_BACKGROUND
    """
    H, W = bev_semantic.shape
    rows = np.clip(np.round(pixel_coords[:, 0]).astype(int), 0, H - 1)
    cols = np.clip(np.round(pixel_coords[:, 1]).astype(int), 0, W - 1)

    # Mark out-of-bounds as background
    oob = ((pixel_coords[:, 0] < 0) | (pixel_coords[:, 0] >= H) |
           (pixel_coords[:, 1] < 0) | (pixel_coords[:, 1] >= W))

    classes = bev_semantic[rows, cols].copy()
    classes[oob] = BEV_BACKGROUND
    return classes


def _detect_line_crossings(classes_along_traj: np.ndarray):
    """
    Detect lane line crossings along a trajectory.

    Returns:
        crosses_solid: bool - whether trajectory crosses a solid line
        crosses_dashed: bool - whether trajectory crosses a dashed line
    """
    crosses_solid = False
    crosses_dashed = False

    for i in range(len(classes_along_traj) - 1):
        c0 = classes_along_traj[i]
        c1 = classes_along_traj[i + 1]
        # A crossing is when we transition from non-line to line or line to non-line
        if c0 != c1:
            if c0 == BEV_SOLID_LINE or c1 == BEV_SOLID_LINE:
                crosses_solid = True
            if c0 == BEV_DASHED_LINE or c1 == BEV_DASHED_LINE:
                crosses_dashed = True

    return crosses_solid, crosses_dashed


def _filter_bev_by_boxes(
    bev_semantic: np.ndarray,
    boxes: list,
    ppm: float = 2.0,
    bev_size: int = 256,
    same_dir_speed_thresh: float = 1.0,
) -> np.ndarray:
    """
    Filter BEV semantic map using boxes info:
    - Remove vehicle/pedestrian pixels for objects that are moving in the same
      direction as ego (they will move away, not a collision risk).
    - Keep pixels for: stationary objects, oncoming objects, slow objects.

    Ego faces forward (x+), yaw=0. An object is "same direction" if its yaw
    is within +-90 degrees of ego's forward direction AND has speed > threshold.

    Args:
        bev_semantic: (H, W) uint8, will NOT be modified in-place
        boxes: list of dicts with keys: class, position, speed, yaw, extent
        ppm: pixels per meter
        bev_size: BEV image size
        same_dir_speed_thresh: speed threshold (m/s) below which object is treated as stationary

    Returns:
        filtered_bev: (H, W) uint8, same-dir moving vehicles replaced with BEV_ROAD
    """
    filtered = bev_semantic.copy()
    center = bev_size / 2.0

    for box in boxes:
        cls = box.get('class', '')
        if cls == 'ego_car':
            continue
        # Only filter vehicles (class contains 'car', 'truck', 'bus', 'motorcycle', etc.)
        # Pedestrians are slow-moving, keep them as collision risk
        is_vehicle = cls in ('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle')
        if not is_vehicle:
            continue

        speed = abs(box.get('speed', 0.0))
        yaw = box.get('yaw', 0.0)  # radians, ego forward = 0

        # Check if same direction: cos(yaw) > 0 means forward-ish
        # yaw ~0 = same dir as ego, yaw ~±π = oncoming
        same_direction = abs(np.cos(yaw)) > 0.5 and np.cos(yaw) < -0.5  # yaw≈π means they face same way in ego frame
        # Actually: in ego frame, ego faces +x (yaw=0).
        # Box yaw is also in ego frame. A car driving same direction has yaw ≈ 0.
        # A car driving opposite direction has yaw ≈ ±π.
        # cos(0)=1 (same dir), cos(π)=-1 (opposite)
        same_direction = np.cos(yaw) > 0  # yaw within ±90° of ego forward

        if same_direction and speed > same_dir_speed_thresh:
            # This vehicle is moving away in same direction — erase from BEV
            pos = box['position'][:2]  # [x_forward, y_lateral]
            extent = box.get('extent', [2.5, 1.0])[:2]  # [half_length, half_width]

            # Compute bounding box in BEV pixels
            # Box corners in ego coords, then convert to pixels
            # Simple axis-aligned approximation (good enough for erasing)
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            half_l, half_w = extent[0], extent[1]

            # 4 corners in ego frame
            corners_local = np.array([
                [-half_l, -half_w],
                [-half_l,  half_w],
                [ half_l,  half_w],
                [ half_l, -half_w],
            ])
            # Rotate by yaw and translate
            rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            corners_ego = corners_local @ rot.T + np.array(pos)

            # Convert to BEV pixels
            cols = center + corners_ego[:, 0] * ppm
            rows = center + corners_ego[:, 1] * ppm

            # Fill bounding box region with road
            r_min = max(0, int(np.floor(rows.min())))
            r_max = min(bev_size - 1, int(np.ceil(rows.max())))
            c_min = max(0, int(np.floor(cols.min())))
            c_max = min(bev_size - 1, int(np.ceil(cols.max())))

            # Only erase vehicle pixels (don't erase road markings etc.)
            region = filtered[r_min:r_max+1, c_min:c_max+1]
            region[region == BEV_VEHICLE] = BEV_ROAD

    return filtered


def _check_dynamic_collision_future(
    anchor_centers_abs: np.ndarray,
    ego_matrix_current: np.ndarray,
    future_frames_data: List[Optional[Tuple[list, np.ndarray]]],
    collision_margin: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check dynamic collision using future frame boxes data (simlingo-style).

    For each anchor waypoint at timestep k, check if it overlaps with any
    vehicle/pedestrian at their actual position in future frame t+k.

    Args:
        anchor_centers_abs: (num_modes, num_points, 2) [x_forward, y_lateral]
        ego_matrix_current: (4, 4) world transform of ego at current frame
        future_frames_data: list of (boxes_list, ego_matrix_4x4) per future timestep,
                           None entries are skipped. Length should match num_points.
        collision_margin: extra margin in meters around actor bounding box

    Returns:
        dynamic_collision_vehicle: (num_modes,) bool per-anchor vehicle collision
        dynamic_collision_pedestrian: (num_modes,) bool per-anchor pedestrian collision
    """
    num_modes, num_points, _ = anchor_centers_abs.shape
    collision_vehicle = np.zeros(num_modes, dtype=bool)
    collision_pedestrian = np.zeros(num_modes, dtype=bool)

    # Inverse of current ego world transform: world → current ego frame
    try:
        ego_inv = np.linalg.inv(ego_matrix_current)
    except np.linalg.LinAlgError:
        return collision_vehicle, collision_pedestrian

    for k, frame_data in enumerate(future_frames_data):
        if frame_data is None or k >= num_points:
            continue
        boxes_future, ego_matrix_future = frame_data

        # Combined transform: future ego frame → world → current ego frame
        # pos_current = ego_inv @ ego_matrix_future @ pos_future_homo
        transform = ego_inv @ np.array(ego_matrix_future)

        for box in boxes_future:
            cls = box.get('class', '')
            if cls == 'ego_car':
                continue

            is_vehicle = cls in ('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle')
            is_pedestrian = (cls == 'walker')
            if not (is_vehicle or is_pedestrian):
                continue

            # Convert box position from future ego frame to current ego frame
            pos = box['position']
            pos_homo = np.array([pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0, 1.0])
            pos_current = transform @ pos_homo
            pos_2d = pos_current[:2]  # [x_forward, y_lateral] in current ego frame

            # Actor collision radius from extent
            extent = box.get('extent', [2.5, 1.0])[:2]
            radius = max(extent[0], extent[1]) + collision_margin

            # Check all anchors' waypoint k against this actor
            anchor_pts = anchor_centers_abs[:, k, :]  # (num_modes, 2)
            dists = np.linalg.norm(anchor_pts - pos_2d[np.newaxis, :], axis=1)
            hits = dists < radius

            if is_vehicle:
                collision_vehicle |= hits
            elif is_pedestrian:
                collision_pedestrian |= hits

    return collision_vehicle, collision_pedestrian


def label_anchors_semantic(
    anchor_centers_abs: np.ndarray,
    bev_semantic: np.ndarray,
    ppm: float = 2.0,
    bev_size: int = 256,
    off_road_threshold: float = 0.3,
    sidewalk_threshold: float = 0.2,
    stop_distance_threshold: float = 2.0,
    collision_near_weight: float = 2.0,
    boxes: Optional[list] = None,
    measurements: Optional[dict] = None,
    ego_matrix_current: Optional[np.ndarray] = None,
    future_frames_data: Optional[List] = None,
    gt_trajectory: Optional[np.ndarray] = None,
    gt_safe_dist: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Label each anchor trajectory with behavior category and allowed/forbidden flag.

    Hybrid approach:
    1. BEV semantic map — static environment (road, sidewalk, lane lines, off-road)
    2. Future frame boxes — dynamic collision with vehicles/pedestrians at future
       positions (simlingo-style, most accurate for moving actors)
    3. Measurements — CARLA simulator hazard flags (light_hazard for red light)

    Priority for collision detection:
    - If future_frames_data is provided → use dynamic collision (most accurate)
    - Else if boxes is provided → filter same-dir vehicles from BEV, use BEV collision
    - Else → use raw BEV collision (least accurate for dynamic objects)

    Args:
        anchor_centers_abs: (num_modes, num_points, 2) absolute ego-centric coords
        bev_semantic: (H, W) uint8 BEV semantic map
        ppm: pixels per meter for BEV
        bev_size: BEV image size
        off_road_threshold: fraction of trajectory off-road to classify as off_road
        sidewalk_threshold: fraction on sidewalk to classify as on_sidewalk
        stop_distance_threshold: total displacement below this = stop behavior
        collision_near_weight: extra weight for near-half BEV collision
        boxes: optional list of dicts from boxes/*.json.gz for BEV dynamic filtering
        measurements: optional dict from measurements/*.json.gz (light_hazard, etc.)
        ego_matrix_current: optional (4,4) ego world transform for coordinate conversion
        future_frames_data: optional list of (boxes, ego_matrix) per future timestep
        gt_trajectory: optional (num_points, 2) expert trajectory waypoints.
            Expert never collides, so anchors close to GT are safe → suppress
            false-positive collision labels for nearby anchors.
        gt_safe_dist: distance threshold (meters) for GT-based correction.
            Anchors with mean distance to GT < gt_safe_dist are considered safe.

    Returns:
        behavior_labels: (num_modes,) int behavior category
        allowed_flags: (num_modes,) bool allowed/forbidden
        semantic_features: (num_modes, 11) float class fractions per anchor
    """
    # GT trajectory reference: compute per-anchor distance to expert trajectory
    # Expert never collides, so nearby anchors should be safe
    gt_nearby_mask = np.zeros(anchor_centers_abs.shape[0], dtype=bool)
    if gt_trajectory is not None:
        num_points = anchor_centers_abs.shape[1]
        gt_pts = gt_trajectory[:num_points]  # align length
        if len(gt_pts) == num_points:
            # Mean L2 distance across all waypoints
            dists = np.linalg.norm(anchor_centers_abs - gt_pts[np.newaxis, :, :], axis=2)
            mean_dists = dists.mean(axis=1)  # (num_modes,)
            gt_nearby_mask = mean_dists < gt_safe_dist

    # Dynamic collision via future frames (simlingo-style, highest priority)
    use_dynamic_collision = (
        future_frames_data is not None
        and ego_matrix_current is not None
        and len(future_frames_data) > 0
    )
    dynamic_veh_collision = None
    dynamic_ped_collision = None
    if use_dynamic_collision:
        dynamic_veh_collision, dynamic_ped_collision = _check_dynamic_collision_future(
            anchor_centers_abs, np.array(ego_matrix_current), future_frames_data
        )
        # Suppress false-positive collisions for anchors near GT trajectory
        if gt_nearby_mask.any():
            dynamic_veh_collision[gt_nearby_mask] = False
            dynamic_ped_collision[gt_nearby_mask] = False

    # Filter BEV: erase same-direction moving vehicles (fallback for BEV collision)
    if boxes is not None:
        bev_semantic = _filter_bev_by_boxes(bev_semantic, boxes, ppm, bev_size)

    # Measurements-based flags
    light_hazard = False
    if measurements is not None:
        light_hazard = measurements.get('light_hazard', False)

    num_modes = anchor_centers_abs.shape[0]
    num_classes = 11  # BEV classes 0-10

    behavior_labels = np.zeros(num_modes, dtype=np.int64)
    allowed_flags = np.ones(num_modes, dtype=bool)
    semantic_features = np.zeros((num_modes, num_classes), dtype=np.float32)

    for m in range(num_modes):
        waypoints = anchor_centers_abs[m]  # (num_points, 2)

        # Total displacement
        total_disp = np.linalg.norm(waypoints[-1] - waypoints[0])

        # Interpolate trajectory
        interp_pts = _interpolate_trajectory(waypoints, step_m=0.5)
        n_pts = len(interp_pts)

        # Convert to BEV pixels
        pixel_coords = _ego_to_bev_pixels(interp_pts, ppm, bev_size)

        # Sample BEV classes
        classes = _sample_bev_classes(bev_semantic, pixel_coords)

        # Compute class fractions (overall, for semantic_features output)
        for c in range(num_classes):
            semantic_features[m, c] = np.mean(classes == c)

        # --- Collision detection ---
        # GT-nearby anchors are safe (expert never collides)
        is_gt_nearby = gt_nearby_mask[m]

        if is_gt_nearby:
            # Expert trajectory is close → no collision possible
            has_vehicle_collision = False
            has_pedestrian_collision = False
        elif use_dynamic_collision:
            # Use future-frame dynamic collision (most accurate)
            has_vehicle_collision = dynamic_veh_collision[m]
            has_pedestrian_collision = dynamic_ped_collision[m]
        else:
            # Fallback: BEV-based collision with near/far weighting
            mid = max(n_pts // 2, 1)
            classes_near = classes[:mid]
            classes_far = classes[mid:]

            vehicle_frac_near = np.mean(classes_near == BEV_VEHICLE) if len(classes_near) > 0 else 0.0
            pedestrian_frac_near = np.mean(classes_near == BEV_PEDESTRIAN) if len(classes_near) > 0 else 0.0
            vehicle_frac_far = np.mean(classes_far == BEV_VEHICLE) if len(classes_far) > 0 else 0.0
            pedestrian_frac_far = np.mean(classes_far == BEV_PEDESTRIAN) if len(classes_far) > 0 else 0.0

            vehicle_score = vehicle_frac_near * collision_near_weight + vehicle_frac_far
            pedestrian_score = pedestrian_frac_near * collision_near_weight + pedestrian_frac_far

            has_vehicle_collision = vehicle_score > 0.05
            has_pedestrian_collision = pedestrian_score > 0.05

        # Detect line crossings (BEV-based, always reliable)
        crosses_solid, crosses_dashed = _detect_line_crossings(classes)

        # Static features (BEV-based, always reliable)
        background_frac = semantic_features[m, BEV_BACKGROUND]
        sidewalk_frac = semantic_features[m, BEV_SIDEWALK]
        red_light_frac = semantic_features[m, BEV_RED_LIGHT]

        # Red light: use measurements flag if available (more reliable than BEV pixels)
        # If light_hazard=True and anchor goes forward significantly, it's running red
        has_red_light = red_light_frac > 0
        if light_hazard and total_disp > 5.0:
            has_red_light = True

        # Lateral displacement of anchor trajectory (for direction classification)
        lateral_disp = waypoints[-1, 1] - waypoints[0, 1]

        # Classification by priority (highest priority first)
        if has_vehicle_collision:
            # Subdivide collision by anchor direction: front/left/right
            if abs(lateral_disp) < LATERAL_DIR_THRESH:
                behavior_labels[m] = COLLISION_FRONT
            elif lateral_disp < 0:
                behavior_labels[m] = COLLISION_LEFT
            else:
                behavior_labels[m] = COLLISION_RIGHT
            allowed_flags[m] = False
        elif has_pedestrian_collision:
            behavior_labels[m] = COLLISION_PEDESTRIAN
            allowed_flags[m] = False
        elif background_frac > off_road_threshold:
            behavior_labels[m] = OFF_ROAD
            allowed_flags[m] = False
        elif sidewalk_frac > sidewalk_threshold:
            behavior_labels[m] = ON_SIDEWALK
            allowed_flags[m] = False
        elif has_red_light:
            behavior_labels[m] = RUN_RED_LIGHT
            allowed_flags[m] = False
        elif crosses_solid or crosses_dashed:
            # Subdivide lane change by direction
            if lateral_disp < 0:
                behavior_labels[m] = LANE_CHANGE_LEFT
            else:
                behavior_labels[m] = LANE_CHANGE_RIGHT
            allowed_flags[m] = crosses_dashed and not crosses_solid
        elif total_disp < stop_distance_threshold:
            behavior_labels[m] = STOP
            allowed_flags[m] = True
        else:
            behavior_labels[m] = FOLLOW_ROAD
            allowed_flags[m] = True

    return behavior_labels, allowed_flags, semantic_features


# ===========================================================================
# Scene Bucket Classification (adapted from simlingo carla_get_buckets.py)
# ===========================================================================

# Bucket category names for scene-level classification
BUCKET_CATEGORIES = [
    'vehicle_hazard',       # 0: vehicle hazard active
    'walker_hazard',        # 1: pedestrian hazard active
    'light_hazard',         # 2: red light hazard active
    'stop_sign_hazard',     # 3: stop sign hazard active
    'junction',             # 4: near junction (<10m)
    'red_light',            # 5: red light + near junction
    'green_light',          # 6: green light + near junction
    'vehicle_front',        # 7: hazard vehicle coming from front
    'vehicle_side',         # 8: hazard vehicle coming from side
    'brake',                # 9: braking
    'start_from_stop',      # 10: starting from standstill
    'high_lateral',         # 11: significant lateral control (turning)
    'high_decel',           # 12: hard deceleration
    'low_speed',            # 13: low target speed (<2 m/s)
]
NUM_BUCKET_CATEGORIES = len(BUCKET_CATEGORIES)


def classify_scene_buckets(
    measurements: Optional[dict] = None,
    boxes: Optional[list] = None,
    ego_waypoints: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Classify the current scene into bucket categories based on measurements,
    boxes, and waypoints. Adapted from simlingo's carla_get_buckets.py.

    Each bucket is a binary flag indicating whether the scene belongs to that
    category. A sample can belong to multiple buckets simultaneously.

    Args:
        measurements: dict from measurements/*.json.gz
        boxes: list of dicts from boxes/*.json.gz
        ego_waypoints: (N, 2) expert trajectory waypoints (origin-excluded)

    Returns:
        bucket_flags: (NUM_BUCKET_CATEGORIES,) bool array
    """
    flags = np.zeros(NUM_BUCKET_CATEGORIES, dtype=bool)

    if measurements is None:
        return flags

    # --- Hazard flags ---
    if measurements.get('vehicle_hazard', False):
        flags[0] = True  # vehicle_hazard
    if measurements.get('walker_hazard', False):
        flags[1] = True  # walker_hazard
    if measurements.get('light_hazard', False):
        flags[2] = True  # light_hazard
    if measurements.get('stop_sign_hazard', False):
        flags[3] = True  # stop_sign_hazard

    # --- Junction / traffic light (from measurements + boxes) ---
    # measurements has 'junction' (bool)
    # traffic_light boxes have 'state', 'affects_ego', 'distance'
    is_junction = measurements.get('junction', False)
    if is_junction:
        flags[4] = True  # junction

    # Find traffic light state from boxes (affects_ego + close)
    traffic_light_state = None
    vehicle_affecting = None
    if boxes is not None:
        for box in boxes:
            if box.get('class') == 'traffic_light' and box.get('affects_ego', False):
                tl_dist = box.get('distance', 100)
                if tl_dist < 20:
                    traffic_light_state = box.get('state', None)
            if box.get('class') == 'car':
                affecting_id = measurements.get('vehicle_affecting_id')
                if affecting_id is not None and box.get('id') == affecting_id:
                    vehicle_affecting = box

    if traffic_light_state == 'Red':
        flags[5] = True  # red_light
    if traffic_light_state == 'Green':
        flags[6] = True  # green_light

    # --- Vehicle hazard direction (front vs side) ---
    is_red_light = flags[5]
    if flags[0] and vehicle_affecting is not None:  # vehicle_hazard
        yaw = vehicle_affecting.get('yaw', 0.0)
        # Front: yaw ≈ π (oncoming), Side: yaw not aligned with ego
        if abs(yaw) > np.pi - 0.6 and abs(yaw) < np.pi + 0.6:
            if not is_red_light:
                flags[7] = True  # vehicle_front
        elif abs(yaw) > 0.5:
            if not is_red_light:
                flags[8] = True  # vehicle_side

    # --- Brake ---
    brake = measurements.get('brake', False) or measurements.get('control_brake', False)
    if brake:
        flags[9] = True  # brake

    # --- Start from stop ---
    current_speed = measurements.get('speed', 0.0)
    target_speed = measurements.get('target_speed', 0.0)
    if current_speed < 0.5 and target_speed > 0.8:
        flags[10] = True  # start_from_stop

    # --- Lateral control (turning) ---
    if ego_waypoints is not None and len(ego_waypoints) >= 2:
        lateral_control = np.abs(np.mean(ego_waypoints[:, 1]))
        if lateral_control > 2.0:
            flags[11] = True  # high_lateral

    # --- Deceleration ---
    speed = measurements.get('speed', 0.0)
    target = measurements.get('target_speed', 0.0)
    if speed > 2.0 and target < speed * 0.5:
        flags[12] = True  # high_decel

    # --- Low target speed ---
    if target_speed < 2.0:
        flags[13] = True  # low_speed

    return flags


# ===========================================================================
# Visualization and standalone testing
# ===========================================================================

def visualize_anchor_labels(
    anchor_centers_abs: np.ndarray,
    bev_semantic: np.ndarray,
    behavior_labels: np.ndarray,
    allowed_flags: np.ndarray,
    save_path: str = '/tmp/anchor_labels_viz.png',
    ppm: float = 2.0,
    bev_size: int = 256,
    gt_trajectory: Optional[np.ndarray] = None,
):
    """
    Visualize anchor trajectories on BEV with behavior color coding.
    """
    from PIL import Image, ImageDraw, ImageFont

    # BEV class colors (RGB)
    bev_colors = {
        0: (30, 30, 30),       # background - dark gray
        1: (180, 180, 180),    # road - light gray
        2: (220, 220, 220),    # sidewalk - white-ish
        3: (255, 255, 0),      # solid line - yellow
        4: (50, 234, 157),     # dashed line - green
        5: (160, 160, 0),      # stop sign
        6: (0, 200, 0),        # green light
        7: (200, 200, 0),      # yellow light
        8: (200, 0, 0),        # red light
        9: (250, 170, 30),     # vehicle - orange
        10: (0, 200, 0),       # pedestrian - green
    }

    # Behavior colors for trajectory drawing
    behavior_colors = {
        FOLLOW_ROAD: (0, 180, 0),            # green
        COLLISION_FRONT: (255, 0, 0),        # red
        COLLISION_LEFT: (255, 80, 80),       # light red
        COLLISION_RIGHT: (180, 0, 0),        # dark red
        COLLISION_PEDESTRIAN: (255, 0, 255), # magenta
        OFF_ROAD: (180, 0, 180),             # purple
        ON_SIDEWALK: (255, 128, 0),          # orange
        RUN_RED_LIGHT: (255, 50, 50),        # bright red
        LANE_CHANGE_LEFT: (0, 150, 255),     # blue
        LANE_CHANGE_RIGHT: (0, 80, 200),     # dark blue
        STOP: (128, 128, 128),               # gray
    }

    # Scale up for visibility
    scale = 3
    sz = bev_size * scale

    # Create color BEV base image
    bev_color = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    for cls, color in bev_colors.items():
        bev_color[bev_semantic == cls] = color

    # Scale up
    from PIL import Image as PILImage
    base_img = PILImage.fromarray(bev_color).resize((sz, sz), PILImage.NEAREST)
    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    center = bev_size / 2.0

    # Draw each anchor
    for m in range(len(anchor_centers_abs)):
        wps = anchor_centers_abs[m]
        behavior = behavior_labels[m]
        allowed = allowed_flags[m]
        color = behavior_colors.get(behavior, (255, 255, 255))

        # If forbidden, make line dashed (draw with reduced alpha / thinner)
        width = 2 if allowed else 1

        # Convert to BEV pixel coords and scale
        pts = []
        for wp in wps:
            c = (center + wp[0] * ppm) * scale
            r = (center + wp[1] * ppm) * scale
            pts.append((c, r))

        # Draw trajectory line
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=width)

        # Draw endpoint marker
        end = pts[-1]
        marker_size = 3
        draw.ellipse([end[0] - marker_size, end[1] - marker_size,
                       end[0] + marker_size, end[1] + marker_size],
                      fill=color, outline=(0, 0, 0))

    # Find closest anchor to GT
    closest_idx = -1
    closest_dist = float('inf')
    if gt_trajectory is not None:
        num_points = anchor_centers_abs.shape[1]
        gt_pts = gt_trajectory[:num_points]
        if len(gt_pts) == num_points:
            dists = np.linalg.norm(anchor_centers_abs - gt_pts[np.newaxis, :, :], axis=2)
            mean_dists = dists.mean(axis=1)
            closest_idx = int(np.argmin(mean_dists))
            closest_dist = mean_dists[closest_idx]

    # Draw GT trajectory if provided
    if gt_trajectory is not None:
        pts = []
        for wp in gt_trajectory:
            c = (center + wp[0] * ppm) * scale
            r = (center + wp[1] * ppm) * scale
            pts.append((c, r))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=(255, 255, 255), width=3)
        for pt in pts:
            draw.ellipse([pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4],
                          fill=(255, 255, 255), outline=(0, 0, 0))

    # Highlight closest-to-GT anchor with yellow outline
    if closest_idx >= 0:
        wps = anchor_centers_abs[closest_idx]
        pts = []
        for wp in wps:
            c = (center + wp[0] * ppm) * scale
            r = (center + wp[1] * ppm) * scale
            pts.append((c, r))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=(255, 255, 0), width=3)
        end = pts[-1]
        draw.ellipse([end[0] - 5, end[1] - 5, end[0] + 5, end[1] + 5],
                      fill=(255, 255, 0), outline=(0, 0, 0))

    # Draw ego position
    ego_c = center * scale
    ego_r = center * scale
    draw.ellipse([ego_c - 6, ego_r - 6, ego_c + 6, ego_r + 6],
                  fill=(0, 255, 255), outline=(0, 0, 0))

    # Add legend
    legend_x = 10
    legend_y = sz - 20 * (NUM_BEHAVIORS + 2) - 10
    for i, name in enumerate(BEHAVIOR_NAMES):
        color = behavior_colors.get(i, (255, 255, 255))
        y = legend_y + i * 20
        draw.rectangle([legend_x, y, legend_x + 15, y + 15], fill=color, outline=(0, 0, 0))
        draw.text((legend_x + 20, y), name, fill=(255, 255, 255))
    # GT legend entries
    y = legend_y + NUM_BEHAVIORS * 20
    draw.rectangle([legend_x, y, legend_x + 15, y + 15], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((legend_x + 20, y), "GT expert", fill=(255, 255, 255))
    y += 20
    draw.rectangle([legend_x, y, legend_x + 15, y + 15], fill=(255, 255, 0), outline=(0, 0, 0))
    draw.text((legend_x + 20, y), "closest to GT", fill=(255, 255, 255))

    # Add stats text
    n_allowed = np.sum(allowed_flags)
    n_forbidden = np.sum(~allowed_flags)
    draw.text((10, 10), f"Allowed: {n_allowed}  Forbidden: {n_forbidden}",
              fill=(255, 255, 255))
    if closest_idx >= 0:
        b = BEHAVIOR_NAMES[behavior_labels[closest_idx]]
        a = "allowed" if allowed_flags[closest_idx] else "FORBIDDEN"
        draw.text((10, 25), f"Closest#{closest_idx}: {b} ({a}) d={closest_dist:.1f}m",
                  fill=(255, 255, 0))

    img.save(save_path)
    print(f"Saved visualization to {save_path}")
    return img


def test_on_dataset(dataset_root: str, anchor_path: str, n_samples: int = 5):
    """
    Test anchor labeling on actual dataset samples.
    """
    import pickle
    import glob
    import os
    from PIL import Image

    # Load anchors
    with open(anchor_path, 'rb') as f:
        anchor_data = pickle.load(f)
    anchor_centers_abs = anchor_data['centers']  # (32, N, 2)
    print(f"Loaded {len(anchor_centers_abs)} anchors, shape: {anchor_centers_abs.shape}")

    # Find sample pkl files
    pkl_files = sorted(glob.glob(os.path.join(dataset_root, 'train', '*.pkl')))
    if not pkl_files:
        pkl_files = sorted(glob.glob(os.path.join(dataset_root, '*.pkl')))
    print(f"Found {len(pkl_files)} samples")

    # Process a few samples
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(pkl_files)), min(n_samples, len(pkl_files)))

    stats = {name: 0 for name in BEHAVIOR_NAMES}
    allowed_count = 0
    total_count = 0

    for idx in sample_indices:
        pkl_path = pkl_files[idx]
        with open(pkl_path, 'rb') as f:
            sample = pickle.load(f)

        # Derive BEV semantic path from transfuser feature path
        # feature: "Accident/Town12_.../transfuser_feature/0006_feature.pt"
        # bev:     "Accident/Town12_.../bev_semantics/0006.png"
        feature_rel = sample.get('transfuser_bev_feature', '')
        bev_rel = feature_rel.replace('transfuser_feature/', 'bev_semantics/').replace('_feature.pt', '.png')
        bev_path = os.path.join(dataset_root, bev_rel)

        if not os.path.exists(bev_path):
            print(f"  BEV not found: {bev_path}")
            continue

        bev_semantic = np.array(Image.open(bev_path))

        # Label anchors
        behavior_labels, allowed_flags, semantic_features = label_anchors_semantic(
            anchor_centers_abs, bev_semantic
        )

        # Stats
        for b in behavior_labels:
            stats[BEHAVIOR_NAMES[b]] += 1
        allowed_count += np.sum(allowed_flags)
        total_count += len(allowed_flags)

        # Visualize
        gt_traj = sample.get('ego_waypoints', None)
        if gt_traj is not None:
            gt_traj = gt_traj[1:]  # Remove origin point

        save_path = f'/tmp/anchor_labels_sample_{idx}.png'
        visualize_anchor_labels(
            anchor_centers_abs, bev_semantic,
            behavior_labels, allowed_flags,
            save_path=save_path,
            gt_trajectory=gt_traj
        )

        print(f"  Sample {idx}: {os.path.basename(pkl_path)}")
        print(f"    Behaviors: {dict(zip(BEHAVIOR_NAMES, [np.sum(behavior_labels == i) for i in range(NUM_BEHAVIORS)]))}")
        print(f"    Allowed: {np.sum(allowed_flags)}/{len(allowed_flags)}")

    print(f"\n=== Overall Stats ({total_count} anchor-sample pairs) ===")
    for name, count in stats.items():
        print(f"  {name}: {count} ({100*count/max(total_count,1):.1f}%)")
    print(f"  Allowed: {allowed_count}/{total_count} ({100*allowed_count/max(total_count,1):.1f}%)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test anchor semantic labeling')
    parser.add_argument('--dataset', type=str,
                        default='/media/z/data/dataset/pdm_lite_mini',
                        help='Dataset root path')
    parser.add_argument('--anchors', type=str,
                        default='/media/z/data/mzq/others/MoT-DP/wp_tokens.pkl',
                        help='Anchor tokens path')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of samples to test')
    args = parser.parse_args()

    test_on_dataset(args.dataset, args.anchors, args.n_samples)
