#!/usr/bin/env python3
"""
Pre-compute lightweight debug/cache fields and inject them into samples_packed.pkl.

This keeps the packed samples aligned with the current stage1 debug workflow.
Each sample gets these fast fields:
  - ego_status:      (obs_horizon, 14) float32
  - conflict_area_family / dir / active / start_frame / end_frame
  - conflict_decision_phase / conflict_go_frame

Usage:
  python scripts/data_tools/precompute_semantic_labels.py \
    --dataset_path /media/z/data/dataset/pdm_lite_mini/train \
    --image_data_root /media/z/data/dataset/pdm_lite_mini

  # Also for val split:
  python scripts/data_tools/precompute_semantic_labels.py \
    --dataset_path /media/z/data/dataset/pdm_lite_mini/val \
    --image_data_root /media/z/data/dataset/pdm_lite_mini
"""

import os
import sys
import argparse
import pickle
import numpy as np
import json
import gzip
import glob
import time
from collections import defaultdict
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


FAST_FIELDS = (
    'ego_status',
    'conflict_area_family',
    'conflict_area_dir',
    'conflict_area_active',
    'conflict_area_start_frame',
    'conflict_area_end_frame',
    'conflict_decision_phase',
    'conflict_go_frame',
)


ACTOR_CLASS_NONE = 0
ACTOR_CLASS_VEHICLE = 1
ACTOR_CLASS_BICYCLE = 2
ACTOR_CLASS_PEDESTRIAN = 3

ACTOR_CLASS_NAMES = {
    ACTOR_CLASS_NONE: 'none',
    ACTOR_CLASS_VEHICLE: 'vehicle',
    ACTOR_CLASS_BICYCLE: 'bicycle',
    ACTOR_CLASS_PEDESTRIAN: 'pedestrian',
}

ACTOR_CLASS_WEIGHTS = {
    ACTOR_CLASS_NONE: 0.0,
    ACTOR_CLASS_VEHICLE: 1.0,
    ACTOR_CLASS_BICYCLE: 2.0,
    ACTOR_CLASS_PEDESTRIAN: 3.0,
}

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'vehicle'}
BICYCLE_CLASSES = {'bicycle', 'bike', 'cyclist'}
PEDESTRIAN_CLASSES = {'pedestrian', 'walker'}
DEFAULT_EGO_EXTENT_2D = np.array([2.5, 1.0], dtype=np.float32)
COMMAND_MAP = {
    1: 'LEFT',
    2: 'RIGHT',
    3: 'STRAIGHT',
    4: 'LANE_FOLLOW',
    5: 'CHANGE_LEFT',
    6: 'CHANGE_RIGHT',
}
LEFT_COMMAND_ID = 1
RIGHT_COMMAND_ID = 2
LANE_FOLLOW_COMMAND_ID = 4
INTERACTION_SAME_DIR_ANGLE_THRESH_DEG = 45.0
INTERACTION_CROSS_MIN_ANGLE_THRESH_DEG = 70.0
CONFLICT_DIR_SAME_MAX_ANGLE_DEG = 45.0
CONFLICT_DIR_OPPOSITE_MIN_ANGLE_DEG = 135.0
MERGE_DEBUG_MIN_DEGO_M = 1.0
MERGE_DEBUG_MIN_GO_DENOM_S = 0.10
NO_ROUTE_EXTENSION_SCENES = {'HazardAtSideLane'}
TWOWAY_START_LATERAL_THRESH_M = 0.5
TWOWAY_RETURN_TAIL_POINTS = 12
TWOWAY_RETURN_FALLBACK_EXTRA_POINT_INDEX = 10
TWOWAY_BORROW_ONEWAY_ROUTE_OVERRIDES = {
    "Town13_Rep0_1313_0_route0_11_09_06_18_06",
    "Town13_Rep0_1313_1_route0_11_08_23_22_02",
    "Town13_Rep0_1315_0_route0_11_09_00_34_48",
    "Town13_Rep0_1315_1_route0_11_09_06_09_23",
}
STAGE1_MERGE_START_CONFIRM_FRAMES = 2
STAGE1_MERGE_SPEED_CAP_MPS = 1000.0
STAGE1_MERGE_CONFLICT_LOOKAHEAD_PROGRESS_M = 15.0
STAGE1_MERGE_CONFLICT_CLUSTER_GAP_M = 4.0
STAGE1_MERGE_AREA_POST_MARGIN_M = 10.0
STAGE1_FUTURE_START_GATE_CHASE_SPEED_THRESH_MPS = 0.5
STAGE1_FUTURE_START_GATE_CHASE_DISTANCE_THRESH_M = 15.0
STAGE1_JUNCTION_CROSS_MIN_CLUSTER_POINTS = 2
STAGE1_JUNCTION_CROSS_FALLBACK_RADIUS_M = 7.5
STAGE1_JUNCTION_AREA_PRE_MARGIN_M = 10.0
STAGE1_JUNCTION_AREA_POST_MARGIN_M = 10.0
STAGE1_CONFLICT_GO_STOP_SPEED_THRESH_MPS = 0.1
STAGE1_CONFLICT_GO_START_SPEED_THRESH_MPS = 0.5
STAGE1_CONFLICT_AREA_ENTRY_TOL_M = 0.5

STAGE1_SPEED_FIELDS = (
    'conflict_area_family',
    'conflict_area_dir',
    'conflict_area_active',
    'conflict_area_start_frame',
    'conflict_area_end_frame',
    'conflict_decision_phase',
    'conflict_go_frame',
)

CONFLICT_FAMILY_TO_CODE = {
    'none': 0,
    'borrow': 1,
    'merge': 2,
    'junction': 3,
}

CONFLICT_DIR_TO_CODE = {
    'none': 0,
    'same': 1,
    'opposite': 2,
    'cross': 3,
}

CONFLICT_FAMILY_PRIORITY = {
    'borrow': 0,
    'merge': 1,
    'junction': 2,
}

CONFLICT_DECISION_PHASE_TO_CODE = {
    'none': 0,
    'yld': 1,
    'go': 2,
}


def _has_all_fast_fields(sample):
    return all(field in sample for field in FAST_FIELDS)


def _infer_num_future_points(samples, default_num_points=6):
    for sample in samples:
        ego_waypoints = sample.get('ego_waypoints')
        if ego_waypoints is None:
            continue
        ego_waypoints = np.asarray(ego_waypoints, dtype=np.float32)
        if ego_waypoints.ndim == 2 and ego_waypoints.shape[0] >= 2:
            return max(int(ego_waypoints.shape[0] - 1), 1)
    return int(default_num_points)


def _extract_target_points(sample):
    target_point_hist = np.asarray(sample['target_point_hist'], dtype=np.float32)
    if target_point_hist.shape[-1] == 4:
        return target_point_hist[..., :2], target_point_hist[..., 2:]

    target_point_next_hist = sample.get('target_point_next_hist', target_point_hist)
    target_point_next_hist = np.asarray(target_point_next_hist, dtype=np.float32)
    return target_point_hist[..., :2], target_point_next_hist[..., :2]


def _build_ego_status(sample):
    speed_hist = sample.get('speed_hist', sample.get('speed'))
    if speed_hist is None:
        raise KeyError("missing 'speed_hist'/'speed' for ego_status precompute")

    target_point_hist, target_point_next_hist = _extract_target_points(sample)
    ego_status = np.concatenate([
        np.asarray(speed_hist, dtype=np.float32)[..., None],
        np.asarray(sample['theta_hist'], dtype=np.float32)[..., None],
        np.asarray(sample['command_hist'], dtype=np.float32),
        target_point_hist,
        target_point_next_hist,
        np.asarray(sample['waypoints_hist'], dtype=np.float32),
    ], axis=-1)
    return ego_status.astype(np.float32)


def _resolve_feature_frame_info(sample):
    feature_rel = sample.get('transfuser_bev_feature', '')
    frame_id = sample.get('frame_id', None)
    if not feature_rel or frame_id is None:
        return None, None

    if 'route_features.pt' in feature_rel:
        base_dir = os.path.dirname(os.path.dirname(feature_rel))
        frame_str = f'{int(frame_id):04d}'
    else:
        base_dir = os.path.dirname(os.path.dirname(feature_rel))
        frame_str = os.path.basename(feature_rel).replace('_feature.pt', '')
    return base_dir, frame_str


def _load_json_gz_if_exists(path):
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    except Exception:
        return None


def _prepend_route_origin(route):
    route = np.asarray(route, dtype=np.float32)
    if route.ndim != 2 or route.shape[0] == 0 or route.shape[1] != 2:
        return None
    if np.linalg.norm(route[0]) < 1e-4:
        return route
    return np.concatenate([np.zeros((1, 2), dtype=np.float32), route], axis=0)


def _route_with_origin(route):
    return _prepend_route_origin(route)


def _interpolate_route_with_arclength(route_points, step_m=0.25):
    route_points = np.asarray(route_points, dtype=np.float32)
    if route_points.shape[0] == 1:
        return route_points.copy(), np.zeros(1, dtype=np.float32)

    dense_pts = [route_points[0]]
    dense_s = [0.0]
    cumulative = 0.0

    for i in range(route_points.shape[0] - 1):
        p0 = route_points[i]
        p1 = route_points[i + 1]
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            continue
        n_steps = max(int(np.ceil(seg_len / step_m)), 1)
        for j in range(1, n_steps + 1):
            t = j / n_steps
            dense_pts.append(p0 + t * seg)
            dense_s.append(cumulative + t * seg_len)
        cumulative += seg_len

    return np.asarray(dense_pts, dtype=np.float32), np.asarray(dense_s, dtype=np.float32)


def _points_inside_oriented_box(points, center, extent, yaw, margin_m=0.0):
    points = np.asarray(points, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    extent = np.asarray(extent, dtype=np.float32)
    if (
        points.ndim != 2 or points.shape[1] != 2 or
        center.shape[0] != 2 or extent.shape[0] != 2
    ):
        return np.zeros(points.shape[0], dtype=np.bool_)

    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))

    # Transform from ego/world frame into the actor local frame.
    local_x = dx * cos_y + dy * sin_y
    local_y = -dx * sin_y + dy * cos_y

    half_len = float(extent[0]) + margin_m
    half_wid = float(extent[1]) + margin_m
    return (np.abs(local_x) <= half_len) & (np.abs(local_y) <= half_wid)


def _oriented_box_corners(center, extent, yaw):
    center = np.asarray(center, dtype=np.float32)
    extent = np.asarray(extent, dtype=np.float32)
    if center.shape != (2,) or extent.shape != (2,):
        return np.zeros((0, 2), dtype=np.float32)

    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))
    rot = np.asarray([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float32)
    local = np.asarray([
        [ extent[0],  extent[1]],
        [ extent[0], -extent[1]],
        [-extent[0], -extent[1]],
        [-extent[0],  extent[1]],
    ], dtype=np.float32)
    return center[None, :] + local @ rot.T


def _project_points_to_axis(points, axis):
    points = np.asarray(points, dtype=np.float32)
    axis = np.asarray(axis, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2 or axis.shape != (2,):
        return np.inf, -np.inf
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        return np.inf, -np.inf
    axis = axis / axis_norm
    proj = points @ axis
    return float(np.min(proj)), float(np.max(proj))


def _oriented_boxes_intersect(center_a, extent_a, yaw_a, center_b, extent_b, yaw_b):
    corners_a = _oriented_box_corners(center_a, extent_a, yaw_a)
    corners_b = _oriented_box_corners(center_b, extent_b, yaw_b)
    if corners_a.shape[0] == 0 or corners_b.shape[0] == 0:
        return False

    axes = [
        np.asarray([np.cos(yaw_a), np.sin(yaw_a)], dtype=np.float32),
        np.asarray([-np.sin(yaw_a), np.cos(yaw_a)], dtype=np.float32),
        np.asarray([np.cos(yaw_b), np.sin(yaw_b)], dtype=np.float32),
        np.asarray([-np.sin(yaw_b), np.cos(yaw_b)], dtype=np.float32),
    ]
    for axis in axes:
        min_a, max_a = _project_points_to_axis(corners_a, axis)
        min_b, max_b = _project_points_to_axis(corners_b, axis)
        if max_a < min_b or max_b < min_a:
            return False
    return True


def _route_corridor_half_lengths(route_s, min_half_len_m=0.0):
    route_s = np.asarray(route_s, dtype=np.float32)
    if route_s.ndim != 1 or route_s.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    min_half_len_m = float(max(float(min_half_len_m), 0.05))
    if route_s.shape[0] == 1:
        return np.asarray([min_half_len_m], dtype=np.float32)

    step = np.maximum(np.diff(route_s), 0.0).astype(np.float32)
    local_step = np.zeros_like(route_s, dtype=np.float32)
    local_step[0] = step[0]
    local_step[-1] = step[-1]
    if route_s.shape[0] > 2:
        local_step[1:-1] = np.maximum(step[:-1], step[1:])

    return np.maximum(0.5 * local_step, min_half_len_m).astype(np.float32)


def _transform_box_to_current_frame(box, transform):
    pos = box.get('position', None)
    if pos is None or len(pos) < 2:
        return None

    pos_h = np.array([pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0, 1.0], dtype=np.float32)
    pos_cur = np.asarray(transform, dtype=np.float32) @ pos_h

    yaw_future = float(box.get('yaw', 0.0))
    heading_future = np.array([np.cos(yaw_future), np.sin(yaw_future), 0.0, 0.0], dtype=np.float32)
    heading_cur = np.asarray(transform, dtype=np.float32) @ heading_future
    yaw_cur = float(np.arctan2(heading_cur[1], heading_cur[0]))

    box_cur = dict(box)
    box_cur['position'] = [float(pos_cur[0]), float(pos_cur[1]), float(pos_cur[2])]
    box_cur['yaw'] = yaw_cur
    return box_cur


def _canonical_actor_class(box):
    cls = str(box.get('class', '')).lower()
    if cls == 'ego_car':
        return ACTOR_CLASS_NONE
    if any(token in cls for token in PEDESTRIAN_CLASSES):
        return ACTOR_CLASS_PEDESTRIAN
    if any(token in cls for token in BICYCLE_CLASSES):
        return ACTOR_CLASS_BICYCLE
    if cls in VEHICLE_CLASSES:
        return ACTOR_CLASS_VEHICLE
    return ACTOR_CLASS_NONE


def _actor_weight(actor_class_id):
    return float(ACTOR_CLASS_WEIGHTS.get(int(actor_class_id), 1.0))


def _blocking_risk_from_distance(distance_m, safe_distance_m):
    if not np.isfinite(distance_m):
        return 0.0
    return float(np.clip((safe_distance_m - distance_m) / max(safe_distance_m, 1e-6), 0.0, 1.0))


def _block_severity_bin(distance_m):
    if not np.isfinite(distance_m):
        return 0
    if distance_m < 5.0:
        return 4
    if distance_m < 10.0:
        return 3
    if distance_m < 20.0:
        return 2
    if distance_m < 30.0:
        return 1
    return 0


def _ttc_severity_bin(ttc_s):
    if not np.isfinite(ttc_s):
        return 0
    if ttc_s < 1.0:
        return 4
    if ttc_s < 2.0:
        return 3
    if ttc_s < 3.0:
        return 2
    if ttc_s < 5.0:
        return 1
    return 0


def _bump_vru_hazard(severity, actor_class_id):
    if actor_class_id in (ACTOR_CLASS_BICYCLE, ACTOR_CLASS_PEDESTRIAN) and severity > 0:
        return min(int(severity) + 1, 4)
    return int(severity)


def _find_route_cover_point(
    route_dense,
    route_s,
    box,
    corridor_margin_m,
    corridor_half_len_m=0.0,
    route_half_lens=None,
):
    actor_class_id = _canonical_actor_class(box)
    if actor_class_id == ACTOR_CLASS_NONE:
        return None

    pos = box.get('position', None)
    extent = box.get('extent', None)
    if pos is None or extent is None or len(pos) < 2 or len(extent) < 2:
        return None

    route_dense = np.asarray(route_dense, dtype=np.float32)
    route_s = np.asarray(route_s, dtype=np.float32)
    if route_dense.ndim != 2 or route_dense.shape[0] == 0 or route_dense.shape[1] != 2:
        return None
    if route_s.ndim != 1 or route_s.shape[0] != route_dense.shape[0]:
        return None
    if route_half_lens is not None:
        route_half_lens = np.asarray(route_half_lens, dtype=np.float32)
        if route_half_lens.ndim != 1 or route_half_lens.shape[0] != route_dense.shape[0]:
            route_half_lens = None

    actor_center = np.asarray(pos[:2], dtype=np.float32)
    actor_extent = np.asarray(extent[:2], dtype=np.float32)
    actor_yaw = float(box.get('yaw', 0.0))
    corridor_half_wid = float(max(float(corridor_margin_m), 1e-3))
    if route_half_lens is None:
        route_half_lens = _route_corridor_half_lengths(route_s, min_half_len_m=corridor_half_len_m)
    actor_radius = float(np.linalg.norm(actor_extent))

    # Broad-phase coarse filter: if the distance between the actor center and a
    # route corridor cell center exceeds the sum of their circumradii, the two
    # oriented boxes cannot intersect. This preserves the exact SAT result while
    # skipping most route indices before the expensive box-box test.
    cell_radii = np.sqrt(np.square(route_half_lens) + float(corridor_half_wid ** 2)).astype(np.float32)
    center_delta = route_dense[:, :2] - actor_center[None, :]
    center_dist_sq = np.einsum('ij,ij->i', center_delta, center_delta).astype(np.float32)
    candidate_indices = np.flatnonzero(center_dist_sq <= np.square(cell_radii + actor_radius))
    if candidate_indices.size == 0:
        return None

    first_idx = None
    first_heading = None
    first_half_len = None
    for route_idx in candidate_indices:
        route_idx = int(route_idx)
        route_heading = _route_heading_at_idx(route_dense, route_idx)
        if route_heading is None:
            continue
        corridor_half_len = float(route_half_lens[route_idx])
        if _oriented_boxes_intersect(
            center_a=route_dense[route_idx],
            extent_a=np.asarray([corridor_half_len, corridor_half_wid], dtype=np.float32),
            yaw_a=float(route_heading),
            center_b=actor_center,
            extent_b=actor_extent,
            yaw_b=actor_yaw,
        ):
            first_idx = int(route_idx)
            first_heading = float(route_heading)
            first_half_len = float(corridor_half_len)
            break
    if first_idx is None:
        return None
    contact_distance = float(route_s[first_idx] + first_half_len)
    if route_s.shape[0] > 0:
        contact_distance = float(min(contact_distance, float(route_s[-1])))
    contact_point = np.asarray(route_dense[first_idx], dtype=np.float32)
    if first_heading is not None and first_half_len is not None:
        contact_point = contact_point + np.asarray(
            [np.cos(first_heading), np.sin(first_heading)],
            dtype=np.float32,
        ) * float(first_half_len)
    return {
        'route_idx': first_idx,
        'route_point': contact_point,
        'route_center_point': route_dense[first_idx],
        'route_distance': contact_distance,
        'route_heading': first_heading,
        'corridor_half_len': first_half_len,
        'corridor_half_wid': corridor_half_wid,
    }


def _find_ego_box(current_boxes, default_extent_2d=DEFAULT_EGO_EXTENT_2D):
    current_boxes = current_boxes or []
    for box in current_boxes:
        if str(box.get('class', '')).lower() != 'ego_car':
            continue
        pos = box.get('position', None)
        extent = box.get('extent', None)
        if pos is None or extent is None or len(pos) < 2 or len(extent) < 2:
            continue
        ego_box = dict(box)
        ego_box['position'] = [float(pos[0]), float(pos[1]), float(pos[2] if len(pos) > 2 else 0.0)]
        ego_box['extent'] = [float(extent[0]), float(extent[1]), float(extent[2] if len(extent) > 2 else 0.0)]
        ego_box['yaw'] = float(box.get('yaw', 0.0))
        return ego_box

    return {
        'class': 'ego_car',
        'position': [0.0, 0.0, 0.0],
        'extent': [float(default_extent_2d[0]), float(default_extent_2d[1]), 0.0],
        'yaw': 0.0,
    }


def _route_progress_inside_box(route_dense, route_s, box, margin_m=0.0):
    pos = box.get('position', None)
    extent = box.get('extent', None)
    if pos is None or extent is None or len(pos) < 2 or len(extent) < 2:
        return 0.0

    mask = _points_inside_oriented_box(
        route_dense,
        center=np.asarray(pos[:2], dtype=np.float32),
        extent=np.asarray(extent[:2], dtype=np.float32),
        yaw=float(box.get('yaw', 0.0)),
        margin_m=margin_m,
    )
    if not np.any(mask):
        return 0.0
    return float(np.max(route_s[mask]))


def _point_to_oriented_box_distance(point, box, margin_m=0.0):
    point = np.asarray(point, dtype=np.float32)
    pos = box.get('position', None)
    extent = box.get('extent', None)
    if point.shape != (2,) or pos is None or extent is None or len(pos) < 2 or len(extent) < 2:
        return np.inf

    center = np.asarray(pos[:2], dtype=np.float32)
    extent = np.asarray(extent[:2], dtype=np.float32)
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    yaw = float(box.get('yaw', 0.0))
    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))

    local_x = dx * cos_y + dy * sin_y
    local_y = -dx * sin_y + dy * cos_y
    qx = abs(local_x) - (float(extent[0]) + margin_m)
    qy = abs(local_y) - (float(extent[1]) + margin_m)
    return float(np.hypot(max(qx, 0.0), max(qy, 0.0)))


def _risk_from_ttc(ttc_s, safe_ttc_s):
    if not np.isfinite(ttc_s):
        return 0.0
    return float(np.clip((safe_ttc_s - ttc_s) / max(safe_ttc_s, 1e-6), 0.0, 1.0))


def _ensure_packed_samples(dataset_path):
    packed_path = os.path.join(dataset_path, 'samples_packed.pkl')
    if os.path.exists(packed_path):
        return packed_path

    sample_files = sorted(
        fp for fp in glob.glob(os.path.join(dataset_path, '*.pkl'))
        if os.path.basename(fp) != 'samples_packed.pkl'
    )
    if not sample_files:
        raise FileNotFoundError(
            f"{packed_path} not found and no per-sample pkl files were found in {dataset_path}"
        )

    print(f"{packed_path} not found. Packing {len(sample_files)} pkl files first...")
    samples = [None] * len(sample_files)
    for i, sample_path in enumerate(tqdm(sample_files, desc="Packing pkl", leave=False)):
        with open(sample_path, 'rb') as f:
            samples[i] = pickle.load(f)

    tmp_path = packed_path + f'.tmp.{os.getpid()}'
    with open(tmp_path, 'wb') as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.rename(tmp_path, packed_path)
    print(f"Packed samples saved to {packed_path} ({os.path.getsize(packed_path) / 1e6:.1f} MB)")
    return packed_path


def _compute_front_route_label(
    route,
    current_boxes,
    ego_speed,
    ego_matrix_current=None,
    future_frames_data=None,
    corridor_margin_m=0.5,
    route_step_m=0.25,
    max_distance_m=40.0,
    safe_ttc_s=3.0,
    max_ttc_s=10.0,
    block_safe_distance_m=30.0,
    return_debug=False,
):
    def _base_result(distance, ttc, risk, block_risk, case, has_lead, actor_class_id):
        block_bin = _block_severity_bin(distance) if int(case) == 1 else 0
        ttc_bin = _ttc_severity_bin(ttc) if int(case) == 2 else 0
        hazard_bin = _bump_vru_hazard(max(block_bin, ttc_bin), actor_class_id)
        return {
            'distance': float(distance),
            'ttc': float(ttc),
            'risk': float(risk),
            'block_risk': float(block_risk),
            'case': int(case),
            'has_lead': float(has_lead),
            'actor_class': int(actor_class_id),
            'actor_weight': float(_actor_weight(actor_class_id)),
            'block_bin': int(block_bin),
            'ttc_bin': int(ttc_bin),
            'hazard_bin': int(hazard_bin),
        }

    debug = {
        'route_poly': None,
        'route_dense': None,
        'route_s': None,
        'best_current': None,
        'best_future': None,
    }

    route_poly = _prepend_route_origin(route)
    debug['route_poly'] = route_poly
    if route_poly is None:
        result = _base_result(max_distance_m, max_ttc_s, 0.0, 0.0, 0, 0.0, ACTOR_CLASS_NONE)
        return (result, debug) if return_debug else result

    route_dense, route_s = _interpolate_route_with_arclength(route_poly, step_m=route_step_m)
    debug['route_dense'] = route_dense
    debug['route_s'] = route_s
    if route_dense.shape[0] == 0:
        result = _base_result(max_distance_m, max_ttc_s, 0.0, 0.0, 0, 0.0, ACTOR_CLASS_NONE)
        return (result, debug) if return_debug else result

    ego_speed = float(max(ego_speed, 0.0))
    current_boxes = current_boxes or []
    current_boxes_by_id = {
        box.get('id'): box for box in current_boxes if box.get('id') is not None
    }
    current_cover_actor_ids = set()
    ego_box = _find_ego_box(current_boxes)
    ego_route_front_s = _route_progress_inside_box(route_dense, route_s, ego_box, margin_m=0.0)
    ego_extent = np.asarray(ego_box.get('extent', DEFAULT_EGO_EXTENT_2D)[:2], dtype=np.float32)
    corridor_half_len_m = float(ego_extent[0]) if ego_extent.shape[0] >= 1 else float(DEFAULT_EGO_EXTENT_2D[0])
    corridor_half_width_m = float(max(
        float(corridor_margin_m),
        float(ego_extent[1]) if ego_extent.shape[0] >= 2 else 0.0,
    ))
    route_half_lens = _route_corridor_half_lengths(route_s, min_half_len_m=corridor_half_len_m)

    # Case 1: a vehicle already covers the current route -> pursuit / following problem.
    best_current = None
    for box in current_boxes:
        cover = _find_route_cover_point(
            route_dense, route_s, box,
            corridor_half_width_m,
            corridor_half_len_m=corridor_half_len_m,
            route_half_lens=route_half_lens,
        )
        if cover is None:
            continue
        actor_id = box.get('id', None)
        if actor_id is not None:
            current_cover_actor_ids.add(actor_id)

        lead_speed = float(abs(box.get('speed', 0.0)))
        closing_speed = max(ego_speed - lead_speed, 0.1)
        gap_distance = max(float(cover['route_distance']) - ego_route_front_s, 0.0)
        ttc = gap_distance / closing_speed
        actor_class_id = _canonical_actor_class(box)
        block_risk = _blocking_risk_from_distance(gap_distance, block_safe_distance_m)
        candidate = _base_result(
            np.clip(gap_distance, 0.0, max_distance_m),
            min(float(ttc), max_ttc_s),
            max(_risk_from_ttc(ttc, safe_ttc_s), block_risk),
            block_risk,
            1,
            1.0,
            actor_class_id,
        )
        if best_current is None or candidate['distance'] < best_current['distance']:
            best_current = candidate
            debug['best_current'] = {
                'box': dict(box),
                'cover': cover,
                'lead_speed': lead_speed,
                'closing_speed': closing_speed,
                'ego_route_front_s': float(ego_route_front_s),
                'gap_distance': float(gap_distance),
                'actor_class': ACTOR_CLASS_NAMES.get(actor_class_id, 'unknown'),
                'block_risk': float(block_risk),
            }

    # Case 2: no current cover, but a future vehicle will intersect the current route.
    if ego_matrix_current is None or future_frames_data is None or len(future_frames_data) == 0:
        result = best_current if best_current is not None else _base_result(
            max_distance_m, max_ttc_s, 0.0, 0.0, 0, 0.0, ACTOR_CLASS_NONE
        )
        return (result, debug) if return_debug else result

    try:
        ego_inv = np.linalg.inv(np.asarray(ego_matrix_current, dtype=np.float32))
    except np.linalg.LinAlgError:
        result = best_current if best_current is not None else _base_result(
            max_distance_m, max_ttc_s, 0.0, 0.0, 0, 0.0, ACTOR_CLASS_NONE
        )
        return (result, debug) if return_debug else result

    best_future = None
    future_ego_cover_cache = {}
    for frame_idx, frame_data in enumerate(future_frames_data):
        if frame_data is None:
            continue
        boxes_future, ego_matrix_future = frame_data
        transform = ego_inv @ np.asarray(ego_matrix_future, dtype=np.float32)

        for box_future in boxes_future:
            box_cur = _transform_box_to_current_frame(box_future, transform)
            if box_cur is None:
                continue

            cover = _find_route_cover_point(
                route_dense, route_s, box_cur,
                corridor_half_width_m,
                corridor_half_len_m=corridor_half_len_m,
                route_half_lens=route_half_lens,
            )
            if cover is None:
                continue

            actor_id = box_future.get('id')
            if actor_id is not None and actor_id in current_cover_actor_ids:
                continue
            current_box = current_boxes_by_id.get(actor_id, None)
            actor_cache_key = None if actor_id is None else (int(actor_id), int(frame_idx))
            if actor_cache_key is not None:
                blocked_by_ego_cover = future_ego_cover_cache.get(actor_cache_key, None)
                if blocked_by_ego_cover is None:
                    blocked_by_ego_cover = _actor_has_ego_cover_before_future_cover(
                        actor_id=int(actor_id),
                        current_box=current_box,
                        future_frames_data=future_frames_data,
                        ego_inv=ego_inv,
                        ego_box=ego_box,
                        stop_frame_idx=int(frame_idx),
                    )
                    future_ego_cover_cache[actor_cache_key] = bool(blocked_by_ego_cover)
                if blocked_by_ego_cover:
                    continue
            elif _boxes_have_cover(current_box, ego_box):
                continue
            if current_box is not None and len(current_box.get('position', [])) >= 2:
                bg_pos = np.asarray(current_box['position'][:2], dtype=np.float32)
                bg_speed = float(abs(current_box.get('speed', box_future.get('speed', 0.0))))
            else:
                bg_pos = np.asarray(box_cur['position'][:2], dtype=np.float32)
                bg_speed = float(abs(box_future.get('speed', 0.0)))

            conflict_pt = np.asarray(cover['route_point'], dtype=np.float32)
            raw_d_ego = float(cover['route_distance']) - ego_route_front_s
            # Ignore future-cover candidates whose first cover point is already
            # behind or inside the current ego-front progress. These usually
            # correspond to rear/behind-ego artifacts rather than actionable
            # future conflicts ahead of the route.
            if raw_d_ego <= 0.25:
                continue
            d_ego = max(raw_d_ego, 0.0)
            bg_box = current_box if current_box is not None else box_cur
            d_bg = _point_to_oriented_box_distance(conflict_pt, bg_box, margin_m=corridor_half_width_m)
            if not np.isfinite(d_bg):
                d_bg = float(np.linalg.norm(bg_pos - conflict_pt))
            meet_dist = d_ego + d_bg
            meet_speed = max(ego_speed + bg_speed, 0.1)
            ttc = meet_dist / meet_speed
            actor_class_id = _canonical_actor_class(current_box if current_box is not None else box_future)

            candidate = _base_result(
                np.clip(meet_dist, 0.0, max_distance_m),
                min(float(ttc), max_ttc_s),
                _risk_from_ttc(ttc, safe_ttc_s),
                0.0,
                2,
                1.0,
                actor_class_id,
            )
            if best_future is None or candidate['ttc'] < best_future['ttc']:
                best_future = candidate
                debug['best_future'] = {
                    'frame_index': int(frame_idx + 1),
                    'box_future': dict(box_future),
                    'box_current_frame': dict(box_cur),
                    'current_box': None if current_box is None else dict(current_box),
                    'cover': cover,
                    'ego_route_front_s': float(ego_route_front_s),
                    'bg_pos': np.asarray(bg_pos, dtype=np.float32),
                    'bg_speed': bg_speed,
                    'd_ego': float(d_ego),
                    'd_bg': float(d_bg),
                    'meet_dist': float(meet_dist),
                    'meet_speed': float(meet_speed),
                    'actor_class': ACTOR_CLASS_NAMES.get(actor_class_id, 'unknown'),
                }

    if best_current is not None:
        return (best_current, debug) if return_debug else best_current

    if best_future is not None:
        return (best_future, debug) if return_debug else best_future

    result = _base_result(max_distance_m, max_ttc_s, 0.0, 0.0, 0, 0.0, ACTOR_CLASS_NONE)
    return (result, debug) if return_debug else result


def _scene_name_from_base_dir(base_dir):
    if not base_dir:
        return None
    norm = str(base_dir).replace("\\", "/")
    return norm.split("/")[0] if "/" in norm else norm


def _box_class_name(box):
    return str((box or {}).get("class", "")).lower()


def _is_pedestrian_box(box):
    return _box_class_name(box) in PEDESTRIAN_CLASSES or any(
        token in _box_class_name(box) for token in PEDESTRIAN_CLASSES
    )


def _measurement_command_id(current_meas):
    if current_meas is None:
        return None
    cmd = current_meas.get("command", None)
    if cmd is None:
        return None
    try:
        return int(cmd)
    except Exception:
        return None


def _is_right_turn_junction_context(current_meas):
    if current_meas is None:
        return False
    return bool(current_meas.get("junction", False) and _measurement_command_id(current_meas) == RIGHT_COMMAND_ID)


def _is_right_turn_scene_context(current_meas, event_name=None):
    return _is_right_turn_junction_context(current_meas) or str(event_name or "") in {
        "NonSignalizedJunctionRightTurn",
        "SignalizedJunctionRightTurn",
    }


def _is_left_turn_junction_context(current_meas):
    if current_meas is None:
        return False
    return bool(current_meas.get("junction", False) and _measurement_command_id(current_meas) == LEFT_COMMAND_ID)


def _is_left_turn_scene_context(current_meas, event_name=None):
    return _is_left_turn_junction_context(current_meas) or str(event_name or "") in {
        "NonSignalizedJunctionLeftTurn",
        "SignalizedJunctionLeftTurn",
    }


def _is_borrow_cross_scene_context(event_name=None):
    return str(event_name or "") in {
        "ConstructionObstacleTwoWays",
        "AccidentTwoWays",
        "ParkedObstacleTwoWays",
        "VehicleOpensDoorTwoWays",
    }


EVENT_NAME_RECORD_EXCLUDED_SCENES = {
    "ConstructionObstacleTwoWays",
    "AccidentTwoWays",
    "ParkedObstacleTwoWays",
    "VehicleOpensDoorTwoWays",
}


def _attach_event_name_record(interaction, event_name=None):
    name = str(event_name or "")
    recorded = bool(name) and name not in {"None", "none", "nan", "NaN", "null", "Null"}
    recorded = recorded and name not in EVENT_NAME_RECORD_EXCLUDED_SCENES
    tagged = dict(interaction)
    tagged["event_name_recorded"] = bool(recorded)
    tagged["event_name_record"] = name if recorded else ""
    return tagged


def _ego_length_m(current_boxes, default_length_m=4.5):
    ego_box = _find_ego_box(current_boxes)
    if ego_box is None:
        return float(default_length_m)
    extent = ego_box.get("extent", None)
    if extent is None or len(extent) < 1:
        return float(default_length_m)
    try:
        return float(max(2.0 * float(extent[0]), 1.0))
    except Exception:
        return float(default_length_m)


def _box_length_m(box, default_length_m=4.5):
    if not isinstance(box, dict):
        return float(default_length_m)
    extent = box.get("extent", None)
    if extent is None or len(extent) < 1:
        return float(default_length_m)
    try:
        return float(max(2.0 * float(extent[0]), 1.0))
    except Exception:
        return float(default_length_m)


def _left_junction_conflict_len_m(current_meas, current_boxes, scale=1.5, event_name=None):
    if not _is_left_turn_scene_context(current_meas, event_name=event_name):
        return np.nan
    ego_length_m = _ego_length_m(current_boxes)
    return float(max(float(scale) * ego_length_m, ego_length_m))


def _wrap_to_pi(angle_rad):
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def _heading_to_deg(angle_rad):
    return float(np.degrees(_wrap_to_pi(angle_rad)))


def _route_heading_at_idx(route_dense, route_idx):
    route_dense = np.asarray(route_dense, dtype=np.float32)
    if route_dense.ndim != 2 or route_dense.shape[0] < 2 or route_dense.shape[1] != 2:
        return None
    idx = int(np.clip(route_idx, 0, route_dense.shape[0] - 1))
    lo = max(idx - 1, 0)
    hi = min(idx + 1, route_dense.shape[0] - 1)
    if hi == lo:
        return None
    vec = route_dense[hi] - route_dense[lo]
    if float(np.linalg.norm(vec)) < 1e-6:
        return None
    return float(np.arctan2(vec[1], vec[0]))


def _mean_angle_rad(angles_rad):
    angles = np.asarray(angles_rad, dtype=np.float32)
    if angles.ndim != 1 or angles.size == 0:
        return None
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return None
    mean_sin = float(np.mean(np.sin(angles)))
    mean_cos = float(np.mean(np.cos(angles)))
    if abs(mean_sin) < 1e-8 and abs(mean_cos) < 1e-8:
        return None
    return float(np.arctan2(mean_sin, mean_cos))


def _route_heading_mean_at_idx(route_dense, route_idx, radius=2):
    route_dense = np.asarray(route_dense, dtype=np.float32)
    if route_dense.ndim != 2 or route_dense.shape[0] < 2 or route_dense.shape[1] != 2:
        return None
    idx = int(np.clip(route_idx, 0, route_dense.shape[0] - 1))
    radius = int(max(radius, 0))
    headings = []
    for heading_idx in range(max(idx - radius, 0), min(idx + radius, route_dense.shape[0] - 1) + 1):
        heading = _route_heading_at_idx(route_dense, heading_idx)
        if heading is not None and np.isfinite(float(heading)):
            headings.append(float(heading))
    mean_heading = _mean_angle_rad(headings)
    if mean_heading is not None and np.isfinite(float(mean_heading)):
        return float(mean_heading)
    return _route_heading_at_idx(route_dense, idx)


def _route_heading_shape_near_idx(route_dense, route_idx, span=2, min_delta_deg=8.0):
    center_idx = int(route_idx)
    heading_prev = _route_heading_mean_at_idx(route_dense, center_idx - int(span), radius=1)
    heading_center = _route_heading_mean_at_idx(route_dense, center_idx, radius=1)
    heading_next = _route_heading_mean_at_idx(route_dense, center_idx + int(span), radius=1)
    if heading_prev is None or heading_center is None or heading_next is None:
        return {
            "valid": False,
            "prev_delta_deg": np.nan,
            "next_delta_deg": np.nan,
            "junction_like": False,
            "borrow_like": False,
        }
    prev_delta_deg = float(_heading_to_deg(float(heading_center) - float(heading_prev)))
    next_delta_deg = float(_heading_to_deg(float(heading_next) - float(heading_center)))
    junction_like = bool(
        abs(prev_delta_deg) >= float(min_delta_deg) and
        abs(next_delta_deg) >= float(min_delta_deg) and
        prev_delta_deg * next_delta_deg > 0.0
    )
    borrow_like = bool(
        abs(prev_delta_deg) >= float(min_delta_deg) and
        abs(next_delta_deg) >= float(min_delta_deg) and
        prev_delta_deg * next_delta_deg < 0.0
    )
    return {
        "valid": True,
        "prev_delta_deg": float(prev_delta_deg),
        "next_delta_deg": float(next_delta_deg),
        "junction_like": bool(junction_like),
        "borrow_like": bool(borrow_like),
    }


def _interaction_route_shape_debug(case, current_meas, event_name, route_shape):
    mismatch_reason = 'none'
    if bool(route_shape.get('valid', False)):
        if (
            _is_left_turn_scene_context(current_meas, event_name=event_name) and
            bool(route_shape.get('borrow_like', False)) and
            not bool(route_shape.get('junction_like', False))
        ):
            mismatch_reason = 'junction_shape_recovery'
        elif (
            _is_borrow_cross_scene_context(event_name=event_name) and
            bool(route_shape.get('junction_like', False)) and
            not bool(route_shape.get('borrow_like', False))
        ):
            mismatch_reason = 'borrow_shape_monotonic'
    shape_label = 'none'
    if bool(route_shape.get('junction_like', False)) and not bool(route_shape.get('borrow_like', False)):
        shape_label = 'junction_like'
    elif bool(route_shape.get('borrow_like', False)) and not bool(route_shape.get('junction_like', False)):
        shape_label = 'borrow_like'
    return {
        "route_shape_valid": bool(route_shape.get("valid", False)),
        "route_shape_prev_delta_deg": float(route_shape.get("prev_delta_deg", np.nan)),
        "route_shape_next_delta_deg": float(route_shape.get("next_delta_deg", np.nan)),
        "route_shape_label": str(shape_label),
        "route_shape_mismatch_reason": str(mismatch_reason),
        "route_shape_case": int(case),
    }


def _cover_route_shape_mismatch_reason(cover):
    interaction = ((cover or {}).get('interaction') or {})
    reason = str(interaction.get('route_shape_mismatch_reason', 'none'))
    return reason if reason and reason != 'none' else 'none'


def _record_window_cover_shape_issue(samples, records, family, positions):
    seen = set()
    for pos in positions or []:
        record = records[int(pos)]
        for cover_key in ('current_cover', 'future_cover'):
            cover = (record or {}).get(cover_key) or {}
            reason = _cover_route_shape_mismatch_reason(cover)
            if reason == 'none':
                continue
            key = (str(reason), int(pos), str(cover_key))
            if key in seen:
                continue
            seen.add(key)
            return _conflict_area_issue(
                samples[record['sample_idx']],
                family,
                f'{family}_cover_shape_mismatch',
                pos=int(pos),
                mismatch_reason=str(reason),
                cover_key=str(cover_key),
                cover_subtype=str(_cover_interaction_subtype(cover)),
            )
    return None


def _heading_from_motion(start_xy, end_xy, min_motion_m=0.5):
    start_xy = np.asarray(start_xy, dtype=np.float32)
    end_xy = np.asarray(end_xy, dtype=np.float32)
    if start_xy.shape != (2,) or end_xy.shape != (2,):
        return None, 0.0
    motion = end_xy - start_xy
    motion_norm = float(np.linalg.norm(motion))
    if motion_norm < float(min_motion_m):
        return None, motion_norm
    return float(np.arctan2(motion[1], motion[0])), motion_norm


def _interaction_signal_from_candidate(
    case,
    best,
    debug,
    current_meas=None,
    event_name=None,
    angle_thresh_deg=INTERACTION_SAME_DIR_ANGLE_THRESH_DEG,
    min_motion_m=0.5,
):
    if case <= 0 or best is None:
        return _attach_event_name_record({
            "mode": 0,
            "name": "none",
            "subtype": "none",
            "source": "none",
            "angle_deg": np.nan,
            "route_heading_deg": np.nan,
            "actor_heading_deg": np.nan,
            "motion_m": 0.0,
        }, event_name=event_name)

    cover = best.get("cover", {})
    actor_box = best.get("box") if case == 1 else (
        best.get("current_box") or best.get("box_future") or best.get("box_current_frame")
    )
    if _is_pedestrian_box(actor_box):
        return _attach_event_name_record({
            "mode": 2,
            "name": "meet",
            "subtype": "ped_cross",
            "source": "ped_corridor_cover",
            "angle_deg": np.nan,
            "route_heading_deg": np.nan,
            "actor_heading_deg": np.nan,
            "motion_m": 0.0,
        }, event_name=event_name)

    if case == 2:
        route_heading = _route_heading_mean_at_idx(debug.get("route_dense"), cover.get("route_idx", 0), radius=2)
    else:
        route_heading = _route_heading_at_idx(debug.get("route_dense"), cover.get("route_idx", 0))
    if route_heading is None:
        return _attach_event_name_record({
            "mode": 0,
            "name": "none",
            "subtype": "none",
            "source": "missing_route_heading",
            "angle_deg": np.nan,
            "route_heading_deg": np.nan,
            "actor_heading_deg": np.nan,
            "motion_m": 0.0,
        }, event_name=event_name)

    actor_heading = None
    motion_m = 0.0
    source = "yaw_vs_route"
    if case == 2:
        current_box = best.get("current_box")
        future_box = best.get("box_future")
        if current_box is not None and future_box is not None:
            _, motion_m = _heading_from_motion(
                np.asarray(current_box.get("position", [0.0, 0.0])[:2], dtype=np.float32),
                np.asarray((best.get("box_current_frame") or {}).get("position", [0.0, 0.0])[:2], dtype=np.float32),
                min_motion_m=min_motion_m,
            )
        if future_box is not None:
            actor_heading = float(future_box.get("yaw", 0.0))
            source = "future_yaw_vs_route_avg"
    else:
        box = best.get("box")
        if box is not None:
            actor_heading = float(box.get("yaw", 0.0))

    if actor_heading is None:
        if case in (1, 2) and _is_right_turn_scene_context(current_meas, event_name=event_name):
            return _attach_event_name_record({
                "mode": 2,
                "name": "meet",
                "subtype": "merge_meet",
                "source": "junction_right_scene_override_missing_heading",
                "angle_deg": np.nan,
                "route_heading_deg": _heading_to_deg(route_heading),
                "actor_heading_deg": np.nan,
                "motion_m": motion_m,
            }, event_name=event_name)
        return _attach_event_name_record({
            "mode": 0,
            "name": "none",
            "subtype": "none",
            "source": "missing_actor_heading",
            "angle_deg": np.nan,
            "route_heading_deg": _heading_to_deg(route_heading),
            "actor_heading_deg": np.nan,
            "motion_m": motion_m,
        }, event_name=event_name)

    route_shape = _route_heading_shape_near_idx(debug.get("route_dense"), cover.get("route_idx", 0), span=2)
    angle_deg = abs(_heading_to_deg(actor_heading - route_heading))
    same_direction = angle_deg <= float(angle_thresh_deg)
    cross_direction = angle_deg >= float(INTERACTION_CROSS_MIN_ANGLE_THRESH_DEG)
    if case == 2 and _is_right_turn_scene_context(current_meas, event_name=event_name):
        return _attach_event_name_record({
            "mode": 2,
            "name": "meet",
            "subtype": "merge_meet",
            "source": "junction_right_scene_future_override",
            "angle_deg": float(angle_deg),
            "route_heading_deg": _heading_to_deg(route_heading),
            "actor_heading_deg": _heading_to_deg(actor_heading),
            "motion_m": float(motion_m),
        }, event_name=event_name)
    if same_direction and case == 1:
        return _attach_event_name_record({
            "mode": 1,
            "name": "chase",
            "subtype": "follow_chase",
            "source": f"{source}+same_dir_current_cover",
            "angle_deg": float(angle_deg),
            "route_heading_deg": _heading_to_deg(route_heading),
            "actor_heading_deg": _heading_to_deg(actor_heading),
            "motion_m": float(motion_m),
        }, event_name=event_name)
    if case == 1 and _is_right_turn_scene_context(current_meas, event_name=event_name):
        return _attach_event_name_record({
            "mode": 2,
            "name": "meet",
            "subtype": "merge_meet",
            "source": f"{source}+junction_right_scene_current_override",
            "angle_deg": float(angle_deg),
            "route_heading_deg": _heading_to_deg(route_heading),
            "actor_heading_deg": _heading_to_deg(actor_heading),
            "motion_m": float(motion_m),
        }, event_name=event_name)

    subtype = "merge_meet"
    if same_direction and case == 2:
        source = f"{source}+same_dir_future_cover"
    elif not cross_direction:
        if case == 1:
            source = f"{source}+mid_angle_merge_current"
        else:
            source = f"{source}+mid_angle_merge_future"
    elif case == 1 and _is_left_turn_scene_context(current_meas, event_name=event_name):
        subtype = "junction_left_cross_meet"
        source = f"{source}+cross_dir+junction_left_cross_current"
    elif case == 1 and _is_borrow_cross_scene_context(event_name=event_name):
        subtype = "borrow_cross_meet"
        source = f"{source}+cross_dir+borrow_cross_current"
    elif case == 2 and _is_left_turn_scene_context(current_meas, event_name=event_name):
        subtype = "junction_left_cross_meet"
        source = f"{source}+cross_dir+junction_left_cross"
    elif case == 2 and _is_borrow_cross_scene_context(event_name=event_name):
        subtype = "borrow_cross_meet"
        source = f"{source}+cross_dir+borrow_cross"
    else:
        subtype = "cross_meet"
        source = f"{source}+cross_dir"
    interaction = {
        "mode": 2,
        "name": "meet",
        "subtype": subtype,
        "source": source,
        "angle_deg": float(angle_deg),
        "route_heading_deg": _heading_to_deg(route_heading),
        "actor_heading_deg": _heading_to_deg(actor_heading),
        "motion_m": float(motion_m),
    }
    interaction.update(_interaction_route_shape_debug(case, current_meas, event_name, route_shape))
    return _attach_event_name_record(interaction, event_name=event_name)


def _cover_candidate_summary(case, best, debug, current_meas=None, event_name=None):
    interaction = _interaction_signal_from_candidate(
        case=case,
        best=best,
        debug=debug,
        current_meas=current_meas,
        event_name=event_name,
    )
    if case <= 0 or best is None:
        return {
            "exists": 0.0,
            "case": int(case),
            "actor_id": -1,
            "frame_index": -1,
            "interaction": interaction,
            "actor_class_name": "none",
            "distance": np.nan,
            "ttc": np.nan,
            "block_risk": np.nan,
            "d_ego": np.nan,
            "d_bg": np.nan,
            "other_speed": np.nan,
            "other_length_m": np.nan,
            "route_distance_m": np.nan,
            "ego_route_front_s_m": np.nan,
            "route_point_local_xy": [],
            "scene_route_conflict_s_m": np.nan,
            "scene_route_conflict_world_xyz": [],
        }

    if case == 1:
        box = best.get("box", {})
        actor_id = int(box.get("id", -1)) if box.get("id", None) is not None else -1
        gap_distance = float(best.get("gap_distance", np.nan))
        closing_speed = float(best.get("closing_speed", np.nan))
        ttc = np.inf if not np.isfinite(closing_speed) or closing_speed <= 1e-6 else gap_distance / closing_speed
        cover = best.get("cover") or {}
        route_point_local = np.asarray(cover.get("route_point", []), dtype=np.float32).reshape(-1)
        return {
            "exists": 1.0,
            "case": int(case),
            "actor_id": int(actor_id),
            "frame_index": 0,
            "interaction": interaction,
            "actor_class_name": _box_class_name(box),
            "distance": float(gap_distance),
            "ttc": float(ttc),
            "block_risk": float(best.get("block_risk", np.nan)),
            "d_ego": np.nan,
            "d_bg": np.nan,
            "other_speed": float(best.get("lead_speed", np.nan)),
            "other_length_m": float(_box_length_m(box, default_length_m=np.nan)),
            "route_distance_m": float(cover.get("route_distance", np.nan)),
            "ego_route_front_s_m": float(best.get("ego_route_front_s", np.nan)),
            "route_point_local_xy": route_point_local[:2].astype(float).tolist() if route_point_local.size >= 2 else [],
            "scene_route_conflict_s_m": np.nan,
            "scene_route_conflict_world_xyz": [],
        }

    current_box = best.get("current_box") or {}
    box_future = best.get("box_future") or {}
    other_box = current_box if current_box else box_future
    actor_id = current_box.get("id", None)
    if actor_id is None:
        actor_id = box_future.get("id", None)
    actor_id = int(actor_id) if actor_id is not None else -1
    meet_dist = float(best.get("meet_dist", np.nan))
    meet_speed = float(best.get("meet_speed", np.nan))
    ttc = np.inf if not np.isfinite(meet_speed) or meet_speed <= 1e-6 else meet_dist / meet_speed
    cover = best.get("cover") or {}
    route_point_local = np.asarray(cover.get("route_point", []), dtype=np.float32).reshape(-1)
    return {
        "exists": 1.0,
        "case": int(case),
        "actor_id": int(actor_id),
        "frame_index": int(best.get("frame_index", -1)),
        "interaction": interaction,
        "actor_class_name": _box_class_name(current_box if current_box else box_future),
        "distance": float(meet_dist),
        "ttc": float(ttc),
        "block_risk": 0.0,
        "d_ego": float(best.get("d_ego", np.nan)),
        "d_bg": float(best.get("d_bg", np.nan)),
        "other_speed": float(best.get("bg_speed", np.nan)),
        "other_length_m": float(_box_length_m(other_box, default_length_m=np.nan)),
        "route_distance_m": float(cover.get("route_distance", np.nan)),
        "ego_route_front_s_m": float(best.get("ego_route_front_s", np.nan)),
        "route_point_local_xy": route_point_local[:2].astype(float).tolist() if route_point_local.size >= 2 else [],
        "scene_route_conflict_s_m": np.nan,
        "scene_route_conflict_world_xyz": [],
    }


def _augment_cover_with_scene_route_fields(cover, ego_matrix_current, scene_route_polyline_world):
    cover_out = dict(cover or {})
    cover_out.setdefault("scene_route_conflict_s_m", np.nan)
    cover_out.setdefault("scene_route_conflict_world_xyz", [])
    if int(cover_out.get("exists", 0.0)) <= 0:
        return cover_out
    route_point_local = np.asarray(cover_out.get("route_point_local_xy", []), dtype=np.float32).reshape(-1)
    scene_route_polyline_world = np.asarray(scene_route_polyline_world, dtype=np.float32)
    if (
        ego_matrix_current is None or route_point_local.size < 2 or
        scene_route_polyline_world.ndim != 2 or scene_route_polyline_world.shape[0] < 2
    ):
        return cover_out
    route_point_world = _transform_points_local_to_world_xyz(
        route_point_local[:2][None, :],
        ego_matrix_current,
    )
    if route_point_world.shape[0] == 0:
        return cover_out
    _, proj_s = _project_point_to_polyline(route_point_world[0, :2], scene_route_polyline_world[:, :2])
    cover_out["scene_route_conflict_world_xyz"] = route_point_world[0, :3].astype(float).tolist()
    cover_out["scene_route_conflict_s_m"] = float(proj_s) if proj_s is not None else np.nan
    return cover_out


def _merge_speed_cap(value, default=np.nan):
    try:
        value = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(np.clip(value, 0.0, float(STAGE1_MERGE_SPEED_CAP_MPS)))


def _find_box_by_id(boxes, actor_id):
    if actor_id is None:
        return None
    for box in boxes or []:
        if box.get("id") == actor_id:
            return box
    return None


def _approx_box_clearance_gap_m(box_a, box_b):
    if box_a is None or box_b is None:
        return np.nan
    pos_a = box_a.get("position", None)
    pos_b = box_b.get("position", None)
    ext_a = box_a.get("extent", None)
    ext_b = box_b.get("extent", None)
    if (
        pos_a is None or pos_b is None or ext_a is None or ext_b is None or
        len(pos_a) < 2 or len(pos_b) < 2 or len(ext_a) < 2 or len(ext_b) < 2
    ):
        return np.nan
    center_dist = float(np.linalg.norm(np.asarray(pos_a[:2], dtype=np.float32) - np.asarray(pos_b[:2], dtype=np.float32)))
    radius_a = float(np.linalg.norm(np.asarray(ext_a[:2], dtype=np.float32)))
    radius_b = float(np.linalg.norm(np.asarray(ext_b[:2], dtype=np.float32)))
    return float(max(center_dist - radius_a - radius_b, 0.0))


def _segment_intersects_oriented_box(point_a, point_b, box, margin_m=0.0):
    point_a = np.asarray(point_a, dtype=np.float32).reshape(-1)
    point_b = np.asarray(point_b, dtype=np.float32).reshape(-1)
    pos = None if box is None else box.get("position", None)
    extent = None if box is None else box.get("extent", None)
    if (
        point_a.size < 2 or point_b.size < 2 or pos is None or extent is None or
        len(pos) < 2 or len(extent) < 2
    ):
        return False
    segment = np.stack([point_a[:2], point_b[:2]], axis=0).astype(np.float32)
    if np.any(
        _points_inside_oriented_box(
            segment,
            center=np.asarray(pos[:2], dtype=np.float32),
            extent=np.asarray(extent[:2], dtype=np.float32),
            yaw=float(box.get("yaw", 0.0)),
            margin_m=margin_m,
        )
    ):
        return True
    corners = _oriented_box_corners(
        center=np.asarray(pos[:2], dtype=np.float32),
        extent=np.asarray(extent[:2], dtype=np.float32) + float(margin_m),
        yaw=float(box.get("yaw", 0.0)),
    )
    if corners.shape != (4, 2):
        return False
    for idx in range(4):
        edge_p0 = corners[idx]
        edge_p1 = corners[(idx + 1) % 4]
        inter_pt, _ = _line_segment_intersection_2d(point_a[:2], point_b[:2], edge_p0, edge_p1)
        if inter_pt is not None:
            return True
    return False


def _polyline_has_ego_cover(polyline_points_xy, ego_box):
    pts = np.asarray(polyline_points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return False
    if np.any(
        _points_inside_oriented_box(
            pts[:, :2],
            center=np.asarray(ego_box.get("position", [0.0, 0.0])[:2], dtype=np.float32),
            extent=np.asarray(ego_box.get("extent", DEFAULT_EGO_EXTENT_2D)[:2], dtype=np.float32),
            yaw=float(ego_box.get("yaw", 0.0)),
        )
    ):
        return True
    for idx in range(pts.shape[0] - 1):
        if _segment_intersects_oriented_box(pts[idx], pts[idx + 1], ego_box):
            return True
    return False


def _actor_has_ego_cover_before_future_cover(
    actor_id,
    current_box,
    future_frames_data,
    ego_inv,
    ego_box,
    stop_frame_idx,
):
    if actor_id is None:
        return False
    trajectory_points = []
    if current_box is not None:
        pos = current_box.get("position", None)
        if pos is not None and len(pos) >= 2:
            trajectory_points.append(np.asarray(pos[:2], dtype=np.float32))
    for prior_frame_idx in range(int(stop_frame_idx) + 1):
        frame_data = future_frames_data[prior_frame_idx]
        if frame_data is None:
            continue
        boxes_future, ego_matrix_future = frame_data
        actor_box_future = _find_box_by_id(boxes_future, actor_id)
        if actor_box_future is None:
            continue
        transform = np.asarray(ego_inv, dtype=np.float32) @ np.asarray(ego_matrix_future, dtype=np.float32)
        actor_box_cur = _transform_box_to_current_frame(actor_box_future, transform)
        pos = None if actor_box_cur is None else actor_box_cur.get("position", None)
        if pos is not None and len(pos) >= 2:
            trajectory_points.append(np.asarray(pos[:2], dtype=np.float32))
    if len(trajectory_points) >= 2 and _polyline_has_ego_cover(np.stack(trajectory_points, axis=0), ego_box):
        return True
    return False


def _pedestrian_corridor_margin_m(current_boxes, base_margin_m=0.5):
    ego_box = _find_ego_box(current_boxes)
    if ego_box is None:
        return float(base_margin_m)
    extent = ego_box.get("extent", None)
    if extent is None or len(extent) < 2:
        return float(base_margin_m)
    return float(max(float(base_margin_m), float(extent[1])))


def _filter_current_boxes_pedestrian(current_boxes):
    return [box for box in (current_boxes or []) if _is_pedestrian_box(box)]


def _filter_future_frames_pedestrian(future_frames):
    filtered = []
    for frame_data in future_frames or []:
        if frame_data is None:
            filtered.append(None)
            continue
        boxes, ego_matrix = frame_data
        ped_boxes = [box for box in (boxes or []) if _is_pedestrian_box(box)]
        filtered.append((ped_boxes, ego_matrix) if ped_boxes else None)
    return filtered


def _box_world_xy(box, ego_matrix_current=None):
    matrix = (box or {}).get("matrix", None)
    if isinstance(matrix, (list, tuple, np.ndarray)):
        matrix_arr = np.asarray(matrix, dtype=np.float32)
        if matrix_arr.shape == (4, 4):
            return matrix_arr[:2, 3].astype(np.float32)
    pos = (box or {}).get("position", None)
    if pos is None or len(pos) < 2:
        return None
    pos_xy = np.asarray(pos[:2], dtype=np.float32)
    if ego_matrix_current is not None:
        world = _transform_points_local_to_world_xyz(pos_xy[None, :], ego_matrix_current)
        if world.shape[0] > 0:
            return world[0, :2].astype(np.float32)
    return pos_xy.astype(np.float32)


def _box_world_xyz(box, ego_matrix_current=None):
    matrix = (box or {}).get("matrix", None)
    if isinstance(matrix, (list, tuple, np.ndarray)):
        matrix_arr = np.asarray(matrix, dtype=np.float32)
        if matrix_arr.shape == (4, 4):
            return matrix_arr[:3, 3].astype(np.float32)
    pos = (box or {}).get("position", None)
    if pos is None or len(pos) < 2:
        return None
    pos_xy = np.asarray(pos[:2], dtype=np.float32)
    if ego_matrix_current is not None:
        world = _transform_points_local_to_world_xyz(pos_xy[None, :], ego_matrix_current)
        if world.shape[0] > 0:
            return world[0, :3].astype(np.float32)
    if len(pos) >= 3:
        return np.asarray(pos[:3], dtype=np.float32)
    return np.asarray([float(pos_xy[0]), float(pos_xy[1]), 0.0], dtype=np.float32)


def _collect_scene_nonstatic_actor_ids(route_samples, image_root, speed_thresh_mps=0.25, motion_thresh_m=1.0):
    stats = {}
    for sample in route_samples or []:
        base_dir, frame_str = _resolve_feature_frame_info(sample)
        if base_dir is None or frame_str is None:
            continue
        boxes_path = os.path.join(image_root, base_dir, "boxes", f"{frame_str}.json.gz")
        boxes = _load_json_gz_if_exists(boxes_path)
        if boxes is None:
            continue
        ego_matrix_current = None
        meas_path = os.path.join(image_root, base_dir, "measurements", f"{frame_str}.json.gz")
        meas = _load_json_gz_if_exists(meas_path)
        if isinstance(meas, dict):
            ego_matrix_current = meas.get("ego_matrix", None)
        for box in boxes:
            cls = str(box.get("class", "")).lower()
            if cls in {"ego_car", "static"}:
                continue
            actor_id = box.get("id", None)
            pos_xy = _box_world_xy(box, ego_matrix_current=ego_matrix_current)
            if actor_id is None or pos_xy is None or pos_xy.shape != (2,):
                continue
            actor_id = int(actor_id)
            speed_abs = abs(float(box.get("speed", 0.0)))
            rec = stats.setdefault(actor_id, {"first_pos": pos_xy, "last_pos": pos_xy, "max_abs_speed": speed_abs})
            rec["last_pos"] = pos_xy
            rec["max_abs_speed"] = max(float(rec["max_abs_speed"]), speed_abs)
    nonstatic_ids = set()
    for actor_id, rec in stats.items():
        displacement = float(np.linalg.norm(np.asarray(rec["last_pos"]) - np.asarray(rec["first_pos"])))
        if float(rec["max_abs_speed"]) > float(speed_thresh_mps) or displacement > float(motion_thresh_m):
            nonstatic_ids.add(int(actor_id))
    return nonstatic_ids


def _filter_current_boxes_dynamic(current_boxes, dynamic_ids):
    filtered = []
    for box in current_boxes or []:
        cls = str(box.get("class", "")).lower()
        if cls == "ego_car":
            filtered.append(box)
            continue
        if cls == "static":
            continue
        actor_id = box.get("id", None)
        if actor_id is not None and int(actor_id) in dynamic_ids:
            filtered.append(box)
    return filtered


def _filter_future_frames_dynamic(future_frames_data, dynamic_ids, speed_thresh_mps=0.25):
    filtered = []
    for frame_data in future_frames_data or []:
        if frame_data is None:
            filtered.append(None)
            continue
        boxes_future, ego_matrix_future = frame_data
        keep = []
        for box in boxes_future or []:
            cls = str(box.get("class", "")).lower()
            if cls in {"ego_car", "static"}:
                continue
            actor_id = box.get("id", None)
            if actor_id is not None and int(actor_id) in dynamic_ids:
                keep.append(box)
                continue
            if abs(float(box.get("speed", 0.0))) > float(speed_thresh_mps):
                keep.append(box)
        filtered.append((keep, ego_matrix_future))
    return filtered


def _risk_from_time_gap(delta_t_s, safe_gap_s):
    if not np.isfinite(delta_t_s):
        return 0.0
    return float(np.clip((safe_gap_s - float(delta_t_s)) / max(float(safe_gap_s), 1e-6), 0.0, 1.0))


def _borrow_corridor_metrics(release_info, current_meas=None, route_local=None):
    info = release_info or {}
    borrow_distance_m = float(info.get("borrow_distance_m", np.nan))
    if not np.isfinite(borrow_distance_m) or borrow_distance_m <= 1e-3:
        return None

    ego_matrix_current = None if current_meas is None else current_meas.get("ego_matrix", None)
    end_local_xy = None
    start_local_xy = None
    if ego_matrix_current is not None:
        borrow_end_world_xyz = np.asarray(info.get("borrow_end_world_xyz", []), dtype=np.float32)
        borrow_start_world_xyz = np.asarray(info.get("borrow_start_world_xyz", []), dtype=np.float32)
        if borrow_end_world_xyz.shape == (3,):
            end_local = _transform_points_world_xyz_to_local(borrow_end_world_xyz[None, :], ego_matrix_current)
            if end_local.shape == (1, 2):
                end_local_xy = end_local[0, :2].astype(np.float32)
        if borrow_start_world_xyz.shape == (3,):
            start_local = _transform_points_world_xyz_to_local(borrow_start_world_xyz[None, :], ego_matrix_current)
            if start_local.shape == (1, 2):
                start_local_xy = start_local[0, :2].astype(np.float32)

    borrow_start_distance_m = float(info.get("borrow_start_distance_m", np.nan))
    borrow_end_distance_m = np.nan
    if start_local_xy is not None:
        if route_local is not None:
            route_poly = _route_with_origin(np.asarray(route_local, dtype=np.float32))
            _, start_s = _project_point_to_polyline(start_local_xy, route_poly)
            if start_s is not None and np.isfinite(float(start_s)):
                borrow_start_distance_m = float(max(float(start_s), 0.0))
            if end_local_xy is not None:
                _, end_s = _project_point_to_polyline(end_local_xy, route_poly)
                if end_s is not None and np.isfinite(float(end_s)):
                    borrow_end_distance_m = float(max(float(end_s), 0.0))
        elif np.isfinite(float(start_local_xy[0])):
            borrow_start_distance_m = float(max(float(start_local_xy[0]), 0.0))
    if not np.isfinite(borrow_end_distance_m) and end_local_xy is not None and np.isfinite(float(end_local_xy[0])):
        borrow_end_distance_m = float(max(float(end_local_xy[0]), 0.0))
    if not np.isfinite(borrow_start_distance_m):
        borrow_start_distance_m = 0.0
    borrow_start_distance_m = float(max(borrow_start_distance_m, 0.0))

    return {
        "borrow_start_distance_m": float(borrow_start_distance_m),
        "borrow_end_distance_m": float(borrow_end_distance_m) if np.isfinite(borrow_end_distance_m) else np.nan,
        "borrow_distance_m": float(borrow_distance_m),
        "borrow_total_clear_distance_m": float(max(borrow_start_distance_m + borrow_distance_m, 0.0)),
        "start_local_xy": None if start_local_xy is None else start_local_xy,
        "end_local_xy": None if end_local_xy is None else end_local_xy,
    }


def _borrow_actor_distance_to_corridor_end_m(current_boxes, actor_id, end_local_xy, fallback_distance_m=np.nan):
    if end_local_xy is None or np.asarray(end_local_xy, dtype=np.float32).shape != (2,):
        return float(fallback_distance_m)
    actor_box = _find_box_by_id(current_boxes, actor_id)
    if actor_box is None:
        return float(fallback_distance_m)
    dist = _point_to_oriented_box_distance(np.asarray(end_local_xy, dtype=np.float32), actor_box, margin_m=0.0)
    if np.isfinite(dist):
        return float(max(dist, 0.0))
    return float(fallback_distance_m)


def _transform_points_local_to_world_xyz(points_xy, ego_matrix):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((0, 3), dtype=np.float32)
    ego_matrix = np.asarray(ego_matrix, dtype=np.float32)
    pts_h = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=np.float32), np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    world = (ego_matrix @ pts_h.T).T
    return world[:, :3].astype(np.float32)


def _transform_points_world_xyz_to_local(points_xyz, ego_matrix):
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.zeros((0, 2), dtype=np.float32)
    ego_matrix = np.asarray(ego_matrix, dtype=np.float32)
    try:
        ego_inv = np.linalg.inv(ego_matrix)
    except np.linalg.LinAlgError:
        return np.zeros((0, 2), dtype=np.float32)
    pts_h = np.concatenate([pts[:, :3], np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    local = (ego_inv @ pts_h.T).T
    return local[:, :2].astype(np.float32)


def _dedupe_polyline(points_xy, min_step_m=0.5):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    kept = [pts[0]]
    for pt in pts[1:]:
        if float(np.linalg.norm(pt[:2] - kept[-1][:2])) >= float(min_step_m):
            kept.append(pt)
    if len(kept) == 1:
        kept.append(pts[-1])
    return np.asarray(kept, dtype=np.float32)


def _polyline_length_m(points_xy):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] <= 1 or pts.shape[1] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).sum())


def _polyline_arclengths(points_xy):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return np.zeros((0,), dtype=np.float32)
    if pts.shape[0] == 1:
        return np.zeros((1,), dtype=np.float32)
    seg_lens = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).astype(np.float32)
    return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(seg_lens, dtype=np.float32)], axis=0)


def _project_point_to_polyline(point_xy, polyline_xy):
    pts = np.asarray(polyline_xy, dtype=np.float32)
    point_xy = np.asarray(point_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2 or point_xy.shape != (2,):
        return None, None
    if pts.shape[0] == 1:
        return pts[0, :2].astype(np.float32), 0.0
    arc = _polyline_arclengths(pts)
    best_dist = np.inf
    best_proj = pts[0]
    best_s = 0.0
    for idx in range(pts.shape[0] - 1):
        p0 = pts[idx, :2]
        p1 = pts[idx + 1, :2]
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-8:
            proj = p0
            t = 0.0
        else:
            t = float(np.clip(np.dot(point_xy - p0, seg) / seg_len_sq, 0.0, 1.0))
            proj = p0 + t * seg
        dist = float(np.linalg.norm(point_xy - proj))
        if dist < best_dist:
            best_dist = dist
            best_proj = proj
            best_s = float(arc[idx] + t * np.linalg.norm(seg))
    return np.asarray(best_proj, dtype=np.float32), best_s


def _line_segment_intersection_2d(line_p0, line_p1, seg_p0, seg_p1, eps=1e-6):
    line_p0 = np.asarray(line_p0, dtype=np.float32)
    line_p1 = np.asarray(line_p1, dtype=np.float32)
    seg_p0 = np.asarray(seg_p0, dtype=np.float32)
    seg_p1 = np.asarray(seg_p1, dtype=np.float32)
    r = line_p1 - line_p0
    s = seg_p1 - seg_p0
    denom = float(r[0] * s[1] - r[1] * s[0])
    if abs(denom) < eps:
        return None, None
    qp = seg_p0 - line_p0
    t = float((qp[0] * s[1] - qp[1] * s[0]) / denom)
    u = float((qp[0] * r[1] - qp[1] * r[0]) / denom)
    if u < -eps or u > 1.0 + eps:
        return None, None
    return (line_p0 + t * r), float(np.clip(u, 0.0, 1.0))


def _sample_polyline_at_arclengths(polyline_xy, query_s):
    pts = np.asarray(polyline_xy, dtype=np.float32)
    query_s = np.asarray(query_s, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    if pts.shape[0] == 1:
        keep = query_s >= 0.0
        return np.repeat(pts[:1], int(np.count_nonzero(keep)), axis=0).astype(np.float32)
    arc = _polyline_arclengths(pts)
    total_len = float(arc[-1]) if arc.size > 0 else 0.0
    query_s = query_s[(query_s >= 0.0) & (query_s <= total_len + 1e-6)]
    if query_s.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    sampled = []
    seg_idx = 0
    for q in query_s:
        while seg_idx + 1 < arc.shape[0] and float(arc[seg_idx + 1]) < float(q):
            seg_idx += 1
        if seg_idx + 1 >= arc.shape[0]:
            sampled.append(pts[-1])
            continue
        seg_len = float(arc[seg_idx + 1] - arc[seg_idx])
        if seg_len < 1e-8:
            sampled.append(pts[seg_idx])
            continue
        t = float((q - arc[seg_idx]) / seg_len)
        sampled.append(pts[seg_idx] + t * (pts[seg_idx + 1] - pts[seg_idx]))
    return np.asarray(sampled, dtype=np.float32)


def _sample_polyline_xyz_at_arclengths(polyline_xyz, query_s):
    pts = np.asarray(polyline_xyz, dtype=np.float32)
    query_s = np.asarray(query_s, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    if pts.shape[0] == 1:
        keep = query_s >= 0.0
        return np.repeat(pts[:1, :3], int(np.count_nonzero(keep)), axis=0).astype(np.float32)
    arc = _polyline_arclengths(pts[:, :2])
    total_len = float(arc[-1]) if arc.size > 0 else 0.0
    query_s = query_s[(query_s >= 0.0) & (query_s <= total_len + 1e-6)]
    if query_s.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    sampled = []
    seg_idx = 0
    for q in query_s:
        while seg_idx + 1 < arc.shape[0] and float(arc[seg_idx + 1]) < float(q):
            seg_idx += 1
        if seg_idx + 1 >= arc.shape[0]:
            sampled.append(pts[-1, :3])
            continue
        seg_len = float(arc[seg_idx + 1] - arc[seg_idx])
        if seg_len < 1e-8:
            sampled.append(pts[seg_idx, :3])
            continue
        t = float((q - arc[seg_idx]) / seg_len)
        sampled.append(pts[seg_idx, :3] + t * (pts[seg_idx + 1, :3] - pts[seg_idx, :3]))
    return np.asarray(sampled, dtype=np.float32)


def _is_two_way_event_corridor_scene_context(event_name=None):
    return str(event_name or "") in {"ConstructionObstacleTwoWays", "AccidentTwoWays"}


def _summarize_signed_route_shift(route_local, shift_sign=-1.0, baseline_points=6):
    route_poly = _prepend_route_origin(route_local)
    if route_poly.shape[0] < 3:
        return None

    arc = _polyline_arclengths(route_poly)
    if arc.shape[0] != route_poly.shape[0]:
        return None

    baseline_count = min(max(int(baseline_points), 2), route_poly.shape[0])
    baseline_y = float(np.median(route_poly[:baseline_count, 1]))
    rel_y = route_poly[:, 1] - baseline_y
    signed_factor = float(np.sign(float(shift_sign)) or -1.0)
    signed_rel_y = signed_factor * rel_y
    peak_idx = int(np.argmax(signed_rel_y))
    return {
        "route_poly": route_poly.astype(np.float32),
        "arc": arc.astype(np.float32),
        "baseline_y": float(baseline_y),
        "rel_y": rel_y.astype(np.float32),
        "signed_rel_y": signed_rel_y.astype(np.float32),
        "peak_idx": int(peak_idx),
        "peak_signed_lateral_m": float(signed_rel_y[peak_idx]),
        "shift_sign": float(signed_factor),
    }


def _build_borrow_geom_from_shift_summary(summary, enter_idx, return_idx, segment_step_m=0.5, mode="strict"):
    route_poly = np.asarray(summary["route_poly"], dtype=np.float32)
    arc = np.asarray(summary["arc"], dtype=np.float32)
    rel_y = np.asarray(summary["rel_y"], dtype=np.float32)
    signed_rel_y = np.asarray(summary["signed_rel_y"], dtype=np.float32)
    enter_idx = int(enter_idx)
    return_idx = int(return_idx)
    if not (0 <= enter_idx < return_idx < route_poly.shape[0]):
        return None

    query_s = np.arange(
        float(arc[enter_idx]),
        float(arc[return_idx]) + 1e-6,
        float(max(segment_step_m, 1e-3)),
        dtype=np.float32,
    )
    if query_s.size == 0 or float(query_s[-1]) < float(arc[return_idx]) - 1e-4:
        query_s = np.concatenate([query_s, np.array([float(arc[return_idx])], dtype=np.float32)], axis=0)
    segment_local = _sample_polyline_at_arclengths(route_poly, query_s)
    return {
        "enter_idx": int(enter_idx),
        "return_idx": int(return_idx),
        "enter_s_m": float(arc[enter_idx]),
        "return_s_m": float(arc[return_idx]),
        "borrow_distance_m": float(max(arc[return_idx] - arc[enter_idx], 0.0)),
        "peak_lateral_m": float(np.max(np.abs(rel_y[enter_idx:return_idx + 1]))),
        "peak_signed_lateral_m": float(np.max(signed_rel_y[enter_idx:return_idx + 1])),
        "baseline_y_m": float(summary["baseline_y"]),
        "shift_sign": float(summary["shift_sign"]),
        "enter_local_xy": route_poly[enter_idx, :2].astype(np.float32),
        "return_local_xy": route_poly[return_idx, :2].astype(np.float32),
        "segment_local_xy": segment_local.astype(np.float32),
        "mode": str(mode),
        "return_abs_m": float(abs(signed_rel_y[return_idx])),
    }


def _estimate_borrow_points_from_signed_route_shift(
    route_local,
    shift_sign=-1.0,
    borrow_enter_lateral_thresh=1.25,
    return_lateral_thresh=0.8,
    min_enter_progress_m=4.0,
    min_return_progress_m=6.0,
    baseline_points=6,
    segment_step_m=0.5,
):
    summary = _summarize_signed_route_shift(
        route_local=route_local,
        shift_sign=shift_sign,
        baseline_points=baseline_points,
    )
    if summary is None:
        return None

    route_poly = np.asarray(summary["route_poly"], dtype=np.float32)
    arc = np.asarray(summary["arc"], dtype=np.float32)
    signed_rel_y = np.asarray(summary["signed_rel_y"], dtype=np.float32)

    enter_idx = None
    for idx in range(1, route_poly.shape[0]):
        if float(arc[idx]) < float(min_enter_progress_m):
            continue
        if float(signed_rel_y[idx]) >= float(borrow_enter_lateral_thresh):
            enter_idx = idx
            break
    if enter_idx is None:
        return None

    return_idx = None
    for idx in range(enter_idx + 1, route_poly.shape[0]):
        if float(arc[idx] - arc[enter_idx]) < float(min_return_progress_m):
            continue
        if float(signed_rel_y[idx]) <= float(return_lateral_thresh):
            return_idx = idx
            break
    if return_idx is None:
        return None

    return _build_borrow_geom_from_shift_summary(
        summary,
        enter_idx=enter_idx,
        return_idx=return_idx,
        segment_step_m=segment_step_m,
        mode="strict",
    )


def _estimate_borrow_points_from_signed_route_shift_relaxed(
    route_local,
    shift_sign=-1.0,
    borrow_enter_lateral_thresh=1.25,
    return_lateral_thresh=0.8,
    min_enter_progress_m=4.0,
    min_return_progress_m=6.0,
    baseline_points=6,
    segment_step_m=0.5,
    fallback_peak_lateral_thresh=2.5,
    fallback_return_abs_thresh=1.5,
    fallback_min_recovery_m=1.0,
):
    summary = _summarize_signed_route_shift(
        route_local=route_local,
        shift_sign=shift_sign,
        baseline_points=baseline_points,
    )
    if summary is None:
        return None

    route_poly = np.asarray(summary["route_poly"], dtype=np.float32)
    arc = np.asarray(summary["arc"], dtype=np.float32)
    signed_rel_y = np.asarray(summary["signed_rel_y"], dtype=np.float32)

    enter_idx = None
    for idx in range(1, route_poly.shape[0]):
        if float(arc[idx]) < float(min_enter_progress_m):
            continue
        if float(signed_rel_y[idx]) >= float(borrow_enter_lateral_thresh):
            enter_idx = idx
            break
    if enter_idx is None:
        return None

    peak_slice = signed_rel_y[enter_idx:]
    if peak_slice.size == 0:
        return None
    peak_idx = int(enter_idx + int(np.argmax(peak_slice)))
    peak_signed = float(signed_rel_y[peak_idx])
    if peak_signed < float(fallback_peak_lateral_thresh):
        return None

    for idx in range(peak_idx + 1, route_poly.shape[0]):
        if float(arc[idx] - arc[enter_idx]) < float(min_return_progress_m):
            continue
        if float(signed_rel_y[idx]) <= float(return_lateral_thresh):
            return _build_borrow_geom_from_shift_summary(
                summary,
                enter_idx=enter_idx,
                return_idx=idx,
                segment_step_m=segment_step_m,
                mode="strict",
            )

    return None


def _two_way_blocker_box_allowed(box, event_name=None, stop_speed_thresh_mps=0.25):
    cls = _box_class_name(box)
    if cls == "ego_car":
        return False
    event_name = str(event_name or "")
    if event_name == "ConstructionObstacleTwoWays":
        return bool(cls == "static")
    if event_name == "AccidentTwoWays":
        if cls not in {"car", "truck", "bus", "van", "vehicle"}:
            return False
        return bool(float(abs(box.get("speed", 0.0))) <= float(stop_speed_thresh_mps))
    return False


def _empty_two_way_borrow_context(
    event_name=None,
    failure_reason="unknown",
    failure_stage="unknown",
    seed_actor=None,
    best_candidate=None,
    route_candidate_count=0,
):
    seed_actor = dict(seed_actor or {})
    best_candidate = dict(best_candidate or {})
    return {
        "valid": 0.0,
        "ready": 0.0,
        "source": "event_blocker_route_missing",
        "failure_reason": str(failure_reason),
        "failure_stage": str(failure_stage),
        "event_name": str(event_name or ""),
        "route_candidate_count": int(route_candidate_count),
        "release_frame_id": -1,
        "enter_frame_id": -1,
        "return_frame_id": -1,
        "borrow_duration_s": 0.0,
        "release_to_return_s": 0.0,
        "borrow_start_distance_m": np.nan,
        "borrow_start_world_xyz": [],
        "borrow_end_world_xyz": [],
        "borrow_segment_world_xyz": [],
        "borrow_distance_m": np.nan,
        "peak_lateral_m": np.nan,
        "peak_signed_lateral_m": np.nan,
        "shift_sign": np.nan,
        "context_frame_id": int(best_candidate.get("frame_id", -1)),
        "anchor_actor_id": int(seed_actor.get("actor_id", best_candidate.get("actor_id", -1))),
        "anchor_distance_m": float(best_candidate.get("local_x", np.nan)),
        "anchor_world_xyz": list(best_candidate.get("world_xyz", seed_actor.get("seed_world_xyz", []))),
        "blocking_actor_id": int(seed_actor.get("actor_id", best_candidate.get("actor_id", -1))),
        "blocking_actor_class": str(seed_actor.get("actor_class", best_candidate.get("actor_class", "none"))),
        "blocking_actor_local_x_m": float(best_candidate.get("local_x", np.nan)),
        "blocking_actor_local_y_m": float(best_candidate.get("local_y", np.nan)),
        "seed_frame_id": int(seed_actor.get("seed_frame_id", -1)),
        "seed_priority": int(seed_actor.get("seed_priority", -1)),
        "route_return_abs_m": np.nan,
        "blocked_frame_id": -1,
    }


def _build_scene_global_two_way_blocker_cluster(
    frame_records,
    event_name=None,
    stop_speed_thresh_mps=0.1,
    link_distance_m=12.0,
    min_cluster_size=3,
):
    event_name = str(event_name or "")
    actor_obs = {}
    for record in frame_records or []:
        current_meas = record.get("current_meas")
        ego_matrix_current = None if current_meas is None else current_meas.get("ego_matrix", None)
        if ego_matrix_current is None:
            continue
        for box in record.get("current_boxes") or []:
            actor_id = box.get("id", None)
            if actor_id is None or not _two_way_blocker_box_allowed(
                box,
                event_name=event_name,
                stop_speed_thresh_mps=stop_speed_thresh_mps,
            ):
                continue
            world_xy = _box_world_xy(box, ego_matrix_current=ego_matrix_current)
            if world_xy is None:
                continue
            obs = actor_obs.setdefault(int(actor_id), {
                "actor_id": int(actor_id),
                "actor_class": str(_box_class_name(box)),
                "world_x": [],
                "world_y": [],
                "frame_ids": [],
            })
            obs["world_x"].append(float(world_xy[0]))
            obs["world_y"].append(float(world_xy[1]))
            obs["frame_ids"].append(int(record.get("frame_id", -1)))

    actor_nodes = []
    for actor_id, obs in actor_obs.items():
        if len(obs["frame_ids"]) <= 0:
            continue
        mean_x = float(np.mean(np.asarray(obs["world_x"], dtype=np.float32)))
        mean_y = float(np.mean(np.asarray(obs["world_y"], dtype=np.float32)))
        actor_nodes.append({
            "actor_id": int(actor_id),
            "actor_class": str(obs["actor_class"]),
            "obs_count": int(len(obs["frame_ids"])),
            "first_frame": int(min(obs["frame_ids"])),
            "last_frame": int(max(obs["frame_ids"])),
            "mean_world_xy": [mean_x, mean_y],
        })
    if not actor_nodes:
        return None

    visited = set()
    clusters = []
    for node in actor_nodes:
        actor_id = int(node["actor_id"])
        if actor_id in visited:
            continue
        stack = [actor_id]
        component_ids = []
        while stack:
            cur = int(stack.pop())
            if cur in visited:
                continue
            visited.add(cur)
            component_ids.append(cur)
            cur_node = next(item for item in actor_nodes if int(item["actor_id"]) == cur)
            cur_xy = np.asarray(cur_node["mean_world_xy"], dtype=np.float32)
            for other in actor_nodes:
                other_id = int(other["actor_id"])
                if other_id in visited or other_id == cur:
                    continue
                other_xy = np.asarray(other["mean_world_xy"], dtype=np.float32)
                if float(np.linalg.norm(cur_xy - other_xy)) <= float(link_distance_m):
                    stack.append(other_id)
        component = [item for item in actor_nodes if int(item["actor_id"]) in component_ids]
        if len(component) < int(min_cluster_size):
            continue
        total_obs = int(sum(int(item["obs_count"]) for item in component))
        first_frame = int(min(int(item["first_frame"]) for item in component))
        clusters.append({
            "actor_ids": [int(item["actor_id"]) for item in component],
            "actor_classes": [str(item["actor_class"]) for item in component],
            "actor_count": int(len(component)),
            "total_obs": int(total_obs),
            "first_frame": int(first_frame),
        })
    if not clusters:
        return None

    clusters.sort(key=lambda item: (-int(item["total_obs"]), -int(item["actor_count"]), int(item["first_frame"]), item["actor_ids"]))
    return dict(clusters[0])


def _find_best_two_way_cluster_candidate(
    route_candidates,
    frame_records,
    cluster_actor_ids,
    lateral_thresh_m=5.0,
):
    cluster_actor_ids = {int(actor_id) for actor_id in (cluster_actor_ids or [])}
    if not cluster_actor_ids:
        return None

    best_candidate = None
    for route_candidate in route_candidates:
        record = frame_records[int(route_candidate["record_idx"])]
        ego_matrix_current = None if record.get("current_meas") is None else record["current_meas"].get("ego_matrix", None)
        for box in record.get("current_boxes") or []:
            actor_id = box.get("id", None)
            if actor_id is None or int(actor_id) not in cluster_actor_ids:
                continue
            pos = box.get("position", None)
            if pos is None or len(pos) < 2:
                continue
            local_x = float(pos[0])
            local_y = float(pos[1])
            if local_x <= 0.0 or float(abs(local_y)) > float(lateral_thresh_m):
                continue
            world_xyz = _box_world_xyz(box, ego_matrix_current=ego_matrix_current)
            candidate = {
                "score": (float(local_x), float(abs(local_y)), -int(route_candidate["frame_id"])),
                "record_idx": int(route_candidate["record_idx"]),
                "frame_id": int(route_candidate["frame_id"]),
                "actor_id": int(actor_id),
                "actor_class": str(_box_class_name(box)),
                "local_x": float(local_x),
                "local_y": float(local_y),
                "world_xyz": [] if world_xyz is None else np.asarray(world_xyz, dtype=np.float32).astype(float).tolist(),
                "geom": dict(route_candidate["geom"]),
                "priority": int(route_candidate["priority"]),
            }
            if best_candidate is None or candidate["score"] < best_candidate["score"]:
                best_candidate = candidate
    return best_candidate


def _build_event_two_way_borrow_context(
    frame_records,
    event_name=None,
    route_step_m=0.5,
    borrow_start_mode="route_head",
    borrow_enter_lateral_thresh=1.25,
    return_lateral_thresh=0.8,
    min_enter_progress_m=4.0,
    min_return_progress_m=6.0,
    blocker_pre_shift_lateral_thresh=1.75,
    fallback_peak_lateral_thresh=2.5,
    fallback_return_abs_thresh=1.5,
    fallback_min_recovery_m=1.0,
):
    if not _is_two_way_event_corridor_scene_context(event_name=event_name):
        return None
    event_name = str(event_name or "")
    scene_global_cluster = None
    if event_name in {"AccidentTwoWays", "ConstructionObstacleTwoWays"}:
        scene_global_cluster = _build_scene_global_two_way_blocker_cluster(
            frame_records,
            event_name=event_name,
        )

    def _best_borrow_geom_for_route(route_local):
        best_geom = None
        best_score = None
        best_priority = None
        for shift_sign in (-1.0, 1.0):
            strict_geom = _estimate_borrow_points_from_signed_route_shift(
                route_local=route_local,
                shift_sign=shift_sign,
                borrow_enter_lateral_thresh=float(borrow_enter_lateral_thresh),
                return_lateral_thresh=float(return_lateral_thresh),
                min_enter_progress_m=float(min_enter_progress_m),
                min_return_progress_m=float(min_return_progress_m),
                segment_step_m=float(max(route_step_m, 0.25)),
            )
            geom = strict_geom
            priority = 0
            if geom is None:
                continue
            score = (
                int(priority),
                -float(geom.get("peak_signed_lateral_m", 0.0)),
                float(geom.get("return_abs_m", np.inf)),
            )
            if best_score is None or score < best_score:
                best_geom = dict(geom)
                best_priority = int(priority)
                best_score = score
        if best_geom is None:
            return None, None
        return best_geom, int(best_priority)

    route_candidates = []
    for record_idx, record in enumerate(frame_records or []):
        current_meas = record.get("current_meas")
        if _measurement_command_id(current_meas) != LANE_FOLLOW_COMMAND_ID:
            continue
        ego_matrix_current = None if current_meas is None else current_meas.get("ego_matrix", None)
        if ego_matrix_current is None:
            continue
        route_local = np.asarray(record.get("route_input_local", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        borrow_geom, priority = _best_borrow_geom_for_route(route_local)
        if borrow_geom is None:
            continue
        route_candidates.append({
            "record_idx": int(record_idx),
            "frame_id": int(record.get("frame_id", -1)),
            "priority": int(priority),
            "geom": dict(borrow_geom),
        })
    if not route_candidates:
        return _empty_two_way_borrow_context(
            event_name=event_name,
            failure_reason="no_route_candidates",
            failure_stage="route_candidates",
            route_candidate_count=0,
        )

    primary_cluster_candidate = None
    if scene_global_cluster is not None:
        primary_cluster_candidate = _find_best_two_way_cluster_candidate(
            route_candidates,
            frame_records,
            scene_global_cluster.get("actor_ids", []),
            lateral_thresh_m=max(float(blocker_pre_shift_lateral_thresh), 5.0),
        )

    seed_actor = None
    if primary_cluster_candidate is not None:
        seed_actor = {
            "actor_id": int(primary_cluster_candidate["actor_id"]),
            "actor_class": str(primary_cluster_candidate["actor_class"]),
            "seed_frame_id": int(primary_cluster_candidate["frame_id"]),
            "seed_priority": int(primary_cluster_candidate.get("priority", -1)),
            "seed_world_xyz": list(primary_cluster_candidate.get("world_xyz", [])),
            "seed_source": "scene_global_cluster",
            "cluster_actor_ids": list(scene_global_cluster.get("actor_ids", [])),
        }
    else:
        for route_candidate in sorted(route_candidates, key=lambda item: (int(item["priority"]), int(item["frame_id"]))):
            record = frame_records[int(route_candidate["record_idx"])]
            eligible_boxes = []
            for box in record.get("current_boxes") or []:
                actor_id = box.get("id", None)
                if actor_id is None or not _two_way_blocker_box_allowed(box, event_name=event_name):
                    continue
                pos = box.get("position", None)
                if pos is None or len(pos) < 2:
                    continue
                local_x = float(pos[0])
                local_y = float(pos[1])
                if local_x <= 0.0 or float(abs(local_y)) > float(blocker_pre_shift_lateral_thresh):
                    continue
                eligible_boxes.append((float(local_x), float(abs(local_y)), int(actor_id), box))
            if not eligible_boxes:
                continue
            eligible_boxes.sort(key=lambda item: (item[0], item[1], item[2]))
            _, _, actor_id, seed_box = eligible_boxes[0]
            seed_world_xyz = _box_world_xyz(seed_box, ego_matrix_current=record.get("current_meas", {}).get("ego_matrix", None))
            seed_actor = {
                "actor_id": int(actor_id),
                "actor_class": str(_box_class_name(seed_box)),
                "seed_frame_id": int(route_candidate["frame_id"]),
                "seed_priority": int(route_candidate["priority"]),
                "seed_world_xyz": [] if seed_world_xyz is None else np.asarray(seed_world_xyz, dtype=np.float32).astype(float).tolist(),
                "seed_source": "route_candidate",
            }
            break
    if seed_actor is None:
        return _empty_two_way_borrow_context(
            event_name=event_name,
            failure_reason="no_blocker_seed",
            failure_stage="seed_actor",
            route_candidate_count=len(route_candidates),
        )

    best_strict = None
    best_fallback = None
    for route_candidate in route_candidates:
        record = frame_records[int(route_candidate["record_idx"])]
        ego_matrix_current = None if record.get("current_meas") is None else record["current_meas"].get("ego_matrix", None)
        for box in record.get("current_boxes") or []:
            actor_id = box.get("id", None)
            if actor_id is None or int(actor_id) != int(seed_actor["actor_id"]):
                continue
            pos = box.get("position", None)
            if pos is None or len(pos) < 2:
                continue
            local_x = float(pos[0])
            local_y = float(pos[1])
            if local_x <= 0.0 or float(abs(local_y)) > float(blocker_pre_shift_lateral_thresh):
                continue
            world_xyz = _box_world_xyz(box, ego_matrix_current=ego_matrix_current)
            candidate = {
                "score": (float(local_x), float(abs(local_y)), -int(route_candidate["frame_id"])),
                "record_idx": int(route_candidate["record_idx"]),
                "frame_id": int(route_candidate["frame_id"]),
                "actor_id": int(seed_actor["actor_id"]),
                "actor_class": str(seed_actor["actor_class"]),
                "local_x": float(local_x),
                "local_y": float(local_y),
                "world_xyz": [] if world_xyz is None else np.asarray(world_xyz, dtype=np.float32).astype(float).tolist(),
                "geom": dict(route_candidate["geom"]),
            }
            if int(route_candidate["priority"]) == 0:
                if best_strict is None or candidate["score"] < best_strict["score"]:
                    best_strict = candidate
            else:
                if best_fallback is None or candidate["score"] < best_fallback["score"]:
                    best_fallback = candidate

    best_candidate = primary_cluster_candidate if primary_cluster_candidate is not None else (best_strict if best_strict is not None else best_fallback)
    if best_candidate is None:
        return _empty_two_way_borrow_context(
            event_name=event_name,
            failure_reason="seed_actor_not_recovered",
            failure_stage="context_frame_search",
            seed_actor=seed_actor,
            route_candidate_count=len(route_candidates),
        )

    record = frame_records[int(best_candidate["record_idx"])]
    current_meas = record.get("current_meas")
    ego_matrix_current = None if current_meas is None else current_meas.get("ego_matrix", None)
    if ego_matrix_current is None:
        return _empty_two_way_borrow_context(
            event_name=event_name,
            failure_reason="missing_ego_matrix",
            failure_stage="context_frame_search",
            seed_actor=seed_actor,
            best_candidate=best_candidate,
            route_candidate_count=len(route_candidates),
        )

    borrow_geom = dict(best_candidate["geom"])
    route_local_context = np.asarray(record.get("route_input_local", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    route_poly_context = _route_with_origin(route_local_context)
    arc_context = _polyline_arclengths(route_poly_context)
    if (
        route_poly_context.ndim != 2 or route_poly_context.shape[0] < 2 or route_poly_context.shape[1] != 2 or
        arc_context.ndim != 1 or arc_context.shape[0] != route_poly_context.shape[0]
    ):
        return _empty_two_way_borrow_context(
            event_name=event_name,
            failure_reason="invalid_context_route",
            failure_stage="context_route",
            seed_actor=seed_actor,
            best_candidate=best_candidate,
            route_candidate_count=len(route_candidates),
        )
    route_head_idx = int(max(route_poly_context.shape[0] - route_local_context.shape[0], 0))
    obstacle_local = np.asarray([best_candidate["local_x"], best_candidate["local_y"]], dtype=np.float32)
    if obstacle_local.shape != (2,) or not np.all(np.isfinite(obstacle_local)):
        return _empty_two_way_borrow_context(
            event_name=event_name,
            failure_reason="invalid_blocker_local_xy",
            failure_stage="context_route",
            seed_actor=seed_actor,
            best_candidate=best_candidate,
            route_candidate_count=len(route_candidates),
        )
    if str(borrow_start_mode) == "obstacle_align":
        start_mask = np.abs(route_poly_context[:, 1]) <= float(TWOWAY_START_LATERAL_THRESH_M)
        if not np.any(start_mask):
            return _empty_two_way_borrow_context(
                event_name=event_name,
                failure_reason="no_start_mask",
                failure_stage="route_start",
                seed_actor=seed_actor,
                best_candidate=best_candidate,
                route_candidate_count=len(route_candidates),
            )
        candidate_idx = np.where(start_mask)[0]
        candidate_x = route_poly_context[candidate_idx, 0]
        start_idx = int(candidate_idx[np.argmin(np.abs(candidate_x - obstacle_local[0]))])
        route_start_s = float(arc_context[start_idx])
    else:
        # Route head means the first actual route point, not the prepended origin.
        route_start_s = float(arc_context[min(route_head_idx, arc_context.shape[0] - 1)])

    shift_summary = _summarize_signed_route_shift(
        route_local=route_local_context,
        shift_sign=float(borrow_geom.get("shift_sign", -1.0)),
    )
    route_return_s = np.nan
    if shift_summary is not None:
        rel_y = np.asarray(shift_summary.get("rel_y", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        peak_idx = int(shift_summary.get("peak_idx", route_head_idx))
        search_start_idx = int(np.clip(max(route_head_idx, peak_idx), 0, max(rel_y.shape[0] - 1, 0)))
        if rel_y.ndim == 1 and rel_y.shape[0] == route_poly_context.shape[0] and search_start_idx < rel_y.shape[0]:
            if float(borrow_geom.get("shift_sign", -1.0)) < 0.0:
                return_idx = int(search_start_idx + np.argmax(rel_y[search_start_idx:]))
            else:
                return_idx = int(search_start_idx + np.argmin(rel_y[search_start_idx:]))
            route_return_s = float(arc_context[min(return_idx, arc_context.shape[0] - 1)])
    if not np.isfinite(route_return_s) or route_return_s <= route_start_s + 1e-3:
        route_front_count = int(record.get("route_front_count", 0))
        fallback_local_idx = route_front_count + int(TWOWAY_RETURN_FALLBACK_EXTRA_POINT_INDEX) - 1
        if route_front_count <= 0 or fallback_local_idx >= route_local_context.shape[0]:
            return _empty_two_way_borrow_context(
                event_name=event_name,
                failure_reason="no_return_point",
                failure_stage="route_end",
                seed_actor=seed_actor,
                best_candidate=best_candidate,
                route_candidate_count=len(route_candidates),
            )
        fallback_poly_idx = int(route_head_idx + fallback_local_idx)
        if fallback_poly_idx < 0 or fallback_poly_idx >= arc_context.shape[0]:
            return _empty_two_way_borrow_context(
                event_name=event_name,
                failure_reason="invalid_return_point",
                failure_stage="route_end",
                seed_actor=seed_actor,
                best_candidate=best_candidate,
                route_candidate_count=len(route_candidates),
            )
        route_return_s = float(arc_context[fallback_poly_idx])
    route_return_s = float(np.clip(route_return_s, route_start_s, float(arc_context[-1])))

    query_s = np.arange(
        route_start_s,
        route_return_s + 1e-6,
        float(max(route_step_m, 0.25)),
        dtype=np.float32,
    )
    if query_s.size == 0 or float(query_s[-1]) < route_return_s - 1e-4:
        query_s = np.concatenate([query_s, np.array([route_return_s], dtype=np.float32)], axis=0)
    corridor_segment_local = _sample_polyline_at_arclengths(route_poly_context, query_s)
    if corridor_segment_local.ndim != 2 or corridor_segment_local.shape[0] < 2 or corridor_segment_local.shape[1] != 2:
        return _empty_two_way_borrow_context(
            event_name=event_name,
            failure_reason="empty_corridor_segment",
            failure_stage="corridor_segment",
            seed_actor=seed_actor,
            best_candidate=best_candidate,
            route_candidate_count=len(route_candidates),
        )

    borrow_geom["segment_local_xy"] = corridor_segment_local.astype(np.float32)
    borrow_geom["enter_local_xy"] = corridor_segment_local[0, :2].astype(np.float32)
    borrow_geom["enter_s_m"] = float(route_start_s)
    borrow_geom["return_s_m"] = float(route_return_s)
    borrow_geom["borrow_distance_m"] = float(max(route_return_s - route_start_s, 0.0))
    borrow_geom["start_from_route_head"] = False
    borrow_segment_world = _transform_points_local_to_world_xyz(
        np.asarray(borrow_geom.get("segment_local_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32),
        ego_matrix_current,
    )
    if borrow_segment_world.ndim != 2 or borrow_segment_world.shape[0] < 2 or borrow_segment_world.shape[1] < 3:
        return _empty_two_way_borrow_context(
            event_name=event_name,
            failure_reason="invalid_corridor_world",
            failure_stage="corridor_world",
            seed_actor=seed_actor,
            best_candidate=best_candidate,
            route_candidate_count=len(route_candidates),
        )
    borrow_world_xyz = np.stack([borrow_segment_world[0, :3], borrow_segment_world[-1, :3]], axis=0)

    return {
        "valid": 1.0,
        "ready": 1.0,
        "source": "event_blocker_route_{}".format(str(borrow_geom.get("mode", "strict"))),
        "failure_reason": "none",
        "failure_stage": "none",
        "event_name": str(event_name or ""),
        "route_candidate_count": int(len(route_candidates)),
        "release_frame_id": -1,
        "enter_frame_id": int(best_candidate["frame_id"]),
        "return_frame_id": int(best_candidate["frame_id"]),
        "borrow_duration_s": 0.0,
        "release_to_return_s": 0.0,
        "borrow_start_distance_m": float(route_start_s),
        "borrow_start_world_xyz": borrow_world_xyz[0, :3].astype(float).tolist(),
        "borrow_end_world_xyz": borrow_world_xyz[1, :3].astype(float).tolist(),
        "borrow_segment_world_xyz": borrow_segment_world[:, :3].astype(float).tolist(),
        "borrow_distance_m": float(borrow_geom["borrow_distance_m"]),
        "peak_lateral_m": float(borrow_geom["peak_lateral_m"]),
        "peak_signed_lateral_m": float(borrow_geom.get("peak_signed_lateral_m", np.nan)),
        "shift_sign": float(borrow_geom.get("shift_sign", np.nan)),
        "context_frame_id": int(best_candidate["frame_id"]),
        "anchor_actor_id": int(best_candidate["actor_id"]),
        "anchor_distance_m": float(best_candidate["local_x"]),
        "anchor_world_xyz": list(best_candidate["world_xyz"]),
        "blocking_actor_id": int(best_candidate["actor_id"]),
        "blocking_actor_class": str(best_candidate["actor_class"]),
        "blocking_actor_local_x_m": float(best_candidate["local_x"]),
        "blocking_actor_local_y_m": float(best_candidate["local_y"]),
        "seed_frame_id": int(seed_actor.get("seed_frame_id", -1)),
        "seed_priority": int(seed_actor.get("seed_priority", -1)),
        "seed_source": str(seed_actor.get("seed_source", "route_candidate")),
        "scene_global_cluster_actor_ids": list(scene_global_cluster.get("actor_ids", [])) if scene_global_cluster is not None else [],
        "scene_global_cluster_actor_count": int(scene_global_cluster.get("actor_count", 0)) if scene_global_cluster is not None else 0,
        "scene_global_cluster_total_obs": int(scene_global_cluster.get("total_obs", 0)) if scene_global_cluster is not None else 0,
        "route_return_abs_m": float(borrow_geom.get("return_abs_m", np.nan)),
        "blocked_frame_id": -1,
    }


def _build_scene_route_polyline_world(route_samples, image_root, dedupe_step_m=0.5, tail_window_points=8, overlap_thresh_m=1.5):
    by_base_dir = {}
    for sample in route_samples:
        base_dir, frame_str = _resolve_feature_frame_info(sample)
        if base_dir is None or frame_str is None:
            continue
        meas_path = os.path.join(image_root, base_dir, "measurements", f"{frame_str}.json.gz")
        meas = _load_json_gz_if_exists(meas_path)
        if meas is None:
            continue
        ego_matrix = meas.get("ego_matrix", None)
        route_raw = meas.get("route", None)
        if ego_matrix is None or route_raw is None:
            continue
        route_raw = np.asarray(route_raw, dtype=np.float32)
        if route_raw.ndim != 2 or route_raw.shape[0] == 0 or route_raw.shape[1] < 2:
            continue
        tail_start = max(0, int(route_raw.shape[0]) - int(max(tail_window_points, 1)))
        route_tail_world = _transform_points_local_to_world_xyz(route_raw[tail_start:, :2], ego_matrix)
        route_tail_world = _dedupe_polyline(route_tail_world, min_step_m=max(0.25, 0.5 * float(dedupe_step_m)))
        if route_tail_world.shape[0] == 0:
            continue
        by_base_dir.setdefault(base_dir, []).append((int(sample.get("frame_id", -1)), route_tail_world))

    scene_polyline_world = {}
    scene_polyline_anchor_s = {}
    for base_dir, frame_points in by_base_dir.items():
        frame_points.sort(key=lambda x: x[0])
        kept = []
        anchor_s = {}
        cumulative_s = 0.0
        prev_pt = None
        for frame_id, tail_slice in frame_points:
            tail_slice = np.asarray(tail_slice, dtype=np.float32)
            if tail_slice.ndim != 2 or tail_slice.shape[0] == 0:
                continue
            if prev_pt is None:
                kept = [pt.copy() for pt in tail_slice]
                prev_pt = np.asarray(kept[-1], dtype=np.float32)
                poly = np.asarray(kept, dtype=np.float32)
                cumulative_s = _polyline_length_m(poly)
                _, tail_s = _project_point_to_polyline(tail_slice[-1, :2], poly[:, :2])
                anchor_s[int(frame_id)] = 0.0 if tail_s is None else float(tail_s)
                continue
            dists_to_last = np.linalg.norm(tail_slice[:, :2] - prev_pt[None, :2], axis=1)
            overlap_idx = int(np.argmin(dists_to_last))
            append_slice = tail_slice[overlap_idx + 1:] if float(dists_to_last[overlap_idx]) <= float(overlap_thresh_m) else tail_slice
            for pt in append_slice:
                pt = np.asarray(pt, dtype=np.float32)
                step = float(np.linalg.norm(pt[:2] - prev_pt[:2]))
                if step < float(dedupe_step_m):
                    continue
                cumulative_s += step
                kept.append(pt)
                prev_pt = pt
            poly = np.asarray(kept, dtype=np.float32)
            _, tail_s = _project_point_to_polyline(tail_slice[-1, :2], poly[:, :2])
            anchor_s[int(frame_id)] = float(cumulative_s if tail_s is None else tail_s)
        if kept:
            scene_polyline_world[base_dir] = np.asarray(kept, dtype=np.float32)
            scene_polyline_anchor_s[base_dir] = anchor_s
    return scene_polyline_world, scene_polyline_anchor_s


def _extend_local_route_with_scene_polyline(route_local, ego_matrix_current, scene_polyline_world, anchor_s=None, extension_step_m=1.0, extension_points=12):
    route_local = np.asarray(route_local, dtype=np.float32)
    scene_polyline_world = np.asarray(scene_polyline_world, dtype=np.float32)
    if (
        route_local.ndim != 2 or route_local.shape[0] == 0 or route_local.shape[1] != 2 or
        ego_matrix_current is None or scene_polyline_world.ndim != 2 or scene_polyline_world.shape[0] < 2
    ):
        return route_local
    if anchor_s is None:
        tail_world = _transform_points_local_to_world_xyz(route_local[-1:, :2], ego_matrix_current)
        if tail_world.shape[0] == 0:
            return route_local
        _, tail_s = _project_point_to_polyline(tail_world[0, :2], scene_polyline_world[:, :2])
    else:
        tail_s = float(anchor_s)
    if tail_s is None:
        return route_local
    query_s = tail_s + float(extension_step_m) * np.arange(1, int(extension_points) + 1, dtype=np.float32)
    extension_world = _sample_polyline_at_arclengths(scene_polyline_world, query_s)
    if extension_world.shape[0] == 0:
        return route_local
    extension_local = _transform_points_world_xyz_to_local(extension_world, ego_matrix_current)
    if extension_local.shape[0] == 0:
        return route_local
    merged = np.concatenate([route_local[:, :2], extension_local], axis=0)
    return _dedupe_polyline(merged, min_step_m=max(0.25, 0.5 * float(extension_step_m)))




def _to_stage1_debug_python(value):
    if isinstance(value, dict):
        return {str(k): _to_stage1_debug_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_stage1_debug_python(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _build_stage1_speed_debug_payload(
    current_cover,
    future_cover,
    merge_motion=None,
    scene_borrow_context=None,
    borrow_motion=None,
):
    return _to_stage1_debug_python({
        "current_cover": dict(current_cover),
        "future_cover": dict(future_cover),
        "merge_motion": dict(merge_motion or {}),
        "scene_borrow_context": None if scene_borrow_context is None else dict(scene_borrow_context),
        "borrow_motion": dict(borrow_motion or {}),
        "conflict_area": _default_conflict_area_debug(),
        "conflict_phase": _default_conflict_phase_debug(),
    })


def _default_conflict_area_debug():
    return {
        'family': 'none',
        'family_code': int(CONFLICT_FAMILY_TO_CODE['none']),
        'dir': 'none',
        'dir_code': int(CONFLICT_DIR_TO_CODE['none']),
        'active': 0.0,
        'start_frame': -1,
        'end_frame': -1,
        'frame_role': 'none',
        'source': 'none',
        'source_episode_id': -1,
        'source_priority': -1,
        'selection_reason': 'none',
        'active_families': [],
        'active_family_count': 0,
        'issue_count': 0,
        'issue_families': [],
        'missing_reason': 'none',
        'area_type': 'none',
        'area_start_s_m': np.nan,
        'area_end_s_m': np.nan,
        'area_start_world_xyz': [],
        'area_end_world_xyz': [],
        'area_segment_world_xyz': [],
        'area_center_world_xyz': [],
        'area_radius_m': np.nan,
        'borrow_start_world_xyz': [],
        'borrow_end_world_xyz': [],
        'borrow_distance_m': np.nan,
        'dir_source': 'none',
        'dir_angle_deg': np.nan,
        'route_heading_deg': np.nan,
        'actor_heading_deg': np.nan,
        'collision_point_world_xyz': [],
        'dir_cover_key': 'none',
        'dir_frame_id': -1,
    }


def _default_conflict_phase_debug():
    return {
        'phase': 'none',
        'phase_code': int(CONFLICT_DECISION_PHASE_TO_CODE['none']),
        'active': 0.0,
        'family': 'none',
        'start_frame': -1,
        'end_frame': -1,
        'entry_frame': -1,
        'go_frame': -1,
        'release_frame': -1,
        'frame_role': 'none',
        'source': 'none',
        'release_reason': 'none',
        'entry_found': 0.0,
        'speed_mps': np.nan,
        'stop_speed_thresh_mps': float(STAGE1_CONFLICT_GO_STOP_SPEED_THRESH_MPS),
    }


def _set_stage1_conflict_area_defaults(sample):
    sample['conflict_area_family'] = np.int64(CONFLICT_FAMILY_TO_CODE['none'])
    sample['conflict_area_dir'] = np.int64(CONFLICT_DIR_TO_CODE['none'])
    sample['conflict_area_active'] = np.float32(0.0)
    sample['conflict_area_start_frame'] = np.int64(-1)
    sample['conflict_area_end_frame'] = np.int64(-1)
    stage1_debug = sample.get('stage1_speed_debug')
    if isinstance(stage1_debug, dict):
        stage1_debug['conflict_area'] = _default_conflict_area_debug()


def _set_stage1_conflict_area_annotation(sample, conflict_info):
    family = str(conflict_info.get('family', 'none'))
    direction = str(conflict_info.get('dir', 'none'))
    sample['conflict_area_family'] = np.int64(CONFLICT_FAMILY_TO_CODE.get(family, 0))
    sample['conflict_area_dir'] = np.int64(CONFLICT_DIR_TO_CODE.get(direction, 0))
    sample['conflict_area_active'] = np.float32(float(conflict_info.get('active', 0.0)))
    sample['conflict_area_start_frame'] = np.int64(int(conflict_info.get('start_frame', -1)))
    sample['conflict_area_end_frame'] = np.int64(int(conflict_info.get('end_frame', -1)))
    stage1_debug = sample.get('stage1_speed_debug')
    if isinstance(stage1_debug, dict):
        stage1_debug['conflict_area'] = _to_stage1_debug_python(dict(conflict_info))


def _set_stage1_conflict_phase_defaults(sample):
    sample['conflict_decision_phase'] = np.int64(CONFLICT_DECISION_PHASE_TO_CODE['none'])
    sample['conflict_go_frame'] = np.int64(-1)
    stage1_debug = sample.get('stage1_speed_debug')
    if isinstance(stage1_debug, dict):
        stage1_debug['conflict_phase'] = _default_conflict_phase_debug()


def _set_stage1_conflict_phase_annotation(sample, phase_info):
    phase = str(phase_info.get('phase', 'none'))
    sample['conflict_decision_phase'] = np.int64(CONFLICT_DECISION_PHASE_TO_CODE.get(phase, 0))
    sample['conflict_go_frame'] = np.int64(int(phase_info.get('go_frame', -1)))
    stage1_debug = sample.get('stage1_speed_debug')
    if isinstance(stage1_debug, dict):
        stage1_debug['conflict_phase'] = _to_stage1_debug_python(dict(phase_info))


def _conflict_area_issue(sample, family, reason, **extra):
    payload = {
        'issue': 1,
        'family': str(family),
        'reason': str(reason),
        'base_dir': str(sample.get('base_dir', 'unknown')),
        'route_name': str(sample.get('route_name', 'unknown')),
        'frame_id': int(sample.get('frame_id', -1)),
    }
    payload.update(extra)
    return payload


def _cover_is_borrow_cross_meet(cover):
    return int((cover or {}).get('exists', 0.0)) > 0 and _cover_interaction_subtype(cover) == 'borrow_cross_meet'


def _route_heading_at_progress(route_local, progress_m):
    route_poly = _route_with_origin(route_local)
    if route_poly is None or route_poly.shape[0] < 2:
        return np.nan
    dense_route, dense_s = _interpolate_route_with_arclength(route_poly, step_m=0.25)
    if dense_route.ndim != 2 or dense_route.shape[0] < 2 or dense_s.ndim != 1 or dense_s.shape[0] != dense_route.shape[0]:
        return np.nan
    progress_m = float(progress_m)
    total_s = float(dense_s[-1]) if dense_s.size > 0 else 0.0
    if not np.isfinite(progress_m):
        return np.nan
    progress_m = float(np.clip(progress_m, 0.0, total_s))
    idx = int(np.searchsorted(dense_s, progress_m, side='left'))
    idx = int(np.clip(idx, 0, dense_route.shape[0] - 1))
    heading = _route_heading_at_idx(dense_route, idx)
    return float(heading) if heading is not None else np.nan


def _cover_actor_heading_rad(cover):
    interaction = ((cover or {}).get('interaction') or {})
    try:
        actor_heading_deg = float(interaction.get('actor_heading_deg', np.nan))
    except Exception:
        actor_heading_deg = np.nan
    if not np.isfinite(actor_heading_deg):
        return np.nan
    return float(np.radians(actor_heading_deg))


def _cover_conflict_world_xyz(cover):
    xyz = np.asarray((cover or {}).get('scene_route_conflict_world_xyz', []), dtype=np.float32).reshape(-1)
    if xyz.size >= 3 and np.all(np.isfinite(xyz[:3])):
        return xyz[:3].astype(float).tolist()
    return []


def _conflict_dir_from_headings(route_heading_rad, actor_heading_rad):
    if not np.isfinite(route_heading_rad) or not np.isfinite(actor_heading_rad):
        return 'none', np.nan
    angle_deg = abs(_heading_to_deg(float(actor_heading_rad) - float(route_heading_rad)))
    if angle_deg <= float(CONFLICT_DIR_SAME_MAX_ANGLE_DEG):
        return 'same', float(angle_deg)
    if angle_deg >= float(CONFLICT_DIR_OPPOSITE_MIN_ANGLE_DEG):
        return 'opposite', float(angle_deg)
    return 'cross', float(angle_deg)


def _record_family_cover_candidates(
    record,
    family,
    area_start_s_m=np.nan,
    area_end_s_m=np.nan,
    borrow_conflict_start_progress_m=np.nan,
    borrow_conflict_end_progress_m=np.nan,
):
    family = str(family or 'none')
    candidates = []
    for cover_key in ('current_cover', 'future_cover'):
        cover = (record or {}).get(cover_key) or {}
        if int(cover.get('exists', 0.0)) <= 0:
            continue
        actor_heading_rad = _cover_actor_heading_rad(cover)
        if not np.isfinite(actor_heading_rad):
            continue
        keep = False
        progress_value = np.nan
        if family == 'borrow':
            if _cover_is_borrow_cross_meet(cover):
                borrow_motion = (record or {}).get('borrow_motion') or {}
                borrow_start_distance_m = float(borrow_motion.get('borrow_start_distance_m', np.nan))
                route_distance_m = float(cover.get('route_distance_m', np.nan))
                if np.isfinite(borrow_start_distance_m) and np.isfinite(route_distance_m):
                    progress_value = float(route_distance_m - borrow_start_distance_m)
                    if (
                        not np.isfinite(borrow_conflict_start_progress_m) or
                        not np.isfinite(borrow_conflict_end_progress_m) or
                        (
                            progress_value >= float(borrow_conflict_start_progress_m) - 0.5 and
                            progress_value <= float(borrow_conflict_end_progress_m) + 0.5
                        )
                    ):
                        keep = True
        elif family == 'merge':
            if _cover_is_merge_meet(cover) or _cover_is_chase(cover):
                progress_value = float(cover.get('scene_route_conflict_s_m', np.nan))
                if (
                    not np.isfinite(area_start_s_m) or
                    not np.isfinite(area_end_s_m) or
                    (
                        np.isfinite(progress_value) and
                        progress_value >= float(area_start_s_m) - 2.0 and
                        progress_value <= float(area_end_s_m) + 2.0
                    )
                ):
                    keep = True
        elif family == 'junction':
            conflict_xyz = _cover_conflict_world_xyz(cover)
            progress_value = float(cover.get('scene_route_conflict_s_m', np.nan))
            if conflict_xyz:
                if (
                    not np.isfinite(area_start_s_m) or
                    not np.isfinite(area_end_s_m) or
                    not np.isfinite(progress_value) or
                    (
                        progress_value >= float(area_start_s_m) - 2.0 and
                        progress_value <= float(area_end_s_m) + 2.0
                    )
                ):
                    keep = True
        if not keep:
            continue
        candidates.append({
            'cover_key': str(cover_key),
            'cover': dict(cover),
            'actor_heading_rad': float(actor_heading_rad),
            'progress_value': float(progress_value) if np.isfinite(progress_value) else np.nan,
        })
    return candidates


def _select_conflict_dir_cover_candidate(
    records,
    start_pos,
    end_pos,
    family,
    area_start_s_m=np.nan,
    area_end_s_m=np.nan,
    borrow_conflict_start_progress_m=np.nan,
    borrow_conflict_end_progress_m=np.nan,
):
    family = str(family or 'none')
    if family == 'borrow' and np.isfinite(borrow_conflict_start_progress_m) and np.isfinite(borrow_conflict_end_progress_m):
        target_progress = 0.5 * (float(borrow_conflict_start_progress_m) + float(borrow_conflict_end_progress_m))
    elif np.isfinite(area_start_s_m) and np.isfinite(area_end_s_m):
        target_progress = 0.5 * (float(area_start_s_m) + float(area_end_s_m))
    else:
        target_progress = np.nan

    best = None
    for pos in range(int(start_pos), int(end_pos) + 1):
        record = records[int(pos)]
        candidates = _record_family_cover_candidates(
            record,
            family=family,
            area_start_s_m=area_start_s_m,
            area_end_s_m=area_end_s_m,
            borrow_conflict_start_progress_m=borrow_conflict_start_progress_m,
            borrow_conflict_end_progress_m=borrow_conflict_end_progress_m,
        )
        for item in candidates:
            progress_value = float(item.get('progress_value', np.nan))
            dist_to_target = abs(progress_value - float(target_progress)) if np.isfinite(progress_value) and np.isfinite(target_progress) else np.inf
            cover_key = str(item.get('cover_key', 'future_cover'))
            cover_rank = 0 if cover_key == 'current_cover' else 1
            score = (float(dist_to_target), int(cover_rank), int(pos))
            payload = dict(item)
            payload['pos'] = int(pos)
            payload['score'] = score
            if best is None or score < best['score']:
                best = payload
    return best


def _borrow_route_heading_rad_from_conflict_area(record, conflict_start_progress_m, conflict_end_progress_m):
    borrow_motion = (record or {}).get('borrow_motion') or {}
    borrow_start_distance_m = float(borrow_motion.get('borrow_start_distance_m', np.nan))
    route_local = np.asarray((record or {}).get('route_local', np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    if not np.isfinite(borrow_start_distance_m):
        return np.nan
    if not np.isfinite(conflict_start_progress_m) or not np.isfinite(conflict_end_progress_m):
        return np.nan
    local_start_s = float(borrow_start_distance_m) + float(conflict_start_progress_m)
    local_end_s = float(borrow_start_distance_m) + float(conflict_end_progress_m)
    heading = _route_heading_at_progress(route_local, 0.5 * (local_start_s + local_end_s))
    return float(heading) if np.isfinite(heading) else np.nan


def _junction_route_heading_rad_from_conflict_area(record, area_start_s_m, area_end_s_m):
    front_s = _record_scene_front_s(record)
    route_local = np.asarray((record or {}).get('route_local', np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    if not np.isfinite(front_s) or not np.isfinite(area_start_s_m) or not np.isfinite(area_end_s_m):
        return np.nan
    local_mid_s = 0.5 * (float(area_start_s_m) + float(area_end_s_m)) - float(front_s)
    heading = _route_heading_at_progress(route_local, local_mid_s)
    return float(heading) if np.isfinite(heading) else np.nan


def _merge_route_heading_rad_from_conflict_area(record, merge_area_end_s_m):
    front_s = _record_scene_front_s(record)
    route_local = np.asarray((record or {}).get('route_local', np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    if not np.isfinite(front_s):
        return np.nan
    distance_to_area_end_m = max(float(merge_area_end_s_m) - float(front_s), 0.0)
    heading = _route_heading_at_progress(route_local, distance_to_area_end_m + 5.0)
    if np.isfinite(heading):
        return float(heading)
    return float(_route_heading_at_progress(route_local, distance_to_area_end_m + 3.0))


def _borrow_conflict_dir_info(records, start_pos, end_pos, conflict_start_progress_m, conflict_end_progress_m):
    selected = _select_conflict_dir_cover_candidate(
        records,
        start_pos=start_pos,
        end_pos=end_pos,
        family='borrow',
        borrow_conflict_start_progress_m=conflict_start_progress_m,
        borrow_conflict_end_progress_m=conflict_end_progress_m,
    )
    if selected is None:
        return {
            'dir': 'opposite',
            'dir_code': int(CONFLICT_DIR_TO_CODE['opposite']),
            'dir_source': 'family_fallback',
            'dir_angle_deg': np.nan,
            'route_heading_deg': np.nan,
            'actor_heading_deg': np.nan,
            'collision_point_world_xyz': [],
            'dir_cover_key': 'none',
            'dir_frame_id': -1,
            'topology_override': 'none',
        }
    record = records[int(selected['pos'])]
    route_heading_rad = _borrow_route_heading_rad_from_conflict_area(
        record,
        conflict_start_progress_m=conflict_start_progress_m,
        conflict_end_progress_m=conflict_end_progress_m,
    )
    actor_heading_rad = float(selected['actor_heading_rad'])
    direction, angle_deg = _conflict_dir_from_headings(route_heading_rad, actor_heading_rad)
    if direction == 'none':
        direction = 'opposite'
    return {
        'dir': str(direction),
        'dir_code': int(CONFLICT_DIR_TO_CODE.get(direction, CONFLICT_DIR_TO_CODE['opposite'])),
        'dir_source': 'borrow_cover_vs_conflict_area',
        'dir_angle_deg': float(angle_deg) if np.isfinite(angle_deg) else np.nan,
        'route_heading_deg': _heading_to_deg(route_heading_rad) if np.isfinite(route_heading_rad) else np.nan,
        'actor_heading_deg': _heading_to_deg(actor_heading_rad) if np.isfinite(actor_heading_rad) else np.nan,
        'collision_point_world_xyz': _cover_conflict_world_xyz(selected.get('cover') or {}),
        'dir_cover_key': str(selected.get('cover_key', 'none')),
        'dir_frame_id': int(records[int(selected['pos'])].get('frame_id', -1)),
    }


def _merge_conflict_dir_info(records, start_pos, end_pos, merge_area_start_s_m, merge_area_end_s_m):
    selected = _select_conflict_dir_cover_candidate(
        records,
        start_pos=start_pos,
        end_pos=end_pos,
        family='merge',
        area_start_s_m=merge_area_start_s_m,
        area_end_s_m=merge_area_end_s_m,
    )
    if selected is None:
        return {
            'dir': 'same',
            'dir_code': int(CONFLICT_DIR_TO_CODE['same']),
            'dir_source': 'family_fallback',
            'dir_angle_deg': np.nan,
            'route_heading_deg': np.nan,
            'actor_heading_deg': np.nan,
            'collision_point_world_xyz': [],
            'dir_cover_key': 'none',
            'dir_frame_id': -1,
        }
    record = records[int(selected['pos'])]
    route_heading_rad = _merge_route_heading_rad_from_conflict_area(record, merge_area_end_s_m)
    actor_heading_rad = float(selected['actor_heading_rad'])
    direction, angle_deg = _conflict_dir_from_headings(route_heading_rad, actor_heading_rad)
    if direction == 'none':
        direction = 'same'
    return {
        'dir': str(direction),
        'dir_code': int(CONFLICT_DIR_TO_CODE.get(direction, CONFLICT_DIR_TO_CODE['same'])),
        'dir_source': 'merge_cover_vs_downstream_route',
        'dir_angle_deg': float(angle_deg) if np.isfinite(angle_deg) else np.nan,
        'route_heading_deg': _heading_to_deg(route_heading_rad) if np.isfinite(route_heading_rad) else np.nan,
        'actor_heading_deg': _heading_to_deg(actor_heading_rad) if np.isfinite(actor_heading_rad) else np.nan,
        'collision_point_world_xyz': _cover_conflict_world_xyz(selected.get('cover') or {}),
        'dir_cover_key': str(selected.get('cover_key', 'none')),
        'dir_frame_id': int(records[int(selected['pos'])].get('frame_id', -1)),
    }


def _junction_conflict_dir_info(records, start_pos, end_pos, area_start_s_m, area_end_s_m):
    selected = _select_conflict_dir_cover_candidate(
        records,
        start_pos=start_pos,
        end_pos=end_pos,
        family='junction',
        area_start_s_m=area_start_s_m,
        area_end_s_m=area_end_s_m,
    )
    if selected is None:
        return {
            'dir': 'none',
            'dir_code': int(CONFLICT_DIR_TO_CODE['none']),
            'dir_source': 'family_fallback',
            'dir_angle_deg': np.nan,
            'route_heading_deg': np.nan,
            'actor_heading_deg': np.nan,
            'collision_point_world_xyz': [],
            'dir_cover_key': 'none',
            'dir_frame_id': -1,
        }
    record = records[int(selected['pos'])]
    route_heading_rad = _junction_route_heading_rad_from_conflict_area(record, area_start_s_m, area_end_s_m)
    actor_heading_rad = float(selected['actor_heading_rad'])
    direction, angle_deg = _conflict_dir_from_headings(route_heading_rad, actor_heading_rad)
    return {
        'dir': str(direction),
        'dir_code': int(CONFLICT_DIR_TO_CODE.get(direction, CONFLICT_DIR_TO_CODE['none'])),
        'dir_source': 'junction_cover_vs_conflict_area',
        'dir_angle_deg': float(angle_deg) if np.isfinite(angle_deg) else np.nan,
        'route_heading_deg': _heading_to_deg(route_heading_rad) if np.isfinite(route_heading_rad) else np.nan,
        'actor_heading_deg': _heading_to_deg(actor_heading_rad) if np.isfinite(actor_heading_rad) else np.nan,
        'collision_point_world_xyz': _cover_conflict_world_xyz(selected.get('cover') or {}),
        'dir_cover_key': str(selected.get('cover_key', 'none')),
        'dir_frame_id': int(records[int(selected['pos'])].get('frame_id', -1)),
    }


def _build_route_conflict_records(samples, route_sample_indices, scene_route_world=None):
    ordered_indices = sorted(route_sample_indices, key=lambda i: int(samples[i].get('frame_id', -1)))
    records = []
    scene_route_world = np.asarray(scene_route_world, dtype=np.float32)
    for sample_idx in ordered_indices:
        sample = samples[int(sample_idx)]
        stage1_debug = sample.get('stage1_speed_debug') or {}
        base_dir, _ = _resolve_feature_frame_info(sample)
        event_name = str(base_dir).split('/', 1)[0] if isinstance(base_dir, str) and base_dir else ''
        route_local = sample.get('_stage1_route_input_local', sample.get('route', np.zeros((0, 2), dtype=np.float32)))
        records.append({
            'sample_idx': int(sample_idx),
            'frame_id': int(sample.get('frame_id', -1)),
            'base_dir': str(base_dir or ''),
            'event_name': str(event_name),
            'route_local': np.asarray(route_local, dtype=np.float32),
            'scene_route_world': scene_route_world.copy() if scene_route_world.ndim == 2 else np.zeros((0, 3), dtype=np.float32),
            'current_cover': stage1_debug.get('current_cover') or {},
            'future_cover': stage1_debug.get('future_cover') or {},
            'merge_motion': stage1_debug.get('merge_motion') or {},
            'borrow_motion': stage1_debug.get('borrow_motion') or {},
            'scene_borrow_context': stage1_debug.get('scene_borrow_context') or {},
            'stage1_speed_debug': stage1_debug if isinstance(stage1_debug, dict) else {},
        })
    return records


def _record_scene_front_s(record):
    merge_motion = (record or {}).get('merge_motion') or {}
    front_s = float(merge_motion.get('scene_route_front_s_m', np.nan))
    return float(front_s) if np.isfinite(front_s) else np.nan


def _record_scene_rear_s(record):
    merge_motion = (record or {}).get('merge_motion') or {}
    rear_s = float(merge_motion.get('scene_route_rear_s_m', np.nan))
    return float(rear_s) if np.isfinite(rear_s) else np.nan


def _record_ego_matrix(record):
    merge_motion = (record or {}).get('merge_motion') or {}
    ego_matrix = np.asarray(merge_motion.get('ego_matrix', []), dtype=np.float32)
    if ego_matrix.shape == (4, 4) and np.all(np.isfinite(ego_matrix)):
        return ego_matrix
    return None


def _record_s_interval_world_geometry(record, area_start_s_m, area_end_s_m, step_m=0.5):
    route_local = np.asarray((record or {}).get('route_local', np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    route_poly = _route_with_origin(route_local)
    ego_matrix = _record_ego_matrix(record)
    front_s = _record_scene_front_s(record)
    if route_poly.ndim != 2 or route_poly.shape[0] < 2 or ego_matrix is None or not np.isfinite(front_s):
        return None
    local_start_s = max(float(area_start_s_m) - float(front_s), 0.0)
    local_end_s = max(float(area_end_s_m) - float(front_s), local_start_s)
    query_s = np.arange(local_start_s, local_end_s + 1e-6, float(max(step_m, 0.25)), dtype=np.float32)
    if query_s.size == 0 or float(query_s[-1]) < local_end_s - 1e-4:
        query_s = np.concatenate([query_s, np.array([local_end_s], dtype=np.float32)], axis=0)
    local_segment = _sample_polyline_at_arclengths(route_poly, query_s)
    if local_segment.ndim != 2 or local_segment.shape[0] == 0 or local_segment.shape[1] < 2:
        return None
    world_segment = _transform_points_local_to_world_xyz(local_segment[:, :2], ego_matrix)
    if world_segment.ndim != 2 or world_segment.shape[0] == 0 or world_segment.shape[1] < 3:
        return None
    return {
        'area_start_world_xyz': world_segment[0, :3].astype(float).tolist(),
        'area_end_world_xyz': world_segment[-1, :3].astype(float).tolist(),
        'area_segment_world_xyz': world_segment[:, :3].astype(float).tolist(),
    }


def _record_scene_s_interval_world_geometry(record, area_start_s_m, area_end_s_m, step_m=0.5):
    scene_route_world = np.asarray((record or {}).get('scene_route_world', np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    if scene_route_world.ndim != 2 or scene_route_world.shape[0] < 2 or scene_route_world.shape[1] < 3:
        return None
    query_s = np.arange(
        float(area_start_s_m),
        float(area_end_s_m) + 1e-6,
        float(max(step_m, 0.25)),
        dtype=np.float32,
    )
    if query_s.size == 0 or float(query_s[-1]) < float(area_end_s_m) - 1e-4:
        query_s = np.concatenate([query_s, np.array([float(area_end_s_m)], dtype=np.float32)], axis=0)
    world_segment = _sample_polyline_xyz_at_arclengths(scene_route_world[:, :3], query_s)
    if world_segment.ndim != 2 or world_segment.shape[0] == 0 or world_segment.shape[1] < 3:
        return None
    return {
        'area_start_world_xyz': world_segment[0, :3].astype(float).tolist(),
        'area_end_world_xyz': world_segment[-1, :3].astype(float).tolist(),
        'area_segment_world_xyz': world_segment[:, :3].astype(float).tolist(),
    }


def _window_s_interval_world_geometry(records, start_pos, end_pos, area_start_s_m, area_end_s_m, step_m=0.5):
    if not records:
        return None
    start_pos = int(max(start_pos, 0))
    end_pos = int(min(end_pos, len(records) - 1))
    for pos in range(start_pos, end_pos + 1):
        world_geometry = _record_scene_s_interval_world_geometry(
            records[pos],
            area_start_s_m=area_start_s_m,
            area_end_s_m=area_end_s_m,
            step_m=step_m,
        )
        if world_geometry is not None:
            return world_geometry
    for pos in range(start_pos, end_pos + 1):
        world_geometry = _record_s_interval_world_geometry(
            records[pos],
            area_start_s_m=area_start_s_m,
            area_end_s_m=area_end_s_m,
            step_m=step_m,
        )
        if world_geometry is not None:
            return world_geometry
    return None


def _borrow_conflict_world_geometry(scene_borrow_context, conflict_start_progress_m, conflict_end_progress_m, step_m=0.5):
    corridor_world = np.asarray((scene_borrow_context or {}).get('borrow_segment_world_xyz', []), dtype=np.float32)
    if corridor_world.ndim != 2 or corridor_world.shape[0] < 2 or corridor_world.shape[1] < 3:
        return None
    query_s = np.arange(
        float(conflict_start_progress_m),
        float(conflict_end_progress_m) + 1e-6,
        float(max(step_m, 0.25)),
        dtype=np.float32,
    )
    if query_s.size == 0 or float(query_s[-1]) < float(conflict_end_progress_m) - 1e-4:
        query_s = np.concatenate([query_s, np.array([float(conflict_end_progress_m)], dtype=np.float32)], axis=0)
    world_segment = _sample_polyline_xyz_at_arclengths(corridor_world[:, :3], query_s)
    if world_segment.ndim != 2 or world_segment.shape[0] == 0 or world_segment.shape[1] < 3:
        return None
    return {
        'area_start_world_xyz': world_segment[0, :3].astype(float).tolist(),
        'area_end_world_xyz': world_segment[-1, :3].astype(float).tolist(),
        'area_segment_world_xyz': world_segment[:, :3].astype(float).tolist(),
    }


def _borrow_record_conflict_progresses(record):
    scene_borrow_context = (record or {}).get('scene_borrow_context') or {}
    borrow_motion = (record or {}).get('borrow_motion') or {}
    if float(scene_borrow_context.get('valid', 0.0)) <= 0.5 or float(scene_borrow_context.get('ready', 0.0)) <= 0.5:
        return []
    borrow_distance_m = float(scene_borrow_context.get('borrow_distance_m', np.nan))
    borrow_start_distance_m = float(borrow_motion.get('borrow_start_distance_m', np.nan))
    if not np.isfinite(borrow_distance_m) or borrow_distance_m <= 1e-3 or not np.isfinite(borrow_start_distance_m):
        return []
    progresses = []
    for cover_key in ('current_cover', 'future_cover'):
        cover = (record or {}).get(cover_key) or {}
        if not _cover_is_borrow_cross_meet(cover):
            continue
        route_distance_m = float(cover.get('route_distance_m', np.nan))
        if not np.isfinite(route_distance_m):
            continue
        progress_m = float(route_distance_m - borrow_start_distance_m)
        if progress_m < -0.5 or progress_m > float(borrow_distance_m) + 0.5:
            continue
        progresses.append(float(np.clip(progress_m, 0.0, float(borrow_distance_m))))
    return progresses


def _borrow_conflict_end_distance_m(record, conflict_end_progress_m):
    scene_borrow_context = (record or {}).get('scene_borrow_context') or {}
    borrow_motion = (record or {}).get('borrow_motion') or {}
    borrow_distance_m = float(scene_borrow_context.get('borrow_distance_m', np.nan))
    borrow_end_distance_m = float(borrow_motion.get('borrow_end_distance_m', np.nan))
    if not np.isfinite(borrow_distance_m) or not np.isfinite(borrow_end_distance_m):
        return np.nan
    tail_after_conflict_m = max(float(borrow_distance_m) - float(conflict_end_progress_m), 0.0)
    return float(borrow_end_distance_m - tail_after_conflict_m)


def _build_borrow_conflict_windows(records, samples):
    windows = []
    issues = []
    if not records:
        return windows, issues
    event_name = str(records[0].get('event_name', ''))
    if event_name not in {"ConstructionObstacleTwoWays", "AccidentTwoWays"}:
        return windows, issues
    route_name = str(samples[records[0]['sample_idx']].get('route_name', 'unknown'))
    if (
        event_name == "ConstructionObstacleTwoWays" and
        route_name in TWOWAY_BORROW_ONEWAY_ROUTE_OVERRIDES
    ):
        issues.append(
            _conflict_area_issue(
                samples[records[0]['sample_idx']],
                'borrow',
                'construction_oneway_topology_override',
                pos=0,
                topology_override='oneway',
            )
        )
        return windows, issues

    scene_borrow_context = None
    for record in records:
        ctx = record.get('scene_borrow_context') or {}
        if float(ctx.get('valid', 0.0)) > 0.5 and float(ctx.get('ready', 0.0)) > 0.5:
            scene_borrow_context = ctx
            break
    if scene_borrow_context is None:
        issues.append(_conflict_area_issue(samples[records[0]['sample_idx']], 'borrow', 'missing_scene_borrow_context', pos=0))
        return windows, issues

    progress_by_pos = {}
    all_progress = []
    for pos, record in enumerate(records):
        progresses = _borrow_record_conflict_progresses(record)
        if not progresses:
            continue
        progress_by_pos[int(pos)] = [float(v) for v in progresses]
        all_progress.extend(progresses)
    if not all_progress:
        issues.append(_conflict_area_issue(samples[records[0]['sample_idx']], 'borrow', 'missing_borrow_conflict_progress_span', pos=0))
        return windows, issues

    conflict_start_progress_m = float(min(all_progress))
    conflict_end_progress_m = float(max(all_progress))
    candidate_positions = sorted(progress_by_pos.keys())
    front_candidates = []
    for pos in candidate_positions:
        front_s = _record_scene_front_s(records[pos])
        if np.isfinite(front_s):
            front_candidates.append((float(front_s), int(pos)))
    start_pos = min(front_candidates, key=lambda item: (float(item[0]), int(item[1])))[1] if front_candidates else int(candidate_positions[0])

    end_pos = None
    for pos in range(int(start_pos), len(records)):
        dist_to_conflict_end_m = _borrow_conflict_end_distance_m(records[pos], conflict_end_progress_m)
        if np.isfinite(dist_to_conflict_end_m) and dist_to_conflict_end_m <= 0.5:
            end_pos = int(pos)
            break
    if end_pos is None:
        issues.append(_conflict_area_issue(samples[records[start_pos]['sample_idx']], 'borrow', 'missing_borrow_conflict_end', pos=int(start_pos)))
        return windows, issues

    shape_issue = _record_window_cover_shape_issue(
        samples,
        records,
        family='borrow',
        positions=range(int(start_pos), int(end_pos) + 1),
    )
    if shape_issue is not None:
        issues.append(shape_issue)

    dir_info = _borrow_conflict_dir_info(
        records,
        start_pos=start_pos,
        end_pos=end_pos,
        conflict_start_progress_m=conflict_start_progress_m,
        conflict_end_progress_m=conflict_end_progress_m,
    )
    world_geometry = _borrow_conflict_world_geometry(
        scene_borrow_context,
        conflict_start_progress_m=conflict_start_progress_m,
        conflict_end_progress_m=conflict_end_progress_m,
    ) or {}
    windows.append({
        'family': 'borrow',
        'family_code': int(CONFLICT_FAMILY_TO_CODE['borrow']),
        'dir': str(dir_info.get('dir', 'opposite')),
        'dir_code': int(dir_info.get('dir_code', CONFLICT_DIR_TO_CODE['opposite'])),
        'source': 'borrow_conflict_area',
        'source_episode_id': 0,
        'source_priority': int(CONFLICT_FAMILY_PRIORITY['borrow']),
        'area_type': 'corridor_subset',
        'start_pos': int(start_pos),
        'end_pos': int(end_pos),
        'start_frame': int(records[start_pos]['frame_id']),
        'end_frame': int(records[end_pos]['frame_id']),
        'borrow_start_world_xyz': list(scene_borrow_context.get('borrow_start_world_xyz', [])),
        'borrow_end_world_xyz': list(scene_borrow_context.get('borrow_end_world_xyz', [])),
        'borrow_distance_m': float(scene_borrow_context.get('borrow_distance_m', np.nan)),
        'borrow_conflict_start_progress_m': float(conflict_start_progress_m),
        'borrow_conflict_end_progress_m': float(conflict_end_progress_m),
        'area_start_world_xyz': list(world_geometry.get('area_start_world_xyz', [])),
        'area_end_world_xyz': list(world_geometry.get('area_end_world_xyz', [])),
        'area_segment_world_xyz': list(world_geometry.get('area_segment_world_xyz', [])),
        'collision_point_world_xyz': list(dir_info.get('collision_point_world_xyz', [])),
        'dir_source': str(dir_info.get('dir_source', 'family_fallback')),
        'dir_angle_deg': float(dir_info.get('dir_angle_deg', np.nan)),
        'route_heading_deg': float(dir_info.get('route_heading_deg', np.nan)),
        'actor_heading_deg': float(dir_info.get('actor_heading_deg', np.nan)),
        'dir_cover_key': str(dir_info.get('dir_cover_key', 'none')),
        'dir_frame_id': int(dir_info.get('dir_frame_id', -1)),
    })
    return windows, issues


def _merge_direction_heading_rad(records, start_pos, merge_area_end_s_m):
    front_s = _merge_record_scene_front_s(records[int(start_pos)])
    route_local = np.asarray(records[int(start_pos)].get('route_local', np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    if not np.isfinite(front_s):
        return np.nan
    distance_to_area_end_m = max(float(merge_area_end_s_m) - float(front_s), 0.0)
    heading = _route_heading_at_progress(route_local, distance_to_area_end_m + 5.0)
    if np.isfinite(heading):
        return float(heading)
    return float(_route_heading_at_progress(route_local, distance_to_area_end_m + 3.0))


def _build_merge_conflict_windows(records, samples):
    windows = []
    issues = []
    num_records = len(records)
    pos = 0
    source_episode_id = 0
    while pos < num_records:
        start_scan_pos = None
        while pos < num_records:
            if np.isfinite(_merge_record_conflict_s(records[pos])):
                start_scan_pos = int(pos)
                break
            pos += 1
        if start_scan_pos is None:
            break

        candidate_positions = _merge_collect_conflict_candidate_positions(records, start_scan_pos)
        if len(candidate_positions) < int(STAGE1_MERGE_START_CONFIRM_FRAMES):
            # Weak or isolated future merge hints are common; they are not actionable
            # conflict-area failures and should not be recorded as per-route issues.
            pos = int(start_scan_pos) + 1
            continue
        area_info = _merge_resolve_conflict_area(records, candidate_positions)
        if area_info is None:
            issues.append(_conflict_area_issue(samples[records[start_scan_pos]['sample_idx']], 'merge', 'missing_merge_conflict_cluster', pos=int(start_scan_pos)))
            pos = int(start_scan_pos) + 1
            continue

        future_merge_positions = [int(p) for p in area_info.get('inlier_positions', [])]
        conflict_s_values = [
            float(_merge_record_conflict_s(records[int(p)]))
            for p in future_merge_positions
            if np.isfinite(_merge_record_conflict_s(records[int(p)]))
        ]
        if not future_merge_positions or not conflict_s_values:
            issues.append(_conflict_area_issue(samples[records[start_scan_pos]['sample_idx']], 'merge', 'missing_merge_conflict_points', pos=int(start_scan_pos)))
            pos = int(start_scan_pos) + 1
            continue

        eligible_start_positions = [
            int(p) for p in future_merge_positions
            if not _current_follow_chase_start_gate(records[int(p)])
        ]
        if not eligible_start_positions:
            issues.append(_conflict_area_issue(samples[records[start_scan_pos]['sample_idx']], 'merge', 'merge_start_blocked_by_current_follow_chase', pos=int(start_scan_pos)))
            pos = int(max(future_merge_positions)) + 1
            continue

        front_candidates = []
        for p in eligible_start_positions:
            front_s = _merge_record_scene_front_s(records[int(p)])
            if np.isfinite(front_s):
                front_candidates.append((float(front_s), int(p)))
        start_pos = min(front_candidates, key=lambda item: (float(item[0]), int(item[1])))[1] if front_candidates else int(min(eligible_start_positions))

        merge_area_start_s_m = float(area_info['first_conflict_s_m'])
        merge_area_end_s_m = float(area_info['last_conflict_s_m'] + float(STAGE1_MERGE_AREA_POST_MARGIN_M))

        end_pos = None
        for scan_pos in range(int(start_pos), num_records):
            if _merge_record_passed_area(records[scan_pos], merge_area_end_s_m):
                end_pos = int(scan_pos)
                break
        if end_pos is None:
            issues.append(_conflict_area_issue(samples[records[start_pos]['sample_idx']], 'merge', 'missing_merge_end', pos=int(start_pos)))
            pos = int(max(future_merge_positions)) + 1
            continue

        dir_info = _merge_conflict_dir_info(
            records,
            start_pos=start_pos,
            end_pos=end_pos,
            merge_area_start_s_m=merge_area_start_s_m,
            merge_area_end_s_m=merge_area_end_s_m,
        )
        world_geometry = _window_s_interval_world_geometry(
            records,
            start_pos=start_pos,
            end_pos=end_pos,
            area_start_s_m=merge_area_start_s_m,
            area_end_s_m=merge_area_end_s_m,
        ) or {}
        windows.append({
            'family': 'merge',
            'family_code': int(CONFLICT_FAMILY_TO_CODE['merge']),
            'dir': str(dir_info.get('dir', 'same')),
            'dir_code': int(dir_info.get('dir_code', CONFLICT_DIR_TO_CODE['same'])),
            'source': 'merge_area',
            'source_episode_id': int(source_episode_id),
            'source_priority': int(CONFLICT_FAMILY_PRIORITY['merge']),
            'area_type': 's_interval',
            'start_pos': int(start_pos),
            'end_pos': int(end_pos),
            'start_frame': int(records[start_pos]['frame_id']),
            'end_frame': int(records[end_pos]['frame_id']),
            'area_start_s_m': float(merge_area_start_s_m),
            'area_end_s_m': float(merge_area_end_s_m),
            'area_start_world_xyz': list(world_geometry.get('area_start_world_xyz', [])),
            'area_end_world_xyz': list(world_geometry.get('area_end_world_xyz', [])),
            'area_segment_world_xyz': list(world_geometry.get('area_segment_world_xyz', [])),
            'collision_point_world_xyz': list(dir_info.get('collision_point_world_xyz', [])),
            'merge_area_first_conflict_s_m': float(area_info['first_conflict_s_m']),
            'merge_area_last_conflict_s_m': float(area_info['last_conflict_s_m']),
            'merge_direction_heading_rad': float(np.radians(dir_info.get('route_heading_deg', np.nan))) if np.isfinite(float(dir_info.get('route_heading_deg', np.nan))) else np.nan,
            'dir_source': str(dir_info.get('dir_source', 'family_fallback')),
            'dir_angle_deg': float(dir_info.get('dir_angle_deg', np.nan)),
            'route_heading_deg': float(dir_info.get('route_heading_deg', np.nan)),
            'actor_heading_deg': float(dir_info.get('actor_heading_deg', np.nan)),
            'dir_cover_key': str(dir_info.get('dir_cover_key', 'none')),
            'dir_frame_id': int(dir_info.get('dir_frame_id', -1)),
        })
        source_episode_id += 1
        pos = int(end_pos) + 1

    return windows, issues


def _build_junction_conflict_windows(records, samples):
    windows = []
    issues = []
    source_episode_id = 0
    prev_end_pos = -1
    for cluster in _junction_cluster_conflict_candidates(records):
        cluster_positions = sorted(
            int(item['pos']) for item in cluster['items']
            if int(item['pos']) > int(prev_end_pos)
        )
        cluster_positions = [
            int(pos) for pos in cluster_positions
            if not _current_follow_chase_start_gate(records[int(pos)])
        ]
        if not cluster_positions:
            issue_pos = int(max(int(prev_end_pos) + 1, 0))
            issues.append(_conflict_area_issue(samples[records[issue_pos]['sample_idx']], 'junction', 'junction_start_blocked_by_current_follow_chase', pos=issue_pos))
            continue

        cluster_conflict_s_m = float(cluster.get('conflict_s_m', np.nan))
        if not np.isfinite(cluster_conflict_s_m):
            issues.append(_conflict_area_issue(samples[records[cluster_positions[0]]['sample_idx']], 'junction', 'missing_junction_conflict_s', pos=int(cluster_positions[0])))
            continue

        area_start_s_m = float(max(cluster_conflict_s_m - float(STAGE1_JUNCTION_AREA_PRE_MARGIN_M), 0.0))
        area_end_s_m = float(max(cluster_conflict_s_m + float(STAGE1_JUNCTION_AREA_POST_MARGIN_M), area_start_s_m))

        start_pos = None
        for pos in range(int(max(int(prev_end_pos) + 1, 0)), len(records)):
            front_s = _record_scene_front_s(records[pos])
            if np.isfinite(front_s) and float(front_s) >= float(area_start_s_m):
                start_pos = int(pos)
                break
        if start_pos is None:
            issues.append(_conflict_area_issue(
                samples[records[cluster_positions[0]]['sample_idx']],
                'junction',
                'missing_junction_start',
                pos=int(cluster_positions[0]),
                cluster_conflict_s_m=float(cluster_conflict_s_m),
            ))
            continue

        end_pos = None
        for pos in range(int(start_pos), len(records)):
            front_s = _record_scene_front_s(records[pos])
            if np.isfinite(front_s) and float(front_s) >= float(area_end_s_m):
                end_pos = int(pos)
                break
        if end_pos is None:
            issues.append(_conflict_area_issue(
                samples[records[start_pos]['sample_idx']],
                'junction',
                'missing_junction_end',
                pos=int(start_pos),
                cluster_conflict_s_m=float(cluster_conflict_s_m),
            ))
            continue

        shape_issue = _record_window_cover_shape_issue(
            samples,
            records,
            family='junction',
            positions=range(int(start_pos), int(end_pos) + 1),
        )
        if shape_issue is not None:
            issues.append(shape_issue)

        center_xyz = np.asarray(cluster.get('center_xyz', []), dtype=np.float32).reshape(-1)
        radius_m = float(cluster.get('radius_m', np.nan))
        if center_xyz.size < 3 or not np.all(np.isfinite(center_xyz[:3])) or not np.isfinite(radius_m):
            issues.append(_conflict_area_issue(samples[records[start_pos]['sample_idx']], 'junction', 'missing_junction_area_geometry', pos=int(start_pos)))
            continue

        dir_info = _junction_conflict_dir_info(
            records,
            start_pos=start_pos,
            end_pos=end_pos,
            area_start_s_m=area_start_s_m,
            area_end_s_m=area_end_s_m,
        )
        world_geometry = _window_s_interval_world_geometry(
            records,
            start_pos=start_pos,
            end_pos=end_pos,
            area_start_s_m=area_start_s_m,
            area_end_s_m=area_end_s_m,
        ) or {}
        windows.append({
            'family': 'junction',
            'family_code': int(CONFLICT_FAMILY_TO_CODE['junction']),
            'dir': str(dir_info.get('dir', 'none')),
            'dir_code': int(dir_info.get('dir_code', CONFLICT_DIR_TO_CODE['none'])),
            'source': 'junction_conflict_area',
            'source_episode_id': int(source_episode_id),
            'source_priority': int(CONFLICT_FAMILY_PRIORITY['junction']),
            'area_type': 'circle',
            'start_pos': int(start_pos),
            'end_pos': int(end_pos),
            'start_frame': int(records[start_pos]['frame_id']),
            'end_frame': int(records[end_pos]['frame_id']),
            'area_start_s_m': float(area_start_s_m),
            'area_end_s_m': float(area_end_s_m),
            'area_start_world_xyz': list(world_geometry.get('area_start_world_xyz', [])),
            'area_end_world_xyz': list(world_geometry.get('area_end_world_xyz', [])),
            'area_segment_world_xyz': list(world_geometry.get('area_segment_world_xyz', [])),
            'collision_point_world_xyz': list(dir_info.get('collision_point_world_xyz', [])),
            'area_center_world_xyz': center_xyz[:3].astype(float).tolist(),
            'area_radius_m': float(radius_m),
            'cluster_conflict_s_m': float(cluster_conflict_s_m),
            'candidate_frame_count': int(len(cluster_positions)),
            'dir_source': str(dir_info.get('dir_source', 'family_fallback')),
            'dir_angle_deg': float(dir_info.get('dir_angle_deg', np.nan)),
            'route_heading_deg': float(dir_info.get('route_heading_deg', np.nan)),
            'actor_heading_deg': float(dir_info.get('actor_heading_deg', np.nan)),
            'dir_cover_key': str(dir_info.get('dir_cover_key', 'none')),
            'dir_frame_id': int(dir_info.get('dir_frame_id', -1)),
        })
        prev_end_pos = int(end_pos)
        source_episode_id += 1
    return windows, issues


def _conflict_window_frame_role(window, pos):
    if int(pos) == int(window.get('start_pos', -1)):
        return 'start'
    if int(pos) == int(window.get('end_pos', -1)):
        return 'end'
    return 'active'


def _active_conflict_window_identity(sample):
    if float(sample.get('conflict_area_active', 0.0)) <= 0.5:
        return None
    stage1_debug = sample.get('stage1_speed_debug')
    if not isinstance(stage1_debug, dict):
        return None
    conflict_info = stage1_debug.get('conflict_area') or {}
    family = str(conflict_info.get('family', 'none'))
    if family == 'none':
        return None
    return (
        str(family),
        int(conflict_info.get('start_frame', -1)),
        int(conflict_info.get('end_frame', -1)),
        str(conflict_info.get('source', 'none')),
        int(conflict_info.get('source_episode_id', -1)),
    )


def _record_conflict_speed_mps(record, family):
    if str(family) == 'borrow':
        borrow_motion = (record or {}).get('borrow_motion') or {}
        speed_mps = float(borrow_motion.get('speed_mps', np.nan))
        if np.isfinite(speed_mps):
            return float(speed_mps)
    merge_motion = (record or {}).get('merge_motion') or {}
    speed_mps = float(merge_motion.get('speed_mps', np.nan))
    return float(speed_mps) if np.isfinite(speed_mps) else np.nan


def _conflict_window_entry_pos(records, family, conflict_info, start_pos, end_pos):
    family = str(family or 'none')
    if family == 'borrow':
        conflict_start_progress_m = float(conflict_info.get('borrow_conflict_start_progress_m', np.nan))
        if not np.isfinite(conflict_start_progress_m):
            return None
        for pos in range(int(start_pos), int(end_pos) + 1):
            borrow_motion = (records[int(pos)].get('borrow_motion') or {})
            borrow_start_distance_m = float(borrow_motion.get('borrow_start_distance_m', np.nan))
            if not np.isfinite(borrow_start_distance_m):
                continue
            dist_to_area_start_m = float(borrow_start_distance_m - conflict_start_progress_m)
            if dist_to_area_start_m <= float(STAGE1_CONFLICT_AREA_ENTRY_TOL_M):
                return int(pos)
        return None

    area_start_s_m = float(conflict_info.get('area_start_s_m', np.nan))
    if not np.isfinite(area_start_s_m):
        return None
    for pos in range(int(start_pos), int(end_pos) + 1):
        front_s_m = _record_scene_front_s(records[int(pos)])
        if not np.isfinite(front_s_m):
            continue
        if front_s_m >= float(area_start_s_m) - float(STAGE1_CONFLICT_AREA_ENTRY_TOL_M):
            return int(pos)
    return None


def _conflict_window_release_pos(records, family, start_pos, entry_pos, end_pos):
    for pos in range(int(entry_pos), int(start_pos) - 1, -1):
        speed_mps = _record_conflict_speed_mps(records[int(pos)], family)
        if np.isfinite(speed_mps) and speed_mps <= float(STAGE1_CONFLICT_GO_STOP_SPEED_THRESH_MPS):
            return int(pos), 'stopped'

    return int(start_pos), 'window_start'


def _conflict_window_go_pos(records, family, start_pos, release_pos, end_pos, release_reason):
    if str(release_reason) != 'stopped':
        return int(start_pos), 'window_start_direct_go'

    for pos in range(int(release_pos), int(end_pos) + 1):
        speed_mps = _record_conflict_speed_mps(records[int(pos)], family)
        if np.isfinite(speed_mps) and speed_mps > float(STAGE1_CONFLICT_GO_START_SPEED_THRESH_MPS):
            return int(pos), 'post_stop_speed_restart'

    return None, 'stopped_no_restart'


def _conflict_phase_frame_role(pos, start_pos, end_pos, entry_pos, go_pos, phase):
    tags = []
    if int(pos) == int(start_pos):
        tags.append('start')
    if entry_pos is not None and int(pos) == int(entry_pos):
        tags.append('entry')
    if go_pos is not None and int(pos) == int(go_pos):
        tags.append('go_start')
    if int(pos) == int(end_pos):
        tags.append('end')
    if tags:
        return '_'.join(tags)
    return str(phase)


def _annotate_route_stage1_conflict_phases(samples, route_sample_indices):
    records = _build_route_conflict_records(samples, route_sample_indices)
    if not records:
        return

    for record in records:
        sample = samples[int(record['sample_idx'])]
        _set_stage1_conflict_phase_defaults(sample)

    pos = 0
    while pos < len(records):
        sample = samples[int(records[int(pos)]['sample_idx'])]
        window_key = _active_conflict_window_identity(sample)
        if window_key is None:
            pos += 1
            continue

        start_pos = int(pos)
        stage1_debug = sample.get('stage1_speed_debug') or {}
        conflict_info = dict(stage1_debug.get('conflict_area') or {})
        while pos + 1 < len(records):
            next_sample = samples[int(records[int(pos) + 1]['sample_idx'])]
            if _active_conflict_window_identity(next_sample) != window_key:
                break
            pos += 1
        end_pos = int(pos)

        family = str(conflict_info.get('family', 'none'))
        start_frame = int(conflict_info.get('start_frame', -1))
        end_frame = int(conflict_info.get('end_frame', -1))
        entry_pos = _conflict_window_entry_pos(
            records,
            family=family,
            conflict_info=conflict_info,
            start_pos=start_pos,
            end_pos=end_pos,
        )
        if entry_pos is None:
            go_pos = None
            go_frame = -1
            release_pos = None
            release_reason = 'entry_missing'
        else:
            release_pos, release_reason = _conflict_window_release_pos(
                records,
                family=family,
                start_pos=start_pos,
                entry_pos=entry_pos,
                end_pos=end_pos,
            )
            go_pos, go_reason = _conflict_window_go_pos(
                records,
                family=family,
                start_pos=start_pos,
                release_pos=release_pos,
                end_pos=end_pos,
                release_reason=release_reason,
            )
            go_frame = int(records[int(go_pos)]['frame_id']) if go_pos is not None else -1

        for window_pos in range(int(start_pos), int(end_pos) + 1):
            record = records[int(window_pos)]
            phase = 'yld' if go_pos is None or int(window_pos) < int(go_pos) else 'go'
            phase_info = {
                'phase': str(phase),
                'phase_code': int(CONFLICT_DECISION_PHASE_TO_CODE.get(phase, 0)),
                'active': 1.0,
                'family': str(family),
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'entry_frame': int(records[int(entry_pos)]['frame_id']) if entry_pos is not None else -1,
                'go_frame': int(go_frame),
                'release_frame': int(records[int(release_pos)]['frame_id']) if release_pos is not None else -1,
                'frame_role': _conflict_phase_frame_role(
                    window_pos,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    entry_pos=entry_pos,
                    go_pos=go_pos,
                    phase=phase,
                ),
                'source': 'area_entry_backward_stop_then_restart',
                'release_reason': str(release_reason if entry_pos is None else go_reason),
                'entry_found': float(entry_pos is not None),
                'speed_mps': float(_record_conflict_speed_mps(record, family)),
                'stop_speed_thresh_mps': float(STAGE1_CONFLICT_GO_STOP_SPEED_THRESH_MPS),
            }
            _set_stage1_conflict_phase_annotation(samples[int(record['sample_idx'])], phase_info)

        pos += 1


def _annotate_route_stage1_conflict_areas(samples, route_sample_indices, scene_route_world=None):
    records = _build_route_conflict_records(samples, route_sample_indices, scene_route_world=scene_route_world)
    if not records:
        return

    for record in records:
        sample = samples[int(record['sample_idx'])]
        _set_stage1_conflict_area_defaults(sample)

    borrow_windows, borrow_issues = _build_borrow_conflict_windows(records, samples)
    merge_windows, merge_issues = _build_merge_conflict_windows(records, samples)
    junction_windows, junction_issues = _build_junction_conflict_windows(records, samples)
    route_windows = borrow_windows + merge_windows + junction_windows
    route_issues = borrow_issues + merge_issues + junction_issues
    issues_by_pos = {}
    for issue in route_issues:
        issue_pos = int(issue.get('pos', -1))
        if 0 <= issue_pos < len(records):
            issues_by_pos.setdefault(issue_pos, []).append(dict(issue))

    for pos, record in enumerate(records):
        sample = samples[int(record['sample_idx'])]
        frame_issues = issues_by_pos.get(int(pos), [])
        active_windows = [
            dict(window) for window in route_windows
            if int(window.get('start_pos', -1)) <= int(pos) <= int(window.get('end_pos', -1))
        ]
        if active_windows:
            active_windows.sort(key=lambda item: (int(item.get('source_priority', 99)), str(item.get('family', 'none'))))
            selected = dict(active_windows[0])
            selected['active'] = 1.0
            selected['frame_role'] = _conflict_window_frame_role(selected, pos)
            selected['active_families'] = [str(item.get('family', 'none')) for item in active_windows]
            selected['active_family_count'] = int(len(active_windows))
            selected['selection_reason'] = 'single_active_family' if len(active_windows) == 1 else 'priority'
            selected['issue_count'] = int(len(frame_issues))
            selected['issue_families'] = [str(item.get('family', 'none')) for item in frame_issues]
            selected['missing_reason'] = str(frame_issues[0].get('reason', 'none')) if frame_issues else 'none'
            selected['topology_override'] = str(frame_issues[0].get('topology_override', 'none')) if frame_issues else 'none'
            _set_stage1_conflict_area_annotation(sample, selected)
            continue

        info = _default_conflict_area_debug()
        if frame_issues:
            info['issue_count'] = int(len(frame_issues))
            info['issue_families'] = [str(item.get('family', 'none')) for item in frame_issues]
            info['missing_reason'] = str(frame_issues[0].get('reason', 'unknown'))
            info['selection_reason'] = 'issue_only'
            info['topology_override'] = str(frame_issues[0].get('topology_override', 'none'))
        _set_stage1_conflict_area_annotation(sample, info)


def _has_stage1_speed_fields(sample):
    return all(field in sample for field in STAGE1_SPEED_FIELDS)


def _set_stage1_speed_fallback(sample):
    sample['stage1_speed_debug'] = _build_stage1_speed_debug_payload(
        current_cover=_cover_candidate_summary(0, None, {}),
        future_cover=_cover_candidate_summary(0, None, {}),
        merge_motion=None,
        scene_borrow_context=None,
        borrow_motion=None,
    )
    _set_stage1_conflict_area_defaults(sample)
    _set_stage1_conflict_phase_defaults(sample)


def _cover_actor_id(cover):
    try:
        return int((cover or {}).get('actor_id', -1))
    except Exception:
        return -1


def _cover_interaction_name(cover):
    return str(((cover or {}).get('interaction') or {}).get('name', 'none'))


def _cover_interaction_subtype(cover):
    interaction = ((cover or {}).get('interaction') or {})
    return str(interaction.get('subtype') or interaction.get('name') or 'none')


def _cover_is_merge_meet(cover):
    return int((cover or {}).get('exists', 0.0)) > 0 and _cover_interaction_subtype(cover) == 'merge_meet'


def _cover_is_chase(cover):
    return int((cover or {}).get('exists', 0.0)) > 0 and _cover_interaction_name(cover) == 'chase'


def _cover_is_cross_meet(cover):
    if int((cover or {}).get('exists', 0.0)) <= 0:
        return False
    if _cover_interaction_name(cover) != 'meet':
        return False
    subtype = _cover_interaction_subtype(cover)
    return 'cross' in subtype


def _build_merge_motion_context(
    current_meas,
    route_dense,
    ego_matrix_current=None,
    current_boxes=None,
    scene_route_polyline_world=None,
):
    steer = np.nan if current_meas is None else float(current_meas.get('steer', np.nan))
    theta = np.nan if current_meas is None else float(current_meas.get('theta', np.nan))
    speed = np.nan if current_meas is None else float(current_meas.get('speed', np.nan))
    light_hazard = 0.0 if current_meas is None else float(bool(current_meas.get('light_hazard', False)))
    stop_sign_hazard = 0.0 if current_meas is None else float(bool(current_meas.get('stop_sign_hazard', False)))
    route_heading_local = _route_heading_at_idx(route_dense, 1)
    if route_heading_local is None:
        route_heading_local = _route_heading_at_idx(route_dense, 0)
    heading_error_deg = np.nan if route_heading_local is None else abs(_heading_to_deg(route_heading_local))
    ego_half_length_m = float(DEFAULT_EGO_EXTENT_2D[0])
    ego_box = _find_ego_box(current_boxes or [])
    if ego_box is not None:
        ego_extent = np.asarray(ego_box.get('extent', DEFAULT_EGO_EXTENT_2D)[:2], dtype=np.float32)
        if ego_extent.size >= 1 and np.isfinite(float(ego_extent[0])):
            ego_half_length_m = float(max(float(ego_extent[0]), 1e-3))

    scene_route_center_s = np.nan
    scene_route_front_s = np.nan
    scene_route_rear_s = np.nan
    ego_world_xyz = np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
    ego_matrix_list = []
    scene_route_polyline_world = np.asarray(scene_route_polyline_world, dtype=np.float32)
    if (
        ego_matrix_current is not None and
        scene_route_polyline_world.ndim == 2 and
        scene_route_polyline_world.shape[0] >= 2
    ):
        ego_matrix_np = np.asarray(ego_matrix_current, dtype=np.float32)
        if ego_matrix_np.ndim == 2 and ego_matrix_np.shape[0] >= 3 and ego_matrix_np.shape[1] >= 4:
            ego_world_xyz = ego_matrix_np[:3, 3].astype(np.float32)
        if ego_matrix_np.shape == (4, 4) and np.all(np.isfinite(ego_matrix_np)):
            ego_matrix_list = ego_matrix_np.astype(float).tolist()
        ego_probe_local = np.asarray(
            [
                [0.0, 0.0],
                [float(ego_half_length_m), 0.0],
                [-float(ego_half_length_m), 0.0],
            ],
            dtype=np.float32,
        )
        ego_probe_world = _transform_points_local_to_world_xyz(ego_probe_local, ego_matrix_current)
        if ego_probe_world.shape[0] == 3:
            _, scene_route_center_s = _project_point_to_polyline(ego_probe_world[0, :2], scene_route_polyline_world[:, :2])
            _, scene_route_front_s = _project_point_to_polyline(ego_probe_world[1, :2], scene_route_polyline_world[:, :2])
            _, scene_route_rear_s = _project_point_to_polyline(ego_probe_world[2, :2], scene_route_polyline_world[:, :2])
    return {
        'steer': float(steer) if np.isfinite(steer) else np.nan,
        'theta_rad': float(theta) if np.isfinite(theta) else np.nan,
        'speed_mps': float(speed) if np.isfinite(speed) else np.nan,
        'light_hazard': float(light_hazard),
        'stop_sign_hazard': float(stop_sign_hazard),
        'route_heading_local_rad': np.nan if route_heading_local is None else float(route_heading_local),
        'heading_error_deg': float(heading_error_deg) if np.isfinite(heading_error_deg) else np.nan,
        'ego_half_length_m': float(ego_half_length_m),
        'ego_matrix': ego_matrix_list,
        'ego_world_xyz': ego_world_xyz.astype(float).tolist() if np.all(np.isfinite(ego_world_xyz)) else [],
        'scene_route_center_s_m': float(scene_route_center_s) if np.isfinite(scene_route_center_s) else np.nan,
        'scene_route_front_s_m': float(scene_route_front_s) if np.isfinite(scene_route_front_s) else np.nan,
        'scene_route_rear_s_m': float(scene_route_rear_s) if np.isfinite(scene_route_rear_s) else np.nan,
    }


def _merge_go_lane_settled(merge_motion):
    steer = float((merge_motion or {}).get('steer', np.nan))
    heading_error_deg = float((merge_motion or {}).get('heading_error_deg', np.nan))
    if not np.isfinite(steer) or not np.isfinite(heading_error_deg):
        return False
    return bool(
        abs(steer) <= float(STAGE1_GO_END_STEER_ABS_THRESH) and
        abs(heading_error_deg) <= float(STAGE1_GO_END_HEADING_ALIGN_THRESH_DEG)
    )


def _merge_is_red_light_wait(merge_motion):
    speed_mps = float((merge_motion or {}).get('speed_mps', np.nan))
    light_hazard = float((merge_motion or {}).get('light_hazard', 0.0))
    if not np.isfinite(speed_mps):
        return False
    return bool(
        light_hazard > 0.5 and
        speed_mps <= float(STAGE1_MERGE_RED_LIGHT_WAIT_SPEED_THRESH)
    )


def _cover_angle_deg(cover):
    interaction = ((cover or {}).get('interaction') or {})
    try:
        angle_deg = float(interaction.get('angle_deg', np.nan))
    except Exception:
        angle_deg = np.nan
    return float(angle_deg) if np.isfinite(angle_deg) else np.nan


def _cover_is_cross_like(cover):
    if int((cover or {}).get('exists', 0.0)) <= 0:
        return False
    subtype = _cover_interaction_subtype(cover)
    if 'cross' in subtype:
        return True
    angle_deg = _cover_angle_deg(cover)
    return bool(np.isfinite(angle_deg) and angle_deg >= float(INTERACTION_CROSS_MIN_ANGLE_THRESH_DEG))


def _current_follow_chase_start_gate(record):
    current_cover = (record or {}).get('current_cover') or {}
    if int(current_cover.get('exists', 0.0)) <= 0:
        return False
    if _cover_interaction_subtype(current_cover) != 'follow_chase':
        return False
    other_speed = float(current_cover.get('other_speed', np.nan))
    route_distance_m = float(current_cover.get('route_distance_m', np.nan))
    if not np.isfinite(other_speed) or not np.isfinite(route_distance_m):
        return False
    return bool(
        other_speed <= float(STAGE1_FUTURE_START_GATE_CHASE_SPEED_THRESH_MPS) and
        route_distance_m <= float(STAGE1_FUTURE_START_GATE_CHASE_DISTANCE_THRESH_M)
    )




def _merge_record_scene_front_s(record):
    merge_motion = (record or {}).get('merge_motion') or {}
    front_s = float(merge_motion.get('scene_route_front_s_m', np.nan))
    if np.isfinite(front_s):
        return float(front_s)
    center_s = float(merge_motion.get('scene_route_center_s_m', np.nan))
    return float(center_s) if np.isfinite(center_s) else np.nan


def _merge_record_scene_rear_s(record):
    merge_motion = (record or {}).get('merge_motion') or {}
    rear_s = float(merge_motion.get('scene_route_rear_s_m', np.nan))
    if np.isfinite(rear_s):
        return float(rear_s)
    center_s = float(merge_motion.get('scene_route_center_s_m', np.nan))
    return float(center_s) if np.isfinite(center_s) else np.nan


def _merge_record_speed_mps(record):
    merge_motion = (record or {}).get('merge_motion') or {}
    speed_mps = float(merge_motion.get('speed_mps', np.nan))
    return float(speed_mps) if np.isfinite(speed_mps) else np.nan


def _merge_record_conflict_s(record):
    future_cover = (record or {}).get('future_cover') or {}
    if not _cover_is_merge_meet(future_cover):
        return np.nan
    conflict_s = float(future_cover.get('scene_route_conflict_s_m', np.nan))
    if np.isfinite(conflict_s):
        return float(conflict_s)
    route_distance_m = float(future_cover.get('route_distance_m', np.nan))
    ego_route_front_s_m = float(future_cover.get('ego_route_front_s_m', np.nan))
    scene_front_s_m = _merge_record_scene_front_s(record)
    if np.isfinite(route_distance_m) and np.isfinite(ego_route_front_s_m) and np.isfinite(scene_front_s_m):
        return float(scene_front_s_m + max(route_distance_m - ego_route_front_s_m, 0.0))
    return np.nan


def _merge_record_thresholds(record):
    meet_debug = (record or {}).get('meet_debug') or {}
    if str(meet_debug.get('subtype', 'none')) != 'merge_meet':
        return np.nan, np.nan, np.nan
    v_yield_max = _merge_speed_cap(meet_debug.get('v_yield_max_mps', np.nan))
    v_go_min = _merge_speed_cap(meet_debug.get('v_go_min_mps', np.nan))
    v_go_need = _merge_speed_cap(meet_debug.get('v_go_need_mps', np.nan))
    go_threshold = float(v_go_need) if np.isfinite(v_go_need) else float(v_go_min)
    return float(v_yield_max), float(v_go_min), float(go_threshold)


def _merge_record_go_signal(record, merge_area_start_s_m):
    front_s = _merge_record_scene_front_s(record)
    if np.isfinite(front_s) and np.isfinite(float(merge_area_start_s_m)) and front_s >= float(merge_area_start_s_m):
        return True, 'merge_area_entry'
    speed_mps = _merge_record_speed_mps(record)
    v_yield_max, _, go_threshold = _merge_record_thresholds(record)
    if np.isfinite(speed_mps):
        if np.isfinite(go_threshold) and speed_mps >= float(go_threshold):
            return True, 'speed_go_threshold'
        if not np.isfinite(v_yield_max) and not np.isfinite(go_threshold):
            return False, 'threshold_missing'
    return False, 'none'


def _merge_record_passed_area(record, merge_area_end_s_m):
    rear_s = _merge_record_scene_rear_s(record)
    if not np.isfinite(rear_s) or not np.isfinite(float(merge_area_end_s_m)):
        return False
    return bool(rear_s >= float(merge_area_end_s_m))


def _merge_collect_conflict_candidate_positions(records, start_scan_pos):
    if int(start_scan_pos) < 0 or int(start_scan_pos) >= len(records):
        return []
    first_conflict_s = _merge_record_conflict_s(records[int(start_scan_pos)])
    if not np.isfinite(first_conflict_s):
        return []
    max_front_s = float(first_conflict_s) + float(STAGE1_MERGE_CONFLICT_LOOKAHEAD_PROGRESS_M)
    candidate_positions = []
    for pos in range(int(start_scan_pos), len(records)):
        front_s = _merge_record_scene_front_s(records[pos])
        if np.isfinite(front_s) and np.isfinite(max_front_s) and float(front_s) > float(max_front_s):
            break
        conflict_s = _merge_record_conflict_s(records[pos])
        if np.isfinite(conflict_s):
            candidate_positions.append(int(pos))
    return candidate_positions


def _merge_resolve_conflict_area(records, candidate_positions):
    points = []
    for pos in candidate_positions:
        conflict_s = _merge_record_conflict_s(records[int(pos)])
        front_s = _merge_record_scene_front_s(records[int(pos)])
        if not np.isfinite(conflict_s):
            continue
        points.append({
            'pos': int(pos),
            'conflict_s': float(conflict_s),
            'front_s': float(front_s) if np.isfinite(front_s) else np.nan,
        })
    if len(points) < int(STAGE1_MERGE_START_CONFIRM_FRAMES):
        return None

    points_by_s = sorted(points, key=lambda item: (float(item['conflict_s']), int(item['pos'])))
    clusters = []
    current_cluster = [points_by_s[0]]
    for item in points_by_s[1:]:
        prev_s = float(current_cluster[-1]['conflict_s'])
        cur_s = float(item['conflict_s'])
        if abs(cur_s - prev_s) <= float(STAGE1_MERGE_CONFLICT_CLUSTER_GAP_M):
            current_cluster.append(item)
        else:
            clusters.append(current_cluster)
            current_cluster = [item]
    clusters.append(current_cluster)

    valid_clusters = [cluster for cluster in clusters if len(cluster) >= int(STAGE1_MERGE_START_CONFIRM_FRAMES)]
    if not valid_clusters:
        return None

    def _cluster_sort_key(cluster):
        conflict_vals = [float(item['conflict_s']) for item in cluster]
        spread = float(max(conflict_vals) - min(conflict_vals)) if conflict_vals else np.inf
        min_front_s = min(
            [float(item['front_s']) for item in cluster if np.isfinite(float(item['front_s']))] or [np.inf]
        )
        return (-len(cluster), spread, min_front_s, min(int(item['pos']) for item in cluster))

    best_cluster = min(valid_clusters, key=_cluster_sort_key)
    inlier_positions = sorted(int(item['pos']) for item in best_cluster)
    conflict_vals = [float(item['conflict_s']) for item in best_cluster]
    front_candidates = [
        (float(item['front_s']), int(item['pos']))
        for item in best_cluster
        if np.isfinite(float(item['front_s']))
    ]
    if front_candidates:
        _, start_pos = min(front_candidates, key=lambda item: (float(item[0]), int(item[1])))
    else:
        start_pos = min(inlier_positions)

    return {
        'start_pos': int(start_pos),
        'inlier_positions': inlier_positions,
        'first_conflict_s_m': float(min(conflict_vals)),
        'last_conflict_s_m': float(max(conflict_vals)),
    }


def _junction_cover_conflict_world_xyz(cover):
    if int((cover or {}).get('exists', 0.0)) <= 0:
        return None
    if str(((cover or {}).get('interaction') or {}).get('subtype', 'none')) != 'junction_left_cross_meet':
        return None
    pt = np.asarray((cover or {}).get('scene_route_conflict_world_xyz', []), dtype=np.float32).reshape(-1)
    if pt.size >= 3 and np.all(np.isfinite(pt[:3])):
        return pt[:3].astype(np.float32)
    return None


def _junction_record_conflict_world_xyz(record):
    future_cover = (record or {}).get('future_cover') or {}
    pt = _junction_cover_conflict_world_xyz(future_cover)
    if pt is not None:
        return pt
    current_cover = (record or {}).get('current_cover') or {}
    return _junction_cover_conflict_world_xyz(current_cover)


def _junction_cover_conflict_s(cover):
    if int((cover or {}).get('exists', 0.0)) <= 0:
        return np.nan
    if str(((cover or {}).get('interaction') or {}).get('subtype', 'none')) != 'junction_left_cross_meet':
        return np.nan
    conflict_s = float((cover or {}).get('scene_route_conflict_s_m', np.nan))
    return float(conflict_s) if np.isfinite(conflict_s) else np.nan


def _junction_record_conflict_s(record):
    future_cover = (record or {}).get('future_cover') or {}
    conflict_s = _junction_cover_conflict_s(future_cover)
    if np.isfinite(conflict_s):
        return float(conflict_s)
    current_cover = (record or {}).get('current_cover') or {}
    conflict_s = _junction_cover_conflict_s(current_cover)
    return float(conflict_s) if np.isfinite(conflict_s) else np.nan


def _junction_record_conflict_radius_m(record):
    meet_debug = (record or {}).get('meet_debug') or {}
    radius_m = float(meet_debug.get('context_conflict_len_m', np.nan))
    if np.isfinite(radius_m) and radius_m > 0.0:
        return float(radius_m)
    return float(STAGE1_JUNCTION_CROSS_FALLBACK_RADIUS_M)


def _junction_record_scene_front_s(record):
    merge_motion = (record or {}).get('merge_motion') or {}
    front_s = float(merge_motion.get('scene_route_front_s_m', np.nan))
    if np.isfinite(front_s):
        return float(front_s)
    center_s = float(merge_motion.get('scene_route_center_s_m', np.nan))
    return float(center_s) if np.isfinite(center_s) else np.nan


def _junction_record_ego_world_xyz(record):
    merge_motion = (record or {}).get('merge_motion') or {}
    pt = np.asarray(merge_motion.get('ego_world_xyz', []), dtype=np.float32).reshape(-1)
    if pt.size >= 3 and np.all(np.isfinite(pt[:3])):
        return pt[:3].astype(np.float32)
    return None


def _junction_cluster_conflict_candidates(records):
    candidate_points = []
    for pos, record in enumerate(records):
        pt_xyz = _junction_record_conflict_world_xyz(record)
        if pt_xyz is None:
            continue
        candidate_points.append({
            'pos': int(pos),
            'frame_id': int(record.get('frame_id', -1)),
            'point_xyz': pt_xyz.astype(np.float32),
            'conflict_s': float(_junction_record_conflict_s(record)),
            'radius_m': float(_junction_record_conflict_radius_m(record)),
            'front_s': float(_junction_record_scene_front_s(record)),
        })
    if not candidate_points:
        return []

    clusters = []
    for item in sorted(candidate_points, key=lambda entry: int(entry['pos'])):
        assigned = None
        for cluster in clusters:
            center = np.asarray(cluster['center_xyz'], dtype=np.float32)
            radius_m = float(max(cluster['radius_m'], item['radius_m']))
            if float(np.linalg.norm(item['point_xyz'][:2] - center[:2])) <= radius_m:
                assigned = cluster
                break
        if assigned is None:
            clusters.append({
                'center_xyz': item['point_xyz'].astype(np.float32),
                'radius_m': float(item['radius_m']),
                'items': [item],
            })
            continue
        assigned['items'].append(item)
        pts = np.stack([entry['point_xyz'] for entry in assigned['items']], axis=0)
        assigned['center_xyz'] = np.mean(pts, axis=0).astype(np.float32)
        assigned['radius_m'] = float(max(float(assigned['radius_m']), float(item['radius_m'])))

    valid_clusters = []
    for cluster in clusters:
        items = list(cluster['items'])
        if len(items) < int(STAGE1_JUNCTION_CROSS_MIN_CLUSTER_POINTS):
            continue
        front_candidates = [
            (float(entry['front_s']), int(entry['pos']))
            for entry in items
            if np.isfinite(float(entry['front_s']))
        ]
        if front_candidates:
            _, start_pos = min(front_candidates, key=lambda entry: (float(entry[0]), int(entry[1])))
        else:
            start_pos = min(int(entry['pos']) for entry in items)
        conflict_s_values = [
            float(entry['conflict_s'])
            for entry in items
            if np.isfinite(float(entry.get('conflict_s', np.nan)))
        ]
        valid_clusters.append({
            'center_xyz': np.asarray(cluster['center_xyz'], dtype=np.float32),
            'radius_m': float(cluster['radius_m']),
            'items': items,
            'start_pos': int(start_pos),
            'conflict_s_m': float(np.median(conflict_s_values)) if conflict_s_values else np.nan,
        })
    valid_clusters.sort(key=lambda cluster: int(cluster['start_pos']))
    return valid_clusters


def _build_borrow_motion_context(release_info, current_meas=None, route_local=None, frame_id=-1):
    speed_mps = np.nan if current_meas is None else float(current_meas.get('speed', np.nan))
    corridor = _borrow_corridor_metrics(
        release_info,
        current_meas=current_meas,
        route_local=route_local,
    )
    if corridor is None:
        return {
            'valid': 0.0,
            'frame_id': int(frame_id),
            'speed_mps': float(speed_mps) if np.isfinite(speed_mps) else np.nan,
            'borrow_start_distance_m': np.nan,
            'borrow_end_distance_m': np.nan,
            'corridor_length_m': np.nan,
            'context_frame_id': int((release_info or {}).get('context_frame_id', -1)),
        }
    return {
        'valid': 1.0,
        'frame_id': int(frame_id),
        'speed_mps': float(speed_mps) if np.isfinite(speed_mps) else np.nan,
        'borrow_start_distance_m': float(corridor.get('borrow_start_distance_m', np.nan)),
        'borrow_end_distance_m': float(corridor.get('borrow_end_distance_m', np.nan)),
        'corridor_length_m': float(corridor.get('borrow_distance_m', np.nan)),
        'context_frame_id': int((release_info or {}).get('context_frame_id', -1)),
    }


def _build_future_frames_data(image_data_root, base_dir, frame_str, num_future):
    future_frames_data = []
    frame_id = int(frame_str)
    for k in range(1, num_future + 1):
        future_frame_str = f'{frame_id + k:04d}'
        fut_boxes_path = os.path.join(image_data_root, base_dir, 'boxes', f'{future_frame_str}.json.gz')
        fut_meas_path = os.path.join(image_data_root, base_dir, 'measurements', f'{future_frame_str}.json.gz')
        fut_boxes = _load_json_gz_if_exists(fut_boxes_path)
        fut_meas = _load_json_gz_if_exists(fut_meas_path)
        if fut_boxes is None or fut_meas is None:
            future_frames_data.append(None)
            continue
        fut_ego_matrix = fut_meas.get('ego_matrix', None)
        if fut_ego_matrix is not None:
            future_frames_data.append((fut_boxes, fut_ego_matrix))
        else:
            future_frames_data.append(None)
    return future_frames_data


def _atomic_pickle_save(obj, target_path):
    tmp_path = target_path + f'.tmp.{os.getpid()}'
    with open(tmp_path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, target_path)


def _atomic_json_save(payload, target_path):
    tmp_path = target_path + f'.tmp.{os.getpid()}'
    with open(tmp_path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, target_path)


def _format_stage1_timing_map(timing_map):
    if not isinstance(timing_map, dict):
        return ""
    parts = []
    for key in ('front_route', 'route_total'):
        if key not in timing_map:
            continue
        value = timing_map[key]
        try:
            value_f = float(value)
        except Exception:
            continue
        parts.append(f"{key}={value_f:.2f}s")
    return " ".join(parts)


def precompute(
    dataset_path,
    image_data_root,
    anchor_path,
    bev_ppm=2.0,
    bev_size=256,
    force=False,
    front_corridor_margin_m=0.5,
    front_route_step_m=0.25,
    front_max_distance_m=40.0,
    front_safe_ttc_s=3.0,
    front_max_ttc_s=10.0,
    front_block_safe_distance_m=30.0,
    checkpoint_every_minutes=20.0,
    stage1_only=False,
    stage1_timing=False,
    stage1_timing_every=1,
):
    num_points = None

    # Load packed samples
    packed_path = _ensure_packed_samples(dataset_path)

    print(f"Loading {packed_path}...")
    load_start_time = time.perf_counter()
    with open(packed_path, 'rb') as f:
        samples = pickle.load(f)
    load_elapsed_s = float(time.perf_counter() - load_start_time)
    print(f"Loaded {len(samples)} samples")
    if stage1_timing:
        packed_size_gb = float(os.path.getsize(packed_path)) / 1e9 if os.path.exists(packed_path) else np.nan
        print(
            f"[stage1_timing] load_samples={load_elapsed_s:.2f}s "
            f"packed_size_gb={packed_size_gb:.2f}"
        )

    already_ego_status = sum(1 for s in samples if 'ego_status' in s)
    already_stage1_speed = sum(1 for s in samples if _has_stage1_speed_fields(s))
    num_points = _infer_num_future_points(samples)
    if stage1_only:
        print(
            f"Existing fields (stage1-only): conflict_area={already_stage1_speed}/{len(samples)}, "
            f"inferred_num_future={num_points}"
        )
        if already_stage1_speed == len(samples) and not force:
            print(f"All {len(samples)} samples already have stage1 conflict-area fields. Use --force to re-compute.")
            return
    else:
        complete_fast_fields = sum(1 for s in samples if _has_all_fast_fields(s))
        print(
            f"Existing fields: ego_status={already_ego_status}/{len(samples)}, "
            f"conflict_area={already_stage1_speed}/{len(samples)}, "
            f"complete={complete_fast_fields}/{len(samples)}"
        )
        if complete_fast_fields == len(samples) and not force:
            print(f"All {len(samples)} samples already have all fast fields. Use --force to re-compute.")
            return

    image_data_root = os.path.realpath(image_data_root)
    ego_status_built = 0
    stage1_speed_built = 0
    skipped = 0
    fallback = 0
    labeling_processed = 0
    stage1_processed = 0
    checkpoint_every_seconds = None
    if checkpoint_every_minutes is not None and float(checkpoint_every_minutes) > 0:
        checkpoint_every_seconds = float(checkpoint_every_minutes) * 60.0
    checkpoint_progress_path = packed_path + '.progress.json'
    last_checkpoint_time = time.time()
    dirty_since_checkpoint = False

    def _checkpoint_payload(phase, reason):
        return {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'phase': phase,
            'reason': reason,
            'dataset_path': os.path.realpath(dataset_path),
            'packed_path': packed_path,
            'num_samples': len(samples),
            'ego_status_built': ego_status_built,
            'stage1_speed_built': stage1_speed_built,
            'labeling_processed': labeling_processed,
            'stage1_processed': stage1_processed,
            'skipped': skipped,
            'fallback': fallback,
        }

    def _save_checkpoint(phase, reason):
        nonlocal last_checkpoint_time, dirty_since_checkpoint
        print(f"\n[checkpoint] {phase}: {reason} -> saving {packed_path} ...")
        _atomic_pickle_save(samples, packed_path)
        size_mb = os.path.getsize(packed_path) / 1e6
        payload = _checkpoint_payload(phase=phase, reason=reason)
        payload['packed_size_mb'] = round(size_mb, 3)
        _atomic_json_save(payload, checkpoint_progress_path)
        last_checkpoint_time = time.time()
        dirty_since_checkpoint = False
        print(f"[checkpoint] saved ({size_mb:.1f} MB)")

    def _maybe_checkpoint(phase, force_reason=None):
        if force_reason is not None:
            if dirty_since_checkpoint:
                _save_checkpoint(phase=phase, reason=force_reason)
            return
        if checkpoint_every_seconds is None or not dirty_since_checkpoint:
            return
        if (time.time() - last_checkpoint_time) < checkpoint_every_seconds:
            return
        _save_checkpoint(phase=phase, reason='periodic')

    try:
        if not stage1_only:
            for sample in tqdm(samples, desc="Ego Status"):
                labeling_processed += 1
                needs_ego_status = force or ('ego_status' not in sample)

                if not needs_ego_status:
                    skipped += 1
                    _maybe_checkpoint(phase='labeling')
                    continue

                dirty_since_checkpoint = True
                if needs_ego_status:
                    sample['ego_status'] = _build_ego_status(sample)
                    ego_status_built += 1

                _maybe_checkpoint(phase='labeling')

            _maybe_checkpoint(phase='labeling', force_reason='phase_complete')

        route_groups = defaultdict(list)
        for sample_idx, sample in enumerate(samples):
            if not force and _has_stage1_speed_fields(sample):
                continue
            base_dir, frame_str = _resolve_feature_frame_info(sample)
            if base_dir is None or frame_str is None:
                route_groups[None].append(sample_idx)
                continue
            route_groups[base_dir].append(sample_idx)

        total_stage1_routes = int(len(route_groups))
        stage1_timing_every = max(int(stage1_timing_every), 1)
        stage1_timing_totals = defaultdict(float)
        for route_group_idx, (base_dir, route_sample_indices) in enumerate(
            tqdm(route_groups.items(), desc="Stage1 conflict-area", leave=False),
            start=1,
        ):
            route_total_start = time.perf_counter()
            route_timing = defaultdict(float)
            route_fallback_count = 0
            if base_dir is None:
                for sample_idx in route_sample_indices:
                    stage1_processed += 1
                    dirty_since_checkpoint = True
                    _set_stage1_speed_fallback(samples[sample_idx])
                    stage1_speed_built += 1
                    fallback += 1
                    route_fallback_count += 1
                    _maybe_checkpoint(phase='stage1_speed')
                if stage1_timing and (route_group_idx % stage1_timing_every == 0):
                    route_total_s = float(time.perf_counter() - route_total_start)
                    print(
                        f"[stage1_timing] route={route_group_idx}/{total_stage1_routes} "
                        f"base_dir=<missing> frames={len(route_sample_indices)} "
                        f"total={route_total_s:.2f}s avg={route_total_s / max(len(route_sample_indices), 1):.3f}s/frame "
                        f"fallback={route_fallback_count}"
                    )
                continue

            route_sample_indices = sorted(route_sample_indices, key=lambda i: int(samples[i].get('frame_id', -1)))
            route_samples = [samples[i] for i in route_sample_indices]
            event_name = _scene_name_from_base_dir(base_dir)
            scene_nonstatic_actor_ids = _collect_scene_nonstatic_actor_ids(route_samples, image_data_root)
            scene_route_polyline_world, scene_route_anchor_s = _build_scene_route_polyline_world(route_samples, image_data_root)
            if _is_two_way_event_corridor_scene_context(event_name=event_name):
                if base_dir not in scene_route_polyline_world:
                    raise RuntimeError(f"Missing scene polyline for two-way route: {base_dir}")
            scene_borrow_context = None
            if _is_two_way_event_corridor_scene_context(event_name=event_name):
                borrow_frame_records = []
                for route_sample in route_samples:
                    base_dir_borrow, frame_str_borrow = _resolve_feature_frame_info(route_sample)
                    if base_dir_borrow is None or frame_str_borrow is None:
                        continue
                    boxes_borrow = _load_json_gz_if_exists(
                        os.path.join(image_data_root, base_dir_borrow, 'boxes', f'{frame_str_borrow}.json.gz')
                    )
                    meas_borrow = _load_json_gz_if_exists(
                        os.path.join(image_data_root, base_dir_borrow, 'measurements', f'{frame_str_borrow}.json.gz')
                    )
                    if boxes_borrow is None or meas_borrow is None:
                        continue
                    ego_matrix_borrow = meas_borrow.get('ego_matrix', None)
                    route_local_borrow = np.asarray(route_sample.get('route', np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
                    if (
                        ego_matrix_borrow is None or route_local_borrow.ndim != 2 or
                        route_local_borrow.shape[0] == 0 or route_local_borrow.shape[1] != 2
                    ):
                        continue
                    if event_name not in NO_ROUTE_EXTENSION_SCENES and base_dir_borrow in scene_route_polyline_world:
                        anchor_s_borrow = scene_route_anchor_s.get(base_dir_borrow, {}).get(int(route_sample.get('frame_id', -1)))
                        route_input_borrow = _extend_local_route_with_scene_polyline(
                            route_local_borrow,
                            ego_matrix_borrow,
                            scene_route_polyline_world[base_dir_borrow],
                            anchor_s=anchor_s_borrow,
                            extension_step_m=1.0,
                            extension_points=12,
                        )
                    else:
                        route_input_borrow = route_local_borrow.astype(np.float32)
                    borrow_frame_records.append({
                        'frame_id': int(route_sample.get('frame_id', -1)),
                        'current_meas': meas_borrow,
                        'current_boxes': boxes_borrow,
                        'route_input_local': route_input_borrow,
                        'route_front_count': int(route_local_borrow.shape[0]),
                    })
                if borrow_frame_records:
                    scene_borrow_context = _build_event_two_way_borrow_context(
                        borrow_frame_records,
                        event_name=event_name,
                        route_step_m=max(0.25, float(front_route_step_m)),
                    )

            for sample_idx in route_sample_indices:
                stage1_processed += 1
                dirty_since_checkpoint = True
                sample = samples[sample_idx]
                base_dir_cur, frame_str = _resolve_feature_frame_info(sample)
                if base_dir_cur is None or frame_str is None:
                    _set_stage1_speed_fallback(sample)
                    stage1_speed_built += 1
                    fallback += 1
                    route_fallback_count += 1
                    _maybe_checkpoint(phase='stage1_speed')
                    continue

                boxes_path = os.path.join(image_data_root, base_dir_cur, 'boxes', f'{frame_str}.json.gz')
                meas_path = os.path.join(image_data_root, base_dir_cur, 'measurements', f'{frame_str}.json.gz')
                current_boxes = _load_json_gz_if_exists(boxes_path)
                current_measurements = _load_json_gz_if_exists(meas_path)
                if current_boxes is None or current_measurements is None:
                    _set_stage1_speed_fallback(sample)
                    stage1_speed_built += 1
                    fallback += 1
                    route_fallback_count += 1
                    _maybe_checkpoint(phase='stage1_speed')
                    continue

                ego_matrix_current = current_measurements.get('ego_matrix', None)
                if ego_matrix_current is None:
                    _set_stage1_speed_fallback(sample)
                    stage1_speed_built += 1
                    fallback += 1
                    route_fallback_count += 1
                    _maybe_checkpoint(phase='stage1_speed')
                    continue

                route_local = np.asarray(sample.get('route', np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
                if route_local.ndim != 2 or route_local.shape[0] == 0 or route_local.shape[1] != 2:
                    _set_stage1_speed_fallback(sample)
                    stage1_speed_built += 1
                    fallback += 1
                    route_fallback_count += 1
                    _maybe_checkpoint(phase='stage1_speed')
                    continue
                if event_name not in NO_ROUTE_EXTENSION_SCENES and base_dir_cur in scene_route_polyline_world:
                    anchor_s = scene_route_anchor_s.get(base_dir_cur, {}).get(int(sample.get('frame_id', -1)))
                    route_input = _extend_local_route_with_scene_polyline(
                        route_local,
                        ego_matrix_current,
                        scene_route_polyline_world[base_dir_cur],
                        anchor_s=anchor_s,
                        extension_step_m=1.0,
                        extension_points=12,
                    )
                else:
                    route_input = route_local.astype(np.float32)
                sample['_stage1_route_input_local'] = np.asarray(route_input, dtype=np.float32)

                ego_speed = float(current_measurements.get('speed', 0.0))
                future_frames_data = _build_future_frames_data(image_data_root, base_dir_cur, frame_str, num_points)

                current_boxes_dynamic = _filter_current_boxes_dynamic(current_boxes, scene_nonstatic_actor_ids)
                future_frames_dynamic = _filter_future_frames_dynamic(future_frames_data, scene_nonstatic_actor_ids)
                timing_start = time.perf_counter()
                _, front_debug = _compute_front_route_label(
                    route=route_input,
                    current_boxes=current_boxes_dynamic,
                    ego_speed=ego_speed,
                    ego_matrix_current=ego_matrix_current,
                    future_frames_data=future_frames_dynamic,
                    corridor_margin_m=front_corridor_margin_m,
                    route_step_m=front_route_step_m,
                    max_distance_m=front_max_distance_m,
                    safe_ttc_s=front_safe_ttc_s,
                    max_ttc_s=front_max_ttc_s,
                    block_safe_distance_m=front_block_safe_distance_m,
                    return_debug=True,
                )
                route_timing['front_route'] += float(time.perf_counter() - timing_start)
                current_cover = _cover_candidate_summary(1, front_debug.get('best_current'), front_debug, current_meas=current_measurements, event_name=event_name)
                future_cover = _cover_candidate_summary(2, front_debug.get('best_future'), front_debug, current_meas=current_measurements, event_name=event_name)

                scene_route_world = scene_route_polyline_world.get(base_dir_cur)
                current_cover = _augment_cover_with_scene_route_fields(
                    current_cover,
                    ego_matrix_current=ego_matrix_current,
                    scene_route_polyline_world=scene_route_world,
                )
                future_cover = _augment_cover_with_scene_route_fields(
                    future_cover,
                    ego_matrix_current=ego_matrix_current,
                    scene_route_polyline_world=scene_route_world,
                )
                borrow_motion = _build_borrow_motion_context(
                    scene_borrow_context,
                    current_meas=current_measurements,
                    route_local=route_input,
                    frame_id=int(sample.get('frame_id', -1)),
                )
                sample['stage1_speed_debug'] = _build_stage1_speed_debug_payload(
                    current_cover=current_cover,
                    future_cover=future_cover,
                    merge_motion=_build_merge_motion_context(
                        current_meas=current_measurements,
                        route_dense=front_debug.get('route_dense'),
                        ego_matrix_current=ego_matrix_current,
                        current_boxes=current_boxes_dynamic,
                        scene_route_polyline_world=scene_route_world,
                    ),
                    scene_borrow_context=scene_borrow_context,
                    borrow_motion=borrow_motion,
                )
                stage1_speed_built += 1
                _maybe_checkpoint(phase='stage1_speed')

            _annotate_route_stage1_conflict_areas(
                samples,
                route_sample_indices,
                scene_route_world=scene_route_polyline_world.get(base_dir),
            )
            _annotate_route_stage1_conflict_phases(samples, route_sample_indices)
            for sample_idx in route_sample_indices:
                samples[int(sample_idx)].pop('_stage1_route_input_local', None)
            dirty_since_checkpoint = True
            _maybe_checkpoint(phase='stage1_speed')
            route_total_s = float(time.perf_counter() - route_total_start)
            route_timing['route_total'] += route_total_s
            for key, value in route_timing.items():
                stage1_timing_totals[key] += float(value)
            if stage1_timing and (route_group_idx % stage1_timing_every == 0):
                route_name = str(base_dir)
                print(
                    f"[stage1_timing] route={route_group_idx}/{total_stage1_routes} "
                    f"event={event_name} frames={len(route_sample_indices)} "
                    f"avg={route_total_s / max(len(route_sample_indices), 1):.3f}s/frame "
                    f"fallback={route_fallback_count} "
                    f"{_format_stage1_timing_map(dict(route_timing))} "
                    f"base_dir={route_name}"
                )

        _maybe_checkpoint(phase='stage1_speed', force_reason='phase_complete')
        if stage1_timing and total_stage1_routes > 0:
            print(
                f"[stage1_timing] totals routes={total_stage1_routes} "
                f"{_format_stage1_timing_map(dict(stage1_timing_totals))}"
            )
    except BaseException as exc:
        try:
            _maybe_checkpoint(phase='exception', force_reason=f'exception_{type(exc).__name__}')
        except Exception as checkpoint_exc:
            print(f"[checkpoint] failed during exception handling: {checkpoint_exc}")
        raise

    print(
        f"\nDone: ego_status={ego_status_built}, "
        f"conflict_area={stage1_speed_built}, "
        f"skipped={skipped}, fallback={fallback}"
    )

    _save_checkpoint(phase='final', reason='complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-compute lightweight debug fields into samples_packed.pkl")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset split (e.g. /media/z/data/dataset/pdm_lite_mini/train)')
    parser.add_argument('--image_data_root', type=str, required=True,
                        help='Root of image data (e.g. /media/z/data/dataset/pdm_lite_mini)')
    parser.add_argument('--anchor_path', type=str, default=None,
                        help='Compatibility no-op. Semantic anchor labeling is disabled in the current debug-focused script.')
    parser.add_argument('--bev_ppm', type=float, default=2.0,
                        help='Compatibility no-op. Kept to avoid breaking old commands.')
    parser.add_argument('--bev_size', type=int, default=256,
                        help='Compatibility no-op. Kept to avoid breaking old commands.')
    parser.add_argument('--front_corridor_margin_m', type=float, default=0.5)
    parser.add_argument('--front_route_step_m', type=float, default=0.25)
    parser.add_argument('--front_max_distance_m', type=float, default=40.0)
    parser.add_argument('--front_safe_ttc_s', type=float, default=3.0)
    parser.add_argument('--front_max_ttc_s', type=float, default=10.0)
    parser.add_argument('--front_block_safe_distance_m', type=float, default=30.0)
    parser.add_argument('--checkpoint_every_minutes', type=float, default=20.0,
                        help='Periodically atomically save updated samples_packed.pkl to make long runs resumable. Set <=0 to disable.')
    parser.add_argument('--stage1_only', action='store_true',
                        help='Skip ego_status refresh and only (re)compute stage1 conflict-area labels.')
    parser.add_argument('--stage1_timing', action='store_true',
                        help='Print lightweight stage1 timing summaries (load/front_route/route_total).')
    parser.add_argument('--stage1_timing_every', type=int, default=1,
                        help='When --stage1_timing is set, print every N route groups.')
    parser.add_argument('--force', action='store_true', help='Re-compute even if labels exist')
    args = parser.parse_args()

    precompute(args.dataset_path, args.image_data_root, args.anchor_path,
               bev_ppm=args.bev_ppm,
               bev_size=args.bev_size,
               force=args.force,
               front_corridor_margin_m=args.front_corridor_margin_m,
               front_route_step_m=args.front_route_step_m,
               front_max_distance_m=args.front_max_distance_m,
               front_safe_ttc_s=args.front_safe_ttc_s,
               front_max_ttc_s=args.front_max_ttc_s,
               front_block_safe_distance_m=args.front_block_safe_distance_m,
               checkpoint_every_minutes=args.checkpoint_every_minutes,
               stage1_only=args.stage1_only,
               stage1_timing=args.stage1_timing,
               stage1_timing_every=args.stage1_timing_every)
