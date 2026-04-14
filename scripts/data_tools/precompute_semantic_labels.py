#!/usr/bin/env python3
"""
Pre-compute training-time supervision/cache fields and inject them into samples_packed.pkl.

This avoids expensive on-the-fly BEV semantic + boxes + measurements IO during training,
and also moves cheap-but-frequent tensor assembly out of the dataloader.
Each sample gets these fast fields:
  - behavior_labels: (num_modes,) int64
  - allowed_flags:   (num_modes,) float32
  - scene_buckets:   (NUM_BUCKET_CATEGORIES,) float32
  - energy_targets:  (num_modes, 5) float32
  - energy_active_mask: (num_modes,) bool
  - ego_status:      (obs_horizon, 14) float32
  - front_route_distance: () float32, route-arc distance to first lead vehicle
  - front_route_ttc:      () float32, TTC under current-cover / future-meeting rule
  - front_route_risk:     () float32, continuous front risk from TTC/blocking
  - front_route_block_risk: () float32, blocking risk from current route occupancy
  - front_route_case:     () int64, 0=none, 1=current_cover, 2=future_cover
  - front_route_has_lead: () float32, 1 if any vehicle covers the current route corridor
  - front_route_actor_class: () int64, 0=none, 1=vehicle, 2=bicycle, 3=pedestrian
  - front_route_actor_weight: () float32, class-dependent training weight
  - front_route_block_bin: () int64, 0..4 blocking severity bin
  - front_route_ttc_bin: () int64, 0..4 TTC severity bin
  - front_route_hazard_bin: () int64, 0..4 overall hazard severity bin

Usage:
  python scripts/data_tools/precompute_semantic_labels.py \
    --dataset_path /media/z/data/dataset/pdm_lite_mini/train \
    --image_data_root /media/z/data/dataset/pdm_lite_mini \
    --anchor_path dd_baseline/anchors/carla_kmeans_32.npy

  # Also for val split:
  python scripts/data_tools/precompute_semantic_labels.py \
    --dataset_path /media/z/data/dataset/pdm_lite_mini/val \
    --image_data_root /media/z/data/dataset/pdm_lite_mini \
    --anchor_path dd_baseline/anchors/carla_kmeans_32.npy
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
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from tools.anchor_semantic_labeler import label_anchors_semantic, classify_scene_buckets, NUM_BUCKET_CATEGORIES


FAST_FIELDS = (
    'behavior_labels',
    'allowed_flags',
    'scene_buckets',
    'energy_targets',
    'energy_active_mask',
    'ego_status',
    'front_route_distance',
    'front_route_ttc',
    'front_route_risk',
    'front_route_block_risk',
    'front_route_case',
    'front_route_has_lead',
    'front_route_actor_class',
    'front_route_actor_weight',
    'front_route_block_bin',
    'front_route_ttc_bin',
    'front_route_hazard_bin',
    'speed_sample_values',
    'speed_sample_valid_mask',
    'speed_sample_exp_index',
    'speed_risk_chase_values',
    'speed_risk_meet_values',
    'speed_risk_junction_cross_yld_values',
    'speed_risk_junction_cross_go_values',
    'speed_risk_merge_yld_values',
    'speed_risk_merge_go_values',
    'speed_risk_borrow_yld_values',
    'speed_risk_borrow_go_values',
    'speed_risk_ped_values',
    'speed_cross_wait_time_s',
    'speed_cross_wait_valid',
    'junction_cross_episode_id',
    'junction_cross_episode_active',
    'junction_cross_episode_start_frame',
    'junction_cross_episode_end_frame',
    'borrow_cross_decision_phase',
    'borrow_cross_episode_id',
    'borrow_cross_episode_active',
    'borrow_cross_active_time_s',
    'borrow_cross_episode_start_frame',
    'borrow_cross_episode_end_frame',
    'borrow_cross_go_frame',
    'borrow_cross_context_frame',
    'merge_decision_phase',
    'merge_episode_id',
    'merge_episode_active',
    'merge_episode_no_go',
    'merge_episode_start_frame',
    'merge_episode_end_frame',
    'merge_go_frame',
    'merge_resolution_actor_id',
    'merge_end_state',
    'merge_hold',
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
MERGE_DEBUG_MIN_DEGO_M = 1.0
MERGE_DEBUG_MIN_GO_DENOM_S = 0.10
NO_ROUTE_EXTENSION_SCENES = {'HazardAtSideLane'}
TWOWAY_START_LATERAL_THRESH_M = 0.5
TWOWAY_RETURN_TAIL_POINTS = 12
TWOWAY_RETURN_FALLBACK_EXTRA_POINT_INDEX = 10
STAGE1_SPEED_OFFSETS_MPS = np.asarray([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0], dtype=np.float32)
STAGE1_MERGE_GRACE_FRAMES = 4
STAGE1_MERGE_START_CONFIRM_FRAMES = 2
STAGE1_MERGE_RED_LIGHT_WAIT_SPEED_THRESH = 0.5
STAGE1_MERGE_ACTOR_CONTINUITY_GRACE_FRAMES = 6
STAGE1_MERGE_RESOLUTION_LOOKAHEAD_FRAMES = 6
STAGE1_GO_END_STEER_ABS_THRESH = 0.08
STAGE1_GO_END_HEADING_ALIGN_THRESH_DEG = 12.0
STAGE1_GO_END_CONFIRM_FRAMES = 3
STAGE1_MERGE_SPEED_CAP_MPS = 1000.0
STAGE1_MERGE_CONFLICT_LOOKAHEAD_PROGRESS_M = 15.0
STAGE1_MERGE_CONFLICT_CLUSTER_GAP_M = 4.0
STAGE1_MERGE_AREA_POST_MARGIN_M = 3.0
STAGE1_JUNCTION_CROSS_MIN_CLUSTER_POINTS = 2
STAGE1_JUNCTION_CROSS_FALLBACK_RADIUS_M = 7.5
STAGE1_PERSISTED_MEET_FRAMES = 3
STAGE1_BORROW_ACTIVE_DT_S = 0.25
STAGE1_CROSS_WAIT_DT_S = 0.25
STAGE1_CROSS_GO_SPEED_THRESH = 1.0
STAGE1_CROSS_WAIT_SPEED_THRESH = 0.5
STAGE1_CROSS_START_DISTANCE_M = 10.0

MERGE_DECISION_PHASE_TO_CODE = {
    'none': 0,
    'yld': 1,
    'go': 2,
}

MERGE_END_STATE_TO_CODE = {
    'none': 0,
    'ended_with_chase': 1,
    'ended_with_cross': 2,
    'ended_with_other_current_actor': 3,
    'ended_empty': 4,
    'ended_route_end': 5,
    'ended_merge_area': 6,
}

STAGE1_SPEED_FIELDS = (
    'speed_sample_values',
    'speed_sample_valid_mask',
    'speed_sample_exp_index',
    'speed_risk_chase_values',
    'speed_risk_meet_values',
    'speed_risk_junction_cross_yld_values',
    'speed_risk_junction_cross_go_values',
    'speed_risk_merge_yld_values',
    'speed_risk_merge_go_values',
    'speed_risk_borrow_yld_values',
    'speed_risk_borrow_go_values',
    'speed_risk_ped_values',
    'speed_cross_wait_time_s',
    'speed_cross_wait_valid',
    'junction_cross_episode_id',
    'junction_cross_episode_active',
    'junction_cross_episode_start_frame',
    'junction_cross_episode_end_frame',
    'borrow_cross_decision_phase',
    'borrow_cross_episode_id',
    'borrow_cross_episode_active',
    'borrow_cross_active_time_s',
    'borrow_cross_episode_start_frame',
    'borrow_cross_episode_end_frame',
    'borrow_cross_go_frame',
    'borrow_cross_context_frame',
    'merge_decision_phase',
    'merge_episode_id',
    'merge_episode_active',
    'merge_episode_no_go',
    'merge_episode_start_frame',
    'merge_episode_end_frame',
    'merge_go_frame',
    'merge_resolution_actor_id',
    'merge_end_state',
    'merge_hold',
)

CROSS_DECISION_PHASE_TO_CODE = {
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


def _set_semantic_fallback(sample, num_modes):
    sample['behavior_labels'] = np.zeros(num_modes, dtype=np.int64)
    sample['allowed_flags'] = np.ones(num_modes, dtype=np.float32)
    sample['scene_buckets'] = np.zeros(NUM_BUCKET_CATEGORIES, dtype=np.float32)


def _build_energy_targets(behavior_labels, allowed_flags):
    behavior_labels = np.asarray(behavior_labels, dtype=np.int64)
    allowed_flags = np.asarray(allowed_flags, dtype=np.float32)
    energy_targets = np.stack([
        (behavior_labels == 1).astype(np.float32),
        (behavior_labels == 2).astype(np.float32),
        (behavior_labels == 3).astype(np.float32),
        (behavior_labels == 4).astype(np.float32),
        ((behavior_labels >= 5) & (behavior_labels <= 6)).astype(np.float32),
    ], axis=-1)
    energy_active_mask = (allowed_flags < 0.5).astype(np.bool_)
    return energy_targets.astype(np.float32), energy_active_mask


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


def _set_front_route_fallback(sample, max_distance_m, max_ttc_s):
    sample['front_route_distance'] = np.float32(max_distance_m)
    sample['front_route_ttc'] = np.float32(max_ttc_s)
    sample['front_route_risk'] = np.float32(0.0)
    sample['front_route_block_risk'] = np.float32(0.0)
    sample['front_route_case'] = np.int64(0)  # 0=none, 1=current_cover, 2=future_cover
    sample['front_route_has_lead'] = np.float32(0.0)
    sample['front_route_actor_class'] = np.int64(ACTOR_CLASS_NONE)
    sample['front_route_actor_weight'] = np.float32(_actor_weight(ACTOR_CLASS_NONE))
    sample['front_route_block_bin'] = np.int64(0)
    sample['front_route_ttc_bin'] = np.int64(0)
    sample['front_route_hazard_bin'] = np.int64(0)


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
    return str(event_name or "") in {"ConstructionObstacleTwoWays", "AccidentTwoWays"}


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
        future_box = best.get("box_current_frame")
        if current_box is not None and future_box is not None:
            actor_heading, motion_m = _heading_from_motion(
                np.asarray(current_box.get("position", [0.0, 0.0])[:2], dtype=np.float32),
                np.asarray(future_box.get("position", [0.0, 0.0])[:2], dtype=np.float32),
                min_motion_m=min_motion_m,
            )
            if actor_heading is not None:
                source = "motion_vs_route"
        if actor_heading is None and future_box is not None:
            actor_heading = float(future_box.get("yaw", 0.0))
            source = "future_yaw_vs_route"
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
    return _attach_event_name_record({
        "mode": 2,
        "name": "meet",
        "subtype": subtype,
        "source": source,
        "angle_deg": float(angle_deg),
        "route_heading_deg": _heading_to_deg(route_heading),
        "actor_heading_deg": _heading_to_deg(actor_heading),
        "motion_m": float(motion_m),
    }, event_name=event_name)


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
            "scene_route_conflict_world_xy": [],
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
            "scene_route_conflict_world_xy": [],
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
        "scene_route_conflict_world_xy": [],
    }


def _augment_cover_with_scene_route_fields(cover, ego_matrix_current, scene_route_polyline_world):
    cover_out = dict(cover or {})
    cover_out.setdefault("scene_route_conflict_s_m", np.nan)
    cover_out.setdefault("scene_route_conflict_world_xy", [])
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
    proj_world, proj_s = _project_point_to_polyline(route_point_world[0, :2], scene_route_polyline_world[:, :2])
    cover_out["scene_route_conflict_world_xy"] = route_point_world[0, :2].astype(float).tolist()
    cover_out["scene_route_conflict_s_m"] = float(proj_s) if proj_s is not None else np.nan
    if proj_world is not None and not cover_out["scene_route_conflict_world_xy"]:
        cover_out["scene_route_conflict_world_xy"] = np.asarray(proj_world, dtype=np.float32).astype(float).tolist()
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


def _speed_risk_samples_fixed(speed_mps, max_speed_mps=20.0):
    speed_mps = float(max(speed_mps, 0.0))
    return np.clip(speed_mps + STAGE1_SPEED_OFFSETS_MPS, 0.0, float(max_speed_mps)).astype(np.float32)


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
        "borrow_start_world_xy": [],
        "borrow_end_world_xy": [],
        "borrow_start_world_xyz": [],
        "borrow_end_world_xyz": [],
        "borrow_segment_world_xy": [],
        "borrow_segment_world_xyz": [],
        "borrow_distance_m": np.nan,
        "peak_lateral_m": np.nan,
        "peak_signed_lateral_m": np.nan,
        "shift_sign": np.nan,
        "context_frame_id": int(best_candidate.get("frame_id", -1)),
        "anchor_actor_id": int(seed_actor.get("actor_id", best_candidate.get("actor_id", -1))),
        "anchor_distance_m": float(best_candidate.get("local_x", np.nan)),
        "anchor_world_xy": list(best_candidate.get("world_xy", seed_actor.get("seed_world_xy", []))),
        "blocking_actor_id": int(seed_actor.get("actor_id", best_candidate.get("actor_id", -1))),
        "blocking_actor_class": str(seed_actor.get("actor_class", best_candidate.get("actor_class", "none"))),
        "blocking_actor_local_x_m": float(best_candidate.get("local_x", np.nan)),
        "blocking_actor_local_y_m": float(best_candidate.get("local_y", np.nan)),
        "seed_frame_id": int(seed_actor.get("seed_frame_id", -1)),
        "seed_priority": int(seed_actor.get("seed_priority", -1)),
        "route_return_abs_m": np.nan,
        "blocked_frame_id": -1,
    }


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

    seed_actor = None
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
        seed_world_xy = _box_world_xy(seed_box, ego_matrix_current=record.get("current_meas", {}).get("ego_matrix", None))
        seed_actor = {
            "actor_id": int(actor_id),
            "actor_class": str(_box_class_name(seed_box)),
            "seed_frame_id": int(route_candidate["frame_id"]),
            "seed_priority": int(route_candidate["priority"]),
            "seed_world_xy": [] if seed_world_xy is None else np.asarray(seed_world_xy, dtype=np.float32).astype(float).tolist(),
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
            world_xy = _box_world_xy(box, ego_matrix_current=ego_matrix_current)
            candidate = {
                "score": (float(local_x), float(abs(local_y)), -int(route_candidate["frame_id"])),
                "record_idx": int(route_candidate["record_idx"]),
                "frame_id": int(route_candidate["frame_id"]),
                "actor_id": int(seed_actor["actor_id"]),
                "actor_class": str(seed_actor["actor_class"]),
                "local_x": float(local_x),
                "local_y": float(local_y),
                "world_xy": [] if world_xy is None else np.asarray(world_xy, dtype=np.float32).astype(float).tolist(),
                "geom": dict(route_candidate["geom"]),
            }
            if int(route_candidate["priority"]) == 0:
                if best_strict is None or candidate["score"] < best_strict["score"]:
                    best_strict = candidate
            else:
                if best_fallback is None or candidate["score"] < best_fallback["score"]:
                    best_fallback = candidate

    best_candidate = best_strict if best_strict is not None else best_fallback
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
    borrow_world = np.stack([borrow_segment_world[0, :2], borrow_segment_world[-1, :2]], axis=0)
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
        "borrow_start_world_xy": borrow_world[0, :2].astype(float).tolist(),
        "borrow_end_world_xy": borrow_world[1, :2].astype(float).tolist(),
        "borrow_start_world_xyz": borrow_world_xyz[0, :3].astype(float).tolist(),
        "borrow_end_world_xyz": borrow_world_xyz[1, :3].astype(float).tolist(),
        "borrow_segment_world_xy": borrow_segment_world[:, :2].astype(float).tolist(),
        "borrow_segment_world_xyz": borrow_segment_world[:, :3].astype(float).tolist(),
        "borrow_distance_m": float(borrow_geom["borrow_distance_m"]),
        "peak_lateral_m": float(borrow_geom["peak_lateral_m"]),
        "peak_signed_lateral_m": float(borrow_geom.get("peak_signed_lateral_m", np.nan)),
        "shift_sign": float(borrow_geom.get("shift_sign", np.nan)),
        "context_frame_id": int(best_candidate["frame_id"]),
        "anchor_actor_id": int(best_candidate["actor_id"]),
        "anchor_distance_m": float(best_candidate["local_x"]),
        "anchor_world_xy": list(best_candidate["world_xy"]),
        "blocking_actor_id": int(best_candidate["actor_id"]),
        "blocking_actor_class": str(best_candidate["actor_class"]),
        "blocking_actor_local_x_m": float(best_candidate["local_x"]),
        "blocking_actor_local_y_m": float(best_candidate["local_y"]),
        "seed_frame_id": int(seed_actor.get("seed_frame_id", -1)),
        "seed_priority": int(seed_actor.get("seed_priority", -1)),
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




def _build_speed_curve_debug(
    current_cover,
    future_cover,
    current_meas,
    current_boxes=None,
    event_name=None,
    release_info=None,
    route_local=None,
    safe_ttc_s=3.0,
    merge_tau_s=0.25,
    merge_clearance_m=6.0,
    merge_follow_base_gap_m=4.0,
    merge_follow_headway_s=0.6,
    merge_band_smooth_mps=1.0,
    chase_follow_base_gap_m=3.0,
    chase_follow_headway_s=0.5,
    rear_hard_ttc_s=1.0,
    rear_safe_ttc_s=3.0,
    ped_hard_ttc_s=1.0,
    ped_safe_ttc_s=3.0,
    junction_left_conflict_len_scale=1.5,
    cross_safe_gap_s=1.0,
    sample_speeds_override=None,
):
    speed = float((current_meas or {}).get("speed", 0.0))
    if sample_speeds_override is None:
        sample_speeds = _speed_risk_samples_fixed(speed)
    else:
        sample_speeds = np.asarray(sample_speeds_override, dtype=np.float32)
        if sample_speeds.ndim != 1:
            sample_speeds = sample_speeds.reshape(-1).astype(np.float32)
    chase_risks = np.zeros(sample_speeds.shape, dtype=np.float32)
    meet_risks = np.zeros(sample_speeds.shape, dtype=np.float32)
    junction_cross_yld_risks = np.zeros(sample_speeds.shape, dtype=np.float32)
    junction_cross_go_risks = np.zeros(sample_speeds.shape, dtype=np.float32)
    merge_yld_risks = np.zeros(sample_speeds.shape, dtype=np.float32)
    merge_go_risks = np.zeros(sample_speeds.shape, dtype=np.float32)
    borrow_yld_risks = np.zeros(sample_speeds.shape, dtype=np.float32)
    borrow_go_risks = np.zeros(sample_speeds.shape, dtype=np.float32)
    ego_length_m = _ego_length_m(current_boxes)
    left_junction_conflict_len_m = _left_junction_conflict_len_m(
        current_meas,
        current_boxes,
        scale=junction_left_conflict_len_scale,
        event_name=event_name,
    )
    borrow_cross_conflict_len_m = np.nan
    if _is_borrow_cross_scene_context(event_name=event_name):
        borrow_distance_m = np.nan if release_info is None else float(release_info.get("borrow_distance_m", np.nan))
        if np.isfinite(borrow_distance_m) and borrow_distance_m > 1e-3:
            borrow_cross_conflict_len_m = float(max(borrow_distance_m, ego_length_m))
        else:
            borrow_cross_conflict_len_m = float(max(3.0 * ego_length_m, ego_length_m))
    borrow_corridor = _borrow_corridor_metrics(
        release_info,
        current_meas=current_meas,
        route_local=route_local,
    )

    chase_info = {
        "valid": 0.0,
        "gap_m": np.nan,
        "lead_speed_mps": np.nan,
        "safe_gap_cur_m": np.nan,
    }
    if int(current_cover.get("exists", 0.0)) > 0 and current_cover["interaction"]["name"] == "chase":
        gap_m = float(current_cover.get("distance", np.nan))
        lead_speed_mps = float(current_cover.get("other_speed", np.nan))
        if np.isfinite(gap_m) and np.isfinite(lead_speed_mps):
            chase_info = {
                "valid": 1.0,
                "gap_m": gap_m,
                "lead_speed_mps": lead_speed_mps,
                "safe_gap_cur_m": float(chase_follow_base_gap_m + chase_follow_headway_s * max(speed, 0.0)),
            }
            for idx, candidate_speed in enumerate(sample_speeds):
                closing = float(candidate_speed) - lead_speed_mps
                ttc = np.inf if closing <= 1e-6 else gap_m / max(closing, 1e-6)
                safe_gap = float(chase_follow_base_gap_m + chase_follow_headway_s * max(float(candidate_speed), 0.0))
                gap_risk = float(np.clip((safe_gap - gap_m) / max(safe_gap, 1e-6), 0.0, 1.0))
                ttc_risk = _risk_from_time_gap(ttc, safe_ttc_s)
                if closing > 1e-6:
                    closing_speed_risk = float(np.clip(closing / max(float(candidate_speed), 1.0), 0.0, 1.0))
                    distance_weight = float(safe_gap / max(gap_m + safe_gap, 1e-6))
                    closing_trend_risk = float(np.clip(closing_speed_risk * distance_weight, 0.0, 1.0))
                else:
                    closing_trend_risk = 0.0
                chase_risks[idx] = max(gap_risk, ttc_risk, closing_trend_risk)

    meet_info = {
        "valid": 0.0,
        "d_ego_m": np.nan,
        "d_bg_m": np.nan,
        "conflict_len_m": np.nan,
        "context_conflict_len_m": float(left_junction_conflict_len_m),
        "borrow_conflict_len_m": float(borrow_cross_conflict_len_m),
        "borrow_start_distance_m": np.nan,
        "borrow_total_distance_m": np.nan,
        "d_bg_to_end_m": np.nan,
        "t_bg_to_end_s": np.nan,
        "ego_clearance_m": np.nan,
        "bg_clearance_m": np.nan,
        "bg_speed_mps": np.nan,
        "t_bg_s": np.nan,
        "t_bg_exit_s": np.nan,
        "t_current_bg_exit_s": np.nan,
        "t_bg_clear_s": np.nan,
        "t_ego_exit_s": np.nan,
        "safe_gap_bg_m": np.nan,
        "rear_gap_m": np.nan,
        "v_equal_mps": np.nan,
        "v_go_min_mps": np.nan,
        "v_go_cap_mps": np.nan,
        "v_behind_min_mps": np.nan,
        "v_go_need_mps": np.nan,
        "v_yield_max_mps": np.nan,
    }

    def _cross_wait_from_meet_debug(info):
        subtype = str(info.get("subtype", "none"))
        if "cross" not in subtype:
            return 0.0, 0.0
        t_exit = float(info.get("t_bg_exit_s", np.nan))
        t_bg = float(info.get("t_bg_s", np.nan))
        t_wait = t_exit if np.isfinite(t_exit) else t_bg
        if not np.isfinite(t_wait):
            return 0.0, 0.0
        return float(max(t_wait, 0.0)), 1.0

    ped_cover = None
    if int(current_cover.get("exists", 0.0)) > 0 and str(current_cover.get("actor_class_name", "")) in PEDESTRIAN_CLASSES:
        ped_cover = {
            "case": "current",
            "distance_m": float(current_cover.get("distance", np.nan)),
            "actor_id": int(current_cover.get("actor_id", -1)),
        }
    elif int(future_cover.get("exists", 0.0)) > 0 and str(future_cover.get("actor_class_name", "")) in PEDESTRIAN_CLASSES:
        d_ego = float(future_cover.get("d_ego", np.nan))
        if not np.isfinite(d_ego):
            d_ego = float(future_cover.get("distance", np.nan))
        ped_cover = {
            "case": "future",
            "distance_m": float(d_ego),
            "actor_id": int(future_cover.get("actor_id", -1)),
        }
    if ped_cover is not None and np.isfinite(float(ped_cover["distance_m"])):
        ped_distance = max(float(ped_cover["distance_m"]), 0.0)
        meet_info = {
            "valid": 1.0,
            "d_ego_m": float(ped_distance),
            "d_bg_m": np.nan,
            "conflict_len_m": np.nan,
            "context_conflict_len_m": float(left_junction_conflict_len_m),
            "borrow_conflict_len_m": float(borrow_cross_conflict_len_m),
            "bg_speed_mps": np.nan,
            "t_bg_s": np.nan,
            "t_bg_exit_s": np.nan,
            "t_bg_clear_s": np.nan,
            "safe_gap_bg_m": np.nan,
            "rear_gap_m": np.nan,
            "v_equal_mps": np.nan,
            "v_go_min_mps": np.nan,
            "v_behind_min_mps": np.nan,
            "v_go_need_mps": np.nan,
            "v_yield_max_mps": np.nan,
            "subtype": "ped_cross",
            "ped_case": str(ped_cover["case"]),
            "ped_actor_id": int(ped_cover["actor_id"]),
        }
        for idx, candidate_speed in enumerate(sample_speeds):
            v = float(candidate_speed)
            if v <= 1e-6:
                meet_risks[idx] = 0.0
                continue
            t_cover = ped_distance / max(v, 1e-6)
            risk = float(
                np.clip(
                    (float(ped_safe_ttc_s) - t_cover) /
                    max(float(ped_safe_ttc_s) - float(ped_hard_ttc_s), 1e-6),
                    0.0,
                    1.0,
                )
            )
            meet_risks[idx] = risk
        total_risks = np.maximum(chase_risks, meet_risks)
        cross_wait_time_s, cross_wait_valid = _cross_wait_from_meet_debug(meet_info)
        return {
            "sample_speeds_mps": sample_speeds.astype(np.float32),
            "chase_risks": chase_risks.astype(np.float32),
            "meet_risks": meet_risks.astype(np.float32),
            "junction_cross_yld_risks": junction_cross_yld_risks.astype(np.float32),
            "junction_cross_go_risks": junction_cross_go_risks.astype(np.float32),
            "merge_yld_risks": merge_yld_risks.astype(np.float32),
            "merge_go_risks": merge_go_risks.astype(np.float32),
            "borrow_yld_risks": borrow_yld_risks.astype(np.float32),
            "borrow_go_risks": borrow_go_risks.astype(np.float32),
            "total_risks": total_risks.astype(np.float32),
            "cross_wait_time_s": float(cross_wait_time_s),
            "cross_wait_valid": float(cross_wait_valid),
            "chase": chase_info,
            "meet": meet_info,
        }

    if int(current_cover.get("exists", 0.0)) > 0 and current_cover["interaction"]["name"] == "meet":
        d_ego = float(current_cover.get("distance", np.nan))
        bg_speed = float(current_cover.get("other_speed", np.nan))
        meet_subtype = str(current_cover.get("interaction", {}).get("subtype", "none"))
        if np.isfinite(d_ego):
            if meet_subtype == "borrow_cross_meet" and borrow_corridor is not None:
                bg_length_m = float(current_cover.get("other_length_m", np.nan))
                if not np.isfinite(bg_length_m):
                    bg_length_m = float(ego_length_m)
                bg_clearance_m = float(max(bg_length_m, 1.0))
                start_distance_m = float(borrow_corridor["borrow_start_distance_m"])
                borrow_distance_m = float(borrow_corridor["borrow_distance_m"])
                distance_to_clear_start_m = float(max(float(d_ego) - start_distance_m, 0.0) + bg_clearance_m)
                t_bg_exit = np.inf if not np.isfinite(bg_speed) or bg_speed <= 1e-6 else distance_to_clear_start_m / max(bg_speed, 1e-6)
                v_yield_max = np.nan
                if np.isfinite(t_bg_exit):
                    denom = float(t_bg_exit) + float(cross_safe_gap_s)
                    v_yield_max = np.inf if denom <= 1e-6 else float(start_distance_m) / max(denom, 1e-6)
                v_go_min = np.inf
                meet_info = {
                    "valid": 1.0,
                    "d_ego_m": float(d_ego),
                    "d_bg_m": 0.0,
                    "conflict_len_m": float(borrow_distance_m),
                    "context_conflict_len_m": float(left_junction_conflict_len_m),
                    "borrow_conflict_len_m": float(borrow_distance_m),
                    "borrow_start_distance_m": float(start_distance_m),
                    "borrow_total_distance_m": float(borrow_corridor["borrow_total_clear_distance_m"]),
                    "d_bg_to_end_m": np.nan,
                    "ego_clearance_m": float(ego_length_m),
                    "bg_clearance_m": float(bg_clearance_m),
                    "bg_speed_mps": float(bg_speed),
                    "t_bg_s": 0.0,
                    "t_bg_exit_s": float(t_bg_exit),
                    "t_bg_to_end_s": np.nan,
                    "t_bg_clear_s": np.nan,
                    "safe_gap_bg_m": np.nan,
                    "rear_gap_m": np.nan,
                    "v_equal_mps": np.nan,
                    "v_go_min_mps": float(v_go_min),
                    "v_behind_min_mps": np.nan,
                    "v_go_need_mps": np.nan,
                    "v_yield_max_mps": float(v_yield_max),
                    "subtype": meet_subtype,
                    "cover_case": "current",
                }
                for idx, candidate_speed in enumerate(sample_speeds):
                    v = float(candidate_speed)
                    if v <= 1e-6:
                        borrow_yld_risks[idx] = 0.0
                        borrow_go_risks[idx] = 1.0
                        meet_risks[idx] = 0.0
                        continue
                    if np.isfinite(v_yield_max):
                        yld_risk = 0.0 if v <= float(v_yield_max) else 1.0
                    elif not np.isfinite(t_bg_exit):
                        yld_risk = 1.0
                    else:
                        t_ego_hit = float(d_ego) / max(v, 1e-6)
                        yld_risk = _risk_from_time_gap(t_ego_hit - float(t_bg_exit), cross_safe_gap_s)
                    borrow_yld_risks[idx] = float(np.clip(yld_risk, 0.0, 1.0))
                    borrow_go_risks[idx] = 1.0
                    meet_risks[idx] = float(np.clip(min(borrow_yld_risks[idx], borrow_go_risks[idx]), 0.0, 1.0))
                total_risks = np.maximum(chase_risks, meet_risks)
                cross_wait_time_s, cross_wait_valid = _cross_wait_from_meet_debug(meet_info)
                return {
                    "sample_speeds_mps": sample_speeds.astype(np.float32),
                    "chase_risks": chase_risks.astype(np.float32),
                    "meet_risks": meet_risks.astype(np.float32),
                    "junction_cross_yld_risks": junction_cross_yld_risks.astype(np.float32),
                    "junction_cross_go_risks": junction_cross_go_risks.astype(np.float32),
                    "merge_yld_risks": merge_yld_risks.astype(np.float32),
                    "merge_go_risks": merge_go_risks.astype(np.float32),
                    "borrow_yld_risks": borrow_yld_risks.astype(np.float32),
                    "borrow_go_risks": borrow_go_risks.astype(np.float32),
                    "total_risks": total_risks.astype(np.float32),
                    "cross_wait_time_s": float(cross_wait_time_s),
                    "cross_wait_valid": float(cross_wait_valid),
                    "chase": chase_info,
                    "meet": meet_info,
                }
            conflict_len_m = np.nan
            if meet_subtype == "junction_left_cross_meet":
                conflict_len_m = float(left_junction_conflict_len_m)
            elif "cross" in meet_subtype and meet_subtype != "borrow_cross_meet":
                conflict_len_m = float(max(ego_length_m, 1.0))
            bg_length_m = float(current_cover.get("other_length_m", np.nan))
            if not np.isfinite(bg_length_m):
                bg_length_m = float(ego_length_m)
            bg_clearance_m = float(max(bg_length_m, 1.0))
            d_ego_entry_m = np.nan
            if not np.isfinite(d_ego_entry_m):
                if np.isfinite(conflict_len_m):
                    d_ego_entry_m = float(max(float(d_ego) - 0.5 * float(conflict_len_m), 0.0))
                else:
                    d_ego_entry_m = float(max(float(d_ego), 0.0))
            t_bg = 0.0
            current_bg_exit_s = np.nan
            junction_go_cap_mps = np.nan
            if meet_subtype == "junction_left_cross_meet" and np.isfinite(conflict_len_m):
                current_bg_occ_len_m = float(max(0.5 * float(conflict_len_m), 0.0) + bg_clearance_m)
                if np.isfinite(bg_speed) and bg_speed > 1e-3:
                    current_bg_exit_s = float(current_bg_occ_len_m / max(bg_speed, 1e-6))
                else:
                    current_bg_exit_s = np.inf
                denom = float(current_bg_exit_s) + float(cross_safe_gap_s)
                if np.isfinite(denom):
                    junction_go_cap_mps = np.inf if denom <= 1e-6 else float(d_ego_entry_m) / max(denom, 1e-6)
                t_bg_exit = float(current_bg_exit_s)
            elif np.isfinite(bg_speed) and bg_speed > 1e-3 and np.isfinite(conflict_len_m):
                t_bg_exit = float(conflict_len_m / max(bg_speed, 1e-6))
            elif np.isfinite(conflict_len_m):
                t_bg_exit = np.inf
            else:
                t_bg_exit = np.nan
            meet_info = {
                "valid": 1.0,
                "d_ego_m": float(d_ego_entry_m),
                "d_bg_m": 0.0,
                "conflict_len_m": float(conflict_len_m),
                "context_conflict_len_m": float(left_junction_conflict_len_m),
                "borrow_conflict_len_m": float(borrow_cross_conflict_len_m),
                "ego_clearance_m": float(ego_length_m) if meet_subtype == "junction_left_cross_meet" else np.nan,
                "bg_clearance_m": float(bg_clearance_m) if meet_subtype == "junction_left_cross_meet" else np.nan,
                "bg_speed_mps": float(bg_speed),
                "t_bg_s": float(t_bg),
                "t_bg_exit_s": float(t_bg_exit),
                "t_current_bg_exit_s": float(current_bg_exit_s),
                "t_bg_clear_s": np.nan,
                "safe_gap_bg_m": np.nan,
                "rear_gap_m": np.nan,
                "v_equal_mps": np.nan,
                "v_go_min_mps": 0.0 if meet_subtype == "junction_left_cross_meet" else np.nan,
                "v_go_cap_mps": float(junction_go_cap_mps),
                "v_behind_min_mps": np.nan,
                "v_go_need_mps": np.nan,
                "v_yield_max_mps": float(junction_go_cap_mps) if meet_subtype == "junction_left_cross_meet" else np.nan,
                "subtype": meet_subtype,
                "cover_case": "current",
            }
            if np.isfinite(conflict_len_m):
                current_cover_dist_horizon_m = float(max(15.0, 2.0 * float(conflict_len_m)))
            else:
                current_cover_dist_horizon_m = 15.0
            occupancy_floor = float(
                np.clip(
                    (current_cover_dist_horizon_m - float(d_ego_entry_m)) /
                    max(current_cover_dist_horizon_m, 1e-6),
                    0.02,
                    1.0,
                )
            )
            current_cover_time_horizon_s = float(max(float(cross_safe_gap_s), 3.0))
            for idx, candidate_speed in enumerate(sample_speeds):
                v = float(candidate_speed)
                if v <= 1e-6:
                    meet_risks[idx] = 0.0
                    if meet_subtype == "junction_left_cross_meet":
                        junction_cross_yld_risks[idx] = 0.0
                        junction_cross_go_risks[idx] = 1.0
                    continue
                t_ego_in = float(max(d_ego_entry_m, 0.0) / max(v, 1e-6))
                if float(d_ego_entry_m) <= 1e-4:
                    risk = 1.0
                else:
                    risk = float(
                        np.clip(
                            (current_cover_time_horizon_s - t_ego_in) / max(current_cover_time_horizon_s, 1e-6),
                            0.0,
                            1.0,
                        )
                    )
                if meet_subtype == "junction_left_cross_meet":
                    current_cap_risk = _risk_from_time_gap(t_ego_in - float(t_bg_exit), cross_safe_gap_s)
                    junction_cross_yld_risks[idx] = float(
                        np.clip(np.nan_to_num(current_cap_risk, nan=1.0, posinf=0.0, neginf=1.0), 0.0, 1.0)
                    )
                    junction_cross_go_risks[idx] = float(
                        np.clip(np.nan_to_num(current_cap_risk, nan=1.0, posinf=0.0, neginf=1.0), 0.0, 1.0)
                    )
                    meet_risks[idx] = float(
                        max(
                            float(np.clip(np.nan_to_num(current_cap_risk, nan=1.0, posinf=0.0, neginf=1.0), 0.0, 1.0)),
                            occupancy_floor,
                        )
                    )
                else:
                    meet_risks[idx] = float(max(risk, occupancy_floor))
            total_risks = np.maximum(chase_risks, meet_risks)
            cross_wait_time_s, cross_wait_valid = _cross_wait_from_meet_debug(meet_info)
            return {
                "sample_speeds_mps": sample_speeds.astype(np.float32),
                "chase_risks": chase_risks.astype(np.float32),
                "meet_risks": meet_risks.astype(np.float32),
                "junction_cross_yld_risks": junction_cross_yld_risks.astype(np.float32),
                "junction_cross_go_risks": junction_cross_go_risks.astype(np.float32),
                "merge_yld_risks": merge_yld_risks.astype(np.float32),
                "merge_go_risks": merge_go_risks.astype(np.float32),
                "borrow_yld_risks": borrow_yld_risks.astype(np.float32),
                "borrow_go_risks": borrow_go_risks.astype(np.float32),
                "total_risks": total_risks.astype(np.float32),
                "cross_wait_time_s": float(cross_wait_time_s),
                "cross_wait_valid": float(cross_wait_valid),
                "chase": chase_info,
                "meet": meet_info,
            }

    if int(future_cover.get("exists", 0.0)) > 0 and future_cover["interaction"]["name"] == "meet":
        d_ego = float(future_cover.get("d_ego", np.nan))
        d_bg = float(future_cover.get("d_bg", np.nan))
        bg_speed = float(future_cover.get("other_speed", np.nan))
        meet_subtype = str(future_cover.get("interaction", {}).get("subtype", "none"))
        if np.isfinite(d_ego) and np.isfinite(d_bg) and np.isfinite(bg_speed) and d_bg > 1e-4 and bg_speed > 1e-4:
            if meet_subtype == "borrow_cross_meet" and borrow_corridor is not None:
                actor_id = int(future_cover.get("actor_id", -1))
                bg_length_m = float(future_cover.get("other_length_m", np.nan))
                if not np.isfinite(bg_length_m):
                    bg_length_m = float(ego_length_m)
                bg_clearance_m = float(max(bg_length_m, 1.0))
                start_distance_m = float(borrow_corridor["borrow_start_distance_m"])
                borrow_distance_m = float(borrow_corridor["borrow_distance_m"])
                d_bg_to_end_m = _borrow_actor_distance_to_corridor_end_m(
                    current_boxes=current_boxes,
                    actor_id=actor_id,
                    end_local_xy=borrow_corridor.get("end_local_xy"),
                    fallback_distance_m=d_bg,
                )
                t_bg_to_end = float(d_bg_to_end_m / max(bg_speed, 1e-6))
                t_bg_exit_to_start = float((d_bg_to_end_m + borrow_distance_m + bg_clearance_m) / max(bg_speed, 1e-6))
                v_yield_max = np.nan
                if np.isfinite(t_bg_exit_to_start):
                    denom = float(t_bg_exit_to_start) + float(cross_safe_gap_s)
                    v_yield_max = np.inf if denom <= 1e-6 else float(start_distance_m) / max(denom, 1e-6)
                v_go_min = np.nan
                if np.isfinite(t_bg_to_end):
                    go_denom = float(t_bg_to_end) - float(cross_safe_gap_s)
                    v_go_min = np.inf if go_denom <= 1e-6 else float(borrow_corridor["borrow_total_clear_distance_m"] + ego_length_m) / max(go_denom, 1e-6)
                meet_info = {
                    "valid": 1.0,
                    "d_ego_m": float(d_ego),
                    "d_bg_m": float(d_bg),
                    "conflict_len_m": float(borrow_distance_m),
                    "context_conflict_len_m": float(left_junction_conflict_len_m),
                    "borrow_conflict_len_m": float(borrow_distance_m),
                    "borrow_start_distance_m": float(start_distance_m),
                    "borrow_total_distance_m": float(borrow_corridor["borrow_total_clear_distance_m"]),
                    "d_bg_to_end_m": float(d_bg_to_end_m),
                    "ego_clearance_m": float(ego_length_m),
                    "bg_clearance_m": float(bg_clearance_m),
                    "bg_speed_mps": float(bg_speed),
                    "t_bg_s": float(d_bg / max(bg_speed, 1e-6)),
                    "t_bg_exit_s": float(t_bg_exit_to_start),
                    "t_bg_to_end_s": float(t_bg_to_end),
                    "t_bg_clear_s": np.nan,
                    "safe_gap_bg_m": np.nan,
                    "rear_gap_m": np.nan,
                    "v_equal_mps": np.nan,
                    "v_go_min_mps": float(v_go_min),
                    "v_behind_min_mps": np.nan,
                    "v_go_need_mps": np.nan,
                    "v_yield_max_mps": float(v_yield_max),
                    "subtype": meet_subtype,
                    "cover_case": "future",
                }
                for idx, candidate_speed in enumerate(sample_speeds):
                    v = float(candidate_speed)
                    if v <= 1e-6:
                        borrow_yld_risks[idx] = 0.0
                        borrow_go_risks[idx] = 1.0
                        meet_risks[idx] = 0.0
                        continue
                    t_ego_enter = float(start_distance_m / max(v, 1e-6))
                    t_ego_clear = float((borrow_corridor["borrow_total_clear_distance_m"] + ego_length_m) / max(v, 1e-6))
                    if np.isfinite(v_yield_max):
                        yld_risk = 0.0 if v <= float(v_yield_max) else 1.0
                    else:
                        yld_risk = _risk_from_time_gap(t_ego_enter - t_bg_exit_to_start, cross_safe_gap_s)
                    go_risk = _risk_from_time_gap(t_bg_to_end - t_ego_clear, cross_safe_gap_s)
                    borrow_yld_risks[idx] = float(np.clip(yld_risk, 0.0, 1.0))
                    borrow_go_risks[idx] = float(np.clip(go_risk, 0.0, 1.0))
                    meet_risks[idx] = float(np.clip(min(borrow_yld_risks[idx], borrow_go_risks[idx]), 0.0, 1.0))
                total_risks = np.maximum(chase_risks, meet_risks)
                return {
                    "sample_speeds_mps": sample_speeds.astype(np.float32),
                    "chase_risks": chase_risks.astype(np.float32),
                    "meet_risks": meet_risks.astype(np.float32),
                    "merge_yld_risks": merge_yld_risks.astype(np.float32),
                    "merge_go_risks": merge_go_risks.astype(np.float32),
                    "borrow_yld_risks": borrow_yld_risks.astype(np.float32),
                    "borrow_go_risks": borrow_go_risks.astype(np.float32),
                    "total_risks": total_risks.astype(np.float32),
                    "chase": chase_info,
                    "meet": meet_info,
                }
            conflict_len_m = np.nan
            risk_d_ego = float(d_ego)
            risk_d_bg = float(d_bg)
            ego_clearance_m = float(max(float(ego_length_m), 1.0))
            bg_length_m = float(future_cover.get("other_length_m", np.nan))
            if not np.isfinite(bg_length_m):
                bg_length_m = float(ego_length_m)
            bg_clearance_m = float(max(bg_length_m, 1.0))
            if meet_subtype == "junction_left_cross_meet":
                conflict_len_m = float(left_junction_conflict_len_m)
                conflict_half_m = 0.5 * conflict_len_m
                risk_d_ego = max(float(d_ego) - conflict_half_m, 0.0)
                risk_d_bg = max(float(d_bg) - conflict_half_m, 0.0)
            current_junction_bg_exit_s = np.nan
            current_junction_go_cap_mps = np.nan
            if (
                meet_subtype == "junction_left_cross_meet" and
                int(current_cover.get("exists", 0.0)) > 0 and
                str(current_cover.get("interaction", {}).get("subtype", "none")) == "junction_left_cross_meet"
            ):
                current_bg_speed = float(current_cover.get("other_speed", np.nan))
                current_bg_length_m = float(current_cover.get("other_length_m", np.nan))
                if not np.isfinite(current_bg_length_m):
                    current_bg_length_m = float(ego_length_m)
                current_bg_clearance_m = float(max(current_bg_length_m, 1.0))
                current_bg_occ_len_m = float(max(0.5 * float(conflict_len_m), 0.0) + current_bg_clearance_m)
                if np.isfinite(current_bg_speed) and current_bg_speed > 1e-3:
                    current_junction_bg_exit_s = float(current_bg_occ_len_m / max(current_bg_speed, 1e-6))
                else:
                    current_junction_bg_exit_s = np.inf
                cap_denom = float(current_junction_bg_exit_s) + float(cross_safe_gap_s)
                if np.isfinite(cap_denom):
                    current_junction_go_cap_mps = (
                        np.inf if cap_denom <= 1e-6 else (risk_d_ego / max(cap_denom, 1e-6))
                    )
            t_bg = risk_d_bg / max(bg_speed, 1e-6)
            cross_conflict_len_m = max(float(conflict_len_m) if np.isfinite(conflict_len_m) else 0.0, 0.0)
            bg_occ_len_m = float(
                cross_conflict_len_m + (bg_clearance_m if meet_subtype == "junction_left_cross_meet" else 0.0)
            )
            ego_cross_occ_len_m = float(
                cross_conflict_len_m + (ego_clearance_m if meet_subtype == "junction_left_cross_meet" else 0.0)
            )
            t_bg_exit = (risk_d_bg + bg_occ_len_m) / max(bg_speed, 1e-6)
            safe_gap_bg = float(merge_follow_base_gap_m + merge_follow_headway_s * max(bg_speed, 0.0))
            merge_clearance_effective_m = float(max(float(merge_clearance_m), float(ego_length_m)))
            t_bg_clear = (risk_d_bg + merge_clearance_effective_m) / max(bg_speed, 1e-6)
            rear_gap_m = float(future_cover.get("rear_gap_m", np.nan))
            if meet_subtype == "merge_meet":
                v_behind_min = _merge_speed_cap(max(float(bg_speed), 0.0), default=0.0)
                v_equal = _merge_speed_cap(
                    (d_ego + ego_clearance_m) / max(t_bg, 1e-3),
                    default=float(STAGE1_MERGE_SPEED_CAP_MPS),
                )
                v_go_min = _merge_speed_cap(
                    (d_ego + ego_clearance_m) / max(t_bg - float(merge_tau_s), 1e-3),
                    default=float(STAGE1_MERGE_SPEED_CAP_MPS),
                )
                v_go_need = _merge_speed_cap(
                    max(float(v_go_min), float(v_behind_min)),
                    default=float(STAGE1_MERGE_SPEED_CAP_MPS),
                )
                v_yield_max = _merge_speed_cap(
                    d_ego / max(t_bg_clear + float(merge_tau_s), 1e-3),
                    default=float(STAGE1_MERGE_SPEED_CAP_MPS),
                )
                debug_v_equal = float(v_equal)
                debug_v_go_min = float(v_go_min)
            else:
                safe_gap_bg = np.nan
                t_bg_clear = np.nan
                v_equal = np.nan
                v_go_min = np.nan
                v_behind_min = np.nan
                v_go_need = np.nan
                v_yield_max = np.nan
                debug_v_equal = np.nan
                debug_v_go_min = np.nan
                if meet_subtype == "junction_left_cross_meet":
                    go_denom = float(t_bg) - float(cross_safe_gap_s)
                    v_go_min = np.inf if go_denom <= 1e-6 else (risk_d_ego + ego_cross_occ_len_m) / max(go_denom, 1e-6)
                    v_yield_max = risk_d_ego / max(t_bg_exit + float(cross_safe_gap_s), 1e-6)
            meet_info = {
                "valid": 1.0,
                "d_ego_m": risk_d_ego,
                "d_bg_m": risk_d_bg,
                "conflict_len_m": float(conflict_len_m),
                "context_conflict_len_m": float(left_junction_conflict_len_m),
                "borrow_conflict_len_m": float(borrow_cross_conflict_len_m),
                "ego_clearance_m": float(ego_clearance_m) if meet_subtype in {"merge_meet", "junction_left_cross_meet"} else np.nan,
                "bg_clearance_m": float(bg_clearance_m) if meet_subtype == "junction_left_cross_meet" else np.nan,
                "bg_speed_mps": bg_speed,
                "t_bg_s": float(t_bg),
                "t_bg_exit_s": float(t_bg_exit),
                "t_current_bg_exit_s": float(current_junction_bg_exit_s),
                "t_bg_clear_s": float(t_bg_clear),
                "t_ego_exit_s": np.nan,
                "safe_gap_bg_m": float(safe_gap_bg),
                "rear_gap_m": float(rear_gap_m),
                "v_equal_mps": float(debug_v_equal),
                "v_go_min_mps": float(debug_v_go_min),
                "v_go_cap_mps": float(current_junction_go_cap_mps),
                "v_behind_min_mps": float(v_behind_min),
                "v_go_need_mps": float(v_go_need),
                "v_yield_max_mps": float(v_yield_max),
                "subtype": meet_subtype,
            }
            for idx, candidate_speed in enumerate(sample_speeds):
                v = float(candidate_speed)
                if v <= 1e-6:
                    meet_risks[idx] = 0.0
                    merge_yld_risks[idx] = 0.0
                    merge_go_risks[idx] = 1.0 if (meet_subtype == "merge_meet" and np.isfinite(v_go_need)) else 0.0
                    continue
                if meet_subtype == "merge_meet":
                    yld_risk = 0.0
                    go_risk = 0.0
                    if not np.isfinite(v_go_need):
                        risk = 0.0
                    elif not np.isfinite(v_yield_max):
                        rear_gap = rear_gap_m
                        if not np.isfinite(rear_gap):
                            rear_gap = d_bg
                        closing_rear = max(bg_speed - v, 0.0)
                        if closing_rear <= 1e-6:
                            risk = 0.0
                        else:
                            rear_ttc = rear_gap / max(closing_rear, 1e-6)
                            risk = float(
                                np.clip(
                                    (float(rear_safe_ttc_s) - rear_ttc) /
                                    max(float(rear_safe_ttc_s) - float(rear_hard_ttc_s), 1e-6),
                                    0.0,
                                    1.0,
                                )
                            )
                        yld_risk = float(risk)
                        go_risk = float(risk)
                    elif not np.isfinite(v_equal):
                        risk = float(np.clip((v_go_need - v) / max(v_go_need - v_yield_max, 1e-6), 0.0, 1.0))
                        yld_risk = float(np.clip((v - v_yield_max) / max(v_go_need - v_yield_max, 1e-6), 0.0, 1.0))
                        go_risk = float(np.clip((v_go_need - v) / max(v_go_need - v_yield_max, 1e-6), 0.0, 1.0))
                    else:
                        if v_go_need <= v_yield_max:
                            meet_risks[idx] = 0.0
                            merge_yld_risks[idx] = 0.0
                            merge_go_risks[idx] = 0.0
                            continue
                        peak_speed = max(float(v_equal), float(v_yield_max))
                        if v <= float(v_yield_max):
                            risk = 0.0
                            yld_risk = 0.0
                            go_risk = 1.0
                        elif peak_speed <= float(v_yield_max) + 1e-6:
                            risk = float(np.clip((v_go_need - v) / max(v_go_need - v_yield_max, 1e-6), 0.0, 1.0))
                            yld_risk = float(np.clip((v - v_yield_max) / max(v_go_need - v_yield_max, 1e-6), 0.0, 1.0))
                            go_risk = float(np.clip((v_go_need - v) / max(v_go_need - v_yield_max, 1e-6), 0.0, 1.0))
                        elif v <= peak_speed:
                            risk = float(np.clip((v - v_yield_max) / max(peak_speed - v_yield_max, 1e-6), 0.0, 1.0))
                            yld_risk = float(np.clip((v - v_yield_max) / max(peak_speed - v_yield_max, 1e-6), 0.0, 1.0))
                            go_risk = 1.0
                        elif v < v_go_need:
                            risk = float(np.clip((v_go_need - v) / max(v_go_need - peak_speed, 1e-6), 0.0, 1.0))
                            yld_risk = 1.0
                            go_risk = float(np.clip((v_go_need - v) / max(v_go_need - peak_speed, 1e-6), 0.0, 1.0))
                        else:
                            risk = 0.0
                            yld_risk = 1.0
                            go_risk = 0.0
                    if (
                        np.isfinite(v_yield_max) and np.isfinite(v_behind_min) and
                        float(v_behind_min) > float(v_yield_max) + 1e-6 and
                        v > float(v_yield_max)
                    ):
                        rear_speed_risk = float(
                            np.clip(
                                (float(v_behind_min) - v) /
                                max(float(v_behind_min) - float(v_yield_max), 1e-6),
                                0.0,
                                1.0,
                            )
                        )
                        risk = max(float(risk), float(rear_speed_risk))
                        yld_risk = max(float(yld_risk), float(rear_speed_risk))
                        go_risk = max(float(go_risk), float(rear_speed_risk))
                    merge_yld_risks[idx] = float(np.clip(np.nan_to_num(yld_risk, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
                    merge_go_risks[idx] = float(np.clip(np.nan_to_num(go_risk, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
                else:
                    conflict_len = max(float(conflict_len_m) if np.isfinite(conflict_len_m) else 0.0, 0.0)
                    t_ego_in = risk_d_ego / max(v, 1e-6)
                    if conflict_len > 1e-6:
                        t_ego_out = (risk_d_ego + ego_cross_occ_len_m) / max(v, 1e-6)
                        if meet_subtype == "junction_left_cross_meet" and idx == int(len(sample_speeds) // 2):
                            meet_info["t_ego_exit_s"] = float(t_ego_out)
                        gap_before = t_bg - t_ego_out
                        gap_after = t_ego_in - t_bg_exit
                        if meet_subtype == "junction_left_cross_meet":
                            yld_risk = float(
                                np.clip(
                                    (float(cross_safe_gap_s) - gap_after) / max(float(cross_safe_gap_s), 1e-6),
                                    0.0,
                                    1.0,
                                )
                            )
                            go_risk = float(
                                np.clip(
                                    (float(cross_safe_gap_s) - gap_before) / max(float(cross_safe_gap_s), 1e-6),
                                    0.0,
                                    1.0,
                                )
                            )
                            if np.isfinite(current_junction_bg_exit_s):
                                current_gap_after = t_ego_in - float(current_junction_bg_exit_s)
                                current_cap_risk = float(
                                    np.clip(
                                        (float(cross_safe_gap_s) - current_gap_after) /
                                        max(float(cross_safe_gap_s), 1e-6),
                                        0.0,
                                        1.0,
                                    )
                                )
                                go_risk = max(float(go_risk), float(current_cap_risk))
                            junction_cross_yld_risks[idx] = float(np.clip(np.nan_to_num(yld_risk, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
                            junction_cross_go_risks[idx] = float(np.clip(np.nan_to_num(go_risk, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
                            risk = float(min(junction_cross_yld_risks[idx], junction_cross_go_risks[idx]))
                        else:
                            time_clearance = max(gap_before, gap_after)
                            risk = float(
                                np.clip(
                                    (float(cross_safe_gap_s) - time_clearance) / max(float(cross_safe_gap_s), 1e-6),
                                    0.0,
                                    1.0,
                                )
                            )
                    else:
                        delta_t = abs(t_ego_in - t_bg)
                        risk = _risk_from_time_gap(delta_t, merge_tau_s)
                meet_risks[idx] = float(np.clip(np.nan_to_num(risk, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))

    total_risks = np.maximum(chase_risks, meet_risks)
    return {
        "sample_speeds_mps": sample_speeds.astype(np.float32),
        "chase_risks": chase_risks.astype(np.float32),
        "meet_risks": meet_risks.astype(np.float32),
        "junction_cross_yld_risks": junction_cross_yld_risks.astype(np.float32),
        "junction_cross_go_risks": junction_cross_go_risks.astype(np.float32),
        "merge_yld_risks": merge_yld_risks.astype(np.float32),
        "merge_go_risks": merge_go_risks.astype(np.float32),
        "borrow_yld_risks": borrow_yld_risks.astype(np.float32),
        "borrow_go_risks": borrow_go_risks.astype(np.float32),
        "total_risks": total_risks.astype(np.float32),
        "chase": chase_info,
        "meet": meet_info,
    }


def _build_speed_curve_targets(
    current_cover,
    future_cover,
    current_meas,
    current_boxes=None,
    route_local=None,
    release_info=None,
    event_name=None,
    safe_ttc_s=3.0,
    merge_tau_s=0.25,
    merge_clearance_m=6.0,
    merge_follow_base_gap_m=4.0,
    merge_follow_headway_s=0.6,
    chase_follow_base_gap_m=3.0,
    chase_follow_headway_s=0.5,
    rear_hard_ttc_s=1.0,
    rear_safe_ttc_s=3.0,
    ped_hard_ttc_s=1.0,
    ped_safe_ttc_s=3.0,
    junction_left_conflict_len_scale=1.5,
    cross_safe_gap_s=1.0,
    return_debug=False,
):
    speed = float((current_meas or {}).get("speed", 0.0))
    sample_speeds = _speed_risk_samples_fixed(speed)
    speed_curve = _build_speed_curve_debug(
        current_cover=current_cover,
        future_cover=future_cover,
        current_meas=current_meas,
        current_boxes=current_boxes,
        event_name=event_name,
        release_info=release_info,
        route_local=route_local,
        safe_ttc_s=safe_ttc_s,
        merge_tau_s=merge_tau_s,
        merge_clearance_m=merge_clearance_m,
        merge_follow_base_gap_m=merge_follow_base_gap_m,
        merge_follow_headway_s=merge_follow_headway_s,
        chase_follow_base_gap_m=chase_follow_base_gap_m,
        chase_follow_headway_s=chase_follow_headway_s,
        rear_hard_ttc_s=rear_hard_ttc_s,
        rear_safe_ttc_s=rear_safe_ttc_s,
        ped_hard_ttc_s=ped_hard_ttc_s,
        ped_safe_ttc_s=ped_safe_ttc_s,
        junction_left_conflict_len_scale=junction_left_conflict_len_scale,
        cross_safe_gap_s=cross_safe_gap_s,
        sample_speeds_override=sample_speeds,
    )

    sample_speeds = np.asarray(speed_curve.get("sample_speeds_mps", sample_speeds), dtype=np.float32)
    valid_mask = np.ones(sample_speeds.shape, dtype=np.float32)
    exp_index = np.int64(int(np.argmin(np.abs(sample_speeds - speed)))) if sample_speeds.size > 0 else np.int64(-1)
    zeros = np.zeros(sample_speeds.shape, dtype=np.float32)

    chase_risks = np.asarray(speed_curve.get("chase_risks", zeros), dtype=np.float32)
    meet_risks_raw = np.asarray(speed_curve.get("meet_risks", zeros), dtype=np.float32)
    junction_cross_yld_risks = np.asarray(speed_curve.get("junction_cross_yld_risks", zeros), dtype=np.float32)
    junction_cross_go_risks = np.asarray(speed_curve.get("junction_cross_go_risks", zeros), dtype=np.float32)
    merge_yld_risks = np.asarray(speed_curve.get("merge_yld_risks", zeros), dtype=np.float32)
    merge_go_risks = np.asarray(speed_curve.get("merge_go_risks", zeros), dtype=np.float32)
    borrow_yld_risks = np.asarray(speed_curve.get("borrow_yld_risks", zeros), dtype=np.float32)
    borrow_go_risks = np.asarray(speed_curve.get("borrow_go_risks", zeros), dtype=np.float32)

    is_ped_context = (
        int(current_cover.get("exists", 0.0)) > 0 and str(current_cover.get("actor_class_name", "")) in PEDESTRIAN_CLASSES
    ) or (
        int(future_cover.get("exists", 0.0)) > 0 and str(future_cover.get("actor_class_name", "")) in PEDESTRIAN_CLASSES
    )

    if is_ped_context:
        ped_risks = meet_risks_raw.copy()
        meet_risks = zeros.copy()
        chase_risks = zeros.copy()
        junction_cross_yld_risks = zeros.copy()
        junction_cross_go_risks = zeros.copy()
        merge_yld_risks = zeros.copy()
        merge_go_risks = zeros.copy()
        borrow_yld_risks = zeros.copy()
        borrow_go_risks = zeros.copy()
    else:
        ped_risks = zeros.copy()
        meet_risks = meet_risks_raw

    if not return_debug:
        return (
            sample_speeds,
            valid_mask,
            exp_index,
            chase_risks,
            meet_risks,
            ped_risks,
            junction_cross_yld_risks,
            junction_cross_go_risks,
            merge_yld_risks,
            merge_go_risks,
            borrow_yld_risks,
            borrow_go_risks,
        )

    def _cross_wait_from_meet_info(info):
        subtype = str((info or {}).get("subtype", "none"))
        if "cross" not in subtype:
            return 0.0, 0.0
        t_exit = float((info or {}).get("t_bg_exit_s", np.nan))
        t_bg = float((info or {}).get("t_bg_s", np.nan))
        t_wait = t_exit if np.isfinite(t_exit) else t_bg
        if not np.isfinite(t_wait):
            return 0.0, 0.0
        return float(max(t_wait, 0.0)), 1.0

    meet_info = dict(speed_curve.get("meet", {}))
    cross_wait_time_s = speed_curve.get("cross_wait_time_s", np.nan)
    cross_wait_valid = speed_curve.get("cross_wait_valid", np.nan)
    if not np.isfinite(float(cross_wait_time_s)) or not np.isfinite(float(cross_wait_valid)):
        cross_wait_time_s, cross_wait_valid = _cross_wait_from_meet_info(meet_info)

    total_risks = np.maximum(np.maximum(chase_risks, meet_risks), ped_risks)
    debug_payload = {
        "sample_speeds_mps": sample_speeds.astype(np.float32),
        "total_risks": total_risks.astype(np.float32),
        "chase_risks": chase_risks.astype(np.float32),
        "meet_risks": meet_risks.astype(np.float32),
        "junction_cross_yld_risks": junction_cross_yld_risks.astype(np.float32),
        "junction_cross_go_risks": junction_cross_go_risks.astype(np.float32),
        "merge_yld_risks": merge_yld_risks.astype(np.float32),
        "merge_go_risks": merge_go_risks.astype(np.float32),
        "borrow_yld_risks": borrow_yld_risks.astype(np.float32),
        "borrow_go_risks": borrow_go_risks.astype(np.float32),
        "ped_risks": ped_risks.astype(np.float32),
        "cross_wait_time_s": float(cross_wait_time_s),
        "cross_wait_valid": float(cross_wait_valid),
        "chase_debug": dict(speed_curve.get("chase", {})),
        "meet_debug": meet_info,
    }
    return (
        sample_speeds,
        valid_mask,
        exp_index,
        chase_risks,
        meet_risks,
        ped_risks,
        junction_cross_yld_risks,
        junction_cross_go_risks,
        merge_yld_risks,
        merge_go_risks,
        borrow_yld_risks,
        borrow_go_risks,
        debug_payload,
    )


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
    ped_current_cover,
    ped_future_cover,
    speed_curve_future_cover,
    speed_curve_future_persisted,
    speed_curve_debug,
    merge_motion=None,
    scene_borrow_context=None,
    borrow_motion=None,
):
    return _to_stage1_debug_python({
        "current_cover": dict(current_cover),
        "future_cover": dict(future_cover),
        "ped_current_cover": dict(ped_current_cover),
        "ped_future_cover": dict(ped_future_cover),
        "speed_curve_future_cover": dict(speed_curve_future_cover),
        "speed_curve_future_persisted": bool(speed_curve_future_persisted),
        "speed_curve": dict(speed_curve_debug),
        "merge_motion": dict(merge_motion or {}),
        "scene_borrow_context": None if scene_borrow_context is None else dict(scene_borrow_context),
        "borrow_motion": dict(borrow_motion or {}),
    })


def _default_merge_episode_debug():
    return {
        'phase': 'none',
        'phase_code': int(MERGE_DECISION_PHASE_TO_CODE['none']),
        'episode_id': -1,
        'active': 0.0,
        'hold': 0.0,
        'hold_reason': 'none',
        'no_go': 0.0,
        'start_frame': -1,
        'end_frame': -1,
        'go_frame': -1,
        'go_reason': 'none',
        'resolution_actor_id': -1,
        'end_state': 'none',
        'end_state_code': int(MERGE_END_STATE_TO_CODE['none']),
        'actor_ids': [],
        'actor_switch_frames': [],
        'future_merge_count': 0,
        'merge_area_start_s_m': np.nan,
        'merge_area_end_s_m': np.nan,
        'merge_area_first_conflict_s_m': np.nan,
        'merge_area_last_conflict_s_m': np.nan,
        'frame_role': 'none',
        'future_grace_index': 0,
        'red_light_hold_index': 0,
        'post_go_grace_index': 0,
        'red_light_hold': 0.0,
        'grace_frames': int(STAGE1_MERGE_GRACE_FRAMES),
        'actor_grace_frames': int(STAGE1_MERGE_ACTOR_CONTINUITY_GRACE_FRAMES),
        'resolution_lookahead_frames': int(STAGE1_MERGE_RESOLUTION_LOOKAHEAD_FRAMES),
    }


def _default_borrow_cross_episode_debug():
    return {
        'phase': 'none',
        'phase_code': int(CROSS_DECISION_PHASE_TO_CODE['none']),
        'episode_id': -1,
        'active': 0.0,
        'active_time_s': 0.0,
        'start_frame': -1,
        'end_frame': -1,
        'go_frame': -1,
        'context_frame': -1,
        'window_start_distance_m': np.nan,
        'window_end_distance_m': np.nan,
        'frame_role': 'none',
    }


def _default_junction_cross_episode_debug():
    return {
        'episode_id': -1,
        'active': 0.0,
        'start_frame': -1,
        'end_frame': -1,
        'frame_role': 'none',
        'area_center_world_xy': [],
        'area_radius_m': np.nan,
        'candidate_frame_count': 0,
    }


def _set_stage1_junction_cross_defaults(sample):
    sample['junction_cross_episode_id'] = np.int64(-1)
    sample['junction_cross_episode_active'] = np.float32(0.0)
    sample['junction_cross_episode_start_frame'] = np.int64(-1)
    sample['junction_cross_episode_end_frame'] = np.int64(-1)
    stage1_debug = sample.get('stage1_speed_debug')
    if isinstance(stage1_debug, dict):
        stage1_debug['junction_cross_episode'] = _default_junction_cross_episode_debug()


def _set_stage1_borrow_cross_defaults(sample):
    sample['borrow_cross_decision_phase'] = np.int64(CROSS_DECISION_PHASE_TO_CODE['none'])
    sample['borrow_cross_episode_id'] = np.int64(-1)
    sample['borrow_cross_episode_active'] = np.float32(0.0)
    sample['borrow_cross_active_time_s'] = np.float32(0.0)
    sample['borrow_cross_episode_start_frame'] = np.int64(-1)
    sample['borrow_cross_episode_end_frame'] = np.int64(-1)
    sample['borrow_cross_go_frame'] = np.int64(-1)
    sample['borrow_cross_context_frame'] = np.int64(-1)
    stage1_debug = sample.get('stage1_speed_debug')
    if isinstance(stage1_debug, dict):
        stage1_debug['borrow_cross_episode'] = _default_borrow_cross_episode_debug()


def _set_stage1_merge_defaults(sample):
    sample['merge_decision_phase'] = np.int64(MERGE_DECISION_PHASE_TO_CODE['none'])
    sample['merge_episode_id'] = np.int64(-1)
    sample['merge_episode_active'] = np.float32(0.0)
    sample['merge_episode_no_go'] = np.float32(0.0)
    sample['merge_episode_start_frame'] = np.int64(-1)
    sample['merge_episode_end_frame'] = np.int64(-1)
    sample['merge_go_frame'] = np.int64(-1)
    sample['merge_resolution_actor_id'] = np.int64(-1)
    sample['merge_end_state'] = np.int64(MERGE_END_STATE_TO_CODE['none'])
    sample['merge_hold'] = np.float32(0.0)
    stage1_debug = sample.get('stage1_speed_debug')
    if isinstance(stage1_debug, dict):
        stage1_debug['merge_episode'] = _default_merge_episode_debug()


def _has_stage1_speed_fields(sample):
    return all(field in sample for field in STAGE1_SPEED_FIELDS)


def _set_stage1_speed_fallback(sample):
    speed_hist = sample.get('speed_hist', sample.get('speed'))
    if speed_hist is None:
        speed_mps = 0.0
    else:
        speed_arr = np.asarray(speed_hist, dtype=np.float32)
        speed_mps = float(speed_arr[-1]) if speed_arr.ndim > 0 else float(speed_arr)
    sample_speeds = _speed_risk_samples_fixed(speed_mps)
    sample['speed_sample_values'] = sample_speeds.astype(np.float32)
    sample['speed_sample_valid_mask'] = np.ones(sample_speeds.shape, dtype=np.float32)
    sample['speed_sample_exp_index'] = np.int64(3)
    sample['speed_risk_chase_values'] = np.zeros(sample_speeds.shape, dtype=np.float32)
    sample['speed_risk_meet_values'] = np.zeros(sample_speeds.shape, dtype=np.float32)
    sample['speed_risk_junction_cross_yld_values'] = np.zeros(sample_speeds.shape, dtype=np.float32)
    sample['speed_risk_junction_cross_go_values'] = np.zeros(sample_speeds.shape, dtype=np.float32)
    sample['speed_risk_merge_yld_values'] = np.zeros(sample_speeds.shape, dtype=np.float32)
    sample['speed_risk_merge_go_values'] = np.zeros(sample_speeds.shape, dtype=np.float32)
    sample['speed_risk_borrow_yld_values'] = np.zeros(sample_speeds.shape, dtype=np.float32)
    sample['speed_risk_borrow_go_values'] = np.zeros(sample_speeds.shape, dtype=np.float32)
    sample['speed_risk_ped_values'] = np.zeros(sample_speeds.shape, dtype=np.float32)
    sample['speed_cross_wait_time_s'] = np.float32(0.0)
    sample['speed_cross_wait_valid'] = np.float32(0.0)
    sample['stage1_speed_debug'] = _build_stage1_speed_debug_payload(
        current_cover=_cover_candidate_summary(0, None, {}),
        future_cover=_cover_candidate_summary(0, None, {}),
        ped_current_cover=_cover_candidate_summary(0, None, {}),
        ped_future_cover=_cover_candidate_summary(0, None, {}),
        speed_curve_future_cover=_cover_candidate_summary(0, None, {}),
        speed_curve_future_persisted=False,
        speed_curve_debug={
            'sample_speeds_mps': sample_speeds.astype(np.float32),
            'total_risks': np.zeros(sample_speeds.shape, dtype=np.float32),
            'chase_risks': np.zeros(sample_speeds.shape, dtype=np.float32),
            'meet_risks': np.zeros(sample_speeds.shape, dtype=np.float32),
            'junction_cross_yld_risks': np.zeros(sample_speeds.shape, dtype=np.float32),
            'junction_cross_go_risks': np.zeros(sample_speeds.shape, dtype=np.float32),
            'merge_yld_risks': np.zeros(sample_speeds.shape, dtype=np.float32),
            'merge_go_risks': np.zeros(sample_speeds.shape, dtype=np.float32),
            'borrow_yld_risks': np.zeros(sample_speeds.shape, dtype=np.float32),
            'borrow_go_risks': np.zeros(sample_speeds.shape, dtype=np.float32),
            'ped_risks': np.zeros(sample_speeds.shape, dtype=np.float32),
            'cross_wait_time_s': 0.0,
            'cross_wait_valid': 0.0,
            'chase_debug': {'valid': 0.0},
            'meet_debug': {'valid': 0.0},
        },
        scene_borrow_context=None,
        borrow_motion=None,
    )
    _set_stage1_junction_cross_defaults(sample)
    _set_stage1_borrow_cross_defaults(sample)
    _set_stage1_merge_defaults(sample)


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
    ego_world_xy = np.asarray([np.nan, np.nan], dtype=np.float32)
    scene_route_polyline_world = np.asarray(scene_route_polyline_world, dtype=np.float32)
    if (
        ego_matrix_current is not None and
        scene_route_polyline_world.ndim == 2 and
        scene_route_polyline_world.shape[0] >= 2
    ):
        ego_matrix_np = np.asarray(ego_matrix_current, dtype=np.float32)
        if ego_matrix_np.ndim == 2 and ego_matrix_np.shape[0] >= 3 and ego_matrix_np.shape[1] >= 4:
            ego_world_xy = ego_matrix_np[:2, 3].astype(np.float32)
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
        'ego_world_xy': ego_world_xy.astype(float).tolist() if np.all(np.isfinite(ego_world_xy)) else [],
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


def _merge_episode_actor_summary(records, start_pos, end_pos):
    actor_ids = []
    actor_switch_frames = []
    last_actor_id = None
    for pos in range(int(start_pos), int(end_pos) + 1):
        future_cover = records[pos]['future_cover']
        if not _cover_is_merge_meet(future_cover):
            continue
        actor_id = _cover_actor_id(future_cover)
        if actor_id < 0:
            continue
        actor_ids.append(int(actor_id))
        if last_actor_id is None or int(actor_id) != int(last_actor_id):
            actor_switch_frames.append(int(records[pos]['frame_id']))
            last_actor_id = int(actor_id)
    if not actor_ids:
        return [], [], -1
    unique_actor_ids = []
    for actor_id in actor_ids:
        if actor_id not in unique_actor_ids:
            unique_actor_ids.append(int(actor_id))
    counts = {}
    for actor_id in actor_ids:
        counts[int(actor_id)] = counts.get(int(actor_id), 0) + 1
    dominant_actor_id = max(sorted(counts.keys()), key=lambda actor_id: (counts[actor_id], -unique_actor_ids.index(actor_id)))
    return unique_actor_ids, actor_switch_frames, int(dominant_actor_id)


def _junction_cover_conflict_world_xy(cover):
    if int((cover or {}).get('exists', 0.0)) <= 0:
        return None
    if str(((cover or {}).get('interaction') or {}).get('subtype', 'none')) != 'junction_left_cross_meet':
        return None
    pt = np.asarray((cover or {}).get('scene_route_conflict_world_xy', []), dtype=np.float32).reshape(-1)
    if pt.size >= 2 and np.all(np.isfinite(pt[:2])):
        return pt[:2].astype(np.float32)
    return None


def _junction_record_conflict_world_xy(record):
    future_cover = (record or {}).get('future_cover') or {}
    pt = _junction_cover_conflict_world_xy(future_cover)
    if pt is not None:
        return pt
    current_cover = (record or {}).get('current_cover') or {}
    return _junction_cover_conflict_world_xy(current_cover)


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


def _junction_record_ego_world_xy(record):
    merge_motion = (record or {}).get('merge_motion') or {}
    pt = np.asarray(merge_motion.get('ego_world_xy', []), dtype=np.float32).reshape(-1)
    if pt.size >= 2 and np.all(np.isfinite(pt[:2])):
        return pt[:2].astype(np.float32)
    return None


def _junction_cluster_conflict_candidates(records):
    candidate_points = []
    for pos, record in enumerate(records):
        pt = _junction_record_conflict_world_xy(record)
        if pt is None:
            continue
        candidate_points.append({
            'pos': int(pos),
            'frame_id': int(record.get('frame_id', -1)),
            'point_xy': pt.astype(np.float32),
            'radius_m': float(_junction_record_conflict_radius_m(record)),
            'front_s': float(_junction_record_scene_front_s(record)),
        })
    if not candidate_points:
        return []

    clusters = []
    for item in sorted(candidate_points, key=lambda entry: int(entry['pos'])):
        assigned = None
        for cluster in clusters:
            center = np.asarray(cluster['center_xy'], dtype=np.float32)
            radius_m = float(max(cluster['radius_m'], item['radius_m']))
            if float(np.linalg.norm(item['point_xy'] - center)) <= radius_m:
                assigned = cluster
                break
        if assigned is None:
            clusters.append({
                'center_xy': item['point_xy'].astype(np.float32),
                'radius_m': float(item['radius_m']),
                'items': [item],
            })
            continue
        assigned['items'].append(item)
        pts = np.stack([entry['point_xy'] for entry in assigned['items']], axis=0)
        assigned['center_xy'] = np.mean(pts, axis=0).astype(np.float32)
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
        valid_clusters.append({
            'center_xy': np.asarray(cluster['center_xy'], dtype=np.float32),
            'radius_m': float(cluster['radius_m']),
            'items': items,
            'start_pos': int(start_pos),
        })
    valid_clusters.sort(key=lambda cluster: int(cluster['start_pos']))
    return valid_clusters


def _set_stage1_junction_cross_annotation(sample, cross_info):
    sample['junction_cross_episode_id'] = np.int64(int(cross_info.get('episode_id', -1)))
    sample['junction_cross_episode_active'] = np.float32(float(cross_info.get('active', 0.0)))
    sample['junction_cross_episode_start_frame'] = np.int64(int(cross_info.get('start_frame', -1)))
    sample['junction_cross_episode_end_frame'] = np.int64(int(cross_info.get('end_frame', -1)))
    stage1_debug = sample.get('stage1_speed_debug')
    if isinstance(stage1_debug, dict):
        stage1_debug['junction_cross_episode'] = _to_stage1_debug_python(dict(cross_info))


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


def _set_stage1_borrow_cross_annotation(sample, cross_info):
    phase = str(cross_info.get('phase', 'none'))
    sample['borrow_cross_decision_phase'] = np.int64(CROSS_DECISION_PHASE_TO_CODE.get(phase, 0))
    sample['borrow_cross_episode_id'] = np.int64(int(cross_info.get('episode_id', -1)))
    sample['borrow_cross_episode_active'] = np.float32(float(cross_info.get('active', 0.0)))
    sample['borrow_cross_active_time_s'] = np.float32(float(cross_info.get('active_time_s', 0.0)))
    sample['borrow_cross_episode_start_frame'] = np.int64(int(cross_info.get('start_frame', -1)))
    sample['borrow_cross_episode_end_frame'] = np.int64(int(cross_info.get('end_frame', -1)))
    sample['borrow_cross_go_frame'] = np.int64(int(cross_info.get('go_frame', -1)))
    sample['borrow_cross_context_frame'] = np.int64(int(cross_info.get('context_frame', -1)))
    stage1_debug = sample.get('stage1_speed_debug')
    if isinstance(stage1_debug, dict):
        stage1_debug['borrow_cross_episode'] = _to_stage1_debug_python(dict(cross_info))


def _borrow_find_slowdown_start_pos(records, candidate_positions, search_stop_pos):
    positions = [int(pos) for pos in candidate_positions if int(pos) <= int(search_stop_pos)]
    for pos in positions:
        if pos + 2 >= len(records):
            continue
        v0 = float((records[pos].get('borrow_motion') or {}).get('speed_mps', np.nan))
        v1 = float((records[pos + 1].get('borrow_motion') or {}).get('speed_mps', np.nan))
        v2 = float((records[pos + 2].get('borrow_motion') or {}).get('speed_mps', np.nan))
        if not (np.isfinite(v0) and np.isfinite(v1) and np.isfinite(v2)):
            continue
        if v1 <= v0 - 0.15 and v2 <= v1 + 0.05:
            return int(pos)
    return None


def _annotate_route_stage1_borrow_cross_decisions(samples, route_sample_indices):
    ordered_indices = sorted(route_sample_indices, key=lambda i: int(samples[i].get('frame_id', -1)))
    records = []
    for sample_idx in ordered_indices:
        sample = samples[sample_idx]
        stage1_debug = sample.get('stage1_speed_debug') or {}
        scene_borrow_context = stage1_debug.get('scene_borrow_context') or {}
        current_cover = stage1_debug.get('current_cover') or {}
        future_cover = stage1_debug.get('future_cover') or {}
        speed_curve_future_cover = stage1_debug.get('speed_curve_future_cover') or {}
        records.append({
            'sample_idx': int(sample_idx),
            'frame_id': int(sample.get('frame_id', -1)),
            'scene_borrow_context': scene_borrow_context,
            'borrow_motion': stage1_debug.get('borrow_motion') or {},
            'current_cover': current_cover,
            'future_cover': future_cover,
            'speed_curve_future_cover': speed_curve_future_cover,
        })

    default_info = _default_borrow_cross_episode_debug()
    for record in records:
        _set_stage1_borrow_cross_annotation(samples[record['sample_idx']], default_info)

    if not records:
        return

    scene_borrow_context = None
    for record in records:
        ctx = record.get('scene_borrow_context') or {}
        if float(ctx.get('valid', 0.0)) > 0.5 and float(ctx.get('ready', 0.0)) > 0.5:
            scene_borrow_context = ctx
            break
    if scene_borrow_context is None:
        return

    event_name = str(scene_borrow_context.get('event_name', ''))
    if event_name not in {"ConstructionObstacleTwoWays", "AccidentTwoWays"}:
        return

    context_frame_id = int(scene_borrow_context.get('context_frame_id', -1))
    context_pos = None
    if context_frame_id >= 0:
        for pos, record in enumerate(records):
            if int(record['frame_id']) == int(context_frame_id):
                context_pos = int(pos)
                break
    if context_pos is None:
        return

    end_pos = None
    for pos in range(int(context_pos), len(records)):
        end_dist = float((records[pos].get('borrow_motion') or {}).get('borrow_end_distance_m', np.nan))
        if np.isfinite(end_dist) and end_dist <= 0.5:
            end_pos = int(pos)
            break
    if end_pos is None:
        return

    go_pos = None
    context_speed = float((records[context_pos].get('borrow_motion') or {}).get('speed_mps', np.nan))
    if np.isfinite(context_speed) and context_speed > 0.5:
        go_pos = int(context_pos)
    else:
        for pos in range(int(context_pos), int(end_pos) + 1):
            motion = records[pos].get('borrow_motion') or {}
            speed_mps = float(motion.get('speed_mps', np.nan))
            start_dist = float(motion.get('borrow_start_distance_m', np.nan))
            if np.isfinite(speed_mps) and speed_mps > 0.5 and np.isfinite(start_dist) and start_dist <= 5.0:
                go_pos = int(pos)
                break

    band_positions = []
    for pos, record in enumerate(records):
        if pos > int(end_pos):
            break
        start_dist = float((record.get('borrow_motion') or {}).get('borrow_start_distance_m', np.nan))
        if np.isfinite(start_dist) and 0.0 <= start_dist <= 10.0:
            band_positions.append(int(pos))
    if not band_positions:
        return

    start_pos = _borrow_find_slowdown_start_pos(
        records,
        candidate_positions=band_positions,
        search_stop_pos=(go_pos if go_pos is not None else end_pos),
    )
    if start_pos is None:
        fallback_positions = [
            int(pos) for pos in band_positions
            if int(pos) <= int(go_pos if go_pos is not None else end_pos)
        ]
        if not fallback_positions:
            return
        start_pos = min(
            fallback_positions,
            key=lambda pos: abs(float((records[pos].get('borrow_motion') or {}).get('borrow_start_distance_m', np.nan)) - 5.0),
        )

    if int(start_pos) > int(end_pos):
        return

    episode_id = 0
    start_frame = int(records[start_pos]['frame_id'])
    end_frame = int(records[end_pos]['frame_id'])
    go_frame = int(records[go_pos]['frame_id']) if go_pos is not None else -1
    for pos in range(int(start_pos), int(end_pos) + 1):
        phase = 'yld' if (go_pos is None or int(pos) < int(go_pos)) else 'go'
        frame_role = 'active'
        if int(pos) == int(start_pos):
            frame_role = 'start'
        if go_pos is not None and int(pos) == int(go_pos):
            frame_role = 'go_start'
        if int(pos) == int(end_pos):
            frame_role = 'end'
        cross_info = {
            'phase': str(phase),
            'phase_code': int(CROSS_DECISION_PHASE_TO_CODE.get(phase, 0)),
            'episode_id': int(episode_id),
            'active': 1.0,
            'active_time_s': float((int(pos) - int(start_pos)) * float(STAGE1_BORROW_ACTIVE_DT_S)),
            'start_frame': int(start_frame),
            'end_frame': int(end_frame),
            'go_frame': int(go_frame),
            'context_frame': int(context_frame_id),
            'window_start_distance_m': 10.0,
            'window_end_distance_m': 0.0,
            'frame_role': str(frame_role),
        }
        _set_stage1_borrow_cross_annotation(samples[records[pos]['sample_idx']], cross_info)


def _annotate_route_stage1_junction_cross_decisions(samples, route_sample_indices):
    ordered_indices = sorted(route_sample_indices, key=lambda i: int(samples[i].get('frame_id', -1)))
    records = []
    for sample_idx in ordered_indices:
        sample = samples[sample_idx]
        stage1_debug = sample.get('stage1_speed_debug') or {}
        current_cover = stage1_debug.get('current_cover') or {}
        future_cover = stage1_debug.get('future_cover') or {}
        speed_curve = stage1_debug.get('speed_curve') or {}
        records.append({
            'sample_idx': int(sample_idx),
            'frame_id': int(sample.get('frame_id', -1)),
            'current_cover': current_cover,
            'future_cover': future_cover,
            'merge_motion': stage1_debug.get('merge_motion') or {},
            'meet_debug': speed_curve.get('meet_debug') or {},
        })

    default_info = _default_junction_cross_episode_debug()
    for record in records:
        _set_stage1_junction_cross_annotation(samples[record['sample_idx']], default_info)

    if not records:
        return

    episode_id = 0
    prev_end_pos = -1
    for cluster in _junction_cluster_conflict_candidates(records):
        start_pos = max(int(cluster['start_pos']), int(prev_end_pos) + 1)
        if start_pos >= len(records):
            continue
        center_xy = np.asarray(cluster['center_xy'], dtype=np.float32)
        radius_m = float(cluster['radius_m'])
        if center_xy.shape != (2,) or not np.all(np.isfinite(center_xy)) or not np.isfinite(radius_m) or radius_m <= 0.0:
            continue

        cluster_positions = sorted(int(item['pos']) for item in cluster['items'] if int(item['pos']) >= int(start_pos))
        if len(cluster_positions) < int(STAGE1_JUNCTION_CROSS_MIN_CLUSTER_POINTS):
            continue

        end_pos = None
        has_entered_area = False
        for pos in range(int(start_pos), len(records)):
            ego_xy = _junction_record_ego_world_xy(records[pos])
            if ego_xy is None:
                continue
            inside_area = float(np.linalg.norm(ego_xy - center_xy)) <= float(radius_m)
            if inside_area:
                has_entered_area = True
                continue
            if has_entered_area and not inside_area:
                end_pos = int(pos)
                break
        if end_pos is None:
            end_pos = int(max(cluster_positions) if not has_entered_area else len(records) - 1)

        if int(start_pos) > int(end_pos):
            continue

        start_frame = int(records[start_pos]['frame_id'])
        end_frame = int(records[end_pos]['frame_id'])
        candidate_frame_count = int(len(cluster_positions))
        for pos in range(int(start_pos), int(end_pos) + 1):
            frame_role = 'active'
            if int(pos) == int(start_pos):
                frame_role = 'start'
            if int(pos) == int(end_pos):
                frame_role = 'end'
            cross_info = {
                'episode_id': int(episode_id),
                'active': 1.0,
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'frame_role': str(frame_role),
                'area_center_world_xy': center_xy.astype(float).tolist(),
                'area_radius_m': float(radius_m),
                'candidate_frame_count': int(candidate_frame_count),
            }
            _set_stage1_junction_cross_annotation(samples[records[pos]['sample_idx']], cross_info)
        prev_end_pos = int(end_pos)
        episode_id += 1


def _set_stage1_merge_annotation(sample, merge_info):
    phase = str(merge_info.get('phase', 'none'))
    end_state = str(merge_info.get('end_state', 'none'))
    sample['merge_decision_phase'] = np.int64(MERGE_DECISION_PHASE_TO_CODE.get(phase, 0))
    sample['merge_episode_id'] = np.int64(int(merge_info.get('episode_id', -1)))
    sample['merge_episode_active'] = np.float32(float(merge_info.get('active', 0.0)))
    sample['merge_episode_no_go'] = np.float32(float(merge_info.get('no_go', 0.0)))
    sample['merge_episode_start_frame'] = np.int64(int(merge_info.get('start_frame', -1)))
    sample['merge_episode_end_frame'] = np.int64(int(merge_info.get('end_frame', -1)))
    sample['merge_go_frame'] = np.int64(int(merge_info.get('go_frame', -1)))
    sample['merge_resolution_actor_id'] = np.int64(int(merge_info.get('resolution_actor_id', -1)))
    sample['merge_end_state'] = np.int64(MERGE_END_STATE_TO_CODE.get(end_state, 0))
    sample['merge_hold'] = np.float32(float(merge_info.get('hold', 0.0)))
    stage1_debug = sample.get('stage1_speed_debug')
    if isinstance(stage1_debug, dict):
        stage1_debug['merge_episode'] = _to_stage1_debug_python(dict(merge_info))


def _gate_route_stage1_merge_speed_risks(samples, route_sample_indices):
    for sample_idx in route_sample_indices:
        sample = samples[int(sample_idx)]
        merge_active = float(sample.get('merge_episode_active', 0.0))
        if merge_active > 0.5:
            continue

        merge_yld = np.asarray(sample.get('speed_risk_merge_yld_values', []), dtype=np.float32).reshape(-1)
        merge_go = np.asarray(sample.get('speed_risk_merge_go_values', []), dtype=np.float32).reshape(-1)
        if merge_yld.size > 0:
            sample['speed_risk_merge_yld_values'] = np.zeros_like(merge_yld, dtype=np.float32)
        if merge_go.size > 0:
            sample['speed_risk_merge_go_values'] = np.zeros_like(merge_go, dtype=np.float32)

        stage1_debug = sample.get('stage1_speed_debug')
        if not isinstance(stage1_debug, dict):
            continue
        speed_curve = stage1_debug.get('speed_curve')
        if not isinstance(speed_curve, dict):
            continue
        debug_merge_yld = np.asarray(speed_curve.get('merge_yld_risks', []), dtype=np.float32).reshape(-1)
        debug_merge_go = np.asarray(speed_curve.get('merge_go_risks', []), dtype=np.float32).reshape(-1)
        if debug_merge_yld.size > 0:
            speed_curve['merge_yld_risks'] = np.zeros_like(debug_merge_yld, dtype=np.float32)
        if debug_merge_go.size > 0:
            speed_curve['merge_go_risks'] = np.zeros_like(debug_merge_go, dtype=np.float32)


def _gate_route_stage1_borrow_speed_risks(samples, route_sample_indices):
    for sample_idx in route_sample_indices:
        sample = samples[int(sample_idx)]
        borrow_active = float(sample.get('borrow_cross_episode_active', 0.0))
        if borrow_active > 0.5:
            continue

        borrow_yld = np.asarray(sample.get('speed_risk_borrow_yld_values', []), dtype=np.float32).reshape(-1)
        borrow_go = np.asarray(sample.get('speed_risk_borrow_go_values', []), dtype=np.float32).reshape(-1)
        if borrow_yld.size > 0:
            sample['speed_risk_borrow_yld_values'] = np.zeros_like(borrow_yld, dtype=np.float32)
        if borrow_go.size > 0:
            sample['speed_risk_borrow_go_values'] = np.zeros_like(borrow_go, dtype=np.float32)

        stage1_debug = sample.get('stage1_speed_debug')
        if not isinstance(stage1_debug, dict):
            continue
        speed_curve = stage1_debug.get('speed_curve')
        if not isinstance(speed_curve, dict):
            continue
        debug_borrow_yld = np.asarray(speed_curve.get('borrow_yld_risks', []), dtype=np.float32).reshape(-1)
        debug_borrow_go = np.asarray(speed_curve.get('borrow_go_risks', []), dtype=np.float32).reshape(-1)
        if debug_borrow_yld.size > 0:
            speed_curve['borrow_yld_risks'] = np.zeros_like(debug_borrow_yld, dtype=np.float32)
        if debug_borrow_go.size > 0:
            speed_curve['borrow_go_risks'] = np.zeros_like(debug_borrow_go, dtype=np.float32)


def _gate_route_stage1_junction_speed_risks(samples, route_sample_indices):
    for sample_idx in route_sample_indices:
        sample = samples[int(sample_idx)]
        junction_active = float(sample.get('junction_cross_episode_active', 0.0))
        if junction_active > 0.5:
            continue

        junction_yld = np.asarray(sample.get('speed_risk_junction_cross_yld_values', []), dtype=np.float32).reshape(-1)
        junction_go = np.asarray(sample.get('speed_risk_junction_cross_go_values', []), dtype=np.float32).reshape(-1)
        if junction_yld.size > 0:
            sample['speed_risk_junction_cross_yld_values'] = np.zeros_like(junction_yld, dtype=np.float32)
        if junction_go.size > 0:
            sample['speed_risk_junction_cross_go_values'] = np.zeros_like(junction_go, dtype=np.float32)

        stage1_debug = sample.get('stage1_speed_debug')
        if not isinstance(stage1_debug, dict):
            continue
        speed_curve = stage1_debug.get('speed_curve')
        if not isinstance(speed_curve, dict):
            continue
        debug_junction_yld = np.asarray(speed_curve.get('junction_cross_yld_risks', []), dtype=np.float32).reshape(-1)
        debug_junction_go = np.asarray(speed_curve.get('junction_cross_go_risks', []), dtype=np.float32).reshape(-1)
        if debug_junction_yld.size > 0:
            speed_curve['junction_cross_yld_risks'] = np.zeros_like(debug_junction_yld, dtype=np.float32)
        if debug_junction_go.size > 0:
            speed_curve['junction_cross_go_risks'] = np.zeros_like(debug_junction_go, dtype=np.float32)


def _annotate_route_stage1_merge_decisions(samples, route_sample_indices, grace_frames=STAGE1_MERGE_GRACE_FRAMES):
    ordered_indices = sorted(route_sample_indices, key=lambda i: int(samples[i].get('frame_id', -1)))
    records = []
    for sample_idx in ordered_indices:
        sample = samples[sample_idx]
        stage1_debug = sample.get('stage1_speed_debug') or {}
        current_cover = stage1_debug.get('current_cover') or _cover_candidate_summary(0, None, {})
        future_cover = stage1_debug.get('future_cover') or _cover_candidate_summary(0, None, {})
        speed_curve = stage1_debug.get('speed_curve') or {}
        records.append({
            'sample_idx': int(sample_idx),
            'frame_id': int(sample.get('frame_id', -1)),
            'current_cover': current_cover,
            'future_cover': future_cover,
            'merge_motion': stage1_debug.get('merge_motion') or {},
            'meet_debug': speed_curve.get('meet_debug') or {},
        })

    default_info = _default_merge_episode_debug()
    for record in records:
        _set_stage1_merge_annotation(samples[record['sample_idx']], default_info)

    num_records = len(records)
    pos = 0
    episode_id = 0
    while pos < num_records:
        start_scan_pos = None
        while pos < num_records:
            if np.isfinite(_merge_record_conflict_s(records[pos])):
                start_scan_pos = int(pos)
                break
            pos += 1
        if start_scan_pos is None:
            break

        area_info = _merge_resolve_conflict_area(
            records,
            _merge_collect_conflict_candidate_positions(records, start_scan_pos),
        )
        if area_info is None:
            pos = int(start_scan_pos) + 1
            continue

        start_pos = int(area_info['start_pos'])
        future_merge_positions = [int(p) for p in area_info.get('inlier_positions', [])]
        conflict_s_values = [
            float(_merge_record_conflict_s(records[int(p)]))
            for p in future_merge_positions
            if np.isfinite(_merge_record_conflict_s(records[int(p)]))
        ]
        if not future_merge_positions or not conflict_s_values:
            pos = int(start_scan_pos) + 1
            continue

        go_pos = None
        go_reason = 'none'
        end_pos = None
        end_state = 'ended_route_end'
        scan_pos = int(start_pos)

        merge_area_first_conflict_s_m = float(area_info['first_conflict_s_m'])
        merge_area_last_conflict_s_m = float(area_info['last_conflict_s_m'])
        merge_area_start_s_m = float(merge_area_first_conflict_s_m)
        merge_area_end_s_m = float(merge_area_last_conflict_s_m + float(STAGE1_MERGE_AREA_POST_MARGIN_M))

        while scan_pos < num_records:
            if go_pos is None:
                has_go_signal, current_go_reason = _merge_record_go_signal(
                    records[scan_pos],
                    merge_area_start_s_m=merge_area_start_s_m,
                )
                if has_go_signal:
                    go_pos = int(scan_pos)
                    go_reason = str(current_go_reason)

            if _merge_record_passed_area(records[scan_pos], merge_area_end_s_m):
                end_pos = int(scan_pos)
                end_state = 'ended_merge_area'
                break
            scan_pos += 1

        if end_pos is None:
            end_pos = num_records - 1
        start_frame = int(records[start_pos]['frame_id'])
        end_frame = int(records[end_pos]['frame_id'])
        no_go = go_pos is None
        actor_ids, actor_switch_frames, dominant_actor_id = _merge_episode_actor_summary(records, start_pos, end_pos)

        red_light_hold_indices = {}
        red_light_hold_run = 0
        for episode_pos in range(int(start_pos), int(end_pos) + 1):
            if _merge_is_red_light_wait(records[episode_pos].get('merge_motion', {})):
                red_light_hold_run += 1
                red_light_hold_indices[int(episode_pos)] = int(red_light_hold_run)
            else:
                red_light_hold_run = 0

        merge_area_first_conflict_s_m = float(min(conflict_s_values))
        merge_area_last_conflict_s_m = float(max(conflict_s_values))
        merge_area_start_s_m = float(merge_area_first_conflict_s_m)
        merge_area_end_s_m = float(merge_area_last_conflict_s_m + float(STAGE1_MERGE_AREA_POST_MARGIN_M))

        for episode_pos in range(int(start_pos), int(end_pos) + 1):
            record = records[episode_pos]
            red_light_hold = _merge_is_red_light_wait(record.get('merge_motion', {}))
            hold_reason = 'red_light' if red_light_hold else 'none'
            if no_go:
                phase = 'yld'
            else:
                phase = 'yld' if int(episode_pos) < int(go_pos) else 'go'
            frame_role = phase
            if int(episode_pos) == int(start_pos):
                frame_role = 'start'
            if red_light_hold:
                frame_role = 'hold'
            if go_pos is not None and int(episode_pos) == int(go_pos):
                frame_role = 'go_start'
            if int(episode_pos) == int(end_pos):
                frame_role = 'end'
            merge_info = {
                'phase': str(phase),
                'phase_code': int(MERGE_DECISION_PHASE_TO_CODE[phase]),
                'episode_id': int(episode_id),
                'active': 1.0,
                'hold': float(red_light_hold),
                'hold_reason': str(hold_reason),
                'no_go': float(no_go),
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'go_frame': int(records[go_pos]['frame_id']) if go_pos is not None else -1,
                'resolution_actor_id': int(dominant_actor_id),
                'end_state': str(end_state),
                'end_state_code': int(MERGE_END_STATE_TO_CODE.get(end_state, 0)),
                'actor_ids': [int(actor_id) for actor_id in actor_ids],
                'actor_switch_frames': [int(frame_id) for frame_id in actor_switch_frames],
                'future_merge_count': int(len(future_merge_positions)),
                'merge_area_start_s_m': float(merge_area_start_s_m),
                'merge_area_end_s_m': float(merge_area_end_s_m),
                'merge_area_first_conflict_s_m': float(merge_area_first_conflict_s_m),
                'merge_area_last_conflict_s_m': float(merge_area_last_conflict_s_m),
                'frame_role': str(frame_role),
                'future_grace_index': 0,
                'red_light_hold_index': int(red_light_hold_indices.get(int(episode_pos), 0)),
                'post_go_grace_index': 0,
                'red_light_hold': float(red_light_hold),
                'grace_frames': 0,
                'actor_grace_frames': 0,
                'resolution_lookahead_frames': 0,
                'go_reason': str(go_reason),
            }
            _set_stage1_merge_annotation(samples[record['sample_idx']], merge_info)

        episode_id += 1
        pos = int(end_pos) + 1


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
    anchor_centers_abs = None
    num_modes = None
    num_points = None
    if not stage1_only:
        if anchor_path is None:
            raise ValueError("anchor_path is required unless --stage1_only is used")
        if anchor_path.endswith('.npy'):
            anchor_centers_abs = np.load(anchor_path)
        else:
            with open(anchor_path, 'rb') as f:
                anchor_centers_abs = pickle.load(f)['centers']
        num_modes = anchor_centers_abs.shape[0]
        num_points = anchor_centers_abs.shape[1]
        print(f"Anchor: {anchor_centers_abs.shape} from {anchor_path}")

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

    already_semantic = sum(1 for s in samples if 'behavior_labels' in s and 'allowed_flags' in s and 'scene_buckets' in s)
    already_energy = sum(1 for s in samples if 'energy_targets' in s and 'energy_active_mask' in s)
    already_ego_status = sum(1 for s in samples if 'ego_status' in s)
    already_stage1_speed = sum(1 for s in samples if _has_stage1_speed_fields(s))
    if stage1_only:
        num_points = _infer_num_future_points(samples)
        print(
            f"Existing fields (stage1-only): stage1_speed={already_stage1_speed}/{len(samples)}, "
            f"inferred_num_future={num_points}"
        )
        if already_stage1_speed == len(samples) and not force:
            print(f"All {len(samples)} samples already have stage1 speed fields. Use --force to re-compute.")
            return
    else:
        complete_fast_fields = sum(1 for s in samples if _has_all_fast_fields(s))
        print(
            f"Existing fields: semantic={already_semantic}/{len(samples)}, "
            f"energy={already_energy}/{len(samples)}, "
            f"ego_status={already_ego_status}/{len(samples)}, "
            f"stage1_speed={already_stage1_speed}/{len(samples)}, "
            f"complete={complete_fast_fields}/{len(samples)}"
        )
        if complete_fast_fields == len(samples) and not force:
            print(f"All {len(samples)} samples already have all fast fields. Use --force to re-compute.")
            return

    image_data_root = os.path.realpath(image_data_root)
    semantic_computed = 0
    energy_built = 0
    ego_status_built = 0
    front_route_built = 0
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
            'semantic_computed': semantic_computed,
            'energy_built': energy_built,
            'ego_status_built': ego_status_built,
            'front_route_built': front_route_built,
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
            for sample in tqdm(samples, desc="Labeling"):
                labeling_processed += 1
                needs_semantic = force or any(
                    field not in sample for field in ('behavior_labels', 'allowed_flags', 'scene_buckets')
                )
                needs_energy = force or any(
                    field not in sample for field in ('energy_targets', 'energy_active_mask')
                )
                needs_ego_status = force or ('ego_status' not in sample)
                needs_front_route = force or any(
                    field not in sample for field in (
                        'front_route_distance',
                        'front_route_ttc',
                        'front_route_risk',
                        'front_route_block_risk',
                        'front_route_case',
                        'front_route_has_lead',
                        'front_route_actor_class',
                        'front_route_actor_weight',
                        'front_route_block_bin',
                        'front_route_ttc_bin',
                        'front_route_hazard_bin',
                    )
                )

                if not (needs_semantic or needs_energy or needs_ego_status or needs_front_route):
                    skipped += 1
                    _maybe_checkpoint(phase='labeling')
                    continue

                dirty_since_checkpoint = True
                base_dir = None
                frame_str = None
                current_boxes = None
                current_measurements = None
                ego_matrix_current = None
                future_frames_data = None

                if needs_semantic:
                    base_dir, frame_str = _resolve_feature_frame_info(sample)
                    if base_dir is None or frame_str is None:
                        _set_semantic_fallback(sample, num_modes)
                        fallback += 1
                    else:
                        bev_rel = os.path.join(base_dir, 'bev_semantics', f'{frame_str}.png')
                        bev_path = os.path.join(image_data_root, bev_rel)

                        if not os.path.exists(bev_path):
                            _set_semantic_fallback(sample, num_modes)
                            fallback += 1
                        else:
                            bev_semantic = np.array(Image.open(bev_path))

                            boxes_rel = os.path.join(base_dir, 'boxes', f'{frame_str}.json.gz')
                            boxes_path = os.path.join(image_data_root, boxes_rel)
                            current_boxes = _load_json_gz_if_exists(boxes_path)

                            meas_rel = os.path.join(base_dir, 'measurements', f'{frame_str}.json.gz')
                            meas_path = os.path.join(image_data_root, meas_rel)
                            current_measurements = _load_json_gz_if_exists(meas_path)

                            if current_measurements is not None:
                                ego_matrix_current = current_measurements.get('ego_matrix', None)

                            if ego_matrix_current is not None:
                                frame_id = int(frame_str)
                                future_frames_data = []
                                for k in range(1, num_points + 1):
                                    future_frame_str = f"{frame_id + k:04d}"
                                    fut_boxes_path = os.path.join(
                                        image_data_root, base_dir,
                                        'boxes', f'{future_frame_str}.json.gz')
                                    fut_meas_path = os.path.join(
                                        image_data_root, base_dir,
                                        'measurements', f'{future_frame_str}.json.gz')
                                    if os.path.exists(fut_boxes_path) and os.path.exists(fut_meas_path):
                                        try:
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
                                        except Exception:
                                            future_frames_data.append(None)
                                    else:
                                        future_frames_data.append(None)
                            gt_traj = sample.get('ego_waypoints', None)
                            if gt_traj is not None:
                                if isinstance(gt_traj, np.ndarray):
                                    gt_traj = gt_traj[1:]
                                else:
                                    gt_traj = np.array(gt_traj)[1:]

                            behavior_labels, allowed_flags, _ = label_anchors_semantic(
                                anchor_centers_abs, bev_semantic,
                                ppm=bev_ppm, bev_size=bev_size,
                                boxes=current_boxes,
                                measurements=current_measurements,
                                ego_matrix_current=ego_matrix_current,
                                future_frames_data=future_frames_data,
                                gt_trajectory=gt_traj,
                            )

                            bucket_flags = classify_scene_buckets(
                                measurements=current_measurements,
                                boxes=current_boxes,
                                ego_waypoints=gt_traj,
                            )

                            sample['behavior_labels'] = behavior_labels
                            sample['allowed_flags'] = allowed_flags.astype(np.float32)
                            sample['scene_buckets'] = bucket_flags.astype(np.float32)
                            semantic_computed += 1

                if needs_front_route:
                    if base_dir is None or frame_str is None:
                        base_dir, frame_str = _resolve_feature_frame_info(sample)
                    if base_dir is None or frame_str is None:
                        _set_front_route_fallback(sample, front_max_distance_m, front_max_ttc_s)
                        fallback += 1
                    else:
                        if current_boxes is None:
                            boxes_rel = os.path.join(base_dir, 'boxes', f'{frame_str}.json.gz')
                            boxes_path = os.path.join(image_data_root, boxes_rel)
                            current_boxes = _load_json_gz_if_exists(boxes_path)
                        if current_measurements is None:
                            meas_rel = os.path.join(base_dir, 'measurements', f'{frame_str}.json.gz')
                            meas_path = os.path.join(image_data_root, meas_rel)
                            current_measurements = _load_json_gz_if_exists(meas_path)
                        if ego_matrix_current is None and current_measurements is not None:
                            ego_matrix_current = current_measurements.get('ego_matrix', None)
                        if future_frames_data is None and ego_matrix_current is not None:
                            frame_id = int(frame_str)
                            future_frames_data = []
                            for k in range(1, num_points + 1):
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

                        route = sample.get('route', None)
                        ego_speed = 0.0
                        if current_measurements is not None:
                            ego_speed = float(current_measurements.get('speed', 0.0))
                        elif sample.get('speed_hist', None) is not None:
                            ego_speed = float(np.asarray(sample['speed_hist'], dtype=np.float32)[-1])

                        if route is None or current_boxes is None:
                            _set_front_route_fallback(sample, front_max_distance_m, front_max_ttc_s)
                            fallback += 1
                        else:
                            front_label = _compute_front_route_label(
                                route=route,
                                current_boxes=current_boxes,
                                ego_speed=ego_speed,
                                ego_matrix_current=ego_matrix_current,
                                future_frames_data=future_frames_data,
                                corridor_margin_m=front_corridor_margin_m,
                                route_step_m=front_route_step_m,
                                max_distance_m=front_max_distance_m,
                                safe_ttc_s=front_safe_ttc_s,
                                max_ttc_s=front_max_ttc_s,
                                block_safe_distance_m=front_block_safe_distance_m,
                            )
                            sample['front_route_distance'] = np.float32(front_label['distance'])
                            sample['front_route_ttc'] = np.float32(front_label['ttc'])
                            sample['front_route_risk'] = np.float32(front_label['risk'])
                            sample['front_route_block_risk'] = np.float32(front_label['block_risk'])
                            sample['front_route_case'] = np.int64(front_label['case'])
                            sample['front_route_has_lead'] = np.float32(front_label['has_lead'])
                            sample['front_route_actor_class'] = np.int64(front_label['actor_class'])
                            sample['front_route_actor_weight'] = np.float32(front_label['actor_weight'])
                            sample['front_route_block_bin'] = np.int64(front_label['block_bin'])
                            sample['front_route_ttc_bin'] = np.int64(front_label['ttc_bin'])
                            sample['front_route_hazard_bin'] = np.int64(front_label['hazard_bin'])
                            front_route_built += 1

                if needs_energy:
                    if 'behavior_labels' not in sample or 'allowed_flags' not in sample:
                        _set_semantic_fallback(sample, num_modes)
                    energy_targets, energy_active_mask = _build_energy_targets(
                        sample['behavior_labels'],
                        sample['allowed_flags'],
                    )
                    sample['energy_targets'] = energy_targets
                    sample['energy_active_mask'] = energy_active_mask
                    energy_built += 1

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
            tqdm(route_groups.items(), desc="Stage1 speed", leave=False),
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

            persisted_meet = None
            persisted_meet_frames_left = 0
            persist_dt_s = 0.25
            cross_wait_state = 0
            cross_wait_frames = 0
            cross_episode_active = False

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
                speed_curve_future_cover = dict(future_cover)
                speed_curve_future_persisted = False

                if int(future_cover.get('exists', 0.0)) > 0 and future_cover['interaction']['name'] == 'meet':
                    persisted_meet = dict(future_cover)
                    persisted_meet['actor_id'] = int(future_cover.get('actor_id', -1))
                    persisted_meet_frames_left = STAGE1_PERSISTED_MEET_FRAMES
                elif persisted_meet is not None and persisted_meet_frames_left > 0:
                    actor_id = int(persisted_meet.get('actor_id', -1))
                    actor_box = _find_box_by_id(current_boxes_dynamic, actor_id)
                    ego_box = _find_ego_box(current_boxes_dynamic)
                    ego_speed_now = float(current_measurements.get('speed', 0.0))
                    bg_speed_now = float(persisted_meet.get('other_speed', np.nan))
                    if actor_box is not None:
                        bg_speed_now = float(abs(actor_box.get('speed', bg_speed_now)))
                    pseudo = dict(persisted_meet)
                    keep_persisted = False
                    if actor_box is not None:
                        pos = np.asarray(actor_box.get('position', [np.nan, np.nan])[:2], dtype=np.float32)
                        if pos.shape == (2,) and np.all(np.isfinite(pos)):
                            if float(pos[0]) < 1.0 and float(np.linalg.norm(pos)) < 12.0:
                                pseudo['d_ego'] = 0.0
                                pseudo['d_bg'] = float(max(np.linalg.norm(pos), 1e-3))
                                pseudo['rear_gap_m'] = float(_approx_box_clearance_gap_m(actor_box, ego_box))
                                keep_persisted = True
                    if not keep_persisted:
                        if np.isfinite(float(pseudo.get('d_ego', np.nan))):
                            pseudo['d_ego'] = float(max(float(pseudo['d_ego']) - ego_speed_now * persist_dt_s, 0.0))
                        if np.isfinite(float(pseudo.get('d_bg', np.nan))):
                            pseudo['d_bg'] = float(max(float(pseudo['d_bg']) - bg_speed_now * persist_dt_s, 0.0))
                        keep_persisted = bool(np.isfinite(float(pseudo.get('d_bg', np.nan))) and float(pseudo.get('d_bg', 0.0)) > 0.25)
                    if keep_persisted:
                        pseudo['other_speed'] = float(bg_speed_now)
                        speed_curve_future_cover = pseudo
                        speed_curve_future_persisted = True
                        persisted_meet = dict(pseudo)
                        persisted_meet_frames_left -= 1
                    else:
                        persisted_meet = None
                        persisted_meet_frames_left = 0

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
                speed_curve_future_cover = _augment_cover_with_scene_route_fields(
                    speed_curve_future_cover,
                    ego_matrix_current=ego_matrix_current,
                    scene_route_polyline_world=scene_route_world,
                )

                raw_cross_active = _cover_is_cross_meet(current_cover) or _cover_is_cross_meet(speed_curve_future_cover)
                if not cross_episode_active and raw_cross_active:
                    cross_episode_active = True
                borrow_corridor = _borrow_corridor_metrics(
                    scene_borrow_context,
                    current_meas=current_measurements,
                    route_local=route_input,
                )
                borrow_motion = _build_borrow_motion_context(
                    scene_borrow_context,
                    current_meas=current_measurements,
                    route_local=route_input,
                    frame_id=int(sample.get('frame_id', -1)),
                )

                def _cross_start_distance_from_cover(cover):
                    if not _cover_is_cross_meet(cover):
                        return np.nan
                    subtype = _cover_interaction_subtype(cover)
                    if subtype == "borrow_cross_meet" and borrow_corridor is not None:
                        return float(borrow_corridor.get("borrow_start_distance_m", np.nan))
                    d_ego = cover.get("distance", cover.get("d_ego", np.nan))
                    return float(d_ego) if np.isfinite(float(d_ego)) else np.nan

                cross_start_distance = np.nan
                dist_current = _cross_start_distance_from_cover(current_cover)
                dist_future = _cross_start_distance_from_cover(speed_curve_future_cover)
                if np.isfinite(dist_current) and np.isfinite(dist_future):
                    cross_start_distance = float(min(dist_current, dist_future))
                elif np.isfinite(dist_current):
                    cross_start_distance = float(dist_current)
                elif np.isfinite(dist_future):
                    cross_start_distance = float(dist_future)

                near_cross_start = (
                    np.isfinite(cross_start_distance) and
                    float(cross_start_distance) <= float(STAGE1_CROSS_START_DISTANCE_M)
                )
                go_now = False
                if cross_episode_active:
                    if cross_wait_state == 0:
                        if near_cross_start and ego_speed <= float(STAGE1_CROSS_WAIT_SPEED_THRESH):
                            cross_wait_state = 1
                            cross_wait_frames = 0
                    if cross_wait_state == 1:
                        if ego_speed >= float(STAGE1_CROSS_GO_SPEED_THRESH):
                            go_now = True
                            cross_wait_state = 2
                        else:
                            cross_wait_frames += 1
                cross_wait_time_s = float(cross_wait_frames) * float(STAGE1_CROSS_WAIT_DT_S)
                cross_wait_valid = 1.0 if (cross_wait_state == 1 or go_now) else 0.0
                if go_now:
                    cross_episode_active = False
                    cross_wait_state = 0
                    cross_wait_frames = 0

                ped_margin_m = _pedestrian_corridor_margin_m(current_boxes)
                ped_current_boxes = _filter_current_boxes_pedestrian(current_boxes)
                ped_future_frames = _filter_future_frames_pedestrian(future_frames_data)
                _, ped_debug = _compute_front_route_label(
                    route=route_input,
                    current_boxes=ped_current_boxes,
                    ego_speed=ego_speed,
                    ego_matrix_current=ego_matrix_current,
                    future_frames_data=ped_future_frames,
                    corridor_margin_m=ped_margin_m,
                    route_step_m=front_route_step_m,
                    max_distance_m=front_max_distance_m,
                    safe_ttc_s=front_safe_ttc_s,
                    max_ttc_s=front_max_ttc_s,
                    block_safe_distance_m=front_block_safe_distance_m,
                    return_debug=True,
                )
                ped_current_cover = _cover_candidate_summary(1, ped_debug.get('best_current'), ped_debug, current_meas=current_measurements, event_name=event_name)
                ped_future_cover = _cover_candidate_summary(2, ped_debug.get('best_future'), ped_debug, current_meas=current_measurements, event_name=event_name)

                (
                    sample_speeds,
                    valid_mask,
                    exp_index,
                    chase_risks,
                    meet_risks,
                    _,
                    junction_cross_yld_risks,
                    junction_cross_go_risks,
                    merge_yld_risks,
                    merge_go_risks,
                    borrow_yld_risks,
                    borrow_go_risks,
                    speed_curve_debug,
                ) = _build_speed_curve_targets(
                    current_cover=current_cover,
                    future_cover=speed_curve_future_cover,
                    current_meas=current_measurements,
                    current_boxes=current_boxes_dynamic,
                    route_local=route_input,
                    release_info=scene_borrow_context,
                    event_name=event_name,
                    return_debug=True,
                )
                (
                    ped_sample_speeds,
                    _,
                    _,
                    _,
                    _,
                    ped_risks,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = _build_speed_curve_targets(
                    current_cover=ped_current_cover,
                    future_cover=ped_future_cover,
                    current_meas=current_measurements,
                    current_boxes=current_boxes,
                    route_local=route_input,
                    release_info=scene_borrow_context,
                    event_name=event_name,
                )
                if ped_sample_speeds.shape != sample_speeds.shape or not np.allclose(ped_sample_speeds, sample_speeds):
                    ped_risks = np.zeros(sample_speeds.shape, dtype=np.float32)

                sample['speed_sample_values'] = sample_speeds.astype(np.float32)
                sample['speed_sample_valid_mask'] = valid_mask.astype(np.float32)
                sample['speed_sample_exp_index'] = np.int64(exp_index)
                sample['speed_risk_chase_values'] = chase_risks.astype(np.float32)
                sample['speed_risk_meet_values'] = meet_risks.astype(np.float32)
                sample['speed_risk_junction_cross_yld_values'] = junction_cross_yld_risks.astype(np.float32)
                sample['speed_risk_junction_cross_go_values'] = junction_cross_go_risks.astype(np.float32)
                sample['speed_risk_merge_yld_values'] = merge_yld_risks.astype(np.float32)
                sample['speed_risk_merge_go_values'] = merge_go_risks.astype(np.float32)
                sample['speed_risk_borrow_yld_values'] = borrow_yld_risks.astype(np.float32)
                sample['speed_risk_borrow_go_values'] = borrow_go_risks.astype(np.float32)
                sample['speed_risk_ped_values'] = ped_risks.astype(np.float32)
                sample['speed_cross_wait_time_s'] = np.float32(cross_wait_time_s)
                sample['speed_cross_wait_valid'] = np.float32(cross_wait_valid)
                speed_curve_debug['junction_cross_yld_risks'] = np.asarray(junction_cross_yld_risks, dtype=np.float32)
                speed_curve_debug['junction_cross_go_risks'] = np.asarray(junction_cross_go_risks, dtype=np.float32)
                speed_curve_debug['merge_yld_risks'] = np.asarray(merge_yld_risks, dtype=np.float32)
                speed_curve_debug['merge_go_risks'] = np.asarray(merge_go_risks, dtype=np.float32)
                speed_curve_debug['borrow_yld_risks'] = np.asarray(borrow_yld_risks, dtype=np.float32)
                speed_curve_debug['borrow_go_risks'] = np.asarray(borrow_go_risks, dtype=np.float32)
                speed_curve_debug['ped_risks'] = np.asarray(ped_risks, dtype=np.float32)
                speed_curve_debug['cross_wait_time_s'] = np.float32(cross_wait_time_s)
                speed_curve_debug['cross_wait_valid'] = np.float32(cross_wait_valid)
                speed_curve_debug['total_risks'] = np.maximum(
                    np.maximum(np.asarray(chase_risks, dtype=np.float32), np.asarray(meet_risks, dtype=np.float32)),
                    np.asarray(ped_risks, dtype=np.float32),
                ).astype(np.float32)
                sample['stage1_speed_debug'] = _build_stage1_speed_debug_payload(
                    current_cover=current_cover,
                    future_cover=future_cover,
                    ped_current_cover=ped_current_cover,
                    ped_future_cover=ped_future_cover,
                    speed_curve_future_cover=speed_curve_future_cover,
                    speed_curve_future_persisted=speed_curve_future_persisted,
                    speed_curve_debug=speed_curve_debug,
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

            _annotate_route_stage1_borrow_cross_decisions(samples, route_sample_indices)
            _annotate_route_stage1_junction_cross_decisions(samples, route_sample_indices)
            _annotate_route_stage1_merge_decisions(samples, route_sample_indices)
            _gate_route_stage1_junction_speed_risks(samples, route_sample_indices)
            _gate_route_stage1_borrow_speed_risks(samples, route_sample_indices)
            _gate_route_stage1_merge_speed_risks(samples, route_sample_indices)
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
        f"\nDone: semantic={semantic_computed}, energy={energy_built}, "
        f"ego_status={ego_status_built}, front_route={front_route_built}, "
        f"stage1_speed={stage1_speed_built}, "
        f"skipped={skipped}, fallback={fallback}"
    )

    _save_checkpoint(phase='final', reason='complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-compute fast training fields into samples_packed.pkl")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset split (e.g. /media/z/data/dataset/pdm_lite_mini/train)')
    parser.add_argument('--image_data_root', type=str, required=True,
                        help='Root of image data (e.g. /media/z/data/dataset/pdm_lite_mini)')
    parser.add_argument('--anchor_path', type=str, default=None,
                        help='Path to anchor file (.npy or .pkl). Required unless --stage1_only is used.')
    parser.add_argument('--bev_ppm', type=float, default=2.0)
    parser.add_argument('--bev_size', type=int, default=256)
    parser.add_argument('--front_corridor_margin_m', type=float, default=0.5)
    parser.add_argument('--front_route_step_m', type=float, default=0.25)
    parser.add_argument('--front_max_distance_m', type=float, default=40.0)
    parser.add_argument('--front_safe_ttc_s', type=float, default=3.0)
    parser.add_argument('--front_max_ttc_s', type=float, default=10.0)
    parser.add_argument('--front_block_safe_distance_m', type=float, default=30.0)
    parser.add_argument('--checkpoint_every_minutes', type=float, default=20.0,
                        help='Periodically atomically save updated samples_packed.pkl to make long runs resumable. Set <=0 to disable.')
    parser.add_argument('--stage1_only', action='store_true',
                        help='Skip old semantic/energy/ego/front_route labeling and only (re)compute stage1 speed-energy labels.')
    parser.add_argument('--stage1_timing', action='store_true',
                        help='Print lightweight stage1 timing summaries (load/front_route/route_total).')
    parser.add_argument('--stage1_timing_every', type=int, default=1,
                        help='When --stage1_timing is set, print every N route groups.')
    parser.add_argument('--force', action='store_true', help='Re-compute even if labels exist')
    args = parser.parse_args()

    if not args.stage1_only and not args.anchor_path:
        parser.error('--anchor_path is required unless --stage1_only is used')

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
