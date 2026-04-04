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
    'speed_risk_ped_values',
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
INTERACTION_SAME_DIR_ANGLE_THRESH_DEG = 45.0
MERGE_DEBUG_MIN_DEGO_M = 1.0
MERGE_DEBUG_MIN_GO_DENOM_S = 0.10
NO_ROUTE_EXTENSION_SCENES = {'HazardAtSideLane'}
STAGE1_SPEED_OFFSETS_MPS = np.asarray([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0], dtype=np.float32)


def _has_all_fast_fields(sample):
    return all(field in sample for field in FAST_FIELDS)


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


def _find_route_cover_point(route_dense, route_s, box, corridor_margin_m):
    actor_class_id = _canonical_actor_class(box)
    if actor_class_id == ACTOR_CLASS_NONE:
        return None

    pos = box.get('position', None)
    extent = box.get('extent', None)
    if pos is None or extent is None or len(pos) < 2 or len(extent) < 2:
        return None

    mask = _points_inside_oriented_box(
        route_dense,
        center=np.asarray(pos[:2], dtype=np.float32),
        extent=np.asarray(extent[:2], dtype=np.float32),
        yaw=float(box.get('yaw', 0.0)),
        margin_m=corridor_margin_m,
    )
    if not np.any(mask):
        return None

    first_idx = int(np.flatnonzero(mask)[0])
    return {
        'route_idx': first_idx,
        'route_point': route_dense[first_idx],
        'route_distance': float(route_s[first_idx]),
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

    # Case 1: a vehicle already covers the current route -> pursuit / following problem.
    best_current = None
    for box in current_boxes:
        cover = _find_route_cover_point(route_dense, route_s, box, corridor_margin_m)
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

            cover = _find_route_cover_point(route_dense, route_s, box_cur, corridor_margin_m)
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
            d_bg = _point_to_oriented_box_distance(conflict_pt, bg_box, margin_m=0.0)
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
    return str(event_name or "") in {"AccidentTwoWays", "ParkedObstacleTwoWays"}


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
        return {
            "mode": 0,
            "name": "none",
            "subtype": "none",
            "source": "none",
            "angle_deg": np.nan,
            "route_heading_deg": np.nan,
            "actor_heading_deg": np.nan,
            "motion_m": 0.0,
        }

    cover = best.get("cover", {})
    actor_box = best.get("box") if case == 1 else (
        best.get("current_box") or best.get("box_future") or best.get("box_current_frame")
    )
    if _is_pedestrian_box(actor_box):
        return {
            "mode": 2,
            "name": "meet",
            "subtype": "ped_cross",
            "source": "ped_corridor_cover",
            "angle_deg": np.nan,
            "route_heading_deg": np.nan,
            "actor_heading_deg": np.nan,
            "motion_m": 0.0,
        }

    route_heading = _route_heading_at_idx(debug.get("route_dense"), cover.get("route_idx", 0))
    if route_heading is None:
        return {
            "mode": 0,
            "name": "none",
            "subtype": "none",
            "source": "missing_route_heading",
            "angle_deg": np.nan,
            "route_heading_deg": np.nan,
            "actor_heading_deg": np.nan,
            "motion_m": 0.0,
        }

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
        if case == 2 and _is_right_turn_junction_context(current_meas):
            return {
                "mode": 2,
                "name": "meet",
                "subtype": "merge_meet",
                "source": "junction_right_future_cover_override",
                "angle_deg": np.nan,
                "route_heading_deg": _heading_to_deg(route_heading),
                "actor_heading_deg": np.nan,
                "motion_m": motion_m,
            }
        return {
            "mode": 0,
            "name": "none",
            "subtype": "none",
            "source": "missing_actor_heading",
            "angle_deg": np.nan,
            "route_heading_deg": _heading_to_deg(route_heading),
            "actor_heading_deg": np.nan,
            "motion_m": motion_m,
        }

    angle_deg = abs(_heading_to_deg(actor_heading - route_heading))
    same_direction = angle_deg <= float(angle_thresh_deg)
    if case == 2 and _is_right_turn_junction_context(current_meas):
        return {
            "mode": 2,
            "name": "meet",
            "subtype": "merge_meet",
            "source": "junction_right_future_cover_override",
            "angle_deg": float(angle_deg),
            "route_heading_deg": _heading_to_deg(route_heading),
            "actor_heading_deg": _heading_to_deg(actor_heading),
            "motion_m": float(motion_m),
        }
    if same_direction and case == 1:
        return {
            "mode": 1,
            "name": "chase",
            "subtype": "follow_chase",
            "source": f"{source}+same_dir_current_cover",
            "angle_deg": float(angle_deg),
            "route_heading_deg": _heading_to_deg(route_heading),
            "actor_heading_deg": _heading_to_deg(actor_heading),
            "motion_m": float(motion_m),
        }

    subtype = "cross_meet"
    source = f"{source}+cross_dir"
    if same_direction and case == 2:
        subtype = "merge_meet"
        source = f"{source}+same_dir_future_cover"
    elif case == 1 and _is_left_turn_scene_context(current_meas, event_name=event_name):
        subtype = "junction_left_cross_meet"
        source = f"{source}+junction_left_cross_current"
    elif case == 1 and _is_borrow_cross_scene_context(event_name=event_name):
        subtype = "borrow_cross_meet"
        source = f"{source}+borrow_cross_current"
    elif case == 2 and _is_left_turn_scene_context(current_meas, event_name=event_name):
        subtype = "junction_left_cross_meet"
        source = f"{source}+junction_left_cross"
    elif case == 2 and _is_borrow_cross_scene_context(event_name=event_name):
        subtype = "borrow_cross_meet"
        source = f"{source}+borrow_cross"
    return {
        "mode": 2,
        "name": "meet",
        "subtype": subtype,
        "source": source,
        "angle_deg": float(angle_deg),
        "route_heading_deg": _heading_to_deg(route_heading),
        "actor_heading_deg": _heading_to_deg(actor_heading),
        "motion_m": float(motion_m),
    }


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
        }

    if case == 1:
        box = best.get("box", {})
        actor_id = int(box.get("id", -1)) if box.get("id", None) is not None else -1
        gap_distance = float(best.get("gap_distance", np.nan))
        closing_speed = float(best.get("closing_speed", np.nan))
        ttc = np.inf if not np.isfinite(closing_speed) or closing_speed <= 1e-6 else gap_distance / closing_speed
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
        }

    current_box = best.get("current_box") or {}
    box_future = best.get("box_future") or {}
    actor_id = current_box.get("id", None)
    if actor_id is None:
        actor_id = box_future.get("id", None)
    actor_id = int(actor_id) if actor_id is not None else -1
    meet_dist = float(best.get("meet_dist", np.nan))
    meet_speed = float(best.get("meet_speed", np.nan))
    ttc = np.inf if not np.isfinite(meet_speed) or meet_speed <= 1e-6 else meet_dist / meet_speed
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
    }


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
        for box in boxes:
            cls = str(box.get("class", "")).lower()
            if cls in {"ego_car", "static"}:
                continue
            actor_id = box.get("id", None)
            pos = box.get("position", None)
            if actor_id is None or pos is None or len(pos) < 2:
                continue
            actor_id = int(actor_id)
            pos_xy = np.asarray(pos[:2], dtype=np.float32)
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
    extension_world = _sample_polyline_at_arclengths(scene_polyline_world[:, :2], query_s)
    if extension_world.shape[0] == 0:
        return route_local
    extension_world_xyz = np.concatenate([extension_world, np.zeros((extension_world.shape[0], 1), dtype=np.float32)], axis=1)
    extension_local = _transform_points_world_xyz_to_local(extension_world_xyz, ego_matrix_current)
    if extension_local.shape[0] == 0:
        return route_local
    merged = np.concatenate([route_local[:, :2], extension_local], axis=0)
    return _dedupe_polyline(merged, min_step_m=max(0.25, 0.5 * float(extension_step_m)))


def _build_speed_curve_targets(
    current_cover,
    future_cover,
    current_meas,
    current_boxes=None,
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
):
    speed = float((current_meas or {}).get("speed", 0.0))
    sample_speeds = _speed_risk_samples_fixed(speed)
    valid_mask = np.ones(sample_speeds.shape, dtype=np.float32)
    exp_index = np.int64(3)
    chase_risks = np.zeros(sample_speeds.shape, dtype=np.float32)
    meet_risks = np.zeros(sample_speeds.shape, dtype=np.float32)
    ped_risks = np.zeros(sample_speeds.shape, dtype=np.float32)

    ego_length_m = _ego_length_m(current_boxes)
    left_junction_conflict_len_m = _left_junction_conflict_len_m(
        current_meas, current_boxes, scale=junction_left_conflict_len_scale, event_name=event_name
    )

    if int(current_cover.get("exists", 0.0)) > 0 and str(current_cover.get("actor_class_name", "")) in PEDESTRIAN_CLASSES:
        ped_distance = max(float(current_cover.get("distance", np.nan)), 0.0)
        if np.isfinite(ped_distance):
            for idx, candidate_speed in enumerate(sample_speeds):
                v = float(candidate_speed)
                if v <= 1e-6:
                    ped_risks[idx] = 0.0
                    continue
                t_cover = ped_distance / max(v, 1e-6)
                ped_risks[idx] = float(np.clip((float(ped_safe_ttc_s) - t_cover) / max(float(ped_safe_ttc_s) - float(ped_hard_ttc_s), 1e-6), 0.0, 1.0))
            return sample_speeds, valid_mask, exp_index, chase_risks, meet_risks, ped_risks

    if int(future_cover.get("exists", 0.0)) > 0 and str(future_cover.get("actor_class_name", "")) in PEDESTRIAN_CLASSES:
        d_ego = float(future_cover.get("d_ego", np.nan))
        if not np.isfinite(d_ego):
            d_ego = float(future_cover.get("distance", np.nan))
        ped_distance = max(d_ego, 0.0)
        if np.isfinite(ped_distance):
            for idx, candidate_speed in enumerate(sample_speeds):
                v = float(candidate_speed)
                if v <= 1e-6:
                    ped_risks[idx] = 0.0
                    continue
                t_cover = ped_distance / max(v, 1e-6)
                ped_risks[idx] = float(np.clip((float(ped_safe_ttc_s) - t_cover) / max(float(ped_safe_ttc_s) - float(ped_hard_ttc_s), 1e-6), 0.0, 1.0))
            return sample_speeds, valid_mask, exp_index, chase_risks, meet_risks, ped_risks

    if int(current_cover.get("exists", 0.0)) > 0 and current_cover["interaction"]["name"] == "chase":
        gap_m = float(current_cover.get("distance", np.nan))
        lead_speed_mps = float(current_cover.get("other_speed", np.nan))
        if np.isfinite(gap_m) and np.isfinite(lead_speed_mps):
            for idx, candidate_speed in enumerate(sample_speeds):
                v = float(candidate_speed)
                closing = v - lead_speed_mps
                ttc = np.inf if closing <= 1e-6 else gap_m / max(closing, 1e-6)
                safe_gap = float(chase_follow_base_gap_m + chase_follow_headway_s * max(v, 0.0))
                gap_risk = float(np.clip((safe_gap - gap_m) / max(safe_gap, 1e-6), 0.0, 1.0))
                ttc_risk = _risk_from_time_gap(ttc, safe_ttc_s)
                if closing > 1e-6:
                    closing_speed_risk = float(np.clip(closing / max(v, 1.0), 0.0, 1.0))
                    distance_weight = float(safe_gap / max(gap_m + safe_gap, 1e-6))
                    closing_trend_risk = float(np.clip(closing_speed_risk * distance_weight, 0.0, 1.0))
                else:
                    closing_trend_risk = 0.0
                chase_risks[idx] = max(gap_risk, ttc_risk, closing_trend_risk)

    if int(current_cover.get("exists", 0.0)) > 0 and current_cover["interaction"]["name"] == "meet":
        d_ego = float(current_cover.get("distance", np.nan))
        bg_speed = float(current_cover.get("other_speed", np.nan))
        meet_subtype = str(current_cover.get("interaction", {}).get("subtype", "none"))
        if np.isfinite(d_ego):
            conflict_len_m = np.nan
            if meet_subtype == "junction_left_cross_meet":
                conflict_len_m = float(left_junction_conflict_len_m)
            elif "cross" in meet_subtype and meet_subtype != "borrow_cross_meet":
                conflict_len_m = float(max(ego_length_m, 1.0))
            if np.isfinite(conflict_len_m):
                d_ego_entry_m = float(max(float(d_ego) - 0.5 * float(conflict_len_m), 0.0))
            else:
                d_ego_entry_m = float(max(float(d_ego), 0.0))
            if np.isfinite(conflict_len_m):
                current_cover_dist_horizon_m = float(max(15.0, 2.0 * float(conflict_len_m)))
            else:
                current_cover_dist_horizon_m = 15.0
            occupancy_floor = float(np.clip((current_cover_dist_horizon_m - float(d_ego_entry_m)) / max(current_cover_dist_horizon_m, 1e-6), 0.02, 1.0))
            current_cover_time_horizon_s = float(max(float(cross_safe_gap_s), 3.0))
            for idx, candidate_speed in enumerate(sample_speeds):
                v = float(candidate_speed)
                if v <= 1e-6:
                    meet_risks[idx] = 0.0
                    continue
                t_ego_in = float(max(d_ego_entry_m, 0.0) / max(v, 1e-6))
                if float(d_ego_entry_m) <= 1e-4:
                    risk = 1.0
                else:
                    risk = float(np.clip((current_cover_time_horizon_s - t_ego_in) / max(current_cover_time_horizon_s, 1e-6), 0.0, 1.0))
                meet_risks[idx] = float(max(risk, occupancy_floor))
            return sample_speeds, valid_mask, exp_index, chase_risks, meet_risks, ped_risks

    if int(future_cover.get("exists", 0.0)) > 0 and future_cover["interaction"]["name"] == "meet":
        d_ego = float(future_cover.get("d_ego", np.nan))
        d_bg = float(future_cover.get("d_bg", np.nan))
        bg_speed = float(future_cover.get("other_speed", np.nan))
        meet_subtype = str(future_cover.get("interaction", {}).get("subtype", "none"))
        if np.isfinite(d_ego) and np.isfinite(d_bg) and np.isfinite(bg_speed) and d_bg > 1e-4 and bg_speed > 1e-4:
            conflict_len_m = np.nan
            risk_d_ego = float(d_ego)
            risk_d_bg = float(d_bg)
            if meet_subtype == "junction_left_cross_meet":
                conflict_len_m = float(left_junction_conflict_len_m)
                conflict_half_m = 0.5 * conflict_len_m
                risk_d_ego = max(float(d_ego) - conflict_half_m, 0.0)
                risk_d_bg = max(float(d_bg) - conflict_half_m, 0.0)
            t_bg = risk_d_bg / max(bg_speed, 1e-6)
            t_bg_exit = (risk_d_bg + max(float(conflict_len_m) if np.isfinite(conflict_len_m) else 0.0, 0.0)) / max(bg_speed, 1e-6)
            safe_gap_bg = float(merge_follow_base_gap_m + merge_follow_headway_s * max(bg_speed, 0.0))
            rear_gap_m = float(future_cover.get("rear_gap_m", np.nan))
            if meet_subtype == "merge_meet":
                v_behind_min = max(float(bg_speed), 0.0)
                if d_ego <= 0.25:
                    v_equal = np.nan
                    v_go_min = np.nan
                    v_go_need = float(v_behind_min)
                    v_yield_max = np.nan
                else:
                    v_equal = d_ego / max(t_bg, 1e-6)
                    go_denom = t_bg - float(merge_tau_s)
                    v_go_min = np.inf if go_denom <= 1e-6 else d_ego / max(go_denom, 1e-6)
                    v_go_need = max(float(v_go_min), float(v_behind_min))
                    v_yield_max = d_ego / max(t_bg + float(merge_tau_s), 1e-6)
                for idx, candidate_speed in enumerate(sample_speeds):
                    v = float(candidate_speed)
                    if v <= 1e-6:
                        meet_risks[idx] = 0.0
                        continue
                    if not np.isfinite(v_go_need):
                        risk = 0.0
                    elif not np.isfinite(v_yield_max):
                        rear_gap = rear_gap_m if np.isfinite(rear_gap_m) else d_bg
                        closing_rear = max(bg_speed - v, 0.0)
                        if closing_rear <= 1e-6:
                            risk = 0.0
                        else:
                            rear_ttc = rear_gap / max(closing_rear, 1e-6)
                            risk = float(np.clip((float(rear_safe_ttc_s) - rear_ttc) / max(float(rear_safe_ttc_s) - float(rear_hard_ttc_s), 1e-6), 0.0, 1.0))
                    elif not np.isfinite(v_equal):
                        risk = float(np.clip((v_go_need - v) / max(v_go_need - v_yield_max, 1e-6), 0.0, 1.0))
                    else:
                        if v_go_need <= v_yield_max:
                            risk = 0.0
                        else:
                            peak_speed = max(float(v_equal), float(v_yield_max))
                            if v <= float(v_yield_max):
                                risk = 0.0
                            elif peak_speed <= float(v_yield_max) + 1e-6:
                                risk = float(np.clip((v_go_need - v) / max(v_go_need - v_yield_max, 1e-6), 0.0, 1.0))
                            elif v <= peak_speed:
                                risk = float(np.clip((v - v_yield_max) / max(peak_speed - v_yield_max, 1e-6), 0.0, 1.0))
                            elif v < v_go_need:
                                risk = float(np.clip((v_go_need - v) / max(v_go_need - peak_speed, 1e-6), 0.0, 1.0))
                            else:
                                risk = 0.0
                    meet_risks[idx] = float(np.clip(np.nan_to_num(risk, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
            else:
                for idx, candidate_speed in enumerate(sample_speeds):
                    v = float(candidate_speed)
                    if v <= 1e-6:
                        meet_risks[idx] = 0.0
                        continue
                    t_ego_in = risk_d_ego / max(v, 1e-6)
                    if np.isfinite(conflict_len_m) and float(conflict_len_m) > 1e-6:
                        t_ego_out = (risk_d_ego + float(conflict_len_m)) / max(v, 1e-6)
                        gap_before = t_bg - t_ego_out
                        gap_after = t_ego_in - t_bg_exit
                        time_clearance = max(gap_before, gap_after)
                        risk = float(np.clip((float(cross_safe_gap_s) - time_clearance) / max(float(cross_safe_gap_s), 1e-6), 0.0, 1.0))
                    else:
                        delta_t = abs(t_ego_in - t_bg)
                        risk = _risk_from_time_gap(delta_t, merge_tau_s)
                    meet_risks[idx] = float(np.clip(np.nan_to_num(risk, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))

    return sample_speeds, valid_mask, exp_index, chase_risks, meet_risks, ped_risks


def _has_stage1_speed_fields(sample):
    return all(
        field in sample for field in (
            'speed_sample_values',
            'speed_sample_valid_mask',
            'speed_sample_exp_index',
            'speed_risk_chase_values',
            'speed_risk_meet_values',
            'speed_risk_ped_values',
        )
    )


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
    sample['speed_risk_ped_values'] = np.zeros(sample_speeds.shape, dtype=np.float32)


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
):
    # Load anchors
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
    with open(packed_path, 'rb') as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples")

    complete_fast_fields = sum(1 for s in samples if _has_all_fast_fields(s))
    already_semantic = sum(1 for s in samples if 'behavior_labels' in s and 'allowed_flags' in s and 'scene_buckets' in s)
    already_energy = sum(1 for s in samples if 'energy_targets' in s and 'energy_active_mask' in s)
    already_ego_status = sum(1 for s in samples if 'ego_status' in s)
    already_stage1_speed = sum(1 for s in samples if _has_stage1_speed_fields(s))
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

    for sample in tqdm(samples, desc="Labeling"):
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
            continue

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

    route_groups = defaultdict(list)
    for sample_idx, sample in enumerate(samples):
        if not force and _has_stage1_speed_fields(sample):
            continue
        base_dir, frame_str = _resolve_feature_frame_info(sample)
        if base_dir is None or frame_str is None:
            route_groups[None].append(sample_idx)
            continue
        route_groups[base_dir].append(sample_idx)

    for base_dir, route_sample_indices in tqdm(route_groups.items(), desc="Stage1 speed", leave=False):
        if base_dir is None:
            for sample_idx in route_sample_indices:
                _set_stage1_speed_fallback(samples[sample_idx])
                stage1_speed_built += 1
                fallback += 1
            continue

        route_sample_indices = sorted(route_sample_indices, key=lambda i: int(samples[i].get('frame_id', -1)))
        route_samples = [samples[i] for i in route_sample_indices]
        event_name = _scene_name_from_base_dir(base_dir)
        scene_nonstatic_actor_ids = _collect_scene_nonstatic_actor_ids(route_samples, image_data_root)
        scene_route_polyline_world, scene_route_anchor_s = _build_scene_route_polyline_world(route_samples, image_data_root)

        persisted_meet = None
        persisted_meet_frames_left = 0
        persist_dt_s = 0.25

        for sample_idx in route_sample_indices:
            sample = samples[sample_idx]
            base_dir_cur, frame_str = _resolve_feature_frame_info(sample)
            if base_dir_cur is None or frame_str is None:
                _set_stage1_speed_fallback(sample)
                stage1_speed_built += 1
                fallback += 1
                continue

            boxes_path = os.path.join(image_data_root, base_dir_cur, 'boxes', f'{frame_str}.json.gz')
            meas_path = os.path.join(image_data_root, base_dir_cur, 'measurements', f'{frame_str}.json.gz')
            current_boxes = _load_json_gz_if_exists(boxes_path)
            current_measurements = _load_json_gz_if_exists(meas_path)
            if current_boxes is None or current_measurements is None:
                _set_stage1_speed_fallback(sample)
                stage1_speed_built += 1
                fallback += 1
                continue

            ego_matrix_current = current_measurements.get('ego_matrix', None)
            if ego_matrix_current is None:
                _set_stage1_speed_fallback(sample)
                stage1_speed_built += 1
                fallback += 1
                continue

            route_local = np.asarray(sample.get('route', np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
            if route_local.ndim != 2 or route_local.shape[0] == 0 or route_local.shape[1] != 2:
                _set_stage1_speed_fallback(sample)
                stage1_speed_built += 1
                fallback += 1
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
            current_cover = _cover_candidate_summary(1, front_debug.get('best_current'), front_debug, current_meas=current_measurements, event_name=event_name)
            future_cover = _cover_candidate_summary(2, front_debug.get('best_future'), front_debug, current_meas=current_measurements, event_name=event_name)
            speed_curve_future_cover = dict(future_cover)

            if int(future_cover.get('exists', 0.0)) > 0 and future_cover['interaction']['name'] == 'meet':
                persisted_meet = dict(future_cover)
                persisted_meet['actor_id'] = int(future_cover.get('actor_id', -1))
                persisted_meet_frames_left = 4
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
                    persisted_meet = dict(pseudo)
                    persisted_meet_frames_left -= 1
                else:
                    persisted_meet = None
                    persisted_meet_frames_left = 0

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

            sample_speeds, valid_mask, exp_index, chase_risks, meet_risks, _ = _build_speed_curve_targets(
                current_cover=current_cover,
                future_cover=speed_curve_future_cover,
                current_meas=current_measurements,
                current_boxes=current_boxes_dynamic,
                event_name=event_name,
            )
            ped_sample_speeds, _, _, _, _, ped_risks = _build_speed_curve_targets(
                current_cover=ped_current_cover,
                future_cover=ped_future_cover,
                current_meas=current_measurements,
                current_boxes=current_boxes,
                event_name=event_name,
            )
            if ped_sample_speeds.shape != sample_speeds.shape or not np.allclose(ped_sample_speeds, sample_speeds):
                ped_risks = np.zeros(sample_speeds.shape, dtype=np.float32)

            sample['speed_sample_values'] = sample_speeds.astype(np.float32)
            sample['speed_sample_valid_mask'] = valid_mask.astype(np.float32)
            sample['speed_sample_exp_index'] = np.int64(exp_index)
            sample['speed_risk_chase_values'] = chase_risks.astype(np.float32)
            sample['speed_risk_meet_values'] = meet_risks.astype(np.float32)
            sample['speed_risk_ped_values'] = ped_risks.astype(np.float32)
            stage1_speed_built += 1

    print(
        f"\nDone: semantic={semantic_computed}, energy={energy_built}, "
        f"ego_status={ego_status_built}, front_route={front_route_built}, "
        f"stage1_speed={stage1_speed_built}, "
        f"skipped={skipped}, fallback={fallback}"
    )

    # Save back (atomic write)
    tmp_path = packed_path + f'.tmp.{os.getpid()}'
    print(f"Saving to {packed_path}...")
    with open(tmp_path, 'wb') as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.rename(tmp_path, packed_path)
    size_mb = os.path.getsize(packed_path) / 1e6
    print(f"Saved ({size_mb:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-compute fast training fields into samples_packed.pkl")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset split (e.g. /media/z/data/dataset/pdm_lite_mini/train)')
    parser.add_argument('--image_data_root', type=str, required=True,
                        help='Root of image data (e.g. /media/z/data/dataset/pdm_lite_mini)')
    parser.add_argument('--anchor_path', type=str, required=True,
                        help='Path to anchor file (.npy or .pkl)')
    parser.add_argument('--bev_ppm', type=float, default=2.0)
    parser.add_argument('--bev_size', type=int, default=256)
    parser.add_argument('--front_corridor_margin_m', type=float, default=0.5)
    parser.add_argument('--front_route_step_m', type=float, default=0.25)
    parser.add_argument('--front_max_distance_m', type=float, default=40.0)
    parser.add_argument('--front_safe_ttc_s', type=float, default=3.0)
    parser.add_argument('--front_max_ttc_s', type=float, default=10.0)
    parser.add_argument('--front_block_safe_distance_m', type=float, default=30.0)
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
               front_block_safe_distance_m=args.front_block_safe_distance_m)
