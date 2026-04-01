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

    # Case 1: a vehicle already covers the current route -> pursuit / following problem.
    best_current = None
    for box in current_boxes:
        cover = _find_route_cover_point(route_dense, route_s, box, corridor_margin_m)
        if cover is None:
            continue

        lead_speed = float(abs(box.get('speed', 0.0)))
        closing_speed = max(ego_speed - lead_speed, 0.1)
        ttc = cover['route_distance'] / closing_speed
        actor_class_id = _canonical_actor_class(box)
        block_risk = _blocking_risk_from_distance(cover['route_distance'], block_safe_distance_m)
        candidate = _base_result(
            np.clip(cover['route_distance'], 0.0, max_distance_m),
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
                'actor_class': ACTOR_CLASS_NAMES.get(actor_class_id, 'unknown'),
                'block_risk': float(block_risk),
            }

    if best_current is not None:
        return (best_current, debug) if return_debug else best_current

    # Case 2: no current cover, but a future vehicle will intersect the current route.
    if ego_matrix_current is None or future_frames_data is None or len(future_frames_data) == 0:
        result = _base_result(max_distance_m, max_ttc_s, 0.0, 0.0, 0, 0.0, ACTOR_CLASS_NONE)
        return (result, debug) if return_debug else result

    try:
        ego_inv = np.linalg.inv(np.asarray(ego_matrix_current, dtype=np.float32))
    except np.linalg.LinAlgError:
        result = _base_result(max_distance_m, max_ttc_s, 0.0, 0.0, 0, 0.0, ACTOR_CLASS_NONE)
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
            current_box = current_boxes_by_id.get(actor_id, None)
            if current_box is not None and len(current_box.get('position', [])) >= 2:
                bg_pos = np.asarray(current_box['position'][:2], dtype=np.float32)
                bg_speed = float(abs(current_box.get('speed', box_future.get('speed', 0.0))))
            else:
                bg_pos = np.asarray(box_cur['position'][:2], dtype=np.float32)
                bg_speed = float(abs(box_future.get('speed', 0.0)))

            conflict_pt = np.asarray(cover['route_point'], dtype=np.float32)
            d_ego = float(cover['route_distance'])
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
                    'bg_pos': np.asarray(bg_pos, dtype=np.float32),
                    'bg_speed': bg_speed,
                    'meet_dist': float(meet_dist),
                    'meet_speed': float(meet_speed),
                    'actor_class': ACTOR_CLASS_NAMES.get(actor_class_id, 'unknown'),
                }

    if best_future is not None:
        return (best_future, debug) if return_debug else best_future

    result = _base_result(max_distance_m, max_ttc_s, 0.0, 0.0, 0, 0.0, ACTOR_CLASS_NONE)
    return (result, debug) if return_debug else result


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
    packed_path = os.path.join(dataset_path, 'samples_packed.pkl')
    if not os.path.exists(packed_path):
        print(f"ERROR: {packed_path} not found. Run training first to generate it.")
        sys.exit(1)

    print(f"Loading {packed_path}...")
    with open(packed_path, 'rb') as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples")

    complete_fast_fields = sum(1 for s in samples if _has_all_fast_fields(s))
    already_semantic = sum(1 for s in samples if 'behavior_labels' in s and 'allowed_flags' in s and 'scene_buckets' in s)
    already_energy = sum(1 for s in samples if 'energy_targets' in s and 'energy_active_mask' in s)
    already_ego_status = sum(1 for s in samples if 'ego_status' in s)
    print(
        f"Existing fields: semantic={already_semantic}/{len(samples)}, "
        f"energy={already_energy}/{len(samples)}, "
        f"ego_status={already_ego_status}/{len(samples)}, "
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

    print(
        f"\nDone: semantic={semantic_computed}, energy={energy_built}, "
        f"ego_status={ego_status_built}, front_route={front_route_built}, "
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
