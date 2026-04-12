#!/usr/bin/env python3
"""
Generate a route-level video for route-constrained front-risk labels.

For every sample on the selected route, this tool:
- computes the front-route label
- draws the current route, vehicles, and selected conflict geometry
- overlays the current RGB frame
- writes a single MP4 under visualizations/front_route_videos/

Usage:
  /home/z/anaconda3/envs/dpauto/bin/python tools/generate_front_route_label_video.py \
    --dataset_path /media/z/data/dataset/pdm_lite_mini/train \
    --image_data_root /media/z/data/dataset/pdm_lite_mini \
    --route_name Town12_Rep0_1152_0_route0_11_08_04_25_14

  # Full scene-instance sequence for one scene/route pair:
  /home/z/anaconda3/envs/dpauto/bin/python tools/generate_front_route_label_video.py \
    --dataset_path /media/z/data/dataset/pdm_lite_mini/train \
    --image_data_root /media/z/data/dataset/pdm_lite_mini \
    --scene_name AccidentTwoWays \
    --route_name Town12_Rep0_1152_0_route0_11_08_04_25_14
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np

from scripts.data_tools.precompute_semantic_labels import (
    _annotate_route_stage1_merge_decisions,
    _build_merge_motion_context,
    _compute_front_route_label,
    _default_merge_episode_debug,
    _find_ego_box,
    _load_json_gz_if_exists,
    _points_inside_oriented_box,
    _resolve_feature_frame_info,
    _transform_box_to_current_frame,
)


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}
PEDESTRIAN_CLASSES = {"pedestrian", "walker"}
CASE_NAMES = {
    0: "none",
    1: "current_cover",
    2: "future_cover",
}
INTERACTION_MODE_NAMES = {
    0: "none",
    1: "chase",
    2: "meet",
}
COMMAND_MAP = {
    1: "LEFT",
    2: "RIGHT",
    3: "STRAIGHT",
    4: "LANE_FOLLOW",
    5: "CHANGE_LEFT",
    6: "CHANGE_RIGHT",
}
MERGE_COMMAND_IDS = {5, 6}
LEFT_COMMAND_ID = 1
RIGHT_COMMAND_ID = 2
LANE_FOLLOW_COMMAND_ID = 4
INTERACTION_SAME_DIR_ANGLE_THRESH_DEG = 45.0
INTERACTION_CROSS_MIN_ANGLE_THRESH_DEG = 70.0
MERGE_DEBUG_MIN_DEGO_M = 1.0
MERGE_DEBUG_MIN_GO_DENOM_S = 0.10
NO_ROUTE_EXTENSION_SCENES = {"HazardAtSideLane"}
TWOWAY_START_LATERAL_THRESH_M = 0.5
TWOWAY_RETURN_TAIL_POINTS = 12
TWOWAY_RETURN_FALLBACK_EXTRA_POINT_INDEX = 10
MERGE_TOP_LEVEL_FIELDS = (
    "merge_decision_phase",
    "merge_episode_id",
    "merge_episode_active",
    "merge_episode_no_go",
    "merge_episode_start_frame",
    "merge_episode_end_frame",
    "merge_go_frame",
    "merge_resolution_actor_id",
    "merge_end_state",
)
BEV_COLORS = np.array([
    [40, 40, 40],
    [128, 128, 128],
    [180, 0, 180],
    [255, 255, 255],
    [0, 255, 255],
    [0, 0, 255],
    [0, 255, 0],
    [0, 200, 255],
    [0, 0, 255],
    [255, 100, 0],
    [255, 255, 0],
], dtype=np.uint8)

def _wrap_to_pi(angle_rad):
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def _box_class_name(box):
    return str((box or {}).get("class", "")).lower()


def _is_pedestrian_box(box):
    cls = _box_class_name(box)
    return any(token in cls for token in PEDESTRIAN_CLASSES)


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
    junction = bool(current_meas.get("junction", False))
    cmd_id = _measurement_command_id(current_meas)
    return bool(junction and cmd_id == RIGHT_COMMAND_ID)


def _is_right_turn_scene_context(current_meas, event_name=None):
    if _is_right_turn_junction_context(current_meas):
        return True
    event_name = str(event_name or "")
    return event_name in {"NonSignalizedJunctionRightTurn", "SignalizedJunctionRightTurn"}


def _is_left_turn_junction_context(current_meas):
    if current_meas is None:
        return False
    junction = bool(current_meas.get("junction", False))
    cmd_id = _measurement_command_id(current_meas)
    return bool(junction and cmd_id == LEFT_COMMAND_ID)


def _is_left_turn_scene_context(current_meas, event_name=None):
    if _is_left_turn_junction_context(current_meas):
        return True
    event_name = str(event_name or "")
    return event_name in {"NonSignalizedJunctionLeftTurn", "SignalizedJunctionLeftTurn"}


def _is_borrow_cross_scene_context(event_name=None):
    event_name = str(event_name or "")
    return event_name in {"ConstructionObstacleTwoWays", "AccidentTwoWays", "ParkedObstacleTwoWays"}


def _is_two_way_event_corridor_scene_context(event_name=None):
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


def _heading_to_deg(angle_rad):
    return float(np.degrees(_wrap_to_pi(angle_rad)))


def _corridor_segment_from_cover(cover_point_xy, route_heading_deg, conflict_len_m):
    cover_point_xy = np.asarray(cover_point_xy, dtype=np.float32)
    if cover_point_xy.shape != (2,):
        return np.zeros((0, 2), dtype=np.float32)
    if not np.isfinite(float(route_heading_deg)):
        return np.stack([cover_point_xy, cover_point_xy], axis=0).astype(np.float32)
    seg_len = 0.0 if not np.isfinite(float(conflict_len_m)) else max(float(conflict_len_m), 0.0)
    heading_rad = np.deg2rad(float(route_heading_deg))
    tangent = np.asarray([np.cos(heading_rad), np.sin(heading_rad)], dtype=np.float32)
    end_point_xy = cover_point_xy + float(seg_len) * tangent
    return np.stack([cover_point_xy, end_point_xy], axis=0).astype(np.float32)


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


def _interaction_signal_from_debug(
    label,
    debug,
    current_meas=None,
    event_name=None,
    angle_thresh_deg=INTERACTION_SAME_DIR_ANGLE_THRESH_DEG,
    min_motion_m=0.5,
):
    case = int(label.get("case", 0))
    if case <= 0:
        return {
            "mode": 0,
            "name": INTERACTION_MODE_NAMES[0],
            "subtype": "none",
            "source": "none",
            "angle_deg": np.nan,
            "route_heading_deg": np.nan,
            "actor_heading_deg": np.nan,
            "motion_m": 0.0,
        }

    best = debug.get("best_current") if case == 1 else debug.get("best_future")
    return _interaction_signal_from_candidate(
        case=case,
        best=best,
        debug=debug,
        current_meas=current_meas,
        event_name=event_name,
        angle_thresh_deg=angle_thresh_deg,
        min_motion_m=min_motion_m,
    )


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
            "name": INTERACTION_MODE_NAMES[0],
            "subtype": "none",
            "source": "none",
            "angle_deg": np.nan,
            "route_heading_deg": np.nan,
            "actor_heading_deg": np.nan,
            "motion_m": 0.0,
        }, event_name=event_name)

    cover = best.get("cover", {})
    actor_box = best.get("box") if case == 1 else (best.get("current_box") or best.get("box_future") or best.get("box_current_frame"))
    if _is_pedestrian_box(actor_box):
        return _attach_event_name_record({
            "mode": 2,
            "name": INTERACTION_MODE_NAMES[2],
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
            "name": INTERACTION_MODE_NAMES[0],
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
                "name": INTERACTION_MODE_NAMES[2],
                "subtype": "merge_meet",
                "source": "junction_right_scene_override_missing_heading",
                "angle_deg": np.nan,
                "route_heading_deg": _heading_to_deg(route_heading),
                "actor_heading_deg": np.nan,
                "motion_m": motion_m,
            }, event_name=event_name)
        return _attach_event_name_record({
            "mode": 0,
            "name": INTERACTION_MODE_NAMES[0],
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
            "name": INTERACTION_MODE_NAMES[2],
            "subtype": "merge_meet",
            "source": "junction_right_scene_future_override",
            "angle_deg": float(angle_deg),
            "route_heading_deg": _heading_to_deg(route_heading),
            "actor_heading_deg": _heading_to_deg(actor_heading),
            "motion_m": float(motion_m),
        }, event_name=event_name)
    if same_direction and case == 1:
        mode = 1
        subtype = "follow_chase"
        source = f"{source}+same_dir_current_cover"
    elif case == 1 and _is_right_turn_scene_context(current_meas, event_name=event_name):
        mode = 2
        subtype = "merge_meet"
        source = f"{source}+junction_right_scene_current_override"
    else:
        mode = 2
        if same_direction and case == 2:
            subtype = "merge_meet"
            source = f"{source}+same_dir_future_cover"
        elif not cross_direction:
            subtype = "merge_meet"
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
        elif not same_direction:
            subtype = "cross_meet"
            source = f"{source}+cross_dir"
    return _attach_event_name_record({
        "mode": int(mode),
        "name": INTERACTION_MODE_NAMES[int(mode)],
        "subtype": subtype,
        "source": source,
        "angle_deg": float(angle_deg),
        "route_heading_deg": _heading_to_deg(route_heading),
        "actor_heading_deg": _heading_to_deg(actor_heading),
        "motion_m": float(motion_m),
    }, event_name=event_name)


def _cover_candidate_summary(case, best, debug, current_meas=None, event_name=None):
    interaction = _interaction_signal_from_candidate(case, best, debug, current_meas=current_meas, event_name=event_name)
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
            "other_length_m": float(_box_length_m(box, default_length_m=np.nan)),
            "cover_point_local_xy": np.asarray(best.get("cover", {}).get("route_point", []), dtype=np.float32).astype(float).tolist(),
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
        "cover_point_local_xy": np.asarray(best.get("cover", {}).get("route_point", []), dtype=np.float32).astype(float).tolist(),
    }


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
    center_dist = float(
        np.linalg.norm(
            np.asarray(pos_a[:2], dtype=np.float32) - np.asarray(pos_b[:2], dtype=np.float32)
        )
    )
    # A simple conservative clearance approximation for local debug:
    # subtract the 2D half-diagonal radii of both boxes.
    radius_a = float(np.linalg.norm(np.asarray(ext_a[:2], dtype=np.float32)))
    radius_b = float(np.linalg.norm(np.asarray(ext_b[:2], dtype=np.float32)))
    return float(max(center_dist - radius_a - radius_b, 0.0))


def _point_to_oriented_box_distance(point, box, margin_m=0.0):
    point = np.asarray(point, dtype=np.float32)
    pos = box.get("position", None)
    extent = box.get("extent", None)
    if point.shape != (2,) or pos is None or extent is None or len(pos) < 2 or len(extent) < 2:
        return np.inf

    center = np.asarray(pos[:2], dtype=np.float32)
    extent = np.asarray(extent[:2], dtype=np.float32)
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    yaw = float(box.get("yaw", 0.0))
    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))

    local_x = dx * cos_y + dy * sin_y
    local_y = -dx * sin_y + dy * cos_y
    qx = abs(local_x) - (float(extent[0]) + margin_m)
    qy = abs(local_y) - (float(extent[1]) + margin_m)
    return float(np.hypot(max(qx, 0.0), max(qy, 0.0)))


def _speed_risk_samples_around_expert(speed_mps, lower_delta=5.0, upper_delta=5.0, max_speed_mps=20.0):
    speed_mps = float(max(speed_mps, 0.0))
    raw = speed_mps + np.asarray([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0], dtype=np.float32)
    raw = np.clip(raw, 0.0, float(max_speed_mps))
    unique = np.unique(np.round(raw, 2))
    return unique.astype(np.float32)


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
    if start_local_xy is not None:
        if route_local is not None:
            route_poly = _route_with_origin(np.asarray(route_local, dtype=np.float32))
            _, start_s = _project_point_to_polyline(start_local_xy, route_poly)
            if start_s is not None and np.isfinite(float(start_s)):
                borrow_start_distance_m = float(max(float(start_s), 0.0))
        elif np.isfinite(float(start_local_xy[0])):
            borrow_start_distance_m = float(max(float(start_local_xy[0]), 0.0))
    if not np.isfinite(borrow_start_distance_m):
        borrow_start_distance_m = 0.0
    borrow_start_distance_m = float(max(borrow_start_distance_m, 0.0))

    return {
        "borrow_start_distance_m": float(borrow_start_distance_m),
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
):
    speed = float((current_meas or {}).get("speed", 0.0))
    sample_speeds = _speed_risk_samples_around_expert(speed)
    chase_risks = np.zeros(sample_speeds.shape, dtype=np.float32)
    meet_risks = np.zeros(sample_speeds.shape, dtype=np.float32)
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
                # Even before TTC becomes short, any positive closing speed should
                # contribute some chase risk. Modulate it by distance so far-away
                # following is milder while close following is stronger.
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
        "t_bg_clear_s": np.nan,
        "t_ego_exit_s": np.nan,
        "safe_gap_bg_m": np.nan,
        "rear_gap_m": np.nan,
        "v_equal_mps": np.nan,
        "v_go_min_mps": np.nan,
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
            d_ego_entry_m = np.nan
            if not np.isfinite(d_ego_entry_m):
                if np.isfinite(conflict_len_m):
                    d_ego_entry_m = float(max(float(d_ego) - 0.5 * float(conflict_len_m), 0.0))
                else:
                    d_ego_entry_m = float(max(float(d_ego), 0.0))
            t_bg = 0.0
            if np.isfinite(bg_speed) and bg_speed > 1e-3 and np.isfinite(conflict_len_m):
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
                "ego_clearance_m": np.nan,
                "bg_speed_mps": float(bg_speed),
                "t_bg_s": float(t_bg),
                "t_bg_exit_s": float(t_bg_exit),
                "t_bg_clear_s": np.nan,
                "safe_gap_bg_m": np.nan,
                "rear_gap_m": np.nan,
                "v_equal_mps": np.nan,
                "v_go_min_mps": np.nan,
                "v_behind_min_mps": np.nan,
                "v_go_need_mps": np.nan,
                "v_yield_max_mps": np.nan,
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
                meet_risks[idx] = float(max(risk, occupancy_floor))
            total_risks = np.maximum(chase_risks, meet_risks)
        cross_wait_time_s, cross_wait_valid = _cross_wait_from_meet_debug(meet_info)
        return {
            "sample_speeds_mps": sample_speeds.astype(np.float32),
            "chase_risks": chase_risks.astype(np.float32),
            "meet_risks": meet_risks.astype(np.float32),
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
                t_bg_exit_to_end = float((d_bg_to_end_m + bg_clearance_m) / max(bg_speed, 1e-6))
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
                v_behind_min = max(float(bg_speed), 0.0)
                if d_ego <= 0.25:
                    # Once ego is already at / just past the merge point, the
                    # front-arrive timing threshold is no longer meaningful.
                    # Keep only the rear-car floor so the post-merge meet risk
                    # decays smoothly with higher ego speeds instead of turning
                    # into inf/NaN.
                    v_equal = np.nan
                    v_go_min = np.nan
                    v_go_need = float(v_behind_min)
                    v_yield_max = np.nan
                else:
                    v_equal = (d_ego + ego_clearance_m) / max(t_bg, 1e-6)
                    go_denom = t_bg - float(merge_tau_s)
                    v_go_min = np.inf if go_denom <= 1e-6 else (d_ego + ego_clearance_m) / max(go_denom, 1e-6)
                    v_go_need = max(float(v_go_min), float(v_behind_min))
                    v_yield_max = d_ego / max(t_bg_clear + float(merge_tau_s), 1e-6)
                debug_v_equal = float(v_equal)
                debug_v_go_min = float(v_go_min)
                if (
                    d_ego <= float(MERGE_DEBUG_MIN_DEGO_M) or
                    (np.isfinite(t_bg) and (t_bg - float(merge_tau_s)) <= float(MERGE_DEBUG_MIN_GO_DENOM_S))
                ):
                    # Near the future->current transition, these timing-derived
                    # speeds become numerically large and are no longer very
                    # interpretable. Keep the risk curve, but hide the raw
                    # debug values from the overlay.
                    debug_v_equal = np.nan
                    debug_v_go_min = np.nan
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
                "t_bg_clear_s": float(t_bg_clear),
                "t_ego_exit_s": np.nan,
                "safe_gap_bg_m": float(safe_gap_bg),
                "rear_gap_m": float(rear_gap_m),
                "v_equal_mps": float(debug_v_equal),
                "v_go_min_mps": float(debug_v_go_min),
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
                        # Post-merge persistence regime: ego has already
                        # committed, so only the rear-car catch-up matters.
                        # Use bbox-adjusted rear clearance and a linear rear TTC
                        # schedule: <=1s -> 1, >=3s -> 0, no catch-up -> 0.
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
                        # For same-direction merge timing, we keep two safe regimes:
                        # 1) yield behind the actor: v <= v_yield_max
                        # 2) commit clearly ahead of the actor: v >= v_go_need
                        #
                        # The peak danger sits around v_equal (same-arrival speed at
                        # the merge point). From there, risk should fall gradually
                        # toward the front-commit threshold instead of staying at 1
                        # all the way until v_go_need.
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
        "merge_yld_risks": merge_yld_risks.astype(np.float32),
        "merge_go_risks": merge_go_risks.astype(np.float32),
        "borrow_yld_risks": borrow_yld_risks.astype(np.float32),
        "borrow_go_risks": borrow_go_risks.astype(np.float32),
        "total_risks": total_risks.astype(np.float32),
        "chase": chase_info,
        "meet": meet_info,
    }


def _occupancy_signal_from_label(label, interaction, safe_ttc_s=3.0):
    case = int(label.get("case", 0))
    if case <= 0:
        occ_risk = 0.0
        occ_source = "none"
    else:
        ttc = float(label.get("ttc", np.inf))
        ttc_risk = 0.0 if not np.isfinite(ttc) else float(np.clip((safe_ttc_s - ttc) / max(safe_ttc_s, 1e-6), 0.0, 1.0))
        block_risk = float(label.get("block_risk", 0.0))
        if int(interaction.get("mode", 0)) == 1:
            occ_risk = max(block_risk, ttc_risk)
            occ_source = "chase=max(block,ttc)"
        elif int(interaction.get("mode", 0)) == 2:
            occ_risk = ttc_risk
            occ_source = "meet=ttc"
        else:
            occ_risk = 0.0
            occ_source = "none"
    return {
        "present": float(case > 0),
        "risk": float(np.clip(occ_risk, 0.0, 1.0)),
        "case_name": CASE_NAMES.get(case, f"case_{case}"),
        "source": occ_source,
    }


def _command_id_from_sample(sample, current_meas):
    cmd = (current_meas or {}).get("command", None)
    if cmd is not None:
        try:
            return int(cmd)
        except Exception:
            pass

    cmd_hist = sample.get("command_hist", None)
    if cmd_hist is not None:
        cmd_hist = np.asarray(cmd_hist, dtype=np.float32)
        if cmd_hist.ndim >= 2 and cmd_hist.shape[0] > 0 and cmd_hist.shape[-1] == 6:
            return int(np.argmax(cmd_hist[-1]) + 1)

    cmd = sample.get("command", None)
    if cmd is not None:
        cmd = np.asarray(cmd, dtype=np.float32)
        if cmd.shape[-1] == 6:
            return int(np.argmax(cmd) + 1)

    return 4


def _safe_gap_m(speed_mps, base_gap_m=4.0, headway_s=1.5, brake_decel_mps2=4.0):
    speed_mps = float(max(speed_mps, 0.0))
    brake_decel_mps2 = float(max(brake_decel_mps2, 1e-3))
    return float(base_gap_m + headway_s * speed_mps + (speed_mps ** 2) / (2.0 * brake_decel_mps2))


def _proceed_signal_from_label(
    sample,
    label,
    interaction,
    current_meas,
    base_gap_m=4.0,
    headway_s=1.5,
    brake_decel_mps2=4.0,
    safe_ttc_s=3.0,
    speed_gamma=1.0,
    merge_min_speed_ratio=0.35,
):
    speed = float((current_meas or {}).get("speed", 0.0))
    target_speed = float((current_meas or {}).get("target_speed", 0.0))
    speed_limit = float((current_meas or {}).get("speed_limit", 0.0))
    nominal_speed = target_speed if target_speed > 1e-3 else max(speed, speed_limit, 0.0)

    command_id = _command_id_from_sample(sample, current_meas)
    command_name = COMMAND_MAP.get(command_id, f"CMD_{command_id}")
    is_merge = command_id in MERGE_COMMAND_IDS

    distance = float(label.get("distance", np.inf))
    ttc = float(label.get("ttc", np.inf))
    block_risk = float(label.get("block_risk", 0.0))
    safe_gap = _safe_gap_m(
        speed_mps=speed,
        base_gap_m=base_gap_m,
        headway_s=headway_s,
        brake_decel_mps2=brake_decel_mps2,
    )

    if np.isfinite(distance):
        gap_risk = float(np.clip((safe_gap - distance) / max(safe_gap, 1e-6), 0.0, 1.0))
    else:
        gap_risk = 0.0
    if np.isfinite(ttc):
        ttc_risk = float(np.clip((safe_ttc_s - ttc) / max(safe_ttc_s, 1e-6), 0.0, 1.0))
    else:
        ttc_risk = 0.0

    interaction_mode = int(interaction.get("mode", 0))
    if speed <= 1e-3:
        proceed_risk = 0.0
        proceed_source = "stopped_zero_speed"
    elif interaction_mode == 1:
        proceed_risk = float(np.clip(max(gap_risk, ttc_risk, block_risk), 0.0, 1.0))
        proceed_source = "chase=max(gap,ttc,block)"
    elif interaction_mode == 2:
        proceed_risk = float(np.clip(ttc_risk, 0.0, 1.0))
        proceed_source = "meet=ttc"
    else:
        proceed_risk = 0.0
        proceed_source = "none"
    speed_cap = float(nominal_speed * max(0.0, 1.0 - proceed_risk) ** max(speed_gamma, 1e-6))
    if is_merge and nominal_speed > 1e-3:
        speed_floor = float(min(speed_cap, merge_min_speed_ratio * nominal_speed))
    else:
        speed_floor = 0.0

    return {
        "risk": proceed_risk,
        "gap_risk": gap_risk,
        "ttc_risk": ttc_risk,
        "safe_gap_m": safe_gap,
        "speed_cap_mps": speed_cap,
        "speed_floor_mps": speed_floor,
        "command_id": int(command_id),
        "command_name": command_name,
        "is_merge": float(is_merge),
        "nominal_speed_mps": nominal_speed,
        "source": proceed_source,
    }


def _update_wait_release_state(
    prev_wait_state,
    interaction,
    current_meas,
    wait_speed_thresh=0.5,
    release_speed_thresh=1.0,
    release_throttle_thresh=0.3,
    prev_speed_mps=None,
):
    speed = float((current_meas or {}).get("speed", 0.0))
    prev_speed = float(speed if prev_speed_mps is None else prev_speed_mps)
    target_speed = float((current_meas or {}).get("target_speed", 0.0))
    throttle = float((current_meas or {}).get("throttle", 0.0))
    brake = bool((current_meas or {}).get("brake", False))
    control_brake = bool((current_meas or {}).get("control_brake", False))
    interaction_active = bool(int(interaction.get("mode", 0)) > 0)
    wait_candidate = bool(interaction_active and speed <= float(wait_speed_thresh))
    if bool(prev_wait_state):
        wait_state = bool(interaction_active and speed < float(release_speed_thresh))
    else:
        wait_state = wait_candidate
    release_intent = bool(
        throttle >= float(release_throttle_thresh)
        and not brake
        and not control_brake
        and target_speed > 0.5
    )
    # Release is triggered either by the usual speed threshold crossing or by a
    # clear throttle-based launch intent while still nearly stopped.
    release_pulse = bool(
        prev_speed <= float(wait_speed_thresh)
        and (speed >= float(release_speed_thresh) or release_intent)
    )
    return {
        "interaction_active": float(interaction_active),
        "speed_mps": float(speed),
        "prev_speed_mps": float(prev_speed),
        "target_speed_mps": float(target_speed),
        "throttle": float(throttle),
        "brake": float(brake),
        "control_brake": float(control_brake),
        "wait_candidate": float(wait_candidate),
        "wait_state": float(wait_state),
        "release_intent": float(release_intent),
        "release_pulse": float(release_pulse),
        "wait_speed_thresh": float(wait_speed_thresh),
        "release_speed_thresh": float(release_speed_thresh),
        "release_throttle_thresh": float(release_throttle_thresh),
    }


def _ego_world_center_xyz(current_meas):
    ego_matrix = None if current_meas is None else current_meas.get("ego_matrix", None)
    if ego_matrix is None:
        return None
    ego_matrix = np.asarray(ego_matrix, dtype=np.float32)
    if ego_matrix.shape != (4, 4):
        return None
    return ego_matrix[:3, 3].astype(np.float32)


def _route_with_origin(route_local):
    route_local = np.asarray(route_local, dtype=np.float32)
    if route_local.ndim != 2 or route_local.shape[0] == 0 or route_local.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    if float(np.linalg.norm(route_local[0])) < 1e-4:
        return route_local
    return np.concatenate([np.zeros((1, 2), dtype=np.float32), route_local], axis=0)


def _clip_local_route_horizon(route_local, horizon_m, step_m=0.5):
    route_poly = _route_with_origin(route_local)
    if route_poly.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    arc = _polyline_arclengths(route_poly)
    if arc.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    horizon_m = float(max(horizon_m, 0.0))
    total_len = float(arc[-1])
    clip_len = min(horizon_m, total_len)
    if clip_len <= 1e-4:
        return route_poly[:1].astype(np.float32)
    query_s = np.arange(0.0, clip_len + 1e-6, float(max(step_m, 1e-3)), dtype=np.float32)
    if query_s.size == 0 or float(query_s[-1]) < clip_len - 1e-4:
        query_s = np.concatenate([query_s, np.array([clip_len], dtype=np.float32)], axis=0)
    return _sample_polyline_at_arclengths(route_poly, query_s)


def _estimate_borrow_points_from_wait_route(
    route_local,
    borrow_enter_lateral_thresh=1.25,
    return_lateral_thresh=0.8,
    min_enter_progress_m=4.0,
    min_return_progress_m=6.0,
    baseline_points=6,
    segment_step_m=0.5,
):
    route_poly = _route_with_origin(route_local)
    if route_poly.shape[0] < 3:
        return None

    arc = _polyline_arclengths(route_poly)
    if arc.shape[0] != route_poly.shape[0]:
        return None

    baseline_count = min(max(int(baseline_points), 2), route_poly.shape[0])
    baseline_y = float(np.median(route_poly[:baseline_count, 1]))
    rel_y = route_poly[:, 1] - baseline_y

    enter_idx = None
    for idx in range(1, route_poly.shape[0]):
        if float(arc[idx]) < float(min_enter_progress_m):
            continue
        if abs(float(rel_y[idx])) >= float(borrow_enter_lateral_thresh):
            enter_idx = idx
            break
    if enter_idx is None:
        return None

    return_idx = None
    for idx in range(enter_idx + 1, route_poly.shape[0]):
        if float(arc[idx] - arc[enter_idx]) < float(min_return_progress_m):
            continue
        if abs(float(rel_y[idx])) <= float(return_lateral_thresh):
            return_idx = idx
            break
    if return_idx is None:
        return None

    peak_lateral_m = float(np.max(np.abs(rel_y[enter_idx:return_idx + 1])))
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
        "peak_lateral_m": float(peak_lateral_m),
        "baseline_y_m": float(baseline_y),
        "enter_local_xy": route_poly[enter_idx, :2].astype(np.float32),
        "return_local_xy": route_poly[return_idx, :2].astype(np.float32),
        "segment_local_xy": segment_local.astype(np.float32),
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
    route_poly = _route_with_origin(route_local)
    if route_poly.shape[0] < 3:
        return None

    arc = _polyline_arclengths(route_poly)
    if arc.shape[0] != route_poly.shape[0]:
        return None

    baseline_count = min(max(int(baseline_points), 2), route_poly.shape[0])
    baseline_y = float(np.median(route_poly[:baseline_count, 1]))
    rel_y = route_poly[:, 1] - baseline_y
    signed_rel_y = float(np.sign(float(shift_sign)) or -1.0) * rel_y

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

    peak_signed_lateral_m = float(np.max(signed_rel_y[enter_idx:return_idx + 1]))
    peak_lateral_m = float(np.max(np.abs(rel_y[enter_idx:return_idx + 1])))
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
        "peak_lateral_m": float(peak_lateral_m),
        "peak_signed_lateral_m": float(peak_signed_lateral_m),
        "baseline_y_m": float(baseline_y),
        "shift_sign": float(np.sign(float(shift_sign)) or -1.0),
        "enter_local_xy": route_poly[enter_idx, :2].astype(np.float32),
        "return_local_xy": route_poly[return_idx, :2].astype(np.float32),
        "segment_local_xy": segment_local.astype(np.float32),
    }


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


def _summarize_signed_route_shift(route_local, shift_sign=-1.0, baseline_points=6):
    route_poly = _route_with_origin(route_local)
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
        best_priority = None
        best_score = None
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
        route_local = np.asarray(
            record.get("sample_vis", {}).get(
                "_route_corridor_input_local",
                record.get("sample_vis", {}).get("_route_input_local", np.zeros((0, 2), dtype=np.float32)),
            ),
            dtype=np.float32,
        )
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
        return None

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
        return None

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
        return None

    record = frame_records[int(best_candidate["record_idx"])]
    current_meas = record.get("current_meas")
    ego_matrix_current = None if current_meas is None else current_meas.get("ego_matrix", None)
    if ego_matrix_current is None:
        return None
    borrow_geom = dict(best_candidate["geom"])
    route_local_context = np.asarray(
        record.get("sample_vis", {}).get(
            "_route_corridor_input_local",
            record.get("sample_vis", {}).get("_route_input_local", np.zeros((0, 2), dtype=np.float32)),
        ),
        dtype=np.float32,
    )
    route_poly_context = _route_with_origin(route_local_context)
    arc_context = _polyline_arclengths(route_poly_context)
    if (
        route_poly_context.ndim != 2 or route_poly_context.shape[0] < 2 or route_poly_context.shape[1] != 2 or
        arc_context.ndim != 1 or arc_context.shape[0] != route_poly_context.shape[0]
    ):
        return None
    route_head_idx = int(max(route_poly_context.shape[0] - route_local_context.shape[0], 0))
    obstacle_local = np.asarray([best_candidate["local_x"], best_candidate["local_y"]], dtype=np.float32)
    if obstacle_local.shape != (2,) or not np.all(np.isfinite(obstacle_local)):
        return None
    # Choose how we anchor the corridor start along the route.
    if str(borrow_start_mode) == "obstacle_align":
        start_mask = np.abs(route_poly_context[:, 1]) <= float(TWOWAY_START_LATERAL_THRESH_M)
        if not np.any(start_mask):
            return None
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
        front_local = np.asarray(
            record.get("sample_vis", {}).get("_route_front_local", np.zeros((0, 2), dtype=np.float32)),
            dtype=np.float32,
        )
        route_front_count = int(front_local.shape[0]) if front_local.ndim == 2 else 0
        fallback_local_idx = route_front_count + int(TWOWAY_RETURN_FALLBACK_EXTRA_POINT_INDEX) - 1
        if route_front_count <= 0 or fallback_local_idx >= route_local_context.shape[0]:
            return None
        fallback_poly_idx = int(route_head_idx + fallback_local_idx)
        if fallback_poly_idx < 0 or fallback_poly_idx >= arc_context.shape[0]:
            return None
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
        return None
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
        return None
    borrow_world = np.stack([borrow_segment_world[0, :2], borrow_segment_world[-1, :2]], axis=0)
    borrow_world_xyz = np.stack([borrow_segment_world[0, :3], borrow_segment_world[-1, :3]], axis=0)

    return {
        "valid": 1.0,
        "ready": 1.0,
        "source": "event_blocker_route_{}".format(str(borrow_geom.get("mode", "strict"))),
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
        "route_return_abs_m": float(borrow_geom.get("return_abs_m", np.nan)),
        "blocked_frame_id": -1,
    }


def _find_nearest_future_frame_to_world_point(frame_records, start_idx, target_world_xy, max_horizon_frames=120):
    target_world_xy = np.asarray(target_world_xy, dtype=np.float32)
    if target_world_xy.shape != (2,):
        return None

    best_idx = None
    best_dist = np.inf
    horizon_end = min(len(frame_records), int(start_idx) + int(max_horizon_frames) + 1)
    for idx in range(max(int(start_idx), 0), horizon_end):
        center_xyz = _ego_world_center_xyz(frame_records[idx]["current_meas"])
        if center_xyz is None:
            continue
        dist = float(np.linalg.norm(center_xyz[:2] - target_world_xy))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    if best_idx is None:
        return None
    return {
        "idx": int(best_idx),
        "frame_id": int(frame_records[best_idx]["frame_id"]),
        "distance_m": float(best_dist),
    }


def _find_next_speed_release_idx(frame_records, start_idx, wait_speed_thresh=0.5, release_speed_thresh=1.0):
    start_idx = max(int(start_idx), 0)
    for idx in range(start_idx + 1, len(frame_records)):
        speed = float((frame_records[idx]["current_meas"] or {}).get("speed", 0.0))
        prev_speed = float((frame_records[idx - 1]["current_meas"] or {}).get("speed", 0.0))
        if prev_speed <= float(wait_speed_thresh) and speed >= float(release_speed_thresh):
            return int(idx)
    return None


def _vehicle_overlaps_route_segment(box, route_dense_local, corridor_margin_m=0.5):
    route_dense_local = np.asarray(route_dense_local, dtype=np.float32)
    if route_dense_local.ndim != 2 or route_dense_local.shape[0] == 0 or route_dense_local.shape[1] != 2:
        return False
    pos = box.get("position", None)
    extent = box.get("extent", None)
    if pos is None or extent is None or len(pos) < 2 or len(extent) < 2:
        return False
    mask = _points_inside_oriented_box(
        route_dense_local,
        center=np.asarray(pos[:2], dtype=np.float32),
        extent=np.asarray(extent[:2], dtype=np.float32),
        yaw=float(box.get("yaw", 0.0)),
        margin_m=float(corridor_margin_m),
    )
    return bool(np.any(mask))


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
        if ped_boxes:
            filtered.append((ped_boxes, ego_matrix))
        else:
            filtered.append(None)
    return filtered


def _annotate_release_ready(
    frame_records,
    fps_hz=4.0,
    route_step_m=0.5,
    corridor_margin_m=0.5,
    dynamic_speed_thresh=0.25,
    max_borrow_horizon_frames=80,
    borrow_enter_lateral_thresh=1.25,
    return_lateral_thresh=0.8,
    min_enter_progress_m=4.0,
    min_return_progress_m=6.0,
    wait_speed_thresh=0.5,
    release_speed_thresh=1.0,
):
    for idx, record in enumerate(frame_records):
        info = {
            "valid": float(0.0),
            "ready": float(0.0),
            "source": "n/a",
            "release_frame_id": -1,
            "enter_frame_id": -1,
            "return_frame_id": -1,
            "borrow_duration_s": 0.0,
            "release_to_return_s": 0.0,
            "borrow_start_distance_m": 0.0,
            "borrow_distance_m": 0.0,
            "peak_lateral_m": 0.0,
            "blocked_frame_id": -1,
            "blocking_actor_id": -1,
            "blocking_actor_class": "none",
            "borrow_start_world_xy": [],
            "borrow_end_world_xy": [],
            "borrow_start_world_xyz": [],
            "borrow_end_world_xyz": [],
            "borrow_segment_world_xy": [],
            "borrow_segment_world_xyz": [],
        }
        if float(record["wait_info"].get("wait_state", 0.0)) <= 0.5:
            record["release_ready"] = info
            continue

        next_release_idx = _find_next_speed_release_idx(
            frame_records,
            start_idx=idx,
            wait_speed_thresh=wait_speed_thresh,
            release_speed_thresh=release_speed_thresh,
        )
        if next_release_idx is None:
            info["source"] = "no_future_release"
            record["release_ready"] = info
            continue

        current_meas = record["current_meas"]
        ego_matrix_current = None if current_meas is None else current_meas.get("ego_matrix", None)
        if ego_matrix_current is None:
            info["source"] = "missing_ego_matrix"
            record["release_ready"] = info
            continue

        route_local = np.asarray(
            record["sample_vis"].get(
                "_route_corridor_input_local",
                record["sample_vis"].get("_route_input_local", np.zeros((0, 2), dtype=np.float32)),
            ),
            dtype=np.float32,
        )
        borrow_geom = _estimate_borrow_points_from_wait_route(
            route_local=route_local,
            borrow_enter_lateral_thresh=borrow_enter_lateral_thresh,
            return_lateral_thresh=return_lateral_thresh,
            min_enter_progress_m=min_enter_progress_m,
            min_return_progress_m=min_return_progress_m,
            segment_step_m=route_step_m,
        )
        if borrow_geom is None:
            info["source"] = "borrow_geom_invalid"
            record["release_ready"] = info
            continue

        borrow_segment_world = _transform_points_local_to_world_xyz(
            np.asarray(borrow_geom["segment_local_xy"], dtype=np.float32),
            ego_matrix_current,
        )
        if borrow_segment_world.ndim != 2 or borrow_segment_world.shape[0] < 2 or borrow_segment_world.shape[1] < 3:
            info["source"] = "borrow_world_invalid"
            record["release_ready"] = info
            continue
        borrow_world = np.stack([borrow_segment_world[0, :2], borrow_segment_world[-1, :2]], axis=0)
        borrow_world_xyz = np.stack([borrow_segment_world[0, :3], borrow_segment_world[-1, :3]], axis=0)

        enter_match = _find_nearest_future_frame_to_world_point(
            frame_records,
            start_idx=idx,
            target_world_xy=borrow_world[0, :2],
            max_horizon_frames=max_borrow_horizon_frames,
        )
        if enter_match is None:
            info["source"] = "enter_match_missing"
            record["release_ready"] = info
            continue

        return_match = _find_nearest_future_frame_to_world_point(
            frame_records,
            start_idx=max(enter_match["idx"], idx),
            target_world_xy=borrow_world[1, :2],
            max_horizon_frames=max_borrow_horizon_frames,
        )
        if return_match is None:
            info["source"] = "return_match_missing"
            record["release_ready"] = info
            continue

        if int(return_match["idx"]) <= int(enter_match["idx"]):
            info["source"] = "return_before_enter"
            record["release_ready"] = info
            continue

        corridor_route = np.asarray(borrow_geom["segment_local_xy"], dtype=np.float32)
        if corridor_route.shape[0] == 0:
            info["source"] = "empty_borrow_segment"
            record["release_ready"] = info
            continue

        borrow_duration_frames = int(return_match["idx"] - enter_match["idx"])
        release_to_return_frames = int(max(return_match["idx"] - next_release_idx, 0))
        if release_to_return_frames <= 0:
            info["source"] = "release_after_return"
            record["release_ready"] = info
            continue

        try:
            ego_inv = np.linalg.inv(np.asarray(ego_matrix_current, dtype=np.float32))
        except np.linalg.LinAlgError:
            info["source"] = "singular_ego_matrix"
            record["release_ready"] = info
            continue

        corridor_route = np.asarray(borrow_geom["segment_local_xy"], dtype=np.float32)
        if corridor_route.shape[0] == 0:
            info["source"] = "empty_borrow_segment"
            record["release_ready"] = info
            continue

        blocked = False
        blocked_frame_id = -1
        blocking_actor_id = -1
        blocking_actor_class = "none"
        horizon_end = min(len(frame_records) - 1, idx + int(release_to_return_frames))
        for future_idx in range(idx + 1, horizon_end + 1):
            future_record = frame_records[future_idx]
            future_meas = future_record["current_meas"]
            future_boxes = future_record["current_boxes"] or []
            ego_matrix_future = None if future_meas is None else future_meas.get("ego_matrix", None)
            if ego_matrix_future is None:
                continue
            transform = ego_inv @ np.asarray(ego_matrix_future, dtype=np.float32)
            for box_future in future_boxes:
                cls = str(box_future.get("class", "")).lower()
                if cls in {"ego_car", "static"}:
                    continue
                if float(abs(box_future.get("speed", 0.0))) <= float(dynamic_speed_thresh):
                    continue
                box_cur = _transform_box_to_current_frame(box_future, transform)
                if box_cur is None:
                    continue
                if _vehicle_overlaps_route_segment(box_cur, corridor_route, corridor_margin_m=corridor_margin_m):
                    blocked = True
                    blocked_frame_id = int(future_record["frame_id"])
                    blocking_actor_id = int(box_future.get("id", -1)) if box_future.get("id", None) is not None else -1
                    blocking_actor_class = str(box_future.get("class", "unknown"))
                    break
            if blocked:
                break

        info.update({
            "valid": float(1.0),
            "ready": float(0.0 if blocked else 1.0),
            "source": "clear_dynamic_corridor" if not blocked else "blocked_dynamic_corridor",
            "release_frame_id": int(frame_records[next_release_idx]["frame_id"]),
            "enter_frame_id": int(enter_match["frame_id"]),
            "return_frame_id": int(return_match["frame_id"]),
            "borrow_duration_s": float(borrow_duration_frames / max(float(fps_hz), 1e-6)),
            "release_to_return_s": float(release_to_return_frames / max(float(fps_hz), 1e-6)),
            "borrow_start_distance_m": float(borrow_geom.get("enter_s_m", 0.0)),
            "borrow_distance_m": float(borrow_geom["borrow_distance_m"]),
            "peak_lateral_m": float(borrow_geom["peak_lateral_m"]),
            "blocked_frame_id": int(blocked_frame_id),
            "blocking_actor_id": int(blocking_actor_id),
            "blocking_actor_class": blocking_actor_class,
            "borrow_start_world_xy": borrow_world[0, :2].astype(float).tolist(),
            "borrow_end_world_xy": borrow_world[1, :2].astype(float).tolist(),
            "borrow_start_world_xyz": borrow_world_xyz[0, :3].astype(float).tolist(),
            "borrow_end_world_xyz": borrow_world_xyz[1, :3].astype(float).tolist(),
            "borrow_segment_world_xy": borrow_segment_world[:, :2].astype(float).tolist(),
            "borrow_segment_world_xyz": borrow_segment_world[:, :3].astype(float).tolist(),
        })
        record["release_ready"] = info


def _select_route_samples(samples, route_name=None, index=None):
    if route_name is None:
        if index is None:
            raise ValueError("Either --route_name or --index must be provided.")
        route_name = samples[index].get("route_name")
        if route_name is None:
            raise ValueError("Selected sample does not contain route_name.")

    route_samples = [s for s in samples if s.get("route_name") == route_name]
    if not route_samples:
        raise ValueError(f"No samples found for route_name={route_name}")
    route_samples.sort(key=lambda s: int(s.get("frame_id", -1)))
    return route_name, route_samples


def _scene_name_from_base_dir(base_dir):
    if not base_dir:
        return None
    if os.sep in base_dir:
        return base_dir.split(os.sep)[0]
    if "/" in base_dir:
        return base_dir.split("/")[0]
    return base_dir


def _select_scene_samples(samples, scene_name):
    scene_samples = []
    for sample in samples:
        base_dir, _ = _resolve_feature_frame_info(sample)
        if _scene_name_from_base_dir(base_dir) != scene_name:
            continue
        scene_samples.append(sample)
    if not scene_samples:
        raise ValueError(f"No samples found for scene_name={scene_name}")
    scene_samples.sort(
        key=lambda s: (
            _resolve_feature_frame_info(s)[0] or "",
            int(s.get("frame_id", -1)),
        )
    )
    return scene_name, scene_samples


def _load_route_samples_from_route_dir(route_dir, image_root):
    route_dir = os.path.realpath(route_dir)
    image_root = os.path.realpath(image_root)
    measurements_dir = os.path.join(route_dir, "measurements")
    if not os.path.isdir(measurements_dir):
        raise RuntimeError(f"measurements dir not found under route_dir={route_dir}")

    try:
        base_dir = os.path.relpath(route_dir, image_root)
    except ValueError:
        base_dir = os.path.basename(route_dir)
    route_name = os.path.basename(route_dir)

    meas_files = sorted(
        fn for fn in os.listdir(measurements_dir)
        if fn.endswith(".json.gz")
    )
    if not meas_files:
        raise RuntimeError(f"No measurement files found under {measurements_dir}")

    route_samples = []
    for fn in meas_files:
        frame_str = fn.replace(".json.gz", "")
        meas = _load_json_gz_if_exists(os.path.join(measurements_dir, fn))
        if meas is None:
            continue
        route = np.asarray(meas.get("route", []), dtype=np.float32)
        if route.ndim != 2 or route.shape[0] == 0 or route.shape[1] < 2:
            continue
        try:
            frame_id = int(frame_str)
        except Exception:
            continue
        route_samples.append({
            "route_name": route_name,
            "frame_id": frame_id,
            "route": route[:, :2].astype(np.float32),
            # Fake feature path so existing helpers can still resolve base_dir/frame_id.
            "transfuser_bev_feature": os.path.join(base_dir, "bev_features", f"{frame_str}_feature.pt"),
        })

    if not route_samples:
        raise RuntimeError(f"No valid route samples could be built from {route_dir}")

    route_samples.sort(key=lambda s: int(s.get("frame_id", -1)))
    selection_name = base_dir.replace(os.sep, "_").replace("/", "_")
    return selection_name, route_samples


def _transform_points_local_to_world(points_xy, ego_matrix):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    ego_matrix = np.asarray(ego_matrix, dtype=np.float32)
    pts_h = np.concatenate(
        [pts, np.zeros((pts.shape[0], 1), dtype=np.float32), np.ones((pts.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    world = (ego_matrix @ pts_h.T).T
    return world[:, :2].astype(np.float32)


def _transform_points_local_to_world_xyz(points_xy, ego_matrix):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((0, 3), dtype=np.float32)
    ego_matrix = np.asarray(ego_matrix, dtype=np.float32)
    pts_h = np.concatenate(
        [pts, np.zeros((pts.shape[0], 1), dtype=np.float32), np.ones((pts.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    world = (ego_matrix @ pts_h.T).T
    return world[:, :3].astype(np.float32)


def _transform_points_world_to_local(points_xy, ego_matrix):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    ego_matrix = np.asarray(ego_matrix, dtype=np.float32)
    try:
        ego_inv = np.linalg.inv(ego_matrix)
    except np.linalg.LinAlgError:
        return np.zeros((0, 2), dtype=np.float32)
    pts_h = np.concatenate(
        [pts, np.zeros((pts.shape[0], 1), dtype=np.float32), np.ones((pts.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    local = (ego_inv @ pts_h.T).T
    return local[:, :2].astype(np.float32)


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
        return np.zeros((0, 2), dtype=np.float32) if (pts.ndim != 2 or pts.shape[-1] < 2) else np.zeros((0, pts.shape[1]), dtype=np.float32)
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


def _trim_polyline_ahead(points_xy, ego_matrix, max_distance_m=60.0):
    world_pts = np.asarray(points_xy, dtype=np.float32)
    if world_pts.ndim != 2 or world_pts.shape[0] == 0 or world_pts.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)

    ego_local = _transform_points_world_to_local(world_pts, ego_matrix)
    ahead_mask = ego_local[:, 0] >= -1.0
    if np.any(ahead_mask):
        first_idx = int(np.flatnonzero(ahead_mask)[0])
        world_pts = world_pts[first_idx:]
    if world_pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if world_pts.shape[0] == 1:
        return ego_local[:1]

    kept = [world_pts[0]]
    dist_acc = 0.0
    for pt in world_pts[1:]:
        dist_acc += float(np.linalg.norm(pt - kept[-1]))
        kept.append(pt)
        if dist_acc >= float(max_distance_m):
            break
    kept = np.asarray(kept, dtype=np.float32)
    local_pts = _transform_points_world_to_local(kept, ego_matrix)
    if local_pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    local_pts = local_pts[local_pts[:, 0] >= -1.0]
    return local_pts.astype(np.float32)


def _polyline_arclengths(points_xy):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return np.zeros((0,), dtype=np.float32)
    if pts.shape[0] == 1:
        return np.zeros((1,), dtype=np.float32)
    seg_lens = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).astype(np.float32)
    return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(seg_lens, dtype=np.float32)], axis=0)


def _flip_local_y(points_xy):
    pts = np.asarray(points_xy, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[1] != 2:
        return pts
    pts[:, 1] *= -1.0
    return pts


def _build_scene_route_polyline_world(
    route_samples,
    image_root,
    dedupe_step_m=0.5,
    flip_local_y=False,
    tail_window_points=8,
    overlap_thresh_m=1.5,
):
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
        route_tail_local = route_raw[tail_start:, :2]
        if flip_local_y:
            route_tail_local = _flip_local_y(route_tail_local)
        route_tail_world = _transform_points_local_to_world_xyz(route_tail_local, ego_matrix)
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
            if tail_slice.ndim != 2 or tail_slice.shape[0] == 0 or tail_slice.shape[1] < 2:
                continue

            if prev_pt is None:
                kept = [pt.copy() for pt in tail_slice]
                prev_pt = np.asarray(kept[-1], dtype=np.float32)
                poly = np.asarray(kept, dtype=np.float32)
                cumulative_s = _polyline_length_m(poly)
                _, tail_s = _project_point_to_polyline(tail_slice[-1], poly)
                anchor_s[int(frame_id)] = 0.0 if tail_s is None else float(tail_s)
                continue

            dists_to_last = np.linalg.norm(tail_slice[:, :2] - prev_pt[None, :2], axis=1)
            overlap_idx = int(np.argmin(dists_to_last))
            if float(dists_to_last[overlap_idx]) <= float(overlap_thresh_m):
                append_slice = tail_slice[overlap_idx + 1:]
            else:
                append_slice = tail_slice

            for pt in append_slice:
                pt = np.asarray(pt, dtype=np.float32)
                step = float(np.linalg.norm(pt - prev_pt))
                if step < float(dedupe_step_m):
                    continue
                cumulative_s += step
                kept.append(pt)
                prev_pt = pt

            poly = np.asarray(kept, dtype=np.float32)
            _, tail_s = _project_point_to_polyline(tail_slice[-1], poly)
            anchor_s[int(frame_id)] = float(cumulative_s if tail_s is None else tail_s)

        if not kept:
            continue
        scene_polyline_world[base_dir] = np.asarray(kept, dtype=np.float32)
        scene_polyline_anchor_s[base_dir] = anchor_s
    return scene_polyline_world, scene_polyline_anchor_s


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
    if arc.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    total_len = float(arc[-1])
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


def _extend_local_route_with_scene_polyline(
    route_local,
    ego_matrix_current,
    scene_polyline_world,
    anchor_s=None,
    extension_step_m=1.0,
    extension_points=12,
    flip_local_y=False,
):
    route_local = np.asarray(route_local, dtype=np.float32)
    scene_polyline_world = np.asarray(scene_polyline_world, dtype=np.float32)
    if (
        route_local.ndim != 2 or route_local.shape[0] == 0 or route_local.shape[1] != 2 or
        ego_matrix_current is None or
        scene_polyline_world.ndim != 2 or scene_polyline_world.shape[0] < 2 or scene_polyline_world.shape[1] < 2
    ):
        return route_local

    if anchor_s is None:
        route_tail_local = route_local[-1:, :2]
        if flip_local_y:
            route_tail_local = _flip_local_y(route_tail_local)
        tail_world = _transform_points_local_to_world_xyz(route_tail_local, ego_matrix_current)
        if tail_world.shape[0] == 0:
            return route_local
        _, tail_s = _project_point_to_polyline(tail_world[0, :2], scene_polyline_world)
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

    if flip_local_y:
        extension_local = _flip_local_y(extension_local)

    merged = np.concatenate([route_local[:, :2], extension_local], axis=0)
    return _dedupe_polyline(merged, min_step_m=max(0.25, 0.5 * float(extension_step_m)))




def _collect_dynamic_actor_ids(current_boxes, future_frames_data, ego_matrix_current, speed_thresh_mps=0.25, motion_thresh_m=1.0):
    dynamic_ids = set()
    current_boxes = current_boxes or []
    future_frames_data = future_frames_data or []

    current_by_id = {}
    for box in current_boxes:
        cls = str(box.get("class", "")).lower()
        if cls in {"ego_car", "static"}:
            continue
        actor_id = box.get("id", None)
        if actor_id is not None:
            current_by_id[actor_id] = box
        if abs(float(box.get("speed", 0.0))) > float(speed_thresh_mps) and actor_id is not None:
            dynamic_ids.add(actor_id)

    if ego_matrix_current is None:
        return dynamic_ids
    try:
        ego_inv = np.linalg.inv(np.asarray(ego_matrix_current, dtype=np.float32))
    except np.linalg.LinAlgError:
        return dynamic_ids

    for frame_data in future_frames_data:
        if frame_data is None:
            continue
        boxes_future, ego_matrix_future = frame_data
        transform = ego_inv @ np.asarray(ego_matrix_future, dtype=np.float32)
        for box_future in boxes_future or []:
            cls = str(box_future.get("class", "")).lower()
            if cls in {"ego_car", "static"}:
                continue
            actor_id = box_future.get("id", None)
            speed = abs(float(box_future.get("speed", 0.0)))
            if actor_id is not None and speed > float(speed_thresh_mps):
                dynamic_ids.add(actor_id)
                continue
            if actor_id is None or actor_id not in current_by_id:
                continue
            box_cur = _transform_box_to_current_frame(box_future, transform)
            if box_cur is None:
                continue
            cur_pos = np.asarray(current_by_id[actor_id].get("position", [0.0, 0.0])[:2], dtype=np.float32)
            fut_pos = np.asarray(box_cur.get("position", [0.0, 0.0])[:2], dtype=np.float32)
            if float(np.linalg.norm(fut_pos - cur_pos)) > float(motion_thresh_m):
                dynamic_ids.add(actor_id)
    return dynamic_ids


def _collect_scene_nonstatic_actor_ids(route_samples, image_root, speed_thresh_mps=0.25, motion_thresh_m=1.0):
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
            rec = stats.setdefault(
                actor_id,
                {
                    "first_pos": pos_xy,
                    "last_pos": pos_xy,
                    "max_abs_speed": speed_abs,
                },
            )
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
        if actor_id is not None and actor_id in dynamic_ids:
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
            if actor_id is not None and actor_id in dynamic_ids:
                keep.append(box)
                continue
            if abs(float(box.get("speed", 0.0))) > float(speed_thresh_mps):
                keep.append(box)
        filtered.append((keep, ego_matrix_future))
    return filtered


def _oriented_box_corners(position, extent, yaw):
    x, y = float(position[0]), float(position[1])
    half_l, half_w = float(extent[0]), float(extent[1])
    corners = np.array(
        [
            [-half_l, -half_w],
            [-half_l, half_w],
            [half_l, half_w],
            [half_l, -half_w],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + np.array([x, y], dtype=np.float32)


def _to_canvas(points_xy, width, height, xlim, ylim):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts[None, :]
    x_min, x_max = xlim
    y_min, y_max = ylim
    px = (pts[:, 0] - x_min) / max(x_max - x_min, 1e-6) * (width - 1)
    # Keep the world-panel convention consistent with the BEV image convention:
    # x_forward -> right in image, y_lateral -> down in image.
    py = (pts[:, 1] - y_min) / max(y_max - y_min, 1e-6) * (height - 1)
    return np.stack([px, py], axis=1).astype(np.int32)


def _draw_polyline(img, pts_xy, color, width, height, xlim, ylim, thickness=1, closed=False):
    pts = np.asarray(pts_xy, dtype=np.float32)
    if len(pts) == 0:
        return
    pts_px = _to_canvas(pts, width, height, xlim, ylim).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_px], isClosed=closed, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_dashed_polyline(
    img,
    pts_xy,
    color,
    width,
    height,
    xlim,
    ylim,
    thickness=1,
    closed=False,
    dash_px=10.0,
    gap_px=6.0,
):
    pts = np.asarray(pts_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2:
        return
    pts_px = _to_canvas(pts, width, height, xlim, ylim).astype(np.float32)
    if closed and pts_px.shape[0] >= 2:
        pts_px = np.concatenate([pts_px, pts_px[:1]], axis=0)
    if pts_px.shape[0] < 2:
        return

    dash_px = float(max(dash_px, 1.0))
    gap_px = float(max(gap_px, 0.0))
    for idx in range(pts_px.shape[0] - 1):
        start = pts_px[idx]
        end = pts_px[idx + 1]
        seg = end - start
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            continue
        direction = seg / seg_len
        cursor = 0.0
        while cursor < seg_len:
            dash_end = min(cursor + dash_px, seg_len)
            p0 = start + direction * cursor
            p1 = start + direction * dash_end
            cv2.line(
                img,
                tuple(np.round(p0).astype(np.int32)),
                tuple(np.round(p1).astype(np.int32)),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
            cursor += dash_px + gap_px


def _draw_box(img, box, color, width, height, xlim, ylim, thickness=2, dashed=False):
    pos = box.get("position", None)
    extent = box.get("extent", None)
    if pos is None or extent is None or len(pos) < 2 or len(extent) < 2:
        return
    poly = _oriented_box_corners(pos[:2], extent[:2], float(box.get("yaw", 0.0)))
    if dashed:
        _draw_dashed_polyline(img, poly, color, width, height, xlim, ylim, thickness=thickness, closed=True)
    else:
        _draw_polyline(img, poly, color, width, height, xlim, ylim, thickness=thickness, closed=True)


def _find_box_by_id(boxes, actor_id):
    if actor_id is None:
        return None
    for box in boxes or []:
        if box.get("id") == actor_id:
            return box
    return None


def _annotate_box_label(img, box, text, color, width, height, xlim, ylim):
    pos = box.get("position", None)
    if pos is None or len(pos) < 2:
        return
    pt = _to_canvas(np.asarray(pos[:2], dtype=np.float32), width, height, xlim, ylim)[0]
    cv2.putText(
        img,
        text,
        (int(pt[0]) + 6, int(pt[1]) - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _colorize_bev(bev_gray):
    bev_gray = np.clip(bev_gray, 0, len(BEV_COLORS) - 1)
    return BEV_COLORS[bev_gray]


def _box_corners_bev(position, extent, yaw, bev_size=256, bev_ppm=2.0):
    cx, cy = float(position[0]), float(position[1])
    hl, hw = float(extent[0]), float(extent[1])
    corners_local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    corners = (rot @ corners_local.T).T + np.array([cx, cy], dtype=np.float32)
    center = bev_size // 2
    cols = center + corners[:, 0] * bev_ppm
    rows = center + corners[:, 1] * bev_ppm
    return np.stack([cols, rows], axis=1).astype(np.int32)


def _draw_boxes_on_bev(bev_color, boxes, affecting_id=None, speed_obj_id=None):
    ego_box = _find_ego_box(boxes)
    if ego_box is not None:
        ego_corners = _box_corners_bev(
            ego_box["position"][:2],
            ego_box["extent"][:2],
            float(ego_box.get("yaw", 0.0)),
        )
        cv2.polylines(
            bev_color,
            [ego_corners],
            isClosed=True,
            color=(255, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    else:
        center = bev_color.shape[0] // 2
        cv2.drawMarker(
            bev_color,
            (center, center),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )

    for box in boxes or []:
        cls = box.get("class", "")
        if cls == "ego_car":
            continue
        pos = box.get("position", None)
        extent = box.get("extent", None)
        if pos is None or extent is None or len(pos) < 2 or len(extent) < 2:
            continue
        color = (0, 255, 0) if cls in VEHICLE_CLASSES else (180, 180, 180)
        corners = _box_corners_bev(pos[:2], extent[:2], float(box.get("yaw", 0.0)))
        cv2.polylines(bev_color, [corners], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    affecting_box = _find_box_by_id(boxes, affecting_id)
    if affecting_box is not None:
        corners = _box_corners_bev(
            affecting_box["position"][:2],
            affecting_box["extent"][:2],
            float(affecting_box.get("yaw", 0.0)),
        )
        cv2.polylines(bev_color, [corners], isClosed=True, color=(255, 0, 255), thickness=3, lineType=cv2.LINE_AA)

    speed_box = _find_box_by_id(boxes, speed_obj_id)
    if speed_box is not None:
        corners = _box_corners_bev(
            speed_box["position"][:2],
            speed_box["extent"][:2],
            float(speed_box.get("yaw", 0.0)),
        )
        cv2.polylines(bev_color, [corners], isClosed=True, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)
    return bev_color


def _load_scene_panel(image_root, base_dir, frame_str, current_boxes, current_meas=None):
    rgb_candidates = [
        os.path.join(image_root, base_dir, "rgb", f"{frame_str}.jpg"),
        os.path.join(image_root, base_dir, "rgb", f"{frame_str}.png"),
        os.path.join(image_root, base_dir, "rgb", f"{frame_str}.jpeg"),
    ]
    for rgb_path in rgb_candidates:
        if os.path.exists(rgb_path):
            rgb = cv2.imread(rgb_path)
            if rgb is not None:
                return rgb

    bev_path = os.path.join(image_root, base_dir, "bev_semantics", f"{frame_str}.png")
    if os.path.exists(bev_path):
        bev_gray = cv2.imread(bev_path, cv2.IMREAD_GRAYSCALE)
        if bev_gray is not None:
            bev_color = _colorize_bev(bev_gray)
            bev_color = _draw_boxes_on_bev(
                bev_color,
                current_boxes,
                affecting_id=None if current_meas is None else current_meas.get("vehicle_affecting_id"),
                speed_obj_id=None if current_meas is None else current_meas.get("speed_reduced_by_obj_id"),
            )
            bev_color = cv2.resize(bev_color, (768, 768), interpolation=cv2.INTER_NEAREST)
            cv2.putText(
                bev_color,
                "RGB unavailable, showing BEV semantics",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return bev_color

    fallback = np.full((512, 1024, 3), 245, dtype=np.uint8)
    cv2.putText(
        fallback,
        "RGB/BEV unavailable for this frame",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    return fallback


def _load_future_frames(image_root, base_dir, frame_id, num_future):
    future_frames = []
    for k in range(1, num_future + 1):
        future_frame = f"{frame_id + k:04d}"
        boxes_path = os.path.join(image_root, base_dir, "boxes", f"{future_frame}.json.gz")
        meas_path = os.path.join(image_root, base_dir, "measurements", f"{future_frame}.json.gz")
        boxes = _load_json_gz_if_exists(boxes_path)
        meas = _load_json_gz_if_exists(meas_path)
        if boxes is None or meas is None or meas.get("ego_matrix") is None:
            future_frames.append(None)
        else:
            future_frames.append((boxes, meas["ego_matrix"]))
    return future_frames


def _render_world_panel(sample, current_boxes, current_meas, label, debug, event_name, width, height, xlim, ylim, interaction, proceed, wait_info, release_info):
    panel = np.full((height, width, 3), 250, dtype=np.uint8)
    occ = _occupancy_signal_from_label(label, interaction)
    current_cover = _cover_candidate_summary(1, debug.get("best_current"), debug, current_meas=current_meas, event_name=event_name)
    future_cover = _cover_candidate_summary(2, debug.get("best_future"), debug, current_meas=current_meas, event_name=event_name)
    veh_aff_id = None if current_meas is None else current_meas.get("vehicle_affecting_id")
    spd_red_id = None if current_meas is None else current_meas.get("speed_reduced_by_obj_id")
    veh_aff_box = _find_box_by_id(current_boxes, veh_aff_id)
    spd_red_box = _find_box_by_id(current_boxes, spd_red_id)

    route_dense = debug.get("route_dense")
    route_poly = debug.get("route_poly")
    route_front = np.asarray(sample.get("_route_front_local", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    route_all = np.asarray(sample.get("_route_input_local", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    route_ext = np.zeros((0, 2), dtype=np.float32)
    if route_all.ndim == 2 and route_front.ndim == 2 and route_all.shape[0] > route_front.shape[0]:
        route_ext = route_all[route_front.shape[0]:].astype(np.float32)

    if route_dense is not None and len(route_dense) > 0:
        dense_px = _to_canvas(route_dense, width, height, xlim, ylim)
        for pt in dense_px:
            cv2.circle(panel, tuple(pt), 1, (210, 210, 210), -1, lineType=cv2.LINE_AA)

    if route_front.ndim == 2 and route_front.shape[0] > 0:
        _draw_polyline(panel, route_front, (20, 20, 20), width, height, xlim, ylim, thickness=2)
    elif route_poly is not None and len(route_poly) > 0:
        _draw_polyline(panel, route_poly, (20, 20, 20), width, height, xlim, ylim, thickness=2)

    if route_ext.ndim == 2 and route_ext.shape[0] > 0:
        ext_start = route_front[-1:, :] if route_front.ndim == 2 and route_front.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
        ext_poly = np.concatenate([ext_start, route_ext], axis=0) if ext_start.shape[0] > 0 else route_ext
        _draw_polyline(panel, ext_poly, (0, 0, 255), width, height, xlim, ylim, thickness=2)
        ext_px = _to_canvas(route_ext, width, height, xlim, ylim)
        for idx, pt in enumerate(ext_px, start=1):
            cv2.circle(panel, tuple(pt), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                panel,
                str(idx),
                (int(pt[0]) + 5, int(pt[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 0, 160),
                1,
                cv2.LINE_AA,
            )

    ego_matrix_current = None if current_meas is None else current_meas.get("ego_matrix", None)
    borrow_start_world_xy = np.asarray(release_info.get("borrow_start_world_xy", []), dtype=np.float32)
    borrow_end_world_xy = np.asarray(release_info.get("borrow_end_world_xy", []), dtype=np.float32)
    borrow_start_world_xyz = np.asarray(release_info.get("borrow_start_world_xyz", []), dtype=np.float32)
    borrow_end_world_xyz = np.asarray(release_info.get("borrow_end_world_xyz", []), dtype=np.float32)
    borrow_segment_world_xy = np.asarray(release_info.get("borrow_segment_world_xy", []), dtype=np.float32)
    borrow_segment_world_xyz = np.asarray(release_info.get("borrow_segment_world_xyz", []), dtype=np.float32)
    if (
        (
            _is_left_turn_scene_context(current_meas, event_name=event_name) or
            _is_borrow_cross_scene_context(event_name=event_name)
        )
        and ego_matrix_current is not None
        and borrow_start_world_xy.shape == (2,)
        and borrow_end_world_xy.shape == (2,)
    ):
        corridor_local = np.zeros((0, 2), dtype=np.float32)
        if borrow_segment_world_xyz.ndim == 2 and borrow_segment_world_xyz.shape[0] >= 2 and borrow_segment_world_xyz.shape[1] >= 3:
            corridor_local = _transform_points_world_xyz_to_local(borrow_segment_world_xyz, ego_matrix_current)
        elif borrow_segment_world_xy.ndim == 2 and borrow_segment_world_xy.shape[0] >= 2 and borrow_segment_world_xy.shape[1] == 2:
            corridor_local = _transform_points_world_to_local(borrow_segment_world_xy, ego_matrix_current)
        elif borrow_start_world_xyz.shape == (3,) and borrow_end_world_xyz.shape == (3,):
            corridor_local = _transform_points_world_xyz_to_local(
                np.stack([borrow_start_world_xyz, borrow_end_world_xyz], axis=0),
                ego_matrix_current,
            )
        else:
            corridor_local = _transform_points_world_to_local(
                np.stack([borrow_start_world_xy, borrow_end_world_xy], axis=0),
                ego_matrix_current,
            )
        if corridor_local.ndim == 2 and corridor_local.shape[0] >= 2 and corridor_local.shape[1] == 2:
            corridor_color = (60, 220, 60)
            if _is_borrow_cross_scene_context(event_name=event_name):
                corridor_color = (80, 235, 120)
            _draw_polyline(panel, corridor_local, corridor_color, width, height, xlim, ylim, thickness=2)
            start_px = _to_canvas(corridor_local[0], width, height, xlim, ylim)[0]
            end_px = _to_canvas(corridor_local[-1], width, height, xlim, ylim)[0]
            cv2.circle(panel, tuple(start_px), 6, corridor_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(panel, tuple(end_px), 6, (255, 200, 0), -1, lineType=cv2.LINE_AA)
            cv2.drawMarker(panel, tuple(start_px), corridor_color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)
            cv2.drawMarker(panel, tuple(end_px), (255, 200, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)
            cv2.putText(panel, "start", (int(start_px[0]) + 6, int(start_px[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 140, 20), 1, cv2.LINE_AA)
            cv2.putText(panel, "end", (int(end_px[0]) + 6, int(end_px[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 110, 0), 1, cv2.LINE_AA)

    compare_front = route_front
    compare_raw_ext = np.asarray(sample.get("_route_extension_raw", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    compare_flip_ext = np.asarray(sample.get("_route_extension_yflip", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    compare_mode = str(sample.get("_route_mode", "")) == "scene_polyline_compare"
    if compare_mode and compare_front.ndim == 2 and compare_front.shape[0] > 0:
        tail = compare_front[-1:, :]
        if compare_raw_ext.ndim == 2 and compare_raw_ext.shape[0] > 0:
            _draw_polyline(
                panel,
                np.concatenate([tail, compare_raw_ext], axis=0),
                (0, 0, 255),
                width,
                height,
                xlim,
                ylim,
                thickness=2,
            )
        if compare_flip_ext.ndim == 2 and compare_flip_ext.shape[0] > 0:
            _draw_polyline(
                panel,
                np.concatenate([tail, compare_flip_ext], axis=0),
                (255, 180, 0),
                width,
                height,
                xlim,
                ylim,
                thickness=2,
            )

    ego_px = _to_canvas(np.array([[0.0, 0.0]], dtype=np.float32), width, height, xlim, ylim)[0]
    ego_box = _find_ego_box(current_boxes)
    if ego_box is not None:
        _draw_box(panel, ego_box, (255, 0, 0), width, height, xlim, ylim, thickness=2)
    cv2.drawMarker(panel, tuple(ego_px), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

    for box in current_boxes or []:
        cls = box.get("class", "")
        if cls == "ego_car":
            continue
        color = (165, 165, 165) if cls in VEHICLE_CLASSES else (200, 200, 200)
        _draw_box(panel, box, color, width, height, xlim, ylim, thickness=1)

    if debug.get("best_current") is not None:
        cur = debug["best_current"]
        cur_color = (0, 0, 255) if int(current_cover["interaction"]["mode"]) == 1 else (0, 165, 255)
        cur_label_color = (0, 0, 180) if int(current_cover["interaction"]["mode"]) == 1 else (0, 120, 220)
        cur_prefix = "cur_chase" if int(current_cover["interaction"]["mode"]) == 1 else "cur_meet"
        _draw_box(panel, cur["box"], cur_color, width, height, xlim, ylim, thickness=3)
        cur_actor_id = cur.get("box", {}).get("id", None)
        if cur_actor_id is not None:
            _annotate_box_label(panel, cur["box"], f"{cur_prefix}:{int(cur_actor_id)}", cur_label_color, width, height, xlim, ylim)
        cover_pt = np.asarray(cur["cover"]["route_point"], dtype=np.float32)
        cover_px = _to_canvas(cover_pt, width, height, xlim, ylim)[0]
        cv2.drawMarker(panel, tuple(cover_px), cur_color, markerType=cv2.MARKER_STAR, markerSize=18, thickness=2)
        cv2.line(panel, tuple(ego_px), tuple(cover_px), cur_color, 1, lineType=cv2.LINE_AA)

    if debug.get("best_future") is not None:
        fut = debug["best_future"]
        frame_index = int(fut.get("frame_index", -1))
        if fut.get("current_box") is not None and 0 < frame_index <= 4:
            fut_color = (0, 0, 255) if int(future_cover["interaction"]["mode"]) == 1 else (0, 165, 255)
            fut_label_color = (0, 0, 180) if int(future_cover["interaction"]["mode"]) == 1 else (0, 120, 220)
            fut_prefix = "fut_chase" if int(future_cover["interaction"]["mode"]) == 1 else "fut_meet"
            _draw_box(panel, fut["current_box"], fut_color, width, height, xlim, ylim, thickness=2)
            fut_actor_id = fut.get("current_box", {}).get("id", None)
            if fut_actor_id is not None:
                meet_text = f"{fut_prefix}:{int(fut_actor_id)}"
                if frame_index > 0:
                    meet_text += f"@+{frame_index}"
                _annotate_box_label(panel, fut["current_box"], meet_text, fut_label_color, width, height, xlim, ylim)
        if 0 < frame_index <= 4:
            fut_color = (0, 0, 255) if int(future_cover["interaction"]["mode"]) == 1 else (0, 165, 255)
            _draw_box(panel, fut["box_current_frame"], fut_color, width, height, xlim, ylim, thickness=3, dashed=True)
            cover_pt = np.asarray(fut["cover"]["route_point"], dtype=np.float32)
            bg_pos = np.asarray(fut["bg_pos"], dtype=np.float32)
            cover_px = _to_canvas(cover_pt, width, height, xlim, ylim)[0]
            bg_px = _to_canvas(bg_pos, width, height, xlim, ylim)[0]
            cv2.drawMarker(panel, tuple(cover_px), fut_color, markerType=cv2.MARKER_STAR, markerSize=18, thickness=2)
            cv2.circle(panel, tuple(bg_px), 5, fut_color, -1, lineType=cv2.LINE_AA)
            cv2.line(panel, tuple(ego_px), tuple(cover_px), fut_color, 1, lineType=cv2.LINE_AA)
            cv2.line(panel, tuple(bg_px), tuple(cover_px), fut_color, 1, lineType=cv2.LINE_AA)

    same_actor = (
        veh_aff_id is not None and spd_red_id is not None and int(veh_aff_id) == int(spd_red_id)
    )
    if same_actor and veh_aff_box is not None:
        _draw_box(panel, veh_aff_box, (0, 220, 255), width, height, xlim, ylim, thickness=4)
        _draw_box(panel, veh_aff_box, (255, 0, 255), width, height, xlim, ylim, thickness=2)
        _annotate_box_label(panel, veh_aff_box, f"aff/spd:{int(veh_aff_id)}", (120, 0, 180), width, height, xlim, ylim)
    else:
        if spd_red_box is not None:
            _draw_box(panel, spd_red_box, (0, 220, 255), width, height, xlim, ylim, thickness=2)
            _annotate_box_label(panel, spd_red_box, f"spd_red:{int(spd_red_id)}", (0, 160, 220), width, height, xlim, ylim)
        if veh_aff_box is not None:
            _draw_box(panel, veh_aff_box, (255, 0, 255), width, height, xlim, ylim, thickness=3)
            _annotate_box_label(panel, veh_aff_box, f"aff:{int(veh_aff_id)}", (180, 0, 180), width, height, xlim, ylim)

    if veh_aff_box is not None:
        aff_pos = np.asarray(veh_aff_box.get("position", [0.0, 0.0])[:2], dtype=np.float32)
        aff_px = _to_canvas(aff_pos, width, height, xlim, ylim)[0]
        cv2.line(panel, tuple(ego_px), tuple(aff_px), (255, 0, 255), 1, lineType=cv2.LINE_AA)

    release_ready_str = "NA" if float(release_info.get("valid", 0.0)) <= 0.5 else str(int(release_info.get("ready", 0.0) > 0.5))
    spd_red_type = "" if current_meas is None else str(current_meas.get("speed_reduced_by_obj_type", "") or "")
    spd_red_type = spd_red_type.split(".")[-1] if spd_red_type else "none"
    throttle = 0.0 if current_meas is None else float(current_meas.get("throttle", 0.0))
    brake = 0 if current_meas is None else int(bool(current_meas.get("brake", False)))
    control_brake = 0 if current_meas is None else int(bool(current_meas.get("control_brake", False)))
    veh_hazard = 0 if current_meas is None else int(bool(current_meas.get("vehicle_hazard", False)))
    spd_red_dist = -1.0 if current_meas is None else float(current_meas.get("speed_reduced_by_obj_distance", -1.0) or -1.0)
    route_name = str(sample.get("route_name", "unknown"))
    if len(route_name) > 58:
        route_name = route_name[:55] + "..."
    def _fmt_panel_val(x, fmt="{:.2f}"):
        try:
            x = float(x)
        except Exception:
            return "NA"
        if not np.isfinite(x):
            return "NA"
        return fmt.format(x)
    cross_wait_time = float(sample.get("_cross_wait_time_s", np.nan))
    cross_wait_valid = float(sample.get("_cross_wait_valid", 0.0))
    cross_wait_active = float(sample.get("_cross_wait_active", 0.0))
    cross_wait_speed = float(sample.get("_cross_wait_speed_mps", np.nan))
    cross_wait_dist = float(sample.get("_cross_wait_start_dist", np.nan))
    lines = [
        f"{event_name} | frame {int(sample.get('frame_id', -1)):04d}",
        route_name,
        f"cover={occ['case_name']}  interact={interaction['name']}  occ={occ['risk']:.3f}  proceed={proceed['risk']:.3f}  v={wait_info['speed_mps']:.2f}",
        f"wait={int(wait_info['wait_state'])}  release={int(wait_info['release_pulse'])}  rel_ready={release_ready_str}",
        f"cross wait: t={_fmt_panel_val(cross_wait_time)}s valid={int(cross_wait_valid>0.5)} act={int(cross_wait_active>0.5)} v={_fmt_panel_val(cross_wait_speed)} dStart={_fmt_panel_val(cross_wait_dist)}",
        f"route_mode={sample.get('_route_mode', 'local')}  route_len={float(sample.get('_route_len_m', 0.0)):.2f}m  route_pts={int(sample.get('_route_num_points', 0))}",
    ]
    if compare_mode:
        lines.append(
            f"compare ext: raw=red ({int(compare_raw_ext.shape[0])})  yflip=cyan ({int(compare_flip_ext.shape[0])})"
        )
    y = 22
    for line in lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
        y += 22

    return panel


def _compose_frame(rgb, panel, sample, label, current_meas, current_boxes, interaction, proceed, wait_info, release_info):
    occ = _occupancy_signal_from_label(label, interaction)
    event_name = sample.get("_event_name")
    stage1_label = _stage1_label_payload(sample, current_meas, current_boxes=current_boxes)
    current_cover = stage1_label.get("current_cover", _cover_candidate_summary(0, None, {}))
    future_cover = stage1_label.get("future_cover", _cover_candidate_summary(0, None, {}))
    speed_curve = stage1_label["speed_curve"]
    merge_episode = stage1_label.get("merge_episode", _default_merge_episode_debug())
    rgb_h, rgb_w = rgb.shape[:2]
    panel_h, panel_w = panel.shape[:2]
    target_h = max(rgb_h, panel_h)
    rgb_resized = cv2.resize(rgb, (int(rgb_w * target_h / max(rgb_h, 1)), target_h), interpolation=cv2.INTER_LINEAR)
    panel_resized = cv2.resize(panel, (int(panel_w * target_h / max(panel_h, 1)), target_h), interpolation=cv2.INTER_LINEAR)
    top = np.concatenate([rgb_resized, panel_resized], axis=1)

    bar_h = 560
    bar = np.zeros((bar_h, top.shape[1], 3), dtype=np.uint8)
    speed = float((current_meas or {}).get("speed", 0.0))
    target_speed = float((current_meas or {}).get("target_speed", 0.0))
    veh_aff_id = (current_meas or {}).get("vehicle_affecting_id", None)
    spd_red_id = (current_meas or {}).get("speed_reduced_by_obj_id", None)
    release_ready_str = "NA" if float(release_info.get("valid", 0.0)) <= 0.5 else str(int(release_info.get("ready", 0.0) > 0.5))
    left_x = 12
    right_x = top.shape[1] // 2 + 12

    def _fmt_val(x, fmt="{:.2f}"):
        return "NA" if x is None or not np.isfinite(float(x)) else fmt.format(float(x))

    def _fmt_sample_pairs(speeds, risks, limit=None):
        speeds = np.asarray(speeds, dtype=np.float32)
        risks = np.asarray(risks, dtype=np.float32)
        if speeds.ndim != 1 or risks.ndim != 1 or speeds.size == 0 or risks.size == 0:
            return "NA"
        n = min((speeds.size if limit is None else int(limit)), speeds.size, risks.size)
        return " ".join(f"{float(speeds[i]):.0f}:{float(risks[i]):.2f}" for i in range(n))

    def _fmt_speeds(speeds, limit=None):
        speeds = np.asarray(speeds, dtype=np.float32)
        if speeds.ndim != 1 or speeds.size == 0:
            return "NA"
        n = min((speeds.size if limit is None else int(limit)), speeds.size)
        return " ".join(f"{float(speeds[i]):.0f}" for i in range(n))

    cross_wait_time = float(speed_curve.get("cross_wait_time_s", np.nan))
    cross_wait_valid = float(speed_curve.get("cross_wait_valid", 0.0))
    cross_wait_dist = float(sample.get("_cross_wait_start_dist", np.nan))
    cross_wait_speed = float(sample.get("_cross_wait_speed_mps", np.nan))
    cross_wait_active = float(sample.get("_cross_wait_active", 0.0))
    left_lines = [
        ("speed={:.2f}  target={:.2f}  wait={}  release={}  rel_ready={}".format(
            speed, target_speed, int(wait_info["wait_state"]), int(wait_info["release_pulse"]), release_ready_str
        ), (255, 255, 255), 0.62),
        ("corridor src={}  ctx_frame={}  anchor={}".format(
            str(release_info.get("source", "n/a")),
            int(release_info.get("context_frame_id", -1)),
            int(release_info.get("anchor_actor_id", -1)),
        ), (180, 235, 180), 0.50),
        ("primary case={}  interact={}  occ={:.3f}  proceed={:.3f}".format(
            int(label["case"]), interaction["name"], float(occ["risk"]), float(proceed["risk"])
        ), (200, 200, 200), 0.58),
    ]
    left_lines.append(
        ("cross wait: t={}s valid={} act={} v={} dStart={}".format(
            _fmt_val(cross_wait_time),
            int(cross_wait_valid > 0.5),
            int(cross_wait_active > 0.5),
            _fmt_val(cross_wait_speed),
            _fmt_val(cross_wait_dist),
        ), (120, 200, 255), 0.52)
    )
    left_lines.extend([
        ("aff_id={}  spd_red_id={}  route={} ({:.1f}m, {}pts)".format(
            -1 if veh_aff_id is None else int(veh_aff_id),
            -1 if spd_red_id is None else int(spd_red_id),
            sample.get("_route_mode", "local"),
            float(sample.get("_route_len_m", 0.0)),
            int(sample.get("_route_num_points", 0)),
        ), (190, 190, 190), 0.56),
        ("corridor route: {:.1f}m, {}pts".format(
            float(sample.get("_route_corridor_len_m", 0.0)),
            int(sample.get("_route_corridor_num_points", 0)),
        ), (190, 190, 190), 0.54),
        ("borrow_t={}s  rel_frame={}  ret_frame={}".format(
            _fmt_val(release_info.get("release_to_return_s", np.nan)),
            int(release_info.get("release_frame_id", -1)),
            int(release_info.get("return_frame_id", -1)),
        ), (255, 210, 140), 0.56),
        ("speed_curve(center=speed): v*={}".format(
            _fmt_speeds(speed_curve["sample_speeds_mps"]),
        ), (150, 220, 255), 0.50),
        ("E_total={}".format(
            _fmt_sample_pairs(speed_curve["sample_speeds_mps"], speed_curve["total_risks"]),
        ), (150, 220, 255), 0.50),
        ("E_chase={}".format(
            _fmt_sample_pairs(speed_curve["sample_speeds_mps"], speed_curve["chase_risks"]),
        ), (150, 220, 255), 0.48),
        ("E_meet={}".format(
            _fmt_sample_pairs(speed_curve["sample_speeds_mps"], speed_curve["meet_risks"]),
        ), (150, 220, 255), 0.48),
    ])
    merge_yld_curve = np.asarray(speed_curve.get("merge_yld_risks", []), dtype=np.float32)
    merge_go_curve = np.asarray(speed_curve.get("merge_go_risks", []), dtype=np.float32)
    borrow_yld_curve = np.asarray(speed_curve.get("borrow_yld_risks", []), dtype=np.float32)
    borrow_go_curve = np.asarray(speed_curve.get("borrow_go_risks", []), dtype=np.float32)
    show_merge_split = (
        merge_yld_curve.shape == np.asarray(speed_curve["sample_speeds_mps"], dtype=np.float32).shape and
        merge_go_curve.shape == np.asarray(speed_curve["sample_speeds_mps"], dtype=np.float32).shape and
        (
            np.any(merge_yld_curve > 1e-4) or
            np.any(merge_go_curve > 1e-4) or
            str(speed_curve.get("meet_debug", {}).get("subtype", "none")) == "merge_meet"
        )
    )
    if show_merge_split:
        left_lines.append(
            ("E_merge_yld={}".format(
                _fmt_sample_pairs(speed_curve["sample_speeds_mps"], merge_yld_curve),
            ), (120, 220, 255), 0.48)
        )
        left_lines.append(
            ("E_merge_go={}".format(
                _fmt_sample_pairs(speed_curve["sample_speeds_mps"], merge_go_curve),
            ), (120, 255, 180), 0.48)
        )
    show_borrow_split = (
        borrow_yld_curve.shape == np.asarray(speed_curve["sample_speeds_mps"], dtype=np.float32).shape and
        borrow_go_curve.shape == np.asarray(speed_curve["sample_speeds_mps"], dtype=np.float32).shape and
        (
            np.any(borrow_yld_curve > 1e-4) or
            np.any(borrow_go_curve > 1e-4) or
            str(speed_curve.get("meet_debug", {}).get("subtype", "none")) == "borrow_cross_meet"
        )
    )
    if show_borrow_split:
        left_lines.append(
            ("E_borrow_yld={}".format(
                _fmt_sample_pairs(speed_curve["sample_speeds_mps"], borrow_yld_curve),
            ), (255, 210, 120), 0.48)
        )
        left_lines.append(
            ("E_borrow_go={}".format(
                _fmt_sample_pairs(speed_curve["sample_speeds_mps"], borrow_go_curve),
            ), (180, 255, 120), 0.48)
        )
    if np.any(np.asarray(speed_curve.get("ped_risks", []), dtype=np.float32) > 1e-4):
        left_lines.append(
            ("E_ped={}".format(
                _fmt_sample_pairs(speed_curve["sample_speeds_mps"], speed_curve["ped_risks"]),
            ), (150, 220, 255), 0.48)
        )

    meet_debug = speed_curve.get("meet_debug", {})
    chase_debug = speed_curve.get("chase_debug", {})

    if float(meet_debug.get("valid", 0.0)) > 0.5:
        meet_subtype = str(meet_debug.get("subtype", "meet"))
        if "cross" in meet_subtype:
            left_lines.append(
                ("{} dbg: dE={} dB={} dBend={} vB={} tBin={} tBout={} tBend={} tEout={} cLen={}".format(
                    meet_subtype,
                    _fmt_val(meet_debug.get("d_ego_m", np.nan)),
                    _fmt_val(meet_debug.get("d_bg_m", np.nan)),
                    _fmt_val(meet_debug.get("d_bg_to_end_m", np.nan)),
                    _fmt_val(meet_debug.get("bg_speed_mps", np.nan)),
                    _fmt_val(meet_debug.get("t_bg_s", np.nan)),
                    _fmt_val(meet_debug.get("t_bg_exit_s", np.nan)),
                    _fmt_val(meet_debug.get("t_bg_to_end_s", np.nan)),
                    _fmt_val(meet_debug.get("t_ego_exit_s", np.nan)),
                    _fmt_val(meet_debug.get("conflict_len_m", np.nan)),
                ), (120, 200, 255), 0.50)
            )
            if meet_subtype == "borrow_cross_meet":
                left_lines.append(
                    ("borrow dbg: dStart={} dBorrow={}".format(
                        _fmt_val(meet_debug.get("borrow_start_distance_m", np.nan)),
                        _fmt_val(meet_debug.get("borrow_total_distance_m", np.nan)),
                    ), (255, 210, 120), 0.48)
                )
                left_lines.append(
                    ("borrow v: v_go_min={}  v_yld_max={}".format(
                        _fmt_val(meet_debug.get("v_go_min_mps", np.nan)),
                        _fmt_val(meet_debug.get("v_yield_max_mps", np.nan)),
                    ), (255, 210, 120), 0.48)
                )
        else:
            left_lines.append(
                ("{} dbg: dE={} dB={} vB={} tB={} gapB={} cLen={}".format(
                    meet_subtype,
                    _fmt_val(meet_debug.get("d_ego_m", np.nan)),
                    _fmt_val(meet_debug.get("d_bg_m", np.nan)),
                    _fmt_val(meet_debug.get("bg_speed_mps", np.nan)),
                    _fmt_val(meet_debug.get("t_bg_s", np.nan)),
                    _fmt_val(meet_debug.get("safe_gap_bg_m", np.nan)),
                    _fmt_val(meet_debug.get("conflict_len_m", np.nan)),
                ), (120, 200, 255), 0.50)
            )
            left_lines.append(
                ("v_eq={}  v_go={}  v_bmin={}  v_need={}  v_yld={}".format(
                    _fmt_val(meet_debug.get("v_equal_mps", np.nan)),
                    _fmt_val(meet_debug.get("v_go_min_mps", np.nan)),
                    _fmt_val(meet_debug.get("v_behind_min_mps", np.nan)),
                    _fmt_val(meet_debug.get("v_go_need_mps", np.nan)),
                    _fmt_val(meet_debug.get("v_yield_max_mps", np.nan)),
                ), (120, 200, 255), 0.50)
            )
        if bool(stage1_label.get("speed_curve_future_persisted", False)):
            left_lines.append(
                ("{} source=persisted_after_merge".format(meet_subtype), (120, 200, 255), 0.48)
            )
    if float(meet_debug.get("valid", 0.0)) <= 0.5 and np.isfinite(float(meet_debug.get("context_conflict_len_m", np.nan))):
        left_lines.append(
            ("left-junction corridor: cLen={}".format(
                _fmt_val(meet_debug.get("context_conflict_len_m", np.nan)),
            ), (120, 200, 255), 0.50)
        )
    if float(chase_debug.get("valid", 0.0)) > 0.5:
        left_lines.append(
            ("chase dbg: gap={} gapS={} lead_v={}".format(
                _fmt_val(chase_debug.get("gap_m", np.nan)),
                _fmt_val(chase_debug.get("safe_gap_cur_m", np.nan)),
                _fmt_val(chase_debug.get("lead_speed_mps", np.nan)),
            ), (120, 200, 255), 0.50)
        )

    cur_type = current_cover["interaction"].get("subtype") or current_cover["interaction"]["name"]
    fut_type = future_cover["interaction"].get("subtype") or future_cover["interaction"]["name"]
    merge_phase = str(merge_episode.get("phase", "none"))
    merge_end_state = str(merge_episode.get("end_state", "none"))
    merge_frame_role = str(merge_episode.get("frame_role", "none"))
    merge_actor_ids = ",".join(str(int(actor_id)) for actor_id in merge_episode.get("actor_ids", [])[:4]) or "-"
    merge_switch_frames = ",".join(str(int(frame_id)) for frame_id in merge_episode.get("actor_switch_frames", [])[:4]) or "-"
    merge_color = (200, 200, 200)
    if merge_phase == "yld":
        merge_color = (120, 220, 255)
    elif merge_phase == "go":
        merge_color = (120, 255, 160)
    right_lines = [
        ("current cover", (255, 255, 255), 0.62),
        ("id={}  type={}  dist={}m  ttc={}s".format(
            int(current_cover["actor_id"]),
            cur_type,
            _fmt_val(current_cover["distance"]),
            _fmt_val(current_cover["ttc"]),
        ), (0, 0, 255) if cur_type == "chase" else (0, 165, 255), 0.58),
        ("block={}  src={}".format(
            _fmt_val(current_cover["block_risk"], "{:.3f}"),
            current_cover["interaction"]["source"],
        ), (160, 160, 160), 0.54),
        ("future cover", (255, 255, 255), 0.62),
        ("id={}@+{}  type={}  dist={}m  ttc={}s".format(
            int(future_cover["actor_id"]),
            int(future_cover["frame_index"]),
            fut_type,
            _fmt_val(future_cover["distance"]),
            _fmt_val(future_cover["ttc"]),
        ), (0, 0, 255) if fut_type == "chase" else (0, 165, 255), 0.58),
        ("d_ego={}m  d_bg={}m  src={}".format(
            _fmt_val(future_cover["d_ego"]),
            _fmt_val(future_cover["d_bg"]),
            future_cover["interaction"]["source"],
        ), (160, 160, 160), 0.54),
        ("merge ep={}  phase={}  no_go={}  end={}".format(
            int(merge_episode.get("episode_id", -1)),
            merge_phase,
            int(float(merge_episode.get("no_go", 0.0)) > 0.5),
            merge_end_state,
        ), merge_color, 0.56),
        ("role={}  fmerge_n={}  fgrace={}  rhold={}  pgrace={}".format(
            merge_frame_role,
            int(merge_episode.get("future_merge_count", 0)),
            int(merge_episode.get("future_grace_index", 0)),
            int(merge_episode.get("red_light_hold_index", 0)),
            int(merge_episode.get("post_go_grace_index", 0)),
        ), (170, 170, 170), 0.52),
        ("start={}  end={}  go={}  res={}".format(
            int(merge_episode.get("start_frame", -1)),
            int(merge_episode.get("end_frame", -1)),
            int(merge_episode.get("go_frame", -1)),
            int(merge_episode.get("resolution_actor_id", -1)),
        ), (170, 170, 170), 0.54),
        ("actors={}  switch={}".format(
            merge_actor_ids,
            merge_switch_frames,
        ), (170, 170, 170), 0.52),
    ]

    y = 30
    for text, color, scale in left_lines:
        cv2.putText(bar, text, (left_x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 28

    y = 30
    for text, color, scale in right_lines:
        cv2.putText(bar, text, (right_x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 28
    return np.concatenate([top, bar], axis=0)


def _stage1_label_payload(sample_vis, current_meas, current_boxes=None):
    merge_default = _default_merge_episode_debug()

    def _merge_episode_payload(precomputed=None):
        merge_episode = sample_vis.get("_merge_episode", None)
        if not isinstance(merge_episode, dict) and isinstance(precomputed, dict):
            merge_episode = precomputed.get("merge_episode", None)
        if not isinstance(merge_episode, dict):
            return dict(merge_default)
        payload = dict(merge_default)
        payload.update(dict(merge_episode))
        payload["actor_ids"] = [int(actor_id) for actor_id in payload.get("actor_ids", [])]
        payload["actor_switch_frames"] = [int(frame_id) for frame_id in payload.get("actor_switch_frames", [])]
        return payload

    def _merge_motion_payload(precomputed=None):
        merge_motion = sample_vis.get("_merge_motion", None)
        if not isinstance(merge_motion, dict) and isinstance(precomputed, dict):
            merge_motion = precomputed.get("merge_motion", None)
        if isinstance(merge_motion, dict):
            return dict(merge_motion)
        debug = sample_vis.get("_debug", {})
        return _build_merge_motion_context(
            current_meas=current_meas,
            route_dense=debug.get("route_dense"),
        )

    precomputed = sample_vis.get("stage1_speed_debug")
    runtime_speed_overrides = (
        "_release_info" in sample_vis or
        "_cross_wait_time_s" in sample_vis or
        "_speed_curve_future_cover" in sample_vis
    )
    if isinstance(precomputed, dict) and "speed_curve" in precomputed and not runtime_speed_overrides:
        speed_curve = {
            "sample_speeds_mps": np.asarray(precomputed.get("speed_curve", {}).get("sample_speeds_mps", []), dtype=np.float32).astype(float).tolist(),
            "total_risks": np.asarray(precomputed.get("speed_curve", {}).get("total_risks", []), dtype=np.float32).astype(float).tolist(),
            "chase_risks": np.asarray(precomputed.get("speed_curve", {}).get("chase_risks", []), dtype=np.float32).astype(float).tolist(),
            "meet_risks": np.asarray(precomputed.get("speed_curve", {}).get("meet_risks", []), dtype=np.float32).astype(float).tolist(),
            "merge_yld_risks": np.asarray(precomputed.get("speed_curve", {}).get("merge_yld_risks", []), dtype=np.float32).astype(float).tolist(),
            "merge_go_risks": np.asarray(precomputed.get("speed_curve", {}).get("merge_go_risks", []), dtype=np.float32).astype(float).tolist(),
            "borrow_yld_risks": np.asarray(precomputed.get("speed_curve", {}).get("borrow_yld_risks", []), dtype=np.float32).astype(float).tolist(),
            "borrow_go_risks": np.asarray(precomputed.get("speed_curve", {}).get("borrow_go_risks", []), dtype=np.float32).astype(float).tolist(),
            "ped_risks": np.asarray(precomputed.get("speed_curve", {}).get("ped_risks", []), dtype=np.float32).astype(float).tolist(),
            "cross_wait_time_s": float(precomputed.get("speed_curve", {}).get("cross_wait_time_s", 0.0)),
            "cross_wait_valid": float(precomputed.get("speed_curve", {}).get("cross_wait_valid", 0.0)),
            "chase_debug": dict(precomputed.get("speed_curve", {}).get("chase_debug", {})),
            "meet_debug": dict(precomputed.get("speed_curve", {}).get("meet_debug", {})),
        }
        if "cross_wait_time_s" not in precomputed.get("speed_curve", {}):
            meet_debug = speed_curve.get("meet_debug", {})
            subtype = str(meet_debug.get("subtype", "none"))
            if "cross" in subtype:
                t_exit = float(meet_debug.get("t_bg_exit_s", np.nan))
                t_bg = float(meet_debug.get("t_bg_s", np.nan))
                t_wait = t_exit if np.isfinite(t_exit) else t_bg
                if np.isfinite(t_wait):
                    speed_curve["cross_wait_time_s"] = float(max(t_wait, 0.0))
                    speed_curve["cross_wait_valid"] = 1.0
                else:
                    speed_curve["cross_wait_time_s"] = 0.0
                    speed_curve["cross_wait_valid"] = 0.0
        if "_cross_wait_time_s" in sample_vis:
            speed_curve["cross_wait_time_s"] = float(sample_vis.get("_cross_wait_time_s", 0.0))
            speed_curve["cross_wait_valid"] = float(sample_vis.get("_cross_wait_valid", 0.0))
        meet_subtype = str(speed_curve.get("meet_debug", {}).get("subtype", "none"))
        if meet_subtype == "borrow_cross_meet":
            borrow_yld = np.asarray(speed_curve.get("borrow_yld_risks", []), dtype=np.float32)
            borrow_go = np.asarray(speed_curve.get("borrow_go_risks", []), dtype=np.float32)
            if (
                borrow_yld.shape == borrow_go.shape and
                borrow_yld.size > 0 and
                (np.any(borrow_yld > 1e-5) or np.any(borrow_go > 1e-5))
            ):
                meet_risks = np.minimum(borrow_yld, borrow_go)
                chase_risks = np.asarray(speed_curve.get("chase_risks", []), dtype=np.float32)
                ped_risks = np.asarray(speed_curve.get("ped_risks", []), dtype=np.float32)
                total_risks = np.maximum(np.maximum(chase_risks, meet_risks), ped_risks)
                speed_curve["meet_risks"] = meet_risks.astype(np.float32).astype(float).tolist()
                speed_curve["total_risks"] = total_risks.astype(np.float32).astype(float).tolist()
        return {
            "current_cover": dict(precomputed.get("current_cover", {})),
            "future_cover": dict(precomputed.get("future_cover", {})),
            "ped_current_cover": dict(precomputed.get("ped_current_cover", {})),
            "ped_future_cover": dict(precomputed.get("ped_future_cover", {})),
            "speed_curve_future_cover": dict(precomputed.get("speed_curve_future_cover", {})),
            "speed_curve_future_persisted": bool(precomputed.get("speed_curve_future_persisted", False)),
            "merge_episode": _merge_episode_payload(precomputed),
            "merge_motion": _merge_motion_payload(precomputed),
            "speed_curve": speed_curve,
        }

    debug = sample_vis.get("_debug", {})
    event_name = sample_vis.get("_event_name")
    current_cover = _cover_candidate_summary(1, debug.get("best_current"), debug, current_meas=current_meas, event_name=event_name)
    future_cover = _cover_candidate_summary(2, debug.get("best_future"), debug, current_meas=current_meas, event_name=event_name)
    speed_curve_future_cover = sample_vis.get("_speed_curve_future_cover", future_cover)
    route_for_corridor = sample_vis.get("_route_corridor_input_local", sample_vis.get("_route_input_local"))
    speed_curve = _build_speed_curve_debug(
        current_cover,
        speed_curve_future_cover,
        current_meas,
        current_boxes=current_boxes,
        event_name=event_name,
        release_info=sample_vis.get("_release_info"),
        route_local=route_for_corridor,
    )
    sample_speeds = np.asarray(speed_curve["sample_speeds_mps"], dtype=np.float32)
    chase_risks = np.asarray(speed_curve["chase_risks"], dtype=np.float32)
    meet_risks = np.asarray(speed_curve["meet_risks"], dtype=np.float32)
    merge_yld_risks = np.asarray(speed_curve.get("merge_yld_risks", np.zeros(sample_speeds.shape, dtype=np.float32)), dtype=np.float32)
    merge_go_risks = np.asarray(speed_curve.get("merge_go_risks", np.zeros(sample_speeds.shape, dtype=np.float32)), dtype=np.float32)
    borrow_yld_risks = np.asarray(speed_curve.get("borrow_yld_risks", np.zeros(sample_speeds.shape, dtype=np.float32)), dtype=np.float32)
    borrow_go_risks = np.asarray(speed_curve.get("borrow_go_risks", np.zeros(sample_speeds.shape, dtype=np.float32)), dtype=np.float32)
    ped_risks = np.zeros(sample_speeds.shape, dtype=np.float32)
    if all(k in sample_vis for k in ("speed_sample_values", "speed_risk_chase_values", "speed_risk_meet_values", "speed_risk_ped_values")):
        pre_speeds = np.asarray(sample_vis.get("speed_sample_values", []), dtype=np.float32)
        pre_chase = np.asarray(sample_vis.get("speed_risk_chase_values", []), dtype=np.float32)
        pre_meet = np.asarray(sample_vis.get("speed_risk_meet_values", []), dtype=np.float32)
        pre_merge_yld = np.asarray(sample_vis.get("speed_risk_merge_yld_values", merge_yld_risks), dtype=np.float32)
        pre_merge_go = np.asarray(sample_vis.get("speed_risk_merge_go_values", merge_go_risks), dtype=np.float32)
        pre_borrow_yld = np.asarray(sample_vis.get("speed_risk_borrow_yld_values", borrow_yld_risks), dtype=np.float32)
        pre_borrow_go = np.asarray(sample_vis.get("speed_risk_borrow_go_values", borrow_go_risks), dtype=np.float32)
        pre_ped = np.asarray(sample_vis.get("speed_risk_ped_values", []), dtype=np.float32)
        if (
            pre_speeds.shape == sample_speeds.shape == pre_chase.shape == pre_meet.shape == pre_ped.shape and
            pre_merge_yld.shape == sample_speeds.shape and
            pre_merge_go.shape == sample_speeds.shape and
            pre_borrow_yld.shape == sample_speeds.shape and
            pre_borrow_go.shape == sample_speeds.shape
        ):
            sample_speeds = pre_speeds
            chase_risks = pre_chase
            meet_risks = pre_meet
            merge_yld_risks = pre_merge_yld
            merge_go_risks = pre_merge_go
            borrow_yld_risks = pre_borrow_yld
            borrow_go_risks = pre_borrow_go
            ped_risks = pre_ped
    meet_subtype = str(speed_curve.get("meet", {}).get("subtype", speed_curve.get("meet_debug", {}).get("subtype", "none")))
    if meet_subtype == "borrow_cross_meet":
        if (
            borrow_yld_risks.shape == borrow_go_risks.shape and
            borrow_yld_risks.size > 0 and
            (np.any(borrow_yld_risks > 1e-5) or np.any(borrow_go_risks > 1e-5))
        ):
            meet_risks = np.minimum(borrow_yld_risks, borrow_go_risks)
    if "_cross_wait_time_s" in sample_vis:
        speed_curve["cross_wait_time_s"] = float(sample_vis.get("_cross_wait_time_s", 0.0))
        speed_curve["cross_wait_valid"] = float(sample_vis.get("_cross_wait_valid", 0.0))
    total_risks = np.maximum(np.maximum(chase_risks, meet_risks), ped_risks)
    return {
        "current_cover": current_cover,
        "future_cover": future_cover,
        "ped_current_cover": _cover_candidate_summary(0, None, {}),
        "ped_future_cover": _cover_candidate_summary(0, None, {}),
        "speed_curve_future_cover": speed_curve_future_cover,
        "speed_curve_future_persisted": bool(sample_vis.get("_speed_curve_future_persisted", False)),
        "merge_episode": _merge_episode_payload(),
        "merge_motion": _merge_motion_payload(),
        "speed_curve": {
            "sample_speeds_mps": sample_speeds.astype(np.float32).astype(float).tolist(),
            "total_risks": total_risks.astype(np.float32).astype(float).tolist(),
            "chase_risks": chase_risks.astype(np.float32).astype(float).tolist(),
            "meet_risks": meet_risks.astype(np.float32).astype(float).tolist(),
            "merge_yld_risks": merge_yld_risks.astype(np.float32).astype(float).tolist(),
            "merge_go_risks": merge_go_risks.astype(np.float32).astype(float).tolist(),
            "borrow_yld_risks": borrow_yld_risks.astype(np.float32).astype(float).tolist(),
            "borrow_go_risks": borrow_go_risks.astype(np.float32).astype(float).tolist(),
            "ped_risks": ped_risks.astype(np.float32).astype(float).tolist(),
            "chase_debug": dict(speed_curve["chase"]),
            "meet_debug": dict(speed_curve["meet"]),
        },
    }


def _annotate_frame_records_merge_episode(frame_records):
    if not frame_records:
        return

    merge_samples = []
    for record in frame_records:
        sample_vis = record.get("sample_vis", {})
        stage1_label = _stage1_label_payload(
            sample_vis,
            record.get("current_meas"),
            current_boxes=record.get("current_boxes"),
        )
        merge_samples.append({
            "frame_id": int(record.get("frame_id", -1)),
            "stage1_speed_debug": {
                "current_cover": dict(stage1_label.get("current_cover", {})),
                "future_cover": dict(stage1_label.get("future_cover", {})),
                "ped_current_cover": dict(stage1_label.get("ped_current_cover", {})),
                "ped_future_cover": dict(stage1_label.get("ped_future_cover", {})),
                "speed_curve_future_cover": dict(stage1_label.get("speed_curve_future_cover", {})),
                "speed_curve_future_persisted": bool(stage1_label.get("speed_curve_future_persisted", False)),
                "merge_motion": dict(stage1_label.get("merge_motion", {})),
                "speed_curve": dict(stage1_label.get("speed_curve", {})),
                "merge_episode": dict(stage1_label.get("merge_episode", _default_merge_episode_debug())),
            },
        })

    _annotate_route_stage1_merge_decisions(merge_samples, list(range(len(merge_samples))))

    for merge_sample, record in zip(merge_samples, frame_records):
        sample_vis = record.get("sample_vis", {})
        stage1_debug = merge_sample.get("stage1_speed_debug", {})
        sample_vis["_merge_episode"] = dict(stage1_debug.get("merge_episode", _default_merge_episode_debug()))
        for field_name in MERGE_TOP_LEVEL_FIELDS:
            if field_name in merge_sample:
                sample_vis[field_name] = merge_sample[field_name]


def _cover_is_cross_meet_local(cover):
    if int((cover or {}).get("exists", 0.0)) <= 0:
        return False
    interaction = (cover or {}).get("interaction", {}) or {}
    if str(interaction.get("name", "none")) != "meet":
        return False
    subtype = str(interaction.get("subtype") or interaction.get("name") or "none")
    return "cross" in subtype


def _annotate_frame_records_cross_wait(
    frame_records,
    scene_borrow_context=None,
    start_distance_m=2.0,
    wait_speed_thresh=0.5,
    go_speed_thresh=1.0,
    dt_s=0.25,
):
    if not frame_records:
        return
    cross_wait_state = 0
    cross_wait_frames = 0
    cross_episode_active = False
    for record in frame_records:
        sample_vis = record.get("sample_vis", {})
        current_meas = record.get("current_meas")
        debug = record.get("debug", {})
        event_name = record.get("event_name", None)
        current_cover = _cover_candidate_summary(1, debug.get("best_current"), debug, current_meas=current_meas, event_name=event_name)
        future_cover = _cover_candidate_summary(2, debug.get("best_future"), debug, current_meas=current_meas, event_name=event_name)
        raw_cross_active = _cover_is_cross_meet_local(current_cover) or _cover_is_cross_meet_local(future_cover)
        if not cross_episode_active and raw_cross_active:
            cross_episode_active = True

        route_local = np.asarray(
            sample_vis.get("_route_corridor_input_local", sample_vis.get("_route_input_local", np.zeros((0, 2), dtype=np.float32))),
            dtype=np.float32,
        )
        borrow_corridor = _borrow_corridor_metrics(
            scene_borrow_context,
            current_meas=current_meas,
            route_local=route_local,
        )

        def _cross_start_distance_from_cover(cover):
            if not _cover_is_cross_meet_local(cover):
                return np.nan
            interaction = (cover or {}).get("interaction", {}) or {}
            subtype = str(interaction.get("subtype") or interaction.get("name") or "none")
            if subtype == "borrow_cross_meet" and borrow_corridor is not None:
                return float(borrow_corridor.get("borrow_start_distance_m", np.nan))
            d_ego = cover.get("distance", cover.get("d_ego", np.nan))
            try:
                return float(d_ego)
            except Exception:
                return np.nan

        dist_current = _cross_start_distance_from_cover(current_cover)
        dist_future = _cross_start_distance_from_cover(future_cover)
        if np.isfinite(dist_current) and np.isfinite(dist_future):
            cross_start_distance = float(min(dist_current, dist_future))
        elif np.isfinite(dist_current):
            cross_start_distance = float(dist_current)
        elif np.isfinite(dist_future):
            cross_start_distance = float(dist_future)
        else:
            cross_start_distance = np.nan

        speed = float((current_meas or {}).get("speed", 0.0))
        near_cross_start = (
            np.isfinite(cross_start_distance) and
            float(cross_start_distance) <= float(start_distance_m)
        )
        go_now = False
        if cross_episode_active:
            if cross_wait_state == 0:
                if near_cross_start and speed <= float(wait_speed_thresh):
                    cross_wait_state = 1
                    cross_wait_frames = 0
            if cross_wait_state == 1:
                if speed >= float(go_speed_thresh):
                    go_now = True
                    cross_wait_state = 2
                else:
                    cross_wait_frames += 1
        cross_active = bool(cross_episode_active or go_now)
        cross_wait_time_s = float(cross_wait_frames) * float(dt_s)
        cross_wait_valid = 1.0 if (cross_wait_state == 1 or go_now) else 0.0
        sample_vis["_cross_wait_time_s"] = float(cross_wait_time_s)
        sample_vis["_cross_wait_valid"] = float(cross_wait_valid)
        sample_vis["_cross_wait_start_dist"] = float(cross_start_distance) if np.isfinite(cross_start_distance) else np.nan
        sample_vis["_cross_wait_speed_mps"] = float(speed)
        sample_vis["_cross_wait_active"] = 1.0 if cross_active else 0.0
        if go_now:
            cross_episode_active = False
            cross_wait_state = 0
            cross_wait_frames = 0


def _save_frame_bundle(save_dir, frame_id, frame_img, sample_vis, release_info, current_meas, current_boxes=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = f"frame_{int(frame_id):04d}"
    image_path = save_dir / f"{stem}.png"
    json_path = save_dir / f"{stem}.json"

    cv2.imwrite(str(image_path), frame_img)

    route_front = np.asarray(sample_vis.get("_route_front_local", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    route_all = np.asarray(sample_vis.get("_route_input_local", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    route_ext = np.zeros((0, 2), dtype=np.float32)
    if route_all.ndim == 2 and route_front.ndim == 2 and route_all.shape[0] > route_front.shape[0]:
        route_ext = route_all[route_front.shape[0]:]

    payload = {
        "frame_id": int(frame_id),
        "route_mode": str(sample_vis.get("_route_mode", "unknown")),
        "route_front_local_xy": route_front.astype(float).tolist(),
        "route_extension_local_xy": route_ext.astype(float).tolist(),
        "route_all_local_xy": route_all.astype(float).tolist(),
        "release_ready": dict(release_info),
        "stage1_label": _stage1_label_payload(sample_vis, current_meas, current_boxes=current_boxes),
    }

    compare_raw_ext = np.asarray(sample_vis.get("_route_extension_raw", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    compare_flip_ext = np.asarray(sample_vis.get("_route_extension_yflip", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    if compare_raw_ext.ndim == 2 and compare_raw_ext.shape[0] > 0:
        payload["route_extension_raw_local_xy"] = compare_raw_ext.astype(float).tolist()
    if compare_flip_ext.ndim == 2 and compare_flip_ext.shape[0] > 0:
        payload["route_extension_yflip_local_xy"] = compare_flip_ext.astype(float).tolist()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate a route-level front-route-label video")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to split dir containing samples_packed.pkl")
    parser.add_argument("--route_dir", type=str, default=None, help="Optional raw route directory under pdm_lite, for complete-scene rendering")
    parser.add_argument("--image_data_root", type=str, required=True, help="Raw image/data root")
    parser.add_argument("--route_name", type=str, default=None)
    parser.add_argument("--scene_name", type=str, default=None, help="Scene/event name (e.g. AccidentTwoWays)")
    parser.add_argument("--index", type=int, default=None, help="Fallback: pick route_name from this sample index")
    parser.add_argument("--num_future", type=int, default=6)
    parser.add_argument("--front_corridor_margin_m", type=float, default=0.5)
    parser.add_argument("--front_route_step_m", type=float, default=0.25)
    parser.add_argument("--front_max_distance_m", type=float, default=32.0)
    parser.add_argument("--front_safe_ttc_s", type=float, default=3.0)
    parser.add_argument("--front_max_ttc_s", type=float, default=10.0)
    parser.add_argument("--proceed_base_gap_m", type=float, default=4.0)
    parser.add_argument("--proceed_headway_s", type=float, default=1.5)
    parser.add_argument("--proceed_brake_decel_mps2", type=float, default=4.0)
    parser.add_argument("--proceed_speed_gamma", type=float, default=1.0)
    parser.add_argument("--merge_min_speed_ratio", type=float, default=0.35)
    parser.add_argument("--wait_speed_thresh", type=float, default=0.5)
    parser.add_argument("--release_speed_thresh", type=float, default=1.0)
    parser.add_argument("--release_throttle_thresh", type=float, default=0.3)
    parser.add_argument(
        "--route_source_mode",
        type=str,
        default="local",
        choices=["local", "stitched", "scene_polyline", "scene_polyline_compare"],
    )
    parser.add_argument("--scene_route_dedupe_step_m", type=float, default=0.5)
    parser.add_argument("--scene_route_extension_step_m", type=float, default=1.0)
    parser.add_argument("--scene_route_extension_points", type=int, default=12)
    parser.add_argument(
        "--borrow_start_mode",
        type=str,
        default="route_head",
        choices=["route_head", "obstacle_align"],
        help="How to anchor two-way corridor start along the route.",
    )
    parser.add_argument("--xlim", type=float, nargs=2, default=[-10.0, 35.0])
    parser.add_argument("--ylim", type=float, nargs=2, default=[-12.0, 12.0])
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_min", type=int, default=None, help="Only render frames with frame_id >= this value")
    parser.add_argument("--frame_max", type=int, default=None, help="Only render frames with frame_id <= this value")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save_frames_dir", type=str, default=None, help="Optional directory to save per-frame PNGs and route JSON")
    parser.add_argument("--save_only_after_release", action="store_true", help="If set, only save frames from the first release pulse onward")
    args = parser.parse_args()

    if bool(args.dataset_path) == bool(args.route_dir):
        raise RuntimeError("Provide exactly one of --dataset_path or --route_dir")

    if args.route_dir is not None:
        selection_name, route_samples = _load_route_samples_from_route_dir(
            route_dir=args.route_dir,
            image_root=args.image_data_root,
        )
    else:
        packed_path = os.path.join(args.dataset_path, "samples_packed.pkl")
        with open(packed_path, "rb") as f:
            samples = pickle.load(f)

        if args.scene_name is not None and args.route_name is not None:
            route_name, route_samples = _select_route_samples(samples, route_name=args.route_name, index=None)
            route_samples = [
                s for s in route_samples
                if _scene_name_from_base_dir(_resolve_feature_frame_info(s)[0]) == args.scene_name
            ]
            if not route_samples:
                raise RuntimeError(f"No samples found for scene_name={args.scene_name} and route_name={args.route_name}")
            selection_name = f"{args.scene_name}_{args.route_name}"
        elif args.scene_name is not None:
            selection_name, route_samples = _select_scene_samples(samples, scene_name=args.scene_name)
        else:
            selection_name, route_samples = _select_route_samples(samples, route_name=args.route_name, index=args.index)
    route_samples_context = list(route_samples)
    if args.frame_min is not None:
        route_samples = [s for s in route_samples if int(s.get("frame_id", -1)) >= args.frame_min]
    if args.frame_max is not None:
        route_samples = [s for s in route_samples if int(s.get("frame_id", -1)) <= args.frame_max]
    if args.max_frames is not None:
        route_samples = route_samples[: args.max_frames]
    if not route_samples:
        raise RuntimeError("No samples left after applying route/frame filters")

    scene_route_polyline_world = {}
    scene_route_polyline_anchor_s = {}
    scene_route_polyline_world_yflip = {}
    scene_route_polyline_anchor_s_yflip = {}
    scene_nonstatic_actor_ids = _collect_scene_nonstatic_actor_ids(
        route_samples_context,
        image_root=args.image_data_root,
        speed_thresh_mps=0.25,
        motion_thresh_m=1.0,
    )
    need_scene_polyline = bool(args.route_source_mode in {"stitched", "scene_polyline", "scene_polyline_compare"})
    if not need_scene_polyline:
        for sample_ctx in route_samples_context:
            base_dir_ctx, _ = _resolve_feature_frame_info(sample_ctx)
            if base_dir_ctx is None:
                continue
            event_ctx = base_dir_ctx.split(os.sep)[0] if os.sep in base_dir_ctx else base_dir_ctx.split("/")[0]
            if _is_two_way_event_corridor_scene_context(event_name=event_ctx):
                need_scene_polyline = True
                break
    if need_scene_polyline:
        scene_route_polyline_world, scene_route_polyline_anchor_s = _build_scene_route_polyline_world(
            route_samples_context,
            image_root=args.image_data_root,
            dedupe_step_m=args.scene_route_dedupe_step_m,
        )
    if args.route_source_mode == "scene_polyline_compare":
        scene_route_polyline_world_yflip, scene_route_polyline_anchor_s_yflip = _build_scene_route_polyline_world(
            route_samples_context,
            image_root=args.image_data_root,
            dedupe_step_m=args.scene_route_dedupe_step_m,
            flip_local_y=True,
        )

    if route_samples_context:
        base_dir_check, _ = _resolve_feature_frame_info(route_samples_context[0])
        event_check = _scene_name_from_base_dir(base_dir_check) if base_dir_check else None
        if _is_two_way_event_corridor_scene_context(event_name=event_check):
            missing = []
            for sample_ctx in route_samples_context:
                base_dir_ctx, _ = _resolve_feature_frame_info(sample_ctx)
                if base_dir_ctx is None:
                    continue
                if base_dir_ctx not in scene_route_polyline_world:
                    missing.append(base_dir_ctx)
            if missing:
                raise RuntimeError(f"Missing scene polyline for two-way routes: {sorted(set(missing))}")

    safe_route = selection_name.replace("/", "_")
    frame_suffix = ""
    if args.frame_min is not None or args.frame_max is not None:
        lo = "start" if args.frame_min is None else f"{int(args.frame_min):04d}"
        hi = "end" if args.frame_max is None else f"{int(args.frame_max):04d}"
        frame_suffix = f"_{lo}-{hi}"
    project_root = Path(__file__).resolve().parents[1]
    if args.output is None:
        output_dir = project_root / "visualizations" / "front_route_videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"front_route_video_{safe_route}{frame_suffix}.mp4"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    frames_written = 0
    event_name = "UnknownEvent"
    prev_wait_state = False
    prev_speed_mps = None
    release_started = False
    frame_records = []

    for sample in route_samples:
        base_dir, frame_str = _resolve_feature_frame_info(sample)
        if base_dir is None or frame_str is None:
            continue
        event_name = base_dir.split(os.sep)[0] if os.sep in base_dir else base_dir.split("/")[0]

        current_boxes = _load_json_gz_if_exists(os.path.join(args.image_data_root, base_dir, "boxes", f"{frame_str}.json.gz"))
        current_meas = _load_json_gz_if_exists(os.path.join(args.image_data_root, base_dir, "measurements", f"{frame_str}.json.gz"))
        if current_boxes is None or current_meas is None:
            continue
        num_future = args.num_future
        if "ego_waypoints" in sample:
            try:
                num_future = max(1, min(num_future, len(sample["ego_waypoints"]) - 1))
            except Exception:
                pass
        future_frames = _load_future_frames(args.image_data_root, base_dir, int(sample["frame_id"]), num_future)

        base_route_local = np.asarray(sample["route"], dtype=np.float32)
        route_input = base_route_local
        route_corridor_input = route_input
        route_mode_name = args.route_source_mode
        compare_raw_ext = np.zeros((0, 2), dtype=np.float32)
        compare_flip_ext = np.zeros((0, 2), dtype=np.float32)
        if event_name in NO_ROUTE_EXTENSION_SCENES:
            route_input = base_route_local
            route_mode_name = "local_noext"
        elif args.route_source_mode in {"stitched", "scene_polyline", "scene_polyline_compare"}:
            ego_matrix_current = current_meas.get("ego_matrix", None)
            if ego_matrix_current is not None:
                scene_polyline_world = scene_route_polyline_world.get(base_dir, None)
                scene_anchor_s = scene_route_polyline_anchor_s.get(base_dir, {}).get(int(sample["frame_id"]))
                extended_local = _extend_local_route_with_scene_polyline(
                    route_local=base_route_local,
                    ego_matrix_current=ego_matrix_current,
                    scene_polyline_world=scene_polyline_world,
                    anchor_s=scene_anchor_s,
                    extension_step_m=args.scene_route_extension_step_m,
                    extension_points=args.scene_route_extension_points,
                )
                if extended_local.shape[0] >= 2 and args.route_source_mode != "scene_polyline_compare":
                    route_input = extended_local
                    route_mode_name = "scene_polyline"
                if args.route_source_mode == "scene_polyline_compare":
                    if extended_local.ndim == 2 and extended_local.shape[0] > base_route_local.shape[0]:
                        compare_raw_ext = extended_local[base_route_local.shape[0]:].astype(np.float32)
                    scene_polyline_world_flip = scene_route_polyline_world_yflip.get(base_dir, None)
                    scene_anchor_s_flip = scene_route_polyline_anchor_s_yflip.get(base_dir, {}).get(int(sample["frame_id"]))
                    extended_flip = _extend_local_route_with_scene_polyline(
                        route_local=base_route_local,
                        ego_matrix_current=ego_matrix_current,
                        scene_polyline_world=scene_polyline_world_flip,
                        anchor_s=scene_anchor_s_flip,
                        extension_step_m=args.scene_route_extension_step_m,
                        extension_points=args.scene_route_extension_points,
                        flip_local_y=True,
                    )
                    if extended_flip.ndim == 2 and extended_flip.shape[0] > base_route_local.shape[0]:
                        compare_flip_ext = extended_flip[base_route_local.shape[0]:].astype(np.float32)
                    route_mode_name = "scene_polyline_compare"
        if _is_two_way_event_corridor_scene_context(event_name=event_name):
            ego_matrix_current = current_meas.get("ego_matrix", None)
            if ego_matrix_current is not None:
                scene_polyline_world = scene_route_polyline_world.get(base_dir, None)
                scene_anchor_s = scene_route_polyline_anchor_s.get(base_dir, {}).get(int(sample["frame_id"]))
                extended_for_corridor = _extend_local_route_with_scene_polyline(
                    route_local=base_route_local,
                    ego_matrix_current=ego_matrix_current,
                    scene_polyline_world=scene_polyline_world,
                    anchor_s=scene_anchor_s,
                    extension_step_m=args.scene_route_extension_step_m,
                    extension_points=args.scene_route_extension_points,
                )
                if extended_for_corridor.ndim == 2 and extended_for_corridor.shape[0] >= 2:
                    route_corridor_input = extended_for_corridor.astype(np.float32)
                else:
                    route_corridor_input = route_input
            if route_corridor_input.ndim == 2 and route_corridor_input.shape[0] >= 2:
                route_input = route_corridor_input
                if route_mode_name == "local":
                    route_mode_name = "scene_polyline_for_twoway"
        dynamic_ids = _collect_dynamic_actor_ids(
            current_boxes=current_boxes,
            future_frames_data=future_frames,
            ego_matrix_current=current_meas.get("ego_matrix", None),
            speed_thresh_mps=0.25,
            motion_thresh_m=1.0,
        )
        dynamic_ids = set(dynamic_ids) | set(scene_nonstatic_actor_ids)
        label_current_boxes = _filter_current_boxes_dynamic(current_boxes, dynamic_ids)
        label_future_frames = _filter_future_frames_dynamic(future_frames, dynamic_ids, speed_thresh_mps=0.25)
        label, debug = _compute_front_route_label(
            route=route_input,
            current_boxes=label_current_boxes,
            ego_speed=float(current_meas.get("speed", 0.0)),
            ego_matrix_current=current_meas.get("ego_matrix"),
            future_frames_data=label_future_frames,
            corridor_margin_m=args.front_corridor_margin_m,
            route_step_m=args.front_route_step_m,
            max_distance_m=args.front_max_distance_m,
            safe_ttc_s=args.front_safe_ttc_s,
            max_ttc_s=args.front_max_ttc_s,
            return_debug=True,
        )
        ped_current_boxes = _filter_current_boxes_pedestrian(current_boxes)
        ped_future_frames = _filter_future_frames_pedestrian(future_frames)
        if ped_current_boxes or any(frame is not None for frame in ped_future_frames):
            ped_label, ped_debug = _compute_front_route_label(
                route=route_input,
                current_boxes=ped_current_boxes,
                ego_speed=float(current_meas.get("speed", 0.0)),
                ego_matrix_current=current_meas.get("ego_matrix"),
                future_frames_data=ped_future_frames,
                corridor_margin_m=_pedestrian_corridor_margin_m(
                    current_boxes=current_boxes,
                    base_margin_m=float(args.front_corridor_margin_m),
                ),
                route_step_m=args.front_route_step_m,
                max_distance_m=args.front_max_distance_m,
                safe_ttc_s=args.front_safe_ttc_s,
                max_ttc_s=args.front_max_ttc_s,
                return_debug=True,
            )
            if int(ped_label.get("case", 0)) > 0:
                label, debug = ped_label, ped_debug
        interaction = _interaction_signal_from_debug(label, debug, current_meas=current_meas)
        proceed = _proceed_signal_from_label(
            sample=sample,
            label=label,
            interaction=interaction,
            current_meas=current_meas,
            base_gap_m=args.proceed_base_gap_m,
            headway_s=args.proceed_headway_s,
            brake_decel_mps2=args.proceed_brake_decel_mps2,
            safe_ttc_s=args.front_safe_ttc_s,
            speed_gamma=args.proceed_speed_gamma,
            merge_min_speed_ratio=args.merge_min_speed_ratio,
        )
        wait_info = _update_wait_release_state(
            prev_wait_state=prev_wait_state,
            interaction=interaction,
            current_meas=current_meas,
            wait_speed_thresh=args.wait_speed_thresh,
            release_speed_thresh=args.release_speed_thresh,
            release_throttle_thresh=args.release_throttle_thresh,
            prev_speed_mps=prev_speed_mps,
        )
        prev_wait_state = bool(wait_info["wait_state"] > 0.5)
        prev_speed_mps = float((current_meas or {}).get("speed", 0.0))
        sample_vis = dict(sample)
        sample_vis["_event_name"] = event_name
        sample_vis["_release_info"] = {}
        sample_vis["_route_mode"] = route_mode_name
        sample_vis["_route_len_m"] = _polyline_length_m(route_input)
        sample_vis["_route_corridor_len_m"] = _polyline_length_m(route_corridor_input)
        sample_vis["_route_front_local"] = base_route_local
        sample_vis["_route_input_local"] = route_input
        sample_vis["_route_corridor_input_local"] = route_corridor_input
        sample_vis["_route_extension_raw"] = compare_raw_ext
        sample_vis["_route_extension_yflip"] = compare_flip_ext
        sample_vis["_debug"] = debug
        route_arr = np.asarray(route_input, dtype=np.float32)
        sample_vis["_route_num_points"] = int(route_arr.shape[0]) if route_arr.ndim == 2 else 0
        corridor_arr = np.asarray(route_corridor_input, dtype=np.float32)
        sample_vis["_route_corridor_num_points"] = int(corridor_arr.shape[0]) if corridor_arr.ndim == 2 else 0

        frame_records.append({
            "sample": sample,
            "sample_vis": sample_vis,
            "base_dir": base_dir,
            "frame_str": frame_str,
            "frame_id": int(sample.get("frame_id", -1)),
            "current_boxes": current_boxes,
            "current_meas": current_meas,
            "label": label,
            "debug": debug,
            "interaction": interaction,
            "proceed": proceed,
            "wait_info": wait_info,
            "event_name": event_name,
        })

    _annotate_frame_records_merge_episode(frame_records)

    _annotate_release_ready(
        frame_records,
        fps_hz=4.0,
        route_step_m=max(0.25, float(args.front_route_step_m)),
        corridor_margin_m=float(args.front_corridor_margin_m),
        wait_speed_thresh=float(args.wait_speed_thresh),
        release_speed_thresh=float(args.release_speed_thresh),
    )

    scene_borrow_context = None
    if _is_borrow_cross_scene_context(event_name=event_name):
        scene_borrow_context = _build_event_two_way_borrow_context(
            frame_records,
            event_name=event_name,
            route_step_m=max(0.25, float(args.front_route_step_m)),
            borrow_start_mode=str(args.borrow_start_mode),
            borrow_enter_lateral_thresh=1.25,
            return_lateral_thresh=0.8,
            min_enter_progress_m=4.0,
            min_return_progress_m=6.0,
        )
        if _is_two_way_event_corridor_scene_context(event_name=event_name) and scene_borrow_context is None:
            raise RuntimeError(
                f"Failed to build scene-level two-way corridor for {selection_name}; "
                "old wait/release and cover-centered fallbacks are disabled."
            )

    _annotate_frame_records_cross_wait(
        frame_records,
        scene_borrow_context=scene_borrow_context,
        start_distance_m=10.0,
        wait_speed_thresh=0.5,
        go_speed_thresh=1.0,
        dt_s=0.25,
    )

    persisted_meet = None
    persisted_meet_frames_left = 0
    persist_dt_s = 0.25
    for record in frame_records:
        sample = record["sample"]
        sample_vis = record["sample_vis"]
        current_boxes = record["current_boxes"]
        current_meas = record["current_meas"]
        label = record["label"]
        debug = record["debug"]
        interaction = record["interaction"]
        proceed = record["proceed"]
        wait_info = record["wait_info"]
        release_info = record.get("release_ready", {
            "valid": 0.0,
            "ready": 0.0,
            "source": "n/a",
            "release_frame_id": -1,
            "enter_frame_id": -1,
            "return_frame_id": -1,
            "borrow_duration_s": 0.0,
            "release_to_return_s": 0.0,
            "borrow_start_distance_m": 0.0,
            "borrow_distance_m": 0.0,
            "peak_lateral_m": 0.0,
            "blocked_frame_id": -1,
            "blocking_actor_id": -1,
            "blocking_actor_class": "none",
            "borrow_start_world_xy": [],
            "borrow_end_world_xy": [],
            "borrow_start_world_xyz": [],
            "borrow_end_world_xyz": [],
            "borrow_segment_world_xy": [],
            "borrow_segment_world_xyz": [],
        })
        rec_event_name = record.get("event_name", event_name)
        if scene_borrow_context is not None and _is_two_way_event_corridor_scene_context(event_name=rec_event_name):
            release_info = dict(scene_borrow_context)
        sample_vis["_release_info"] = dict(release_info)

        current_cover = _cover_candidate_summary(1, debug.get("best_current"), debug, current_meas=current_meas, event_name=event_name)
        future_cover = _cover_candidate_summary(2, debug.get("best_future"), debug, current_meas=current_meas, event_name=event_name)
        sample_vis["_speed_curve_future_persisted"] = False
        sample_vis["_speed_curve_future_cover"] = future_cover

        if int(future_cover.get("exists", 0.0)) > 0 and future_cover["interaction"]["name"] == "meet":
            persisted_meet = dict(future_cover)
            persisted_meet_actor_id = int(future_cover.get("actor_id", -1))
            persisted_meet["actor_id"] = persisted_meet_actor_id
            persisted_meet_frames_left = 4
        elif persisted_meet is not None and persisted_meet_frames_left > 0:
            actor_id = int(persisted_meet.get("actor_id", -1))
            actor_box = _find_box_by_id(current_boxes, actor_id)
            ego_box = _find_ego_box(current_boxes)
            ego_speed_now = float((current_meas or {}).get("speed", 0.0))
            bg_speed_now = float(persisted_meet.get("other_speed", np.nan))
            if actor_box is not None:
                bg_speed_now = float(abs(actor_box.get("speed", bg_speed_now)))
            pseudo = dict(persisted_meet)
            keep_persisted = False
            if actor_box is not None:
                pos = np.asarray(actor_box.get("position", [np.nan, np.nan])[:2], dtype=np.float32)
                if pos.shape == (2,) and np.all(np.isfinite(pos)):
                    # After merge, the original future-meet actor may become a
                    # close rear actor. Keep a rear-floor constraint alive while
                    # it stays near and behind ego.
                    if float(pos[0]) < 1.0 and float(np.linalg.norm(pos)) < 12.0:
                        pseudo["d_ego"] = 0.0
                        pseudo["d_bg"] = float(max(np.linalg.norm(pos), 1e-3))
                        pseudo["rear_gap_m"] = float(_approx_box_clearance_gap_m(actor_box, ego_box))
                        keep_persisted = True
            if not keep_persisted:
                if np.isfinite(float(pseudo.get("d_ego", np.nan))):
                    pseudo["d_ego"] = float(max(float(pseudo["d_ego"]) - ego_speed_now * persist_dt_s, 0.0))
                if np.isfinite(float(pseudo.get("d_bg", np.nan))):
                    pseudo["d_bg"] = float(max(float(pseudo["d_bg"]) - bg_speed_now * persist_dt_s, 0.0))
                keep_persisted = bool(
                    np.isfinite(float(pseudo.get("d_bg", np.nan))) and float(pseudo.get("d_bg", 0.0)) > 0.25
                )
            if keep_persisted:
                pseudo["other_speed"] = float(bg_speed_now)
                sample_vis["_speed_curve_future_cover"] = pseudo
                sample_vis["_speed_curve_future_persisted"] = True
                persisted_meet = dict(pseudo)
                persisted_meet_frames_left -= 1
            else:
                persisted_meet = None
                persisted_meet_frames_left = 0

        rgb = _load_scene_panel(
            args.image_data_root,
            record["base_dir"],
            record["frame_str"],
            current_boxes,
            current_meas=current_meas,
        )
        panel = _render_world_panel(
            sample=sample_vis,
            current_boxes=current_boxes,
            current_meas=current_meas,
            label=label,
            debug=debug,
            event_name=record["event_name"],
            width=720,
            height=540,
            xlim=args.xlim,
            ylim=args.ylim,
            interaction=interaction,
            proceed=proceed,
            wait_info=wait_info,
            release_info=release_info,
        )
        frame = _compose_frame(
            rgb,
            panel,
            sample_vis,
            label,
            current_meas,
            current_boxes,
            interaction,
            proceed,
            wait_info,
            release_info,
        )

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {output_path}")

        writer.write(frame)
        frames_written += 1

        if bool(wait_info["release_pulse"] > 0.5):
            release_started = True
        if args.save_frames_dir is not None:
            should_save = True
            if args.save_only_after_release and not release_started:
                should_save = False
            if should_save:
                _save_frame_bundle(
                    save_dir=args.save_frames_dir,
                    frame_id=int(sample.get("frame_id", -1)),
                    frame_img=frame,
                    sample_vis=sample_vis,
                    release_info=release_info,
                    current_meas=current_meas,
                    current_boxes=current_boxes,
                )

    if writer is not None:
        writer.release()

    if frames_written == 0:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"No frames were written for selection {selection_name}")

    print(f"Saved video to {output_path}")
    print(f"selection={selection_name}")
    print(f"event_name={event_name}")
    print(f"frames={frames_written}")


if __name__ == "__main__":
    main()
