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
    _compute_front_route_label,
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
INTERACTION_SAME_DIR_ANGLE_THRESH_DEG = 45.0
MERGE_DEBUG_MIN_DEGO_M = 1.0
MERGE_DEBUG_MIN_GO_DENOM_S = 0.10
NO_ROUTE_EXTENSION_SCENES = {"HazardAtSideLane"}
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
    return event_name in {"AccidentTwoWays", "ParkedObstacleTwoWays"}


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

    cover = best.get("cover", {})
    actor_box = best.get("box") if case == 1 else (best.get("current_box") or best.get("box_future") or best.get("box_current_frame"))
    if _is_pedestrian_box(actor_box):
        return {
            "mode": 2,
            "name": INTERACTION_MODE_NAMES[2],
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
            "name": INTERACTION_MODE_NAMES[0],
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
                "name": INTERACTION_MODE_NAMES[2],
                "subtype": "merge_meet",
                "source": "junction_right_future_cover_override",
                "angle_deg": np.nan,
                "route_heading_deg": _heading_to_deg(route_heading),
                "actor_heading_deg": np.nan,
                "motion_m": motion_m,
            }
        return {
            "mode": 0,
            "name": INTERACTION_MODE_NAMES[0],
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
            "name": INTERACTION_MODE_NAMES[2],
            "subtype": "merge_meet",
            "source": "junction_right_future_cover_override",
            "angle_deg": float(angle_deg),
            "route_heading_deg": _heading_to_deg(route_heading),
            "actor_heading_deg": _heading_to_deg(actor_heading),
            "motion_m": float(motion_m),
        }
    if same_direction and case == 1:
        mode = 1
        subtype = "follow_chase"
        source = f"{source}+same_dir_current_cover"
    else:
        mode = 2
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
        elif not same_direction:
            subtype = "cross_meet"
            source = f"{source}+cross_dir"
    return {
        "mode": int(mode),
        "name": INTERACTION_MODE_NAMES[int(mode)],
        "subtype": subtype,
        "source": source,
        "angle_deg": float(angle_deg),
        "route_heading_deg": _heading_to_deg(route_heading),
        "actor_heading_deg": _heading_to_deg(actor_heading),
        "motion_m": float(motion_m),
    }


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
            "cover_point_local_xy": np.asarray(best.get("cover", {}).get("route_point", []), dtype=np.float32).astype(float).tolist(),
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


def _build_speed_curve_debug(
    current_cover,
    future_cover,
    current_meas,
    current_boxes=None,
    event_name=None,
    release_info=None,
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
    }
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
        return {
            "sample_speeds_mps": sample_speeds.astype(np.float32),
            "chase_risks": chase_risks.astype(np.float32),
            "meet_risks": meet_risks.astype(np.float32),
            "total_risks": total_risks.astype(np.float32),
            "chase": chase_info,
            "meet": meet_info,
        }
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
            return {
                "sample_speeds_mps": sample_speeds.astype(np.float32),
                "chase_risks": chase_risks.astype(np.float32),
                "meet_risks": meet_risks.astype(np.float32),
                "total_risks": total_risks.astype(np.float32),
                "chase": chase_info,
                "meet": meet_info,
            }
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
            t_bg_clear = (risk_d_bg + float(merge_clearance_m)) / max(bg_speed, 1e-6)
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
                    v_equal = d_ego / max(t_bg, 1e-6)
                    go_denom = t_bg - float(merge_tau_s)
                    v_go_min = np.inf if go_denom <= 1e-6 else d_ego / max(go_denom, 1e-6)
                    v_go_need = max(float(v_go_min), float(v_behind_min))
                    v_yield_max = d_ego / max(t_bg + float(merge_tau_s), 1e-6)
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
                "bg_speed_mps": bg_speed,
                "t_bg_s": float(t_bg),
                "t_bg_exit_s": float(t_bg_exit),
                "t_bg_clear_s": float(t_bg_clear),
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
                    continue
                if meet_subtype == "merge_meet":
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
                    elif not np.isfinite(v_equal):
                        risk = float(np.clip((v_go_need - v) / max(v_go_need - v_yield_max, 1e-6), 0.0, 1.0))
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
                            continue
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
                else:
                    conflict_len = max(float(conflict_len_m) if np.isfinite(conflict_len_m) else 0.0, 0.0)
                    t_ego_in = risk_d_ego / max(v, 1e-6)
                    if conflict_len > 1e-6:
                        t_ego_out = (risk_d_ego + conflict_len) / max(v, 1e-6)
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
            "borrow_distance_m": 0.0,
            "peak_lateral_m": 0.0,
            "blocked_frame_id": -1,
            "blocking_actor_id": -1,
            "blocking_actor_class": "none",
            "borrow_start_world_xy": [],
            "borrow_end_world_xy": [],
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

        route_local = np.asarray(record["sample_vis"].get("_route_input_local", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
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

        borrow_world = _transform_points_local_to_world_xyz(
            np.stack([borrow_geom["enter_local_xy"], borrow_geom["return_local_xy"]], axis=0),
            ego_matrix_current,
        )
        if borrow_world.shape[0] != 2:
            info["source"] = "borrow_world_invalid"
            record["release_ready"] = info
            continue

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
            "borrow_distance_m": float(borrow_geom["borrow_distance_m"]),
            "peak_lateral_m": float(borrow_geom["peak_lateral_m"]),
            "blocked_frame_id": int(blocked_frame_id),
            "blocking_actor_id": int(blocking_actor_id),
            "blocking_actor_class": blocking_actor_class,
            "borrow_start_world_xy": borrow_world[0, :2].astype(float).tolist(),
            "borrow_end_world_xy": borrow_world[1, :2].astype(float).tolist(),
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


def _draw_box(img, box, color, width, height, xlim, ylim, thickness=2):
    pos = box.get("position", None)
    extent = box.get("extent", None)
    if pos is None or extent is None or len(pos) < 2 or len(extent) < 2:
        return
    poly = _oriented_box_corners(pos[:2], extent[:2], float(box.get("yaw", 0.0)))
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
    if (
        _is_left_turn_scene_context(current_meas, event_name=event_name)
        and ego_matrix_current is not None
        and borrow_start_world_xy.shape == (2,)
        and borrow_end_world_xy.shape == (2,)
    ):
        corridor_local = _transform_points_world_to_local(
            np.stack([borrow_start_world_xy, borrow_end_world_xy], axis=0),
            ego_matrix_current,
        )
        if corridor_local.shape == (2, 2):
            _draw_polyline(panel, corridor_local, (60, 220, 60), width, height, xlim, ylim, thickness=2)
            start_px = _to_canvas(corridor_local[0], width, height, xlim, ylim)[0]
            end_px = _to_canvas(corridor_local[1], width, height, xlim, ylim)[0]
            cv2.circle(panel, tuple(start_px), 6, (60, 220, 60), -1, lineType=cv2.LINE_AA)
            cv2.circle(panel, tuple(end_px), 6, (255, 200, 0), -1, lineType=cv2.LINE_AA)
            cv2.putText(panel, "b_in", (int(start_px[0]) + 6, int(start_px[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 140, 20), 1, cv2.LINE_AA)
            cv2.putText(panel, "b_out", (int(end_px[0]) + 6, int(end_px[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 110, 0), 1, cv2.LINE_AA)

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
            _draw_box(panel, fut["box_current_frame"], fut_color, width, height, xlim, ylim, thickness=3)
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
    lines = [
        f"{event_name} | frame {int(sample.get('frame_id', -1)):04d}",
        route_name,
        f"cover={occ['case_name']}  interact={interaction['name']}  occ={occ['risk']:.3f}  proceed={proceed['risk']:.3f}  v={wait_info['speed_mps']:.2f}",
        f"wait={int(wait_info['wait_state'])}  release={int(wait_info['release_pulse'])}  rel_ready={release_ready_str}",
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
    debug = sample.get("_debug", {})
    event_name = sample.get("_event_name")
    current_cover = _cover_candidate_summary(1, debug.get("best_current"), debug, current_meas=current_meas, event_name=event_name)
    future_cover = _cover_candidate_summary(2, debug.get("best_future"), debug, current_meas=current_meas, event_name=event_name)
    speed_curve_future_cover = sample.get("_speed_curve_future_cover", future_cover)
    speed_curve = _build_speed_curve_debug(
        current_cover,
        speed_curve_future_cover,
        current_meas,
        current_boxes=current_boxes,
        event_name=event_name,
        release_info=release_info,
    )
    rgb_h, rgb_w = rgb.shape[:2]
    panel_h, panel_w = panel.shape[:2]
    target_h = max(rgb_h, panel_h)
    rgb_resized = cv2.resize(rgb, (int(rgb_w * target_h / max(rgb_h, 1)), target_h), interpolation=cv2.INTER_LINEAR)
    panel_resized = cv2.resize(panel, (int(panel_w * target_h / max(panel_h, 1)), target_h), interpolation=cv2.INTER_LINEAR)
    top = np.concatenate([rgb_resized, panel_resized], axis=1)

    bar_h = 264
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

    left_lines = [
        ("speed={:.2f}  target={:.2f}  wait={}  release={}  rel_ready={}".format(
            speed, target_speed, int(wait_info["wait_state"]), int(wait_info["release_pulse"]), release_ready_str
        ), (255, 255, 255), 0.62),
        ("primary case={}  interact={}  occ={:.3f}  proceed={:.3f}".format(
            int(label["case"]), interaction["name"], float(occ["risk"]), float(proceed["risk"])
        ), (200, 200, 200), 0.58),
        ("aff_id={}  spd_red_id={}  route={} ({:.1f}m, {}pts)".format(
            -1 if veh_aff_id is None else int(veh_aff_id),
            -1 if spd_red_id is None else int(spd_red_id),
            sample.get("_route_mode", "local"),
            float(sample.get("_route_len_m", 0.0)),
            int(sample.get("_route_num_points", 0)),
        ), (190, 190, 190), 0.56),
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
        ("E_chase={}  E_meet={}".format(
            _fmt_sample_pairs(speed_curve["sample_speeds_mps"], speed_curve["chase_risks"]),
            _fmt_sample_pairs(speed_curve["sample_speeds_mps"], speed_curve["meet_risks"]),
        ), (150, 220, 255), 0.48),
    ]

    if float(speed_curve["meet"]["valid"]) > 0.5:
        meet_subtype = str(speed_curve["meet"].get("subtype", "meet"))
        if "cross" in meet_subtype:
            left_lines.append(
                ("{} dbg: dE={} dB={} vB={} tBin={} tBout={} cLen={}".format(
                    meet_subtype,
                    _fmt_val(speed_curve["meet"]["d_ego_m"]),
                    _fmt_val(speed_curve["meet"]["d_bg_m"]),
                    _fmt_val(speed_curve["meet"]["bg_speed_mps"]),
                    _fmt_val(speed_curve["meet"]["t_bg_s"]),
                    _fmt_val(speed_curve["meet"].get("t_bg_exit_s", np.nan)),
                    _fmt_val(speed_curve["meet"].get("conflict_len_m", np.nan)),
                ), (120, 200, 255), 0.50)
            )
        else:
            left_lines.append(
                ("{} dbg: dE={} dB={} vB={} tB={} gapB={} cLen={}".format(
                    meet_subtype,
                    _fmt_val(speed_curve["meet"]["d_ego_m"]),
                    _fmt_val(speed_curve["meet"]["d_bg_m"]),
                    _fmt_val(speed_curve["meet"]["bg_speed_mps"]),
                    _fmt_val(speed_curve["meet"]["t_bg_s"]),
                    _fmt_val(speed_curve["meet"]["safe_gap_bg_m"]),
                    _fmt_val(speed_curve["meet"].get("conflict_len_m", np.nan)),
                ), (120, 200, 255), 0.50)
            )
            left_lines.append(
                ("v_eq={}  v_go={}  v_bmin={}  v_need={}  v_yld={}".format(
                    _fmt_val(speed_curve["meet"]["v_equal_mps"]),
                    _fmt_val(speed_curve["meet"]["v_go_min_mps"]),
                    _fmt_val(speed_curve["meet"]["v_behind_min_mps"]),
                    _fmt_val(speed_curve["meet"]["v_go_need_mps"]),
                    _fmt_val(speed_curve["meet"]["v_yield_max_mps"]),
                ), (120, 200, 255), 0.50)
            )
        if bool(sample.get("_speed_curve_future_persisted", False)):
            left_lines.append(
                ("{} source=persisted_after_merge".format(meet_subtype), (120, 200, 255), 0.48)
            )
    if float(speed_curve["meet"]["valid"]) <= 0.5 and np.isfinite(float(speed_curve["meet"].get("context_conflict_len_m", np.nan))):
        left_lines.append(
            ("left-junction corridor: cLen={}".format(
                _fmt_val(speed_curve["meet"].get("context_conflict_len_m", np.nan)),
            ), (120, 200, 255), 0.50)
        )
    if float(speed_curve["chase"]["valid"]) > 0.5:
        left_lines.append(
            ("chase dbg: gap={} gapS={} lead_v={}".format(
                _fmt_val(speed_curve["chase"]["gap_m"]),
                _fmt_val(speed_curve["chase"]["safe_gap_cur_m"]),
                _fmt_val(speed_curve["chase"]["lead_speed_mps"]),
            ), (120, 200, 255), 0.50)
        )

    cur_type = current_cover["interaction"].get("subtype") or current_cover["interaction"]["name"]
    fut_type = future_cover["interaction"].get("subtype") or future_cover["interaction"]["name"]
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
    debug = sample_vis.get("_debug", {})
    event_name = sample_vis.get("_event_name")
    current_cover = _cover_candidate_summary(1, debug.get("best_current"), debug, current_meas=current_meas, event_name=event_name)
    future_cover = _cover_candidate_summary(2, debug.get("best_future"), debug, current_meas=current_meas, event_name=event_name)
    speed_curve_future_cover = sample_vis.get("_speed_curve_future_cover", future_cover)
    speed_curve = _build_speed_curve_debug(
        current_cover,
        speed_curve_future_cover,
        current_meas,
        current_boxes=current_boxes,
        event_name=event_name,
        release_info=sample_vis.get("_release_info"),
    )
    return {
        "current_cover": current_cover,
        "future_cover": future_cover,
        "speed_curve_future_cover": speed_curve_future_cover,
        "speed_curve_future_persisted": bool(sample_vis.get("_speed_curve_future_persisted", False)),
        "speed_curve": {
            "sample_speeds_mps": np.asarray(speed_curve["sample_speeds_mps"], dtype=np.float32).astype(float).tolist(),
            "total_risks": np.asarray(speed_curve["total_risks"], dtype=np.float32).astype(float).tolist(),
            "chase_risks": np.asarray(speed_curve["chase_risks"], dtype=np.float32).astype(float).tolist(),
            "meet_risks": np.asarray(speed_curve["meet_risks"], dtype=np.float32).astype(float).tolist(),
            "chase_debug": dict(speed_curve["chase"]),
            "meet_debug": dict(speed_curve["meet"]),
        },
    }


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
    if args.route_source_mode in {"stitched", "scene_polyline", "scene_polyline_compare"}:
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
        sample_vis["_route_front_local"] = base_route_local
        sample_vis["_route_input_local"] = route_input
        sample_vis["_route_extension_raw"] = compare_raw_ext
        sample_vis["_route_extension_yflip"] = compare_flip_ext
        sample_vis["_debug"] = debug
        route_arr = np.asarray(route_input, dtype=np.float32)
        sample_vis["_route_num_points"] = int(route_arr.shape[0]) if route_arr.ndim == 2 else 0

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
        for rec in frame_records:
            info = rec.get("release_ready", {})
            start_xy = info.get("borrow_start_world_xy", [])
            end_xy = info.get("borrow_end_world_xy", [])
            if len(start_xy) == 2 and len(end_xy) == 2:
                scene_borrow_context = {
                    "borrow_start_world_xy": list(start_xy),
                    "borrow_end_world_xy": list(end_xy),
                    "borrow_distance_m": float(info.get("borrow_distance_m", 0.0)),
                    "peak_lateral_m": float(info.get("peak_lateral_m", 0.0)),
                    "release_to_return_s": float(info.get("release_to_return_s", 0.0)),
                    "borrow_duration_s": float(info.get("borrow_duration_s", 0.0)),
                }
                break
        if scene_borrow_context is None:
            for rec in frame_records:
                route_local = np.asarray(
                    rec.get("sample_vis", {}).get("_route_input_local", np.zeros((0, 2), dtype=np.float32)),
                    dtype=np.float32,
                )
                current_meas = rec.get("current_meas")
                ego_matrix_current = None if current_meas is None else current_meas.get("ego_matrix", None)
                if ego_matrix_current is None:
                    continue
                enter_thresh = 1.25
                return_thresh = 0.8
                min_enter_m = 4.0
                min_return_m = 6.0
                if _is_borrow_cross_scene_context(event_name=event_name):
                    enter_thresh = 0.6
                    return_thresh = 0.5
                    min_enter_m = 2.0
                    min_return_m = 3.0
                borrow_geom = _estimate_borrow_points_from_wait_route(
                    route_local=route_local,
                    borrow_enter_lateral_thresh=float(enter_thresh),
                    return_lateral_thresh=float(return_thresh),
                    min_enter_progress_m=float(min_enter_m),
                    min_return_progress_m=float(min_return_m),
                    segment_step_m=max(0.25, float(args.front_route_step_m)),
                )
                if borrow_geom is None:
                    continue
                borrow_world = _transform_points_local_to_world_xyz(
                    np.stack([borrow_geom["enter_local_xy"], borrow_geom["return_local_xy"]], axis=0),
                    ego_matrix_current,
                )
                if borrow_world.shape[0] != 2:
                    continue
                scene_borrow_context = {
                    "borrow_start_world_xy": borrow_world[0, :2].astype(float).tolist(),
                    "borrow_end_world_xy": borrow_world[1, :2].astype(float).tolist(),
                    "borrow_distance_m": float(borrow_geom["borrow_distance_m"]),
                    "peak_lateral_m": float(borrow_geom["peak_lateral_m"]),
                    "release_to_return_s": 0.0,
                    "borrow_duration_s": 0.0,
                }
                break
        if scene_borrow_context is None:
            for rec in frame_records:
                current_meas = rec.get("current_meas")
                route_local = np.asarray(
                    rec.get("sample_vis", {}).get("_route_input_local", np.zeros((0, 2), dtype=np.float32)),
                    dtype=np.float32,
                )
                ego_matrix_current = None if current_meas is None else current_meas.get("ego_matrix", None)
                if ego_matrix_current is None or route_local.ndim != 2 or route_local.shape[0] < 2:
                    continue
                debug = rec.get("debug", {})
                current_cover = _cover_candidate_summary(1, debug.get("best_current"), debug, current_meas=current_meas, event_name=event_name)
                future_cover = _cover_candidate_summary(2, debug.get("best_future"), debug, current_meas=current_meas, event_name=event_name)
                if str(current_cover.get("interaction", {}).get("subtype", "none")) == "borrow_cross_meet":
                    cover_pt = None if debug.get("best_current") is None else np.asarray(debug["best_current"].get("cover", {}).get("route_point", None), dtype=np.float32)
                elif str(future_cover.get("interaction", {}).get("subtype", "none")) == "borrow_cross_meet":
                    cover_pt = None if debug.get("best_future") is None else np.asarray(debug["best_future"].get("cover", {}).get("route_point", None), dtype=np.float32)
                else:
                    cover_pt = None
                if cover_pt is None or np.asarray(cover_pt).shape != (2,):
                    continue
                cover_proj, cover_s = _project_point_to_polyline(np.asarray(cover_pt, dtype=np.float32), route_local)
                if cover_proj is None or cover_s is None:
                    continue
                ego_length_m = _ego_length_m(rec.get("current_boxes"))
                corridor_len_m = float(max(3.0 * ego_length_m, ego_length_m))
                start_s = max(float(cover_s) - 0.5 * corridor_len_m, 0.0)
                end_s = min(float(cover_s) + 0.5 * corridor_len_m, float(_polyline_length_m(route_local)))
                corridor_local = _sample_polyline_at_arclengths(
                    route_local,
                    np.asarray([start_s, end_s], dtype=np.float32),
                )
                corridor_world = _transform_points_local_to_world_xyz(corridor_local, ego_matrix_current)
                if corridor_world.shape[0] != 2:
                    continue
                scene_borrow_context = {
                    "borrow_start_world_xy": corridor_world[0, :2].astype(float).tolist(),
                    "borrow_end_world_xy": corridor_world[1, :2].astype(float).tolist(),
                    "borrow_distance_m": float(max(end_s - start_s, 0.0)),
                    "peak_lateral_m": 0.0,
                    "release_to_return_s": 0.0,
                    "borrow_duration_s": 0.0,
                }
                break

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
            "borrow_distance_m": 0.0,
            "peak_lateral_m": 0.0,
            "blocked_frame_id": -1,
            "blocking_actor_id": -1,
            "blocking_actor_class": "none",
        })
        if scene_borrow_context is not None:
            start_xy = release_info.get("borrow_start_world_xy", [])
            end_xy = release_info.get("borrow_end_world_xy", [])
            if not (len(start_xy) == 2 and len(end_xy) == 2):
                release_info = dict(release_info)
                release_info.update(scene_borrow_context)
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
