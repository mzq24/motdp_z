#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter

import numpy as np


DEBUG_KEY = "cover_relation_graph_boundary"

CURRENT_EDGE_VALID_KEY = "current_cover_edge_valid"
CURRENT_EDGE_OCCUPIED_KEY = "current_cover_edge_occupied"
CURRENT_EDGE_MODE_KEY = "current_cover_edge_mode"
CURRENT_EDGE_MODE_VALID_KEY = "current_cover_edge_mode_valid"

FUTURE_EDGE_VALID_KEY = "future_cover_edge_valid"
FUTURE_EDGE_MODE_KEY = "future_cover_edge_mode"
FUTURE_EDGE_MODE_VALID_KEY = "future_cover_edge_mode_valid"

CURRENT_UPPER_SPEED_KEY = "current_cover_upper_speed_mps"
CURRENT_UPPER_VALID_KEY = "current_cover_upper_speed_valid"
CURRENT_UPPER_SOURCE_KEY = "current_cover_upper_speed_source"

FUTURE_LOWER_SPEED_KEY = "future_cover_lower_speed_mps"
FUTURE_LOWER_VALID_KEY = "future_cover_lower_speed_valid"
FUTURE_LOWER_SOURCE_KEY = "future_cover_lower_speed_source"

FRONT_FOLLOW_UPPER_SPEED_KEY = "front_follow_upper_speed_mps"
FRONT_FOLLOW_UPPER_VALID_KEY = "front_follow_upper_speed_valid"

MERGE_FLOW_LOWER_SPEED_KEY = "merge_flow_lower_speed_mps"
MERGE_FLOW_LOWER_VALID_KEY = "merge_flow_lower_speed_valid"

MODE_NONE = 0
MODE_PASS_AFTER_CURRENT = 1
MODE_GO_BEFORE_FUTURE = 2
MODE_YIELD_AFTER_FUTURE = 3
MODE_AMBIGUOUS = 4

MODE_NAMES = {
    MODE_NONE: "none",
    MODE_PASS_AFTER_CURRENT: "pass_after_current",
    MODE_GO_BEFORE_FUTURE: "go_before_future",
    MODE_YIELD_AFTER_FUTURE: "yield_after_future",
    MODE_AMBIGUOUS: "ambiguous",
}

SOURCE_NONE = 0
SOURCE_JUNCTION_YLD_MAX = 1
SOURCE_FAMILY_GO_MIN = 2
SOURCE_CHASE_SPEED_MAX = 3
SOURCE_MERGE_FOLLOW_THROUGH_VBMIN = 4

SOURCE_NAMES = {
    SOURCE_NONE: "none",
    SOURCE_JUNCTION_YLD_MAX: "junction_yld_max",
    SOURCE_FAMILY_GO_MIN: "family_go_min",
    SOURCE_CHASE_SPEED_MAX: "chase_speed_max",
    SOURCE_MERGE_FOLLOW_THROUGH_VBMIN: "merge_follow_through_vbmin",
}

ROLE_NONE = 0
ROLE_YLD_TARGET_ACTOR = 1
ROLE_GO_BEFORE_NEXT_ACTOR = 2
ROLE_CURRENT_AREA_ACTOR = 3
ROLE_OPEN_UNBOUNDED = 4

PHASE_NONE = 0
PHASE_YLD = 1
PHASE_GO = 2

FAMILY_CODE_TO_NAME = {
    0: "none",
    1: "borrow",
    2: "merge",
    3: "junction",
}
ACTIVE_FAMILIES = {"borrow", "merge", "junction"}
AREA_STATUS_AFTER = 3
MERGE_CURRENT_COVER_AREA_START_MARGIN_M = 1.0


def _atomic_pickle_dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _to_float(value, default=np.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    return float(value)


def _finite_float(value, default=np.nan) -> float:
    value = _to_float(value, default=default)
    return float(value) if np.isfinite(value) else float(default)


def _to_int(value, default: int = -1) -> int:
    try:
        if value is None:
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _base_dir_from_sample(sample: dict) -> str:
    feat = str(sample.get("transfuser_bev_feature", "") or "")
    if feat:
        return os.path.dirname(os.path.dirname(feat))
    return str(sample.get("route_name", "") or "")


def _frame_id(sample: dict) -> int:
    return _to_int(sample.get("frame_id", -1), default=-1)


def _group_indices_by_route(samples: list[dict]):
    route_to_indices: dict[str, list[int]] = {}
    route_order: list[str] = []
    for idx, sample in enumerate(samples):
        route_key = _base_dir_from_sample(sample)
        if route_key not in route_to_indices:
            route_to_indices[route_key] = []
            route_order.append(route_key)
        route_to_indices[route_key].append(int(idx))

    grouped = []
    for route_key in route_order:
        indices = route_to_indices[route_key]
        indices.sort(key=lambda idx: _frame_id(samples[int(idx)]))
        grouped.append((route_key, indices))
    return grouped


def _ensure_stage1_debug(sample: dict) -> dict:
    stage1_debug = sample.get("stage1_speed_debug")
    if not isinstance(stage1_debug, dict):
        stage1_debug = {}
        sample["stage1_speed_debug"] = stage1_debug
    return stage1_debug


def _stage1_block(sample: dict, key: str) -> dict:
    stage1_debug = sample.get("stage1_speed_debug")
    if not isinstance(stage1_debug, dict):
        return {}
    block = stage1_debug.get(key)
    return block if isinstance(block, dict) else {}


def _cover_exists(cover: dict) -> bool:
    return bool(_to_float((cover or {}).get("exists", 0.0), default=0.0) > 0.5)


def _cover_actor_id(cover: dict) -> int:
    return _to_int((cover or {}).get("actor_id", -1), default=-1)


def _actor_valid(actor_id: int) -> bool:
    return bool(int(actor_id) >= 0)


def _family_name(sample: dict) -> str:
    conflict_area = _stage1_block(sample, "conflict_area")
    family = str(conflict_area.get("family", "none"))
    if family != "none":
        return family
    return FAMILY_CODE_TO_NAME.get(_to_int(sample.get("conflict_area_family", 0), default=0), "none")


def _is_active_conflict_sample(sample: dict) -> bool:
    if _to_float(sample.get("conflict_area_active", 0.0), default=0.0) <= 0.5:
        return False
    return _family_name(sample) in ACTIVE_FAMILIES


def _conflict_window_identity(sample: dict):
    if not _is_active_conflict_sample(sample):
        return None
    family = _family_name(sample)
    conflict_info = _stage1_block(sample, "conflict_area")
    return (
        str(family),
        _to_int(conflict_info.get("start_frame", sample.get("conflict_area_start_frame", -1)), default=-1),
        _to_int(conflict_info.get("end_frame", sample.get("conflict_area_end_frame", -1)), default=-1),
        str(conflict_info.get("source", "none")),
        _to_int(conflict_info.get("source_episode_id", -1), default=-1),
    )


def _project_point_to_polyline_progress(polyline_xyz, point_xyz) -> float:
    polyline_xyz = np.asarray(polyline_xyz, dtype=np.float32)
    point_xyz = np.asarray(point_xyz, dtype=np.float32).reshape(-1)
    if polyline_xyz.ndim != 2 or polyline_xyz.shape[0] < 2 or polyline_xyz.shape[1] < 3 or point_xyz.shape[0] < 3:
        return np.nan

    best_dist = np.inf
    best_progress = np.nan
    prefix = 0.0
    for seg_idx in range(int(polyline_xyz.shape[0]) - 1):
        p0 = polyline_xyz[seg_idx, :3].astype(np.float64)
        p1 = polyline_xyz[seg_idx + 1, :3].astype(np.float64)
        seg = p1 - p0
        seg_norm_sq = float(np.dot(seg, seg))
        seg_len = float(np.sqrt(max(seg_norm_sq, 0.0)))
        if seg_len <= 1e-6:
            continue
        rel = point_xyz[:3].astype(np.float64) - p0
        t = float(np.clip(np.dot(rel, seg) / seg_norm_sq, 0.0, 1.0))
        proj = p0 + t * seg
        dist = float(np.linalg.norm(point_xyz[:3].astype(np.float64) - proj))
        if dist < best_dist:
            best_dist = dist
            best_progress = float(prefix + t * seg_len)
        prefix += seg_len
    return float(best_progress)


def _merge_current_cover_overlaps_area(sample: dict, current_cover: dict) -> tuple[bool, str]:
    if not _cover_exists(current_cover):
        return False, "missing_current_cover"

    conflict_area = _stage1_block(sample, "conflict_area")
    area_start_s_m = _finite_float(conflict_area.get("area_start_s_m", np.nan), default=np.nan)
    area_end_s_m = _finite_float(conflict_area.get("area_end_s_m", np.nan), default=np.nan)
    if not (np.isfinite(area_start_s_m) and np.isfinite(area_end_s_m)):
        return False, "missing_merge_area_progress"

    conflict_s_m = _finite_float(current_cover.get("scene_route_conflict_s_m", np.nan), default=np.nan)
    if np.isfinite(conflict_s_m):
        ok = bool(
            float(area_start_s_m) - float(MERGE_CURRENT_COVER_AREA_START_MARGIN_M)
            <= float(conflict_s_m)
            <= float(area_end_s_m)
        )
        return ok, "none" if ok else "merge_current_cover_outside_area"

    area_segment_xyz = np.asarray(conflict_area.get("area_segment_world_xyz", []), dtype=np.float32)
    conflict_world_xyz = np.asarray(current_cover.get("scene_route_conflict_world_xyz", []), dtype=np.float32).reshape(-1)
    if area_segment_xyz.ndim != 2 or area_segment_xyz.shape[0] < 2 or conflict_world_xyz.shape[0] < 3:
        return False, "missing_merge_cover_projection"

    segment_progress_m = _project_point_to_polyline_progress(area_segment_xyz, conflict_world_xyz)
    if not np.isfinite(segment_progress_m):
        return False, "merge_cover_projection_failed"
    projected_conflict_s_m = float(area_start_s_m) + float(segment_progress_m)
    ok = bool(
        float(area_start_s_m) - float(MERGE_CURRENT_COVER_AREA_START_MARGIN_M)
        <= float(projected_conflict_s_m)
        <= float(area_end_s_m)
    )
    return ok, "none" if ok else "merge_current_cover_outside_area"


def _current_cover_in_family_area(sample: dict, family: str, current_cover: dict) -> tuple[bool, str]:
    if not _cover_exists(current_cover):
        return False, "missing_current_cover"
    if family == "merge":
        return _merge_current_cover_overlaps_area(sample, current_cover)
    # Borrow / junction current cover is already family-filtered by precompute.
    return True, "none"


def _future_gate_info(future_cover: dict, frame_gate: int, distance_gate_m: float) -> dict:
    actor_id = _cover_actor_id(future_cover)
    exists = _cover_exists(future_cover)
    frame_index = _to_int((future_cover or {}).get("frame_index", -1), default=-1)
    d_bg = _finite_float((future_cover or {}).get("d_bg", np.nan), default=np.nan)

    if not exists:
        gate_reason = "no_future_cover"
    elif frame_index <= 0 or frame_index > int(frame_gate):
        gate_reason = "future_frame_gate_fail"
    elif not np.isfinite(d_bg):
        gate_reason = "future_distance_missing"
    elif float(d_bg) > float(distance_gate_m):
        gate_reason = "future_distance_gate_fail"
    else:
        gate_reason = "none"

    return {
        "exists": float(1.0 if exists else 0.0),
        "actor_id": int(actor_id),
        "actor_valid": float(1.0 if _actor_valid(actor_id) else 0.0),
        "frame_index": int(frame_index),
        "distance_m": float(d_bg) if np.isfinite(d_bg) else np.nan,
        "gate_passed": float(1.0 if exists and gate_reason == "none" else 0.0),
        "gate_reason": str(gate_reason),
        "frame_gate": int(frame_gate),
        "distance_gate_m": float(distance_gate_m),
    }


def _ego_cleared_area(sample: dict) -> bool:
    status = _to_int(sample.get("conflict_area_status", 0), default=0)
    if status >= AREA_STATUS_AFTER:
        return True
    dist_exit = _finite_float(sample.get("conflict_dist_to_exit_m", np.nan), default=np.nan)
    return bool(np.isfinite(dist_exit) and dist_exit <= 0.0)


def _phase_object_future_mode(sample: dict, future_actor_id: int) -> tuple[int, bool, str]:
    role = _to_int(sample.get("conflict_phase_ref_role", ROLE_NONE), default=ROLE_NONE)
    actor_id = _to_int(sample.get("conflict_phase_ref_actor_id", -1), default=-1)
    actor_valid = _to_float(sample.get("conflict_phase_ref_actor_valid", 0.0), default=0.0) > 0.5
    if not (actor_valid and _actor_valid(future_actor_id) and int(actor_id) == int(future_actor_id)):
        return MODE_NONE, False, "phase_ref_not_future_actor"
    if role == ROLE_GO_BEFORE_NEXT_ACTOR:
        return MODE_GO_BEFORE_FUTURE, True, "phase_object_binding"
    if role == ROLE_YLD_TARGET_ACTOR:
        return MODE_YIELD_AFTER_FUTURE, True, "phase_object_binding"
    return MODE_NONE, False, "phase_ref_role_not_future"


def _temporal_future_mode(
    samples: list[dict],
    route_indices: list[int],
    anchor_pos: int,
    anchor_window,
    family: str,
    future_actor_id: int,
    lookahead: int,
) -> tuple[int, bool, dict]:
    transition = {
        "source": "temporal_actor_transition",
        "lookahead": int(lookahead),
        "future_actor_id": int(future_actor_id),
        "future_becomes_current_offset": -1,
        "future_becomes_current_frame": -1,
        "ego_clears_offset": -1,
        "ego_clears_frame": -1,
        "issue_reason": "none",
    }
    if not _actor_valid(future_actor_id):
        transition["issue_reason"] = "future_actor_id_invalid"
        return MODE_AMBIGUOUS, False, transition

    future_current_offset = None
    ego_clear_offset = None
    max_pos = min(len(route_indices) - 1, int(anchor_pos) + int(max(lookahead, 0)))
    for pos in range(int(anchor_pos), max_pos + 1):
        sample = samples[int(route_indices[pos])]
        if _conflict_window_identity(sample) != anchor_window:
            break
        rel_offset = int(pos - int(anchor_pos))
        if ego_clear_offset is None and _ego_cleared_area(sample):
            ego_clear_offset = rel_offset
            transition["ego_clears_offset"] = rel_offset
            transition["ego_clears_frame"] = _frame_id(sample)
        if future_current_offset is None:
            current_cover = _stage1_block(sample, "current_cover")
            current_ok, _ = _current_cover_in_family_area(sample, family, current_cover)
            if current_ok and _cover_actor_id(current_cover) == int(future_actor_id):
                future_current_offset = rel_offset
                transition["future_becomes_current_offset"] = rel_offset
                transition["future_becomes_current_frame"] = _frame_id(sample)
        if future_current_offset is not None and ego_clear_offset is not None:
            break

    if future_current_offset is None and ego_clear_offset is None:
        transition["issue_reason"] = "temporal_order_missing"
        return MODE_AMBIGUOUS, False, transition
    if ego_clear_offset is not None and (future_current_offset is None or ego_clear_offset <= future_current_offset):
        transition["issue_reason"] = "none"
        return MODE_GO_BEFORE_FUTURE, True, transition
    if future_current_offset is not None:
        transition["issue_reason"] = "none"
        return MODE_YIELD_AFTER_FUTURE, True, transition

    transition["issue_reason"] = "temporal_order_ambiguous"
    return MODE_AMBIGUOUS, False, transition


def _family_go_min_field(family: str) -> tuple[str, str]:
    if family == "merge":
        return "merge_go_min_speed", "merge_go_min_speed_valid"
    if family == "borrow":
        return "borrow_go_min_speed", "borrow_go_min_speed_valid"
    if family == "junction":
        return "junction_go_min_speed", "junction_go_min_speed_valid"
    return "", ""


def _default_values() -> dict:
    return {
        CURRENT_EDGE_VALID_KEY: np.float32(0.0),
        CURRENT_EDGE_OCCUPIED_KEY: np.float32(0.0),
        CURRENT_EDGE_MODE_KEY: np.int64(MODE_NONE),
        CURRENT_EDGE_MODE_VALID_KEY: np.float32(0.0),
        FUTURE_EDGE_VALID_KEY: np.float32(0.0),
        FUTURE_EDGE_MODE_KEY: np.int64(MODE_NONE),
        FUTURE_EDGE_MODE_VALID_KEY: np.float32(0.0),
        CURRENT_UPPER_SPEED_KEY: np.float32(np.nan),
        CURRENT_UPPER_VALID_KEY: np.float32(0.0),
        CURRENT_UPPER_SOURCE_KEY: np.int64(SOURCE_NONE),
        FUTURE_LOWER_SPEED_KEY: np.float32(np.nan),
        FUTURE_LOWER_VALID_KEY: np.float32(0.0),
        FUTURE_LOWER_SOURCE_KEY: np.int64(SOURCE_NONE),
        FRONT_FOLLOW_UPPER_SPEED_KEY: np.float32(np.nan),
        FRONT_FOLLOW_UPPER_VALID_KEY: np.float32(0.0),
        MERGE_FLOW_LOWER_SPEED_KEY: np.float32(np.nan),
        MERGE_FLOW_LOWER_VALID_KEY: np.float32(0.0),
    }


def _default_debug() -> dict:
    return {
        "active": 0.0,
        "family": "none",
        "current_edge": {
            "valid": 0.0,
            "occupied": 0.0,
            "mode": int(MODE_NONE),
            "mode_name": MODE_NAMES[MODE_NONE],
            "mode_valid": 0.0,
            "actor_id": -1,
            "area_reason": "none",
        },
        "future_edge": {
            "valid": 0.0,
            "mode": int(MODE_NONE),
            "mode_name": MODE_NAMES[MODE_NONE],
            "mode_valid": 0.0,
            "actor_id": -1,
            "gate_reason": "none",
            "mode_source": "none",
        },
        "future_transition": {
            "source": "none",
            "issue_reason": "none",
        },
        "speed_sources": {
            "current_upper": SOURCE_NAMES[SOURCE_NONE],
            "future_lower": SOURCE_NAMES[SOURCE_NONE],
            "front_follow_upper": SOURCE_NAMES[SOURCE_NONE],
            "merge_flow_lower": SOURCE_NAMES[SOURCE_NONE],
        },
        "issue_reason": "not_computed",
    }


def _has_existing_fields(sample: dict) -> bool:
    stage1_debug = sample.get("stage1_speed_debug")
    debug = stage1_debug.get(DEBUG_KEY) if isinstance(stage1_debug, dict) else None
    return bool(
        CURRENT_EDGE_VALID_KEY in sample
        and CURRENT_EDGE_OCCUPIED_KEY in sample
        and CURRENT_EDGE_MODE_KEY in sample
        and CURRENT_EDGE_MODE_VALID_KEY in sample
        and FUTURE_EDGE_VALID_KEY in sample
        and FUTURE_EDGE_MODE_KEY in sample
        and FUTURE_EDGE_MODE_VALID_KEY in sample
        and CURRENT_UPPER_SPEED_KEY in sample
        and CURRENT_UPPER_VALID_KEY in sample
        and CURRENT_UPPER_SOURCE_KEY in sample
        and FUTURE_LOWER_SPEED_KEY in sample
        and FUTURE_LOWER_VALID_KEY in sample
        and FUTURE_LOWER_SOURCE_KEY in sample
        and FRONT_FOLLOW_UPPER_SPEED_KEY in sample
        and FRONT_FOLLOW_UPPER_VALID_KEY in sample
        and MERGE_FLOW_LOWER_SPEED_KEY in sample
        and MERGE_FLOW_LOWER_VALID_KEY in sample
        and isinstance(debug, dict)
    )


def _compute_annotation(
    samples: list[dict],
    route_indices: list[int],
    anchor_pos: int,
    args,
) -> tuple[dict, dict]:
    values = _default_values()
    debug = _default_debug()

    sample = samples[int(route_indices[int(anchor_pos)])]
    family = _family_name(sample)
    active = _is_active_conflict_sample(sample)
    anchor_window = _conflict_window_identity(sample)
    debug.update({"active": float(1.0 if active else 0.0), "family": str(family)})
    if not active or anchor_window is None:
        debug["issue_reason"] = "inactive_or_unsupported_family"
        return values, debug

    current_cover = _stage1_block(sample, "current_cover")
    current_actor_id = _cover_actor_id(current_cover)
    current_ok, current_reason = _current_cover_in_family_area(sample, family, current_cover)
    if current_ok:
        values[CURRENT_EDGE_VALID_KEY] = np.float32(1.0)
        values[CURRENT_EDGE_OCCUPIED_KEY] = np.float32(1.0)
        values[CURRENT_EDGE_MODE_KEY] = np.int64(MODE_PASS_AFTER_CURRENT)
        values[CURRENT_EDGE_MODE_VALID_KEY] = np.float32(1.0)
    debug["current_edge"] = {
        "valid": float(values[CURRENT_EDGE_VALID_KEY]),
        "occupied": float(values[CURRENT_EDGE_OCCUPIED_KEY]),
        "mode": int(values[CURRENT_EDGE_MODE_KEY]),
        "mode_name": MODE_NAMES[int(values[CURRENT_EDGE_MODE_KEY])],
        "mode_valid": float(values[CURRENT_EDGE_MODE_VALID_KEY]),
        "actor_id": int(current_actor_id),
        "area_reason": str(current_reason),
    }

    future_cover = _stage1_block(sample, "future_cover")
    future_gate = _future_gate_info(future_cover, args.future_frame_gate, args.future_distance_gate_m)
    future_actor_id = int(future_gate["actor_id"])
    future_edge_valid = float(future_gate["gate_passed"]) > 0.5
    future_mode = MODE_NONE
    future_mode_valid = False
    future_mode_source = "none"
    transition_debug = {"source": "none", "issue_reason": "none"}
    if future_edge_valid:
        values[FUTURE_EDGE_VALID_KEY] = np.float32(1.0)
        future_mode, future_mode_valid, future_mode_source = _phase_object_future_mode(sample, future_actor_id)
        if not future_mode_valid:
            future_mode, future_mode_valid, transition_debug = _temporal_future_mode(
                samples,
                route_indices,
                int(anchor_pos),
                anchor_window,
                family,
                future_actor_id,
                args.temporal_lookahead_frames,
            )
            future_mode_source = "temporal_actor_transition"
    else:
        future_mode_source = str(future_gate["gate_reason"])

    values[FUTURE_EDGE_MODE_KEY] = np.int64(int(future_mode))
    values[FUTURE_EDGE_MODE_VALID_KEY] = np.float32(1.0 if future_mode_valid else 0.0)
    debug["future_edge"] = {
        "valid": float(values[FUTURE_EDGE_VALID_KEY]),
        "mode": int(values[FUTURE_EDGE_MODE_KEY]),
        "mode_name": MODE_NAMES[int(values[FUTURE_EDGE_MODE_KEY])],
        "mode_valid": float(values[FUTURE_EDGE_MODE_VALID_KEY]),
        "actor_id": int(future_actor_id),
        "frame_index": int(future_gate["frame_index"]),
        "distance_m": float(future_gate["distance_m"]) if np.isfinite(future_gate["distance_m"]) else np.nan,
        "gate_reason": str(future_gate["gate_reason"]),
        "mode_source": str(future_mode_source),
    }
    debug["future_transition"] = transition_debug

    if family == "junction" and current_ok:
        upper = _finite_float(sample.get("junction_yld_max_speed", np.nan), default=np.nan)
        upper_valid = bool(_to_float(sample.get("junction_yld_max_speed_valid", 0.0), default=0.0) > 0.5 and np.isfinite(upper))
        if upper_valid:
            values[CURRENT_UPPER_SPEED_KEY] = np.float32(float(upper))
            values[CURRENT_UPPER_VALID_KEY] = np.float32(1.0)
            values[CURRENT_UPPER_SOURCE_KEY] = np.int64(SOURCE_JUNCTION_YLD_MAX)

    go_field, go_valid_field = _family_go_min_field(family)
    if future_mode == MODE_GO_BEFORE_FUTURE and future_mode_valid and go_field:
        lower = _finite_float(sample.get(go_field, np.nan), default=np.nan)
        lower_valid = bool(_to_float(sample.get(go_valid_field, 0.0), default=0.0) > 0.5 and np.isfinite(lower))
        if lower_valid:
            values[FUTURE_LOWER_SPEED_KEY] = np.float32(float(lower))
            values[FUTURE_LOWER_VALID_KEY] = np.float32(1.0)
            values[FUTURE_LOWER_SOURCE_KEY] = np.int64(SOURCE_FAMILY_GO_MIN)

    front_upper = _finite_float(sample.get("chase_speed_max", np.nan), default=np.nan)
    front_valid = bool(_to_float(sample.get("chase_speed_max_valid", 0.0), default=0.0) > 0.5 and np.isfinite(front_upper))
    if front_valid:
        values[FRONT_FOLLOW_UPPER_SPEED_KEY] = np.float32(float(front_upper))
        values[FRONT_FOLLOW_UPPER_VALID_KEY] = np.float32(1.0)

    merge_flow = _finite_float(sample.get("merge_follow_through_vbmin", np.nan), default=np.nan)
    merge_flow_valid = bool(_to_float(sample.get("merge_follow_through_vbmin_valid", 0.0), default=0.0) > 0.5 and np.isfinite(merge_flow))
    if merge_flow_valid:
        values[MERGE_FLOW_LOWER_SPEED_KEY] = np.float32(float(merge_flow))
        values[MERGE_FLOW_LOWER_VALID_KEY] = np.float32(1.0)

    debug["speed_sources"] = {
        "current_upper": SOURCE_NAMES[int(values[CURRENT_UPPER_SOURCE_KEY])],
        "future_lower": SOURCE_NAMES[int(values[FUTURE_LOWER_SOURCE_KEY])],
        "front_follow_upper": SOURCE_NAMES[SOURCE_CHASE_SPEED_MAX] if front_valid else SOURCE_NAMES[SOURCE_NONE],
        "merge_flow_lower": SOURCE_NAMES[SOURCE_MERGE_FOLLOW_THROUGH_VBMIN] if merge_flow_valid else SOURCE_NAMES[SOURCE_NONE],
        "current_upper_mps": float(values[CURRENT_UPPER_SPEED_KEY]) if float(values[CURRENT_UPPER_VALID_KEY]) > 0.5 else np.nan,
        "future_lower_mps": float(values[FUTURE_LOWER_SPEED_KEY]) if float(values[FUTURE_LOWER_VALID_KEY]) > 0.5 else np.nan,
        "front_follow_upper_mps": float(values[FRONT_FOLLOW_UPPER_SPEED_KEY]) if front_valid else np.nan,
        "merge_flow_lower_mps": float(values[MERGE_FLOW_LOWER_SPEED_KEY]) if merge_flow_valid else np.nan,
    }
    issue_reasons = [
        str(current_reason),
        str(future_gate["gate_reason"]),
        str(transition_debug.get("issue_reason", "none")),
    ]
    debug["issue_reason"] = "none" if all(reason == "none" for reason in issue_reasons) else "|".join(
        reason for reason in issue_reasons if reason != "none"
    )
    return values, debug


def _write_annotation(sample: dict, values: dict, debug: dict) -> None:
    sample.update(values)
    stage1_debug = _ensure_stage1_debug(sample)
    stage1_debug[DEBUG_KEY] = debug


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Postprocess stage1 cover relation graph edge mode and boundary labels."
    )
    parser.add_argument("--input_path", required=True, help="Existing samples_packed.pkl")
    parser.add_argument("--output_path", required=True, help="Output pickle path")
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Recompute labels even when cover relation graph fields already exist.",
    )
    parser.add_argument(
        "--allow_inplace",
        action="store_true",
        help="Allow --input_path and --output_path to be the same file.",
    )
    parser.add_argument("--future_frame_gate", type=int, default=6)
    parser.add_argument("--future_distance_gate_m", type=float, default=20.0)
    parser.add_argument("--temporal_lookahead_frames", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path)
    if input_path == output_path and not bool(args.allow_inplace):
        raise ValueError("Refusing in-place overwrite without --allow_inplace")

    with open(input_path, "rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"Expected list in {input_path}, got {type(samples).__name__}")

    written = 0
    skipped_existing = 0
    active_count = 0
    family_counts = Counter()
    current_mode_counts = Counter()
    future_mode_counts = Counter()
    future_mode_source_counts = Counter()
    current_upper_valid_count = 0
    future_lower_valid_count = 0
    front_follow_upper_valid_count = 0
    merge_flow_lower_valid_count = 0
    issue_counts = Counter()

    for _, route_indices in _group_indices_by_route(samples):
        for anchor_pos, sample_idx in enumerate(route_indices):
            sample = samples[int(sample_idx)]
            if _has_existing_fields(sample) and not bool(args.overwrite_existing):
                skipped_existing += 1
                current_mode_counts[MODE_NAMES.get(_to_int(sample.get(CURRENT_EDGE_MODE_KEY, MODE_NONE), MODE_NONE), "unknown")] += 1
                future_mode_counts[MODE_NAMES.get(_to_int(sample.get(FUTURE_EDGE_MODE_KEY, MODE_NONE), MODE_NONE), "unknown")] += 1
                continue

            values, debug = _compute_annotation(samples, route_indices, int(anchor_pos), args)
            _write_annotation(sample, values, debug)
            written += 1

            if float(debug.get("active", 0.0)) > 0.5:
                active_count += 1
                family_counts[str(debug.get("family", "none"))] += 1
            current_mode_counts[MODE_NAMES[int(values[CURRENT_EDGE_MODE_KEY])]] += 1
            future_mode_counts[MODE_NAMES[int(values[FUTURE_EDGE_MODE_KEY])]] += 1
            future_edge = debug.get("future_edge", {}) if isinstance(debug, dict) else {}
            future_mode_source_counts[str(future_edge.get("mode_source", "none"))] += 1
            if float(values[CURRENT_UPPER_VALID_KEY]) > 0.5:
                current_upper_valid_count += 1
            if float(values[FUTURE_LOWER_VALID_KEY]) > 0.5:
                future_lower_valid_count += 1
            if float(values[FRONT_FOLLOW_UPPER_VALID_KEY]) > 0.5:
                front_follow_upper_valid_count += 1
            if float(values[MERGE_FLOW_LOWER_VALID_KEY]) > 0.5:
                merge_flow_lower_valid_count += 1
            issue_counts[str(debug.get("issue_reason", "none"))] += 1

    _atomic_pickle_dump(samples, output_path)

    def _summary(counter: Counter) -> str:
        return ",".join(f"{key}:{value}" for key, value in sorted(counter.items()))

    print(
        "postprocess cover relation graph boundary done: "
        f"samples={len(samples)} "
        f"written={written} "
        f"skipped_existing={skipped_existing} "
        f"active={active_count} "
        f"current_upper_valid={current_upper_valid_count} "
        f"future_lower_valid={future_lower_valid_count} "
        f"front_follow_upper_valid={front_follow_upper_valid_count} "
        f"merge_flow_lower_valid={merge_flow_lower_valid_count} "
        f"family_counts={{{_summary(family_counts)}}} "
        f"current_mode_counts={{{_summary(current_mode_counts)}}} "
        f"future_mode_counts={{{_summary(future_mode_counts)}}} "
        f"future_mode_source_counts={{{_summary(future_mode_source_counts)}}} "
        f"issue_counts={{{_summary(issue_counts)}}} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
