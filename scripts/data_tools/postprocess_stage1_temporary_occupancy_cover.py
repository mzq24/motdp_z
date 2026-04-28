#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle

import numpy as np


TEMP_OCCUPANCY_COVER_KEY = "temporary_occupancy_cover_bins"
TEMP_OCCUPANCY_COVER_VALID_KEY = "temporary_occupancy_cover_valid"
TEMP_OCCUPANCY_COVER_DEBUG_KEY = "temporary_occupancy_cover"
TEMP_OCCUPANCY_COVER_TOTAL_SLOTS = 13

GO_OPPORTUNITY_PROB_KEY = "go_opportunity_prob"
YLD_PRESSURE_PROB_KEY = "yld_pressure_prob"
GO_OPPORTUNITY_VALID_KEY = "go_opportunity_valid"

DISTANCE_EXTRA_BIN_THRESH_1_M = 4.0
DISTANCE_EXTRA_BIN_THRESH_2_M = 8.0
MERGE_CURRENT_COVER_AREA_START_MARGIN_M = 1.0
GO_PROB_GO_PHASE_FLOOR_NON_COLLISION = 0.5
CONFLICT_DECISION_GO_CODE = 2

CONFLICT_FAMILY_CODE_TO_NAME = {
    0: "none",
    1: "borrow",
    2: "merge",
    3: "junction",
}


def _atomic_pickle_dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _load_json_gz_if_exists(path: str):
    if not path or not os.path.isfile(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _has_nonempty_infraction(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        try:
            return float(value) > 0.0
        except Exception:
            return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, dict):
        return any(_has_nonempty_infraction(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_nonempty_infraction(v) for v in value)
    return True


def _infer_image_data_root_from_packed_path(path: str) -> str:
    cur = os.path.abspath(str(path or ""))
    while cur and cur != os.path.dirname(cur):
        if os.path.basename(cur) == "tmp_data":
            return os.path.dirname(cur)
        cur = os.path.dirname(cur)
    return ""


def _base_dir_from_sample(sample: dict) -> str:
    feat = str(sample.get("transfuser_bev_feature", "") or "")
    if feat:
        return os.path.dirname(os.path.dirname(feat))
    route_name = str(sample.get("route_name", "") or "")
    return route_name


def _frame_id_from_sample(sample: dict) -> int:
    return int(sample.get("frame_id", -1))


def _group_indices_by_route(samples):
    route_to_indices = {}
    route_order = []
    for idx, sample in enumerate(samples):
        route_key = _base_dir_from_sample(sample)
        if route_key not in route_to_indices:
            route_to_indices[route_key] = []
            route_order.append(route_key)
        route_to_indices[route_key].append(idx)

    grouped = []
    for route_key in route_order:
        indices = route_to_indices[route_key]
        indices.sort(key=lambda idx: _frame_id_from_sample(samples[idx]))
        grouped.append((route_key, indices))
    return grouped


def _ensure_stage1_debug(sample: dict) -> dict:
    stage1_debug = sample.get("stage1_speed_debug")
    if not isinstance(stage1_debug, dict):
        stage1_debug = {}
        sample["stage1_speed_debug"] = stage1_debug
    return stage1_debug


def _sample_stage1_block(sample: dict, key: str) -> dict:
    stage1_debug = sample.get("stage1_speed_debug")
    if not isinstance(stage1_debug, dict):
        return {}
    block = stage1_debug.get(key)
    return block if isinstance(block, dict) else {}


def _sample_conflict_family_name(sample: dict) -> str:
    conflict_info = _sample_stage1_block(sample, "conflict_area")
    family = str(conflict_info.get("family", "none"))
    if family != "none":
        return family
    family_code = int(sample.get("conflict_area_family", 0))
    return str(CONFLICT_FAMILY_CODE_TO_NAME.get(family_code, "none"))


def _conflict_window_identity(sample: dict):
    if float(sample.get("conflict_area_active", 0.0)) <= 0.5:
        return None

    family = _sample_conflict_family_name(sample)
    if family not in {"borrow", "merge", "junction"}:
        return None

    conflict_info = _sample_stage1_block(sample, "conflict_area")
    return (
        str(family),
        int(conflict_info.get("start_frame", sample.get("conflict_area_start_frame", -1))),
        int(conflict_info.get("end_frame", sample.get("conflict_area_end_frame", -1))),
        str(conflict_info.get("source", "none")),
        int(conflict_info.get("source_episode_id", -1)),
    )


def _cover_interaction_name(cover):
    return str(((cover or {}).get("interaction") or {}).get("name", "none"))


def _cover_interaction_subtype(cover):
    interaction = ((cover or {}).get("interaction") or {})
    return str(interaction.get("subtype") or interaction.get("name") or "none")


def _cover_is_merge_meet(cover):
    return int((cover or {}).get("exists", 0.0)) > 0 and _cover_interaction_subtype(cover) == "merge_meet"


def _cover_exists(cover):
    return int((cover or {}).get("exists", 0.0)) > 0


def _merge_current_cover_overlaps_area(sample: dict) -> bool:
    current_cover = _sample_stage1_block(sample, "current_cover")
    if not _cover_exists(current_cover):
        return False

    conflict_area = _sample_stage1_block(sample, "conflict_area")
    area_start_s_m = float(conflict_area.get("area_start_s_m", np.nan))
    area_end_s_m = float(conflict_area.get("area_end_s_m", np.nan))
    if not (np.isfinite(area_start_s_m) and np.isfinite(area_end_s_m)):
        return False

    conflict_s_m = float(current_cover.get("scene_route_conflict_s_m", np.nan))
    if np.isfinite(conflict_s_m):
        return bool(
            float(area_start_s_m) - float(MERGE_CURRENT_COVER_AREA_START_MARGIN_M)
            <= float(conflict_s_m)
            <= float(area_end_s_m)
        )

    area_segment_xyz = np.asarray(conflict_area.get("area_segment_world_xyz", []), dtype=np.float32)
    conflict_world_xyz = np.asarray(current_cover.get("scene_route_conflict_world_xyz", []), dtype=np.float32).reshape(-1)
    if area_segment_xyz.ndim != 2 or area_segment_xyz.shape[0] < 2 or conflict_world_xyz.shape[0] < 3:
        return False

    segment_progress_m = _project_point_to_polyline_progress(area_segment_xyz, conflict_world_xyz)
    if not np.isfinite(segment_progress_m):
        return False
    projected_conflict_s_m = float(area_start_s_m) + float(segment_progress_m)
    return bool(
        float(area_start_s_m) - float(MERGE_CURRENT_COVER_AREA_START_MARGIN_M)
        <= float(projected_conflict_s_m)
        <= float(area_end_s_m)
    )


def _cover_is_borrow_cross_meet(cover):
    return int((cover or {}).get("exists", 0.0)) > 0 and _cover_interaction_subtype(cover) == "borrow_cross_meet"


def _cover_is_cross_meet(cover):
    if int((cover or {}).get("exists", 0.0)) <= 0:
        return False
    if _cover_interaction_name(cover) != "meet":
        return False
    return "cross" in _cover_interaction_subtype(cover)


def _current_cover_matches_family(sample: dict, family: str) -> bool:
    current_cover = _sample_stage1_block(sample, "current_cover")
    if family == "merge":
        return bool(_merge_current_cover_overlaps_area(sample))
    if family == "borrow":
        return bool(_cover_is_borrow_cross_meet(current_cover))
    if family == "junction":
        return bool(_cover_is_cross_meet(current_cover))
    return False


def _default_temporary_occupancy_cover_debug() -> dict:
    return {
        "active": 0.0,
        "family": "none",
        "source": "current_cover_rollout",
        "bins": [0.0] * TEMP_OCCUPANCY_COVER_TOTAL_SLOTS,
        "valid": [0.0] * TEMP_OCCUPANCY_COVER_TOTAL_SLOTS,
        "reference_frame": -1,
        "reference_run_start_expert": -1,
        "reference_run_start_geom": -1,
        "reference_run_start_final": -1,
        "reference_run_len": -1,
        "raw_reference_pass_time_bins": -1,
        "adjusted_run_start_bins": -1,
        "distance_to_area_start_m": np.nan,
        "distance_adjustment_bins": -1,
        "current_run_start": -1,
        "current_run_len": -1,
        "current_remaining_run_len": -1,
        "area_start_frame": -1,
        "area_start_source": "none",
        "goable": 0.0,
        "cycle_id": -1,
        "accepted_cycle": 0.0,
        "go_prob_floor_clamped": 0.0,
        "issue_reason": "none",
        # Backward-compatible aliases kept for existing local debug readers.
        "reference_lead_blocked_len": -1,
        "reference_passable_len": -1,
        "adjusted_pass_time_bins": -1,
        "distance_to_go_frame_progress_m": np.nan,
        "current_lead_blocked_len": -1,
        "current_passable_len": -1,
    }


def _set_temporary_occupancy_defaults(sample: dict) -> None:
    sample[TEMP_OCCUPANCY_COVER_KEY] = np.zeros((TEMP_OCCUPANCY_COVER_TOTAL_SLOTS,), dtype=np.float32)
    sample[TEMP_OCCUPANCY_COVER_VALID_KEY] = np.zeros((TEMP_OCCUPANCY_COVER_TOTAL_SLOTS,), dtype=np.float32)
    sample[GO_OPPORTUNITY_PROB_KEY] = np.float32(0.0)
    sample[YLD_PRESSURE_PROB_KEY] = np.float32(1.0)
    sample[GO_OPPORTUNITY_VALID_KEY] = np.float32(0.0)
    stage1_debug = _ensure_stage1_debug(sample)
    stage1_debug[TEMP_OCCUPANCY_COVER_DEBUG_KEY] = _default_temporary_occupancy_cover_debug()


def _vector_has_expected_len(value, expected_len: int) -> bool:
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return bool(arr.shape == (int(expected_len),))


def _should_write_sample(sample: dict, overwrite_existing: bool) -> bool:
    if overwrite_existing:
        return True
    stage1_debug = sample.get("stage1_speed_debug")
    tempocc_debug = stage1_debug.get(TEMP_OCCUPANCY_COVER_DEBUG_KEY) if isinstance(stage1_debug, dict) else None
    has_debug = isinstance(tempocc_debug, dict)
    return bool(
        not _vector_has_expected_len(sample.get(TEMP_OCCUPANCY_COVER_KEY, None), TEMP_OCCUPANCY_COVER_TOTAL_SLOTS) or
        not _vector_has_expected_len(sample.get(TEMP_OCCUPANCY_COVER_VALID_KEY, None), TEMP_OCCUPANCY_COVER_TOTAL_SLOTS) or
        GO_OPPORTUNITY_PROB_KEY not in sample or
        YLD_PRESSURE_PROB_KEY not in sample or
        GO_OPPORTUNITY_VALID_KEY not in sample or
        not has_debug or
        "goable" not in tempocc_debug or
        "cycle_id" not in tempocc_debug or
        "go_prob_floor_clamped" not in tempocc_debug
    )


def _set_temporary_occupancy_bins_annotation(sample: dict, family: str, bins, valid) -> None:
    bins_arr = np.asarray(bins, dtype=np.float32).reshape(TEMP_OCCUPANCY_COVER_TOTAL_SLOTS)
    valid_arr = np.asarray(valid, dtype=np.float32).reshape(TEMP_OCCUPANCY_COVER_TOTAL_SLOTS)
    sample[TEMP_OCCUPANCY_COVER_KEY] = bins_arr.astype(np.float32)
    sample[TEMP_OCCUPANCY_COVER_VALID_KEY] = valid_arr.astype(np.float32)
    stage1_debug = _ensure_stage1_debug(sample)
    debug = _default_temporary_occupancy_cover_debug()
    current = stage1_debug.get(TEMP_OCCUPANCY_COVER_DEBUG_KEY)
    if isinstance(current, dict):
        debug.update(dict(current))
    debug.update({
        "active": float(valid_arr[0] > 0.5),
        "family": str(family),
        "source": "current_cover_rollout",
        "bins": bins_arr.astype(float).tolist(),
        "valid": valid_arr.astype(float).tolist(),
    })
    stage1_debug[TEMP_OCCUPANCY_COVER_DEBUG_KEY] = debug


def _set_go_opportunity_annotation(
    sample: dict,
    go_prob: float,
    valid: float,
    cycle_id: int,
    accepted_cycle: float,
    goable: float,
    go_prob_floor_clamped: float,
    reference_frame: int,
    reference_run_start_expert: int,
    reference_run_start_geom: int,
    reference_run_start_final: int,
    reference_run_len: int,
    raw_reference_pass_time_bins: int,
    adjusted_run_start_bins: int,
    distance_to_area_start_m: float,
    distance_adjustment_bins: int,
    current_run_start: int,
    current_run_len: int,
    current_remaining_run_len: int,
    area_start_frame: int,
    area_start_source: str,
    issue_reason: str,
) -> None:
    go_prob = float(np.clip(float(go_prob), 0.0, 1.0))
    valid = float(valid)
    sample[GO_OPPORTUNITY_PROB_KEY] = np.float32(go_prob)
    sample[YLD_PRESSURE_PROB_KEY] = np.float32(1.0 - go_prob)
    sample[GO_OPPORTUNITY_VALID_KEY] = np.float32(valid)
    stage1_debug = _ensure_stage1_debug(sample)
    debug = _default_temporary_occupancy_cover_debug()
    current = stage1_debug.get(TEMP_OCCUPANCY_COVER_DEBUG_KEY)
    if isinstance(current, dict):
        debug.update(dict(current))
    debug.update({
        "reference_frame": int(reference_frame),
        "reference_run_start_expert": int(reference_run_start_expert),
        "reference_run_start_geom": int(reference_run_start_geom),
        "reference_run_start_final": int(reference_run_start_final),
        "reference_run_len": int(reference_run_len),
        "raw_reference_pass_time_bins": int(raw_reference_pass_time_bins),
        "adjusted_run_start_bins": int(adjusted_run_start_bins),
        "distance_to_area_start_m": float(distance_to_area_start_m) if np.isfinite(distance_to_area_start_m) else np.nan,
        "distance_adjustment_bins": int(distance_adjustment_bins),
        "current_run_start": int(current_run_start),
        "current_run_len": int(current_run_len),
        "current_remaining_run_len": int(current_remaining_run_len),
        "area_start_frame": int(area_start_frame),
        "area_start_source": str(area_start_source),
        "goable": float(goable),
        "cycle_id": int(cycle_id),
        "accepted_cycle": float(accepted_cycle),
        "go_prob_floor_clamped": float(go_prob_floor_clamped),
        "issue_reason": str(issue_reason),
        # Backward-compatible aliases.
        "reference_lead_blocked_len": int(reference_run_start_expert),
        "reference_passable_len": int(reference_run_len),
        "adjusted_pass_time_bins": int(adjusted_run_start_bins),
        "distance_to_go_frame_progress_m": float(distance_to_area_start_m) if np.isfinite(distance_to_area_start_m) else np.nan,
        "current_lead_blocked_len": int(current_run_start),
        "current_passable_len": int(current_run_len),
    })
    stage1_debug[TEMP_OCCUPANCY_COVER_DEBUG_KEY] = debug


def _iter_route_windows(route_indices, samples):
    route_window_ids = [_conflict_window_identity(samples[int(idx)]) for idx in route_indices]
    pos = 0
    while pos < len(route_indices):
        window_id = route_window_ids[int(pos)]
        if window_id is None:
            pos += 1
            continue
        start_pos = int(pos)
        while pos + 1 < len(route_indices) and route_window_ids[int(pos) + 1] == window_id:
            pos += 1
        end_pos = int(pos)
        yield start_pos, end_pos, window_id
        pos += 1


def _first_qualifying_zero_run(bins, valid, required_run_len=1):
    bins_arr = np.asarray(bins, dtype=np.float32).reshape(-1)
    valid_arr = np.asarray(valid, dtype=np.float32).reshape(-1)
    total = min(int(bins_arr.shape[0]), int(valid_arr.shape[0]))
    required_run_len = int(max(int(required_run_len), 1))
    max_run_len = 0
    pos = 0
    while pos < total and float(valid_arr[pos]) > 0.5:
        if float(bins_arr[pos]) > 0.5:
            pos += 1
            continue
        run_start = int(pos)
        run_len = 0
        while pos < total and float(valid_arr[pos]) > 0.5 and float(bins_arr[pos]) <= 0.5:
            run_len += 1
            pos += 1
        max_run_len = max(int(max_run_len), int(run_len))
        if int(run_len) >= int(required_run_len):
            return int(run_start), int(run_len), int(max_run_len), True
    return int(total), 0, int(max_run_len), False


def _project_point_to_polyline_progress(polyline_xyz, point_xyz):
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


def _sample_scene_front_s(sample: dict) -> float:
    junction_thresholds = _sample_stage1_block(sample, "junction_thresholds")
    ego_front_s = float(junction_thresholds.get("ego_front_s_m", np.nan))
    if np.isfinite(ego_front_s):
        return float(ego_front_s)
    merge_motion = _sample_stage1_block(sample, "merge_motion")
    front_s = float(merge_motion.get("scene_route_front_s_m", np.nan))
    return float(front_s)


def _sample_area_start_s_m(sample: dict) -> float:
    conflict_area = _sample_stage1_block(sample, "conflict_area")
    area_start_s_m = float(conflict_area.get("area_start_s_m", np.nan))
    return float(area_start_s_m) if np.isfinite(area_start_s_m) else np.nan


def _sample_conflict_distance_m(sample: dict, family: str) -> float:
    conflict_area = _sample_stage1_block(sample, "conflict_area")
    area_segment_xyz = np.asarray(conflict_area.get("area_segment_world_xyz", []), dtype=np.float32)
    collision_point_xyz = np.asarray(conflict_area.get("collision_point_world_xyz", []), dtype=np.float32).reshape(-1)

    if family in {"merge", "junction"}:
        front_s = _sample_scene_front_s(sample)
        area_start_s = float(conflict_area.get("area_start_s_m", np.nan))
        if np.isfinite(front_s) and np.isfinite(area_start_s):
            collision_progress = _project_point_to_polyline_progress(area_segment_xyz, collision_point_xyz)
            if np.isfinite(collision_progress):
                return float(area_start_s + float(collision_progress) - float(front_s))
            return float(area_start_s - float(front_s))
        threshold_block = _sample_stage1_block(sample, f"{family}_thresholds")
        d_ego_m = float(threshold_block.get("d_ego_m", np.nan))
        if np.isfinite(d_ego_m):
            return float(d_ego_m)
        return np.nan

    if family == "borrow":
        borrow_motion = _sample_stage1_block(sample, "borrow_motion")
        borrow_start_distance_m = float(borrow_motion.get("borrow_start_distance_m", np.nan))
        conflict_start_progress_m = float(conflict_area.get("borrow_conflict_start_progress_m", np.nan))
        if np.isfinite(borrow_start_distance_m) and np.isfinite(conflict_start_progress_m):
            base_distance_m = float(borrow_start_distance_m - conflict_start_progress_m)
            collision_progress = _project_point_to_polyline_progress(area_segment_xyz, collision_point_xyz)
            if np.isfinite(collision_progress):
                return float(base_distance_m + float(collision_progress))
            return float(base_distance_m)
        threshold_block = _sample_stage1_block(sample, "borrow_thresholds")
        d_ego_m = float(threshold_block.get("d_ego_m", np.nan))
        if np.isfinite(d_ego_m):
            return float(d_ego_m)
    return np.nan


def _distance_adjustment_bins(current_distance_m: float, reference_distance_m: float) -> int:
    if not (np.isfinite(current_distance_m) and np.isfinite(reference_distance_m)):
        return 0
    extra_distance_m = float(current_distance_m) - float(reference_distance_m)
    if extra_distance_m > float(DISTANCE_EXTRA_BIN_THRESH_2_M):
        return 2
    if extra_distance_m > float(DISTANCE_EXTRA_BIN_THRESH_1_M):
        return 1
    return 0


def _merge_issue_reason(*parts) -> str:
    merged = []
    for part in parts:
        text = str(part or "none")
        if text in {"", "none"}:
            continue
        if text not in merged:
            merged.append(text)
    return "+".join(merged) if merged else "none"


def _sample_distance_to_area_start_m(
    sample: dict,
    family: str,
    area_start_s_m: float,
    reference_mode: bool = False,
) -> tuple[float, str]:
    front_s_m = _sample_scene_front_s(sample)
    if np.isfinite(front_s_m) and np.isfinite(area_start_s_m):
        return float(max(float(area_start_s_m) - float(front_s_m), 0.0)), "none"

    fallback_distance_m = _sample_conflict_distance_m(sample, family)
    if np.isfinite(fallback_distance_m):
        if reference_mode:
            return float(fallback_distance_m), "missing_go_frame_area_start_distance_fallback_conflict_distance"
        return float(fallback_distance_m), "missing_area_start_distance_fallback_conflict_distance"
    if reference_mode:
        return np.nan, "missing_go_frame_area_start_distance_no_fallback"
    return np.nan, "missing_area_start_distance_no_fallback"


def _window_go_frame(route_indices, samples, start_pos: int, end_pos: int) -> tuple[int | None, int]:
    go_frame = int(samples[int(route_indices[int(start_pos)])].get("conflict_go_frame", -1))
    if go_frame < 0:
        return None, -1
    for route_pos in range(int(start_pos), int(end_pos) + 1):
        if _frame_id_from_sample(samples[int(route_indices[int(route_pos)])]) == int(go_frame):
            return int(route_pos), int(go_frame)
    return None, int(go_frame)


def _window_entry_frame(route_indices, samples, start_pos: int, end_pos: int) -> tuple[int | None, int]:
    conflict_phase = _sample_stage1_block(samples[int(route_indices[int(start_pos)])], "conflict_phase")
    entry_frame = int(conflict_phase.get("entry_frame", -1))
    if entry_frame < 0:
        return None, -1
    for route_pos in range(int(start_pos), int(end_pos) + 1):
        if _frame_id_from_sample(samples[int(route_indices[int(route_pos)])]) == int(entry_frame):
            return int(route_pos), int(entry_frame)
    return None, int(entry_frame)


def _window_area_start_pos(route_indices, samples, start_pos: int, end_pos: int) -> tuple[int | None, int, str]:
    anchor_sample = samples[int(route_indices[int(start_pos)])]
    area_start_s_m = _sample_area_start_s_m(anchor_sample)
    if np.isfinite(area_start_s_m):
        for route_pos in range(int(start_pos), int(end_pos) + 1):
            sample = samples[int(route_indices[int(route_pos)])]
            front_s_m = _sample_scene_front_s(sample)
            if np.isfinite(front_s_m) and float(front_s_m) >= float(area_start_s_m):
                return int(route_pos), int(_frame_id_from_sample(sample)), "area_start"
        return None, -1, "area_start_not_reached"

    entry_route_pos, entry_frame = _window_entry_frame(route_indices, samples, start_pos, end_pos)
    if entry_route_pos is not None:
        return int(entry_route_pos), int(entry_frame), "area_start_missing_fallback_entry_frame"
    return None, -1, "area_start_missing"


def _window_goable_segments(goable_flags):
    segments = []
    segment_start = None
    for idx, goable in enumerate(goable_flags):
        if goable and segment_start is None:
            segment_start = int(idx)
            continue
        if not goable and segment_start is not None:
            segments.append((int(segment_start), int(idx - 1)))
            segment_start = None
    if segment_start is not None:
        segments.append((int(segment_start), int(len(goable_flags) - 1)))
    return segments


def _segment_probability(offset_in_segment: int, segment_len: int) -> float:
    if int(segment_len) <= 1:
        return 1.0
    return float(max(1.0 - (float(offset_in_segment) / float(segment_len)), 1.0 / float(segment_len)))


def _route_collision_info(sample: dict, image_data_root: str, cache: dict) -> dict:
    base_dir = str(_base_dir_from_sample(sample) or "")
    cache_key = (str(image_data_root or ""), base_dir)
    if cache_key in cache:
        return dict(cache[cache_key])

    info = {
        "known": False,
        "non_collision": False,
        "collision_route": False,
        "results_path": "",
        "issue_reason": "none",
    }
    if not image_data_root:
        info["issue_reason"] = "missing_image_data_root"
        cache[cache_key] = dict(info)
        return info
    if not base_dir:
        info["issue_reason"] = "missing_base_dir"
        cache[cache_key] = dict(info)
        return info

    results_path = os.path.join(str(image_data_root), base_dir, "results.json.gz")
    payload = _load_json_gz_if_exists(results_path)
    info["results_path"] = str(results_path)
    if not isinstance(payload, dict):
        info["issue_reason"] = "missing_route_results"
        cache[cache_key] = dict(info)
        return info

    infractions = payload.get("infractions", {})
    collision_route = False
    for key in ("collisions_vehicle", "collisions_pedestrian", "collisions_layout"):
        value = infractions.get(key) if isinstance(infractions, dict) else None
        if _has_nonempty_infraction(value):
            collision_route = True
            break

    info.update({
        "known": True,
        "non_collision": not bool(collision_route),
        "collision_route": bool(collision_route),
        "issue_reason": "none",
    })
    cache[cache_key] = dict(info)
    return info


def _fill_bins_for_route(samples, route_indices, can_write):
    assigned = 0
    family_counts = {"borrow": 0, "merge": 0, "junction": 0}
    route_has_active = False
    for pos, sample_idx in enumerate(route_indices):
        sample_idx = int(sample_idx)
        if not can_write[sample_idx]:
            continue
        anchor_sample = samples[sample_idx]
        anchor_window = _conflict_window_identity(anchor_sample)
        if anchor_window is None:
            continue

        route_has_active = True
        family = str(anchor_window[0])
        bins = np.zeros((TEMP_OCCUPANCY_COVER_TOTAL_SLOTS,), dtype=np.float32)
        valid = np.zeros((TEMP_OCCUPANCY_COVER_TOTAL_SLOTS,), dtype=np.float32)
        for offset in range(int(TEMP_OCCUPANCY_COVER_TOTAL_SLOTS)):
            target_pos = int(pos) + int(offset)
            if target_pos >= len(route_indices):
                break
            target_sample = samples[int(route_indices[int(target_pos)])]
            if _conflict_window_identity(target_sample) != anchor_window:
                break
            valid[offset] = 1.0
            bins[offset] = 1.0 if _current_cover_matches_family(target_sample, family) else 0.0

        _set_temporary_occupancy_bins_annotation(anchor_sample, family=family, bins=bins, valid=valid)
        assigned += 1
        family_counts[family] += 1
    return route_has_active, int(assigned), family_counts


def _annotate_window_go_opportunity(
    samples,
    route_indices,
    can_write,
    start_pos: int,
    end_pos: int,
    family: str,
    non_collision_good_route: bool,
):
    go_route_pos, reference_frame = _window_go_frame(route_indices, samples, start_pos, end_pos)
    if go_route_pos is None:
        return 0, 0, 0
    area_start_route_pos, area_start_frame, area_start_source = _window_area_start_pos(
        route_indices,
        samples,
        start_pos,
        end_pos,
    )
    if area_start_route_pos is None:
        area_start_route_pos = int(end_pos)

    ref_sample = samples[int(route_indices[int(go_route_pos)])]
    ref_bins = ref_sample.get(TEMP_OCCUPANCY_COVER_KEY, np.zeros((TEMP_OCCUPANCY_COVER_TOTAL_SLOTS,), dtype=np.float32))
    ref_valid = ref_sample.get(TEMP_OCCUPANCY_COVER_VALID_KEY, np.zeros((TEMP_OCCUPANCY_COVER_TOTAL_SLOTS,), dtype=np.float32))
    raw_reference_pass_time_bins = int(max(int(end_pos) - int(go_route_pos) + 1, 1))
    reference_run_start_geom = int(max(int(area_start_route_pos) - int(go_route_pos), 0))
    reference_run_len = int(max(int(raw_reference_pass_time_bins) - int(reference_run_start_geom), 1))
    ref_run_start_expert, _, ref_max_zero_run_len, ref_has_run = _first_qualifying_zero_run(
        ref_bins,
        ref_valid,
        required_run_len=reference_run_len,
    )

    issue_reason = "none"
    if not ref_has_run:
        ref_run_start_expert = int(reference_run_start_geom)
        issue_reason = "reference_no_qualifying_run_fallback_geom"
    run_start_diff = abs(int(ref_run_start_expert) - int(reference_run_start_geom))
    if int(run_start_diff) <= 1:
        reference_run_start_final = int(ref_run_start_expert)
    else:
        reference_run_start_final = int(round((float(ref_run_start_expert) + float(reference_run_start_geom)) * 0.5))
    reference_run_start_final = int(min(int(reference_run_start_final), int(reference_run_start_geom)))
    reference_run_start_final = int(max(int(reference_run_start_final), 0))

    area_start_s_m = _sample_area_start_s_m(ref_sample)
    reference_distance_to_area_start_m, reference_distance_issue_reason = _sample_distance_to_area_start_m(
        ref_sample,
        family,
        area_start_s_m=area_start_s_m,
        reference_mode=True,
    )
    area_start_issue_reason = "none" if str(area_start_source) == "area_start" else str(area_start_source)
    base_issue_reason = _merge_issue_reason(issue_reason, area_start_issue_reason, reference_distance_issue_reason)

    if int(reference_run_len) <= 0:
        base_issue_reason = _merge_issue_reason(base_issue_reason, "reference_run_len_invalid")
        for route_pos in range(int(start_pos), int(end_pos) + 1):
            sample_idx = int(route_indices[int(route_pos)])
            if not can_write[sample_idx]:
                continue
            _set_go_opportunity_annotation(
                samples[sample_idx],
                go_prob=0.0,
                valid=0.0,
                cycle_id=-1,
                accepted_cycle=0.0,
                goable=0.0,
                go_prob_floor_clamped=0.0,
                reference_frame=reference_frame,
                reference_run_start_expert=ref_run_start_expert,
                reference_run_start_geom=reference_run_start_geom,
                reference_run_start_final=reference_run_start_final,
                reference_run_len=reference_run_len,
                raw_reference_pass_time_bins=raw_reference_pass_time_bins,
                adjusted_run_start_bins=-1,
                distance_to_area_start_m=np.nan,
                distance_adjustment_bins=-1,
                current_run_start=-1,
                current_run_len=-1,
                current_remaining_run_len=-1,
                area_start_frame=area_start_frame,
                area_start_source=area_start_source,
                issue_reason=base_issue_reason,
            )
        return 0, 0, 0

    goable_flags = []
    current_goable_by_pos = []
    adjusted_run_start_by_pos = []
    distance_to_area_start_by_pos = []
    distance_extra_bins_by_pos = []
    current_run_start_by_pos = []
    current_run_len_by_pos = []
    current_remaining_run_len_by_pos = []
    per_pos_issue_reason = []
    for route_pos in range(int(start_pos), int(end_pos) + 1):
        sample = samples[int(route_indices[int(route_pos)])]
        bins = sample.get(TEMP_OCCUPANCY_COVER_KEY, np.zeros((TEMP_OCCUPANCY_COVER_TOTAL_SLOTS,), dtype=np.float32))
        valid = sample.get(TEMP_OCCUPANCY_COVER_VALID_KEY, np.zeros((TEMP_OCCUPANCY_COVER_TOTAL_SLOTS,), dtype=np.float32))
        current_distance_m, distance_issue_reason = _sample_distance_to_area_start_m(
            sample,
            family,
            area_start_s_m=area_start_s_m,
            reference_mode=False,
        )
        distance_extra_bins = int(_distance_adjustment_bins(current_distance_m, reference_distance_to_area_start_m))
        adjusted_run_start_bins = int(reference_run_start_final + distance_extra_bins)
        current_run_start, current_run_len, _, has_run = _first_qualifying_zero_run(
            bins,
            valid,
            required_run_len=reference_run_len,
        )
        adjusted_run_start_by_pos.append(int(adjusted_run_start_bins))
        distance_to_area_start_by_pos.append(float(current_distance_m) if np.isfinite(current_distance_m) else np.nan)
        distance_extra_bins_by_pos.append(int(distance_extra_bins))
        current_run_start_by_pos.append(int(current_run_start))
        current_run_len_by_pos.append(int(current_run_len))
        current_remaining_run_len = int(max(int(current_run_start) + int(current_run_len) - int(adjusted_run_start_bins), 0))
        current_remaining_run_len_by_pos.append(int(current_remaining_run_len))
        per_pos_issue_reason.append(_merge_issue_reason(base_issue_reason, distance_issue_reason))
        is_goable = bool(
            float(np.asarray(valid, dtype=np.float32).reshape(-1)[0]) > 0.5 and
            bool(has_run) and
            int(current_run_start) <= int(adjusted_run_start_bins) and
            int(current_remaining_run_len) >= int(reference_run_len)
        )
        current_goable_by_pos.append(bool(is_goable))
        goable_flags.append(bool(is_goable) if int(route_pos) <= int(area_start_route_pos) else False)

    segments = _window_goable_segments(goable_flags)
    accepted_cycle_id = -1
    for cycle_id, (seg_start_rel, seg_end_rel) in enumerate(segments):
        if int(seg_start_rel) <= int(go_route_pos - start_pos) <= int(seg_end_rel):
            accepted_cycle_id = int(cycle_id)
            break

    clamped_frame_count = 0
    for rel_pos, route_pos in enumerate(range(int(start_pos), int(end_pos) + 1)):
        sample_idx = int(route_indices[int(route_pos)])
        if not can_write[sample_idx]:
            continue

        if int(route_pos) > int(go_route_pos):
            _set_go_opportunity_annotation(
                samples[sample_idx],
                go_prob=1.0,
                valid=1.0,
                cycle_id=int(accepted_cycle_id),
                accepted_cycle=1.0 if int(accepted_cycle_id) >= 0 else 0.0,
                goable=1.0 if current_goable_by_pos[int(rel_pos)] else 0.0,
                go_prob_floor_clamped=0.0,
                reference_frame=reference_frame,
                reference_run_start_expert=ref_run_start_expert,
                reference_run_start_geom=reference_run_start_geom,
                reference_run_start_final=reference_run_start_final,
                reference_run_len=reference_run_len,
                raw_reference_pass_time_bins=raw_reference_pass_time_bins,
                adjusted_run_start_bins=adjusted_run_start_by_pos[int(rel_pos)],
                distance_to_area_start_m=distance_to_area_start_by_pos[int(rel_pos)],
                distance_adjustment_bins=distance_extra_bins_by_pos[int(rel_pos)],
                current_run_start=current_run_start_by_pos[int(rel_pos)],
                current_run_len=current_run_len_by_pos[int(rel_pos)],
                current_remaining_run_len=current_remaining_run_len_by_pos[int(rel_pos)],
                area_start_frame=area_start_frame,
                area_start_source=area_start_source,
                issue_reason=per_pos_issue_reason[int(rel_pos)],
            )
            continue

        cycle_id = -1
        accepted_cycle = 0.0
        go_prob = 0.0
        goable = 1.0 if current_goable_by_pos[int(rel_pos)] else 0.0
        go_prob_floor_clamped = 0.0
        for current_cycle_id, (seg_start_rel, seg_end_rel) in enumerate(segments):
            if int(seg_start_rel) <= int(rel_pos) <= int(seg_end_rel):
                cycle_id = int(current_cycle_id)
                accepted_cycle = 1.0 if int(current_cycle_id) == int(accepted_cycle_id) else 0.0
                seg_len = int(seg_end_rel) - int(seg_start_rel) + 1
                go_prob = _segment_probability(int(rel_pos) - int(seg_start_rel), seg_len)
                break

        if (
            bool(non_collision_good_route)
            and int(samples[sample_idx].get("conflict_decision_phase", 0)) == int(CONFLICT_DECISION_GO_CODE)
            and float(go_prob) < float(GO_PROB_GO_PHASE_FLOOR_NON_COLLISION)
        ):
            go_prob = float(GO_PROB_GO_PHASE_FLOOR_NON_COLLISION)
            go_prob_floor_clamped = 1.0
            clamped_frame_count += 1

        _set_go_opportunity_annotation(
            samples[sample_idx],
            go_prob=go_prob,
            valid=1.0,
            cycle_id=cycle_id,
            accepted_cycle=accepted_cycle,
            goable=goable,
            go_prob_floor_clamped=go_prob_floor_clamped,
            reference_frame=reference_frame,
            reference_run_start_expert=ref_run_start_expert,
            reference_run_start_geom=reference_run_start_geom,
            reference_run_start_final=reference_run_start_final,
            reference_run_len=reference_run_len,
            raw_reference_pass_time_bins=raw_reference_pass_time_bins,
            adjusted_run_start_bins=adjusted_run_start_by_pos[int(rel_pos)],
            distance_to_area_start_m=distance_to_area_start_by_pos[int(rel_pos)],
            distance_adjustment_bins=distance_extra_bins_by_pos[int(rel_pos)],
            current_run_start=current_run_start_by_pos[int(rel_pos)],
            current_run_len=current_run_len_by_pos[int(rel_pos)],
            current_remaining_run_len=current_remaining_run_len_by_pos[int(rel_pos)],
            area_start_frame=area_start_frame,
            area_start_source=area_start_source,
            issue_reason=per_pos_issue_reason[int(rel_pos)],
        )
    return len(segments), int(accepted_cycle_id >= 0), int(clamped_frame_count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill cover-based temporary occupancy bins and go/yld soft labels on an existing stage1 merged samples_packed.pkl."
    )
    parser.add_argument("--input_path", required=True, help="Existing merged samples_packed.pkl")
    parser.add_argument("--output_path", required=True, help="Output pickle path")
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Overwrite existing temporary occupancy values instead of only filling missing ones.",
    )
    args = parser.parse_args()

    with open(args.input_path, "rb") as f:
        samples = pickle.load(f)

    total_samples = 0
    filled_defaults = 0
    skipped_existing = 0
    assigned_active = 0
    active_route_count = 0
    active_family_counts = {"borrow": 0, "merge": 0, "junction": 0}
    window_count = 0
    cycle_count = 0
    accepted_cycle_count = 0
    non_collision_route_count = 0
    collision_route_count = 0
    go_phase_floor_clamp_frames = 0
    go_phase_floor_clamp_windows = 0
    image_data_root = _infer_image_data_root_from_packed_path(args.input_path)
    route_results_cache = {}

    can_write = []
    for sample in samples:
        total_samples += 1
        write_this_sample = _should_write_sample(sample, overwrite_existing=bool(args.overwrite_existing))
        can_write.append(write_this_sample)
        if write_this_sample:
            _set_temporary_occupancy_defaults(sample)
            filled_defaults += 1
        else:
            skipped_existing += 1

    for _, route_indices in _group_indices_by_route(samples):
        route_collision_info = (
            _route_collision_info(samples[int(route_indices[0])], image_data_root=image_data_root, cache=route_results_cache)
            if route_indices else
            {"known": False, "non_collision": False, "collision_route": False}
        )
        non_collision_good_route = bool(
            route_collision_info.get("known", False) and
            route_collision_info.get("non_collision", False)
        )
        if bool(route_collision_info.get("known", False)):
            if bool(route_collision_info.get("collision_route", False)):
                collision_route_count += 1
            else:
                non_collision_route_count += 1
        route_has_active, route_assigned, route_family_counts = _fill_bins_for_route(samples, route_indices, can_write)
        assigned_active += int(route_assigned)
        for family_name, family_count in route_family_counts.items():
            active_family_counts[str(family_name)] += int(family_count)
        if route_has_active:
            active_route_count += 1

        for start_pos, end_pos, window_id in _iter_route_windows(route_indices, samples):
            family = str(window_id[0])
            num_cycles, has_accepted_cycle, clamped_frame_count = _annotate_window_go_opportunity(
                samples,
                route_indices,
                can_write,
                start_pos=start_pos,
                end_pos=end_pos,
                family=family,
                non_collision_good_route=non_collision_good_route,
            )
            window_count += 1
            cycle_count += int(num_cycles)
            accepted_cycle_count += int(has_accepted_cycle)
            go_phase_floor_clamp_frames += int(clamped_frame_count)
            if int(clamped_frame_count) > 0:
                go_phase_floor_clamp_windows += 1

    _atomic_pickle_dump(samples, args.output_path)
    print(
        "postprocess temporary occupancy cover done: "
        f"samples={total_samples} "
        f"active_routes={active_route_count} "
        f"filled_defaults={filled_defaults} "
        f"assigned_active={assigned_active} "
        f"borrow={active_family_counts['borrow']} "
        f"merge={active_family_counts['merge']} "
        f"junction={active_family_counts['junction']} "
        f"windows={window_count} "
        f"cycles={cycle_count} "
        f"accepted_cycles={accepted_cycle_count} "
        f"non_collision_routes={non_collision_route_count} "
        f"collision_routes={collision_route_count} "
        f"go_phase_floor_clamp_frames={go_phase_floor_clamp_frames} "
        f"go_phase_floor_clamp_windows={go_phase_floor_clamp_windows} "
        f"skipped_existing={skipped_existing} "
        f"key={TEMP_OCCUPANCY_COVER_KEY} "
        f"valid_key={TEMP_OCCUPANCY_COVER_VALID_KEY} "
        f"go_key={GO_OPPORTUNITY_PROB_KEY} "
        f"yld_key={YLD_PRESSURE_PROB_KEY} "
        f"output={args.output_path}"
    )


if __name__ == "__main__":
    main()
