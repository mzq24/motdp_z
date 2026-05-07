#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
from collections import Counter

import numpy as np


DEBUG_KEY = "boundary_speed_consistency"

CONSISTENCY_VALID_KEY = "boundary_speed_consistency_valid"
CONSISTENCY_ISSUE_KEY = "boundary_speed_consistency_issue"
CONSISTENCY_ISSUE_FLAG_KEY = "boundary_speed_consistency_issue_flag"
CONSISTENCY_REQUIRED_ACTION_KEY = "boundary_speed_consistency_required_action"
CONSISTENCY_SPEED_DELTA_KEY = "boundary_speed_consistency_speed_delta_mps"

PHASE_NONE = 0
PHASE_YLD = 1
PHASE_GO = 2

ACTION_NONE = 0
ACTION_DECEL = 1
ACTION_ACCEL = 2
ACTION_EITHER = 3

ACTION_NAMES = {
    ACTION_NONE: "none",
    ACTION_DECEL: "decel",
    ACTION_ACCEL: "accel",
    ACTION_EITHER: "either",
}

CONFLICTING_BOUND_BEHAVIOR_NONE = 0
CONFLICTING_BOUND_BEHAVIOR_UPPER = 1
CONFLICTING_BOUND_BEHAVIOR_LOWER = 2
CONFLICTING_BOUND_BEHAVIOR_BOTH = 3

CONFLICTING_BOUND_BEHAVIOR_NAMES = {
    CONFLICTING_BOUND_BEHAVIOR_NONE: "none",
    CONFLICTING_BOUND_BEHAVIOR_UPPER: "upper_decel",
    CONFLICTING_BOUND_BEHAVIOR_LOWER: "lower_accel",
    CONFLICTING_BOUND_BEHAVIOR_BOTH: "both",
}

ISSUE_NONE = 0
ISSUE_INACTIVE_OR_UNSUPPORTED = 1
ISSUE_COLLISION_ROUTE_SKIPPED = 2
ISSUE_MISSING_NEXT_SPEED = 3
ISSUE_NO_APPLICABLE_CONSTRAINT = 4
ISSUE_CHASE_OVER_MAX_NO_DECEL = 5
ISSUE_YLD_OVER_MAX_NO_DECEL = 6
ISSUE_GO_UNDER_MIN_NO_ACCEL = 7
ISSUE_CONFLICTING_BOUNDS_NO_ACTION = 8
ISSUE_JUNCTION_GO_OVER_MAX_NO_DECEL = 9
ISSUE_TIGHT_BOUNDS_FILTERED = 10

ISSUE_NAMES = {
    ISSUE_NONE: "none",
    ISSUE_INACTIVE_OR_UNSUPPORTED: "inactive_or_unsupported",
    ISSUE_COLLISION_ROUTE_SKIPPED: "collision_route_skipped",
    ISSUE_MISSING_NEXT_SPEED: "missing_next_speed",
    ISSUE_NO_APPLICABLE_CONSTRAINT: "no_applicable_constraint",
    ISSUE_CHASE_OVER_MAX_NO_DECEL: "chase_over_max_no_decel",
    ISSUE_YLD_OVER_MAX_NO_DECEL: "yld_over_max_no_decel",
    ISSUE_GO_UNDER_MIN_NO_ACCEL: "go_under_min_no_accel",
    ISSUE_CONFLICTING_BOUNDS_NO_ACTION: "conflicting_bounds_no_action",
    ISSUE_JUNCTION_GO_OVER_MAX_NO_DECEL: "junction_go_over_max_no_decel",
    ISSUE_TIGHT_BOUNDS_FILTERED: "tight_bounds_filtered",
}

FAMILY_CODE_TO_NAME = {
    0: "none",
    1: "borrow",
    2: "merge",
    3: "junction",
}
ACTIVE_FAMILIES = {"borrow", "merge", "junction"}
COLLISION_KEYS = ("collisions_vehicle", "collisions_pedestrian", "collisions_layout")


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


def _speed_from_sample(sample: dict) -> float:
    for key in ("speed", "speed_mps"):
        value = _finite_float(sample.get(key, np.nan), default=np.nan)
        if np.isfinite(value):
            return float(value)
    speed_hist = np.asarray(sample.get("speed_hist", []), dtype=np.float32).reshape(-1)
    if speed_hist.size > 0 and np.isfinite(float(speed_hist[-1])):
        return float(speed_hist[-1])
    return np.nan


def _hist_last(sample: dict, key: str) -> float:
    arr = np.asarray(sample.get(key, []), dtype=np.float32).reshape(-1)
    if arr.size > 0 and np.isfinite(float(arr[-1])):
        return float(arr[-1])
    return np.nan


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


def _load_json_gz_if_exists(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _route_name_from_route_key(route_key: str) -> str:
    route_key = str(route_key or "").rstrip("/")
    if not route_key:
        return ""
    return os.path.basename(route_key)


def _load_collision_route_overrides(paths: list[str] | None) -> set[str]:
    routes: set[str] = set()
    for path in paths or []:
        if not path:
            continue
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                # Support either "route_name" or "scene route_name" lists.
                route_name = parts[-1].strip()
                if route_name:
                    routes.add(route_name)
    return routes


def _route_collision_info(route_key: str, image_data_root: str, cache: dict) -> dict:
    cache_key = (str(image_data_root or ""), str(route_key or ""))
    if cache_key in cache:
        return dict(cache[cache_key])
    info = {
        "known": False,
        "collision_route": False,
        "non_collision": False,
        "results_path": "",
        "collision_infractions": [],
        "issue_reason": "none",
    }
    if not image_data_root:
        info["issue_reason"] = "missing_image_data_root"
        cache[cache_key] = dict(info)
        return info
    if not route_key:
        info["issue_reason"] = "missing_route_key"
        cache[cache_key] = dict(info)
        return info

    results_path = os.path.join(str(image_data_root), str(route_key), "results.json.gz")
    info["results_path"] = str(results_path)
    payload = _load_json_gz_if_exists(results_path)
    if not isinstance(payload, dict):
        info["issue_reason"] = "missing_route_results"
        cache[cache_key] = dict(info)
        return info

    infractions = payload.get("infractions", {})
    collision_infractions = []
    for key in COLLISION_KEYS:
        value = infractions.get(key) if isinstance(infractions, dict) else None
        if _has_nonempty_infraction(value):
            collision_infractions.append(str(key))
    info.update({
        "known": True,
        "collision_route": bool(collision_infractions),
        "non_collision": not bool(collision_infractions),
        "collision_infractions": collision_infractions,
        "issue_reason": "none",
    })
    cache[cache_key] = dict(info)
    return info


def _apply_collision_override(route_key: str, info: dict, override_routes: set[str]) -> dict:
    route_name = _route_name_from_route_key(route_key)
    if route_name and route_name in override_routes:
        updated = dict(info)
        updated.update({
            "known": True,
            "collision_route": True,
            "non_collision": False,
            "collision_infractions": list(updated.get("collision_infractions", [])) + ["collision_route_override"],
            "issue_reason": "collision_route_override",
        })
        return updated
    return info



def _default_values() -> dict:
    return {
        CONSISTENCY_VALID_KEY: np.float32(0.0),
        CONSISTENCY_ISSUE_KEY: np.int64(ISSUE_NONE),
        CONSISTENCY_ISSUE_FLAG_KEY: np.float32(0.0),
        CONSISTENCY_REQUIRED_ACTION_KEY: np.int64(ACTION_NONE),
        CONSISTENCY_SPEED_DELTA_KEY: np.float32(np.nan),
    }


def _default_debug(issue_name: str = "not_computed") -> dict:
    return {
        "active": 0.0,
        "non_collision_route": 0.0,
        "family": "none",
        "phase": "none",
        "valid": 0.0,
        "issue": int(ISSUE_NONE),
        "issue_name": str(issue_name),
        "issue_cause_name": "none",
        "issue_flag": 0.0,
        "required_action": int(ACTION_NONE),
        "required_action_name": ACTION_NAMES[ACTION_NONE],
        "current_speed_mps": np.nan,
        "next_speed_mps": np.nan,
        "speed_delta_mps": np.nan,
        "decel_observed": 0.0,
        "accel_observed": 0.0,
        "throttle": np.nan,
        "brake": np.nan,
        "chase_speed_max_mps": np.nan,
        "chase_speed_max_valid": 0.0,
        "junction_go_speed_max_mps": np.nan,
        "junction_go_speed_max_valid": 0.0,
        "effective_speed_max_mps": np.nan,
        "effective_speed_max_valid": 0.0,
        "effective_speed_max_source": "none",
        "yld_max_speed_mps": np.nan,
        "yld_max_valid": 0.0,
        "go_min_speed_mps": np.nan,
        "go_min_valid": 0.0,
        "merge_vbmin_mps": np.nan,
        "merge_vbmin_valid": 0.0,
        "effective_go_min_mps": np.nan,
        "both_bounds_active": 0.0,
        "both_bounds_behavior": int(CONFLICTING_BOUND_BEHAVIOR_NONE),
        "both_bounds_behavior_name": CONFLICTING_BOUND_BEHAVIOR_NAMES[CONFLICTING_BOUND_BEHAVIOR_NONE],
        "boundary_scalar_loss_valid": 0.0,
        "bound_gap_mps": np.nan,
        "tight_bound_gap_mps": np.nan,
        "tight_bounds_filtered": 0.0,
        "bound_violation_tolerance_mps": np.nan,
        "need_decel": 0.0,
        "need_accel": 0.0,
        "conflicting_bounds": 0.0,
        "conflicting_bound_behavior": int(CONFLICTING_BOUND_BEHAVIOR_NONE),
        "conflicting_bound_behavior_name": CONFLICTING_BOUND_BEHAVIOR_NAMES[CONFLICTING_BOUND_BEHAVIOR_NONE],
        "margin_mps": np.nan,
        "route_results_issue": "none",
    }


def _has_existing_fields(sample: dict) -> bool:
    stage1_debug = sample.get("stage1_speed_debug")
    debug = stage1_debug.get(DEBUG_KEY) if isinstance(stage1_debug, dict) else None
    return bool(
        CONSISTENCY_VALID_KEY in sample and
        CONSISTENCY_ISSUE_KEY in sample and
        CONSISTENCY_ISSUE_FLAG_KEY in sample and
        CONSISTENCY_REQUIRED_ACTION_KEY in sample and
        CONSISTENCY_SPEED_DELTA_KEY in sample and
        isinstance(debug, dict)
    )


def _family_boundary_fields(family: str, phase: int) -> tuple[str, str]:
    if family == "merge":
        return (
            "merge_yld_max_speed" if phase == PHASE_YLD else "merge_go_min_speed",
            "merge_yld_max_speed_valid" if phase == PHASE_YLD else "merge_go_min_speed_valid",
        )
    if family == "borrow":
        return (
            "borrow_yld_max_speed" if phase == PHASE_YLD else "borrow_go_min_speed",
            "borrow_yld_max_speed_valid" if phase == PHASE_YLD else "borrow_go_min_speed_valid",
        )
    if family == "junction":
        return (
            "junction_yld_max_speed" if phase == PHASE_YLD else "junction_go_min_speed",
            "junction_yld_max_speed_valid" if phase == PHASE_YLD else "junction_go_min_speed_valid",
        )
    return "", ""


def _family_yld_boundary_fields(family: str) -> tuple[str, str]:
    if family == "merge":
        return "merge_yld_max_speed", "merge_yld_max_speed_valid"
    if family == "borrow":
        return "borrow_yld_max_speed", "borrow_yld_max_speed_valid"
    if family == "junction":
        return "junction_yld_max_speed", "junction_yld_max_speed_valid"
    return "", ""


def _speed_action_flags(speed_delta_mps: float, args) -> tuple[bool, bool]:
    if not np.isfinite(speed_delta_mps):
        return False, False
    decel = bool(float(speed_delta_mps) <= -float(args.speed_delta_eps_mps))
    accel = bool(float(speed_delta_mps) >= float(args.speed_delta_eps_mps))
    return decel, accel


def _conflicting_bound_behavior(decel_observed: bool, accel_observed: bool) -> int:
    if decel_observed and accel_observed:
        return int(CONFLICTING_BOUND_BEHAVIOR_BOTH)
    if decel_observed:
        return int(CONFLICTING_BOUND_BEHAVIOR_UPPER)
    if accel_observed:
        return int(CONFLICTING_BOUND_BEHAVIOR_LOWER)
    return int(CONFLICTING_BOUND_BEHAVIOR_NONE)


def _set_issue(
    values: dict,
    debug: dict,
    issue: int,
    required_action: int = ACTION_NONE,
    valid: bool = False,
    flag: bool = False,
) -> None:
    values[CONSISTENCY_VALID_KEY] = np.float32(1.0 if valid else 0.0)
    values[CONSISTENCY_ISSUE_KEY] = np.int64(int(issue))
    values[CONSISTENCY_ISSUE_FLAG_KEY] = np.float32(1.0 if flag else 0.0)
    values[CONSISTENCY_REQUIRED_ACTION_KEY] = np.int64(int(required_action))
    debug.update({
        "valid": float(1.0 if valid else 0.0),
        "issue": int(issue),
        "issue_name": ISSUE_NAMES.get(int(issue), str(issue)),
        "issue_flag": float(1.0 if flag else 0.0),
        "required_action": int(required_action),
        "required_action_name": ACTION_NAMES.get(int(required_action), str(required_action)),
    })


def _issue_cause_name(sample: dict, debug: dict, args) -> str:
    if float(debug.get("issue_flag", 0.0)) <= 0.5:
        return "none"

    issue_name = str(debug.get("issue_name", "none"))
    family = str(debug.get("family", "none"))
    phase = str(debug.get("phase", "none"))
    phase_ref_role = _to_int(sample.get("conflict_phase_ref_role", -1), default=-1)
    boundary_ref_role = _to_int(sample.get("conflict_phase_boundary_ref_role", -1), default=-1)
    actor_match = float(sample.get("conflict_phase_boundary_actor_match", 0.0)) > 0.5
    scalar_loss_valid = float(sample.get("conflict_phase_boundary_scalar_loss_valid", 0.0)) > 0.5
    current_speed = _finite_float(debug.get("current_speed_mps", np.nan), default=np.nan)
    next_speed = _finite_float(debug.get("next_speed_mps", np.nan), default=np.nan)
    speed_delta = _finite_float(debug.get("speed_delta_mps", np.nan), default=np.nan)
    yld_max = _finite_float(debug.get("yld_max_speed_mps", np.nan), default=np.nan)
    go_min = _finite_float(debug.get("go_min_speed_mps", np.nan), default=np.nan)
    chase_max = _finite_float(debug.get("chase_speed_max_mps", np.nan), default=np.nan)

    if (
        issue_name == "yld_over_max_no_decel"
        and family == "merge"
        and phase == "yld"
        and int(phase_ref_role) == 1
        and int(boundary_ref_role) == 1
        and actor_match
        and scalar_loss_valid
        and np.isfinite(chase_max)
    ):
        if np.isfinite(current_speed) and abs(float(current_speed) - float(chase_max)) <= float(args.phase_cause_speed_near_mps):
            return "candidate_chase_limited_go_cur_near_chase"
        if np.isfinite(next_speed) and abs(float(next_speed) - float(chase_max)) <= float(args.phase_cause_speed_near_mps):
            return "candidate_chase_limited_go_next_near_chase"
        if np.isfinite(speed_delta) and float(speed_delta) >= -float(args.speed_delta_eps_mps):
            return "candidate_chase_limited_go_non_decel"

    if (
        issue_name == "go_under_min_no_accel"
        and family == "merge"
        and phase == "go"
        and np.isfinite(go_min)
        and float(go_min) >= float(args.high_go_min_cause_mps)
    ):
        if np.isfinite(yld_max) and np.isfinite(current_speed):
            return "candidate_yield_after_actor_no_decel_needed"
        return "candidate_high_go_min_relation_review"

    if issue_name == "yld_over_max_no_decel" and family == "borrow" and phase == "yld":
        return "borrow_yld_over_max_check_surrounding_slowdown"

    return "normal"


def _compute_annotation(sample: dict, next_sample: dict | None, route_collision_info: dict, args) -> tuple[dict, dict]:
    values = _default_values()
    debug = _default_debug()
    family = _family_name(sample)
    phase = _to_int(sample.get("conflict_decision_phase", PHASE_NONE), default=PHASE_NONE)
    phase_name = {PHASE_NONE: "none", PHASE_YLD: "yld", PHASE_GO: "go"}.get(int(phase), str(phase))
    active = _is_active_conflict_sample(sample)
    non_collision = bool(route_collision_info.get("known", False) and route_collision_info.get("non_collision", False))
    current_speed = _speed_from_sample(sample)
    next_speed = _speed_from_sample(next_sample or {}) if next_sample is not None else np.nan
    speed_delta = float(next_speed - current_speed) if np.isfinite(current_speed) and np.isfinite(next_speed) else np.nan
    decel_observed, accel_observed = _speed_action_flags(speed_delta, args)
    throttle = _hist_last(sample, "throttle_hist")
    brake = _hist_last(sample, "brake_hist")

    values[CONSISTENCY_SPEED_DELTA_KEY] = np.float32(float(speed_delta) if np.isfinite(speed_delta) else np.nan)
    debug.update({
        "active": float(1.0 if active else 0.0),
        "non_collision_route": float(1.0 if non_collision else 0.0),
        "family": str(family),
        "phase": str(phase_name),
        "current_speed_mps": float(current_speed) if np.isfinite(current_speed) else np.nan,
        "next_speed_mps": float(next_speed) if np.isfinite(next_speed) else np.nan,
        "speed_delta_mps": float(speed_delta) if np.isfinite(speed_delta) else np.nan,
        "decel_observed": float(1.0 if decel_observed else 0.0),
        "accel_observed": float(1.0 if accel_observed else 0.0),
        "throttle": float(throttle) if np.isfinite(throttle) else np.nan,
        "brake": float(brake) if np.isfinite(brake) else np.nan,
        "margin_mps": float(args.speed_margin_mps),
        "route_results_issue": str(route_collision_info.get("issue_reason", "none")),
    })

    if not active or phase not in {PHASE_YLD, PHASE_GO}:
        _set_issue(values, debug, ISSUE_INACTIVE_OR_UNSUPPORTED, valid=False, flag=False)
        return values, debug
    if not non_collision and not bool(args.include_collision_routes):
        _set_issue(values, debug, ISSUE_COLLISION_ROUTE_SKIPPED, valid=False, flag=False)
        return values, debug
    if not (np.isfinite(current_speed) and np.isfinite(next_speed)):
        _set_issue(values, debug, ISSUE_MISSING_NEXT_SPEED, valid=False, flag=False)
        return values, debug

    margin = float(args.speed_margin_mps)
    violation_tol = max(float(args.speed_margin_mps), float(args.bound_violation_tolerance_mps))
    scalar_loss_valid = float(sample.get("conflict_phase_boundary_scalar_loss_valid", 1.0)) > 0.5
    debug.update({
        "boundary_scalar_loss_valid": float(1.0 if scalar_loss_valid else 0.0),
        "bound_violation_tolerance_mps": float(args.bound_violation_tolerance_mps),
    })

    chase_max = _finite_float(sample.get("chase_speed_max", np.nan), default=np.nan)
    # Junction conflict windows use yld/go boundary labels for cross-traffic
    # timing. Current-cover chase caps are not semantically reliable there.
    chase_valid = bool(family != "junction" and float(sample.get("chase_speed_max_valid", 0.0)) > 0.5)
    debug.update({
        "chase_speed_max_mps": float(chase_max) if chase_valid and np.isfinite(chase_max) else np.nan,
        "chase_speed_max_valid": float(1.0 if chase_valid else 0.0),
    })

    speed_field, valid_field = _family_boundary_fields(family, phase)
    boundary_speed = _finite_float(sample.get(speed_field, np.nan), default=np.nan) if speed_field else np.nan
    boundary_valid = bool(valid_field and float(sample.get(valid_field, 0.0)) > 0.5 and scalar_loss_valid)
    need_chase_decel = bool(
        chase_valid and np.isfinite(chase_max) and float(current_speed) > float(chase_max) + violation_tol
    )

    if phase == PHASE_YLD:
        need_boundary_decel = bool(
            boundary_valid and np.isfinite(boundary_speed) and
            float(current_speed) > float(boundary_speed) + violation_tol
        )
        need_decel = bool(need_chase_decel or need_boundary_decel)
        debug.update({
            "yld_max_speed_mps": float(boundary_speed) if np.isfinite(boundary_speed) else np.nan,
            "yld_max_valid": float(1.0 if boundary_valid else 0.0),
            "need_decel": float(1.0 if need_decel else 0.0),
        })
        if need_decel:
            if not decel_observed:
                issue = ISSUE_CHASE_OVER_MAX_NO_DECEL if need_chase_decel else ISSUE_YLD_OVER_MAX_NO_DECEL
                _set_issue(values, debug, issue, ACTION_DECEL, valid=True, flag=True)
                return values, debug
            _set_issue(values, debug, ISSUE_NONE, ACTION_DECEL, valid=True, flag=False)
            return values, debug
        if boundary_valid and np.isfinite(boundary_speed):
            _set_issue(values, debug, ISSUE_NONE, ACTION_NONE, valid=True, flag=False)
            return values, debug
        if need_chase_decel:
            _set_issue(values, debug, ISSUE_NONE, ACTION_DECEL, valid=True, flag=False)
            return values, debug
        _set_issue(values, debug, ISSUE_NO_APPLICABLE_CONSTRAINT, valid=False, flag=False)
        return values, debug

    go_min = boundary_speed if boundary_valid and np.isfinite(boundary_speed) else np.nan
    yld_speed_field, yld_valid_field = _family_yld_boundary_fields(family)
    yld_cap_speed = _finite_float(sample.get(yld_speed_field, np.nan), default=np.nan) if yld_speed_field else np.nan
    yld_cap_valid = bool(yld_valid_field and float(sample.get(yld_valid_field, 0.0)) > 0.5 and scalar_loss_valid)
    phase_ref_role = _to_int(sample.get("conflict_phase_ref_role", -1), default=-1)
    junction_go_role3_cap_valid = bool(
        family == "junction" and
        phase == PHASE_GO and
        int(phase_ref_role) == 3 and
        yld_cap_valid and
        np.isfinite(yld_cap_speed)
    )
    need_junction_go_decel = bool(
        junction_go_role3_cap_valid and float(current_speed) > float(yld_cap_speed) + violation_tol
    )
    vbmin = _finite_float(sample.get("merge_follow_through_vbmin", np.nan), default=np.nan)
    vbmin_valid = bool(family == "merge" and float(sample.get("merge_follow_through_vbmin_valid", 0.0)) > 0.5 and np.isfinite(vbmin))
    candidates = [value for value in (go_min, vbmin if vbmin_valid else np.nan) if np.isfinite(value)]
    effective_go_min = max(candidates) if candidates else np.nan
    max_candidates = []
    if chase_valid and np.isfinite(chase_max):
        max_candidates.append((float(chase_max), "chase"))
    if junction_go_role3_cap_valid:
        max_candidates.append((float(yld_cap_speed), "junction_yld_as_go_max"))
    effective_speed_max, effective_speed_max_source = (np.nan, "none")
    if max_candidates:
        effective_speed_max, effective_speed_max_source = min(max_candidates, key=lambda item: item[0])
    bound_gap = (
        float(effective_speed_max) - float(effective_go_min)
        if np.isfinite(effective_speed_max) and np.isfinite(effective_go_min)
        else np.nan
    )
    both_bounds_active = bool(np.isfinite(effective_speed_max) and np.isfinite(effective_go_min))
    both_bounds_behavior = (
        _conflicting_bound_behavior(decel_observed, accel_observed)
        if both_bounds_active
        else int(CONFLICTING_BOUND_BEHAVIOR_NONE)
    )
    tight_bounds_filtered = bool(
        np.isfinite(bound_gap) and float(bound_gap) <= float(args.tight_bound_gap_mps)
    )
    debug.update({
        "go_min_speed_mps": float(go_min) if np.isfinite(go_min) else np.nan,
        "go_min_valid": float(1.0 if np.isfinite(go_min) else 0.0),
        "yld_max_speed_mps": float(yld_cap_speed) if np.isfinite(yld_cap_speed) else np.nan,
        "yld_max_valid": float(1.0 if yld_cap_valid else 0.0),
        "junction_go_speed_max_mps": float(yld_cap_speed) if junction_go_role3_cap_valid else np.nan,
        "junction_go_speed_max_valid": float(1.0 if junction_go_role3_cap_valid else 0.0),
        "effective_speed_max_mps": float(effective_speed_max) if np.isfinite(effective_speed_max) else np.nan,
        "effective_speed_max_valid": float(1.0 if np.isfinite(effective_speed_max) else 0.0),
        "effective_speed_max_source": str(effective_speed_max_source),
        "merge_vbmin_mps": float(vbmin) if np.isfinite(vbmin) else np.nan,
        "merge_vbmin_valid": float(1.0 if vbmin_valid else 0.0),
        "effective_go_min_mps": float(effective_go_min) if np.isfinite(effective_go_min) else np.nan,
        "both_bounds_active": float(1.0 if both_bounds_active else 0.0),
        "both_bounds_behavior": int(both_bounds_behavior),
        "both_bounds_behavior_name": CONFLICTING_BOUND_BEHAVIOR_NAMES.get(
            int(both_bounds_behavior), str(both_bounds_behavior)
        ),
        "bound_gap_mps": float(bound_gap) if np.isfinite(bound_gap) else np.nan,
        "tight_bound_gap_mps": float(args.tight_bound_gap_mps),
        "tight_bounds_filtered": float(1.0 if tight_bounds_filtered else 0.0),
    })
    if tight_bounds_filtered:
        _set_issue(values, debug, ISSUE_TIGHT_BOUNDS_FILTERED, ACTION_NONE, valid=False, flag=False)
        return values, debug

    need_accel = bool(np.isfinite(effective_go_min) and float(current_speed) < float(effective_go_min) - violation_tol)
    need_decel = bool(need_chase_decel or need_junction_go_decel)
    conflicting_bounds = bool(need_decel and need_accel)
    conflicting_behavior = (
        _conflicting_bound_behavior(decel_observed, accel_observed)
        if conflicting_bounds
        else int(CONFLICTING_BOUND_BEHAVIOR_NONE)
    )
    debug.update({
        "need_decel": float(1.0 if need_decel else 0.0),
        "need_accel": float(1.0 if need_accel else 0.0),
        "conflicting_bounds": float(1.0 if conflicting_bounds else 0.0),
        "conflicting_bound_behavior": int(conflicting_behavior),
        "conflicting_bound_behavior_name": CONFLICTING_BOUND_BEHAVIOR_NAMES.get(
            int(conflicting_behavior), str(conflicting_behavior)
        ),
    })
    if conflicting_bounds:
        if decel_observed or accel_observed:
            _set_issue(values, debug, ISSUE_NONE, ACTION_EITHER, valid=True, flag=False)
            return values, debug
        _set_issue(values, debug, ISSUE_CONFLICTING_BOUNDS_NO_ACTION, ACTION_EITHER, valid=True, flag=True)
        return values, debug
    if need_decel:
        if not decel_observed:
            issue = ISSUE_JUNCTION_GO_OVER_MAX_NO_DECEL if need_junction_go_decel else ISSUE_CHASE_OVER_MAX_NO_DECEL
            _set_issue(values, debug, issue, ACTION_DECEL, valid=True, flag=True)
            return values, debug
        _set_issue(values, debug, ISSUE_NONE, ACTION_DECEL, valid=True, flag=False)
        return values, debug
    if np.isfinite(effective_go_min):
        if need_accel:
            if not accel_observed:
                _set_issue(values, debug, ISSUE_GO_UNDER_MIN_NO_ACCEL, ACTION_ACCEL, valid=True, flag=True)
                return values, debug
            _set_issue(values, debug, ISSUE_NONE, ACTION_ACCEL, valid=True, flag=False)
            return values, debug
        _set_issue(values, debug, ISSUE_NONE, ACTION_NONE, valid=True, flag=False)
        return values, debug

    _set_issue(values, debug, ISSUE_NO_APPLICABLE_CONSTRAINT, valid=False, flag=False)
    return values, debug


def _write_annotation(sample: dict, values: dict, debug: dict) -> None:
    sample.update(values)
    stage1_debug = _ensure_stage1_debug(sample)
    stage1_debug[DEBUG_KEY] = debug


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Audit non-collision route speed trend consistency against chase/yld/go/vbmin constraints."
    )
    parser.add_argument("--input_path", required=True, help="Existing samples_packed.pkl")
    parser.add_argument("--output_path", required=True, help="Output pickle path")
    parser.add_argument("--image_data_root", required=True, help="Dataset root containing route results.json.gz")
    parser.add_argument("--issue_csv", default=None, help="Optional CSV of issue frames")
    parser.add_argument(
        "--collision_routes_txt",
        action="append",
        default=[],
        help="Optional text list of collision route names to skip in addition to results.json.gz infractions.",
    )
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--allow_inplace", action="store_true")
    parser.add_argument("--include_collision_routes", action="store_true")
    parser.add_argument("--speed_margin_mps", type=float, default=0.25)
    parser.add_argument(
        "--bound_violation_tolerance_mps",
        type=float,
        default=1.0,
        help="Ignore speed-bound violations smaller than this absolute m/s gap.",
    )
    parser.add_argument("--speed_delta_eps_mps", type=float, default=0.05)
    parser.add_argument(
        "--tight_bound_gap_mps",
        type=float,
        default=1.0,
        help="Filter go-phase checks when effective upper/lower bounds are within this gap.",
    )
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

    issue_counts = Counter()
    action_counts = Counter()
    family_counts = Counter()
    route_results_cache = {}
    issue_rows = []
    written = 0
    skipped_existing = 0
    valid_count = 0
    issue_flag_count = 0
    non_collision_routes = 0
    collision_routes = 0
    override_routes = _load_collision_route_overrides(args.collision_routes_txt)

    for route_key, route_indices in _group_indices_by_route(samples):
        route_info = _apply_collision_override(
            route_key,
            _route_collision_info(route_key, args.image_data_root, route_results_cache),
            override_routes,
        )
        if bool(route_info.get("known", False)):
            if bool(route_info.get("collision_route", False)):
                collision_routes += 1
            else:
                non_collision_routes += 1
        for rel_pos, sample_idx in enumerate(route_indices):
            sample = samples[int(sample_idx)]
            if _has_existing_fields(sample) and not bool(args.overwrite_existing):
                skipped_existing += 1
                issue = _to_int(sample.get(CONSISTENCY_ISSUE_KEY, ISSUE_NONE), default=ISSUE_NONE)
                issue_counts[ISSUE_NAMES.get(issue, str(issue))] += 1
                continue
            next_sample = samples[int(route_indices[rel_pos + 1])] if rel_pos + 1 < len(route_indices) else None
            values, debug = _compute_annotation(sample, next_sample, route_info, args)
            _write_annotation(sample, values, debug)
            written += 1
            issue_name = str(debug.get("issue_name", "none"))
            issue_counts[issue_name] += 1
            action_counts[str(debug.get("required_action_name", "none"))] += 1
            if float(debug.get("active", 0.0)) > 0.5:
                family_counts[str(debug.get("family", "none"))] += 1
            if float(values[CONSISTENCY_VALID_KEY]) > 0.5:
                valid_count += 1
            if float(values[CONSISTENCY_ISSUE_FLAG_KEY]) > 0.5:
                issue_flag_count += 1
                issue_rows.append({
                    "route_key": str(route_key),
                    "route_name": str(sample.get("route_name", "")),
                    "frame_id": int(_frame_id(sample)),
                    "family": str(debug.get("family", "none")),
                    "phase": str(debug.get("phase", "none")),
                    "issue_name": issue_name,
                    "required_action": str(debug.get("required_action_name", "none")),
                    "current_speed_mps": float(debug.get("current_speed_mps", np.nan)),
                    "next_speed_mps": float(debug.get("next_speed_mps", np.nan)),
                    "speed_delta_mps": float(debug.get("speed_delta_mps", np.nan)),
                    "chase_speed_max_mps": float(debug.get("chase_speed_max_mps", np.nan)),
                    "yld_max_speed_mps": float(debug.get("yld_max_speed_mps", np.nan)),
                    "effective_speed_max_mps": float(debug.get("effective_speed_max_mps", np.nan)),
                    "effective_speed_max_source": str(debug.get("effective_speed_max_source", "none")),
                    "effective_go_min_mps": float(debug.get("effective_go_min_mps", np.nan)),
                    "bound_gap_mps": float(debug.get("bound_gap_mps", np.nan)),
                    "merge_vbmin_mps": float(debug.get("merge_vbmin_mps", np.nan)),
                })

    _atomic_pickle_dump(samples, output_path)

    if args.issue_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.issue_csv)), exist_ok=True)
        fieldnames = [
            "route_key",
            "route_name",
            "frame_id",
            "family",
            "phase",
            "issue_name",
            "required_action",
            "current_speed_mps",
            "next_speed_mps",
            "speed_delta_mps",
            "chase_speed_max_mps",
            "yld_max_speed_mps",
            "effective_speed_max_mps",
            "effective_speed_max_source",
            "effective_go_min_mps",
            "bound_gap_mps",
            "merge_vbmin_mps",
        ]
        with open(args.issue_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(issue_rows)

    issue_summary = ",".join(f"{key}:{value}" for key, value in sorted(issue_counts.items()))
    action_summary = ",".join(f"{key}:{value}" for key, value in sorted(action_counts.items()))
    family_summary = ",".join(f"{key}:{value}" for key, value in sorted(family_counts.items()))
    print(
        "postprocess boundary speed consistency done: "
        f"samples={len(samples)} "
        f"written={written} "
        f"skipped_existing={skipped_existing} "
        f"valid={valid_count} "
        f"issue_flags={issue_flag_count} "
        f"non_collision_routes={non_collision_routes} "
        f"collision_routes={collision_routes} "
        f"issue_counts={{{issue_summary}}} "
        f"action_counts={{{action_summary}}} "
        f"family_counts={{{family_summary}}} "
        f"issue_csv={args.issue_csv or ''} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
