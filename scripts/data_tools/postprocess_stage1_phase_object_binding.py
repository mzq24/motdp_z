#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter

import numpy as np


DEBUG_KEY = "phase_object_binding"

PHASE_REF_ROLE_KEY = "conflict_phase_ref_role"
PHASE_REF_ACTOR_ID_KEY = "conflict_phase_ref_actor_id"
PHASE_REF_ACTOR_VALID_KEY = "conflict_phase_ref_actor_valid"
PHASE_OPEN_UNBOUNDED_KEY = "conflict_phase_open_unbounded"
BOUNDARY_REF_ACTOR_ID_KEY = "conflict_phase_boundary_ref_actor_id"
BOUNDARY_REF_VALID_KEY = "conflict_phase_boundary_ref_valid"
BOUNDARY_REF_ROLE_KEY = "conflict_phase_boundary_ref_role"
BOUNDARY_MODE_KEY = "conflict_phase_boundary_mode"
BOUNDARY_STATE_VALID_KEY = "conflict_phase_boundary_state_valid"
BOUNDARY_OBJECT_MISSING_KEY = "conflict_phase_boundary_object_missing"
BOUNDARY_SCALAR_LOSS_VALID_KEY = "conflict_phase_boundary_scalar_loss_valid"
BOUNDARY_ACTOR_MATCH_KEY = "conflict_phase_boundary_actor_match"

ROLE_NONE = 0
ROLE_YLD_TARGET_ACTOR = 1
ROLE_GO_BEFORE_NEXT_ACTOR = 2
ROLE_CURRENT_AREA_ACTOR = 3
ROLE_OPEN_UNBOUNDED = 4

ROLE_NAMES = {
    ROLE_NONE: "none",
    ROLE_YLD_TARGET_ACTOR: "yld_target_actor",
    ROLE_GO_BEFORE_NEXT_ACTOR: "go_before_next_actor",
    ROLE_CURRENT_AREA_ACTOR: "current_area_actor",
    ROLE_OPEN_UNBOUNDED: "open_unbounded",
}

BOUNDARY_MODE_NONE = 0
BOUNDARY_MODE_FUTURE_ACTOR = 1
BOUNDARY_MODE_CURRENT_CLEAR = 2
BOUNDARY_MODE_OPEN_UNBOUNDED = 3

BOUNDARY_MODE_NAMES = {
    BOUNDARY_MODE_NONE: "none",
    BOUNDARY_MODE_FUTURE_ACTOR: "future_actor_boundary",
    BOUNDARY_MODE_CURRENT_CLEAR: "current_clear_transition",
    BOUNDARY_MODE_OPEN_UNBOUNDED: "open_unbounded",
}

PHASE_NONE = 0
PHASE_YLD = 1
PHASE_GO = 2

PHASE_NAMES = {
    PHASE_NONE: "none",
    PHASE_YLD: "yld",
    PHASE_GO: "go",
}

FAMILY_CODE_TO_NAME = {
    0: "none",
    1: "borrow",
    2: "merge",
    3: "junction",
}
ACTIVE_FAMILIES = {"borrow", "merge", "junction"}
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


def _current_cover_is_phase_area_actor(sample: dict, family: str, current_cover: dict) -> tuple[bool, str]:
    if not _cover_exists(current_cover):
        return False, "missing_current_cover"
    if family == "merge":
        return _merge_current_cover_overlaps_area(sample, current_cover)
    # For borrow / junction, current_cover is already family-filtered upstream
    # in the current labeler. Merge needs the extra area-exit guard because the
    # same actor can remain a front/chase cover after leaving the merge area.
    return True, "none"


def _family_name(sample: dict) -> str:
    conflict_area = _stage1_block(sample, "conflict_area")
    family = str(conflict_area.get("family", "none"))
    if family != "none":
        return family
    return FAMILY_CODE_TO_NAME.get(_to_int(sample.get("conflict_area_family", 0), default=0), "none")


def _phase_code(sample: dict) -> int:
    value = _to_int(sample.get("conflict_decision_phase", PHASE_NONE), default=PHASE_NONE)
    if value in PHASE_NAMES:
        return int(value)
    phase_debug = _stage1_block(sample, "conflict_phase")
    phase_name = str(phase_debug.get("phase", "none"))
    if phase_name == "yld":
        return PHASE_YLD
    if phase_name == "go":
        return PHASE_GO
    return PHASE_NONE


def _is_active_conflict_sample(sample: dict) -> bool:
    if _to_float(sample.get("conflict_area_active", 0.0), default=0.0) <= 0.5:
        return False
    return _family_name(sample) in ACTIVE_FAMILIES


def _role_name(role: int) -> str:
    return ROLE_NAMES.get(int(role), str(role))


def _boundary_mode_name(mode: int) -> str:
    return BOUNDARY_MODE_NAMES.get(int(mode), str(mode))


def _default_values() -> dict:
    return {
        PHASE_REF_ROLE_KEY: np.int64(ROLE_NONE),
        PHASE_REF_ACTOR_ID_KEY: np.int64(-1),
        PHASE_REF_ACTOR_VALID_KEY: np.float32(0.0),
        PHASE_OPEN_UNBOUNDED_KEY: np.float32(0.0),
        BOUNDARY_REF_ACTOR_ID_KEY: np.int64(-1),
        BOUNDARY_REF_VALID_KEY: np.float32(0.0),
        BOUNDARY_REF_ROLE_KEY: np.int64(ROLE_NONE),
        BOUNDARY_MODE_KEY: np.int64(BOUNDARY_MODE_NONE),
        BOUNDARY_STATE_VALID_KEY: np.float32(0.0),
        BOUNDARY_OBJECT_MISSING_KEY: np.float32(0.0),
        BOUNDARY_SCALAR_LOSS_VALID_KEY: np.float32(0.0),
        BOUNDARY_ACTOR_MATCH_KEY: np.float32(0.0),
    }


def _default_debug(args, issue_reason: str = "not_computed") -> dict:
    return {
        "active": 0.0,
        "family": "none",
        "phase": "none",
        "source": "stage1_cover_thresholds",
        "current_candidate": {
            "exists": 0.0,
            "actor_id": -1,
            "actor_valid": 0.0,
            "role": int(ROLE_NONE),
            "role_name": ROLE_NAMES[ROLE_NONE],
        },
        "future_candidate": {
            "exists": 0.0,
            "actor_id": -1,
            "actor_valid": 0.0,
            "role": int(ROLE_NONE),
            "role_name": ROLE_NAMES[ROLE_NONE],
            "frame_index": -1,
            "distance_m": np.nan,
            "distance_source": "none",
            "gate_passed": 0.0,
            "gate_reason": "none",
            "frame_gate": int(args.future_frame_gate),
            "distance_gate_m": float(args.future_distance_gate_m),
        },
        "phase_ref": {
            "role": int(ROLE_NONE),
            "role_name": ROLE_NAMES[ROLE_NONE],
            "actor_id": -1,
            "actor_valid": 0.0,
            "open_unbounded": 0.0,
            "source": "none",
        },
        "boundary_ref": {
            "role": int(ROLE_NONE),
            "role_name": ROLE_NAMES[ROLE_NONE],
            "mode": int(BOUNDARY_MODE_NONE),
            "mode_name": BOUNDARY_MODE_NAMES[BOUNDARY_MODE_NONE],
            "alignment_source": "none",
            "actor_id": -1,
            "actor_valid": 0.0,
            "state_valid": 0.0,
            "source_frame": -1,
            "cover_case": "none",
            "threshold_source": "none",
            "branch": "none",
            "branch_valid": 0.0,
            "issue_reason": "none",
        },
        "boundary_actor_match": 0.0,
        "boundary_object_missing": 0.0,
        "boundary_scalar_loss_valid": 0.0,
        "issue_reason": str(issue_reason),
    }


def _has_existing_fields(sample: dict) -> bool:
    stage1_debug = sample.get("stage1_speed_debug")
    debug = stage1_debug.get(DEBUG_KEY) if isinstance(stage1_debug, dict) else None
    return bool(
        PHASE_REF_ROLE_KEY in sample
        and PHASE_REF_ACTOR_ID_KEY in sample
        and PHASE_REF_ACTOR_VALID_KEY in sample
        and PHASE_OPEN_UNBOUNDED_KEY in sample
        and BOUNDARY_REF_ACTOR_ID_KEY in sample
        and BOUNDARY_REF_VALID_KEY in sample
        and BOUNDARY_REF_ROLE_KEY in sample
        and BOUNDARY_MODE_KEY in sample
        and BOUNDARY_STATE_VALID_KEY in sample
        and BOUNDARY_OBJECT_MISSING_KEY in sample
        and BOUNDARY_SCALAR_LOSS_VALID_KEY in sample
        and BOUNDARY_ACTOR_MATCH_KEY in sample
        and isinstance(debug, dict)
    )


def _future_gate_info(future_cover: dict, args) -> dict:
    actor_id = _cover_actor_id(future_cover)
    exists = _cover_exists(future_cover)
    frame_index = _to_int((future_cover or {}).get("frame_index", -1), default=-1)
    d_bg = _finite_float((future_cover or {}).get("d_bg", np.nan), default=np.nan)
    if np.isfinite(d_bg):
        gate_distance = float(d_bg)
        distance_source = "d_bg"
    else:
        gate_distance = np.nan
        distance_source = "none"

    if not exists:
        gate_reason = "no_future_cover"
    elif frame_index <= 0 or frame_index > int(args.future_frame_gate):
        gate_reason = "future_frame_gate_fail"
    elif not np.isfinite(gate_distance):
        gate_reason = "future_distance_missing"
    elif gate_distance > float(args.future_distance_gate_m):
        gate_reason = "future_distance_gate_fail"
    else:
        gate_reason = "none"

    gate_passed = bool(exists and gate_reason == "none")
    return {
        "exists": float(1.0 if exists else 0.0),
        "actor_id": int(actor_id),
        "actor_valid": float(1.0 if _actor_valid(actor_id) else 0.0),
        "role": int(ROLE_NONE),
        "role_name": ROLE_NAMES[ROLE_NONE],
        "frame_index": int(frame_index),
        "distance_m": float(gate_distance) if np.isfinite(gate_distance) else np.nan,
        "distance_source": str(distance_source),
        "gate_passed": float(1.0 if gate_passed else 0.0),
        "gate_reason": str(gate_reason),
        "frame_gate": int(args.future_frame_gate),
        "distance_gate_m": float(args.future_distance_gate_m),
    }


def _candidate_debug(sample: dict, family: str, current_cover: dict, future_cover: dict, args) -> tuple[dict, dict]:
    current_actor_id = _cover_actor_id(current_cover)
    current_cover_exists = _cover_exists(current_cover)
    current_is_area_actor, current_gate_reason = _current_cover_is_phase_area_actor(sample, family, current_cover)
    current_candidate = {
        "exists": float(1.0 if current_cover_exists else 0.0),
        "area_actor": float(1.0 if current_is_area_actor else 0.0),
        "actor_id": int(current_actor_id),
        "actor_valid": float(1.0 if _actor_valid(current_actor_id) else 0.0),
        "role": int(ROLE_CURRENT_AREA_ACTOR if current_is_area_actor else ROLE_NONE),
        "role_name": _role_name(ROLE_CURRENT_AREA_ACTOR if current_is_area_actor else ROLE_NONE),
        "gate_reason": str(current_gate_reason),
    }
    future_candidate = _future_gate_info(future_cover, args)
    return current_candidate, future_candidate


def _threshold_block(sample: dict, family: str) -> dict:
    if family not in ACTIVE_FAMILIES:
        return {}
    return _stage1_block(sample, f"{family}_thresholds")


def _branch_valid(threshold_debug: dict, phase_code: int) -> tuple[str, bool]:
    if phase_code == PHASE_YLD:
        return "yld", bool(_to_float((threshold_debug or {}).get("yld_valid", 0.0), default=0.0) > 0.5)
    if phase_code == PHASE_GO:
        return "go", bool(_to_float((threshold_debug or {}).get("go_valid", 0.0), default=0.0) > 0.5)
    return "none", False


def _boundary_role_for_future_branch(branch: str) -> int:
    if branch == "yld":
        return int(ROLE_YLD_TARGET_ACTOR)
    if branch == "go":
        return int(ROLE_GO_BEFORE_NEXT_ACTOR)
    return int(ROLE_NONE)


def _boundary_payload(
    *,
    role: int,
    mode: int,
    actor_id: int = -1,
    actor_valid: bool = False,
    state_valid: bool = False,
    source_frame: int = -1,
    cover_case: str = "none",
    threshold_source: str = "none",
    branch: str = "none",
    branch_valid: bool = False,
    issue_reason: str = "none",
    alignment_source: str = "threshold_debug",
) -> dict:
    return {
        "role": int(role),
        "role_name": _role_name(int(role)),
        "mode": int(mode),
        "mode_name": _boundary_mode_name(int(mode)),
        "alignment_source": str(alignment_source),
        "actor_id": int(actor_id),
        "actor_valid": float(1.0 if actor_valid else 0.0),
        "state_valid": float(1.0 if state_valid else 0.0),
        "source_frame": int(source_frame),
        "cover_case": str(cover_case),
        "threshold_source": str(threshold_source),
        "branch": str(branch),
        "branch_valid": float(1.0 if branch_valid else 0.0),
        "issue_reason": str(issue_reason),
    }


def _lookup_cover_actor(route_frame_lookup: dict[int, dict], source_frame: int, cover_case: str) -> tuple[int, bool, str]:
    source_sample = route_frame_lookup.get(int(source_frame))
    if source_sample is None:
        return -1, False, "boundary_source_frame_not_found"
    cover_key = "current_cover" if cover_case == "current" else "future_cover"
    cover = _stage1_block(source_sample, cover_key)
    actor_id = _cover_actor_id(cover)
    actor_valid = _actor_valid(actor_id)
    if not actor_valid:
        return int(actor_id), False, "boundary_actor_id_invalid"
    return int(actor_id), True, "none"


def _resolve_boundary_ref(sample: dict, route_frame_lookup: dict[int, dict], family: str, phase_code: int) -> dict:
    threshold_debug = _threshold_block(sample, family)
    branch, valid_branch = _branch_valid(threshold_debug, phase_code)
    cover_case = str((threshold_debug or {}).get("cover_case", "none"))
    source_frame = _to_int((threshold_debug or {}).get("source_frame", -1), default=-1)
    threshold_source = str((threshold_debug or {}).get("source", "none"))

    if not threshold_debug:
        return _boundary_payload(
            role=ROLE_NONE,
            mode=BOUNDARY_MODE_NONE,
            source_frame=source_frame,
            cover_case=cover_case,
            threshold_source=threshold_source,
            branch=branch,
            branch_valid=valid_branch,
            issue_reason="missing_threshold_debug",
        )
    if _to_float(threshold_debug.get("active", 0.0), default=0.0) <= 0.5:
        return _boundary_payload(
            role=ROLE_NONE,
            mode=BOUNDARY_MODE_NONE,
            source_frame=source_frame,
            cover_case=cover_case,
            threshold_source=threshold_source,
            branch=branch,
            branch_valid=valid_branch,
            issue_reason="boundary_threshold_inactive",
        )
    if branch == "none":
        return _boundary_payload(
            role=ROLE_NONE,
            mode=BOUNDARY_MODE_NONE,
            source_frame=source_frame,
            cover_case=cover_case,
            threshold_source=threshold_source,
            branch=branch,
            branch_valid=valid_branch,
            issue_reason="boundary_phase_none",
        )

    if cover_case not in {"current", "future"}:
        if "unconstrained" in threshold_source:
            return _boundary_payload(
                role=ROLE_OPEN_UNBOUNDED,
                mode=BOUNDARY_MODE_OPEN_UNBOUNDED,
                source_frame=source_frame,
                cover_case=cover_case,
                threshold_source=threshold_source,
                branch=branch,
                branch_valid=valid_branch,
                state_valid=True,
                issue_reason="none",
            )
        return _boundary_payload(
            role=ROLE_NONE,
            mode=BOUNDARY_MODE_NONE,
            source_frame=source_frame,
            cover_case=cover_case,
            threshold_source=threshold_source,
            branch=branch,
            branch_valid=valid_branch,
            issue_reason="boundary_cover_case_invalid",
        )

    role = ROLE_CURRENT_AREA_ACTOR if cover_case == "current" else _boundary_role_for_future_branch(branch)
    mode = BOUNDARY_MODE_CURRENT_CLEAR if cover_case == "current" else BOUNDARY_MODE_FUTURE_ACTOR
    if source_frame < 0:
        return _boundary_payload(
            role=role,
            mode=mode,
            source_frame=source_frame,
            cover_case=cover_case,
            threshold_source=threshold_source,
            branch=branch,
            branch_valid=valid_branch,
            issue_reason="boundary_source_frame_missing",
        )

    actor_id, actor_valid, actor_issue = _lookup_cover_actor(route_frame_lookup, source_frame, cover_case)
    if cover_case == "current":
        state_valid = bool(actor_valid)
        issue_reason = actor_issue
    else:
        state_valid = bool(actor_valid and valid_branch)
        issue_reason = actor_issue if actor_issue != "none" else ("none" if valid_branch else "boundary_branch_invalid")

    return _boundary_payload(
        role=role,
        mode=mode,
        actor_id=actor_id,
        actor_valid=actor_valid,
        state_valid=state_valid,
        source_frame=source_frame,
        cover_case=cover_case,
        threshold_source=threshold_source,
        branch=branch,
        branch_valid=valid_branch,
        issue_reason=issue_reason,
    )


def _future_phase_ref_payload(
    *,
    role: int,
    actor_id: int,
    actor_valid: bool,
    source: str = "future_cover",
) -> dict:
    issue_reason = "none" if actor_valid else "future_actor_id_invalid"
    return {
        "role": int(role),
        "role_name": ROLE_NAMES[int(role)],
        "actor_id": int(actor_id),
        "actor_valid": float(1.0 if actor_valid else 0.0),
        "open_unbounded": 0.0,
        "source": str(source),
        "issue_reason": issue_reason,
    }


def _select_phase_ref(phase_code: int, family: str, current_candidate: dict, future_candidate: dict) -> dict:
    current_exists = int(current_candidate.get("role", ROLE_NONE)) == ROLE_CURRENT_AREA_ACTOR
    current_actor_id = int(current_candidate.get("actor_id", -1))
    current_actor_valid = _actor_valid(current_actor_id)
    future_gate_passed = float(future_candidate.get("gate_passed", 0.0)) > 0.5
    future_actor_id = int(future_candidate.get("actor_id", -1))
    future_actor_valid = _actor_valid(future_actor_id)

    if family == "merge" and current_exists and future_gate_passed:
        # Merge has a common mixed state: the current cover is still clearing
        # the merge area while the next future cover already determines the
        # scalar yld/go boundary. Bind the phase object to that future actor so
        # phase-object and boundary-object diagnostics stay on the same object.
        if phase_code == PHASE_YLD:
            return _future_phase_ref_payload(
                role=ROLE_YLD_TARGET_ACTOR,
                actor_id=future_actor_id,
                actor_valid=future_actor_valid,
                source="future_cover_merge_current_area_mixed",
            )
        if phase_code == PHASE_GO:
            return _future_phase_ref_payload(
                role=ROLE_GO_BEFORE_NEXT_ACTOR,
                actor_id=future_actor_id,
                actor_valid=future_actor_valid,
                source="future_cover_merge_current_area_mixed",
            )

    if current_exists:
        issue_reason = "none" if current_actor_valid else "current_actor_id_invalid"
        return {
            "role": int(ROLE_CURRENT_AREA_ACTOR),
            "role_name": ROLE_NAMES[ROLE_CURRENT_AREA_ACTOR],
            "actor_id": int(current_actor_id),
            "actor_valid": float(1.0 if current_actor_valid else 0.0),
            "open_unbounded": 0.0,
            "source": "current_cover",
            "issue_reason": issue_reason,
        }

    if phase_code == PHASE_YLD:
        if future_gate_passed:
            return _future_phase_ref_payload(
                role=ROLE_YLD_TARGET_ACTOR,
                actor_id=future_actor_id,
                actor_valid=future_actor_valid,
            )
        return {
            "role": int(ROLE_NONE),
            "role_name": ROLE_NAMES[ROLE_NONE],
            "actor_id": -1,
            "actor_valid": 0.0,
            "open_unbounded": 0.0,
            "source": "none",
            "issue_reason": "yld_missing_reference_actor",
        }

    if phase_code == PHASE_GO:
        if future_gate_passed:
            return _future_phase_ref_payload(
                role=ROLE_GO_BEFORE_NEXT_ACTOR,
                actor_id=future_actor_id,
                actor_valid=future_actor_valid,
            )
        return {
            "role": int(ROLE_OPEN_UNBOUNDED),
            "role_name": ROLE_NAMES[ROLE_OPEN_UNBOUNDED],
            "actor_id": -1,
            "actor_valid": 0.0,
            "open_unbounded": 1.0,
            "source": "open_unbounded",
            "issue_reason": "none",
        }

    return {
        "role": int(ROLE_NONE),
        "role_name": ROLE_NAMES[ROLE_NONE],
        "actor_id": -1,
        "actor_valid": 0.0,
        "open_unbounded": 0.0,
        "source": "none",
        "issue_reason": "phase_none",
    }


def _phase_ref_state_valid(phase_ref: dict) -> bool:
    role = int(phase_ref.get("role", ROLE_NONE))
    if role == ROLE_OPEN_UNBOUNDED:
        return bool(float(phase_ref.get("open_unbounded", 0.0)) > 0.5)
    if role in {ROLE_YLD_TARGET_ACTOR, ROLE_GO_BEFORE_NEXT_ACTOR, ROLE_CURRENT_AREA_ACTOR}:
        return bool(float(phase_ref.get("actor_valid", 0.0)) > 0.5)
    return False


def _boundary_mode_for_phase_role(role: int) -> int:
    role = int(role)
    if role == ROLE_CURRENT_AREA_ACTOR:
        return int(BOUNDARY_MODE_CURRENT_CLEAR)
    if role in {ROLE_YLD_TARGET_ACTOR, ROLE_GO_BEFORE_NEXT_ACTOR}:
        return int(BOUNDARY_MODE_FUTURE_ACTOR)
    if role == ROLE_OPEN_UNBOUNDED:
        return int(BOUNDARY_MODE_OPEN_UNBOUNDED)
    return int(BOUNDARY_MODE_NONE)


def _boundary_cover_case_for_phase_role(role: int) -> str:
    role = int(role)
    if role == ROLE_CURRENT_AREA_ACTOR:
        return "current"
    if role in {ROLE_YLD_TARGET_ACTOR, ROLE_GO_BEFORE_NEXT_ACTOR}:
        return "future"
    return "none"


def _boundary_from_phase_ref(phase_ref: dict, boundary_ref: dict) -> dict:
    role = int(phase_ref.get("role", ROLE_NONE))
    mode = _boundary_mode_for_phase_role(role)
    actor_id = int(phase_ref.get("actor_id", -1))
    actor_valid = bool(float(phase_ref.get("actor_valid", 0.0)) > 0.5)
    if role == ROLE_OPEN_UNBOUNDED:
        actor_id = -1
        actor_valid = False
    aligned = _boundary_payload(
        role=role,
        mode=mode,
        actor_id=actor_id,
        actor_valid=actor_valid,
        state_valid=True,
        source_frame=int(boundary_ref.get("source_frame", -1)),
        cover_case=_boundary_cover_case_for_phase_role(role),
        threshold_source=str(boundary_ref.get("threshold_source", "none")),
        branch=str(boundary_ref.get("branch", "none")),
        branch_valid=bool(float(boundary_ref.get("branch_valid", 0.0)) > 0.5),
        issue_reason="none",
        alignment_source="phase_ref_relation",
    )
    aligned["pre_alignment"] = {
        "role": int(boundary_ref.get("role", ROLE_NONE)),
        "role_name": str(boundary_ref.get("role_name", _role_name(ROLE_NONE))),
        "mode": int(boundary_ref.get("mode", BOUNDARY_MODE_NONE)),
        "mode_name": str(boundary_ref.get("mode_name", _boundary_mode_name(BOUNDARY_MODE_NONE))),
        "actor_id": int(boundary_ref.get("actor_id", -1)),
        "actor_valid": float(boundary_ref.get("actor_valid", 0.0)),
        "state_valid": float(boundary_ref.get("state_valid", 0.0)),
        "issue_reason": str(boundary_ref.get("issue_reason", "none")),
        "alignment_source": str(boundary_ref.get("alignment_source", "threshold_debug")),
    }
    return aligned


def _actor_match(phase_ref: dict, boundary_ref: dict) -> float:
    phase_role = int(phase_ref.get("role", ROLE_NONE))
    boundary_role = int(boundary_ref.get("role", ROLE_NONE))
    phase_valid = float(phase_ref.get("actor_valid", 0.0)) > 0.5
    boundary_valid = float(boundary_ref.get("actor_valid", 0.0)) > 0.5
    boundary_state_valid = float(boundary_ref.get("state_valid", 0.0)) > 0.5
    if not boundary_state_valid:
        return 0.0
    if phase_role == ROLE_OPEN_UNBOUNDED and boundary_role == ROLE_OPEN_UNBOUNDED:
        return 1.0
    if (
        phase_role == boundary_role
        and phase_valid
        and boundary_valid
        and int(phase_ref.get("actor_id", -1)) == int(boundary_ref.get("actor_id", -1))
    ):
        return 1.0
    return 0.0


def _align_boundary_ref_to_phase_ref(phase_ref: dict, boundary_ref: dict) -> dict:
    # Keep boundary provenance faithful to the threshold scalar source. If a
    # future-cover scalar boundary coexists with a current-cover phase ref, the
    # mismatch is important supervision/debug signal and should not be hidden by
    # rewriting the boundary object to the phase object.
    return boundary_ref


def _boundary_object_missing(phase_code: int, phase_ref: dict) -> bool:
    if int(phase_code) == PHASE_NONE:
        return False
    return bool(int(phase_ref.get("role", ROLE_NONE)) == ROLE_NONE)


def _boundary_scalar_loss_valid(boundary_object_missing: bool, boundary_ref: dict, match: float) -> bool:
    if boundary_object_missing:
        return False
    if float(boundary_ref.get("actor_valid", 0.0)) > 0.5 and float(match) <= 0.5:
        # Actor-conditioned scalar boundary should only supervise the current
        # phase object when the object relation actually matches.
        return False
    role = int(boundary_ref.get("role", ROLE_NONE))
    if role == ROLE_OPEN_UNBOUNDED:
        return False
    return bool(float(boundary_ref.get("state_valid", 0.0)) > 0.5)


def _compute_annotation(sample: dict, route_frame_lookup: dict[int, dict], args) -> tuple[dict, dict]:
    values = _default_values()
    debug = _default_debug(args)

    family = _family_name(sample)
    phase_code = _phase_code(sample)
    phase_name = PHASE_NAMES.get(int(phase_code), str(phase_code))
    current_cover = _stage1_block(sample, "current_cover")
    future_cover = _stage1_block(sample, "future_cover")
    current_candidate, future_candidate = _candidate_debug(sample, family, current_cover, future_cover, args)

    debug.update({
        "active": float(1.0 if _is_active_conflict_sample(sample) else 0.0),
        "family": str(family),
        "phase": str(phase_name),
        "current_candidate": current_candidate,
        "future_candidate": future_candidate,
    })

    if not _is_active_conflict_sample(sample):
        debug["issue_reason"] = "inactive_or_unsupported_family"
        debug["boundary_ref"]["issue_reason"] = "inactive_or_unsupported_family"
        debug["future_candidate"]["gate_reason"] = "inactive_or_unsupported_family"
        return values, debug

    phase_ref = _select_phase_ref(phase_code, family, current_candidate, future_candidate)
    if str(phase_ref.get("source", "none")).startswith("future_cover"):
        future_candidate["role"] = int(phase_ref.get("role", ROLE_NONE))
        future_candidate["role_name"] = _role_name(int(phase_ref.get("role", ROLE_NONE)))
    boundary_ref = _resolve_boundary_ref(sample, route_frame_lookup, family, phase_code)
    boundary_ref = _align_boundary_ref_to_phase_ref(phase_ref, boundary_ref)
    match = _actor_match(phase_ref, boundary_ref)
    object_missing = _boundary_object_missing(phase_code, phase_ref)
    scalar_loss_valid = _boundary_scalar_loss_valid(object_missing, boundary_ref, match)

    values.update({
        PHASE_REF_ROLE_KEY: np.int64(int(phase_ref["role"])),
        PHASE_REF_ACTOR_ID_KEY: np.int64(int(phase_ref["actor_id"])),
        PHASE_REF_ACTOR_VALID_KEY: np.float32(float(phase_ref["actor_valid"])),
        PHASE_OPEN_UNBOUNDED_KEY: np.float32(float(phase_ref["open_unbounded"])),
        BOUNDARY_REF_ACTOR_ID_KEY: np.int64(int(boundary_ref["actor_id"])),
        BOUNDARY_REF_VALID_KEY: np.float32(float(boundary_ref["actor_valid"])),
        BOUNDARY_REF_ROLE_KEY: np.int64(int(boundary_ref["role"])),
        BOUNDARY_MODE_KEY: np.int64(int(boundary_ref["mode"])),
        BOUNDARY_STATE_VALID_KEY: np.float32(float(boundary_ref["state_valid"])),
        BOUNDARY_OBJECT_MISSING_KEY: np.float32(1.0 if object_missing else 0.0),
        BOUNDARY_SCALAR_LOSS_VALID_KEY: np.float32(1.0 if scalar_loss_valid else 0.0),
        BOUNDARY_ACTOR_MATCH_KEY: np.float32(float(match)),
    })
    debug.update({
        "phase_ref": phase_ref,
        "future_candidate": future_candidate,
        "boundary_ref": boundary_ref,
        "boundary_actor_match": float(match),
        "boundary_object_missing": float(1.0 if object_missing else 0.0),
        "boundary_scalar_loss_valid": float(1.0 if scalar_loss_valid else 0.0),
        "issue_reason": str(phase_ref.get("issue_reason", "none")),
    })
    return values, debug


def _write_annotation(sample: dict, values: dict, debug: dict) -> None:
    sample.update(values)
    stage1_debug = _ensure_stage1_debug(sample)
    stage1_debug[DEBUG_KEY] = debug


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Postprocess stage1 phase/object binding labels from current_cover, future_cover, phase, and thresholds."
    )
    parser.add_argument("--input_path", required=True, help="Existing samples_packed.pkl")
    parser.add_argument("--output_path", required=True, help="Output pickle path")
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Recompute labels even when phase-object binding fields already exist.",
    )
    parser.add_argument(
        "--allow_inplace",
        action="store_true",
        help="Allow --input_path and --output_path to be the same file.",
    )
    parser.add_argument("--future_frame_gate", type=int, default=6)
    parser.add_argument("--future_distance_gate_m", type=float, default=20.0)
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

    role_counts = Counter()
    boundary_role_counts = Counter()
    boundary_mode_counts = Counter()
    boundary_alignment_counts = Counter()
    family_counts = Counter()
    issue_counts = Counter()
    boundary_issue_counts = Counter()
    future_gate_counts = Counter()
    written = 0
    skipped_existing = 0
    active_count = 0
    open_unbounded_count = 0
    boundary_valid_count = 0
    boundary_state_valid_count = 0
    boundary_match_count = 0
    boundary_object_missing_count = 0
    boundary_scalar_loss_valid_count = 0

    for _, route_indices in _group_indices_by_route(samples):
        route_frame_lookup = {
            _frame_id(samples[int(sample_idx)]): samples[int(sample_idx)]
            for sample_idx in route_indices
        }
        for sample_idx in route_indices:
            sample = samples[int(sample_idx)]
            if _has_existing_fields(sample) and not bool(args.overwrite_existing):
                skipped_existing += 1
                role = int(sample.get(PHASE_REF_ROLE_KEY, ROLE_NONE))
                role_counts[ROLE_NAMES.get(role, str(role))] += 1
                boundary_role = int(sample.get(BOUNDARY_REF_ROLE_KEY, ROLE_NONE))
                boundary_role_counts[ROLE_NAMES.get(boundary_role, str(boundary_role))] += 1
                boundary_mode = int(sample.get(BOUNDARY_MODE_KEY, BOUNDARY_MODE_NONE))
                boundary_mode_counts[BOUNDARY_MODE_NAMES.get(boundary_mode, str(boundary_mode))] += 1
                stage1_debug = sample.get("stage1_speed_debug")
                debug = stage1_debug.get(DEBUG_KEY) if isinstance(stage1_debug, dict) else {}
                boundary_ref = debug.get("boundary_ref", {}) if isinstance(debug, dict) else {}
                boundary_alignment_counts[str(boundary_ref.get("alignment_source", "unknown"))] += 1
                continue

            values, debug = _compute_annotation(sample, route_frame_lookup, args)
            _write_annotation(sample, values, debug)
            written += 1

            role = int(values[PHASE_REF_ROLE_KEY])
            role_counts[ROLE_NAMES.get(role, str(role))] += 1
            boundary_role = int(values[BOUNDARY_REF_ROLE_KEY])
            boundary_role_counts[ROLE_NAMES.get(boundary_role, str(boundary_role))] += 1
            boundary_mode = int(values[BOUNDARY_MODE_KEY])
            boundary_mode_counts[BOUNDARY_MODE_NAMES.get(boundary_mode, str(boundary_mode))] += 1
            if float(debug.get("active", 0.0)) > 0.5:
                active_count += 1
                family_counts[str(debug.get("family", "none"))] += 1
            issue_counts[str(debug.get("issue_reason", "none"))] += 1
            boundary_ref = debug.get("boundary_ref", {})
            boundary_issue_counts[str(boundary_ref.get("issue_reason", "none"))] += 1
            boundary_alignment_counts[str(boundary_ref.get("alignment_source", "unknown"))] += 1
            future_candidate = debug.get("future_candidate", {})
            future_gate_counts[str(future_candidate.get("gate_reason", "none"))] += 1
            if float(values[PHASE_OPEN_UNBOUNDED_KEY]) > 0.5:
                open_unbounded_count += 1
            if float(values[BOUNDARY_REF_VALID_KEY]) > 0.5:
                boundary_valid_count += 1
            if float(values[BOUNDARY_STATE_VALID_KEY]) > 0.5:
                boundary_state_valid_count += 1
            if float(values[BOUNDARY_OBJECT_MISSING_KEY]) > 0.5:
                boundary_object_missing_count += 1
            if float(values[BOUNDARY_SCALAR_LOSS_VALID_KEY]) > 0.5:
                boundary_scalar_loss_valid_count += 1
            if float(values[BOUNDARY_ACTOR_MATCH_KEY]) > 0.5:
                boundary_match_count += 1

    _atomic_pickle_dump(samples, output_path)

    role_summary = ",".join(f"{key}:{value}" for key, value in sorted(role_counts.items()))
    boundary_role_summary = ",".join(f"{key}:{value}" for key, value in sorted(boundary_role_counts.items()))
    boundary_mode_summary = ",".join(f"{key}:{value}" for key, value in sorted(boundary_mode_counts.items()))
    boundary_alignment_summary = ",".join(f"{key}:{value}" for key, value in sorted(boundary_alignment_counts.items()))
    family_summary = ",".join(f"{key}:{value}" for key, value in sorted(family_counts.items()))
    issue_summary = ",".join(f"{key}:{value}" for key, value in sorted(issue_counts.items()))
    boundary_issue_summary = ",".join(f"{key}:{value}" for key, value in sorted(boundary_issue_counts.items()))
    future_gate_summary = ",".join(f"{key}:{value}" for key, value in sorted(future_gate_counts.items()))
    print(
        "postprocess phase-object binding done: "
        f"samples={len(samples)} "
        f"written={written} "
        f"skipped_existing={skipped_existing} "
        f"active={active_count} "
        f"open_unbounded={open_unbounded_count} "
        f"boundary_valid={boundary_valid_count} "
        f"boundary_state_valid={boundary_state_valid_count} "
        f"boundary_object_missing={boundary_object_missing_count} "
        f"boundary_scalar_loss_valid={boundary_scalar_loss_valid_count} "
        f"boundary_match={boundary_match_count} "
        f"role_counts={{{role_summary}}} "
        f"boundary_role_counts={{{boundary_role_summary}}} "
        f"boundary_mode_counts={{{boundary_mode_summary}}} "
        f"boundary_alignment_counts={{{boundary_alignment_summary}}} "
        f"family_counts={{{family_summary}}} "
        f"issue_counts={{{issue_summary}}} "
        f"boundary_issue_counts={{{boundary_issue_summary}}} "
        f"future_gate_counts={{{future_gate_summary}}} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
