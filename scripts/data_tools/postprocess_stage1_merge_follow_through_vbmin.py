#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter

import numpy as np


DEBUG_KEY = "merge_follow_through_vbmin"

VBMIN_KEY = "merge_follow_through_vbmin"
VBMIN_VALID_KEY = "merge_follow_through_vbmin_valid"
VBMIN_ACTOR_ID_KEY = "merge_follow_through_vbmin_actor_id"
VBMIN_ACTOR_VALID_KEY = "merge_follow_through_vbmin_actor_valid"

FAMILY_CODE_TO_NAME = {
    0: "none",
    1: "borrow",
    2: "merge",
    3: "junction",
}

AREA_STATUS_INSIDE = 2
AREA_STATUS_AFTER = 3
PHASE_YLD = 1
PHASE_GO = 2


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


def _is_active_merge(sample: dict) -> bool:
    return bool(
        _to_float(sample.get("conflict_area_active", 0.0), default=0.0) > 0.5 and
        _family_name(sample) == "merge"
    )


def _is_follow_through_region(sample: dict) -> bool:
    phase = _to_int(sample.get("conflict_decision_phase", 0), default=0)
    if phase not in {PHASE_YLD, PHASE_GO}:
        return False

    status = _to_int(sample.get("conflict_area_status", 0), default=0)
    if status == AREA_STATUS_INSIDE:
        return True
    if status == AREA_STATUS_AFTER:
        # The conflict window/phase intentionally outlives the strict area end.
        # Keep merge follow-through vbmin alive during that tail instead of
        # dropping it exactly at the geometric exit boundary.
        return True

    dist_to_entry = _finite_float(sample.get("conflict_dist_to_entry_m", np.nan), default=np.nan)
    dist_to_exit = _finite_float(sample.get("conflict_dist_to_exit_m", np.nan), default=np.nan)
    if np.isfinite(dist_to_entry) and np.isfinite(dist_to_exit):
        return bool(float(dist_to_entry) <= 0.0 and float(dist_to_exit) > 0.0)
    return False


def _default_values() -> dict:
    return {
        VBMIN_KEY: np.float32(np.nan),
        VBMIN_VALID_KEY: np.float32(0.0),
        VBMIN_ACTOR_ID_KEY: np.int64(-1),
        VBMIN_ACTOR_VALID_KEY: np.float32(0.0),
    }


def _default_debug(issue_reason: str = "not_computed") -> dict:
    return {
        "active": 0.0,
        "family": "none",
        "follow_through_active": 0.0,
        "source": "none",
        "speed_min_mps": np.nan,
        "speed_min_valid": 0.0,
        "actor_id": -1,
        "actor_valid": 0.0,
        "source_frame": -1,
        "source_cover_case": "none",
        "area_status": 0,
        "dist_to_entry_m": np.nan,
        "dist_to_exit_m": np.nan,
        "carry_speed_mps": np.nan,
        "carry_actor_id": -1,
        "issue_reason": str(issue_reason),
    }


def _has_existing_fields(sample: dict) -> bool:
    stage1_debug = sample.get("stage1_speed_debug")
    debug = stage1_debug.get(DEBUG_KEY) if isinstance(stage1_debug, dict) else None
    return bool(
        VBMIN_KEY in sample and
        VBMIN_VALID_KEY in sample and
        VBMIN_ACTOR_ID_KEY in sample and
        VBMIN_ACTOR_VALID_KEY in sample and
        isinstance(debug, dict)
    )


def _cap_speed(speed_mps: float, speed_cap_mps: float) -> float:
    if not np.isfinite(speed_mps):
        return np.nan
    return float(np.clip(float(speed_mps), 0.0, float(speed_cap_mps)))


def _candidate_from_cover(sample: dict, cover_key: str, speed_cap_mps: float) -> dict | None:
    cover = _stage1_block(sample, cover_key)
    if not _cover_exists(cover):
        return None
    speed = _cap_speed(_finite_float(cover.get("other_speed", np.nan), default=np.nan), speed_cap_mps)
    if not np.isfinite(speed):
        return None
    actor_id = _cover_actor_id(cover)
    return {
        "speed_mps": float(speed),
        "actor_id": int(actor_id),
        "actor_valid": float(1.0 if _actor_valid(actor_id) else 0.0),
        "source": str(cover_key),
        "source_frame": int(_frame_id(sample)),
        "source_cover_case": "current" if cover_key == "current_cover" else "future",
    }


def _candidate_from_threshold(sample: dict, speed_cap_mps: float) -> dict | None:
    threshold = _stage1_block(sample, "merge_thresholds")
    if not threshold or _to_float(threshold.get("active", 0.0), default=0.0) <= 0.5:
        return None
    cover_case = str(threshold.get("cover_case", "none"))
    speed = _cap_speed(_finite_float(threshold.get("bg_speed_mps", np.nan), default=np.nan), speed_cap_mps)
    if not np.isfinite(speed):
        return None
    cover_key = "future_cover" if cover_case == "future" else "current_cover" if cover_case == "current" else "none"
    cover = _stage1_block(sample, cover_key) if cover_key != "none" else {}
    actor_id = _cover_actor_id(cover)
    return {
        "speed_mps": float(speed),
        "actor_id": int(actor_id),
        "actor_valid": float(1.0 if _actor_valid(actor_id) else 0.0),
        "source": "merge_thresholds.bg_speed_mps",
        "source_frame": int(_to_int(threshold.get("source_frame", _frame_id(sample)), default=_frame_id(sample))),
        "source_cover_case": str(cover_case),
    }


def _best_vbmin_candidate(sample: dict, speed_cap_mps: float) -> dict | None:
    candidates = [
        _candidate_from_threshold(sample, speed_cap_mps),
        _candidate_from_cover(sample, "future_cover", speed_cap_mps),
        _candidate_from_cover(sample, "current_cover", speed_cap_mps),
    ]
    candidates = [cand for cand in candidates if cand is not None and np.isfinite(float(cand["speed_mps"]))]
    if not candidates:
        return None
    return max(candidates, key=lambda cand: float(cand["speed_mps"]))


def _write_sample(sample: dict, values: dict, debug: dict) -> None:
    sample.update(values)
    stage1_debug = _ensure_stage1_debug(sample)
    stage1_debug[DEBUG_KEY] = debug


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Postprocess merge-only follow-through vbmin label after ego enters conflict area."
    )
    parser.add_argument("--input_path", required=True, help="Existing samples_packed.pkl")
    parser.add_argument("--output_path", required=True, help="Output pickle path")
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Recompute labels even when merge follow-through vbmin fields already exist.",
    )
    parser.add_argument(
        "--allow_inplace",
        action="store_true",
        help="Allow --input_path and --output_path to be the same file.",
    )
    parser.add_argument("--speed_cap_mps", type=float, default=30.0)
    parser.add_argument(
        "--min_candidate_speed_mps",
        type=float,
        default=0.5,
        help="Ignore near-zero actor speeds when carrying vbmin.",
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
    source_counts = Counter()
    written = 0
    skipped_existing = 0
    active_merge_count = 0
    follow_count = 0
    valid_count = 0

    for _, route_indices in _group_indices_by_route(samples):
        carry_speed = np.nan
        carry_actor_id = -1
        carry_actor_valid = 0.0
        carry_source = "none"
        carry_source_frame = -1
        carry_cover_case = "none"

        for sample_idx in route_indices:
            sample = samples[int(sample_idx)]
            if _has_existing_fields(sample) and not bool(args.overwrite_existing):
                skipped_existing += 1
                if float(sample.get(VBMIN_VALID_KEY, 0.0)) > 0.5:
                    valid_count += 1
                continue

            values = _default_values()
            debug = _default_debug()
            family = _family_name(sample)
            active_merge = _is_active_merge(sample)
            follow_active = bool(active_merge and _is_follow_through_region(sample))

            if not active_merge:
                carry_speed = np.nan
                carry_actor_id = -1
                carry_actor_valid = 0.0
                carry_source = "none"
                carry_source_frame = -1
                carry_cover_case = "none"
            else:
                active_merge_count += 1
                candidate = _best_vbmin_candidate(sample, float(args.speed_cap_mps))
                if candidate is not None and float(candidate["speed_mps"]) >= float(args.min_candidate_speed_mps):
                    # Keep the strongest observed traffic floor so ego-induced braking
                    # does not erase the follow-through constraint.
                    if not np.isfinite(carry_speed) or float(candidate["speed_mps"]) > float(carry_speed):
                        carry_speed = float(candidate["speed_mps"])
                        carry_actor_id = int(candidate["actor_id"])
                        carry_actor_valid = float(candidate["actor_valid"])
                        carry_source = str(candidate["source"])
                        carry_source_frame = int(candidate["source_frame"])
                        carry_cover_case = str(candidate["source_cover_case"])

            issue_reason = "inactive_or_non_merge"
            if active_merge:
                issue_reason = "before_entry_or_inactive_phase"
            if follow_active:
                follow_count += 1
                if np.isfinite(carry_speed):
                    values.update({
                        VBMIN_KEY: np.float32(float(carry_speed)),
                        VBMIN_VALID_KEY: np.float32(1.0),
                        VBMIN_ACTOR_ID_KEY: np.int64(int(carry_actor_id)),
                        VBMIN_ACTOR_VALID_KEY: np.float32(float(carry_actor_valid)),
                    })
                    valid_count += 1
                    issue_reason = "none"
                else:
                    issue_reason = "missing_vbmin_source"

            debug.update({
                "active": float(1.0 if active_merge else 0.0),
                "family": str(family),
                "follow_through_active": float(1.0 if follow_active else 0.0),
                "source": str(carry_source if np.isfinite(carry_speed) else "none"),
                "speed_min_mps": float(carry_speed) if np.isfinite(carry_speed) else np.nan,
                "speed_min_valid": float(values[VBMIN_VALID_KEY]),
                "actor_id": int(carry_actor_id),
                "actor_valid": float(carry_actor_valid),
                "source_frame": int(carry_source_frame),
                "source_cover_case": str(carry_cover_case),
                "area_status": int(_to_int(sample.get("conflict_area_status", 0), default=0)),
                "dist_to_entry_m": float(_finite_float(sample.get("conflict_dist_to_entry_m", np.nan), default=np.nan)),
                "dist_to_exit_m": float(_finite_float(sample.get("conflict_dist_to_exit_m", np.nan), default=np.nan)),
                "carry_speed_mps": float(carry_speed) if np.isfinite(carry_speed) else np.nan,
                "carry_actor_id": int(carry_actor_id),
                "issue_reason": str(issue_reason),
            })
            _write_sample(sample, values, debug)
            written += 1
            issue_counts[str(issue_reason)] += 1
            source_counts[str(debug.get("source", "none"))] += 1

    _atomic_pickle_dump(samples, output_path)

    issue_summary = ",".join(f"{key}:{value}" for key, value in sorted(issue_counts.items()))
    source_summary = ",".join(f"{key}:{value}" for key, value in sorted(source_counts.items()))
    print(
        "postprocess merge follow-through vbmin done: "
        f"samples={len(samples)} "
        f"written={written} "
        f"skipped_existing={skipped_existing} "
        f"active_merge={active_merge_count} "
        f"follow_through={follow_count} "
        f"valid={valid_count} "
        f"issue_counts={{{issue_summary}}} "
        f"source_counts={{{source_summary}}} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
