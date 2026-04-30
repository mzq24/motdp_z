#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter

import numpy as np


CHASE_DEBUG_KEY = "chase_front_following"

CHASE_HAS_LEAD_KEY = "chase_has_lead"
CHASE_STATUS_KEY = "chase_status"
CHASE_DIST_KEY = "chase_dist_m"
CHASE_DIST_VALID_KEY = "chase_dist_valid"
CHASE_TTC_KEY = "chase_ttc_s"
CHASE_TTC_VALID_KEY = "chase_ttc_valid"
CHASE_SPEED_MAX_KEY = "chase_speed_max"
CHASE_SPEED_MAX_VALID_KEY = "chase_speed_max_valid"

CHASE_STATUS_NONE = 0
CHASE_STATUS_LEAD_FAR = 1
CHASE_STATUS_LEAD_CLOSE = 2
CHASE_STATUS_BLOCKED_OR_TTC_LOW = 3

CHASE_STATUS_NAMES = {
    CHASE_STATUS_NONE: "none",
    CHASE_STATUS_LEAD_FAR: "lead_far",
    CHASE_STATUS_LEAD_CLOSE: "lead_close",
    CHASE_STATUS_BLOCKED_OR_TTC_LOW: "blocked_or_ttc_low",
}


def _atomic_pickle_dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


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


def _cover_interaction(cover: dict) -> dict:
    interaction = (cover or {}).get("interaction")
    return interaction if isinstance(interaction, dict) else {}


def _cover_interaction_name(cover: dict) -> str:
    return str(_cover_interaction(cover).get("name", "none"))


def _cover_interaction_subtype(cover: dict) -> str:
    interaction = _cover_interaction(cover)
    return str(interaction.get("subtype") or interaction.get("name") or "none")


def _to_float(value, default=np.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    return float(value)


def _finite_float(value, default=np.nan) -> float:
    value = _to_float(value, default=default)
    return float(value) if np.isfinite(value) else float(default)


def _sample_current_speed_mps(sample: dict) -> float:
    for key in ("speed", "speed_mps"):
        value = _finite_float(sample.get(key, np.nan), default=np.nan)
        if np.isfinite(value):
            return float(value)
    speed_hist = np.asarray(sample.get("speed_hist", []), dtype=np.float32).reshape(-1)
    if speed_hist.size > 0 and np.isfinite(float(speed_hist[-1])):
        return float(speed_hist[-1])
    return np.nan


def _default_chase_debug() -> dict:
    return {
        "active": 0.0,
        "source": "current_cover",
        "filter": "any_current_cover",
        "actor_id": -1,
        "actor_class_name": "none",
        "interaction_name": "none",
        "interaction_subtype": "none",
        "distance_m": np.nan,
        "ttc_s": np.nan,
        "lead_speed_mps": np.nan,
        "speed_max_mps": np.nan,
        "status": int(CHASE_STATUS_NONE),
        "status_name": CHASE_STATUS_NAMES[CHASE_STATUS_NONE],
        "dist_valid": 0.0,
        "ttc_valid": 0.0,
        "speed_max_valid": 0.0,
        "issue_reason": "none",
    }


def _set_chase_defaults(
    sample: dict,
    dist_cap_m: float,
    ttc_cap_s: float,
    speed_cap_mps: float,
    issue_reason: str = "not_computed",
) -> None:
    sample[CHASE_HAS_LEAD_KEY] = np.float32(0.0)
    sample[CHASE_STATUS_KEY] = np.int64(CHASE_STATUS_NONE)
    sample[CHASE_DIST_KEY] = np.float32(float(dist_cap_m))
    sample[CHASE_DIST_VALID_KEY] = np.float32(0.0)
    sample[CHASE_TTC_KEY] = np.float32(float(ttc_cap_s))
    sample[CHASE_TTC_VALID_KEY] = np.float32(0.0)
    sample[CHASE_SPEED_MAX_KEY] = np.float32(float(speed_cap_mps))
    sample[CHASE_SPEED_MAX_VALID_KEY] = np.float32(0.0)

    stage1_debug = _ensure_stage1_debug(sample)
    debug = _default_chase_debug()
    debug.update({
        "distance_m": float(dist_cap_m),
        "ttc_s": float(ttc_cap_s),
        "speed_max_mps": float(speed_cap_mps),
        "issue_reason": str(issue_reason),
    })
    stage1_debug[CHASE_DEBUG_KEY] = debug


def _has_existing_chase_fields(sample: dict) -> bool:
    stage1_debug = sample.get("stage1_speed_debug")
    debug = stage1_debug.get(CHASE_DEBUG_KEY) if isinstance(stage1_debug, dict) else None
    return bool(
        CHASE_HAS_LEAD_KEY in sample and
        CHASE_STATUS_KEY in sample and
        CHASE_DIST_KEY in sample and
        CHASE_DIST_VALID_KEY in sample and
        CHASE_TTC_KEY in sample and
        CHASE_TTC_VALID_KEY in sample and
        CHASE_SPEED_MAX_KEY in sample and
        CHASE_SPEED_MAX_VALID_KEY in sample and
        isinstance(debug, dict)
    )


def _cover_is_current_lead(cover: dict, follow_chase_only: bool) -> bool:
    if int(_to_float((cover or {}).get("exists", 0.0), default=0.0)) <= 0:
        return False
    if int(_to_float((cover or {}).get("case", 1), default=1.0)) != 1:
        return False
    if int(_to_float((cover or {}).get("frame_index", 0), default=0.0)) not in (0, -1):
        return False
    if follow_chase_only:
        return bool(
            _cover_interaction_name(cover) == "chase" or
            _cover_interaction_subtype(cover) == "follow_chase"
        )
    return True


def _status_from_dist_ttc(
    has_lead: bool,
    dist_m: float,
    ttc_s: float,
    close_dist_m: float,
    blocked_dist_m: float,
    close_ttc_s: float,
    blocked_ttc_s: float,
) -> int:
    if not has_lead:
        return CHASE_STATUS_NONE

    dist_finite = np.isfinite(float(dist_m))
    ttc_finite = np.isfinite(float(ttc_s))
    if (dist_finite and float(dist_m) <= float(blocked_dist_m)) or (
        ttc_finite and float(ttc_s) <= float(blocked_ttc_s)
    ):
        return CHASE_STATUS_BLOCKED_OR_TTC_LOW
    if (dist_finite and float(dist_m) <= float(close_dist_m)) or (
        ttc_finite and float(ttc_s) <= float(close_ttc_s)
    ):
        return CHASE_STATUS_LEAD_CLOSE
    return CHASE_STATUS_LEAD_FAR


def _compute_chase_annotation(sample: dict, args) -> tuple[dict, dict]:
    follow_chase_only = bool(args.follow_chase_only)
    current_cover = _stage1_block(sample, "current_cover")
    route_valid = isinstance(current_cover, dict) and bool(current_cover)
    if not route_valid:
        return (
            {
                CHASE_HAS_LEAD_KEY: np.float32(0.0),
                CHASE_STATUS_KEY: np.int64(CHASE_STATUS_NONE),
                CHASE_DIST_KEY: np.float32(float(args.dist_cap_m)),
                CHASE_DIST_VALID_KEY: np.float32(0.0),
                CHASE_TTC_KEY: np.float32(float(args.ttc_cap_s)),
                CHASE_TTC_VALID_KEY: np.float32(0.0),
                CHASE_SPEED_MAX_KEY: np.float32(float(args.speed_cap_mps)),
                CHASE_SPEED_MAX_VALID_KEY: np.float32(0.0),
            },
            {
                **_default_chase_debug(),
                "filter": "follow_chase_only" if follow_chase_only else "any_current_cover",
                "distance_m": float(args.dist_cap_m),
                "ttc_s": float(args.ttc_cap_s),
                "speed_max_mps": float(args.speed_cap_mps),
                "issue_reason": "missing_current_cover_debug",
            },
        )

    has_lead = _cover_is_current_lead(current_cover, follow_chase_only=follow_chase_only)
    dist_raw = _finite_float(current_cover.get("distance", np.nan), default=np.nan)
    ttc_raw = _finite_float(current_cover.get("ttc", np.nan), default=np.nan)
    lead_speed = _finite_float(current_cover.get("other_speed", np.nan), default=np.nan)

    if has_lead and not np.isfinite(dist_raw):
        has_lead = False
        issue_reason = "lead_missing_distance"
    else:
        issue_reason = "none"

    dist_m = float(np.clip(dist_raw, 0.0, float(args.dist_cap_m))) if has_lead else float(args.dist_cap_m)
    if has_lead and np.isfinite(ttc_raw):
        ttc_s = float(np.clip(ttc_raw, 0.0, float(args.ttc_cap_s)))
    elif has_lead:
        ttc_s = float(args.ttc_cap_s)
        if issue_reason == "none":
            issue_reason = "lead_missing_ttc"
    else:
        ttc_s = float(args.ttc_cap_s)

    interaction_name = _cover_interaction_name(current_cover)
    interaction_subtype = _cover_interaction_subtype(current_cover)
    uses_lead_speed_for_cap = bool(interaction_name == "chase" or interaction_subtype == "follow_chase")
    cap_base_speed = float(lead_speed) if uses_lead_speed_for_cap and np.isfinite(lead_speed) else 0.0

    if has_lead and (np.isfinite(lead_speed) or not uses_lead_speed_for_cap):
        available_gap_m = max(float(dist_m) - float(args.safe_gap_m), 0.0)
        speed_max = float(cap_base_speed) + available_gap_m / max(float(args.safe_ttc_s), 1e-6)
        speed_max = float(np.clip(speed_max, 0.0, float(args.speed_cap_mps)))
        speed_valid = 1.0
    elif has_lead:
        speed_max = float(args.speed_cap_mps)
        speed_valid = 0.0
        if issue_reason == "none":
            issue_reason = "lead_missing_speed"
    else:
        speed_max = float(args.speed_cap_mps)
        speed_valid = 1.0

    status = _status_from_dist_ttc(
        has_lead=bool(has_lead),
        dist_m=dist_m,
        ttc_s=ttc_s,
        close_dist_m=float(args.close_dist_m),
        blocked_dist_m=float(args.blocked_dist_m),
        close_ttc_s=float(args.close_ttc_s),
        blocked_ttc_s=float(args.blocked_ttc_s),
    )

    values = {
        CHASE_HAS_LEAD_KEY: np.float32(1.0 if has_lead else 0.0),
        CHASE_STATUS_KEY: np.int64(int(status)),
        CHASE_DIST_KEY: np.float32(float(dist_m)),
        CHASE_DIST_VALID_KEY: np.float32(1.0 if route_valid else 0.0),
        CHASE_TTC_KEY: np.float32(float(ttc_s)),
        CHASE_TTC_VALID_KEY: np.float32(1.0 if has_lead else 0.0),
        CHASE_SPEED_MAX_KEY: np.float32(float(speed_max)),
        CHASE_SPEED_MAX_VALID_KEY: np.float32(float(speed_valid) if route_valid else 0.0),
    }
    debug = _default_chase_debug()
    debug.update({
        "active": float(1.0 if has_lead else 0.0),
        "source": "current_cover",
        "filter": "follow_chase_only" if follow_chase_only else "any_current_cover",
        "actor_id": int(_to_float(current_cover.get("actor_id", -1), default=-1)),
        "actor_class_name": str(current_cover.get("actor_class_name", "none")),
        "interaction_name": interaction_name,
        "interaction_subtype": interaction_subtype,
        "distance_m": float(dist_m),
        "ttc_s": float(ttc_s),
        "lead_speed_mps": float(lead_speed) if np.isfinite(lead_speed) else np.nan,
        "cap_base_speed_mps": float(cap_base_speed),
        "cap_uses_lead_speed": float(1.0 if uses_lead_speed_for_cap else 0.0),
        "speed_max_mps": float(speed_max),
        "status": int(status),
        "status_name": CHASE_STATUS_NAMES.get(int(status), str(status)),
        "dist_valid": float(values[CHASE_DIST_VALID_KEY]),
        "ttc_valid": float(values[CHASE_TTC_VALID_KEY]),
        "speed_max_valid": float(values[CHASE_SPEED_MAX_VALID_KEY]),
        "safe_gap_m": float(args.safe_gap_m),
        "safe_ttc_s": float(args.safe_ttc_s),
        "issue_reason": str(issue_reason),
    })
    return values, debug


def _write_chase_annotation(sample: dict, values: dict, debug: dict) -> None:
    sample.update(values)
    stage1_debug = _ensure_stage1_debug(sample)
    stage1_debug[CHASE_DEBUG_KEY] = debug


def _parse_args():
    parser = argparse.ArgumentParser(description="Postprocess stage1 chase/front-following labels from current_cover.")
    parser.add_argument("--input_path", required=True, help="Existing samples_packed.pkl")
    parser.add_argument("--output_path", required=True, help="Output pickle path")
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Recompute labels even when chase_front_following fields already exist.",
    )
    parser.add_argument(
        "--allow_inplace",
        action="store_true",
        help="Allow --input_path and --output_path to be the same file.",
    )
    parser.add_argument(
        "--follow_chase_only",
        action="store_true",
        help="Only treat current_cover interaction=chase/follow_chase as lead. Default accepts any current-frame route cover.",
    )
    parser.add_argument("--dist_cap_m", type=float, default=40.0)
    parser.add_argument("--ttc_cap_s", type=float, default=10.0)
    parser.add_argument("--speed_cap_mps", type=float, default=30.0)
    parser.add_argument("--safe_gap_m", type=float, default=5.0)
    parser.add_argument("--safe_ttc_s", type=float, default=3.0)
    parser.add_argument("--close_dist_m", type=float, default=20.0)
    parser.add_argument("--blocked_dist_m", type=float, default=8.0)
    parser.add_argument("--close_ttc_s", type=float, default=5.0)
    parser.add_argument("--blocked_ttc_s", type=float, default=2.0)
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

    status_counts = Counter()
    issue_counts = Counter()
    interaction_counts = Counter()
    written = 0
    skipped_existing = 0
    has_lead_count = 0
    speed_cap_violations = 0

    for sample in samples:
        if _has_existing_chase_fields(sample) and not bool(args.overwrite_existing):
            skipped_existing += 1
            status = int(sample.get(CHASE_STATUS_KEY, CHASE_STATUS_NONE))
            status_counts[CHASE_STATUS_NAMES.get(status, str(status))] += 1
            if float(sample.get(CHASE_HAS_LEAD_KEY, 0.0)) > 0.5:
                has_lead_count += 1
            continue

        values, debug = _compute_chase_annotation(sample, args)
        _write_chase_annotation(sample, values, debug)
        written += 1

        status = int(values[CHASE_STATUS_KEY])
        status_counts[CHASE_STATUS_NAMES.get(status, str(status))] += 1
        issue_counts[str(debug.get("issue_reason", "none"))] += 1
        if float(values[CHASE_HAS_LEAD_KEY]) > 0.5:
            has_lead_count += 1
            interaction_counts[str(debug.get("interaction_subtype", "none"))] += 1

        current_speed = _sample_current_speed_mps(sample)
        if (
            np.isfinite(current_speed) and
            float(values[CHASE_SPEED_MAX_VALID_KEY]) > 0.5 and
            float(current_speed) > float(values[CHASE_SPEED_MAX_KEY]) + 1e-3
        ):
            speed_cap_violations += 1

    _atomic_pickle_dump(samples, output_path)

    status_summary = ",".join(f"{key}:{value}" for key, value in sorted(status_counts.items()))
    issue_summary = ",".join(f"{key}:{value}" for key, value in sorted(issue_counts.items()))
    interaction_summary = ",".join(f"{key}:{value}" for key, value in sorted(interaction_counts.items()))
    print(
        "postprocess chase front-following done: "
        f"samples={len(samples)} "
        f"written={written} "
        f"skipped_existing={skipped_existing} "
        f"has_lead={has_lead_count} "
        f"speed_cap_violations={speed_cap_violations} "
        f"status_counts={{{status_summary}}} "
        f"issue_counts={{{issue_summary}}} "
        f"interaction_counts={{{interaction_summary}}} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
