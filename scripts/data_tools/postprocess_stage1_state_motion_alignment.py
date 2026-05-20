#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter

import numpy as np


DEBUG_KEY = "state_motion_alignment"

PHASE_SPEED_LOWER_KEY = "phase_speed_lower_mps"
PHASE_SPEED_UPPER_KEY = "phase_speed_upper_mps"
PHASE_SPEED_LOWER_VALID_KEY = "phase_speed_lower_valid"
PHASE_SPEED_UPPER_VALID_KEY = "phase_speed_upper_valid"
PHASE_SPEED_LOWER_SOURCE_KEY = "phase_speed_lower_source"
PHASE_SPEED_UPPER_SOURCE_KEY = "phase_speed_upper_source"

INTERVAL_VALID_KEY = "state_motion_speed_interval_valid"
RELATION_KEY = "state_motion_expert_speed_relation"
MARGIN_LOWER_KEY = "state_motion_expert_speed_margin_lower_mps"
MARGIN_UPPER_KEY = "state_motion_expert_speed_margin_upper_mps"
RISKY_PASSABLE_KEY = "state_motion_risky_passable"

SOURCE_NONE = 0
SOURCE_FUTURE_COVER_LOWER = 1
SOURCE_MERGE_FLOW_LOWER = 2
SOURCE_CURRENT_COVER_UPPER = 3
SOURCE_FRONT_FOLLOW_UPPER = 4

SOURCE_NAMES = {
    SOURCE_NONE: "none",
    SOURCE_FUTURE_COVER_LOWER: "future_cover_lower",
    SOURCE_MERGE_FLOW_LOWER: "merge_flow_lower",
    SOURCE_CURRENT_COVER_UPPER: "current_cover_upper",
    SOURCE_FRONT_FOLLOW_UPPER: "front_follow_upper",
}

REL_INVALID_OR_UNBOUNDED = 0
REL_INSIDE_INTERVAL = 1
REL_BELOW_LOWER = 2
REL_ABOVE_UPPER = 3
REL_RISKY_PASSABLE = 4

RELATION_NAMES = {
    REL_INVALID_OR_UNBOUNDED: "invalid_or_unbounded",
    REL_INSIDE_INTERVAL: "inside_interval",
    REL_BELOW_LOWER: "below_lower",
    REL_ABOVE_UPPER: "above_upper",
    REL_RISKY_PASSABLE: "risky_passable",
}

ALIGNMENT_FIELDS = (
    PHASE_SPEED_LOWER_KEY,
    PHASE_SPEED_UPPER_KEY,
    PHASE_SPEED_LOWER_VALID_KEY,
    PHASE_SPEED_UPPER_VALID_KEY,
    PHASE_SPEED_LOWER_SOURCE_KEY,
    PHASE_SPEED_UPPER_SOURCE_KEY,
    INTERVAL_VALID_KEY,
    RELATION_KEY,
    MARGIN_LOWER_KEY,
    MARGIN_UPPER_KEY,
    RISKY_PASSABLE_KEY,
)


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


def _bool_field(sample: dict, key: str, default: float = 0.0) -> bool:
    return bool(_to_float(sample.get(key, default), default=default) > 0.5)


def _speed_target(sample: dict) -> float:
    for key in ("next_speed_target_mps", "expert_next_speed_mps"):
        value = _finite_float(sample.get(key, np.nan), default=np.nan)
        if np.isfinite(value):
            return float(value)
    # Last-resort fallback is current speed. This makes the audit robust on old
    # packed files, but summary will still expose missing next-speed coverage.
    for key in ("speed", "speed_mps"):
        value = _finite_float(sample.get(key, np.nan), default=np.nan)
        if np.isfinite(value):
            return float(value)
    speed_hist = np.asarray(sample.get("speed_hist", []), dtype=np.float32).reshape(-1)
    if speed_hist.size > 0 and np.isfinite(float(speed_hist[-1])):
        return float(speed_hist[-1])
    return np.nan


def _phase_constraint_allowed(sample: dict) -> bool:
    scalar_valid = _bool_field(sample, "conflict_phase_boundary_scalar_loss_valid", default=1.0)
    actor_match = _bool_field(sample, "conflict_phase_boundary_actor_match", default=1.0)
    object_missing = _bool_field(sample, "conflict_phase_boundary_object_missing", default=0.0)
    open_unbounded = _bool_field(sample, "conflict_phase_open_unbounded", default=0.0)
    return bool(scalar_valid and actor_match and not object_missing and not open_unbounded)


def _valid_speed(sample: dict, speed_key: str, valid_key: str, *, require_phase_ok: bool = False) -> tuple[bool, float]:
    if require_phase_ok and not _phase_constraint_allowed(sample):
        return False, np.nan
    value = _finite_float(sample.get(speed_key, np.nan), default=np.nan)
    valid = _bool_field(sample, valid_key, default=0.0)
    return bool(valid and np.isfinite(value)), float(value)


def _pick_lower(sample: dict) -> tuple[bool, float, int]:
    candidates: list[tuple[float, int]] = []
    valid, value = _valid_speed(
        sample,
        "future_cover_lower_speed_mps",
        "future_cover_lower_speed_valid",
        require_phase_ok=True,
    )
    if valid:
        candidates.append((value, SOURCE_FUTURE_COVER_LOWER))
    valid, value = _valid_speed(sample, "merge_flow_lower_speed_mps", "merge_flow_lower_speed_valid")
    if valid:
        candidates.append((value, SOURCE_MERGE_FLOW_LOWER))
    if not candidates:
        return False, np.nan, SOURCE_NONE
    value, source = max(candidates, key=lambda item: item[0])
    return True, float(value), int(source)


def _pick_upper(sample: dict) -> tuple[bool, float, int]:
    candidates: list[tuple[float, int]] = []
    valid, value = _valid_speed(
        sample,
        "current_cover_upper_speed_mps",
        "current_cover_upper_speed_valid",
        require_phase_ok=True,
    )
    if valid:
        candidates.append((value, SOURCE_CURRENT_COVER_UPPER))
    valid, value = _valid_speed(sample, "front_follow_upper_speed_mps", "front_follow_upper_speed_valid")
    if valid:
        candidates.append((value, SOURCE_FRONT_FOLLOW_UPPER))
    if not candidates:
        return False, np.nan, SOURCE_NONE
    value, source = min(candidates, key=lambda item: item[0])
    return True, float(value), int(source)


def _risky_passable(sample: dict, relation: int) -> bool:
    if relation not in (REL_BELOW_LOWER, REL_ABOVE_UPPER):
        return False
    consistency_valid = _bool_field(sample, "boundary_speed_consistency_valid", default=0.0)
    issue_flag = _bool_field(sample, "boundary_speed_consistency_issue_flag", default=0.0)
    # These are expert samples that survived the route/filtering pipeline but
    # are outside a usable speed interval. Keep them as medium-score examples.
    return bool(consistency_valid and issue_flag)


def _default_values() -> dict:
    return {
        PHASE_SPEED_LOWER_KEY: np.float32(np.nan),
        PHASE_SPEED_UPPER_KEY: np.float32(np.nan),
        PHASE_SPEED_LOWER_VALID_KEY: np.float32(0.0),
        PHASE_SPEED_UPPER_VALID_KEY: np.float32(0.0),
        PHASE_SPEED_LOWER_SOURCE_KEY: np.int64(SOURCE_NONE),
        PHASE_SPEED_UPPER_SOURCE_KEY: np.int64(SOURCE_NONE),
        INTERVAL_VALID_KEY: np.float32(0.0),
        RELATION_KEY: np.int64(REL_INVALID_OR_UNBOUNDED),
        MARGIN_LOWER_KEY: np.float32(np.nan),
        MARGIN_UPPER_KEY: np.float32(np.nan),
        RISKY_PASSABLE_KEY: np.float32(0.0),
    }


def _compute_alignment(sample: dict) -> tuple[dict, dict]:
    values = _default_values()
    lower_valid, lower, lower_source = _pick_lower(sample)
    upper_valid, upper, upper_source = _pick_upper(sample)
    target_speed = _speed_target(sample)

    values[PHASE_SPEED_LOWER_VALID_KEY] = np.float32(1.0 if lower_valid else 0.0)
    values[PHASE_SPEED_UPPER_VALID_KEY] = np.float32(1.0 if upper_valid else 0.0)
    values[PHASE_SPEED_LOWER_SOURCE_KEY] = np.int64(lower_source)
    values[PHASE_SPEED_UPPER_SOURCE_KEY] = np.int64(upper_source)
    if lower_valid:
        values[PHASE_SPEED_LOWER_KEY] = np.float32(lower)
    if upper_valid:
        values[PHASE_SPEED_UPPER_KEY] = np.float32(upper)

    interval_valid = bool((lower_valid or upper_valid) and np.isfinite(target_speed))
    values[INTERVAL_VALID_KEY] = np.float32(1.0 if interval_valid else 0.0)

    relation = REL_INVALID_OR_UNBOUNDED
    margin_lower = np.nan
    margin_upper = np.nan
    if interval_valid:
        if lower_valid:
            margin_lower = float(target_speed - lower)
            if target_speed < lower:
                relation = REL_BELOW_LOWER
        if upper_valid:
            margin_upper = float(upper - target_speed)
            if target_speed > upper:
                relation = REL_ABOVE_UPPER
        if relation == REL_INVALID_OR_UNBOUNDED:
            relation = REL_INSIDE_INTERVAL
        if _risky_passable(sample, relation):
            relation = REL_RISKY_PASSABLE
            values[RISKY_PASSABLE_KEY] = np.float32(1.0)

    values[RELATION_KEY] = np.int64(relation)
    values[MARGIN_LOWER_KEY] = np.float32(margin_lower)
    values[MARGIN_UPPER_KEY] = np.float32(margin_upper)

    debug = {
        "target_speed_mps": float(target_speed) if np.isfinite(target_speed) else np.nan,
        "lower_mps": float(lower) if lower_valid else np.nan,
        "upper_mps": float(upper) if upper_valid else np.nan,
        "lower_valid": float(1.0 if lower_valid else 0.0),
        "upper_valid": float(1.0 if upper_valid else 0.0),
        "lower_source": SOURCE_NAMES.get(lower_source, "unknown"),
        "upper_source": SOURCE_NAMES.get(upper_source, "unknown"),
        "interval_valid": float(values[INTERVAL_VALID_KEY]),
        "relation": int(relation),
        "relation_name": RELATION_NAMES.get(relation, "unknown"),
        "margin_lower_mps": float(margin_lower) if np.isfinite(margin_lower) else np.nan,
        "margin_upper_mps": float(margin_upper) if np.isfinite(margin_upper) else np.nan,
        "risky_passable": float(values[RISKY_PASSABLE_KEY]),
        "phase_constraint_allowed": float(1.0 if _phase_constraint_allowed(sample) else 0.0),
        "boundary_scalar_loss_valid": float(sample.get("conflict_phase_boundary_scalar_loss_valid", np.nan)),
        "boundary_actor_match": float(sample.get("conflict_phase_boundary_actor_match", np.nan)),
        "boundary_object_missing": float(sample.get("conflict_phase_boundary_object_missing", np.nan)),
        "phase_open_unbounded": float(sample.get("conflict_phase_open_unbounded", np.nan)),
    }
    return values, debug


def _has_existing_fields(sample: dict) -> bool:
    stage1_debug = sample.get("stage1_speed_debug")
    debug = stage1_debug.get(DEBUG_KEY) if isinstance(stage1_debug, dict) else None
    return bool(all(key in sample for key in ALIGNMENT_FIELDS) and isinstance(debug, dict))


def _write_annotation(sample: dict, values: dict, debug: dict) -> None:
    sample.update(values)
    stage1_debug = sample.get("stage1_speed_debug")
    if not isinstance(stage1_debug, dict):
        stage1_debug = {}
        sample["stage1_speed_debug"] = stage1_debug
    stage1_debug[DEBUG_KEY] = debug


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive state-motion alignment interval labels from edge-aware stage1 labels."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--allow_inplace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path)
    summary_path = os.path.abspath(args.summary_json or (args.output_path + ".summary.json"))
    if input_path == output_path and not bool(args.allow_inplace):
        raise ValueError("Refusing in-place overwrite without --allow_inplace")
    if os.path.exists(output_path) and not bool(args.overwrite_existing):
        raise FileExistsError("Output exists; pass --overwrite_existing to replace it.")

    with open(input_path, "rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"Expected list in {input_path}, got {type(samples).__name__}")

    written = 0
    skipped_existing = 0
    interval_valid = 0
    lower_valid = 0
    upper_valid = 0
    risky_passable = 0
    next_speed_coverage = 0
    relation_counts = Counter()
    lower_source_counts = Counter()
    upper_source_counts = Counter()
    mask_counts = Counter()

    for sample in samples:
        if _has_existing_fields(sample) and not bool(args.overwrite_existing):
            skipped_existing += 1
            relation = int(sample.get(RELATION_KEY, REL_INVALID_OR_UNBOUNDED))
            relation_counts[RELATION_NAMES.get(relation, str(relation))] += 1
            continue

        values, debug = _compute_alignment(sample)
        _write_annotation(sample, values, debug)
        written += 1

        if "next_speed_target_mps" in sample and np.isfinite(_finite_float(sample.get("next_speed_target_mps", np.nan))):
            next_speed_coverage += 1
        if float(values[INTERVAL_VALID_KEY]) > 0.5:
            interval_valid += 1
        if float(values[PHASE_SPEED_LOWER_VALID_KEY]) > 0.5:
            lower_valid += 1
        if float(values[PHASE_SPEED_UPPER_VALID_KEY]) > 0.5:
            upper_valid += 1
        if float(values[RISKY_PASSABLE_KEY]) > 0.5:
            risky_passable += 1
        relation = int(values[RELATION_KEY])
        relation_counts[RELATION_NAMES.get(relation, str(relation))] += 1
        lower_source_counts[SOURCE_NAMES.get(int(values[PHASE_SPEED_LOWER_SOURCE_KEY]), "unknown")] += 1
        upper_source_counts[SOURCE_NAMES.get(int(values[PHASE_SPEED_UPPER_SOURCE_KEY]), "unknown")] += 1
        if _bool_field(sample, "conflict_phase_open_unbounded", default=0.0):
            mask_counts["open_unbounded"] += 1
        if _bool_field(sample, "conflict_phase_boundary_object_missing", default=0.0):
            mask_counts["object_missing"] += 1
        if not _bool_field(sample, "conflict_phase_boundary_actor_match", default=1.0):
            mask_counts["actor_mismatch"] += 1
        if not _bool_field(sample, "conflict_phase_boundary_scalar_loss_valid", default=1.0):
            mask_counts["scalar_invalid"] += 1

    _atomic_pickle_dump(samples, output_path)

    summary = {
        "input": input_path,
        "output": output_path,
        "summary_json": summary_path,
        "num_samples": len(samples),
        "written": written,
        "skipped_existing": skipped_existing,
        "next_speed_coverage": next_speed_coverage,
        "interval_valid": interval_valid,
        "lower_valid": lower_valid,
        "upper_valid": upper_valid,
        "risky_passable": risky_passable,
        "relation_counts": dict(sorted(relation_counts.items())),
        "lower_source_counts": dict(sorted(lower_source_counts.items())),
        "upper_source_counts": dict(sorted(upper_source_counts.items())),
        "mask_counts": dict(sorted(mask_counts.items())),
    }
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
