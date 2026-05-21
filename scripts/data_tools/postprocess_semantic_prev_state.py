#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import os
import pickle
from collections import defaultdict
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


CATEGORICAL_FIELDS = (
    "conflict_area_family",
    "conflict_area_dir",
    "conflict_area_status",
    "conflict_decision_phase",
    "conflict_control_phase",
    "current_cover_edge_mode",
    "future_cover_edge_mode",
    "state_motion_expert_speed_relation",
)

VECTOR_FIELDS = (
    "conflict_area_route_mask",
    "temporary_occupancy_cover_bins",
)

SCALAR_FIELDS = (
    "go_opportunity_prob",
    "yld_pressure_prob",
    "conflict_dist_to_entry_m",
    "conflict_dist_to_exit_m",
    "conflict_time_to_entry_s",
    "merge_yld_max_speed",
    "merge_go_min_speed",
    "junction_yld_max_speed",
    "junction_go_min_speed",
    "borrow_yld_max_speed",
    "borrow_go_min_speed",
    "chase_has_lead",
    "chase_speed_max",
    "current_cover_edge_valid",
    "future_cover_edge_valid",
    "current_cover_upper_speed_mps",
    "future_cover_lower_speed_mps",
    "front_follow_upper_speed_mps",
    "merge_flow_lower_speed_mps",
    "phase_speed_lower_mps",
    "phase_speed_upper_mps",
    "state_motion_speed_interval_valid",
    "state_motion_expert_speed_margin_lower_mps",
    "state_motion_expert_speed_margin_upper_mps",
    "state_motion_risky_passable",
)

MASK_FIELDS = (
    "temporary_occupancy_cover_valid",
    "go_opportunity_valid",
    "conflict_timing_valid",
    "current_cover_edge_mode_valid",
    "future_cover_edge_mode_valid",
    "current_cover_upper_speed_valid",
    "future_cover_lower_speed_valid",
    "front_follow_upper_speed_valid",
    "merge_flow_lower_speed_valid",
    "phase_speed_lower_valid",
    "phase_speed_upper_valid",
)

PREV_FIELDS = (
    tuple(f"prev_{field}" for field in CATEGORICAL_FIELDS)
    + tuple(f"prev_{field}" for field in VECTOR_FIELDS)
    + tuple(f"prev_{field}" for field in SCALAR_FIELDS)
    + tuple(f"prev_{field}" for field in MASK_FIELDS)
    + ("prev_semantic_state_valid",)
)

SEMANTIC_FIELDS = CATEGORICAL_FIELDS + VECTOR_FIELDS + SCALAR_FIELDS + MASK_FIELDS


def _atomic_pickle_save(obj: Any, target_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = target_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, target_path)


def _as_copy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _basename_frame_id(path: str) -> Optional[int]:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _frame_id(sample: Dict[str, Any]) -> Optional[int]:
    for key in ("frame_id", "frame", "tick"):
        if key in sample:
            try:
                return int(sample[key])
            except (TypeError, ValueError):
                pass
    for key in ("transfuser_bev_feature", "bev_feature", "image_path"):
        value = sample.get(key)
        if value:
            frame = _basename_frame_id(str(value))
            if frame is not None:
                return frame
    return None


def _route_key(sample: Dict[str, Any]) -> Tuple[str, ...]:
    parts: List[str] = []
    for key in ("scene_id", "route_id", "route_name", "scenario_id", "town", "log_id"):
        value = sample.get(key)
        if value is not None and str(value) != "":
            parts.append(f"{key}={value}")
    feat_rel = str(sample.get("transfuser_bev_feature", "") or "")
    if feat_rel:
        parent = os.path.dirname(os.path.dirname(feat_rel))
        if parent:
            parts.append(f"feat_parent={parent}")
    if not parts:
        parts.append(f"singleton={id(sample)}")
    return tuple(parts)


def _neutral_value(field: str, *, route_bins: int, temp_bins: int, speed_cap: float) -> Any:
    if field in CATEGORICAL_FIELDS:
        return np.int64(0)
    if field == "conflict_area_route_mask":
        return np.zeros((route_bins,), dtype=np.float32)
    if field == "temporary_occupancy_cover_bins":
        return np.zeros((temp_bins,), dtype=np.float32)
    if field == "temporary_occupancy_cover_valid":
        return np.zeros((temp_bins,), dtype=np.float32)
    if field in MASK_FIELDS:
        return np.float32(0.0)
    if field in ("go_opportunity_prob", "yld_pressure_prob"):
        return np.float32(0.5)
    if field.endswith("_go_min_speed"):
        return np.float32(speed_cap)
    if field in (
        "chase_speed_max",
        "front_follow_upper_speed_mps",
        "phase_speed_upper_mps",
    ):
        return np.float32(speed_cap)
    if field in (
        "current_cover_upper_speed_mps",
        "future_cover_lower_speed_mps",
        "merge_flow_lower_speed_mps",
        "phase_speed_lower_mps",
    ):
        return np.float32(0.0)
    return np.float32(0.0)


def _write_neutral_prefixed(
    sample: Dict[str, Any],
    prefix: str,
    *,
    route_bins: int,
    temp_bins: int,
    speed_cap: float,
) -> None:
    for field in SEMANTIC_FIELDS:
        sample[f"{prefix}_{field}"] = _neutral_value(
            field, route_bins=route_bins, temp_bins=temp_bins, speed_cap=speed_cap
        )
    sample[f"{prefix}_semantic_state_valid"] = np.float32(0.0)


def _copy_prefixed(
    sample: Dict[str, Any],
    source_sample: Dict[str, Any],
    prefix: str,
    *,
    route_bins: int,
    temp_bins: int,
    speed_cap: float,
) -> int:
    copied = 0
    for field in SEMANTIC_FIELDS:
        if field in source_sample:
            sample[f"{prefix}_{field}"] = _as_copy(source_sample[field])
            copied += 1
        else:
            sample[f"{prefix}_{field}"] = _neutral_value(
                field, route_bins=route_bins, temp_bins=temp_bins, speed_cap=speed_cap
            )
    sample[f"{prefix}_semantic_state_valid"] = np.float32(1.0)
    return copied


def _write_neutral_prev(
    sample: Dict[str, Any],
    *,
    route_bins: int,
    temp_bins: int,
    speed_cap: float,
) -> None:
    _write_neutral_prefixed(
        sample,
        "prev",
        route_bins=route_bins,
        temp_bins=temp_bins,
        speed_cap=speed_cap,
    )


def _copy_prev(
    sample: Dict[str, Any],
    prev_sample: Dict[str, Any],
    *,
    route_bins: int,
    temp_bins: int,
    speed_cap: float,
) -> int:
    return _copy_prefixed(
        sample,
        prev_sample,
        "prev",
        route_bins=route_bins,
        temp_bins=temp_bins,
        speed_cap=speed_cap,
    )


def _history_fields(history_depth: int) -> Tuple[str, ...]:
    fields: List[str] = []
    for depth in range(1, max(int(history_depth), 0) + 1):
        prefix = f"hist{depth}"
        fields.extend(f"{prefix}_{field}" for field in SEMANTIC_FIELDS)
        fields.append(f"{prefix}_semantic_state_valid")
    return tuple(fields)


def _clear_existing_prev(sample: Dict[str, Any], *, history_depth: int = 0) -> None:
    for key in PREV_FIELDS:
        sample.pop(key, None)
    hist_pat = re.compile(r"^hist\d+_(semantic_state_valid|" + "|".join(map(re.escape, SEMANTIC_FIELDS)) + r")$")
    for key in list(sample.keys()):
        if hist_pat.match(str(key)):
            sample.pop(key, None)


def _group_samples(samples: Iterable[Dict[str, Any]]):
    grouped = defaultdict(list)
    missing_frame = 0
    for idx, sample in enumerate(samples):
        frame = _frame_id(sample)
        if frame is None:
            missing_frame += 1
            frame = idx
        grouped[_route_key(sample)].append((frame, idx))
    return grouped, missing_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write offline previous semantic-state fields into a packed dataset. "
            "Run independently on train/val packed files after scene split."
        )
    )
    parser.add_argument("--input", required=True, help="Input samples_packed.pkl")
    parser.add_argument("--output", required=True, help="Output samples_packed.pkl")
    parser.add_argument("--route-bins", type=int, default=20)
    parser.add_argument("--temp-bins", type=int, default=13)
    parser.add_argument("--speed-cap", type=float, default=30.0)
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=20,
        help="Maximum allowed frame_id gap for prev link; <=0 disables the gap check.",
    )
    parser.add_argument("--summary-json", default=None)
    parser.add_argument(
        "--history-depth",
        type=int,
        default=0,
        help=(
            "Also write hist1_*..histN_* semantic history fields. "
            "Default 0 preserves the legacy prev_* only behavior."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = os.path.realpath(args.input)
    output_path = os.path.realpath(args.output)
    summary_path = os.path.realpath(args.summary_json or (args.output + ".summary.json"))
    if not args.overwrite and (os.path.exists(output_path) or os.path.exists(summary_path)):
        raise FileExistsError("Output exists; pass --overwrite to replace it.")

    with open(input_path, "rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"Expected list in {input_path}, got {type(samples).__name__}")

    grouped, missing_frame_count = _group_samples(samples)
    history_depth = max(int(args.history_depth), 0)
    for sample in samples:
        _clear_existing_prev(sample, history_depth=history_depth)

    linked = 0
    neutral = 0
    gap_rejected = 0
    copied_field_total = 0
    history_linked = [0 for _ in range(history_depth)]
    history_neutral = [0 for _ in range(history_depth)]
    history_gap_rejected = [0 for _ in range(history_depth)]
    for _, rows in grouped.items():
        rows.sort(key=lambda item: (item[0], item[1]))
        prev_frame: Optional[int] = None
        prev_idx: Optional[int] = None
        history: deque[Tuple[int, int]] = deque(maxlen=max(history_depth, 1))
        for frame, idx in rows:
            sample = samples[idx]
            has_prev = prev_idx is not None
            if has_prev and args.max_frame_gap > 0 and prev_frame is not None:
                if frame - prev_frame > args.max_frame_gap:
                    has_prev = False
                    gap_rejected += 1
                    history.clear()
            if has_prev and prev_idx is not None:
                copied_field_total += _copy_prev(
                    sample,
                    samples[prev_idx],
                    route_bins=args.route_bins,
                    temp_bins=args.temp_bins,
                    speed_cap=args.speed_cap,
                )
                linked += 1
            else:
                _write_neutral_prev(
                    sample,
                    route_bins=args.route_bins,
                    temp_bins=args.temp_bins,
                    speed_cap=args.speed_cap,
                )
                neutral += 1

            history_rows = list(history)
            for depth in range(1, history_depth + 1):
                prefix = f"hist{depth}"
                hist_pos = len(history_rows) - depth
                has_hist = hist_pos >= 0
                if has_hist:
                    hist_frame, hist_idx = history_rows[hist_pos]
                    if args.max_frame_gap > 0 and frame - hist_frame > args.max_frame_gap * depth:
                        has_hist = False
                        history_gap_rejected[depth - 1] += 1
                if has_hist:
                    copied_field_total += _copy_prefixed(
                        sample,
                        samples[hist_idx],
                        prefix,
                        route_bins=args.route_bins,
                        temp_bins=args.temp_bins,
                        speed_cap=args.speed_cap,
                    )
                    history_linked[depth - 1] += 1
                else:
                    _write_neutral_prefixed(
                        sample,
                        prefix,
                        route_bins=args.route_bins,
                        temp_bins=args.temp_bins,
                        speed_cap=args.speed_cap,
                    )
                    history_neutral[depth - 1] += 1

            history.append((frame, idx))
            prev_frame = frame
            prev_idx = idx

    _atomic_pickle_save(samples, output_path)
    summary = {
        "input": input_path,
        "output": output_path,
        "num_samples": len(samples),
        "num_groups": len(grouped),
        "linked_prev": linked,
        "neutral_prev": neutral,
        "gap_rejected": gap_rejected,
        "copied_field_total": copied_field_total,
        "missing_frame_count": missing_frame_count,
        "max_frame_gap": args.max_frame_gap,
        "route_bins": args.route_bins,
        "temp_bins": args.temp_bins,
        "speed_cap": args.speed_cap,
        "prev_field_count": len(PREV_FIELDS),
        "history_depth": history_depth,
        "history_linked": {f"hist{i + 1}": int(v) for i, v in enumerate(history_linked)},
        "history_neutral": {f"hist{i + 1}": int(v) for i, v in enumerate(history_neutral)},
        "history_gap_rejected": {f"hist{i + 1}": int(v) for i, v in enumerate(history_gap_rejected)},
        "history_field_count": len(_history_fields(history_depth)),
    }
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
