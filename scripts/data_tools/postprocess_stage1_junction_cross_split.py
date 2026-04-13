#!/usr/bin/env python3
import argparse
import os
import pickle

import numpy as np


DEFAULT_CROSS_SAFE_GAP_S = 1.0


def _as_float_array(value, fallback_shape=None):
    if value is None:
        if fallback_shape is None:
            return np.zeros((0,), dtype=np.float32)
        return np.zeros(fallback_shape, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(np.float32, copy=False)


def _empty_like_speed(sample, speed_curve):
    sample_speeds = _as_float_array(sample.get("speed_sample_values"))
    if sample_speeds.size > 0:
        return np.zeros(sample_speeds.shape, dtype=np.float32), sample_speeds
    curve_speeds = _as_float_array((speed_curve or {}).get("sample_speeds_mps"))
    if curve_speeds.size > 0:
        return np.zeros(curve_speeds.shape, dtype=np.float32), curve_speeds
    meet_risks = _as_float_array(sample.get("speed_risk_meet_values"))
    if meet_risks.size > 0:
        return np.zeros(meet_risks.shape, dtype=np.float32), meet_risks
    return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)


def _derive_junction_cross_split(sample, cross_safe_gap_s):
    stage1_debug = sample.get("stage1_speed_debug")
    speed_curve = stage1_debug.get("speed_curve", {}) if isinstance(stage1_debug, dict) else {}
    meet_debug = speed_curve.get("meet_debug", {}) if isinstance(speed_curve, dict) else {}

    zeros, sample_speeds = _empty_like_speed(sample, speed_curve)
    yld = zeros.copy()
    go = zeros.copy()

    subtype = str(meet_debug.get("subtype", "none"))
    if subtype != "junction_left_cross_meet":
        return yld, go

    meet_risks = _as_float_array(sample.get("speed_risk_meet_values"), fallback_shape=zeros.shape)
    if meet_risks.shape != zeros.shape:
        meet_risks = _as_float_array(speed_curve.get("meet_risks"), fallback_shape=zeros.shape)
    if meet_risks.shape != zeros.shape:
        meet_risks = zeros.copy()

    cover_case = str(meet_debug.get("cover_case", "")).lower()
    if cover_case == "current":
        yld = np.clip(meet_risks, 0.0, 1.0).astype(np.float32, copy=False)
        go = np.ones_like(yld, dtype=np.float32)
        return yld, go

    risk_d_ego = float(meet_debug.get("d_ego_m", np.nan))
    t_bg = float(meet_debug.get("t_bg_s", np.nan))
    t_bg_exit = float(meet_debug.get("t_bg_exit_s", np.nan))
    conflict_len = float(meet_debug.get("conflict_len_m", np.nan))
    ego_clearance = float(meet_debug.get("ego_clearance_m", np.nan))
    if not np.isfinite(ego_clearance):
        ego_clearance = 0.0
    if not np.isfinite(conflict_len):
        conflict_len = 0.0

    if not (np.isfinite(risk_d_ego) and np.isfinite(t_bg) and np.isfinite(t_bg_exit)):
        return yld, go

    ego_cross_occ_len = max(conflict_len, 0.0) + max(ego_clearance, 0.0)
    gap = max(float(cross_safe_gap_s), 1e-6)
    for idx, candidate_speed in enumerate(sample_speeds):
        v = float(candidate_speed)
        if v <= 1e-6:
            yld[idx] = 0.0
            go[idx] = 1.0
            continue
        t_ego_in = float(risk_d_ego) / max(v, 1e-6)
        t_ego_out = float(risk_d_ego + ego_cross_occ_len) / max(v, 1e-6)
        gap_before = float(t_bg) - float(t_ego_out)
        gap_after = float(t_ego_in) - float(t_bg_exit)
        yld[idx] = float(np.clip((gap - gap_after) / gap, 0.0, 1.0))
        go[idx] = float(np.clip((gap - gap_before) / gap, 0.0, 1.0))
    return yld.astype(np.float32, copy=False), go.astype(np.float32, copy=False)


def _ensure_debug_speed_curve(sample):
    stage1_debug = sample.get("stage1_speed_debug")
    if not isinstance(stage1_debug, dict):
        return None
    speed_curve = stage1_debug.get("speed_curve")
    if not isinstance(speed_curve, dict):
        speed_curve = {}
        stage1_debug["speed_curve"] = speed_curve
    return speed_curve


def main():
    parser = argparse.ArgumentParser(description="Add junction-cross yld/go fields to an existing samples_packed.pkl.")
    parser.add_argument("--input_path", required=True, help="Existing samples_packed.pkl")
    parser.add_argument("--output_path", required=True, help="Output pickle path")
    parser.add_argument("--cross_safe_gap_s", type=float, default=DEFAULT_CROSS_SAFE_GAP_S)
    parser.add_argument("--overwrite_existing", action="store_true", help="Overwrite existing top-level/debug junction-cross fields")
    args = parser.parse_args()

    with open(args.input_path, "rb") as f:
        samples = pickle.load(f)

    total = 0
    added_top = 0
    added_debug = 0
    skipped_existing = 0
    junction_samples = 0

    for sample in samples:
        total += 1
        stage1_debug = sample.get("stage1_speed_debug")
        speed_curve = stage1_debug.get("speed_curve", {}) if isinstance(stage1_debug, dict) else {}
        meet_debug = speed_curve.get("meet_debug", {}) if isinstance(speed_curve, dict) else {}
        if str(meet_debug.get("subtype", "none")) == "junction_left_cross_meet":
            junction_samples += 1

        top_has_yld = "speed_risk_junction_cross_yld_values" in sample
        top_has_go = "speed_risk_junction_cross_go_values" in sample

        debug_speed_curve = _ensure_debug_speed_curve(sample)
        debug_has_yld = isinstance(debug_speed_curve, dict) and "junction_cross_yld_risks" in debug_speed_curve
        debug_has_go = isinstance(debug_speed_curve, dict) and "junction_cross_go_risks" in debug_speed_curve

        if (
            not args.overwrite_existing and
            top_has_yld and top_has_go and
            (debug_speed_curve is None or (debug_has_yld and debug_has_go))
        ):
            skipped_existing += 1
            continue

        yld, go = _derive_junction_cross_split(sample, cross_safe_gap_s=float(args.cross_safe_gap_s))

        if args.overwrite_existing or not top_has_yld:
            sample["speed_risk_junction_cross_yld_values"] = yld.astype(np.float32)
            added_top += 1
        if args.overwrite_existing or not top_has_go:
            sample["speed_risk_junction_cross_go_values"] = go.astype(np.float32)
            added_top += 1

        if isinstance(debug_speed_curve, dict):
            if args.overwrite_existing or not debug_has_yld:
                debug_speed_curve["junction_cross_yld_risks"] = yld.astype(np.float32).tolist()
                added_debug += 1
            if args.overwrite_existing or not debug_has_go:
                debug_speed_curve["junction_cross_go_risks"] = go.astype(np.float32).tolist()
                added_debug += 1

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        "postprocess junction split done: "
        f"samples={total} junction_samples={junction_samples} "
        f"added_top={added_top} added_debug={added_debug} skipped_existing={skipped_existing} "
        f"output={args.output_path}"
    )


if __name__ == "__main__":
    main()
