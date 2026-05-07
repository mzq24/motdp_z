#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict


STAGE1_FIELDS = (
    "ego_status",
    "conflict_area_family",
    "conflict_area_dir",
    "conflict_area_active",
    "conflict_area_start_frame",
    "conflict_area_end_frame",
    "conflict_dist_to_entry_m",
    "conflict_dist_to_exit_m",
    "conflict_time_to_entry_s",
    "conflict_area_status",
    "conflict_decision_phase",
    "conflict_control_phase",
    "conflict_go_frame",
    "merge_yld_max_speed",
    "merge_go_min_speed",
    "merge_yld_max_speed_valid",
    "merge_go_min_speed_valid",
    "merge_threshold_train_only_negative_tail",
    "merge_follow_through_vbmin",
    "merge_follow_through_vbmin_valid",
    "merge_follow_through_vbmin_actor_id",
    "merge_follow_through_vbmin_actor_valid",
    "boundary_speed_consistency_valid",
    "boundary_speed_consistency_issue",
    "boundary_speed_consistency_issue_flag",
    "boundary_speed_consistency_required_action",
    "boundary_speed_consistency_speed_delta_mps",
    "chase_max_speed",
    "chase_max_speed_valid",
    "chase_has_lead",
    "chase_status",
    "chase_dist_m",
    "chase_dist_valid",
    "chase_ttc_s",
    "chase_ttc_valid",
    "chase_speed_max",
    "chase_speed_max_valid",
    "borrow_yld_max_speed",
    "borrow_go_min_speed",
    "borrow_yld_max_speed_valid",
    "borrow_go_min_speed_valid",
    "junction_yld_max_speed",
    "junction_go_min_speed",
    "junction_yld_max_speed_valid",
    "junction_go_min_speed_valid",
    "temporary_occupancy_cover_bins",
    "temporary_occupancy_cover_valid",
    "go_opportunity_prob",
    "yld_pressure_prob",
    "go_opportunity_valid",
    "conflict_phase_ref_role",
    "conflict_phase_ref_actor_id",
    "conflict_phase_ref_actor_valid",
    "conflict_phase_open_unbounded",
    "conflict_phase_boundary_ref_actor_id",
    "conflict_phase_boundary_ref_valid",
    "conflict_phase_boundary_ref_role",
    "conflict_phase_boundary_mode",
    "conflict_phase_boundary_state_valid",
    "conflict_phase_boundary_object_missing",
    "conflict_phase_boundary_scalar_loss_valid",
    "conflict_phase_boundary_actor_match",
    "stage1_speed_debug",
)


def _sample_key(sample: Dict) -> str:
    feat_rel = str(sample.get("transfuser_bev_feature", "") or "")
    route_name = str(sample.get("route_name", "") or "")
    frame_id = int(sample.get("frame_id", -1))
    return f"{feat_rel}|{frame_id}" if feat_rel else f"{route_name}|{frame_id}"


def _atomic_pickle_save(obj, target_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = target_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, target_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project stage1 labels from a relabeled padded packed dataset back onto the "
            "original trimmed packed dataset."
        )
    )
    parser.add_argument("--base", required=True, help="Original trimmed samples_packed.pkl")
    parser.add_argument("--padded_relabel", required=True, help="Relabeled padded samples_packed.pkl")
    parser.add_argument("--output", required=True, help="Projected output samples_packed.pkl")
    parser.add_argument(
        "--summary_json",
        default=None,
        help="Optional summary JSON path. Defaults to <output>.summary.json",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = os.path.realpath(args.output)
    summary_json = os.path.realpath(args.summary_json or (args.output + ".summary.json"))

    if not args.overwrite and (os.path.exists(output_path) or os.path.exists(summary_json)):
        raise FileExistsError("Output exists; pass --overwrite to replace it.")

    with open(args.base, "rb") as f:
        base_samples = pickle.load(f)
    with open(args.padded_relabel, "rb") as f:
        padded_samples = pickle.load(f)
    if not isinstance(base_samples, list):
        raise TypeError(f"Expected list in {args.base}, got {type(base_samples).__name__}")
    if not isinstance(padded_samples, list):
        raise TypeError(f"Expected list in {args.padded_relabel}, got {type(padded_samples).__name__}")

    base_index = {}
    for idx, sample in enumerate(base_samples):
        key = _sample_key(sample)
        if key in base_index:
            raise ValueError(f"duplicate key in base dataset: {key}")
        base_index[key] = idx

    replaced_keys = set()
    ignored_padded_only = 0
    for sample in padded_samples:
        key = _sample_key(sample)
        base_idx = base_index.get(key)
        if base_idx is None:
            ignored_padded_only += 1
            continue
        dst = base_samples[base_idx]
        for field in STAGE1_FIELDS:
            if field in sample:
                dst[field] = sample[field]
        replaced_keys.add(key)

    if len(replaced_keys) != len(base_samples):
        raise RuntimeError(
            "projection coverage mismatch: "
            f"replaced={len(replaced_keys)} base={len(base_samples)}"
        )

    _atomic_pickle_save(base_samples, output_path)
    summary = {
        "base": os.path.realpath(args.base),
        "padded_relabel": os.path.realpath(args.padded_relabel),
        "output": output_path,
        "base_samples": len(base_samples),
        "replaced_samples": len(replaced_keys),
        "ignored_padded_only_samples": int(ignored_padded_only),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
