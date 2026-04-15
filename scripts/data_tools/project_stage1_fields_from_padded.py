#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict


STAGE1_FIELDS = (
    "speed_sample_values",
    "speed_sample_valid_mask",
    "speed_sample_exp_index",
    "speed_risk_chase_values",
    "speed_risk_meet_values",
    "speed_risk_junction_cross_yld_values",
    "speed_risk_junction_cross_go_values",
    "speed_risk_merge_yld_values",
    "speed_risk_merge_go_values",
    "speed_risk_borrow_yld_values",
    "speed_risk_borrow_go_values",
    "speed_risk_ped_values",
    "speed_cross_wait_time_s",
    "speed_cross_wait_valid",
    "junction_cross_episode_id",
    "junction_cross_episode_active",
    "junction_cross_episode_start_frame",
    "junction_cross_episode_end_frame",
    "borrow_cross_decision_phase",
    "borrow_cross_episode_id",
    "borrow_cross_episode_active",
    "borrow_cross_active_time_s",
    "borrow_cross_episode_start_frame",
    "borrow_cross_episode_end_frame",
    "borrow_cross_go_frame",
    "borrow_cross_context_frame",
    "merge_decision_phase",
    "merge_episode_id",
    "merge_episode_active",
    "merge_episode_no_go",
    "merge_episode_start_frame",
    "merge_episode_end_frame",
    "merge_go_frame",
    "merge_resolution_actor_id",
    "merge_end_state",
    "merge_hold",
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
