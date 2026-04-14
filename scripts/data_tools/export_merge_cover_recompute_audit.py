#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np

from tools.generate_front_route_label_video import (
    _collect_dynamic_actor_ids,
    _collect_scene_nonstatic_actor_ids,
    _compute_front_route_label,
    _cover_candidate_summary,
    _filter_current_boxes_dynamic,
    _filter_future_frames_dynamic,
    _interaction_name,
    _interaction_subtype,
    _load_future_frames,
    _load_json_gz_if_exists,
    _resolve_feature_frame_info,
    _scene_name_from_base_dir,
)


def _atomic_json_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)
    os.replace(tmp_path, path)


def _frame_id_from_sample(sample: Dict[str, Any]) -> int:
    return int(sample.get("frame_id", -1))


def _base_dir_from_sample(sample: Dict[str, Any]) -> str:
    feat = str(sample.get("transfuser_bev_feature", "") or "")
    return os.path.dirname(os.path.dirname(feat)) if feat else ""


def _group_indices_by_scene(samples: List[Dict[str, Any]]) -> List[Tuple[str, List[int]]]:
    scene_to_indices: Dict[str, List[int]] = {}
    scene_order: List[str] = []
    for idx, sample in enumerate(samples):
        base_dir = _base_dir_from_sample(sample)
        if base_dir not in scene_to_indices:
            scene_to_indices[base_dir] = []
            scene_order.append(base_dir)
        scene_to_indices[base_dir].append(idx)
    grouped: List[Tuple[str, List[int]]] = []
    for base_dir in scene_order:
        indices = scene_to_indices[base_dir]
        indices.sort(key=lambda i: _frame_id_from_sample(samples[i]))
        grouped.append((base_dir, indices))
    return grouped


def _is_merge_cover(cover: Dict[str, Any]) -> bool:
    if int((cover or {}).get("exists", 0.0)) <= 0:
        return False
    if _interaction_name(cover) != "meet":
        return False
    return _interaction_subtype(cover) == "merge_meet"


def _cover_summary(cover: Dict[str, Any]) -> Dict[str, Any]:
    interaction = ((cover or {}).get("interaction") or {})
    return {
        "exists": int((cover or {}).get("exists", 0.0)),
        "actor_id": int((cover or {}).get("actor_id", -1)),
        "interaction_name": str(interaction.get("name", "none")),
        "interaction_subtype": str(interaction.get("subtype") or interaction.get("name") or "none"),
        "distance": None if not np.isfinite(float((cover or {}).get("distance", np.nan))) else float((cover or {}).get("distance", np.nan)),
        "d_ego": None if not np.isfinite(float((cover or {}).get("d_ego", np.nan))) else float((cover or {}).get("d_ego", np.nan)),
        "d_bg": None if not np.isfinite(float((cover or {}).get("d_bg", np.nan))) else float((cover or {}).get("d_bg", np.nan)),
        "source": str(interaction.get("source", "none")),
    }


def _packed_raw_merge_cover(sample: Dict[str, Any]) -> bool:
    stage1_debug = sample.get("stage1_speed_debug")
    if not isinstance(stage1_debug, dict):
        return False
    return _is_merge_cover(stage1_debug.get("current_cover") or {}) or _is_merge_cover(stage1_debug.get("future_cover") or {})


def _packed_cover_pair(sample: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    stage1_debug = sample.get("stage1_speed_debug")
    if not isinstance(stage1_debug, dict):
        stage1_debug = {}
    return dict(stage1_debug.get("current_cover", {})), dict(stage1_debug.get("future_cover", {}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit packed merge cover vs recomputed merge cover.")
    parser.add_argument("--packed_path", required=True)
    parser.add_argument("--image_data_root", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--frames_jsonl", required=True)
    parser.add_argument("--scene_name", default=None)
    parser.add_argument("--route_name", default=None)
    parser.add_argument("--num_future", type=int, default=6)
    parser.add_argument("--front_corridor_margin_m", type=float, default=0.5)
    parser.add_argument("--front_route_step_m", type=float, default=0.25)
    parser.add_argument("--front_max_distance_m", type=float, default=32.0)
    parser.add_argument("--front_safe_ttc_s", type=float, default=3.0)
    parser.add_argument("--front_max_ttc_s", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.packed_path, "rb") as f:
        samples = pickle.load(f)

    grouped_scenes = _group_indices_by_scene(samples)
    if args.scene_name is not None:
        grouped_scenes = [
            (scene, indices) for scene, indices in grouped_scenes
            if _scene_name_from_base_dir(scene) == args.scene_name
        ]
    if args.route_name is not None:
        grouped_scenes = [
            (scene, indices) for scene, indices in grouped_scenes
            if os.path.basename(scene) == args.route_name
        ]

    summary_records = []
    frames_dir = os.path.dirname(os.path.abspath(args.frames_jsonl))
    os.makedirs(frames_dir, exist_ok=True)
    tmp_jsonl = args.frames_jsonl + f".tmp.{os.getpid()}"

    with open(tmp_jsonl, "w", encoding="utf-8") as f_jsonl:
        for scene_name, indices in grouped_scenes:
            route_samples = [samples[idx] for idx in indices]
            scene_nonstatic_actor_ids = _collect_scene_nonstatic_actor_ids(
                route_samples,
                image_root=args.image_data_root,
                speed_thresh_mps=0.25,
                motion_thresh_m=1.0,
            )

            packed_ids: List[int] = []
            recomputed_ids: List[int] = []
            packed_only_ids: List[int] = []
            recomputed_only_ids: List[int] = []

            for sample in route_samples:
                base_dir, frame_str = _resolve_feature_frame_info(sample)
                if base_dir is None or frame_str is None:
                    continue
                current_boxes = _load_json_gz_if_exists(os.path.join(args.image_data_root, base_dir, "boxes", f"{frame_str}.json.gz"))
                current_meas = _load_json_gz_if_exists(os.path.join(args.image_data_root, base_dir, "measurements", f"{frame_str}.json.gz"))
                if current_boxes is None or current_meas is None:
                    continue

                num_future = args.num_future
                if "ego_waypoints" in sample:
                    try:
                        num_future = max(1, min(num_future, len(sample["ego_waypoints"]) - 1))
                    except Exception:
                        pass
                future_frames = _load_future_frames(args.image_data_root, base_dir, int(sample["frame_id"]), num_future)
                dynamic_ids = _collect_dynamic_actor_ids(
                    current_boxes=current_boxes,
                    future_frames_data=future_frames,
                    ego_matrix_current=current_meas.get("ego_matrix", None),
                    speed_thresh_mps=0.25,
                    motion_thresh_m=1.0,
                )
                dynamic_ids = set(dynamic_ids) | set(scene_nonstatic_actor_ids)
                label_current_boxes = _filter_current_boxes_dynamic(current_boxes, dynamic_ids)
                label_future_frames = _filter_future_frames_dynamic(future_frames, dynamic_ids, speed_thresh_mps=0.25)

                _, debug = _compute_front_route_label(
                    route=np.asarray(sample["route"], dtype=np.float32),
                    current_boxes=label_current_boxes,
                    ego_speed=float(current_meas.get("speed", 0.0)),
                    ego_matrix_current=current_meas.get("ego_matrix"),
                    future_frames_data=label_future_frames,
                    corridor_margin_m=args.front_corridor_margin_m,
                    route_step_m=args.front_route_step_m,
                    max_distance_m=args.front_max_distance_m,
                    safe_ttc_s=args.front_safe_ttc_s,
                    max_ttc_s=args.front_max_ttc_s,
                    return_debug=True,
                )

                event_name = _scene_name_from_base_dir(base_dir)
                recomputed_current = _cover_candidate_summary(1, debug.get("best_current"), debug, current_meas=current_meas, event_name=event_name)
                recomputed_future = _cover_candidate_summary(2, debug.get("best_future"), debug, current_meas=current_meas, event_name=event_name)
                recomputed_merge = _is_merge_cover(recomputed_current) or _is_merge_cover(recomputed_future)

                packed_current, packed_future = _packed_cover_pair(sample)
                packed_merge = _is_merge_cover(packed_current) or _is_merge_cover(packed_future)

                frame_id = int(sample.get("frame_id", -1))
                if packed_merge:
                    packed_ids.append(frame_id)
                if recomputed_merge:
                    recomputed_ids.append(frame_id)
                if packed_merge and not recomputed_merge:
                    packed_only_ids.append(frame_id)
                if recomputed_merge and not packed_merge:
                    recomputed_only_ids.append(frame_id)

                f_jsonl.write(json.dumps({
                    "scene": scene_name,
                    "frame_id": frame_id,
                    "packed_raw_merge_cover": bool(packed_merge),
                    "recomputed_raw_merge_cover": bool(recomputed_merge),
                    "packed_current_cover": _cover_summary(packed_current),
                    "packed_raw_future_cover": _cover_summary(packed_future),
                    "recomputed_current_cover": _cover_summary(recomputed_current),
                    "recomputed_raw_future_cover": _cover_summary(recomputed_future),
                }, ensure_ascii=True) + "\n")

            packed_ids = sorted(set(packed_ids))
            recomputed_ids = sorted(set(recomputed_ids))
            packed_only_ids = sorted(set(packed_only_ids))
            recomputed_only_ids = sorted(set(recomputed_only_ids))
            summary_records.append({
                "scene": scene_name,
                "num_frames": int(len(route_samples)),
                "packed_raw_merge_cover_frame_count": int(len(packed_ids)),
                "packed_raw_merge_cover_frame_ids": packed_ids,
                "recomputed_raw_merge_cover_frame_count": int(len(recomputed_ids)),
                "recomputed_raw_merge_cover_frame_ids": recomputed_ids,
                "packed_only_frame_count": int(len(packed_only_ids)),
                "packed_only_frame_ids": packed_only_ids,
                "recomputed_only_frame_count": int(len(recomputed_only_ids)),
                "recomputed_only_frame_ids": recomputed_only_ids,
            })

    os.replace(tmp_jsonl, args.frames_jsonl)
    _atomic_json_dump(summary_records, args.summary_json)


if __name__ == "__main__":
    main()
