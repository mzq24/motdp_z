#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


MERGE_DECISION_PHASE_FROM_CODE = {
    0: "none",
    1: "yld",
    2: "go",
}

MERGE_END_STATE_FROM_CODE = {
    0: "none",
    1: "ended_with_chase",
    2: "ended_with_cross",
    3: "ended_with_other_current_actor",
    4: "ended_empty",
    5: "ended_route_end",
}


def _atomic_json_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)
    os.replace(tmp_path, path)


def _sanitize_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def _sanitize_list(values: Iterable[Any]) -> List[Any]:
    return [_sanitize_scalar(v) for v in values]


def _base_dir_from_sample(sample: Dict[str, Any]) -> str:
    feat = str(sample.get("transfuser_bev_feature", "") or "")
    return os.path.dirname(os.path.dirname(feat)) if feat else ""


def _frame_id_from_sample(sample: Dict[str, Any]) -> int:
    return int(sample.get("frame_id", -1))


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


def _interaction_name(cover: Dict[str, Any]) -> str:
    return str(((cover or {}).get("interaction") or {}).get("name", "none"))


def _interaction_subtype(cover: Dict[str, Any]) -> str:
    inter = ((cover or {}).get("interaction") or {})
    return str(inter.get("subtype") or inter.get("name") or "none")


def _is_merge_cover(cover: Dict[str, Any]) -> bool:
    if int((cover or {}).get("exists", 0.0)) <= 0:
        return False
    if _interaction_name(cover) != "meet":
        return False
    return _interaction_subtype(cover) == "merge_meet"


def _sample_current_speed_mps(sample: Dict[str, Any]) -> float:
    speed_hist = np.asarray(sample.get("speed_hist"), dtype=np.float32).reshape(-1)
    if speed_hist.size > 0:
        v = float(speed_hist[-1])
        if np.isfinite(v):
            return v
    ego_status = np.asarray(sample.get("ego_status"), dtype=np.float32)
    if ego_status.ndim >= 2 and ego_status.shape[-1] >= 1:
        v = float(ego_status.reshape(-1, ego_status.shape[-1])[-1, 0])
        if np.isfinite(v):
            return v
    return 0.0


def _cover_summary(cover: Dict[str, Any]) -> Dict[str, Any]:
    inter = ((cover or {}).get("interaction") or {})
    return {
        "exists": int((cover or {}).get("exists", 0.0)),
        "actor_id": int((cover or {}).get("actor_id", -1)),
        "interaction_name": str(inter.get("name", "none")),
        "interaction_subtype": str(inter.get("subtype") or inter.get("name") or "none"),
        "distance": _sanitize_scalar((cover or {}).get("distance", None)),
        "d_ego": _sanitize_scalar((cover or {}).get("d_ego", None)),
        "d_bg": _sanitize_scalar((cover or {}).get("d_bg", None)),
        "other_speed": _sanitize_scalar((cover or {}).get("other_speed", None)),
        "rear_gap_m": _sanitize_scalar((cover or {}).get("rear_gap_m", None)),
        "other_length_m": _sanitize_scalar((cover or {}).get("other_length_m", None)),
        "angle_deg": _sanitize_scalar(inter.get("angle_deg", None)),
    }


def _merge_meet_debug_summary(speed_curve: Dict[str, Any]) -> Dict[str, Any]:
    meet_debug = (speed_curve or {}).get("meet_debug", {})
    if not isinstance(meet_debug, dict):
        meet_debug = {}
    keys = [
        "valid",
        "subtype",
        "cover_case",
        "d_ego_m",
        "d_bg_m",
        "conflict_len_m",
        "context_conflict_len_m",
        "ego_clearance_m",
        "bg_clearance_m",
        "bg_speed_mps",
        "t_bg_s",
        "t_bg_exit_s",
        "t_bg_clear_s",
        "safe_gap_bg_m",
        "rear_gap_m",
        "v_equal_mps",
        "v_go_min_mps",
        "v_behind_min_mps",
        "v_go_need_mps",
        "v_yield_max_mps",
    ]
    out = {}
    for key in keys:
        out[key] = _sanitize_scalar(meet_debug.get(key))
    return out


def _sample_has_merge_signal(sample: Dict[str, Any]) -> bool:
    if float(sample.get("merge_episode_active", 0.0)) > 0.5:
        return True
    if int(sample.get("merge_episode_id", -1)) >= 0:
        return True
    if int(sample.get("merge_go_frame", -1)) >= 0:
        return True
    if float(sample.get("merge_episode_no_go", 0.0)) > 0.5:
        return True
    stage1_debug = sample.get("stage1_speed_debug")
    if not isinstance(stage1_debug, dict):
        return False
    if _is_merge_cover(stage1_debug.get("current_cover") or {}):
        return True
    if _is_merge_cover(stage1_debug.get("speed_curve_future_cover") or {}):
        return True
    speed_curve = stage1_debug.get("speed_curve")
    if isinstance(speed_curve, dict):
        meet_debug = speed_curve.get("meet_debug", {})
        if isinstance(meet_debug, dict) and str(meet_debug.get("subtype", "none")) == "merge_meet":
            return True
    return False


def _frame_debug_record(sample: Dict[str, Any]) -> Dict[str, Any]:
    stage1_debug = sample.get("stage1_speed_debug")
    if not isinstance(stage1_debug, dict):
        stage1_debug = {}
    speed_curve = stage1_debug.get("speed_curve")
    if not isinstance(speed_curve, dict):
        speed_curve = {}

    sample_speeds = np.asarray(sample.get("speed_sample_values", speed_curve.get("sample_speeds_mps", [])), dtype=np.float32).reshape(-1)
    merge_yld = np.asarray(sample.get("speed_risk_merge_yld_values", speed_curve.get("merge_yld_risks", [])), dtype=np.float32).reshape(-1)
    merge_go = np.asarray(sample.get("speed_risk_merge_go_values", speed_curve.get("merge_go_risks", [])), dtype=np.float32).reshape(-1)
    meet_risks = np.asarray(sample.get("speed_risk_meet_values", speed_curve.get("meet_risks", [])), dtype=np.float32).reshape(-1)

    return {
        "frame_id": int(sample.get("frame_id", -1)),
        "current_speed_mps": _sanitize_scalar(_sample_current_speed_mps(sample)),
        "merge_episode_active": float(sample.get("merge_episode_active", 0.0)),
        "merge_episode_start_frame": int(sample.get("merge_episode_start_frame", -1)),
        "merge_episode_end_frame": int(sample.get("merge_episode_end_frame", -1)),
        "merge_go_frame": int(sample.get("merge_go_frame", -1)),
        "merge_episode_no_go": float(sample.get("merge_episode_no_go", 0.0)),
        "merge_decision_phase_code": int(sample.get("merge_decision_phase", 0)),
        "merge_decision_phase": MERGE_DECISION_PHASE_FROM_CODE.get(int(sample.get("merge_decision_phase", 0)), "none"),
        "merge_resolution_actor_id": int(sample.get("merge_resolution_actor_id", -1)),
        "merge_end_state_code": int(sample.get("merge_end_state", 0)),
        "merge_end_state": MERGE_END_STATE_FROM_CODE.get(int(sample.get("merge_end_state", 0)), "none"),
        "current_cover": _cover_summary(stage1_debug.get("current_cover") or {}),
        "future_cover": _cover_summary(stage1_debug.get("speed_curve_future_cover") or {}),
        "merge_meet_debug": _merge_meet_debug_summary(speed_curve),
        "sample_speeds_mps": _sanitize_list(sample_speeds.tolist()),
        "meet_risks": _sanitize_list(meet_risks.tolist()),
        "merge_yld_risks": _sanitize_list(merge_yld.tolist()),
        "merge_go_risks": _sanitize_list(merge_go.tolist()),
    }


def _episode_summary(samples: List[Dict[str, Any]], indices: List[int], episode_id: int) -> Dict[str, Any]:
    episode_indices = [idx for idx in indices if int(samples[idx].get("merge_episode_id", -1)) == int(episode_id)]
    episode_indices.sort(key=lambda idx: _frame_id_from_sample(samples[idx]))
    first = samples[episode_indices[0]]
    last = samples[episode_indices[-1]]
    phase_counter = Counter(
        MERGE_DECISION_PHASE_FROM_CODE.get(int(samples[idx].get("merge_decision_phase", 0)), "none")
        for idx in episode_indices
    )
    return {
        "episode_id": int(episode_id),
        "frame_count": int(len(episode_indices)),
        "frame_range": [int(_frame_id_from_sample(first)), int(_frame_id_from_sample(last))],
        "start_frame": int(first.get("merge_episode_start_frame", -1)),
        "end_frame": int(first.get("merge_episode_end_frame", -1)),
        "go_frame": int(first.get("merge_go_frame", -1)),
        "no_go": float(first.get("merge_episode_no_go", 0.0)),
        "resolution_actor_id": int(first.get("merge_resolution_actor_id", -1)),
        "end_state_code": int(first.get("merge_end_state", 0)),
        "end_state": MERGE_END_STATE_FROM_CODE.get(int(first.get("merge_end_state", 0)), "none"),
        "phase_counts": dict(phase_counter),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export merge episode/cover/threshold/energy debug data from samples_packed.pkl.")
    parser.add_argument("--packed_path", required=True)
    parser.add_argument("--summary_json", required=True, help="Scene/episode summary JSON output path.")
    parser.add_argument("--frames_jsonl", required=True, help="Per-frame merge debug JSONL output path.")
    parser.add_argument("--merge_signal_only", action="store_true", help="Export only scenes/frames with merge-related signals.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.packed_path, "rb") as f:
        samples = pickle.load(f)
    grouped_scenes = _group_indices_by_scene(samples)

    scene_summaries = []
    frames_dir = os.path.dirname(os.path.abspath(args.frames_jsonl))
    os.makedirs(frames_dir, exist_ok=True)
    tmp_jsonl = args.frames_jsonl + f".tmp.{os.getpid()}"

    total_merge_signal_frames = 0
    merge_signal_scenes = 0

    with open(tmp_jsonl, "w", encoding="utf-8") as f_jsonl:
        for scene_name, indices in grouped_scenes:
            signal_indices = [idx for idx in indices if _sample_has_merge_signal(samples[idx])]
            if args.merge_signal_only and not signal_indices:
                continue
            if signal_indices:
                merge_signal_scenes += 1
            total_merge_signal_frames += len(signal_indices)

            episode_ids = sorted(
                set(
                    int(samples[idx].get("merge_episode_id", -1))
                    for idx in signal_indices
                    if int(samples[idx].get("merge_episode_id", -1)) >= 0
                )
            )

            scene_summary = {
                "scene": scene_name,
                "num_frames": int(len(indices)),
                "merge_signal_frame_count": int(len(signal_indices)),
                "merge_active_frame_count": int(
                    sum(float(samples[idx].get("merge_episode_active", 0.0)) > 0.5 for idx in indices)
                ),
                "episode_ids": [int(episode_id) for episode_id in episode_ids],
                "episodes": [
                    _episode_summary(samples, indices, episode_id)
                    for episode_id in episode_ids
                ],
            }
            if signal_indices:
                frame_ids = [_frame_id_from_sample(samples[idx]) for idx in signal_indices]
                scene_summary["merge_signal_frame_range"] = [int(min(frame_ids)), int(max(frame_ids))]
            scene_summaries.append(scene_summary)

            for idx in signal_indices:
                record = {
                    "scene": scene_name,
                    **_frame_debug_record(samples[idx]),
                }
                f_jsonl.write(json.dumps(record, ensure_ascii=True) + "\n")

    os.replace(tmp_jsonl, args.frames_jsonl)
    summary_payload = {
        "packed_path": args.packed_path,
        "total_samples": int(len(samples)),
        "source_total_scenes": int(len(grouped_scenes)),
        "exported_scenes": int(len(scene_summaries)),
        "merge_signal_scenes": int(merge_signal_scenes),
        "merge_signal_frames": int(total_merge_signal_frames),
        "scenes": scene_summaries,
    }
    _atomic_json_dump(summary_payload, args.summary_json)

    print(
        {
            "total_samples": len(samples),
            "source_total_scenes": len(grouped_scenes),
            "exported_scenes": len(summary_payload["scenes"]),
            "merge_signal_scenes": merge_signal_scenes,
            "merge_signal_frames": total_merge_signal_frames,
            "summary_json": args.summary_json,
            "frames_jsonl": args.frames_jsonl,
        }
    )


if __name__ == "__main__":
    main()
