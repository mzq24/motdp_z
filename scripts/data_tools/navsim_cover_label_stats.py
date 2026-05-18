#!/usr/bin/env python3
"""Prototype NAVSIM cover-based label statistics.

This script is the NAVSIM-facing adapter for the existing MoT-DP stage1 cover
logic. It deliberately starts from bbox-route cover, not from lateral-distance
heuristics:

  NAVSIM raw log -> long future ego path route -> current/future boxes
  -> _compute_front_route_label() -> current_cover / future_cover stats

NAVSIM raw pkl files contain multiple short scene_token chunks, but the
sample_prev/sample_next chain can continue across those boundaries. We process
continuous token chains inside each pkl instead of stopping at scene_token
boundaries, so route/future-cover can use a longer temporal context.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_tools.precompute_semantic_labels import _compute_front_route_label  # noqa: E402


NAVSIM_DT = 0.5
BOX_X = 0
BOX_Y = 1
BOX_Z = 2
BOX_LENGTH = 3
BOX_WIDTH = 4
BOX_HEIGHT = 5
BOX_HEADING = 6

COMMAND_NAMES = ("none", "left", "straight", "right")
CLASS_MAP = {
    "vehicle": "vehicle",
    "bicycle": "bicycle",
    "pedestrian": "pedestrian",
}


def frame_transform(frame: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(frame["ego2global"], dtype=np.float64)


def command_name(command: Any) -> str:
    arr = np.asarray(command).astype(int).reshape(-1)
    if arr.size == 4 and int(arr.sum()) == 1:
        return COMMAND_NAMES[int(arr.argmax())]
    return "other"


def coerce_boxes(value: Any) -> np.ndarray:
    arr = np.asarray(value if value is not None else [], dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def coerce_velocities(value: Any) -> np.ndarray:
    arr = np.asarray(value if value is not None else [], dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def anns(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    return frame.get("anns", {})


def position_xy(frame: Mapping[str, Any]) -> np.ndarray:
    transform = frame_transform(frame)
    return np.asarray([transform[0, 3], transform[1, 3]], dtype=np.float64)


def ego_speed_mps(frames: Sequence[Mapping[str, Any]], idx: int, segment_start: int, segment_end: int) -> float:
    if idx > segment_start:
        dt = max((int(frames[idx]["timestamp"]) - int(frames[idx - 1]["timestamp"])) / 1e6, 1e-3)
        return float(np.linalg.norm(position_xy(frames[idx]) - position_xy(frames[idx - 1])) / dt)
    if idx < segment_end:
        dt = max((int(frames[idx + 1]["timestamp"]) - int(frames[idx]["timestamp"])) / 1e6, 1e-3)
        return float(np.linalg.norm(position_xy(frames[idx + 1]) - position_xy(frames[idx])) / dt)
    return 0.0


def build_track_id_map(frames: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    next_id = 1
    for frame in frames:
        for token in anns(frame).get("track_tokens", []) or []:
            token = str(token)
            if token and token not in mapping:
                mapping[token] = next_id
                next_id += 1
    return mapping


def navsim_boxes_to_stage1(
    frame: Mapping[str, Any],
    track_id_map: Mapping[str, int],
    include_ego: bool = True,
) -> List[Dict[str, Any]]:
    frame_anns = anns(frame)
    boxes_np = coerce_boxes(frame_anns.get("gt_boxes", []))
    names = list(map(str, frame_anns.get("gt_names", [])))
    velocities = coerce_velocities(frame_anns.get("gt_velocity_3d", []))
    track_tokens = list(map(str, frame_anns.get("track_tokens", [])))

    boxes: List[Dict[str, Any]] = []
    if include_ego:
        boxes.append(
            {
                "id": 0,
                "class": "ego_car",
                "position": [0.0, 0.0, 0.0],
                "extent": [2.5, 1.0, 0.8],
                "yaw": 0.0,
                "speed": 0.0,
            }
        )

    for idx, box in enumerate(boxes_np):
        raw_name = names[idx] if idx < len(names) else "unknown"
        class_name = CLASS_MAP.get(raw_name)
        if class_name is None:
            continue
        if box.shape[0] <= BOX_HEADING:
            continue
        token = track_tokens[idx] if idx < len(track_tokens) else ""
        actor_id = track_id_map.get(token) if token else None
        speed = float(np.linalg.norm(velocities[idx, :2])) if idx < len(velocities) else 0.0
        boxes.append(
            {
                "id": actor_id,
                "track_token": token,
                "class": class_name,
                "position": [float(box[BOX_X]), float(box[BOX_Y]), float(box[BOX_Z])],
                # MoT-DP stage1 helpers expect half extents; NAVSIM boxes store full size.
                "extent": [
                    float(box[BOX_LENGTH]) * 0.5,
                    float(box[BOX_WIDTH]) * 0.5,
                    float(box[BOX_HEIGHT]) * 0.5,
                ],
                "yaw": float(box[BOX_HEADING]),
                "speed": speed,
            }
        )
    return boxes


def split_contiguous_segments(
    frames: Sequence[Mapping[str, Any]],
    min_dt_s: float = 0.35,
    max_dt_s: float = 0.75,
) -> List[Tuple[int, int]]:
    if not frames:
        return []
    segments: List[Tuple[int, int]] = []
    start = 0
    for idx in range(1, len(frames)):
        prev = frames[idx - 1]
        cur = frames[idx]
        dt = (int(cur["timestamp"]) - int(prev["timestamp"])) / 1e6
        linked = (
            prev.get("sample_next") == cur.get("token")
            and cur.get("sample_prev") == prev.get("token")
            and min_dt_s <= dt <= max_dt_s
        )
        if not linked:
            segments.append((start, idx - 1))
            start = idx
    segments.append((start, len(frames) - 1))
    return segments


def relative_future_points(
    frames: Sequence[Mapping[str, Any]],
    current_idx: int,
    segment_end: int,
    lookahead_frames: int,
) -> np.ndarray:
    origin = frame_transform(frames[current_idx])
    origin_inv = np.linalg.inv(origin)
    end = min(segment_end, current_idx + int(lookahead_frames))
    positions = []
    for idx in range(current_idx, end + 1):
        transform = frame_transform(frames[idx])
        point = origin_inv @ np.array([transform[0, 3], transform[1, 3], transform[2, 3], 1.0], dtype=np.float64)
        positions.append([point[0], point[1]])
    return np.asarray(positions, dtype=np.float32)


def sample_path_by_distance(points: np.ndarray, step_m: float, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    path = np.zeros((int(num_points), 2), dtype=np.float32)
    mask = np.zeros(int(num_points), dtype=bool)
    if points.shape[0] == 0:
        return path, mask
    if points.shape[0] == 1:
        path[:] = points[0, :2]
        return path, mask

    seg_lens = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
    keep = np.concatenate([[True], np.diff(dist) > 1e-3])
    dist = dist[keep]
    points = points[keep]
    if points.shape[0] < 2:
        path[:] = points[-1, :2]
        return path, mask

    targets = np.arange(step_m, step_m * (int(num_points) + 1), step_m, dtype=np.float32)
    path[:, 0] = np.interp(targets, dist, points[:, 0])
    path[:, 1] = np.interp(targets, dist, points[:, 1])
    valid = targets <= dist[-1]
    mask[:] = valid
    return path, mask


def build_route(
    frames: Sequence[Mapping[str, Any]],
    current_idx: int,
    segment_end: int,
    lookahead_frames: int,
    route_points: int,
    route_point_step_m: float,
) -> Tuple[np.ndarray, int, float, bool]:
    future_points = relative_future_points(frames, current_idx, segment_end, lookahead_frames)
    route, mask = sample_path_by_distance(future_points, route_point_step_m, route_points)
    valid = int(mask.sum())
    length = float(valid * route_point_step_m)
    crosses_scene = any(
        frames[idx].get("scene_token") != frames[idx - 1].get("scene_token")
        for idx in range(current_idx + 1, min(segment_end, current_idx + lookahead_frames) + 1)
    )
    return route[mask], valid, length, bool(crosses_scene)


class ScalarCollector:
    def __init__(self) -> None:
        self.values: List[float] = []

    def add(self, value: float) -> None:
        if np.isfinite(value):
            self.values.append(float(value))

    def as_dict(self) -> Dict[str, float]:
        if not self.values:
            return {"count": 0}
        arr = np.asarray(self.values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        }


class ExampleStore:
    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.keys = set()

    def add(self, category: str, sample: Mapping[str, Any]) -> None:
        key = (str(category), sample.get("log_name"), sample.get("global_idx"), sample.get("token"))
        if key in self.keys:
            return
        bucket = self.items[str(category)]
        if len(bucket) < self.limit:
            bucket.append(dict(sample))
            self.keys.add(key)

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for category, samples in sorted(self.items.items()):
                for sample in samples:
                    f.write(json.dumps({"category": category, **sample}, sort_keys=True) + "\n")

    def counts(self) -> Dict[str, int]:
        return {key: len(value) for key, value in sorted(self.items.items())}


def empty_cover() -> Dict[str, Any]:
    return {
        "exists": 0.0,
        "actor_id": -1,
        "actor_class_name": "none",
        "route_distance_m": float("nan"),
        "ttc_s": float("nan"),
    }


def cover_summary(debug_block: Optional[Mapping[str, Any]], label: Mapping[str, Any], cover_case: str) -> Dict[str, Any]:
    if not debug_block:
        return empty_cover()
    box = debug_block.get("box") or debug_block.get("box_current_frame") or debug_block.get("box_future") or {}
    cover = debug_block.get("cover") or {}
    summary = {
        "exists": 1.0,
        "cover_case": cover_case,
        "actor_id": int(box.get("id", -1)) if box.get("id", None) is not None else -1,
        "track_token": str(box.get("track_token", "")),
        "actor_class_name": str(debug_block.get("actor_class", box.get("class", "unknown"))),
        "route_idx": int(cover.get("route_idx", -1)),
        "route_distance_m": float(cover.get("route_distance", np.nan)),
        "route_point_local_xy": np.asarray(cover.get("route_point", []), dtype=np.float32).reshape(-1)[:2].astype(float).tolist(),
        "ttc_s": float(label.get("ttc", np.nan)),
        "risk": float(label.get("risk", np.nan)),
    }
    for key in ("gap_distance", "d_ego", "d_bg", "meet_dist", "bg_speed", "lead_speed"):
        if key in debug_block:
            summary[key] = float(debug_block.get(key, np.nan))
    if "frame_index" in debug_block:
        summary["future_frame_index"] = int(debug_block.get("frame_index", -1))
    return summary


def serializable_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def init_stats(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "args": serializable_args(args),
        "counts": Counter(),
        "command_counts": Counter(),
        "case_counts": Counter(),
        "actor_class_counts": Counter(),
        "current_cover_route_distance_m": ScalarCollector(),
        "future_cover_route_distance_m": ScalarCollector(),
        "future_cover_d_ego_m": ScalarCollector(),
        "future_cover_d_bg_m": ScalarCollector(),
        "cover_ttc_s": ScalarCollector(),
        "route_valid_points": ScalarCollector(),
        "route_length_m": ScalarCollector(),
    }


def process_window(
    frames: Sequence[Mapping[str, Any]],
    log_name: str,
    current_idx: int,
    segment_start: int,
    segment_end: int,
    track_id_map: Mapping[str, int],
    args: argparse.Namespace,
    stats: Dict[str, Any],
    examples: ExampleStore,
) -> None:
    current = frames[current_idx]
    stats["counts"]["windows_total"] += 1
    if args.require_route and len(current.get("roadblock_ids", [])) == 0:
        stats["counts"]["windows_skipped_no_route"] += 1
        return

    route, route_valid, route_length, route_crosses_scene = build_route(
        frames,
        current_idx,
        segment_end,
        args.route_lookahead_frames,
        args.route_points,
        args.route_point_step_m,
    )
    stats["route_valid_points"].add(route_valid)
    stats["route_length_m"].add(route_length)
    if route_valid < args.min_route_points or route_length < args.min_route_length_m:
        stats["counts"]["windows_skipped_short_route"] += 1
        return

    future_end = current_idx + args.future_cover_frames
    if future_end > segment_end:
        stats["counts"]["windows_skipped_short_future"] += 1
        return

    current_boxes = navsim_boxes_to_stage1(current, track_id_map, include_ego=True)
    future_frames_data = [
        (navsim_boxes_to_stage1(frames[idx], track_id_map, include_ego=False), frame_transform(frames[idx]))
        for idx in range(current_idx + 1, future_end + 1)
    ]
    speed = ego_speed_mps(frames, current_idx, segment_start, segment_end)
    label, debug = _compute_front_route_label(
        route,
        current_boxes,
        speed,
        ego_matrix_current=frame_transform(current),
        future_frames_data=future_frames_data,
        corridor_margin_m=args.corridor_margin_m,
        route_step_m=args.route_dense_step_m,
        max_distance_m=args.max_distance_m,
        safe_ttc_s=args.safe_ttc_s,
        max_ttc_s=args.max_ttc_s,
        return_debug=True,
    )

    case = int(label.get("case", 0))
    stats["counts"]["windows_processed"] += 1
    stats["case_counts"][str(case)] += 1
    stats["command_counts"][command_name(current.get("driving_command", []))] += 1
    if route_crosses_scene:
        stats["counts"]["route_crosses_scene_token_boundary"] += 1

    sample_id = {
        "log_name": log_name,
        "global_idx": int(current_idx),
        "frame_idx": int(current.get("frame_idx", -1)),
        "scene_token": str(current.get("scene_token", "")),
        "scene_name": str(current.get("scene_name", "")),
        "token": str(current.get("token", "")),
        "command": command_name(current.get("driving_command", [])),
        "route_length_m": float(route_length),
        "ego_speed_mps": float(speed),
    }

    current_cover = cover_summary(debug.get("best_current"), label, "current")
    future_cover = cover_summary(debug.get("best_future"), label, "future")
    if current_cover["exists"] > 0.5:
        stats["counts"]["current_cover_exists"] += 1
        stats["actor_class_counts"][current_cover["actor_class_name"]] += 1
        stats["current_cover_route_distance_m"].add(current_cover["route_distance_m"])
        stats["cover_ttc_s"].add(current_cover["ttc_s"])
        examples.add("current_cover", {**sample_id, **current_cover})
    if future_cover["exists"] > 0.5:
        stats["counts"]["future_cover_exists"] += 1
        stats["actor_class_counts"][future_cover["actor_class_name"]] += 1
        stats["future_cover_route_distance_m"].add(future_cover["route_distance_m"])
        stats["future_cover_d_ego_m"].add(future_cover.get("d_ego", np.nan))
        stats["future_cover_d_bg_m"].add(future_cover.get("d_bg", np.nan))
        stats["cover_ttc_s"].add(future_cover["ttc_s"])
        examples.add("future_cover", {**sample_id, **future_cover})
    if current_cover["exists"] > 0.5 or future_cover["exists"] > 0.5:
        stats["counts"]["any_cover_exists"] += 1
        examples.add("any_cover", sample_id)


def process_log(path: Path, args: argparse.Namespace, stats: Dict[str, Any], examples: ExampleStore) -> bool:
    frames = pickle.load(open(path, "rb"))
    track_id_map = build_track_id_map(frames)
    segments = split_contiguous_segments(frames)
    stats["counts"]["logs"] += 1
    stats["counts"]["frames"] += len(frames)
    stats["counts"]["contiguous_segments"] += len(segments)
    stats["counts"]["scene_tokens"] += len(set(frame.get("scene_token") for frame in frames))

    for segment_start, segment_end in segments:
        if segment_end - segment_start + 1 < args.history + args.future_cover_frames:
            stats["counts"]["segments_too_short"] += 1
            continue
        first_current = segment_start + args.history - 1
        last_current = segment_end - args.future_cover_frames
        for current_idx in range(first_current, last_current + 1, args.stride):
            process_window(frames, path.stem, current_idx, segment_start, segment_end, track_id_map, args, stats, examples)
            if args.max_windows is not None and stats["counts"]["windows_processed"] >= args.max_windows:
                return True
    return False


def finalize_stats(stats: Dict[str, Any], examples: ExampleStore) -> Dict[str, Any]:
    return {
        "args": stats["args"],
        "counts": dict(stats["counts"]),
        "command_counts": dict(stats["command_counts"]),
        "case_counts": dict(stats["case_counts"]),
        "actor_class_counts": dict(stats["actor_class_counts"]),
        "route": {
            "valid_points": stats["route_valid_points"].as_dict(),
            "length_m": stats["route_length_m"].as_dict(),
        },
        "covers": {
            "current_route_distance_m": stats["current_cover_route_distance_m"].as_dict(),
            "future_route_distance_m": stats["future_cover_route_distance_m"].as_dict(),
            "future_d_ego_m": stats["future_cover_d_ego_m"].as_dict(),
            "future_d_bg_m": stats["future_cover_d_bg_m"].as_dict(),
            "ttc_s": stats["cover_ttc_s"].as_dict(),
        },
        "example_counts": examples.counts(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-examples", type=Path, required=True)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--require-route", action="store_true")
    parser.add_argument("--future-cover-frames", type=int, default=40)
    parser.add_argument("--route-lookahead-frames", type=int, default=160)
    parser.add_argument("--route-points", type=int, default=80)
    parser.add_argument("--route-point-step-m", type=float, default=1.0)
    parser.add_argument("--min-route-points", type=int, default=20)
    parser.add_argument("--min-route-length-m", type=float, default=20.0)
    parser.add_argument("--corridor-margin-m", type=float, default=0.5)
    parser.add_argument("--route-dense-step-m", type=float, default=0.25)
    parser.add_argument("--max-distance-m", type=float, default=40.0)
    parser.add_argument("--safe-ttc-s", type=float, default=3.0)
    parser.add_argument("--max-ttc-s", type=float, default=10.0)
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--examples-per-category", type=int, default=40)
    parser.add_argument("--progress-every-logs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.history < 1 or args.stride < 1 or args.future_cover_frames < 1:
        raise ValueError("history/stride/future-cover-frames must be positive")
    log_paths = sorted(args.log_root.glob("*.pkl"))
    if args.max_logs is not None:
        log_paths = log_paths[: args.max_logs]
    if not log_paths:
        raise RuntimeError(f"No .pkl logs found under {args.log_root}")

    stats = init_stats(args)
    examples = ExampleStore(args.examples_per_category)
    stop = False
    for log_index, log_path in enumerate(log_paths, start=1):
        if args.progress_every_logs > 0 and (log_index == 1 or log_index % args.progress_every_logs == 0):
            print(f"[navsim_cover_label_stats] {log_index}/{len(log_paths)} {log_path.name}", file=sys.stderr, flush=True)
        stop = process_log(log_path, args, stats, examples)
        if stop:
            break

    result = finalize_stats(stats, examples)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    examples.write_jsonl(args.output_examples)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
