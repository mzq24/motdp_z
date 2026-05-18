#!/usr/bin/env python3
"""Prototype NAVSIM sliding-window label statistics.

This script intentionally stays outside the training path. It reads raw NAVSIM
log pickle files, builds lightweight trajectory/path/object-track labels, and
reports candidate junction/merge-like distributions for migration planning.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


NAVSIM_DT = 0.5
BOX_X = 0
BOX_Y = 1
BOX_Z = 2
BOX_LENGTH = 3
BOX_WIDTH = 4
BOX_HEIGHT = 5
BOX_HEADING = 6

CLASS_NAMES = (
    "generic_object",
    "vehicle",
    "pedestrian",
    "traffic_cone",
    "barrier",
    "bicycle",
    "czone_sign",
)
COMMAND_NAMES = ("none", "left", "straight", "right")


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_transform(transform: np.ndarray) -> float:
    return math.atan2(float(transform[1, 0]), float(transform[0, 0]))


def command_name(command: Any) -> str:
    arr = np.asarray(command).astype(int).reshape(-1)
    if arr.size == 4 and int(arr.sum()) == 1:
        return COMMAND_NAMES[int(arr.argmax())]
    return "other"


def frame_transform(frame: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(frame["ego2global"], dtype=np.float64)


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


def prepare_log_cache(frames: List[Mapping[str, Any]]) -> Dict[str, Any]:
    transforms = [frame_transform(frame) for frame in frames]
    inv_transforms = [np.linalg.inv(transform) for transform in transforms]
    yaws = np.asarray([yaw_from_transform(transform) for transform in transforms], dtype=np.float64)
    positions_xyz = np.asarray(
        [[transform[0, 3], transform[1, 3], transform[2, 3]] for transform in transforms],
        dtype=np.float64,
    )

    boxes: List[np.ndarray] = []
    names: List[List[str]] = []
    velocities: List[np.ndarray] = []
    track_tokens: List[List[str]] = []
    token_maps: List[Dict[str, int]] = []
    for frame in frames:
        frame_anns = anns(frame)
        frame_boxes = coerce_boxes(frame_anns.get("gt_boxes", []))
        frame_names = list(map(str, frame_anns.get("gt_names", [])))
        frame_velocities = coerce_velocities(frame_anns.get("gt_velocity_3d", []))
        frame_tokens = list(map(str, frame_anns.get("track_tokens", [])))
        boxes.append(frame_boxes)
        names.append(frame_names)
        velocities.append(frame_velocities)
        track_tokens.append(frame_tokens)
        token_maps.append({token: idx for idx, token in enumerate(frame_tokens)})

    return {
        "transforms": transforms,
        "inv_transforms": inv_transforms,
        "yaws": yaws,
        "positions_xyz": positions_xyz,
        "boxes": boxes,
        "names": names,
        "velocities": velocities,
        "track_tokens": track_tokens,
        "token_maps": token_maps,
    }


def trajectory_from_cache(cache: Mapping[str, Any], current_idx: int, future: int) -> np.ndarray:
    origin_inv = cache["inv_transforms"][current_idx]
    origin_yaw = float(cache["yaws"][current_idx])
    positions = cache["positions_xyz"][current_idx + 1 : current_idx + future + 1]
    homogeneous = np.concatenate([positions, np.ones((positions.shape[0], 1), dtype=np.float64)], axis=1)
    local = (origin_inv @ homogeneous.T).T
    yaw_delta = [wrap_angle(float(yaw) - origin_yaw) for yaw in cache["yaws"][current_idx + 1 : current_idx + future + 1]]
    return np.column_stack([local[:, 0], local[:, 1], yaw_delta]).astype(np.float32)


def speed_from_cache(cache: Mapping[str, Any], current_idx: int, future: int) -> np.ndarray:
    positions = cache["positions_xyz"][current_idx : current_idx + future + 1, :2]
    return (np.linalg.norm(np.diff(positions, axis=0), axis=1) / NAVSIM_DT).astype(np.float32)


def relative_future_points(
    cache: Mapping[str, Any],
    current_idx: int,
    lookahead_frames: int,
) -> np.ndarray:
    origin_inv = cache["inv_transforms"][current_idx]
    end = min(len(cache["positions_xyz"]) - 1, current_idx + int(lookahead_frames))
    positions = cache["positions_xyz"][current_idx : end + 1]
    homogeneous = np.concatenate([positions, np.ones((positions.shape[0], 1), dtype=np.float64)], axis=1)
    local = (origin_inv @ homogeneous.T).T
    return local[:, :2].astype(np.float32)


def sample_path_by_distance(
    points: np.ndarray,
    step_m: float,
    num_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
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


def make_route_polyline(route: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = route[mask]
    if valid.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate([np.zeros((1, 2), dtype=np.float32), valid.astype(np.float32)], axis=0)


def project_point_to_polyline(point: np.ndarray, polyline: np.ndarray) -> Optional[Dict[str, float]]:
    if polyline.shape[0] < 2:
        return None
    best: Optional[Dict[str, float]] = None
    progress_offset = 0.0
    point = np.asarray(point, dtype=np.float32)
    for start, end in zip(polyline[:-1], polyline[1:]):
        seg = end - start
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-4:
            continue
        unit = seg / seg_len
        rel = point - start
        t = float(np.clip(np.dot(rel, unit) / seg_len, 0.0, 1.0))
        proj = start + t * seg
        delta = point - proj
        lateral = float(unit[0] * delta[1] - unit[1] * delta[0])
        distance = float(np.linalg.norm(delta))
        progress = progress_offset + t * seg_len
        heading = math.atan2(float(unit[1]), float(unit[0]))
        candidate = {
            "distance": distance,
            "lateral": lateral,
            "progress": progress,
            "heading": heading,
        }
        if best is None or candidate["distance"] < best["distance"]:
            best = candidate
        progress_offset += seg_len
    return best


def transform_box_to_origin(
    box: np.ndarray,
    frame: Mapping[str, Any],
    origin_inv: np.ndarray,
    origin_yaw: float,
) -> np.ndarray:
    transform = frame_transform(frame)
    point_global = transform @ np.array([box[BOX_X], box[BOX_Y], box[BOX_Z], 1.0], dtype=np.float64)
    point_origin = origin_inv @ point_global
    frame_yaw = yaw_from_transform(transform)
    out = np.asarray(box, dtype=np.float32).copy()
    out[BOX_X] = point_origin[0]
    out[BOX_Y] = point_origin[1]
    out[BOX_Z] = point_origin[2]
    out[BOX_HEADING] = wrap_angle(float(box[BOX_HEADING]) + frame_yaw - origin_yaw)
    return out


def transform_box_to_origin_cached(
    box: np.ndarray,
    frame_transform_value: np.ndarray,
    frame_yaw: float,
    origin_inv: np.ndarray,
    origin_yaw: float,
) -> np.ndarray:
    point_global = frame_transform_value @ np.array([box[BOX_X], box[BOX_Y], box[BOX_Z], 1.0], dtype=np.float64)
    point_origin = origin_inv @ point_global
    out = np.asarray(box, dtype=np.float32).copy()
    out[BOX_X] = point_origin[0]
    out[BOX_Y] = point_origin[1]
    out[BOX_Z] = point_origin[2]
    out[BOX_HEADING] = wrap_angle(float(box[BOX_HEADING]) + float(frame_yaw) - origin_yaw)
    return out


def anns(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    return frame.get("anns", {})


def token_index_map(frame: Mapping[str, Any]) -> Dict[str, int]:
    tokens = anns(frame).get("track_tokens", []) or []
    return {str(token): idx for idx, token in enumerate(tokens)}


def in_roi(box: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> bool:
    return x_min <= float(box[BOX_X]) <= x_max and y_min <= float(box[BOX_Y]) <= y_max


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

    def add(self, category: str, sample: Mapping[str, Any]) -> None:
        bucket = self.items[str(category)]
        if len(bucket) < self.limit:
            bucket.append(dict(sample))

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for category, samples in sorted(self.items.items()):
                for sample in samples:
                    record = {"category": category, **sample}
                    f.write(json.dumps(record, sort_keys=True) + "\n")

    def counts(self) -> Dict[str, int]:
        return {key: len(value) for key, value in sorted(self.items.items())}


def update_track_stats(
    stats: Dict[str, Any],
    examples: ExampleStore,
    cache: Mapping[str, Any],
    current_idx: int,
    future: int,
    route_polyline: np.ndarray,
    sample_id: Mapping[str, Any],
) -> None:
    origin_inv = cache["inv_transforms"][current_idx]
    origin_yaw = float(cache["yaws"][current_idx])

    boxes = cache["boxes"][current_idx]
    names = cache["names"][current_idx]
    velocities = cache["velocities"][current_idx]
    track_tokens = cache["track_tokens"][current_idx]
    future_maps = cache["token_maps"][current_idx + 1 : current_idx + future + 1]

    for idx, box in enumerate(boxes):
        name = names[idx] if idx < len(names) else "unknown"
        if name not in CLASS_NAMES:
            name = "unknown"
        stats["object_instances_by_class"][name] += 1
        if idx < len(velocities):
            stats["object_speed_mps_by_class"][name].add(float(np.linalg.norm(velocities[idx, :2])))

        box_in_lidar_roi = in_roi(box, -32.0, 32.0, -32.0, 32.0)
        if box_in_lidar_roi:
            stats["object_lidar_roi_instances_by_class"][name] += 1
        else:
            continue

        token = track_tokens[idx] if idx < len(track_tokens) else ""
        if not token:
            continue

        matched_refs: List[Tuple[int, int]] = []
        for step, token_map in enumerate(future_maps, start=1):
            future_idx = token_map.get(token)
            frame_idx = current_idx + step
            if future_idx is None or future_idx >= len(cache["boxes"][frame_idx]):
                continue
            matched_refs.append((frame_idx, future_idx))

        coverage = len(matched_refs)
        stats["object_track_coverage_total_by_class"][name] += coverage
        if coverage >= 1:
            stats["object_track_coverage_ge1_by_class"][name] += 1
        if coverage >= 4:
            stats["object_track_coverage_ge4_by_class"][name] += 1
        if coverage == future:
            stats["object_track_coverage_full_by_class"][name] += 1

        if name != "vehicle":
            continue
        current_proj = project_point_to_polyline(box[[BOX_X, BOX_Y]], route_polyline)
        if current_proj is None:
            continue
        heading_diff = abs(wrap_angle(float(box[BOX_HEADING]) - current_proj["heading"]))
        same_direction = heading_diff <= math.radians(45.0)
        if not same_direction:
            continue

        current_lateral_abs = abs(current_proj["lateral"])
        current_progress = current_proj["progress"]
        if current_lateral_abs <= 1.75 and 0.0 <= current_progress <= 35.0:
            stats["candidate_counts"]["front_chase_like"] += 1
            examples.add("front_chase_like", sample_id)

        if coverage < 4 or current_lateral_abs <= 2.0:
            continue
        matched_boxes = [
            transform_box_to_origin_cached(
                cache["boxes"][frame_idx][future_idx],
                cache["transforms"][frame_idx],
                float(cache["yaws"][frame_idx]),
                origin_inv,
                origin_yaw,
            )
            for frame_idx, future_idx in matched_refs
        ]
        future_projs = [project_point_to_polyline(matched_box[[BOX_X, BOX_Y]], route_polyline) for matched_box in matched_boxes]
        future_projs = [proj for proj in future_projs if proj is not None]
        if not future_projs:
            continue
        min_future_lateral = min(abs(proj["lateral"]) for proj in future_projs)
        min_progress = min(proj["progress"] for proj in future_projs + [current_proj])
        max_progress = max(proj["progress"] for proj in future_projs + [current_proj])
        if min_future_lateral <= 2.0 and max_progress >= 0.0 and min_progress <= 40.0:
            stats["candidate_counts"]["merge_like"] += 1
            examples.add("merge_like", sample_id)


def init_stats(args: argparse.Namespace) -> Dict[str, Any]:
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "args": serializable_args,
        "counts": Counter(),
        "command_counts": Counter(),
        "candidate_counts": Counter(),
        "object_instances_by_class": Counter(),
        "object_lidar_roi_instances_by_class": Counter(),
        "object_track_coverage_ge1_by_class": Counter(),
        "object_track_coverage_ge4_by_class": Counter(),
        "object_track_coverage_full_by_class": Counter(),
        "object_track_coverage_total_by_class": Counter(),
        "object_speed_mps_by_class": defaultdict(ScalarCollector),
        "trajectory_displacement_m": ScalarCollector(),
        "trajectory_lateral_m": ScalarCollector(),
        "trajectory_abs_yaw_delta_rad": ScalarCollector(),
        "speed_mps": ScalarCollector(),
        "path_valid_points": ScalarCollector(),
        "route_valid_points": ScalarCollector(),
    }


def finalize_stats(stats: Dict[str, Any], examples: ExampleStore) -> Dict[str, Any]:
    object_speed = {
        key: collector.as_dict()
        for key, collector in sorted(stats["object_speed_mps_by_class"].items())
    }
    result = {
        "args": stats["args"],
        "counts": dict(stats["counts"]),
        "command_counts": dict(stats["command_counts"]),
        "candidate_counts": dict(stats["candidate_counts"]),
        "trajectory": {
            "displacement_m": stats["trajectory_displacement_m"].as_dict(),
            "lateral_m": stats["trajectory_lateral_m"].as_dict(),
            "abs_yaw_delta_rad": stats["trajectory_abs_yaw_delta_rad"].as_dict(),
        },
        "speed_mps": stats["speed_mps"].as_dict(),
        "path": {
            "path_valid_points": stats["path_valid_points"].as_dict(),
            "route_valid_points": stats["route_valid_points"].as_dict(),
        },
        "objects": {
            "instances_by_class": dict(stats["object_instances_by_class"]),
            "lidar_roi_instances_by_class": dict(stats["object_lidar_roi_instances_by_class"]),
            "track_coverage_ge1_by_class": dict(stats["object_track_coverage_ge1_by_class"]),
            "track_coverage_ge4_by_class": dict(stats["object_track_coverage_ge4_by_class"]),
            "track_coverage_full_by_class": dict(stats["object_track_coverage_full_by_class"]),
            "track_coverage_total_by_class": dict(stats["object_track_coverage_total_by_class"]),
            "speed_mps_by_class": object_speed,
        },
        "example_counts": examples.counts(),
    }
    return result


def process_log(path: Path, args: argparse.Namespace, stats: Dict[str, Any], examples: ExampleStore) -> bool:
    frames = pickle.load(open(path, "rb"))
    cache = prepare_log_cache(frames)
    stats["counts"]["logs"] += 1
    stats["counts"]["frames"] += len(frames)
    windows_seen = 0

    max_start = len(frames) - args.history - args.future + 1
    if max_start <= 0:
        stats["counts"]["logs_too_short"] += 1
        return False

    for start in range(0, max_start, args.stride):
        current_idx = start + args.history - 1
        current = frames[current_idx]
        windows_seen += 1
        stats["counts"]["windows_total"] += 1

        if args.require_route and len(current.get("roadblock_ids", [])) == 0:
            stats["counts"]["windows_skipped_no_route"] += 1
            continue

        stats["counts"]["windows_processed"] += 1
        sample_id = {
            "log_name": current.get("log_name", path.stem),
            "frame_idx": int(current.get("frame_idx", current_idx)),
            "token": current.get("token", ""),
        }

        command = command_name(current.get("driving_command", []))
        stats["command_counts"][command] += 1

        trajectory = trajectory_from_cache(cache, current_idx, args.future)
        speeds = speed_from_cache(cache, current_idx, args.future)
        stats["trajectory_displacement_m"].add(float(np.linalg.norm(trajectory[-1, :2])))
        stats["trajectory_lateral_m"].add(float(trajectory[-1, 1]))
        stats["trajectory_abs_yaw_delta_rad"].add(abs(float(trajectory[-1, 2])))
        for speed in speeds:
            stats["speed_mps"].add(float(speed))

        future_points = relative_future_points(cache, current_idx, args.path_lookahead_frames)
        path50, path_mask = sample_path_by_distance(future_points, 1.0, args.path_points)
        route = path50[: args.route_points]
        route_mask = path_mask[: args.route_points]
        stats["path_valid_points"].add(float(path_mask.sum()))
        stats["route_valid_points"].add(float(route_mask.sum()))
        if path_mask.sum() == args.path_points:
            stats["counts"]["path_full"] += 1
        if route_mask.sum() == args.route_points:
            stats["counts"]["route_full"] += 1

        yaw_delta = abs(float(trajectory[-1, 2]))
        junction_like = command in {"left", "right"} or yaw_delta >= args.junction_yaw_threshold
        if junction_like:
            stats["candidate_counts"]["junction_like"] += 1
            examples.add("junction_like", sample_id)
            if len(current.get("traffic_lights", [])) > 0:
                stats["candidate_counts"]["signalized_junction_like"] += 1
                examples.add("signalized_junction_like", sample_id)

        route_polyline = make_route_polyline(route, route_mask)
        update_track_stats(stats, examples, cache, current_idx, args.future, route_polyline, sample_id)

        if args.max_windows is not None and stats["counts"]["windows_processed"] >= args.max_windows:
            return True

    stats["counts"]["windows_seen_in_logs"] += windows_seen
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-examples", type=Path, required=True)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--future", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--require-route", action="store_true")
    parser.add_argument("--path-lookahead-frames", type=int, default=80)
    parser.add_argument("--path-points", type=int, default=50)
    parser.add_argument("--route-points", type=int, default=20)
    parser.add_argument("--junction-yaw-threshold", type=float, default=0.45)
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--examples-per-category", type=int, default=20)
    parser.add_argument("--progress-every-logs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.history < 1 or args.future < 1 or args.stride < 1:
        raise ValueError("history/future/stride must be positive")
    if args.route_points > args.path_points:
        raise ValueError("route-points must be <= path-points")

    stats = init_stats(args)
    examples = ExampleStore(args.examples_per_category)
    log_paths = sorted(args.log_root.glob("*.pkl"))
    if args.max_logs is not None:
        log_paths = log_paths[: args.max_logs]
    if not log_paths:
        raise RuntimeError(f"No .pkl logs found under {args.log_root}")

    stop = False
    for log_index, log_path in enumerate(log_paths, start=1):
        if args.progress_every_logs > 0 and (log_index == 1 or log_index % args.progress_every_logs == 0):
            print(
                f"[navsim_label_stats] processing {log_index}/{len(log_paths)} {log_path.name}",
                file=sys.stderr,
                flush=True,
            )
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
