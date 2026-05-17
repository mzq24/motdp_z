#!/usr/bin/env python3
"""Generate quick NAVSIM label verification videos from example JSONL records."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


NAVSIM_DT = 0.5
BOX_X = 0
BOX_Y = 1
BOX_Z = 2
BOX_LENGTH = 3
BOX_WIDTH = 4
BOX_HEIGHT = 5
BOX_HEADING = 6

COMMAND_NAMES = ("none", "left", "straight", "right")
VEHICLE_NAMES = {"vehicle"}


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


def anns(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    return frame.get("anns", {})


def coerce_boxes(value: Any) -> np.ndarray:
    arr = np.asarray(value if value is not None else [], dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def token_index_map(frame: Mapping[str, Any]) -> Dict[str, int]:
    tokens = anns(frame).get("track_tokens", []) or []
    return {str(token): idx for idx, token in enumerate(tokens)}


def relative_future_points(
    frames: Sequence[Mapping[str, Any]],
    current_idx: int,
    lookahead_frames: int,
) -> np.ndarray:
    origin = frame_transform(frames[current_idx])
    origin_inv = np.linalg.inv(origin)
    end = min(len(frames) - 1, current_idx + int(lookahead_frames))
    points = []
    for idx in range(current_idx, end + 1):
        transform = frame_transform(frames[idx])
        point = origin_inv @ np.array([transform[0, 3], transform[1, 3], transform[2, 3], 1.0])
        points.append([point[0], point[1]])
    return np.asarray(points, dtype=np.float32)


def sample_path_by_distance(points: np.ndarray, step_m: float, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    path = np.zeros((int(num_points), 2), dtype=np.float32)
    mask = np.zeros(int(num_points), dtype=bool)
    if points.shape[0] < 2:
        return path, mask

    seg_lens = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
    keep = np.concatenate([[True], np.diff(dist) > 1e-3])
    dist = dist[keep]
    points = points[keep]
    if points.shape[0] < 2 or dist[-1] < step_m:
        return path, mask

    targets = np.arange(step_m, step_m * (int(num_points) + 1), step_m, dtype=np.float32)
    valid = targets <= dist[-1]
    targets_valid = targets[valid]
    if targets_valid.size == 0:
        return path, mask

    path[: len(targets_valid), 0] = np.interp(targets_valid, dist, points[:, 0])
    path[: len(targets_valid), 1] = np.interp(targets_valid, dist, points[:, 1])
    mask[: len(targets_valid)] = True
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


def future_yaw_delta(frames: Sequence[Mapping[str, Any]], current_idx: int, future: int) -> float:
    end = min(len(frames) - 1, current_idx + int(future))
    current_yaw = yaw_from_transform(frame_transform(frames[current_idx]))
    future_yaw = yaw_from_transform(frame_transform(frames[end]))
    return abs(wrap_angle(future_yaw - current_yaw))


def build_route_polyline(
    frames: Sequence[Mapping[str, Any]],
    current_idx: int,
    route_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    future_points = relative_future_points(frames, current_idx, 80)
    path50, path_mask = sample_path_by_distance(future_points, 1.0, 50)
    route = path50[:route_points]
    route_mask = path_mask[:route_points]
    return future_points, route, make_route_polyline(route, route_mask)


def candidate_box_indices(
    frames: Sequence[Mapping[str, Any]],
    current_idx: int,
    route_polyline: np.ndarray,
    future: int = 8,
) -> Tuple[Set[int], Set[int]]:
    current = frames[current_idx]
    current_anns = anns(current)
    boxes = coerce_boxes(current_anns.get("gt_boxes", []))
    names = list(map(str, current_anns.get("gt_names", [])))
    track_tokens = list(map(str, current_anns.get("track_tokens", [])))
    future_maps = [token_index_map(frames[current_idx + step]) for step in range(1, min(future, len(frames) - current_idx - 1) + 1)]
    future_boxes = [coerce_boxes(anns(frames[current_idx + step]).get("gt_boxes", [])) for step in range(1, len(future_maps) + 1)]

    origin = frame_transform(current)
    origin_inv = np.linalg.inv(origin)
    origin_yaw = yaw_from_transform(origin)

    merge_indices: Set[int] = set()
    front_indices: Set[int] = set()
    for idx, box in enumerate(boxes):
        name = names[idx] if idx < len(names) else "unknown"
        if name not in VEHICLE_NAMES:
            continue
        current_proj = project_point_to_polyline(box[[BOX_X, BOX_Y]], route_polyline)
        if current_proj is None:
            continue
        heading_diff = abs(wrap_angle(float(box[BOX_HEADING]) - current_proj["heading"]))
        if heading_diff > math.radians(45.0):
            continue
        current_lateral_abs = abs(current_proj["lateral"])
        current_progress = current_proj["progress"]
        if current_lateral_abs <= 1.75 and 0.0 <= current_progress <= 35.0:
            front_indices.add(idx)

        token = track_tokens[idx] if idx < len(track_tokens) else ""
        if not token or current_lateral_abs <= 2.0:
            continue
        matched_boxes = []
        for step, token_map in enumerate(future_maps):
            future_idx = token_map.get(token)
            if future_idx is None or future_idx >= len(future_boxes[step]):
                continue
            matched_boxes.append(
                transform_box_to_origin(
                    future_boxes[step][future_idx],
                    frames[current_idx + step + 1],
                    origin_inv,
                    origin_yaw,
                )
            )
        if len(matched_boxes) < 4:
            continue
        future_projs = [project_point_to_polyline(box2[[BOX_X, BOX_Y]], route_polyline) for box2 in matched_boxes]
        future_projs = [proj for proj in future_projs if proj is not None]
        if not future_projs:
            continue
        min_future_lateral = min(abs(proj["lateral"]) for proj in future_projs)
        min_progress = min(proj["progress"] for proj in future_projs + [current_proj])
        max_progress = max(proj["progress"] for proj in future_projs + [current_proj])
        if min_future_lateral <= 2.0 and max_progress >= 0.0 and min_progress <= 40.0:
            merge_indices.add(idx)
    return merge_indices, front_indices


class BevProjector:
    def __init__(self, size: int = 560, x_min: float = -10.0, x_max: float = 60.0, y_min: float = -25.0, y_max: float = 25.0):
        self.size = int(size)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

    def point(self, xy: Sequence[float]) -> Tuple[int, int]:
        x, y = float(xy[0]), float(xy[1])
        u = (self.y_max - y) / (self.y_max - self.y_min) * (self.size - 1)
        v = (self.x_max - x) / (self.x_max - self.x_min) * (self.size - 1)
        return int(round(u)), int(round(v))

    def in_view(self, xy: Sequence[float], margin: float = 2.0) -> bool:
        x, y = float(xy[0]), float(xy[1])
        return self.x_min - margin <= x <= self.x_max + margin and self.y_min - margin <= y <= self.y_max + margin


def draw_polyline(panel: np.ndarray, projector: BevProjector, points: np.ndarray, color: Tuple[int, int, int], thickness: int) -> None:
    visible = [projector.point(point) for point in points if projector.in_view(point)]
    if len(visible) >= 2:
        cv2.polylines(panel, [np.asarray(visible, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
    for point in visible:
        cv2.circle(panel, point, max(2, thickness), color, -1, cv2.LINE_AA)


def box_corners(box: np.ndarray) -> np.ndarray:
    length = float(box[BOX_LENGTH])
    width = float(box[BOX_WIDTH])
    yaw = float(box[BOX_HEADING])
    corners = np.asarray(
        [[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]],
        dtype=np.float32,
    )
    rot = np.asarray([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]], dtype=np.float32)
    return corners @ rot.T + box[[BOX_X, BOX_Y]]


def draw_box(panel: np.ndarray, projector: BevProjector, box: np.ndarray, color: Tuple[int, int, int], thickness: int) -> None:
    if not projector.in_view(box[[BOX_X, BOX_Y]], margin=8.0):
        return
    corners = np.asarray([projector.point(point) for point in box_corners(box)], dtype=np.int32)
    cv2.polylines(panel, [corners], True, color, thickness, cv2.LINE_AA)
    center = projector.point(box[[BOX_X, BOX_Y]])
    front = projector.point((box[BOX_X] + math.cos(float(box[BOX_HEADING])) * 2.5, box[BOX_Y] + math.sin(float(box[BOX_HEADING])) * 2.5))
    cv2.arrowedLine(panel, center, front, color, max(1, thickness), cv2.LINE_AA, tipLength=0.35)


def draw_bev_panel(
    frames: Sequence[Mapping[str, Any]],
    current_idx: int,
    target_idx: int,
    category: str,
    route_points: int,
) -> np.ndarray:
    projector = BevProjector()
    panel = np.full((projector.size, projector.size, 3), (24, 26, 30), dtype=np.uint8)

    for x in range(0, 61, 10):
        y0 = projector.point((x, projector.y_min))
        y1 = projector.point((x, projector.y_max))
        cv2.line(panel, y0, y1, (54, 58, 65), 1, cv2.LINE_AA)
    for y in range(-20, 21, 10):
        p0 = projector.point((projector.x_min, y))
        p1 = projector.point((projector.x_max, y))
        cv2.line(panel, p0, p1, (54, 58, 65), 1, cv2.LINE_AA)

    future_points, route, route_polyline = build_route_polyline(frames, current_idx, route_points)
    route_mask = np.linalg.norm(route, axis=1) > 1e-6

    draw_polyline(panel, projector, future_points[: min(len(future_points), 9)], (80, 220, 120), 2)
    draw_polyline(panel, projector, route[route_mask], (255, 210, 70), 3)

    boxes = coerce_boxes(anns(frames[current_idx]).get("gt_boxes", []))
    names = list(map(str, anns(frames[current_idx]).get("gt_names", [])))
    merge_indices, front_indices = candidate_box_indices(frames, current_idx, route_polyline)
    for idx, box in enumerate(boxes):
        name = names[idx] if idx < len(names) else "unknown"
        if idx in merge_indices:
            color, thickness = (255, 80, 255), 3
        elif idx in front_indices:
            color, thickness = (40, 230, 255), 3
        elif name == "vehicle":
            color, thickness = (60, 120, 255), 1
        elif name == "pedestrian":
            color, thickness = (255, 220, 80), 1
        elif name == "bicycle":
            color, thickness = (180, 120, 255), 1
        else:
            color, thickness = (120, 130, 145), 1
        draw_box(panel, projector, box, color, thickness)

    ego_box = np.asarray([0.0, 0.0, 0.0, 4.8, 2.0, 1.5, 0.0], dtype=np.float32)
    draw_box(panel, projector, ego_box, (245, 245, 245), 2)

    if current_idx == target_idx:
        cv2.rectangle(panel, (4, 4), (projector.size - 5, projector.size - 5), (40, 40, 255), 4)

    lines = [
        "BEV x-forward / y-left",
        f"route pts {int(route_mask.sum())}/20",
        f"merge boxes {len(merge_indices)}",
        f"front boxes {len(front_indices)}",
        "magenta=merge, yellow=front",
    ]
    y = 24
    for line in lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 235, 240), 1, cv2.LINE_AA)
        y += 22
    cv2.putText(panel, category, (12, projector.size - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 230, 255), 2, cv2.LINE_AA)
    return panel


def draw_text_overlay(image: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    out = image.copy()
    overlay = out.copy()
    height = 30 + 28 * len(lines)
    cv2.rectangle(overlay, (0, 0), (out.shape[1], height), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0.0)
    y = 28
    for idx, line in enumerate(lines):
        color = (80, 230, 255) if idx == 0 else (245, 245, 245)
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2 if idx == 0 else 1, cv2.LINE_AA)
        y += 28
    return out


def read_front_image(frame: Mapping[str, Any], sensor_root: Path, cam: str, size: Tuple[int, int]) -> np.ndarray:
    cam_info = frame.get("cams", {}).get(cam, {})
    image_path = sensor_root / str(cam_info.get("data_path", ""))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        image = np.full((size[1], size[0], 3), (35, 35, 35), dtype=np.uint8)
        cv2.putText(image, f"missing {cam}", (30, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2, cv2.LINE_AA)
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def find_frame_index(frames: Sequence[Mapping[str, Any]], record: Mapping[str, Any]) -> int:
    token = str(record.get("token", ""))
    frame_idx = int(record.get("frame_idx", -1))
    for idx, frame in enumerate(frames):
        if token and str(frame.get("token", "")) == token:
            return idx
    for idx, frame in enumerate(frames):
        if int(frame.get("frame_idx", -999999)) == frame_idx:
            return idx
    raise ValueError(f"Could not find frame for {record}")


def load_examples(path: Path, categories: Sequence[str], per_category: int, min_frame_gap: int) -> List[Dict[str, Any]]:
    wanted = set(categories)
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen = set()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def try_add(record: Dict[str, Any], enforce_gap: bool) -> None:
        category = str(record.get("category", ""))
        if category not in wanted or len(by_category[category]) >= per_category:
            return
        key = (category, record.get("log_name"), int(record.get("frame_idx", -1)))
        if key in seen:
            return
        if enforce_gap:
            for old in by_category[category]:
                same_log = old.get("log_name") == record.get("log_name")
                close = abs(int(old.get("frame_idx", -1)) - int(record.get("frame_idx", -1))) < min_frame_gap
                if same_log and close:
                    return
        by_category[category].append(record)
        seen.add(key)

    for record in records:
        try_add(record, enforce_gap=True)
    for record in records:
        try_add(record, enforce_gap=False)

    selected: List[Dict[str, Any]] = []
    for category in categories:
        selected.extend(by_category.get(category, []))
    return selected


def frame_candidate_categories(
    frames: Sequence[Mapping[str, Any]],
    current_idx: int,
    route_points: int,
    junction_yaw_threshold: float,
) -> Set[str]:
    current = frames[current_idx]
    if len(current.get("roadblock_ids", [])) == 0:
        return set()

    _, _, route_polyline = build_route_polyline(frames, current_idx, route_points)
    categories: Set[str] = set()
    command = command_name(current.get("driving_command", []))
    yaw_delta = future_yaw_delta(frames, current_idx, 8)
    if command in {"left", "right"} or yaw_delta >= junction_yaw_threshold:
        categories.add("junction_like")
        if len(current.get("traffic_lights", [])) > 0:
            categories.add("signalized_junction_like")

    merge_indices, front_indices = candidate_box_indices(frames, current_idx, route_polyline)
    if merge_indices:
        categories.add("merge_like")
    if front_indices:
        categories.add("front_chase_like")
    return categories


def scan_examples(
    log_root: Path,
    categories: Sequence[str],
    per_category: int,
    min_frame_gap: int,
    unique_log_per_category: bool,
    scan_stride: int,
    route_points: int,
    junction_yaw_threshold: float,
    max_logs: Optional[int],
    min_target_index: int,
    required_future_frames: int,
) -> List[Dict[str, Any]]:
    wanted = set(categories)
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    log_paths = sorted(log_root.glob("*.pkl"))
    if max_logs is not None:
        log_paths = log_paths[:max_logs]

    def full() -> bool:
        return all(len(by_category[category]) >= per_category for category in categories)

    for log_path in log_paths:
        if full():
            break
        frames = pickle.load(open(log_path, "rb"))
        max_current = len(frames) - max(8, int(required_future_frames)) - 1
        start_current = max(3, int(min_target_index))
        if max_current < start_current:
            continue
        for current_idx in range(start_current, max_current + 1, max(1, int(scan_stride))):
            found = frame_candidate_categories(frames, current_idx, route_points, junction_yaw_threshold)
            for category in categories:
                if category not in found or category not in wanted or len(by_category[category]) >= per_category:
                    continue
                if unique_log_per_category and any(item["log_name"] == log_path.stem for item in by_category[category]):
                    continue
                if any(
                    item["log_name"] == log_path.stem
                    and abs(int(item["frame_idx"]) - int(frames[current_idx].get("frame_idx", current_idx))) < min_frame_gap
                    for item in by_category[category]
                ):
                    continue
                by_category[category].append(
                    {
                        "category": category,
                        "log_name": str(frames[current_idx].get("log_name", log_path.stem)),
                        "frame_idx": int(frames[current_idx].get("frame_idx", current_idx)),
                        "token": str(frames[current_idx].get("token", "")),
                    }
                )
            if full():
                break

    selected: List[Dict[str, Any]] = []
    for category in categories:
        selected.extend(by_category.get(category, []))
    return selected


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def generate_video(
    record: Mapping[str, Any],
    log_root: Path,
    sensor_root: Path,
    output_dir: Path,
    cam: str,
    before: int,
    after: int,
    fps: float,
    route_points: int,
) -> Path:
    category = str(record["category"])
    log_name = str(record["log_name"])
    log_path = log_root / f"{log_name}.pkl"
    frames = pickle.load(open(log_path, "rb"))
    target_idx = find_frame_index(frames, record)
    start = max(0, target_idx - before)
    end = min(len(frames) - 1, target_idx + after)

    front_size = (960, 560)
    bev_size = 560
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_name(category)}_{safe_name(log_name)}_f{int(record['frame_idx']):05d}.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (front_size[0] + bev_size, front_size[1]))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    for frame_idx in range(start, end + 1):
        frame = frames[frame_idx]
        front = read_front_image(frame, sensor_root, cam, front_size)
        command = command_name(frame.get("driving_command", []))
        yaw_delta = future_yaw_delta(frames, frame_idx, 8)
        traffic_lights = len(frame.get("traffic_lights", []))
        target = " TARGET" if frame_idx == target_idx else ""
        lines = [
            f"{category}{target}",
            f"log={log_name}",
            f"frame={frame.get('frame_idx', frame_idx)} command={command} yaw8={yaw_delta:.2f} lights={traffic_lights}",
        ]
        front = draw_text_overlay(front, lines)
        bev = draw_bev_panel(frames, frame_idx, target_idx, category, route_points)
        writer.write(np.concatenate([front, bev], axis=1))
    writer.release()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["merge_like", "junction_like", "front_chase_like", "signalized_junction_like"])
    parser.add_argument("--per-category", type=int, default=1)
    parser.add_argument("--min-frame-gap", type=int, default=20)
    parser.add_argument("--before", type=int, default=12)
    parser.add_argument("--after", type=int, default=28)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--cam", default="CAM_F0")
    parser.add_argument("--route-points", type=int, default=20)
    parser.add_argument("--scan-candidates", action="store_true")
    parser.add_argument("--scan-stride", type=int, default=5)
    parser.add_argument("--unique-log-per-category", action="store_true")
    parser.add_argument("--max-scan-logs", type=int, default=None)
    parser.add_argument("--junction-yaw-threshold", type=float, default=0.45)
    parser.add_argument("--min-target-index", type=int, default=3)
    parser.add_argument("--required-future-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scan_candidates:
        selected = scan_examples(
            args.log_root,
            args.categories,
            args.per_category,
            args.min_frame_gap,
            args.unique_log_per_category,
            args.scan_stride,
            args.route_points,
            args.junction_yaw_threshold,
            args.max_scan_logs,
            args.min_target_index,
            args.required_future_frames,
        )
    else:
        selected = load_examples(args.examples_jsonl, args.categories, args.per_category, args.min_frame_gap)
    if not selected:
        raise RuntimeError("No examples selected")
    for record in selected:
        output = generate_video(
            record,
            args.log_root,
            args.sensor_root,
            args.output_dir,
            args.cam,
            args.before,
            args.after,
            args.fps,
            args.route_points,
        )
        print(output)


if __name__ == "__main__":
    main()
