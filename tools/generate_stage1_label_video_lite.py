#!/usr/bin/env python3
"""
Generate a lightweight stage1 label video by reading precomputed labels only.

This script does NOT recompute:
- cover
- merge / borrow / junction windows
- conflict-area labels
- speed risks

It only reads:
- packed samples
- per-sample stage1 labels / debug already stored in packed
- raw RGB / boxes / measurements for visualization

Typical usage:

  python tools/generate_stage1_label_video_lite.py \
    --dataset_path /workspace1/z_project/dataset/pdm_lite/tmp_data/full_scene_refresh_scene_split_95_5/train \
    --image_data_root /workspace1/z_project/dataset/pdm_lite \
    --scene_name AccidentTwoWays \
    --route_name Town12_Rep0_26_0_route0_11_08_18_12_42
"""

import argparse
import gzip
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle", "van"}
COMMAND_MAP = {
    1: "LEFT",
    2: "RIGHT",
    3: "STRAIGHT",
    4: "LANE_FOLLOW",
    5: "CHANGE_LEFT",
    6: "CHANGE_RIGHT",
}
CONFLICT_FAMILY_NAMES = {
    0: "none",
    1: "borrow",
    2: "merge",
    3: "junction",
}
CONFLICT_DIR_NAMES = {
    0: "none",
    1: "same",
    2: "opposite",
    3: "cross",
}


def _resolve_packed_path(dataset_path=None, packed_path=None):
    if packed_path:
        return os.path.realpath(packed_path)
    if not dataset_path:
        raise ValueError("Either --dataset_path or --packed_path is required.")
    return os.path.realpath(os.path.join(dataset_path, "samples_packed.pkl"))


def _resolve_feature_frame_info(sample):
    feature_rel = sample.get("transfuser_bev_feature", "")
    frame_id = sample.get("frame_id", None)
    if not feature_rel or frame_id is None:
        return None, None
    base_dir = os.path.dirname(os.path.dirname(feature_rel))
    if "route_features.pt" in feature_rel:
        frame_str = f"{int(frame_id):04d}"
    else:
        frame_str = os.path.basename(feature_rel).replace("_feature.pt", "")
    return base_dir, frame_str


def _load_json_gz_if_exists(path):
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt") as f:
            return json.load(f)
    except Exception:
        return None


def _scene_name_from_base_dir(base_dir):
    if not base_dir:
        return None
    if os.sep in base_dir:
        return base_dir.split(os.sep)[0]
    if "/" in base_dir:
        return base_dir.split("/")[0]
    return base_dir


def _select_route_samples(samples, route_name=None, scene_name=None, index=None):
    if route_name is None:
        if index is None:
            raise ValueError("Either --route_name or --index must be provided.")
        route_name = samples[index].get("route_name")
        if route_name is None:
            raise ValueError("Selected sample does not contain route_name.")

    route_samples = []
    for sample in samples:
        if sample.get("route_name") != route_name:
            continue
        base_dir, _ = _resolve_feature_frame_info(sample)
        if scene_name is not None and _scene_name_from_base_dir(base_dir) != scene_name:
            continue
        route_samples.append(sample)

    if not route_samples:
        if scene_name is None:
            raise ValueError(f"No samples found for route_name={route_name}")
        raise ValueError(f"No samples found for scene_name={scene_name} route_name={route_name}")

    route_samples.sort(key=lambda s: int(s.get("frame_id", -1)))
    return route_name, route_samples


def _load_route_requests(list_path):
    requests = []
    with open(list_path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                requests.append({"scene_name": None, "route_name": parts[0]})
                continue
            if len(parts) == 2:
                requests.append({"scene_name": parts[0], "route_name": parts[1]})
                continue
            raise ValueError(
                f"Invalid route list line {lineno} in {list_path}: expected "
                "'route_name' or 'scene_name route_name', got: {raw_line.rstrip()}"
            )
    if not requests:
        raise ValueError(f"No valid route requests found in {list_path}")
    return requests


def _load_scene_rgb(image_root, base_dir, frame_str):
    rgb_candidates = [
        os.path.join(image_root, base_dir, "rgb", f"{frame_str}.jpg"),
        os.path.join(image_root, base_dir, "rgb", f"{frame_str}.png"),
        os.path.join(image_root, base_dir, "rgb", f"{frame_str}.jpeg"),
    ]
    for rgb_path in rgb_candidates:
        if os.path.exists(rgb_path):
            rgb = cv2.imread(rgb_path)
            if rgb is not None:
                return rgb
    return np.zeros((900, 1600, 3), dtype=np.uint8)


def _transform_points_world_xyz_to_local(points_xyz, ego_matrix):
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.zeros((0, 2), dtype=np.float32)
    ego_matrix = np.asarray(ego_matrix, dtype=np.float32)
    try:
        ego_inv = np.linalg.inv(ego_matrix)
    except np.linalg.LinAlgError:
        return np.zeros((0, 2), dtype=np.float32)
    pts_h = np.concatenate([pts[:, :3], np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    local = (ego_inv @ pts_h.T).T
    return local[:, :2].astype(np.float32)


def _transform_points_local_to_world_xyz(points_xy, ego_matrix):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.zeros((0, 3), dtype=np.float32)
    ego_matrix = np.asarray(ego_matrix, dtype=np.float32)
    pts_h = np.concatenate(
        [pts[:, :2], np.zeros((pts.shape[0], 1), dtype=np.float32), np.ones((pts.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    world = (ego_matrix @ pts_h.T).T
    return world[:, :3].astype(np.float32)

def _transform_world_points_to_local_xy(points_world, ego_matrix):
    pts = np.asarray(points_world, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if pts.shape[1] >= 3:
        return _transform_points_world_xyz_to_local(pts[:, :3], ego_matrix)
    return np.zeros((0, 2), dtype=np.float32)


def _transform_single_world_point_to_local_xy(world_xyz, world_xy, ego_matrix):
    xyz = np.asarray(world_xyz, dtype=np.float32).reshape(-1)
    if xyz.size >= 3 and np.all(np.isfinite(xyz[:3])):
        local_xy = _transform_points_world_xyz_to_local(xyz[:3][None, :], ego_matrix)
        if local_xy.shape == (1, 2):
            return local_xy[0]
    return None


def _transform_box_to_current_frame(box, transform):
    pos = box.get("position", None)
    if pos is None or len(pos) < 2:
        return None
    pos_h = np.array([pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0, 1.0], dtype=np.float32)
    pos_cur = np.asarray(transform, dtype=np.float32) @ pos_h

    yaw_future = float(box.get("yaw", 0.0))
    heading_future = np.array([np.cos(yaw_future), np.sin(yaw_future), 0.0, 0.0], dtype=np.float32)
    heading_cur = np.asarray(transform, dtype=np.float32) @ heading_future
    yaw_cur = float(np.arctan2(heading_cur[1], heading_cur[0]))

    box_cur = dict(box)
    box_cur["position"] = [float(pos_cur[0]), float(pos_cur[1]), float(pos_cur[2])]
    box_cur["yaw"] = yaw_cur
    return box_cur


def _route_with_origin(route):
    route = np.asarray(route, dtype=np.float32)
    if route.ndim != 2 or route.shape[0] == 0 or route.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    if np.linalg.norm(route[0]) < 1e-4:
        return route
    return np.concatenate([np.zeros((1, 2), dtype=np.float32), route], axis=0)


def _interpolate_route_with_arclength(route_xy, step_m=0.25):
    pts = np.asarray(route_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)], axis=0).astype(np.float32)
    total = float(cum[-1])
    if total <= 1e-6:
        return pts[:1].copy(), np.zeros((1,), dtype=np.float32)
    query = np.arange(0.0, total + step_m * 0.5, step_m, dtype=np.float32)
    x = np.interp(query, cum, pts[:, 0]).astype(np.float32)
    y = np.interp(query, cum, pts[:, 1]).astype(np.float32)
    return np.stack([x, y], axis=1), query


def _sample_route_point_at_s(route_xy, progress_m):
    dense, dense_s = _interpolate_route_with_arclength(_route_with_origin(route_xy), step_m=0.25)
    if dense.shape[0] == 0 or dense_s.shape[0] == 0:
        return None
    progress_m = float(np.clip(float(progress_m), 0.0, float(dense_s[-1])))
    idx = int(np.searchsorted(dense_s, progress_m, side="left"))
    idx = int(np.clip(idx, 0, dense.shape[0] - 1))
    return dense[idx]


def _sample_polyline_point_at_s(poly_xy, progress_m):
    dense, dense_s = _interpolate_route_with_arclength(np.asarray(poly_xy, dtype=np.float32), step_m=0.25)
    if dense.shape[0] == 0 or dense_s.shape[0] == 0:
        return None
    progress_m = float(np.clip(float(progress_m), 0.0, float(dense_s[-1])))
    idx = int(np.searchsorted(dense_s, progress_m, side="left"))
    idx = int(np.clip(idx, 0, dense.shape[0] - 1))
    return dense[idx]


def _sample_polyline_segment_at_s(poly_xy, start_s, end_s, step_m=0.5):
    dense, dense_s = _interpolate_route_with_arclength(np.asarray(poly_xy, dtype=np.float32), step_m=step_m)
    if dense.shape[0] == 0 or dense_s.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    start_s = float(max(start_s, 0.0))
    end_s = float(max(end_s, start_s))
    total_s = float(dense_s[-1])
    if total_s <= 1e-6 or start_s > total_s:
        return np.zeros((0, 2), dtype=np.float32)
    end_s = float(min(end_s, total_s))
    query = np.arange(start_s, end_s + float(step_m) * 0.5, float(step_m), dtype=np.float32)
    if query.size == 0 or float(query[-1]) < end_s - 1e-4:
        query = np.concatenate([query, np.array([end_s], dtype=np.float32)], axis=0)
    x = np.interp(query, dense_s, dense[:, 0]).astype(np.float32)
    y = np.interp(query, dense_s, dense[:, 1]).astype(np.float32)
    return np.stack([x, y], axis=1)


def _oriented_box_corners(position_xy, extent_xy, yaw):
    x, y = float(position_xy[0]), float(position_xy[1])
    half_l, half_w = float(extent_xy[0]), float(extent_xy[1])
    corners = np.array(
        [
            [-half_l, -half_w],
            [-half_l, half_w],
            [half_l, half_w],
            [half_l, -half_w],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + np.array([x, y], dtype=np.float32)


def _draw_dashed_polyline(
    canvas,
    points_xy,
    color,
    x_range,
    y_range,
    thickness=2,
    dash_px=10.0,
    gap_px=6.0,
    closed=False,
):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return
    if closed:
        pts = np.concatenate([pts, pts[:1]], axis=0)
    pts_px = _local_to_canvas(pts, canvas.shape[1], canvas.shape[0], x_range, y_range).astype(np.float32)
    if pts_px.shape[0] < 2:
        return
    dash_px = float(max(dash_px, 1.0))
    gap_px = float(max(gap_px, 0.0))
    for idx in range(pts_px.shape[0] - 1):
        start = pts_px[idx]
        end = pts_px[idx + 1]
        seg = end - start
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            continue
        direction = seg / seg_len
        cursor = 0.0
        while cursor < seg_len:
            dash_end = min(cursor + dash_px, seg_len)
            p0 = start + direction * cursor
            p1 = start + direction * dash_end
            cv2.line(
                canvas,
                tuple(np.round(p0).astype(np.int32)),
                tuple(np.round(p1).astype(np.int32)),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
            cursor += dash_px + gap_px


def _local_to_canvas(points_xy, canvas_w, canvas_h, x_range, y_range):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.int32)
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    # Match the old world-panel convention exactly:
    # x_forward -> right in image, y_lateral -> down in image.
    xs = (pts[:, 0] - x0) / max(x1 - x0, 1e-6) * float(max(canvas_w - 1, 1))
    ys = (pts[:, 1] - y0) / max(y1 - y0, 1e-6) * float(max(canvas_h - 1, 1))
    return np.stack([xs, ys], axis=1).round().astype(np.int32)


def _draw_box(canvas, box, color, x_range, y_range, thickness=2, dashed=False):
    pos = box.get("position", None)
    extent = box.get("extent", None)
    if pos is None or extent is None or len(pos) < 2 or len(extent) < 2:
        return
    corners = _oriented_box_corners(pos[:2], extent[:2], float(box.get("yaw", 0.0)))
    if dashed:
        _draw_dashed_polyline(
            canvas,
            corners,
            color=color,
            x_range=x_range,
            y_range=y_range,
            thickness=thickness,
            closed=True,
        )
        return
    corners_px = _local_to_canvas(corners, canvas.shape[1], canvas.shape[0], x_range, y_range)
    if corners_px.shape[0] == 4:
        cv2.polylines(canvas, [corners_px], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_cover_route_point(canvas, cover, color, label, x_range, y_range):
    route_point = np.asarray((cover or {}).get("route_point_local_xy", []), dtype=np.float32)
    if route_point.shape != (2,):
        return
    route_px = _local_to_canvas(route_point[None, :], canvas.shape[1], canvas.shape[0], x_range, y_range)
    if route_px.shape != (1, 2):
        return
    px = tuple(route_px[0])
    cv2.drawMarker(canvas, px, color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)
    if label:
        cv2.putText(
            canvas,
            str(label),
            (px[0] + 8, px[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def _resolve_collision_point_world_xyz(payload):
    payload = payload or {}
    world_xyz = np.asarray(payload.get("collision_point_world_xyz", []), dtype=np.float32).reshape(-1)
    if world_xyz.size >= 3 and np.all(np.isfinite(world_xyz[:3])):
        return world_xyz[:3]
    world_xyz = np.asarray(payload.get("scene_route_conflict_world_xyz", []), dtype=np.float32).reshape(-1)
    if world_xyz.size >= 3 and np.all(np.isfinite(world_xyz[:3])):
        return world_xyz[:3]
    return np.zeros((0,), dtype=np.float32)


def _draw_cover_collision_point(canvas, cover, ego_matrix, color, label, x_range, y_range):
    world_xyz = _resolve_collision_point_world_xyz(cover)
    if ego_matrix is None:
        return None
    if world_xyz.shape == (3,):
        local_xy = _transform_points_world_xyz_to_local(world_xyz[None, :], ego_matrix)
    else:
        return None
    if local_xy.shape != (1, 2):
        return None
    px = _local_to_canvas(local_xy, canvas.shape[1], canvas.shape[0], x_range, y_range)
    if px.shape != (1, 2):
        return None
    point_px = tuple(px[0])
    cv2.drawMarker(canvas, point_px, color, markerType=cv2.MARKER_STAR, markerSize=22, thickness=2)
    cv2.circle(canvas, point_px, 10, color, 1, cv2.LINE_AA)
    if label:
        cv2.putText(
            canvas,
            str(label),
            (point_px[0] + 10, point_px[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return local_xy[0]


def _draw_local_point_marker(canvas, local_xy, color, label, x_range, y_range):
    local_xy = np.asarray(local_xy, dtype=np.float32)
    if local_xy.shape != (2,):
        return
    pt_px = _local_to_canvas(local_xy[None, :], canvas.shape[1], canvas.shape[0], x_range, y_range)
    if pt_px.shape != (1, 2):
        return
    px = tuple(pt_px[0])
    cv2.circle(canvas, px, 16, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, px, 15, color, 2, cv2.LINE_AA)
    cv2.circle(canvas, px, 8, color, -1, cv2.LINE_AA)
    if label not in (None, ""):
        cv2.putText(
            canvas,
            str(label),
            (px[0] + 10, px[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_cross_labeled_marker(canvas, local_xy, color, label, x_range, y_range):
    local_xy = np.asarray(local_xy, dtype=np.float32)
    if local_xy.shape != (2,):
        return
    pt_px = _local_to_canvas(local_xy[None, :], canvas.shape[1], canvas.shape[0], x_range, y_range)
    if pt_px.shape != (1, 2):
        return
    px = tuple(pt_px[0])
    cv2.circle(canvas, px, 15, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, px, 15, color, 2, cv2.LINE_AA)
    cv2.drawMarker(canvas, px, (255, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=4)
    cv2.drawMarker(canvas, px, color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)
    if label not in (None, ""):
        cv2.putText(
            canvas,
            str(label),
            (px[0] + 8, px[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_segment_end_markers(canvas, segment_local_xy, color, x_range, y_range):
    pts = np.asarray(segment_local_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return
    _draw_local_point_marker(canvas, pts[0], color, None, x_range, y_range)
    _draw_cross_labeled_marker(canvas, pts[-1], color, None, x_range, y_range)


def _draw_world_xyz_marker(canvas, world_xyz, ego_matrix, color, marker_kind, x_range, y_range):
    if ego_matrix is None:
        return
    world_xyz = np.asarray(world_xyz, dtype=np.float32).reshape(-1)
    if world_xyz.shape[0] < 3:
        return
    local_xy = _transform_points_world_xyz_to_local(world_xyz[None, :3], ego_matrix)
    if local_xy.shape != (1, 2):
        return
    if marker_kind == "start":
        _draw_local_point_marker(canvas, local_xy[0], color, None, x_range, y_range)
    elif marker_kind == "end":
        _draw_cross_labeled_marker(canvas, local_xy[0], color, None, x_range, y_range)


def _draw_route_progress_marker(canvas, route_xy, progress_m, color, label, x_range, y_range):
    if route_xy is None:
        return
    pt = _sample_route_point_at_s(route_xy, progress_m)
    if pt is None:
        return
    _draw_local_point_marker(canvas, pt, color, label, x_range, y_range)


def _draw_panel_header(panel, title, subtitle=None, bg_color=(28, 38, 54), fg_color=(245, 245, 245)):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 40), bg_color, -1, cv2.LINE_AA)
    cv2.putText(panel, str(title), (14, 26), cv2.FONT_HERSHEY_DUPLEX, 0.70, fg_color, 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(panel, str(subtitle), (panel.shape[1] - 250, 26), cv2.FONT_HERSHEY_DUPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)


def _draw_text_section(panel, x, y, width, title, lines, color):
    cv2.rectangle(panel, (x, y), (x + width, y + 26), color, -1, cv2.LINE_AA)
    cv2.putText(panel, str(title), (x + 10, y + 18), cv2.FONT_HERSHEY_DUPLEX, 0.50, (248, 248, 248), 1, cv2.LINE_AA)
    y += 38
    for line in lines:
        cv2.putText(panel, str(line), (x + 10, y), cv2.FONT_HERSHEY_DUPLEX, 0.50, (28, 28, 28), 1, cv2.LINE_AA)
        y += 24
        if y >= panel.shape[0] - 24:
            break
    return y + 8


def _find_box_by_id(boxes, actor_id):
    if actor_id is None:
        return None
    try:
        actor_id = int(actor_id)
    except Exception:
        return None
    for box in boxes or []:
        try:
            if int(box.get("id", -1)) == actor_id:
                return box
        except Exception:
            continue
    return None


def _load_future_cover_box_current_frame(image_data_root, base_dir, current_frame_id, current_meas, future_cover):
    if current_meas is None:
        return None
    if int((future_cover or {}).get("exists", 0.0)) <= 0:
        return None
    actor_id = future_cover.get("actor_id", None)
    frame_index = int((future_cover or {}).get("frame_index", -1))
    ego_matrix_current = current_meas.get("ego_matrix", None)
    if actor_id is None or frame_index <= 0 or ego_matrix_current is None:
        return None
    future_frame_id = int(current_frame_id) + int(frame_index)
    future_frame_str = f"{future_frame_id:04d}"
    future_boxes = _load_json_gz_if_exists(
        os.path.join(image_data_root, base_dir, "boxes", f"{future_frame_str}.json.gz")
    ) or []
    future_meas = _load_json_gz_if_exists(
        os.path.join(image_data_root, base_dir, "measurements", f"{future_frame_str}.json.gz")
    ) or {}
    ego_matrix_future = future_meas.get("ego_matrix", None) if isinstance(future_meas, dict) else None
    future_box = _find_box_by_id(future_boxes, actor_id)
    if future_box is None or ego_matrix_future is None:
        return None
    try:
        transform = np.linalg.inv(np.asarray(ego_matrix_current, dtype=np.float32)) @ np.asarray(ego_matrix_future, dtype=np.float32)
    except np.linalg.LinAlgError:
        return None
    return _transform_box_to_current_frame(future_box, transform)


def _cover_subtype(cover):
    return str((cover or {}).get("interaction", {}).get("subtype", "none"))


def _cover_name(cover):
    return str((cover or {}).get("interaction", {}).get("name", "none"))


def _fmt_float(x, fmt="{:.2f}"):
    try:
        x = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return fmt.format(x)


def _fmt_int(x):
    try:
        return str(int(x))
    except Exception:
        return "NA"


def _build_bev_panel(sample, current_boxes, current_meas, x_range, y_range, future_cover_current_box=None):
    canvas = np.full((760, 760, 3), 248, dtype=np.uint8)
    route = np.asarray(sample.get("route", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    route_xy = _route_with_origin(route)
    family_code = int(sample.get("conflict_area_family", 0))
    frame_role = str(((sample.get("stage1_speed_debug") or {}).get("conflict_area") or {}).get("frame_role", "none"))
    _draw_panel_header(
        canvas,
        "BEV / Route / Conflict",
        f"{CONFLICT_FAMILY_NAMES.get(family_code, str(family_code))} | {frame_role}",
    )

    for x in np.arange(x_range[0], x_range[1] + 1e-3, 5.0):
        pts = np.array([[x, y_range[0]], [x, y_range[1]]], dtype=np.float32)
        px = _local_to_canvas(pts, canvas.shape[1], canvas.shape[0], x_range, y_range)
        cv2.line(canvas, tuple(px[0]), tuple(px[1]), (234, 234, 234), 1, cv2.LINE_AA)
    for y in np.arange(y_range[0], y_range[1] + 1e-3, 5.0):
        pts = np.array([[x_range[0], y], [x_range[1], y]], dtype=np.float32)
        px = _local_to_canvas(pts, canvas.shape[1], canvas.shape[0], x_range, y_range)
        cv2.line(canvas, tuple(px[0]), tuple(px[1]), (234, 234, 234), 1, cv2.LINE_AA)

    if route_xy.shape[0] >= 2:
        route_dense, _ = _interpolate_route_with_arclength(route_xy, step_m=0.5)
        if route_dense.shape[0] > 0:
            dense_px = _local_to_canvas(route_dense, canvas.shape[1], canvas.shape[0], x_range, y_range)
            for pt in dense_px:
                cv2.circle(canvas, tuple(pt), 1, (214, 214, 214), -1, cv2.LINE_AA)
        route_px = _local_to_canvas(route_xy, canvas.shape[1], canvas.shape[0], x_range, y_range)
        cv2.polylines(canvas, [route_px], isClosed=False, color=(30, 30, 30), thickness=3, lineType=cv2.LINE_AA)

    for box in current_boxes or []:
        cls = str(box.get("class", "")).lower()
        if cls == "ego_car":
            _draw_box(canvas, box, color=(255, 0, 0), x_range=x_range, y_range=y_range, thickness=3)
            continue
        color = (180, 180, 180) if cls in VEHICLE_CLASSES else (205, 205, 205)
        _draw_box(canvas, box, color=color, x_range=x_range, y_range=y_range, thickness=1)

    stage1_debug = sample.get("stage1_speed_debug") or {}
    current_cover = stage1_debug.get("current_cover") or {}
    future_cover = stage1_debug.get("future_cover") or {}
    conflict_area = stage1_debug.get("conflict_area") or {}
    scene_borrow_context = stage1_debug.get("scene_borrow_context") or {}
    ego_matrix = None if current_meas is None else current_meas.get("ego_matrix", None)

    current_box = _find_box_by_id(current_boxes, current_cover.get("actor_id"))
    future_box = _find_box_by_id(current_boxes, future_cover.get("actor_id"))
    if current_box is not None:
        _draw_box(canvas, current_box, color=(0, 0, 255), x_range=x_range, y_range=y_range, thickness=3)
    if future_box is not None:
        _draw_box(canvas, future_box, color=(0, 215, 255), x_range=x_range, y_range=y_range, thickness=3)
    if future_cover_current_box is not None:
        _draw_box(
            canvas,
            future_cover_current_box,
            color=(0, 215, 255),
            x_range=x_range,
            y_range=y_range,
            thickness=3,
            dashed=True,
        )
    borrow_area_drawn = False
    if ego_matrix is not None:
        corridor_world = np.asarray(scene_borrow_context.get("borrow_segment_world_xyz", []), dtype=np.float32)
        corridor_local_xy = _transform_world_points_to_local_xy(corridor_world, ego_matrix)
        if corridor_local_xy.shape[0] >= 2:
            corridor_px = _local_to_canvas(corridor_local_xy, canvas.shape[1], canvas.shape[0], x_range, y_range)
            cv2.polylines(canvas, [corridor_px], isClosed=False, color=(0, 135, 180), thickness=3, lineType=cv2.LINE_AA)
            _draw_segment_end_markers(canvas, corridor_local_xy, (0, 135, 180), x_range, y_range)

            conflict_start_progress_m = float(conflict_area.get("borrow_conflict_start_progress_m", np.nan))
            conflict_end_progress_m = float(conflict_area.get("borrow_conflict_end_progress_m", np.nan))
            stored_area_segment_world = np.asarray(conflict_area.get("area_segment_world_xyz", []), dtype=np.float32)
            stored_area_start_world = np.asarray(conflict_area.get("area_start_world_xyz", []), dtype=np.float32).reshape(-1)
            stored_area_end_world = np.asarray(conflict_area.get("area_end_world_xyz", []), dtype=np.float32).reshape(-1)
            stored_area_segment_local = _transform_world_points_to_local_xy(stored_area_segment_world, ego_matrix)
            if stored_area_segment_local.shape[0] >= 2:
                seg_px = _local_to_canvas(stored_area_segment_local, canvas.shape[1], canvas.shape[0], x_range, y_range)
                cv2.polylines(canvas, [seg_px], isClosed=False, color=(50, 205, 50), thickness=5, lineType=cv2.LINE_AA)
                _draw_segment_end_markers(canvas, stored_area_segment_local, (50, 205, 50), x_range, y_range)
                _draw_world_xyz_marker(canvas, stored_area_start_world, ego_matrix, (50, 205, 50), "start", x_range, y_range)
                _draw_world_xyz_marker(canvas, stored_area_end_world, ego_matrix, (50, 205, 50), "end", x_range, y_range)
                borrow_area_drawn = True
            elif np.isfinite(conflict_start_progress_m) and np.isfinite(conflict_end_progress_m):
                seg_pts = []
                for s in np.linspace(
                    conflict_start_progress_m,
                    conflict_end_progress_m,
                    num=max(int((conflict_end_progress_m - conflict_start_progress_m) / 0.5) + 2, 2),
                ):
                    pt = _sample_polyline_point_at_s(corridor_local_xy, s)
                    if pt is not None:
                        seg_pts.append(pt)
                if len(seg_pts) >= 2:
                    seg_pts = np.asarray(seg_pts, dtype=np.float32)
                    seg_px = _local_to_canvas(seg_pts, canvas.shape[1], canvas.shape[0], x_range, y_range)
                    cv2.polylines(canvas, [seg_px], isClosed=False, color=(50, 205, 50), thickness=5, lineType=cv2.LINE_AA)
                    _draw_segment_end_markers(canvas, seg_pts, (50, 205, 50), x_range, y_range)
                    _draw_world_xyz_marker(canvas, stored_area_start_world, ego_matrix, (50, 205, 50), "start", x_range, y_range)
                    _draw_world_xyz_marker(canvas, stored_area_end_world, ego_matrix, (50, 205, 50), "end", x_range, y_range)
                    borrow_area_drawn = True
            else:
                _draw_world_xyz_marker(canvas, stored_area_start_world, ego_matrix, (50, 205, 50), "start", x_range, y_range)
                _draw_world_xyz_marker(canvas, stored_area_end_world, ego_matrix, (50, 205, 50), "end", x_range, y_range)

    stored_area_segment_world = np.asarray(conflict_area.get("area_segment_world_xyz", []), dtype=np.float32)
    stored_area_start_world = np.asarray(conflict_area.get("area_start_world_xyz", []), dtype=np.float32).reshape(-1)
    stored_area_end_world = np.asarray(conflict_area.get("area_end_world_xyz", []), dtype=np.float32).reshape(-1)
    if ego_matrix is not None and not borrow_area_drawn and stored_area_segment_world.ndim == 2 and stored_area_segment_world.shape[0] >= 2:
        area_segment_local = _transform_world_points_to_local_xy(stored_area_segment_world, ego_matrix)
        if area_segment_local.shape[0] >= 2:
            seg_px = _local_to_canvas(area_segment_local, canvas.shape[1], canvas.shape[0], x_range, y_range)
            cv2.polylines(canvas, [seg_px], isClosed=False, color=(50, 205, 50), thickness=5, lineType=cv2.LINE_AA)
            _draw_segment_end_markers(canvas, area_segment_local, (50, 205, 50), x_range, y_range)
            _draw_world_xyz_marker(canvas, stored_area_start_world, ego_matrix, (50, 205, 50), "start", x_range, y_range)
            _draw_world_xyz_marker(canvas, stored_area_end_world, ego_matrix, (50, 205, 50), "end", x_range, y_range)
    elif ego_matrix is not None and not borrow_area_drawn:
        _draw_world_xyz_marker(canvas, stored_area_start_world, ego_matrix, (50, 205, 50), "start", x_range, y_range)
        _draw_world_xyz_marker(canvas, stored_area_end_world, ego_matrix, (50, 205, 50), "end", x_range, y_range)

    area_type = str(conflict_area.get("area_type", "none"))
    if area_type == "circle":
        center_world_xyz = np.asarray(conflict_area.get("area_center_world_xyz", []), dtype=np.float32)
        radius_m = float(conflict_area.get("area_radius_m", np.nan))
        ego_matrix = None if current_meas is None else current_meas.get("ego_matrix", None)
        if center_world_xyz.shape == (3,) and ego_matrix is not None and np.isfinite(radius_m):
            center_local = _transform_points_world_xyz_to_local(
                center_world_xyz[None, :3].astype(np.float32),
                ego_matrix,
            )
            if center_local.shape == (1, 2):
                center_px = _local_to_canvas(center_local, canvas.shape[1], canvas.shape[0], x_range, y_range)[0]
                scale_x = canvas.shape[1] / max(float(x_range[1] - x_range[0]), 1e-6)
                radius_px = int(max(radius_m * scale_x, 2.0))
                cv2.circle(canvas, tuple(center_px), radius_px, (180, 0, 180), 3, cv2.LINE_AA)

    return canvas


def _build_text_panel(sample, current_meas):
    panel = np.full((460, 960, 3), 248, dtype=np.uint8)
    stage1_debug = sample.get("stage1_speed_debug") or {}
    current_cover = stage1_debug.get("current_cover") or {}
    future_cover = stage1_debug.get("future_cover") or {}
    conflict_area = stage1_debug.get("conflict_area") or {}
    conflict_phase = stage1_debug.get("conflict_phase") or {}

    base_dir, _ = _resolve_feature_frame_info(sample)
    event_name = _scene_name_from_base_dir(base_dir) or "unknown"
    route_name = str(sample.get("route_name", "unknown"))
    if len(route_name) > 52:
        route_name = route_name[:49] + "..."
    cmd_id = None if current_meas is None else current_meas.get("command", None)
    junction_flag = 0 if current_meas is None else int(bool(current_meas.get("junction", False)))
    speed = 0.0 if current_meas is None else float(current_meas.get("speed", 0.0))

    family_code = int(sample.get("conflict_area_family", 0))
    dir_code = int(sample.get("conflict_area_dir", 0))
    issue_count = int(conflict_area.get("issue_count", 0))
    issue_families = ",".join(str(x) for x in conflict_area.get("issue_families", [])) or "none"
    if issue_count > 0:
        issue_line = f"issue={conflict_area.get('missing_reason', 'none')}"
        if issue_families != "none":
            issue_line += f" [{issue_families}]"
    else:
        issue_line = "issue=none"
    _draw_panel_header(panel, f"{event_name} | frame {int(sample.get('frame_id', -1)):04d}", route_name)
    col_gap = 18
    col_x0 = 10
    col_w = (panel.shape[1] - col_gap - 30) // 2
    col_x1 = col_x0 + col_w + col_gap
    y_left = 58
    y_right = 58
    y_left = _draw_text_section(
        panel,
        col_x0,
        y_left,
        col_w,
        "Scene",
        [
            f"speed={speed:.2f}  cmd={COMMAND_MAP.get(int(cmd_id), str(cmd_id)) if cmd_id is not None else 'NA'}  junction={junction_flag}",
        ],
        (58, 80, 116),
    )
    y_left = _draw_text_section(
        panel,
        col_x0,
        y_left,
        col_w,
        "Conflict",
        [
            f"family={CONFLICT_FAMILY_NAMES.get(family_code, str(family_code))} dir={CONFLICT_DIR_NAMES.get(dir_code, str(dir_code))} active={int(float(sample.get('conflict_area_active', 0.0)) > 0.5)}",
            f"frame start={_fmt_int(sample.get('conflict_area_start_frame', -1))} end={_fmt_int(sample.get('conflict_area_end_frame', -1))} role={conflict_area.get('frame_role', 'none')}",
            f"area s={_fmt_float(conflict_area.get('area_start_s_m', np.nan))} e={_fmt_float(conflict_area.get('area_end_s_m', np.nan))}",
            f"borrow prog s={_fmt_float(conflict_area.get('borrow_conflict_start_progress_m', np.nan))} e={_fmt_float(conflict_area.get('borrow_conflict_end_progress_m', np.nan))}",
            f"src={conflict_area.get('source', 'none')} type={conflict_area.get('area_type', 'none')} reason={conflict_area.get('selection_reason', 'none')}",
            issue_line,
        ],
        (56, 122, 78),
    )
    y_left = _draw_text_section(
        panel,
        col_x0,
        y_left,
        col_w,
        "Phase",
        [
            f"phase={conflict_phase.get('phase', 'none')} active={int(float(conflict_phase.get('active', 0.0)) > 0.5)} family={conflict_phase.get('family', 'none')}",
            f"go_frame={_fmt_int(sample.get('conflict_go_frame', -1))} entry={_fmt_int(conflict_phase.get('entry_frame', -1))} release={_fmt_int(conflict_phase.get('release_frame', -1))}",
            f"role={conflict_phase.get('frame_role', 'none')} src={conflict_phase.get('source', 'none')} reason={conflict_phase.get('release_reason', 'none')} issue={conflict_phase.get('issue_reason', 'none')}",
            f"speed={_fmt_float(conflict_phase.get('speed_mps', np.nan))} stop_th={_fmt_float(conflict_phase.get('stop_speed_thresh_mps', np.nan))}",
        ],
        (146, 98, 42),
    )
    _draw_text_section(
        panel,
        col_x0,
        y_left,
        col_w,
        "Area Debug",
        [
            f"issue_count={issue_count} issue_families={issue_families}",
            f"dir src={conflict_area.get('dir_source', 'none')} cover={conflict_area.get('dir_cover_key', 'none')} frame={_fmt_int(conflict_area.get('dir_frame_id', -1))}",
            f"angle={_fmt_float(conflict_area.get('dir_angle_deg', np.nan))} route={_fmt_float(conflict_area.get('route_heading_deg', np.nan))} actor={_fmt_float(conflict_area.get('actor_heading_deg', np.nan))}",
        ],
        (120, 90, 44),
    )
    y_right = _draw_text_section(
        panel,
        col_x1,
        y_right,
        col_w,
        "Cover",
        [
            f"current: name={_cover_name(current_cover)} subtype={_cover_subtype(current_cover)} actor={_fmt_int(current_cover.get('actor_id', -1))}",
            f"current: dist={_fmt_float(current_cover.get('distance', np.nan))} route_d={_fmt_float(current_cover.get('route_distance_m', np.nan))} cp_s={_fmt_float(current_cover.get('scene_route_conflict_s_m', np.nan))}",
            f"future : name={_cover_name(future_cover)} subtype={_cover_subtype(future_cover)} actor={_fmt_int(future_cover.get('actor_id', -1))}",
            f"future : dE={_fmt_float(future_cover.get('d_ego', np.nan))} dB={_fmt_float(future_cover.get('d_bg', np.nan))} route_d={_fmt_float(future_cover.get('route_distance_m', np.nan))}",
            f"future : conflict_s={_fmt_float(future_cover.get('scene_route_conflict_s_m', np.nan))}",
        ],
        (112, 74, 136),
    )
    _draw_text_section(
        panel,
        col_x1,
        y_right,
        col_w,
        "Debug",
        [
            f"area xyz start={len(conflict_area.get('area_start_world_xyz', []))} end={len(conflict_area.get('area_end_world_xyz', []))} seg={len(conflict_area.get('area_segment_world_xyz', []))}",
            f"borrow xyz s={len(conflict_area.get('borrow_start_world_xyz', []))} e={len(conflict_area.get('borrow_end_world_xyz', []))}",
            f"collision xyz={len(conflict_area.get('collision_point_world_xyz', []))} radius={_fmt_float(conflict_area.get('area_radius_m', np.nan))}",
            f"yld f={_fmt_int(conflict_phase.get('yld_frame_count', 0))} low={_fmt_int(conflict_phase.get('yld_low_speed_frame_count', 0))} stop={_fmt_int(conflict_phase.get('yld_stop_frame_count', 0))}",
            f"v0={_fmt_float(conflict_phase.get('yld_start_speed_mps', np.nan))} vmin={_fmt_float(conflict_phase.get('window_min_speed_mps', np.nan))}@{_fmt_int(conflict_phase.get('window_min_speed_frame', -1))}/{conflict_phase.get('window_min_speed_phase', 'none')} vgo={_fmt_float(conflict_phase.get('yld_go_speed_mps', np.nan))}",
            f"drop={_fmt_float(conflict_phase.get('yld_speed_drop_from_start_mps', np.nan))} ratio={_fmt_float(conflict_phase.get('yld_speed_drop_ratio', np.nan))} prog={_fmt_float(conflict_phase.get('yld_progress_span_m', np.nan))}",
        ],
        (86, 86, 86),
    )
    return panel


def _fit_to_canvas(image, canvas_w, canvas_h, bg_color=(245, 245, 245)):
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
    scale = min(float(canvas_w) / max(src_w, 1), float(canvas_h) / max(src_h, 1))
    new_w = max(int(round(src_w * scale)), 1)
    new_h = max(int(round(src_h * scale)), 1)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
    off_x = (canvas_w - new_w) // 2
    off_y = (canvas_h - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas


def _compose_frame(rgb, bev_panel, text_panel):
    target_h = 960
    right_w = 960
    left_w = 960
    bev_h = 500
    rgb_resized = _fit_to_canvas(rgb, left_w, target_h, bg_color=(0, 0, 0))
    bev_resized = cv2.resize(bev_panel, (right_w, bev_h), interpolation=cv2.INTER_LINEAR)
    text_resized = cv2.resize(text_panel, (right_w, target_h - bev_h), interpolation=cv2.INTER_LINEAR)
    right = np.concatenate([bev_resized, text_resized], axis=0)
    gutter = np.full((target_h, 12, 3), 238, dtype=np.uint8)
    frame = np.concatenate([rgb_resized, gutter, right], axis=1)
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (225, 225, 225), 1, cv2.LINE_AA)
    return frame


def _selection_name(scene_name, route_name):
    return route_name if scene_name is None else f"{scene_name}_{route_name}"


def _default_output_dir():
    return Path(__file__).resolve().parents[1] / "visualizations" / "stage1_label_videos"


def _resolve_output_path(scene_name, route_name, output=None, output_dir=None):
    selection_name = _selection_name(scene_name, route_name)
    safe_name = selection_name.replace("/", "_")
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    if output_dir is None:
        output_dir = _default_output_dir()
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{safe_name}.mp4"


def _render_route_video(route_name, route_samples, image_data_root, fps, x_range, y_range, output_path):
    writer = None
    for sample in route_samples:
        base_dir, frame_str = _resolve_feature_frame_info(sample)
        if base_dir is None or frame_str is None:
            raise RuntimeError(
                f"Missing base_dir/frame_str for route_name={route_name} "
                f"frame_id={sample.get('frame_id')}"
            )
        current_boxes = _load_json_gz_if_exists(
            os.path.join(image_data_root, base_dir, "boxes", f"{frame_str}.json.gz")
        ) or []
        current_meas = _load_json_gz_if_exists(
            os.path.join(image_data_root, base_dir, "measurements", f"{frame_str}.json.gz")
        ) or {}
        rgb = _load_scene_rgb(image_data_root, base_dir, frame_str)
        future_cover = ((sample.get("stage1_speed_debug") or {}).get("future_cover") or {})
        future_cover_current_box = _load_future_cover_box_current_frame(
            image_data_root=image_data_root,
            base_dir=base_dir,
            current_frame_id=int(sample.get("frame_id", -1)),
            current_meas=current_meas,
            future_cover=future_cover,
        )
        bev_panel = _build_bev_panel(
            sample,
            current_boxes,
            current_meas,
            x_range,
            y_range,
            future_cover_current_box=future_cover_current_box,
        )
        text_panel = _build_text_panel(sample, current_meas)
        frame = _compose_frame(rgb, bev_panel, text_panel)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {output_path}")
        writer.write(frame)

    if writer is not None:
        writer.release()


def main():
    parser = argparse.ArgumentParser(description="Generate a lightweight video from precomputed stage1 labels")
    parser.add_argument("--dataset_path", type=str, default=None, help="Directory containing samples_packed.pkl")
    parser.add_argument("--packed_path", type=str, default=None, help="Explicit samples_packed.pkl path")
    parser.add_argument("--image_data_root", type=str, required=True, help="Raw data root")
    parser.add_argument("--route_name", type=str, default=None)
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument(
        "--route_list_txt",
        type=str,
        default=None,
        help="Batch mode. Text file with one route per line: 'route_name' or 'scene_name route_name'.",
    )
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for batch outputs when --route_list_txt is used.",
    )
    parser.add_argument("--x_range", type=float, nargs=2, default=[-15.0, 55.0])
    parser.add_argument("--y_range", type=float, nargs=2, default=[-18.0, 18.0])
    args = parser.parse_args()

    packed_path = _resolve_packed_path(dataset_path=args.dataset_path, packed_path=args.packed_path)
    with open(packed_path, "rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"Expected list in {packed_path}, got {type(samples).__name__}")

    if args.route_list_txt is not None:
        if args.route_name is not None or args.scene_name is not None or args.index is not None:
            raise ValueError("--route_list_txt cannot be combined with --route_name / --scene_name / --index")
        if args.output is not None:
            raise ValueError("Use --output_dir instead of --output in batch mode.")
        requests = _load_route_requests(args.route_list_txt)
    else:
        requests = [{
            "scene_name": args.scene_name,
            "route_name": args.route_name,
            "index": args.index,
        }]

    generated = []
    for req in requests:
        route_name, route_samples = _select_route_samples(
            samples,
            route_name=req.get("route_name"),
            scene_name=req.get("scene_name"),
            index=req.get("index"),
        )

        if args.frame_start is not None:
            route_samples = [s for s in route_samples if int(s.get("frame_id", -1)) >= int(args.frame_start)]
        if args.frame_end is not None:
            route_samples = [s for s in route_samples if int(s.get("frame_id", -1)) <= int(args.frame_end)]
        if args.max_frames is not None:
            route_samples = route_samples[: max(int(args.max_frames), 0)]
        if not route_samples:
            raise ValueError(f"No samples left after frame filtering for route_name={route_name}")

        output_path = _resolve_output_path(
            scene_name=req.get("scene_name"),
            route_name=route_name,
            output=args.output if len(requests) == 1 else None,
            output_dir=args.output_dir,
        )
        _render_route_video(
            route_name=route_name,
            route_samples=route_samples,
            image_data_root=args.image_data_root,
            fps=args.fps,
            x_range=args.x_range,
            y_range=args.y_range,
            output_path=output_path,
        )
        generated.append(str(output_path))
        print(f"Saved lightweight stage1 label video to {output_path}")

    if len(generated) > 1:
        print(f"Generated {len(generated)} videos from one packed load.")


if __name__ == "__main__":
    main()
