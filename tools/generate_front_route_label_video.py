#!/usr/bin/env python3
"""
Generate a route-level video for route-constrained front-risk labels.

For every sample on the selected route, this tool:
- computes the front-route label
- draws the current route, vehicles, and selected conflict geometry
- overlays the current RGB frame
- writes a single MP4 under visualizations/front_route_videos/

Usage:
  /home/z/anaconda3/envs/dpauto/bin/python tools/generate_front_route_label_video.py \
    --dataset_path /media/z/data/dataset/pdm_lite_mini/train \
    --image_data_root /media/z/data/dataset/pdm_lite_mini \
    --route_name Town12_Rep0_1152_0_route0_11_08_04_25_14
"""

import argparse
import os
import pickle
from pathlib import Path

import cv2
import numpy as np

from scripts.data_tools.precompute_semantic_labels import (
    _compute_front_route_label,
    _load_json_gz_if_exists,
    _resolve_feature_frame_info,
)


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}
BEV_COLORS = np.array([
    [40, 40, 40],
    [128, 128, 128],
    [180, 0, 180],
    [255, 255, 255],
    [0, 255, 255],
    [0, 0, 255],
    [0, 255, 0],
    [0, 200, 255],
    [0, 0, 255],
    [255, 100, 0],
    [255, 255, 0],
], dtype=np.uint8)


def _select_route_samples(samples, route_name=None, index=None):
    if route_name is None:
        if index is None:
            raise ValueError("Either --route_name or --index must be provided.")
        route_name = samples[index].get("route_name")
        if route_name is None:
            raise ValueError("Selected sample does not contain route_name.")

    route_samples = [s for s in samples if s.get("route_name") == route_name]
    if not route_samples:
        raise ValueError(f"No samples found for route_name={route_name}")
    route_samples.sort(key=lambda s: int(s.get("frame_id", -1)))
    return route_name, route_samples


def _oriented_box_corners(position, extent, yaw):
    x, y = float(position[0]), float(position[1])
    half_l, half_w = float(extent[0]), float(extent[1])
    corners = np.array(
        [
            [-half_l, -half_w],
            [-half_l, half_w],
            [half_l, half_w],
            [half_l, -half_w],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + np.array([x, y], dtype=np.float32)


def _to_canvas(points_xy, width, height, xlim, ylim):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts[None, :]
    x_min, x_max = xlim
    y_min, y_max = ylim
    px = (pts[:, 0] - x_min) / max(x_max - x_min, 1e-6) * (width - 1)
    py = (y_max - pts[:, 1]) / max(y_max - y_min, 1e-6) * (height - 1)
    return np.stack([px, py], axis=1).astype(np.int32)


def _draw_polyline(img, pts_xy, color, width, height, xlim, ylim, thickness=1, closed=False):
    pts = np.asarray(pts_xy, dtype=np.float32)
    if len(pts) == 0:
        return
    pts_px = _to_canvas(pts, width, height, xlim, ylim).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_px], isClosed=closed, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_box(img, box, color, width, height, xlim, ylim, thickness=2):
    pos = box.get("position", None)
    extent = box.get("extent", None)
    if pos is None or extent is None or len(pos) < 2 or len(extent) < 2:
        return
    poly = _oriented_box_corners(pos[:2], extent[:2], float(box.get("yaw", 0.0)))
    _draw_polyline(img, poly, color, width, height, xlim, ylim, thickness=thickness, closed=True)


def _colorize_bev(bev_gray):
    bev_gray = np.clip(bev_gray, 0, len(BEV_COLORS) - 1)
    return BEV_COLORS[bev_gray]


def _box_corners_bev(position, extent, yaw, bev_size=256, bev_ppm=2.0):
    cx, cy = float(position[0]), float(position[1])
    hl, hw = float(extent[0]), float(extent[1])
    corners_local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    corners = (rot @ corners_local.T).T + np.array([cx, cy], dtype=np.float32)
    center = bev_size // 2
    cols = center + corners[:, 0] * bev_ppm
    rows = center + corners[:, 1] * bev_ppm
    return np.stack([cols, rows], axis=1).astype(np.int32)


def _draw_boxes_on_bev(bev_color, boxes):
    for box in boxes or []:
        cls = box.get("class", "")
        if cls == "ego_car":
            continue
        pos = box.get("position", None)
        extent = box.get("extent", None)
        if pos is None or extent is None or len(pos) < 2 or len(extent) < 2:
            continue
        color = (0, 255, 0) if cls in VEHICLE_CLASSES else (180, 180, 180)
        corners = _box_corners_bev(pos[:2], extent[:2], float(box.get("yaw", 0.0)))
        cv2.polylines(bev_color, [corners], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    return bev_color


def _load_scene_panel(image_root, base_dir, frame_str, current_boxes):
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

    bev_path = os.path.join(image_root, base_dir, "bev_semantics", f"{frame_str}.png")
    if os.path.exists(bev_path):
        bev_gray = cv2.imread(bev_path, cv2.IMREAD_GRAYSCALE)
        if bev_gray is not None:
            bev_color = _colorize_bev(bev_gray)
            bev_color = _draw_boxes_on_bev(bev_color, current_boxes)
            bev_color = cv2.resize(bev_color, (1024, 512), interpolation=cv2.INTER_NEAREST)
            cv2.putText(
                bev_color,
                "RGB unavailable, showing BEV semantics",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return bev_color

    fallback = np.full((512, 1024, 3), 245, dtype=np.uint8)
    cv2.putText(
        fallback,
        "RGB/BEV unavailable for this frame",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    return fallback


def _load_future_frames(image_root, base_dir, frame_id, num_future):
    future_frames = []
    for k in range(1, num_future + 1):
        future_frame = f"{frame_id + k:04d}"
        boxes_path = os.path.join(image_root, base_dir, "boxes", f"{future_frame}.json.gz")
        meas_path = os.path.join(image_root, base_dir, "measurements", f"{future_frame}.json.gz")
        boxes = _load_json_gz_if_exists(boxes_path)
        meas = _load_json_gz_if_exists(meas_path)
        if boxes is None or meas is None or meas.get("ego_matrix") is None:
            future_frames.append(None)
        else:
            future_frames.append((boxes, meas["ego_matrix"]))
    return future_frames


def _render_world_panel(sample, current_boxes, label, debug, event_name, width, height, xlim, ylim):
    panel = np.full((height, width, 3), 250, dtype=np.uint8)

    route_dense = debug.get("route_dense")
    route_poly = debug.get("route_poly")
    if route_dense is not None and len(route_dense) > 0:
        dense_px = _to_canvas(route_dense, width, height, xlim, ylim)
        for pt in dense_px:
            cv2.circle(panel, tuple(pt), 1, (210, 210, 210), -1, lineType=cv2.LINE_AA)
    if route_poly is not None and len(route_poly) > 0:
        _draw_polyline(panel, route_poly, (20, 20, 20), width, height, xlim, ylim, thickness=2)

    ego_px = _to_canvas(np.array([[0.0, 0.0]], dtype=np.float32), width, height, xlim, ylim)[0]
    cv2.drawMarker(panel, tuple(ego_px), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

    for box in current_boxes or []:
        cls = box.get("class", "")
        if cls == "ego_car":
            continue
        color = (165, 165, 165) if cls in VEHICLE_CLASSES else (200, 200, 200)
        _draw_box(panel, box, color, width, height, xlim, ylim, thickness=1)

    if debug.get("best_current") is not None:
        cur = debug["best_current"]
        _draw_box(panel, cur["box"], (0, 0, 255), width, height, xlim, ylim, thickness=3)
        cover_pt = np.asarray(cur["cover"]["route_point"], dtype=np.float32)
        cover_px = _to_canvas(cover_pt, width, height, xlim, ylim)[0]
        cv2.drawMarker(panel, tuple(cover_px), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=18, thickness=2)
        cv2.line(panel, tuple(ego_px), tuple(cover_px), (0, 0, 255), 1, lineType=cv2.LINE_AA)

    if debug.get("best_future") is not None:
        fut = debug["best_future"]
        if fut.get("current_box") is not None:
            _draw_box(panel, fut["current_box"], (0, 165, 255), width, height, xlim, ylim, thickness=2)
        _draw_box(panel, fut["box_current_frame"], (0, 0, 255), width, height, xlim, ylim, thickness=3)
        cover_pt = np.asarray(fut["cover"]["route_point"], dtype=np.float32)
        bg_pos = np.asarray(fut["bg_pos"], dtype=np.float32)
        cover_px = _to_canvas(cover_pt, width, height, xlim, ylim)[0]
        bg_px = _to_canvas(bg_pos, width, height, xlim, ylim)[0]
        cv2.drawMarker(panel, tuple(cover_px), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=18, thickness=2)
        cv2.circle(panel, tuple(bg_px), 5, (0, 165, 255), -1, lineType=cv2.LINE_AA)
        cv2.line(panel, tuple(ego_px), tuple(cover_px), (0, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.line(panel, tuple(bg_px), tuple(cover_px), (0, 165, 255), 1, lineType=cv2.LINE_AA)

    lines = [
        f"{event_name} | {sample.get('route_name')} | frame {int(sample.get('frame_id', -1)):04d}",
        f"case={int(label['case'])}  dist={float(label['distance']):.2f}m  ttc={float(label['ttc']):.2f}s  risk={float(label['risk']):.3f}",
        f"block={float(label['block_risk']):.3f}  actor={int(label['actor_class'])}  hazard_bin={int(label['hazard_bin'])}",
    ]
    y = 22
    for line in lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
        y += 22

    return panel


def _compose_frame(rgb, panel, sample, label, current_meas):
    rgb_h, rgb_w = rgb.shape[:2]
    panel_h, panel_w = panel.shape[:2]
    target_h = max(rgb_h, panel_h)
    rgb_resized = cv2.resize(rgb, (int(rgb_w * target_h / max(rgb_h, 1)), target_h), interpolation=cv2.INTER_LINEAR)
    panel_resized = cv2.resize(panel, (int(panel_w * target_h / max(panel_h, 1)), target_h), interpolation=cv2.INTER_LINEAR)
    top = np.concatenate([rgb_resized, panel_resized], axis=1)

    bar_h = 108
    bar = np.zeros((bar_h, top.shape[1], 3), dtype=np.uint8)
    speed = float((current_meas or {}).get("speed", 0.0))
    target_speed = float((current_meas or {}).get("target_speed", 0.0))
    line1 = f"speed={speed:.2f} m/s  target_speed={target_speed:.2f} m/s  case={int(label['case'])}  has_lead={float(label['has_lead']):.0f}"
    line2 = f"front_route_distance={float(label['distance']):.2f} m  front_route_ttc={float(label['ttc']):.2f} s  front_route_risk={float(label['risk']):.3f}"
    line3 = f"block_risk={float(label['block_risk']):.3f}  actor_class={int(label['actor_class'])}  block_bin={int(label['block_bin'])}  ttc_bin={int(label['ttc_bin'])}  hazard_bin={int(label['hazard_bin'])}"
    cv2.putText(bar, line1, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, line2, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(bar, line3, (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1, cv2.LINE_AA)
    return np.concatenate([top, bar], axis=0)


def main():
    parser = argparse.ArgumentParser(description="Generate a route-level front-route-label video")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to split dir containing samples_packed.pkl")
    parser.add_argument("--image_data_root", type=str, required=True, help="Raw image/data root")
    parser.add_argument("--route_name", type=str, default=None)
    parser.add_argument("--index", type=int, default=None, help="Fallback: pick route_name from this sample index")
    parser.add_argument("--num_future", type=int, default=6)
    parser.add_argument("--front_corridor_margin_m", type=float, default=0.5)
    parser.add_argument("--front_route_step_m", type=float, default=0.25)
    parser.add_argument("--front_max_distance_m", type=float, default=40.0)
    parser.add_argument("--front_safe_ttc_s", type=float, default=3.0)
    parser.add_argument("--front_max_ttc_s", type=float, default=10.0)
    parser.add_argument("--xlim", type=float, nargs=2, default=[-10.0, 35.0])
    parser.add_argument("--ylim", type=float, nargs=2, default=[-12.0, 12.0])
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    packed_path = os.path.join(args.dataset_path, "samples_packed.pkl")
    with open(packed_path, "rb") as f:
        samples = pickle.load(f)

    route_name, route_samples = _select_route_samples(samples, route_name=args.route_name, index=args.index)
    if args.max_frames is not None:
        route_samples = route_samples[: args.max_frames]

    safe_route = route_name.replace("/", "_")
    project_root = Path(__file__).resolve().parents[1]
    if args.output is None:
        output_dir = project_root / "visualizations" / "front_route_videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"front_route_video_{safe_route}.mp4"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    frames_written = 0
    event_name = "UnknownEvent"

    for sample in route_samples:
        base_dir, frame_str = _resolve_feature_frame_info(sample)
        if base_dir is None or frame_str is None:
            continue
        event_name = base_dir.split(os.sep)[0] if os.sep in base_dir else base_dir.split("/")[0]

        current_boxes = _load_json_gz_if_exists(os.path.join(args.image_data_root, base_dir, "boxes", f"{frame_str}.json.gz"))
        current_meas = _load_json_gz_if_exists(os.path.join(args.image_data_root, base_dir, "measurements", f"{frame_str}.json.gz"))
        if current_boxes is None or current_meas is None:
            continue
        rgb = _load_scene_panel(args.image_data_root, base_dir, frame_str, current_boxes)

        num_future = args.num_future
        if "ego_waypoints" in sample:
            try:
                num_future = max(1, min(num_future, len(sample["ego_waypoints"]) - 1))
            except Exception:
                pass
        future_frames = _load_future_frames(args.image_data_root, base_dir, int(sample["frame_id"]), num_future)

        label, debug = _compute_front_route_label(
            route=sample["route"],
            current_boxes=current_boxes,
            ego_speed=float(current_meas.get("speed", 0.0)),
            ego_matrix_current=current_meas.get("ego_matrix"),
            future_frames_data=future_frames,
            corridor_margin_m=args.front_corridor_margin_m,
            route_step_m=args.front_route_step_m,
            max_distance_m=args.front_max_distance_m,
            safe_ttc_s=args.front_safe_ttc_s,
            max_ttc_s=args.front_max_ttc_s,
            return_debug=True,
        )

        panel = _render_world_panel(
            sample=sample,
            current_boxes=current_boxes,
            label=label,
            debug=debug,
            event_name=event_name,
            width=720,
            height=540,
            xlim=args.xlim,
            ylim=args.ylim,
        )
        frame = _compose_frame(rgb, panel, sample, label, current_meas)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {output_path}")

        writer.write(frame)
        frames_written += 1

    if writer is not None:
        writer.release()

    if frames_written == 0:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"No frames were written for route {route_name}")

    print(f"Saved video to {output_path}")
    print(f"route_name={route_name}")
    print(f"event_name={event_name}")
    print(f"frames={frames_written}")


if __name__ == "__main__":
    main()
