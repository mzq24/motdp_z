#!/usr/bin/env python3
"""
Visualize route-constrained front-risk labels for a single sample.

This draws:
- current route polyline and dense route samples
- current-frame vehicle boxes
- the selected current-cover or future-cover vehicle
- the route conflict point used by the label
- summary text with distance / TTC / risk / case

Usage:
  python tools/visualize_front_route_label.py \
    --dataset_path /media/z/data/dataset/pdm_lite_mini/train \
    --image_data_root /media/z/data/dataset/pdm_lite_mini \
    --index 50
"""

import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from scripts.data_tools.precompute_semantic_labels import (
    _compute_front_route_label,
    _load_json_gz_if_exists,
    _resolve_feature_frame_info,
)


def _select_sample(samples, index=None, route_name=None, frame_id=None):
    if index is not None:
        return index, samples[index]

    for idx, sample in enumerate(samples):
        if route_name is not None and sample.get('route_name') != route_name:
            continue
        if frame_id is not None and int(sample.get('frame_id', -1)) != int(frame_id):
            continue
        return idx, sample

    raise ValueError("Could not find a sample matching the requested route_name/frame_id.")


def _oriented_box_corners(position, extent, yaw):
    x, y = float(position[0]), float(position[1])
    half_l, half_w = float(extent[0]), float(extent[1])
    corners = np.array([
        [-half_l, -half_w],
        [-half_l, half_w],
        [half_l, half_w],
        [half_l, -half_w],
    ], dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + np.array([x, y], dtype=np.float32)


def _draw_box(ax, box, color, linestyle='-', linewidth=1.5, alpha=0.8, label=None):
    pos = box.get('position', None)
    extent = box.get('extent', None)
    if pos is None or extent is None or len(pos) < 2 or len(extent) < 2:
        return
    poly = _oriented_box_corners(pos[:2], extent[:2], float(box.get('yaw', 0.0)))
    patch = Polygon(poly, closed=True, fill=False, edgecolor=color, linestyle=linestyle,
                    linewidth=linewidth, alpha=alpha, label=label)
    ax.add_patch(patch)


def _build_future_frames(image_root, base_dir, frame_id, num_future):
    future_frames = []
    for k in range(1, num_future + 1):
        future_frame = f"{frame_id + k:04d}"
        boxes_path = os.path.join(image_root, base_dir, 'boxes', f'{future_frame}.json.gz')
        meas_path = os.path.join(image_root, base_dir, 'measurements', f'{future_frame}.json.gz')
        boxes = _load_json_gz_if_exists(boxes_path)
        meas = _load_json_gz_if_exists(meas_path)
        if boxes is None or meas is None or meas.get('ego_matrix') is None:
            future_frames.append(None)
        else:
            future_frames.append((boxes, meas['ego_matrix']))
    return future_frames


def main():
    parser = argparse.ArgumentParser(description="Visualize route-based front-risk label for one sample")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to split dir containing samples_packed.pkl')
    parser.add_argument('--image_data_root', type=str, required=True,
                        help='Raw image/data root')
    parser.add_argument('--index', type=int, default=None)
    parser.add_argument('--route_name', type=str, default=None)
    parser.add_argument('--frame_id', type=int, default=None)
    parser.add_argument('--num_future', type=int, default=6)
    parser.add_argument('--front_corridor_margin_m', type=float, default=0.5)
    parser.add_argument('--front_route_step_m', type=float, default=0.25)
    parser.add_argument('--front_max_distance_m', type=float, default=40.0)
    parser.add_argument('--front_safe_ttc_s', type=float, default=3.0)
    parser.add_argument('--front_max_ttc_s', type=float, default=10.0)
    parser.add_argument('--xlim', type=float, nargs=2, default=[-10.0, 35.0])
    parser.add_argument('--ylim', type=float, nargs=2, default=[-12.0, 12.0])
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    packed_path = os.path.join(args.dataset_path, 'samples_packed.pkl')
    with open(packed_path, 'rb') as f:
        samples = pickle.load(f)

    sample_idx, sample = _select_sample(
        samples,
        index=args.index,
        route_name=args.route_name,
        frame_id=args.frame_id,
    )

    base_dir, frame_str = _resolve_feature_frame_info(sample)
    if base_dir is None or frame_str is None:
        raise ValueError("Sample does not contain transfuser feature path / frame_id metadata.")

    current_boxes = _load_json_gz_if_exists(
        os.path.join(args.image_data_root, base_dir, 'boxes', f'{frame_str}.json.gz')
    )
    current_meas = _load_json_gz_if_exists(
        os.path.join(args.image_data_root, base_dir, 'measurements', f'{frame_str}.json.gz')
    )
    if current_boxes is None or current_meas is None:
        raise FileNotFoundError("Missing current boxes/measurements for selected sample.")

    num_future = args.num_future
    if 'ego_waypoints' in sample:
        try:
            num_future = max(1, min(num_future, len(sample['ego_waypoints']) - 1))
        except Exception:
            pass
    future_frames = _build_future_frames(
        args.image_data_root,
        base_dir,
        int(sample['frame_id']),
        num_future,
    )

    label, debug = _compute_front_route_label(
        route=sample['route'],
        current_boxes=current_boxes,
        ego_speed=float(current_meas.get('speed', 0.0)),
        ego_matrix_current=current_meas.get('ego_matrix'),
        future_frames_data=future_frames,
        corridor_margin_m=args.front_corridor_margin_m,
        route_step_m=args.front_route_step_m,
        max_distance_m=args.front_max_distance_m,
        safe_ttc_s=args.front_safe_ttc_s,
        max_ttc_s=args.front_max_ttc_s,
        return_debug=True,
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    route_poly = debug['route_poly']
    route_dense = debug['route_dense']
    if route_dense is not None and len(route_dense) > 0:
        ax.scatter(route_dense[:, 0], route_dense[:, 1], s=6, c='lightgray', alpha=0.6, label='route_dense')
    if route_poly is not None and len(route_poly) > 0:
        ax.plot(route_poly[:, 0], route_poly[:, 1], color='black', linewidth=2.0, label='route')

    ax.scatter([0.0], [0.0], c='blue', s=80, marker='x', label='ego')

    for box in current_boxes:
        cls = box.get('class', '')
        if cls == 'ego_car':
            continue
        color = '0.65' if cls in {'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle'} else '0.8'
        _draw_box(ax, box, color=color, linewidth=1.0, alpha=0.5)

    if debug['best_current'] is not None:
        _draw_box(ax, debug['best_current']['box'], color='red', linewidth=2.5, alpha=1.0, label='selected_current_box')
        cover_pt = np.asarray(debug['best_current']['cover']['route_point'])
        ax.scatter([cover_pt[0]], [cover_pt[1]], c='red', s=100, marker='*', label='conflict_point')
        ax.plot([0.0, cover_pt[0]], [0.0, cover_pt[1]], color='red', linestyle=':', linewidth=1.5)

    if debug['best_future'] is not None:
        fut = debug['best_future']
        if fut['current_box'] is not None:
            _draw_box(ax, fut['current_box'], color='orange', linewidth=2.0, alpha=0.9, label='same_actor_current')
        _draw_box(ax, fut['box_current_frame'], color='red', linestyle='--', linewidth=2.5, alpha=1.0,
                  label='selected_future_box_in_current_frame')
        cover_pt = np.asarray(fut['cover']['route_point'])
        bg_pos = np.asarray(fut['bg_pos'])
        ax.scatter([cover_pt[0]], [cover_pt[1]], c='red', s=100, marker='*', label='future_conflict_point')
        ax.scatter([bg_pos[0]], [bg_pos[1]], c='orange', s=60, marker='o', label='bg_current_position')
        ax.plot([bg_pos[0], cover_pt[0]], [bg_pos[1], cover_pt[1]], color='orange', linestyle=':', linewidth=1.5)
        ax.plot([0.0, cover_pt[0]], [0.0, cover_pt[1]], color='red', linestyle=':', linewidth=1.5)

    title = (
        f"front_route_label | sample={sample_idx} | route={sample.get('route_name')} | frame={sample.get('frame_id')}\n"
        f"case={label['case']}  distance={label['distance']:.2f}m  "
        f"ttc={label['ttc']:.2f}s  risk={label['risk']:.3f}  block={label['block_risk']:.3f}\n"
        f"actor_class={label['actor_class']}  block_bin={label['block_bin']}  "
        f"ttc_bin={label['ttc_bin']}  hazard_bin={label['hazard_bin']}"
    )
    ax.set_title(title)
    ax.set_xlabel('x_forward (m)')
    ax.set_ylabel('y_lateral (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(args.xlim[0], args.xlim[1])
    ax.set_ylim(args.ylim[0], args.ylim[1])
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc='upper right', fontsize=9)

    if args.output is None:
        safe_route = str(sample.get('route_name', 'sample')).replace('/', '_')
        project_root = Path(__file__).resolve().parents[1]
        output_dir = project_root / 'visualizations' / 'front_route_labels'
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"front_route_label_{safe_route}_{int(sample.get('frame_id', 0)):04d}.png")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output = str(output_path)
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    print(f"Saved visualization to {args.output}")
    print(label)


if __name__ == '__main__':
    main()
