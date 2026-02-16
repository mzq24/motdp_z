"""
Visualize anchor semantic behavior labels on BEV map.

Supports hybrid labeling: BEV + future frame boxes + measurements.

Usage:
    python tools/visualize_anchor_labels.py
    python tools/visualize_anchor_labels.py --n_samples 20 --output tools/viz_output
    python tools/visualize_anchor_labels.py --dataset /path/to/dataset --anchors wp_tokens.pkl
    python tools/visualize_anchor_labels.py --no-future-boxes   # BEV-only mode
"""

import numpy as np
import pickle
import glob
import os
import random
import json
import gzip
import argparse
from collections import defaultdict
from PIL import Image

from anchor_semantic_labeler import (
    label_anchors_semantic, visualize_anchor_labels,
    BEHAVIOR_NAMES, NUM_BEHAVIORS,
)


def _load_json_gz(path):
    """Load a gzip-compressed JSON file."""
    try:
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    except Exception:
        return None


def _load_future_frames(dataset_root, base_dir, frame_id, num_points):
    """Load future frame boxes and measurements for dynamic collision."""
    future_frames_data = []
    for k in range(1, num_points + 1):
        future_str = f"{frame_id + k:04d}"
        fut_boxes_path = os.path.join(dataset_root, base_dir, 'boxes', f'{future_str}.json.gz')
        fut_meas_path = os.path.join(dataset_root, base_dir, 'measurements', f'{future_str}.json.gz')
        if os.path.exists(fut_boxes_path) and os.path.exists(fut_meas_path):
            fut_boxes = _load_json_gz(fut_boxes_path)
            fut_meas = _load_json_gz(fut_meas_path)
            if fut_boxes is not None and fut_meas is not None:
                fut_ego_matrix = fut_meas.get('ego_matrix', None)
                if fut_ego_matrix is not None:
                    future_frames_data.append((fut_boxes, fut_ego_matrix))
                else:
                    future_frames_data.append(None)
            else:
                future_frames_data.append(None)
        else:
            future_frames_data.append(None)
    return future_frames_data


def main():
    parser = argparse.ArgumentParser(description='Visualize anchor semantic labels on BEV')
    parser.add_argument('--dataset', type=str,
                        default='/media/z/data/dataset/pdm_lite_mini',
                        help='Dataset root path')
    parser.add_argument('--anchors', type=str,
                        default='wp_tokens.pkl',
                        help='Anchor tokens pkl path')
    parser.add_argument('--output', type=str,
                        default='tools/viz_output',
                        help='Output directory for visualizations')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'],
                        help='Dataset split')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sample selection')
    parser.add_argument('--diverse', action='store_true', default=True,
                        help='Sample from different scenarios (default: True)')
    parser.add_argument('--no-future-boxes', action='store_true',
                        help='Disable future frame boxes (BEV-only mode)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load anchors
    with open(args.anchors, 'rb') as f:
        anchor_data = pickle.load(f)
    anchor_centers_abs = anchor_data['centers']  # (num_modes, num_points, 2)
    num_points = anchor_centers_abs.shape[1]
    print(f"Anchors: {anchor_centers_abs.shape}")
    print(f"Mode: {'BEV-only' if args.no_future_boxes else 'Hybrid (BEV + future boxes + measurements)'}")

    # Find pkl files
    pkl_files = sorted(glob.glob(os.path.join(args.dataset, args.split, '*.pkl')))
    if not pkl_files:
        pkl_files = sorted(glob.glob(os.path.join(args.dataset, '*.pkl')))
    print(f"Total {args.split} samples: {len(pkl_files)}")

    # Select samples
    random.seed(args.seed)
    if args.diverse:
        scenarios = defaultdict(list)
        for p in pkl_files:
            scenario = os.path.basename(p).rsplit('_', 1)[0]
            scenarios[scenario].append(p)
        keys = list(scenarios.keys())
        print(f"Unique scenarios: {len(keys)}")
        if len(keys) >= args.n_samples:
            chosen = random.sample(keys, args.n_samples)
            selected = [random.choice(scenarios[s]) for s in chosen]
        else:
            selected = random.sample(pkl_files, min(args.n_samples, len(pkl_files)))
    else:
        selected = random.sample(pkl_files, min(args.n_samples, len(pkl_files)))

    # Process and visualize
    stats_all = {name: 0 for name in BEHAVIOR_NAMES}
    total = 0
    dynamic_used = 0

    for i, pkl_path in enumerate(selected):
        with open(pkl_path, 'rb') as f:
            sample = pickle.load(f)

        # Derive paths from feature path
        feature_rel = sample.get('transfuser_bev_feature', '')
        base_dir = os.path.dirname(os.path.dirname(feature_rel))
        frame_str = os.path.basename(feature_rel).replace('_feature.pt', '')

        bev_rel = feature_rel.replace('transfuser_feature/', 'bev_semantics/').replace('_feature.pt', '.png')
        bev_path = os.path.join(args.dataset, bev_rel)

        if not os.path.exists(bev_path):
            print(f"  [{i}] BEV not found: {bev_path}, skipping")
            continue

        bev_semantic = np.array(Image.open(bev_path))

        # Load current frame boxes
        boxes_rel = feature_rel.replace('transfuser_feature/', 'boxes/').replace('_feature.pt', '.json.gz')
        boxes_path = os.path.join(args.dataset, boxes_rel)
        boxes = _load_json_gz(boxes_path)

        # Load measurements
        meas_rel = feature_rel.replace('transfuser_feature/', 'measurements/').replace('_feature.pt', '.json.gz')
        meas_path = os.path.join(args.dataset, meas_rel)
        measurements = _load_json_gz(meas_path)

        # Load future frames for dynamic collision
        ego_matrix_current = None
        future_frames_data = None
        if not args.no_future_boxes and measurements is not None:
            ego_matrix_current = measurements.get('ego_matrix', None)
            if ego_matrix_current is not None:
                frame_id = int(frame_str)
                future_frames_data = _load_future_frames(
                    args.dataset, base_dir, frame_id, num_points)
                n_valid = sum(1 for f in future_frames_data if f is not None)
                if n_valid > 0:
                    dynamic_used += 1

        # GT trajectory (expert never collides)
        gt_traj = sample.get('ego_waypoints', None)
        if gt_traj is not None:
            gt_traj = gt_traj[1:]  # skip origin

        # Label with hybrid approach + GT reference
        behavior_labels, allowed_flags, _ = label_anchors_semantic(
            anchor_centers_abs, bev_semantic,
            boxes=boxes,
            measurements=measurements,
            ego_matrix_current=ego_matrix_current,
            future_frames_data=future_frames_data,
            gt_trajectory=gt_traj,
        )

        # Save visualization
        name = os.path.basename(pkl_path).replace('.pkl', '')
        save_path = os.path.join(args.output, f'{i:02d}_{name}.png')
        visualize_anchor_labels(
            anchor_centers_abs, bev_semantic,
            behavior_labels, allowed_flags,
            save_path=save_path,
            gt_trajectory=gt_traj,
        )

        # Stats
        for b in behavior_labels:
            stats_all[BEHAVIOR_NAMES[b]] += 1
        total += len(behavior_labels)

        n_a = int(np.sum(allowed_flags))
        meas_info = ""
        if measurements:
            flags = []
            if measurements.get('vehicle_hazard'): flags.append('veh_haz')
            if measurements.get('walker_hazard'): flags.append('wal_haz')
            if measurements.get('light_hazard'): flags.append('light_haz')
            if measurements.get('junction'): flags.append('junction')
            if flags:
                meas_info = f"  meas=[{','.join(flags)}]"

        # Find closest anchor to GT
        gt_info = ""
        if gt_traj is not None and len(gt_traj) == num_points:
            dists = np.linalg.norm(
                anchor_centers_abs - gt_traj[np.newaxis, :num_points, :], axis=2)
            mean_dists = dists.mean(axis=1)
            ci = int(np.argmin(mean_dists))
            cb = BEHAVIOR_NAMES[behavior_labels[ci]]
            ca = "ok" if allowed_flags[ci] else "BAD"
            gt_info = f"  GT-closest#{ci}:{cb}({ca},d={mean_dists[ci]:.1f}m)"

        behaviors_str = ', '.join(
            f'{BEHAVIOR_NAMES[b]}={np.sum(behavior_labels == b)}'
            for b in range(NUM_BEHAVIORS) if np.sum(behavior_labels == b) > 0
        )
        print(f"  [{i}] {name[:60]}")
        print(f"       allowed={n_a}/32  {behaviors_str}{meas_info}{gt_info}")

    # Print overall stats
    print(f"\n{'='*60}")
    print(f"Overall Stats ({total} anchor evaluations, {len(selected)} samples)")
    print(f"Dynamic collision used: {dynamic_used}/{len(selected)} samples")
    print(f"{'='*60}")
    for name, count in stats_all.items():
        pct = 100 * count / max(total, 1)
        bar = '#' * int(pct / 2)
        print(f"  {name:22s}: {count:4d} ({pct:5.1f}%) {bar}")
    print(f"\nSaved to: {os.path.abspath(args.output)}/")


if __name__ == '__main__':
    main()
