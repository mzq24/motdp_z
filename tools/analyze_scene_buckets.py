"""
Analyze scene bucket distribution across the dataset.

Scans all samples, classifies each into scene buckets, and reports:
  - Per-bucket sample counts and percentages
  - Long-tail (rare) buckets
  - Accident/corner-case moment identification
  - Per-bucket sample paths for filtering/oversampling

Usage:
    python tools/analyze_scene_buckets.py
    python tools/analyze_scene_buckets.py --dataset /path/to/dataset --split train
    python tools/analyze_scene_buckets.py --save-buckets tools/bucket_paths.pkl
"""

import numpy as np
import pickle
import glob
import os
import gzip
import json
import argparse
from collections import defaultdict
from PIL import Image

from anchor_semantic_labeler import (
    classify_scene_buckets, BUCKET_CATEGORIES, NUM_BUCKET_CATEGORIES,
    label_anchors_semantic, BEHAVIOR_NAMES, NUM_BEHAVIORS,
)


def _load_json_gz(path):
    try:
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Analyze scene bucket distribution')
    parser.add_argument('--dataset', type=str,
                        default='/media/z/data/dataset/pdm_lite_mini',
                        help='Dataset root path')
    parser.add_argument('--anchors', type=str,
                        default='wp_tokens.pkl',
                        help='Anchor tokens pkl path')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'],
                        help='Dataset split')
    parser.add_argument('--save-buckets', type=str, default=None,
                        help='Save per-bucket sample paths to pkl (for oversampling)')
    parser.add_argument('--show-samples', type=int, default=3,
                        help='Show N example samples per bucket')
    args = parser.parse_args()

    # Load anchors
    with open(args.anchors, 'rb') as f:
        anchor_data = pickle.load(f)
    anchor_centers_abs = anchor_data['centers']
    num_points = anchor_centers_abs.shape[1]
    print(f"Anchors: {anchor_centers_abs.shape}")

    # Find pkl files
    pkl_files = sorted(glob.glob(os.path.join(args.dataset, args.split, '*.pkl')))
    if not pkl_files:
        pkl_files = sorted(glob.glob(os.path.join(args.dataset, '*.pkl')))
    print(f"Total {args.split} samples: {len(pkl_files)}")

    # Per-bucket tracking
    bucket_counts = np.zeros(NUM_BUCKET_CATEGORIES, dtype=int)
    bucket_sample_paths = defaultdict(list)

    # Behavior stats (co-occurrence with buckets)
    behavior_counts = np.zeros(NUM_BEHAVIORS, dtype=int)

    # Accident moment detection: samples with any hazard active
    accident_samples = []

    # Combined corner-case score: samples with multiple hazards/rare conditions
    corner_case_scores = []

    for i, pkl_path in enumerate(pkl_files):
        with open(pkl_path, 'rb') as f:
            sample = pickle.load(f)

        feature_rel = sample.get('transfuser_bev_feature', '')
        base_dir = os.path.dirname(os.path.dirname(feature_rel))
        frame_str = os.path.basename(feature_rel).replace('_feature.pt', '')

        # Load measurements
        meas_rel = feature_rel.replace('transfuser_feature/', 'measurements/').replace('_feature.pt', '.json.gz')
        meas_path = os.path.join(args.dataset, meas_rel)
        measurements = _load_json_gz(meas_path)

        # Load boxes
        boxes_rel = feature_rel.replace('transfuser_feature/', 'boxes/').replace('_feature.pt', '.json.gz')
        boxes_path = os.path.join(args.dataset, boxes_rel)
        boxes = _load_json_gz(boxes_path)

        # GT trajectory
        gt_traj = sample.get('ego_waypoints', None)
        if gt_traj is not None:
            gt_traj = gt_traj[1:]

        # Classify scene buckets
        flags = classify_scene_buckets(
            measurements=measurements,
            boxes=boxes,
            ego_waypoints=gt_traj,
        )
        bucket_counts += flags.astype(int)

        # Track per-bucket sample paths
        for b_idx in range(NUM_BUCKET_CATEGORIES):
            if flags[b_idx]:
                bucket_sample_paths[BUCKET_CATEGORIES[b_idx]].append(pkl_path)

        # Corner-case score = number of active hazard/rare buckets
        # Hazard buckets: 0-3 (vehicle/walker/light/stop_sign hazard)
        # Other rare: 7,8 (vehicle_front/side), 12 (high_decel)
        rare_indices = [0, 1, 2, 3, 7, 8, 12]
        score = sum(flags[j] for j in rare_indices)
        corner_case_scores.append((score, pkl_path, flags.copy()))

        # Detect accident moments (any hazard active)
        if any(flags[j] for j in [0, 1, 2, 3]):
            active_hazards = [BUCKET_CATEGORIES[j] for j in [0, 1, 2, 3] if flags[j]]
            accident_samples.append((pkl_path, active_hazards))

        # Also compute behavior labels for co-occurrence
        bev_rel = feature_rel.replace('transfuser_feature/', 'bev_semantics/').replace('_feature.pt', '.png')
        bev_path = os.path.join(args.dataset, bev_rel)
        if os.path.exists(bev_path):
            bev_semantic = np.array(Image.open(bev_path))
            # Load future frames for dynamic collision
            ego_matrix_current = None
            future_frames_data = None
            if measurements is not None:
                ego_matrix_current = measurements.get('ego_matrix', None)
            if ego_matrix_current is not None:
                frame_id = int(frame_str)
                future_frames_data = []
                for k in range(1, num_points + 1):
                    future_str = f"{frame_id + k:04d}"
                    fut_b = os.path.join(args.dataset, base_dir, 'boxes', f'{future_str}.json.gz')
                    fut_m = os.path.join(args.dataset, base_dir, 'measurements', f'{future_str}.json.gz')
                    if os.path.exists(fut_b) and os.path.exists(fut_m):
                        fb = _load_json_gz(fut_b)
                        fm = _load_json_gz(fut_m)
                        if fb and fm and fm.get('ego_matrix'):
                            future_frames_data.append((fb, fm['ego_matrix']))
                        else:
                            future_frames_data.append(None)
                    else:
                        future_frames_data.append(None)

            behavior_labels, _, _ = label_anchors_semantic(
                anchor_centers_abs, bev_semantic,
                boxes=boxes, measurements=measurements,
                ego_matrix_current=ego_matrix_current,
                future_frames_data=future_frames_data,
                gt_trajectory=gt_traj,
            )
            for b in behavior_labels:
                behavior_counts[b] += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(pkl_files)} samples...")

    # ===== Report =====
    total = len(pkl_files)
    print(f"\n{'=' * 70}")
    print(f"Scene Bucket Distribution ({total} samples)")
    print(f"{'=' * 70}")

    # Sort by count (ascending = rarest first)
    sorted_indices = np.argsort(bucket_counts)
    for idx in sorted_indices:
        name = BUCKET_CATEGORIES[idx]
        count = bucket_counts[idx]
        pct = 100 * count / max(total, 1)
        bar = '#' * int(pct)
        rarity = " [RARE]" if pct < 5 else (" [LONG-TAIL]" if pct < 10 else "")
        print(f"  {name:22s}: {count:5d} ({pct:5.1f}%){rarity}  {bar}")

    # Accident moments
    print(f"\n{'=' * 70}")
    print(f"Accident/Hazard Moments: {len(accident_samples)}/{total} "
          f"({100 * len(accident_samples) / max(total, 1):.1f}%)")
    print(f"{'=' * 70}")
    if accident_samples and args.show_samples > 0:
        print(f"  First {min(args.show_samples, len(accident_samples))} examples:")
        for path, hazards in accident_samples[:args.show_samples]:
            name = os.path.basename(path).replace('.pkl', '')
            print(f"    {name[:60]}  hazards={hazards}")

    # Top corner-case samples
    corner_case_scores.sort(key=lambda x: -x[0])
    top_corner = [s for s in corner_case_scores if s[0] >= 2]
    print(f"\n{'=' * 70}")
    print(f"High Corner-Case Score (>=2 rare flags): {len(top_corner)}/{total}")
    print(f"{'=' * 70}")
    if top_corner and args.show_samples > 0:
        for score, path, flags in top_corner[:args.show_samples]:
            name = os.path.basename(path).replace('.pkl', '')
            active = [BUCKET_CATEGORIES[j] for j in range(NUM_BUCKET_CATEGORIES) if flags[j]]
            print(f"  score={score}  {name[:50]}  buckets={active}")

    # Behavior distribution (if computed)
    total_anchors = behavior_counts.sum()
    if total_anchors > 0:
        print(f"\n{'=' * 70}")
        print(f"Behavior Label Distribution ({total_anchors} anchor evaluations)")
        print(f"{'=' * 70}")
        for b_idx in range(NUM_BEHAVIORS):
            name = BEHAVIOR_NAMES[b_idx]
            count = behavior_counts[b_idx]
            pct = 100 * count / max(total_anchors, 1)
            bar = '#' * int(pct / 2)
            print(f"  {name:22s}: {count:6d} ({pct:5.1f}%) {bar}")

    # Save per-bucket paths for oversampling
    if args.save_buckets:
        save_data = {
            'bucket_paths': dict(bucket_sample_paths),
            'bucket_counts': {BUCKET_CATEGORIES[i]: int(bucket_counts[i])
                              for i in range(NUM_BUCKET_CATEGORIES)},
            'total_samples': total,
            'accident_samples': [p for p, _ in accident_samples],
            'corner_case_samples': [p for s, p, _ in corner_case_scores if s >= 2],
        }
        os.makedirs(os.path.dirname(args.save_buckets) or '.', exist_ok=True)
        with open(args.save_buckets, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"\nSaved bucket paths to: {os.path.abspath(args.save_buckets)}")


if __name__ == '__main__':
    main()
