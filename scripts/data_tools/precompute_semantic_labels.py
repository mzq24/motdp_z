#!/usr/bin/env python3
"""
Pre-compute semantic behavior labels for all samples and inject into samples_packed.pkl.

This avoids expensive on-the-fly BEV semantic + boxes + measurements IO during training.
Each sample gets 3 new fields:
  - behavior_labels: (num_modes,) int64
  - allowed_flags:   (num_modes,) float32
  - scene_buckets:   (NUM_BUCKET_CATEGORIES,) float32

Usage:
  python scripts/data_tools/precompute_semantic_labels.py \
    --dataset_path /media/z/data/dataset/pdm_lite_mini/train \
    --image_data_root /media/z/data/dataset/pdm_lite_mini \
    --anchor_path dd_baseline/anchors/carla_kmeans_32.npy

  # Also for val split:
  python scripts/data_tools/precompute_semantic_labels.py \
    --dataset_path /media/z/data/dataset/pdm_lite_mini/val \
    --image_data_root /media/z/data/dataset/pdm_lite_mini \
    --anchor_path dd_baseline/anchors/carla_kmeans_32.npy
"""

import os
import sys
import argparse
import pickle
import numpy as np
import json
import gzip
from tqdm import tqdm
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from tools.anchor_semantic_labeler import label_anchors_semantic, classify_scene_buckets, NUM_BUCKET_CATEGORIES


def precompute(dataset_path, image_data_root, anchor_path, bev_ppm=2.0, bev_size=256):
    # Load anchors
    if anchor_path.endswith('.npy'):
        anchor_centers_abs = np.load(anchor_path)
    else:
        with open(anchor_path, 'rb') as f:
            anchor_centers_abs = pickle.load(f)['centers']
    num_modes = anchor_centers_abs.shape[0]
    num_points = anchor_centers_abs.shape[1]
    print(f"Anchor: {anchor_centers_abs.shape} from {anchor_path}")

    # Load packed samples
    packed_path = os.path.join(dataset_path, 'samples_packed.pkl')
    if not os.path.exists(packed_path):
        print(f"ERROR: {packed_path} not found. Run training first to generate it.")
        sys.exit(1)

    print(f"Loading {packed_path}...")
    with open(packed_path, 'rb') as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples")

    # Check how many already have labels
    already_labeled = sum(1 for s in samples if 'behavior_labels' in s)
    if already_labeled == len(samples):
        print(f"All {len(samples)} samples already have labels. Use --force to re-compute.")
        if not args.force:
            return
    elif already_labeled > 0:
        print(f"{already_labeled}/{len(samples)} samples already labeled, will fill missing ones.")

    image_data_root = os.path.realpath(image_data_root)
    labeled = 0
    skipped = 0
    fallback = 0

    for sample in tqdm(samples, desc="Labeling"):
        if 'behavior_labels' in sample and not args.force:
            skipped += 1
            continue

        feature_rel = sample.get('transfuser_bev_feature', '')
        if not feature_rel:
            # No feature path — fallback
            sample['behavior_labels'] = np.zeros(num_modes, dtype=np.int64)
            sample['allowed_flags'] = np.ones(num_modes, dtype=np.float32)
            sample['scene_buckets'] = np.zeros(NUM_BUCKET_CATEGORIES, dtype=np.float32)
            fallback += 1
            continue

        base_dir = os.path.dirname(os.path.dirname(feature_rel))
        frame_str = os.path.basename(feature_rel).replace('_feature.pt', '')

        # Load BEV semantic
        bev_rel = feature_rel.replace('transfuser_feature/', 'bev_semantics/').replace('_feature.pt', '.png')
        bev_path = os.path.join(image_data_root, bev_rel)

        if not os.path.exists(bev_path):
            sample['behavior_labels'] = np.zeros(num_modes, dtype=np.int64)
            sample['allowed_flags'] = np.ones(num_modes, dtype=np.float32)
            sample['scene_buckets'] = np.zeros(NUM_BUCKET_CATEGORIES, dtype=np.float32)
            fallback += 1
            continue

        bev_semantic = np.array(Image.open(bev_path))

        # Load boxes
        boxes = None
        boxes_rel = feature_rel.replace('transfuser_feature/', 'boxes/').replace('_feature.pt', '.json.gz')
        boxes_path = os.path.join(image_data_root, boxes_rel)
        if os.path.exists(boxes_path):
            try:
                with gzip.open(boxes_path, 'rt') as bf:
                    boxes = json.load(bf)
            except Exception:
                pass

        # Load measurements
        measurements = None
        meas_rel = feature_rel.replace('transfuser_feature/', 'measurements/').replace('_feature.pt', '.json.gz')
        meas_path = os.path.join(image_data_root, meas_rel)
        if os.path.exists(meas_path):
            try:
                with gzip.open(meas_path, 'rt') as mf:
                    measurements = json.load(mf)
            except Exception:
                pass

        # Load future frame data for dynamic collision
        ego_matrix_current = None
        future_frames_data = None
        if measurements is not None:
            ego_matrix_current = measurements.get('ego_matrix', None)

        if ego_matrix_current is not None:
            frame_id = int(frame_str)
            future_frames_data = []
            for k in range(1, num_points + 1):
                future_frame_str = f"{frame_id + k:04d}"
                fut_boxes_path = os.path.join(
                    image_data_root, base_dir,
                    'boxes', f'{future_frame_str}.json.gz')
                fut_meas_path = os.path.join(
                    image_data_root, base_dir,
                    'measurements', f'{future_frame_str}.json.gz')
                if os.path.exists(fut_boxes_path) and os.path.exists(fut_meas_path):
                    try:
                        with gzip.open(fut_boxes_path, 'rt') as bf:
                            fut_boxes = json.load(bf)
                        with gzip.open(fut_meas_path, 'rt') as mf:
                            fut_meas = json.load(mf)
                        fut_ego_matrix = fut_meas.get('ego_matrix', None)
                        if fut_ego_matrix is not None:
                            future_frames_data.append((fut_boxes, fut_ego_matrix))
                        else:
                            future_frames_data.append(None)
                    except Exception:
                        future_frames_data.append(None)
                else:
                    future_frames_data.append(None)

        # GT trajectory
        gt_traj = sample.get('ego_waypoints', None)
        if gt_traj is not None:
            if isinstance(gt_traj, np.ndarray):
                gt_traj = gt_traj[1:]
            else:
                gt_traj = np.array(gt_traj)[1:]

        # Compute labels
        behavior_labels, allowed_flags, _ = label_anchors_semantic(
            anchor_centers_abs, bev_semantic,
            ppm=bev_ppm, bev_size=bev_size,
            boxes=boxes,
            measurements=measurements,
            ego_matrix_current=ego_matrix_current,
            future_frames_data=future_frames_data,
            gt_trajectory=gt_traj,
        )

        # Scene buckets
        bucket_flags = classify_scene_buckets(
            measurements=measurements,
            boxes=boxes,
            ego_waypoints=gt_traj,
        )

        sample['behavior_labels'] = behavior_labels          # (num_modes,) int64
        sample['allowed_flags'] = allowed_flags.astype(np.float32)  # (num_modes,) float32
        sample['scene_buckets'] = bucket_flags.astype(np.float32)   # (NUM_BUCKET_CATEGORIES,) float32
        labeled += 1

    print(f"\nDone: labeled={labeled}, skipped={skipped}, fallback={fallback}")

    # Save back (atomic write)
    tmp_path = packed_path + f'.tmp.{os.getpid()}'
    print(f"Saving to {packed_path}...")
    with open(tmp_path, 'wb') as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.rename(tmp_path, packed_path)
    size_mb = os.path.getsize(packed_path) / 1e6
    print(f"Saved ({size_mb:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-compute semantic behavior labels into samples_packed.pkl")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset split (e.g. /media/z/data/dataset/pdm_lite_mini/train)')
    parser.add_argument('--image_data_root', type=str, required=True,
                        help='Root of image data (e.g. /media/z/data/dataset/pdm_lite_mini)')
    parser.add_argument('--anchor_path', type=str, required=True,
                        help='Path to anchor file (.npy or .pkl)')
    parser.add_argument('--bev_ppm', type=float, default=2.0)
    parser.add_argument('--bev_size', type=int, default=256)
    parser.add_argument('--force', action='store_true', help='Re-compute even if labels exist')
    args = parser.parse_args()

    precompute(args.dataset_path, args.image_data_root, args.anchor_path,
               bev_ppm=args.bev_ppm, bev_size=args.bev_size)
