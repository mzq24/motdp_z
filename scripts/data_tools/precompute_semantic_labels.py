#!/usr/bin/env python3
"""
Pre-compute training-time supervision/cache fields and inject them into samples_packed.pkl.

This avoids expensive on-the-fly BEV semantic + boxes + measurements IO during training,
and also moves cheap-but-frequent tensor assembly out of the dataloader.
Each sample gets these fast fields:
  - behavior_labels: (num_modes,) int64
  - allowed_flags:   (num_modes,) float32
  - scene_buckets:   (NUM_BUCKET_CATEGORIES,) float32
  - energy_targets:  (num_modes, 5) float32
  - energy_active_mask: (num_modes,) bool
  - ego_status:      (obs_horizon, 14) float32

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


FAST_FIELDS = (
    'behavior_labels',
    'allowed_flags',
    'scene_buckets',
    'energy_targets',
    'energy_active_mask',
    'ego_status',
)


def _has_all_fast_fields(sample):
    return all(field in sample for field in FAST_FIELDS)


def _set_semantic_fallback(sample, num_modes):
    sample['behavior_labels'] = np.zeros(num_modes, dtype=np.int64)
    sample['allowed_flags'] = np.ones(num_modes, dtype=np.float32)
    sample['scene_buckets'] = np.zeros(NUM_BUCKET_CATEGORIES, dtype=np.float32)


def _build_energy_targets(behavior_labels, allowed_flags):
    behavior_labels = np.asarray(behavior_labels, dtype=np.int64)
    allowed_flags = np.asarray(allowed_flags, dtype=np.float32)
    energy_targets = np.stack([
        (behavior_labels == 1).astype(np.float32),
        (behavior_labels == 2).astype(np.float32),
        (behavior_labels == 3).astype(np.float32),
        (behavior_labels == 4).astype(np.float32),
        ((behavior_labels >= 5) & (behavior_labels <= 6)).astype(np.float32),
    ], axis=-1)
    energy_active_mask = (allowed_flags < 0.5).astype(np.bool_)
    return energy_targets.astype(np.float32), energy_active_mask


def _extract_target_points(sample):
    target_point_hist = np.asarray(sample['target_point_hist'], dtype=np.float32)
    if target_point_hist.shape[-1] == 4:
        return target_point_hist[..., :2], target_point_hist[..., 2:]

    target_point_next_hist = sample.get('target_point_next_hist', target_point_hist)
    target_point_next_hist = np.asarray(target_point_next_hist, dtype=np.float32)
    return target_point_hist[..., :2], target_point_next_hist[..., :2]


def _build_ego_status(sample):
    speed_hist = sample.get('speed_hist', sample.get('speed'))
    if speed_hist is None:
        raise KeyError("missing 'speed_hist'/'speed' for ego_status precompute")

    target_point_hist, target_point_next_hist = _extract_target_points(sample)
    ego_status = np.concatenate([
        np.asarray(speed_hist, dtype=np.float32)[..., None],
        np.asarray(sample['theta_hist'], dtype=np.float32)[..., None],
        np.asarray(sample['command_hist'], dtype=np.float32),
        target_point_hist,
        target_point_next_hist,
        np.asarray(sample['waypoints_hist'], dtype=np.float32),
    ], axis=-1)
    return ego_status.astype(np.float32)


def precompute(dataset_path, image_data_root, anchor_path, bev_ppm=2.0, bev_size=256, force=False):
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

    complete_fast_fields = sum(1 for s in samples if _has_all_fast_fields(s))
    already_semantic = sum(1 for s in samples if 'behavior_labels' in s and 'allowed_flags' in s and 'scene_buckets' in s)
    already_energy = sum(1 for s in samples if 'energy_targets' in s and 'energy_active_mask' in s)
    already_ego_status = sum(1 for s in samples if 'ego_status' in s)
    print(
        f"Existing fields: semantic={already_semantic}/{len(samples)}, "
        f"energy={already_energy}/{len(samples)}, "
        f"ego_status={already_ego_status}/{len(samples)}, "
        f"complete={complete_fast_fields}/{len(samples)}"
    )
    if complete_fast_fields == len(samples) and not force:
        print(f"All {len(samples)} samples already have all fast fields. Use --force to re-compute.")
        return

    image_data_root = os.path.realpath(image_data_root)
    semantic_computed = 0
    energy_built = 0
    ego_status_built = 0
    skipped = 0
    fallback = 0

    for sample in tqdm(samples, desc="Labeling"):
        needs_semantic = force or any(
            field not in sample for field in ('behavior_labels', 'allowed_flags', 'scene_buckets')
        )
        needs_energy = force or any(
            field not in sample for field in ('energy_targets', 'energy_active_mask')
        )
        needs_ego_status = force or ('ego_status' not in sample)

        if not (needs_semantic or needs_energy or needs_ego_status):
            skipped += 1
            continue

        if needs_semantic:
            feature_rel = sample.get('transfuser_bev_feature', '')
            frame_id = sample.get('frame_id', None)
            if not feature_rel or frame_id is None:
                _set_semantic_fallback(sample, num_modes)
                fallback += 1
            else:
                # Derive base_dir and frame_str from feature_rel + frame_id
                # Supports both per-frame (XXXX_feature.pt) and route-level (route_features.pt) formats
                if 'route_features.pt' in feature_rel:
                    base_dir = os.path.dirname(os.path.dirname(feature_rel))
                    frame_str = f"{frame_id:04d}"
                else:
                    base_dir = os.path.dirname(os.path.dirname(feature_rel))
                    frame_str = os.path.basename(feature_rel).replace('_feature.pt', '')

                bev_rel = os.path.join(base_dir, 'bev_semantics', f'{frame_str}.png')
                bev_path = os.path.join(image_data_root, bev_rel)

                if not os.path.exists(bev_path):
                    _set_semantic_fallback(sample, num_modes)
                    fallback += 1
                else:
                    bev_semantic = np.array(Image.open(bev_path))

                    boxes = None
                    boxes_rel = os.path.join(base_dir, 'boxes', f'{frame_str}.json.gz')
                    boxes_path = os.path.join(image_data_root, boxes_rel)
                    if os.path.exists(boxes_path):
                        try:
                            with gzip.open(boxes_path, 'rt') as bf:
                                boxes = json.load(bf)
                        except Exception:
                            pass

                    measurements = None
                    meas_rel = os.path.join(base_dir, 'measurements', f'{frame_str}.json.gz')
                    meas_path = os.path.join(image_data_root, meas_rel)
                    if os.path.exists(meas_path):
                        try:
                            with gzip.open(meas_path, 'rt') as mf:
                                measurements = json.load(mf)
                        except Exception:
                            pass

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

                    gt_traj = sample.get('ego_waypoints', None)
                    if gt_traj is not None:
                        if isinstance(gt_traj, np.ndarray):
                            gt_traj = gt_traj[1:]
                        else:
                            gt_traj = np.array(gt_traj)[1:]

                    behavior_labels, allowed_flags, _ = label_anchors_semantic(
                        anchor_centers_abs, bev_semantic,
                        ppm=bev_ppm, bev_size=bev_size,
                        boxes=boxes,
                        measurements=measurements,
                        ego_matrix_current=ego_matrix_current,
                        future_frames_data=future_frames_data,
                        gt_trajectory=gt_traj,
                    )

                    bucket_flags = classify_scene_buckets(
                        measurements=measurements,
                        boxes=boxes,
                        ego_waypoints=gt_traj,
                    )

                    sample['behavior_labels'] = behavior_labels
                    sample['allowed_flags'] = allowed_flags.astype(np.float32)
                    sample['scene_buckets'] = bucket_flags.astype(np.float32)
                    semantic_computed += 1

        if needs_energy:
            if 'behavior_labels' not in sample or 'allowed_flags' not in sample:
                _set_semantic_fallback(sample, num_modes)
            energy_targets, energy_active_mask = _build_energy_targets(
                sample['behavior_labels'],
                sample['allowed_flags'],
            )
            sample['energy_targets'] = energy_targets
            sample['energy_active_mask'] = energy_active_mask
            energy_built += 1

        if needs_ego_status:
            sample['ego_status'] = _build_ego_status(sample)
            ego_status_built += 1

    print(
        f"\nDone: semantic={semantic_computed}, energy={energy_built}, "
        f"ego_status={ego_status_built}, skipped={skipped}, fallback={fallback}"
    )

    # Save back (atomic write)
    tmp_path = packed_path + f'.tmp.{os.getpid()}'
    print(f"Saving to {packed_path}...")
    with open(tmp_path, 'wb') as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.rename(tmp_path, packed_path)
    size_mb = os.path.getsize(packed_path) / 1e6
    print(f"Saved ({size_mb:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-compute fast training fields into samples_packed.pkl")
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
               bev_ppm=args.bev_ppm, bev_size=args.bev_size, force=args.force)
