"""
Patch existing pkl samples to add target_point_next_hist field.

Reads the measurements.json.gz files (which already contain target_point_next)
and adds target_point_next_hist to each pkl sample that is missing it.
Also removes old samples_packed.pkl so it will be regenerated.

Usage:
    python scripts/patch_target_point_next.py --dataset_path /path/to/tmp_data/train --image_data_root /path/to/pdm_lite
    python scripts/patch_target_point_next.py --dataset_path /path/to/tmp_data/val --image_data_root /path/to/pdm_lite
"""

import argparse
import os
import pickle
import gzip
import json
import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def compute_target_point_next_hist(all_measurements, frame_id, obs_horizon, hz_interval):
    """
    Compute target_point_next_hist for a given frame, matching preprocess_pdm_lite.py logic.

    all_measurements: dict mapping frame_id (int) -> measurement dict
    frame_id: current frame id
    obs_horizon: number of history frames
    hz_interval: frame interval for history sampling
    """
    if frame_id not in all_measurements:
        return None

    current_anno = all_measurements[frame_id]
    current_matrix = np.array(current_anno['ego_matrix'])[:3]
    current_translation = current_matrix[:, 3:4]
    current_rotation = current_matrix[:, :3]

    # History frames at hz_interval spacing, same as preprocess
    history_frame_ids = []
    for step in range(obs_horizon):
        fid = frame_id - (obs_horizon - 1 - step) * hz_interval
        history_frame_ids.append(fid)

    target_points_next_hist = []
    for fid in history_frame_ids:
        if fid not in all_measurements:
            return None  # Can't compute, skip this sample

        frame_anno = all_measurements[fid]

        if 'target_point_next' in frame_anno:
            tp_next = np.array(frame_anno['target_point_next'])[:2]

            past_matrix = np.array(frame_anno['ego_matrix'])[:3]
            past_translation = past_matrix[:, 3:4]
            past_rotation = past_matrix[:, :3]

            tp_next_3d = np.array([tp_next[0], tp_next[1], 0.0]).reshape(3, 1)
            tp_next_world = past_rotation @ tp_next_3d + past_translation
            tp_next_current = current_rotation.T @ (tp_next_world - current_translation)

            target_points_next_hist.append(tp_next_current[:2, 0])
        else:
            target_points_next_hist.append(np.array([0.0, 0.0]))

    return np.array(target_points_next_hist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='Path to train or val pkl directory')
    parser.add_argument('--image_data_root', required=True, help='Root path of image/measurement data')
    parser.add_argument('--obs_horizon', type=int, default=4)
    parser.add_argument('--hz_interval', type=int, default=2, help='Must match preprocess hz_interval (default: 2 for 2Hz from 4Hz data)')
    args = parser.parse_args()

    # Find all pkl files
    pkl_files = sorted(glob.glob(os.path.join(args.dataset_path, '*.pkl')))
    if not pkl_files:
        print(f"No pkl files found in {args.dataset_path}")
        return

    print(f"Found {len(pkl_files)} pkl files")

    # Group by route, skip those that already have the field
    route_to_pkls = defaultdict(list)
    already_done = 0
    for pkl_path in tqdm(pkl_files, desc="Scanning"):
        with open(pkl_path, 'rb') as f:
            sample = pickle.load(f)
        if 'target_point_next_hist' in sample:
            already_done += 1
            continue
        feat_rel = sample.get('transfuser_bev_feature', '')
        route_dir = os.path.dirname(os.path.dirname(feat_rel))
        frame_str = os.path.basename(feat_rel).replace('_feature.pt', '')
        route_to_pkls[route_dir].append((pkl_path, sample, int(frame_str)))

    print(f"Already patched: {already_done}, to patch: {sum(len(v) for v in route_to_pkls.values())} across {len(route_to_pkls)} routes")

    if not route_to_pkls:
        print("Nothing to do.")
        return

    total_patched = 0
    total_skipped = 0

    for route_dir, pkl_list in tqdm(route_to_pkls.items(), desc="Patching routes"):
        meas_dir = os.path.join(args.image_data_root, route_dir, 'measurements')
        if not os.path.isdir(meas_dir):
            total_skipped += len(pkl_list)
            continue

        # Load all measurements for this route into a dict: frame_id -> measurement
        all_measurements = {}
        for mf in sorted(glob.glob(os.path.join(meas_dir, '*.json.gz'))):
            frame_str = os.path.basename(mf).replace('.json.gz', '')
            try:
                with gzip.open(mf, 'rt') as gf:
                    meas = json.load(gf)
                all_measurements[int(frame_str)] = meas
            except Exception:
                continue

        for pkl_path, sample, frame_id in pkl_list:
            tp_next_hist = compute_target_point_next_hist(
                all_measurements, frame_id, args.obs_horizon, args.hz_interval)

            if tp_next_hist is None or tp_next_hist.shape != (args.obs_horizon, 2):
                total_skipped += 1
                continue

            sample['target_point_next_hist'] = tp_next_hist

            tmp_path = pkl_path + '.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.rename(tmp_path, pkl_path)
            total_patched += 1

    print(f"\nDone! Patched {total_patched}, skipped {total_skipped}")

    # Remove old packed file
    packed_path = os.path.join(args.dataset_path, 'samples_packed.pkl')
    if os.path.exists(packed_path):
        os.remove(packed_path)
        print(f"Removed old {packed_path} (will be regenerated on next training run)")


if __name__ == '__main__':
    main()
