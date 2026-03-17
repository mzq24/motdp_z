"""
Verify that transfuser_bev_scene.pkl and route_features.pt produce matching features
for routes that have both files.

Usage: python scripts/verify_feature_match.py --dataset_path /path/to/pdm_lite [--num_routes 10]
"""
import argparse
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--num_routes', type=int, default=10, help='Number of routes to check')
    args = parser.parse_args()

    # Find routes that have BOTH files
    both = []
    for event_entry in os.scandir(args.dataset_path):
        if not event_entry.is_dir() or event_entry.name in ('tmp_data', 'tmp_data_tg_next'):
            continue
        for route_entry in os.scandir(event_entry.path):
            if not route_entry.is_dir():
                continue
            route_feat = os.path.join(route_entry.path, 'transfuser_feature', 'route_features.pt')
            scene_pkl = os.path.join(route_entry.path, 'transfuser_bev_scene.pkl')
            if os.path.exists(route_feat) and os.path.exists(scene_pkl):
                both.append((event_entry.name, route_entry.name, route_feat, scene_pkl))

    print(f"Found {len(both)} routes with both files. Checking {min(args.num_routes, len(both))}...")

    for event, route, rf_path, sp_path in tqdm(both[:args.num_routes]):
        # Load route_features.pt
        rf = torch.load(rf_path, weights_only=True)
        rf_frame_to_idx = {fn: i for i, fn in enumerate(rf['frame_nums'])}

        # Load scene pkl
        with open(sp_path, 'rb') as f:
            sp = pickle.load(f)
        sp_frame_to_idx = {f'{int(fid):04d}': i for i, fid in enumerate(sp['frame_ids'])}

        # Find common frames
        common_frames = set(rf_frame_to_idx.keys()) & set(sp_frame_to_idx.keys())
        if not common_frames:
            print(f"  {event}/{route}: NO common frames! rf={len(rf_frame_to_idx)}, sp={len(sp_frame_to_idx)}")
            continue

        # Compare features for all common frames
        bev_diffs = []
        up_diffs = []
        for frame in sorted(common_frames)[:20]:  # check up to 20 frames per route
            rf_idx = rf_frame_to_idx[frame]
            sp_idx = sp_frame_to_idx[frame]

            # route_features.pt: bev_features(1512,8,8) ↔ scene_pkl: fused_features(1512,8,8)
            rf_bev = rf['bev_features'][rf_idx].numpy()
            sp_bev = sp['fused_features'][sp_idx]
            bev_diff = np.abs(rf_bev - sp_bev)
            bev_diffs.append(bev_diff.mean())

            # route_features.pt: bev_upsamples(64,64,64) ↔ scene_pkl: bev_features(64,64,64)
            rf_up = rf['bev_upsamples'][rf_idx].numpy()
            sp_up = sp['bev_features'][sp_idx]
            up_diff = np.abs(rf_up - sp_up)
            up_diffs.append(up_diff.mean())

        bev_mean = np.mean(bev_diffs)
        up_mean = np.mean(up_diffs)
        print(f"  {event}/{route}: "
              f"common={len(common_frames)}, checked={min(20, len(common_frames))}, "
              f"bev_MAE={bev_mean:.6f}, upsample_MAE={up_mean:.6f}")

        del rf, sp


if __name__ == '__main__':
    main()
