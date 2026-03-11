"""
Convert transfuser_bev_scene.pkl → transfuser_feature/route_features.pt
for routes that are missing route_features.pt.

Drops image_features to save space and unify format.

Usage:
    python scripts/convert_scene_pkl_to_route_features.py --dataset_path /path/to/pdm_lite
"""

import argparse
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='Root path of pdm_lite dataset')
    parser.add_argument('--dry_run', action='store_true', help='Only print what would be done')
    args = parser.parse_args()

    # Scan all routes
    routes = []
    for event_entry in os.scandir(args.dataset_path):
        if not event_entry.is_dir() or event_entry.name in ('tmp_data', 'tmp_data_tg_next'):
            continue
        for route_entry in os.scandir(event_entry.path):
            if route_entry.is_dir():
                routes.append((event_entry.name, route_entry.name, route_entry.path))

    print(f"Found {len(routes)} routes total.")

    to_convert = []
    already_have = 0
    no_source = 0

    for event, route_name, route_path in routes:
        feat_dir = os.path.join(route_path, 'transfuser_feature')
        route_feat_path = os.path.join(feat_dir, 'route_features.pt')
        scene_pkl_path = os.path.join(route_path, 'transfuser_bev_scene.pkl')

        if os.path.exists(route_feat_path):
            already_have += 1
        elif os.path.exists(scene_pkl_path):
            to_convert.append((event, route_name, route_path, scene_pkl_path, feat_dir, route_feat_path))
        else:
            no_source += 1

    print(f"Already have route_features.pt: {already_have}")
    print(f"To convert from scene_pkl: {len(to_convert)}")
    print(f"No source at all: {no_source}")

    if args.dry_run:
        for event, route_name, _, scene_pkl, _, route_feat in to_convert[:10]:
            print(f"  Would convert: {event}/{route_name}")
        return

    converted = 0
    errors = 0
    for event, route_name, route_path, scene_pkl_path, feat_dir, route_feat_path in tqdm(
            to_convert, desc="Converting"):
        try:
            with open(scene_pkl_path, 'rb') as f:
                scene = pickle.load(f)

            frame_ids = scene['frame_ids']
            fused_features = scene['fused_features']    # (N, 1512, 8, 8) numpy
            bev_features = scene['bev_features']        # (N, 64, 64, 64) numpy

            # Convert to route_features.pt format
            # frame_nums: list of zero-padded strings like '0006'
            frame_nums = [f'{int(fid):04d}' for fid in frame_ids]

            os.makedirs(feat_dir, exist_ok=True)

            pack = {
                'frame_nums': frame_nums,
                'bev_features': torch.from_numpy(fused_features),    # (N, 1512, 8, 8)
                'bev_upsamples': torch.from_numpy(bev_features),     # (N, 64, 64, 64)
            }

            # Atomic write
            tmp_path = route_feat_path + f'.tmp.{os.getpid()}'
            torch.save(pack, tmp_path)
            os.rename(tmp_path, route_feat_path)
            converted += 1

        except Exception as e:
            print(f"Error converting {event}/{route_name}: {e}")
            errors += 1

    print(f"\nDone! Converted: {converted}, Errors: {errors}")


if __name__ == '__main__':
    main()
