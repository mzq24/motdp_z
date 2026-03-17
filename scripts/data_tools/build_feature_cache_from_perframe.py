"""
Build memmap cache from per-frame .pt files (local SSD layout).

Reads {frame_id}_feature.pt and {frame_id}_feature_upsample.pt directly,
producing the same output as build_feature_cache_fp16.py:

Output (in {dataset_root}/tmp_data/):
  feature_index.pkl       - dict: packed_path -> {offset, n_frames}
  bev_features_fp16.bin   - flat (total_frames, 1512, 8, 8) float16
  bev_upsamples_fp16.bin  - flat (total_frames, 64, 32, 32) float16  [2x spatial downsample]

Usage:
  python scripts/build_feature_cache_from_perframe.py --dataset_root /media/z/data/dataset/pdm_lite_mini
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str,
                        default='/media/z/data/dataset/pdm_lite_mini')
    args = parser.parse_args()

    dataset_root = os.path.realpath(args.dataset_root)
    cache_dir = os.path.join(dataset_root, 'tmp_data')
    os.makedirs(cache_dir, exist_ok=True)

    # Step 1: Find all transfuser_feature directories
    print("Scanning for transfuser_feature directories...")
    route_dirs = []  # (event/route/transfuser_feature path, route_features.pt-compatible key)
    for event_entry in sorted(os.scandir(dataset_root), key=lambda e: e.name):
        if not event_entry.is_dir() or event_entry.name in ('tmp_data', 'tmp_data_tg_next'):
            continue
        for route_entry in sorted(os.scandir(event_entry.path), key=lambda e: e.name):
            if not route_entry.is_dir():
                continue
            tf_dir = os.path.join(route_entry.path, 'transfuser_feature')
            if os.path.isdir(tf_dir):
                # Key must match what dataset uses for _feat_index lookup:
                # packed_path = os.path.join(os.path.dirname(bev_feature_path), 'route_features.pt')
                packed_key = os.path.join(tf_dir, 'route_features.pt')
                route_dirs.append((tf_dir, packed_key))

    print(f"Found {len(route_dirs)} routes with transfuser_feature/.")

    # Step 2: Build binary cache
    feat_bin_path = os.path.join(cache_dir, 'bev_features_fp16.bin')
    ups_bin_path = os.path.join(cache_dir, 'bev_upsamples_fp16.bin')
    index_path = os.path.join(cache_dir, 'feature_index.pkl')

    index = {}
    total_frames = 0
    offset = 0

    with open(feat_bin_path, 'wb') as feat_f, open(ups_bin_path, 'wb') as ups_f:
        for tf_dir, packed_key in tqdm(route_dirs, desc="Building cache"):
            # Find all frame IDs in this route
            feat_files = sorted([
                f for f in os.listdir(tf_dir)
                if f.endswith('_feature.pt') and not f.endswith('_feature_upsample.pt')
            ])

            if not feat_files:
                continue

            n_frames = len(feat_files)
            frame_nums = []

            for feat_file in feat_files:
                frame_str = feat_file.replace('_feature.pt', '')
                frame_id = int(frame_str)
                frame_nums.append(frame_id)

                feat_path = os.path.join(tf_dir, feat_file)
                ups_path = os.path.join(tf_dir, feat_file.replace('_feature.pt', '_feature_upsample.pt'))

                # Load and convert to fp16 numpy
                bev_feat = torch.load(feat_path, weights_only=True).squeeze(0).half()  # (1512, 8, 8)
                if os.path.exists(ups_path):
                    bev_ups = torch.load(ups_path, weights_only=True).squeeze(0).half()  # (64, 64, 64)
                    # Spatial 2x downsample: (64, 64, 64) -> (64, 32, 32)
                    bev_ups = F.avg_pool2d(bev_ups.unsqueeze(0), kernel_size=2).squeeze(0)
                else:
                    bev_ups = torch.zeros(64, 32, 32, dtype=torch.float16)

                feat_f.write(bev_feat.numpy().tobytes())
                ups_f.write(bev_ups.numpy().tobytes())

            fn_to_idx = {fn: i for i, fn in enumerate(frame_nums)}
            index[packed_key] = {
                'offset': offset,
                'n_frames': n_frames,
                'frame_num_to_idx': fn_to_idx,
            }
            offset += n_frames
            total_frames += n_frames

    print(f"\nTotal frames: {total_frames}")
    bev_feat_shape = (total_frames, 1512, 8, 8)
    bev_ups_shape = (total_frames, 64, 32, 32)

    feat_bytes = total_frames * 1512 * 8 * 8 * 2
    ups_bytes = total_frames * 64 * 32 * 32 * 2
    print(f"bev_features_fp16.bin:  {feat_bytes / 1e9:.2f} GB")
    print(f"bev_upsamples_fp16.bin: {ups_bytes / 1e9:.2f} GB")
    print(f"Total:                  {(feat_bytes + ups_bytes) / 1e9:.2f} GB")

    # Save index
    with open(index_path, 'wb') as f:
        pickle.dump({
            'index': index,
            'total_frames': total_frames,
            'bev_feat_shape': bev_feat_shape,
            'bev_ups_shape': bev_ups_shape,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nDone!")
    print(f"  {feat_bin_path} ({os.path.getsize(feat_bin_path) / 1e9:.2f} GB)")
    print(f"  {ups_bin_path} ({os.path.getsize(ups_bin_path) / 1e9:.2f} GB)")
    print(f"  {index_path} ({os.path.getsize(index_path) / 1e6:.1f} MB)")
    print(f"\nNow set use_per_frame: false in config to use memmap.")


if __name__ == '__main__':
    main()
