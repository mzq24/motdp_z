"""
One-time script: merge all route_features.pt into flat float16 binary files + index.

Output (in {dataset_root}/tmp_data/):
  feature_index.pkl       - dict: packed_path -> {offset, n_frames, frame_num_to_idx}
  bev_features_fp16.bin   - flat (total_frames, 1512, 8, 8) float16
  bev_upsamples_fp16.bin  - flat (total_frames, 64, 32, 32) float16  [2x spatial downsample]

bev_upsamples are spatially downsampled 2x via avg_pool2d to save ~75% space.
Dataset should F.interpolate back to 64x64 when loading.

All DDP ranks can numpy.memmap these files → OS shares physical pages automatically.

Usage:
  python scripts/build_feature_cache_fp16.py
  # Or on HPC:
  python scripts/build_feature_cache_fp16.py --dataset_root /path/to/pdm_lite
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
                        default='/home/users/ntu/wh.huang/scratch/z_projects/dataset/pdm_lite')
    args = parser.parse_args()

    dataset_root = os.path.realpath(args.dataset_root)
    cache_dir = os.path.join(dataset_root, 'tmp_data')
    os.makedirs(cache_dir, exist_ok=True)

    # Step 1: Find all route_features.pt
    print("Scanning for route_features.pt files...")
    route_files = []
    for event_entry in os.scandir(dataset_root):
        if not event_entry.is_dir() or event_entry.name in ('tmp_data', 'tmp_data_tg_next'):
            continue
        for route_entry in os.scandir(event_entry.path):
            if not route_entry.is_dir():
                continue
            feat_path = os.path.join(route_entry.path, 'transfuser_feature', 'route_features.pt')
            if os.path.exists(feat_path):
                route_files.append(feat_path)

    route_files.sort()
    print(f"Found {len(route_files)} route_features.pt files.")

    # Step 2: Single-pass — load each route_features.pt once, append to binary files
    feat_bin_path = os.path.join(cache_dir, 'bev_features_fp16.bin')
    ups_bin_path = os.path.join(cache_dir, 'bev_upsamples_fp16.bin')
    index_path = os.path.join(cache_dir, 'feature_index.pkl')

    index = {}
    total_frames = 0
    offset = 0

    with open(feat_bin_path, 'wb') as feat_f, open(ups_bin_path, 'wb') as ups_f:
        for pp in tqdm(route_files, desc="Building cache"):
            pack = torch.load(pp, weights_only=True)
            frame_nums = pack['frame_nums']
            n_frames = len(frame_nums)
            fn_to_idx = {fn: i for i, fn in enumerate(frame_nums)}

            bev_f = pack['bev_features'].half().numpy()   # (N, 1512, 8, 8)
            # Spatial 2x downsample: (N, 64, 64, 64) -> (N, 64, 32, 32)
            bev_u = F.avg_pool2d(pack['bev_upsamples'].half(), kernel_size=2).numpy()

            feat_f.write(bev_f.tobytes())
            ups_f.write(bev_u.tobytes())

            index[pp] = {
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
    print(f"\nTo use: set feature_cache in dataset config, or it will be auto-detected.")


if __name__ == '__main__':
    main()
