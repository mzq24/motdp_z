#!/usr/bin/env python3
"""
Build a full-resolution (64x64) BEV upsample memmap cache from an existing
downsampled cache.

Input (in {cache_dir}):
  feature_index{suffix}.pkl
  bev_upsamples_fp16{suffix}.bin         # (N, 64, 32, 32) or already (N, 64, 64, 64)

Output:
  feature_index{suffix}_fullres.pkl
  bev_upsamples_fp16{suffix}_fullres.bin # (N, 64, 64, 64)

The dataset loader auto-detects the *_fullres files and skips per-sample
F.interpolate in __getitem__, which reduces dataloader CPU cost.

Example:
  python scripts/data_tools/upsample_bev_upsample_cache.py \
    --dataset_root /workspace1/z_project/dataset/pdm_lite \
    --feature_suffix ensemble \
    --batch_size 256
"""
import argparse
import os
import pickle
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/workspace1/z_project/dataset/pdm_lite',
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='Override cache dir. Defaults to {dataset_root}/tmp_data.',
    )
    parser.add_argument(
        '--feature_suffix',
        type=str,
        default='',
        help="Suffix for cache files, e.g. 'ensemble' -> *_ensemble.bin",
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Number of frames to upsample per batch.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing *_fullres outputs if present.',
    )
    args = parser.parse_args()

    dataset_root = os.path.realpath(args.dataset_root)
    cache_dir = os.path.realpath(args.cache_dir) if args.cache_dir else os.path.join(dataset_root, 'tmp_data')
    sfx = f'_{args.feature_suffix}' if args.feature_suffix else ''

    src_index_path = os.path.join(cache_dir, f'feature_index{sfx}.pkl')
    src_ups_path = os.path.join(cache_dir, f'bev_upsamples_fp16{sfx}.bin')
    dst_index_path = os.path.join(cache_dir, f'feature_index{sfx}_fullres.pkl')
    dst_ups_path = os.path.join(cache_dir, f'bev_upsamples_fp16{sfx}_fullres.bin')

    if not os.path.exists(src_index_path):
        raise FileNotFoundError(f"Missing source index: {src_index_path}")
    if not os.path.exists(src_ups_path):
        raise FileNotFoundError(f"Missing source upsample bin: {src_ups_path}")
    if (os.path.exists(dst_index_path) or os.path.exists(dst_ups_path)) and not args.overwrite:
        raise FileExistsError(
            f"Fullres outputs already exist. Re-run with --overwrite:\n"
            f"  {dst_index_path}\n"
            f"  {dst_ups_path}"
        )

    with open(src_index_path, 'rb') as f:
        src_meta = pickle.load(f)

    src_shape = tuple(src_meta['bev_ups_shape'])
    if len(src_shape) != 4 or src_shape[1] != 64:
        raise ValueError(f"Unexpected bev_ups_shape in {src_index_path}: {src_shape}")

    total_frames = int(src_shape[0])
    dst_shape = (total_frames, 64, 64, 64)

    if src_shape[-2:] == (64, 64):
        print("Source cache is already full-resolution; copying to *_fullres outputs...")
        shutil.copyfile(src_ups_path, dst_ups_path)
    else:
        if src_shape[-2:] != (32, 32):
            raise ValueError(f"Unsupported source spatial shape: {src_shape[-2:]}")

        src_mmap = np.memmap(src_ups_path, dtype=np.float16, mode='r', shape=src_shape)
        tmp_dst_path = dst_ups_path + f'.tmp.{os.getpid()}'
        with open(tmp_dst_path, 'wb') as dst_f:
            for start in tqdm(range(0, total_frames, args.batch_size), desc='Upsampling cache'):
                end = min(start + args.batch_size, total_frames)
                batch = torch.from_numpy(src_mmap[start:end].copy()).float()
                batch = F.interpolate(batch, size=(64, 64), mode='bilinear', align_corners=False)
                dst_f.write(batch.half().numpy().tobytes())
        os.replace(tmp_dst_path, dst_ups_path)

    dst_meta = dict(src_meta)
    dst_meta['bev_ups_shape'] = dst_shape
    dst_meta['ups_cache_variant'] = 'fullres_64x64'
    dst_meta['ups_cache_source'] = os.path.basename(src_ups_path)
    with open(dst_index_path, 'wb') as f:
        pickle.dump(dst_meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    ups_bytes = total_frames * 64 * 64 * 64 * 2
    print("Done!")
    print(f"  source: {src_ups_path}")
    print(f"  output: {dst_ups_path} ({ups_bytes / 1e9:.2f} GB expected)")
    print(f"  index:  {dst_index_path}")
    print("\nDataset will auto-detect *_fullres cache and skip per-sample interpolate.")


if __name__ == '__main__':
    main()
