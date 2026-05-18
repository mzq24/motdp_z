#!/usr/bin/env python3
"""Smoke-test NAVSIM cached dataset with joint label sidecar."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.navsim_cached_dataset import NavSimCachedDataset, collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument("--load-mode", choices=("memmap", "ram"), default="memmap")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ego-input-dim", type=int, default=8)
    parser.add_argument("--max-print", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = NavSimCachedDataset(
        cache_dir=args.cache_dir,
        label_dir=args.label_dir,
        split=args.split,
        load_mode=args.load_mode,
        preload=False,
        dedupe_tokens=True,
        ego_input_dim=args.ego_input_dim,
        require_labels=True,
    )
    print(f"dataset_len={len(dataset)}")
    for i in range(min(args.max_print, len(dataset))):
        item = dataset[i]
        print(
            f"sample[{i}] token={item.get('token')} "
            f"traj={tuple(item['trajectory'].shape)} "
            f"route={tuple(item['route'].shape)} "
            f"route_mask_sum={float(item['route_mask'].sum()):.1f} "
            f"speed={tuple(item['speed_profile'].shape)}"
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    batch = next(iter(loader))
    print("batch:")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"  {key}: len={len(value)} type={type(value[0]).__name__ if value else 'empty'}")


if __name__ == "__main__":
    main()
