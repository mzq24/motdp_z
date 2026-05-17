"""
Dataset utilities for cached NavSim BEV features.

Two cache layouts are supported:

1. Legacy NPZ shards:
   bev_cache_shard*.npz plus optional *_meta.npz sidecars.
   This path preloads arrays into RAM and is slow because npz is a zip
   container. It is kept for compatibility.

2. Consolidated NPY cache:
   cache_index.npz plus bev_grid.npy / bev_feature.npy / ego_status.npy /
   trajectory.npy. This path supports memory mapping and starts quickly.
"""

from __future__ import annotations

import glob
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NavSimCachedDataset(Dataset):
    def __init__(
        self,
        cache_dir="/workspace2/z_project/motdp_bev_cache_official_npy",
        split="train",
        val_ratio=0.05,
        seed=42,
        preload=True,
        token_filter_file=None,
        dedupe_tokens=False,
        load_mode="auto",
    ):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.load_mode = load_mode
        t0 = time.time()

        if (self.cache_dir / "cache_index.npz").is_file():
            self._load_npy_cache(load_mode=load_mode)
        else:
            self._load_npz_shards(preload=preload)

        n_total = len(self.trajectory)
        print(f"  Total: {n_total} tokens ({time.time() - t0:.1f}s)")
        print(f"  Metadata: {'yes' if self.has_metadata else 'no'}")
        print(f"  Cache layout: {self.cache_layout}, load_mode={self.load_mode}")

        base_indices = np.arange(n_total, dtype=np.int64)
        if token_filter_file:
            if not self.has_metadata:
                raise ValueError("token_filter_file requires cache token metadata")
            allowed_tokens = set(Path(token_filter_file).read_text().split())
            base_indices = np.asarray(
                [i for i in base_indices if self.tokens[i] in allowed_tokens],
                dtype=np.int64,
            )
            print(f"  Token filter: {len(base_indices)} / {n_total} samples")

        if dedupe_tokens:
            if not self.has_metadata:
                raise ValueError("dedupe_tokens requires cache token metadata")
            seen = set()
            deduped = []
            for i in base_indices:
                token = self.tokens[i]
                if token in seen:
                    continue
                seen.add(token)
                deduped.append(i)
            base_indices = np.asarray(deduped, dtype=np.int64)
            print(f"  Dedupe tokens: {len(base_indices)} samples")

        rng = np.random.RandomState(seed)
        perm = rng.permutation(base_indices)
        n_val = max(1, int(len(perm) * val_ratio))
        if split in ("train", "training"):
            self.indices = perm[n_val:]
        elif split in ("val", "valid", "validation"):
            self.indices = perm[:n_val]
        elif split in ("all", None):
            self.indices = perm
        else:
            raise ValueError(f"Unknown split: {split}")
        print(f"  {split}: {len(self.indices)} samples")

        stats_indices = self.indices[: min(4096, len(self.indices))]
        stats_traj = np.asarray(self.trajectory[stats_indices], dtype=np.float32)
        self.traj_mean = stats_traj.mean(axis=(0, 1)).astype(np.float32)
        self.traj_std = np.maximum(stats_traj.std(axis=(0, 1)).astype(np.float32), 0.01)
        print(f"  Stats: mean={self.traj_mean.round(2)} std={self.traj_std.round(2)}")

    def _load_npy_cache(self, load_mode: str) -> None:
        if load_mode == "auto":
            load_mode = "memmap"
        if load_mode not in ("memmap", "ram"):
            raise ValueError(f"NPY cache load_mode must be memmap or ram, got {load_mode}")
        self.load_mode = load_mode
        self.cache_layout = "npy"

        mmap_mode: Optional[str] = "r" if load_mode == "memmap" else None
        print(f"Loading consolidated NPY cache from {self.cache_dir}...")
        self.bev_grid = np.load(self.cache_dir / "bev_grid.npy", mmap_mode=mmap_mode)
        self.bev_feature = np.load(self.cache_dir / "bev_feature.npy", mmap_mode=mmap_mode)
        self.ego_status = np.load(self.cache_dir / "ego_status.npy", mmap_mode=mmap_mode)
        self.trajectory = np.load(self.cache_dir / "trajectory.npy", mmap_mode=mmap_mode)

        index = np.load(self.cache_dir / "cache_index.npz", allow_pickle=False)
        self.tokens = index["tokens"].astype(str)
        self.log_names = index["log_names"].astype(str)
        self.frame_indices = index["frame_indices"].astype(np.int32)
        self.has_metadata = True

    def _load_npz_shards(self, preload: bool) -> None:
        if not preload:
            raise ValueError("Legacy NPZ shards require preload=True; convert to NPY for memmap mode")
        self.cache_layout = "npz"
        self.load_mode = "ram"
        paths = sorted(
            p for p in glob.glob(str(self.cache_dir / "bev_cache_shard*.npz"))
            if not Path(p).stem.endswith("_meta") and not Path(p).stem.endswith("_tmp")
        )
        if not paths:
            raise FileNotFoundError(f"No supported cache files in {self.cache_dir}")
        print(f"Loading {len(paths)} NPZ shards from {self.cache_dir}...")

        bg, bf, eg, tr = [], [], [], []
        tokens, log_names, frame_indices = [], [], []
        has_metadata = True
        t0 = time.time()
        for p in paths:
            shard_path = Path(p)
            d = np.load(shard_path)
            bg.append(d["bev_grid"])
            bf.append(d["bev_feature"])
            eg.append(d["ego_status"].astype(np.float32))
            tr.append(d["trajectory"].astype(np.float32))

            if {"tokens", "log_names", "frame_indices"}.issubset(d.files):
                meta = d
            else:
                sidecar = shard_path.with_name(shard_path.stem + "_meta.npz")
                meta = np.load(sidecar) if sidecar.is_file() else None

            if meta is not None and {"tokens", "log_names", "frame_indices"}.issubset(meta.files):
                tokens.append(meta["tokens"].astype(str))
                log_names.append(meta["log_names"].astype(str))
                frame_indices.append(meta["frame_indices"].astype(np.int32))
            else:
                has_metadata = False
            print(f"  {shard_path.name}: {len(tr[-1])} tokens ({time.time() - t0:.0f}s)")

        self.bev_grid = np.concatenate(bg)
        self.bev_feature = np.concatenate(bf)
        self.ego_status = np.concatenate(eg)
        self.trajectory = np.concatenate(tr)
        self.has_metadata = has_metadata
        if has_metadata:
            self.tokens = np.concatenate(tokens)
            self.log_names = np.concatenate(log_names)
            self.frame_indices = np.concatenate(frame_indices)
        else:
            self.tokens = None
            self.log_names = None
            self.frame_indices = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        item = {
            "bev_grid": torch.from_numpy(np.asarray(self.bev_grid[i]).copy()),
            "bev_feature": torch.from_numpy(np.asarray(self.bev_feature[i]).copy()),
            "ego_status": torch.from_numpy(np.asarray(self.ego_status[i], dtype=np.float32).copy()),
            "trajectory": torch.from_numpy(np.asarray(self.trajectory[i], dtype=np.float32).copy()),
        }
        if self.has_metadata:
            item.update(
                {
                    "token": self.tokens[i],
                    "log_name": self.log_names[i],
                    "frame_idx": int(self.frame_indices[i]),
                }
            )
        return item

    def get_traj_stats(self):
        return self.traj_mean, self.traj_std


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        values = [b[k] for b in batch]
        out[k] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return out
