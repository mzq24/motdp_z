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

Optionally, a NAVSIM joint-label sidecar can be provided with route/path/speed
labels generated from raw NAVSIM logs. Labels are matched by token, not by
array position.
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
        ego_input_dim=8,
        label_dir: Optional[str] = None,
        require_labels: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.load_mode = load_mode
        self.ego_input_dim = int(ego_input_dim)
        self.label_dir = Path(label_dir) if label_dir else None
        self.require_labels = bool(require_labels)
        self.has_labels = False
        t0 = time.time()

        if (self.cache_dir / "cache_index.npz").is_file():
            self._load_npy_cache(load_mode=load_mode)
        else:
            self._load_npz_shards(preload=preload)

        n_total = len(self.trajectory)
        print(f"  Total: {n_total} tokens ({time.time() - t0:.1f}s)")
        print(f"  Metadata: {'yes' if self.has_metadata else 'no'}")
        print(f"  Cache layout: {self.cache_layout}, load_mode={self.load_mode}")

        if self.label_dir is not None:
            self._load_label_sidecar(load_mode=load_mode)

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

        if self.has_labels:
            before = len(base_indices)
            label_mask = self.label_row_for_cache[base_indices] >= 0
            if self.require_labels:
                base_indices = base_indices[label_mask]
            matched = int(label_mask.sum())
            print(f"  Label filter: {matched} / {before} samples matched")
            if self.require_labels and len(base_indices) == 0:
                raise ValueError(f"label_dir={self.label_dir} produced zero usable samples")

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
        n_val = max(1, int(len(perm) * val_ratio)) if len(perm) > 0 else 0
        if split in ("train", "training"):
            self.indices = perm[n_val:]
        elif split in ("val", "valid", "validation"):
            self.indices = perm[:n_val]
        elif split in ("all", None):
            self.indices = perm
        else:
            raise ValueError(f"Unknown split: {split}")
        print(f"  {split}: {len(self.indices)} samples")
        if len(self.indices) == 0:
            raise ValueError(f"Split {split} is empty after filtering")

        stats_indices = self.indices[: min(4096, len(self.indices))]
        stats_traj = np.stack([self._trajectory_for_cache_index(int(i)) for i in stats_indices], axis=0)
        self.traj_mean = stats_traj.mean(axis=(0, 1)).astype(np.float32)
        self.traj_std = np.maximum(stats_traj.std(axis=(0, 1)).astype(np.float32), 0.01)
        print(f"  Stats: mean={self.traj_mean.round(2)} std={self.traj_std.round(2)}")

    def _fit_ego_status(self, ego_status: np.ndarray) -> np.ndarray:
        ego_status = np.asarray(ego_status, dtype=np.float32)
        cur_dim = ego_status.shape[-1]
        if cur_dim == self.ego_input_dim:
            return ego_status
        if cur_dim > self.ego_input_dim:
            return ego_status[..., : self.ego_input_dim]
        pad_shape = (*ego_status.shape[:-1], self.ego_input_dim - cur_dim)
        pad = np.zeros(pad_shape, dtype=np.float32)
        return np.concatenate([ego_status, pad], axis=-1)

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

    def _load_label_sidecar(self, load_mode: str) -> None:
        if not self.has_metadata:
            raise ValueError("label_dir requires cache token metadata")
        if self.label_dir is None:
            return
        if not (self.label_dir / "label_index.npz").is_file():
            raise FileNotFoundError(f"missing label_index.npz in {self.label_dir}")

        mmap_mode: Optional[str] = "r" if load_mode in ("auto", "memmap") else None
        index = np.load(self.label_dir / "label_index.npz", allow_pickle=False)
        self.label_tokens = index["tokens"].astype(str)
        self.label_log_names = index["log_names"].astype(str) if "log_names" in index.files else None
        self.label_frame_indices = index["frame_indices"].astype(np.int32) if "frame_indices" in index.files else None
        self.label_trajectory = np.load(self.label_dir / "trajectory.npy", mmap_mode=mmap_mode)
        self.label_route = np.load(self.label_dir / "route.npy", mmap_mode=mmap_mode)
        self.label_route_mask = np.load(self.label_dir / "route_mask.npy", mmap_mode=mmap_mode)
        self.label_path = np.load(self.label_dir / "path.npy", mmap_mode=mmap_mode)
        self.label_path_mask = np.load(self.label_dir / "path_mask.npy", mmap_mode=mmap_mode)
        self.label_speed_profile = np.load(self.label_dir / "speed_profile.npy", mmap_mode=mmap_mode)

        label_row_by_token = {token: row for row, token in enumerate(self.label_tokens)}
        self.label_row_for_cache = np.full(len(self.tokens), -1, dtype=np.int64)
        matched = 0
        for cache_row, token in enumerate(self.tokens):
            label_row = label_row_by_token.get(str(token), -1)
            if label_row >= 0:
                self.label_row_for_cache[cache_row] = label_row
                matched += 1
        self.has_labels = True
        print(f"  Label sidecar: {self.label_dir}")
        print(f"  Label tokens: {len(self.label_tokens)} | cache rows matched: {matched} / {len(self.tokens)}")

    def _label_row(self, cache_index: int) -> int:
        if not self.has_labels:
            return -1
        row = int(self.label_row_for_cache[cache_index])
        if row < 0 and self.require_labels:
            token = self.tokens[cache_index] if self.has_metadata else cache_index
            raise KeyError(f"missing joint label for cache token {token}")
        return row

    def _trajectory_for_cache_index(self, cache_index: int) -> np.ndarray:
        label_row = self._label_row(cache_index)
        if label_row >= 0:
            return np.asarray(self.label_trajectory[label_row], dtype=np.float32)
        traj = np.asarray(self.trajectory[cache_index], dtype=np.float32)
        return traj[..., :2]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        item = {
            "bev_grid": torch.from_numpy(np.asarray(self.bev_grid[i]).copy()),
            "bev_feature": torch.from_numpy(np.asarray(self.bev_feature[i]).copy()),
            "ego_status": torch.from_numpy(self._fit_ego_status(self.ego_status[i]).copy()),
            "trajectory": torch.from_numpy(self._trajectory_for_cache_index(i).copy()),
        }

        label_row = self._label_row(i)
        if label_row >= 0:
            item.update(
                {
                    "route": torch.from_numpy(np.asarray(self.label_route[label_row], dtype=np.float32).copy()),
                    "route_mask": torch.from_numpy(np.asarray(self.label_route_mask[label_row], dtype=np.float32).copy()),
                    "path": torch.from_numpy(np.asarray(self.label_path[label_row], dtype=np.float32).copy()),
                    "path_mask": torch.from_numpy(np.asarray(self.label_path_mask[label_row], dtype=np.float32).copy()),
                    "speed_profile": torch.from_numpy(np.asarray(self.label_speed_profile[label_row], dtype=np.float32).copy()),
                }
            )

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
