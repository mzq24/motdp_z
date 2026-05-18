#!/usr/bin/env python3
"""Smoke-test one NAVSIM joint model forward/backward pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.navsim_cached_dataset import NavSimCachedDataset, collate_fn
from model.navsim_joint_route_speed_diffusion import NavSimJointRouteSpeedDiffusion
from training.train_navsim_joint_route_speed_ddp import read_npz_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--d-ffn", type=int, default=1024)
    parser.add_argument("--ego-input-dim", type=int, default=8)
    parser.add_argument(
        "--traj-stats-path",
        default="/workspace2/z_project/motdp_navsim_norm_stats/navtrain_official_h8_r50_20260518_sparsemask/navtrain_official_h8_r50_sparsemask_abs_stats.npz",
    )
    parser.add_argument(
        "--route-stats-path",
        default="/workspace2/z_project/motdp_navsim_norm_stats/navtrain_official_h8_r50_20260518_sparsemask/navtrain_official_h8_r50_sparsemask_route_abs_stats.npz",
    )
    parser.add_argument(
        "--speed-stats-path",
        default="/workspace2/z_project/motdp_navsim_norm_stats/navtrain_official_h8_r50_20260518_sparsemask/navtrain_official_h8_r50_sparsemask_speed_profile_stats.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dataset = NavSimCachedDataset(
        cache_dir=args.cache_dir,
        label_dir=args.label_dir,
        split="all",
        load_mode="memmap",
        preload=False,
        dedupe_tokens=True,
        ego_input_dim=args.ego_input_dim,
        require_labels=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

    model = NavSimJointRouteSpeedDiffusion(
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        d_ffn=args.d_ffn,
        ego_input_dim=args.ego_input_dim,
    ).to(device)
    traj_mean, traj_std = read_npz_pair(args.traj_stats_path, ("abs_mean",), ("abs_std",))
    route_mean, route_std = read_npz_pair(args.route_stats_path, ("route_abs_mean", "abs_mean"), ("route_abs_std", "abs_std"))
    speed_mean, speed_std = read_npz_pair(args.speed_stats_path, ("speed_profile_mean",), ("speed_profile_std",))
    model.set_normalization_stats(
        np.asarray(traj_mean),
        np.asarray(traj_std),
        np.asarray(route_mean),
        np.asarray(route_std),
        np.asarray(speed_mean),
        np.asarray(speed_std),
    )

    model.train()
    loss, info = model.compute_loss(batch)
    loss.backward()
    print(f"loss={float(loss.item()):.6f}")
    for key, value in info.items():
        print(f"{key}={value}")
    finite_grads = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    print(f"finite_grads={finite_grads}")
    if not torch.isfinite(loss) or not finite_grads:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
