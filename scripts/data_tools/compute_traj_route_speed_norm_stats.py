#!/usr/bin/env python3
"""Compute per-point normalization stats for traj, route, and speed profile.

The output intentionally keeps the existing Route-B file contract:
- abs_stats.npz: abs_mean / abs_std for trajectory, shape (T, 2)
- route_abs_stats.npz: route_abs_mean / route_abs_std, shape (R, 2)

It also writes:
- speed_profile_stats.npz: speed_profile_mean / speed_profile_std, shape (T,)
- traj_route_speed_norm_stats.npz: all arrays plus counts in one bundle

Inputs are raw MoT-DP samples. Packed datasets with train/val/samples_packed.pkl
are supported and preferred because they avoid many small-file opens.
"""

import argparse
import glob
import os
import pickle
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import torch
except Exception:  # pragma: no cover - torch is present in normal envs.
    torch = None


class RunningPointStats:
    def __init__(self, shape: Tuple[int, ...], name: str):
        self.name = name
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sumsq = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape, dtype=np.float64)

    def update(self, values: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self.sum.shape:
            raise ValueError(f"{self.name}: expected {self.sum.shape}, got {values.shape}")
        finite = np.isfinite(values)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != values.shape:
                if values.ndim == 2 and mask.shape == values.shape[:-1]:
                    mask = np.broadcast_to(mask[..., None], values.shape)
                else:
                    raise ValueError(f"{self.name}: mask {mask.shape} incompatible with {values.shape}")
            finite = finite & mask
        safe_values = np.where(finite, values, 0.0)
        self.sum += safe_values
        self.sumsq += safe_values * safe_values
        self.count += finite.astype(np.float64)

    def finalize(self, min_std: float) -> Tuple[np.ndarray, np.ndarray]:
        if np.any(self.count <= 0):
            missing = int(np.sum(self.count <= 0))
            raise ValueError(f"{self.name}: {missing} positions have zero valid samples")
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - mean * mean, 0.0)
        std = np.maximum(np.sqrt(var), float(min_std))
        return mean.astype(np.float32), std.astype(np.float32)


def _to_numpy(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _iter_sample_files(dataset_path: Path) -> Iterator[Path]:
    for pattern in ("train/*.pkl", "val/*.pkl", "*.pkl"):
        for p in sorted(dataset_path.glob(pattern)):
            if p.name == "samples_packed.pkl":
                continue
            yield p


def _iter_samples(dataset_path: Path, split: str, max_samples: Optional[int]) -> Iterator[dict]:
    packed_candidates = []
    if split in ("train", "all"):
        packed_candidates.append(dataset_path / "train" / "samples_packed.pkl")
    if split in ("val", "all"):
        packed_candidates.append(dataset_path / "val" / "samples_packed.pkl")
    if split in ("direct", "all"):
        packed_candidates.append(dataset_path / "samples_packed.pkl")

    packed_paths = [p for p in packed_candidates if p.exists()]
    yielded = 0
    if packed_paths:
        for packed_path in packed_paths:
            with packed_path.open("rb") as f:
                samples = pickle.load(f)
            desc = f"{packed_path.parent.name}/samples_packed"
            for sample in tqdm(samples, desc=desc):
                yield sample
                yielded += 1
                if max_samples is not None and yielded >= max_samples:
                    return
        return

    files = list(_iter_sample_files(dataset_path))
    if split != "all":
        files = [p for p in files if p.parent.name == split or split == "direct"]
    if max_samples is not None:
        files = files[:max_samples]
    for pkl_path in tqdm(files, desc="sample pkl"):
        with pkl_path.open("rb") as f:
            yield pickle.load(f)


def _get_traj(sample: dict, horizon: int) -> Optional[np.ndarray]:
    traj = _to_numpy(sample.get("agent_pos"))
    if traj is None:
        ego_waypoints = _to_numpy(sample.get("ego_waypoints"))
        if ego_waypoints is not None and ego_waypoints.shape[0] > 1:
            traj = ego_waypoints[1:]
    if traj is None or traj.ndim < 2 or traj.shape[-1] < 2:
        return None
    return _fit_polyline_num_points(traj, horizon)



def _fit_polyline_num_points(points: np.ndarray, target_count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[-1] < 2:
        raise ValueError(f"polyline must be (N,2+), got {points.shape}")
    points = points[:, :2]
    n = points.shape[0]
    if n == target_count:
        return points
    if n == 0:
        return np.zeros((target_count, 2), dtype=np.float32)
    if n > target_count:
        return points[:target_count]
    if n == 1:
        step = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        step = points[-1] - points[-2]
        if float(np.linalg.norm(step)) < 1e-4:
            step = np.asarray([1.0, 0.0], dtype=np.float32)
    extra_steps = np.arange(1, target_count - n + 1, dtype=np.float32)[:, None]
    extra = points[-1:] + extra_steps * step[None]
    return np.concatenate([points, extra.astype(np.float32)], axis=0)


def _fit_mask_num_points(mask: Optional[np.ndarray], target_count: int) -> Optional[np.ndarray]:
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=np.float32).reshape(-1) > 0.5
    if mask.shape[0] == target_count:
        return mask
    if mask.shape[0] > target_count:
        return mask[:target_count]
    fill = bool(mask[-1]) if mask.shape[0] else True
    pad = np.full((target_count - mask.shape[0],), fill, dtype=bool)
    return np.concatenate([mask, pad], axis=0)

def _get_route(sample: dict, route_points: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    route = _to_numpy(sample.get("route"))
    if route is None or route.ndim < 2 or route.shape[-1] < 2:
        return None, None
    route = _fit_polyline_num_points(route, route_points)
    mask = _to_numpy(sample.get("route_mask"))
    if mask is None:
        mask = _to_numpy(sample.get("path_mask"))
    mask = _fit_mask_num_points(mask, route_points)
    return route, mask


def _derive_speed_profile(traj: np.ndarray, dt: float) -> np.ndarray:
    first_disp = traj[:1]
    if traj.shape[0] > 1:
        disp = np.concatenate([first_disp, traj[1:] - traj[:-1]], axis=0)
    else:
        disp = first_disp
    return np.linalg.norm(disp, axis=-1).astype(np.float32) / max(float(dt), 1e-6)


def _get_speed_profile(sample: dict, traj: np.ndarray, horizon: int, dt: float) -> Optional[np.ndarray]:
    speed = _to_numpy(sample.get("speed_profile_target_mps"))
    if speed is not None:
        speed = np.asarray(speed, dtype=np.float32).reshape(-1)
        if speed.shape[0] >= horizon:
            return speed[:horizon]
    if traj is None:
        return None
    return _derive_speed_profile(traj, dt)[:horizon]


def _print_stats(name: str, mean: np.ndarray, std: np.ndarray, max_rows: int = 12) -> None:
    print(f"\n{name}: shape={mean.shape}")
    rows = min(mean.shape[0], max_rows) if mean.ndim > 0 else 1
    for i in range(rows):
        m = mean[i] if mean.ndim > 0 else mean
        s = std[i] if std.ndim > 0 else std
        print(f"  {i:02d}: mean={np.asarray(m).tolist()} std={np.asarray(s).tolist()}")
    if mean.ndim > 0 and mean.shape[0] > rows:
        print(f"  ... {mean.shape[0] - rows} more")


def compute_stats(args: argparse.Namespace) -> dict:
    dataset_path = Path(args.dataset_path)
    traj_stats = RunningPointStats((args.traj_horizon, 2), "traj")
    route_stats = RunningPointStats((args.route_points, 2), "route")
    speed_stats = RunningPointStats((args.traj_horizon,), "speed_profile")
    accel_stats = RunningPointStats((max(args.traj_horizon - 1, 1),), "acceleration_profile")

    total = valid_traj = valid_route = valid_speed = failed = 0
    for sample in _iter_samples(dataset_path, args.split, args.max_samples):
        total += 1
        try:
            traj = _get_traj(sample, args.traj_horizon)
            if traj is not None:
                traj_stats.update(traj)
                valid_traj += 1

            route, route_mask = _get_route(sample, args.route_points)
            if route is not None:
                route_stats.update(route, route_mask)
                valid_route += 1

            speed = _get_speed_profile(sample, traj, args.traj_horizon, args.dt)
            if speed is not None and speed.shape[0] >= args.traj_horizon:
                speed = np.clip(speed[:args.traj_horizon], 0.0, args.max_speed_mps)
                speed_stats.update(speed)
                valid_speed += 1
                if args.traj_horizon > 1:
                    accel = np.diff(speed) / max(float(args.dt), 1e-6)
                    accel = np.clip(accel, -args.max_abs_accel_mps2, args.max_abs_accel_mps2)
                    accel_stats.update(accel)
        except Exception:
            failed += 1
            if args.strict:
                raise

    print(f"\nProcessed samples: {total}")
    print(f"  valid traj:  {valid_traj}")
    print(f"  valid route: {valid_route}")
    print(f"  valid speed: {valid_speed}")
    print(f"  failed:      {failed}")
    if valid_traj == 0 or valid_route == 0 or valid_speed == 0:
        raise ValueError("Not enough valid samples to compute all requested stats")

    abs_mean, abs_std = traj_stats.finalize(args.min_std)
    route_abs_mean, route_abs_std = route_stats.finalize(args.min_std)
    speed_profile_mean, speed_profile_std = speed_stats.finalize(args.speed_min_std)
    acceleration_profile_mean, acceleration_profile_std = accel_stats.finalize(args.accel_min_std)

    _print_stats("traj abs", abs_mean, abs_std)
    _print_stats("route abs", route_abs_mean, route_abs_std)
    _print_stats("speed profile", speed_profile_mean, speed_profile_std)
    _print_stats("acceleration profile", acceleration_profile_mean, acceleration_profile_std)

    return {
        "abs_mean": abs_mean,
        "abs_std": abs_std,
        "route_abs_mean": route_abs_mean,
        "route_abs_std": route_abs_std,
        "speed_profile_mean": speed_profile_mean,
        "speed_profile_std": speed_profile_std,
        "acceleration_profile_mean": acceleration_profile_mean,
        "acceleration_profile_std": acceleration_profile_std,
        "traj_count": traj_stats.count.astype(np.float32),
        "route_count": route_stats.count.astype(np.float32),
        "speed_profile_count": speed_stats.count.astype(np.float32),
        "acceleration_profile_count": accel_stats.count.astype(np.float32),
        "dt": np.asarray(args.dt, dtype=np.float32),
    }


def save_outputs(stats: dict, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / f"{prefix}_traj_route_speed_norm_stats.npz"
    abs_path = output_dir / f"{prefix}_abs_stats.npz"
    route_path = output_dir / f"{prefix}_route_abs_stats.npz"
    speed_path = output_dir / f"{prefix}_speed_profile_stats.npz"

    np.savez(bundle_path, **stats)
    np.savez(abs_path, abs_mean=stats["abs_mean"], abs_std=stats["abs_std"])
    np.savez(route_path, route_abs_mean=stats["route_abs_mean"], route_abs_std=stats["route_abs_std"])
    np.savez(
        speed_path,
        speed_profile_mean=stats["speed_profile_mean"],
        speed_profile_std=stats["speed_profile_std"],
        acceleration_profile_mean=stats["acceleration_profile_mean"],
        acceleration_profile_std=stats["acceleration_profile_std"],
        dt=stats["dt"],
    )
    print("\nSaved:")
    print(f"  {bundle_path}")
    print(f"  {abs_path}")
    print(f"  {route_path}")
    print(f"  {speed_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--split", choices=("train", "val", "direct", "all"), default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--traj_horizon", type=int, default=8)
    parser.add_argument("--route_points", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--min_std", type=float, default=1e-3)
    parser.add_argument("--speed_min_std", type=float, default=1e-3)
    parser.add_argument("--accel_min_std", type=float, default=1e-3)
    parser.add_argument("--max_speed_mps", type=float, default=40.0)
    parser.add_argument("--max_abs_accel_mps2", type=float, default=20.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if not args.prefix:
        args.prefix = "train" if args.split == "train" else args.split
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.dataset_path)
    stats = compute_stats(args)
    save_outputs(stats, output_dir, args.prefix)


if __name__ == "__main__":
    main()
