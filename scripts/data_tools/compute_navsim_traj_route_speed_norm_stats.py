#!/usr/bin/env python3
"""Compute NAVSIM trajectory / SparseDrive-style route / speed normalization stats.

This uses official NAVSIM cache tokens as the sample list, then reads raw NAVSIM
logs to reconstruct labels:

- trajectory: 8 future ego waypoints in the current ego frame
- route/path: future ego path sampled by arc length every 1m, SparseDrive-style
- speed: future displacement / fixed NAVSIM dt

Unlike the older MoT-DP PDMLite stats script, this script is NAVSIM-specific and
never reads PDMLite packed samples.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

NAVSIM_DT = 0.5


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


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_transform(transform: np.ndarray) -> float:
    return math.atan2(float(transform[1, 0]), float(transform[0, 0]))


def frame_transform(frame: Mapping) -> np.ndarray:
    return np.asarray(frame["ego2global"], dtype=np.float64)


def split_contiguous_segments(
    frames: Sequence[Mapping],
    min_dt_s: float = 0.35,
    max_dt_s: float = 0.75,
) -> list[tuple[int, int]]:
    if not frames:
        return []
    segments: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, len(frames)):
        prev = frames[idx - 1]
        cur = frames[idx]
        dt = (int(cur["timestamp"]) - int(prev["timestamp"])) / 1e6
        linked = (
            prev.get("sample_next") == cur.get("token")
            and cur.get("sample_prev") == prev.get("token")
            and min_dt_s <= dt <= max_dt_s
        )
        if not linked:
            segments.append((start, idx - 1))
            start = idx
    segments.append((start, len(frames) - 1))
    return segments


def prepare_pose_cache(frames: Sequence[Mapping]) -> dict[str, np.ndarray | list[np.ndarray]]:
    transforms = [frame_transform(frame) for frame in frames]
    inv_transforms = [np.linalg.inv(transform) for transform in transforms]
    yaws = np.asarray([yaw_from_transform(transform) for transform in transforms], dtype=np.float64)
    positions_xyz = np.asarray(
        [[transform[0, 3], transform[1, 3], transform[2, 3]] for transform in transforms],
        dtype=np.float64,
    )
    return {"transforms": transforms, "inv_transforms": inv_transforms, "yaws": yaws, "positions_xyz": positions_xyz}


def trajectory_from_cache(cache: Mapping, current_idx: int, segment_end: int, horizon: int) -> Optional[np.ndarray]:
    if current_idx + horizon > segment_end:
        return None
    origin_inv = cache["inv_transforms"][current_idx]
    origin_yaw = float(cache["yaws"][current_idx])
    positions = cache["positions_xyz"][current_idx + 1 : current_idx + horizon + 1]
    homogeneous = np.concatenate([positions, np.ones((positions.shape[0], 1), dtype=np.float64)], axis=1)
    local = (origin_inv @ homogeneous.T).T
    yaw_delta = [wrap_angle(float(yaw) - origin_yaw) for yaw in cache["yaws"][current_idx + 1 : current_idx + horizon + 1]]
    return np.column_stack([local[:, 0], local[:, 1], yaw_delta]).astype(np.float32)


def speed_from_cache(cache: Mapping, current_idx: int, segment_end: int, horizon: int, dt: float) -> Optional[np.ndarray]:
    if current_idx + horizon > segment_end:
        return None
    positions = cache["positions_xyz"][current_idx : current_idx + horizon + 1, :2]
    return (np.linalg.norm(np.diff(positions, axis=0), axis=1) / max(float(dt), 1e-6)).astype(np.float32)


def relative_future_points(cache: Mapping, current_idx: int, segment_end: int, lookahead_frames: int) -> np.ndarray:
    origin_inv = cache["inv_transforms"][current_idx]
    end = min(segment_end, current_idx + int(lookahead_frames))
    positions = cache["positions_xyz"][current_idx : end + 1]
    homogeneous = np.concatenate([positions, np.ones((positions.shape[0], 1), dtype=np.float64)], axis=1)
    local = (origin_inv @ homogeneous.T).T
    return local[:, :2].astype(np.float32)


def sample_path_by_distance(points: np.ndarray, step_m: float, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    path = np.zeros((int(num_points), 2), dtype=np.float32)
    mask = np.zeros(int(num_points), dtype=bool)
    if points.shape[0] == 0:
        return path, mask
    if points.shape[0] == 1:
        path[:] = points[0, :2]
        return path, mask

    seg_lens = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
    keep = np.concatenate([[True], np.diff(dist) > 1e-3])
    dist = dist[keep]
    points = points[keep]
    if points.shape[0] < 2:
        path[:] = points[-1, :2]
        return path, mask

    targets = np.arange(step_m, step_m * (int(num_points) + 1), step_m, dtype=np.float32)
    path[:, 0] = np.interp(targets, dist, points[:, 0])
    path[:, 1] = np.interp(targets, dist, points[:, 1])
    valid = targets <= dist[-1]
    mask[:] = valid
    return path, mask


def load_cache_tokens(cache_dir: Path) -> set[str]:
    index_path = cache_dir / "cache_index.npz"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing cache_index.npz: {index_path}")
    with np.load(index_path, allow_pickle=False) as index:
        return set(map(str, index["tokens"].astype(str)))


def iter_log_paths(log_root: Path, max_logs: Optional[int]) -> Iterable[Path]:
    paths = sorted(log_root.glob("*.pkl"))
    if max_logs is not None:
        paths = paths[: int(max_logs)]
    return paths


def compute_stats(args: argparse.Namespace) -> dict[str, np.ndarray | int | float | str]:
    allowed_tokens = load_cache_tokens(args.cache_dir)
    print(f"Allowed official cache tokens: {len(allowed_tokens)}", flush=True)

    traj_stats = RunningPointStats((args.traj_horizon, 2), "traj")
    route_stats = RunningPointStats((args.route_points, 2), "route")
    speed_stats = RunningPointStats((args.traj_horizon,), "speed_profile")
    accel_stats = RunningPointStats((max(args.traj_horizon - 1, 1),), "acceleration_profile")

    counts = {
        "logs": 0,
        "frames": 0,
        "segments": 0,
        "tokens_seen_in_logs": 0,
        "tokens_matched": 0,
        "tokens_skipped_no_route": 0,
        "tokens_skipped_future_short": 0,
        "valid_traj": 0,
        "valid_route_any": 0,
        "valid_route_full": 0,
        "valid_speed": 0,
        "failed": 0,
    }

    t0 = time.time()
    stop = False
    for log_path in tqdm(list(iter_log_paths(args.log_root, args.max_logs)), desc="navsim logs"):
        counts["logs"] += 1
        with log_path.open("rb") as f:
            frames = pickle.load(f)
        counts["frames"] += len(frames)
        cache = prepare_pose_cache(frames)
        segments = split_contiguous_segments(frames)
        counts["segments"] += len(segments)
        for seg_start, seg_end in segments:
            for idx in range(seg_start, seg_end + 1):
                token = str(frames[idx].get("token", ""))
                if token not in allowed_tokens:
                    continue
                counts["tokens_seen_in_logs"] += 1
                if args.require_route and len(frames[idx].get("roadblock_ids", [])) == 0:
                    counts["tokens_skipped_no_route"] += 1
                    continue
                try:
                    traj = trajectory_from_cache(cache, idx, seg_end, args.traj_horizon)
                    speed = speed_from_cache(cache, idx, seg_end, args.traj_horizon, args.dt)
                    if traj is None or speed is None:
                        counts["tokens_skipped_future_short"] += 1
                        continue
                    future_points = relative_future_points(cache, idx, seg_end, args.path_lookahead_frames)
                    path, path_mask = sample_path_by_distance(future_points, args.route_step_m, args.route_points)

                    traj_stats.update(traj[:, :2])
                    route_stats.update(path, path_mask)
                    speed = np.clip(speed, 0.0, args.max_speed_mps)
                    speed_stats.update(speed)
                    if args.traj_horizon > 1:
                        accel = np.diff(speed) / max(float(args.dt), 1e-6)
                        accel = np.clip(accel, -args.max_abs_accel_mps2, args.max_abs_accel_mps2)
                        accel_stats.update(accel)

                    counts["tokens_matched"] += 1
                    counts["valid_traj"] += 1
                    counts["valid_speed"] += 1
                    if bool(path_mask.any()):
                        counts["valid_route_any"] += 1
                    if int(path_mask.sum()) == args.route_points:
                        counts["valid_route_full"] += 1
                except Exception:
                    counts["failed"] += 1
                    if args.strict:
                        raise
                if args.max_tokens is not None and counts["tokens_matched"] >= args.max_tokens:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    if counts["valid_traj"] == 0 or counts["valid_speed"] == 0 or counts["valid_route_any"] == 0:
        raise RuntimeError(f"not enough valid NAVSIM samples: {counts}")

    abs_mean, abs_std = traj_stats.finalize(args.min_std)
    route_abs_mean, route_abs_std = route_stats.finalize(args.min_std)
    speed_profile_mean, speed_profile_std = speed_stats.finalize(args.speed_min_std)
    acceleration_profile_mean, acceleration_profile_std = accel_stats.finalize(args.accel_min_std)

    print(json.dumps(counts, indent=2, sort_keys=True), flush=True)
    print(f"traj abs {abs_mean.shape} route abs {route_abs_mean.shape} speed {speed_profile_mean.shape}", flush=True)
    print(f"elapsed_sec={time.time() - t0:.1f}", flush=True)

    out: dict[str, np.ndarray | int | float | str] = {
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
    for key, value in counts.items():
        out[key] = np.asarray(value, dtype=np.int64)
    out["cache_dir"] = str(args.cache_dir)
    out["log_root"] = str(args.log_root)
    return out


def save_outputs(stats: dict, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / f"{prefix}_navsim_traj_route_speed_norm_stats.npz"
    abs_path = output_dir / f"{prefix}_abs_stats.npz"
    route_path = output_dir / f"{prefix}_route_abs_stats.npz"
    speed_path = output_dir / f"{prefix}_speed_profile_stats.npz"
    metadata_path = output_dir / f"{prefix}_metadata.json"

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
    metadata = {
        key: (int(value) if isinstance(value, np.ndarray) and value.shape == () and value.dtype.kind in "iu" else str(value))
        for key, value in stats.items()
        if key in {"cache_dir", "log_root"} or (isinstance(value, np.ndarray) and value.shape == () and value.dtype.kind in "iu")
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print("Saved:")
    for path in (bundle_path, abs_path, route_path, speed_path, metadata_path):
        print(f"  {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=Path("/workspace2/data/navsim/navsim_logs/trainval"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/workspace2/z_project/motdp_bev_cache_train_official4cam_officialacc_npy"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="navtrain_official_h8_r50")
    parser.add_argument("--traj-horizon", type=int, default=8)
    parser.add_argument("--route-points", type=int, default=50)
    parser.add_argument("--path-lookahead-frames", type=int, default=80)
    parser.add_argument("--route-step-m", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=NAVSIM_DT)
    parser.add_argument("--require-route", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--min-std", type=float, default=1e-3)
    parser.add_argument("--speed-min-std", type=float, default=1e-3)
    parser.add_argument("--accel-min-std", type=float, default=1e-3)
    parser.add_argument("--max-speed-mps", type=float, default=40.0)
    parser.add_argument("--max-abs-accel-mps2", type=float, default=20.0)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = compute_stats(args)
    save_outputs(stats, args.output_dir, args.prefix)


if __name__ == "__main__":
    main()
