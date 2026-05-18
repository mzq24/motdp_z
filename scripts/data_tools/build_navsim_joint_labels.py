#!/usr/bin/env python3
"""Build NAVSIM joint trajectory / route / speed label sidecars.

The output is indexed by token and aligned to the canonical cached-BEV train
cache. It deliberately reuses the NAVSIM stats path helpers so label generation
and normalization statistics stay in lockstep.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_tools.compute_navsim_traj_route_speed_norm_stats import (  # noqa: E402
    NAVSIM_DT,
    prepare_pose_cache,
    relative_future_points,
    sample_path_by_distance,
    speed_from_cache,
    split_contiguous_segments,
    trajectory_from_cache,
)


def iter_log_paths(log_root: Path, max_logs: Optional[int]) -> Iterable[Path]:
    paths = sorted(log_root.glob("*.pkl"))
    if max_logs is not None:
        paths = paths[: int(max_logs)]
    return paths


def load_cache_index(cache_dir: Path, max_tokens: Optional[int]) -> dict[str, np.ndarray]:
    index_path = cache_dir / "cache_index.npz"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing cache_index.npz: {index_path}")
    with np.load(index_path, allow_pickle=False) as index:
        tokens = index["tokens"].astype(str)
        log_names = index["log_names"].astype(str)
        frame_indices = index["frame_indices"].astype(np.int32)

    keep_rows = []
    seen = set()
    for row, token in enumerate(tokens):
        token = str(token)
        if token in seen:
            continue
        seen.add(token)
        keep_rows.append(row)
        if max_tokens is not None and len(keep_rows) >= int(max_tokens):
            break
    keep_rows_np = np.asarray(keep_rows, dtype=np.int64)
    return {
        "tokens": tokens[keep_rows_np],
        "log_names": log_names[keep_rows_np],
        "frame_indices": frame_indices[keep_rows_np],
        "cache_indices": keep_rows_np,
    }


def build_labels(args: argparse.Namespace) -> dict[str, Any]:
    cache_index = load_cache_index(args.cache_dir, args.max_tokens)
    ordered_tokens = [str(t) for t in cache_index["tokens"]]
    wanted_tokens = set(ordered_tokens)
    found: dict[str, dict[str, Any]] = {}

    counts: dict[str, int] = {
        "cache_unique_tokens": len(ordered_tokens),
        "logs": 0,
        "frames": 0,
        "segments": 0,
        "tokens_seen_in_logs": 0,
        "tokens_matched": 0,
        "tokens_skipped_future_short": 0,
        "tokens_skipped_empty_route": 0,
        "tokens_skipped_duplicate": 0,
        "failed": 0,
    }
    failed_examples: list[dict[str, Any]] = []
    t0 = time.time()

    stop = False
    log_paths = list(iter_log_paths(args.log_root, args.max_logs))
    for log_path in tqdm(log_paths, desc="navsim logs"):
        counts["logs"] += 1
        with log_path.open("rb") as f:
            frames = pickle.load(f)
        counts["frames"] += len(frames)
        cache = prepare_pose_cache(frames)
        segments = split_contiguous_segments(frames)
        counts["segments"] += len(segments)

        for seg_start, seg_end in segments:
            for idx in range(seg_start, seg_end + 1):
                frame = frames[idx]
                token = str(frame.get("token", ""))
                if token not in wanted_tokens:
                    continue
                counts["tokens_seen_in_logs"] += 1
                if token in found:
                    counts["tokens_skipped_duplicate"] += 1
                    continue
                try:
                    traj = trajectory_from_cache(cache, idx, seg_end, args.traj_horizon)
                    speed = speed_from_cache(cache, idx, seg_end, args.speed_horizon, args.dt)
                    if traj is None or speed is None:
                        counts["tokens_skipped_future_short"] += 1
                        continue

                    future_points = relative_future_points(cache, idx, seg_end, args.path_lookahead_frames)
                    path, path_mask = sample_path_by_distance(future_points, args.route_step_m, args.route_points)
                    if args.require_route_any and not bool(path_mask.any()):
                        counts["tokens_skipped_empty_route"] += 1
                        continue

                    speed = np.clip(speed, 0.0, args.max_speed_mps).astype(np.float32)
                    traj_xy = np.asarray(traj[:, :2], dtype=np.float32)
                    path = np.asarray(path, dtype=np.float32)
                    path_mask = np.asarray(path_mask, dtype=bool)
                    found[token] = {
                        "trajectory": traj_xy,
                        "route": path,
                        "route_mask": path_mask,
                        "path": path,
                        "path_mask": path_mask,
                        "speed_profile": speed,
                    }
                    counts["tokens_matched"] += 1
                    if args.stop_after_matched is not None and counts["tokens_matched"] >= int(args.stop_after_matched):
                        stop = True
                        break
                except Exception as exc:  # pragma: no cover - diagnostic path
                    counts["failed"] += 1
                    if len(failed_examples) < 20:
                        failed_examples.append({"token": token, "log": log_path.name, "frame_idx": idx, "error": repr(exc)})
                    if args.strict:
                        raise
            if stop:
                break
        if stop:
            break

    label_tokens: list[str] = []
    label_log_names: list[str] = []
    label_frame_indices: list[int] = []
    label_cache_indices: list[int] = []
    trajectories: list[np.ndarray] = []
    routes: list[np.ndarray] = []
    route_masks: list[np.ndarray] = []
    paths: list[np.ndarray] = []
    path_masks: list[np.ndarray] = []
    speeds: list[np.ndarray] = []

    for row, token in enumerate(ordered_tokens):
        data = found.get(token)
        if data is None:
            continue
        label_tokens.append(token)
        label_log_names.append(str(cache_index["log_names"][row]))
        label_frame_indices.append(int(cache_index["frame_indices"][row]))
        label_cache_indices.append(int(cache_index["cache_indices"][row]))
        trajectories.append(data["trajectory"])
        routes.append(data["route"])
        route_masks.append(data["route_mask"])
        paths.append(data["path"])
        path_masks.append(data["path_mask"])
        speeds.append(data["speed_profile"])

    if not label_tokens:
        raise RuntimeError(f"No labels were built. counts={counts}")

    elapsed = time.time() - t0
    metadata = {
        **counts,
        "labels_written": len(label_tokens),
        "missing_after_scan": len(ordered_tokens) - len(label_tokens),
        "elapsed_sec": round(elapsed, 3),
        "cache_dir": str(args.cache_dir),
        "log_root": str(args.log_root),
        "traj_horizon": int(args.traj_horizon),
        "speed_horizon": int(args.speed_horizon),
        "route_points": int(args.route_points),
        "path_lookahead_frames": int(args.path_lookahead_frames),
        "route_step_m": float(args.route_step_m),
        "dt": float(args.dt),
        "failed_examples": failed_examples,
    }

    return {
        "tokens": np.asarray(label_tokens),
        "log_names": np.asarray(label_log_names),
        "frame_indices": np.asarray(label_frame_indices, dtype=np.int32),
        "cache_indices": np.asarray(label_cache_indices, dtype=np.int64),
        "trajectory": np.stack(trajectories).astype(np.float32),
        "route": np.stack(routes).astype(np.float32),
        "route_mask": np.stack(route_masks).astype(bool),
        "path": np.stack(paths).astype(np.float32),
        "path_mask": np.stack(path_masks).astype(bool),
        "speed_profile": np.stack(speeds).astype(np.float32),
        "metadata": metadata,
    }


def save_outputs(labels: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "label_index.npz",
        tokens=labels["tokens"],
        log_names=labels["log_names"],
        frame_indices=labels["frame_indices"],
        cache_indices=labels["cache_indices"],
    )
    np.save(output_dir / "trajectory.npy", labels["trajectory"])
    np.save(output_dir / "route.npy", labels["route"])
    np.save(output_dir / "route_mask.npy", labels["route_mask"])
    np.save(output_dir / "path.npy", labels["path"])
    np.save(output_dir / "path_mask.npy", labels["path_mask"])
    np.save(output_dir / "speed_profile.npy", labels["speed_profile"])
    (output_dir / "metadata.json").write_text(json.dumps(labels["metadata"], indent=2, sort_keys=True), encoding="utf-8")

    print("Saved NAVSIM joint labels:")
    for name in (
        "label_index.npz",
        "trajectory.npy",
        "route.npy",
        "route_mask.npy",
        "path.npy",
        "path_mask.npy",
        "speed_profile.npy",
        "metadata.json",
    ):
        print(f"  {output_dir / name}")
    print(json.dumps(labels["metadata"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=Path("/workspace2/data/navsim/navsim_logs/trainval"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/workspace2/z_project/motdp_bev_cache_train_official4cam_officialacc_npy"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--traj-horizon", type=int, default=8)
    parser.add_argument("--speed-horizon", type=int, default=8)
    parser.add_argument("--route-points", type=int, default=50)
    parser.add_argument("--path-lookahead-frames", type=int, default=80)
    parser.add_argument("--route-step-m", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=NAVSIM_DT)
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None, help="Limit first N unique cache tokens before scanning logs.")
    parser.add_argument("--stop-after-matched", type=int, default=None, help="Stop scan after N labels are matched.")
    parser.add_argument("--max-speed-mps", type=float, default=40.0)
    parser.add_argument("--require-route-any", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = build_labels(args)
    save_outputs(labels, args.output_dir)


if __name__ == "__main__":
    main()
