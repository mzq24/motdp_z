#!/usr/bin/env python3
"""Precompute exact next-step speed targets into packed samples.

This script writes `next_speed_target_mps` into `samples_packed.pkl` so training
can read the exact t+0.5s speed target without opening raw measurements online
inside the dataset.

It supports either:
  1. a split directory that directly contains `samples_packed.pkl`
  2. a dataset root that contains `train/` and `val/` split directories
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import time
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm


def _atomic_pickle_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _atomic_json_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)
    os.replace(tmp_path, path)


def _resolve_split_dirs(dataset_path: str) -> List[str]:
    dataset_path = os.path.abspath(dataset_path)
    direct_packed = os.path.join(dataset_path, "samples_packed.pkl")
    if os.path.exists(direct_packed):
        return [dataset_path]

    split_dirs = []
    for split in ("train", "val"):
        split_dir = os.path.join(dataset_path, split)
        split_packed = os.path.join(split_dir, "samples_packed.pkl")
        if os.path.exists(split_packed):
            split_dirs.append(split_dir)

    if split_dirs:
        return split_dirs

    raise FileNotFoundError(
        f"Could not find samples_packed.pkl under {dataset_path} or its train/val subdirectories."
    )


def _build_route_name_to_event(image_data_root: str) -> Dict[str, str]:
    route_name_to_event: Dict[str, str] = {}
    try:
        for event_entry in os.scandir(image_data_root):
            if not event_entry.is_dir():
                continue
            if event_entry.name.startswith("tmp_data"):
                continue
            try:
                for route_entry in os.scandir(event_entry.path):
                    if route_entry.is_dir():
                        route_name_to_event[route_entry.name] = event_entry.name
            except OSError:
                continue
    except OSError:
        return {}
    return route_name_to_event


def _get_route_rel_from_sample(sample: Dict[str, Any], route_name_to_event: Dict[str, str]) -> Optional[str]:
    feat_rel = sample.get("transfuser_bev_feature", "")
    if isinstance(feat_rel, str) and feat_rel:
        return os.path.dirname(os.path.dirname(feat_rel))

    route_name = sample.get("route_name")
    if route_name:
        event_name = route_name_to_event.get(str(route_name))
        if event_name:
            return os.path.join(event_name, str(route_name))

    town_name = sample.get("town_name")
    if route_name and town_name:
        return os.path.join(str(town_name), str(route_name))
    return None


def _load_route_speed_map(measurements_dir: str) -> Dict[int, float]:
    speed_map: Dict[int, float] = {}
    if not os.path.isdir(measurements_dir):
        return speed_map

    try:
        entries = sorted(os.listdir(measurements_dir))
    except OSError:
        return speed_map

    for fname in entries:
        if not fname.endswith(".json.gz"):
            continue
        try:
            frame_id = int(fname.split(".")[0])
        except ValueError:
            continue
        path = os.path.join(measurements_dir, fname)
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                measurement = json.load(f)
            speed_map[frame_id] = float(measurement.get("speed", 0.0))
        except Exception:
            continue

    return speed_map


def _save_progress(progress_path: str, payload: Dict[str, Any]) -> None:
    _atomic_json_dump(payload, progress_path)


def _process_split(
    split_dir: str,
    image_data_root: str,
    frame_offset: int,
    force: bool,
    checkpoint_every_minutes: float,
    route_name_to_event: Dict[str, str],
) -> None:
    packed_path = os.path.join(split_dir, "samples_packed.pkl")
    progress_path = packed_path + ".next_speed.progress.json"
    if not os.path.exists(packed_path):
        raise FileNotFoundError(f"Packed samples not found: {packed_path}")

    with open(packed_path, "rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"Expected list in {packed_path}, got {type(samples).__name__}")

    total = len(samples)
    route_speed_cache: Dict[str, Dict[int, float]] = {}
    route_speed_cache_maxsize = 128
    updated = 0
    skipped_existing = 0
    missing_route = 0
    missing_frame = 0
    seen_routes = set()
    dirty = False

    checkpoint_seconds = max(float(checkpoint_every_minutes), 0.0) * 60.0
    last_checkpoint_time = time.time()

    def get_route_speed_map(route_rel: str) -> Dict[int, float]:
        cached = route_speed_cache.get(route_rel)
        if cached is not None:
            return cached
        if len(route_speed_cache) >= route_speed_cache_maxsize:
            route_speed_cache.pop(next(iter(route_speed_cache)))
        measurements_dir = os.path.join(image_data_root, route_rel, "measurements")
        speed_map = _load_route_speed_map(measurements_dir)
        route_speed_cache[route_rel] = speed_map
        seen_routes.add(route_rel)
        return speed_map

    def maybe_checkpoint(reason: str, force_save: bool = False) -> None:
        nonlocal dirty, last_checkpoint_time
        now = time.time()
        should_save = force_save or (
            dirty and checkpoint_seconds > 0.0 and (now - last_checkpoint_time) >= checkpoint_seconds
        )
        payload = {
            "split_dir": os.path.abspath(split_dir),
            "packed_path": os.path.abspath(packed_path),
            "frame_offset": int(frame_offset),
            "total_samples": total,
            "updated_samples": updated,
            "skipped_existing": skipped_existing,
            "missing_route": missing_route,
            "missing_target_frame": missing_frame,
            "loaded_route_caches": len(seen_routes),
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if should_save:
            _atomic_pickle_dump(samples, packed_path)
            dirty = False
            last_checkpoint_time = now
            payload["saved_packed"] = True
        else:
            payload["saved_packed"] = False
        _save_progress(progress_path, payload)

    pbar = tqdm(samples, desc=f"next_speed {os.path.basename(split_dir)}", leave=True)
    for sample in pbar:
        if not force and sample.get("next_speed_target_mps") is not None:
            skipped_existing += 1
            continue

        frame_id = sample.get("frame_id")
        route_rel = _get_route_rel_from_sample(sample, route_name_to_event)
        if frame_id is None or route_rel is None:
            missing_route += 1
            continue

        speed_map = get_route_speed_map(route_rel)
        target_speed = speed_map.get(int(frame_id) + frame_offset)
        if target_speed is None:
            missing_frame += 1
            continue

        sample["next_speed_target_mps"] = float(target_speed)
        updated += 1
        dirty = True

        pbar.set_postfix({
            "updated": updated,
            "missing": missing_frame,
            "routes": len(seen_routes),
        })
        maybe_checkpoint(reason="periodic")

    maybe_checkpoint(reason="final", force_save=True)

    print("========================================")
    print(f"Finished next speed preprocess for: {split_dir}")
    print(f"Packed path:        {packed_path}")
    print(f"Total samples:      {total}")
    print(f"Updated samples:    {updated}")
    print(f"Skipped existing:   {skipped_existing}")
    print(f"Missing route/frame:{missing_route}")
    print(f"Missing target:     {missing_frame}")
    print(f"Routes scanned:     {len(seen_routes)}")
    print(f"Progress file:      {progress_path}")
    print("========================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Split dir with samples_packed.pkl, or dataset root with train/val subdirs.",
    )
    parser.add_argument(
        "--image_data_root",
        required=True,
        help="Raw dataset root that contains <event>/<route>/measurements/*.json.gz",
    )
    parser.add_argument(
        "--frame_offset",
        type=int,
        default=2,
        help="Future frame offset for next-speed target. Default 2 corresponds to +0.5s.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if next_speed_target_mps already exists.",
    )
    parser.add_argument(
        "--checkpoint_every_minutes",
        type=float,
        default=20.0,
        help="Periodically save packed file while processing. Set <=0 to disable periodic saves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dirs = _resolve_split_dirs(args.dataset_path)
    route_name_to_event = _build_route_name_to_event(args.image_data_root)
    print(f"Resolved split dirs: {split_dirs}")
    print(f"Scanned {len(route_name_to_event)} route->event mappings.")
    for split_dir in split_dirs:
        _process_split(
            split_dir=split_dir,
            image_data_root=args.image_data_root,
            frame_offset=args.frame_offset,
            force=args.force,
            checkpoint_every_minutes=args.checkpoint_every_minutes,
            route_name_to_event=route_name_to_event,
        )


if __name__ == "__main__":
    main()
