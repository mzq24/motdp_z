#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set


def _atomic_json_dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp_path, path)


def _base_dir_from_sample(sample: Dict) -> str:
    feat = str(sample.get("transfuser_bev_feature", "") or "")
    return os.path.dirname(os.path.dirname(feat)) if feat else ""


def _event_name_from_scene(scene: str) -> str:
    return scene.split("/", 1)[0] if "/" in scene else scene


def _extract_frame_id(filename: str) -> Optional[int]:
    stem = filename.split(".", 1)[0]
    if not stem.isdigit():
        return None
    return int(stem)


def _list_numeric_frame_ids(dir_path: str, *, suffix_filter: Optional[Sequence[str]] = None) -> List[int]:
    if not os.path.isdir(dir_path):
        return []
    suffixes = tuple(suffix_filter or ())
    frame_ids: List[int] = []
    for name in os.listdir(dir_path):
        if suffixes and not name.endswith(suffixes):
            continue
        frame_id = _extract_frame_id(name)
        if frame_id is None:
            continue
        frame_ids.append(frame_id)
    frame_ids.sort()
    return frame_ids


def _discover_route_dirs(image_root: str) -> List[str]:
    route_dirs: List[str] = []
    for event_name in sorted(os.listdir(image_root)):
        event_dir = os.path.join(image_root, event_name)
        if not os.path.isdir(event_dir):
            continue
        for route_name in sorted(os.listdir(event_dir)):
            route_dir = os.path.join(event_dir, route_name)
            if not os.path.isdir(route_dir):
                continue
            if os.path.isdir(os.path.join(route_dir, "measurements")) or os.path.isdir(os.path.join(route_dir, "rgb")):
                route_dirs.append(f"{event_name}/{route_name}")
    return route_dirs


def _coverage_detail(raw_ids: Sequence[int], packed_ids: Sequence[int]) -> Dict:
    raw_set = set(int(fid) for fid in raw_ids)
    packed_set = set(int(fid) for fid in packed_ids)
    if not raw_ids:
        return {
            "raw_frame_min": None,
            "raw_frame_max": None,
            "packed_frame_min": packed_ids[0] if packed_ids else None,
            "packed_frame_max": packed_ids[-1] if packed_ids else None,
            "missing_head_count": 0,
            "missing_tail_count": 0,
            "missing_internal_count": 0,
            "missing_head_frames_preview": [],
            "missing_tail_frames_preview": [],
            "missing_internal_frames_preview": [],
        }

    raw_min = int(raw_ids[0])
    raw_max = int(raw_ids[-1])
    packed_min = int(packed_ids[0]) if packed_ids else None
    packed_max = int(packed_ids[-1]) if packed_ids else None

    if packed_min is None or packed_max is None:
        missing_head = list(raw_ids)
        missing_tail: List[int] = []
        missing_internal: List[int] = []
    else:
        missing_head = [fid for fid in raw_ids if fid < packed_min]
        missing_tail = [fid for fid in raw_ids if fid > packed_max]
        missing_internal = [fid for fid in raw_ids if packed_min <= fid <= packed_max and fid not in packed_set]

    return {
        "raw_frame_min": raw_min,
        "raw_frame_max": raw_max,
        "packed_frame_min": packed_min,
        "packed_frame_max": packed_max,
        "missing_head_count": len(missing_head),
        "missing_tail_count": len(missing_tail),
        "missing_internal_count": len(missing_internal),
        "missing_head_frames_preview": missing_head[:20],
        "missing_tail_frames_preview": missing_tail[:20],
        "missing_internal_frames_preview": missing_internal[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a raw-vs-packed route frame coverage index for later labeling/debug checks."
    )
    parser.add_argument("--image_root", required=True, help="Raw dataset root, e.g. /workspace1/.../pdm_lite")
    parser.add_argument("--packed_path", required=True, help="samples_packed.pkl or merged packed path")
    parser.add_argument("--output_json", required=True, help="Where to write the coverage index JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.packed_path, "rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"Expected list in {args.packed_path}, got {type(samples).__name__}")

    packed_frame_ids_by_scene: Dict[str, List[int]] = defaultdict(list)
    for sample in samples:
        scene = _base_dir_from_sample(sample)
        if not scene:
            continue
        packed_frame_ids_by_scene[scene].append(int(sample.get("frame_id", -1)))
    for scene, frame_ids in packed_frame_ids_by_scene.items():
        frame_ids.sort()

    route_dirs = _discover_route_dirs(args.image_root)
    event_summary: Dict[str, Counter] = defaultdict(Counter)
    route_entries: List[Dict] = []

    for scene in route_dirs:
        route_dir = os.path.join(args.image_root, scene)
        event_name = _event_name_from_scene(scene)
        measurement_ids = _list_numeric_frame_ids(os.path.join(route_dir, "measurements"), suffix_filter=(".json.gz",))
        rgb_ids = _list_numeric_frame_ids(os.path.join(route_dir, "rgb"), suffix_filter=(".jpg", ".png", ".jpeg"))
        lidar_ids = _list_numeric_frame_ids(os.path.join(route_dir, "lidar"))
        packed_ids = packed_frame_ids_by_scene.get(scene, [])

        coverage = _coverage_detail(measurement_ids, packed_ids)
        entry = {
            "scene": scene,
            "event_name": event_name,
            "raw_measurement_count": len(measurement_ids),
            "raw_rgb_count": len(rgb_ids),
            "raw_lidar_count": len(lidar_ids),
            "packed_sample_count": len(packed_ids),
            **coverage,
        }
        route_entries.append(entry)

        event_summary[event_name]["route_count"] += 1
        if len(packed_ids) == 0:
            event_summary[event_name]["missing_from_packed_count"] += 1
        if coverage["missing_head_count"] > 0:
            event_summary[event_name]["missing_head_route_count"] += 1
        if coverage["missing_tail_count"] > 0:
            event_summary[event_name]["missing_tail_route_count"] += 1
        if coverage["missing_internal_count"] > 0:
            event_summary[event_name]["missing_internal_route_count"] += 1

    summary = {
        "image_root": os.path.realpath(args.image_root),
        "packed_path": os.path.realpath(args.packed_path),
        "num_routes": len(route_entries),
        "num_events": len(event_summary),
        "events": {
            event_name: dict(sorted(counter.items()))
            for event_name, counter in sorted(event_summary.items())
        },
        "routes": route_entries,
    }
    _atomic_json_dump(summary, args.output_json)
    print(
        json.dumps(
            {
                "output_json": os.path.realpath(args.output_json),
                "num_routes": len(route_entries),
                "num_events": len(event_summary),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
