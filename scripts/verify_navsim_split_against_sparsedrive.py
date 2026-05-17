"""
Verify our NavSim precompute split against the official/SparseDrive navtrain flow.

SparseDrive delegates split construction to NAVSIM's SceneLoader:
  1. load train_test_split=navtrain
  2. iterate sliding windows from navsim_logs/trainval
  3. keep windows with route and token in scene_filter.tokens
  4. cache by log_name/token

This script mirrors that filtering without loading images into tensors.  It also
checks which official tokens have the sensor files needed by different backbones.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import yaml


DEFAULT_NAVTRAIN_YAML = (
    "/home/z/code/navsim/navsim/planning/script/config/common/"
    "train_test_split/scene_filter/navtrain.yaml"
)
DEFAULT_LOG_ROOT = "/workspace2/data/navsim/navsim_logs/trainval"
DEFAULT_SENSOR_ROOT = "/workspace2/data/navsim/sensor_blobs/trainval"
DEFAULT_CACHE_ROOT = "/workspace2/z_project/motdp_bev_cache"

CAMERA_SETS = {
    "sparsedrive_3cam": ("CAM_L0", "CAM_F0", "CAM_R0"),
    "lead_4cam": ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0"),
    "all_8cam": ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0"),
}


def split_list(input_list: List[dict], num_frames: int, frame_interval: int) -> Iterable[List[dict]]:
    for idx in range(0, len(input_list), frame_interval):
        yield input_list[idx : idx + num_frames]


def is_continuous(frame_list: List[dict]) -> bool:
    for prev, cur in zip(frame_list[:-1], frame_list[1:]):
        if prev["sample_next"] != cur["token"]:
            return False
        if cur["sample_prev"] != prev["token"]:
            return False
        dt = (cur["timestamp"] - prev["timestamp"]) / 1e6
        if not 0.35 <= dt <= 0.75:
            return False
    return True


def has_cameras(frame: dict, sensor_root: str, cam_names: Iterable[str]) -> bool:
    cams = frame.get("cams", {})
    for cam_name in cam_names:
        cam = cams.get(cam_name)
        if cam is None:
            return False
        if not os.path.exists(os.path.join(sensor_root, cam["data_path"])):
            return False
    return True


def load_official_records(args: argparse.Namespace):
    scene_filter = yaml.safe_load(open(args.navtrain_yaml))
    official_logs = set(scene_filter["log_names"])
    official_tokens = set(scene_filter["tokens"])
    num_history_frames = int(scene_filter["num_history_frames"])
    num_frames = num_history_frames + int(scene_filter["num_future_frames"])
    frame_interval = int(scene_filter["frame_interval"])
    has_route = bool(scene_filter["has_route"])

    records: Dict[str, dict] = {}
    duplicates = Counter()
    logs_seen = 0

    for log_path in sorted(Path(args.log_root).glob("*.pkl")):
        log_name = log_path.stem
        if log_name not in official_logs:
            continue
        logs_seen += 1
        with open(log_path, "rb") as f:
            frames = pickle.load(f)
        for start, frame_list in enumerate(split_list(frames, num_frames, frame_interval)):
            if len(frame_list) < num_frames:
                continue
            current = frame_list[num_history_frames - 1]
            if has_route and len(current["roadblock_ids"]) == 0:
                continue
            token = current["token"]
            if token not in official_tokens:
                continue
            if token in records:
                duplicates[token] += 1
                continue
            records[token] = {
                "token": token,
                "log_name": log_name,
                "frame_idx": start + num_history_frames - 1,
                "window_start": start,
                "continuous": is_continuous(frame_list),
                "camera_ok": {
                    name: has_cameras(current, args.sensor_root, cams)
                    for name, cams in CAMERA_SETS.items()
                },
            }

    return scene_filter, official_tokens, logs_seen, records, duplicates


def summarize_cache(cache_root: str):
    paths = sorted(glob.glob(str(Path(cache_root) / "bev_cache_shard*.npz")))
    if not paths:
        return {"paths": [], "count": 0, "has_tokens": False}

    count = 0
    has_tokens = True
    cache_tokens = set()
    for path in paths:
        data = np.load(path)
        count += int(data["trajectory"].shape[0])
        if "tokens" in data.files:
            cache_tokens.update(map(str, data["tokens"].tolist()))
        else:
            has_tokens = False
    return {
        "paths": paths,
        "count": count,
        "has_tokens": has_tokens,
        "tokens": cache_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--navtrain_yaml", default=DEFAULT_NAVTRAIN_YAML)
    parser.add_argument("--log_root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sensor_root", default=DEFAULT_SENSOR_ROOT)
    parser.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args()

    scene_filter, official_tokens, logs_seen, records, duplicates = load_official_records(args)
    missing_from_logs = official_tokens - set(records)
    by_log = defaultdict(int)
    for rec in records.values():
        by_log[rec["log_name"]] += 1

    print("== Official/SparseDrive navtrain filter ==")
    print(f"yaml logs: {len(scene_filter['log_names'])}")
    print(f"logs present on disk: {logs_seen}")
    print(f"yaml tokens: {len(official_tokens)}")
    print(f"tokens reconstructed from logs: {len(records)}")
    print(f"tokens missing from logs/windows: {len(missing_from_logs)}")
    print(f"duplicate token windows: {sum(duplicates.values())}")
    print(f"logs with at least one token: {len(by_log)}")

    continuous_count = sum(1 for rec in records.values() if rec["continuous"])
    print("\n== Window continuity ==")
    print(f"continuous official windows: {continuous_count}")
    print(f"non-continuous official windows: {len(records) - continuous_count}")

    print("\n== Sensor availability on official tokens ==")
    for name in CAMERA_SETS:
        ok = sum(1 for rec in records.values() if rec["camera_ok"][name])
        print(f"{name}: {ok} / {len(records)}")

    cache = summarize_cache(args.cache_root)
    print("\n== Existing BEV cache ==")
    print(f"cache shards: {len(cache['paths'])}")
    print(f"cache samples: {cache['count']}")
    print(f"cache has token metadata: {cache['has_tokens']}")
    if cache.get("has_tokens"):
        cache_tokens = cache["tokens"]
        print(f"cache unique tokens: {len(cache_tokens)}")
        print(f"cache ∩ official: {len(cache_tokens & official_tokens)}")
        print(f"cache - official: {len(cache_tokens - official_tokens)}")
        print(f"official - cache: {len(official_tokens - cache_tokens)}")

    if missing_from_logs:
        print("\nfirst missing tokens:")
        for token in sorted(missing_from_logs)[:10]:
            print(token)


if __name__ == "__main__":
    main()
