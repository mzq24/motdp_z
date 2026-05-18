"""
Recover token metadata for legacy NavSim BEV cache shards.

The first BEV cache run saved only arrays, not token/log metadata.  This script
rebuilds the old log-only-continuous traversal, matches cached samples by
trajectory+ego_status fingerprint, and writes tiny sidecar files:

  bev_cache_shard000_meta.npz

It also writes missing_official_tokens.txt, which can be used to precompute only
the official navtrain tokens absent from the recovered legacy cache.
"""

from __future__ import annotations

import argparse
import bisect
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from pyquaternion import Quaternion


NAVTRAIN_YAML = (
    "/home/z/code/navsim/navsim/planning/script/config/common/"
    "train_test_split/scene_filter/navtrain.yaml"
)
LOG_ROOT = Path("/workspace2/data/navsim/navsim_logs/trainval")
SENSOR_ROOT = Path("/workspace2/data/navsim/sensor_blobs/trainval")
CACHE_ROOT = Path("/workspace2/z_project/motdp_bev_cache")
NUM_HISTORY = 4
NUM_FUTURE = 10
WINDOW_SIZE = NUM_HISTORY + NUM_FUTURE
TRAJ_HORIZON = 8
CAM_NAMES = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0")


@dataclass
class Record:
    token: str
    log_name: str
    frame_idx: int
    official: bool


def is_pair_continuous(prev, cur) -> bool:
    if prev["sample_next"] != cur["token"]:
        return False
    if cur["sample_prev"] != prev["token"]:
        return False
    dt = (cur["timestamp"] - prev["timestamp"]) / 1e6
    return 0.35 <= dt <= 0.75


def has_sensor(frame) -> bool:
    cams = frame.get("cams", {})
    for cam_name in CAM_NAMES:
        cam = cams.get(cam_name)
        if cam is None:
            return False
        if not (SENSOR_ROOT / cam["data_path"]).exists():
            return False
    return True


def build_ego_status(raw_frames, current_idx):
    feats = []
    for i in range(current_idx - 3, current_idx + 1):
        f = raw_frames[i]
        vel = np.array(f["ego_dynamic_state"][:2], dtype=np.float32)
        acc = np.zeros(2, dtype=np.float32)
        if i > current_idx - 3:
            pv = np.array(raw_frames[i - 1]["ego_dynamic_state"][:2], dtype=np.float32)
            dt = (f["timestamp"] - raw_frames[i - 1]["timestamp"]) / 1e6
            if dt > 0:
                acc = (vel - pv) / dt
        cmd = np.array(f["driving_command"], dtype=np.float32)
        feats.append(np.concatenate([vel, acc, cmd]))
    return np.stack(feats, axis=0).astype(np.float32)


def build_trajectory(raw_frames, current_idx, horizon=TRAJ_HORIZON):
    cur_f = raw_frames[current_idx]
    ego_t = np.array(cur_f["ego2global_translation"][:2], dtype=np.float64)
    q = Quaternion(*cur_f["ego2global_rotation"])
    ego_yaw = q.yaw_pitch_roll[0]
    cos_y, sin_y = np.cos(ego_yaw), np.sin(ego_yaw)
    waypoints = []
    for fi in range(current_idx + 1, min(current_idx + horizon + 1, len(raw_frames))):
        f = raw_frames[fi]
        t = np.array(f["ego2global_translation"][:2], dtype=np.float64)
        dx, dy = t[0] - ego_t[0], t[1] - ego_t[1]
        waypoints.append([dx * cos_y + dy * sin_y, -dx * sin_y + dy * cos_y])
    while len(waypoints) < horizon:
        waypoints.append(waypoints[-1] if waypoints else [0.0, 0.0])
    return np.array(waypoints, dtype=np.float32)


def fingerprint(traj: np.ndarray, ego: np.ndarray) -> bytes:
    return traj.astype(np.float32, copy=False).tobytes() + ego.astype(np.float32, copy=False).tobytes()


def build_candidates(navtrain_yaml: str):
    nav = yaml.safe_load(open(navtrain_yaml))
    navtrain_logs = set(nav["log_names"])
    navtrain_tokens = set(nav["tokens"])
    pkls = [p for p in sorted(LOG_ROOT.glob("*.pkl")) if p.stem in navtrain_logs]

    records: List[Record] = []
    fp_to_indices: Dict[bytes, List[int]] = defaultdict(list)
    for pkl_path in pkls:
        with open(pkl_path, "rb") as f:
            frames = pickle.load(f)
        if len(frames) < WINDOW_SIZE:
            continue
        for start in range(0, len(frames) - WINDOW_SIZE + 1):
            window = frames[start:start + WINDOW_SIZE]
            cur_idx = start + NUM_HISTORY - 1
            cur = window[NUM_HISTORY - 1]
            if not has_sensor(cur):
                continue
            if not all(is_pair_continuous(window[i], window[i + 1]) for i in range(WINDOW_SIZE - 1)):
                continue
            traj = build_trajectory(frames, cur_idx)
            ego = build_ego_status(frames, cur_idx)
            idx = len(records)
            records.append(
                Record(
                    token=cur["token"],
                    log_name=pkl_path.stem,
                    frame_idx=cur_idx,
                    official=cur["token"] in navtrain_tokens,
                )
            )
            fp_to_indices[fingerprint(traj, ego)].append(idx)
    return records, fp_to_indices, navtrain_tokens


def choose_index(indices: List[int], lower_bound: int) -> Optional[int]:
    pos = bisect.bisect_left(indices, lower_bound)
    if pos < len(indices):
        return indices[pos]
    return None


def recover_shard(cache_path: Path, records: List[Record], fp_to_indices: Dict[bytes, List[int]]):
    data = np.load(cache_path)
    traj = data["trajectory"].astype(np.float32)
    ego = data["ego_status"].astype(np.float32)

    mapped_records: List[Record] = []
    candidate_indices: List[int] = []
    lower_bound = 0
    resets = 0
    for i in range(len(traj)):
        fp = fingerprint(traj[i], ego[i])
        candidates = fp_to_indices.get(fp, [])
        idx = choose_index(candidates, lower_bound)
        if idx is None:
            idx = choose_index(candidates, 0)
            if idx is not None:
                resets += 1
        if idx is None:
            raise RuntimeError(f"Could not match {cache_path.name} sample {i}")
        mapped_records.append(records[idx])
        candidate_indices.append(idx)
        lower_bound = idx + 1

    return mapped_records, np.asarray(candidate_indices, dtype=np.int64), resets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", default=str(CACHE_ROOT))
    parser.add_argument("--navtrain_yaml", default=NAVTRAIN_YAML)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    records, fp_to_indices, navtrain_tokens = build_candidates(args.navtrain_yaml)
    print(
        f"candidate records={len(records)}, "
        f"official_in_candidates={sum(r.official for r in records)}",
        flush=True,
    )

    recovered_official = []
    for cache_path in sorted(cache_root.glob("bev_cache_shard[0-9][0-9][0-9].npz")):
        mapped_records, candidate_indices, resets = recover_shard(cache_path, records, fp_to_indices)
        tokens = np.asarray([r.token for r in mapped_records])
        log_names = np.asarray([r.log_name for r in mapped_records])
        frame_indices = np.asarray([r.frame_idx for r in mapped_records], dtype=np.int32)
        official_mask = np.asarray([r.official for r in mapped_records], dtype=bool)
        recovered_official.extend(tokens[official_mask].tolist())

        print(
            f"{cache_path.name}: matched={len(tokens)} official={int(official_mask.sum())} resets={resets}",
            flush=True,
        )

        if args.write:
            sidecar = cache_path.with_name(cache_path.stem + "_meta.npz")
            np.savez_compressed(
                sidecar,
                tokens=tokens,
                log_names=log_names,
                frame_indices=frame_indices,
                official_mask=official_mask,
                candidate_indices=candidate_indices,
            )
            print(f"  wrote {sidecar}", flush=True)

    recovered_unique = set(recovered_official)
    missing = sorted(navtrain_tokens - recovered_unique)
    print(
        f"official_total={len(navtrain_tokens)} recovered_unique={len(recovered_unique)} "
        f"missing={len(missing)} duplicate_official={len(recovered_official) - len(recovered_unique)}",
        flush=True,
    )

    if args.write:
        missing_path = cache_root / "missing_official_tokens.txt"
        missing_path.write_text("\n".join(missing) + "\n")
        print(f"wrote {missing_path}", flush=True)


if __name__ == "__main__":
    main()
