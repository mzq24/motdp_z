#!/usr/bin/env python3
"""Analyze PDM-Lite distributions in model-visible and audit spaces.

This script is intentionally read-only: it never writes back to
``samples_packed.pkl``.  It produces aggregate tables, optional target-point
sequence sidecars, and optional baseline clusters for representation-conditioned
distribution analysis.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import pickle
import random
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


FAMILY_CODE_TO_NAME = {
    0: "none",
    1: "borrow",
    2: "merge",
    3: "junction",
}

FACTOR_ORDER = ["none", "borrow", "merge", "junction", "unknown"]
SPEED_BINS = [0.0, 1.0, 3.0, 5.0, 8.0, 12.0, 20.0, float("inf")]
DIST_BINS = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0, float("inf")]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_json_gz(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except Exception:
        return None


def _to_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[-1]
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[-1]
    try:
        return int(value)
    except Exception:
        return default


def _last_vec(sample: Dict[str, Any], key: str, width: Optional[int] = None) -> Optional[np.ndarray]:
    arr = _to_numpy(sample.get(key))
    if arr is None or arr.size == 0:
        return None
    if arr.ndim == 0:
        vec = arr.reshape(1)
    elif arr.ndim == 1:
        vec = arr
    else:
        vec = arr[-1]
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if width is not None:
        if vec.shape[0] < width:
            out = np.zeros(width, dtype=np.float32)
            out[: vec.shape[0]] = vec
            return out
        return vec[:width]
    return vec


def _last_scalar(sample: Dict[str, Any], key: str) -> float:
    arr = _to_numpy(sample.get(key))
    if arr is None or arr.size == 0:
        return float("nan")
    return _to_float(arr)


def _target_points(sample: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    tp = _last_vec(sample, "target_point_hist")
    if tp is None:
        tp = np.zeros(2, dtype=np.float32)
    if tp.shape[0] >= 4:
        tp_current = tp[:2]
        tp_next = tp[2:4]
    else:
        tp_current = tp[:2]
        tp_next_arr = _last_vec(sample, "target_point_next_hist", width=2)
        tp_next = tp_next_arr if tp_next_arr is not None else tp_current.copy()
    return tp_current.astype(np.float32), tp_next.astype(np.float32)


def _command_id(sample: Dict[str, Any]) -> int:
    vec = _last_vec(sample, "command_hist")
    if vec is None or vec.size == 0:
        return -1
    if vec.size == 1:
        return _to_int(vec[0], default=-1)
    return int(np.argmax(vec))


def _command_onehot(command_id: int, width: int = 6) -> np.ndarray:
    out = np.zeros(width, dtype=np.float32)
    if 0 <= command_id < width:
        out[command_id] = 1.0
    return out


def _route_rel_from_sample(sample: Dict[str, Any]) -> str:
    feat_rel = str(sample.get("transfuser_bev_feature", "") or "")
    if feat_rel:
        # e.g. Accident/Town12.../transfuser_feature/0006_feature.pt
        route_rel = os.path.dirname(os.path.dirname(feat_rel))
        if route_rel and route_rel != ".":
            return route_rel
    route_name = str(sample.get("route_name", "") or "")
    event_name = str(sample.get("event_name", sample.get("scenario_name", "")) or "")
    return os.path.join(event_name, route_name) if event_name else route_name


def _route_label_from_rel(route_rel: str) -> str:
    parts = [p for p in route_rel.replace("\\", "/").split("/") if p]
    return parts[0] if parts else "unknown"


def _packed_path_for_feature(image_root: str, sample: Dict[str, Any]) -> Optional[str]:
    feat_rel = str(sample.get("transfuser_bev_feature", "") or "")
    if not feat_rel:
        return None
    bev_feature_path = os.path.join(image_root, feat_rel)
    return os.path.join(os.path.dirname(bev_feature_path), "route_features.pt")


def _sample_key(sample: Dict[str, Any]) -> str:
    return f"{_route_rel_from_sample(sample)}|{_to_int(sample.get('frame_id'), -1):04d}"


def _active_flag(sample: Dict[str, Any], key: str) -> bool:
    return _to_float(sample.get(key), 0.0) > 0.5


def _factor_label(sample: Dict[str, Any]) -> str:
    family_code = _to_int(sample.get("conflict_area_family"), 0)
    if family_code in FAMILY_CODE_TO_NAME and family_code != 0:
        return FAMILY_CODE_TO_NAME[family_code]
    if _active_flag(sample, "borrow_cross_episode_active"):
        return "borrow"
    if _active_flag(sample, "junction_cross_episode_active"):
        return "junction"
    if _active_flag(sample, "merge_episode_active") or _active_flag(sample, "merge_active"):
        return "merge"
    if family_code == 0:
        return "none"
    return "unknown"


def _bin_value(value: float, bins: Sequence[float], prefix: str) -> str:
    if not math.isfinite(value):
        return f"{prefix}:nan"
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo <= value < hi:
            return f"{prefix}:[{lo:g},{hi:g})"
    return f"{prefix}:overflow"


def _target_direction(tp: np.ndarray) -> str:
    x = float(tp[0]) if tp.shape[0] > 0 else 0.0
    y = float(tp[1]) if tp.shape[0] > 1 else 0.0
    if x < -1.0:
        return "behind"
    angle_deg = math.degrees(math.atan2(y, max(x, 1e-6)))
    if angle_deg > 15.0:
        return "left"
    if angle_deg < -15.0:
        return "right"
    return "straight"


def _model_visible_vector(sample: Dict[str, Any]) -> np.ndarray:
    speed = _last_scalar(sample, "speed_hist")
    theta = _last_scalar(sample, "theta_hist")
    command_id = _command_id(sample)
    tp, tp_next = _target_points(sample)
    wp = _last_vec(sample, "waypoints_hist", width=2)
    if wp is None:
        wp = np.zeros(2, dtype=np.float32)
    tp_dist = float(np.linalg.norm(tp))
    tp_next_dist = float(np.linalg.norm(tp_next))
    tp_angle = math.atan2(float(tp[1]), max(float(tp[0]), 1e-6))
    tp_next_angle = math.atan2(float(tp_next[1]), max(float(tp_next[0]), 1e-6))
    values = [
        speed,
        theta,
        *(_command_onehot(command_id).tolist()),
        float(tp[0]),
        float(tp[1]),
        float(tp_next[0]),
        float(tp_next[1]),
        float(wp[0]),
        float(wp[1]),
        tp_dist,
        tp_angle,
        tp_next_dist,
        tp_next_angle,
    ]
    return np.asarray(values, dtype=np.float32)


def _sample_summary_row(split: str, sample_idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
    route_rel = _route_rel_from_sample(sample)
    tp, tp_next = _target_points(sample)
    speed = _last_scalar(sample, "speed_hist")
    theta = _last_scalar(sample, "theta_hist")
    factor = _factor_label(sample)
    row = {
        "split": split,
        "sample_idx": sample_idx,
        "sample_key": _sample_key(sample),
        "route_rel": route_rel,
        "route_label": _route_label_from_rel(route_rel),
        "route_name": os.path.basename(route_rel),
        "frame_id": _to_int(sample.get("frame_id"), -1),
        "factor_label": factor,
        "command_id": _command_id(sample),
        "speed_mps": speed,
        "theta": theta,
        "target_x": float(tp[0]),
        "target_y": float(tp[1]),
        "target_dist": float(np.linalg.norm(tp)),
        "target_dir": _target_direction(tp),
        "target_next_x": float(tp_next[0]),
        "target_next_y": float(tp_next[1]),
        "target_next_dist": float(np.linalg.norm(tp_next)),
        "target_next_dir": _target_direction(tp_next),
        "conflict_area_family": _to_int(sample.get("conflict_area_family"), 0),
        "merge_active": int(_active_flag(sample, "merge_active") or _active_flag(sample, "merge_episode_active")),
        "junction_active": int(_active_flag(sample, "junction_cross_episode_active")),
        "borrow_active": int(_active_flag(sample, "borrow_cross_episode_active")),
    }
    return row


def _write_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_counter_csv(path: str, counter: Counter, key_name: str, value_name: str = "count") -> None:
    rows = [{key_name: key, value_name: value} for key, value in counter.most_common()]
    _write_csv(path, rows, [key_name, value_name])


def _write_event_factor_csv(path: str, event_factor: Dict[str, Counter]) -> None:
    fieldnames = ["route_label", *FACTOR_ORDER, "total"]
    rows = []
    for event in sorted(event_factor):
        counter = event_factor[event]
        total = sum(counter.values())
        row = {"route_label": event, "total": total}
        for factor in FACTOR_ORDER:
            row[factor] = counter.get(factor, 0)
        rows.append(row)
    _write_csv(path, rows, fieldnames)


def _route_measurements_dir(image_root: str, route_rel: str) -> str:
    return os.path.join(image_root, route_rel, "measurements")


def _load_route_measurements(image_root: str, route_rel: str) -> Dict[int, Dict[str, Any]]:
    meas_dir = _route_measurements_dir(image_root, route_rel)
    out: Dict[int, Dict[str, Any]] = {}
    if not os.path.isdir(meas_dir):
        return out
    for fname in sorted(os.listdir(meas_dir)):
        if not fname.endswith(".json.gz"):
            continue
        try:
            frame_id = int(fname.split(".")[0])
        except Exception:
            continue
        meas = _load_json_gz(os.path.join(meas_dir, fname))
        if meas is not None:
            out[frame_id] = meas
    return out


def _matrix_rt(ego_matrix: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    mat = _to_numpy(ego_matrix)
    if mat is None or mat.ndim != 2 or mat.shape[0] < 3 or mat.shape[1] < 4:
        return None
    mat = mat.astype(np.float32)
    return mat[:3, :3], mat[:3, 3:4]


def _local_point_to_world_xy(point_xy: Any, ego_matrix: Any) -> Optional[np.ndarray]:
    rt = _matrix_rt(ego_matrix)
    point = _to_numpy(point_xy)
    if rt is None or point is None or point.size < 2:
        return None
    rotation, translation = rt
    local = np.asarray([float(point.reshape(-1)[0]), float(point.reshape(-1)[1]), 0.0], dtype=np.float32).reshape(3, 1)
    world = rotation @ local + translation
    return world[:2, 0].astype(np.float32)


def _world_xy_to_current_local(world_xy: np.ndarray, current_ego_matrix: Any) -> Optional[np.ndarray]:
    rt = _matrix_rt(current_ego_matrix)
    if rt is None or world_xy is None or world_xy.size < 2:
        return None
    rotation, translation = rt
    world = np.asarray([float(world_xy[0]), float(world_xy[1]), 0.0], dtype=np.float32).reshape(3, 1)
    local = rotation.T @ (world - translation)
    return local[:2, 0].astype(np.float32)


def _future_target_in_current(
    future_meas: Dict[str, Any],
    current_ego_matrix: Any,
    key: str,
) -> Optional[np.ndarray]:
    if key not in future_meas:
        return None
    world_xy = _local_point_to_world_xy(future_meas.get(key), future_meas.get("ego_matrix"))
    if world_xy is None:
        return None
    return _world_xy_to_current_local(world_xy, current_ego_matrix)


def _build_tg_list_sidecar(
    split: str,
    samples: Sequence[Dict[str, Any]],
    image_root: str,
    output_dir: str,
    horizon: int,
    frame_step: int,
    seed: int,
    smoke_routes: int,
) -> Dict[str, Any]:
    route_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        route_to_indices[_route_rel_from_sample(sample)].append(idx)

    n_samples = len(samples)
    target = np.zeros((n_samples, horizon, 2), dtype=np.float32)
    target_next = np.zeros((n_samples, horizon, 2), dtype=np.float32)
    valid = np.zeros((n_samples, horizon), dtype=np.bool_)
    sample_keys = np.asarray([_sample_key(s) for s in samples])
    frame_ids = np.asarray([_to_int(s.get("frame_id"), -1) for s in samples], dtype=np.int32)

    stats = Counter()
    route_stats = []
    rng = random.Random(seed)
    smoke_route_set = set(rng.sample(list(route_to_indices), min(smoke_routes, len(route_to_indices)))) if smoke_routes > 0 else set()
    smoke_rows: List[Dict[str, Any]] = []

    for route_rel, indices in route_to_indices.items():
        measurements = _load_route_measurements(image_root, route_rel)
        if not measurements:
            stats["missing_route_measurements"] += len(indices)
            route_stats.append({"route_rel": route_rel, "samples": len(indices), "valid_slots": 0, "total_slots": len(indices) * horizon})
            continue
        valid_slots = 0
        for sample_idx in indices:
            sample = samples[sample_idx]
            frame_id = _to_int(sample.get("frame_id"), -1)
            current_meas = measurements.get(frame_id)
            if current_meas is None:
                stats["missing_current_measurement"] += 1
                continue
            current_ego_matrix = current_meas.get("ego_matrix")
            if _matrix_rt(current_ego_matrix) is None:
                stats["missing_current_ego_matrix"] += 1
                continue
            for slot in range(horizon):
                future_frame = frame_id + slot * frame_step
                future_meas = measurements.get(future_frame)
                if future_meas is None:
                    stats["missing_future_measurement"] += 1
                    continue
                tg = _future_target_in_current(future_meas, current_ego_matrix, "target_point")
                tg_next = _future_target_in_current(future_meas, current_ego_matrix, "target_point_next")
                if tg is None:
                    stats["missing_future_target_point"] += 1
                    continue
                if tg_next is None:
                    tg_next = tg
                    stats["missing_future_target_point_next"] += 1
                target[sample_idx, slot] = tg
                target_next[sample_idx, slot] = tg_next
                valid[sample_idx, slot] = True
                valid_slots += 1
            if route_rel in smoke_route_set:
                tp, tp_next = _target_points(sample)
                smoke_rows.append(
                    {
                        "split": split,
                        "sample_key": _sample_key(sample),
                        "route_rel": route_rel,
                        "frame_id": frame_id,
                        "sample_target_x": float(tp[0]),
                        "sample_target_y": float(tp[1]),
                        "tg_list0_x": float(target[sample_idx, 0, 0]),
                        "tg_list0_y": float(target[sample_idx, 0, 1]),
                        "tg_list0_valid": int(valid[sample_idx, 0]),
                        "sample_target_next_x": float(tp_next[0]),
                        "sample_target_next_y": float(tp_next[1]),
                        "tg_next_list0_x": float(target_next[sample_idx, 0, 0]),
                        "tg_next_list0_y": float(target_next[sample_idx, 0, 1]),
                    }
                )
        route_stats.append(
            {
                "route_rel": route_rel,
                "samples": len(indices),
                "valid_slots": int(valid_slots),
                "total_slots": int(len(indices) * horizon),
            }
        )

    out_npz = os.path.join(output_dir, f"tg_list_{split}.npz")
    np.savez_compressed(
        out_npz,
        tg_list_current_ego=target,
        tg_next_list_current_ego=target_next,
        valid_mask=valid,
        sample_keys=sample_keys,
        frame_ids=frame_ids,
        horizon=np.asarray([horizon], dtype=np.int32),
        frame_step=np.asarray([frame_step], dtype=np.int32),
    )

    total_slots = int(n_samples * horizon)
    summary = {
        "split": split,
        "samples": int(n_samples),
        "horizon": int(horizon),
        "frame_step": int(frame_step),
        "valid_slots": int(valid.sum()),
        "total_slots": total_slots,
        "valid_ratio": float(valid.sum() / max(total_slots, 1)),
        "stats": dict(stats),
        "npz_path": out_npz,
    }
    with open(os.path.join(output_dir, f"tg_list_{split}.summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _write_csv(
        os.path.join(output_dir, f"tg_list_{split}.route_coverage.csv"),
        route_stats,
        ["route_rel", "samples", "valid_slots", "total_slots"],
    )
    if smoke_rows:
        _write_csv(
            os.path.join(output_dir, f"tg_list_{split}.smoke.csv"),
            smoke_rows[: max(1, smoke_routes * 20)],
            [
                "split",
                "sample_key",
                "route_rel",
                "frame_id",
                "sample_target_x",
                "sample_target_y",
                "tg_list0_x",
                "tg_list0_y",
                "tg_list0_valid",
                "sample_target_next_x",
                "sample_target_next_y",
                "tg_next_list0_x",
                "tg_next_list0_y",
            ],
        )
    return summary


def _raw_metadata_row(image_root: str, sample: Dict[str, Any]) -> Tuple[Dict[str, Any], Counter]:
    stats = Counter()
    route_rel = _route_rel_from_sample(sample)
    frame_id = _to_int(sample.get("frame_id"), -1)
    meas_path = os.path.join(image_root, route_rel, "measurements", f"{frame_id:04d}.json.gz")
    boxes_path = os.path.join(image_root, route_rel, "boxes", f"{frame_id:04d}.json.gz")
    meas = _load_json_gz(meas_path)
    boxes = _load_json_gz(boxes_path)
    if meas is None:
        stats["missing_measurement"] += 1
        meas = {}
    if boxes is None:
        stats["missing_boxes"] += 1
        boxes = []
    vehicle_distances = []
    walker_distances = []
    for box in boxes if isinstance(boxes, list) else []:
        cls = str(box.get("class", ""))
        dist = _to_float(box.get("distance"), float("nan"))
        if cls == "Car" or "vehicle" in cls.lower():
            vehicle_distances.append(dist)
        if "walker" in cls.lower() or "pedestrian" in cls.lower():
            walker_distances.append(dist)
    row = {
        "speed_limit": _to_float(meas.get("speed_limit")),
        "target_speed": _to_float(meas.get("target_speed")),
        "junction": int(bool(meas.get("junction", False))),
        "vehicle_hazard": int(bool(meas.get("vehicle_hazard", False))),
        "walker_hazard": int(bool(meas.get("walker_hazard", False))),
        "light_hazard": int(bool(meas.get("light_hazard", False))),
        "stop_sign_hazard": int(bool(meas.get("stop_sign_hazard", False))),
        "num_boxes": len(boxes) if isinstance(boxes, list) else 0,
        "nearest_vehicle_dist": min(vehicle_distances) if vehicle_distances else float("nan"),
        "nearest_walker_dist": min(walker_distances) if walker_distances else float("nan"),
    }
    return row, stats


def _load_bev_pooler(args: argparse.Namespace) -> Tuple[Optional[np.memmap], Optional[Dict[str, Any]], Optional[Tuple[int, ...]]]:
    if not args.bev_feature_index or not args.bev_feature_bin:
        return None, None, None
    meta = _load_pickle(args.bev_feature_index)
    index = meta.get("index", meta if isinstance(meta, dict) else {})
    shape = tuple(meta.get("bev_feat_shape", ())) if isinstance(meta, dict) else ()
    if not shape:
        raise ValueError("--bev-feature-index does not contain bev_feat_shape")
    mmap = np.memmap(args.bev_feature_bin, dtype=np.float16, mode="r", shape=shape)
    return mmap, index, shape


def _bev_pooled_feature(
    sample: Dict[str, Any],
    image_root: str,
    mmap: Optional[np.memmap],
    index: Optional[Dict[str, Any]],
) -> Optional[np.ndarray]:
    if mmap is None or index is None:
        return None
    packed_path = _packed_path_for_feature(image_root, sample)
    frame_id = _to_int(sample.get("frame_id"), -1)
    if packed_path is None or frame_id < 0:
        return None
    route_info = index.get(packed_path)
    if route_info is None:
        return None
    n_frames = int(route_info.get("n_frames", len(route_info.get("frame_num_to_idx", []))))
    if frame_id >= n_frames:
        return None
    abs_idx = int(route_info["offset"]) + frame_id
    feat = np.asarray(mmap[abs_idx], dtype=np.float32)
    if feat.ndim != 3:
        return None
    return feat.mean(axis=(1, 2)).astype(np.float32)


def _run_clustering(
    split: str,
    samples: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: str,
) -> Dict[str, Any]:
    try:
        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError("Clustering requires scikit-learn in the active environment") from exc

    rng = np.random.default_rng(args.seed)
    n = len(samples)
    chosen = np.arange(n)
    if args.cluster_max_samples > 0 and n > args.cluster_max_samples:
        chosen = rng.choice(n, size=args.cluster_max_samples, replace=False)
        chosen.sort()

    bev_mmap, bev_index, _ = _load_bev_pooler(args)
    feature_vectors = []
    kept_indices = []
    missing_bev = 0
    for idx in chosen:
        vec = _model_visible_vector(samples[int(idx)])
        if args.bev_feature_index and args.bev_feature_bin:
            bev_vec = _bev_pooled_feature(samples[int(idx)], args.image_data_root, bev_mmap, bev_index)
            if bev_vec is None:
                missing_bev += 1
                continue
            vec = np.concatenate([vec, bev_vec], axis=0)
        if np.all(np.isfinite(vec)):
            feature_vectors.append(vec)
            kept_indices.append(int(idx))

    if not feature_vectors:
        raise RuntimeError(f"No finite cluster features for split={split}")
    x = np.stack(feature_vectors, axis=0)
    x = StandardScaler().fit_transform(x)

    summaries = {
        "split": split,
        "requested_samples": int(len(chosen)),
        "cluster_samples": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "missing_bev": int(missing_bev),
        "methods": {},
    }
    assignment_rows = []
    for method in args.cluster_methods:
        n_clusters = min(args.num_clusters, int(x.shape[0]))
        if n_clusters <= 0:
            summaries["methods"][method] = {"skipped": "no_samples"}
            continue
        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
            labels = model.fit_predict(x)
        elif method == "gmm":
            if x.shape[0] < 2:
                summaries["methods"][method] = {"skipped": "gmm_requires_at_least_2_samples"}
                continue
            model = GaussianMixture(n_components=n_clusters, random_state=args.seed)
            labels = model.fit_predict(x)
        else:
            raise ValueError(f"Unsupported cluster method: {method}")
        cluster_counts = Counter(map(int, labels))
        summaries["methods"][method] = {"n_clusters": int(n_clusters), "cluster_counts": dict(cluster_counts)}
        for sample_idx, label in zip(kept_indices, labels):
            row = dict(rows[sample_idx])
            row["cluster_method"] = method
            row["cluster_id"] = int(label)
            assignment_rows.append(row)

    cluster_path = os.path.join(output_dir, f"clusters_{split}.csv")
    fieldnames = list(rows[0].keys()) + ["cluster_method", "cluster_id"] if rows else ["cluster_method", "cluster_id"]
    _write_csv(cluster_path, assignment_rows, fieldnames)
    summaries["assignment_csv"] = cluster_path
    with open(os.path.join(output_dir, f"clusters_{split}.summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    return summaries


def _analyze_split(split: str, split_dir: str, args: argparse.Namespace, output_dir: str) -> Dict[str, Any]:
    packed_path = os.path.join(split_dir, "samples_packed.pkl")
    samples = _load_pickle(packed_path)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    rows: List[Dict[str, Any]] = []
    event_counts: Counter = Counter()
    factor_counts: Counter = Counter()
    event_factor: Dict[str, Counter] = defaultdict(Counter)
    command_counts: Counter = Counter()
    speed_bins: Counter = Counter()
    target_dist_bins: Counter = Counter()
    target_direction_counts: Counter = Counter()
    route_stats: Dict[str, Counter] = defaultdict(Counter)
    raw_missing: Counter = Counter()

    for idx, sample in enumerate(samples):
        row = _sample_summary_row(split, idx, sample)
        if args.join_raw_metadata:
            raw_row, raw_stats = _raw_metadata_row(args.image_data_root, sample)
            row.update(raw_row)
            raw_missing.update(raw_stats)
        rows.append(row)
        event = row["route_label"]
        factor = row["factor_label"]
        event_counts[event] += 1
        factor_counts[factor] += 1
        event_factor[event][factor] += 1
        command_counts[str(row["command_id"])] += 1
        speed_bins[_bin_value(float(row["speed_mps"]), SPEED_BINS, "speed")] += 1
        target_dist_bins[_bin_value(float(row["target_dist"]), DIST_BINS, "target_dist")] += 1
        target_direction_counts[str(row["target_dir"])] += 1
        route_stats[row["route_rel"]]["samples"] += 1
        route_stats[row["route_rel"]][f"factor_{factor}"] += 1
        if factor != "none":
            route_stats[row["route_rel"]]["active_samples"] += 1

    fieldnames = list(rows[0].keys()) if rows else []
    if args.write_sample_csv and rows:
        _write_csv(os.path.join(output_dir, f"samples_{split}.csv"), rows, fieldnames)
    _write_counter_csv(os.path.join(output_dir, f"route_label_counts_{split}.csv"), event_counts, "route_label")
    _write_counter_csv(os.path.join(output_dir, f"factor_counts_{split}.csv"), factor_counts, "factor_label")
    _write_counter_csv(os.path.join(output_dir, f"command_counts_{split}.csv"), command_counts, "command_id")
    _write_counter_csv(os.path.join(output_dir, f"speed_bins_{split}.csv"), speed_bins, "speed_bin")
    _write_counter_csv(os.path.join(output_dir, f"target_dist_bins_{split}.csv"), target_dist_bins, "target_dist_bin")
    _write_counter_csv(os.path.join(output_dir, f"target_direction_counts_{split}.csv"), target_direction_counts, "target_direction")
    _write_event_factor_csv(os.path.join(output_dir, f"route_label_x_factor_{split}.csv"), event_factor)

    route_rows = []
    for route_rel, counter in sorted(route_stats.items()):
        samples_count = int(counter.get("samples", 0))
        route_rows.append(
            {
                "route_rel": route_rel,
                "route_label": _route_label_from_rel(route_rel),
                "samples": samples_count,
                "active_samples": int(counter.get("active_samples", 0)),
                "active_fraction": float(counter.get("active_samples", 0) / max(samples_count, 1)),
                "none": int(counter.get("factor_none", 0)),
                "borrow": int(counter.get("factor_borrow", 0)),
                "merge": int(counter.get("factor_merge", 0)),
                "junction": int(counter.get("factor_junction", 0)),
                "unknown": int(counter.get("factor_unknown", 0)),
            }
        )
    _write_csv(
        os.path.join(output_dir, f"route_factor_summary_{split}.csv"),
        route_rows,
        ["route_rel", "route_label", "samples", "active_samples", "active_fraction", "none", "borrow", "merge", "junction", "unknown"],
    )

    tg_summary = None
    if args.build_tg_list:
        if not args.image_data_root:
            raise ValueError("--build-tg-list requires --image-data-root")
        tg_summary = _build_tg_list_sidecar(
            split=split,
            samples=samples,
            image_root=args.image_data_root,
            output_dir=output_dir,
            horizon=args.tg_list_horizon,
            frame_step=args.tg_list_frame_step,
            seed=args.seed,
            smoke_routes=args.tg_list_smoke_routes,
        )

    cluster_summary = None
    if args.cluster:
        cluster_summary = _run_clustering(split, samples, rows, args, output_dir)

    summary = {
        "split": split,
        "packed_path": packed_path,
        "samples": len(samples),
        "routes": len(route_stats),
        "route_labels": dict(event_counts),
        "factor_counts": dict(factor_counts),
        "raw_metadata_missing": dict(raw_missing),
        "tg_list": tg_summary,
        "clusters": cluster_summary,
        "model_visible_feature_policy": {
            "included": ["bev_feature(optional pooled)", "command", "target_point", "target_point_next", "speed", "theta", "waypoints"],
            "excluded_from_model_visible": ["speed_limit", "scenario_name", "raw hazards", "boxes"],
        },
    }
    with open(os.path.join(output_dir, f"summary_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only PDM-Lite model-visible distribution and tg_list analysis."
    )
    parser.add_argument("--split-root", required=True, help="Root containing train/val subdirs with samples_packed.pkl.")
    parser.add_argument("--image-data-root", default="", help="Raw PDM-Lite root for measurements/boxes and optional BEV index joins.")
    parser.add_argument("--output-dir", default="", help="Output dir. Defaults to eval/analysis/pdmlite_distribution_<date>.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits under --split-root to analyze.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional per-split sample cap for smoke tests.")
    parser.add_argument("--no-sample-csv", dest="write_sample_csv", action="store_false", help="Skip per-sample CSV.")
    parser.set_defaults(write_sample_csv=True)

    parser.add_argument("--join-raw-metadata", action="store_true", help="Join speed_limit/hazard/boxes metadata for audit-only tables.")
    parser.add_argument("--build-tg-list", action="store_true", help="Build tg_list sidecar NPZ from raw measurements.")
    parser.add_argument("--tg-list-horizon", type=int, default=32, help="Number of future raw frames in tg_list.")
    parser.add_argument("--tg-list-frame-step", type=int, default=1, help="Raw frame step between tg_list slots.")
    parser.add_argument("--tg-list-smoke-routes", type=int, default=10, help="Number of routes to sample for tg_list smoke CSV.")

    parser.add_argument("--cluster", action="store_true", help="Run simple clustering on model-visible features.")
    parser.add_argument("--cluster-methods", nargs="+", default=["kmeans"], choices=["kmeans", "gmm"])
    parser.add_argument("--num-clusters", type=int, default=12)
    parser.add_argument("--cluster-max-samples", type=int, default=50000)
    parser.add_argument("--bev-feature-index", default="", help="Optional feature_index.pkl for pooled BEV clustering.")
    parser.add_argument("--bev-feature-bin", default="", help="Optional bev_features_fp16.bin for pooled BEV clustering.")
    parser.add_argument("--seed", type=int, default=20260630)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir:
        stamp = datetime.now().strftime("%Y%m%d")
        output_dir = os.path.join("eval", "analysis", f"pdmlite_distribution_{stamp}")
    _ensure_dir(output_dir)

    all_summaries = {}
    for split in args.splits:
        split_dir = os.path.join(args.split_root, split)
        if not os.path.exists(os.path.join(split_dir, "samples_packed.pkl")):
            raise FileNotFoundError(f"Missing samples_packed.pkl for split={split}: {split_dir}")
        all_summaries[split] = _analyze_split(split, split_dir, args, output_dir)

    top_summary = {
        "split_root": args.split_root,
        "image_data_root": args.image_data_root,
        "output_dir": output_dir,
        "splits": args.splits,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summaries": all_summaries,
    }
    with open(os.path.join(output_dir, "summary_all.json"), "w", encoding="utf-8") as f:
        json.dump(top_summary, f, indent=2, ensure_ascii=False)
    print(f"Done. Wrote analysis to: {output_dir}")


if __name__ == "__main__":
    main()
