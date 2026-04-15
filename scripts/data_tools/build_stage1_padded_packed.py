#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def _atomic_pickle_save(obj, target_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = target_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, target_path)


def _atomic_json_dump(obj, target_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = target_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp_path, target_path)


def _base_dir_from_sample(sample: Dict) -> str:
    feat = str(sample.get("transfuser_bev_feature", "") or "")
    return os.path.dirname(os.path.dirname(feat)) if feat else ""


def _sample_frame_id(sample: Dict) -> int:
    return int(sample.get("frame_id", -1))


def _load_json_gz(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _route_from_measurement(measurement: Dict) -> Optional[np.ndarray]:
    route = measurement.get("route")
    if route is None:
        return None
    route = np.asarray(route, dtype=np.float32)
    if route.ndim != 2 or route.shape[1] != 2 or route.shape[0] == 0:
        return None
    if route.shape[0] < 20:
        pad = np.repeat(route[-1:, :], 20 - route.shape[0], axis=0)
        route = np.concatenate([route, pad], axis=0)
    else:
        route = route[:20]
    return route.astype(np.float32)


def _one_hot_command(command_id: int, num_classes: int = 6) -> np.ndarray:
    out = np.zeros((num_classes,), dtype=np.float32)
    if 1 <= int(command_id) <= num_classes:
        out[int(command_id) - 1] = 1.0
    return out


def _feature_path_for_frame(template_path: str, frame_id: int) -> str:
    template_path = str(template_path or "")
    if not template_path:
        return ""
    if template_path.endswith("route_features.pt"):
        return template_path
    dirname = os.path.dirname(template_path)
    basename = os.path.basename(template_path)
    if basename.endswith("_feature_upsample.pt"):
        return os.path.join(dirname, f"{int(frame_id):04d}_feature_upsample.pt")
    if basename.endswith("_feature.pt"):
        return os.path.join(dirname, f"{int(frame_id):04d}_feature.pt")
    return template_path


def _template_vector(template_value, *, fill_value: float, default_shape: Tuple[int, ...]) -> np.ndarray:
    if template_value is None:
        return np.full(default_shape, fill_value, dtype=np.float32)
    arr = np.asarray(template_value, dtype=np.float32)
    if arr.size == 0:
        return np.full(default_shape, fill_value, dtype=np.float32)
    return np.full(arr.shape, fill_value, dtype=np.float32)


def _template_command_hist(template_value) -> np.ndarray:
    if template_value is None:
        return _one_hot_command(-1)[None, :]
    arr = np.asarray(template_value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] <= 0:
        return _one_hot_command(-1)[None, :]
    return np.zeros(arr.shape, dtype=np.float32)


def _build_stage1_padded_sample(
    *,
    scene: str,
    frame_id: int,
    measurement: Dict,
    template_sample: Dict,
    pad_kind: str,
) -> Optional[Dict]:
    route = _route_from_measurement(measurement)
    if route is None:
        return None

    speed = float(measurement.get("speed", 0.0))
    theta = float(measurement.get("theta", 0.0))
    throttle = float(measurement.get("throttle", 0.0))
    brake = float(measurement.get("brake", 0.0))
    command_vec = _one_hot_command(int(measurement.get("command", -1)))

    sample = {
        "town_name": template_sample.get("town_name", scene.split("/", 1)[0]),
        "event_name": template_sample.get("event_name", scene.split("/", 1)[0]),
        "route_name": template_sample.get("route_name", scene.rsplit("/", 1)[-1]),
        "frame_id": int(frame_id),
        "route": route,
        "speed_hist": _template_vector(
            template_sample.get("speed_hist"), fill_value=speed, default_shape=(1,)
        ),
        "theta_hist": _template_vector(
            template_sample.get("theta_hist"), fill_value=theta, default_shape=(1,)
        ),
        "throttle_hist": _template_vector(
            template_sample.get("throttle_hist"), fill_value=throttle, default_shape=(1,)
        ),
        "brake_hist": _template_vector(
            template_sample.get("brake_hist"), fill_value=brake, default_shape=(1,)
        ),
        "command_hist": _template_command_hist(template_sample.get("command_hist")),
        "transfuser_bev_feature": _feature_path_for_frame(
            template_sample.get("transfuser_bev_feature", ""), frame_id
        ),
        "transfuser_bev_feature_upsample": _feature_path_for_frame(
            template_sample.get("transfuser_bev_feature_upsample", ""), frame_id
        ),
        "stage1_padding_kind": str(pad_kind),
        "stage1_padding_source_frame": int(template_sample.get("frame_id", -1)),
    }

    command_hist = np.asarray(sample["command_hist"], dtype=np.float32)
    if command_hist.ndim == 2 and command_hist.shape[1] == command_vec.shape[0]:
        command_hist[...] = command_vec[None, :]
        sample["command_hist"] = command_hist

    return sample


def _load_coverage_map(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    routes = payload.get("routes", [])
    return {str(entry.get("scene")): entry for entry in routes if entry.get("scene")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a temporary stage1-only padded packed dataset by filling raw head/tail "
            "frames that are systematically trimmed from samples_packed.pkl."
        )
    )
    parser.add_argument("--packed_path", required=True, help="Original trimmed full samples_packed.pkl")
    parser.add_argument("--coverage_json", required=True, help="route_frame_coverage_index.json")
    parser.add_argument("--image_root", required=True, help="Raw dataset root")
    parser.add_argument("--output_path", required=True, help="Where to write the padded packed")
    parser.add_argument(
        "--summary_json",
        default=None,
        help="Optional summary JSON path. Defaults to <output_path>.summary.json",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = os.path.realpath(args.output_path)
    summary_json = os.path.realpath(args.summary_json or (args.output_path + ".summary.json"))

    if not args.overwrite and (os.path.exists(output_path) or os.path.exists(summary_json)):
        raise FileExistsError("Output exists; pass --overwrite to replace it.")

    with open(args.packed_path, "rb") as f:
        base_samples = pickle.load(f)
    if not isinstance(base_samples, list):
        raise TypeError(f"Expected list in {args.packed_path}, got {type(base_samples).__name__}")

    coverage_by_scene = _load_coverage_map(args.coverage_json)

    scene_to_samples: Dict[str, List[Dict]] = defaultdict(list)
    scene_order: List[str] = []
    for sample in base_samples:
        scene = _base_dir_from_sample(sample)
        if not scene:
            continue
        if scene not in scene_to_samples:
            scene_order.append(scene)
        scene_to_samples[scene].append(sample)

    padded_samples: List[Dict] = []
    event_counter = Counter()
    scene_reports: List[Dict] = []
    total_head_added = 0
    total_tail_added = 0
    total_failed_additions = 0

    for scene in scene_order:
        scene_samples = sorted(scene_to_samples[scene], key=_sample_frame_id)
        coverage = coverage_by_scene.get(scene)
        if coverage is None or not scene_samples:
            padded_samples.extend(scene_samples)
            continue

        raw_min = coverage.get("raw_frame_min")
        raw_max = coverage.get("raw_frame_max")
        packed_min = coverage.get("packed_frame_min")
        packed_max = coverage.get("packed_frame_max")
        if raw_min is None or raw_max is None or packed_min is None or packed_max is None:
            padded_samples.extend(scene_samples)
            continue

        head_frames = list(range(int(raw_min), int(packed_min)))
        tail_frames = list(range(int(packed_max) + 1, int(raw_max) + 1))
        if not head_frames and not tail_frames:
            padded_samples.extend(scene_samples)
            continue

        head_template = scene_samples[0]
        tail_template = scene_samples[-1]
        head_padded: List[Dict] = []
        tail_padded: List[Dict] = []
        failed_frames: List[int] = []

        for frame_id in head_frames:
            meas = _load_json_gz(os.path.join(args.image_root, scene, "measurements", f"{int(frame_id):04d}.json.gz"))
            if meas is None:
                failed_frames.append(int(frame_id))
                continue
            padded = _build_stage1_padded_sample(
                scene=scene,
                frame_id=int(frame_id),
                measurement=meas,
                template_sample=head_template,
                pad_kind="head",
            )
            if padded is None:
                failed_frames.append(int(frame_id))
                continue
            head_padded.append(padded)

        for frame_id in tail_frames:
            meas = _load_json_gz(os.path.join(args.image_root, scene, "measurements", f"{int(frame_id):04d}.json.gz"))
            if meas is None:
                failed_frames.append(int(frame_id))
                continue
            padded = _build_stage1_padded_sample(
                scene=scene,
                frame_id=int(frame_id),
                measurement=meas,
                template_sample=tail_template,
                pad_kind="tail",
            )
            if padded is None:
                failed_frames.append(int(frame_id))
                continue
            tail_padded.append(padded)

        scene_output = [*head_padded, *scene_samples, *tail_padded]
        scene_output.sort(key=_sample_frame_id)
        padded_samples.extend(scene_output)

        total_head_added += len(head_padded)
        total_tail_added += len(tail_padded)
        total_failed_additions += len(failed_frames)
        event_counter[scene.split("/", 1)[0]] += 1
        scene_reports.append(
            {
                "scene": scene,
                "original_sample_count": len(scene_samples),
                "padded_sample_count": len(scene_output),
                "head_added_count": len(head_padded),
                "tail_added_count": len(tail_padded),
                "failed_frame_count": len(failed_frames),
                "failed_frames_preview": failed_frames[:20],
                "packed_frame_min": int(packed_min),
                "packed_frame_max": int(packed_max),
                "raw_frame_min": int(raw_min),
                "raw_frame_max": int(raw_max),
            }
        )

    _atomic_pickle_save(padded_samples, output_path)
    summary = {
        "packed_path": os.path.realpath(args.packed_path),
        "coverage_json": os.path.realpath(args.coverage_json),
        "image_root": os.path.realpath(args.image_root),
        "output_path": output_path,
        "base_sample_count": len(base_samples),
        "padded_sample_count": len(padded_samples),
        "added_sample_count": len(padded_samples) - len(base_samples),
        "added_head_sample_count": total_head_added,
        "added_tail_sample_count": total_tail_added,
        "failed_added_sample_count": total_failed_additions,
        "padded_scene_count": len(scene_reports),
        "padded_scene_count_by_event": dict(sorted(event_counter.items())),
        "scene_reports": scene_reports,
    }
    _atomic_json_dump(summary, summary_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
