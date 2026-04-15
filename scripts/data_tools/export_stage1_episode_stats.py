#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def _atomic_json_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True, sort_keys=True)
    os.replace(tmp_path, path)


def _base_dir_from_sample(sample: Dict[str, Any]) -> str:
    feat = str(sample.get("transfuser_bev_feature", "") or "")
    return os.path.dirname(os.path.dirname(feat)) if feat else ""


def _event_name_from_scene(scene: str) -> str:
    scene = str(scene or "")
    return scene.split("/", 1)[0] if "/" in scene else scene


def _frame_id_from_sample(sample: Dict[str, Any]) -> int:
    return int(sample.get("frame_id", -1))


def _group_indices_by_scene(samples: List[Dict[str, Any]]) -> List[Tuple[str, List[int]]]:
    scene_to_indices: Dict[str, List[int]] = {}
    scene_order: List[str] = []
    for idx, sample in enumerate(samples):
        base_dir = _base_dir_from_sample(sample)
        if base_dir not in scene_to_indices:
            scene_to_indices[base_dir] = []
            scene_order.append(base_dir)
        scene_to_indices[base_dir].append(idx)
    grouped: List[Tuple[str, List[int]]] = []
    for base_dir in scene_order:
        indices = scene_to_indices[base_dir]
        indices.sort(key=lambda i: _frame_id_from_sample(samples[i]))
        grouped.append((base_dir, indices))
    return grouped


def _scene_episode_ids(samples: List[Dict[str, Any]], indices: List[int], field_name: str) -> List[int]:
    episode_ids = []
    seen = set()
    for idx in indices:
        episode_id = int(samples[idx].get(field_name, -1))
        if episode_id < 0 or episode_id in seen:
            continue
        seen.add(episode_id)
        episode_ids.append(episode_id)
    return episode_ids


def _merge_stats(samples: List[Dict[str, Any]], grouped_scenes: List[Tuple[str, List[int]]]) -> Dict[str, Any]:
    active_samples = 0
    yld_samples = 0
    go_samples = 0
    active_scenes = 0
    start_events = 0
    end_events = 0
    go_events = 0
    no_go_episodes = 0
    event_scene_counter: Counter = Counter()
    event_sample_counter: Counter = Counter()
    end_state_counter: Counter = Counter()

    for scene, indices in grouped_scenes:
        event_name = _event_name_from_scene(scene)
        scene_has_active = False
        episode_ids = _scene_episode_ids(samples, indices, "merge_episode_id")
        start_events += len(episode_ids)
        end_events += len(episode_ids)

        seen_go_frames = set()
        seen_episode_no_go = set()
        for idx in indices:
            sample = samples[idx]
            if float(sample.get("merge_episode_active", 0.0)) > 0.5:
                active_samples += 1
                event_sample_counter[event_name] += 1
                scene_has_active = True
            phase = int(sample.get("merge_decision_phase", 0))
            if phase == 1:
                yld_samples += 1
            elif phase == 2:
                go_samples += 1

            go_frame = int(sample.get("merge_go_frame", -1))
            episode_id = int(sample.get("merge_episode_id", -1))
            if go_frame >= 0 and episode_id >= 0 and episode_id not in seen_go_frames:
                seen_go_frames.add(episode_id)
                go_events += 1
            if float(sample.get("merge_episode_no_go", 0.0)) > 0.5 and episode_id >= 0 and episode_id not in seen_episode_no_go:
                seen_episode_no_go.add(episode_id)
                no_go_episodes += 1
            end_state = int(sample.get("merge_end_state", 0))
            if end_state > 0 and episode_id >= 0:
                end_state_counter[end_state] += 1

        if scene_has_active:
            active_scenes += 1
            event_scene_counter[event_name] += 1

    return {
        "active_samples": active_samples,
        "active_scenes": active_scenes,
        "episode_count": start_events,
        "yld_samples": yld_samples,
        "go_samples": go_samples,
        "start_events": start_events,
        "end_events": end_events,
        "go_events": go_events,
        "no_go_episodes": no_go_episodes,
        "active_scenes_by_event": dict(sorted(event_scene_counter.items())),
        "active_samples_by_event": dict(sorted(event_sample_counter.items())),
        "end_state_counts": dict(sorted(end_state_counter.items())),
    }


def _borrow_stats(samples: List[Dict[str, Any]], grouped_scenes: List[Tuple[str, List[int]]]) -> Dict[str, Any]:
    active_samples = 0
    yld_samples = 0
    go_samples = 0
    active_scenes = 0
    start_events = 0
    end_events = 0
    go_events = 0
    context_events = 0
    event_scene_counter: Counter = Counter()
    event_sample_counter: Counter = Counter()

    for scene, indices in grouped_scenes:
        event_name = _event_name_from_scene(scene)
        scene_has_active = False
        episode_ids = _scene_episode_ids(samples, indices, "borrow_cross_episode_id")
        start_events += len(episode_ids)
        end_events += len(episode_ids)

        seen_go_frames = set()
        seen_contexts = set()
        for idx in indices:
            sample = samples[idx]
            if float(sample.get("borrow_cross_episode_active", 0.0)) > 0.5:
                active_samples += 1
                event_sample_counter[event_name] += 1
                scene_has_active = True
            phase = int(sample.get("borrow_cross_decision_phase", 0))
            if phase == 1:
                yld_samples += 1
            elif phase == 2:
                go_samples += 1

            episode_id = int(sample.get("borrow_cross_episode_id", -1))
            go_frame = int(sample.get("borrow_cross_go_frame", -1))
            context_frame = int(sample.get("borrow_cross_context_frame", -1))
            if go_frame >= 0 and episode_id >= 0 and episode_id not in seen_go_frames:
                seen_go_frames.add(episode_id)
                go_events += 1
            if context_frame >= 0 and episode_id >= 0 and episode_id not in seen_contexts:
                seen_contexts.add(episode_id)
                context_events += 1

        if scene_has_active:
            active_scenes += 1
            event_scene_counter[event_name] += 1

    return {
        "active_samples": active_samples,
        "active_scenes": active_scenes,
        "episode_count": start_events,
        "yld_samples": yld_samples,
        "go_samples": go_samples,
        "start_events": start_events,
        "end_events": end_events,
        "go_events": go_events,
        "context_events": context_events,
        "active_scenes_by_event": dict(sorted(event_scene_counter.items())),
        "active_samples_by_event": dict(sorted(event_sample_counter.items())),
    }


def _junction_stats(samples: List[Dict[str, Any]], grouped_scenes: List[Tuple[str, List[int]]]) -> Dict[str, Any]:
    active_samples = 0
    active_scenes = 0
    start_events = 0
    end_events = 0
    yld_positive_samples = 0
    go_positive_samples = 0
    event_scene_counter: Counter = Counter()
    event_sample_counter: Counter = Counter()

    for scene, indices in grouped_scenes:
        event_name = _event_name_from_scene(scene)
        scene_has_active = False
        episode_ids = _scene_episode_ids(samples, indices, "junction_cross_episode_id")
        start_events += len(episode_ids)
        end_events += len(episode_ids)
        for idx in indices:
            sample = samples[idx]
            if float(sample.get("junction_cross_episode_active", 0.0)) > 0.5:
                active_samples += 1
                event_sample_counter[event_name] += 1
                scene_has_active = True

            yld = sample.get("speed_risk_junction_cross_yld_values")
            go = sample.get("speed_risk_junction_cross_go_values")
            if yld is not None and any(float(v) > 0.0 for v in yld):
                yld_positive_samples += 1
            if go is not None and any(float(v) > 0.0 for v in go):
                go_positive_samples += 1

        if scene_has_active:
            active_scenes += 1
            event_scene_counter[event_name] += 1

    return {
        "active_samples": active_samples,
        "active_scenes": active_scenes,
        "episode_count": start_events,
        "start_events": start_events,
        "end_events": end_events,
        "yld_positive_samples": yld_positive_samples,
        "go_positive_samples": go_positive_samples,
        "active_scenes_by_event": dict(sorted(event_scene_counter.items())),
        "active_samples_by_event": dict(sorted(event_sample_counter.items())),
        "note": "junction currently has active/start/end hard labels; yld/go are still represented as risk curves, not hard phases",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full-dataset stage1 episode stats.")
    parser.add_argument("--packed_path", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.packed_path, "rb") as f:
        samples = pickle.load(f)
    grouped_scenes = _group_indices_by_scene(samples)

    summary = {
        "packed_path": os.path.realpath(args.packed_path),
        "total_samples": len(samples),
        "total_scenes": len(grouped_scenes),
        "merge": _merge_stats(samples, grouped_scenes),
        "borrow": _borrow_stats(samples, grouped_scenes),
        "junction": _junction_stats(samples, grouped_scenes),
    }
    _atomic_json_dump(summary, args.summary_json)
    print(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
