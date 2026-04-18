#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Tuple


def _atomic_json_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
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


def _family_name(sample: Dict[str, Any]) -> str:
    mapping = {
        0: "none",
        1: "borrow",
        2: "merge",
        3: "junction",
    }
    return mapping.get(int(sample.get("conflict_area_family", 0)), "unknown")


def _issue_debug(sample: Dict[str, Any]) -> Dict[str, Any]:
    stage1_debug = sample.get("stage1_speed_debug") or {}
    conflict_debug = stage1_debug.get("conflict_area") or {}
    return conflict_debug if isinstance(conflict_debug, dict) else {}


def _summarize_conflict_area(samples: List[Dict[str, Any]], grouped_scenes: List[Tuple[str, List[int]]]) -> Dict[str, Any]:
    active_samples_by_family: Counter = Counter()
    active_scenes_by_family: Counter = Counter()
    active_scene_samples_by_event: Dict[str, Counter] = {
        "borrow": Counter(),
        "merge": Counter(),
        "junction": Counter(),
    }
    active_scene_counts_by_event: Dict[str, Counter] = {
        "borrow": Counter(),
        "merge": Counter(),
        "junction": Counter(),
    }
    issue_frames = 0
    issue_scenes_by_family: Counter = Counter()
    missing_reason_counts: Counter = Counter()
    topology_override_counts: Counter = Counter()

    for scene, indices in grouped_scenes:
        event_name = _event_name_from_scene(scene)
        scene_active_families = set()
        scene_issue_families = set()
        scene_missing_reasons = set()
        for idx in indices:
            sample = samples[idx]
            if float(sample.get("conflict_area_active", 0.0)) > 0.5:
                family = _family_name(sample)
                if family in {"borrow", "merge", "junction"}:
                    active_samples_by_family[family] += 1
                    active_scene_samples_by_event[family][event_name] += 1
                    scene_active_families.add(family)

            conflict_debug = _issue_debug(sample)
            issue_count = int(conflict_debug.get("issue_count", 0))
            if issue_count > 0:
                issue_frames += 1
                for family in conflict_debug.get("issue_families", []):
                    family = str(family)
                    if family in {"borrow", "merge", "junction"}:
                        scene_issue_families.add(family)
                reason = str(conflict_debug.get("missing_reason", "none"))
                if reason and reason != "none":
                    scene_missing_reasons.add(reason)
                topology_override = str(conflict_debug.get("topology_override", "none"))
                if topology_override and topology_override != "none":
                    topology_override_counts[topology_override] += 1

        for family in scene_active_families:
            active_scenes_by_family[family] += 1
            active_scene_counts_by_event[family][event_name] += 1
        for family in scene_issue_families:
            issue_scenes_by_family[family] += 1
        for reason in scene_missing_reasons:
            missing_reason_counts[reason] += 1

    return {
        "active_samples_by_family": dict(sorted(active_samples_by_family.items())),
        "active_scenes_by_family": dict(sorted(active_scenes_by_family.items())),
        "active_samples_by_event": {
            family: dict(sorted(counter.items()))
            for family, counter in active_scene_samples_by_event.items()
        },
        "active_scenes_by_event": {
            family: dict(sorted(counter.items()))
            for family, counter in active_scene_counts_by_event.items()
        },
        "issue_frames": int(issue_frames),
        "issue_scenes_by_family": dict(sorted(issue_scenes_by_family.items())),
        "missing_reason_counts": dict(sorted(missing_reason_counts.items())),
        "topology_override_counts": dict(sorted(topology_override_counts.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export unified stage1 conflict-area stats.")
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
        "conflict_area": _summarize_conflict_area(samples, grouped_scenes),
    }
    _atomic_json_dump(summary, args.summary_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
