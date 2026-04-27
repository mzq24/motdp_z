#!/usr/bin/env python3
"""Build a scene-level train/val split from a single packed dataset.

This is intended for the post-relabel workflow where we already have one full
packed file and want to create a cleaner validation split without scene
leakage. The split is performed by scene (`base_dir` inferred from
`transfuser_bev_feature`), while preserving the original sample order inside
each output packed file.

Outputs:
  <out_root>/train/samples_packed.pkl
  <out_root>/val/samples_packed.pkl
  <out_root>/scene_holdout_manifest.json
  <out_root>/val_scenes.json

Optional:
  - exclude specific scenes/routes from validation via a text list
  - listed scenes are forced into train and never sampled into val
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from collections import Counter
from typing import Any, Dict, List, Set


def _load_packed_samples(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Packed samples not found: {path}")
    with open(path, "rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"Expected a list in {path}, got {type(samples).__name__}")
    return samples


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


def _sample_key(sample: Dict[str, Any]) -> str:
    feat_rel = str(sample.get("transfuser_bev_feature", "") or "")
    route_name = str(sample.get("route_name", "") or "")
    frame_id = int(sample.get("frame_id", -1))
    if feat_rel:
        return f"{feat_rel}|{frame_id}"
    return f"{route_name}|{frame_id}"


def _base_dir(sample: Dict[str, Any]) -> str:
    feat_rel = str(sample.get("transfuser_bev_feature", "") or "")
    if feat_rel:
        return os.path.dirname(os.path.dirname(feat_rel))
    route_name = str(sample.get("route_name", "") or "")
    return route_name


def _ordered_unique_scenes(samples: List[Dict[str, Any]]) -> List[str]:
    ordered = []
    seen = set()
    for sample in samples:
        base_dir = _base_dir(sample)
        if base_dir in seen:
            continue
        seen.add(base_dir)
        ordered.append(base_dir)
    return ordered


def _count_scene_samples(samples: List[Dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for sample in samples:
        counts[_base_dir(sample)] += 1
    return counts


def _scene_match_keys(scene_name: str) -> Set[str]:
    scene_name = str(scene_name or "")
    keys = {scene_name}
    if "/" in scene_name:
        keys.add(scene_name.split("/", 1)[1])
    return keys


def _parse_scene_list_line(line: str, lineno: int, list_path: str) -> str:
    parts = line.split()
    if len(parts) == 1:
        token = str(parts[0]).strip()
        return token
    if len(parts) == 2:
        scene_name = str(parts[0]).strip()
        route_name = str(parts[1]).strip()
        return f"{scene_name}/{route_name}"
    raise ValueError(
        f"Invalid scene list line {lineno} in {list_path}: expected "
        f"'route_name', 'scene_name route_name', or 'scene_name/route_name', got: {line}"
    )


def _load_excluded_val_scenes(list_path: str) -> Set[str]:
    excluded: Set[str] = set()
    if not list_path:
        return excluded
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"Excluded val scene list not found: {list_path}")
    with open(list_path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            excluded.add(_parse_scene_list_line(line, lineno, list_path))
    return excluded


def build_scene_holdout(
    packed_path: str,
    out_root: str,
    val_scene_ratio: float,
    seed: int,
    overwrite: bool,
    exclude_val_scene_list: str,
) -> None:
    samples = _load_packed_samples(packed_path)
    if not samples:
        raise ValueError(f"No samples found in {packed_path}")

    if not (0.0 < float(val_scene_ratio) < 1.0):
        raise ValueError(f"val_scene_ratio must be in (0,1), got {val_scene_ratio}")

    ordered_scenes = _ordered_unique_scenes(samples)
    total_scenes = len(ordered_scenes)
    if total_scenes < 2:
        raise ValueError(f"Need at least 2 scenes to split, got {total_scenes}")

    excluded_val_scenes = _load_excluded_val_scenes(exclude_val_scene_list)
    matched_excluded_val_scenes: Set[str] = set()
    matched_exclusion_tokens: Set[str] = set()
    for scene_name in ordered_scenes:
        match_keys = _scene_match_keys(scene_name)
        if match_keys & excluded_val_scenes:
            matched_excluded_val_scenes.add(scene_name)
            matched_exclusion_tokens.update(match_keys & excluded_val_scenes)
    unmatched_excluded_val_scenes = excluded_val_scenes - matched_exclusion_tokens
    eligible_val_scenes = [scene for scene in ordered_scenes if scene not in matched_excluded_val_scenes]
    if not eligible_val_scenes:
        raise ValueError(
            "No scenes left for validation after applying excluded val scene list"
        )

    val_scene_count = max(1, min(total_scenes - 1, int(math.ceil(total_scenes * float(val_scene_ratio)))))
    val_scene_count = min(val_scene_count, len(eligible_val_scenes))
    rng = random.Random(seed)
    shuffled_scenes = list(eligible_val_scenes)
    rng.shuffle(shuffled_scenes)
    val_scene_set = set(shuffled_scenes[:val_scene_count])

    train_samples = [sample for sample in samples if _base_dir(sample) not in val_scene_set]
    val_samples = [sample for sample in samples if _base_dir(sample) in val_scene_set]
    if not train_samples or not val_samples:
        raise RuntimeError(
            f"Invalid scene split: train={len(train_samples)} val={len(val_samples)}"
        )

    train_out_dir = os.path.join(out_root, "train")
    val_out_dir = os.path.join(out_root, "val")
    train_out_packed = os.path.join(train_out_dir, "samples_packed.pkl")
    val_out_packed = os.path.join(val_out_dir, "samples_packed.pkl")
    manifest_path = os.path.join(out_root, "scene_holdout_manifest.json")
    val_scenes_path = os.path.join(out_root, "val_scenes.json")

    existing_outputs = [
        train_out_packed,
        val_out_packed,
        manifest_path,
        val_scenes_path,
    ]
    if not overwrite and any(os.path.exists(path) for path in existing_outputs):
        raise FileExistsError(
            f"Output already exists under {out_root}. Use --overwrite to replace it."
        )

    _atomic_pickle_dump(train_samples, train_out_packed)
    _atomic_pickle_dump(val_samples, val_out_packed)
    _atomic_json_dump(sorted(val_scene_set), val_scenes_path)

    train_scene_counts = _count_scene_samples(train_samples)
    val_scene_counts = _count_scene_samples(val_samples)
    overlap = set(train_scene_counts) & set(val_scene_counts)
    if overlap:
        raise RuntimeError(f"Scene overlap detected after split: {len(overlap)} scenes")

    manifest = {
        "source_packed": os.path.abspath(packed_path),
        "out_root": os.path.abspath(out_root),
        "seed": int(seed),
        "val_scene_ratio": float(val_scene_ratio),
        "exclude_val_scene_list": os.path.abspath(exclude_val_scene_list) if exclude_val_scene_list else "",
        "total_samples": len(samples),
        "total_scenes": total_scenes,
        "eligible_val_scenes": len(eligible_val_scenes),
        "requested_excluded_val_scenes": len(excluded_val_scenes),
        "matched_excluded_val_scenes": len(matched_excluded_val_scenes),
        "unmatched_excluded_val_scenes": len(unmatched_excluded_val_scenes),
        "unmatched_excluded_val_scene_examples": sorted(unmatched_excluded_val_scenes)[:20],
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_sample_ratio": len(train_samples) / float(len(samples)),
        "val_sample_ratio": len(val_samples) / float(len(samples)),
        "train_scenes": len(train_scene_counts),
        "val_scenes": len(val_scene_counts),
        "train_examples": [_sample_key(sample) for sample in train_samples[:5]],
        "val_examples": [_sample_key(sample) for sample in val_samples[:10]],
    }
    _atomic_json_dump(manifest, manifest_path)

    print("========================================")
    print("Scene-level holdout split created")
    print(f"Source packed:       {packed_path}")
    print(f"Output root:         {out_root}")
    print(f"Train scenes:        {len(train_scene_counts)}")
    print(f"Val scenes:          {len(val_scene_counts)}")
    print(f"Train samples:       {len(train_samples)}")
    print(f"Val samples:         {len(val_samples)}")
    print(f"Val scene ratio:     {len(val_scene_counts) / float(total_scenes):.4%}")
    print(f"Val sample ratio:    {len(val_samples) / float(len(samples)):.4%}")
    print(f"Excluded val scenes: {len(matched_excluded_val_scenes)}")
    print(f"Manifest:            {manifest_path}")
    print(f"Val scenes:          {val_scenes_path}")
    print("========================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packed-path",
        required=True,
        help="Full relabeled packed file path",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Output dataset root containing train/ and val/",
    )
    parser.add_argument(
        "--val-scene-ratio",
        type=float,
        default=0.05,
        help="Fraction of scenes to place into validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for deterministic scene sampling",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs under out-root",
    )
    parser.add_argument(
        "--exclude-val-scene-list",
        type=str,
        default="",
        help=(
            "Optional txt list of scenes/routes that must stay in train. "
            "Each line may be 'route_name', 'scene_name route_name', or 'scene_name/route_name'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_scene_holdout(
        packed_path=args.packed_path,
        out_root=args.out_root,
        val_scene_ratio=args.val_scene_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
        exclude_val_scene_list=args.exclude_val_scene_list,
    )


if __name__ == "__main__":
    main()
