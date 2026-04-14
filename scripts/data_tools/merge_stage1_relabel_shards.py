#!/usr/bin/env python3
import argparse
import glob
import json
import os
import pickle
from typing import Dict, List, Sequence


def _sample_key(sample: Dict) -> str:
    feat_rel = str(sample.get("transfuser_bev_feature", "") or "")
    route_name = str(sample.get("route_name", "") or "")
    frame_id = int(sample.get("frame_id", -1))
    return f"{feat_rel}|{frame_id}" if feat_rel else f"{route_name}|{frame_id}"


def _atomic_pickle_save(obj, target_path: str) -> None:
    tmp_path = target_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, target_path)


def _discover_overlay_paths(overlay_root: str) -> List[str]:
    pattern = os.path.join(os.path.realpath(overlay_root), "shard_*", "samples_packed.pkl")
    return sorted(glob.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge sharded stage1 relabel overlays back onto a base packed dataset."
    )
    parser.add_argument("--base", required=True, help="Base samples_packed.pkl")
    parser.add_argument("--output", required=True, help="Merged output samples_packed.pkl")
    parser.add_argument(
        "--overlay_root",
        default=None,
        help="Root directory containing shard_*/samples_packed.pkl",
    )
    parser.add_argument(
        "--overlay",
        dest="overlays",
        action="append",
        default=[],
        help="Explicit overlay samples_packed.pkl paths. Can be repeated.",
    )
    args = parser.parse_args()

    overlay_paths = list(args.overlays)
    if args.overlay_root:
        overlay_paths.extend(_discover_overlay_paths(args.overlay_root))
    overlay_paths = [os.path.realpath(path) for path in overlay_paths]
    overlay_paths = sorted(dict.fromkeys(overlay_paths))
    if not overlay_paths:
        raise ValueError("No overlay paths found. Provide --overlay_root or at least one --overlay.")

    with open(args.base, "rb") as f:
        base_samples = pickle.load(f)
    if not isinstance(base_samples, list):
        raise TypeError(f"Expected list in {args.base}, got {type(base_samples).__name__}")

    index = {}
    for sample_idx, sample in enumerate(base_samples):
        key = _sample_key(sample)
        if key in index:
            raise ValueError(f"duplicate key in base dataset: {key}")
        index[key] = sample_idx

    replaced_keys = set()
    overlay_reports = []
    for overlay_path in overlay_paths:
        with open(overlay_path, "rb") as f:
            overlay_samples = pickle.load(f)
        if not isinstance(overlay_samples, list):
            raise TypeError(
                f"Expected list in overlay {overlay_path}, got {type(overlay_samples).__name__}"
            )
        replaced_here = 0
        for sample in overlay_samples:
            key = _sample_key(sample)
            if key not in index:
                raise KeyError(f"overlay key missing from base dataset: {key}")
            if key in replaced_keys:
                raise ValueError(f"duplicate overlay key across shards: {key}")
            base_samples[index[key]] = sample
            replaced_keys.add(key)
            replaced_here += 1
        overlay_reports.append({
            "overlay": overlay_path,
            "sample_count": len(overlay_samples),
            "replaced_count": replaced_here,
        })

    if len(replaced_keys) != len(base_samples):
        raise RuntimeError(
            "overlay coverage mismatch: "
            f"replaced={len(replaced_keys)} base={len(base_samples)}"
        )

    _atomic_pickle_save(base_samples, args.output)
    summary = {
        "base": os.path.realpath(args.base),
        "output": os.path.realpath(args.output),
        "overlay_count": len(overlay_paths),
        "base_samples": len(base_samples),
        "replaced_samples": len(replaced_keys),
        "overlays": overlay_reports,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
