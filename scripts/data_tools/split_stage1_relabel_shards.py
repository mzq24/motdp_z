#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from typing import Dict, Iterable, List, Sequence, Tuple


def _load_stage1_field_names() -> Sequence[str]:
    try:
        from scripts.data_tools.precompute_semantic_labels import STAGE1_SPEED_FIELDS

        return tuple(STAGE1_SPEED_FIELDS)
    except Exception:
        return ()


STAGE1_SPEED_FIELDS = set(_load_stage1_field_names())


def _sample_key(sample: Dict) -> str:
    feat_rel = str(sample.get("transfuser_bev_feature", "") or "")
    route_name = str(sample.get("route_name", "") or "")
    frame_id = int(sample.get("frame_id", -1))
    return f"{feat_rel}|{frame_id}" if feat_rel else f"{route_name}|{frame_id}"


def _resolve_base_dir(sample: Dict) -> str:
    feat_rel = str(sample.get("transfuser_bev_feature", "") or "")
    if not feat_rel:
        return ""
    return os.path.dirname(os.path.dirname(feat_rel))


def _is_stage1_field(field_name: str) -> bool:
    if field_name in STAGE1_SPEED_FIELDS:
        return True
    if field_name == "stage1_speed_debug":
        return True
    if field_name.startswith("speed_sample_"):
        return True
    if field_name.startswith("speed_risk_"):
        return True
    if field_name in {"speed_cross_wait_time_s", "speed_cross_wait_valid"}:
        return True
    if field_name.startswith("junction_cross_episode_"):
        return True
    if field_name.startswith("borrow_cross_"):
        return True
    if field_name.startswith("merge_episode_"):
        return True
    if field_name in {
        "merge_decision_phase",
        "merge_go_frame",
        "merge_resolution_actor_id",
        "merge_end_state",
        "merge_hold",
    }:
        return True
    return False


def _clear_stage1_fields(sample: Dict) -> Dict:
    out = dict(sample)
    for key in list(out.keys()):
        if _is_stage1_field(key):
            out.pop(key, None)
    return out


def _atomic_pickle_save(obj, target_path: str) -> None:
    tmp_path = target_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, target_path)


def _atomic_json_save(payload, target_path: str) -> None:
    tmp_path = target_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, target_path)


def _build_route_groups(samples: Sequence[Dict]) -> List[Dict]:
    groups_by_base: Dict[str, Dict] = {}
    ordered_bases: List[str] = []
    for sample_idx, sample in enumerate(samples):
        base_dir = _resolve_base_dir(sample)
        group = groups_by_base.get(base_dir)
        if group is None:
            group = {
                "base_dir": base_dir,
                "indices": [],
                "start_index": sample_idx,
                "end_index": sample_idx,
                "sample_count": 0,
            }
            groups_by_base[base_dir] = group
            ordered_bases.append(base_dir)
        group["indices"].append(sample_idx)
        group["end_index"] = sample_idx
        group["sample_count"] += 1
    return [groups_by_base[base_dir] for base_dir in ordered_bases]


def _split_groups_evenly(groups: Sequence[Dict], num_shards: int) -> List[List[Dict]]:
    if not groups:
        return []
    num_shards = max(1, min(int(num_shards), len(groups)))
    shards: List[List[Dict]] = []
    next_group_idx = 0
    remaining_samples = sum(int(group["sample_count"]) for group in groups)

    for shard_idx in range(num_shards):
        remaining_shards = num_shards - shard_idx
        target_samples = remaining_samples / max(remaining_shards, 1)
        shard_groups: List[Dict] = []
        shard_samples = 0

        while next_group_idx < len(groups):
            group = groups[next_group_idx]
            group_samples = int(group["sample_count"])
            remaining_groups_after_take = len(groups) - (next_group_idx + 1)
            need_leave = remaining_shards - 1
            would_exceed = shard_groups and (shard_samples + group_samples > target_samples)
            if would_exceed and remaining_groups_after_take >= need_leave:
                break
            shard_groups.append(group)
            shard_samples += group_samples
            next_group_idx += 1

        if not shard_groups and next_group_idx < len(groups):
            group = groups[next_group_idx]
            shard_groups.append(group)
            shard_samples += int(group["sample_count"])
            next_group_idx += 1

        shards.append(shard_groups)
        remaining_samples -= shard_samples

    if next_group_idx != len(groups):
        raise RuntimeError(
            f"failed to assign all groups: assigned={next_group_idx}, total={len(groups)}"
        )
    return shards


def _summarize_groups(groups: Sequence[Dict]) -> Dict:
    if not groups:
        return {
            "route_group_count": 0,
            "scene_count": 0,
            "sample_count": 0,
            "first_scene": "",
            "last_scene": "",
            "group_index_lo": -1,
            "group_index_hi": -1,
        }
    return {
        "route_group_count": len(groups),
        "scene_count": len(groups),
        "sample_count": sum(int(group["sample_count"]) for group in groups),
        "first_scene": str(groups[0]["base_dir"]),
        "last_scene": str(groups[-1]["base_dir"]),
        "group_index_lo": int(groups[0]["group_index"]),
        "group_index_hi": int(groups[-1]["group_index"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a packed dataset into stage1 relabel shards by contiguous route groups."
    )
    parser.add_argument("--source", required=True, help="Input samples_packed.pkl")
    parser.add_argument("--output_root", required=True, help="Directory for shard_* outputs")
    parser.add_argument("--num_shards", type=int, required=True, help="Number of shards to create")
    parser.add_argument(
        "--keep_existing_stage1",
        action="store_true",
        help="Do not clear existing stage1 fields before writing shards.",
    )
    args = parser.parse_args()

    with open(args.source, "rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"Expected list in {args.source}, got {type(samples).__name__}")

    groups = _build_route_groups(samples)
    for group_idx, group in enumerate(groups, start=1):
        group["group_index"] = group_idx

    shards = _split_groups_evenly(groups, args.num_shards)
    os.makedirs(args.output_root, exist_ok=True)

    summary = {
        "source": os.path.realpath(args.source),
        "output_root": os.path.realpath(args.output_root),
        "num_input_samples": len(samples),
        "num_route_groups": len(groups),
        "num_scenes": len(groups),
        "num_shards": len(shards),
        "clear_existing_stage1": not args.keep_existing_stage1,
        "shards": [],
    }

    seen_keys = set()
    for shard_idx, shard_groups in enumerate(shards, start=1):
        shard_dir = os.path.join(args.output_root, f"shard_{shard_idx:02d}_of_{len(shards):02d}")
        os.makedirs(shard_dir, exist_ok=True)
        shard_samples: List[Dict] = []
        for group in shard_groups:
            for sample_idx in group["indices"]:
                sample = samples[sample_idx]
                key = _sample_key(sample)
                if key in seen_keys:
                    raise ValueError(f"duplicate sample key across shards: {key}")
                seen_keys.add(key)
                shard_samples.append(
                    dict(sample) if args.keep_existing_stage1 else _clear_stage1_fields(sample)
                )

        shard_path = os.path.join(shard_dir, "samples_packed.pkl")
        meta_path = os.path.join(shard_dir, "split_meta.json")
        _atomic_pickle_save(shard_samples, shard_path)
        meta = {
            "name": os.path.basename(shard_dir),
            "shard_index": shard_idx,
            **_summarize_groups(shard_groups),
            "source": os.path.realpath(args.source),
            "output_path": os.path.realpath(shard_path),
        }
        _atomic_json_save(meta, meta_path)
        summary["shards"].append(meta)
        print(meta)

    if len(seen_keys) != len(samples):
        raise RuntimeError(f"shard coverage mismatch: seen={len(seen_keys)} total={len(samples)}")

    summary_path = os.path.join(args.output_root, "split_summary.json")
    _atomic_json_save(summary, summary_path)
    print(
        json.dumps(
            {
                "num_shards": len(shards),
                "num_route_groups": len(groups),
                "num_samples": len(samples),
                "summary_path": os.path.realpath(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
