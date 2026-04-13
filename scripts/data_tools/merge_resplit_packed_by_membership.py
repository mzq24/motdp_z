#!/usr/bin/env python3
"""
Merge train/val packed samples into one temporary packed file, or split a merged
packed file back into train/val using the original membership.

This is meant for scene-aware relabeling workflows where the current train/val
packed files were split at sample level and therefore no longer contain full
scene context per split.

Usage:
  1) Merge while saving original membership:
     python scripts/data_tools/merge_resplit_packed_by_membership.py merge \
       --train_packed /path/to/train/samples_packed.pkl \
       --val_packed /path/to/val/samples_packed.pkl \
       --merged_packed /path/to/full_refresh/samples_packed.pkl \
       --membership_json /path/to/full_refresh/original_membership.json

  2) After relabeling the merged pack, split it back:
     python scripts/data_tools/merge_resplit_packed_by_membership.py split \
       --merged_packed /path/to/full_refresh/samples_packed.pkl \
       --membership_json /path/to/full_refresh/original_membership.json \
       --train_out /path/to/train/samples_packed.refreshed.pkl \
       --val_out /path/to/val/samples_packed.refreshed.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Tuple


def _load_packed(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"{path} does not contain a list")
    return samples


def _atomic_pickle_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _atomic_json_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
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


def _dedupe_by_key(samples: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    dropped = 0
    for sample in samples:
        key = _sample_key(sample)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(sample)
    return deduped, dropped


def _build_membership(samples: Iterable[Dict[str, Any]]) -> List[str]:
    return [_sample_key(sample) for sample in samples]


def cmd_merge(args: argparse.Namespace) -> None:
    train_samples = _load_packed(args.train_packed)
    val_samples = _load_packed(args.val_packed)

    membership = {
        "train_keys": _build_membership(train_samples),
        "val_keys": _build_membership(val_samples),
    }

    merged_samples, dropped = _dedupe_by_key(list(train_samples) + list(val_samples))

    _atomic_pickle_dump(merged_samples, args.merged_packed)
    _atomic_json_dump(membership, args.membership_json)

    print("merge done")
    print(f"  train_samples={len(train_samples)}")
    print(f"  val_samples={len(val_samples)}")
    print(f"  merged_samples={len(merged_samples)}")
    print(f"  dropped_duplicates={dropped}")
    print(f"  merged_packed={args.merged_packed}")
    print(f"  membership_json={args.membership_json}")


def cmd_split(args: argparse.Namespace) -> None:
    merged_samples = _load_packed(args.merged_packed)
    with open(args.membership_json, "r", encoding="utf-8") as f:
        membership = json.load(f)

    train_keys = list(membership.get("train_keys", []))
    val_keys = list(membership.get("val_keys", []))
    train_set = set(train_keys)
    val_set = set(val_keys)

    sample_by_key = {}
    for sample in merged_samples:
        key = _sample_key(sample)
        sample_by_key[key] = sample

    missing_train = [key for key in train_keys if key not in sample_by_key]
    missing_val = [key for key in val_keys if key not in sample_by_key]
    if missing_train or missing_val:
        raise KeyError(
            "Missing keys while splitting merged packed: "
            f"missing_train={len(missing_train)} missing_val={len(missing_val)}"
        )

    train_out = [sample_by_key[key] for key in train_keys]
    val_out = [sample_by_key[key] for key in val_keys]

    overlap = train_set & val_set
    if overlap:
        print(f"warning: membership overlap count={len(overlap)}")

    _atomic_pickle_dump(train_out, args.train_out)
    _atomic_pickle_dump(val_out, args.val_out)

    print("split done")
    print(f"  merged_samples={len(merged_samples)}")
    print(f"  train_out={len(train_out)} -> {args.train_out}")
    print(f"  val_out={len(val_out)} -> {args.val_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge", help="Merge train/val packed files and save original membership.")
    merge_parser.add_argument("--train_packed", required=True)
    merge_parser.add_argument("--val_packed", required=True)
    merge_parser.add_argument("--merged_packed", required=True)
    merge_parser.add_argument("--membership_json", required=True)
    merge_parser.set_defaults(func=cmd_merge)

    split_parser = subparsers.add_parser("split", help="Split a merged packed file back into train/val using saved membership.")
    split_parser.add_argument("--merged_packed", required=True)
    split_parser.add_argument("--membership_json", required=True)
    split_parser.add_argument("--train_out", required=True)
    split_parser.add_argument("--val_out", required=True)
    split_parser.set_defaults(func=cmd_split)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
