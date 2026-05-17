#!/usr/bin/env python3
"""Convert legacy NavSim BEV NPZ shards into one official-token NPY cache.

The legacy shards are npz zip containers. They are okay for archival storage but
slow to reopen during every training run. This script filters/dedupes tokens once
and writes contiguous .npy arrays that can be memory-mapped by the trainer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


ARRAY_SPECS = {
    "bev_grid": ((64, 64, 64), np.float16),
    "bev_feature": ((512, 8, 8), np.float16),
    "ego_status": ((4, 14), np.float32),
    "trajectory": ((8, 2), np.float32),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert NavSim BEV NPZ cache to consolidated NPY cache.")
    parser.add_argument("--input-dir", default="/workspace2/z_project/motdp_bev_cache_official_final")
    parser.add_argument("--output-dir", default="/workspace2/z_project/motdp_bev_cache_official_npy")
    parser.add_argument("--token-filter-file", default=None)
    parser.add_argument("--dedupe-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only scan metadata and report selected sample count.")
    return parser.parse_args()


def discover_shards(input_dir: Path):
    paths = sorted(
        p for p in input_dir.glob("bev_cache_shard*.npz")
        if not p.stem.endswith("_meta") and not p.stem.endswith("_tmp")
    )
    if not paths:
        raise FileNotFoundError(f"No bev_cache_shard*.npz files found in {input_dir}")
    return paths


def load_meta(path: Path):
    with np.load(path, allow_pickle=False) as d:
        if {"tokens", "log_names", "frame_indices"}.issubset(d.files):
            return (
                d["tokens"].astype(str),
                d["log_names"].astype(str),
                d["frame_indices"].astype(np.int32),
            )

    sidecar = path.with_name(path.stem + "_meta.npz")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Metadata missing for {path}; expected {sidecar}")
    with np.load(sidecar, allow_pickle=False) as m:
        return (
            m["tokens"].astype(str),
            m["log_names"].astype(str),
            m["frame_indices"].astype(np.int32),
        )


def build_selection(paths, allowed_tokens, dedupe_tokens):
    seen = set()
    selections = []
    total_raw = 0
    total_kept = 0
    for path in paths:
        tokens, log_names, frame_indices = load_meta(path)
        keep = []
        for i, token in enumerate(tokens):
            if allowed_tokens is not None and token not in allowed_tokens:
                continue
            if dedupe_tokens and token in seen:
                continue
            seen.add(token)
            keep.append(i)
        keep = np.asarray(keep, dtype=np.int64)
        selections.append((path, keep, tokens[keep], log_names[keep], frame_indices[keep]))
        total_raw += len(tokens)
        total_kept += len(keep)
        print(f"{path.name}: raw={len(tokens)} kept={len(keep)}")
    print(f"Raw entries: {total_raw}")
    print(f"Kept entries: {total_kept}")
    print(f"Unique kept tokens: {len(seen)}")
    return selections, total_kept


def create_memmaps(output_dir: Path, count: int):
    arrays = {}
    for name, (shape_tail, dtype) in ARRAY_SPECS.items():
        arrays[name] = np.lib.format.open_memmap(
            output_dir / f"{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=(count, *shape_tail),
        )
    return arrays


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace files")
    output_dir.mkdir(parents=True, exist_ok=True)

    token_filter_file = args.token_filter_file
    if token_filter_file is None:
        candidate = input_dir / "navtrain_official_tokens.txt"
        token_filter_file = str(candidate) if candidate.is_file() else None
    allowed_tokens = set(Path(token_filter_file).read_text().split()) if token_filter_file else None
    if allowed_tokens is not None:
        print(f"Allowed tokens: {len(allowed_tokens)} from {token_filter_file}")

    paths = discover_shards(input_dir)
    selections, count = build_selection(paths, allowed_tokens, args.dedupe_tokens)
    if args.dry_run:
        print(f"Dry run complete. Would write {count} samples to {output_dir}")
        return
    arrays = create_memmaps(output_dir, count)

    all_tokens = []
    all_log_names = []
    all_frame_indices = []
    offset = 0
    t0 = time.time()
    for path, keep, tokens, log_names, frame_indices in selections:
        if len(keep) == 0:
            continue
        print(f"Writing {path.name}: {len(keep)} samples")
        with np.load(path, allow_pickle=False) as d:
            end = offset + len(keep)
            for name, (_, dtype) in ARRAY_SPECS.items():
                src = d[name]
                for start in tqdm(range(0, len(keep), 1024), desc=f"{path.name}:{name}", leave=False):
                    chunk_idx = keep[start : start + 1024]
                    out_slice = slice(offset + start, offset + start + len(chunk_idx))
                    arrays[name][out_slice] = src[chunk_idx].astype(dtype, copy=False)
            offset = end
        all_tokens.append(tokens)
        all_log_names.append(log_names)
        all_frame_indices.append(frame_indices)
        for arr in arrays.values():
            arr.flush()
        print(f"  done offset={offset}/{count} elapsed={(time.time() - t0) / 60:.1f}min")

    tokens = np.concatenate(all_tokens)
    log_names = np.concatenate(all_log_names)
    frame_indices = np.concatenate(all_frame_indices).astype(np.int32)
    np.savez(
        output_dir / "cache_index.npz",
        tokens=tokens,
        log_names=log_names,
        frame_indices=frame_indices,
    )
    if token_filter_file:
        target = output_dir / "navtrain_official_tokens.txt"
        target.write_text(Path(token_filter_file).read_text())

    traj = np.load(output_dir / "trajectory.npy", mmap_mode="r")
    stats = {
        "count": int(count),
        "traj_mean": np.asarray(traj[: min(count, 4096)], dtype=np.float32).mean(axis=(0, 1)).tolist(),
        "traj_std": np.asarray(traj[: min(count, 4096)], dtype=np.float32).std(axis=(0, 1)).tolist(),
        "input_dir": str(input_dir),
        "token_filter_file": token_filter_file,
        "dedupe_tokens": bool(args.dedupe_tokens),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (output_dir / "metadata.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
