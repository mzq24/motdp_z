#!/usr/bin/env python3
"""Evaluate initial-noise sensitivity for clean paper motion policies.

The experiment fixes each validation scene and reruns DDIM sampling with
different initial noise seeds. If the final route/traj variance is tiny, the
policy is behaving mostly like a deterministic scene-conditioned refiner.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.unified_carla_dataset import CARLAImageDataset
from policy.paper_motion_policy import PaperMotionPolicy


def load_config(path: str) -> Dict:
    with open(path, "rt", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_val_dataset(config: Dict) -> CARLAImageDataset:
    training_cfg = config["training"]
    dataset_cfg = config.get("dataset", {})
    validation_cfg = config.get("validation", {})
    root = training_cfg["dataset_path"]
    return CARLAImageDataset(
        dataset_path=os.path.join(root, "val"),
        image_data_root=training_cfg["image_data_root"],
        mode="val",
        skip_memmap=not bool(validation_cfg.get("use_memmap", True)),
        use_per_frame=bool(dataset_cfg.get("use_per_frame", False)),
        cache_dir=dataset_cfg.get("cache_dir"),
        feature_suffix=dataset_cfg.get("feature_suffix", "") or "",
        load_transfuser_lidar_bev=False,
        filter_bad_routes=bool(dataset_cfg.get("val_filter_bad_routes", True)),
        retain_bad_routes_for_energy=False,
        use_fullres_upsample_cache=bool(dataset_cfg.get("use_fullres_upsample_cache", False)),
    )


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def make_indices(dataset_len: int, num_samples: int, seed: int, random_subset: bool) -> List[int]:
    n = min(int(num_samples), int(dataset_len))
    if random_subset:
        rng = random.Random(seed)
        indices = list(range(dataset_len))
        rng.shuffle(indices)
        return indices[:n]
    return list(range(n))


def offdiag_pairwise_l2(samples: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal pairwise L2.

    Args:
        samples: Tensor shaped [K, B, ..., D].

    Returns:
        Tensor shaped [B].
    """
    k, b = samples.shape[:2]
    if k <= 1:
        return torch.zeros(b, device=samples.device, dtype=samples.dtype)
    flat = samples.reshape(k, b, -1).permute(1, 0, 2)
    dist = torch.cdist(flat, flat, p=2)
    mask = ~torch.eye(k, device=samples.device, dtype=torch.bool)
    return dist[:, mask].reshape(b, k * (k - 1)).mean(dim=1)


def summarize_values(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def sample_with_seed(policy: PaperMotionPolicy, batch: Dict, seed: int, steps: int) -> Dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return policy.sample(batch, num_inference_steps=steps)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Tuple[Dict, List[Dict]]:
    config = load_config(args.config)
    if args.paper_sampling_mode:
        config.setdefault("route_b", {})["paper_sampling_mode"] = args.paper_sampling_mode
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy, _ = PaperMotionPolicy.load_checkpoint(args.checkpoint, config, device=str(device))
    policy.eval()

    dataset = build_val_dataset(config)
    indices = make_indices(len(dataset), args.num_samples, args.subset_seed, args.random_subset)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    seeds = [args.base_seed + i for i in range(args.num_seeds)]
    rows: List[Dict] = []
    global_idx = 0

    for batch in tqdm(loader, desc="noise sensitivity", dynamic_ncols=True):
        batch = move_batch(batch, device)
        routes = []
        trajs = []
        speeds = []
        speed_logits = []
        for seed in seeds:
            out = sample_with_seed(policy, batch, seed, args.num_inference_steps)
            routes.append(out["route"].detach().float())
            trajs.append(out["trajectory"].detach().float())
            if out.get("speed_mps") is not None:
                speeds.append(out["speed_mps"].detach().float())
            if out.get("speed_logits") is not None:
                speed_logits.append(out["speed_logits"].detach().float())

        route_stack = torch.stack(routes, dim=0)
        traj_stack = torch.stack(trajs, dim=0)
        bsz = route_stack.shape[1]

        route_std = route_stack.std(dim=0, unbiased=False)
        traj_std = traj_stack.std(dim=0, unbiased=False)
        route_point_std = torch.linalg.norm(route_std, dim=-1).mean(dim=-1)
        traj_point_std = torch.linalg.norm(traj_std, dim=-1).mean(dim=-1)
        route_endpoint_std = torch.linalg.norm(route_std[:, -1], dim=-1)
        traj_endpoint_std = torch.linalg.norm(traj_std[:, -1], dim=-1)
        route_pairwise = offdiag_pairwise_l2(route_stack)
        traj_pairwise = offdiag_pairwise_l2(traj_stack)
        route_pairwise_per_point = route_pairwise / float(np.sqrt(route_stack.shape[2]))
        traj_pairwise_per_point = traj_pairwise / float(np.sqrt(traj_stack.shape[2]))

        if speeds:
            speed_stack = torch.stack(speeds, dim=0)
            speed_std = speed_stack.std(dim=0, unbiased=False)
        else:
            speed_std = torch.full((bsz,), float("nan"), device=device)

        if speed_logits:
            probs = torch.softmax(torch.stack(speed_logits, dim=0).float(), dim=-1).mean(dim=0)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        else:
            entropy = torch.full((bsz,), float("nan"), device=device)

        for i in range(bsz):
            source_index = indices[global_idx] if global_idx < len(indices) else global_idx
            rows.append(
                {
                    "subset_index": int(global_idx),
                    "dataset_index": int(source_index),
                    "route_point_std": float(route_point_std[i].cpu()),
                    "route_endpoint_std": float(route_endpoint_std[i].cpu()),
                    "route_pairwise_l2": float(route_pairwise[i].cpu()),
                    "route_pairwise_l2_per_point": float(route_pairwise_per_point[i].cpu()),
                    "traj_point_std": float(traj_point_std[i].cpu()),
                    "traj_endpoint_std": float(traj_endpoint_std[i].cpu()),
                    "traj_pairwise_l2": float(traj_pairwise[i].cpu()),
                    "traj_pairwise_l2_per_point": float(traj_pairwise_per_point[i].cpu()),
                    "speed_mps_std": float(speed_std[i].cpu()),
                    "speed_entropy": float(entropy[i].cpu()),
                }
            )
            global_idx += 1

    metric_names = [
        "route_point_std",
        "route_endpoint_std",
        "route_pairwise_l2",
        "route_pairwise_l2_per_point",
        "traj_point_std",
        "traj_endpoint_std",
        "traj_pairwise_l2",
        "traj_pairwise_l2_per_point",
        "speed_mps_std",
        "speed_entropy",
    ]
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "paper_sampling_mode": config.get("route_b", {}).get("paper_sampling_mode"),
        "num_inference_steps": int(args.num_inference_steps),
        "num_seeds": int(args.num_seeds),
        "num_samples": int(len(rows)),
        "subset_seed": int(args.subset_seed),
        "random_subset": bool(args.random_subset),
        "metrics": {name: summarize_values(row[name] for row in rows) for name in metric_names},
    }
    return summary, rows


def write_outputs(summary: Dict, rows: List[Dict], output_json: str, output_csv: str) -> None:
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "wt", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["subset_index", "dataset_index"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=20260610)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--random-subset", action="store_true")
    parser.add_argument("--paper-sampling-mode", default=None, choices=[None, "diffusers_step", "old_pred_x0_ddim"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    summary, rows = evaluate(args)
    write_outputs(summary, rows, args.output_json, args.output_csv)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
