#!/usr/bin/env python3
"""Backfill CL7 all-target validation for task-end checkpoints.

The training-time continual_validation.json used validation_mode=seen_targets, so it
cannot answer future-target / upper-triangle questions. This script evaluates a
single checkpoint on every target task, then merges the resulting parts into a
matrix-friendly JSON/Markdown summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from policy.nuplan_diffusion_policy import NuPlanDiffusionPolicy  # noqa: E402
from training.train_nuplan import (  # noqa: E402
    DEFAULT_VAL_CACHE_DIRS,
    build_eval_dataloader,
    load_config,
    load_pegp_state,
    load_policy_checkpoint,
    validate,
)

METRIC_KEYS = [
    "loss",
    "ego_loss",
    "neighbor_loss",
    "ego_ADE",
    "ego_FDE",
    "pred_final_disp",
    "gt_final_disp",
]


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _target_tasks(config: Dict[str, Any]) -> List[str]:
    tasks = config.get("continual_learning", {}).get("target_tasks", [])
    if not tasks:
        raise ValueError("Config does not define continual_learning.target_tasks")
    return list(tasks)


def _aggregate(per_task: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    total = sum(float(m.get("sample_count", 0.0)) for m in per_task.values())
    out: Dict[str, float] = {"sample_count": total}
    if total <= 0:
        return out
    for key in METRIC_KEYS:
        weighted = 0.0
        seen = False
        for metrics in per_task.values():
            count = float(metrics.get("sample_count", 0.0))
            if key in metrics and count > 0:
                weighted += float(metrics[key]) * count
                seen = True
        if seen:
            out[key] = weighted / total
    return out


def _load_policy(config: Dict[str, Any], checkpoint: Path) -> NuPlanDiffusionPolicy:
    policy = NuPlanDiffusionPolicy(config).cuda()
    norm_stats_path = config.get("norm_stats_path")
    if norm_stats_path:
        with open(norm_stats_path, "r", encoding="utf-8") as f:
            policy.load_norm_stats(json.load(f))

    ckpt = torch.load(str(checkpoint), map_location="cpu")
    model_state = ckpt.get("model", ckpt)
    load_policy_checkpoint(policy, model_state, 0, str(checkpoint))
    load_pegp_state(policy, ckpt, 0, str(checkpoint))
    policy.eval()
    return policy


def _build_task_loader(config: Dict[str, Any], task: str):
    cache_dirs = config.get("val_cache_dirs", DEFAULT_VAL_CACHE_DIRS)
    allowed_scenario_types = config.get(
        "val_allowed_scenario_types", config.get("allowed_scenario_types")
    )
    max_samples = config.get("val_task_max_samples", config.get("val_max_samples"))
    samples_per_target_type = config.get(
        "val_task_samples_per_target_type", config.get("val_samples_per_target_type")
    )
    repeat_small_target_types = config.get("val_repeat_small_target_types", False)
    sampling_seed = config.get("val_sampling_seed", config.get("sampling_seed", 0))
    default_partition = config.get("val_default_target_type_partition_index", 0)
    partition_indices = config.get("val_target_type_partition_indices")
    return build_eval_dataloader(
        config,
        cache_dirs=cache_dirs,
        allowed_scenario_types=allowed_scenario_types,
        allowed_target_types=[task],
        max_samples=max_samples,
        samples_per_target_type=samples_per_target_type,
        repeat_small_target_types=repeat_small_target_types,
        sampling_seed=sampling_seed,
        default_target_type_partition_index=default_partition,
        target_type_partition_indices=partition_indices,
    )


def eval_one(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    config["use_tqdm"] = False
    if args.max_batches is not None:
        config["val_task_num_batches"] = args.max_batches
    if args.val_batch_size is not None:
        config["val_batch_size"] = args.val_batch_size
    if args.num_workers is not None:
        config["val_num_workers"] = args.num_workers

    target_tasks = _target_tasks(config)
    checkpoint = Path(args.checkpoint)
    policy = _load_policy(config, checkpoint)

    per_task: Dict[str, Dict[str, Any]] = {}
    max_batches = config.get("val_task_num_batches", config.get("val_num_batches"))
    for task in target_tasks:
        _, dataloader = _build_task_loader(config, task)
        metrics = validate(
            policy,
            dataloader,
            rank=0,
            config=config,
            progress_desc=f"{args.run_label} after_t{args.completed_task_index} {task}",
            max_batches=max_batches,
        )
        per_task[task] = metrics
        print(
            f"[{args.run_label}] after_t{args.completed_task_index} eval {task}: "
            f"ADE={metrics.get('ego_ADE', float('nan')):.4f} "
            f"FDE={metrics.get('ego_FDE', float('nan')):.4f} "
            f"n={metrics.get('sample_count', 0)}",
            flush=True,
        )

    payload = {
        "run_label": args.run_label,
        "validation_mode": "all_targets_backfill",
        "completed_task_index": args.completed_task_index,
        "current_task": args.current_task,
        "checkpoint": str(checkpoint),
        "target_tasks": target_tasks,
        "max_batches_per_task": max_batches,
        "per_task_metrics": per_task,
        "aggregate_metrics": _aggregate(per_task),
    }
    _write_json(Path(args.output_json), payload)
    torch.cuda.empty_cache()


def _load_parts(parts_dir: Path) -> List[Dict[str, Any]]:
    parts = []
    for path in sorted(parts_dir.glob("task_*.json")):
        with path.open("r", encoding="utf-8") as f:
            parts.append(json.load(f))
    parts.sort(key=lambda x: int(x["completed_task_index"]))
    return parts


def _metric(snapshot: Dict[str, Any], task: str, key: str) -> str:
    metrics = snapshot.get("per_task_metrics", {}).get(task)
    if not metrics or key not in metrics:
        return ""
    return f"{float(metrics[key]):.3f}"


def _short(task: str) -> str:
    mapping = {
        "starting_straight_traffic_light_intersection_traversal": "traffic_light_straight",
        "following_lane_with_lead": "following",
        "high_lateral_acceleration": "high_lat",
        "near_multiple_vehicles": "near_multi",
        "waiting_for_pedestrian_to_cross": "ped_wait",
        "traversing_pickup_dropoff": "pickup",
        "stationary_in_traffic": "stationary",
    }
    return mapping.get(task, task)


def _matrix_md(title: str, snapshots: List[Dict[str, Any]], tasks: List[str], key: str) -> List[str]:
    lines = [f"### {title}", ""]
    header = ["after"] + [_short(t) for t in tasks]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for snap in snapshots:
        row = [f"T{snap['completed_task_index']} {_short(snap['current_task'])}"]
        row.extend(_metric(snap, task, key) for task in tasks)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def merge(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    target_tasks = _target_tasks(config)
    snapshots = _load_parts(Path(args.parts_dir))
    if len(snapshots) != len(target_tasks):
        print(
            f"WARNING: expected {len(target_tasks)} snapshots, found {len(snapshots)} in {args.parts_dir}",
            file=sys.stderr,
        )

    payload = {
        "run_label": args.run_label,
        "validation_mode": "all_targets_backfill",
        "target_tasks": target_tasks,
        "snapshots": snapshots,
    }
    _write_json(Path(args.output_json), payload)

    lines = [
        f"# {args.run_label} CL7 All-Target Backfill",
        "",
        "Training-time continual validation used `seen_targets`, so this file is a backfilled full target matrix.",
        "Rows are task-end checkpoints; columns are all target tasks, including future tasks.",
        "",
    ]
    lines.extend(_matrix_md("ego_ADE", snapshots, target_tasks, "ego_ADE"))
    lines.extend(_matrix_md("ego_FDE", snapshots, target_tasks, "ego_FDE"))
    lines.append("### Aggregate all-target metrics")
    lines.append("")
    lines.append("| after | ADE | FDE | loss | sample_count |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for snap in snapshots:
        agg = snap.get("aggregate_metrics", {})
        lines.append(
            f"| T{snap['completed_task_index']} {_short(snap['current_task'])} | "
            f"{float(agg.get('ego_ADE', 0.0)):.3f} | "
            f"{float(agg.get('ego_FDE', 0.0)):.3f} | "
            f"{float(agg.get('loss', 0.0)):.4f} | "
            f"{int(float(agg.get('sample_count', 0.0)))} |"
        )
    lines.append("")
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare(args: argparse.Namespace) -> None:
    normal = _read_json(args.normal_json)
    pegp = _read_json(args.pegp_json)
    tasks = normal["target_tasks"]
    normal_snaps = normal["snapshots"]
    pegp_snaps = pegp["snapshots"]

    lines = [
        "# CL7 LoRA All-Target Backfill Comparison",
        "",
        "Lower ADE/FDE is better. `PEGP - Normal` below: negative means PEGP is better.",
        "",
        "## Aggregate All-Target ADE/FDE",
        "",
        "| after | Normal ADE | PEGP ADE | ΔADE | Normal FDE | PEGP FDE | ΔFDE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for n_snap, p_snap in zip(normal_snaps, pegp_snaps):
        n_agg = n_snap["aggregate_metrics"]
        p_agg = p_snap["aggregate_metrics"]
        lines.append(
            f"| T{n_snap['completed_task_index']} {_short(n_snap['current_task'])} | "
            f"{float(n_agg['ego_ADE']):.3f} | {float(p_agg['ego_ADE']):.3f} | "
            f"{float(p_agg['ego_ADE']) - float(n_agg['ego_ADE']):+.3f} | "
            f"{float(n_agg['ego_FDE']):.3f} | {float(p_agg['ego_FDE']):.3f} | "
            f"{float(p_agg['ego_FDE']) - float(n_agg['ego_FDE']):+.3f} |"
        )
    lines.append("")
    lines.append("## Final Per-Task ADE/FDE")
    lines.append("")
    lines.append("| task | Normal ADE | PEGP ADE | ΔADE | Normal FDE | PEGP FDE | ΔFDE |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    n_final = normal_snaps[-1]["per_task_metrics"]
    p_final = pegp_snaps[-1]["per_task_metrics"]
    for task in tasks:
        n = n_final[task]
        p = p_final[task]
        lines.append(
            f"| {_short(task)} | {float(n['ego_ADE']):.3f} | {float(p['ego_ADE']):.3f} | "
            f"{float(p['ego_ADE']) - float(n['ego_ADE']):+.3f} | "
            f"{float(n['ego_FDE']):.3f} | {float(p['ego_FDE']):.3f} | "
            f"{float(p['ego_FDE']) - float(n['ego_FDE']):+.3f} |"
        )
    lines.append("")
    lines.append("## ADE Δ Matrix: PEGP - Normal")
    lines.append("")
    header = ["after"] + [_short(t) for t in tasks]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * len(tasks)) + " |")
    for n_snap, p_snap in zip(normal_snaps, pegp_snaps):
        row = [f"T{n_snap['completed_task_index']} {_short(n_snap['current_task'])}"]
        for task in tasks:
            delta = float(p_snap["per_task_metrics"][task]["ego_ADE"]) - float(
                n_snap["per_task_metrics"][task]["ego_ADE"]
            )
            row.append(f"{delta:+.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_eval = sub.add_parser("eval-one")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--completed-task-index", type=int, required=True)
    p_eval.add_argument("--current-task", required=True)
    p_eval.add_argument("--run-label", required=True)
    p_eval.add_argument("--output-json", required=True)
    p_eval.add_argument("--max-batches", type=int, default=None)
    p_eval.add_argument("--val-batch-size", type=int, default=None)
    p_eval.add_argument("--num-workers", type=int, default=None)
    p_eval.set_defaults(func=eval_one)

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--config", required=True)
    p_merge.add_argument("--parts-dir", required=True)
    p_merge.add_argument("--run-label", required=True)
    p_merge.add_argument("--output-json", required=True)
    p_merge.add_argument("--output-md", required=True)
    p_merge.set_defaults(func=merge)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--normal-json", required=True)
    p_compare.add_argument("--pegp-json", required=True)
    p_compare.add_argument("--output-md", required=True)
    p_compare.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
