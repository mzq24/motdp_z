#!/usr/bin/env python3
"""Offline open-loop evaluation for the cached NavSim diffusion planner."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.navsim_cached_dataset import NavSimCachedDataset, collate_fn
from model.navsim_simple_diffusion import NavSimSimpleDiffusion


DEFAULT_CKPT = "/workspace2/z_project/motdp_logs/navsim_simple_official_lr1e4_e60_b64/best_model.pt"
DEFAULT_CACHE = "/workspace2/z_project/motdp_bev_cache_official_npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NavSim cached BEV diffusion model on held-out split.")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--token-filter-file", default=None)
    parser.add_argument("--dedupe-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-mode", default="auto", choices=["auto", "memmap", "ram"])
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def resolve_token_filter(cache_dir: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    candidate = Path(cache_dir) / "navtrain_official_tokens.txt"
    return str(candidate) if candidate.is_file() else None


def build_model(config: Dict[str, Any], device: torch.device) -> NavSimSimpleDiffusion:
    return NavSimSimpleDiffusion(
        d_model=config.get("d_model", 512),
        n_head=config.get("n_head", 8),
        n_layer=config.get("n_layer", 4),
        d_ffn=config.get("d_ffn", 2048),
        p_drop_attn=config.get("p_drop_attn", 0.1),
        p_drop_emb=config.get("p_drop_emb", 0.1),
        traj_horizon=config.get("traj_horizon", 8),
        traj_dim=config.get("traj_dim", 2),
        ego_input_dim=config.get("ego_input_dim", 14),
        ego_history_frames=config.get("ego_history_frames", 4),
        prediction_type=config.get("prediction_type", "sample"),
        num_inference_steps=config.get("num_inference_steps", 10),
        num_train_timesteps=config.get("num_train_timesteps", 1000),
        beta_schedule=config.get("beta_schedule", "cosine"),
    ).to(device)


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = build_model(config, device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    if isinstance(ckpt, dict):
        if "traj_mean" in ckpt:
            model.traj_mean.copy_(torch.as_tensor(ckpt["traj_mean"], device=device, dtype=torch.float32))
        if "traj_std" in ckpt:
            model.traj_std.copy_(torch.as_tensor(ckpt["traj_std"], device=device, dtype=torch.float32))
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}


def make_loader(args: argparse.Namespace) -> DataLoader:
    dataset = NavSimCachedDataset(
        cache_dir=args.cache_dir,
        split=args.split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        preload=(args.load_mode != "memmap"),
        token_filter_file=resolve_token_filter(args.cache_dir, args.token_filter_file),
        dedupe_tokens=args.dedupe_tokens,
        load_mode=args.load_mode,
    )
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    kwargs: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_fn,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def amp_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, ckpt = load_checkpoint(args.checkpoint, device)
    if args.num_inference_steps is not None:
        model.num_inference_steps = args.num_inference_steps
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {ckpt.get('epoch')} val_loss={ckpt.get('val_loss')}")
    print(f"Sampler steps: {model.num_inference_steps}")
    print(f"Traj stats: mean={model.traj_mean.detach().cpu().numpy()} std={model.traj_std.detach().cpu().numpy()}")

    loader = make_loader(args)
    total_samples = 0
    total_ade = 0.0
    total_fde = 0.0
    total_max_l2 = 0.0
    total_target_norm = 0.0
    total_pred_norm = 0.0
    step_l2_sum = None
    t0 = time.time()

    use_amp = bool(args.use_amp and device.type == "cuda" and args.amp_dtype != "fp32")
    pbar = tqdm(loader, desc=f"Evaluating {args.split}", total=len(loader))
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            bev_grid = batch["bev_grid"].to(device, non_blocking=True).float()
            ego_status = batch["ego_status"].to(device, non_blocking=True).float()
            target = batch["trajectory"].to(device, non_blocking=True).float()

            with torch.autocast(device_type="cuda", dtype=amp_dtype(args.amp_dtype), enabled=use_amp):
                pred = model.sample(bev_grid, ego_status, return_trajectory=True)
            pred = pred.float()

            l2 = torch.linalg.norm(pred - target, dim=-1)
            ade = l2.mean(dim=1)
            fde = l2[:, -1]
            max_l2 = l2.max(dim=1).values
            n = target.shape[0]

            if step_l2_sum is None:
                step_l2_sum = torch.zeros(l2.shape[1], dtype=torch.float64)
            step_l2_sum += l2.detach().double().sum(dim=0).cpu()
            total_samples += n
            total_ade += float(ade.sum().item())
            total_fde += float(fde.sum().item())
            total_max_l2 += float(max_l2.sum().item())
            total_target_norm += float(torch.linalg.norm(target, dim=-1).mean(dim=1).sum().item())
            total_pred_norm += float(torch.linalg.norm(pred, dim=-1).mean(dim=1).sum().item())
            pbar.set_postfix(
                ade=f"{total_ade / max(1, total_samples):.3f}",
                fde=f"{total_fde / max(1, total_samples):.3f}",
            )

    if total_samples == 0:
        raise RuntimeError("No samples evaluated")
    per_step_l2 = (step_l2_sum / total_samples).tolist() if step_l2_sum is not None else []
    result = {
        "checkpoint": args.checkpoint,
        "cache_dir": args.cache_dir,
        "split": args.split,
        "num_samples": int(total_samples),
        "num_batches": int(math.ceil(total_samples / args.batch_size)),
        "batch_size": int(args.batch_size),
        "num_inference_steps": int(model.num_inference_steps),
        "ade_m": total_ade / total_samples,
        "fde_m": total_fde / total_samples,
        "max_l2_m": total_max_l2 / total_samples,
        "target_path_norm_m": total_target_norm / total_samples,
        "pred_path_norm_m": total_pred_norm / total_samples,
        "per_step_l2_m": per_step_l2,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
