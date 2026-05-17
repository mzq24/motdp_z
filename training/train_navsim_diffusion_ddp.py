"""DDP training entrypoint for NavSim cached-BEV diffusion."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast as torch_autocast

    def autocast_cuda(enabled: bool, dtype):
        return torch_autocast("cuda", enabled=enabled, dtype=dtype)

    def make_grad_scaler(enabled: bool):
        return GradScaler("cuda", enabled=enabled)

except ImportError:
    from torch.cuda.amp import GradScaler, autocast as torch_autocast

    def autocast_cuda(enabled: bool, dtype):
        return torch_autocast(enabled=enabled, dtype=dtype)

    def make_grad_scaler(enabled: bool):
        return GradScaler(enabled=enabled)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.navsim_cached_dataset import NavSimCachedDataset, collate_fn
from model.navsim_simple_diffusion import NavSimSimpleDiffusion


class LossWrapper(nn.Module):
    def __init__(self, model: NavSimSimpleDiffusion) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: Dict[str, torch.Tensor]):
        loss, info = self.model.compute_loss(batch)
        l2 = torch.as_tensor(float(info["l2_err_m"]), device=loss.device)
        return loss, l2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DDP train NavSim cached-BEV diffusion policy.")
    parser.add_argument("--cache-dir", default="/workspace2/z_project/motdp_bev_cache_train_official4cam_npy")
    parser.add_argument("--log-dir", default="/workspace2/z_project/motdp_logs/navsim_official4cam_ddp")
    parser.add_argument("--token-filter-file", default=None)
    parser.add_argument("--load-mode", choices=("memmap", "ram"), default="memmap")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-GPU batch size.")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--lr-final", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--val-every-epochs", type=int, default=5)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--max-keep-ckpts", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dedupe-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--d-ffn", type=int, default=2048)
    parser.add_argument("--p-drop-attn", type=float, default=0.1)
    parser.add_argument("--p-drop-emb", type=float, default=0.1)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", default="cosine")
    parser.add_argument("--prediction-type", default="sample")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, device


def is_main(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_limit_dataset(dataset, limit: int | None):
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def amp_dtype(name: str):
    return torch.bfloat16 if name == "bf16" else torch.float16


def make_loader(dataset, args, sampler, shuffle: bool, drop_last: bool) -> DataLoader:
    kwargs = dict(
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def reduce_triplet(values, device, world_size: int):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().numpy().tolist()


def validate(raw_model, val_loader, device, args, dtype, world_size: int):
    raw_model.eval()
    amp_enabled = bool(args.use_amp and torch.cuda.is_available())
    loss_sum, l2_sum, count = 0.0, 0.0, 0
    if dist.is_initialized() and dist.get_rank() == 0:
        iterator = tqdm(val_loader, desc="Validating", leave=False)
    else:
        iterator = val_loader
    with torch.no_grad():
        for batch in iterator:
            batch = move_batch_to_device(batch, device)
            with autocast_cuda(amp_enabled, dtype):
                loss, info = raw_model.compute_loss(batch)
            loss_sum += float(loss.item())
            l2_sum += float(info["l2_err_m"])
            count += 1
    raw_model.train()
    loss_sum, l2_sum, count = reduce_triplet([loss_sum, l2_sum, count], device, world_size)
    return loss_sum / max(1.0, count), l2_sum / max(1.0, count)


def save_checkpoint(path: str, model, optimizer, scheduler, scaler, args, epoch, global_step, train_loss, val_loss, val_l2, world_size):
    cfg = vars(args).copy()
    cfg["world_size"] = world_size
    cfg["global_batch_size"] = args.batch_size * world_size
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_l2": val_l2,
            "traj_mean": model.traj_mean.detach().cpu().numpy(),
            "traj_std": model.traj_std.detach().cpu().numpy(),
            "config": cfg,
        },
        path,
    )


def prune_checkpoints(log_dir: str, max_keep: int) -> None:
    if max_keep <= 0:
        return
    ckpts = sorted(glob.glob(os.path.join(log_dir, "model_epoch*.pt")), key=os.path.getmtime)
    while len(ckpts) > max_keep:
        old = ckpts.pop(0)
        os.remove(old)
        print(f"  Removed old checkpoint: {old}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    set_seed(args.seed, rank)
    main = is_main(rank)

    if args.token_filter_file is None:
        candidate = Path(args.cache_dir) / "navtrain_official_tokens.txt"
        args.token_filter_file = str(candidate) if candidate.is_file() else None

    if main:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file = os.path.join(args.log_dir, "train_log.txt")
        metrics_file = os.path.join(args.log_dir, "metrics.jsonl")

        def log(msg: str) -> None:
            print(msg, flush=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

        def log_json(row: Dict[str, Any]) -> None:
            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

        log(f"DDP training started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"rank={rank} local_rank={local_rank} world_size={world_size} device={device}")
        log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        log(f"Config: {vars(args)}")
        log(f"Global batch size: {args.batch_size * world_size}")
    else:
        log = lambda msg: None
        log_json = lambda row: None

    train_dataset = NavSimCachedDataset(
        cache_dir=args.cache_dir,
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        preload=False,
        token_filter_file=args.token_filter_file,
        dedupe_tokens=args.dedupe_tokens,
        load_mode=args.load_mode,
    )
    val_dataset = NavSimCachedDataset(
        cache_dir=args.cache_dir,
        split="val",
        val_ratio=args.val_ratio,
        seed=args.seed,
        preload=False,
        token_filter_file=args.token_filter_file,
        dedupe_tokens=args.dedupe_tokens,
        load_mode=args.load_mode,
    )
    stats_dataset = train_dataset
    train_dataset = maybe_limit_dataset(train_dataset, args.max_train_samples)
    val_dataset = maybe_limit_dataset(val_dataset, args.max_val_samples)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed, drop_last=False) if world_size > 1 else None
    train_loader = make_loader(train_dataset, args, train_sampler, shuffle=(train_sampler is None), drop_last=True)
    val_loader = make_loader(val_dataset, args, val_sampler, shuffle=False, drop_last=False)

    raw_model = NavSimSimpleDiffusion(
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        d_ffn=args.d_ffn,
        p_drop_attn=args.p_drop_attn,
        p_drop_emb=args.p_drop_emb,
        traj_horizon=8,
        traj_dim=2,
        ego_input_dim=14,
        ego_history_frames=4,
        prediction_type=args.prediction_type,
        num_inference_steps=args.num_inference_steps,
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
    ).to(device)
    mean, std = stats_dataset.get_traj_stats()
    raw_model.traj_mean.copy_(torch.tensor(mean, device=device))
    raw_model.traj_std.copy_(torch.tensor(std, device=device))
    wrapper = LossWrapper(raw_model).to(device)
    model = DDP(wrapper, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if world_size > 1 else wrapper

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return 0.1 + 0.9 * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        min_lr_ratio = args.lr_final / args.lr
        return np.cos(progress * np.pi / 2) * (1.0 - min_lr_ratio) + min_lr_ratio

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler_enabled = bool(args.use_amp and args.amp_dtype == "fp16" and torch.cuda.is_available())
    scaler = make_grad_scaler(scaler_enabled)
    dtype = amp_dtype(args.amp_dtype)
    amp_enabled = bool(args.use_amp and torch.cuda.is_available())

    log(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    log(f"Steps per epoch per rank: {len(train_loader)}")
    log(f"Traj stats: mean={mean}, std={std}")
    log(f"Params: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.1f}M")

    best_val_loss = float("inf")
    best_val_l2 = float("inf")
    last_val_loss = float("inf")
    last_val_l2 = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        raw_model.train()
        local_loss, local_l2, local_count = 0.0, 0.0, 0
        t0 = time.time()
        iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True) if main else train_loader
        for batch_idx, batch in enumerate(iterator):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_cuda(amp_enabled, dtype):
                loss, l2 = model(batch)
            if not torch.isfinite(loss):
                if main:
                    print(f"Warning: non-finite loss at batch {batch_idx}, skipping", flush=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if main:
                    print(f"Warning: non-finite grad norm at batch {batch_idx}, skipping", flush=True)
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            local_loss += float(loss.item())
            local_l2 += float(l2.item())
            local_count += 1
            global_step += 1
            if main:
                iterator.set_postfix(loss=f"{loss.item():.4f}", l2=f"{l2.item():.3f}m", lr=f"{scheduler.get_last_lr()[0]:.2e}", grad=f"{float(grad_norm):.2f}")

        loss_sum, l2_sum, count = reduce_triplet([local_loss, local_l2, local_count], device, world_size)
        train_loss = loss_sum / max(1.0, count)
        train_l2 = l2_sum / max(1.0, count)
        epoch_time = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        if main:
            log(f"Epoch {epoch + 1}/{args.epochs} | loss={train_loss:.4f} l2={train_l2:.3f}m lr={lr_now:.2e} time={epoch_time:.0f}s")
            log_json({"epoch": epoch + 1, "global_step": global_step, "train_loss": train_loss, "train_l2": train_l2, "lr": lr_now, "epoch_time_sec": round(epoch_time, 3)})

        if args.val_every_epochs > 0 and (epoch + 1) % args.val_every_epochs == 0:
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            last_val_loss, last_val_l2 = validate(raw_model, val_loader, device, args, dtype, world_size)
            if main:
                log(f"  Val epoch {epoch + 1}: loss={last_val_loss:.4f} l2={last_val_l2:.3f}m")
                log_json({"epoch": epoch + 1, "global_step": global_step, "val_loss": last_val_loss, "val_l2": last_val_l2})
                if last_val_loss < best_val_loss:
                    best_val_loss = last_val_loss
                    best_val_l2 = last_val_l2
                    best_path = os.path.join(args.log_dir, "best_model.pt")
                    save_checkpoint(best_path, raw_model, optimizer, scheduler, scaler, args, epoch, global_step, train_loss, last_val_loss, last_val_l2, world_size)
                    log(f"  Saved best model: {best_path}")

        if main and args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
            ckpt_path = os.path.join(args.log_dir, f"model_epoch{epoch + 1}.pt")
            save_checkpoint(ckpt_path, raw_model, optimizer, scheduler, scaler, args, epoch, global_step, train_loss, last_val_loss, last_val_l2, world_size)
            log(f"  Saved checkpoint: {ckpt_path}")
            prune_checkpoints(args.log_dir, args.max_keep_ckpts)

    if main:
        log(f"Training finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Best val loss: {best_val_loss:.4f} | best val l2: {best_val_l2:.3f}m")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
