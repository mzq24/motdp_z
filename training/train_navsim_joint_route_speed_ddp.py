"""DDP training entrypoint for NAVSIM joint traj/route/speed diffusion."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.navsim_cached_dataset import NavSimCachedDataset, collate_fn
from model.navsim_joint_route_speed_diffusion import NavSimJointRouteSpeedDiffusion
from training.train_navsim_diffusion_ddp import (
    amp_dtype,
    autocast_cuda,
    is_main,
    make_grad_scaler,
    maybe_limit_dataset,
    move_batch_to_device,
    set_seed,
    setup_distributed,
)


METRIC_KEYS = ("l2_err_m", "traj_loss", "route_loss", "speed_loss", "route_l2_m", "speed_mae_mps")


class LossWrapper(nn.Module):
    def __init__(self, model: NavSimJointRouteSpeedDiffusion) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: Dict[str, torch.Tensor]):
        loss, info = self.model.compute_loss(batch)
        metrics = torch.tensor([float(info[key]) for key in METRIC_KEYS], dtype=loss.dtype, device=loss.device)
        return loss, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DDP train NAVSIM joint traj/route/speed diffusion.")
    parser.add_argument("--cache-dir", default="/workspace2/z_project/motdp_bev_cache_train_official4cam_officialacc_npy")
    parser.add_argument("--label-dir", default="/workspace2/z_project/motdp_navsim_labels/navtrain_official_h8_r50_sparsemask")
    parser.add_argument("--log-dir", default="/workspace2/z_project/motdp_logs/navsim_joint_route_speed")
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
    parser.add_argument("--require-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--d-ffn", type=int, default=2048)
    parser.add_argument("--p-drop-attn", type=float, default=0.1)
    parser.add_argument("--p-drop-emb", type=float, default=0.1)
    parser.add_argument("--traj-horizon", type=int, default=8)
    parser.add_argument("--traj-dim", type=int, default=2)
    parser.add_argument("--route-points", type=int, default=50)
    parser.add_argument("--speed-horizon", type=int, default=8)
    parser.add_argument("--ego-input-dim", type=int, default=8)
    parser.add_argument("--ego-history-frames", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", default="cosine")
    parser.add_argument("--prediction-type", default="sample")
    parser.add_argument("--route-loss-weight", type=float, default=1.0)
    parser.add_argument("--speed-loss-weight", type=float, default=0.5)
    parser.add_argument("--use-raw-bev-feature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-bev-dim", type=int, default=512)
    parser.add_argument(
        "--traj-stats-path",
        default="/workspace2/z_project/motdp_navsim_norm_stats/navtrain_official_h8_r50_20260518_sparsemask/navtrain_official_h8_r50_sparsemask_abs_stats.npz",
    )
    parser.add_argument(
        "--route-stats-path",
        default="/workspace2/z_project/motdp_navsim_norm_stats/navtrain_official_h8_r50_20260518_sparsemask/navtrain_official_h8_r50_sparsemask_route_abs_stats.npz",
    )
    parser.add_argument(
        "--speed-stats-path",
        default="/workspace2/z_project/motdp_navsim_norm_stats/navtrain_official_h8_r50_20260518_sparsemask/navtrain_official_h8_r50_sparsemask_speed_profile_stats.npz",
    )
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser


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


def reduce_vector(values, device, world_size: int) -> np.ndarray:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().numpy()


def read_npz_pair(path: str, mean_keys: tuple[str, ...], std_keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        mean_key = next((key for key in mean_keys if key in data.files), None)
        std_key = next((key for key in std_keys if key in data.files), None)
        if mean_key is None or std_key is None:
            raise KeyError(f"{path} missing expected mean/std keys. keys={data.files}")
        return data[mean_key].astype(np.float32), data[std_key].astype(np.float32)


def load_normalization_stats(raw_model: NavSimJointRouteSpeedDiffusion, args, device) -> dict[str, Any]:
    traj_mean, traj_std = read_npz_pair(args.traj_stats_path, ("abs_mean", "traj_mean"), ("abs_std", "traj_std"))
    route_mean, route_std = read_npz_pair(args.route_stats_path, ("route_abs_mean", "abs_mean"), ("route_abs_std", "abs_std"))
    speed_mean, speed_std = read_npz_pair(args.speed_stats_path, ("speed_profile_mean", "speed_mean"), ("speed_profile_std", "speed_std"))
    raw_model.set_normalization_stats(
        torch.as_tensor(traj_mean, device=device),
        torch.as_tensor(traj_std, device=device),
        torch.as_tensor(route_mean, device=device),
        torch.as_tensor(route_std, device=device),
        torch.as_tensor(speed_mean, device=device),
        torch.as_tensor(speed_std, device=device),
    )
    return {
        "traj": [list(traj_mean.shape), list(traj_std.shape)],
        "route": [list(route_mean.shape), list(route_std.shape)],
        "speed": [list(speed_mean.shape), list(speed_std.shape)],
    }


def validate(raw_model, val_loader, device, args, dtype, world_size: int) -> dict[str, float]:
    raw_model.eval()
    amp_enabled = bool(args.use_amp and torch.cuda.is_available())
    sums = np.zeros(1 + len(METRIC_KEYS), dtype=np.float64)
    iterator = tqdm(val_loader, desc="Validating", leave=False) if dist.is_initialized() and dist.get_rank() == 0 else val_loader
    with torch.no_grad():
        for batch in iterator:
            batch = move_batch_to_device(batch, device)
            with autocast_cuda(amp_enabled, dtype):
                loss, info = raw_model.compute_loss(batch)
            sums += np.asarray([float(loss.item()), *[float(info[key]) for key in METRIC_KEYS]], dtype=np.float64)
    reduced = reduce_vector([*sums.tolist(), len(val_loader)], device, world_size)
    denom = max(1.0, reduced[-1])
    return {
        "val_loss": reduced[0] / denom,
        "val_l2": reduced[1] / denom,
        "val_traj_loss": reduced[2] / denom,
        "val_route_loss": reduced[3] / denom,
        "val_speed_loss": reduced[4] / denom,
        "val_route_l2": reduced[5] / denom,
        "val_speed_mae": reduced[6] / denom,
    }


def save_checkpoint(path: str, model, optimizer, scheduler, scaler, args, epoch, global_step, train_metrics, val_metrics, world_size):
    cfg = vars(args).copy()
    cfg.update(
        {
            "model_type": "navsim_joint_route_speed_diffusion",
            "state_tokens": 0,
            "bev_conditioning": "raw_bev_feature_plus_top_down_grid",
            "world_size": world_size,
            "global_batch_size": args.batch_size * world_size,
        }
    )
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "traj_mean": model.traj_mean.detach().cpu().numpy(),
            "traj_std": model.traj_std.detach().cpu().numpy(),
            "route_mean": model.route_mean.detach().cpu().numpy(),
            "route_std": model.route_std.detach().cpu().numpy(),
            "speed_mean": model.speed_mean.detach().cpu().numpy(),
            "speed_std": model.speed_std.detach().cpu().numpy(),
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
    main_rank = is_main(rank)

    if args.token_filter_file is None:
        candidate = Path(args.cache_dir) / "navtrain_official_tokens.txt"
        args.token_filter_file = str(candidate) if candidate.is_file() else None

    if main_rank:
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

        log(f"NAVSIM joint DDP training started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"rank={rank} local_rank={local_rank} world_size={world_size} device={device}")
        log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        log(f"Config: {vars(args)}")
        log(f"Global batch size: {args.batch_size * world_size}")
    else:
        log = lambda msg: None
        log_json = lambda row: None

    dataset_kwargs = dict(
        cache_dir=args.cache_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        preload=False,
        token_filter_file=args.token_filter_file,
        dedupe_tokens=args.dedupe_tokens,
        load_mode=args.load_mode,
        ego_input_dim=args.ego_input_dim,
        label_dir=args.label_dir,
        require_labels=args.require_labels,
    )
    train_dataset = NavSimCachedDataset(split="train", **dataset_kwargs)
    val_dataset = NavSimCachedDataset(split="val", **dataset_kwargs)
    train_dataset = maybe_limit_dataset(train_dataset, args.max_train_samples)
    val_dataset = maybe_limit_dataset(val_dataset, args.max_val_samples)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed, drop_last=False) if world_size > 1 else None
    train_loader = make_loader(train_dataset, args, train_sampler, shuffle=(train_sampler is None), drop_last=True)
    val_loader = make_loader(val_dataset, args, val_sampler, shuffle=False, drop_last=False)

    raw_model = NavSimJointRouteSpeedDiffusion(
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        d_ffn=args.d_ffn,
        p_drop_attn=args.p_drop_attn,
        p_drop_emb=args.p_drop_emb,
        traj_horizon=args.traj_horizon,
        traj_dim=args.traj_dim,
        route_points=args.route_points,
        speed_horizon=args.speed_horizon,
        ego_input_dim=args.ego_input_dim,
        ego_history_frames=args.ego_history_frames,
        prediction_type=args.prediction_type,
        num_inference_steps=args.num_inference_steps,
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        route_loss_weight=args.route_loss_weight,
        speed_loss_weight=args.speed_loss_weight,
        use_raw_bev_feature=args.use_raw_bev_feature,
        raw_bev_dim=args.raw_bev_dim,
    ).to(device)
    stats_info = load_normalization_stats(raw_model, args, device)
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
    scaler = make_grad_scaler(bool(args.use_amp and args.amp_dtype == "fp16" and torch.cuda.is_available()))
    dtype = amp_dtype(args.amp_dtype)
    amp_enabled = bool(args.use_amp and torch.cuda.is_available())

    log(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    log(f"Steps per epoch per rank: {len(train_loader)}")
    log(f"Stats: {stats_info}")
    log(f"Params: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.1f}M")

    best_val_loss = float("inf")
    best_val_l2 = float("inf")
    last_val_metrics = {"val_loss": float("inf"), "val_l2": float("inf")}
    global_step = 0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        raw_model.train()
        local = np.zeros(1 + len(METRIC_KEYS), dtype=np.float64)
        t0 = time.time()
        iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True) if main_rank else train_loader
        for batch_idx, batch in enumerate(iterator):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_cuda(amp_enabled, dtype):
                loss, metrics = model(batch)
            if not torch.isfinite(loss):
                if main_rank:
                    print(f"Warning: non-finite loss at batch {batch_idx}, skipping", flush=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if main_rank:
                    print(f"Warning: non-finite grad norm at batch {batch_idx}, skipping", flush=True)
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            metric_values = metrics.detach().float().cpu().numpy()
            local += np.asarray([float(loss.item()), *metric_values.tolist()], dtype=np.float64)
            global_step += 1
            if main_rank:
                iterator.set_postfix(
                    loss=f"{loss.item():.4f}",
                    l2=f"{metric_values[0]:.3f}m",
                    tr=f"{metric_values[1]:.3f}",
                    rt=f"{metric_values[2]:.3f}",
                    sp=f"{metric_values[3]:.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        reduced = reduce_vector([*local.tolist(), len(train_loader)], device, world_size)
        denom = max(1.0, reduced[-1])
        train_metrics = {
            "train_loss": reduced[0] / denom,
            "train_l2": reduced[1] / denom,
            "train_traj_loss": reduced[2] / denom,
            "train_route_loss": reduced[3] / denom,
            "train_speed_loss": reduced[4] / denom,
            "train_route_l2": reduced[5] / denom,
            "train_speed_mae": reduced[6] / denom,
        }
        epoch_time = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        if main_rank:
            log(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"loss={train_metrics['train_loss']:.4f} "
                f"l2={train_metrics['train_l2']:.3f}m "
                f"traj={train_metrics['train_traj_loss']:.4f} "
                f"route={train_metrics['train_route_loss']:.4f} "
                f"speed={train_metrics['train_speed_loss']:.4f} "
                f"lr={lr_now:.2e} time={epoch_time:.0f}s"
            )
            log_json({"epoch": epoch + 1, "global_step": global_step, **train_metrics, "lr": lr_now, "epoch_time_sec": round(epoch_time, 3)})

        if args.val_every_epochs > 0 and (epoch + 1) % args.val_every_epochs == 0:
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            last_val_metrics = validate(raw_model, val_loader, device, args, dtype, world_size)
            if main_rank:
                log(
                    f"  Val epoch {epoch + 1}: "
                    f"loss={last_val_metrics['val_loss']:.4f} "
                    f"l2={last_val_metrics['val_l2']:.3f}m "
                    f"traj={last_val_metrics['val_traj_loss']:.4f} "
                    f"route={last_val_metrics['val_route_loss']:.4f} "
                    f"speed={last_val_metrics['val_speed_loss']:.4f}"
                )
                log_json({"epoch": epoch + 1, "global_step": global_step, **last_val_metrics})
                if last_val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = last_val_metrics["val_loss"]
                    best_val_l2 = last_val_metrics["val_l2"]
                    best_path = os.path.join(args.log_dir, "best_model.pt")
                    save_checkpoint(best_path, raw_model, optimizer, scheduler, scaler, args, epoch, global_step, train_metrics, last_val_metrics, world_size)
                    log(f"  Saved best model: {best_path}")

        if main_rank and args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
            ckpt_path = os.path.join(args.log_dir, f"model_epoch{epoch + 1}.pt")
            save_checkpoint(ckpt_path, raw_model, optimizer, scheduler, scaler, args, epoch, global_step, train_metrics, last_val_metrics, world_size)
            log(f"  Saved checkpoint: {ckpt_path}")
            prune_checkpoints(args.log_dir, args.max_keep_ckpts)

    if main_rank:
        log(f"Training finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Best val loss: {best_val_loss:.4f} | best val l2: {best_val_l2:.3f}m")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
