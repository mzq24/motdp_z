"""
CLI training entrypoint for NavSim cached-BEV diffusion.

The script mirrors the useful ergonomics of train_carla_bev.py: tqdm progress,
periodic validation, periodic checkpoints, best checkpoint tracking, and clear
per-epoch resource/logging output. It prefers the consolidated NPY cache because
legacy NPZ shards are very slow to reopen.
"""

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
from torch.utils.data import DataLoader, Subset
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


DEFAULTS: Dict[str, Any] = dict(
    cache_dir="/workspace2/z_project/motdp_bev_cache_official_npy",
    fallback_cache_dir="/workspace2/z_project/motdp_bev_cache_official_final",
    log_dir="/workspace2/z_project/motdp_logs/navsim_simple_official",
    token_filter_file=None,
    load_mode="auto",
    batch_size=64,
    epochs=60,
    lr=1e-4,
    weight_decay=1e-4,
    warmup_epochs=3,
    lr_final=1e-6,
    max_grad_norm=1.0,
    val_ratio=0.05,
    val_every_epochs=5,
    save_every_epochs=10,
    max_keep_ckpts=5,
    num_workers=4,
    prefetch_factor=2,
    seed=42,
    dedupe_tokens=True,
    max_train_samples=None,
    max_val_samples=None,
    d_model=512,
    n_head=8,
    n_layer=4,
    d_ffn=2048,
    p_drop_attn=0.1,
    p_drop_emb=0.1,
    traj_horizon=8,
    traj_dim=2,
    ego_input_dim=8,
    ego_history_frames=4,
    prediction_type="sample",
    num_inference_steps=10,
    num_train_timesteps=1000,
    beta_schedule="cosine",
    use_amp=True,
    amp_dtype="bf16",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train NavSim cached-BEV diffusion policy.")
    parser.add_argument("--cache-dir", default=DEFAULTS["cache_dir"])
    parser.add_argument("--fallback-cache-dir", default=DEFAULTS["fallback_cache_dir"])
    parser.add_argument("--log-dir", default=DEFAULTS["log_dir"])
    parser.add_argument("--token-filter-file", default=DEFAULTS["token_filter_file"])
    parser.add_argument("--load-mode", choices=("auto", "memmap", "ram"), default=DEFAULTS["load_mode"])

    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULTS["warmup_epochs"])
    parser.add_argument("--lr-final", type=float, default=DEFAULTS["lr_final"])
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULTS["max_grad_norm"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULTS["val_ratio"])
    parser.add_argument("--val-every-epochs", type=int, default=DEFAULTS["val_every_epochs"])
    parser.add_argument("--save-every-epochs", type=int, default=DEFAULTS["save_every_epochs"])
    parser.add_argument("--max-keep-ckpts", type=int, default=DEFAULTS["max_keep_ckpts"])
    parser.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--prefetch-factor", type=int, default=DEFAULTS["prefetch_factor"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--dedupe-tokens", action=argparse.BooleanOptionalAction, default=DEFAULTS["dedupe_tokens"])
    parser.add_argument("--max-train-samples", type=int, default=DEFAULTS["max_train_samples"])
    parser.add_argument("--max-val-samples", type=int, default=DEFAULTS["max_val_samples"])

    parser.add_argument("--d-model", type=int, default=DEFAULTS["d_model"])
    parser.add_argument("--n-head", type=int, default=DEFAULTS["n_head"])
    parser.add_argument("--n-layer", type=int, default=DEFAULTS["n_layer"])
    parser.add_argument("--d-ffn", type=int, default=DEFAULTS["d_ffn"])
    parser.add_argument("--p-drop-attn", type=float, default=DEFAULTS["p_drop_attn"])
    parser.add_argument("--p-drop-emb", type=float, default=DEFAULTS["p_drop_emb"])
    parser.add_argument("--traj-horizon", type=int, default=DEFAULTS["traj_horizon"])
    parser.add_argument("--traj-dim", type=int, default=DEFAULTS["traj_dim"])
    parser.add_argument("--ego-input-dim", type=int, default=DEFAULTS["ego_input_dim"])
    parser.add_argument("--ego-history-frames", type=int, default=DEFAULTS["ego_history_frames"])
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULTS["num_inference_steps"])
    parser.add_argument("--num-train-timesteps", type=int, default=DEFAULTS["num_train_timesteps"])
    parser.add_argument("--beta-schedule", default=DEFAULTS["beta_schedule"])
    parser.add_argument("--prediction-type", default=DEFAULTS["prediction_type"])

    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=DEFAULTS["use_amp"])
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default=DEFAULTS["amp_dtype"])
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_limit_dataset(dataset, limit: int | None):
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def make_loader(dataset, cfg: Dict[str, Any], shuffle: bool) -> DataLoader:
    kwargs = dict(
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    if cfg["num_workers"] > 0:
        kwargs["prefetch_factor"] = cfg["prefetch_factor"]
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def amp_dtype(cfg: Dict[str, Any]):
    return torch.bfloat16 if cfg["amp_dtype"] == "bf16" else torch.float16


def resolve_cache_dir(cfg: Dict[str, Any]) -> None:
    cache_dir = Path(cfg["cache_dir"])
    if not cache_dir.exists() and cfg.get("fallback_cache_dir"):
        fallback = Path(cfg["fallback_cache_dir"])
        if fallback.exists():
            print(f"Cache dir not found: {cache_dir}; falling back to {fallback}")
            cfg["cache_dir"] = str(fallback)
            cache_dir = fallback
    if cfg.get("token_filter_file") is None:
        candidate = cache_dir / "navtrain_official_tokens.txt"
        cfg["token_filter_file"] = str(candidate) if candidate.is_file() else None


def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def save_checkpoint(path: str, model, optimizer, scheduler, scaler, cfg, epoch, global_step, train_loss, val_loss, val_l2):
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
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
        print(f"  Removed old checkpoint: {old}")


def print_resource_snapshot(prefix: str = "") -> None:
    try:
        import psutil

        mem = psutil.virtual_memory()
        print(
            f"{prefix}RAM: {mem.used / 1e9:.1f}GB used / {mem.total / 1e9:.1f}GB total "
            f"({mem.percent}%), available={mem.available / 1e9:.1f}GB, "
            f"cached={getattr(mem, 'cached', 0) / 1e9:.1f}GB"
        )
    except Exception:
        pass
    if torch.cuda.is_available():
        for gi in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(gi) / 1e9
            reserved = torch.cuda.memory_reserved(gi) / 1e9
            print(f"{prefix}GPU{gi}: {alloc:.2f}GB alloc / {reserved:.2f}GB reserved")


def validate(model, val_loader, device, cfg, autocast_dtype):
    model.eval()
    amp_enabled = bool(cfg["use_amp"] and torch.cuda.is_available())
    total_loss = 0.0
    total_l2 = 0.0
    count = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False, total=len(val_loader))
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            with autocast_cuda(amp_enabled, autocast_dtype):
                loss, info = model.compute_loss(batch)
            total_loss += float(loss.item())
            total_l2 += float(info["l2_err_m"])
            count += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", l2=f"{float(info['l2_err_m']):.3f}")
    model.train()
    return total_loss / max(1, count), total_l2 / max(1, count)


def train(cfg: Dict[str, Any]) -> None:
    resolve_cache_dir(cfg)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    t_load = time.time()
    print("Loading train dataset...")
    train_dataset = NavSimCachedDataset(
        cache_dir=cfg["cache_dir"],
        split="train",
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
        preload=(cfg["load_mode"] != "memmap"),
        token_filter_file=cfg["token_filter_file"],
        dedupe_tokens=cfg["dedupe_tokens"],
        load_mode=cfg["load_mode"],
        ego_input_dim=cfg["ego_input_dim"],
    )
    print("Loading val dataset...")
    val_dataset = NavSimCachedDataset(
        cache_dir=cfg["cache_dir"],
        split="val",
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
        preload=(cfg["load_mode"] != "memmap"),
        token_filter_file=cfg["token_filter_file"],
        dedupe_tokens=cfg["dedupe_tokens"],
        load_mode=cfg["load_mode"],
        ego_input_dim=cfg["ego_input_dim"],
    )
    print(f"Dataset init finished in {(time.time() - t_load) / 60:.1f}min")

    stats_dataset = train_dataset
    train_dataset = maybe_limit_dataset(train_dataset, cfg["max_train_samples"])
    val_dataset = maybe_limit_dataset(val_dataset, cfg["max_val_samples"])
    train_loader = make_loader(train_dataset, cfg, shuffle=True)
    val_loader = make_loader(val_dataset, cfg, shuffle=False)

    print("Creating model...")
    model = NavSimSimpleDiffusion(
        d_model=cfg["d_model"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        d_ffn=cfg["d_ffn"],
        p_drop_attn=cfg["p_drop_attn"],
        p_drop_emb=cfg["p_drop_emb"],
        traj_horizon=cfg["traj_horizon"],
        traj_dim=cfg["traj_dim"],
        ego_input_dim=cfg["ego_input_dim"],
        ego_history_frames=cfg["ego_history_frames"],
        prediction_type=cfg["prediction_type"],
        num_inference_steps=cfg["num_inference_steps"],
        num_train_timesteps=cfg["num_train_timesteps"],
        beta_schedule=cfg["beta_schedule"],
    ).to(device)

    mean, std = stats_dataset.get_traj_stats()
    model.traj_mean.copy_(torch.tensor(mean, device=device))
    model.traj_std.copy_(torch.tensor(std, device=device))
    print(f"Traj stats: mean={mean}, std={std}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = len(train_loader) * cfg["warmup_epochs"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return 0.1 + 0.9 * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        min_lr_ratio = cfg["lr_final"] / cfg["lr"]
        return np.cos(progress * np.pi / 2) * (1.0 - min_lr_ratio) + min_lr_ratio

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler_enabled = bool(cfg["use_amp"] and cfg["amp_dtype"] == "fp16" and torch.cuda.is_available())
    scaler = make_grad_scaler(scaler_enabled)
    autocast_dtype = amp_dtype(cfg)
    amp_enabled = bool(cfg["use_amp"] and torch.cuda.is_available())

    os.makedirs(cfg["log_dir"], exist_ok=True)
    log_file = os.path.join(cfg["log_dir"], "train_log.txt")
    metrics_file = os.path.join(cfg["log_dir"], "metrics.jsonl")

    def log(msg: str) -> None:
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def log_json(row: Dict[str, Any]) -> None:
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    log(f"Training started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Config: {cfg}")
    log(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    global_step = 0
    best_val_loss = float("inf")
    best_val_l2 = float("inf")
    last_val_loss = float("inf")
    last_val_l2 = float("inf")

    for epoch in range(cfg["epochs"]):
        print_resource_snapshot(prefix=f"\n[Epoch {epoch + 1}] ")
        model.train()
        epoch_loss = 0.0
        epoch_l2 = 0.0
        seen_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']}", leave=True, total=len(train_loader))
        for batch_idx, batch in enumerate(pbar):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_cuda(amp_enabled, autocast_dtype):
                loss, info = model.compute_loss(batch)

            if not torch.isfinite(loss):
                print(f"Warning: non-finite loss at batch {batch_idx}, skipping")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            if not torch.isfinite(grad_norm):
                print(f"Warning: non-finite grad norm at batch {batch_idx}, skipping")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_value = float(loss.item())
            l2_value = float(info["l2_err_m"])
            epoch_loss += loss_value
            epoch_l2 += l2_value
            seen_batches += 1
            global_step += 1
            pbar.set_postfix(
                loss=f"{loss_value:.4f}",
                l2=f"{l2_value:.3f}m",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                grad=f"{float(grad_norm):.2f}",
            )

        avg_loss = epoch_loss / max(1, seen_batches)
        avg_l2 = epoch_l2 / max(1, seen_batches)
        epoch_time = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        log(
            f"Epoch {epoch + 1}/{cfg['epochs']} | loss={avg_loss:.4f} "
            f"l2={avg_l2:.3f}m lr={lr_now:.2e} time={epoch_time:.0f}s"
        )
        log_json(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": avg_loss,
                "train_l2": avg_l2,
                "lr": lr_now,
                "epoch_time_sec": round(epoch_time, 3),
            }
        )

        if cfg["val_every_epochs"] > 0 and (epoch + 1) % cfg["val_every_epochs"] == 0:
            log(f"Validating (Epoch {epoch + 1}/{cfg['epochs']})...")
            last_val_loss, last_val_l2 = validate(model, val_loader, device, cfg, autocast_dtype)
            log(f"  Val epoch {epoch + 1}: loss={last_val_loss:.4f} l2={last_val_l2:.3f}m")
            log_json(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "val_loss": last_val_loss,
                    "val_l2": last_val_l2,
                }
            )

            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                best_val_l2 = last_val_l2
                best_path = os.path.join(cfg["log_dir"], "best_model.pt")
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    cfg,
                    epoch,
                    global_step,
                    avg_loss,
                    last_val_loss,
                    last_val_l2,
                )
                log(f"  Saved best model: {best_path}")

        if cfg["save_every_epochs"] > 0 and (epoch + 1) % cfg["save_every_epochs"] == 0:
            ckpt_path = os.path.join(cfg["log_dir"], f"model_epoch{epoch + 1}.pt")
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scheduler,
                scaler,
                cfg,
                epoch,
                global_step,
                avg_loss,
                last_val_loss,
                last_val_l2,
            )
            log(f"  Saved checkpoint: {ckpt_path}")
            prune_checkpoints(cfg["log_dir"], cfg["max_keep_ckpts"])

    log(f"Training finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Best val loss: {best_val_loss:.4f} | best val l2: {best_val_l2:.3f}m")


def main() -> None:
    cfg = dict(DEFAULTS)
    cfg.update(vars(build_parser().parse_args()))
    train(cfg)


if __name__ == "__main__":
    main()
