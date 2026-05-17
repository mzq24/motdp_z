"""
Improved training: tqdm, wandb, ADE/FDE val metrics.
Run on newhpc:
    conda activate z_navsim_motdp
    cd /home/z/code/motdp_z_navsim_motdp
    python training/train_navsim_diffusion.py --cache_dir /path/to/cache --log_dir /path/to/logs
"""

import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.navsim_simple_diffusion import NavSimSimpleDiffusion
from dataset.navsim_cached_dataset import NavSimCachedDataset, collate_fn

# Optional wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def compute_ade_fde(model, val_loader, device, use_amp, num_samples=None):
    """Compute ADE/FDE via DDIM sampling."""
    model.eval()
    total_ade, total_fde, count = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Val ADE/FDE', leave=False):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            gt = batch['trajectory']  # (B, 8, 2)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model.sample(batch['bev_grid'], batch['ego_status'],
                                    return_trajectory=True)
            # ADE: average L2 distance over all waypoints
            ade = torch.norm(pred - gt, dim=-1).mean(dim=-1)  # (B,)
            # FDE: L2 distance at final waypoint
            fde = torch.norm(pred[:, -1] - gt[:, -1], dim=-1)  # (B,)
            total_ade += ade.sum().item()
            total_fde += fde.sum().item()
            count += len(gt)
            if num_samples and count >= num_samples:
                break
    model.train()
    return total_ade / count, total_fde / count


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default='/workspace2/z_project/motdp_bev_cache_official_npy')
    parser.add_argument('--log_dir', type=str, default='/workspace2/z_project/motdp_logs/navsim_v2')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--lr_final', type=float, default=1e-7)
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='navsim-diffusion')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = True

    # ---- Logging ----
    os.makedirs(args.log_dir, exist_ok=True)
    run_name = args.wandb_name or time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.log_dir, f'{run_name}.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    log(f'Device: {device}')
    log(f'Args: {vars(args)}')

    # ---- WandB ----
    if args.use_wandb and HAS_WANDB:
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        log('WandB initialized')
    elif args.use_wandb and not HAS_WANDB:
        log('WARNING: wandb not installed, skipping')

    # ---- Dataset ----
    log('Loading datasets...')
    train_ds = NavSimCachedDataset(cache_dir=args.cache_dir, split='train',
                                    val_ratio=args.val_ratio)
    val_ds = NavSimCachedDataset(cache_dir=args.cache_dir, split='val',
                                  val_ratio=args.val_ratio)
    log(f'Train: {len(train_ds)}, Val: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_fn)

    # ---- Model ----
    log('Creating model...')
    model = NavSimSimpleDiffusion(
        d_model=args.d_model, n_head=8, n_layer=args.n_layer, d_ffn=2048,
        traj_horizon=8, traj_dim=2, ego_input_dim=14, ego_history_frames=4,
        p_drop_attn=0.1, p_drop_emb=0.1,
        prediction_type='sample', num_inference_steps=10,
        num_train_timesteps=1000, beta_schedule='cosine',
    ).to(device)

    mean, std = train_ds.get_traj_stats()
    model.traj_mean.copy_(torch.tensor(mean, dtype=torch.float32))
    model.traj_std.copy_(torch.tensor(std, dtype=torch.float32))
    log(f'Model: {sum(p.numel()/1e6 for p in model.parameters()):.1f}M params')
    log(f'Traj stats: mean={mean.round(2)} std={std.round(2)}')

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return np.cos(progress * np.pi / 2) * (1.0 - args.lr_final / args.lr) + args.lr_final / args.lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- Resume ----
    start_epoch = 0
    if args.resume:
        log(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        log(f'Resumed at epoch {start_epoch}')

    # ---- Training Loop ----
    global_step = start_epoch * len(train_loader)
    best_val_loss = float('inf')
    best_ade = float('inf')

    log(f'Training started: {time.strftime("%Y-%m-%d %H:%M:%S")}')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss, epoch_l2 = 0.0, 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:02d}/{args.epochs}',
                    unit='batch', ncols=100)
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, info = model.compute_loss(batch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_l2 += info['l2_err_m']
            global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                             'l2': f'{info["l2_err_m"]:.3f}m',
                             'lr': f'{scheduler.get_last_lr()[0]:.1e}'})

            if args.use_wandb and HAS_WANDB and global_step % 50 == 0:
                wandb.log({'train/loss': loss.item(), 'train/l2_m': info['l2_err_m'],
                          'lr': scheduler.get_last_lr()[0]}, step=global_step)

        epoch_loss /= len(train_loader)
        epoch_l2 /= len(train_loader)
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        log(f'Epoch {epoch+1:02d}/{args.epochs} | loss={epoch_loss:.4f} l2={epoch_l2:.3f}m '
            f'lr={lr_now:.1e} time={elapsed/60:.1f}min')

        if args.use_wandb and HAS_WANDB:
            wandb.log({'epoch': epoch+1, 'train/epoch_loss': epoch_loss,
                      'train/epoch_l2_m': epoch_l2}, step=global_step)

        # ---- Validation ----
        if (epoch + 1) % args.val_every == 0:
            model.eval()
            val_loss, val_l2 = 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                             for k, v in batch.items()}
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        loss, info = model.compute_loss(batch)
                    val_loss += loss.item()
                    val_l2 += info['l2_err_m']
            val_loss /= len(val_loader)
            val_l2 /= len(val_loader)

            # ADE/FDE (on subset for speed)
            val_ade, val_fde = compute_ade_fde(model, val_loader, device, use_amp, num_samples=1024)

            log(f'  Val epoch {epoch+1}: loss={val_loss:.4f} l2={val_l2:.3f}m '
                f'ADE={val_ade:.3f}m FDE={val_fde:.3f}m')

            if args.use_wandb and HAS_WANDB:
                wandb.log({'val/loss': val_loss, 'val/l2_m': val_l2,
                          'val/ADE_m': val_ade, 'val/FDE_m': val_fde}, step=global_step)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ade = val_ade
                ckpt = {
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'traj_mean': model.traj_mean.cpu().numpy(),
                    'traj_std': model.traj_std.cpu().numpy(),
                    'val_loss': val_loss, 'val_ade': val_ade, 'val_fde': val_fde,
                    'config': vars(args),
                }
                torch.save(ckpt, os.path.join(args.log_dir, 'best_model.pt'))
                log(f'  Saved best (val_loss={val_loss:.4f}, ADE={val_ade:.3f}m)')

            model.train()

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}
            torch.save(ckpt, os.path.join(args.log_dir, f'model_epoch{epoch+1:03d}.pt'))

    log(f'Training finished. Best val_loss={best_val_loss:.4f}, best ADE={best_ade:.3f}m')


if __name__ == '__main__':
    main()
