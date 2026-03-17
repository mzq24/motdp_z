#!/usr/bin/env python3
"""
Training script for Bridge Baseline (BridgeDrive DDBM reproduction).

Reuses MoT-DP's distributed training infrastructure, dataset, and dataloader.
GT trajectory is route[:10] (geometric equidistant route) instead of agent_pos.

Usage:
    # Single GPU
    python bridge_baseline/train.py --config_path bridge_baseline/bd_config.yaml

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=2 bridge_baseline/train.py --config_path bridge_baseline/bd_config.yaml
"""
import os
import sys
import torch
from torch.amp import autocast, GradScaler
import yaml
import wandb
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from collections import defaultdict
import argparse
import datetime
from torch.distributed.elastic.multiprocessing.errors import record
from diffusers.training_utils import EMAModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from dataset.unified_carla_dataset import CARLAImageDataset
from bridge_baseline.policy import BDBaselinePolicy, BDBaselinePolicyV2


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def safe_wandb_log(data, use_wandb=True):
    if not use_wandb:
        return
    try:
        wandb.log({k: v.item() if isinstance(v, (torch.Tensor, np.generic)) else v
                   for k, v in data.items() if isinstance(v, (int, float, torch.Tensor, np.generic))})
    except Exception:
        pass


def compute_driving_metrics(predicted_trajectories, target_trajectories):
    predicted_trajectories = predicted_trajectories.detach().cpu().numpy()
    target_trajectories = target_trajectories.detach().cpu().numpy()
    B, T, _ = predicted_trajectories.shape
    l2_errors = np.linalg.norm(predicted_trajectories - target_trajectories, axis=-1)

    metrics = {}
    if T >= 2:
        metrics['L2_1s'] = np.mean(l2_errors[:, 1])
    if T >= 4:
        metrics['L2_2s'] = np.mean(l2_errors[:, 3])
    if T >= 6:
        metrics['L2_3s'] = np.mean(l2_errors[:, 5])

    l2_avg_values = []
    if T >= 2: l2_avg_values.append(l2_errors[:, 1])
    if T >= 4: l2_avg_values.append(l2_errors[:, 3])
    if T >= 6: l2_avg_values.append(l2_errors[:, 5])
    metrics['L2_avg'] = np.mean(np.concatenate(l2_avg_values)) if l2_avg_values else 0.0
    return metrics


def validate_model(policy, val_loader, device, rank=0, world_size=1,
                   use_amp=False, amp_dtype=torch.float16, max_batches=None):
    policy.eval()
    model_for_inference = policy.module if world_size > 1 else policy
    val_metrics = defaultdict(list)

    with torch.no_grad():
        if max_batches is not None and max_batches <= 0:
            max_batches = None
        total_batches = min(len(val_loader), max_batches) if max_batches else len(val_loader)
        pbar = tqdm(val_loader, desc="Validating", leave=False, total=total_batches) if rank == 0 else val_loader

        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                model_for_inference.train()
                loss_dict = model_for_inference.compute_loss(batch)
                model_for_inference.eval()
                loss = loss_dict['total_loss']

            if rank == 0:
                val_metrics['loss'].append(loss.item())
                for k, v in loss_dict.items():
                    if k != 'total_loss' and isinstance(v, torch.Tensor):
                        val_metrics[k].append(v.item())

                obs_dict = {k: batch[k] for k in (
                    'transfuser_bev_feature_upsample', 'speed',
                    'command_hist', 'target_point_hist', 'target_point_next_hist',
                ) if k in batch}
                # GT for L2 metric: match the model's actual prediction space
                num_poses = model_for_inference.model.trajectory_head.num_poses
                if model_for_inference.bd_config.predict_traj:
                    target_route = batch['agent_pos'][:, :num_poses]
                else:
                    target_route = batch['route'][:, :num_poses]

                try:
                    result = model_for_inference.predict_action(obs_dict)
                    predicted_actions = torch.from_numpy(result['action']).to(device)
                    target_eval = target_route[:, :predicted_actions.shape[1]]

                    driving_metrics = compute_driving_metrics(predicted_actions, target_eval)
                    for key, value in driving_metrics.items():
                        val_metrics[key].append(value)

                    if rank == 0 and hasattr(pbar, 'set_postfix'):
                        postfix = {'val_loss': f'{loss.item():.4f}'}
                        if 'L2_avg' in driving_metrics:
                            postfix['L2_avg'] = f'{driving_metrics["L2_avg"]:.3f}'
                        pbar.set_postfix(postfix)
                except Exception:
                    continue

        if rank == 0 and hasattr(pbar, 'close'):
            pbar.close()

    averaged_metrics = {f'val_{k}': np.mean(v) for k, v in val_metrics.items() if v}
    return averaged_metrics


@record
def train_bridge_baseline(config_path, resume_path=None, val_only=False):
    if val_only and resume_path is None:
        raise ValueError("--val_only requires --resume to specify a checkpoint")

    torch.cuda.empty_cache()
    config = load_config(config_path)

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size == 1:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(f'cuda:{local_rank}')
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://',
            world_size=world_size, rank=rank,
            timeout=datetime.timedelta(minutes=15)
        )
        torch.cuda.set_device(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if rank == 0:
        print(f'[Bridge Baseline] Rank: {rank}, Device: {device}, World size: {world_size}')

    # WandB
    use_wandb = config.get('logging', {}).get('use_wandb', True) and (rank == 0)
    if use_wandb:
        try:
            logging_cfg = config.get('logging', {})
            wandb_api_key = logging_cfg.get('wandb_api_key')
            if wandb_api_key:
                os.environ['WANDB_API_KEY'] = str(wandb_api_key)
            wandb.init(
                project=logging_cfg.get('wandb_project', "bridge-baseline"),
                name=logging_cfg.get('run_name', "bd_baseline_train"),
                mode=os.environ.get('WANDB_MODE', 'online'),
                resume='allow',
                config=config,
            )
        except Exception as e:
            print(f"WandB init failed: {e}")
            use_wandb = False

    # Dataset
    dataset_path_root = config.get('training', {}).get('dataset_path')
    train_dataset_path = os.path.join(dataset_path_root, 'train')
    val_dataset_path = os.path.join(dataset_path_root, 'val')
    image_data_root = config.get('training', {}).get('image_data_root')

    use_per_frame = config.get('dataset', {}).get('use_per_frame', False)
    cache_dir = config.get('dataset', {}).get('cache_dir', None)  # e.g. /tmp/tmp_data for tmpfs
    train_dataset = CARLAImageDataset(dataset_path=train_dataset_path, image_data_root=image_data_root,
                                       use_per_frame=use_per_frame, cache_dir=cache_dir)
    val_dataset = CARLAImageDataset(dataset_path=val_dataset_path, image_data_root=image_data_root,
                                     skip_memmap=True, use_per_frame=use_per_frame)

    if rank == 0:
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    dataloader_cfg = config.get('dataloader', {})
    training_cfg = config.get('training', {})
    validation_cfg = config.get('validation', {})
    train_batch_size = dataloader_cfg.get('batch_size', 32)
    val_batch_size = dataloader_cfg.get('val_batch_size', train_batch_size)
    num_workers = dataloader_cfg.get('num_workers', 4)
    validation_freq = int(validation_cfg.get('freq', training_cfg.get('validation_freq', 5)))
    raw_val_max_batches = validation_cfg.get('max_batches', 16)
    val_max_batches = int(raw_val_max_batches) if raw_val_max_batches not in (None, 0, "0") else None
    if val_max_batches is not None and val_max_batches <= 0:
        val_max_batches = None

    if not use_per_frame:
        _max_val_per_rank = val_max_batches * val_batch_size if val_max_batches else None
        val_dataset.inject_ram_features(train_dataset, rank=rank, world_size=world_size,
                                        max_val_samples=_max_val_per_rank)

    def safe_collate(batch):
        try:
            return default_collate(batch)
        except RuntimeError:
            raise

    if world_size > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_size=train_batch_size, sampler=sampler_train,
            num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=safe_collate)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True)
        val_loader = DataLoader(
            val_dataset, batch_size=val_batch_size, sampler=val_sampler,
            shuffle=False, num_workers=0, drop_last=True, collate_fn=safe_collate)
    else:
        sampler_train = None
        train_loader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=safe_collate)
        val_loader = DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=False,
            num_workers=0, drop_last=True, collate_fn=safe_collate)

    # AMP
    use_amp = config.get('model_optimization', {}).get('use_mixed_precision', True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    if rank == 0:
        print(f"Mixed precision: {'enabled' if use_amp else 'disabled'} (dtype={amp_dtype})")

    # Model
    if rank == 0:
        print("Initializing Bridge Baseline policy...")
    policy_version = config.get('policy_version', 'v1')
    PolicyClass = BDBaselinePolicyV2 if policy_version == 'v2' else BDBaselinePolicy
    policy = PolicyClass(config).to(device)
    if rank == 0:
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Resume
    start_epoch = 0
    checkpoint = None
    if resume_path is not None:
        if rank == 0:
            print(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        policy.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        if rank == 0:
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # DDP
    if world_size > 1:
        policy = torch.nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)

    # Optimizer
    lr = config.get('optimizer', {}).get('lr', 1e-4)
    weight_decay = config.get('optimizer', {}).get('weight_decay', 1e-4)
    scale_lr = config.get('optimizer', {}).get('scale_lr', True)
    if scale_lr and world_size > 1:
        lr = lr * world_size
        if rank == 0:
            print(f"LR scaled for {world_size} GPUs: {lr}")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    # LR Scheduler
    num_epochs = int(training_cfg.get('num_epochs', 200))
    warmup_epochs = int(training_cfg.get('warmup_epochs', 5))
    lr_final = float(training_cfg.get('lr_final', 1e-7))
    use_lr_scheduler = training_cfg.get('use_lr_scheduler', True)
    scheduler = None
    if use_lr_scheduler:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=lr_final)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                  milestones=[warmup_epochs])

    # EMA
    ema_cfg = config.get('ema', {})
    model_for_ema = policy.module if world_size > 1 else policy
    ema_model = EMAModel(model_for_ema.parameters(), max_value=ema_cfg.get('max_value', 0.9999))
    ema_model.to(device)
    ema_update_interval = ema_cfg.get('update_interval', 10)
    if checkpoint is not None and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])

    # Checkpoint dir
    checkpoint_dir = training_cfg.get('checkpoint_dir',
                                      os.path.join(project_root, 'checkpoints', 'bridge_baseline'))
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint dir: {checkpoint_dir}")

    best_l2_avg = float('inf')
    val_loss = None
    val_metrics = {}

    # ========== Val Only Mode ==========
    if val_only:
        if rank == 0:
            print("\n" + "="*50)
            print(f"Running Validation Only on {resume_path}")
            print("="*50)

        ema_model.store(model_for_ema.parameters())
        ema_model.copy_to(model_for_ema.parameters())

        try:
            val_metrics = validate_model(
                policy, val_loader, device, rank=rank, world_size=world_size,
                use_amp=use_amp, amp_dtype=amp_dtype, max_batches=val_max_batches)
        except Exception as e:
            if rank == 0:
                print(f"Error during validation: {e}")

        if rank == 0:
            print("\nValidation Results:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
            safe_wandb_log(val_metrics, use_wandb)

        ema_model.restore(model_for_ema.parameters())
        if use_wandb:
            wandb.finish(quiet=True)
        if world_size > 1:
            torch.distributed.destroy_process_group()
        return

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        if world_size > 1:
            sampler_train.set_epoch(epoch)

        policy.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True) if rank == 0 else train_loader

        for batch_idx, batch in enumerate(pbar):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                loss_dict = policy(batch)
                loss = loss_dict['total_loss']

            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            max_grad_norm = training_cfg.get('max_grad_norm', 1.0)
            params = policy.module.parameters() if world_size > 1 else policy.parameters()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                if rank == 0:
                    print(f"Warning: NaN/Inf gradient at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                continue

            scaler.step(optimizer)
            scaler.update()

            if batch_idx % ema_update_interval == 0:
                ema_model.step(model_for_ema.parameters())

            train_losses.append(loss.item())

            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'grad': f'{grad_norm.item():.2f}'
                })

            log_freq = config.get('logging', {}).get('log_freq', 50)
            if batch_idx % log_freq == 0 and rank == 0:
                log_data = {
                    "train/loss_step": loss.item(),
                    "train/epoch": epoch,
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/grad_norm": grad_norm.item(),
                }
                for k, v in loss_dict.items():
                    if k != 'total_loss':
                        log_data[f"train/{k}"] = v.item() if isinstance(v, torch.Tensor) else v
                safe_wandb_log(log_data, use_wandb)

        if rank == 0 and hasattr(pbar, 'close'):
            pbar.close()

        avg_train_loss = np.mean(train_losses) if train_losses else 0
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Avg loss: {avg_train_loss:.4f}")
            safe_wandb_log({
                "train/loss_epoch": avg_train_loss,
                "train/epoch": epoch,
                "train/lr": optimizer.param_groups[0]['lr'],
            }, use_wandb)

        if scheduler is not None:
            scheduler.step()

        model_to_save = policy.module if world_size > 1 else policy

        save_freq = training_cfg.get('save_freq', 5)
        max_keep_ckpts = training_cfg.get('max_keep_ckpts', 5)
        if rank == 0 and (epoch + 1) % save_freq == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"bd_baseline_epoch{epoch+1}.pt")
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict() if use_amp else None,
                'config': config, 'epoch': epoch,
                'val_loss': val_loss, 'train_loss': avg_train_loss,
                'val_metrics': val_metrics,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

            import glob as glob_module
            periodic_ckpts = sorted(
                glob_module.glob(os.path.join(checkpoint_dir, "bd_baseline_epoch*.pt")),
                key=os.path.getmtime)
            while len(periodic_ckpts) > max_keep_ckpts:
                os.remove(periodic_ckpts.pop(0))

        # Validation
        if (epoch + 1) % validation_freq == 0:
            ema_model.store(model_for_ema.parameters())
            ema_model.copy_to(model_for_ema.parameters())

            try:
                val_metrics = validate_model(
                    policy, val_loader, device, rank=rank, world_size=world_size,
                    use_amp=use_amp, amp_dtype=amp_dtype, max_batches=val_max_batches)
            except Exception as e:
                if rank == 0:
                    print(f"Error during validation: {e}")
                ema_model.restore(model_for_ema.parameters())
                continue

            torch.cuda.empty_cache()

            if rank == 0:
                for key, value in val_metrics.items():
                    print(f"  {key}: {value:.4f}")
                log_dict = {"train/epoch": epoch}
                for key, value in val_metrics.items():
                    log_dict[f"val/{key.removeprefix('val_')}"] = value
                safe_wandb_log(log_dict, use_wandb)

                val_loss = val_metrics.get('val_loss', float('inf'))
                l2_avg = val_metrics.get('val_L2_avg', float('inf'))
                if l2_avg < best_l2_avg:
                    best_l2_avg = l2_avg
                    torch.save({
                        'model_state_dict': model_to_save.state_dict(),
                        'ema_state_dict': ema_model.state_dict(),
                        'config': config, 'epoch': epoch,
                        'val_loss': val_loss, 'train_loss': avg_train_loss,
                        'val_metrics': val_metrics,
                    }, os.path.join(checkpoint_dir, "bd_baseline_best.pt"))
                    print(f"New best model! L2_avg: {l2_avg:.4f}")

            ema_model.restore(model_for_ema.parameters())

    if rank == 0:
        print(f"Training completed! Best L2_avg: {best_l2_avg:.4f}")
        if use_wandb:
            wandb.finish(quiet=True)

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bridge Baseline (BridgeDrive DDBM reproduction)")
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'bd_config.yaml'))
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--val_only', action='store_true')
    args = parser.parse_args()
    train_bridge_baseline(config_path=args.config_path, resume_path=args.resume, val_only=args.val_only)
