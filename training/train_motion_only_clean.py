#!/usr/bin/env python3
"""Clean motion-only training entry point for paper reproduction."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import yaml
from diffusers.training_utils import EMAModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler, WeightedRandomSampler
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast

    def autocast_cuda(enabled: bool, dtype: torch.dtype):
        return autocast('cuda', enabled=enabled, dtype=dtype)
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast as _autocast

    def autocast_cuda(enabled: bool, dtype: torch.dtype):
        return _autocast(enabled=enabled, dtype=dtype)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.unified_carla_dataset import CARLAImageDataset
from policy.paper_motion_policy import PaperMotionPolicy


def setup_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    rank = dist.get_rank() if dist.is_initialized() else 0
    return device, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}




class WeightedDistributedSampler(Sampler):
    """DDP sampler for per-sample weights.

    All ranks draw the same weighted global index list for each epoch and take
    rank-strided slices, keeping per-rank batch counts aligned.
    """

    def __init__(
        self,
        weights,
        num_replicas=None,
        rank=None,
        replacement=True,
        drop_last=True,
        seed=0,
    ):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank={rank}, num_replicas={num_replicas}")

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.dim() != 1:
            raise ValueError(f"weights must be 1-D, got {tuple(self.weights.shape)}")
        if len(self.weights) == 0:
            raise ValueError("weights must be non-empty")
        if not torch.isfinite(self.weights).all() or float(self.weights.sum()) <= 0.0:
            raise ValueError("weights must be finite and have positive sum")
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        dataset_len = len(self.weights)
        if self.drop_last and dataset_len % self.num_replicas != 0:
            self.num_samples = int(np.ceil(max(dataset_len - self.num_replicas, 0) / self.num_replicas))
        else:
            self.num_samples = int(np.ceil(dataset_len / self.num_replicas))
        self.total_size = int(self.num_samples * self.num_replicas)
        if not self.replacement and self.total_size > dataset_len:
            raise ValueError(
                "replacement=False requires total_size <= dataset length; "
                f"got total_size={self.total_size}, dataset_len={dataset_len}"
            )

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def _sample_float(sample, keys, default=0.0):
    for key in keys:
        if key not in sample:
            continue
        try:
            value = sample.get(key)
            if isinstance(value, np.ndarray):
                value = float(np.asarray(value).reshape(-1)[0])
            else:
                value = float(value)
            if np.isfinite(value):
                return value
        except Exception:
            continue
    return float(default)


def _sample_int(sample, keys, default=-1):
    value = _sample_float(sample, keys, default=float(default))
    if not np.isfinite(value):
        return int(default)
    return int(round(value))


def build_window_sample_weights(dataset, dataloader_cfg):
    """Build window/transition-aware weights from cached semantic labels."""
    samples = getattr(dataset, '_sample_cache', None)
    if samples is None:
        raise AttributeError("window-aware sampler requires dataset._sample_cache")

    weight_cfg = dataloader_cfg.get('window_sampler_weights', {}) or {}
    none_weight = float(weight_cfg.get('none', dataloader_cfg.get('window_sampler_none_weight', 1.0)))
    merge_weight = float(weight_cfg.get('merge', dataloader_cfg.get('window_sampler_merge_weight', 3.0)))
    junction_weight = float(weight_cfg.get('junction', dataloader_cfg.get('window_sampler_junction_weight', 3.0)))
    borrow_weight = float(weight_cfg.get('borrow', dataloader_cfg.get('window_sampler_borrow_weight', 5.0)))
    use_semantic_shift_sampler = bool(dataloader_cfg.get('use_semantic_shift_sampler', False))
    shift_window_weight = float(dataloader_cfg.get('semantic_shift_window_weight', 4.0))
    shift_phase_weight = float(dataloader_cfg.get('semantic_shift_phase_weight', 3.0))
    shift_area_weight = float(dataloader_cfg.get('semantic_shift_area_weight', 2.0))
    shift_edge_weight = float(dataloader_cfg.get('semantic_shift_edge_weight', 2.0))
    shift_opportunity_weight = float(dataloader_cfg.get('semantic_shift_opportunity_weight', 2.0))
    shift_max_weight = float(dataloader_cfg.get('semantic_shift_max_weight', 8.0))

    weights = np.full(len(samples), none_weight, dtype=np.float64)
    counts = {'none': 0, 'merge': 0, 'junction': 0, 'borrow': 0}
    shift_counts = {'window': 0, 'phase': 0, 'area': 0, 'edge': 0, 'opportunity': 0}
    for idx, sample in enumerate(samples):
        family = int(round(_sample_float(sample, ('conflict_area_family',), default=-1.0)))
        merge_active = _sample_float(sample, ('merge_active', 'merge_episode_active')) > 0.5
        junction_active = _sample_float(
            sample,
            ('junction_cross_active', 'junction_cross_episode_active', 'cross_active', 'cross_episode_active'),
        ) > 0.5
        borrow_active = _sample_float(sample, ('borrow_cross_active', 'borrow_cross_episode_active')) > 0.5
        borrow_active = borrow_active or family == 1
        merge_active = merge_active or family == 2
        junction_active = junction_active or family == 3

        sample_weight = none_weight
        if merge_active:
            sample_weight = max(sample_weight, merge_weight)
            counts['merge'] += 1
        if junction_active:
            sample_weight = max(sample_weight, junction_weight)
            counts['junction'] += 1
        if borrow_active:
            sample_weight = max(sample_weight, borrow_weight)
            counts['borrow'] += 1
        if not (merge_active or junction_active or borrow_active):
            counts['none'] += 1

        if use_semantic_shift_sampler:
            shift_bonus = 0.0
            prev_family = _sample_int(sample, ('prev_conflict_area_family',), default=-1)
            if prev_family >= 0 and family >= 0 and prev_family != family:
                shift_bonus += shift_window_weight
                shift_counts['window'] += 1

            decision = _sample_int(sample, ('conflict_decision_phase',), default=-1)
            prev_decision = _sample_int(sample, ('prev_conflict_decision_phase',), default=-1)
            control = _sample_int(sample, ('conflict_control_phase',), default=-1)
            prev_control = _sample_int(sample, ('prev_conflict_control_phase',), default=-1)
            phase_shift = (
                prev_decision > 0 and decision > 0 and prev_decision != decision
            ) or (
                prev_control > 0 and control > 0 and prev_control != control
            )
            if phase_shift:
                shift_bonus += shift_phase_weight
                shift_counts['phase'] += 1

            status = _sample_int(sample, ('conflict_area_status',), default=-1)
            prev_status = _sample_int(sample, ('prev_conflict_area_status',), default=-1)
            if prev_status >= 0 and status >= 0 and prev_status != status:
                shift_bonus += shift_area_weight
                shift_counts['area'] += 1

            has_prev_edge = (
                'prev_current_cover_edge_valid' in sample
                or 'prev_future_cover_edge_valid' in sample
            )
            if has_prev_edge:
                current_edge_valid = _sample_float(sample, ('current_cover_edge_valid',), default=0.0) > 0.5
                prev_current_edge_valid = _sample_float(sample, ('prev_current_cover_edge_valid',), default=0.0) > 0.5
                future_edge_valid = _sample_float(sample, ('future_cover_edge_valid',), default=0.0) > 0.5
                prev_future_edge_valid = _sample_float(sample, ('prev_future_cover_edge_valid',), default=0.0) > 0.5
                current_mode = _sample_int(sample, ('current_cover_edge_mode',), default=-1)
                prev_current_mode = _sample_int(sample, ('prev_current_cover_edge_mode',), default=-1)
                future_mode = _sample_int(sample, ('future_cover_edge_mode',), default=-1)
                prev_future_mode = _sample_int(sample, ('prev_future_cover_edge_mode',), default=-1)
                edge_shift = (
                    current_edge_valid != prev_current_edge_valid
                    or future_edge_valid != prev_future_edge_valid
                    or (
                        current_edge_valid and prev_current_edge_valid
                        and current_mode >= 0 and prev_current_mode >= 0
                        and current_mode != prev_current_mode
                    )
                    or (
                        future_edge_valid and prev_future_edge_valid
                        and future_mode >= 0 and prev_future_mode >= 0
                        and future_mode != prev_future_mode
                    )
                )
                if edge_shift:
                    shift_bonus += shift_edge_weight
                    shift_counts['edge'] += 1

            go_prob = _sample_float(sample, ('go_opportunity_prob',), default=0.5)
            prev_go_prob = _sample_float(sample, ('prev_go_opportunity_prob',), default=0.5)
            yld_prob = _sample_float(sample, ('yld_pressure_prob',), default=0.5)
            prev_yld_prob = _sample_float(sample, ('prev_yld_pressure_prob',), default=0.5)
            go_cross = (prev_go_prob < 0.5 <= go_prob) or (prev_go_prob >= 0.5 > go_prob)
            yld_cross = (prev_yld_prob < 0.5 <= yld_prob) or (prev_yld_prob >= 0.5 > yld_prob)
            if go_cross or yld_cross:
                shift_bonus += shift_opportunity_weight
                shift_counts['opportunity'] += 1

            sample_weight = min(sample_weight + shift_bonus, shift_max_weight)
        weights[idx] = sample_weight

    weights = np.maximum(weights, 1e-6)
    summary = {
        'counts': counts,
        'shift_counts': shift_counts,
        'weights': {
            'none': none_weight,
            'merge': merge_weight,
            'junction': junction_weight,
            'borrow': borrow_weight,
            'semantic_shift_enabled': float(use_semantic_shift_sampler),
            'semantic_shift_max': shift_max_weight,
        },
        'mean_weight': float(weights.mean()) if weights.size else 0.0,
        'max_weight': float(weights.max()) if weights.size else 0.0,
    }
    return weights, summary

def build_datasets(config: Dict):
    training_cfg = config['training']
    dataset_cfg = config.get('dataset', {})
    root = training_cfg['dataset_path']
    image_root = training_cfg['image_data_root']
    feature_suffix = dataset_cfg.get('feature_suffix', '') or ''
    use_per_frame = bool(dataset_cfg.get('use_per_frame', False))
    cache_dir = dataset_cfg.get('cache_dir')
    use_fullres = bool(dataset_cfg.get('use_fullres_upsample_cache', False))
    gps_noise_cfg = config.get('augmentation', {}).get('gps_noise', {})
    train_set = CARLAImageDataset(
        dataset_path=os.path.join(root, 'train'),
        image_data_root=image_root,
        mode='train',
        use_per_frame=use_per_frame,
        cache_dir=cache_dir,
        feature_suffix=feature_suffix,
        gps_noise_cfg=gps_noise_cfg,
        load_transfuser_lidar_bev=False,
        filter_bad_routes=bool(dataset_cfg.get('train_filter_bad_routes', False)),
        retain_bad_routes_for_energy=False,
        use_fullres_upsample_cache=use_fullres,
    )
    val_set = None
    if config.get('validation', {}).get('enabled', True):
        val_set = CARLAImageDataset(
            dataset_path=os.path.join(root, 'val'),
            image_data_root=image_root,
            mode='val',
            skip_memmap=not bool(config.get('validation', {}).get('use_memmap', True)),
            use_per_frame=use_per_frame,
            cache_dir=cache_dir,
            feature_suffix=feature_suffix,
            load_transfuser_lidar_bev=False,
            filter_bad_routes=bool(dataset_cfg.get('val_filter_bad_routes', True)),
            retain_bad_routes_for_energy=False,
            use_fullres_upsample_cache=use_fullres,
        )
    return train_set, val_set


def make_loader(dataset, config: Dict, train: bool, rank: int, world_size: int):
    dl_cfg = config.get('dataloader', {})
    if train:
        batch_size = int(dl_cfg.get('batch_size', 128))
        workers = int(dl_cfg.get('train_num_workers', dl_cfg.get('num_workers', 2)))
        prefetch = int(dl_cfg.get('train_prefetch_factor', dl_cfg.get('prefetch_factor', 1)))
        persistent = bool(dl_cfg.get('train_persistent_workers', dl_cfg.get('persistent_workers', False)))
    else:
        batch_size = int(dl_cfg.get('val_batch_size', dl_cfg.get('batch_size', 128)))
        workers = int(dl_cfg.get('val_num_workers', 1))
        prefetch = int(dl_cfg.get('val_prefetch_factor', 1))
        persistent = bool(dl_cfg.get('val_persistent_workers', False))
    sampler = None
    use_weighted = train and bool(dl_cfg.get('use_window_weighted_sampler', False))
    if use_weighted:
        weights, summary = build_window_sample_weights(dataset, dl_cfg)
        if is_main(rank):
            print(
                "Using motion-only window-aware weighted sampler: "
                f"counts={summary['counts']}, shift_counts={summary['shift_counts']}, "
                f"weights={summary['weights']}, mean_weight={summary['mean_weight']:.3f}, "
                f"max_weight={summary['max_weight']:.3f}, "
                f"replacement={bool(dl_cfg.get('window_sampler_replacement', True))}"
            )
        replacement = bool(dl_cfg.get('window_sampler_replacement', True))
        seed = int(dl_cfg.get('window_sampler_seed', 0))
        if world_size > 1:
            sampler = WeightedDistributedSampler(
                weights,
                num_replicas=world_size,
                rank=rank,
                replacement=replacement,
                drop_last=train,
                seed=seed,
            )
        else:
            generator = torch.Generator()
            generator.manual_seed(seed)
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=replacement,
                generator=generator,
            )
    elif world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=train, drop_last=train)
    kwargs = {
        'batch_size': batch_size,
        'shuffle': train and sampler is None,
        'sampler': sampler,
        'num_workers': workers,
        'pin_memory': bool(dl_cfg.get('pin_memory', False)),
        'drop_last': train,
        'persistent_workers': persistent and workers > 0,
    }
    if workers > 0:
        kwargs['prefetch_factor'] = prefetch
    return DataLoader(dataset, **kwargs), sampler


def lr_for_epoch(config: Dict, epoch_idx: int) -> float:
    opt_cfg = config.get('optimizer', {})
    train_cfg = config.get('training', {})
    lr0 = float(opt_cfg.get('lr', 5e-5))
    lr_final = float(train_cfg.get('lr_final', 1e-7))
    total = max(int(train_cfg.get('scheduler_total_epochs', train_cfg.get('num_epochs', 60))), 1)
    warmup = int(train_cfg.get('warmup_epochs', 0))
    if not bool(train_cfg.get('use_lr_scheduler', True)):
        return lr0
    if warmup > 0 and epoch_idx < warmup:
        return lr0 * float(epoch_idx + 1) / float(warmup)
    span = max(total - warmup, 1)
    progress = min(max(epoch_idx + 1 - warmup, 0) / span, 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return lr_final + (lr0 - lr_final) * cosine


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group['lr'] = lr


def reduce_metrics(metrics: Dict[str, float], rank: int, world_size: int) -> Dict[str, float]:
    if world_size <= 1:
        return metrics
    keys = sorted(metrics.keys())
    values = torch.tensor([metrics[k] for k in keys], dtype=torch.float64, device='cuda')
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {k: float(v.item()) for k, v in zip(keys, values)}


@torch.no_grad()
def validate(policy: PaperMotionPolicy, val_loader, config: Dict, device: torch.device, rank: int, world_size: int) -> Dict[str, float]:
    if val_loader is None:
        return {}
    policy.eval()
    sums = defaultdict(float)
    count = 0
    max_batches = int(config.get('validation', {}).get('max_batches', 0) or 0)
    iterator = val_loader if not is_main(rank) else tqdm(val_loader, desc='val', leave=False)
    for batch_idx, batch in enumerate(iterator):
        if max_batches and batch_idx >= max_batches:
            break
        batch = move_batch(batch, device)
        loss_dict = policy.compute_loss(batch)
        sample = policy.sample(batch)
        traj_gt = batch['agent_pos'][:, :policy.horizon]
        route_gt = batch['route'][:, :policy.num_waypoints]
        traj_l2 = torch.linalg.norm(sample['trajectory'] - traj_gt, dim=-1)
        route_l2 = torch.linalg.norm(sample['route'] - route_gt, dim=-1)
        speed_target = batch.get('next_speed_target_mps')
        if speed_target is not None and sample['speed_mps'] is not None:
            speed_mae = (sample['speed_mps'] - speed_target.to(device=device, dtype=sample['speed_mps'].dtype).reshape(-1)).abs().mean()
        else:
            speed_mae = torch.zeros((), device=device)
        bsz = traj_gt.shape[0]
        count += bsz
        l2_1 = traj_l2[:, min(1, traj_l2.shape[1] - 1)].mean()
        l2_2 = traj_l2[:, min(3, traj_l2.shape[1] - 1)].mean()
        l2_3 = traj_l2[:, min(5, traj_l2.shape[1] - 1)].mean()
        batch_metrics = {
            'val_loss': loss_dict['total_loss'],
            'val_ADE': traj_l2.mean(),
            'val_L2_1s': l2_1,
            'val_L2_2s': l2_2,
            'val_L2_3s': l2_3,
            'val_L2_avg': (l2_1 + l2_2 + l2_3) / 3.0,
            'val_route_L2': route_l2.mean(),
            'val_route_L2_final': route_l2[:, -1].mean(),
            'val_speed_loss': loss_dict['speed_loss'],
            'val_speed_mae_speed_head': speed_mae,
        }
        for key, value in batch_metrics.items():
            sums[key] += float(value.detach().item()) * bsz
    reduced = reduce_metrics({**sums, '_count': float(count)}, rank, world_size)
    total_count = max(reduced.pop('_count', 0.0), 1.0)
    return {key: value / total_count for key, value in reduced.items()}


def save_checkpoint(path: Path, policy, optimizer, scaler, ema_model, epoch: int, config: Dict) -> None:
    model = policy.module if isinstance(policy, DDP) else policy
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'ema_state_dict': ema_model.state_dict() if ema_model is not None else None,
        'config': config,
    }, path)


def append_val_csv(path: Path, epoch: int, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    keys = ['epoch'] + sorted(metrics.keys())
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not exists:
            writer.writeheader()
        row = {'epoch': epoch}
        row.update(metrics)
        writer.writerow(row)


def train(config_path: str, resume: Optional[str] = None, val_only: bool = False):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    device, rank, world_size, _ = setup_distributed()
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    if is_main(rank):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Clean motion-only training: {config_path}")
        print(f"checkpoint_dir={checkpoint_dir}")
        print(f"world_size={world_size}")

    train_set, val_set = build_datasets(config)
    train_loader, train_sampler = make_loader(train_set, config, True, rank, world_size)
    val_loader, _ = make_loader(val_set, config, False, rank, world_size) if val_set is not None else (None, None)

    policy = PaperMotionPolicy(config).to(device)
    policy.register_norm_stats_from_config(config)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(config.get('optimizer', {}).get('lr', 5e-5)),
        weight_decay=float(config.get('optimizer', {}).get('weight_decay', 1e-4)),
        betas=tuple(config.get('optimizer', {}).get('betas', [0.9, 0.999])),
        eps=float(config.get('optimizer', {}).get('eps', 1e-8)),
    )
    use_amp = bool(config.get('model_optimization', {}).get('use_mixed_precision', True)) and device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp) if 'cuda' in str(device) else GradScaler(enabled=False)
    ema_cfg = config.get('ema', {})
    ema_model = EMAModel(policy.parameters(), max_value=float(ema_cfg.get('max_value', 0.9999)))
    ema_model.to(device)

    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location='cpu', weights_only=False)
        policy.load_state_dict(ckpt['model_state_dict'], strict=True)
        start_epoch = int(ckpt.get('epoch', 0))
        if not bool(config.get('training', {}).get('resume_weights_only', False)):
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if ckpt.get('scaler_state_dict'):
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            if ckpt.get('ema_state_dict'):
                ema_model.load_state_dict(ckpt['ema_state_dict'])
        if is_main(rank):
            print(f"Resumed {resume} at epoch {start_epoch}")

    if world_size > 1:
        policy = DDP(policy, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    if val_only:
        model = policy.module if isinstance(policy, DDP) else policy
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
        metrics = validate(model, val_loader, config, device, rank, world_size)
        ema_model.restore(model.parameters())
        if is_main(rank):
            print(metrics)
        cleanup_distributed()
        return

    num_epochs = int(config['training'].get('num_epochs', 60))
    val_freq = int(config['training'].get('validation_freq', 5))
    save_freq = int(config['training'].get('save_freq', 5))
    log_freq = int(config.get('logging', {}).get('log_freq', 50))
    max_grad_norm = float(config['training'].get('max_grad_norm', 1.0))
    best_metric_name = str(config['training'].get('best_checkpoint_metric', 'l2_avg'))
    best_value = float('inf')

    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
        set_optimizer_lr(optimizer, lr_for_epoch(config, epoch))
        policy.train()
        iterator = train_loader if not is_main(rank) else tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        running = defaultdict(float)
        for batch_idx, batch in enumerate(iterator):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_cuda(use_amp, torch.float16):
                loss_dict = policy(batch, return_loss_dict=True)
                loss = loss_dict['total_loss']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            model = policy.module if isinstance(policy, DDP) else policy
            ema_model.step(model.parameters())
            for key, value in loss_dict.items():
                running[key] += float(value.detach().item())
            if is_main(rank) and (batch_idx + 1) % log_freq == 0:
                denom = float(log_freq)
                summary = {k: v / denom for k, v in running.items()}
                running.clear()
                iterator.set_postfix({k: f'{v:.4f}' for k, v in summary.items() if k in ('total_loss', 'reg_loss', 'route_loss', 'speed_loss')})

        if is_main(rank) and ((epoch + 1) % save_freq == 0 or epoch + 1 == num_epochs):
            save_checkpoint(checkpoint_dir / f'dit_policy_epoch{epoch + 1}.pt', policy, optimizer, scaler, ema_model, epoch + 1, config)

        if val_loader is not None and ((epoch + 1) % val_freq == 0 or epoch + 1 == num_epochs):
            model = policy.module if isinstance(policy, DDP) else policy
            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())
            metrics = validate(model, val_loader, config, device, rank, world_size)
            ema_model.restore(model.parameters())
            if is_main(rank):
                append_val_csv(checkpoint_dir / 'val_metrics_summary.csv', epoch + 1, metrics)
                print(f"Epoch {epoch + 1} val: " + ', '.join(f'{k}={v:.4f}' for k, v in sorted(metrics.items())))
                metric_key = 'val_' + best_metric_name if not best_metric_name.startswith('val_') else best_metric_name
                value = metrics.get(metric_key, metrics.get('val_L2_avg', float('inf')))
                if value < best_value:
                    best_value = value
                    save_checkpoint(checkpoint_dir / 'dit_policy_best.pt', policy, optimizer, scaler, ema_model, epoch + 1, config)
                    print(f"Saved best checkpoint by {metric_key}: {best_value:.4f}")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--val_only', action='store_true')
    args = parser.parse_args()
    train(args.config_path, resume=args.resume, val_only=args.val_only)


if __name__ == '__main__':
    main()
