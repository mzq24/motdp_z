#!/usr/bin/env python3
import os
import sys
import torch
import json
import csv
import shutil
try:
    from torch.amp import autocast as torch_autocast, GradScaler

    def autocast_cuda(enabled, dtype):
        return torch_autocast('cuda', enabled=enabled, dtype=dtype)
except ImportError:
    from torch.cuda.amp import autocast as torch_autocast, GradScaler

    def autocast_cuda(enabled, dtype):
        return torch_autocast(enabled=enabled, dtype=dtype)
import yaml
import wandb
import numpy as np
from torch.utils.data import DataLoader, Sampler
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import argparse
import datetime
from typing import Any, Dict, List, Optional, Tuple
from torch.distributed.elastic.multiprocessing.errors import record
from diffusers.training_utils import EMAModel


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from dataset.unified_carla_dataset import CARLAImageDataset
from policy.annealed_energy_guidance_policy import AnnealedEnergyGuidancePolicy


CURRENT_ADAPTIVE_WEIGHTS = {
    'startup': [0.50, 0.35, 0.15],
    'low': [0.29, 0.30, 0.41],
    'medium': [0.36, 0.34, 0.30],
    'high': [0.30, 0.37, 0.33],
}
REGIME_ORDER = ['startup', 'low', 'medium', 'high']


def _stage_feature_cache_to_ram(
    *,
    source_dir: str,
    target_dir: str,
    feature_suffix: str = '',
    use_fullres_upsample_cache: bool = True,
    include_lidar: bool = False,
    rank: int = 0,
):
    """Copy shared feature memmap files to a RAM-backed cache directory.

    This keeps the dataset on the memmap path, but moves the underlying files to
    tmpfs (for example /dev/shm) so random training access is not at the mercy of
    the OS page cache being evicted by other jobs.
    """
    source_dir = os.path.realpath(source_dir)
    target_dir = os.path.realpath(target_dir)
    if source_dir == target_dir:
        if rank == 0:
            print(f"Feature RAM staging skipped: source and target are both {target_dir}")
        return

    sfx = f'_{feature_suffix}' if feature_suffix else ''
    required_files = [
        f'feature_index{sfx}.pkl',
        f'bev_features_fp16{sfx}.bin',
        f'bev_upsamples_fp16{sfx}.bin',
    ]
    optional_files = []
    if use_fullres_upsample_cache:
        optional_files.extend([
            f'feature_index{sfx}_fullres.pkl',
            f'bev_upsamples_fp16{sfx}_fullres.bin',
        ])
    if include_lidar:
        optional_files.extend([
            'lidar_bev_index.pkl',
            'lidar_bev_fp16.bin',
        ])

    os.makedirs(target_dir, exist_ok=True)

    def _copy_one(name: str, *, required: bool):
        src = os.path.join(source_dir, name)
        dst = os.path.join(target_dir, name)
        if not os.path.exists(src):
            if required:
                raise FileNotFoundError(f"Required feature cache file missing: {src}")
            return
        src_size = os.path.getsize(src)
        if os.path.exists(dst) and os.path.getsize(dst) == src_size:
            return
        print(f"  staging {name} -> {target_dir}")
        shutil.copy2(src, dst)

    if rank == 0:
        print(f"Staging feature cache to RAM: {source_dir} -> {target_dir}")
        for filename in required_files:
            _copy_one(filename, required=True)
        for filename in optional_files:
            _copy_one(filename, required=False)
        print("Feature RAM staging complete.")


class DistributedRouteBatchSampler(Sampler):
    """DDP route-grouped batch sampler to reduce random memmap page faults.

    It builds route-local batches globally, drops the tail to make the batch
    count divisible by world_size, then gives each rank a strided slice.
    """

    def __init__(
        self,
        route_groups,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=True,
        drop_last=True,
        seed=0,
    ):
        if num_replicas is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                num_replicas = 1
            else:
                num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                rank = 0
            else:
                rank = torch.distributed.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank={rank}, num_replicas={num_replicas}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.route_groups = [list(g) for g in route_groups if len(g) > 0]
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def _build_batches(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            group_order = torch.randperm(len(self.route_groups), generator=generator).tolist()
        else:
            group_order = list(range(len(self.route_groups)))

        batches = []
        for group_idx in group_order:
            group = list(self.route_groups[group_idx])
            if self.shuffle and len(group) > 1:
                perm = torch.randperm(len(group), generator=generator).tolist()
                group = [group[i] for i in perm]
            for start in range(0, len(group), self.batch_size):
                batch = group[start:start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)

        usable = (len(batches) // self.num_replicas) * self.num_replicas
        return batches[:usable]

    def __iter__(self):
        batches = self._build_batches()
        return iter(batches[self.rank::self.num_replicas])

    def __len__(self):
        return len(self._build_batches()) // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


class WeightedDistributedSampler(Sampler):
    """Distributed weighted sampler for DDP.

    Each rank builds the same weighted global draw for the epoch, then takes its
    rank-specific strided slice. This keeps per-rank sample counts aligned.
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
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                num_replicas = 1
            else:
                num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                rank = 0
            else:
                rank = torch.distributed.get_rank()
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
    """Build simple window-aware sample weights from cached stage1 active fields."""
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
        # Newer packed stage1 labels use conflict_area_family:
        # 0 none, 1 borrow, 2 merge, 3 junction. Keep the legacy active-flag
        # paths too so old packed datasets behave exactly as before.
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


def _speed_regime_masks(median3, current_speed, rough_threshold=2.5, high_threshold=10.0, startup_speed_threshold=0.2):
    startup_mask = (median3 < rough_threshold) & (current_speed < startup_speed_threshold)
    low_mask = (median3 < rough_threshold) & (~startup_mask)
    medium_mask = (median3 >= rough_threshold) & (median3 < high_threshold)
    high_mask = median3 >= high_threshold
    return {
        'startup': startup_mask,
        'low': low_mask,
        'medium': medium_mask,
        'high': high_mask,
    }


def _mae(pred, target):
    return float(np.mean(np.abs(pred - target)))

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(project_root, "config", "pdm_server.yaml")
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


def safe_wandb_finish(use_wandb=True):
    if not use_wandb:
        return
    wandb.finish(quiet=True, exit_code=0)




def compute_driving_metrics(predicted_trajectories, target_trajectories, fut_obstacles=None):
    """
    计算驾驶性能指标
    
    Args:
        predicted_trajectories: (B, T, 2) 预测轨迹
        target_trajectories: (B, T, 2) 真实轨迹
        fut_obstacles: List of B lists, each containing T dicts with obstacle info
                      Each dict has 'gt_boxes' (N, 7), 'gt_names' (N,), 'gt_velocity' (N, 2)
                      Optional - if None, collision metrics will not be computed
    
    Returns:
        metrics: 包含L2误差和碰撞率的字典
        
    """
    predicted_trajectories = predicted_trajectories.detach().cpu().numpy()
    target_trajectories = target_trajectories.detach().cpu().numpy()
    
    B, T, _ = predicted_trajectories.shape
    
    l2_errors = np.linalg.norm(
        predicted_trajectories - target_trajectories, axis=-1
    )
    
    metrics = {}
    
    # === L2 误差指标 ===
    metrics['ADE'] = np.mean(l2_errors)

    if T >= 2:  
        metrics['L2_1s'] = np.mean(l2_errors[:, 1])
    
    if T >= 4:  
        metrics['L2_2s'] = np.mean(l2_errors[:, 3])
    
    if T >= 6: 
        metrics['L2_3s'] = np.mean(l2_errors[:, 5])
    
    # L2_avg: 只计算1s, 2s, 3s时间步的平均
    l2_avg_values = []
    if T >= 2:
        l2_avg_values.append(l2_errors[:, 1])
    if T >= 4:
        l2_avg_values.append(l2_errors[:, 3])
    if T >= 6:
        l2_avg_values.append(l2_errors[:, 5])
    
    if len(l2_avg_values) > 0:
        metrics['L2_avg'] = np.mean(np.concatenate(l2_avg_values))
    else:
        metrics['L2_avg'] = 0.0
    
    return metrics


def _to_numpy_array(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _append_classification_val_metrics(
    val_metrics,
    prefix,
    probs,
    target,
    valid_mask=None,
    class_names=None,
    active_is_nonzero=False,
):
    probs_np = _to_numpy_array(probs)
    target_np = _to_numpy_array(target)
    if probs_np is None or target_np is None:
        return

    probs_np = np.asarray(probs_np)
    target_np = np.asarray(target_np).reshape(-1).astype(np.int64)
    if probs_np.ndim == 1:
        probs_np = probs_np.reshape(1, -1)
    elif probs_np.ndim > 2:
        probs_np = probs_np.reshape(-1, probs_np.shape[-1])
    if probs_np.shape[0] != target_np.shape[0]:
        return

    pred_np = np.argmax(probs_np, axis=-1).astype(np.int64)
    finite_mask = np.isfinite(probs_np).all(axis=-1)
    if valid_mask is not None:
        valid_np = _to_numpy_array(valid_mask)
        valid_np = np.asarray(valid_np).reshape(-1).astype(bool)
        if valid_np.shape[0] != target_np.shape[0]:
            return
        finite_mask &= valid_np
    if not np.any(finite_mask):
        return

    pred_valid = pred_np[finite_mask]
    target_valid = target_np[finite_mask]
    val_metrics[f'{prefix}_acc'].append(float(np.mean(pred_valid == target_valid)))
    val_metrics[f'{prefix}_count'].append(float(target_valid.shape[0]))

    if active_is_nonzero:
        pred_active = pred_valid != 0
        target_active = target_valid != 0
        tp = float(np.sum(pred_active & target_active))
        fp = float(np.sum(pred_active & ~target_active))
        fn = float(np.sum(~pred_active & target_active))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-6)
        val_metrics[f'{prefix}_active_precision'].append(precision)
        val_metrics[f'{prefix}_active_recall'].append(recall)
        val_metrics[f'{prefix}_active_f1'].append(f1)

    if class_names is not None:
        for class_idx, class_name in enumerate(class_names):
            class_mask = target_valid == class_idx
            if np.any(class_mask):
                val_metrics[f'{prefix}_{class_name}_recall'].append(
                    float(np.mean(pred_valid[class_mask] == class_idx))
                )


def _append_boundary_val_metric(val_metrics, prefix, pred, target, valid=None):
    pred_np = _to_numpy_array(pred)
    target_np = _to_numpy_array(target)
    if pred_np is None or target_np is None:
        return

    pred_np = np.asarray(pred_np).reshape(-1).astype(np.float32)
    target_np = np.asarray(target_np).reshape(-1).astype(np.float32)
    if pred_np.shape[0] != target_np.shape[0]:
        return

    finite_mask = np.isfinite(pred_np) & np.isfinite(target_np)
    if valid is not None:
        valid_np = _to_numpy_array(valid)
        if valid_np is None:
            return
        valid_np = np.asarray(valid_np).reshape(-1).astype(bool)
        if valid_np.shape[0] != finite_mask.shape[0]:
            return
        finite_mask &= valid_np
    if not np.any(finite_mask):
        return
    mae = np.abs(pred_np[finite_mask] - target_np[finite_mask])
    val_metrics[f'{prefix}_mae'].append(float(np.mean(mae)))
    val_metrics[f'{prefix}_count'].append(float(np.sum(finite_mask)))


def _append_binary_prob_val_metrics(val_metrics, prefix, prob, target, valid_mask=None, threshold=0.5):
    prob_np = _to_numpy_array(prob)
    target_np = _to_numpy_array(target)
    if prob_np is None or target_np is None:
        return

    prob_np = np.asarray(prob_np).reshape(-1).astype(np.float32)
    target_np = np.asarray(target_np).reshape(-1).astype(np.float32)
    if prob_np.shape[0] != target_np.shape[0]:
        return

    finite_mask = np.isfinite(prob_np) & np.isfinite(target_np)
    if valid_mask is not None:
        valid_np = _to_numpy_array(valid_mask)
        if valid_np is None:
            return
        valid_np = np.asarray(valid_np).reshape(-1).astype(bool)
        if valid_np.shape[0] != finite_mask.shape[0]:
            return
        finite_mask &= valid_np
    if not np.any(finite_mask):
        return

    pred_pos = prob_np[finite_mask] >= threshold
    true_pos = target_np[finite_mask] >= 0.5
    tp = float(np.sum(pred_pos & true_pos))
    fp = float(np.sum(pred_pos & ~true_pos))
    fn = float(np.sum(~pred_pos & true_pos))
    tn = float(np.sum(~pred_pos & ~true_pos))
    denom = max(tp + fp + fn + tn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-6)

    val_metrics[f'{prefix}_acc'].append((tp + tn) / denom)
    val_metrics[f'{prefix}_precision'].append(precision)
    val_metrics[f'{prefix}_recall'].append(recall)
    val_metrics[f'{prefix}_f1'].append(f1)
    val_metrics[f'{prefix}_count'].append(float(np.sum(finite_mask)))


def _append_binary_prob_calibration_metrics(
    val_metrics,
    prefix,
    prob,
    target,
    valid_mask=None,
    thresholds=(0.2, 0.3, 0.4, 0.5),
):
    prob_np = _to_numpy_array(prob)
    target_np = _to_numpy_array(target)
    if prob_np is None or target_np is None:
        return

    prob_np = np.asarray(prob_np).reshape(-1).astype(np.float32)
    target_np = np.asarray(target_np).reshape(-1).astype(np.float32)
    if prob_np.shape[0] != target_np.shape[0]:
        return

    finite_mask = np.isfinite(prob_np) & np.isfinite(target_np)
    if valid_mask is not None:
        valid_np = _to_numpy_array(valid_mask)
        if valid_np is None:
            return
        valid_np = np.asarray(valid_np).reshape(-1).astype(bool)
        if valid_np.shape[0] != finite_mask.shape[0]:
            return
        finite_mask &= valid_np
    if not np.any(finite_mask):
        return

    prob_eval = prob_np[finite_mask]
    target_pos = target_np[finite_mask] >= 0.5
    val_metrics[f'{prefix}_target_pos_rate'].append(float(np.mean(target_pos)))
    val_metrics[f'{prefix}_prob_mean'].append(float(np.mean(prob_eval)))
    if np.any(target_pos):
        val_metrics[f'{prefix}_pos_prob_mean'].append(float(np.mean(prob_eval[target_pos])))
    if np.any(~target_pos):
        val_metrics[f'{prefix}_neg_prob_mean'].append(float(np.mean(prob_eval[~target_pos])))

    for threshold in thresholds:
        pred_pos = prob_eval >= float(threshold)
        tp = float(np.sum(pred_pos & target_pos))
        fp = float(np.sum(pred_pos & ~target_pos))
        fn = float(np.sum(~pred_pos & target_pos))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-6)
        suffix = f'thr{int(round(float(threshold) * 100)):02d}'
        val_metrics[f'{prefix}_{suffix}_precision'].append(precision)
        val_metrics[f'{prefix}_{suffix}_recall'].append(recall)
        val_metrics[f'{prefix}_{suffix}_f1'].append(f1)
        val_metrics[f'{prefix}_{suffix}_pred_pos_rate'].append(float(np.mean(pred_pos)))


def _get_stage1_result(result, suffix):
    """Prefer renamed stage1 outputs while accepting older speed_energy aliases."""
    if f'stage1_{suffix}' in result:
        return result.get(f'stage1_{suffix}')
    return result.get(f'speed_energy_{suffix}')


def _append_new_stage1_val_metrics(val_metrics, batch, result, alias_prefix=None):
    """Evaluate new direct-label stage1 heads on inference outputs."""
    before_counts = {key: len(value) for key, value in val_metrics.items()}

    family = batch.get('conflict_area_family')
    if family is not None and _get_stage1_result(result, 'window_probs') is not None:
        family_np = _to_numpy_array(family).reshape(-1).astype(np.int64)
        # Label order: 0 none, 1 borrow, 2 merge, 3 junction.
        # Window head order: 0 none, 1 merge, 2 junction, 3 borrow.
        window_target = np.zeros_like(family_np)
        window_target[family_np == 2] = 1
        window_target[family_np == 3] = 2
        window_target[family_np == 1] = 3
        _append_classification_val_metrics(
            val_metrics,
            'stage1_window',
            _get_stage1_result(result, 'window_probs'),
            window_target,
            class_names=('none', 'merge', 'junction', 'borrow'),
            active_is_nonzero=True,
        )

    if _get_stage1_result(result, 'dir_probs') is not None and batch.get('conflict_area_dir') is not None:
        dir_target = np.clip(_to_numpy_array(batch['conflict_area_dir']).reshape(-1).astype(np.int64), 0, 3)
        _append_classification_val_metrics(
            val_metrics,
            'stage1_dir',
            _get_stage1_result(result, 'dir_probs'),
            dir_target,
            class_names=('none', 'same', 'opposite', 'cross'),
            active_is_nonzero=True,
        )

    if _get_stage1_result(result, 'decision_phase_probs') is not None and batch.get('conflict_decision_phase') is not None:
        decision_codes = _to_numpy_array(batch['conflict_decision_phase']).reshape(-1).astype(np.int64)
        decision_valid = decision_codes > 0
        decision_target = np.clip(decision_codes - 1, 0, 1)
        _append_classification_val_metrics(
            val_metrics,
            'stage1_decision_phase',
            _get_stage1_result(result, 'decision_phase_probs'),
            decision_target,
            valid_mask=decision_valid,
            class_names=('yld', 'go'),
        )

    if _get_stage1_result(result, 'control_phase_probs') is not None and batch.get('conflict_control_phase') is not None:
        control_codes = _to_numpy_array(batch['conflict_control_phase']).reshape(-1).astype(np.int64)
        control_valid = control_codes > 0
        control_target = np.clip(control_codes - 1, 0, 3)
        _append_classification_val_metrics(
            val_metrics,
            'stage1_control_phase',
            _get_stage1_result(result, 'control_phase_probs'),
            control_target,
            valid_mask=control_valid,
            class_names=('coast_yld', 'slow_yld', 'stop_yld', 'go'),
        )

    temp_probs = _to_numpy_array(_get_stage1_result(result, 'temporary_occupancy_probs'))
    temp_bins = _to_numpy_array(batch.get('temporary_occupancy_cover_bins'))
    temp_valid = _to_numpy_array(batch.get('temporary_occupancy_cover_valid'))
    if temp_probs is not None and temp_bins is not None and temp_valid is not None:
        temp_probs = np.asarray(temp_probs).reshape(-1, np.asarray(temp_probs).shape[-1]).astype(np.float32)
        temp_bins = np.asarray(temp_bins).reshape(temp_probs.shape[0], -1).astype(np.float32)
        temp_valid = np.asarray(temp_valid).reshape(temp_probs.shape[0], -1) > 0.5
        if temp_bins.shape == temp_probs.shape and temp_valid.shape == temp_probs.shape and np.any(temp_valid):
            clipped = np.clip(temp_probs, 1e-5, 1.0 - 1e-5)
            bce = -(temp_bins * np.log(clipped) + (1.0 - temp_bins) * np.log(1.0 - clipped))
            val_metrics['stage1_tempocc_bce'].append(float(np.mean(bce[temp_valid])))
            val_metrics['stage1_tempocc_count'].append(float(np.sum(temp_valid)))

    go_probs = _to_numpy_array(_get_stage1_result(result, 'go_opportunity_probs'))
    yld_target = _to_numpy_array(batch.get('yld_pressure_prob'))
    go_target = _to_numpy_array(batch.get('go_opportunity_prob'))
    go_valid = _to_numpy_array(batch.get('go_opportunity_valid'))
    if go_probs is not None and yld_target is not None and go_target is not None and go_valid is not None:
        go_probs = np.asarray(go_probs).reshape(-1, 2).astype(np.float32)
        target_probs = np.stack([
            np.asarray(yld_target).reshape(-1).astype(np.float32),
            np.asarray(go_target).reshape(-1).astype(np.float32),
        ], axis=-1)
        denom = np.sum(target_probs, axis=-1, keepdims=True)
        target_probs = np.where(denom > 1e-6, target_probs / np.maximum(denom, 1e-6), 0.5)
        go_valid = np.asarray(go_valid).reshape(-1) > 0.5
        if go_probs.shape[0] == target_probs.shape[0] and go_valid.shape[0] == go_probs.shape[0]:
            finite = go_valid & np.isfinite(go_probs).all(axis=-1) & np.isfinite(target_probs).all(axis=-1)
        else:
            finite = np.zeros((0,), dtype=bool)
        if finite.shape[0] == go_probs.shape[0] and np.any(finite):
            clipped = np.clip(go_probs[finite], 1e-5, 1.0)
            ce = -np.sum(target_probs[finite] * np.log(clipped), axis=-1)
            mae = np.abs(go_probs[finite, 1] - target_probs[finite, 1])
            val_metrics['stage1_go_opportunity_ce'].append(float(np.mean(ce)))
            val_metrics['stage1_go_opportunity_mae'].append(float(np.mean(mae)))
            val_metrics['stage1_go_opportunity_count'].append(float(np.sum(finite)))

    if _get_stage1_result(result, 'conflict_area_status_probs') is not None and batch.get('conflict_area_status') is not None:
        status_target = np.clip(
            _to_numpy_array(batch['conflict_area_status']).reshape(-1).astype(np.int64),
            0,
            3,
        )
        _append_classification_val_metrics(
            val_metrics,
            'stage1_conflict_area_status',
            _get_stage1_result(result, 'conflict_area_status_probs'),
            status_target,
            class_names=('none', 'approaching', 'inside', 'past'),
            active_is_nonzero=True,
        )

    timing_specs = (
        ('stage1_conflict_dist_to_entry', 'conflict_dist_to_entry_m', 'conflict_dist_to_entry_m'),
        ('stage1_conflict_dist_to_exit', 'conflict_dist_to_exit_m', 'conflict_dist_to_exit_m'),
        ('stage1_conflict_time_to_entry', 'conflict_time_to_entry_s', 'conflict_time_to_entry_s'),
    )
    timing_valid = None
    for valid_key in ('conflict_timing_valid', 'conflict_area_timing_valid', 'conflict_area_status_valid'):
        if batch.get(valid_key) is not None:
            timing_valid = _to_numpy_array(batch[valid_key]).reshape(-1) > 0.5
            break
    if timing_valid is None and all(batch.get(k) is not None for k in (
        'conflict_dist_to_entry_valid',
        'conflict_dist_to_exit_valid',
        'conflict_time_to_entry_valid',
    )):
        timing_valid = (
            (_to_numpy_array(batch['conflict_dist_to_entry_valid']).reshape(-1) > 0.5)
            & (_to_numpy_array(batch['conflict_dist_to_exit_valid']).reshape(-1) > 0.5)
            & (_to_numpy_array(batch['conflict_time_to_entry_valid']).reshape(-1) > 0.5)
        )
    if timing_valid is None and batch.get('conflict_area_family') is not None:
        timing_valid = _to_numpy_array(batch['conflict_area_family']).reshape(-1).astype(np.int64) > 0
    for metric_prefix, pred_key, target_key in timing_specs:
        pred_np = _to_numpy_array(_get_stage1_result(result, pred_key))
        target_np = _to_numpy_array(batch.get(target_key))
        if pred_np is None or target_np is None:
            continue
        pred_np = np.asarray(pred_np).reshape(-1).astype(np.float32)
        target_np = np.asarray(target_np).reshape(-1).astype(np.float32)
        finite = np.isfinite(pred_np) & np.isfinite(target_np)
        if timing_valid is not None and timing_valid.shape[0] == finite.shape[0]:
            finite &= timing_valid
        if pred_np.shape[0] == target_np.shape[0] and np.any(finite):
            val_metrics[f'{metric_prefix}_mae'].append(float(np.mean(np.abs(pred_np[finite] - target_np[finite]))))
            val_metrics[f'{metric_prefix}_count'].append(float(np.sum(finite)))

    chase_prob = _to_numpy_array(_get_stage1_result(result, 'chase_has_lead_prob'))
    chase_target = _to_numpy_array(batch.get('chase_has_lead'))
    if chase_prob is not None and chase_target is not None:
        chase_prob = np.asarray(chase_prob).reshape(-1).astype(np.float32)
        chase_target = np.asarray(chase_target).reshape(-1).astype(np.float32) > 0.5
        if chase_prob.shape[0] == chase_target.shape[0]:
            finite = np.isfinite(chase_prob)
            if np.any(finite):
                pred_pos = chase_prob[finite] >= 0.5
                true_pos = chase_target[finite]
                tp = float(np.sum(pred_pos & true_pos))
                fp = float(np.sum(pred_pos & ~true_pos))
                fn = float(np.sum(~pred_pos & true_pos))
                tn = float(np.sum(~pred_pos & ~true_pos))
                denom = max(tp + fp + fn + tn, 1.0)
                precision = tp / max(tp + fp, 1.0)
                recall = tp / max(tp + fn, 1.0)
                f1 = 2.0 * precision * recall / max(precision + recall, 1e-6)
                val_metrics['stage1_chase_has_lead_acc'].append((tp + tn) / denom)
                val_metrics['stage1_chase_has_lead_precision'].append(precision)
                val_metrics['stage1_chase_has_lead_recall'].append(recall)
                val_metrics['stage1_chase_has_lead_f1'].append(f1)
                val_metrics['stage1_chase_has_lead_count'].append(float(np.sum(finite)))

    chase_speed = _to_numpy_array(_get_stage1_result(result, 'chase_speed_max_mps'))
    chase_speed_target = _to_numpy_array(batch.get('chase_speed_max'))
    if chase_speed is not None and chase_speed_target is not None:
        chase_speed = np.asarray(chase_speed).reshape(-1).astype(np.float32)
        chase_speed_target = np.asarray(chase_speed_target).reshape(-1).astype(np.float32)
        finite = np.isfinite(chase_speed) & np.isfinite(chase_speed_target)
        if chase_speed.shape[0] == chase_speed_target.shape[0] and np.any(finite):
            val_metrics['stage1_chase_speed_max_mae'].append(
                float(np.mean(np.abs(chase_speed[finite] - chase_speed_target[finite])))
            )
            val_metrics['stage1_chase_speed_max_count'].append(float(np.sum(finite)))

    graph_edge_mode_names = (
        'none',
        'pass_after_current',
        'go_before_future',
        'yield_after_future',
        'ambiguous',
    )
    graph_specs = (
        (
            'stage1_graph_current_cover_edge',
            'current_cover_edge_valid_prob',
            'current_cover_edge_valid',
            'current_cover_edge_mode_probs',
            'current_cover_edge_mode',
            'current_cover_edge_mode_valid',
        ),
        (
            'stage1_graph_future_cover_edge',
            'future_cover_edge_valid_prob',
            'future_cover_edge_valid',
            'future_cover_edge_mode_probs',
            'future_cover_edge_mode',
            'future_cover_edge_mode_valid',
        ),
    )
    for (
        metric_prefix,
        valid_pred_key,
        valid_target_key,
        mode_pred_key,
        mode_target_key,
        mode_valid_key,
    ) in graph_specs:
        _append_binary_prob_val_metrics(
            val_metrics,
            f'{metric_prefix}_valid',
            _get_stage1_result(result, valid_pred_key),
            batch.get(valid_target_key),
        )
        _append_binary_prob_calibration_metrics(
            val_metrics,
            f'{metric_prefix}_valid_calib',
            _get_stage1_result(result, valid_pred_key),
            batch.get(valid_target_key),
        )
        mode_valid = batch.get(mode_valid_key)
        if mode_valid is None:
            mode_valid = batch.get(valid_target_key)
        _append_classification_val_metrics(
            val_metrics,
            f'{metric_prefix}_mode',
            _get_stage1_result(result, mode_pred_key),
            batch.get(mode_target_key),
            valid_mask=mode_valid,
            class_names=graph_edge_mode_names,
            active_is_nonzero=True,
        )

    graph_speed_specs = (
        (
            'stage1_graph_current_cover_upper_speed',
            'current_cover_upper_speed_mps',
            'current_cover_upper_speed_mps',
            'current_cover_upper_speed_valid',
        ),
        (
            'stage1_graph_future_cover_lower_speed',
            'future_cover_lower_speed_mps',
            'future_cover_lower_speed_mps',
            'future_cover_lower_speed_valid',
        ),
        (
            'stage1_graph_front_follow_upper_speed',
            'front_follow_upper_speed_mps',
            'front_follow_upper_speed_mps',
            'front_follow_upper_speed_valid',
        ),
        (
            'stage1_graph_merge_flow_lower_speed',
            'merge_flow_lower_speed_mps',
            'merge_flow_lower_speed_mps',
            'merge_flow_lower_speed_valid',
        ),
    )
    for metric_prefix, pred_key, target_key, valid_key in graph_speed_specs:
        _append_boundary_val_metric(
            val_metrics,
            metric_prefix,
            _get_stage1_result(result, pred_key),
            batch.get(target_key),
            valid=batch.get(valid_key),
        )

    boundary_specs = (
        ('stage1_merge_yld_max', 'merge_yld_max_mps', 'merge_yld_max_speed'),
        ('stage1_merge_go_min', 'merge_go_min_mps', 'merge_go_min_speed'),
        ('stage1_junction_yld_max', 'junction_yld_max_mps', 'junction_yld_max_speed'),
        ('stage1_junction_go_min', 'junction_go_min_mps', 'junction_go_min_speed'),
        ('stage1_borrow_yld_max', 'borrow_yld_max_mps', 'borrow_yld_max_speed'),
        ('stage1_borrow_go_min', 'borrow_go_min_mps', 'borrow_go_min_speed'),
    )
    all_abs_errors = []
    all_valid_counts = []
    for metric_prefix, pred_key, target_key in boundary_specs:
        _append_boundary_val_metric(
            val_metrics,
            metric_prefix,
            _get_stage1_result(result, pred_key),
            batch.get(target_key),
        )
        pred_np = _to_numpy_array(_get_stage1_result(result, pred_key))
        target_np = _to_numpy_array(batch.get(target_key))
        if pred_np is None or target_np is None:
            continue
        pred_np = np.asarray(pred_np).reshape(-1).astype(np.float32)
        target_np = np.asarray(target_np).reshape(-1).astype(np.float32)
        finite_mask = np.isfinite(pred_np) & np.isfinite(target_np)
        if np.any(finite_mask):
            all_abs_errors.append(np.abs(pred_np[finite_mask] - target_np[finite_mask]))
            all_valid_counts.append(float(np.sum(finite_mask)))
    if all_abs_errors:
        merged_errors = np.concatenate(all_abs_errors, axis=0)
        val_metrics['stage1_boundary_mae'].append(float(np.mean(merged_errors)))
        val_metrics['stage1_boundary_count'].append(float(np.sum(all_valid_counts)))

    if alias_prefix:
        for key, values in list(val_metrics.items()):
            if not key.startswith('stage1_'):
                continue
            start = before_counts.get(key, 0)
            if len(values) <= start:
                continue
            alias_key = key.replace('stage1_', f'{alias_prefix}_', 1)
            val_metrics[alias_key].extend(values[start:])


def _ordered_basename_frame_id(path: str) -> Optional[int]:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _ordered_frame_id(sample: Dict[str, Any]) -> Optional[int]:
    for key in ("frame_id", "frame", "tick"):
        if key in sample:
            try:
                return int(sample[key])
            except (TypeError, ValueError):
                pass
    for key in ("transfuser_bev_feature", "bev_feature", "image_path"):
        value = sample.get(key)
        if value:
            frame = _ordered_basename_frame_id(str(value))
            if frame is not None:
                return frame
    return None


def _ordered_route_key(sample: Dict[str, Any]) -> Tuple[str, ...]:
    parts: List[str] = []
    for key in (
        "scene_id",
        "route_id",
        "route_name",
        "scenario_id",
        "town",
        "log_id",
    ):
        value = sample.get(key)
        if value is not None and str(value) != "":
            parts.append(f"{key}={value}")
    feat_rel = str(sample.get("transfuser_bev_feature", "") or "")
    if feat_rel:
        parent = os.path.dirname(os.path.dirname(feat_rel))
        if parent:
            parts.append(f"feat_parent={parent}")
    if not parts:
        parts.append(f"singleton={id(sample)}")
    return tuple(parts)


def _ordered_dataset_groups(dataset):
    """Rebuild route/frame order from packed sample metadata."""
    base_dataset = dataset
    local_to_base = None
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
        local_to_base = list(dataset.indices)
    samples = getattr(base_dataset, 'samples', None)
    if samples is None:
        return None, "dataset has no samples metadata"
    if local_to_base is None:
        local_to_base = list(range(len(base_dataset)))

    grouped = defaultdict(list)
    for local_idx, base_idx in enumerate(local_to_base):
        sample = samples[int(base_idx)]
        frame = _ordered_frame_id(sample)
        if frame is None:
            return None, f"sample {base_idx} has no frame id"
        grouped[_ordered_route_key(sample)].append((frame, local_idx))
    groups = []
    for items in grouped.values():
        items.sort(key=lambda item: item[0])
        groups.append([local_idx for _, local_idx in items])
    groups.sort(key=lambda group: _ordered_route_key(samples[int(local_to_base[group[0]])]))
    return groups, None


def _move_batch_to_device(batch, device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def _build_route_b_obs_dict(batch, model_for_inference):
    obs_dict = {
        'transfuser_bev_feature': batch['transfuser_bev_feature'],
        'transfuser_bev_feature_upsample': batch['transfuser_bev_feature_upsample'],
        'transfuser_lidar_bev': batch['transfuser_lidar_bev'],
        'ego_status': batch['ego_status'][:, :model_for_inference.n_obs_steps],
    }
    if 'borrow_cross_active_time_s' in batch:
        obs_dict['borrow_cross_active_time_s'] = batch['borrow_cross_active_time_s']
    return obs_dict


def _gather_metric_lists(metric_lists, world_size):
    if (
        world_size <= 1
        or not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        return metric_lists
    gathered = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, {k: list(v) for k, v in metric_lists.items()})
    merged = defaultdict(list)
    for item in gathered:
        if not item:
            continue
        for key, values in item.items():
            merged[key].extend(values)
    return merged


def validate_ordered_semantic_rollout(
    policy,
    val_loader,
    device,
    rank=0,
    world_size=1,
    use_amp=False,
    amp_dtype=torch.float16,
    max_routes=0,
    max_frames=0,
):
    """Route-ordered semantic cache rollout validation for next-token state."""
    model_for_inference = policy.module if world_size > 1 else policy
    groups, warning = _ordered_dataset_groups(val_loader.dataset)
    if groups is None:
        if rank == 0:
            print(f"[Val ordered rollout] skipped: {warning}")
        return {}
    if max_routes and max_routes > 0:
        groups = groups[:int(max_routes)]

    local_metrics = defaultdict(list)
    local_groups = [
        (idx, group) for idx, group in enumerate(groups)
        if (idx % max(world_size, 1)) == rank
    ]
    total_frames = sum(len(group) for _, group in local_groups)
    if max_frames and max_frames > 0:
        total_frames = min(total_frames, int(max_frames))
    iterator = tqdm(
        local_groups,
        desc="Ordered semantic rollout val",
        leave=False,
    ) if rank == 0 else local_groups

    frames_seen = 0
    with torch.no_grad():
        for _, group in iterator:
            if hasattr(model_for_inference, 'reset_semantic_state_cache'):
                model_for_inference.reset_semantic_state_cache()
            for frame_pos, dataset_idx in enumerate(group):
                if max_frames and max_frames > 0 and frames_seen >= int(max_frames):
                    break
                batch = default_collate([val_loader.dataset[dataset_idx]])
                batch = _move_batch_to_device(batch, device)
                obs_dict = _build_route_b_obs_dict(batch, model_for_inference)
                target_actions = batch['agent_pos']
                try:
                    with autocast_cuda(use_amp, amp_dtype):
                        result = model_for_inference.predict_action(
                            obs_dict,
                            no_noise=False,
                            use_server_style=False,
                            gt_trajectory=target_actions,
                            reset_semantic_state_cache=(frame_pos == 0),
                            disable_semantic_state_cache=False,
                        )
                    predicted_actions = torch.from_numpy(result['action']).to(device)
                    target_actions_eval = target_actions
                    if target_actions_eval.dim() == 3:
                        target_actions_eval = target_actions_eval[:, :predicted_actions.shape[1]]
                    elif target_actions_eval.dim() == 2:
                        target_actions_eval = target_actions_eval.unsqueeze(1)
                    driving_metrics = compute_driving_metrics(
                        predicted_actions,
                        target_actions_eval,
                        fut_obstacles=batch.get('fut_obstacles', None),
                    )
                    for key, value in driving_metrics.items():
                        local_metrics[f'ordered_{key}'].append(value)
                    _append_new_stage1_val_metrics(
                        local_metrics,
                        batch,
                        result,
                        alias_prefix='ordered_semantic_rollout',
                    )
                    if 'route' in batch and batch['route'] is not None and 'route_pred' in result:
                        route_gt = batch['route'].to(device)
                        route_pred = result['route_pred']
                        route_l2 = torch.sqrt(((route_pred - route_gt) ** 2).sum(dim=-1))
                        local_metrics['ordered_route_L2'].append(route_l2.mean().item())
                        local_metrics['ordered_route_L2_final'].append(route_l2[:, -1].mean().item())
                except Exception as e:
                    if rank == 0:
                        print(f"[Val ordered rollout] sample failed: {e}")
                    continue
                frames_seen += 1
            if max_frames and max_frames > 0 and frames_seen >= int(max_frames):
                break
    if hasattr(model_for_inference, 'reset_semantic_state_cache'):
        model_for_inference.reset_semantic_state_cache()

    merged = _gather_metric_lists(local_metrics, world_size)
    return {
        f'val_{key}': float(np.mean(values))
        for key, values in merged.items()
        if values and (key.startswith('ordered_') or key.startswith('ordered_semantic_rollout_'))
    }


def validate_model(
    policy,
    val_loader,
    device,
    rank=0,
    world_size=1,
    use_amp=False,
    amp_dtype=torch.float16,
    max_batches=None,
    speed_adaptive_json_path=None,
    ordered_semantic_rollout_enabled=True,
    ordered_semantic_rollout_max_routes=0,
    ordered_semantic_rollout_max_frames=0,
):
    """
    Validation function for distributed training
    Only rank 0 will compute and log metrics
    max_batches: limit number of val batches to avoid evicting training page cache
                 None or <=0 means full validation.
    """
    policy.eval()

    # Get the actual model (unwrap DDP if needed)
    model_for_inference = policy.module if world_size > 1 else policy
    route_b_cfg = getattr(model_for_inference, 'route_b_cfg', {}) or {}
    route_b_phase = 'split' if route_b_cfg.get('use_split_forward', False) else 'unified'

    val_metrics = defaultdict(list)
    speed_adaptive_cache = defaultdict(list)

    # All ranks perform validation to avoid NCCL timeout
    # (rank 0 logs metrics, others just run forward to stay in sync)
    with torch.no_grad():
        if max_batches is not None and max_batches <= 0:
            max_batches = None
        total_batches = min(len(val_loader), max_batches) if max_batches is not None else len(val_loader)
        pbar = tqdm(val_loader, desc="Validating", leave=False, total=total_batches) if rank == 0 else val_loader

        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = _move_batch_to_device(batch, device)

            with autocast_cuda(use_amp, amp_dtype):
                loss_dict = model_for_inference(batch, return_loss_dict=True, phase=route_b_phase)
                loss = loss_dict['total_loss']
            if rank == 0:
                val_metrics['loss'].append(loss.item())
                val_metrics['cls_loss'].append(loss_dict['cls_loss'].item())
                val_metrics['reg_loss'].append(loss_dict['reg_loss'].item())
                val_metrics['route_loss'].append(loss_dict['route_loss'].item())
                if 'stage1_loss' in loss_dict:
                    val_metrics['stage1_loss'].append(loss_dict['stage1_loss'].item())
                    val_metrics['energy_loss'].append(loss_dict['stage1_loss'].item())
                    val_metrics['alignment_loss'].append(loss_dict['alignment_loss'].item())
                elif 'energy_loss' in loss_dict:
                    val_metrics['energy_loss'].append(loss_dict['energy_loss'].item())
                    val_metrics['alignment_loss'].append(loss_dict['alignment_loss'].item())
                if 'energy_front_loss' in loss_dict:
                    val_metrics['energy_front_loss'].append(loss_dict['energy_front_loss'].item())
                if 'energy_chase_loss' in loss_dict:
                    val_metrics['energy_chase_loss'].append(loss_dict['energy_chase_loss'].item())
                if 'energy_left_loss' in loss_dict:
                    val_metrics['energy_left_loss'].append(loss_dict['energy_left_loss'].item())
                if 'energy_merge_loss' in loss_dict:
                    val_metrics['energy_merge_loss'].append(loss_dict['energy_merge_loss'].item())
                if 'energy_ped_loss' in loss_dict:
                    val_metrics['energy_ped_loss'].append(loss_dict['energy_ped_loss'].item())
                if 'energy_pedestrian_loss' in loss_dict:
                    val_metrics['energy_pedestrian_loss'].append(loss_dict['energy_pedestrian_loss'].item())
                if 'energy_right_loss' in loss_dict:
                    val_metrics['energy_right_loss'].append(loss_dict['energy_right_loss'].item())
                if 'energy_cross_loss' in loss_dict:
                    val_metrics['energy_cross_loss'].append(loss_dict['energy_cross_loss'].item())
                if 'energy_off_loss' in loss_dict:
                    val_metrics['energy_off_loss'].append(loss_dict['energy_off_loss'].item())
                if 'energy_route_loss' in loss_dict:
                    val_metrics['energy_route_loss'].append(loss_dict['energy_route_loss'].item())
                if 'speed_loss' in loss_dict:
                    val_metrics['speed_loss'].append(loss_dict['speed_loss'].item())
                if 'speed_profile_loss' in loss_dict:
                    val_metrics['speed_profile_loss'].append(loss_dict['speed_profile_loss'].item())
                for key in (
                    'stage1_merge_yld_loss',
                    'stage1_merge_go_loss',
                    'stage1_junction_yld_loss',
                    'stage1_junction_go_loss',
                    'stage1_borrow_yld_loss',
                    'stage1_borrow_go_loss',
                    'stage1_cross_yld_loss',
                    'stage1_cross_go_loss',
                    'stage1_merge_active_loss',
                    'stage1_junction_active_loss',
                    'stage1_borrow_active_loss',
                    'stage1_cross_active_loss',
                    'stage1_dir_loss',
                    'stage1_conflict_area_loss',
                    'stage1_window_loss',
                    'stage1_phase_loss',
                    'stage1_decision_phase_loss',
                    'stage1_control_phase_loss',
                    'stage1_temporary_occupancy_loss',
                    'stage1_go_opportunity_loss',
                    'stage1_conflict_area_status_loss',
                    'stage1_conflict_timing_loss',
                    'stage1_inside_area_go_loss',
                    'stage1_chase_loss',
                    'stage1_chase_has_lead_loss',
                    'stage1_chase_speed_max_loss',
                    'stage1_state_consistency_loss',
                    'stage1_state_consistency_window_loss',
                    'stage1_state_consistency_phase_loss',
                    'stage1_state_consistency_timing_loss',
                    'stage1_state_consistency_boundary_loss',
                    'stage1_state_consistency_area_loss',
                    'stage1_state_consistency_tempocc_loss',
                    'stage1_state_consistency_opportunity_loss',
                    'stage1_semantic_direct_aux_loss',
                    'stage1_semantic_next_token_loss',
                    'stage1_semantic_transition_loss',
                    'stage1_semantic_transition_consistency_loss',
                    'stage1_merge_yld_max_loss',
                    'stage1_merge_go_min_loss',
                    'stage1_junction_yld_max_loss',
                    'stage1_junction_go_min_loss',
                    'stage1_borrow_yld_max_loss',
                    'stage1_borrow_go_min_loss',
                    'energy_merge_yld_loss',
                    'energy_merge_go_loss',
                    'energy_junction_yld_loss',
                    'energy_junction_go_loss',
                    'energy_borrow_yld_loss',
                    'energy_borrow_go_loss',
                    'energy_cross_yld_loss',
                    'energy_cross_go_loss',
                    'energy_merge_active_loss',
                    'energy_junction_active_loss',
                    'energy_borrow_active_loss',
                    'energy_cross_active_loss',
                    'energy_dir_loss',
                    'energy_conflict_area_loss',
                    'energy_window_loss',
                    'energy_phase_loss',
                    'energy_decision_phase_loss',
                    'energy_control_phase_loss',
                    'energy_temporary_occupancy_loss',
                    'energy_go_opportunity_loss',
                    'energy_conflict_area_status_loss',
                    'energy_conflict_timing_loss',
                    'energy_inside_area_go_loss',
                    'energy_chase_loss',
                    'energy_chase_has_lead_loss',
                    'energy_chase_speed_max_loss',
                    'energy_state_consistency_loss',
                    'energy_state_consistency_window_loss',
                    'energy_state_consistency_phase_loss',
                    'energy_state_consistency_timing_loss',
                    'energy_state_consistency_boundary_loss',
                    'energy_state_consistency_area_loss',
                    'energy_state_consistency_tempocc_loss',
                    'energy_state_consistency_opportunity_loss',
                    'energy_merge_yld_max_loss',
                    'energy_merge_go_min_loss',
                    'energy_junction_yld_max_loss',
                    'energy_junction_go_min_loss',
                    'energy_borrow_yld_max_loss',
                    'energy_borrow_go_min_loss',
                ):
                    if key in loss_dict:
                        val_metrics[key].append(loss_dict[key].item())
                for key, value in loss_dict.items():
                    if key.startswith('speed_profile_step') and key.endswith('_loss'):
                        val_metrics[key].append(value.item() if isinstance(value, torch.Tensor) else value)
                if 'stage1_semantic_transition_loss' in loss_dict:
                    val_metrics['gtprev_semantic_transition_loss'].append(
                        loss_dict['stage1_semantic_transition_loss'].item()
                    )
                
                # Route B model only needs BEV/detail features plus ego status.
                obs_dict = _build_route_b_obs_dict(batch, model_for_inference)
                target_actions = batch['agent_pos']
                
                try:
                    # Model always returns route prediction
                    result = model_for_inference.predict_action(
                        obs_dict,
                        no_noise=False,
                        use_server_style=False,
                        gt_trajectory=target_actions,
                        disable_semantic_state_cache=True,
                    )
                    predicted_actions = torch.from_numpy(result['action']).to(device)

                    target_actions_eval = target_actions
                    if target_actions_eval.dim() == 3:  # (B, T, 2)
                        target_actions_eval = target_actions_eval[:, :predicted_actions.shape[1]]
                    elif target_actions_eval.dim() == 2:  # (B, 2)
                        target_actions_eval = target_actions_eval.unsqueeze(1)  # (B, 1, 2)

                    fut_obstacles = batch.get('fut_obstacles', None)

                    driving_metrics = compute_driving_metrics(
                        predicted_actions,
                        target_actions_eval,
                        fut_obstacles=fut_obstacles
                    )
                    for key, value in driving_metrics.items():
                        val_metrics[key].append(value)
                    _append_new_stage1_val_metrics(
                        val_metrics,
                        batch,
                        result,
                        alias_prefix='semantic_direct',
                    )

                    target_speed = result.get('target_speed', None)
                    if target_speed is not None and target_actions_eval.shape[1] >= 3:
                        gt_speed = torch.norm(
                            target_actions_eval[:, 2] - target_actions_eval[:, 0], dim=-1
                        ).detach().cpu().numpy()
                        current_speed_hist = batch.get('speed', None)
                        if current_speed_hist is not None:
                            current_speed = current_speed_hist[:, -1].detach().cpu().numpy()
                        else:
                            current_speed = np.zeros_like(gt_speed)

                        pred_traj_np = predicted_actions.detach().cpu().numpy()
                        traj_1s = np.linalg.norm(pred_traj_np[:, 2] - pred_traj_np[:, 0], axis=-1)
                        traj_05 = np.linalg.norm(pred_traj_np[:, 1] - pred_traj_np[:, 0], axis=-1) * 2.0
                        speed_head = np.asarray(target_speed).reshape(-1)
                        stacked = np.stack([speed_head, traj_1s, traj_05], axis=1)
                        mean3 = stacked.mean(axis=1)
                        median3 = np.median(stacked, axis=1)

                        masks = _speed_regime_masks(median3, current_speed)
                        adaptive_current = np.empty_like(mean3)
                        for regime in REGIME_ORDER:
                            if np.any(masks[regime]):
                                adaptive_current[masks[regime]] = stacked[masks[regime]] @ np.asarray(
                                    CURRENT_ADAPTIVE_WEIGHTS[regime], dtype=np.float32
                                )

                        valid = np.isfinite(gt_speed)
                        valid &= np.isfinite(speed_head)
                        valid &= np.isfinite(traj_1s)
                        valid &= np.isfinite(traj_05)
                        valid &= np.isfinite(mean3)
                        valid &= np.isfinite(median3)
                        valid &= np.isfinite(adaptive_current)
                        if np.any(valid):
                            speed_adaptive_cache['gt_speed'].append(gt_speed[valid])
                            speed_adaptive_cache['current_speed'].append(current_speed[valid])
                            speed_adaptive_cache['speed_head'].append(speed_head[valid])
                            speed_adaptive_cache['traj_1s'].append(traj_1s[valid])
                            speed_adaptive_cache['traj_05'].append(traj_05[valid])
                            speed_adaptive_cache['mean3'].append(mean3[valid])
                            speed_adaptive_cache['median3'].append(median3[valid])
                            speed_adaptive_cache['adaptive_current'].append(adaptive_current[valid])

                    # Also log 1-step DDIM L2 metrics for direct comparison
                    # Support both Route A (num_diffusion_steps) and Route B (num_inference_steps)
                    steps_attr = None
                    if hasattr(model_for_inference, 'num_diffusion_steps'):
                        steps_attr = 'num_diffusion_steps'
                    elif hasattr(model_for_inference, 'num_inference_steps'):
                        steps_attr = 'num_inference_steps'
                    if steps_attr is not None:
                        original_num_steps = getattr(model_for_inference, steps_attr)
                        try:
                            setattr(model_for_inference, steps_attr, 1)
                            result_1step = model_for_inference.predict_action(
                                obs_dict,
                                no_noise=False,
                                use_server_style=False,
                                gt_trajectory=target_actions,
                                disable_semantic_state_cache=True,
                            )
                            predicted_actions_1step = torch.from_numpy(result_1step['action']).to(device)

                            target_actions_1step = target_actions
                            if target_actions_1step.dim() == 3:
                                target_actions_1step = target_actions_1step[:, :predicted_actions_1step.shape[1]]
                            elif target_actions_1step.dim() == 2:
                                target_actions_1step = target_actions_1step.unsqueeze(1)

                            driving_metrics_1step = compute_driving_metrics(
                                predicted_actions_1step,
                                target_actions_1step,
                                fut_obstacles=fut_obstacles
                            )
                            for key, value in driving_metrics_1step.items():
                                val_metrics[f'{key}_1step'].append(value)
                        finally:
                            setattr(model_for_inference, steps_attr, original_num_steps)
                    
                    # Compute route prediction metrics if route ground truth is available
                    if 'route' in batch and batch['route'] is not None and 'route_pred' in result:
                        try:
                            route_gt = batch['route'].to(device)  # (B, num_waypoints, 2)
                            route_pred = result['route_pred']  # Already a tensor (B, num_waypoints, 2)
                            
                            # Compute L2 distance per waypoint and average
                            route_l2 = torch.sqrt(((route_pred - route_gt) ** 2).sum(dim=-1))  # (B, num_waypoints)
                            route_l2_mean = route_l2.mean().item()  # Average over all waypoints and batch
                            route_l2_final = route_l2[:, -1].mean().item()  # L2 at final waypoint
                            
                            val_metrics['route_L2'].append(route_l2_mean)
                            val_metrics['route_L2_final'].append(route_l2_final)
                        except Exception:
                            pass

                    pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                except Exception as e:
                    if batch_idx == 0:
                        import traceback
                        print(f"\n[Val] predict_action failed: {e}")
                        traceback.print_exc()
                    continue
        
        if rank == 0:
            pbar.close()

    # Compute averaged metrics
    averaged_metrics = {f'val_{k}': np.mean(v) for k, v in val_metrics.items() if v}

    if rank == 0 and speed_adaptive_cache.get('gt_speed'):
        gt_all = np.concatenate(speed_adaptive_cache['gt_speed'])
        current_speed_all = np.concatenate(speed_adaptive_cache['current_speed'])
        speed_head_all = np.concatenate(speed_adaptive_cache['speed_head'])
        traj_1s_all = np.concatenate(speed_adaptive_cache['traj_1s'])
        traj_05_all = np.concatenate(speed_adaptive_cache['traj_05'])
        mean3_all = np.concatenate(speed_adaptive_cache['mean3'])
        median3_all = np.concatenate(speed_adaptive_cache['median3'])
        adaptive_current_all = np.concatenate(speed_adaptive_cache['adaptive_current'])

        sources = {
            'speed_head': speed_head_all,
            'traj_1s': traj_1s_all,
            'traj_05': traj_05_all,
            'mean3': mean3_all,
            'median3': median3_all,
            'adaptive_current': adaptive_current_all,
        }
        regime_masks = _speed_regime_masks(median3_all, current_speed_all)

        for key, value in sources.items():
            averaged_metrics[f'val_speed_mae_{key}'] = _mae(value, gt_all)
            averaged_metrics[f'val_speed_bias_{key}'] = float(np.mean(value - gt_all))

        summary = {
            'num_samples': int(gt_all.shape[0]),
            'source_order': ['speed_head', 'traj_1s', 'traj_05'],
            'current_adaptive_weights': CURRENT_ADAPTIVE_WEIGHTS,
            'regime_thresholds': {
                'rough_threshold': 2.5,
                'high_threshold': 10.0,
                'startup_speed_threshold': 0.2,
            },
            'overall_mae': {key: _mae(value, gt_all) for key, value in sources.items()},
            'overall_bias': {key: float(np.mean(value - gt_all)) for key, value in sources.items()},
            'regimes': {},
        }

        for regime in REGIME_ORDER:
            mask = regime_masks[regime]
            count = int(mask.sum())
            averaged_metrics[f'val_speed_count_{regime}'] = float(count)
            item = {'count': count, 'current_weights': CURRENT_ADAPTIVE_WEIGHTS[regime]}
            if count > 0:
                item['source_mae'] = {key: _mae(value[mask], gt_all[mask]) for key, value in sources.items()}
                for key, value in sources.items():
                    averaged_metrics[f'val_speed_mae_{key}_{regime}'] = _mae(value[mask], gt_all[mask])
            else:
                item['source_mae'] = {}
            summary['regimes'][regime] = item

        if speed_adaptive_json_path:
            os.makedirs(os.path.dirname(speed_adaptive_json_path), exist_ok=True)
            with open(speed_adaptive_json_path, 'w') as f:
                json.dump(summary, f, indent=2)

    if ordered_semantic_rollout_enabled:
        ordered_metrics = validate_ordered_semantic_rollout(
            policy,
            val_loader,
            device,
            rank=rank,
            world_size=world_size,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_routes=ordered_semantic_rollout_max_routes,
            max_frames=ordered_semantic_rollout_max_frames,
        )
        averaged_metrics.update(ordered_metrics)

    return averaged_metrics


def _print_validation_metrics(val_metrics, show_speed_metrics=False):
    print(f"Validation metrics: (total {len(val_metrics)} metrics)")
    l2_keys = [
        'val_ADE', 'val_L2_1s', 'val_L2_2s', 'val_L2_3s', 'val_L2_avg',
        'val_ADE_1step', 'val_L2_1s_1step', 'val_L2_2s_1step', 'val_L2_3s_1step', 'val_L2_avg_1step'
    ]
    for key in l2_keys:
        if key in val_metrics:
            tag = " (1-step)" if "_1step" in key else ""
            print(f"  >>> {key}: {val_metrics[key]:.4f}{tag}")

    for key, value in val_metrics.items():
        if key in l2_keys:
            continue
        if (not show_speed_metrics) and key.startswith('val_speed_') and (not key.endswith('_loss')):
            continue
        print(f"  {key}: {value:.4f}")


def _json_safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(value):
        return None
    return value


def _write_validation_metrics_artifacts(checkpoint_dir, epoch, train_loss, val_metrics):
    """Persist one validation row per checkpoint epoch for offline ckpt selection."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch_1based = int(epoch) + 1
    metrics_clean = {k: _json_safe_float(v) for k, v in sorted(val_metrics.items())}
    payload = {
        'epoch': int(epoch),
        'epoch_1based': epoch_1based,
        'checkpoint': f'dit_policy_epoch{epoch_1based}.pt',
        'train_loss': _json_safe_float(train_loss),
        'val_metrics': metrics_clean,
    }

    epoch_json = os.path.join(checkpoint_dir, f'val_metrics_epoch{epoch_1based:04d}.json')
    latest_json = os.path.join(checkpoint_dir, 'val_metrics_latest.json')
    for path in (epoch_json, latest_json):
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    csv_path = os.path.join(checkpoint_dir, 'val_metrics_summary.csv')
    row = {
        'epoch': int(epoch),
        'epoch_1based': epoch_1based,
        'checkpoint': f'dit_policy_epoch{epoch_1based}.pt',
        'train_loss': _json_safe_float(train_loss),
    }
    row.update(metrics_clean)

    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    rows = [r for r in rows if str(r.get('epoch_1based')) != str(epoch_1based)]
    rows.append(row)
    rows.sort(key=lambda r: int(float(r.get('epoch_1based', 0) or 0)))

    base_fields = ['epoch', 'epoch_1based', 'checkpoint', 'train_loss']
    metric_fields = sorted({key for r in rows for key in r.keys()} - set(base_fields))
    fieldnames = base_fields + metric_fields
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    return epoch_json, csv_path

@record  # Records error and tracebacks in case of failure
def train_pdm_policy(config_path, resume_path=None, val_only=False, init_checkpoint_path=None):
    """
    Multi-GPU distributed training for PDM policy

    Args:
        config_path: 配置文件路径
        resume_path: checkpoint路径，用于恢复训练或只跑验证
        val_only: 如果为True，只跑验证不训练（需要配合resume_path使用）
    """
    if val_only and resume_path is None:
        raise ValueError("--val_only requires --resume to specify a checkpoint")

    torch.cuda.empty_cache()
    
    config = load_config(config_path=config_path)

    # Initialize distributed training
    rank = int(os.environ.get('RANK', 0))  # Rank across all processes
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  # Rank on Node
    world_size = int(os.environ.get('WORLD_SIZE', 1))  # Number of processes
    
    # Single GPU fallback
    if world_size == 1:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(f'cuda:{local_rank}')
        
        # Initialize process group
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=15)
        )
        
        torch.cuda.set_device(device)
    
    # Enable performance optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    
    if rank == 0:
        print(f'Rank: {rank}, Device: {device}, World size: {world_size}')
    
    # Only rank 0 should initialize wandb
    wandb_mode = os.environ.get('WANDB_MODE', 'offline') 
    use_wandb = config.get('logging', {}).get('use_wandb', True) and (rank == 0)
    
    if use_wandb:
        try:
            logging_cfg = config.get('logging', {})
            wandb_entity = logging_cfg.get('wandb_entity')
            wandb_api_key = logging_cfg.get('wandb_api_key')

            if wandb_api_key:
                # 优先通过环境变量注入，然后尝试login，这样在CI中也能工作
                os.environ['WANDB_API_KEY'] = str(wandb_api_key)
                try:
                    wandb.login(key=str(wandb_api_key))
                except Exception as e:
                    print(f"⚠ WandB login failed: {e}")

            init_kwargs = dict(
                project=logging_cfg.get('wandb_project', "carla-diffusion-policy"),
                name=logging_cfg.get('run_name', "carla_dit_full_validation"),
                mode=wandb_mode,
                resume='allow',
                config={
                    "learning_rate": config.get('optimizer', {}).get('lr', 5e-5),
                    "epochs": config.get('training', {}).get('num_epochs', 50),
                    "batch_size": config.get('dataloader', {}).get('batch_size', 16),
                    "obs_horizon": config.get('obs_horizon', 2),
                    "action_horizon": config.get('action_horizon', 4),
                    "pred_horizon": config.get('pred_horizon', 8),
                    "dataset_path": config.get('training', {}).get('dataset_path', ""),
                    "max_files": None,
                    "train_split": 0.8,
                    "weight_decay": config.get('optimizer', {}).get('weight_decay', 1e-5),
                    "num_workers": config.get('dataloader', {}).get('num_workers', 4)
                }
            )

            # 如果config里指定了entity（账号/组织），把它传给wandb.init
            if wandb_entity:
                init_kwargs['entity'] = wandb_entity

            wandb.init(**init_kwargs)
        except Exception as e:
            print(f"⚠ WandB initialization failed: {e}")
            use_wandb = False

    # dataset
    dataset_path_root = config.get('training', {}).get('dataset_path')
    train_dataset_path = os.path.join(dataset_path_root, 'train')
    val_dataset_path = os.path.join(dataset_path_root, 'val')
    image_data_root = config.get('training', {}).get('image_data_root')
    
    use_per_frame = config.get('dataset', {}).get('use_per_frame', False)
    cache_dir = config.get('dataset', {}).get('cache_dir', None)  # e.g. /tmp/tmp_data for tmpfs
    train_filter_bad_routes = config.get('dataset', {}).get('train_filter_bad_routes', True)
    val_filter_bad_routes = config.get('dataset', {}).get('val_filter_bad_routes', True)
    train_retain_bad_routes_for_energy = config.get('dataset', {}).get('train_retain_bad_routes_for_energy', False)
    val_retain_bad_routes_for_energy = config.get('dataset', {}).get('val_retain_bad_routes_for_energy', False)
    use_fullres_upsample_cache = config.get('dataset', {}).get('use_fullres_upsample_cache', True)
    train_warmup_memmap_page_cache = bool(config.get('dataset', {}).get('train_warmup_memmap_page_cache', False))
    train_warmup_lidar_page_cache = bool(config.get('dataset', {}).get('train_warmup_lidar_page_cache', False))
    train_warmup_max_samples = config.get('dataset', {}).get('train_warmup_max_samples', None)
    if train_warmup_max_samples in (None, 0, "0"):
        train_warmup_max_samples = None
    else:
        train_warmup_max_samples = int(train_warmup_max_samples)
    stage_feature_cache_to_ram = bool(config.get('dataset', {}).get('stage_feature_cache_to_ram', False))
    cache_source_dir = config.get('dataset', {}).get(
        'cache_source_dir',
        os.path.join(image_data_root, 'tmp_data') if image_data_root else None,
    )
    use_lidar_bev_detail = config.get('route_b', {}).get('use_lidar_bev_detail', False)

    policy_type = config.get('policy_type', 'anchor_free')
    if policy_type != 'anchor_free':
        raise ValueError(
            f"This training entry is Route B-only after cleanup; expected policy_type='anchor_free', "
            f"got {policy_type!r}."
        )

    gps_noise_cfg = config.get('augmentation', {}).get('gps_noise', {})
    route_b_cfg = config.get('route_b', {})
    policy_cfg = config.get('policy', {})
    lidar_history_frames = max(
        int(route_b_cfg.get('lidar_history_frames', policy_cfg.get('ego_status_seq_len', config.get('obs_horizon', 1)))),
        1,
    )

    feature_suffix = config.get('dataset', {}).get('feature_suffix', '')
    if stage_feature_cache_to_ram:
        if not cache_dir:
            raise ValueError("dataset.stage_feature_cache_to_ram=true requires dataset.cache_dir")
        if cache_source_dir is None:
            raise ValueError("dataset.stage_feature_cache_to_ram=true requires dataset.cache_source_dir")
        if rank == 0:
            _stage_feature_cache_to_ram(
                source_dir=cache_source_dir,
                target_dir=cache_dir,
                feature_suffix=feature_suffix,
                use_fullres_upsample_cache=use_fullres_upsample_cache,
                include_lidar=use_lidar_bev_detail,
                rank=rank,
            )
        if world_size > 1:
            torch.distributed.barrier()
    validation_cfg = config.get('validation', {})
    validation_enabled = validation_cfg.get('enabled', True)
    validation_use_memmap = validation_cfg.get('use_memmap', True)
    validation_preload_to_ram = validation_cfg.get('preload_to_ram', not validation_use_memmap)
    build_validation = validation_enabled or val_only

    train_dataset = CARLAImageDataset(
        dataset_path=train_dataset_path, image_data_root=image_data_root,
        mode='train',
        use_per_frame=use_per_frame,
        cache_dir=cache_dir, feature_suffix=feature_suffix,
        gps_noise_cfg=gps_noise_cfg,
        load_transfuser_lidar_bev=use_lidar_bev_detail,
        lidar_history_frames=lidar_history_frames,
        filter_bad_routes=train_filter_bad_routes,
        retain_bad_routes_for_energy=train_retain_bad_routes_for_energy,
        use_fullres_upsample_cache=use_fullres_upsample_cache,
    )
    val_dataset_orig = None
    val_dataset = None
    if build_validation:
        val_dataset_orig = CARLAImageDataset(
            dataset_path=val_dataset_path, image_data_root=image_data_root,
            mode='val',
            skip_memmap=not validation_use_memmap,
            use_per_frame=use_per_frame,
            cache_dir=cache_dir, feature_suffix=feature_suffix,
            load_transfuser_lidar_bev=use_lidar_bev_detail,
            lidar_history_frames=lidar_history_frames,
            filter_bad_routes=val_filter_bad_routes,
            retain_bad_routes_for_energy=val_retain_bad_routes_for_energy,
            use_fullres_upsample_cache=use_fullres_upsample_cache,
        )
        val_dataset = val_dataset_orig

    if rank == 0:
        print(f"\nTraining samples: {len(train_dataset)}")
        if val_dataset is not None:
            print(f"Validation samples: {len(val_dataset)}" + (" (train+val combined)" if val_only else ""))
        else:
            print("Validation: disabled by config (validation.enabled=false)")
    

    
    dataloader_cfg = config.get('dataloader', {})
    training_cfg = config.get('training', {})
    train_batch_size = dataloader_cfg.get('batch_size', 32)
    val_batch_size = dataloader_cfg.get('val_batch_size', train_batch_size)

    base_num_workers = dataloader_cfg.get('num_workers', 4)
    base_persistent_workers = dataloader_cfg.get('persistent_workers', True)
    base_prefetch_factor = dataloader_cfg.get('prefetch_factor', 2)
    base_pin_memory = dataloader_cfg.get('pin_memory', True)

    train_num_workers = dataloader_cfg.get('train_num_workers', base_num_workers)
    val_num_workers = dataloader_cfg.get('val_num_workers', base_num_workers)
    train_persistent_workers = dataloader_cfg.get('train_persistent_workers', base_persistent_workers)
    val_persistent_workers = dataloader_cfg.get('val_persistent_workers', False)
    train_prefetch_factor = dataloader_cfg.get('train_prefetch_factor', base_prefetch_factor)
    val_prefetch_factor = dataloader_cfg.get('val_prefetch_factor', 1)
    train_pin_memory = dataloader_cfg.get('train_pin_memory', base_pin_memory)
    val_pin_memory = dataloader_cfg.get('val_pin_memory', False)
    use_route_group_sampler = dataloader_cfg.get('use_route_group_sampler', False)
    use_window_weighted_sampler = bool(dataloader_cfg.get('use_window_weighted_sampler', False))
    window_sampler_replacement = bool(dataloader_cfg.get('window_sampler_replacement', True))
    window_sampler_seed = int(dataloader_cfg.get('window_sampler_seed', 0))
    train_sample_weights = None
    if use_window_weighted_sampler:
        train_sample_weights, weight_summary = build_window_sample_weights(train_dataset, dataloader_cfg)
        if rank == 0:
            counts = weight_summary['counts']
            shift_counts = weight_summary.get('shift_counts', {})
            weights_cfg = weight_summary['weights']
            print(
                "Using window-aware weighted sampler: "
                f"counts={counts}, shift_counts={shift_counts}, weights={weights_cfg}, "
                f"mean_weight={weight_summary['mean_weight']:.3f}, "
                f"max_weight={weight_summary['max_weight']:.3f}, "
                f"replacement={window_sampler_replacement}"
            )
            if use_route_group_sampler:
                print("window-aware weighted sampler is enabled; route-grouped sampler will be ignored.")

    validation_freq = int(validation_cfg.get('freq', training_cfg.get('validation_freq', 1)))
    raw_val_max_batches = validation_cfg.get('max_batches', 16)
    if raw_val_max_batches in (None, 0, "0"):
        val_max_batches = None
    else:
        val_max_batches = int(raw_val_max_batches)
        if val_max_batches <= 0:
            val_max_batches = None
    ordered_semantic_rollout_enabled = bool(
        validation_cfg.get('ordered_semantic_rollout_enabled', True)
    )
    ordered_semantic_rollout_max_routes = int(
        validation_cfg.get('ordered_semantic_rollout_max_routes', 0) or 0
    )
    ordered_semantic_rollout_max_frames = int(
        validation_cfg.get('ordered_semantic_rollout_max_frames', 0) or 0
    )

    val_subset_indices = None
    raw_val_subset_size = validation_cfg.get('subset_size', None)
    if val_dataset_orig is not None and raw_val_subset_size not in (None, 0, "0"):
        val_subset_size = int(raw_val_subset_size)
        if val_subset_size > 0 and val_subset_size < len(val_dataset_orig):
            val_subset_indices = np.linspace(
                0, len(val_dataset_orig) - 1, num=val_subset_size, dtype=np.int64
            ).tolist()
            val_dataset = torch.utils.data.Subset(val_dataset_orig, val_subset_indices)
            if rank == 0:
                print(
                    f"Validation subset enabled: {len(val_subset_indices)}/{len(val_dataset_orig)} "
                    f"samples (deterministic evenly spaced selection)"
                )

    # Optionally inject val features from train's memmap into RAM (old-HPC path).
    _max_val_per_rank = val_max_batches * val_batch_size if val_max_batches else None
    if not use_per_frame and val_dataset_orig is not None and validation_preload_to_ram:
        preload_indices = None
        if val_subset_indices is not None:
            preload_indices = val_subset_indices[rank::world_size] if world_size > 1 else val_subset_indices
        val_dataset_orig.inject_ram_features(
            train_dataset,
            rank=rank,
            world_size=world_size,
            max_val_samples=_max_val_per_rank,
            sample_indices=preload_indices,
        )
    elif rank == 0 and val_dataset_orig is not None:
        if validation_use_memmap and not use_per_frame:
            print("Validation features: using packed samples + memmap directly.")
        elif use_per_frame:
            print("Validation features: using per-frame loading.")

    if train_warmup_memmap_page_cache:
        if rank == 0:
            train_dataset.warmup_train_memmap_page_cache(
                rank=rank,
                include_lidar=train_warmup_lidar_page_cache,
                max_samples=train_warmup_max_samples,
            )
        if world_size > 1:
            torch.distributed.barrier()

    def safe_collate(batch):
        try:
            return default_collate(batch)
        except (RuntimeError, KeyError, AttributeError) as e:
            # Print mismatched shapes for quick diagnosis
            key_sets = [set(b.keys()) for b in batch if isinstance(b, dict)]
            if key_sets:
                shared_keys = set.intersection(*key_sets)
                union_keys = set.union(*key_sets)
                missing_keys = sorted(union_keys - shared_keys)
                if missing_keys:
                    print(f"[COLLATE] missing keys across batch: {missing_keys}", flush=True)
            for key in batch[0]:
                vals = [b[key] for b in batch if isinstance(b.get(key), torch.Tensor)]
                if vals:
                    shapes = set(v.shape for v in vals)
                    if len(shapes) > 1:
                        print(f"[COLLATE] shape mismatch '{key}': {shapes}", flush=True)
                types = {type(b.get(key)).__name__ for b in batch if isinstance(b, dict) and key in b}
                if len(types) > 1:
                    print(f"[COLLATE] type mismatch '{key}': {sorted(types)}", flush=True)
            raise

    def print_nonfinite_loss_debug(loss_dict, batch, batch_idx, rank):
        if rank != 0:
            return
        print(f"[NONFINITE] total_loss became non-finite at batch {batch_idx}", flush=True)
        scalar_items = []
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value_detached = value.detach().float()
                scalar_items.append(
                    (
                        key,
                        float(value_detached.item()) if torch.isfinite(value_detached) else str(value_detached.item()),
                        bool(torch.isfinite(value_detached).item()),
                    )
                )
        scalar_items.sort(key=lambda item: item[0])
        for key, value_repr, is_finite in scalar_items:
            status = "finite" if is_finite else "NONFINITE"
            print(f"[NONFINITE] loss[{key}]={value_repr} ({status})", flush=True)

        debug_batch_keys = (
            'conflict_area_family',
            'conflict_area_dir',
            'conflict_decision_phase',
            'conflict_control_phase',
            'merge_yld_max_speed',
            'merge_go_min_speed',
            'junction_yld_max_speed',
            'junction_go_min_speed',
            'borrow_yld_max_speed',
            'borrow_go_min_speed',
            'merge_yld_max_speed_valid',
            'merge_go_min_speed_valid',
            'junction_yld_max_speed_valid',
            'junction_go_min_speed_valid',
            'borrow_yld_max_speed_valid',
            'borrow_go_min_speed_valid',
            'borrow_cross_active_time_s',
        )
        for key in debug_batch_keys:
            value = batch.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            value_detached = value.detach().float().cpu()
            finite_mask = torch.isfinite(value_detached)
            finite_count = int(finite_mask.sum().item())
            total_count = int(value_detached.numel())
            if finite_count > 0:
                finite_values = value_detached[finite_mask]
                min_value = float(finite_values.min().item())
                max_value = float(finite_values.max().item())
            else:
                min_value = float('nan')
                max_value = float('nan')
            print(
                f"[NONFINITE] batch[{key}] shape={tuple(value_detached.shape)} "
                f"finite={finite_count}/{total_count} min={min_value} max={max_value}",
                flush=True,
            )

    # Use DistributedSampler for multi-GPU training
    if world_size > 1:
        if use_window_weighted_sampler:
            sampler_train = WeightedDistributedSampler(
                train_sample_weights,
                num_replicas=world_size,
                rank=rank,
                replacement=window_sampler_replacement,
                drop_last=True,
                seed=window_sampler_seed,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                sampler=sampler_train,
                num_workers=train_num_workers,
                pin_memory=train_pin_memory,
                persistent_workers=train_persistent_workers if train_num_workers > 0 else False,
                prefetch_factor=train_prefetch_factor if train_num_workers > 0 else None,
                drop_last=True,
                collate_fn=safe_collate,
            )
        elif use_route_group_sampler:
            sampler_train = DistributedRouteBatchSampler(
                train_dataset._route_groups,
                batch_size=train_batch_size,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=window_sampler_seed,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=sampler_train,
                num_workers=train_num_workers,
                pin_memory=train_pin_memory,
                persistent_workers=train_persistent_workers if train_num_workers > 0 else False,
                prefetch_factor=train_prefetch_factor if train_num_workers > 0 else None,
                collate_fn=safe_collate,
            )
            if rank == 0:
                print("Using distributed route-grouped batch sampler")
        else:
            sampler_train = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                shuffle=True,
                num_replicas=world_size,
                rank=rank,
                drop_last=True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                sampler=sampler_train,
                num_workers=train_num_workers,
                pin_memory=train_pin_memory,
                persistent_workers=train_persistent_workers if train_num_workers > 0 else False,
                prefetch_factor=train_prefetch_factor if train_num_workers > 0 else None,
                drop_last=True,
                collate_fn=safe_collate,
            )
        # For validation, only rank 0 needs the full dataset
        # Other ranks don't participate in validation
        sampler_val = None
    else:
        sampler_train = None
        sampler_val = None
        if use_window_weighted_sampler:
            sampler_train = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(train_sample_weights, dtype=torch.double),
                num_samples=len(train_dataset),
                replacement=window_sampler_replacement,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                sampler=sampler_train,
                num_workers=train_num_workers,
                pin_memory=train_pin_memory,
                persistent_workers=train_persistent_workers if train_num_workers > 0 else False,
                prefetch_factor=train_prefetch_factor if train_num_workers > 0 else None,
                drop_last=True,
                collate_fn=safe_collate,
            )
            if rank == 0:
                print("Using window-aware weighted sampler (single GPU)")
        elif use_route_group_sampler:
            # Route-grouped batching: samples in the same batch come from the same/adjacent routes.
            train_batch_sampler = train_dataset.get_route_batch_sampler(
                batch_size=train_batch_size, shuffle=True, drop_last=True)
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,
                num_workers=train_num_workers,
                pin_memory=train_pin_memory,
                persistent_workers=train_persistent_workers if train_num_workers > 0 else False,
                prefetch_factor=train_prefetch_factor if train_num_workers > 0 else None,
                collate_fn=safe_collate,
            )
            if rank == 0:
                print("Using route-grouped batch sampler (single GPU)")
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=train_num_workers,
                pin_memory=train_pin_memory,
                persistent_workers=train_persistent_workers if train_num_workers > 0 else False,
                prefetch_factor=train_prefetch_factor if train_num_workers > 0 else None,
                drop_last=True,
                collate_fn=safe_collate,
            )
            if rank == 0:
                print("Using random shuffle sampler (single GPU)")
    
    # Validation loader: only create meaningful loader for rank 0
    # Other ranks get an empty loader since they don't validate
    val_loader = None
    if val_dataset is not None:
        if world_size > 1:
            # All ranks participate in validation to avoid NCCL timeout
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                shuffle=False,
                num_replicas=world_size,
                rank=rank,
                drop_last=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=val_num_workers,
                pin_memory=val_pin_memory,
                persistent_workers=val_persistent_workers if val_num_workers > 0 else False,
                prefetch_factor=val_prefetch_factor if val_num_workers > 0 else None,
                drop_last=True,
                collate_fn=safe_collate,
            )
        else:
            # Single GPU: use full validation dataset
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                sampler=sampler_val,
                shuffle=False,
                num_workers=val_num_workers,
                pin_memory=val_pin_memory,
                persistent_workers=val_persistent_workers if val_num_workers > 0 else False,
                prefetch_factor=val_prefetch_factor if val_num_workers > 0 else None,
                drop_last=True,
                collate_fn=safe_collate,
            )

    # Mixed precision (AMP) setup — prefer BF16 on supported hardware
    use_amp = config.get('model_optimization', {}).get('use_mixed_precision', True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    if rank == 0:
        print(f"✓ Mixed precision (AMP): {'enabled' if use_amp else 'disabled'} (dtype={amp_dtype})")

    if rank == 0:
        print("Initializing policy model...")
    policy_type = config.get('policy_type', 'anchor_free')
    if policy_type != 'anchor_free':
        raise ValueError(
            f"This training entry is Route B-only after cleanup; expected policy_type='anchor_free', "
            f"got {policy_type!r}."
        )
    policy = AnnealedEnergyGuidancePolicy(config).to(device)
    if rank == 0:
        print("  Policy: AnnealedEnergyGuidancePolicy (Route B semantic-state)")
    # Register normalization stats (before DDP wrapping).
    # Route B uses strict config selection: exactly one normalization family
    # must be specified, and checkpoint loading must not silently override it.
    norm_mode = policy.register_norm_stats_from_config(config)
    route_abs_stats_path = config.get('route_abs_stats_path', None)
    if rank == 0:
        if norm_mode == 'global_abs':
            print(
                "  ✓ Global abs stats registered: "
                f"mean={tuple(policy.global_abs_mean.shape)}, std={tuple(policy.global_abs_std.shape)}"
            )
        elif norm_mode == 'abs':
            print(
                "  ✓ Per-step abs stats registered: "
                f"mean={tuple(policy.abs_mean.shape)}, std={tuple(policy.abs_std.shape)}"
            )
        else:
            print(
                "  ✓ Delta stats registered (legacy): "
                f"mean={tuple(policy.delta_mean.shape)}, std={tuple(policy.delta_std.shape)}"
            )
    if route_abs_stats_path:
        route_data = np.load(route_abs_stats_path)
        policy.register_route_abs_stats(route_data['route_abs_mean'], route_data['route_abs_std'])
        if rank == 0:
            print(
                f"  ✓ Route abs stats registered: mean={route_data['route_abs_mean'].shape}, "
                f"std={route_data['route_abs_std'].shape}"
            )
    else:
        raise ValueError("route_abs_stats_path is required for Route B ego diffusion")

    init_checkpoint_path = init_checkpoint_path or config.get('training', {}).get('init_checkpoint_path', None)
    if init_checkpoint_path is not None and resume_path is None:
        if rank == 0:
            print(f"Initializing model from checkpoint (partial/non-strict): {init_checkpoint_path}")
        init_ckpt = torch.load(init_checkpoint_path, map_location=device)
        init_state = init_ckpt.get('model_state_dict', init_ckpt)
        current_state = policy.state_dict()
        compatible = {}
        skipped_missing = []
        skipped_shape = []
        for key, value in init_state.items():
            if key not in current_state:
                skipped_missing.append(key)
                continue
            if tuple(current_state[key].shape) != tuple(value.shape):
                skipped_shape.append((key, tuple(value.shape), tuple(current_state[key].shape)))
                continue
            compatible[key] = value
        current_state.update(compatible)
        policy.load_state_dict(current_state, strict=True)
        if rank == 0:
            print(
                f"  ✓ Partial init loaded {len(compatible)} tensors; "
                f"skipped_missing={len(skipped_missing)}, skipped_shape={len(skipped_shape)}"
            )
            if skipped_missing:
                print(f"  skipped missing preview: {skipped_missing[:8]}{'...' if len(skipped_missing) > 8 else ''}")
            if skipped_shape:
                preview = [f"{k}: ckpt{s} -> model{m}" for k, s, m in skipped_shape[:8]]
                print(f"  skipped shape preview: {preview}{'...' if len(skipped_shape) > 8 else ''}")

    # Resume from checkpoint if specified
    start_epoch = 0
    checkpoint = None
    if resume_path is not None:
        if rank == 0:
            print(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        saved_state = checkpoint['model_state_dict']
        current_state = policy.state_dict()
        saved_keys = set(saved_state.keys())
        current_keys = set(current_state.keys())
        missing_keys = sorted(current_keys - saved_keys)
        unexpected_keys = sorted(saved_keys - current_keys)
        mismatched_shapes = []
        for key in sorted(saved_keys & current_keys):
            current_shape = tuple(current_state[key].shape)
            saved_shape = tuple(saved_state[key].shape)
            if current_shape != saved_shape:
                mismatched_shapes.append((key, saved_shape, current_shape))

        if missing_keys or unexpected_keys or mismatched_shapes:
            if rank == 0:
                print("  Checkpoint/model mismatch detected during strict resume.")
                if missing_keys:
                    preview = missing_keys[:8]
                    suffix = "..." if len(missing_keys) > 8 else ""
                    print(f"  Missing keys ({len(missing_keys)}): {preview}{suffix}")
                if unexpected_keys:
                    preview = unexpected_keys[:8]
                    suffix = "..." if len(unexpected_keys) > 8 else ""
                    print(f"  Unexpected keys ({len(unexpected_keys)}): {preview}{suffix}")
                if mismatched_shapes:
                    preview = mismatched_shapes[:8]
                    formatted = [f"{k}: ckpt{saved_shape} -> model{current_shape}" for k, saved_shape, current_shape in preview]
                    suffix = "..." if len(mismatched_shapes) > 8 else ""
                    print(f"  Shape mismatches ({len(mismatched_shapes)}): {formatted}{suffix}")
            raise RuntimeError(
                "Strict checkpoint resume failed due to model/ckpt mismatch. "
                "See grouped summary above."
            )

        policy.load_state_dict(saved_state, strict=True)
        start_epoch = checkpoint.get('epoch', 0) + 1
        if rank == 0:
            print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'val_metrics' in checkpoint:
                print(f"  Previous val_metrics: {checkpoint['val_metrics']}")
        # Restore scaler state if available (for AMP resume)
        if use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if rank == 0:
                print("  ✓ AMP scaler state restored")

    # Wrap model with DistributedDataParallel for multi-GPU training
    if world_size > 1:
        policy = torch.nn.parallel.DistributedDataParallel(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True  # Required: some parameters in obs_encoder may not be used in all forward passes
        )
        if rank == 0:
            print(f"DDP enabled | n_action_steps: {policy.module.n_action_steps}")
    else:
        if rank == 0:
            print(f"Single GPU | n_action_steps: {policy.n_action_steps}")
    
    lr = config.get('optimizer', {}).get('lr', 5e-5)
    weight_decay = config.get('optimizer', {}).get('weight_decay', 1e-5)
    
    # Linear learning rate scaling for distributed training
    # With N GPUs and same batch_size per GPU, effective batch_size = N * batch_size
    # Scale learning rate linearly: lr_scaled = lr * world_size
    scale_lr = config.get('optimizer', {}).get('scale_lr', True)  # Default to True
    if scale_lr and world_size > 1:
        lr_scaled = lr * world_size
        if rank == 0:
            print(f"✓ Learning rate scaled for {world_size} GPUs: {lr} -> {lr_scaled}")
        lr = lr_scaled
    
    # ========== Optimizer Setup ==========
    policy_for_params = policy.module if world_size > 1 else policy
    route_b_cfg = config.get('route_b', {})
    route_b_phase = 'split' if route_b_cfg.get('use_split_forward', False) else 'unified'
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    if rank == 0:
        print("✓ Single optimizer: Route B semantic-state model")

    # Learning rate scheduler with warmup for multi-GPU training stability
    warmup_epochs = int(config.get('training', {}).get('warmup_epochs', 5))
    lr_final = float(config.get('training', {}).get('lr_final', 1e-7))
    use_lr_scheduler = config.get('training', {}).get('use_lr_scheduler', True)
    resume_rebuild_scheduler = bool(config.get('training', {}).get('resume_rebuild_scheduler', False))

    if use_lr_scheduler:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ConstantLR

        total_epochs = int(config.get('training', {}).get('num_epochs', 50))
        scheduler_total_epochs = int(config.get('training', {}).get('scheduler_total_epochs', total_epochs))
        scheduler_total_epochs = max(scheduler_total_epochs, warmup_epochs + 1)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_total_epochs - warmup_epochs,
            eta_min=lr_final
        )
        if scheduler_total_epochs < total_epochs:
            hold_scheduler = ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=total_epochs - scheduler_total_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler, hold_scheduler],
                milestones=[warmup_epochs, scheduler_total_epochs],
            )
        else:
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )

        if rank == 0:
            print(
                f"✓ Learning rate scheduler: {warmup_epochs} epochs warmup + "
                f"cosine annealing over {scheduler_total_epochs} epochs to {lr_final}"
                + (f", then hold to epoch {total_epochs}" if scheduler_total_epochs < total_epochs else "")
            )
    else:
        scheduler = None
        if rank == 0:
            print("✓ No learning rate scheduler used")

    def _set_optimizer_lr(optim, lr_value: float):
        if optim is None:
            return
        for group in optim.param_groups:
            group['lr'] = float(lr_value)
            group['initial_lr'] = float(lr_value)

    def _set_scheduler_base_lrs(sched, lr_value: float):
        if sched is None:
            return
        if hasattr(sched, 'base_lrs'):
            sched.base_lrs = [float(lr_value) for _ in sched.base_lrs]
        if hasattr(sched, '_last_lr'):
            sched._last_lr = [float(lr_value) for _ in sched._last_lr]
        if hasattr(sched, '_schedulers'):
            for sub_sched in sched._schedulers:
                _set_scheduler_base_lrs(sub_sched, lr_value)

    def _fast_forward_scheduler(sched, steps: int):
        if sched is None or steps <= 0:
            return
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(int(steps)):
                sched.step()

    # EMA (Exponential Moving Average) for stable inference
    ema_cfg = config.get('ema', {})
    model_for_ema = policy.module if world_size > 1 else policy
    ema_model = EMAModel(model_for_ema.parameters(), max_value=ema_cfg.get('max_value', 0.9999))
    ema_model.to(device)  # Keep EMA on same device as model for .step() compatibility
    ema_update_interval = ema_cfg.get('update_interval', 10)  # Update every N steps
    # Restore EMA state from checkpoint if available
    if checkpoint is not None and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
        ema_saved = checkpoint['ema_state_dict']
        # Check if shadow_params shapes match current model (may differ after architecture changes)
        current_params = list(model_for_ema.parameters())
        saved_shadows = ema_saved.get('shadow_params', [])
        if len(saved_shadows) == len(current_params) and all(
            s.shape == p.shape for s, p in zip(saved_shadows, current_params)
        ):
            ema_model.load_state_dict(ema_saved)
            if rank == 0:
                print("  ✓ EMA state restored")
        else:
            if rank == 0:
                print("  ⚠ EMA shape mismatch (model architecture changed), re-initializing EMA from current model")
    if rank == 0:
        print(f"✓ EMA initialized (max_value={ema_cfg.get('max_value', 0.9999)})")

    # Restore optimizer states from checkpoint if available
    if checkpoint is not None:
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if rank == 0:
                    print("  ✓ Optimizer state restored")
            except Exception:
                if rank == 0:
                    print("  ⚠ Could not restore optimizer state (param groups changed)")
        if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            if resume_rebuild_scheduler:
                if rank == 0:
                    print("  ↺ Skipping decoder scheduler state restore (resume_rebuild_scheduler=true)")
            else:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    if rank == 0:
                        print("  ✓ Scheduler state restored")
                except Exception:
                    if rank == 0:
                        print("  ⚠ Could not restore scheduler state")

        resume_override_lr = config.get('optimizer', {}).get('resume_override_lr', None)

        if resume_override_lr is not None:
            resume_override_lr = float(resume_override_lr)
            if scale_lr and world_size > 1:
                resume_override_lr *= world_size
            _set_optimizer_lr(optimizer, resume_override_lr)
            _set_scheduler_base_lrs(scheduler, resume_override_lr)
            if rank == 0:
                print(f"  ✓ Resume override decoder lr -> {resume_override_lr:.2e}")

        if resume_rebuild_scheduler:
            if scheduler is not None:
                _fast_forward_scheduler(scheduler, start_epoch)
                if rank == 0:
                    print(
                        "  ✓ Rebuilt decoder scheduler from config and "
                        f"fast-forwarded to start_epoch={start_epoch} "
                        f"(lr={optimizer.param_groups[0]['lr']:.2e})"
                    )

    # 设置 checkpoint 目录
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', "/media/z/data/mzq/others/MoT-DP/checkpoints/carla_dit")
    speed_adaptive_json_path = os.path.join(checkpoint_dir, "speed_stats_val_latest.json")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"✓ Checkpoint directory: {checkpoint_dir}")
    
    num_epochs = config.get('training', {}).get('num_epochs', 50)
    best_val_loss = float('inf')
    best_l2_avg = float('inf')  # Use average L2 error as best metric
    val_loss = None  # 初始化验证损失
    val_metrics = {}  # 初始化验证指标

    # ========== Val Only Mode ==========
    if val_only:
        if val_loader is None:
            raise ValueError("Validation is disabled by config, but --val_only was requested.")
        if rank == 0:
            print("=" * 60)
            print("Running validation only (--val_only mode)")
            print("=" * 60)
        try:
            ema_model.store(model_for_ema.parameters())
            ema_model.copy_to(model_for_ema.parameters())
            val_metrics = validate_model(
                policy, val_loader, device, rank=rank, world_size=world_size,
                use_amp=use_amp, amp_dtype=amp_dtype, max_batches=val_max_batches,
                speed_adaptive_json_path=speed_adaptive_json_path,
                ordered_semantic_rollout_enabled=ordered_semantic_rollout_enabled,
                ordered_semantic_rollout_max_routes=ordered_semantic_rollout_max_routes,
                ordered_semantic_rollout_max_frames=ordered_semantic_rollout_max_frames,
            )
            if rank == 0:
                print(f"\n✓ Validation completed")
                _print_validation_metrics(val_metrics, show_speed_metrics=False)
        except Exception as e:
            if rank == 0:
                print(f"✗ Error during validation: {e}")
                import traceback
                traceback.print_exc()
        finally:
            ema_model.restore(model_for_ema.parameters())

        # Clean up and exit
        if world_size > 1:
            torch.distributed.destroy_process_group()
        return

    for epoch in range(start_epoch, num_epochs):
        # Memory monitoring
        if rank == 0:
            import psutil
            mem = psutil.virtual_memory()
            print(f"\n[Epoch {epoch+1}] RAM: {mem.used/1e9:.1f}GB used / {mem.total/1e9:.1f}GB total "
                  f"({mem.percent}%), available={mem.available/1e9:.1f}GB, "
                  f"cached={getattr(mem, 'cached', 0)/1e9:.1f}GB")
            for gi in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(gi) / 1e9
                reserved = torch.cuda.memory_reserved(gi) / 1e9
                print(f"  GPU{gi}: {alloc:.2f}GB alloc / {reserved:.2f}GB reserved")

        # Update the seed depending on the epoch for distributed sampler
        if world_size > 1:
            sampler_train.set_epoch(epoch)

        policy.train()
        policy_unwrapped = policy.module if world_size > 1 else policy
        if hasattr(policy_unwrapped, '_current_epoch'):
            policy_unwrapped._current_epoch = epoch
        train_losses = []
        nonfinite_debug_budget = 3

        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        else:
            pbar = train_loader
            
        for batch_idx, batch in enumerate(pbar):
            if hasattr(policy_unwrapped, '_current_batch_idx'):
                policy_unwrapped._current_batch_idx = batch_idx
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            max_grad_norm = config.get('training', {}).get('max_grad_norm', 1.0)

            optimizer.zero_grad(set_to_none=True)
            with autocast_cuda(use_amp, amp_dtype):
                loss_dict = policy(batch, return_loss_dict=True, phase=route_b_phase)
                loss = loss_dict['total_loss']

            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping")
                    if nonfinite_debug_budget > 0:
                        print_nonfinite_loss_debug(loss_dict, batch, batch_idx, rank)
                        nonfinite_debug_budget -= 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            params = policy.module.parameters() if world_size > 1 else policy.parameters()
            grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)

            if torch.isnan(grad_norm_before_clip) or torch.isinf(grad_norm_before_clip):
                if rank == 0:
                    print(f"Warning: NaN/Inf gradient at batch {batch_idx}, skipping")
                    if nonfinite_debug_budget > 0:
                        print_nonfinite_loss_debug(loss_dict, batch, batch_idx, rank)
                        nonfinite_debug_budget -= 1
                optimizer.zero_grad()
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()

            # Update EMA every N steps
            if batch_idx % ema_update_interval == 0:
                ema_model.step(model_for_ema.parameters())

            train_losses.append(loss.item())

            # Calculate if clipping occurred
            grad_norm_value = grad_norm_before_clip.item() if isinstance(grad_norm_before_clip, torch.Tensor) else grad_norm_before_clip
            was_clipped = grad_norm_value > max_grad_norm

            if rank == 0:
                postfix = {
                    'loss': f'{loss.item():.4f}',
                }
                if 'cls_loss' in loss_dict:
                    postfix['cls'] = f'{loss_dict["cls_loss"].item():.3f}'
                if 'reg_loss' in loss_dict:
                    postfix['reg'] = f'{loss_dict["reg_loss"].item():.3f}'
                if 'route_loss' in loss_dict:
                    postfix['route'] = f'{loss_dict["route_loss"].item():.3f}'
                postfix['grad'] = f'{grad_norm_value:.2f}{"✂" if was_clipped else ""}'
                if 'stage1_loss' in loss_dict:
                    postfix['S1'] = f'{loss_dict["stage1_loss"].item():.3f}'
                elif 'energy_loss' in loss_dict:
                    postfix['E'] = f'{loss_dict["energy_loss"].item():.3f}'
                if 'alignment_loss' in loss_dict:
                    postfix['align'] = f'{loss_dict["alignment_loss"].item():.3f}'
                if 'speed_loss' in loss_dict:
                    postfix['spd'] = f'{loss_dict["speed_loss"].item():.3f}'
                if 'speed_profile_loss' in loss_dict:
                    postfix['spf'] = f'{loss_dict["speed_profile_loss"].item():.3f}'
                pbar.set_postfix(postfix)

            # Log to wandb less frequently to reduce overhead
            log_freq = config.get('logging', {}).get('log_freq', 50)
            if batch_idx % log_freq == 0 and rank == 0:
                step = epoch * len(train_loader) + batch_idx
                log_data = {
                    "train/loss_step": loss.item(),
                    "train/epoch":  epoch,
                    "train/step": step,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/batch_idx": batch_idx,
                    "train/grad_norm_before_clip": grad_norm_value,
                    "train/grad_norm_clipped": min(grad_norm_value, max_grad_norm),
                    "train/grad_clipping_ratio": grad_norm_value / max_grad_norm if max_grad_norm > 0 else 0,
                }
                # Individual losses
                for lk in ('cls_loss', 'reg_loss', 'route_loss', 'speed_loss', 'speed_profile_loss',
                           'stage1_loss', 'energy_loss', 'alignment_loss',
                           'energy_front_loss', 'energy_left_loss', 'energy_right_loss',
                           'energy_chase_loss', 'energy_merge_loss', 'energy_cross_loss',
                           'energy_ped_loss', 'energy_pedestrian_loss',
                           'energy_off_loss', 'energy_route_loss',
                           'stage1_merge_yld_loss', 'stage1_merge_go_loss',
                           'stage1_junction_yld_loss', 'stage1_junction_go_loss',
                           'stage1_borrow_yld_loss', 'stage1_borrow_go_loss',
                           'stage1_cross_yld_loss', 'stage1_cross_go_loss',
                           'stage1_merge_active_loss',
                           'stage1_junction_active_loss', 'stage1_borrow_active_loss',
                           'stage1_cross_active_loss',
                           'stage1_dir_loss',
                           'stage1_conflict_area_loss',
                           'stage1_window_loss', 'stage1_phase_loss',
                           'stage1_decision_phase_loss', 'stage1_control_phase_loss',
                           'stage1_temporary_occupancy_loss', 'stage1_go_opportunity_loss',
                           'stage1_conflict_area_status_loss', 'stage1_conflict_timing_loss',
                           'stage1_inside_area_go_loss',
                           'stage1_chase_loss',
                           'stage1_chase_has_lead_loss',
                           'stage1_chase_speed_max_loss',
                           'stage1_state_consistency_loss',
                           'stage1_state_consistency_window_loss',
                           'stage1_state_consistency_phase_loss',
                           'stage1_state_consistency_timing_loss',
                           'stage1_state_consistency_boundary_loss',
                           'stage1_state_consistency_area_loss',
                           'stage1_state_consistency_tempocc_loss',
                           'stage1_state_consistency_opportunity_loss',
                           'stage1_merge_yld_max_loss', 'stage1_merge_go_min_loss',
                           'stage1_junction_yld_max_loss', 'stage1_junction_go_min_loss',
                           'stage1_borrow_yld_max_loss', 'stage1_borrow_go_min_loss',
                           'energy_merge_yld_loss', 'energy_merge_go_loss',
                           'energy_junction_yld_loss', 'energy_junction_go_loss',
                           'energy_borrow_yld_loss', 'energy_borrow_go_loss',
                           'energy_cross_yld_loss', 'energy_cross_go_loss',
                           'energy_merge_active_loss',
                           'energy_junction_active_loss', 'energy_borrow_active_loss',
                           'energy_cross_active_loss',
                           'energy_dir_loss',
                           'energy_conflict_area_loss',
                           'energy_window_loss', 'energy_phase_loss',
                           'energy_decision_phase_loss', 'energy_control_phase_loss',
                           'energy_temporary_occupancy_loss', 'energy_go_opportunity_loss',
                           'energy_conflict_area_status_loss', 'energy_conflict_timing_loss',
                           'energy_inside_area_go_loss',
                           'energy_chase_loss',
                           'energy_chase_has_lead_loss',
                           'energy_chase_speed_max_loss',
                           'energy_state_consistency_loss',
                           'energy_state_consistency_window_loss',
                           'energy_state_consistency_phase_loss',
                           'energy_state_consistency_timing_loss',
                           'energy_state_consistency_boundary_loss',
                           'energy_state_consistency_area_loss',
                           'energy_state_consistency_tempocc_loss',
                           'energy_state_consistency_opportunity_loss',
                           'energy_merge_yld_max_loss', 'energy_merge_go_min_loss',
                           'energy_junction_yld_max_loss', 'energy_junction_go_min_loss',
                           'energy_borrow_yld_max_loss', 'energy_borrow_go_min_loss'):
                    if lk in loss_dict:
                        val = loss_dict[lk]
                        log_data[f"train/{lk}"] = val.item() if isinstance(val, torch.Tensor) else val
                for lk, val in loss_dict.items():
                    if lk.startswith('speed_profile_step') and lk.endswith('_loss'):
                        log_data[f"train/{lk}"] = val.item() if isinstance(val, torch.Tensor) else val
                # Weighted losses (Route A only)
                for wk in ('cls_loss_weighted', 'reg_loss_weighted', 'route_loss_weighted'):
                    if wk in loss_dict:
                        log_data[f"train/{wk}"] = loss_dict[wk].item()
                # Semantic behavior losses (if available)
                if 'behavior_loss' in loss_dict:
                    log_data["train/behavior_loss"] = loss_dict['behavior_loss'].item()
                    log_data["train/allowed_loss"] = loss_dict['allowed_loss'].item()
                safe_wandb_log(log_data, use_wandb)
        
        if rank == 0:
            pbar.close() 
        
        avg_train_loss = np.mean(train_losses)
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Average training loss: {avg_train_loss:.4f}")
        
        # Update learning rate scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Learning rate: {current_lr:.2e}")
        
        # Get model state dict (handle DDP wrapper)
        model_to_save = policy.module if world_size > 1 else policy
        
        # Save checkpoint at save_freq intervals, keep only the latest max_keep checkpoints
        save_freq = config.get('training', {}).get('save_freq', 5)
        max_keep_ckpts = config.get('training', {}).get('max_keep_ckpts', 5)
        if rank == 0 and (epoch + 1) % save_freq == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"dit_policy_epoch{epoch+1}.pt")
            ckpt_data = {
                        'model_state_dict': model_to_save.state_dict(),
                        'ema_state_dict': ema_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'scaler_state_dict': scaler.state_dict() if use_amp else None,
                        'config': config,
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'train_loss': avg_train_loss,
                        'val_metrics': val_metrics,
                        }
            torch.save(ckpt_data, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

            # Remove old periodic checkpoints, keep only the latest max_keep_ckpts
            import glob as glob_module
            periodic_ckpts = sorted(
                glob_module.glob(os.path.join(checkpoint_dir, "dit_policy_epoch*.pt")),
                key=os.path.getmtime
            )
            while len(periodic_ckpts) > max_keep_ckpts:
                old_ckpt = periodic_ckpts.pop(0)
                os.remove(old_ckpt)
                print(f"  Removed old checkpoint: {old_ckpt}")
        
        if rank == 0:
            safe_wandb_log({
                "train/loss_epoch": avg_train_loss,
                "train/epoch": epoch,
                "train/samples_processed": (epoch + 1) * len(train_dataset)
            }, use_wandb)

        if val_loader is not None and validation_freq > 0 and (epoch + 1) % validation_freq == 0:
            # Apply EMA weights for validation
            ema_model.store(model_for_ema.parameters())
            ema_model.copy_to(model_for_ema.parameters())

            if rank == 0:
                print(f"Validating with EMA weights (Epoch {epoch+1}/{num_epochs})...")
            try:
                val_metrics = validate_model(
                    policy, val_loader, device, rank=rank, world_size=world_size,
                    use_amp=use_amp, amp_dtype=amp_dtype, max_batches=val_max_batches,
                    speed_adaptive_json_path=speed_adaptive_json_path,
                    ordered_semantic_rollout_enabled=ordered_semantic_rollout_enabled,
                    ordered_semantic_rollout_max_routes=ordered_semantic_rollout_max_routes,
                    ordered_semantic_rollout_max_frames=ordered_semantic_rollout_max_frames,
                )
            except Exception as e:
                if rank == 0:
                    print(f"✗ Error during validation: {e}")
                    import traceback
                    traceback.print_exc()
                ema_model.restore(model_for_ema.parameters())
                torch.cuda.empty_cache()
                continue

            # Free GPU memory allocated during diffusion sampling in validation
            torch.cuda.empty_cache()

            if rank == 0:
                log_dict = {"epoch": epoch, "train/loss": avg_train_loss}
                for key, value in val_metrics.items():
                    log_dict[f"val/{key.removeprefix('val_')}"] = value
                safe_wandb_log(log_dict, use_wandb)

                _print_validation_metrics(val_metrics, show_speed_metrics=False)
                metrics_json_path, metrics_csv_path = _write_validation_metrics_artifacts(
                    checkpoint_dir, epoch, avg_train_loss, val_metrics
                )
                print(f"  Validation metrics saved: {metrics_json_path}")
                print(f"  Validation summary CSV: {metrics_csv_path}")

                val_loss = val_metrics.get('val_loss', float('inf'))
                l2_avg = val_metrics.get('val_L2_avg', float('inf'))
                
                # Save best model based on L2_avg (average L2 error across all timesteps)
                if l2_avg < best_l2_avg:
                    best_l2_avg = l2_avg
                    # Append epoch number to filename if epoch > 100 to avoid overwriting
                    if epoch > 100:
                        # best_model_filename = f"dit_policy_best_epoch{epoch}.pt"
                        best_model_filename = "dit_policy_best.pt"
                    else:
                        best_model_filename = "dit_policy_best.pt"
                    # Save EMA weights as the best model (already applied to model_for_ema)
                    torch.save({
                            'model_state_dict': model_to_save.state_dict(),
                            'ema_state_dict': ema_model.state_dict(),
                            'config': config,
                            'epoch': epoch,
                            'val_loss': val_loss,
                            'train_loss': avg_train_loss,
                            'val_metrics': val_metrics
                            }, os.path.join(checkpoint_dir, best_model_filename))
                    print(f"✓ New best model saved with L2_avg: {l2_avg:.4f} (val_loss: {val_loss:.4f})")
                   
                    safe_wandb_log({
                            "best_model/epoch": epoch,
                            "best_model/L2_avg": l2_avg,
                            "best_model/val_loss": val_loss,
                            "best_model/train_loss": avg_train_loss
                        }, use_wandb)

            # Restore training weights after validation/save
            ema_model.restore(model_for_ema.parameters())
    
    if rank == 0:
        print("Training completed!")
        if val_loader is not None:
            print(f"Best L2_avg: {best_l2_avg:.4f}")
        else:
            print("Validation was disabled for this run.")
        safe_wandb_log({
            "training/completed": 0.0,
            "training/total_epochs": num_epochs,
            "training/best_l2_avg": best_l2_avg,
            "training/final_train_loss": avg_train_loss
        }, use_wandb)
        
        safe_wandb_finish(use_wandb)
        print("✓ Training session finished")
    
    # Clean up distributed process group
    if world_size > 1:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pdm Driving Policy with Diffusion DiT - Multi-GPU Distributed Training")
    parser.add_argument('--config_path', type=str, default="/home/wang/Project/MoT-DP/config/pdm_local.yaml",
                        help='Path to the configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='Path to checkpoint for partial/non-strict weight initialization')
    parser.add_argument('--val_only', action='store_true',
                        help='Only run validation (requires --resume)')
    args = parser.parse_args()
    train_pdm_policy(
        config_path=args.config_path,
        resume_path=args.resume,
        val_only=args.val_only,
        init_checkpoint_path=args.init_checkpoint,
    )
