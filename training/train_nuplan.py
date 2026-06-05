"""Training script for NuPlan White-Noise Diffusion.

DDP training with EMA, AMP, AdamW, cosine schedule.
Adapted from MoT-DP's train_carla_bev.py.
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from tqdm.auto import tqdm

try:
    from torch.amp import autocast, GradScaler
    def autocast_cuda(enabled, dtype):
        return autocast('cuda', enabled=enabled, dtype=dtype)
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    def autocast_cuda(enabled, dtype):
        return autocast(enabled=enabled)


# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataset.nuplan_dataset import NuPlanDataset
from model.adapter_layer import AdapterLayer
from model.lora_linear import LoRALinear
from policy.nuplan_diffusion_policy import NuPlanDiffusionPolicy, compute_global_norm_stats
from training.gradient_projector import GradientProjector


DEFAULT_VAL_CACHE_DIRS = [
    '/workspace2/z_project/exp/nuplan/cache_plantf_val/',
]


def move_batch_to_device(batch: tuple) -> tuple:
    return tuple(
        item.cuda(non_blocking=True) if isinstance(item, torch.Tensor) else item
        for item in batch
    )


def format_metrics(prefix: str, metrics: dict) -> str:
    return (
        f"{prefix} | "
        f"Loss: {metrics['loss']:.4f} | "
        f"Ego: {metrics['ego_loss']:.4f} | "
        f"Neighbor: {metrics['neighbor_loss']:.4f} | "
        f"ADE: {metrics['ego_ADE']:.4f} | "
        f"FDE: {metrics['ego_FDE']:.4f} | "
        f"PredFinal: {metrics['pred_final_disp']:.4f} | "
        f"GtFinal: {metrics['gt_final_disp']:.4f} | "
        f"N: {metrics['sample_count']}"
    )


def infer_completed_steps(ckpt: dict, steps_per_epoch: int) -> int:
    scheduler_state = ckpt.get('scheduler') or {}
    last_epoch = scheduler_state.get('last_epoch')
    if isinstance(last_epoch, int) and last_epoch >= 0:
        return last_epoch
    return max(0, (ckpt.get('epoch', -1) + 1) * steps_per_epoch)


def restore_scheduler_state(
    scheduler,
    optimizer,
    ckpt: dict,
    steps_per_epoch: int,
    total_epochs: int,
    rank: int,
):
    scheduler_state = ckpt.get('scheduler')
    if scheduler_state is None:
        return

    completed_steps = infer_completed_steps(ckpt, steps_per_epoch)
    completed_epoch_steps = max(0, (ckpt.get('epoch', -1) + 1) * steps_per_epoch)
    target_total_steps = max(1, total_epochs * steps_per_epoch)
    saved_total_steps = ckpt.get('scheduler_total_steps')
    saved_steps_per_epoch = ckpt.get('steps_per_epoch')
    uses_step_schedule = ckpt.get('lr_schedule_unit') == 'step'

    if (
        uses_step_schedule
        and saved_total_steps == target_total_steps
        and saved_steps_per_epoch == steps_per_epoch
    ):
        scheduler.load_state_dict(scheduler_state)
        return

    if target_total_steps <= completed_epoch_steps:
        scheduler.load_state_dict(scheduler_state)
        return

    current_lrs = [group['lr'] for group in optimizer.param_groups]
    schedule_factor = max(float(scheduler.lr_lambdas[0](completed_epoch_steps)), 1e-8)
    base_lrs = [lr / schedule_factor for lr in current_lrs]

    scheduler.base_lrs = base_lrs
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group['initial_lr'] = base_lr
    scheduler.last_epoch = completed_epoch_steps
    scheduler._step_count = completed_epoch_steps + 1
    scheduler._last_lr = current_lrs

    if rank == 0:
        if uses_step_schedule:
            print(
                f"Re-anchored LR schedule from {saved_total_steps} to {target_total_steps} steps "
                f"at epoch-aligned step {completed_epoch_steps} with lr {current_lrs[0]:.6g}"
            )
        else:
            print(
                f"Detected legacy LR schedule in checkpoint; re-anchored step-based schedule "
                f"at epoch-aligned step {completed_epoch_steps}/{target_total_steps} with lr {current_lrs[0]:.6g}"
            )


def build_eval_dataloader(
    config: dict,
    cache_dirs,
    allowed_scenario_types,
    allowed_target_types,
    max_samples,
    samples_per_target_type,
    repeat_small_target_types,
    sampling_seed,
    default_target_type_partition_index,
    target_type_partition_indices,
):
    num_workers = config.get('val_num_workers', config.get('num_workers', 8))
    dataset = NuPlanDataset(
        cache_dirs=cache_dirs,
        max_samples=max_samples,
        allowed_scenario_types=allowed_scenario_types,
        allowed_target_types=allowed_target_types,
        samples_per_target_type=samples_per_target_type,
        repeat_small_target_types=repeat_small_target_types,
        sampling_seed=sampling_seed,
        default_target_type_partition_index=default_target_type_partition_index,
        target_type_partition_indices=target_type_partition_indices,
    )
    dataloader_kwargs = {}
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = config.get(
            'val_prefetch_factor',
            config.get('prefetch_factor', 2),
        )
        dataloader_kwargs['persistent_workers'] = config.get(
            'val_persistent_workers',
            config.get('persistent_workers', True),
        )
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('val_batch_size', config.get('batch_size', 64)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=default_collate,
        **dataloader_kwargs,
    )
    return dataset, dataloader


def _dedupe_tasks(task_names):
    ordered = []
    seen = set()
    for task_name in task_names or []:
        if task_name not in seen:
            ordered.append(task_name)
            seen.add(task_name)
    return ordered


def _sanitize_name(name: str) -> str:
    return name.replace('/', '_').replace(' ', '_')


def build_train_dataloader(
    config: dict,
    rank: int,
    world_size: int,
    cache_dirs,
    allowed_scenario_types,
    allowed_target_types,
    max_samples,
    samples_per_target_type,
    repeat_small_target_types,
    sampling_seed,
    default_target_type_partition_index,
    target_type_partition_indices,
    dataset_label: str = 'Dataset',
):
    dataset = NuPlanDataset(
        cache_dirs=cache_dirs,
        max_samples=max_samples,
        allowed_scenario_types=allowed_scenario_types,
        allowed_target_types=allowed_target_types,
        samples_per_target_type=samples_per_target_type,
        repeat_small_target_types=repeat_small_target_types,
        sampling_seed=sampling_seed,
        default_target_type_partition_index=default_target_type_partition_index,
        target_type_partition_indices=target_type_partition_indices,
    )

    if rank == 0:
        if allowed_scenario_types:
            print(f"{dataset_label} scenario filter: {allowed_scenario_types}")
        if allowed_target_types:
            print(f"{dataset_label} target filter: {allowed_target_types}")
        if samples_per_target_type is not None:
            print(
                f"{dataset_label} per-target budget: {samples_per_target_type} | "
                f"repeat_small_target_types={repeat_small_target_types} | "
                f"sampling_seed={sampling_seed}"
            )
        if default_target_type_partition_index != 0 or target_type_partition_indices:
            print(
                f"{dataset_label} partitioning: default_index={default_target_type_partition_index} | "
                f"per_type_indices={target_type_partition_indices}"
            )
        if dataset.available_sample_counts_by_target_type:
            print(f"{dataset_label} available counts by target type: {dataset.available_sample_counts_by_target_type}")
            if dataset.available_partition_sample_counts_by_target_type:
                print(
                    f"{dataset_label} available partition counts by target type: "
                    f"{dataset.available_partition_sample_counts_by_target_type}"
                )
            if dataset.partition_index_by_target_type:
                print(f"{dataset_label} partition index by target type: {dataset.partition_index_by_target_type}")
            print(f"{dataset_label} selected counts by target type: {dataset.selected_sample_counts_by_target_type}")
        print(f"{dataset_label}: {len(dataset)} samples from {len(cache_dirs)} train cache dirs")

    if world_size > 1:
        sampler = DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader_kwargs = {}
    train_num_workers = config.get('num_workers', 8)
    if train_num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = config.get('prefetch_factor', 2)
        dataloader_kwargs['persistent_workers'] = config.get('persistent_workers', True)

    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 64) // world_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=train_num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=default_collate,
        **dataloader_kwargs,
    )
    return dataset, dataloader, sampler


def resolve_continual_validation_tasks(
    source_tasks,
    seen_target_tasks,
    current_task,
    mode: str,
):
    if mode == 'source_only':
        return _dedupe_tasks(source_tasks)
    if mode == 'source_plus_current':
        return _dedupe_tasks(list(source_tasks or []) + ([current_task] if current_task else []))
    if mode == 'source_plus_seen_targets':
        return _dedupe_tasks(list(source_tasks or []) + list(seen_target_tasks or []))
    if mode == 'seen_targets':
        return _dedupe_tasks(seen_target_tasks)
    if mode == 'current_target':
        return [current_task] if current_task else []
    raise ValueError(
        f"Unsupported continual validation mode '{mode}'. "
        "Expected one of: source_only, source_plus_current, source_plus_seen_targets, seen_targets, current_target"
    )


def build_validation_dataloaders(
    config: dict,
    rank: int,
    can_run_validation: bool,
    cache_dirs,
    allowed_scenario_types,
    aggregate_target_types,
    task_metric_types,
    max_samples,
    samples_per_target_type,
    repeat_small_target_types,
    sampling_seed,
    default_target_type_partition_index,
    target_type_partition_indices,
    dataset_label: str = 'Val dataset',
    task_label_prefix: str = 'Val task dataset',
):
    val_dataloader = None
    val_task_dataloaders = {}

    if rank != 0 or not can_run_validation:
        return val_dataloader, val_task_dataloaders

    aggregate_target_types = _dedupe_tasks(aggregate_target_types)
    val_dataset, val_dataloader = build_eval_dataloader(
        config,
        cache_dirs=cache_dirs,
        allowed_scenario_types=allowed_scenario_types,
        allowed_target_types=aggregate_target_types or None,
        max_samples=max_samples,
        samples_per_target_type=samples_per_target_type,
        repeat_small_target_types=repeat_small_target_types,
        sampling_seed=sampling_seed,
        default_target_type_partition_index=default_target_type_partition_index,
        target_type_partition_indices=target_type_partition_indices,
    )
    print(f"{dataset_label}: {len(val_dataset)} samples from {len(cache_dirs)} val cache dirs")

    val_task_num_batches = config.get(
        'val_task_num_batches',
        config.get('val_num_batches', 20),
    )
    for task_name in _dedupe_tasks(task_metric_types):
        task_dataset, task_dataloader = build_eval_dataloader(
            config,
            cache_dirs=cache_dirs,
            allowed_scenario_types=allowed_scenario_types,
            allowed_target_types=[task_name],
            max_samples=config.get('val_task_max_samples', max_samples),
            samples_per_target_type=config.get('val_task_samples_per_target_type', None),
            repeat_small_target_types=repeat_small_target_types,
            sampling_seed=sampling_seed,
            default_target_type_partition_index=default_target_type_partition_index,
            target_type_partition_indices=target_type_partition_indices,
        )
        val_task_dataloaders[task_name] = {
            'dataset': task_dataset,
            'dataloader': task_dataloader,
            'max_batches': val_task_num_batches,
        }
        print(f"{task_label_prefix} [{task_name}]: {len(task_dataset)} samples")

    return val_dataloader, val_task_dataloaders


def write_json_file(path: str, payload: dict):
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def setup_distributed():
    """Initialize DDP."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl', timeout=datetime.timedelta(hours=1))
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def load_config(config_path: str) -> dict:
    """Load YAML config and flatten into a single-level dict."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Flatten: merge nested dicts for easy access
    flat = {}
    for k, v in config.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f'{k}_{kk}'] = vv
        else:
            flat[k] = v
    return config


def remap_lora_base_weights_for_warm_start(policy_module, state_dict: dict, rank: int) -> dict:
    if not getattr(policy_module.model, 'peft_active', False):
        return state_dict
    if not any(isinstance(module, LoRALinear) for module in policy_module.model.modules()):
        return state_dict

    remapped = dict(state_dict)
    mapped = 0
    dropped = 0
    for module_name, module in policy_module.model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        prefix = f'model.{module_name}'
        old_weight = f'{prefix}.weight'
        new_weight = f'{prefix}.base_weight'
        if old_weight in remapped:
            if new_weight not in remapped:
                remapped[new_weight] = remapped[old_weight]
                mapped += 1
            del remapped[old_weight]
            dropped += 1

        old_bias = f'{prefix}.bias'
        new_bias = f'{prefix}.base_bias'
        if old_bias in remapped:
            if getattr(module, 'base_bias', None) is not None and new_bias not in remapped:
                remapped[new_bias] = remapped[old_bias]
                mapped += 1
            del remapped[old_bias]
            dropped += 1

    if rank == 0 and (mapped or dropped):
        print(
            f"Remapped {mapped} pretrained Linear tensors into LoRA base buffers "
            f"and dropped {dropped} legacy Linear keys"
        )
    return remapped


def load_policy_checkpoint(policy_module, state_dict: dict, rank: int, context: str):
    allow_missing_peft = bool(getattr(policy_module.model, 'peft_active', False))
    if allow_missing_peft:
        state_dict = remap_lora_base_weights_for_warm_start(policy_module, state_dict, rank)

    incompatible = policy_module.load_state_dict(
        state_dict,
        strict=not allow_missing_peft,
    )

    if not allow_missing_peft:
        return

    non_peft_missing = [
        key for key in incompatible.missing_keys
        if 'adapter_' not in key and '.lora_A' not in key and '.lora_B' not in key
    ]
    if non_peft_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{context} produced incompatible checkpoint load. "
            f"Missing non-PEFT keys: {non_peft_missing} | "
            f"Unexpected keys: {list(incompatible.unexpected_keys)}"
        )

    if rank == 0 and incompatible.missing_keys:
        print(
            f"Initialized PEFT parameters from scratch for {len(incompatible.missing_keys)} keys"
        )


def _unwrap_policy(policy):
    return policy.module if isinstance(policy, DDP) else policy


def _pegp_cfg(config: dict) -> dict:
    return config.get('pegp') or {}


def pegp_enabled(config: dict) -> bool:
    return bool(_pegp_cfg(config).get('enable', False))


def ensure_pegp_state(policy_module) -> None:
    if not hasattr(policy_module, '_pegp_layer_bases') or policy_module._pegp_layer_bases is None:
        policy_module._pegp_layer_bases = {}
    if not hasattr(policy_module, '_pegp_task_history') or policy_module._pegp_task_history is None:
        policy_module._pegp_task_history = []


def iter_pegp_modules(policy_module):
    for name, module in policy_module.model.named_modules():
        if isinstance(module, (AdapterLayer, LoRALinear)):
            yield name, module


def validate_pegp_configuration(policy_module, config: dict) -> None:
    if not pegp_enabled(config):
        return

    if not getattr(policy_module.model, 'peft_active', False):
        raise RuntimeError(
            'PEGP requires PEFT to be enabled. '
            'Use a config with peft_config.type=adapter or peft_config.type=lora.'
        )

    if not list(iter_pegp_modules(policy_module)):
        raise RuntimeError('PEGP is enabled but no AdapterLayer or LoRALinear modules were found in the model.')

    ensure_pegp_state(policy_module)


def serialize_pegp_state(policy_module) -> dict:
    ensure_pegp_state(policy_module)
    return {
        'pegp_layer_bases': {
            name: basis.detach().cpu()
            for name, basis in policy_module._pegp_layer_bases.items()
        },
        'pegp_task_history': list(policy_module._pegp_task_history),
    }


def load_pegp_state(policy_module, ckpt: dict, rank: int, context: str) -> None:
    ensure_pegp_state(policy_module)
    layer_bases = ckpt.get('pegp_layer_bases') or {}
    task_history = ckpt.get('pegp_task_history') or []
    if not layer_bases and not task_history:
        return

    device = next(policy_module.parameters()).device
    policy_module._pegp_layer_bases = {
        name: basis.to(device).contiguous()
        for name, basis in layer_bases.items()
    }
    policy_module._pegp_task_history = list(task_history)

    if rank == 0:
        summary = {
            name: int(basis.shape[1])
            for name, basis in policy_module._pegp_layer_bases.items()
        }
        print(f'Loaded PEGP state from {context}: {summary}')


def pegp_feature_dim(module) -> int:
    if isinstance(module, AdapterLayer):
        return module.down_proj.in_features
    if isinstance(module, LoRALinear):
        return module.in_features
    raise TypeError(f'Unsupported PEGP module type: {type(module).__name__}')


def pegp_projected_grad(module):
    if isinstance(module, AdapterLayer):
        return module.down_proj.weight.grad
    if isinstance(module, LoRALinear):
        return module.lora_A.grad
    raise TypeError(f'Unsupported PEGP module type: {type(module).__name__}')


def apply_pegp_gradient_projection(policy, config: dict) -> None:
    if not pegp_enabled(config):
        return

    policy_module = _unwrap_policy(policy)
    ensure_pegp_state(policy_module)
    if not policy_module._pegp_layer_bases:
        return

    with torch.no_grad():
        for name, module in iter_pegp_modules(policy_module):
            basis = policy_module._pegp_layer_bases.get(name)
            grad = pegp_projected_grad(module)
            if basis is None or grad is None:
                continue
            projected = GradientProjector.project_linear_weight_grad(
                grad,
                basis.to(device=grad.device, dtype=grad.dtype),
            )
            if projected is not grad:
                grad.copy_(projected)


@torch.no_grad()
def update_pegp_memory_from_dataloader(
    policy,
    dataloader,
    config: dict,
    rank: int,
    world_size: int,
    task_name: str = None,
) -> None:
    if not pegp_enabled(config):
        return

    policy_module = _unwrap_policy(policy)
    validate_pegp_configuration(policy_module, config)
    ensure_pegp_state(policy_module)

    pegp_cfg = _pegp_cfg(config)
    max_batches = int(pegp_cfg.get('sample_batches', 16))
    max_rank = int(pegp_cfg.get('max_basis_rank', 16))
    energy_threshold = float(pegp_cfg.get('energy_threshold', 0.9))
    min_rank = int(pegp_cfg.get('min_basis_rank', 1))
    if max_batches <= 0:
        return

    peft_modules = list(iter_pegp_modules(policy_module))
    if not peft_modules:
        return

    device = next(policy_module.parameters()).device
    accum = {
        name: {
            'sum': torch.zeros(pegp_feature_dim(module), device=device),
            'outer_sum': torch.zeros(
                pegp_feature_dim(module),
                pegp_feature_dim(module),
                device=device,
            ),
            'count': torch.zeros(1, device=device, dtype=torch.long),
        }
        for name, module in peft_modules
    }

    def _accumulate(module_name: str, tensor: torch.Tensor) -> None:
        if tensor is None or module_name not in accum or not isinstance(tensor, torch.Tensor):
            return
        x = tensor.detach()
        if x.numel() == 0:
            return
        x = x.reshape(-1, x.shape[-1]).float()
        slot = accum[module_name]
        if x.shape[-1] != slot['sum'].shape[0]:
            return
        slot['sum'] += x.sum(0)
        slot['outer_sum'] += x.transpose(0, 1) @ x
        slot['count'] += x.shape[0]

    def make_hook(module_name: str):
        def _hook(module, inputs):
            if not inputs:
                return
            _accumulate(module_name, inputs[0])
        return _hook

    hooks = [
        module.register_forward_pre_hook(make_hook(name))
        for name, module in peft_modules
    ]

    was_training = policy.training
    policy.eval()
    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            gpu_batch = move_batch_to_device(batch)
            policy(gpu_batch)
    finally:
        for hook in hooks:
            hook.remove()
        policy.train(mode=was_training)

    if world_size > 1:
        for slot in accum.values():
            dist.all_reduce(slot['sum'], op=dist.ReduceOp.SUM)
            dist.all_reduce(slot['outer_sum'], op=dist.ReduceOp.SUM)
            dist.all_reduce(slot['count'], op=dist.ReduceOp.SUM)

    summary = {}
    for module_name, slot in accum.items():
        sample_count = int(slot['count'].item())
        if sample_count < 2:
            continue
        new_basis = GradientProjector.estimate_basis(
            feature_sum=slot['sum'],
            feature_outer_sum=slot['outer_sum'],
            sample_count=sample_count,
            energy_threshold=energy_threshold,
            max_rank=max_rank,
            min_rank=min_rank,
        )
        if new_basis.numel() == 0:
            continue

        existing_basis = policy_module._pegp_layer_bases.get(module_name)
        if existing_basis is not None:
            existing_basis = existing_basis.to(device)
        merged = GradientProjector.merge_bases(
            existing_basis=existing_basis,
            new_basis=new_basis.to(device),
            max_rank=max_rank,
        ).to(device)
        policy_module._pegp_layer_bases[module_name] = merged.contiguous()
        summary[module_name] = int(merged.shape[1])

    if task_name and (
        not policy_module._pegp_task_history
        or policy_module._pegp_task_history[-1] != task_name
    ):
        policy_module._pegp_task_history.append(task_name)

    if rank == 0 and summary:
        print(f'Updated PEGP bases after task {task_name or "<unknown>"}: {summary}')


class ConfigWrapper:
    """Simple config wrapper for attribute access."""

    def __init__(self, config: dict):
        for k, v in config.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    setattr(self, f'{k}_{kk}', vv)
            else:
                setattr(self, k, v)

    def __getattr__(self, name):
        # Return None for missing attributes (some configs are optional)
        return None


def train_epoch(
    policy,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    rank: int,
    config,
    ema_model=None,
    progress_desc: str = None,
):
    """Train one epoch."""
    policy.train()
    total_loss = 0.0
    total_ego_loss = 0.0
    total_neighbor_loss = 0.0
    show_progress = rank == 0 and config.get('use_tqdm', True)
    progress = tqdm(
        dataloader,
        desc=progress_desc or f'Train {epoch}',
        leave=False,
        dynamic_ncols=True,
    ) if show_progress else dataloader

    for batch_idx, batch in enumerate(progress):
        # Move batch to device
        gpu_batch = move_batch_to_device(batch)

        use_amp = config.get('use_amp', True)
        with autocast_cuda(use_amp, torch.float16):
            loss_dict = policy(gpu_batch)

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss_dict['loss']).backward()
            scaler.unscale_(optimizer)
            apply_pegp_gradient_projection(policy, config)
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), config.get('max_grad_norm', 5.0)
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict['loss'].backward()
            apply_pegp_gradient_projection(policy, config)
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), config.get('max_grad_norm', 5.0)
            )
            optimizer.step()

        scheduler.step()

        # EMA update
        if ema_model is not None:
            ema_model.step(policy.parameters())

        total_loss += loss_dict['loss'].item()
        total_ego_loss += loss_dict['ego_loss'].item()
        total_neighbor_loss += loss_dict['neighbor_loss'].item()

        if show_progress:
            progress.set_postfix(
                loss=f"{loss_dict['loss'].item():.4f}",
                ego=f"{loss_dict['ego_loss'].item():.4f}",
                nbr=f"{loss_dict['neighbor_loss'].item():.4f}",
            )
        elif rank == 0 and batch_idx % 50 == 0:
            print(
                f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss_dict['loss'].item():.4f} | "
                f"Ego: {loss_dict['ego_loss'].item():.4f} | "
                f"Neighbor: {loss_dict['neighbor_loss'].item():.4f}"
            )

    n = len(dataloader)
    return total_loss / n, total_ego_loss / n, total_neighbor_loss / n


@torch.no_grad()
def validate(
    policy,
    dataloader,
    rank: int,
    config: dict,
    progress_desc: str = 'Val',
    max_batches: int = None,
):
    """Validate with teacher-forced loss and sampled trajectory metrics."""
    policy.eval()
    total_loss = 0.0
    total_ego_loss = 0.0
    total_neighbor_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_pred_final_disp = 0.0
    total_gt_final_disp = 0.0
    total_samples = 0
    max_batches = config.get('val_num_batches', 20) if max_batches is None else max_batches
    inference_steps = config.get(
        'val_num_inference_steps',
        config.get('num_inference_steps', 10),
    )
    show_progress = rank == 0 and config.get('use_tqdm', True)
    progress = tqdm(
        dataloader,
        desc=progress_desc,
        leave=False,
        dynamic_ncols=True,
    ) if show_progress else dataloader

    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        gpu_batch = move_batch_to_device(batch)
        (ego_current, ego_future, neighbor_past, neighbor_future,
         lanes, lanes_sl, lanes_hsl, route_lanes, static_objs) = gpu_batch

        loss_dict = policy(gpu_batch)
        sampled = policy.conditional_sample(
            {
                'neighbor_agents_past': neighbor_past,
                'static_objects': static_objs,
                'lanes': lanes,
                'lanes_speed_limit': lanes_sl,
                'lanes_has_speed_limit': lanes_hsl,
                'route_lanes': route_lanes,
            },
            num_steps=inference_steps,
        )
        pred_ego = sampled['trajectory'][:, 0, :, :2]
        ego_error = torch.linalg.vector_norm(pred_ego - ego_future, dim=-1)
        ego_ade = ego_error.mean(dim=-1)
        ego_fde = ego_error[:, -1]
        pred_final_disp = torch.linalg.vector_norm(pred_ego[:, -1, :], dim=-1)
        gt_final_disp = torch.linalg.vector_norm(ego_future[:, -1, :], dim=-1)

        batch_size = ego_future.shape[0]
        total_samples += batch_size
        total_loss += loss_dict['loss'].item() * batch_size
        total_ego_loss += loss_dict['ego_loss'].item() * batch_size
        total_neighbor_loss += loss_dict['neighbor_loss'].item() * batch_size
        total_ade += ego_ade.sum().item()
        total_fde += ego_fde.sum().item()
        total_pred_final_disp += pred_final_disp.sum().item()
        total_gt_final_disp += gt_final_disp.sum().item()

        if show_progress:
            progress.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                ade=f"{total_ade / total_samples:.4f}",
                fde=f"{total_fde / total_samples:.4f}",
            )

    if total_samples == 0:
        return None

    return {
        'loss': total_loss / total_samples,
        'ego_loss': total_ego_loss / total_samples,
        'neighbor_loss': total_neighbor_loss / total_samples,
        'ego_ADE': total_ade / total_samples,
        'ego_FDE': total_fde / total_samples,
        'pred_final_disp': total_pred_final_disp / total_samples,
        'gt_final_disp': total_gt_final_disp / total_samples,
        'sample_count': total_samples,
    }


def compute_norm_stats(dataset_dir: str, data_list_path: str):
    """Compute global normalization stats from dataset."""
    print("Computing normalization statistics...")
    dataset = NuPlanDataset(
        data_dir=dataset_dir,
        data_list=data_list_path,
    )
    # Use a subset for efficiency
    indices = list(range(0, len(dataset), max(1, len(dataset) // 1000)))
    subset = [dataset[i] for i in indices]

    mean, std = compute_global_norm_stats(subset)
    stats = {'mean': mean.tolist(), 'std': std.tolist()}

    with open(os.path.join(dataset_dir, 'norm_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved norm stats: mean={mean}, std={std}")
    return stats


def main():
    parser = argparse.ArgumentParser(description='Train NuPlan Diffusion')
    parser.add_argument('--config', type=str, default='configs/nuplan_diffusion.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--init_ckpt', type=str, default=None, help='Initialize model weights from checkpoint without restoring optimizer or scheduler state')
    parser.add_argument('--compute_stats', action='store_true', help='Compute norm stats only')
    args_cli = parser.parse_args()

    if args_cli.resume and args_cli.init_ckpt:
        raise ValueError('--resume and --init_ckpt are mutually exclusive')

    rank, local_rank, world_size = setup_distributed()

    # Load config
    config = load_config(args_cli.config)
    config['use_ema'] = config.get('use_ema', True)

    # Create model
    policy = NuPlanDiffusionPolicy(config).cuda()

    # Load normalization stats if provided, otherwise keep identity scaling.
    norm_stats_path = config.get('norm_stats_path')
    if norm_stats_path:
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        policy.load_norm_stats(norm_stats)
        if rank == 0:
            print(
                f"Loaded normalization stats from {norm_stats_path}: "
                f"mean={norm_stats['mean']} std={norm_stats['std']}"
            )
    elif rank == 0:
        print("Using identity normalization (no stats file)")

    # DDP wrapping
    if world_size > 1:
        policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=True)

    policy_module = _unwrap_policy(policy)
    if pegp_enabled(config):
        validate_pegp_configuration(policy_module, config)

    cl_cfg = config.get('continual_learning') or {}
    if cl_cfg.get('enable', False):
        train_cache_dirs = config.get('train_cache_dirs', config.get('cache_dirs', [
            '/workspace2/z_project/exp/nuplan/cache_plantf_train_singapore/',
            '/workspace2/z_project/exp/nuplan/cache_plantf_train_boston/',
            '/workspace2/z_project/exp/nuplan/cache_plantf_train_pittsburgh/',
        ]))
        max_samples = config.get('max_samples', None)
        allowed_scenario_types = config.get('allowed_scenario_types', None)
        samples_per_target_type = config.get('samples_per_target_type', None)
        repeat_small_target_types = config.get('repeat_small_target_types', False)
        sampling_seed = config.get('sampling_seed', 0)
        default_target_type_partition_index = config.get('default_target_type_partition_index', 0)
        target_type_partition_indices = config.get('target_type_partition_indices', None)

        val_every_epochs = config.get('val_every_epochs', 5)
        enable_validation_in_ddp = config.get('enable_validation_in_ddp', True)
        can_run_validation = (
            val_every_epochs > 0 and (world_size <= 1 or enable_validation_in_ddp)
        )
        if rank == 0 and val_every_epochs > 0 and world_size > 1 and not enable_validation_in_ddp:
            print('Skipping in-process validation during DDP training; set enable_validation_in_ddp=true to override.')

        val_cache_dirs = config.get('val_cache_dirs', DEFAULT_VAL_CACHE_DIRS)
        val_allowed_scenario_types = config.get('val_allowed_scenario_types', allowed_scenario_types)
        val_max_samples = config.get('val_max_samples', None)
        val_samples_per_target_type = config.get('val_samples_per_target_type', None)
        val_repeat_small_target_types = config.get('val_repeat_small_target_types', False)
        val_sampling_seed = config.get('val_sampling_seed', sampling_seed)
        val_default_target_type_partition_index = config.get('val_default_target_type_partition_index', 0)
        val_target_type_partition_indices = config.get('val_target_type_partition_indices', None)

        checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_freq = config.get('save_freq', 20)
        experiment_dir = os.path.dirname(os.path.normpath(checkpoint_dir))

        target_tasks = _dedupe_tasks(cl_cfg.get('target_tasks'))
        source_tasks = _dedupe_tasks(cl_cfg.get('source_tasks'))
        if not target_tasks:
            raise ValueError('continual_learning.enable=true requires continual_learning.target_tasks')

        overlap_tasks = sorted(set(source_tasks) & set(target_tasks))
        if overlap_tasks:
            raise ValueError(
                f'continual_learning source/target overlap is not allowed: {overlap_tasks}'
            )

        epochs_per_task = int(cl_cfg.get('epochs_per_task', config.get('train_epochs', 500)))
        validation_cfg = cl_cfg.get('validation') or {}
        validation_mode = validation_cfg.get(
            'mode',
            'source_plus_seen_targets' if source_tasks else 'seen_targets',
        )
        continual_history_path = os.path.join(experiment_dir, 'continual_validation.json')
        continual_history = {
            'source_tasks': source_tasks,
            'target_tasks': target_tasks,
            'validation_mode': validation_mode,
            'snapshots': [],
        }
        if rank == 0 and os.path.exists(continual_history_path):
            try:
                with open(continual_history_path, 'r') as handle:
                    existing_history = json.load(handle)
                if isinstance(existing_history, dict) and isinstance(existing_history.get('snapshots'), list):
                    continual_history['snapshots'] = existing_history['snapshots']
            except (OSError, json.JSONDecodeError):
                pass

        if rank == 0:
            print(f'Continual target task order: {target_tasks}')
            print(f'Continual source tasks: {source_tasks}')
            print(
                f'Continual validation mode: {validation_mode} | '
                f'epochs_per_task={epochs_per_task}'
            )

        resume_ckpt = None
        resume_task_index = 0
        resume_start_epoch = 0
        if args_cli.resume:
            resume_ckpt = torch.load(args_cli.resume, map_location='cpu')
            if 'task_index' not in resume_ckpt:
                raise ValueError(
                    'Sequential continual-learning resume expects a stage checkpoint with task_index metadata. '
                    'Use --init_ckpt for a source pretrain warm start.'
                )
            if world_size <= 1:
                load_policy_checkpoint(policy, resume_ckpt['model'], rank, args_cli.resume)
            else:
                load_policy_checkpoint(policy.module, resume_ckpt['model'], rank, args_cli.resume)
            load_pegp_state(policy_module, resume_ckpt, rank, args_cli.resume)
            resume_task_index = int(resume_ckpt.get('task_index', 0))
            resume_start_epoch = int(resume_ckpt.get('epoch', -1)) + 1
            if resume_start_epoch >= epochs_per_task:
                resume_task_index += 1
                resume_start_epoch = 0
            if rank == 0:
                print(
                    f'Resuming continual training from task_index={resume_task_index} '
                    f'start_epoch={resume_start_epoch}'
                )
        elif args_cli.init_ckpt:
            init_ckpt = torch.load(args_cli.init_ckpt, map_location='cpu')
            if world_size <= 1:
                load_policy_checkpoint(policy, init_ckpt['model'], rank, args_cli.init_ckpt)
            else:
                load_policy_checkpoint(policy.module, init_ckpt['model'], rank, args_cli.init_ckpt)
            load_pegp_state(policy_module, init_ckpt, rank, args_cli.init_ckpt)
            if rank == 0:
                print(f'Initialized continual model weights from {args_cli.init_ckpt}')

        if resume_task_index >= len(target_tasks):
            if rank == 0:
                print('Continual training already complete for all configured target tasks.')
            if world_size > 1:
                dist.destroy_process_group()
            return

        from diffusers.training_utils import EMAModel

        stage_config = dict(config)
        stage_config['train_epochs'] = epochs_per_task

        for task_index, task_name in enumerate(target_tasks):
            if task_index < resume_task_index:
                continue

            if rank == 0:
                print(
                    f'=== Continual stage {task_index + 1}/{len(target_tasks)} | '
                    f'train_task={task_name} ==='
                )

            _, dataloader, sampler = build_train_dataloader(
                config,
                rank,
                world_size,
                cache_dirs=train_cache_dirs,
                allowed_scenario_types=allowed_scenario_types,
                allowed_target_types=[task_name],
                max_samples=max_samples,
                samples_per_target_type=samples_per_target_type,
                repeat_small_target_types=repeat_small_target_types,
                sampling_seed=sampling_seed,
                default_target_type_partition_index=default_target_type_partition_index,
                target_type_partition_indices=target_type_partition_indices,
                dataset_label=f'Train dataset [{task_name}]',
            )
            steps_per_epoch = len(dataloader)
            if steps_per_epoch <= 0:
                raise ValueError(
                    f'Train dataset for continual task {task_name} is too small for the configured batch size '
                    'after drop_last=True.'
                )

            optimizer, scheduler = (
                policy.configure_optimizers(stage_config, steps_per_epoch=steps_per_epoch)
                if world_size <= 1 else
                policy.module.configure_optimizers(stage_config, steps_per_epoch=steps_per_epoch)
            )
            scaler = GradScaler('cuda', enabled=config.get('use_amp', True))
            ema_model = EMAModel(
                policy.parameters() if world_size <= 1 else policy.module.parameters(),
                decay=config.get('ema_decay', 0.999),
            )
            ema_model.to(torch.device(f'cuda:{local_rank}'))

            start_epoch = 0
            if resume_ckpt is not None and task_index == resume_task_index:
                optimizer.load_state_dict(resume_ckpt['optimizer'])
                restore_scheduler_state(
                    scheduler,
                    optimizer,
                    resume_ckpt,
                    steps_per_epoch=steps_per_epoch,
                    total_epochs=epochs_per_task,
                    rank=rank,
                )
                if resume_ckpt.get('ema') is not None:
                    ema_model.load_state_dict(resume_ckpt['ema'])
                    ema_model.to(torch.device(f'cuda:{local_rank}'))
                if resume_ckpt.get('scaler') is not None:
                    scaler.load_state_dict(resume_ckpt['scaler'])
                start_epoch = resume_start_epoch
                if rank == 0:
                    print(
                        f'Resumed continual stage {task_index + 1}/{len(target_tasks)} '
                        f'for {task_name} from epoch {start_epoch}'
                    )

            seen_target_tasks = target_tasks[:task_index + 1]
            eval_tasks = resolve_continual_validation_tasks(
                source_tasks,
                seen_target_tasks,
                task_name,
                validation_mode,
            )
            val_dataloader, val_task_dataloaders = build_validation_dataloaders(
                config,
                rank,
                can_run_validation,
                cache_dirs=val_cache_dirs,
                allowed_scenario_types=val_allowed_scenario_types,
                aggregate_target_types=eval_tasks,
                task_metric_types=eval_tasks,
                max_samples=val_max_samples,
                samples_per_target_type=val_samples_per_target_type,
                repeat_small_target_types=val_repeat_small_target_types,
                sampling_seed=val_sampling_seed,
                default_target_type_partition_index=val_default_target_type_partition_index,
                target_type_partition_indices=val_target_type_partition_indices,
                dataset_label=(
                    f'Val dataset [after task {task_index + 1}: '
                    f'{_dedupe_tasks(eval_tasks)}]'
                ),
                task_label_prefix=f'Val task dataset [after task {task_index + 1}]',
            )
            if rank == 0:
                print(f'Continual stage {task_index + 1} validation tasks: {eval_tasks}')

            for epoch in range(start_epoch, epochs_per_task):
                if world_size > 1:
                    sampler.set_epoch(task_index * epochs_per_task + epoch)

                train_loss, ego_loss, neighbor_loss = train_epoch(
                    policy,
                    dataloader,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    rank,
                    config,
                    ema_model,
                    progress_desc=(
                        f'Train task {task_index + 1}/{len(target_tasks)} '
                        f'{task_name} epoch {epoch + 1}/{epochs_per_task}'
                    ),
                )

                if rank == 0:
                    print(
                        f'Stage {task_index + 1}/{len(target_tasks)} | '
                        f'Task: {task_name} | '
                        f'Epoch {epoch}/{epochs_per_task} | '
                        f'Loss: {train_loss:.4f} | '
                        f'Ego: {ego_loss:.4f} | '
                        f'Neighbor: {neighbor_loss:.4f}'
                    )

                should_validate = (
                    can_run_validation
                    and ((epoch + 1) % val_every_epochs == 0 or epoch == epochs_per_task - 1)
                )
                if should_validate and rank == 0 and val_dataloader is not None:
                    model_for_val = policy if world_size <= 1 else policy.module
                    aggregate_metrics = validate(
                        model_for_val,
                        val_dataloader,
                        rank,
                        config,
                        progress_desc=(
                            f'Val stage {task_index + 1}/{len(target_tasks)} '
                            f'{task_name}'
                        ),
                    )
                    if aggregate_metrics is not None:
                        print(
                            format_metrics(
                                f'ValStage {task_index + 1}/{len(target_tasks)} '
                                f'{task_name} {epoch}/{epochs_per_task}',
                                aggregate_metrics,
                            )
                        )

                    per_task_metrics = {}
                    for eval_task_name, task_bundle in val_task_dataloaders.items():
                        task_metrics = validate(
                            model_for_val,
                            task_bundle['dataloader'],
                            rank,
                            config,
                            progress_desc=f'Val {eval_task_name}',
                            max_batches=task_bundle['max_batches'],
                        )
                        if task_metrics is not None:
                            per_task_metrics[eval_task_name] = task_metrics
                            print(
                                format_metrics(
                                    f'ValTask {eval_task_name} after_task_{task_index + 1} '
                                    f'{task_name} {epoch}/{epochs_per_task}',
                                    task_metrics,
                                )
                            )

                    if epoch == epochs_per_task - 1:
                        continual_history['snapshots'].append({
                            'completed_task_index': task_index + 1,
                            'current_task': task_name,
                            'seen_target_tasks': seen_target_tasks,
                            'aggregate_metrics': aggregate_metrics,
                            'per_task_metrics': per_task_metrics,
                        })
                        write_json_file(continual_history_path, continual_history)

                if should_validate and world_size > 1:
                    dist.barrier()

                if epoch == epochs_per_task - 1:
                    update_pegp_memory_from_dataloader(
                        policy,
                        dataloader,
                        config,
                        rank,
                        world_size,
                        task_name=task_name,
                    )
                    if world_size > 1:
                        dist.barrier()

                if rank == 0:
                    if (epoch + 1) % save_freq == 0 or epoch == epochs_per_task - 1:
                        model_state = policy.state_dict() if world_size <= 1 else policy.module.state_dict()
                        pegp_state = serialize_pegp_state(policy_module)
                        ckpt_name = (
                            f'task_{task_index + 1:02d}_{_sanitize_name(task_name)}_'
                            f'epoch_{epoch:04d}.pth'
                        )
                        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                        torch.save({
                            'epoch': epoch,
                            'model': model_state,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'ema': ema_model.state_dict() if ema_model else None,
                            'scaler': scaler.state_dict(),
                            'global_step': scheduler.last_epoch,
                            'steps_per_epoch': steps_per_epoch,
                            'scheduler_total_steps': getattr(scheduler, 'total_steps', epochs_per_task * steps_per_epoch),
                            'lr_schedule_unit': getattr(scheduler, 'lr_schedule_unit', 'step'),
                            'task_index': task_index,
                            'task_name': task_name,
                            'source_tasks': source_tasks,
                            'target_tasks': target_tasks,
                            'seen_target_tasks': seen_target_tasks,
                            'continual_validation_mode': validation_mode,
                            'continual_epochs_per_task': epochs_per_task,
                            **pegp_state,
                        }, ckpt_path)
                        print(f'Saved checkpoint: {ckpt_path}')

            resume_ckpt = None
            resume_start_epoch = 0

        if rank == 0:
            print('Continual training complete!')

        if world_size > 1:
            dist.destroy_process_group()
        return

    # Dataset: direct from PlanTF cache
    train_cache_dirs = config.get('train_cache_dirs', config.get('cache_dirs', [
        '/workspace2/z_project/exp/nuplan/cache_plantf_train_singapore/',
        '/workspace2/z_project/exp/nuplan/cache_plantf_train_boston/',
        '/workspace2/z_project/exp/nuplan/cache_plantf_train_pittsburgh/',
    ]))
    max_samples = config.get('max_samples', None)
    allowed_scenario_types = config.get('allowed_scenario_types', None)
    allowed_target_types = config.get('allowed_target_types', None)
    samples_per_target_type = config.get('samples_per_target_type', None)
    repeat_small_target_types = config.get('repeat_small_target_types', False)
    sampling_seed = config.get('sampling_seed', 0)
    default_target_type_partition_index = config.get('default_target_type_partition_index', 0)
    target_type_partition_indices = config.get('target_type_partition_indices', None)
    dataset = NuPlanDataset(
        cache_dirs=train_cache_dirs,
        max_samples=max_samples,
        allowed_scenario_types=allowed_scenario_types,
        allowed_target_types=allowed_target_types,
        samples_per_target_type=samples_per_target_type,
        repeat_small_target_types=repeat_small_target_types,
        sampling_seed=sampling_seed,
        default_target_type_partition_index=default_target_type_partition_index,
        target_type_partition_indices=target_type_partition_indices,
    )

    if rank == 0:
        if allowed_scenario_types:
            print(f"Filtering to scenario types: {allowed_scenario_types}")
        if allowed_target_types:
            print(f"Filtering to target types: {allowed_target_types}")
        if samples_per_target_type is not None:
            print(
                f"Per-target sample budget: {samples_per_target_type} | "
                f"repeat_small_target_types={repeat_small_target_types} | "
                f"sampling_seed={sampling_seed}"
            )
        if default_target_type_partition_index != 0 or target_type_partition_indices:
            print(
                f"Target-type partitioning: default_index={default_target_type_partition_index} | "
                f"per_type_indices={target_type_partition_indices}"
            )
        if dataset.available_sample_counts_by_target_type:
            print(f"Available counts by target type: {dataset.available_sample_counts_by_target_type}")
            if dataset.available_partition_sample_counts_by_target_type:
                print(f"Available partition counts by target type: {dataset.available_partition_sample_counts_by_target_type}")
            if dataset.partition_index_by_target_type:
                print(f"Partition index by target type: {dataset.partition_index_by_target_type}")
            print(f"Selected counts by target type: {dataset.selected_sample_counts_by_target_type}")
        print(f"Dataset: {len(dataset)} samples from {len(train_cache_dirs)} train cache dirs")

    if world_size > 1:
        sampler = DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader_kwargs = {}
    train_num_workers = config.get('num_workers', 8)
    if train_num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = config.get('prefetch_factor', 2)
        dataloader_kwargs['persistent_workers'] = config.get('persistent_workers', True)

    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 64) // world_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=train_num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=default_collate,
        **dataloader_kwargs,
    )

    val_every_epochs = config.get('val_every_epochs', 5)
    enable_validation_in_ddp = config.get('enable_validation_in_ddp', True)
    val_dataloader = None
    val_task_dataloaders = {}
    can_run_validation = (
        val_every_epochs > 0 and (world_size <= 1 or enable_validation_in_ddp)
    )
    if rank == 0 and val_every_epochs > 0 and world_size > 1 and not enable_validation_in_ddp:
        print('Skipping in-process validation during DDP training; set enable_validation_in_ddp=true to override.')

    if rank == 0 and can_run_validation:
        val_cache_dirs = config.get('val_cache_dirs', DEFAULT_VAL_CACHE_DIRS)
        val_allowed_scenario_types = config.get('val_allowed_scenario_types', allowed_scenario_types)
        val_allowed_target_types = config.get('val_allowed_target_types', allowed_target_types)
        val_max_samples = config.get('val_max_samples', None)
        val_samples_per_target_type = config.get('val_samples_per_target_type', None)
        val_repeat_small_target_types = config.get('val_repeat_small_target_types', False)
        val_sampling_seed = config.get('val_sampling_seed', sampling_seed)
        val_default_target_type_partition_index = config.get('val_default_target_type_partition_index', 0)
        val_target_type_partition_indices = config.get('val_target_type_partition_indices', None)

        val_dataset, val_dataloader = build_eval_dataloader(
            config,
            cache_dirs=val_cache_dirs,
            allowed_scenario_types=val_allowed_scenario_types,
            allowed_target_types=val_allowed_target_types,
            max_samples=val_max_samples,
            samples_per_target_type=val_samples_per_target_type,
            repeat_small_target_types=val_repeat_small_target_types,
            sampling_seed=val_sampling_seed,
            default_target_type_partition_index=val_default_target_type_partition_index,
            target_type_partition_indices=val_target_type_partition_indices,
        )
        print(f"Val dataset: {len(val_dataset)} samples from {len(val_cache_dirs)} val cache dirs")

        val_task_metrics = config.get('val_task_metrics', False)
        val_task_metric_types = config.get('val_task_metric_types', None)
        if val_task_metric_types is None:
            val_task_metric_types = val_allowed_target_types or allowed_target_types or []

        if val_task_metrics:
            val_task_max_samples = config.get('val_task_max_samples', val_max_samples)
            val_task_num_batches = config.get('val_task_num_batches', config.get('val_num_batches', 20))
            for task_name in val_task_metric_types:
                task_dataset, task_dataloader = build_eval_dataloader(
                    config,
                    cache_dirs=val_cache_dirs,
                    allowed_scenario_types=val_allowed_scenario_types,
                    allowed_target_types=[task_name],
                    max_samples=val_task_max_samples,
                    samples_per_target_type=config.get('val_task_samples_per_target_type', None),
                    repeat_small_target_types=val_repeat_small_target_types,
                    sampling_seed=val_sampling_seed,
                    default_target_type_partition_index=val_default_target_type_partition_index,
                    target_type_partition_indices=val_target_type_partition_indices,
                )
                val_task_dataloaders[task_name] = {
                    'dataset': task_dataset,
                    'dataloader': task_dataloader,
                    'max_batches': val_task_num_batches,
                }
                print(f"Val task dataset [{task_name}]: {len(task_dataset)} samples")

    steps_per_epoch = len(dataloader)

    # Optimizer
    optimizer, scheduler = policy.configure_optimizers(config, steps_per_epoch=steps_per_epoch) if world_size <= 1 else \
                           policy.module.configure_optimizers(config, steps_per_epoch=steps_per_epoch)

    # AMP
    scaler = GradScaler('cuda', enabled=config.get('use_amp', True))

    # EMA
    ema_model = None
    from diffusers.training_utils import EMAModel
    decay = config.get('ema_decay', 0.999)
    ema_model = EMAModel(
        policy.parameters() if world_size <= 1 else policy.module.parameters(),
        decay=decay,
    )
    ema_model.to(torch.device(f'cuda:{local_rank}'))

    # Resume
    start_epoch = 0
    total_epochs = config.get('train_epochs', 500)
    if args_cli.resume:
        ckpt = torch.load(args_cli.resume, map_location='cpu')
        if world_size <= 1:
            load_policy_checkpoint(policy, ckpt['model'], rank, args_cli.resume)
        else:
            load_policy_checkpoint(policy.module, ckpt['model'], rank, args_cli.resume)
        load_pegp_state(policy_module, ckpt, rank, args_cli.resume)
        optimizer.load_state_dict(ckpt['optimizer'])
        restore_scheduler_state(
            scheduler,
            optimizer,
            ckpt,
            steps_per_epoch=steps_per_epoch,
            total_epochs=total_epochs,
            rank=rank,
        )
        if ema_model is not None and ckpt.get('ema') is not None:
            ema_model.load_state_dict(ckpt['ema'])
            ema_model.to(torch.device(f'cuda:{local_rank}'))
        if ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}")
    elif args_cli.init_ckpt:
        ckpt = torch.load(args_cli.init_ckpt, map_location='cpu')
        if world_size <= 1:
            load_policy_checkpoint(policy, ckpt['model'], rank, args_cli.init_ckpt)
        else:
            load_policy_checkpoint(policy.module, ckpt['model'], rank, args_cli.init_ckpt)
        load_pegp_state(policy_module, ckpt, rank, args_cli.init_ckpt)
        if rank == 0:
            print(f"Initialized model weights from {args_cli.init_ckpt}")

    # Training loop
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_freq = config.get('save_freq', 20)

    for epoch in range(start_epoch, total_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)

        train_loss, ego_loss, neighbor_loss = train_epoch(
            policy, dataloader, optimizer, scheduler, scaler,
            epoch, rank, config, ema_model,
        )

        if rank == 0:
            print(
                f"Epoch {epoch}/{total_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Ego: {ego_loss:.4f} | "
                f"Neighbor: {neighbor_loss:.4f}"
            )

        should_validate = (
            can_run_validation
            and ((epoch + 1) % val_every_epochs == 0 or epoch == total_epochs - 1)
        )
        if should_validate and rank == 0 and val_dataloader is not None:
            model_for_val = policy if world_size <= 1 else policy.module
            val_metrics = validate(model_for_val, val_dataloader, rank, config)
            if val_metrics is not None:
                print(format_metrics(f"Val {epoch}/{total_epochs}", val_metrics))

            for task_name, task_bundle in val_task_dataloaders.items():
                task_metrics = validate(
                    model_for_val,
                    task_bundle['dataloader'],
                    rank,
                    config,
                    progress_desc=f'Val {task_name}',
                    max_batches=task_bundle['max_batches'],
                )
                if task_metrics is not None:
                    print(
                        format_metrics(
                            f"ValTask {task_name} {epoch}/{total_epochs}",
                            task_metrics,
                        )
                    )

        if should_validate and world_size > 1:
            dist.barrier()

        if rank == 0:
            # Save checkpoint
            if (epoch + 1) % save_freq == 0 or epoch == total_epochs - 1:
                model_state = policy.state_dict() if world_size <= 1 else policy.module.state_dict()
                pegp_state = serialize_pegp_state(policy_module)
                ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch:04d}.pth')
                torch.save({
                    'epoch': epoch,
                    'model': model_state,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'ema': ema_model.state_dict() if ema_model else None,
                    'scaler': scaler.state_dict(),
                    'global_step': scheduler.last_epoch,
                    'steps_per_epoch': steps_per_epoch,
                    'scheduler_total_steps': getattr(scheduler, 'total_steps', total_epochs * steps_per_epoch),
                    'scheduler_final_learning_rate': getattr(scheduler, 'final_learning_rate', 0.0),
                    'lr_schedule_unit': getattr(scheduler, 'lr_schedule_unit', 'step'),
                    **pegp_state,
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

    if rank == 0:
        print("Training complete!")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
