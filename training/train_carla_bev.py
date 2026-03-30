#!/usr/bin/env python3
import os
import sys
import torch
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
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import argparse
import pickle
import datetime
from torch.distributed.elastic.multiprocessing.errors import record
from diffusers.training_utils import EMAModel


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from dataset.unified_carla_dataset import CARLAImageDataset
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy
from policy.annealed_energy_guidance_policy import AnnealedEnergyGuidancePolicy

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

def validate_model(policy, val_loader, device, rank=0, world_size=1, use_amp=False, amp_dtype=torch.float16, max_batches=None):
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
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            with autocast_cuda(use_amp, amp_dtype):
                # Route B can explicitly choose unified or split-forward validation.
                if hasattr(model_for_inference, 'compute_unified_loss') and \
                   getattr(model_for_inference, 'anchor_centers_abs', None) is not None:
                    loss_dict = model_for_inference(batch, return_loss_dict=True, phase=route_b_phase)
                else:
                    loss_dict = model_for_inference.compute_loss(batch)
                loss = loss_dict['total_loss']
            if rank == 0:
                val_metrics['loss'].append(loss.item())
                val_metrics['cls_loss'].append(loss_dict['cls_loss'].item())
                val_metrics['reg_loss'].append(loss_dict['reg_loss'].item())
                val_metrics['route_loss'].append(loss_dict['route_loss'].item())
                if 'energy_loss' in loss_dict:
                    val_metrics['energy_loss'].append(loss_dict['energy_loss'].item())
                    val_metrics['alignment_loss'].append(loss_dict['alignment_loss'].item())
                if 'speed_loss' in loss_dict:
                    val_metrics['speed_loss'].append(loss_dict['speed_loss'].item())
                
                # Multimodal model: only needs bev_feature, bev_feature_upsample, ego_status
                # No more reasoning_query_tokens or anchor needed (anchor is loaded from wp_tokens.pkl)
                obs_dict = {
                    'transfuser_bev_feature': batch['transfuser_bev_feature'],
                    'transfuser_bev_feature_upsample': batch['transfuser_bev_feature_upsample'],
                    'ego_status': batch['ego_status'][:, :model_for_inference.n_obs_steps],
                }
                if getattr(model_for_inference, 'use_vqa_anchor', False) and 'vqa_anchor' in batch:
                    obs_dict['vqa_anchor'] = batch['vqa_anchor']
                target_actions = batch['agent_pos']
                
                try:
                    # Model always returns route prediction
                    result = model_for_inference.predict_action(obs_dict, no_noise=False,use_server_style=False, gt_trajectory=target_actions)
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
                                obs_dict, no_noise=False, use_server_style=False, gt_trajectory=target_actions
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
    return averaged_metrics

@record  # Records error and tracebacks in case of failure
def train_pdm_policy(config_path, resume_path=None, val_only=False):
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
    use_vqa_anchor = config.get('use_vqa_anchor', False)

    # Load anchor_centers_abs for semantic behavior labeling
    policy_type = config.get('policy_type', 'anchor')  # 'anchor' (Route A) or 'anchor_free' (Route B)
    semantic_behavior_cfg = config.get('semantic_behavior', {})
    anchor_centers_abs = None
    # Route B+ automatically enables semantic behavior (energy heads need labels)
    if policy_type == 'anchor_free' and not semantic_behavior_cfg.get('enabled', False):
        anchor_path = config.get('anchor_path', None)
        if anchor_path and os.path.exists(anchor_path):
            semantic_behavior_cfg = {'enabled': True}
            if rank == 0:
                print(f"[Route B+] Auto-enabling semantic behavior for energy head training")
    if semantic_behavior_cfg.get('enabled', False):
        anchor_path = config.get('anchor_path', None)
        if anchor_path and os.path.exists(anchor_path):
            if anchor_path.endswith('.npy'):
                anchor_centers_abs = np.load(anchor_path)  # (M, T, 2)
            else:
                with open(anchor_path, 'rb') as f:
                    anchor_data = pickle.load(f)
                anchor_centers_abs = anchor_data['centers']  # (M, T, 2)
            if rank == 0:
                print(f"[Semantic Behavior] Loaded anchor centers: {anchor_centers_abs.shape}")
        else:
            if rank == 0:
                print(f"[Semantic Behavior] WARNING: anchor file not found: {anchor_path}, disabling")
            semantic_behavior_cfg = {}

    gps_noise_cfg = config.get('augmentation', {}).get('gps_noise', {})

    feature_suffix = config.get('dataset', {}).get('feature_suffix', '')
    train_dataset = CARLAImageDataset(
        dataset_path=train_dataset_path, image_data_root=image_data_root,
        use_per_frame=use_per_frame, use_vqa_anchor=use_vqa_anchor,
        anchor_centers_abs=anchor_centers_abs, semantic_behavior_cfg=semantic_behavior_cfg,
        cache_dir=cache_dir, feature_suffix=feature_suffix,
        gps_noise_cfg=gps_noise_cfg,
    )
    # Val dataset: skip memmap, will inject RAM features after config is parsed
    val_dataset_orig = CARLAImageDataset(
        dataset_path=val_dataset_path, image_data_root=image_data_root,
        skip_memmap=True, use_per_frame=use_per_frame, use_vqa_anchor=use_vqa_anchor,
        anchor_centers_abs=anchor_centers_abs, semantic_behavior_cfg=semantic_behavior_cfg,
    )
    # if val_only:
    #     val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset_orig])
    # else:
    val_dataset = val_dataset_orig

    if rank == 0:
        print(f"\nTraining samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}" + (" (train+val combined)" if val_only else ""))
    

    
    dataloader_cfg = config.get('dataloader', {})
    training_cfg = config.get('training', {})
    validation_cfg = config.get('validation', {})

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

    validation_freq = int(validation_cfg.get('freq', training_cfg.get('validation_freq', 1)))
    raw_val_max_batches = validation_cfg.get('max_batches', 16)
    if raw_val_max_batches in (None, 0, "0"):
        val_max_batches = None
    else:
        val_max_batches = int(raw_val_max_batches)
        if val_max_batches <= 0:
            val_max_batches = None

    # Inject val features from train's memmap into RAM (only this rank's samples)
    _max_val_per_rank = val_max_batches * val_batch_size if val_max_batches else None
    if not use_per_frame:
        val_dataset_orig.inject_ram_features(train_dataset, rank=rank, world_size=world_size, max_val_samples=_max_val_per_rank)

    def safe_collate(batch):
        try:
            return default_collate(batch)
        except RuntimeError as e:
            # Print mismatched shapes for quick diagnosis
            for key in batch[0]:
                vals = [b[key] for b in batch if isinstance(b.get(key), torch.Tensor)]
                if vals:
                    shapes = set(v.shape for v in vals)
                    if len(shapes) > 1:
                        print(f"[COLLATE] shape mismatch '{key}': {shapes}", flush=True)
            raise

    # Use DistributedSampler for multi-GPU training
    if world_size > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True
        )
        # For validation, only rank 0 needs the full dataset
        # Other ranks don't participate in validation
        sampler_val = None
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
    else:
        sampler_train = None
        sampler_val = None
        if use_route_group_sampler:
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
    policy_type = config.get('policy_type', 'anchor')  # 'anchor' (Route A) or 'anchor_free' (Route B)
    if policy_type == 'anchor_free':
        policy = AnnealedEnergyGuidancePolicy(config).to(device)
        if rank == 0:
            print(f"  Policy: AnnealedEnergyGuidancePolicy (Route B+ - anchor-free)")
        # Register normalization stats (before DDP wrapping)
        # Priority: global_abs > per-step abs > delta (legacy)
        global_abs_stats_path = config.get('global_abs_stats_path', None)
        abs_stats_path = config.get('abs_stats_path', None)
        delta_stats_path = config.get('delta_stats_path', None)
        route_abs_stats_path = config.get('route_abs_stats_path', None)
        if global_abs_stats_path:
            gdata = np.load(global_abs_stats_path)
            policy.register_global_abs_stats(gdata['global_abs_mean'], gdata['global_abs_std'])
            if rank == 0:
                print(f"  ✓ Global abs stats registered: mean={gdata['global_abs_mean']}, std={gdata['global_abs_std']}")
        elif abs_stats_path:
            abs_data = np.load(abs_stats_path)
            policy.register_abs_stats(abs_data['abs_mean'], abs_data['abs_std'])
            if rank == 0:
                print(f"  ✓ Per-step abs stats registered: mean={abs_data['abs_mean'].shape}, std={abs_data['abs_std'].shape}")
        elif delta_stats_path:
            delta_data = np.load(delta_stats_path)
            policy.register_delta_stats(delta_data['delta_mean'], delta_data['delta_std'])
            if rank == 0:
                print(f"  ✓ Delta stats registered (legacy): mean={delta_data['delta_mean'].shape}, std={delta_data['delta_std'].shape}")
        else:
            if rank == 0:
                print(f"  ⚠ No normalization stats configured — will fail!")
        if route_abs_stats_path:
            route_data = np.load(route_abs_stats_path)
            policy.register_route_abs_stats(route_data['route_abs_mean'], route_data['route_abs_std'])
            if rank == 0:
                print(
                    f"  ✓ Route abs stats registered: mean={route_data['route_abs_mean'].shape}, "
                    f"std={route_data['route_abs_std'].shape}"
                )
        else:
            raise ValueError("route_abs_stats_path is required for joint Route B ego diffusion")
        # Register anchor trajectories for energy head training (before DDP wrapping)
        anchor_path = config.get('anchor_path', None)
        if anchor_path:
            if anchor_path.endswith('.npy'):
                anchor_centers = np.load(anchor_path)  # (M, T, 2)
            else:
                import pickle
                with open(anchor_path, 'rb') as f:
                    anchor_data = pickle.load(f)
                anchor_centers = anchor_data['centers']  # (M, T, 2)
            policy.register_anchor_centers(anchor_centers)
            if rank == 0:
                print(f"  ✓ Anchor centers registered for energy training: {anchor_centers.shape}")
        else:
            if rank == 0:
                print(f"  ⚠ No anchor_path configured — energy heads will not be trained on anchors")
    else:
        policy = DiffusionDiTCarlaPolicy(config).to(device)
        if rank == 0:
            print(f"  Policy: DiffusionDiTCarlaPolicy (Route A - anchor-based)")

    # Resume from checkpoint if specified
    start_epoch = 0
    checkpoint = None
    if resume_path is not None:
        if rank == 0:
            print(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        missing_keys, unexpected_keys = policy.load_state_dict(
            checkpoint['model_state_dict'],
            strict=False,
        )
        start_epoch = checkpoint.get('epoch', 0) + 1
        if rank == 0:
            print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if missing_keys:
                preview = missing_keys[:8]
                suffix = "..." if len(missing_keys) > 8 else ""
                print(f"  Missing model keys during warm-start: {preview}{suffix}")
            if unexpected_keys:
                preview = unexpected_keys[:8]
                suffix = "..." if len(unexpected_keys) > 8 else ""
                print(f"  Unexpected model keys during warm-start: {preview}{suffix}")
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
    # Route B+: dual optimizers (GAN-style D/G isolation)
    # Route A: single optimizer (unchanged)
    optimizer_energy = None
    policy_for_params = policy.module if world_size > 1 else policy
    route_b_cfg = config.get('route_b', {})
    route_b_phase = 'split' if route_b_cfg.get('use_split_forward', False) else 'unified'

    if policy_type == 'anchor_free':
        # Separate energy head params from decoder params
        energy_param_ids = set()
        energy_params = []
        for head_name in ['energy_front_head', 'energy_left_head', 'energy_right_head',
                          'energy_pedestrian_head', 'energy_offroad_head', 'energy_route_head']:
            head = getattr(policy_for_params.model, head_name)
            for p in head.parameters():
                energy_param_ids.add(id(p))
                energy_params.append(p)

        diff_params = [p for p in policy_for_params.parameters() if id(p) not in energy_param_ids]

        optimizer = torch.optim.AdamW(diff_params, lr=lr, weight_decay=weight_decay)
        optimizer_energy = torch.optim.AdamW(energy_params, lr=lr, weight_decay=weight_decay)
        if rank == 0:
            print(f"✓ Dual optimizers: decoder ({len(diff_params)} param groups) + energy ({len(energy_params)} param groups)")
    else:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler with warmup for multi-GPU training stability
    warmup_epochs = int(config.get('training', {}).get('warmup_epochs', 5))
    lr_final = float(config.get('training', {}).get('lr_final', 1e-7))
    use_lr_scheduler = config.get('training', {}).get('use_lr_scheduler', True)

    if use_lr_scheduler:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

        total_epochs = int(config.get('training', {}).get('num_epochs', 50))

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=lr_final
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        # Energy optimizer also gets a scheduler (Route B+)
        scheduler_energy = None
        if optimizer_energy is not None:
            warmup_energy = LinearLR(optimizer_energy, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            cosine_energy = CosineAnnealingLR(optimizer_energy, T_max=total_epochs - warmup_epochs, eta_min=lr_final)
            scheduler_energy = SequentialLR(optimizer_energy, schedulers=[warmup_energy, cosine_energy], milestones=[warmup_epochs])

        if rank == 0:
            print(f"✓ Learning rate scheduler: {warmup_epochs} epochs warmup + cosine annealing to {lr_final}")
    else:
        scheduler = None
        scheduler_energy = None
        if rank == 0:
            print("✓ No learning rate scheduler used")

    # EMA (Exponential Moving Average) for stable inference
    ema_cfg = config.get('ema', {})
    model_for_ema = policy.module if world_size > 1 else policy
    ema_model = EMAModel(model_for_ema.parameters(), max_value=ema_cfg.get('max_value', 0.9999))
    ema_model.to(device)  # Keep EMA on same device as model for .step() compatibility
    ema_update_interval = ema_cfg.get('update_interval', 10)  # Update every N steps
    # Restore EMA state from checkpoint if available
    if checkpoint is not None and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        if rank == 0:
            print("  ✓ EMA state restored")
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
        if optimizer_energy is not None and 'optimizer_energy_state_dict' in checkpoint:
            try:
                optimizer_energy.load_state_dict(checkpoint['optimizer_energy_state_dict'])
                if rank == 0:
                    print("  ✓ Energy optimizer state restored")
            except Exception:
                if rank == 0:
                    print("  ⚠ Could not restore energy optimizer state")

    # 设置 checkpoint 目录
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', "/media/z/data/mzq/others/MoT-DP/checkpoints/carla_dit")
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
        if rank == 0:
            print("=" * 60)
            print("Running validation only (--val_only mode)")
            print("=" * 60)
        try:
            val_metrics = validate_model(
                policy, val_loader, device, rank=rank, world_size=world_size,
                use_amp=use_amp, amp_dtype=amp_dtype, max_batches=val_max_batches
            )
            if rank == 0:
                print(f"\n✓ Validation completed")
                print(f"Validation metrics: (total {len(val_metrics)} metrics)")
                l2_keys = ['val_ADE', 'val_L2_1s', 'val_L2_2s', 'val_L2_3s', 'val_L2_avg',
                            'val_ADE_1step', 'val_L2_1s_1step', 'val_L2_2s_1step', 'val_L2_3s_1step', 'val_L2_avg_1step']
                for key in l2_keys:
                    if key in val_metrics:
                        tag = " (1-step)" if "_1step" in key else ""
                        print(f"  >>> {key}: {val_metrics[key]:.4f}{tag}")
                for key, value in val_metrics.items():
                    if key not in l2_keys:
                        print(f"  {key}: {value:.4f}")
        except Exception as e:
            if rank == 0:
                print(f"✗ Error during validation: {e}")
                import traceback
                traceback.print_exc()

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

        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        else:
            pbar = train_loader
            
        for batch_idx, batch in enumerate(pbar):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            max_grad_norm = config.get('training', {}).get('max_grad_norm', 1.0)

            if policy_type == 'anchor_free' and optimizer_energy is not None:
                # ===== Route B+: Explicit unified/split training path =====
                # Split-forward is the default for the MOA refactor branch.

                optimizer_energy.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)

                with autocast_cuda(use_amp, amp_dtype):
                    loss_dict = policy(batch, return_loss_dict=True, phase=route_b_phase)
                    total_loss = loss_dict['total_loss']

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    if rank == 0:
                        print(f"Warning: NaN/Inf total_loss at batch {batch_idx}, skipping")
                    continue

                scaler.scale(total_loss).backward()

                # Clip and step energy optimizer
                scaler.unscale_(optimizer_energy)
                torch.nn.utils.clip_grad_norm_(energy_params, max_norm=max_grad_norm)
                scaler.step(optimizer_energy)

                # Clip and step diffusion optimizer
                scaler.unscale_(optimizer)
                diff_params_for_clip = [p for p in (policy.module if world_size > 1 else policy).parameters()
                                        if id(p) not in energy_param_ids]
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(diff_params_for_clip, max_norm=max_grad_norm)

                if torch.isnan(grad_norm_before_clip) or torch.isinf(grad_norm_before_clip):
                    if rank == 0:
                        print(f"Warning: NaN/Inf gradient at batch {batch_idx}, skipping")
                    optimizer.zero_grad()
                    optimizer_energy.zero_grad()
                    scaler.update()
                    continue

                scaler.step(optimizer)
                scaler.update()

                loss = total_loss
            else:
                # ===== Route A: Single-phase training (unchanged) =====
                optimizer.zero_grad(set_to_none=True)
                with autocast_cuda(use_amp, amp_dtype):
                    loss_dict = policy(batch, return_loss_dict=True)
                    loss = loss_dict['total_loss']

                if torch.isnan(loss) or torch.isinf(loss):
                    if rank == 0:
                        print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                params = policy.module.parameters() if world_size > 1 else policy.parameters()
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)

                if torch.isnan(grad_norm_before_clip) or torch.isinf(grad_norm_before_clip):
                    if rank == 0:
                        print(f"Warning: NaN/Inf gradient at batch {batch_idx}, skipping")
                    optimizer.zero_grad()
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
                if 'energy_loss' in loss_dict:
                    postfix['E'] = f'{loss_dict["energy_loss"].item():.3f}'
                if 'alignment_loss' in loss_dict:
                    postfix['align'] = f'{loss_dict["alignment_loss"].item():.3f}'
                if 'speed_loss' in loss_dict:
                    postfix['spd'] = f'{loss_dict["speed_loss"].item():.3f}'
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
                for lk in ('cls_loss', 'reg_loss', 'route_loss', 'speed_loss',
                           'energy_loss', 'alignment_loss',
                           'energy_front_loss', 'energy_left_loss', 'energy_right_loss',
                           'energy_ped_loss', 'energy_off_loss', 'energy_route_loss'):
                    if lk in loss_dict:
                        val = loss_dict[lk]
                        log_data[f"train/{lk}"] = val.item() if isinstance(val, torch.Tensor) else val
                # Weighted losses (Route A only)
                for wk in ('cls_loss_weighted', 'reg_loss_weighted', 'route_loss_weighted'):
                    if wk in loss_dict:
                        log_data[f"train/{wk}"] = loss_dict[wk].item()
                # Semantic behavior losses (if available)
                if 'behavior_loss' in loss_dict:
                    log_data["train/behavior_loss"] = loss_dict['behavior_loss'].item()
                    log_data["train/allowed_loss"] = loss_dict['allowed_loss'].item()
                # Energy optimizer LR (Route B+)
                if optimizer_energy is not None:
                    log_data["train/lr_energy"] = optimizer_energy.param_groups[0]['lr']
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
                print(f"  Learning rate (decoder): {current_lr:.2e}")
        if scheduler_energy is not None:
            scheduler_energy.step()
            if rank == 0:
                current_lr_e = optimizer_energy.param_groups[0]['lr']
                print(f"  Learning rate (energy): {current_lr_e:.2e}")
        
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
            if optimizer_energy is not None:
                ckpt_data['optimizer_energy_state_dict'] = optimizer_energy.state_dict()
                ckpt_data['scheduler_energy_state_dict'] = scheduler_energy.state_dict() if scheduler_energy is not None else None
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

        if (epoch + 1) % validation_freq == 0:
            # Apply EMA weights for validation
            ema_model.store(model_for_ema.parameters())
            ema_model.copy_to(model_for_ema.parameters())

            if rank == 0:
                print(f"Validating with EMA weights (Epoch {epoch+1}/{num_epochs})...")
            try:
                val_metrics = validate_model(
                    policy, val_loader, device, rank=rank, world_size=world_size,
                    use_amp=use_amp, amp_dtype=amp_dtype, max_batches=val_max_batches
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

                print(f"Validation metrics: (total {len(val_metrics)} metrics)")
                l2_keys = ['val_ADE', 'val_L2_1s', 'val_L2_2s', 'val_L2_3s', 'val_L2_avg',
                            'val_ADE_1step', 'val_L2_1s_1step', 'val_L2_2s_1step', 'val_L2_3s_1step', 'val_L2_avg_1step']
                for key in l2_keys:
                    if key in val_metrics:
                        tag = " (1-step)" if "_1step" in key else ""
                        print(f"  >>> {key}: {val_metrics[key]:.4f}{tag}")
                for key, value in val_metrics.items():
                    if key not in l2_keys:
                        print(f"  {key}: {value:.4f}")

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
        print(f"Best L2_avg: {best_l2_avg:.4f}")
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
    parser.add_argument('--val_only', action='store_true',
                        help='Only run validation (requires --resume)')
    args = parser.parse_args()
    train_pdm_policy(config_path=args.config_path, resume_path=args.resume, val_only=args.val_only)
