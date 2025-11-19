#!/usr/bin/env python3
import os
import sys
import torch
import yaml
import wandb
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import time
import psutil
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from dataset.unified_carla_dataset import CARLAImageDataset
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy
import yaml

def create_carla_config(config_path=None):
    if config_path is None:
        config_path = "/root/z_projects/code/MoT-DP/config/carla.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def safe_wandb_log(data, use_wandb=True):
    """安全地记录wandb日志，处理断网情况"""
    if use_wandb:
        try:
            wandb.log(data)
        except Exception as e:
            # 断网或其他错误时，只打印警告但不中断训练
            if "ConnectionError" in str(e) or "timeout" in str(e).lower():
                print(f"⚠ WandB connection lost, continuing training without logging...")
            # 静默处理，不打印过多错误信息


def compute_driving_metrics(predicted_trajectories, target_trajectories):
    predicted_trajectories = predicted_trajectories.detach().cpu().numpy()
    target_trajectories = target_trajectories.detach().cpu().numpy()

    # L2距离误差
    l2_errors = np.linalg.norm(predicted_trajectories - target_trajectories, axis=-1)  # (B, T)

    # 不同时间步的L2误差
    metrics = {}
    time_steps = [1, 3, 5, 7]  
    time_labels = ['step_1', 'step_2', 'step_3', 'step_4']
    
    for i, (step, label) in enumerate(zip(time_steps, time_labels)):
        if step < l2_errors.shape[1]:
            metrics[f'trajectory_error_{label}'] = np.mean(l2_errors[:, step])
    
    # 平均轨迹误差 
    metrics['trajectory_error_mean'] = np.mean(l2_errors)
    
    # ADE (Average Displacement Error): 所有时间步的平均L2距离
    metrics['ADE'] = np.mean(l2_errors)
    
    # FDE (Final Displacement Error): 最后一个时间步的平均L2距离
    metrics['FDE'] = np.mean(l2_errors[:, -1])
    
    # x和y坐标的误差
    x_error = np.mean(np.abs(predicted_trajectories[:, :, 0] - target_trajectories[:, :, 0]))
    y_error = np.mean(np.abs(predicted_trajectories[:, :, 1] - target_trajectories[:, :, 1]))
    
    metrics['x_coordinate_mae'] = x_error
    metrics['y_coordinate_mae'] = y_error
    
    # 轨迹变化平滑性评估
    pred_diff = np.diff(predicted_trajectories, axis=1)
    target_diff = np.diff(target_trajectories, axis=1)
    smoothness_error = np.mean(np.linalg.norm(pred_diff - target_diff, axis=-1))
    metrics['trajectory_smoothness_error'] = smoothness_error
    
    return metrics

def validate_model(policy, val_loader, device):
    policy.eval()
    val_metrics = defaultdict(list)
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for batch_idx, batch in enumerate(pbar):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            loss = policy.compute_loss(batch)
            val_metrics['loss'].append(loss.item())
            
            obs_dict = {
                'lidar_token': batch['lidar_token'][:, :policy.n_obs_steps],  # (B, obs_horizon, seq_len, 512)
                'lidar_token_global': batch['lidar_token_global'][:, :policy.n_obs_steps],  # (B, obs_horizon, 1, 512)
                # 'lidar_bev': batch['lidar_bev'][:, :policy.n_obs_steps],  # (B, obs_horizon, 3, 448, 448)
                'agent_pos': batch['agent_pos'][:, :policy.n_obs_steps],  # (B, obs_horizon, 2) - 观测步的agent_pos
                'speed': batch['speed'][:, :policy.n_obs_steps],
                'target_point': batch['target_point'][:, :policy.n_obs_steps],
                'next_command': batch['next_command'][:, :policy.n_obs_steps],  
            }
            
            try:
                result = policy.predict_action(obs_dict)
                predicted_actions = torch.from_numpy(result['action']).to(device)
                target_actions = batch['agent_pos'][:, :predicted_actions.shape[1]]
                
                if batch_idx == 0:
                    print(f"\n=== Debug Info ===")
                    print(f"Predicted actions shape: {predicted_actions.shape}")
                    print(f"Predicted actions range: [{predicted_actions.min():.4f}, {predicted_actions.max():.4f}]")
                    print(f"Predicted actions mean: {predicted_actions.mean():.4f}, std: {predicted_actions.std():.4f}")
                    print(f"Target actions shape: {target_actions.shape}")
                    print(f"Target actions range: [{target_actions.min():.4f}, {target_actions.max():.4f}]")
                    print(f"Target actions mean: {target_actions.mean():.4f}, std: {target_actions.std():.4f}")
                    print(f"Sample predicted: {predicted_actions[0, :2]}")
                    print(f"Sample target: {target_actions[0, :2]}")
                    print(f"==================\n")
                
                driving_metrics = compute_driving_metrics(predicted_actions, target_actions)
                for key, value in driving_metrics.items():
                    val_metrics[key].append(value)
                    
                # 更新验证进度条信息
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
            except Exception as e:
                print(f"Warning: Error in action prediction during validation: {e}")
                continue
    
    pbar.close()  # 关闭验证进度条
    
    averaged_metrics = {}
    for key, values in val_metrics.items():
        if values:  
            averaged_metrics[f'val_{key}'] = np.mean(values)
        
    return averaged_metrics

def train_carla_policy(config_path, device_str, use_vlm_features=True):
    print("Initializing CARLA driving policy training...")
    config = create_carla_config(config_path=config_path)
    
    # 支持wandb离线模式和断网恢复
    wandb_mode = os.environ.get('WANDB_MODE', 'offline')  # 可通过环境变量设置: export WANDB_MODE=offline
    use_wandb = config.get('logging', {}).get('use_wandb', True)
    
    if use_wandb:
        try:
            # wandb.login(key="bddb5a05a0820c1157702750c9f0ce60bcac2bba", anonymous="must")
            wandb.init(
                project=config.get('logging', {}).get('wandb_project', "carla-diffusion-policy"),
                name=config.get('logging', {}).get('run_name', "carla_dit_full_validation"),
                mode=wandb_mode,  # online, offline, or disabled
                resume='allow',  # 允许断点续传
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
            print(f"✓ WandB initialized in {wandb_mode} mode")
        except Exception as e:
            print(f"⚠ WandB initialization failed: {e}")
            print("  Continuing training without WandB logging...")
            use_wandb = False
    else:
        print("⚠ WandB disabled in config")
        use_wandb = False
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # dataset
    dataset_path_root = config.get('training', {}).get('dataset_path')
    train_dataset_path = os.path.join(dataset_path_root, 'train')
    val_dataset_path = os.path.join(dataset_path_root, 'val')
    image_data_root = config.get('training', {}).get('image_data_root')
    dataset_args = config.get('training', {}).get('dataset_args', {})
    train_dataset = CARLAImageDataset(dataset_path=train_dataset_path, image_data_root=image_data_root, **dataset_args)
    val_dataset = CARLAImageDataset(dataset_path=val_dataset_path, image_data_root=image_data_root, **dataset_args)

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    '''
    TODO:  how to load action stats from config file
    '''

    action_stats = {
    'min': torch.tensor([-11.77335262298584, -59.26432800292969]),
    'max': torch.tensor([98.34003448486328, 55.585079193115234]),
    'mean': torch.tensor([9.755727767944336, 0.03559679538011551]),
    'std': torch.tensor([14.527670860290527, 3.224050521850586]),
    } 
    
    batch_size = config.get('dataloader', {}).get('batch_size', 32)
    num_workers = config.get('dataloader', {}).get('num_workers', 4)
    persistent_workers = config.get('dataloader', {}).get('persistent_workers', True)
    prefetch_factor = config.get('dataloader', {}).get('prefetch_factor', 2)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    print("Initializing policy model...")
    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats, device=device, use_vlm_features=use_vlm_features).to(device)  
    print(f"Policy action steps (n_action_steps): {policy.n_action_steps}")
    
    lr = config.get('optimizer', {}).get('lr', 5e-5)
    weight_decay = config.get('optimizer', {}).get('weight_decay', 1e-5)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    # 设置 checkpoint 目录
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', "/home/wang/Project/MoT-DP/checkpoints/carla_dit")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"✓ Checkpoint directory: {checkpoint_dir}")
    
    num_epochs = config.get('training', {}).get('num_epochs', 50)
    best_val_loss = float('inf')
    val_loss = None  # 初始化验证损失
    val_metrics = {}  # 初始化验证指标
    
    for epoch in range(num_epochs):
        policy.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for batch_idx, batch in enumerate(pbar):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            
            loss = policy.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            # 更新进度条信息，显示当前loss
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{np.mean(train_losses):.4f}'
            })
            
            if batch_idx % 10 == 0:
                step = epoch * len(train_loader) + batch_idx
                safe_wandb_log({
                    "train/loss_step": loss.item(),
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/batch_idx": batch_idx
                }, use_wandb)
        
        pbar.close()  # 关闭当前epoch的进度条
        
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{num_epochs} - Average training loss: {avg_train_loss:.4f}")
        torch.save({
                    'model_state_dict': policy.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_loss': avg_train_loss,
                    'val_metrics': val_metrics
                    }, os.path.join(checkpoint_dir, "carla_policy.pt"))
        
        safe_wandb_log({
            "train/loss_epoch": avg_train_loss,
            "train/epoch": epoch,
            "train/samples_processed": (epoch + 1) * len(train_dataset)
        }, use_wandb)

        validation_freq = config.get('training', {}).get('validation_freq', 1)
        if (epoch + 1) % validation_freq == 0:
            print(f"Validating (Epoch {epoch+1}/{num_epochs})...")
            val_metrics = validate_model(policy, val_loader, device)
        
            log_dict = {
                "epoch": epoch,
                "train/loss": avg_train_loss,
            }
        
            if 'val_loss' in val_metrics:
                log_dict["val/loss"] = val_metrics['val_loss']
        
            trajectory_metrics = {}
            for key, value in val_metrics.items():
                if 'trajectory_error' in key:
                    new_key = f"val/trajectory/{key.replace('val_', '')}"
                    trajectory_metrics[new_key] = value
                elif 'coordinate' in key:
                    new_key = f"val/coordinate/{key.replace('val_', '')}"
                    trajectory_metrics[new_key] = value
                elif 'smoothness' in key:
                    new_key = f"val/smoothness/{key.replace('val_', '')}"
                    trajectory_metrics[new_key] = value
                else:
        
                    new_key = f"val/{key.replace('val_', '')}"
                    trajectory_metrics[new_key] = value
        
            log_dict.update(trajectory_metrics)
            safe_wandb_log(log_dict, use_wandb)
        
            print(f"Validation metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
        
            val_loss = val_metrics.get('val_loss', float('inf'))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
                torch.save({
                    'model_state_dict': policy.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_loss': avg_train_loss,
                    'val_metrics': val_metrics
                    }, os.path.join(checkpoint_dir, "carla_policy_best.pt"))
            
                print(f"✓ New best model saved with val_loss: {val_loss:.4f}")
            
                safe_wandb_log({
                    "best_model/epoch": epoch,
                    "best_model/val_loss": val_loss,
                    "best_model/train_loss": avg_train_loss
                }, use_wandb)
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    safe_wandb_log({
        "training/completed": True,
        "training/total_epochs": num_epochs,
        "training/best_val_loss": best_val_loss,
        "training/final_train_loss": avg_train_loss
    }, use_wandb)
    
    if use_wandb:
        wandb.finish()
        print("✓ WandB session finished")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CARLA Driving Policy with Diffusion DiT")
    parser.add_argument('--config_path', type=str, default="/home/wang/Project/MoT-DP/config/carla.yaml", help='Path to the configuration YAML file')
    parser.add_argument('--device', type=str, default="cuda:1", help='Device to use for training (e.g., "cuda:0", "cpu")')
    parser.add_argument('--use_vlm_features', action='store_true', help='Whether to use VLM features during training')
    args = parser.parse_args()
    train_carla_policy(config_path=args.config_path, device_str=args.device, use_vlm_features=args.use_vlm_features)