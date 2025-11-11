#!/usr/bin/env python3
"""
测试脚本：加载最优模型权重，在完整测试集上评估性能并统计指标
"""
import os
import sys
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dataset.unified_carla_dataset import CARLAImageDataset
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_best_model(checkpoint_path, config, device):
    """加载最优模型权重"""
    print(f"Loading best model from: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 提取action统计信息
    action_stats = {
    'min': torch.tensor([-11.77335262298584, -59.26432800292969]),
    'max': torch.tensor([98.34003448486328, 55.585079193115234]),
    'mean': torch.tensor([10.975193977355957, 0.04004639387130737]),
    'std': torch.tensor([14.96833324432373, 3.419595956802368]),
}
    
    # 初始化模型
    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats).to(device)
    
    # 加载模型权重
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    # 打印checkpoint信息
    print(f"✓ Model loaded successfully!")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Validation Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  - Training Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    
    if 'val_metrics' in checkpoint:
        print(f"  - Validation Metrics:")
        for key, value in checkpoint['val_metrics'].items():
            if isinstance(value, (int, float)):
                print(f"    {key}: {value:.4f}")
    
    return policy


def compute_driving_metrics(predicted_trajectories, target_trajectories):
    """
    计算驾驶相关指标（来自train_carla_bev.py）
    
    Args:
        predicted_trajectories: (B, T, 2) 预测轨迹
        target_trajectories: (B, T, 2) 真实轨迹
    
    Returns:
        metrics: 评估指标字典
    """
    if isinstance(predicted_trajectories, torch.Tensor):
        predicted_trajectories = predicted_trajectories.detach().cpu().numpy()
    if isinstance(target_trajectories, torch.Tensor):
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


def compute_basic_metrics(predictions, ground_truths):
    """
    计算基础评估指标
    
    Args:
        predictions: 预测轨迹列表 [(T, 2), ...]
        ground_truths: 真实轨迹列表 [(T, 2), ...]
    
    Returns:
        metrics: 评估指标字典
    """
    all_mse = []
    all_mae = []
    all_l2_errors = []
    all_ade = []  # Average Displacement Error
    all_fde = []  # Final Displacement Error
    
    for pred, gt in zip(predictions, ground_truths):
        # MSE
        mse = np.mean((pred - gt) ** 2)
        all_mse.append(mse)
        
        # MAE
        mae = np.mean(np.abs(pred - gt))
        all_mae.append(mae)
        
        # L2 distance per timestep
        l2_dist = np.linalg.norm(pred - gt, axis=1)
        all_l2_errors.append(l2_dist)
        
        # ADE: Average Displacement Error (平均位移误差)
        # 所有时间步的平均欧氏距离
        ade = np.mean(l2_dist)
        all_ade.append(ade)
        
        # FDE: Final Displacement Error (最终位移误差)
        # 最后一个时间步的欧氏距离
        fde = l2_dist[-1]
        all_fde.append(fde)
    
    # 汇总所有样本的L2误差
    all_l2_errors = np.concatenate(all_l2_errors)
    
    metrics = {
        'MSE': np.mean(all_mse),
        'MSE_std': np.std(all_mse),
        'MAE': np.mean(all_mae),
        'MAE_std': np.std(all_mae),
        'L2_mean': np.mean(all_l2_errors),
        'L2_std': np.std(all_l2_errors),
        'L2_median': np.median(all_l2_errors),
        'L2_max': np.max(all_l2_errors),
        'L2_min': np.min(all_l2_errors),
        'ADE': np.mean(all_ade),
        'ADE_std': np.std(all_ade),
        'FDE': np.mean(all_fde),
        'FDE_std': np.std(all_fde),
    }
    
    return metrics


def test_model(policy, test_loader, config, device='cuda'):
    """
    在完整测试集上测试模型
    
    Args:
        policy: 训练好的模型
        test_loader: 测试数据加载器
        config: 配置字典
        device: 设备
    """
    print(f"\n{'='*60}")
    print(f"Testing model on full test dataset")
    print(f"Total samples: {len(test_loader.dataset)}")
    print(f"{'='*60}\n")
    
    policy.eval()
    
    # 用于存储所有预测和真实值
    all_predictions = []
    all_ground_truths = []
    
    # 用于累积驾驶指标
    driving_metrics_accumulator = defaultdict(list)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # 将数据移到设备上
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # 提取观测数据
            obs_horizon = config.get('obs_horizon', 2)
            obs_dict = {
                'lidar_token': batch['lidar_token'][:, :obs_horizon],  # (B, obs_horizon, seq_len, 512)
                'lidar_token_global': batch['lidar_token_global'][:, :obs_horizon],  # (B, obs_horizon, 1, 512)
                # 'lidar_bev': batch['lidar_bev'][:, :policy.n_obs_steps],  # (B, obs_horizon, 3, 448, 448)
                'agent_pos': batch['agent_pos'][:, :obs_horizon],  # (B, obs_horizon, 2) - 观测步的agent_pos
                'speed': batch['speed'][:, :obs_horizon],
                'target_point': batch['target_point'][:, :obs_horizon],
                'next_command': batch['next_command'][:, :obs_horizon],  
                'vqa': batch['vqa'] if 'vqa' in batch else None,
            }
            
            try:
                # 预测动作/轨迹
                result = policy.predict_action(obs_dict)
                predicted_actions = torch.from_numpy(result['action']).to(device)  # (B, T, 2)
                target_actions = batch['agent_pos'][:, :predicted_actions.shape[1]]  # (B, T, 2)
                
                # 计算驾驶指标（batch级别）
                batch_driving_metrics = compute_driving_metrics(predicted_actions, target_actions)
                for key, value in batch_driving_metrics.items():
                    driving_metrics_accumulator[key].append(value)
                
                # 保存预测和真实值用于后续计算基础指标
                for i in range(predicted_actions.shape[0]):
                    pred = predicted_actions[i].cpu().numpy()  # (T, 2)
                    gt = target_actions[i].cpu().numpy()  # (T, 2)
                    all_predictions.append(pred)
                    all_ground_truths.append(gt)
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # 计算所有指标
    print(f"\n{'='*60}")
    print("Computing overall metrics...")
    print(f"{'='*60}\n")
    
    # 1. 基础指标
    basic_metrics = compute_basic_metrics(all_predictions, all_ground_truths)
    
    # 2. 驾驶指标（平均）
    driving_metrics = {}
    for key, values in driving_metrics_accumulator.items():
        if values:
            driving_metrics[key] = np.mean(values)
    
    # 合并所有指标
    all_metrics = {**basic_metrics, **driving_metrics}
    
    return all_metrics, all_predictions, all_ground_truths


def print_and_save_metrics(metrics, num_samples, save_path):
    """打印并保存指标到文件"""
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60 + "\n")
    
    print(f"Number of samples: {num_samples}\n")
    
    print("Basic Metrics:")
    print(f"  MSE: {metrics['MSE']:.6f} ± {metrics['MSE_std']:.6f}")
    print(f"  MAE: {metrics['MAE']:.6f} ± {metrics['MAE_std']:.6f}")
    print(f"  ADE: {metrics['ADE']:.6f} ± {metrics['ADE_std']:.6f}  (Average Displacement Error)")
    print(f"  FDE: {metrics['FDE']:.6f} ± {metrics['FDE_std']:.6f}  (Final Displacement Error)")
    print(f"  L2 Distance (mean): {metrics['L2_mean']:.6f} ± {metrics['L2_std']:.6f}")
    print(f"  L2 Distance (median): {metrics['L2_median']:.6f}")
    print(f"  L2 Distance (max): {metrics['L2_max']:.6f}")
    print(f"  L2 Distance (min): {metrics['L2_min']:.6f}")
    
    print("\nDriving Metrics (from train_carla_bev.py):")
    print(f"  Trajectory Error (mean): {metrics.get('trajectory_error_mean', 'N/A'):.6f}")
    if 'trajectory_error_step_1' in metrics:
        print(f"  Trajectory Error (step 1): {metrics['trajectory_error_step_1']:.6f}")
    if 'trajectory_error_step_2' in metrics:
        print(f"  Trajectory Error (step 2): {metrics['trajectory_error_step_2']:.6f}")
    if 'trajectory_error_step_3' in metrics:
        print(f"  Trajectory Error (step 3): {metrics['trajectory_error_step_3']:.6f}")
    if 'trajectory_error_step_4' in metrics:
        print(f"  Trajectory Error (step 4): {metrics['trajectory_error_step_4']:.6f}")
    print(f"  X Coordinate MAE: {metrics.get('x_coordinate_mae', 'N/A'):.6f}")
    print(f"  Y Coordinate MAE: {metrics.get('y_coordinate_mae', 'N/A'):.6f}")
    print(f"  Trajectory Smoothness Error: {metrics.get('trajectory_smoothness_error', 'N/A'):.6f}")
    
    # 保存到文件
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Test Results Summary\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Number of samples: {num_samples}\n\n")
        
        f.write("Basic Metrics:\n")
        f.write(f"  MSE: {metrics['MSE']:.6f} ± {metrics['MSE_std']:.6f}\n")
        f.write(f"  MAE: {metrics['MAE']:.6f} ± {metrics['MAE_std']:.6f}\n")
        f.write(f"  ADE: {metrics['ADE']:.6f} ± {metrics['ADE_std']:.6f}  (Average Displacement Error)\n")
        f.write(f"  FDE: {metrics['FDE']:.6f} ± {metrics['FDE_std']:.6f}  (Final Displacement Error)\n")
        f.write(f"  L2 Distance (mean): {metrics['L2_mean']:.6f} ± {metrics['L2_std']:.6f}\n")
        f.write(f"  L2 Distance (median): {metrics['L2_median']:.6f}\n")
        f.write(f"  L2 Distance (max): {metrics['L2_max']:.6f}\n")
        f.write(f"  L2 Distance (min): {metrics['L2_min']:.6f}\n\n")
        
        f.write("Driving Metrics (from train_carla_bev.py):\n")
        f.write(f"  Trajectory Error (mean): {metrics.get('trajectory_error_mean', 'N/A'):.6f}\n")
        if 'trajectory_error_step_1' in metrics:
            f.write(f"  Trajectory Error (step 1): {metrics['trajectory_error_step_1']:.6f}\n")
        if 'trajectory_error_step_2' in metrics:
            f.write(f"  Trajectory Error (step 2): {metrics['trajectory_error_step_2']:.6f}\n")
        if 'trajectory_error_step_3' in metrics:
            f.write(f"  Trajectory Error (step 3): {metrics['trajectory_error_step_3']:.6f}\n")
        if 'trajectory_error_step_4' in metrics:
            f.write(f"  Trajectory Error (step 4): {metrics['trajectory_error_step_4']:.6f}\n")
        f.write(f"  X Coordinate MAE: {metrics.get('x_coordinate_mae', 'N/A'):.6f}\n")
        f.write(f"  Y Coordinate MAE: {metrics.get('y_coordinate_mae', 'N/A'):.6f}\n")
        f.write(f"  Trajectory Smoothness Error: {metrics.get('trajectory_smoothness_error', 'N/A'):.6f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("All Metrics (key-value pairs):\n")
        f.write("="*60 + "\n")
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"\n✓ Metrics saved to: {save_path}")


def main(args):
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # 加载配置
    config_path = args.config_path
    # config_path = os.path.join(project_root, "config/carla.yaml")
    config = load_config(config_path)
    
    # 加载最优模型
    checkpoint_base_path = config.get('training', {}).get('checkpoint_dir', "/root/z_projects/code/MoT-DP/checkpoints/pdm_linearnorm_2obs_8pred")
    checkpoint_path = os.path.join(checkpoint_base_path, "carla_policy_best.pt")
    #checkpoint_path = "/home/wang/Project/MoT-DP/checkpoints/carla_dit/carla_policy_best.pt"
    policy = load_best_model(checkpoint_path, config, device)
    
    # 加载测试数据集
    print("\nLoading test dataset...")
    dataset_path = config.get('training', {}).get('dataset_path')
    val_dataset_path = os.path.join(dataset_path, 'val')
    image_data_root = config.get('training', {}).get('image_data_root')
    
    test_dataset = CARLAImageDataset(
        dataset_path=val_dataset_path, 
        image_data_root=image_data_root
    )
    
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples\n")
    
    # 创建数据加载器
    batch_size = config.get('dataloader', {}).get('batch_size', 16)
    num_workers = config.get('dataloader', {}).get('num_workers', 4)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"DataLoader created with batch_size={batch_size}, num_workers={num_workers}\n")
    
    # 运行测试
    metrics, predictions, ground_truths = test_model(
        policy=policy,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # 打印并保存结果
    save_dir = os.path.join(project_root, 'image', 'test_results')
    metrics_path = os.path.join(save_dir, 'test_metrics.txt')
    
    print_and_save_metrics(metrics, len(predictions), metrics_path)
    
    print(f"\n{'='*60}")
    print("Testing completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test DiffusionDiTCarlaPolicy on full test dataset")
    parser.add_argument('--config_path', type=str, default=os.path.join(project_root, "config/carla.yaml"), help='Path to the configuration file')
    args = parser.parse_args()
    main(args)