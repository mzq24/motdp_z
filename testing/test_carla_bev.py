#!/usr/bin/env python3
"""
测试脚本：加载最优模型权重，从测试集抽样并可视化预测结果与ground truth
"""
import os
import sys
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random
from PIL import Image
from scipy.ndimage import zoom

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# from dataset.generate_pdm_dataset import CARLAImageDataset
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
#     action_stats = {
#     'min': torch.tensor([-21.217477798461914, -22.13955307006836]),
#     'max': torch.tensor([33.02915954589844, 26.23844337463379]),
#     'mean': torch.tensor([3.9731080532073975, -0.05837925150990486]),
#     'std': torch.tensor([4.942440032958984, 1.4319489002227783]),
# }
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


def project_waypoints_to_image(waypoints, camera_intrinsics=None, image_shape=(256, 928)):
    """
    将3D路点投影到图像平面，使用CARLA相机内外参
    
    Args:
        waypoints: (T, 2) 相对路点 [x, y] 单位：米，车辆坐标系
        camera_intrinsics: 相机内参字典（如果为None则使用CARLA默认值）
        image_shape: 图像尺寸 (H, W)
    
    Returns:
        image_points: (T, 2) 图像坐标 [u, v]
        valid_mask: (T,) 是否在图像范围内
        
    CARLA相机配置:
        传感器类型: sensor.camera.rgb
        外参 (相对于车辆中心):
            位置: x=0.80m (向前), y=0.0m (左右居中), z=1.60m (向上)
            旋转: roll=0°, pitch=0°, yaw=0°
        图像参数:
            原始尺寸: width=1600, height=900
            实际使用: width=928, height=256 (经过resize)
            视野角: fov=70°
            
    相机内参矩阵 K (基于原始尺寸 1600x900, fov=70°):
        焦距 f = width / (2 * tan(fov/2)) = 1600 / (2 * tan(35°)) ≈ 1142.154
        主点 cx = width/2 = 800.0
        主点 cy = height/2 = 450.0
        K = [[1142.154,    0.0,      800.0],
             [   0.0,   1142.154,    450.0],
             [   0.0,      0.0,        1.0]]
             
    坐标系说明:
        - 车辆坐标系: x向前(前进方向), y向左, z向上
        - 相机坐标系: x向右, y向下, z向前(光轴方向)
        - 图像坐标系: u向右, v向下, 原点在左上角
    """
    
    # CARLA相机外参（相对于车辆中心）
    camera_x = 0.80  # 相机在车辆前方0.8米
    camera_y = 0.0   # 相机在车辆中心线上
    camera_z = 1.60  # 相机离地面1.6米
    camera_pitch = 0.0  # 俯仰角（度）
    camera_roll = 0.0   # 横滚角（度）
    camera_yaw = 0.0    # 偏航角（度）
    
    # 计算相机内参（需要根据resize后的图像尺寸调整）
    if camera_intrinsics is None:
        # 原始图像: 1600x900, FOV=70°
        original_width = 1600
        original_height = 900
        fov = 70.0  # 度
        
        # 计算原始焦距
        f_original = original_width / (2.0 * np.tan(np.deg2rad(fov / 2.0)))
        
        # 根据resize比例调整内参
        scale_x = image_shape[1] / original_width  # 928 / 1600
        scale_y = image_shape[0] / original_height  # 256 / 900
        
        focal_length = f_original * scale_x  # 调整后的焦距
        cx = image_shape[1] / 2.0  # 调整后的主点x
        cy = image_shape[0] / 2.0  # 调整后的主点y
    else:
        focal_length = camera_intrinsics['f']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
    
    # 坐标变换和投影
    image_points = []
    valid_mask = []
    
    pitch_rad = np.deg2rad(camera_pitch)
    
    for wp in waypoints:
        x_vehicle, y_vehicle = wp  # 路点在车辆坐标系中的位置
        
        # 1. 将路点从车辆坐标系转换到相机坐标系
        #    路点在地面上，所以z_vehicle = 0
        #    相机在车辆上方camera_z米，前方camera_x米
        
        # 相对于相机的位置（车辆坐标系）
        x_rel = x_vehicle - camera_x  # 向前的距离
        y_rel = y_vehicle - camera_y  # 向左的距离
        z_rel = 0.0 - camera_z        # 向上的距离（路点在地面，相机在空中）
        
        # 2. 从车辆坐标系转换到相机坐标系
        #    车辆坐标系: x前, y左, z上
        #    相机坐标系: x右, y下, z前
        #    考虑相机俯仰角（如果pitch=0，相机水平朝前）
        
        # 先进行俯仰旋转（绕相机y轴）
        x_cam_before_pitch = x_rel
        z_cam_before_pitch = -z_rel  # 车辆z上 -> 相机y下（取负）
        # 应用俯仰角旋转
        z_cam = x_cam_before_pitch * np.cos(pitch_rad) - z_cam_before_pitch * np.sin(pitch_rad)
        y_cam = x_cam_before_pitch * np.sin(pitch_rad) + z_cam_before_pitch * np.cos(pitch_rad)
        
        # x轴转换: 车辆y左 -> 相机x右
        # 车辆y为正=向左，相机x为正=向右
        x_cam = y_rel
        
        # 3. 投影到图像平面
        if z_cam > 0.1:  # 只投影前方的点（深度为正）
            u = focal_length * (x_cam / z_cam) + cx
            v = focal_length * (y_cam / z_cam) + cy
            
            # 检查是否在图像范围内
            if 0 <= u < image_shape[1] and 0 <= v < image_shape[0]:
                image_points.append([u, v])
                valid_mask.append(True)
            else:
                image_points.append([u, v])
                valid_mask.append(False)
        else:
            image_points.append([0, 0])
            valid_mask.append(False)
    
    return np.array(image_points), np.array(valid_mask)


def visualize_prediction(sample, prediction, ground_truth, sample_idx, save_dir, 
                         filename=None, attention_map=None, denoising_steps=None, policy=None):
    """
    可视化预测结果，包含四个部分：
    1. 预测轨迹和真实轨迹比较
    2. 在观测图像上绘制3D投影的预测路点
    3. 在LiDAR BEV上绘制attention map热力图
    4. 展示去噪过程的关键步骤
    
    Args:
        sample: 包含所有输入数据的字典
        prediction: 预测的agent_pos轨迹 (T, 2)
        ground_truth: 真实的agent_pos轨迹 (T, 2)
        sample_idx: 样本索引
        save_dir: 保存目录
        attention_map: 注意力图 (可选)
        denoising_steps: 去噪步骤的中间结果 (可选)
        policy: 策略模型（用于获取attention）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建2x2的子图布局
    fig = plt.figure(figsize=(20, 16))
    
    # ====== 子图1: 轨迹比较 ======
    ax1 = plt.subplot(2, 2, 1)
    
    # 获取观测时间步的agent_pos
    obs_horizon = sample['agent_pos'].shape[0] - ground_truth.shape[0]
    obs_agent_pos = sample['agent_pos'][:obs_horizon].cpu().numpy()
    
    # 绘制观测轨迹
    if len(obs_agent_pos) > 0:
        ax1.plot(obs_agent_pos[:, 1], obs_agent_pos[:, 0], 'go-', 
                label='Observed', markersize=10, linewidth=2.5, alpha=0.8)
        ax1.plot(obs_agent_pos[-1, 1], obs_agent_pos[-1, 0], 'ks', 
                markersize=12, label='Reference (last obs)', 
                markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=2)
    
    # 绘制预测轨迹
    ax1.plot(prediction[:, 1], prediction[:, 0], 'b^--', 
            label='Prediction', markersize=8, linewidth=2, alpha=0.8)
    
    # 绘制ground truth轨迹
    ax1.plot(ground_truth[:, 1], ground_truth[:, 0], 'ro-', 
            label='Ground Truth', markersize=8, linewidth=2, alpha=0.8)
    
    # 绘制目标点
    if 'target_point' in sample and sample['target_point'] is not None and False:
        target_point = sample['target_point'][0].cpu().numpy()
        ax1.plot(target_point[1], target_point[0], 'g*', 
                markersize=20, label='Target Point', 
                markeredgecolor='black', markeredgewidth=1.5)
    
    # 计算误差
    mse = np.mean((prediction - ground_truth) ** 2)
    mae = np.mean(np.abs(prediction - ground_truth))
    l2_dist = np.linalg.norm(prediction - ground_truth, axis=1)
    
    ax1.set_xlabel('Y (meters)', fontsize=11)
    ax1.set_ylabel('X (meters)', fontsize=11)
    ax1.set_title(f'1. Trajectory Comparison\nMSE: {mse:.4f}, MAE: {mae:.4f}', 
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # ====== 子图2: 3D投影到观测图像 ======
    ax2 = plt.subplot(2, 2, 2)
    if 'image' in sample and sample['image'] is not None:
        # 取最后一帧观测图像
        img = sample['image'][-1].cpu().numpy()  # (3, H, W)
        
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # 转换为 (H, W, 3)
        img = np.transpose(img, (1, 2, 0))
        
        ax2.imshow(img)
        
        # 投影预测的路点到图像
        image_points, valid_mask = project_waypoints_to_image(prediction, image_shape=img.shape[:2])
        
        # 绘制投影的路点（红色圆点，不显示标签）
        if np.any(valid_mask):
            valid_points = image_points[valid_mask]
            # 绘制路点为红色圆点
            ax2.scatter(valid_points[:, 0], valid_points[:, 1], 
                       c='red', s=100, alpha=0.8, edgecolors='white', 
                       linewidths=2, zorder=5, label='Predicted Waypoints')
        
        ax2.set_title('2. Predicted Waypoints on Camera View\n(3D Perspective Projection)', 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        if np.any(valid_mask):
            ax2.legend(fontsize=9, loc='upper right')
    
    # ====== 子图3: LiDAR BEV + Attention Map ======
    ax3 = plt.subplot(2, 2, 3)
    if 'lidar_bev' in sample and sample['lidar_bev'] is not None:
        # 取最后一帧LiDAR BEV
        bev = sample['lidar_bev'][-1].cpu().numpy()  # (3, H, W)
        
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        bev = bev * std + mean
        bev = np.clip(bev, 0, 1)
        
        # 转换为 (H, W, 3)
        bev = np.transpose(bev, (1, 2, 0))
        
        # 显示BEV图像
        ax3.imshow(bev, alpha=0.7)
        
        # 如果有attention map，叠加显示
        if attention_map is not None:
            # attention_map 是 (feat_h, feat_w) 的热力图，通常是 (14, 14)
            # 需要resize到BEV图像尺寸
            from scipy.ndimage import zoom
            
            # 计算缩放因子
            scale_h = bev.shape[0] / attention_map.shape[0]
            scale_w = bev.shape[1] / attention_map.shape[1]
            
            # 使用双线性插值放大attention map
            attention_resized = zoom(attention_map, (scale_h, scale_w), order=1)
            
            # 归一化到 [0, 1]
            attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
            
            im = ax3.imshow(attention_resized, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label='Attention Weight')
            
            ax3.text(0.02, 0.98, f'Attention shape: {attention_map.shape}', 
                    transform=ax3.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            # 如果没有真实的attention map，创建一个基于路点距离的模拟热力图
            H, W = bev.shape[:2]
            y_grid, x_grid = np.meshgrid(np.linspace(-20, 20, W), np.linspace(-20, 20, H))
            
            # 基于预测路点创建热力图
            heatmap = np.zeros((H, W))
            for wp in prediction:
                # 计算每个网格点到路点的距离
                dist = np.sqrt((x_grid - wp[0])**2 + (y_grid - wp[1])**2)
                heatmap += np.exp(-dist**2 / 10.0)  # 高斯核
            
            heatmap = heatmap / (heatmap.max() + 1e-8)
            im = ax3.imshow(heatmap, cmap='jet', alpha=0.4, vmin=0, vmax=1)
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label='Attention (Simulated)')
        
        ax3.set_title('3. LiDAR BEV with Spatial Attention Map', 
                     fontsize=12, fontweight='bold')
        ax3.axis('off')
    
    # ====== 子图4: 去噪过程 ======
    ax4 = plt.subplot(2, 2, 4)
    
    if denoising_steps is not None and len(denoising_steps) >= 4:
        # 显示4个关键步骤：初始噪声、两个中间状态、最终结果
        steps_to_show = [0, len(denoising_steps)//3, 2*len(denoising_steps)//3, -1]
        step_labels = ['Initial Noise', 'Denoising 33%', 'Denoising 67%', 'Final Result']
        
        for i, (step_idx, label) in enumerate(zip(steps_to_show, step_labels)):
            step_data = denoising_steps[step_idx]  # (T, 2)
            
            # 使用不同的颜色和透明度
            alpha = 0.3 + 0.7 * (i / 3)
            color = plt.cm.viridis(i / 3)
            
            ax4.plot(step_data[:, 1], step_data[:, 0], 'o-', 
                    color=color, alpha=alpha, linewidth=2, markersize=6,
                    label=label)
        
        # 添加ground truth作为参考
        ax4.plot(ground_truth[:, 1], ground_truth[:, 0], 'r*--', 
                markersize=8, linewidth=1.5, alpha=0.5, label='Ground Truth')
        
        ax4.set_xlabel('Y (meters)', fontsize=11)
        ax4.set_ylabel('X (meters)', fontsize=11)
        ax4.set_title('4. Denoising Process Visualization', 
                     fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9, loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
    else:
        # 如果没有去噪步骤数据，显示误差分布
        ax4.bar(range(len(l2_dist)), l2_dist, color='skyblue', alpha=0.7)
        ax4.axhline(y=np.mean(l2_dist), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(l2_dist):.3f}')
        ax4.set_xlabel('Time Step', fontsize=11)
        ax4.set_ylabel('L2 Error (meters)', fontsize=11)
        ax4.set_title('4. Per-Step Trajectory Error', 
                     fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加总体信息
    info_text = f'Sample {sample_idx}\n'
    info_text += f'Avg L2 Error: {np.mean(l2_dist):.4f}m | '
    info_text += f'Max L2 Error: {np.max(l2_dist):.4f}m'
    
    plt.suptitle(info_text, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存图像
    save_path = os.path.join(save_dir, f'sample_{sample_idx}_prediction.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization to: {save_path}")


def compute_metrics(predictions, ground_truths):
    """
    计算整体评估指标
    
    Args:
        predictions: 预测轨迹列表 [(T, 2), ...]
        ground_truths: 真实轨迹列表 [(T, 2), ...]
    
    Returns:
        metrics: 评估指标字典
    """
    all_mse = []
    all_mae = []
    all_l2_errors = []
    
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
    }
    
    return metrics


def test_model(policy, test_dataset, config, num_samples=10, visualize_samples=5, device='cuda', folder_name=None):
    """
    测试模型并可视化结果
    
    Args:
        policy: 训练好的模型
        test_dataset: 测试数据集
        config: 配置字典
        num_samples: 测试样本数量
        visualize_samples: 可视化的样本数量
        device: 设备
    """
    print(f"\n{'='*60}")
    print(f"Testing model on {num_samples} samples from test dataset")
    print(f"Visualizing {visualize_samples} random samples")
    print(f"{'='*60}\n")
    
    # 随机抽取测试样本
    total_samples = len(test_dataset)
    if num_samples > total_samples:
        num_samples = total_samples
        print(f"Warning: Requested {num_samples} samples but only {total_samples} available")
    
    # 随机选择样本索引
    sample_indices = random.sample(range(total_samples), num_samples)
    visualize_indices = random.sample(sample_indices, min(visualize_samples, num_samples))
    
    print(f"Selected sample indices: {sample_indices[:10]}..." if len(sample_indices) > 10 else f"Selected sample indices: {sample_indices}")
    print(f"Visualizing samples: {visualize_indices}\n")
    
    # 创建保存目录
    if folder_name is not None:
        save_dir = os.path.join(project_root, 'image', folder_name)
    else:
        save_dir = os.path.join(project_root, 'image', 'test_results')
    os.makedirs(save_dir, exist_ok=True)
    
    predictions = []
    ground_truths = []
    
    policy.eval()
    with torch.no_grad():
        for idx, sample_idx in enumerate(tqdm(sample_indices, desc="Testing")):
            # 获取样本
            sample = test_dataset[sample_idx]
            
            # 准备输入数据
            batch = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.unsqueeze(0).to(device)  # 添加batch维度
            
            # 提取观测数据
            obs_horizon = config.get('obs_horizon', 2)
            obs_dict = {
                'lidar_token': batch['lidar_token'][:, :obs_horizon],  # (B, obs_horizon, seq_len, 512)
                'lidar_token_global': batch['lidar_token_global'][:, :obs_horizon],  # (B, obs_horizon, 1, 512)
                'agent_pos': batch['agent_pos'][:, :obs_horizon],
                'speed': batch['speed'][:, :obs_horizon],
                'target_point': batch['target_point'][:, :obs_horizon],
                'next_command': batch['next_command'][:, :obs_horizon],
            }
            if 'image' in batch:
                obs_dict['image'] = batch['image'][:, :obs_horizon]
            if 'lidar_bev' in batch:
                obs_dict['lidar_bev'] = batch['lidar_bev'][:, :obs_horizon]
            if 'vqa' in batch:
                obs_dict['vqa'] = batch['vqa']

            # 预测动作/轨迹
            try:
                # 对于可视化的样本，捕获去噪步骤和attention map
                if sample_idx in visualize_indices:
                    result, denoising_steps = policy.predict_action_with_steps(obs_dict)
                    # 提取attention map
                    attention_map = policy.extract_attention_map(obs_dict)
                else:
                    result = policy.predict_action(obs_dict)
                    denoising_steps = None
                    attention_map = None
                
                predicted_actions = result['action'][0]  # 移除batch维度 (T, 2)
                print(f"Sample {sample_idx}: Predicted actions shape: {predicted_actions.shape}")
                
                # 获取ground truth
                gt_actions = batch['agent_pos'][0, :].cpu().numpy()

                predictions.append(predicted_actions)
                ground_truths.append(gt_actions)
                
                # 可视化选定的样本
                if sample_idx in visualize_indices:
                    visualize_prediction(
                        sample=sample,
                        prediction=predicted_actions,
                        ground_truth=gt_actions,
                        sample_idx=sample_idx,
                        save_dir=save_dir,
                        attention_map=attention_map,  # 使用真实的attention map
                        denoising_steps=denoising_steps,
                        policy=policy
                    )
                    
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                continue
    
    # 计算整体指标
    print(f"\n{'='*60}")
    print("Computing overall metrics...")
    print(f"{'='*60}\n")
    
    metrics = compute_metrics(predictions, ground_truths)
    
    print("Test Results:")
    print(f"  Number of samples: {len(predictions)}")
    print(f"  MSE: {metrics['MSE']:.6f} ± {metrics['MSE_std']:.6f}")
    print(f"  MAE: {metrics['MAE']:.6f} ± {metrics['MAE_std']:.6f}")
    print(f"  L2 Distance (mean): {metrics['L2_mean']:.6f} ± {metrics['L2_std']:.6f}")
    print(f"  L2 Distance (median): {metrics['L2_median']:.6f}")
    
    # 保存指标到文件
    metrics_path = os.path.join(save_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Test Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of samples: {len(predictions)}\n")
        f.write(f"MSE: {metrics['MSE']:.6f} ± {metrics['MSE_std']:.6f}\n")
        f.write(f"MAE: {metrics['MAE']:.6f} ± {metrics['MAE_std']:.6f}\n")
        f.write(f"L2 Distance (mean): {metrics['L2_mean']:.6f} ± {metrics['L2_std']:.6f}\n")
        f.write(f"L2 Distance (median): {metrics['L2_median']:.6f}\n")
        f.write(f"\nSample indices: {sample_indices}\n")
        f.write(f"Visualized indices: {visualize_indices}\n")
    
    print(f"\n✓ Metrics saved to: {metrics_path}")
    print(f"✓ Visualizations saved to: {save_dir}")
    
    return metrics, predictions, ground_truths


def main(args):
    """主函数"""
    # 使用时间戳作为随机种子，每次运行都不同
    import time
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Random seed for this run: {seed}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # 加载配置
    # config_path = os.path.join(project_root, "config/carla.yaml")
    config_path = args.config_path
    config = load_config(config_path)
    
    # 加载最优模型
    # checkpoint_path = "/home/wang/Project/MoT-DP/checkpoints/carla_policy_best.pt"
    checkpoint_path_base = config.get('training', {}).get('checkpoint_dir')
    checkpoint_path = os.path.join(checkpoint_path_base, 'carla_policy_best.pt')
    policy = load_best_model(checkpoint_path, config, device)
    
    # 加载测试数据集
    print("\nLoading test dataset...")
    dataset_path = config.get('training', {}).get('dataset_path')
    val_dataset_path = os.path.join(dataset_path, 'val')
    image_data_root = config.get('training', {}).get('image_data_root')
    test_dataset = CARLAImageDataset(
        dataset_path=val_dataset_path, 
        image_data_root=image_data_root,
        mode='val',
    )
    
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples\n")
    
    # 运行测试
    num_test_samples = 50  # 测试样本数量
    num_visualize = 50    # 可视化样本数量
    folder_name = args.get('evaluation', {}).get("folder_name", None)
    metrics, predictions, ground_truths = test_model(
        policy=policy,
        test_dataset=test_dataset,
        config=config,
        num_samples=num_test_samples,
        visualize_samples=num_visualize,
        device=device,
        folder_name=folder_name
    )
    
    print(f"\n{'='*60}")
    print("Testing completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test DiffusionDiTCarlaPolicy and visualize predictions')
    parser.add_argument('--config-path', type=str, default=os.path.join(project_root, "config/carla.yaml"), help='Path to the config file')
    args = parser.parse_args()
    main(args)
