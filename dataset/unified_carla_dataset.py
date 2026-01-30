
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import io
import sys
import pickle
import glob
from tqdm import tqdm  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import textwrap


class CARLAImageDataset(torch.utils.data.Dataset):
    
    
    def __init__(self,
                 dataset_path: str,
                 image_data_root: str,
                 mode: str = 'train'        # train or val
                 ):

        self.image_data_root = image_data_root
        self.dataset_path = dataset_path
        self.mode = mode

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 928)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.lidar_bev_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Processed data directory not found: {dataset_path}")

        # Load pkl files - support both direct directory and train/val subdirectories
        train_files = glob.glob(os.path.join(dataset_path, "train", "*.pkl"))
        val_files = glob.glob(os.path.join(dataset_path, "val", "*.pkl"))
        direct_files = glob.glob(os.path.join(dataset_path, "*.pkl"))
        
        if train_files or val_files:
            self.sample_files = sorted(train_files + val_files)
            print(f"Found {len(self.sample_files)} preprocessed samples in '{dataset_path}' "
                  f"({len(train_files)} train, {len(val_files)} val).")
        elif direct_files:
            self.sample_files = sorted(direct_files)
            print(f"Found {len(self.sample_files)} preprocessed samples in '{dataset_path}'.")
        else:
            raise FileNotFoundError(f"No pkl files found in {dataset_path} or its train/val subdirectories.")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        # Load the pickle file
        sample_path = self.sample_files[idx]
        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)

        # load image data for visualization 
        if self.mode == 'val':
            image_paths = sample.get('rgb_hist_jpg', [])
            images_tensor = self.load_image(image_paths, sample_path)

        # --- Load Transfuser Features (single frame, no temporal) ---
        # Following DiffusionDriveV2: only use bev_feature and bev_feature_upsample
        # bev_feature: (1512, 8, 8), bev_feature_upsample: (64, 64, 64)
        transfuser_bev_feature = None
        transfuser_bev_feature_upsample = None
        
        if 'transfuser_bev_feature' in sample:
            bev_feature_path = os.path.join(self.image_data_root, sample['transfuser_bev_feature'])
            transfuser_bev_feature = torch.load(bev_feature_path, weights_only=True).squeeze(0)  # Remove batch dim
        
        if 'transfuser_bev_feature_upsample' in sample:
            bev_feature_upsample_path = os.path.join(self.image_data_root, sample['transfuser_bev_feature_upsample'])
            transfuser_bev_feature_upsample = torch.load(bev_feature_upsample_path, weights_only=True).squeeze(0)
        
        # # Load VQA feature from pt file
        # vqa_path = sample.get('vqa', None)
        # vqa_feature = {}
        # full_vqa_path = os.path.join(self.image_data_root, vqa_path)
        # vqa_feature = torch.load(full_vqa_path, weights_only=True)
  
        
        # Convert sample data
        final_sample = dict()
        for key, value in sample.items():
            if key == 'rgb_hist_jpg' and self.mode == 'val':
                final_sample['rgb_hist_jpg'] = image_paths  
                final_sample['image'] = images_tensor
            elif key == 'speed_hist':
                speed_data = sample['speed_hist']
                final_sample['speed'] = torch.from_numpy(speed_data).float()
            elif key == 'ego_waypoints':
                ego_waypoints = torch.from_numpy(sample['ego_waypoints'][1:]).float()
                final_sample['agent_pos'] = ego_waypoints
            elif key == 'vqa':
                # Skip VQA field - we no longer use it
                continue
            elif key == 'route':
                # Load route waypoints (expected shape: (20, 2))
                route_data = torch.from_numpy(value).float()
                final_sample['route'] = route_data
            elif key.startswith('transfuser_'):
                # Skip transfuser paths, we already loaded them as tensors
                continue
            elif value is None:
                # Skip None values to avoid DataLoader collate errors
                continue
            elif isinstance(value, np.ndarray):
                final_sample[key] = torch.from_numpy(value).float()
            else:
                final_sample[key] = value

        # Add transfuser features to final_sample
        # Following DiffusionDriveV2: only use bev_feature and bev_feature_upsample
        if transfuser_bev_feature is not None:
            final_sample['transfuser_bev_feature'] = transfuser_bev_feature
        if transfuser_bev_feature_upsample is not None:
            final_sample['transfuser_bev_feature_upsample'] = transfuser_bev_feature_upsample

        # Build ego_status: concatenate historical low-dimensional states
        # Order: speed_hist, theta_hist, command_hist, waypoints_hist
        ego_status_components = []
        
        # 1. speed_hist
        speed_data = final_sample['speed']
        ego_status_components.append(speed_data.unsqueeze(-1))  # (obs_horizon, 1)
        
        # 2. theta_hist
        theta_data = final_sample['theta_hist']
        ego_status_components.append(theta_data.unsqueeze(-1))  # (obs_horizon, 1)
        
        # 5. command_hist (one-hot, shape: (obs_horizon, 6))
        command_data = final_sample['command_hist']
        ego_status_components.append(command_data)  # (obs_horizon, 6)

        # 6. target point
        target_point_data = final_sample['target_point_hist']
        ego_status_components.append(target_point_data)  # (obs_horizon, 2)
        
        # 7. target point next
        target_point_next_data = final_sample['target_point_next_hist']
        ego_status_components.append(target_point_next_data)  # (obs_horizon, 2)
        
        # 8. waypoints_hist (shape: (obs_horizon, 2))
        waypoints_data = final_sample['waypoints_hist']
        ego_status_components.append(waypoints_data)  # (obs_horizon, 2)
        
        # Concatenate all components along the feature dimension
        final_sample['ego_status'] = torch.cat(ego_status_components, dim=-1)  # (obs_horizon, feature_dim)


        return final_sample



    def load_image(self, image_paths, sample_path):
        images = []
        for img_path in image_paths:
            full_img_path = os.path.join(self.image_data_root, img_path)
            try:
                img = Image.open(full_img_path)
                img_tensor = self.image_transform(img)
                images.append(img_tensor)
                img.close()  # Close image object to prevent file handle leak
            except Exception as e:
                print(f"Error loading image {full_img_path}: {e}")
                images.append(torch.zeros(3, 256, 928))
        
        if len(images) > 0:
            images_tensor = torch.stack(images)
        else:
            images_tensor = torch.zeros(2, 3, 256, 928)  # 默认obs_horizon=2
        
        images.clear()  # Clear list to release references

        return images_tensor

    def load_lidar_bev(self, bev_paths, sample_path):
        images = []
        for bev_path in bev_paths:
            full_bev_path = os.path.join(self.image_data_root, bev_path)
            bev_image = Image.open(full_bev_path)
            bev_tensor = self.lidar_bev_transform(bev_image)
            images.append(bev_tensor)
            bev_image.close()  # Close image object to prevent file handle leak

        images_tensor = torch.stack(images)
        images.clear()  # Clear list to release references

        return images_tensor

# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_trajectory(sample, obs_horizon, rand_idx, save_dir='/home/wang/Project/MoT-DP/image'):

    agent_pos = sample.get('agent_pos')
    waypoints_hist = sample.get('waypoints_hist')
    target_point = sample.get('target_point')
    target_point_hist = sample.get('target_point_hist')
    anchor = sample.get('anchor')
    route = sample.get('route')
    
    if isinstance(agent_pos, torch.Tensor):
        agent_pos = agent_pos.float().numpy()
    if isinstance(waypoints_hist, torch.Tensor):
        waypoints_hist = waypoints_hist.float().numpy()
    if isinstance(target_point, torch.Tensor):
        target_point = target_point.float().numpy()
    if isinstance(target_point_hist, torch.Tensor):
        target_point_hist = target_point_hist.float().numpy()
    if isinstance(anchor, torch.Tensor):
        # Handle bfloat16 by converting to float32 first
        anchor = anchor.float().numpy()
        # Remove extra dimension if shape is (1, N, 2)
        if anchor.ndim == 3 and anchor.shape[0] == 1:
            anchor = anchor[0]
    if isinstance(route, torch.Tensor):
        route = route.float().numpy()
    
    plt.figure(figsize=(12, 12))
    
    # ========== Plot Historical Waypoints ==========
    if waypoints_hist is not None and len(waypoints_hist) > 0:
        # waypoints_hist shape: (obs_horizon, 2)
        # Plot line connecting historical points
        plt.plot(waypoints_hist[:, 1], waypoints_hist[:, 0], 'b-', 
                linewidth=1.5, alpha=0.5, label='History trajectory', zorder=2)
        
        # Plot each historical waypoint as discrete point
        for i, waypoint in enumerate(waypoints_hist[:-1]):  # Exclude last (current frame)
            plt.plot(waypoint[1], waypoint[0], 'bo', markersize=8, zorder=3)
    
    # ========== Plot Historical Target Points ==========
    if target_point_hist is not None and len(target_point_hist) > 0:
        # target_point_hist shape: (obs_horizon, 2)
        # Plot each historical target point with different colors based on time
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(target_point_hist)))
        for i, target_pt in enumerate(target_point_hist):
            plt.plot(target_pt[1], target_pt[0], 'D', color=colors[i], 
                    markersize=6, alpha=0.7, zorder=3)
            # Add text label for frame index
            plt.text(target_pt[1], target_pt[0] + 1.5, f't-{len(target_point_hist)-1-i}', 
                    fontsize=8, ha='center', alpha=0.6)
    
    # ========== Plot Current Position (Origin) ==========
    plt.plot(0, 0, 'ko', markersize=15, label='Current position (t=0)', 
            markeredgecolor='yellow', markeredgewidth=2, zorder=5)
    
    # ========== Plot Future Waypoints (GT trajectory - agent_pos) ==========
    if agent_pos is not None and len(agent_pos) > 0:
        # Plot line connecting future points (RED for GT)
        future_waypoints = np.vstack([[[0, 0]], agent_pos])
        plt.plot(future_waypoints[:, 1], future_waypoints[:, 0], 'r-', 
                linewidth=2.5, alpha=0.8, label='GT trajectory (agent_pos)', zorder=4)
        
        # Plot each future waypoint as discrete point
        for i, waypoint in enumerate(agent_pos, 1):
            plt.plot(waypoint[1], waypoint[0], 'ro', markersize=10, 
                    markeredgecolor='darkred', markeredgewidth=1.5, zorder=5)
    
    # ========== Plot Anchor Waypoints (predicted trajectory) ==========
    if anchor is not None and len(anchor) > 0:
        # Plot line connecting anchor points (CYAN for anchor/prediction)
        # Ensure anchor has shape (N, 2)
        if anchor.ndim == 1:
            anchor = anchor.reshape(-1, 2)
        anchor_waypoints = np.vstack([[0, 0], anchor])
        plt.plot(anchor_waypoints[:, 1], anchor_waypoints[:, 0], 'c-', 
                linewidth=2.5, alpha=0.7, label='Anchor trajectory (pred_traj)', zorder=3)
        
        # Plot each anchor waypoint as discrete point
        for i, waypoint in enumerate(anchor, 1):
            plt.plot(waypoint[1], waypoint[0], 'c^', markersize=8, 
                    markeredgecolor='darkcyan', markeredgewidth=1.5, zorder=4)
    
    # ========== Plot Route Waypoints (planned route) ==========
    if route is not None and len(route) > 0:
        # Plot line connecting route points (MAGENTA for route)
        # route shape: (20, 2) - [x, y] in ego frame
        plt.plot(route[:, 1], route[:, 0], 'm--', 
                linewidth=2.0, alpha=0.6, label='Route waypoints', zorder=2)
        
        # Plot discrete route points
        for i, waypoint in enumerate(route):
            plt.plot(waypoint[1], waypoint[0], 'ms', markersize=6, 
                    markeredgecolor='darkmagenta', markeredgewidth=1.0, alpha=0.6, zorder=3)
    
    # ========== Plot Current Target Point ==========
    if target_point is not None:
        if target_point.ndim == 2:
            target_point = target_point[0]
        plt.plot(target_point[1], target_point[0], 'g*', markersize=30, 
                label='Current target point (t=0)', markeredgecolor='darkgreen', markeredgewidth=2, zorder=6)
    
    # ========== Formatting ==========
    plt.xlabel('Y (ego frame, lateral / m)', fontsize=12, fontweight='bold')
    plt.ylabel('X (ego frame, longitudinal / m)', fontsize=12, fontweight='bold')
    plt.title(f'Sample {rand_idx}: Trajectory with History Target Points', fontsize=13, fontweight='bold')
    
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Add legend with custom formatting
    plt.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='black')
    
    # Add axis line at origin
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set axis range to be square and centered at origin
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    
    # Calculate the maximum range needed
    x_range = max(abs(xlim[0]), abs(xlim[1]))
    y_range = max(abs(ylim[0]), abs(ylim[1]))
    max_range = max(x_range, y_range, 1.0)
    
    # Set symmetric square limits around origin
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    # Skip tight_layout due to numpy/matplotlib compatibility issues
    # plt.tight_layout() is already handled by bbox_inches='tight' in savefig
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'sample_{rand_idx}_trajectory.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ 保存轨迹可视化到: {save_path}")


def visualize_observation_images(sample, obs_horizon, rand_idx, save_dir='/root/z_projects/code/MoT-DP-1/image'):

    images = sample['image']  # shape: (obs_horizon, C, H, W)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for t, img_tensor in enumerate(images):
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor * std + mean
            img_tensor = torch.clamp(img_tensor, 0, 1)
            img_arr = img_tensor.numpy()
        else:
            img_arr = img_tensor
            
        if img_arr.shape[0] == 3:
            img_vis = np.moveaxis(img_arr, 0, -1)
            img_vis = (img_vis * 255).astype(np.uint8)
        else:
            img_vis = img_arr.astype(np.uint8)
        
        plt.figure(figsize=(20, 5))
        plt.imshow(img_vis)
        plt.title(f'Random Sample {rand_idx} - Obs Image t={t}', fontsize=12)
        plt.axis('off')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)

        save_path = os.path.join(save_dir, f'sample_{rand_idx}_obs_image_t{t}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存观测图像到: {save_path}")




def print_sample_details(sample, dataset, rand_idx, obs_horizon, 
                        save_dir='/home/wang/Project/MoT-DP/image'):
    print(f"\n样本 {rand_idx} 的详细信息:")
    
    agent_pos = sample.get('agent_pos')
    if agent_pos is not None:
        obs_agent_pos = agent_pos[:obs_horizon]
        pred_agent_pos = agent_pos[obs_horizon:]
        print(f"观测位置: {obs_agent_pos}")
        print(f"预测位置: {pred_agent_pos}")

    target_point = sample.get('target_point')
    if target_point is not None:
        if target_point.ndim == 2:
            target_point = target_point[0]
        print(f"目标点 (相对): {target_point}")
        if agent_pos is not None and len(agent_pos) > obs_horizon - 1:
            last_obs = agent_pos[obs_horizon - 1]
            distance_to_target = np.linalg.norm(target_point - last_obs)
            print(f"目标点距离最后观测点的距离: {distance_to_target:.3f}")

    target_point_hist = sample.get('target_point_hist')
    if target_point_hist is not None:
        print(f"\n历史目标点 (target_point_hist):")
        print(f"  形状: {target_point_hist.shape}")
        for i, tp in enumerate(target_point_hist):
            frame_idx = len(target_point_hist) - 1 - i
            print(f"  t-{frame_idx}: {tp}")

    print(f"\nDataset length: {len(dataset)}")
    first_sample = dataset[0]
    print("Sample keys:", first_sample.keys())
    if 'image' in first_sample:
        print("Image shape:", first_sample['image'].shape)
    if 'agent_pos' in first_sample:
        print("Agent pos shape:", first_sample['agent_pos'].shape)

    print("\nKey fields:")
    for key in ['town_name', 'speed', 'command', 'next_command', 'target_point', 
                'target_point_hist', 'ego_waypoints', 'image', 'agent_pos', 'meta_action_direction', 'meta_action_speed', 'gen_vit_tokens', 'route']:
        if key in first_sample:
            value = first_sample[key]
            print(f"  {key}: shape={getattr(value, 'shape', 'N/A')}, type={type(value)}")
            if hasattr(value, '__len__') and not isinstance(value, str):
                print(f"    Length: {len(value)}")



def test_pdm():
    """Test with PDM Lite dataset."""
    import random
    
    dataset_path = '/home/wang/Dataset/pdm_lite_mini/tmp_data/val'
    obs_horizon = 4
    """
    Prints detailed information about a sample and the dataset.
    
    Args:
        sample (dict): Data sample
        dataset (CARLAImageDataset): Dataset object
        rand_idx (int): Sample index
        obs_horizon (int): Number of observation frames
        save_dir (str): Directory for saving visualizations
    """
    print("\n========== Testing PDM Lite Dataset ==========")
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        image_data_root='/home/wang/Dataset/pdm_lite_mini'
    )
    
    print(f"\n总样本数: {len(dataset)}")
    if len(dataset) == 0:
        print("数据为空，无法进行测试。")
        return

    rand_idx = random.choice(range(len(dataset)))
    rand_sample = dataset[rand_idx]
    print(f"\n随机选择的样本索引: {rand_idx}")

    print("\nSample keys:", rand_sample.keys())
    for key, value in rand_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}")

    visualize_trajectory(rand_sample, obs_horizon, rand_idx)
    print_sample_details(rand_sample, dataset, rand_idx, obs_horizon)

    
if __name__ == "__main__":
    test_pdm()
    


