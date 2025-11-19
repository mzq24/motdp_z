"""
Unified CARLA Dataset Loader for PDM Lite and Bench2Drive (B2D) datasets.

This module provides a single CARLAImageDataset class that can handle:
- PDM Lite dataset (without VQA and meta_action)
- Bench2Drive (B2D) dataset (with VQA and meta_action)

The dataset automatically handles both types transparently by checking for
optional fields in the loaded samples.
"""

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

bagel_path = "/root/z_projects/code/Bagel"
if bagel_path not in sys.path:
    sys.path.insert(0, bagel_path)
from modeling.bagel import SiglipVisionConfig, SiglipVisionModel
from data.data_utils import pil_img2rgb, patchify, add_special_tokens
from data.transforms import ImageTransform
from modeling.qwen2 import Qwen2Tokenizer

class CARLAImageDataset(torch.utils.data.Dataset):
    """
    Unified Dataset for preprocessed PDM Lite and Bench2Drive (B2D) data in 'frame' mode.
    
    Each pickle file is a complete training sample with image paths.
    Images are loaded dynamically in __getitem__.
    
    Supports both:
    - PDM Lite: Basic driving data with waypoints and sensors
    - Bench2Drive (B2D): Extended driving data with VQA and meta-actions
    
    Args:
        dataset_path (str): Directory containing individual frame pkl files
        image_data_root (str): Base path for images
    """
    
    def __init__(self,
                 dataset_path: str,
                 image_data_root: str,
                 mode: str = 'train',        # train or val
                 use_image: bool = False, # Whether to use vit_transform for image processing
                 vit_transform_args: dict = None,
                 ):

        self.image_data_root = image_data_root
        self.dataset_path = dataset_path
        self.mode = mode
        self.use_image = use_image
        if vit_transform_args is None:
            vit_transform_args = {'image_stride': 14, 'max_image_size': 980, 'min_image_size': 518,  'max_pixels': 2_007_040}
        self.vit_transform = ImageTransform(**vit_transform_args)
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
        image_paths = sample.get('rgb_hist_jpg', [])
        if self.use_image or self.mode == 'val':
            images_tensor, images_demo_tensor = self.load_image(image_paths, sample_path)

        # Load LiDAR BEV features
        lidar_bev_paths = sample.get('lidar_bev_hist', [])
        if self.mode == 'val':
            lidar_bev_tensor = self.load_lidar_bev(lidar_bev_paths, sample_path)
        
        lidar_tokens = []
        lidar_tokens_global = []
        for bev_path in lidar_bev_paths:
            if bev_path is None:
                continue
            
            # Infer feature paths from BEV image paths
            # Example: Town01/data/Route_0/lidar_bev/0000.png 
            #       -> Town01/data/Route_0/lidar_bev_features/0000_token.pt
            bev_dir = os.path.dirname(bev_path)
            frame_id = os.path.splitext(os.path.basename(bev_path))[0]
            feature_dir = bev_dir.replace('lidar_bev', 'lidar_bev_features')
                
            token_path = os.path.join(self.image_data_root, feature_dir, f'{frame_id}_token.pt')
            token_global_path = os.path.join(self.image_data_root, feature_dir, f'{frame_id}_token_global.pt')

            try:
                lidar_token = torch.load(token_path, weights_only=True)
                lidar_token_global = torch.load(token_global_path, weights_only=True)
                lidar_tokens.append(lidar_token)
                lidar_tokens_global.append(lidar_token_global)
            except FileNotFoundError:
                print(f"Warning: LiDAR feature files not found for {bev_path}")
                continue
        
        if lidar_tokens:
            lidar_token_tensor = torch.stack(lidar_tokens)  # (obs_horizon, seq_len, 512)
            lidar_token_global_tensor = torch.stack(lidar_tokens_global)  # (obs_horizon, 1, 512)
        else:
            # Fallback if no LiDAR features found
            lidar_token_tensor = None
            lidar_token_global_tensor = None
        
        # Load VQA features from pt files
        vqa_paths = sample.get('vqa', [])
        vqa_features = []
        for vqa_path in vqa_paths:
            if vqa_path is None:
                continue
            
            # Load the .pt file
            full_vqa_path = os.path.join(self.image_data_root, vqa_path)
            try:
                vqa_feature = torch.load(full_vqa_path, weights_only=True)
                vqa_features.append(vqa_feature)
            except FileNotFoundError:
                print(f"Warning: VQA feature file not found for {vqa_path}")
                continue
        
        if vqa_features:
            vqa_feature_tensor = torch.stack(vqa_features)  # (obs_horizon, feature_dim, ...)
        else:
            # Fallback if no VQA features found
            vqa_feature_tensor = None
        
        # Convert sample data
        final_sample = dict()
        for key, value in sample.items():
            if key == 'rgb_hist_jpg':
                final_sample['rgb_hist_jpg'] = image_paths  
                if self.use_image or self.mode == 'val':
                    final_sample['image'] = images_tensor
                    if images_demo_tensor is not None:
                        final_sample['image_demo'] = images_demo_tensor
            if key == 'lidar_bev_hist':
                if self.mode == 'val':
                    final_sample['lidar_bev'] = lidar_bev_tensor
                    final_sample['lidar_bev_hist'] = lidar_bev_paths
                if lidar_token_tensor is not None:
                    final_sample['lidar_token'] = lidar_token_tensor
                    final_sample['lidar_token_global'] = lidar_token_global_tensor
            elif key == 'speed_hist':
                final_sample['speed'] = self.get_certain_data(sample, key_name='speed_hist')
            elif key == 'theta_hist':   
                final_sample['heading'] = self.get_certain_data(sample, key_name='theta_hist')
            elif key == 'ego_waypoints':
                final_sample['agent_pos'] = torch.from_numpy(sample['ego_waypoints'][1:]).float()
            elif key == 'vqa':
                # Load VQA features from pt files
                if vqa_feature_tensor is not None:
                    final_sample['vqa'] = vqa_feature_tensor
                else:
                    final_sample['vqa'] = None
            elif isinstance(value, np.ndarray):
                final_sample[key] = torch.from_numpy(value).float()
            else:
                final_sample[key] = value
        
        # Get obs_horizon for processing
        obs_horizon = lidar_token_tensor.shape[0] if lidar_token_tensor is not None else 2
        final_sample = self.hard_process(final_sample, obs_horizon=obs_horizon)

        return final_sample

    def hard_process(self, final_sample, obs_horizon):
        """
        Process sample data:
        1. Repeat command, next_command, and target_point to match obs_horizon.
        2. For VQA features, remove time dimension and keep only the last timestep.
        
        Args:
            final_sample (dict): Sample data
            obs_horizon (int): Number of observation frames
            
        Returns:
            dict: Processed sample with repeated fields and processed VQA
        """
        for key in ['command', 'next_command', 'target_point']:
            if key in final_sample:
                data = final_sample[key]
                if isinstance(data, torch.Tensor):
                    if data.ndim == 1:
                        data = data.unsqueeze(0)
                    repeated_data = data.repeat(obs_horizon, 1)
                    final_sample[key] = repeated_data
                elif isinstance(data, np.ndarray):
                    if data.ndim == 1:
                        data = data[np.newaxis, :]
                    repeated_data = np.tile(data, (obs_horizon, 1))
                    final_sample[key] = torch.from_numpy(repeated_data).float()
                else:
                    print(f"Warning: Unsupported data type for key '{key}' during hard_process.")
        
        # Process VQA features: remove time dimension, keep only last timestep
        if 'vqa' in final_sample and final_sample['vqa'] is not None:
            vqa_data = final_sample['vqa']
            if isinstance(vqa_data, torch.Tensor):
                if vqa_data.ndim >= 2:
                    final_sample['vqa'] = vqa_data[-1]  
                else:
                    print(f"Warning: VQA data has unexpected shape {vqa_data.shape}")
            elif isinstance(vqa_data, np.ndarray):
                if vqa_data.ndim >= 2:
                    final_sample['vqa'] = torch.from_numpy(vqa_data[-1]).float()
                else:
                    print(f"Warning: VQA data has unexpected shape {vqa_data.shape}")
            else:
                print(f"Warning: Unsupported data type for VQA during hard_process.")
        
        return final_sample

    def get_certain_data(self, sample, key_name=None):
        assert key_name is not None, "key_name must be provided"
        certain_data = sample[key_name]
        if isinstance(certain_data, np.ndarray):
            certain_data = torch.from_numpy(certain_data).float()
        elif isinstance(certain_data, (list, tuple)):
            certain_data = torch.tensor(certain_data, dtype=torch.float32)
        else:
            certain_data = certain_data
        return certain_data

    def load_image(self, image_paths, sample_path):
        images = []
        images_demo = []
        for img_path in image_paths:
            if img_path is None:
                # 如果路径为None，创建一个黑色图像
                images.append(torch.zeros(3, 256, 928))
                print(f"Warning: Image path is None in sample {sample_path}. Using black image.")
            else:
                full_img_path = os.path.join(self.image_data_root, img_path)
                try:
                    img = Image.open(full_img_path).convert('RGB')
                    image = pil_img2rgb(img)
                    image_tensor = self.vit_transform(image, img_num=1)
                    images.append(image_tensor)
                    if self.mode == 'val':
                        img_tensor_demo = self.image_transform(img)
                        images_demo.append(img_tensor_demo)
                except Exception as e:
                    print(f"Error loading image {full_img_path}: {e}")
                    images.append(torch.zeros(3, 490, 980))
                    images_demo.append(torch.zeros(3, 256, 928))
        
        # 堆叠图像
        images_demo_tensor = None
        if len(images) > 0:
            images_tensor = torch.stack(images)
            if self.mode == 'val':
                images_demo_tensor = torch.stack(images_demo)
        else:
            images_tensor = torch.zeros(2, 3, 490, 980)  # 默认obs_horizon=2
            if self.mode == 'val':
                images_demo_tensor = torch.zeros(2, 3, 256, 928)
        return images_tensor, images_demo_tensor
    
    def load_lidar_bev(self, bev_paths, sample_path):
        images = []
        for bev_path in bev_paths:
            if bev_path is None:
                images.append(torch.zeros(3, 448, 448))
                print(f"Warning: LiDAR BEV path is None in sample {sample_path}. Using black image.")
            else:
                full_bev_path = os.path.join(self.image_data_root, bev_path)
                bev_image = Image.open(full_bev_path)
                bev_tensor = self.lidar_bev_transform(bev_image)
                images.append(bev_tensor)

        if len(images) > 0:
            images_tensor = torch.stack(images)
        else:
            images_tensor = torch.zeros(2, 3, 448, 448)  # 默认obs_horizon=2

        return images_tensor

    # no space for image feature preprocessing
    def load_image_feature(self, image_path):
        """Load a single image and apply transformations."""
        full_img_path = os.path.join(self.image_data_root, image_path)
        image_feat = torch.load(full_img_path, weights_only=True)
        return image_feat

    def get_image_tensor_paths(self, lidar_bev_paths, folder_dir='rgb_features'):
        """Generate image tensor paths from LiDAR BEV paths."""
        if folder_dir is None:
            folder_dir = 'rgb_features'
        image_tensor_paths = []
        for bev_path in lidar_bev_paths:
            if bev_path is None:
                continue
            
            bev_dir = os.path.dirname(bev_path)
            frame_id = os.path.splitext(os.path.basename(bev_path))[0]
            image_tensor_dir = bev_dir.replace('lidar_bev', folder_dir)
            image_tensor_path = os.path.join(image_tensor_dir, f'{frame_id}_tensor.pt')
            image_tensor_paths.append(image_tensor_path)
        
        return image_tensor_paths
# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_trajectory(sample, obs_horizon, rand_idx, save_dir='/home/wang/Project/MoT-DP/image'):
    """
    Visualizes and saves the agent's trajectory and target point from a sample.
    
    Args:
        sample (dict): Data sample
        obs_horizon (int): Number of observation frames
        rand_idx (int): Sample index for saving
        save_dir (str): Directory to save visualization
    """
    agent_pos = sample.get('agent_pos')
    target_point = sample.get('target_point')
    
    if agent_pos is None:
        print("'agent_pos' not in sample. Skipping trajectory visualization.")
        return

    if agent_pos.ndim == 3 and agent_pos.shape[1] == 1:
        agent_pos = agent_pos.squeeze(1)
    
    pred_agent_pos = agent_pos
    
    plt.figure(figsize=(10, 10))

    # 收集所有y值用于自动扩展y轴
    y_values = []
    if len(pred_agent_pos) > 0:
        plt.plot(pred_agent_pos[:, 1], pred_agent_pos[:, 0], 'ro--', 
                label='Predicted agent_pos', markersize=8, linewidth=2)
        y_values.extend(pred_agent_pos[:, 1].tolist())

    if target_point is not None:
        if target_point.ndim == 2:
            target_point = target_point[0]
        plt.plot(target_point[1], target_point[0], 'g*', markersize=15, 
                label='Target point', markeredgecolor='black', markeredgewidth=1)
        y_values.append(target_point[1])

    # 加入原点y值
    y_values.append(0)

    # 自动扩展y轴范围
    if len(y_values) > 0:
        y_min = min(y_values)
        y_max = max(y_values)
        margin = max(1.0, 0.2 * (y_max - y_min))  # 至少1，或20%区间
        plt.ylim(y_min - margin, y_max + margin)

    plt.xlabel('Y (ego frame, lateral)', fontsize=12)
    plt.ylabel('X (ego frame, longitudinal)', fontsize=12)
    plt.title(f'Sample {rand_idx}: Predicted Trajectory\n'
              f'(Ego coordinate frame, origin at current position)', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.plot(0, 0, 'ko', markersize=10, label='Current position (origin)', zorder=5)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'sample_{rand_idx}_agent_pos.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"保存agent_pos图像到: {save_path}")


def visualize_observation_images(sample, obs_horizon, rand_idx, save_dir='/home/wang/Project/MoT-DP/image'):
    """
    Visualizes and saves the observation images from a sample.
    
    Args:
        sample (dict): Data sample
        obs_horizon (int): Number of observation frames
        rand_idx (int): Sample index for saving
        save_dir (str): Directory to save visualization
    """
    if 'image' not in sample:
        print("'image' not in sample. Skipping image visualization.")
        return
        
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
        
        plt.figure(figsize=(8, 2))
        plt.imshow(img_vis)
        plt.title(f'Random Sample {rand_idx} - Obs Image t={t}')
        plt.axis('off')
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'sample_{rand_idx}_obs_image_t{t}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"保存观测图像到: {save_path}")


def visualize_vqa_on_image(sample, obs_horizon, rand_idx, max_qa_pairs=5, 
                          save_dir='/home/wang/Project/MoT-DP/image'):
    """
    Visualizes VQA content and meta-actions as overlay on the observation image.
    
    B2D dataset specific visualization that includes VQA and meta-action information.
    
    Args:
        sample (dict): Data sample containing 'image' and optional 'vqa' fields
        obs_horizon (int): Number of observation frames
        rand_idx (int): Sample index for saving
        max_qa_pairs (int): Maximum number of QA pairs to display per category
        save_dir (str): Directory to save visualization
    """
    if 'image' not in sample:
        print("'image' not in sample. Skipping VQA visualization.")
        return
    
    images = sample['image']
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Use the last observation image (most recent frame)
    img_tensor = images[-1]
    
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
    
    # Convert to PIL Image for drawing
    pil_img = Image.fromarray(img_vis)
    img_width, img_height = pil_img.size
    
    canvas_width = img_width + 400
    canvas_height = img_height
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=(255, 255, 255))
    canvas.paste(pil_img, (0, 0))
    
    draw = ImageDraw.Draw(canvas)
    
    # Load fonts
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    text_x = img_width + 10
    text_y = 10
    
    # Draw Meta Actions
    draw.text((text_x, text_y), "Meta Actions", fill=(0, 0, 0), font=font_title)
    text_y += 30
    text_y += 20
    draw.text((text_x, text_y), "═══ META ACTIONS ═══", fill=(150, 50, 50), font=font_title)
    text_y += 25
    
    # Display meta_action_direction
    if 'meta_action_direction' in sample:
        meta_dir = sample['meta_action_direction']
        direction_value = _extract_value(meta_dir)
        
        direction_map = {
            'FOLLOW_LANE': 'Follow Lane',
            'CHANGE_LANE_LEFT': 'Change Lane Left',
            'CHANGE_LANE_RIGHT': 'Change Lane Right',
            'GO_STRAIGHT': 'Go Straight',
            'TURN_LEFT': 'Turn Left',
            'TURN_RIGHT': 'Turn Right'
        }
        direction_str = direction_map.get(direction_value, str(direction_value))
        draw.text((text_x, text_y), f"Direction: {direction_str}", 
                 fill=(0, 0, 150), font=font_text)
        text_y += 20
    
    # Display meta_action_speed
    if 'meta_action_speed' in sample:
        meta_speed = sample['meta_action_speed']
        speed_value = _extract_value(meta_speed)
        
        speed_map = {
            'KEEP': 'Keep Speed',
            'ACCELERATE': 'Accelerate',
            'DECELERATE': 'Decelerate',
            'STOP': 'Stop'
        }
        speed_str = speed_map.get(speed_value, str(speed_value))
        draw.text((text_x, text_y), f"Speed: {speed_str}", 
                 fill=(0, 150, 0), font=font_text)
        text_y += 20
    
    canvas_np = np.array(canvas)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(canvas_np)
    plt.title(f'Sample {rand_idx} - Image with Meta Actions')
    plt.axis('off')
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'sample_{rand_idx}_meta_actions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"保存Meta Actions图像到: {save_path}")


def _extract_value(data):
    """Helper function to extract value from different data types."""
    if isinstance(data, str):
        return data
    elif isinstance(data, (list, np.ndarray)):
        return data[0]
    elif isinstance(data, torch.Tensor):
        if data.numel() == 1:
            return data.item()
        else:
            return data[0].item() if data[0].numel() == 1 else str(data[0])
    else:
        return str(data)


def print_sample_details(sample, dataset, rand_idx, obs_horizon, 
                        save_dir='/home/wang/Project/MoT-DP/image'):
    """
    Prints detailed information about a sample and the dataset.
    
    Args:
        sample (dict): Data sample
        dataset (CARLAImageDataset): Dataset object
        rand_idx (int): Sample index
        obs_horizon (int): Number of observation frames
        save_dir (str): Directory for saving visualizations
    """
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

    print(f"\nDataset length: {len(dataset)}")
    first_sample = dataset[0]
    print("Sample keys:", first_sample.keys())
    if 'image' in first_sample:
        print("Image shape:", first_sample['image'].shape)
    if 'agent_pos' in first_sample:
        print("Agent pos shape:", first_sample['agent_pos'].shape)

    print("\nKey fields:")
    for key in ['town_name', 'speed', 'command', 'next_command', 'target_point', 
                'ego_waypoints', 'image', 'agent_pos', 'meta_action_direction', 'meta_action_speed', 'vqa']:
        if key in first_sample:
            value = first_sample[key]
            print(f"  {key}: shape={getattr(value, 'shape', 'N/A')}, type={type(value)}")
            if hasattr(value, '__len__') and not isinstance(value, str):
                print(f"    Length: {len(value)}")


# ============================================================================
# Test Function
# ============================================================================

def test_pdm():
    """Test with PDM Lite dataset."""
    import random
    
    dataset_path = '/home/wang/Project/carla_garage/tmp_data'
    obs_horizon = 2
    
    print("\n========== Testing PDM Lite Dataset ==========")
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        image_data_root='/home/wang/Project/carla_garage/data'
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


def test_b2d():
    """Test with Bench2Drive (B2D) dataset."""
    import random
    
    dataset_path = '/home/wang/Dataset/b2d_10scene/tmp_data'
    obs_horizon = 2
    
    print("\n========== Testing Bench2Drive (B2D) Dataset ==========")
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        image_data_root='/home/wang/Dataset/b2d_10scene'
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
    visualize_vqa_on_image(rand_sample, obs_horizon, rand_idx)
    print_sample_details(rand_sample, dataset, rand_idx, obs_horizon)


def test():
    """Test with default configuration."""
    import random
    
    # dataset_path = '/home/wang/Dataset/b2d_10scene/tmp_data'
    dataset_path = '/root/data/z_projects/PDM_Lite_processed_2obs_4hz'
    obs_horizon = 2
    
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        # image_data_root='/home/wang/Dataset/b2d_10scene'
        image_data_root= '/root/data/pdm_lite/'
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
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
        else:
            print(f"  {key}: value={value}")

    visualize_trajectory(rand_sample, obs_horizon, rand_idx)
    print_sample_details(rand_sample, dataset, rand_idx, obs_horizon)

    
if __name__ == "__main__":
    # test_pdm()
    # test_b2d()
    test()
