import numpy as np
from PIL import Image
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


class CARLAImageDataset(torch.utils.data.Dataset):
    """
    一个为预处理好的、单个样本文件设计的高效Dataset类。
    """
    def __init__(self,
                 dataset_path: str, # 指向处理好的数据根目录 (包含 train/val)
                 mode: str = 'train',):

        self.mode = mode
    
        data_dir = os.path.join(dataset_path, self.mode)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Processed data directory not found: {data_dir}")

        self.sample_files = sorted(glob.glob(os.path.join(data_dir, "sample_*.pkl")))
        if not self.sample_files:
            raise FileNotFoundError(f"No sample files found in {data_dir}. Did you run the preprocessing script?")
        
        print(f"Found {len(self.sample_files)} preprocessed samples in '{data_dir}'.")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        # 1. 获取文件路径并加载单个样本
        sample_path = self.sample_files[idx]
        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)

        # 2. 数据已经基本处理好，只需做最后的tensor转换和数据复制
        # 注意：图像已经是Tensor了
        final_sample = dict()
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                final_sample[key] = torch.from_numpy(value).float()
            else:
                final_sample[key] = value
        
        return final_sample


def visualize_trajectory(sample, obs_horizon, rand_idx):
    """
    Visualizes and saves the agent's trajectory and target point from a sample.
    """
    agent_pos = sample.get('agent_pos')
    target_point = sample.get('target_point')
    
    if agent_pos is None:
        print("'agent_pos' not in sample. Skipping trajectory visualization.")
        return

    if agent_pos.ndim == 3 and agent_pos.shape[1] == 1:
        agent_pos = agent_pos.squeeze(1)
    
    obs_agent_pos = agent_pos[:obs_horizon]
    pred_agent_pos = agent_pos[obs_horizon:]
    
    plt.figure(figsize=(10, 8))
    
    # Plot observed and predicted points
    if len(obs_agent_pos) > 0:
        plt.plot(obs_agent_pos[:, 1], obs_agent_pos[:, 0], 'bo-', label='Observed agent_pos', markersize=8, linewidth=2)
    if len(pred_agent_pos) > 0:
        plt.plot(pred_agent_pos[:, 1], pred_agent_pos[:, 0], 'ro--', label='Predicted agent_pos', markersize=8, linewidth=2)
    
    # Plot target point
    if target_point is not None:
        if target_point.ndim == 2:
            target_point = target_point[0]
        plt.plot(target_point[1], target_point[0], 'g*', markersize=15, label='Target point (relative)', markeredgecolor='black', markeredgewidth=1)
        if len(obs_agent_pos) > 0:
            last_obs = obs_agent_pos[-1]
            plt.plot([last_obs[1], target_point[1]], [last_obs[0], target_point[0]], 'g--', alpha=0.5, linewidth=1, label='Direction to target')

    # Mark the reference point
    if len(obs_agent_pos) > 0:
        last_obs = obs_agent_pos[-1]
        plt.plot(last_obs[1], last_obs[0], 'ks', markersize=10, label='Reference point (last obs)', markerfacecolor='yellow', markeredgecolor='black')
    
    plt.xlabel('Y (relative to last obs)')
    plt.ylabel('X (relative to last obs)')
    plt.title(f'Sample {rand_idx}: Trajectory & Target Point\n(All positions relative to last observation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    save_path = f'/media/z/data/mzq/others/MoT-DP/sample_{rand_idx}_agent_pos_with_target.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"保存agent_pos和target_point图像到: {save_path}")

def visualize_observation_images(sample, obs_horizon, rand_idx):
    """
    Visualizes and saves the observation images from a sample.
    """
    if 'image' not in sample:
        print("'image' not in sample. Skipping image visualization.")
        return
        
    images = sample['image']  # shape: (obs_horizon, C, H, W)
    for t, img_arr in enumerate(images):
        if img_arr.shape[0] == 3:  # (C, H, W)
            img_vis = np.moveaxis(img_arr, 0, -1).astype(np.uint8)
        else:
            img_vis = img_arr.astype(np.uint8)
        
        plt.figure(figsize=(8, 2))
        plt.imshow(img_vis)
        plt.title(f'Random Sample {rand_idx} - Obs Image t={t}')
        plt.axis('off')
        plt.tight_layout()
        
        save_path = f'/media/z/data/mzq/others/MoT-DP/sample_{rand_idx}_obs_image_t{t}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"保存观测图像到: {save_path}")

def print_sample_details(sample, dataset, rand_idx, obs_horizon):
    """
    Prints detailed information about a sample and the dataset.
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
        if agent_pos is not None and len(agent_pos) > obs_horizon -1:
            last_obs = agent_pos[obs_horizon-1]
            distance_to_target = np.linalg.norm(target_point - last_obs)
            print(f"目标点距离最后观测点的距离: {distance_to_target:.3f}")

    print(f"\nDataset length: {len(dataset)}")
    first_sample = dataset[0]
    print("Sample keys:", first_sample.keys())
    print("Image shape:", first_sample['image'].shape)
    print("Agent pos shape:", first_sample['agent_pos'].shape)

    print("\nNew CARLA fields:")
    for key in ['town_name', 'speed', 'command', 'next_command', 'target_point', 'ego_waypoints', 'image', 'agent_pos']:
        if key in first_sample:
            value = first_sample[key]
            print(f"  {key}: shape={getattr(value, 'shape', 'N/A')}, type={type(value)}")
            if hasattr(value, '__len__') and not isinstance(value, str):
                print(f"    Length: {len(value)}")

def test():
    import random
    # 重要：现在路径应该指向预处理好的数据集
    dataset_path = '/media/z/data/dataset/carla/processed_mini_data'
    obs_horizon = 2
    
    # 使用新的Dataset类
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        mode='train' # or 'val'
    )
    
    print(f"\n总样本数: {len(dataset)}")
    if len(dataset) == 0:
        print("数据为空，无法进行测试。")
        return

    rand_idx = random.choice(range(len(dataset)))
    rand_sample = dataset[rand_idx]
    print(f"\n随机选择的样本索引: {rand_idx}")

    # 检查样本内容
    print("\nSample keys:", rand_sample.keys())
    for key, value in rand_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
        else:
            print(f"  {key}: value={value}")

    # visualize_trajectory(rand_sample, obs_horizon, rand_idx)
    # visualize_observation_images(rand_sample, obs_horizon, rand_idx)
    # print_sample_details(rand_sample, dataset, rand_idx, obs_horizon)

    
if __name__ == "__main__":
    test()
