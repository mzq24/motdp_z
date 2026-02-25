"""
数据集预处理脚本
================
加载 TransFuser backbone，处理 pdm_lite_mini 数据集，
保存 BEV feature 和上采样后的 BEV feature。

Following DiffusionDriveV2: only save bev_feature and bev_feature_upsample.
fused_features and image_feature_grid are NOT used.

使用方法:
    python preprocess_dataset.py --dataset_path /home/wang/Dataset/pdm_lite_mini \
                                 --config_path /home/wang/Project/carla_garage/leaderboard/leaderboard/pretrained_models/all_towns

输出:
    在每个 route 目录下创建 transfuser_feature 子文件夹，保存:
    - 0001_feature.pt: 融合后的原始 BEV 特征
    - 0001_feature_upsample.pt: 上采样后的 BEV 特征
"""

import os
import sys
import argparse
import gzip
from pathlib import Path
from tqdm import tqdm
import re

import torch
import numpy as np
import cv2
import laspy
import ujson

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from backbone_extractor import TransFuserBackboneExtractor
import transfuser_utils as t_u


class DatasetPreprocessor:
    """
    数据集预处理器
    
    遍历 pdm_lite 数据集，提取并保存 TransFuser BEV 特征
    """
    
    def __init__(self, 
                 dataset_path: str, 
                 config_path: str, 
                 model_path: str = None,
                 device: str = 'cuda:0',
                 batch_size: int = 1,
                 skip_existing: bool = True):
        """
        初始化预处理器
        
        Args:
            dataset_path: 数据集根目录路径
            config_path: TransFuser 配置文件路径
            model_path: 模型权重路径 (可选)
            device: 运行设备
            batch_size: 批处理大小
            skip_existing: 是否跳过已存在的特征文件
        """
        self.dataset_path = Path(dataset_path)
        self.config_path = config_path
        self.device = device
        self.batch_size = batch_size
        self.skip_existing = skip_existing
        
        # 创建特征提取器
        print("Initializing TransFuser Backbone Extractor...")
        self.extractor = TransFuserBackboneExtractor(
            config_path=config_path,
            model_path=model_path,
            device=device
        )
        self.config = self.extractor.config
        
        # 特征保存目录名
        self.feature_dir_name = "transfuser_feature"
        
    def find_all_routes(self):
        """
        查找数据集中所有有效的 route 目录
        
        Returns:
            list: route 目录路径列表
        """
        routes = []
        
        # 遍历数据集目录
        for scenario_dir in self.dataset_path.iterdir():
            if not scenario_dir.is_dir():
                continue
            
            # 遍历 scenario 下的 route 目录
            for route_dir in scenario_dir.iterdir():
                if not route_dir.is_dir():
                    continue
                
                # 跳过失败的 route
                if route_dir.name.startswith('FAILED_'):
                    continue
                
                # 检查必要的文件是否存在
                lidar_dir = route_dir / 'lidar'
                rgb_dir = route_dir / 'rgb'
                results_file = route_dir / 'results.json.gz'
                
                if lidar_dir.exists() and rgb_dir.exists() and results_file.exists():
                    # 验证数据完整性
                    try:
                        with gzip.open(results_file, 'rt', encoding='utf-8') as f:
                            results = ujson.load(f)
                        # 可以在这里添加更多验证条件
                        routes.append(route_dir)
                    except Exception as e:
                        print(f"Warning: Failed to read results for {route_dir}: {e}")
                        continue
        
        return routes
    
    def get_frame_count(self, route_dir: Path) -> int:
        """获取 route 中的帧数"""
        lidar_dir = route_dir / 'lidar'
        return len(list(lidar_dir.glob('*.laz')))
    
    def preprocess_rgb(self, rgb_path: str) -> torch.Tensor:
        """
        预处理 RGB 图像 (与训练时完全一致)
        """
        # 读取图像
        image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 裁剪
        image = t_u.crop_array(self.config, image)
        
        # 转换为 PyTorch 格式 (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        # 转换为张量
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        return image
    
    def preprocess_lidar(self, lidar_path: str) -> torch.Tensor:
        """
        预处理 LiDAR 点云 (与训练时完全一致)
        """
        # 读取 LiDAR 数据
        las_object = laspy.read(lidar_path)
        lidar = las_object.xyz
        
        # 转换为 histogram features
        lidar_bev = self.extractor.lidar_to_histogram_features(
            lidar, use_ground_plane=self.config.use_ground_plane)
        
        # 转换为张量
        lidar_bev = torch.from_numpy(lidar_bev).float().unsqueeze(0)
        
        return lidar_bev
    
    def _load_frame(self, args):
        """线程安全的单帧读取，供 ThreadPoolExecutor 调用"""
        lidar_file, rgb_file, frame_num = args
        try:
            rgb = self.preprocess_rgb(str(rgb_file))
            lidar_bev = self.preprocess_lidar(str(lidar_file))
            return frame_num, rgb, lidar_bev
        except Exception as e:
            print(f"Error reading frame {frame_num}: {e}")
            return frame_num, None, None

    def process_route(self, route_dir: Path):
        """
        处理单个 route，提取并保存所有帧的特征。
        使用批处理 (batch_size) 和线程预取来提升吞吐量。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        feature_dir = route_dir / self.feature_dir_name
        feature_dir.mkdir(exist_ok=True)

        lidar_dir = route_dir / 'lidar'
        rgb_dir = route_dir / 'rgb'

        frame_files = sorted(lidar_dir.glob('*.laz'))

        # 过滤已存在的帧
        pending = []
        for lidar_file in frame_files:
            frame_num = lidar_file.stem
            feature_path = feature_dir / f"{frame_num}_feature.pt"
            feature_upsample_path = feature_dir / f"{frame_num}_feature_upsample.pt"
            if self.skip_existing and feature_path.exists() and feature_upsample_path.exists():
                continue
            rgb_file = rgb_dir / f"{frame_num}.jpg"
            if not rgb_file.exists():
                print(f"Warning: RGB file not found: {rgb_file}")
                continue
            pending.append((lidar_file, rgb_file, frame_num))

        if not pending:
            return

        # 按 batch_size 分批，使用线程池并行读取 IO
        io_workers = min(self.batch_size, 8)
        for batch_start in range(0, len(pending), self.batch_size):
            batch_args = pending[batch_start: batch_start + self.batch_size]

            # 并行读取本批帧
            loaded = {}
            with ThreadPoolExecutor(max_workers=io_workers) as executor:
                futures = {executor.submit(self._load_frame, args): args[2] for args in batch_args}
                for future in as_completed(futures):
                    frame_num, rgb, lidar_bev = future.result()
                    if rgb is not None:
                        loaded[frame_num] = (rgb, lidar_bev)

            if not loaded:
                continue

            # 按顺序 stack 成 batch（保持帧号顺序）
            ordered = [(fn, loaded[fn]) for _, _, fn in batch_args if fn in loaded]
            frame_nums = [fn for fn, _ in ordered]
            rgb_batch = torch.cat([d[0] for _, d in ordered], dim=0)    # (B, 3, H, W)
            lidar_batch = torch.cat([d[1] for _, d in ordered], dim=0)  # (B, C, H, W)

            try:
                with torch.no_grad():
                    output = self.extractor(rgb_batch, lidar_batch)

                bev_feature = output['bev_feature']          # (B, 1512, 8, 8)
                bev_upsample = output['bev_feature_upscale'] # (B, 64, 64, 64)

                for j, frame_num in enumerate(frame_nums):
                    if bev_feature is not None:
                        torch.save(bev_feature[j:j+1].cpu(),
                                   feature_dir / f"{frame_num}_feature.pt")
                    if bev_upsample is not None:
                        torch.save(bev_upsample[j:j+1].cpu(),
                                   feature_dir / f"{frame_num}_feature_upsample.pt")

            except Exception as e:
                print(f"Error processing batch in {route_dir}: {e}")
                continue
    
    def run(self, num_workers: int = 1):
        """
        运行数据集预处理
        
        Args:
            num_workers: 工作进程数 (目前仅支持单进程)
        """
        print("\n" + "=" * 60)
        print("Starting Dataset Preprocessing")
        print("=" * 60)
        print(f"Dataset path: {self.dataset_path}")
        print(f"Feature directory name: {self.feature_dir_name}")
        print(f"Skip existing: {self.skip_existing}")
        
        # 查找所有 route
        print("\nFinding all routes...")
        routes = self.find_all_routes()
        print(f"Found {len(routes)} valid routes")
        
        if len(routes) == 0:
            print("No valid routes found. Exiting.")
            return
        
        # 统计总帧数
        total_frames = sum(self.get_frame_count(r) for r in routes)
        print(f"Total frames to process: {total_frames}")
        
        # 处理每个 route
        print("\nProcessing routes...")
        for route_dir in tqdm(routes, desc="Routes"):
            frame_count = self.get_frame_count(route_dir)
            
            # 使用内部进度条
            tqdm.write(f"\nProcessing: {route_dir.name} ({frame_count} frames)")
            
            self.process_route(route_dir)
        
        print("\n" + "=" * 60)
        print("Dataset preprocessing completed!")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess pdm_lite dataset and extract TransFuser BEV features"
    )
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        default='/home/wang/Dataset/pdm_lite_mini',
        help='Path to the pdm_lite_mini dataset'
    )
    parser.add_argument(
        '--config_path', 
        type=str, 
        default='/home/wang/Project/carla_garage/leaderboard/leaderboard/pretrained_models/all_towns',
        help='Path to TransFuser config directory'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=None,
        help='Path to model weights (optional, auto-detected from config_path)'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda:0',
        help='Device to use for inference'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--no_skip_existing', 
        action='store_true',
        help='Do not skip existing feature files'
    )
    
    args = parser.parse_args()
    
    # 创建预处理器并运行
    preprocessor = DatasetPreprocessor(
        dataset_path=args.dataset_path,
        config_path=args.config_path,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing
    )
    
    preprocessor.run()


if __name__ == "__main__":
    main()
