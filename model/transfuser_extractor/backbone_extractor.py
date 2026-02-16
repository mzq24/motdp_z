"""
TransFuser Backbone Feature Extractor
=====================================
提取 TransFuser backbone 的 BEV 特征，供下游端到端规划器使用。

输出:
- bev_feature: 融合后的原始 BEV 特征 (x4)
- bev_feature_upscale: 经过 FPN 上采样后的 BEV 特征 (p3)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import laspy
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

jsonpickle_numpy.register_handlers()

from config import GlobalConfig
from transfuser import TransfuserBackbone
import transfuser_utils as t_u


class TransFuserBackboneExtractor(nn.Module):
    """
    TransFuser Backbone 特征提取器
    
    该类封装了 TransFuser 的 backbone，用于提取融合后的 BEV 特征。
    - 加载预训练权重
    - 冻结所有参数
    - 输出原始 BEV 特征和上采样后的 BEV 特征
    """
    
    def __init__(self, config_path: str, model_path: str = None, device: str = 'cuda:0'):
        """
        初始化 TransFuser Backbone 提取器
        
        Args:
            config_path: 配置文件路径 (包含 config.json 的目录)
            model_path: 模型权重文件路径 (可选, 默认使用 config_path 下的 model_*.pth)
            device: 运行设备
        """
        super().__init__()
        
        self.device = torch.device(device)
        self.config_path = config_path
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 创建 backbone
        self.backbone = TransfuserBackbone(self.config)
        
        # 加载权重
        self._load_weights(config_path, model_path)
        
        # 冻结参数
        self._freeze_parameters()
        
        # 移动到指定设备
        self.backbone.to(self.device)
        self.backbone.eval()
        
        print(f"TransFuser Backbone Extractor initialized on {device}")
        print(f"  - Config: {config_path}")
        print(f"  - All parameters frozen")
        
    def _load_config(self, config_path: str) -> GlobalConfig:
        """加载训练时保存的配置"""
        config_file = os.path.join(config_path, 'config.json')
        
        with open(config_file, 'rt', encoding='utf-8') as f:
            json_config = f.read()
        
        loaded_config = jsonpickle.decode(json_config)
        
        # 创建新配置并更新
        config = GlobalConfig()
        config.__dict__.update(loaded_config.__dict__)
        
        return config
    
    def _load_weights(self, config_path: str, model_path: str = None):
        """加载模型权重"""
        if model_path is None:
            # 自动查找权重文件
            for file in os.listdir(config_path):
                if file.endswith('.pth') and file.startswith('model'):
                    model_path = os.path.join(config_path, file)
                    break
        
        if model_path is None:
            raise FileNotFoundError(f"No model weights found in {config_path}")
        
        print(f"Loading weights from: {model_path}")
        
        # 加载完整模型的 state_dict
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 提取 backbone 相关的权重
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key[len('backbone.'):]
                backbone_state_dict[new_key] = value
        
        # 加载 backbone 权重
        missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")
            
    def _freeze_parameters(self):
        """冻结所有参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 统计参数数量
        total_params = sum(p.numel() for p in self.backbone.parameters())
        frozen_params = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
    
    @torch.no_grad()
    def forward(self, rgb: torch.Tensor, lidar_bev: torch.Tensor):
        """
        前向传播，提取 BEV 特征
        
        Args:
            rgb: RGB 图像张量 [B, 3, H, W], 值范围 [0, 255] 的 float32
            lidar_bev: LiDAR BEV 张量 [B, C, H, W]
            
        Returns:
            dict: 包含以下键:
                - 'bev_feature': 融合后的原始 BEV 特征 (x4)
                - 'bev_feature_upscale': 经过 FPN 上采样后的 BEV 特征 (p3)
                - 'fused_features': 全局融合特征
                - 'image_feature_grid': 图像特征网格 (如果配置启用)
        """
        # 获取模型参数所在设备，确保输入在正确的设备上
        param_device = next(self.backbone.parameters()).device
        rgb = rgb.to(param_device)
        lidar_bev = lidar_bev.to(param_device)
        
        # 调用原始的 forward 方法，但我们需要修改以获取中间特征
        # 原始 forward 返回: features (p3), fused_features, image_feature_grid
        # 我们需要额外获取 x4 (融合后的原始 BEV 特征)
        
        bev_feature, bev_feature_upscale, fused_features, image_feature_grid = \
            self._forward_with_intermediate(rgb, lidar_bev)
        
        return {
            'bev_feature': bev_feature,  # 原始 BEV 特征 (x4)
            'bev_feature_upscale': bev_feature_upscale,  # 上采样后的 BEV 特征 (p3)
            'fused_features': fused_features,
            'image_feature_grid': image_feature_grid
        }
    
    def _forward_with_intermediate(self, image, lidar):
        """
        修改后的 forward，返回中间特征
        """
        if self.config.normalize_imagenet:
            image_features = t_u.normalize_imagenet(image)
        else:
            image_features = image
        
        if self.backbone.lidar_video:
            batch_size = lidar.shape[0]
            lidar_features = lidar.view(batch_size, -1, self.config.lidar_seq_len, 
                                        self.config.lidar_resolution_height,
                                        self.config.lidar_resolution_width)
        else:
            lidar_features = lidar
        
        # 生成层迭代器
        image_layers = iter(self.backbone.image_encoder.items())
        lidar_layers = iter(self.backbone.lidar_encoder.items())
        
        # Stem layer
        if len(self.backbone.image_encoder.return_layers) > 4:
            image_features = self.backbone.forward_layer_block(
                image_layers, self.backbone.image_encoder.return_layers, image_features)
        if len(self.backbone.lidar_encoder.return_layers) > 4:
            lidar_features = self.backbone.forward_layer_block(
                lidar_layers, self.backbone.lidar_encoder.return_layers, lidar_features)
        
        # 4 个 block
        for i in range(4):
            image_features = self.backbone.forward_layer_block(
                image_layers, self.backbone.image_encoder.return_layers, image_features)
            lidar_features = self.backbone.forward_layer_block(
                lidar_layers, self.backbone.lidar_encoder.return_layers, lidar_features)
            image_features, lidar_features = self.backbone.fuse_features(image_features, lidar_features, i)
        
        # 获取原始 BEV 特征 (x4)
        if self.config.detect_boxes or self.config.use_bev_semantic:
            if self.backbone.lidar_video:
                lidar_features_for_bev = torch.mean(lidar_features, dim=2)
            else:
                lidar_features_for_bev = lidar_features
            x4 = lidar_features_for_bev  # 这就是融合后的原始 BEV 特征
        else:
            x4 = None
        
        # 图像特征网格
        image_feature_grid = None
        if self.config.use_semantic or self.config.use_depth:
            image_feature_grid = image_features
        
        # 全局融合特征
        if self.config.transformer_decoder_join:
            fused_features = lidar_features
        else:
            image_features_pooled = self.backbone.global_pool_img(image_features)
            image_features_pooled = torch.flatten(image_features_pooled, 1)
            lidar_features_pooled = self.backbone.global_pool_lidar(lidar_features)
            lidar_features_pooled = torch.flatten(lidar_features_pooled, 1)
            
            if self.config.add_features:
                lidar_features_pooled = self.backbone.lidar_to_img_features_end(lidar_features_pooled)
                fused_features = image_features_pooled + lidar_features_pooled
            else:
                fused_features = torch.cat((image_features_pooled, lidar_features_pooled), dim=1)
        
        # 获取上采样后的 BEV 特征 (p3)
        if self.config.detect_boxes or self.config.use_bev_semantic:
            bev_feature_upscale = self.backbone.top_down(x4)
        else:
            bev_feature_upscale = None
        
        return x4, bev_feature_upscale, fused_features, image_feature_grid
    
    def preprocess_rgb(self, rgb_path: str) -> torch.Tensor:
        """
        预处理 RGB 图像 (与训练时完全一致)
        
        Args:
            rgb_path: RGB 图像文件路径
            
        Returns:
            torch.Tensor: 预处理后的 RGB 张量 [1, 3, H, W]
        """
        # 读取图像
        image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 裁剪 (与 data.py 中一致)
        image = t_u.crop_array(self.config, image)
        
        # 转换为 PyTorch 格式 (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        # 转换为张量
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        return image
    
    def preprocess_lidar(self, lidar_path: str) -> torch.Tensor:
        """
        预处理 LiDAR 点云 (与训练时完全一致)
        
        Args:
            lidar_path: LiDAR .laz 文件路径
            
        Returns:
            torch.Tensor: 预处理后的 LiDAR BEV 张量 [1, C, H, W]
        """
        # 读取 LiDAR 数据
        las_object = laspy.read(lidar_path)
        lidar = las_object.xyz
        
        # 转换为 histogram features (与 data.py 中的 lidar_to_histogram_features 一致)
        lidar_bev = self.lidar_to_histogram_features(lidar, use_ground_plane=self.config.use_ground_plane)
        
        # 转换为张量
        lidar_bev = torch.from_numpy(lidar_bev).float().unsqueeze(0)
        
        return lidar_bev
    
    def lidar_to_histogram_features(self, lidar: np.ndarray, use_ground_plane: bool) -> np.ndarray:
        """
        将 LiDAR 点云转换为 2-bin histogram BEV 表示
        (与 data.py 中的实现完全一致)
        
        Args:
            lidar: (N, 3) numpy, LiDAR 点云
            use_ground_plane: 是否使用地平面
            
        Returns:
            (C, H, W) numpy, LiDAR BEV 特征
        """
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(self.config.min_x, self.config.max_x,
                                (self.config.max_x - self.config.min_x) * int(self.config.pixels_per_meter) + 1)
            ybins = np.linspace(self.config.min_y, self.config.max_y,
                                (self.config.max_y - self.config.min_y) * int(self.config.pixels_per_meter) + 1)
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self.config.hist_max_per_pixel] = self.config.hist_max_per_pixel
            overhead_splat = hist / self.config.hist_max_per_pixel
            return overhead_splat.T
        
        # 移除车辆上方的点
        lidar = lidar[lidar[..., 2] < self.config.max_height_lidar]
        below = lidar[lidar[..., 2] <= self.config.lidar_split_height]
        above = lidar[lidar[..., 2] > self.config.lidar_split_height]
        below_features = splat_points(below)
        above_features = splat_points(above)
        
        if use_ground_plane:
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        return features
    
    def extract_features(self, rgb_path: str, lidar_path: str):
        """
        便捷方法: 从文件路径直接提取特征
        
        Args:
            rgb_path: RGB 图像路径
            lidar_path: LiDAR .laz 文件路径
            
        Returns:
            dict: BEV 特征字典
        """
        rgb = self.preprocess_rgb(rgb_path)
        lidar_bev = self.preprocess_lidar(lidar_path)
        
        return self.forward(rgb, lidar_bev)


def test_extractor():
    """
    Dummy 测试函数
    """
    print("=" * 60)
    print("TransFuser Backbone Extractor - Dummy Test")
    print("=" * 60)
    
    # 配置路径
    config_path = "/home/wang/Project/carla_garage/leaderboard/leaderboard/pretrained_models/all_towns"
    
    # 创建提取器
    extractor = TransFuserBackboneExtractor(config_path=config_path, device='cuda:0')
    
    # 创建 dummy 输入
    print("\nCreating dummy inputs...")
    batch_size = 1
    
    # RGB: [B, 3, H, W] - 裁剪后的尺寸
    rgb_height = extractor.config.cropped_height if extractor.config.crop_image else extractor.config.camera_height
    rgb_width = extractor.config.cropped_width if extractor.config.crop_image else extractor.config.camera_width
    dummy_rgb = torch.randn(batch_size, 3, rgb_height, rgb_width) * 255
    dummy_rgb = dummy_rgb.clamp(0, 255)
    
    # LiDAR BEV: [B, C, H, W]
    lidar_channels = 2 if extractor.config.use_ground_plane else 1
    lidar_channels *= extractor.config.lidar_seq_len  # 考虑时序
    dummy_lidar = torch.randn(batch_size, lidar_channels, 
                              extractor.config.lidar_resolution_height,
                              extractor.config.lidar_resolution_width)
    
    print(f"  RGB shape: {dummy_rgb.shape}")
    print(f"  LiDAR BEV shape: {dummy_lidar.shape}")
    
    # 前向传播
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = extractor(dummy_rgb, dummy_lidar)
    
    print("\nOutput shapes:")
    for key, value in output.items():
        if value is not None:
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: None")
    
    print("\n" + "=" * 60)
    print("Dummy test completed successfully!")
    print("=" * 60)
    
    return extractor, output


if __name__ == "__main__":
    test_extractor()
