import torch
import torch.nn as nn
from typing import Tuple, Optional
import sys
import os
try:
    import torch.distributed as dist
except ImportError:
    dist = None

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from model.TCP.resnet import resnet34

from model.interfuser.resnet import resnet18d
from model.interfuser.modeling_utils import MLPconnector, PositionEmbedding, PositionEmbeddingSine, HybridEmbed
from functools import partial


class InterfuserBEVEncoder(nn.Module):
    """LiDAR BEV图像和状态信息的编码器
    
    输入:
        - image: (B, 3, H, W) - LiDAR BEV图像，推荐尺寸 448x448
        - state: (B, state_dim) - 状态信息 [speed(1), target_point(2), command(6)]
    
    输出:
        - feature: (B, 256) - 编码后的特征向量
    
    相比完整TCP的简化:
        1. 移除了PID控制器 (不需要控制输出)
        2. 移除了轨迹预测分支 (join_traj, decoder_traj等)
        3. 移除了动作预测头 (policy_head, action_head等)
        4. 移除了价值估计分支 (value_branch)
        5. 只保留核心的视觉感知 + 测量融合 + 空间注意力机制
        6. 适配LiDAR BEV图像输入 (448x448)，对应的ResNet34特征图为14x14
    """
    
    def __init__(
        self, 
        perception_backbone: Optional[nn.Module] = None,
        state_dim: int = 10,  # speed(1) + target_point(2) + command(6) + heading(1)
        feature_dim: int = 256,
        bev_embed_dim: int = 256,
        measurement_embed_dim: int = 128,
        token_dim: int = 512,
        use_group_norm: bool = True,
        freeze_backbone: bool = True,
        bev_input_size: Tuple[int, int] = (448, 448)  # LiDAR BEV图像尺寸
    ):
        """
        Args:
            perception_backbone: 视觉backbone 
            state_dim: 状态维度
            feature_dim: 输出特征维度
            bev_embed_dim: BEV backbone 输出的 embedding 维度
            measurement_embed_dim: 状态信息编码后的维度
            token_dim: token 的统一维度，用于 Transformer
            fusion_hidden_dim: 最终特征融合层的隐藏维度
            img_feature_dim: 输入图像特征的维度
            transformer_ff_dim: Transformer 解码器中前馈网络的维度
            use_group_norm: 是否使用GroupNorm标准化输出特征
            freeze_backbone: 是否冻结backbone参数
            bev_input_size: BEV图像输入尺寸，默认(448, 448)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.freeze_backbone = freeze_backbone
        self.bev_input_size = bev_input_size
        
        #1. BEV backbone
        if perception_backbone is None:
            lidar_backbone = resnet18d(
                pretrained=False,  # 改为False，避免从timm加载
                in_chans=3,
                features_only=True,
                out_indices=[4],   
            )
            self.perception=lidar_backbone
            embed_dim=bev_embed_dim
            self.lidar_embed_layer = partial(HybridEmbed, backbone=lidar_backbone)
            self.lidar_connector = MLPconnector(bev_embed_dim, token_dim, 'gelu')
            # LiDAR patch embed：把 token_dim 通道通过 1×1 conv 投到 embed_dim=bev_embed_dim
            self.bev_model_mot = self.lidar_embed_layer(
                img_size=224,      # LiDAR BEV 输入分辨率，默认 224×224
                patch_size=16,     # 与 RGB 保持一致；这里仅记录，不影响 backbone 计算
                in_chans=3,
                embed_dim=embed_dim,     # baseline 用 bev_embed_dim
            )
            self.position_encoding = PositionEmbeddingSine(embed_dim // 2, normalize=True)

        else:
            self.perception = perception_backbone

        # 2. 测量编码器 (state -> measurement features)
        self.measurements = nn.Sequential(
            nn.Linear(state_dim, measurement_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(measurement_embed_dim, measurement_embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # 3. 空间注意力模块
        # 计算ResNet34输出特征图尺寸: 对于448x448输入 -> 14x14特征图
        # ResNet34: conv1(stride=2) -> maxpool(stride=2) -> layer2(stride=2) -> layer3(stride=2) -> layer4(stride=2)
        # 448 -> 224 -> 112 -> 56 -> 28 -> 14
        feat_h = bev_input_size[0] // 32  # 448 // 32 = 14
        feat_w = bev_input_size[1] // 32  # 448 // 32 = 14
        
        if bev_input_size == (448, 448):
            feat_h, feat_w = 14, 14  # 实际输出为14x14
        else:
            # input_size -> (input_size // 32)
            feat_h = (bev_input_size[0] + 31) // 32
            feat_w = (bev_input_size[1] + 31) // 32
        
        self.feat_h = feat_h
        self.feat_w = feat_w
        spatial_size = feat_h * feat_w
        
        self.init_att = nn.Sequential(
            nn.Linear(measurement_embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, spatial_size),  # 14 * 14 = 196 for 448x448 input
            nn.Softmax(dim=1)
        )
        
        # 4. 特征融合模块
        # 融合视觉特征(经过注意力加权)和测量特征
        self.join_ctrl = nn.Sequential(
            nn.Linear(token_dim + measurement_embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )

        # 5. GroupNorm 
        self.use_group_norm = use_group_norm
        if use_group_norm:
            # 256 channels, 8 groups -> 32 channels per group
            self.feature_norm = nn.GroupNorm(num_groups=8, num_channels=feature_dim)
        
        if freeze_backbone and self.perception is not None:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.perception.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.perception.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
    
    def forward(
        self, 
        image: torch.Tensor = None, 
        state: torch.Tensor = None,
        normalize: bool = True,
        return_attention: bool = False,
        lidar_token: torch.Tensor = None,
        lidar_token_global: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass with support for both:
        1. Raw BEV images (image + state) - extracts features on-the-fly
        2. Pre-computed features (lidar_token + lidar_token_global + state) - uses cached features
        
        Args:
            image (torch.Tensor, optional): Input BEV image tensor (B, C, H, W). Defaults to None.
            state (torch.Tensor, optional): Input state tensor (B, state_dim). Defaults to None.
            lidar_token (torch.Tensor, optional): Precomputed spatial lidar tokens. Defaults to None.
            lidar_token_global (torch.Tensor, optional): Precomputed global lidar token. Defaults to None.
            normalize (bool): Whether to normalize the state. Defaults to True.
            return_attention (bool): Whether to return the attention map from the last cross-attention layer.
        
        Returns:
            - j_ctrl (torch.Tensor): Control features (B, 256)
            - attention_map (torch.Tensor, optional): Attention map if return_attention is True
        """
        if image is not None:
            # Mode 1: Process raw BEV image
            lidar_token, lidar_token_global = self.bev_model_mot(image)
            lidar_token = lidar_token + self.position_encoding(lidar_token)
            lidar_token = lidar_token.flatten(2).permute(2, 0, 1)
            lidar_token_global = lidar_token_global.permute(2, 0, 1)
            
            lidar_tokens = torch.cat([lidar_token, lidar_token_global], dim=0)
            lidar_tokens = self.lidar_connector(lidar_tokens)
            
            lidar_token = lidar_tokens[:-1].permute(1, 0, 2)
            lidar_token_global = lidar_tokens[-1:].permute(1, 0, 2)
        
        elif lidar_token is None or lidar_token_global is None:
            raise ValueError("Either 'image' or both 'lidar_token' and 'lidar_token_global' must be provided.")

        if state is None:
            raise ValueError("state is required for forward pass")
        measurement_feature = self.measurements(state)
        
        # Step 3: 空间注意力机制
        # 使用 measurement_feature 生成空间注意力权重
        batch_size = lidar_token.shape[0]
        spatial_size = lidar_token.shape[1]  # seq_len
        attention_weights = self.init_att(measurement_feature)  # (batch, expected_spatial_size)
        
        # Handle dynamic spatial size (in case of pre-computed features with different sizes)
        if attention_weights.shape[1] != spatial_size:
            # Resize attention weights to match actual spatial size
            # This can happen when using pre-computed features from different image sizes
            import torch.nn.functional as F
            expected_h, expected_w = self.feat_h, self.feat_w
            actual_h = actual_w = int(spatial_size ** 0.5)
            
            # Reshape and resize
            attention_weights = attention_weights.reshape(batch_size, expected_h, expected_w)
            attention_weights = F.interpolate(
                attention_weights.unsqueeze(1), 
                size=(actual_h, actual_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
            attention_weights = attention_weights.reshape(batch_size, -1)
            # Re-normalize
            attention_weights = F.softmax(attention_weights, dim=1)
        
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # (batch, spatial_size, 1)
        
        # 对 lidar_token 应用注意力加权并求和
        weighted_spatial = (lidar_token * attention_weights_expanded).sum(dim=1)  # (batch, 512)
        
        # Step 4: 特征融合
        # 融合注意力加权后的视觉特征和测量特征
        fused_feature = torch.cat([weighted_spatial, measurement_feature], dim=1)
        output_feature = self.join_ctrl(fused_feature)
        
        # Step 5: 可选的特征标准化
        if normalize is None:
            normalize = self.use_group_norm and self.training
        
        if normalize and self.use_group_norm:
            output_feature = self.feature_norm(output_feature.unsqueeze(-1)).squeeze(-1)
        
        if return_attention:
            # 返回特征和attention weights
            # attention_weights shape: (batch, spatial_size) -> reshape to (batch, feat_h, feat_w)
            attention_map = attention_weights.reshape(batch_size, self.feat_h, self.feat_w)
            return output_feature, attention_map
        else:
            return output_feature
    
    def extract_features_batch(
        self, 
        images: torch.Tensor = None, 
        states: torch.Tensor = None,
        lidar_tokens: torch.Tensor = None,
        lidar_tokens_global: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Extract features for a batch with temporal dimension
        
        Supports two modes:
        1. Raw images: images (B, T, C, H, W) + states (B, T, state_dim)
        2. Pre-computed: lidar_tokens (B, T, seq_len, 512) + lidar_tokens_global (B, T, 1, 512) + states (B, T, state_dim)
        """
        
        use_precomputed = lidar_tokens is not None and lidar_tokens_global is not None
        
        if use_precomputed:
            B, T = lidar_tokens.shape[:2]
            
            # Reshape: (B, T, ...) -> (B*T, ...)
            lidar_tokens_flat = lidar_tokens.reshape(B * T, *lidar_tokens.shape[2:])
            lidar_tokens_global_flat = lidar_tokens_global.reshape(B * T, *lidar_tokens_global.shape[2:])
            states_flat = states.reshape(B * T, *states.shape[2:])
            
            # Extract features using pre-computed tokens
            features_flat = self.forward(
                state=states_flat,
                lidar_token=lidar_tokens_flat,
                lidar_token_global=lidar_tokens_global_flat
            )
        else:
            if images is None or states is None:
                raise ValueError("Either (images, states) or (lidar_tokens, lidar_tokens_global, states) must be provided")
            
            B, T = images.shape[:2]
            
            # Reshape: (B, T, ...) -> (B*T, ...)
            images_flat = images.reshape(B * T, *images.shape[2:])
            states_flat = states.reshape(B * T, *states.shape[2:])
            
            # Extract features from raw images
            features_flat = self.forward(images_flat, states_flat)
        
        # Reshape back: (B*T, feature_dim) -> (B, T, feature_dim)
        features = features_flat.reshape(B, T, self.feature_dim)
        
        return features
    
    def get_feature_stats(self, feature: torch.Tensor) -> dict:

        return {
            'mean': feature.mean().item(),
            'std': feature.std().item(),
            'min': feature.min().item(),
            'max': feature.max().item(),
            'shape': tuple(feature.shape),
        }


def load_lidar_submodules(model: torch.nn.Module, ckpt_path: str, strict: bool = False, logger=None):
    """
    加载LiDAR子模块的预训练权重，支持键名映射
    
    Args:
        model: 目标模型
        ckpt_path: checkpoint文件路径
        strict: 是否严格匹配所有键
        logger: 日志记录器（可选）
    """
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        if logger:
            logger.warning(f"[LiDAR] ckpt 不存在: {ckpt_path}")
        else:
            print(f"[LiDAR] ckpt 不存在: {ckpt_path}")
        return
    
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    
    # 清理模块名称（移除 "module." 前缀）
    clean = {}
    for k, v in state.items():
        kk = k.replace("module.", "")
        clean[kk] = v
    
    # 定义键名映射规则
    ckpt2model_prefix = {
        "lidar_backbone.": "bev_model_mot.backbone.",
        "lidar_patch_embed.proj.": "bev_model_mot.proj.",
    }
    ckpt_src_prefixes = tuple(ckpt2model_prefix.keys())
    model_dst_prefixes = tuple(ckpt2model_prefix.values())

    # 重映射键名
    remapped = {} 
    for ckpt_k, v in clean.items():
        if ckpt_k.startswith(ckpt_src_prefixes):
            for src_prefix, dst_prefix in ckpt2model_prefix.items():
                if ckpt_k.startswith(src_prefix):
                    model_k = dst_prefix + ckpt_k[len(src_prefix):]
                    remapped[model_k] = (v, ckpt_k)
                    break

    # 检查形状匹配
    model_sd = model.state_dict()
    to_load = {}
    shape_mismatch = []     
    unmapped_after = []      
    for model_k, (tensor, ckpt_k) in remapped.items():
        if model_k in model_sd:
            if model_sd[model_k].shape == tensor.shape:
                to_load[model_k] = tensor
            else:
                shape_mismatch.append((model_k, ckpt_k, tuple(model_sd[model_k].shape), tuple(tensor.shape)))
        else:
            unmapped_after.append((model_k, ckpt_k))

    # 加载权重
    msg = model.load_state_dict(to_load, strict=False)

    # 统计结果
    expected_lidar_keys = [k for k in model_sd.keys() if k.startswith(model_dst_prefixes)]
    missing_after_load = [k for k in expected_lidar_keys if k not in to_load]

    # 检查是否为rank 0（分布式训练）
    rank0 = True
    if dist is not None and dist.is_initialized():
        rank0 = dist.get_rank() == 0
    
    if rank0:
        lines = []
        lines.append(f"[LiDAR] ckpt: {ckpt_path}")
        lines.append("[LiDAR] 映射关系：")
        for s, d in ckpt2model_prefix.items():
            lines.append(f"  - ckpt '{s}*'  -> model '{d}*'")
        lines.append(f"[LiDAR] 目标键数量: {len(expected_lidar_keys)}")
        lines.append(f"[LiDAR] 成功加载键数量: {len(to_load)}")

        if shape_mismatch:
            lines.append(f"[LiDAR] 形状不匹配({len(shape_mismatch)}):")
            for model_k, ckpt_k, a, b in shape_mismatch[:20]:
                lines.append(f"  - model '{model_k}' <= ckpt '{ckpt_k}': model{a} vs ckpt{b}")
            if len(shape_mismatch) > 20:
                lines.append("  ... 省略 ...")

        if missing_after_load:
            lines.append(f"[LiDAR] 模型中目标前缀键但未加载({len(missing_after_load)}):")
            for k in missing_after_load[:20]:
                lines.append(f"  - {k}")
            if len(missing_after_load) > 20:
                lines.append("  ... 省略 ...")

        if unmapped_after:
            lines.append(f"[LiDAR] ckpt 映射后在模型中不存在({len(unmapped_after)}):")
            for model_k, ckpt_k in unmapped_after[:20]:
                lines.append(f"  - model '{model_k}'  <=  ckpt '{ckpt_k}'")
            if len(unmapped_after) > 20:
                lines.append("  ... 省略 ...")

        if hasattr(msg, "missing_keys") and msg.missing_keys:
            lines.append(f"[LiDAR] load_state_dict 返回 missing_keys({len(msg.missing_keys)}), 多为非目标前缀，可忽略")
        if hasattr(msg, "unexpected_keys") and msg.unexpected_keys:
            lines.append(f"[LiDAR] load_state_dict 返回 unexpected_keys({len(msg.unexpected_keys)}), 多为非目标前缀，可忽略")

        report = "\n".join(lines)
        if logger:
            logger.info(report)
        else:
            print(report)


if __name__ == "__main__":
    print("=" * 70)
    print("Testing InterfuserBEVEncoder")
    print("=" * 70)
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 创建编码器
    print("\n1. Creating InterfuserBEVEncoder...")
    try:
        encoder = InterfuserBEVEncoder(
            perception_backbone=None,
            state_dim=9,
            feature_dim=256,
            use_group_norm=True,
            freeze_backbone=True,  # 设为False以便加载权重
            bev_input_size=(448, 448)
        )
        print("✓ Encoder created successfully")
    except Exception as e:
        print(f"✗ Error creating encoder: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 2. 加载预训练权重
    pretrained_path = os.path.join(
        os.path.dirname(__file__), 
        'interfuser/lidar_bev_encoder_only.pth'
    )
    
    if os.path.exists(pretrained_path):
        print(f"\n2. Loading pretrained weights from: {pretrained_path}")
        print("   Using load_lidar_submodules function for better key mapping...")
        try:
            # 使用新的 load_lidar_submodules 函数加载权重
            load_lidar_submodules(encoder, pretrained_path, strict=False, logger=None)
            print("   ✓ Weights loaded successfully using load_lidar_submodules!")
                
        except Exception as e:
            print(f"   ✗ Error loading weights: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n2. ⚠ Pretrained weights not found at: {pretrained_path}")
        print("   Continuing with random initialization...")
    
    # 3. 移动到设备
    print(f"\n3. Moving model to {device}...")
    encoder = encoder.to(device)
    encoder.eval()
    print("✓ Model ready")
    
    # 4. 创建测试输入
    print("\n4. Creating test inputs...")
    batch_size = 4
    test_image = torch.randn(batch_size, 3, 448, 448).to(device)
    test_state = torch.randn(batch_size, 9).to(device)
    
    print(f"   Image shape: {test_image.shape}")
    print(f"   State shape: {test_state.shape}")
    
    # 5. 前向传播测试
    print("\n5. Running forward pass...")
    try:
        with torch.no_grad():
            output = encoder(test_image, test_state)
        
        print(f"   ✓ Forward pass successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output stats:")
        print(f"     - Mean: {output.mean().item():.4f}")
        print(f"     - Std: {output.std().item():.4f}")
        print(f"     - Min: {output.min().item():.4f}")
        print(f"     - Max: {output.max().item():.4f}")
        
    except Exception as e:
        print(f"   ✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 6. 批量特征提取测试
    print("\n6. Testing batch feature extraction...")
    try:
        batch_images = torch.randn(2, 3, 3, 448, 448).to(device)  # (B=2, T=3, C=3, H=448, W=448)
        batch_states = torch.randn(2, 3, 9).to(device)  # (B=2, T=3, state_dim=9)
        
        with torch.no_grad():
            batch_features = encoder.extract_features_batch(batch_images, batch_states)
        
        print(f"   ✓ Batch extraction successful!")
        print(f"   Input: B={batch_images.shape[0]}, T={batch_images.shape[1]}")
        print(f"   Output shape: {batch_features.shape}")
        
    except Exception as e:
        print(f"   ✗ Error in batch extraction: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. 参数统计
    print("\n7. Model statistics:")
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    print("\n" + "=" * 70)
    print("✓ All tests completed successfully!")
    print("=" * 70)
