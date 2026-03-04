"""Configuration dataclass for DD Baseline model."""
from dataclasses import dataclass


@dataclass
class DDBaselineConfig:
    # BEV feature input dimensions (from MoT-DP Transfuser backbone)
    bev_feature_dim: int = 1512       # (1512, 8, 8)
    bev_feature_upsample_dim: int = 64  # (64, 64, 64)
    bev_spatial_size: int = 64        # spatial resolution of upsampled BEV

    # Transformer dimensions
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_head: int = 8
    tf_num_layers: int = 3            # standard transformer decoder layers
    tf_dropout: float = 0.0

    # Trajectory head
    num_poses: int = 6                # waypoints per trajectory (matching MoT-DP data)
    ego_fut_mode: int = 20            # number of anchor modes
    num_decoder_layers: int = 2       # stacked diffusion decoder layers
    plan_anchor_path: str = ""

    # BEV normalization range for GridSampleCrossBEVAttention
    lidar_max_x: float = 32.0
    lidar_max_y: float = 32.0

    # Diffusion config
    num_train_timesteps: int = 1000
    trunc_timesteps: int = 50         # training noise range [0, trunc_timesteps)
    num_diffusion_steps: int = 2      # inference denoising steps

    # Status encoding
    status_dim: int = 14              # ego_status feature dim

    # Loss weights
    trajectory_cls_weight: float = 10.0
    trajectory_reg_weight: float = 8.0

    # Learnable agent queries (replacement for agent detection head)
    num_agent_queries: int = 16

    # Normalization parameters for trajectory coordinates -> [-1, 1]
    norm_x_offset: float = 16.0
    norm_x_range: float = 92.0
    norm_y_offset: float = 45.0
    norm_y_range: float = 88.0
