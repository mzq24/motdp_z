"""Configuration dataclass for Bridge Baseline (BridgeDrive DDBM replica)."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class BridgeBaselineConfig:
    # BEV feature input dimensions (from MoT-DP Transfuser backbone)
    bev_feature_dim: int = 1512         # (1512, 8, 8)
    bev_feature_upsample_dim: int = 64  # (64, 64, 64)
    bev_spatial_size: int = 64          # spatial resolution of upsampled BEV

    # Transformer dimensions
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_head: int = 8
    tf_num_layers: int = 3
    tf_dropout: float = 0.0

    # Trajectory head
    num_poses: int = 10                 # route[:10] waypoints
    ego_fut_mode: int = 20              # number of anchor modes
    num_decoder_layers: int = 2         # stacked DDBM decoder layers
    plan_anchor_path: str = ""

    # BEV normalization range for GridSampleCrossBEVAttention
    lidar_max_x: float = 32.0
    lidar_max_y: float = 32.0

    # DDBM schedule parameters (VP = Variance Preserving)
    beta_d: float = 2.0
    beta_min: float = 0.1
    step_num: int = 20                  # inference denoising steps (bridge loop)

    # PlanningContextEncoder (v2)
    max_speed: float = 25.0              # velocity normalization divisor
    tp_norm: List[float] = field(default_factory=lambda: [200.0, 50.0])  # target_point normalization

    # Status encoding (v1, kept for backward compat)
    status_dim: int = 14

    # Loss weights
    trajectory_cls_weight: float = 10.0
    trajectory_reg_weight: float = 8.0

    # Learnable agent queries
    num_agent_queries: int = 16

    # Per-waypoint normalization statistics (z-score, shape: [num_poses])
    # Filled by scripts/compute_route_stats.py from the training dataset
    norm_x_mean: List[float] = field(default_factory=lambda: [0.0] * 10)
    norm_x_std:  List[float] = field(default_factory=lambda: [1.0] * 10)
    norm_y_mean: List[float] = field(default_factory=lambda: [0.0] * 10)
    norm_y_std:  List[float] = field(default_factory=lambda: [1.0] * 10)

    # Speed prediction (BridgeDrive Method 1: independent decoder)
    predict_target_speed: bool = True
    target_speed_classes: List[float] = field(
        default_factory=lambda: [0.0, 4.0, 8.0, 10.0, 13.89, 16.0, 17.78, 20.0]
    )
    speed_loss_weight: float = 1.0

    # Trajectory mode: switch GT label from route[:10] to agent_pos[:6]
    predict_traj: bool = False
    traj_anchor_path: str = "/media/z/data/mzq/others/MoT-DP/dd_baseline/anchors/carla_kmeans_20.npy"
    traj_norm_x_mean: List[float] = field(default_factory=lambda: [0.0] * 6)
    traj_norm_x_std:  List[float] = field(default_factory=lambda: [1.0] * 6)
    traj_norm_y_mean: List[float] = field(default_factory=lambda: [0.0] * 6)
    traj_norm_y_std:  List[float] = field(default_factory=lambda: [1.0] * 6)

    # Speed prediction (BridgeDrive Method 2: speed as 11th DDBM waypoint)
    diffusion_speed: bool = False
    diffusion_speed_anchor: float = 8.0   # anchor speed for DDBM (m/s), appended to route anchors
    diffusion_speed_mean: float = 8.0     # z-score mean for speed dimension
    diffusion_speed_std: float = 5.0      # z-score std for speed dimension
