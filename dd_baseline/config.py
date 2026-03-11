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

    # Delta prediction mode: predict per-step displacement instead of absolute trajectory
    # delta[0] = pos[0], delta[i] = pos[i] - pos[i-1], cumsum(delta) = abs_traj
    # When True, BEV grid_sample always uses anchor locations (fixed)
    predict_delta: bool = False

    # Per-step delta Z-score normalization (from data statistics)
    norm_delta_x_mean: float = 1.84
    norm_delta_x_std: float = 2.37
    norm_delta_y_mean: float = -0.07
    norm_delta_y_std: float = 0.90

    # Normalized forward mode (abs branch only): run model forward entirely in normalized [-1,1] space
    # BEV grid_sample receives normalized coords instead of physical coords, decoupling spatial dependency
    use_normalized_forward: bool = False

    # Ablation: fix BEV grid_sample at anchor positions (abs mode only)
    # When True, BEV always samples at clean anchor locations instead of predicted trajectory
    # Used to test if BEV's dynamic spatial feedback is the key factor for multi-step DDIM stability
    fix_bev_at_anchor: bool = False

    # Experiment 4b: dynamic BEV in delta mode
    # When True + predict_delta, decoder converts delta predictions to abs (cumsum)
    # and uses them for BEV sampling between layers (like abs mode does)
    delta_dynamic_bev: bool = False

    # Full diffusion: noise GT (not anchor), train with t ∈ [0, num_train_timesteps)
    # Inference starts from pure noise, DDIM skip-step denoising
    # BEV always at anchor positions (anchor provides spatial prior)
    # Works with both abs and delta modes
    use_full_diffusion: bool = False
