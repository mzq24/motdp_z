import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Callable, Union
from collections import defaultdict
import numpy as np
import pickle
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from model.transformer_for_diffusion_multi_head import TransformerForDiffusion
import os



def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result




class DiffusionDiTCarlaPolicy(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        
        # config
        self.cfg = config
        policy_cfg = config['policy']

        obs_as_global_cond = policy_cfg.get('obs_as_global_cond', True)
        self.obs_as_global_cond = obs_as_global_cond
        shape_meta = config['shape_meta']
        action_shape = shape_meta['action']['shape']
        action_dim = action_shape[0]
        
        # Action normalization settings
        self.enable_action_normalization = config.get('enable_action_normalization', True)
        

        self.n_obs_steps = policy_cfg.get('n_obs_steps', config.get('obs_horizon', 1))
        
        # Transfuser feature dimensions (based on backbone output)
        # Following DiffusionDriveV2: only use bev_feature and bev_feature_upsample
        # bev_feature: (1512, 8, 8), bev_feature_upsample: (64, 64, 64)
        transfuser_cfg = config.get('transfuser_encoder', {})
        self.bev_feature_dim = transfuser_cfg.get('bev_feature_dim', 1512)
        self.bev_feature_upsample_dim = transfuser_cfg.get('bev_feature_upsample_dim', 64)

        # ========== Load Anchor Centers from wp_tokens.pkl ==========
        anchor_path = config.get('anchor_path', 'wp_tokens.pkl')
        self.num_modes = config.get('num_modes', 32)  # Number of anchor modes
        self._load_anchor_centers(anchor_path)

        obs_feature_dim = 256  

        # Get status_dim from config
        status_dim = config.get('bev_encoder', {}).get('state_dim', 15)
        
        # Get ego_status_seq_len from policy config (defaults to n_obs_steps)
        ego_status_seq_len = policy_cfg.get('ego_status_seq_len', self.n_obs_steps)
        
        # Number of waypoints for route prediction
        num_waypoints = policy_cfg.get('num_waypoints', 20)
        self.num_waypoints = num_waypoints

        model = TransformerForDiffusion(
            input_dim=policy_cfg.get('input_dim', 2),
            output_dim=policy_cfg.get('output_dim', 2),
            horizon=policy_cfg.get('horizon', 16),
            n_obs_steps=self.n_obs_steps,  
            cond_dim=256,   
            n_layer=policy_cfg.get('n_layer', 8),
            n_head=policy_cfg.get('n_head', 8),
            n_emb=policy_cfg.get('n_emb', 512),
            p_drop_emb=policy_cfg.get('p_drop_emb', 0.1),
            p_drop_attn=policy_cfg.get('p_drop_attn', 0.1),
            causal_attn=policy_cfg.get('causal_attn', True),
            obs_as_cond=obs_as_global_cond,
            n_cond_layers=policy_cfg.get('n_cond_layers', 4),
            status_dim=status_dim,
            ego_status_seq_len=ego_status_seq_len,
            # Transfuser feature dimensions (from transfuser backbone)
            # Following DiffusionDriveV2: only use bev_feature and bev_feature_upsample
            # bev_feature: (1512, 8, 8), bev_feature_upsample: (64, 64, 64)
            transfuser_bev_dim=self.bev_feature_dim,
            transfuser_bev_upsample_dim=self.bev_feature_upsample_dim,
            num_waypoints=num_waypoints,  # Number of route waypoints
            num_modes=self.num_modes,  # Number of anchor modes for multimodal prediction
        )

        self.model = model
        
        # ========== Truncated Diffusion Configuration (DiffusionDriveV2 style) ==========
        diffusion_cfg = config.get('truncated_diffusion', {})
        self.num_train_timesteps = diffusion_cfg.get('num_train_timesteps', 1000)
        self.trunc_timesteps = diffusion_cfg.get('trunc_timesteps', 8)  # Truncated timestep for anchor during inference
        self.train_trunc_timesteps = diffusion_cfg.get('train_trunc_timesteps', 50)  # Max timestep during training (DiffusionDrive uses 50)
        self.num_diffusion_steps = diffusion_cfg.get('num_diffusion_steps', 2)  # Number of denoising steps
        self.diffusion_eta = diffusion_cfg.get('eta', 1.0)  # 1.0 for stochastic multiplicative noise
        
        # Normalization parameters for DELTA (per-step displacement)
        # Based on anchor statistics: dx [-0.31, 11.13], dy [-9.84, 7.88]
        # Formula: 2*(x + offset)/range - 1
        self.norm_delta_x_offset = diffusion_cfg.get('norm_delta_x_offset', 1.0)   # maps [-1, 13] to [-1, 1]
        self.norm_delta_x_range = diffusion_cfg.get('norm_delta_x_range', 14.0)
        self.norm_delta_y_offset = diffusion_cfg.get('norm_delta_y_offset', 10.0)  # maps [-10, 10] to [-1, 1]
        self.norm_delta_y_range = diffusion_cfg.get('norm_delta_y_range', 20.0)

        # Keep old params for absolute coords (used for anchor matching)
        self.norm_x_offset = diffusion_cfg.get('norm_x_offset', 2.0)  # x range: [-2, 78]
        self.norm_x_range = diffusion_cfg.get('norm_x_range', 80.0)
        self.norm_y_offset = diffusion_cfg.get('norm_y_offset', 20.0)  # y range: [-20, 36]
        self.norm_y_range = diffusion_cfg.get('norm_y_range', 56.0)
        
        # Route prediction auxiliary loss weight (横向控制重要性)
        self.route_loss_weight = diffusion_cfg.get('route_loss_weight', 0.5)
        
        # DiffusionDrive-style multimodal loss weights
        self.cls_loss_weight = config.get('cls_loss_weight', 0.5)
        self.reg_loss_weight = config.get('reg_loss_weight', 1.0)
        
        # DDIMScheduler for variance computation (DiffusionDriveV2 style)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            steps_offset=1,
            beta_schedule="scaled_linear",
            prediction_type="sample",  # Predict clean sample directly
        )

        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.horizon = policy_cfg.get('horizon', 16)
        self.n_action_steps = policy_cfg.get('action_horizon', 8)

    def _cumulate_trajectory(self, traj_deltas: torch.Tensor) -> torch.Tensor:
        """
        Convert per-step (dx, dy) deltas into absolute trajectory by cumulative sum.

        Args:
            traj_deltas: (..., T, 2) trajectory deltas

        Returns:
            (..., T, 2) cumulative trajectory
        """
        return torch.cumsum(traj_deltas, dim=-2)
    
    # ========== Normalization Functions ==========
    def norm_odo(self, odo_info_fut: torch.Tensor) -> torch.Tensor:
        """
        Normalize trajectory coordinates to [-1, 1] range.
        Following DiffusionDrive v1: 2*(x + offset)/range - 1
        
        For our data (x: [-0.066, 74.045], y: [-17.526, 32.736]):
        - x: 2*(x + 1)/76 - 1, maps [-1, 75] to [-1, 1]
        - y: 2*(y + 18)/52 - 1, maps [-18, 34] to [-1, 1]
        """
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        
        # Linear mapping to [-1, 1]
        odo_info_fut_x = 2 * (odo_info_fut_x + self.norm_x_offset) / self.norm_x_range - 1
        odo_info_fut_y = 2 * (odo_info_fut_y + self.norm_y_offset) / self.norm_y_range - 1
        
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
    
    def denorm_odo(self, odo_info_fut: torch.Tensor) -> torch.Tensor:
        """
        Denormalize trajectory from [-1, 1] back to original scale.
        Following DiffusionDrive v1: (x + 1)/2 * range - offset
        """
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]

        # Inverse linear mapping from [-1, 1]
        odo_info_fut_x = (odo_info_fut_x + 1) / 2 * self.norm_x_range - self.norm_x_offset
        odo_info_fut_y = (odo_info_fut_y + 1) / 2 * self.norm_y_range - self.norm_y_offset

        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)

    def norm_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Normalize per-step delta (displacement) to [-1, 1] range.
        Based on anchor statistics: dx [-0.31, 11.13], dy [-9.84, 7.88]
        """
        delta_x = delta[..., 0:1]
        delta_y = delta[..., 1:2]

        # Linear mapping to [-1, 1]
        delta_x = 2 * (delta_x + self.norm_delta_x_offset) / self.norm_delta_x_range - 1
        delta_y = 2 * (delta_y + self.norm_delta_y_offset) / self.norm_delta_y_range - 1

        return torch.cat([delta_x, delta_y], dim=-1)

    def denorm_delta(self, delta_normed: torch.Tensor) -> torch.Tensor:
        """
        Denormalize delta from [-1, 1] back to original scale.
        """
        delta_x = delta_normed[..., 0:1]
        delta_y = delta_normed[..., 1:2]

        # Inverse linear mapping from [-1, 1]
        delta_x = (delta_x + 1) / 2 * self.norm_delta_x_range - self.norm_delta_x_offset
        delta_y = (delta_y + 1) / 2 * self.norm_delta_y_range - self.norm_delta_y_offset

        return torch.cat([delta_x, delta_y], dim=-1)

    def _traj_to_delta(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Convert absolute trajectory to per-step deltas.
        delta[0] = pos[0], delta[i] = pos[i] - pos[i-1]
        """
        delta = torch.zeros_like(trajectory)
        delta[..., 0, :] = trajectory[..., 0, :]  # First point is absolute (from origin)
        delta[..., 1:, :] = trajectory[..., 1:, :] - trajectory[..., :-1, :]
        return delta

    def _load_anchor_centers(self, anchor_path: str):
        """
        Load anchor centers from wp_tokens.pkl file.

        The file contains:
        - centers: (num_modes, num_points, 2) - cluster centers as trajectories
        - labels: (N,) - cluster labels for each sample
        - centers_flat: (num_modes, num_points*2) - flattened centers

        We store two versions:
        - anchor_centers: per-step deltas (for model input, cumsum to get trajectory)
        - anchor_centers_abs: absolute coordinates (for anchor matching with GT)
        """
        if os.path.exists(anchor_path):
            with open(anchor_path, 'rb') as f:
                data = pickle.load(f)

            centers = data['centers']  # (32, 5, 2) - absolute coordinates
            self.anchor_num_points = centers.shape[1]  # 5 waypoints per anchor

            # Convert to per-step deltas: delta[0] = pos[0], delta[i] = pos[i] - pos[i-1]
            centers_tensor = torch.from_numpy(centers).float()
            centers_delta = torch.zeros_like(centers_tensor)
            centers_delta[:, 0, :] = centers_tensor[:, 0, :]  # First point is absolute
            centers_delta[:, 1:, :] = centers_tensor[:, 1:, :] - centers_tensor[:, :-1, :]  # Subsequent are deltas

            # Register as buffers (not trainable, but moves with model)
            self.register_buffer('anchor_centers', centers_delta)  # Per-step deltas for model input
            self.register_buffer('anchor_centers_abs', centers_tensor)  # Absolute coords for matching

            print(f"[DiffusionDiTCarlaPolicy] Loaded {self.num_modes} anchor centers from {anchor_path}")
            print(f"  - Shape: {centers.shape} (num_modes, num_points, 2)")
            print(f"  - Converted to per-step deltas for model input")
        else:
            print(f"[Warning] Anchor file not found: {anchor_path}, using default initialization")
            # Initialize with zeros - should be loaded later
            self.anchor_num_points = 5
            self.register_buffer('anchor_centers', torch.zeros(self.num_modes, self.anchor_num_points, 2))
            self.register_buffer('anchor_centers_abs', torch.zeros(self.num_modes, self.anchor_num_points, 2))
    
    def get_best_anchor_idx(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Find the closest anchor center for each trajectory in the batch.
        Uses anchor_centers_abs (absolute coordinates) for distance computation.

        Args:
            trajectory: (B, T, 2) - ground truth trajectory (absolute coordinates)

        Returns:
            best_idx: (B,) - index of closest anchor for each sample
        """
        B = trajectory.shape[0]
        T = trajectory.shape[1]

        # Sample trajectory at anchor waypoint positions
        # anchor_centers_abs: (num_modes, 5, 2), trajectory: (B, T, 2)
        # We need to interpolate trajectory to match anchor's 5 points
        if T != self.anchor_num_points:
            # Linearly interpolate trajectory to anchor_num_points
            indices = torch.linspace(0, T-1, self.anchor_num_points).long()
            traj_sampled = trajectory[:, indices, :]  # (B, anchor_num_points, 2)
        else:
            traj_sampled = trajectory

        # Compute L2 distance between trajectory and each anchor (use absolute coords)
        # traj_sampled: (B, 5, 2), anchor_centers_abs: (32, 5, 2)
        # Expand for broadcasting: (B, 1, 5, 2) - (1, 32, 5, 2) -> (B, 32, 5, 2)
        traj_expanded = traj_sampled.unsqueeze(1)  # (B, 1, 5, 2)
        anchor_expanded = self.anchor_centers_abs.unsqueeze(0)  # (1, 32, 5, 2)

        # L2 distance per point, then mean over points
        dist = torch.norm(traj_expanded - anchor_expanded, dim=-1)  # (B, 32, 5)
        dist = dist.mean(dim=-1)  # (B, 32)

        # Find closest anchor
        best_idx = torch.argmin(dist, dim=-1)  # (B,)

        return best_idx
    
    def get_anchor_for_sample(self, batch_size: int, device: torch.device, dtype: torch.dtype, 
                               mode_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get anchor trajectories for the batch, optionally using specific mode indices.
        
        Args:
            batch_size: number of samples
            device: target device
            dtype: target dtype
            mode_idx: (B,) optional - specific mode indices to use
            
        Returns:
            anchors: (B, anchor_num_points, 2) - anchor trajectories
        """
        if mode_idx is not None:
            # Use specified mode indices
            anchors = self.anchor_centers[mode_idx]  # (B, 5, 2)
        else:
            # Use all modes (for inference - return all anchors)
            anchors = self.anchor_centers.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, 32, 5, 2)
        
        return anchors.to(device=device, dtype=dtype)

    def add_multiplicative_noise_scheduled(
        self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, int],
        eta: float = 1.0,
        std_min: float = 0.04
    ) -> torch.Tensor:
        """
        Add multiplicative noise with scheduler-based variance (DiffusionDriveV2 style).
        The noise level is determined by the diffusion scheduler's variance at the given timestep.
        
        DiffusionDriveV2 formula:
            prev_sample = prev_sample_mean * variance_noise_mul + std_dev_t_add * variance_noise_add
        
        When eta > 0:
            - std_dev_t_mul = clip(std_dev_t, min=0.04) for multiplicative noise
            - std_dev_t_add = 0.0 (no additive noise)
        
        Multiplicative noise is applied separately to x (horizon) and y (vert) directions,
        then combined: sample * noise_mul
        
        Args:
            sample: (B, T, 2) normalized trajectory
            timestep: current diffusion timestep (scalar or tensor)
            eta: scaling factor for variance (0.0 = deterministic, 1.0 = full stochasticity)
            std_min: minimum standard deviation to prevent zero noise (V2 uses 0.04)
            
        Returns:
            Noisy sample with timestep-scheduled multiplicative noise applied
        """
        device = sample.device
        dtype = sample.dtype
        bs = sample.shape[0]
        T = sample.shape[1]  # trajectory length (num_points)
        
        # Get timestep as integer
        if torch.is_tensor(timestep):
            t = timestep.item() if timestep.numel() == 1 else timestep[0].item()
        else:
            t = timestep
        t = int(t)
        
        # Compute variance from scheduler (DDIM style)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        prev_t = t - self.num_train_timesteps // max(self.num_diffusion_steps, 1)
        prev_t = max(prev_t, 0)
        
        alpha_prod_t = self.diffusion_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.diffusion_scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.diffusion_scheduler.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Variance formula from DDIM
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        variance = max(variance.item(), 1e-10)
        
        # std_dev_t with eta scaling
        std_dev_t = eta * (variance ** 0.5)
        
        # DiffusionDriveV2 style: std_dev_t_mul = clip(std_dev_t, min=0.04)
        std_dev_t_mul = max(std_dev_t, std_min)
        
        # Generate multiplicative noise for horizon (x) and vert (y) separately
        # DiffusionDriveV2: variance_noise_horizon/vert shape is (B, G, 1, 1), then repeat
        # Our shape: (B, 1, 1) for horizon and vert, then cat to (B, 1, 2), then repeat to (B, T, 2)
        
        # variance_noise_horizon = randn * std_dev_t_mul + 1.0  (for x direction)
        variance_noise_horizon = torch.randn([bs, 1, 1], device=device, dtype=dtype) * std_dev_t_mul + 1.0
        # variance_noise_vert = randn * std_dev_t_mul + 1.0  (for y direction)
        variance_noise_vert = torch.randn([bs, 1, 1], device=device, dtype=dtype) * std_dev_t_mul + 1.0
        
        # Concatenate horizon and vert: (B, 1, 1) + (B, 1, 1) -> (B, 1, 2)
        variance_noise_mul = torch.cat([variance_noise_horizon, variance_noise_vert], dim=-1)
        
        # Repeat across trajectory length: (B, 1, 2) -> (B, T, 2)
        variance_noise_mul = variance_noise_mul.expand(-1, T, -1)
        
        # Apply multiplicative noise: sample * variance_noise_mul
        # This matches DiffusionDriveV2: prev_sample = prev_sample_mean * variance_noise_mul
        # (when std_dev_t_add = 0, the additive term is zero)
        noisy_sample = sample * variance_noise_mul
        
        return noisy_sample

    def add_multiplicative_noise_scheduled_batch(
        self, 
        sample: torch.Tensor, 
        timesteps: torch.Tensor,
        eta: float = 1.0,
        std_min: float = 0.04
    ) -> torch.Tensor:
        """
        Add multiplicative noise with per-sample timesteps (batch version).
        Each sample in the batch gets noise corresponding to its own timestep.
        
        This is the correct implementation for training where each sample should have
        noise added according to its own sampled timestep.
        
        Args:
            sample: (B, T, 2) normalized trajectory
            timesteps: (B,) tensor of timesteps, one per sample
            eta: scaling factor for variance (0.0 = deterministic, 1.0 = full stochasticity)
            std_min: minimum standard deviation to prevent zero noise (V2 uses 0.04)
            
        Returns:
            Noisy sample with per-sample timestep-scheduled multiplicative noise applied
        """
        device = sample.device
        dtype = sample.dtype
        bs = sample.shape[0]
        T = sample.shape[1]  # trajectory length (num_points)
        
        # Compute variance for each sample based on its timestep
        # Pre-compute alpha_cumprod values on CPU then move to device
        alphas_cumprod = self.diffusion_scheduler.alphas_cumprod
        
        # Get prev_t for each sample
        step_ratio = self.num_train_timesteps // max(self.num_diffusion_steps, 1)
        prev_timesteps = (timesteps - step_ratio).clamp(min=0)
        
        # Gather alpha_prod values for each sample
        alpha_prod_t = alphas_cumprod[timesteps.cpu()].to(device=device, dtype=dtype)  # (B,)
        alpha_prod_t_prev = alphas_cumprod[prev_timesteps.cpu()].to(device=device, dtype=dtype)  # (B,)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Variance formula from DDIM: (B,)
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        variance = variance.clamp(min=1e-10)
        
        # std_dev_t with eta scaling: (B,)
        std_dev_t = eta * (variance ** 0.5)
        
        # DiffusionDriveV2 style: std_dev_t_mul = clip(std_dev_t, min=0.04)
        std_dev_t_mul = std_dev_t.clamp(min=std_min)  # (B,)
        
        # Reshape for broadcasting: (B,) -> (B, 1, 1)
        std_dev_t_mul = std_dev_t_mul.view(bs, 1, 1)
        
        # Generate multiplicative noise for horizon (x) and vert (y) separately
        # variance_noise_horizon = randn * std_dev_t_mul + 1.0  (for x direction)
        variance_noise_horizon = torch.randn([bs, 1, 1], device=device, dtype=dtype) * std_dev_t_mul + 1.0
        # variance_noise_vert = randn * std_dev_t_mul + 1.0  (for y direction)
        variance_noise_vert = torch.randn([bs, 1, 1], device=device, dtype=dtype) * std_dev_t_mul + 1.0
        
        # Concatenate horizon and vert: (B, 1, 1) + (B, 1, 1) -> (B, 1, 2)
        variance_noise_mul = torch.cat([variance_noise_horizon, variance_noise_vert], dim=-1)
        
        # Repeat across trajectory length: (B, 1, 2) -> (B, T, 2)
        variance_noise_mul = variance_noise_mul.expand(-1, T, -1)
        
        # Apply multiplicative noise: sample * variance_noise_mul
        noisy_sample = sample * variance_noise_mul
        
        return noisy_sample

    def forward(self, batch: Dict[str, torch.Tensor], return_loss_dict: bool = False):
        """
        Forward method for DDP compatibility.
        DDP only synchronizes gradients when forward() is called, not for other methods.
        This method simply calls compute_loss() to enable proper gradient synchronization
        in distributed training.

        Args:
            batch: input batch dict
            return_loss_dict: if True, return dict with all losses; if False, return total_loss only

        Returns:
            If return_loss_dict=False: total_loss tensor (for backward)
            If return_loss_dict=True: dict with total_loss, cls_loss, reg_loss, route_loss, etc.
        """
        loss_dict = self.compute_loss(batch)
        if return_loss_dict:
            return loss_dict
        else:
            return loss_dict['total_loss']

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        DiffusionDrive-style multimodal loss computation.
        
        batch: {
            # Transfuser features (single frame, no temporal)
            'transfuser_bev_feature': (B, 1512, 8, 8) - BEV feature
            'transfuser_bev_feature_upsample': (B, 64, 64, 64) - Upscaled BEV feature
            
            'agent_pos': (B, horizon, 2) - 未来轨迹点 (GT trajectory)
            'ego_status': (B, obs_horizon, state_dim) - 车辆状态
            'route': (B, num_waypoints, 2) - 路线waypoints（可选，用于route预测辅助任务）
        }
        
        The model predicts:
        - poses_reg: (B, num_modes, horizon, 2) - trajectory regression for each mode
        - poses_cls: (B, num_modes) - classification scores for each mode
        - route_pred: (B, num_waypoints, 2) - route prediction
        
        Loss:
        - Classification loss: focal loss to select the best matching anchor
        - Regression loss: L1 loss on the best matching mode's prediction
        - Route loss: L1 loss on route prediction (optional)
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        raw_agent_pos = batch['agent_pos'].to(device)
        batch_size = raw_agent_pos.shape[0]
        horizon = raw_agent_pos.shape[1]
        
        # Get ground truth trajectory
        trajectory = raw_agent_pos.to(dtype=model_dtype)  # (B, horizon, 2)
        
        # Get route ground truth for auxiliary task
        route_gt = batch.get('route', None)
        if route_gt is not None:
            route_gt = route_gt.to(device=device, dtype=model_dtype)  # (B, num_waypoints, 2)
        
        # Load transfuser features (single frame, no temporal)
        transfuser_bev_feature = batch['transfuser_bev_feature'].to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = batch['transfuser_bev_feature_upsample'].to(device=device, dtype=model_dtype)
        
        # Get ego_status
        ego_status = batch['ego_status'].to(device=device, dtype=model_dtype)

        # ========== Compute Multimodal Loss (DiffusionDrive style) ==========
        loss_dict = self._compute_multimodal_loss(
            trajectory=trajectory,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
            route_gt=route_gt,
            device=device,
            model_dtype=model_dtype
        )

        return loss_dict
    
    def _compute_multimodal_loss(
        self,
        trajectory: torch.Tensor,
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        ego_status: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype,
        route_gt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute DiffusionDrive-style multimodal loss with truncated diffusion.

        Truncated Diffusion Training Flow:
        1. Sample timestep from truncated range [0, train_trunc_timesteps)
        2. Normalize anchors, add multiplicative noise, then denormalize
        3. Model predicts in ORIGINAL delta scale (not normalized!)
        4. Loss = focal_cls (select best mode) + L1_reg (in original delta scale)

        Key insight: Normalization is ONLY for noise addition (to ensure proper noise scale).
        Model predicts and loss is computed in original delta scale to match route loss scale.

        Args:
            trajectory: (B, horizon, 2) - ground truth trajectory (clean)
            transfuser_bev_feature: (B, 1512, 8, 8) - BEV feature
            transfuser_bev_feature_upsample: (B, 64, 64, 64) - Upscaled BEV feature
            ego_status: (B, To, status_dim) - ego vehicle status
            device: torch device
            model_dtype: model dtype (e.g., bfloat16)
            route_gt: (B, num_waypoints, 2) - optional ground truth route for auxiliary loss
        """
        batch_size = trajectory.shape[0]
        horizon = trajectory.shape[1]

        # ========== Convert GT trajectory to delta (original scale, NOT normalized) ==========
        # Model predicts in original delta scale for proper loss weighting
        trajectory_delta = self._traj_to_delta(trajectory)  # (B, horizon, 2) delta in original scale

        # ========== Sample timestep ==========
        # Truncated range [0, train_trunc_timesteps) instead of [0, 1000)
        timesteps = torch.randint(
            0, self.train_trunc_timesteps,
            (batch_size,), device=device
        ).long()

        # ========== Prepare anchors (delta) and add noise ==========
        # anchor_centers is already in delta form (original scale)
        all_anchors_delta = self.anchor_centers.unsqueeze(0).expand(batch_size, -1, -1, -1)
        all_anchors_delta = all_anchors_delta.to(device=device, dtype=model_dtype)

        # Normalize delta to [-1, 1] for noise addition only
        all_anchors_delta_normed = self.norm_delta(all_anchors_delta)  # (B, M, T_anchor, 2)

        # ========== Add noise to normalized delta anchors (Truncated Diffusion) ==========
        B, M, T_anchor, D = all_anchors_delta_normed.shape
        anchors_flat = all_anchors_delta_normed.contiguous().view(B * M, T_anchor, D)

        # Expand timesteps for all modes: (B,) -> (B*M,)
        timesteps_expanded = timesteps.unsqueeze(1).expand(-1, M).reshape(B * M)

        # Add multiplicative noise to normalized delta
        noisy_anchors_flat = self.add_multiplicative_noise_scheduled_batch(
            anchors_flat, timesteps_expanded, eta=self.diffusion_eta
        )

        # Reshape back to (B, M, T_anchor, 2)
        noisy_anchors_delta_normed = noisy_anchors_flat.view(B, M, T_anchor, D)

        # ========== CRITICAL: Denormalize after noise addition ==========
        # Model operates in original delta scale, not normalized space
        noisy_anchors_delta = self.denorm_delta(noisy_anchors_delta_normed)  # (B, M, T_anchor, 2)

        # ========== Forward pass with noisy delta anchors (original scale) ==========
        # Model receives noisy anchors (original delta scale) and predicts clean delta
        # poses_reg: (B, num_modes, horizon, 2) - original delta scale
        poses_reg, poses_cls, route_pred = self.model(
            anchors=noisy_anchors_delta,  # Noisy anchors (original delta scale)
            timestep=timesteps,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status
        )

        # poses_reg is in original delta scale (model output = residual + anchor)

        # ========== Find best matching anchor (in absolute space) ==========
        # Use anchor_centers_abs (absolute coordinates) for distance computation
        # Ensure anchor_num_points == horizon (no interpolation for delta prediction)
        assert horizon == self.anchor_num_points, \
            f"horizon ({horizon}) must equal anchor_num_points ({self.anchor_num_points}). " \
            f"Interpolating deltas is incorrect - ensure config aligns these values."

        all_anchors_abs = self.anchor_centers_abs.unsqueeze(0).expand(batch_size, -1, -1, -1)
        all_anchors_abs = all_anchors_abs.to(device=device, dtype=model_dtype)

        # Compute L2 distance in absolute space: (B, num_modes)
        # Use absolute trajectory for anchor matching (more intuitive)
        traj_expanded = trajectory.unsqueeze(1)  # (B, 1, horizon, 2)
        dist = torch.norm(traj_expanded - all_anchors_abs, dim=-1)  # (B, num_modes, horizon)
        dist = dist.mean(dim=-1)  # (B, num_modes)

        # Best mode index
        mode_idx = torch.argmin(dist, dim=-1)  # (B,)

        # ========== Classification Loss (Focal Loss) ==========
        # Create one-hot target
        target_onehot = torch.zeros(batch_size, self.num_modes, device=device, dtype=model_dtype)
        target_onehot.scatter_(1, mode_idx.unsqueeze(1), 1)

        # Focal loss
        loss_cls = self._focal_loss(poses_cls, target_onehot)

        # ========== Regression Loss (L1 on best mode in ORIGINAL delta scale) ==========
        # Gather best mode predictions: (B, horizon, 2)
        mode_idx_expanded = mode_idx.view(batch_size, 1, 1, 1).expand(-1, 1, horizon, 2)
        best_reg = torch.gather(poses_reg, 1, mode_idx_expanded).squeeze(1)  # (B, horizon, 2)

        # L1 loss in ORIGINAL delta scale (same scale as route loss)
        loss_reg = F.l1_loss(best_reg, trajectory_delta, reduction='mean')

        # ========== Route Loss (Optional) ==========
        total_loss = self.cls_loss_weight * loss_cls + self.reg_loss_weight * loss_reg

        route_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        if route_gt is not None and route_pred is not None:
            route_loss = F.l1_loss(route_pred, route_gt, reduction='mean')
            total_loss = total_loss + self.route_loss_weight * route_loss

        # Return dict with all losses for logging
        loss_dict = {
            'total_loss': total_loss,
            'cls_loss': loss_cls,
            'reg_loss': loss_reg,
            'route_loss': route_loss,
            # Weighted losses (for debugging loss scale)
            'cls_loss_weighted': self.cls_loss_weight * loss_cls,
            'reg_loss_weighted': self.reg_loss_weight * loss_reg,
            'route_loss_weighted': self.route_loss_weight * route_loss,
        }

        return loss_dict
    
    def _focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0,
        alpha: float = 0.25
    ) -> torch.Tensor:
        """
        Compute focal loss for classification.
        
        Args:
            pred: (B, num_modes) - predicted logits
            target: (B, num_modes) - one-hot target
            gamma: focusing parameter
            alpha: balancing parameter
        """
        pred_sigmoid = pred.sigmoid()
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        return loss.mean()

    def _add_noise_to_anchors_normed(
        self,
        anchors_normed: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add multiplicative noise to normalized anchor trajectories for truncated diffusion.

        Args:
            anchors_normed: (B, num_modes, anchor_num_points, 2) - normalized anchors in absolute coords
            timesteps: (B,) - timesteps for noise level

        Returns:
            noisy_anchors: (B, num_modes, anchor_num_points, 2) - noisy anchors (normalized, absolute)
        """
        B, M, T_anchor, D = anchors_normed.shape

        # Flatten for noise addition: (B, M, T_anchor, 2) -> (B*M, T_anchor, 2)
        anchors_flat = anchors_normed.contiguous().view(B * M, T_anchor, D)

        # Expand timesteps for all modes: (B,) -> (B*M,)
        timesteps_expanded = timesteps.unsqueeze(1).expand(-1, M).reshape(B * M)

        # Add multiplicative noise to normalized anchors
        noisy_anchors_flat = self.add_multiplicative_noise_scheduled_batch(
            anchors_flat, timesteps_expanded, eta=self.diffusion_eta
        )

        # Reshape back
        return noisy_anchors_flat.view(B, M, T_anchor, D)

    def conditional_sample(self,
            transfuser_bev_feature: torch.Tensor,
            transfuser_bev_feature_upsample: torch.Tensor,
            ego_status: torch.Tensor,
            device: torch.device,
            model_dtype: torch.dtype,
            generator=None,
            num_denoise_steps: Optional[int] = None,
            no_noise: bool = False,
            **kwargs
            ):
        """
        Generate trajectory samples using multimodal prediction with truncated diffusion.

        Truncated Diffusion Inference (Single-step, matching training):
        1. Start from anchors + noise (at truncated timestep, e.g., t=8)
        2. Model directly predicts clean delta in one step
        3. Select best mode based on classification scores

        Note: Multi-step denoising with multiplicative noise is complex.
        For simplicity, we use single-step inference which matches the training objective.

        Args:
            transfuser_bev_feature: (B, 1512, 8, 8) - BEV feature
            transfuser_bev_feature_upsample: (B, 64, 64, 64) - Upscaled BEV feature
            ego_status: (B, To, status_dim) - ego status history
            num_denoise_steps: ignored, always use single-step for multiplicative noise
            no_noise: if True, skip noise addition (for debugging model capability)

        Returns:
            (trajectory, route_pred) tuple - trajectory (B, T, 2), route_pred (B, 20, 2)
        """
        bs = transfuser_bev_feature.shape[0]

        # Get all anchor centers (per-step deltas) in original scale
        all_anchors_delta = self.anchor_centers.unsqueeze(0).expand(bs, -1, -1, -1)
        all_anchors_delta = all_anchors_delta.to(device=device, dtype=model_dtype)  # (B, M, T_anchor, 2)

        # ========== Single-step Truncated Diffusion Inference ==========
        # Use truncated timestep (small noise level, e.g., t=8)
        timesteps = torch.full((bs,), self.trunc_timesteps, dtype=torch.long, device=device)

        if no_noise:
            # Debug mode: no noise, just use clean anchors
            noisy_anchors_delta = all_anchors_delta
        else:
            # Normalize delta for noise addition
            all_anchors_delta_normed = self.norm_delta(all_anchors_delta)  # (B, M, T_anchor, 2)

            # Add noise to normalized delta anchors (only once!)
            noisy_anchors_normed = self._add_noise_to_anchors_normed(all_anchors_delta_normed, timesteps)

            # Denormalize after noise addition - model operates in original scale
            noisy_anchors_delta = self.denorm_delta(noisy_anchors_normed)  # (B, M, T_anchor, 2)

        # Forward pass: model directly predicts clean delta (original scale)
        poses_reg, poses_cls, route_pred = self.model(
            anchors=noisy_anchors_delta,  # Noisy anchors (original delta scale)
            timestep=timesteps,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status
        )

        # poses_reg is in original delta scale (B, num_modes, horizon, 2)

        # Select best mode based on classification scores
        best_mode_idx = torch.argmax(poses_cls, dim=-1)  # (B,)

        # Gather best mode trajectory (in original delta scale)
        horizon = poses_reg.shape[2]
        mode_idx_expanded = best_mode_idx.view(bs, 1, 1, 1).expand(-1, 1, horizon, 2)
        best_delta = torch.gather(poses_reg, 1, mode_idx_expanded).squeeze(1)  # (B, horizon, 2)

        # Cumsum to get absolute trajectory (already in original scale, no denorm needed)
        best_trajectory = self._cumulate_trajectory(best_delta)  # (B, horizon, 2)

        return best_trajectory, route_pred

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], no_noise: bool = False) -> Dict[str, torch.Tensor]:
        """
        Predict action from observation.

        Args:
            obs_dict: observation dictionary
            no_noise: if True, skip noise addition in inference (for debugging)

        Returns:
            dict with 'action', 'action_pred', 'route_pred'
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        nobs = dict_apply(obs_dict, lambda x: x.to(device))

        value = next(iter(nobs.values()))
        B = value.shape[0]
        Da = self.action_dim

        # Load transfuser features (single frame, no temporal)
        transfuser_bev_feature = nobs['transfuser_bev_feature'].to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = nobs['transfuser_bev_feature_upsample'].to(device=device, dtype=model_dtype)

        # Get ego_status
        ego_status = nobs['ego_status']
        ego_status = ego_status.to(dtype=model_dtype)

        # Generate samples using multimodal prediction
        nsample, route_pred = self.conditional_sample(
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            no_noise=no_noise,
            ego_status=ego_status,
            device=device,
            model_dtype=model_dtype,
        )
        
        naction_pred = nsample[...,:Da]
        
        # Convert to float32 before numpy
        action_pred = naction_pred.detach().float().cpu().numpy()
        action = action_pred
        result = {
            'action': action,
            'action_pred': action_pred,
            'route_pred': route_pred,
        }
        
        return result