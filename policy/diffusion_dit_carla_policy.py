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
        self.use_vqa_anchor = config.get('use_vqa_anchor', False)
        self._load_anchor_centers(anchor_path)

        obs_feature_dim = 256

        # Semantic behavior configuration
        sem_cfg = config.get('semantic_behavior', {})
        self.semantic_behavior_enabled = sem_cfg.get('enabled', False)
        self.num_behaviors = sem_cfg.get('num_behaviors', 11) if self.semantic_behavior_enabled else 0
        self.allowed_loss_weight = sem_cfg.get('allowed_loss_weight', 0.5)
        self.behavior_loss_weight = sem_cfg.get('behavior_loss_weight', 0.1)

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
            num_behaviors=self.num_behaviors,  # Semantic behavior categories
            traj_can_attend_route=policy_cfg.get('traj_can_attend_route', True),
        )

        self.model = model

        # Behavior prediction heads (auxiliary tasks)
        n_emb = policy_cfg.get('n_emb', 512)
        if self.semantic_behavior_enabled:
            self.allowed_pred_head = nn.Linear(n_emb, 1)
            self.behavior_pred_head = nn.Linear(n_emb, self.num_behaviors)
        
        # ========== Truncated Diffusion Configuration (DiffusionDriveV2 style) ==========
        diffusion_cfg = config.get('truncated_diffusion', {})
        self.num_train_timesteps = diffusion_cfg.get('num_train_timesteps', 1000)
        self.trunc_timesteps = diffusion_cfg.get('trunc_timesteps', 100)
        self.train_trunc_timesteps = diffusion_cfg.get('train_trunc_timesteps', 100)
        self.num_diffusion_steps = diffusion_cfg.get('num_diffusion_steps', 2)
        self.diffusion_eta = diffusion_cfg.get('eta', 0.0)
        self.prediction_type = diffusion_cfg.get('prediction_type', 'sample')  # "sample" or "epsilon"
        
        # Absolute coordinate normalization to [-1, 1] (DiffusionDrive v1 style)
        self.norm_x_offset = diffusion_cfg.get('norm_x_offset', 2.0)
        self.norm_x_range = diffusion_cfg.get('norm_x_range', 80.0)
        self.norm_y_offset = diffusion_cfg.get('norm_y_offset', 20.0)
        self.norm_y_range = diffusion_cfg.get('norm_y_range', 56.0)
        
        # Normalized forward mode: model forward in normalized [-1,1] space
        # BEV grid_sample receives normalized coords, decoupling spatial dependency
        self.use_normalized_forward = diffusion_cfg.get('use_normalized_forward', False)

        # Ablation: fix BEV grid_sample at clean anchor positions (disable dynamic spatial feedback)
        self.fix_bev_at_anchor = diffusion_cfg.get('fix_bev_at_anchor', False)

        # Route prediction auxiliary loss weight (横向控制重要性)
        self.route_loss_weight = diffusion_cfg.get('route_loss_weight', 0.5)
        
        # DiffusionDrive-style multimodal loss weights
        self.cls_loss_weight = config.get('cls_loss_weight', 0.5)
        self.reg_loss_weight = config.get('reg_loss_weight', 1.0)
        
        # DDIMScheduler for noise schedule (alphas_cumprod) and add_noise
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            steps_offset=1,
            beta_schedule="scaled_linear",
            prediction_type=self.prediction_type,
        )

        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.horizon = policy_cfg.get('horizon', 16)
        self.n_action_steps = policy_cfg.get('action_horizon', 8)

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

    def _load_anchor_centers(self, anchor_path: str):
        """
        Load anchor centers (absolute coordinates) from .npy or .pkl file.

        Stores anchor_centers as absolute (x, y) coordinates for:
        - BEV grid_sample spatial attention (needs real positions)
        - Diffusion noise addition (in normalized space via norm_odo)
        - Residual prediction (model predicts offset from anchor)
        """
        if os.path.exists(anchor_path):
            if anchor_path.endswith('.npy'):
                centers = np.load(anchor_path)  # (M, T, 2) absolute coords
            else:
                with open(anchor_path, 'rb') as f:
                    data = pickle.load(f)
                centers = data['centers']  # (M, T, 2) absolute coords

            self.anchor_num_points = centers.shape[1]
            centers_tensor = torch.from_numpy(centers).float()

            # Single buffer: absolute coordinates throughout
            self.register_buffer('anchor_centers', centers_tensor)

            print(f"[DiffusionDiTCarlaPolicy] Loaded {self.num_modes} anchor centers from {anchor_path}")
            print(f"  - Shape: {centers.shape} (num_modes, num_points, 2)")
            print(f"  - x range: [{centers[...,0].min():.2f}, {centers[...,0].max():.2f}]")
            print(f"  - y range: [{centers[...,1].min():.2f}, {centers[...,1].max():.2f}]")
        else:
            print(f"[Warning] Anchor file not found: {anchor_path}, using default initialization")
            self.anchor_num_points = 6
            self.register_buffer('anchor_centers', torch.zeros(self.num_modes, self.anchor_num_points, 2))
    
    def get_best_anchor_idx(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Find the closest anchor center for each trajectory in the batch.
        Uses anchor_centers (absolute coordinates) for distance computation.

        Args:
            trajectory: (B, T, 2) - ground truth trajectory (absolute coordinates)

        Returns:
            best_idx: (B,) - index of closest anchor for each sample
        """
        B = trajectory.shape[0]
        T = trajectory.shape[1]

        # Sample trajectory at anchor waypoint positions
        # anchor_centers: (num_modes, 5, 2), trajectory: (B, T, 2)
        # We need to interpolate trajectory to match anchor's 5 points
        if T != self.anchor_num_points:
            # Linearly interpolate trajectory to anchor_num_points
            indices = torch.linspace(0, T-1, self.anchor_num_points).long()
            traj_sampled = trajectory[:, indices, :]  # (B, anchor_num_points, 2)
        else:
            traj_sampled = trajectory

        # Compute L2 distance between trajectory and each anchor (use absolute coords)
        # traj_sampled: (B, 5, 2), anchor_centers: (32, 5, 2)
        # Expand for broadcasting: (B, 1, 5, 2) - (1, 32, 5, 2) -> (B, 32, 5, 2)
        traj_expanded = traj_sampled.unsqueeze(1)  # (B, 1, 5, 2)
        anchor_expanded = self.anchor_centers.unsqueeze(0)  # (1, 32, 5, 2)

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

        # VQA anchor for 33rd mode experiment
        vqa_anchor = None
        if self.use_vqa_anchor and 'vqa_anchor' in batch:
            vqa_anchor = batch['vqa_anchor'].to(device=device, dtype=model_dtype)  # (B, 6, 2)

        # Extract behavior labels (if available)
        behavior_labels = batch.get('behavior_labels', None)
        allowed_flags = batch.get('allowed_flags', None)

        # ========== Compute Multimodal Loss (DiffusionDrive style) ==========
        loss_dict = self._compute_multimodal_loss(
            trajectory=trajectory,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
            route_gt=route_gt,
            device=device,
            model_dtype=model_dtype,
            vqa_anchor=vqa_anchor,
            behavior_labels=behavior_labels,
            allowed_flags=allowed_flags,
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
        route_gt: Optional[torch.Tensor] = None,
        vqa_anchor: Optional[torch.Tensor] = None,
        behavior_labels: Optional[torch.Tensor] = None,
        allowed_flags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute DiffusionDrive-style multimodal loss with truncated diffusion.

        Absolute trajectory version:
        1. Sample timestep from truncated range [0, train_trunc_timesteps)
        2. Normalize abs anchors to [-1,1], add DDIM noise, clamp, denormalize back
        3. Model receives noisy abs anchors → BEV grid_sample at real positions
        4. Model predicts clean abs trajectory (normalized space), denorm for L1 loss

        Args:
            trajectory: (B, horizon, 2) - ground truth absolute trajectory
            transfuser_bev_feature: (B, 1512, 8, 8) - BEV feature
            transfuser_bev_feature_upsample: (B, 64, 64, 64) - Upscaled BEV feature
            ego_status: (B, To, status_dim) - ego vehicle status
            device: torch device
            model_dtype: model dtype (e.g., bfloat16)
            route_gt: (B, num_waypoints, 2) - optional ground truth route for auxiliary loss
        """
        batch_size = trajectory.shape[0]
        horizon = trajectory.shape[1]

        assert horizon == self.anchor_num_points, \
            f"horizon ({horizon}) must equal anchor_num_points ({self.anchor_num_points})."

        # ========== Sample timestep ==========
        timesteps = torch.randint(
            0, self.train_trunc_timesteps,
            (batch_size,), device=device
        ).long()

        # ========== Prepare anchors (absolute) and add noise in normalized space ==========
        all_anchors = self.anchor_centers.unsqueeze(0).expand(batch_size, -1, -1, -1)
        all_anchors = all_anchors.to(device=device, dtype=model_dtype)  # (B, M, T, 2)

        # Concatenate VLM anchor as extra mode if enabled
        if vqa_anchor is not None:
            vqa_anchor_4d = vqa_anchor.unsqueeze(1)  # (B, 1, T, 2)
            all_anchors = torch.cat([all_anchors, vqa_anchor_4d], dim=1)  # (B, M+1, T, 2)

        # Normalize to [-1, 1] for noise addition
        all_anchors_normed = self.norm_odo(all_anchors)  # (B, M, T, 2)

        B, M, T_anchor, D = all_anchors_normed.shape
        anchors_flat = all_anchors_normed.contiguous().view(B * M, T_anchor, D)

        # Expand timesteps for all modes: (B,) -> (B*M,)
        timesteps_expanded = timesteps.unsqueeze(1).expand(-1, M).reshape(B * M)

        # DDIM additive noise
        noise = torch.randn(anchors_flat.shape, dtype=torch.float32, device=device)
        noisy_anchors_flat = self.diffusion_scheduler.add_noise(
            original_samples=anchors_flat,
            noise=noise,
            timesteps=timesteps_expanded
        )

        # Reshape back and clamp to valid normalized range
        noisy_anchors_normed = noisy_anchors_flat.view(B, M, T_anchor, D)
        noisy_anchors_normed = torch.clamp(noisy_anchors_normed, -1, 1)
        noise = noise.view(B, M, T_anchor, D)

        if self.use_normalized_forward:
            # Normalized forward: model operates in normalized space
            # BEV grid_sample receives normalized coords (decouples spatial dependency)
            noisy_anchors_abs = noisy_anchors_normed  # pass normalized as "abs"
        else:
            # Denormalize back to absolute coords for model input (BEV sampling needs real positions)
            noisy_anchors_abs = self.denorm_odo(noisy_anchors_normed)  # (B, M, T, 2)

        # ========== Forward pass ==========
        # Model receives noisy anchors; BEV grid_sample and trajectory regression
        # use anchors_abs (physical space normally, normalized space if use_normalized_forward).
        # Ablation: fix BEV at clean anchor positions for training consistency
        bev_abs = all_anchors if self.fix_bev_at_anchor else noisy_anchors_abs
        poses_reg_out, poses_cls, route_pred, mode_out = self.model(
            anchors=noisy_anchors_normed,
            anchors_abs=bev_abs,
            timestep=timesteps,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
            behavior_labels=behavior_labels,
            allowed_flags=allowed_flags,
        )
        # Denorm model output if in normalized forward mode
        if self.use_normalized_forward:
            poses_reg_abs = self.denorm_odo(poses_reg_out)
        else:
            poses_reg_abs = poses_reg_out
        # poses_reg_abs: (B, num_modes, horizon, 2) in absolute coords

        # ========== Find best matching anchor (in absolute space) ==========
        traj_expanded = trajectory.unsqueeze(1)  # (B, 1, horizon, 2)
        dist = torch.norm(traj_expanded - all_anchors, dim=-1)  # (B, num_modes, horizon)
        dist = dist.mean(dim=-1)  # (B, num_modes)
        mode_idx = torch.argmin(dist, dim=-1)  # (B,)

        # ========== Classification Loss (Focal Loss) ==========
        target_onehot = torch.zeros(batch_size, M, device=device, dtype=model_dtype)
        target_onehot.scatter_(1, mode_idx.unsqueeze(1), 1)
        loss_cls = self._focal_loss(poses_cls, target_onehot)

        # ========== Regression Loss ==========
        mode_idx_expanded = mode_idx.view(batch_size, 1, 1, 1).expand(-1, 1, horizon, 2)

        if self.prediction_type == "sample":
            # Model predicts clean x_0 directly in absolute space
            best_reg_abs = torch.gather(poses_reg_abs, 1, mode_idx_expanded).squeeze(1)
            loss_reg = F.l1_loss(best_reg_abs, trajectory, reduction='mean')
        else:
            # prediction_type == "epsilon"
            alphas_cumprod = self.diffusion_scheduler.alphas_cumprod.to(device)
            alpha_t = alphas_cumprod[timesteps].view(batch_size, 1, 1, 1)

            best_noise = torch.gather(noise, 1, mode_idx_expanded).squeeze(1)
            best_reg_abs = torch.gather(poses_reg_abs, 1, mode_idx_expanded).squeeze(1)
            best_reg_normed = self.norm_odo(best_reg_abs)
            best_noisy_normed = torch.gather(noisy_anchors_normed, 1, mode_idx_expanded).squeeze(1)

            pred_eps = (best_noisy_normed - alpha_t.squeeze(1).sqrt() * best_reg_normed) / (1 - alpha_t.squeeze(1)).sqrt()
            loss_reg = F.mse_loss(pred_eps, best_noise, reduction='mean')

        # ========== Route Loss (Optional) ==========
        total_loss = self.cls_loss_weight * loss_cls + self.reg_loss_weight * loss_reg

        route_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        if route_gt is not None and route_pred is not None:
            route_loss = F.l1_loss(route_pred, route_gt, reduction='mean')
            total_loss = total_loss + self.route_loss_weight * route_loss

        # ========== Semantic Behavior Loss (Optional) ==========
        behavior_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        allowed_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        if self.semantic_behavior_enabled and behavior_labels is not None and allowed_flags is not None:
            behavior_labels_dev = behavior_labels.to(device=device)
            allowed_flags_dev = allowed_flags.to(device=device, dtype=model_dtype)

            # Allowed prediction: binary classification per anchor
            pred_allowed = self.allowed_pred_head(mode_out).squeeze(-1)  # (B, num_modes)
            allowed_loss = F.binary_cross_entropy_with_logits(pred_allowed, allowed_flags_dev)

            # Behavior prediction: multi-class classification per anchor
            pred_behavior = self.behavior_pred_head(mode_out)  # (B, num_modes, num_behaviors)
            behavior_loss = F.cross_entropy(
                pred_behavior.reshape(-1, self.num_behaviors),
                behavior_labels_dev.reshape(-1),
            )

            total_loss = total_loss + self.allowed_loss_weight * allowed_loss + \
                         self.behavior_loss_weight * behavior_loss

        loss_dict = {
            'total_loss': total_loss,
            'cls_loss': loss_cls,
            'reg_loss': loss_reg,
            'route_loss': route_loss,
            'behavior_loss': behavior_loss,
            'allowed_loss': allowed_loss,
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
            use_server_style: bool = False,
            server_noise_type: str = "additive",
            gt_trajectory: Optional[torch.Tensor] = None,
            **kwargs
            ):
        """
        Generate trajectory samples using DDIM multi-step denoising (DiffusionDriveV2 style).

        DDIM Multi-step Inference:
        1. Normalize anchors to [-1, 1]
        2. Add DDIM additive noise at truncated timestep (e.g., t=8)
        3. Multi-step DDIM denoising loop:
           a. Clamp & denormalize → original delta scale (model input)
           b. Model predicts clean delta (original scale)
           c. Normalize prediction → [-1, 1]
           d. scheduler.step(eta=0) → deterministic DDIM update → x_{t-1}
        4. Final denormalization → original delta scale
        5. Select best mode via cls scores, cumsum → absolute trajectory

        Args:
            transfuser_bev_feature: (B, 1512, 8, 8) - BEV feature
            transfuser_bev_feature_upsample: (B, 64, 64, 64) - Upscaled BEV feature
            ego_status: (B, To, status_dim) - ego status history
            num_denoise_steps: number of DDIM denoising steps (default: self.num_diffusion_steps)
            no_noise: if True, skip noise addition (for debugging model capability)
            use_server_style: if True, run server-style iterative forward (no DDIM step)
            server_noise_type: noise type for server-style path ('additive' or 'multiplicative')

        Returns:
            (trajectory, route_pred) tuple - trajectory (B, T, 2), route_pred (B, 20, 2)
        """
        bs = transfuser_bev_feature.shape[0]
        num_steps = num_denoise_steps or self.num_diffusion_steps
        horizon = self.anchor_centers.shape[1]
        poses_cls = None
        route_pred = None

        # Get all anchors in absolute coords
        all_anchors = self.anchor_centers.unsqueeze(0).expand(bs, -1, -1, -1)
        all_anchors = all_anchors.to(device=device, dtype=model_dtype)  # (B, M, T, 2)

        # Concatenate VLM anchor as extra mode if enabled
        vqa_anchor = kwargs.get('vqa_anchor', None)
        if vqa_anchor is not None:
            vqa_anchor_4d = vqa_anchor.unsqueeze(1)  # (B, 1, T, 2)
            all_anchors = torch.cat([all_anchors, vqa_anchor_4d], dim=1)  # (B, M+1, T, 2)

        num_modes_effective = all_anchors.shape[1]

        # Normalize to [-1, 1] for diffusion
        all_anchors_normed = self.norm_odo(all_anchors)  # (B, M, T, 2)

        # Inference behavior conditioning: follow_road(0) + allowed(1)
        infer_behavior = None
        infer_allowed = None
        if self.semantic_behavior_enabled:
            M = all_anchors.shape[1]
            infer_behavior = torch.zeros((bs, M), dtype=torch.long, device=device)  # follow_road
            infer_allowed = torch.ones((bs, M), dtype=torch.long, device=device)    # allowed

        if no_noise:
            # Debug mode: single forward pass with clean anchors
            timesteps = torch.zeros((bs,), dtype=torch.long, device=device)
            anchors_abs_input = all_anchors_normed if self.use_normalized_forward else all_anchors
            poses_reg_out, poses_cls, route_pred, mode_out = self.model(
                anchors=all_anchors_normed,
                anchors_abs=anchors_abs_input,
                timestep=timesteps,
                transfuser_bev_feature=transfuser_bev_feature,
                transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                ego_status=ego_status,
                behavior_labels=infer_behavior,
                allowed_flags=infer_allowed,
            )
            if self.use_normalized_forward:
                final_abs = self.denorm_odo(poses_reg_out)
            else:
                final_abs = poses_reg_out  # (B, M, T, 2)
        else:
            # ========== Standard DDIM Multi-step Denoising ==========
            noise = torch.randn(all_anchors_normed.shape, dtype=torch.float32, device=device)
            trunc_ts = torch.full((bs,), self.trunc_timesteps - 1, dtype=torch.long, device=device)
            x_t = self.diffusion_scheduler.add_noise(
                original_samples=all_anchors_normed,
                noise=noise,
                timesteps=trunc_ts
            )

            step_ratio = self.trunc_timesteps / num_steps
            roll_timesteps = (np.arange(0, num_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

            alphas_cumprod = self.diffusion_scheduler.alphas_cumprod.to(device)

            # Prepare GT in normed space for diagnostic
            gt_normed = None
            if gt_trajectory is not None:
                gt_normed = self.norm_odo(gt_trajectory.to(device=device, dtype=x_t.dtype))

            prev_best_idx = None
            _debug_first_batch = False
            poses_reg_abs = None

            for step_i, k in enumerate(roll_timesteps):
                t_cur = k.item()
                t_next = roll_timesteps[step_i + 1].item() if step_i + 1 < len(roll_timesteps) else 0

                # Clamp and prepare model input
                x_clamped = torch.clamp(x_t, -1, 1)
                if self.use_normalized_forward:
                    x_abs = x_clamped  # Stay in normalized space
                else:
                    x_abs = self.denorm_odo(x_clamped)  # (B, M, T, 2) absolute coords

                # Ablation: fix BEV at clean anchor positions (disable dynamic spatial feedback)
                bev_abs = all_anchors if self.fix_bev_at_anchor else x_abs

                t_tensor = torch.full((bs,), t_cur, dtype=torch.long, device=device)
                poses_reg_out, poses_cls, route_pred, mode_out = self.model(
                    anchors=x_clamped,
                    anchors_abs=bev_abs,
                    timestep=t_tensor,
                    transfuser_bev_feature=transfuser_bev_feature,
                    transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                    ego_status=ego_status,
                    behavior_labels=infer_behavior,
                    allowed_flags=infer_allowed,
                )

                if self.use_normalized_forward:
                    # Normalized forward: model output is already in normalized space
                    pred_x0_normed = poses_reg_out
                    poses_reg_abs = self.denorm_odo(poses_reg_out)  # for debug logging
                else:
                    poses_reg_abs = poses_reg_out
                    pred_x0_normed = self.norm_odo(poses_reg_abs)
                alpha_t = alphas_cumprod[t_cur]
                alpha_next = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
                pred_eps = (x_t - alpha_t.sqrt() * pred_x0_normed) / (1 - alpha_t).sqrt().clamp(min=1e-8)

                if _debug_first_batch:
                    best_idx = torch.argmax(poses_cls, dim=-1)
                    print(f"\n  [DDIM step {step_i}] t={t_cur}->{t_next}, sqrt(1-alpha_t)={(1-alpha_t).sqrt().item():.4f}")
                    for b in range(min(bs, 1)):
                        bi = best_idx[b].item()
                        print(f"    batch {b}: best_mode={bi}, cls_top3={torch.topk(poses_cls[b], 3).indices.tolist()}")

                        if gt_normed is not None:
                            gt_n = gt_normed[b]  # (T, 2)
                            anchor_gt_dist = (all_anchors_normed[b] - gt_n.unsqueeze(0)).abs().mean(dim=(-2, -1))
                            pred_error_per_mode = (pred_x0_normed[b] - gt_n.unsqueeze(0)).abs().mean(dim=(-2, -1))
                            x_t_anchor_dist = (x_t[b] - all_anchors_normed[b]).abs().mean(dim=(-2, -1))
                            # Physical L2
                            pred_x0_abs = poses_reg_abs[b]
                            gt_abs = gt_trajectory[b].to(device=device, dtype=x_t.dtype)
                            l2_per_mode = (pred_x0_abs - gt_abs.unsqueeze(0)).norm(dim=-1).mean(dim=-1)
                            print(f"      anchor vs GT (normed MAE):  best={anchor_gt_dist[bi]:.4f}, "
                                  f"others_mean={anchor_gt_dist.sum().sub(anchor_gt_dist[bi]).div(num_modes_effective-1):.4f}")
                            print(f"      pred_x0 vs GT (normed MAE): best={pred_error_per_mode[bi]:.4f}, "
                                  f"others_mean={pred_error_per_mode.sum().sub(pred_error_per_mode[bi]).div(num_modes_effective-1):.4f}")
                            print(f"      pred_x0 vs GT (L2 meters):  best={l2_per_mode[bi]:.4f}, "
                                  f"others_mean={l2_per_mode.sum().sub(l2_per_mode[bi]).div(num_modes_effective-1):.4f}")
                            print(f"      x_t vs anchor (normed MAE): best={x_t_anchor_dist[bi]:.4f}, "
                                  f"others_mean={x_t_anchor_dist.sum().sub(x_t_anchor_dist[bi]).div(num_modes_effective-1):.4f}"
                                  f"  <- {'~noise level' if step_i == 0 else 'OOD if >> noise level'}")

                        pred_eps_mag = pred_eps[b].abs().mean(dim=(-2, -1))
                        print(f"      pred_eps_mag (should~1.0): best={pred_eps_mag[bi]:.4f}, "
                              f"others_mean={pred_eps_mag.sum().sub(pred_eps_mag[bi]).div(num_modes_effective-1):.4f}, "
                              f"others_max={pred_eps_mag.clone().scatter_(0, best_idx[b:b+1], 0).max():.4f}")

                        if prev_best_idx is not None:
                            print(f"      mode_changed: {bi != prev_best_idx[b].item()} (was {prev_best_idx[b].item()})")

                    prev_best_idx = best_idx

                # DDIM step
                x_t = alpha_next.sqrt() * pred_x0_normed + (1 - alpha_next).sqrt() * pred_eps

            if _debug_first_batch:
                self._test_debug_printed = True

            # dd_baseline-style: use model's last clean prediction in absolute space.
            if poses_reg_abs is not None:
                final_abs = poses_reg_abs
            elif self.use_normalized_forward:
                final_abs = self.denorm_odo(torch.clamp(x_t, -1, 1))
            else:
                final_abs = self.denorm_odo(x_t)

        # Select best mode, filtered by allowed prediction (Energy Shielding)
        if self.semantic_behavior_enabled and mode_out is not None:
            pred_allowed_logits = self.allowed_pred_head(mode_out).squeeze(-1)  # (B, M)
            pred_allowed_prob = torch.sigmoid(pred_allowed_logits)
            # Mask forbidden modes by setting their cls scores to -inf
            forbidden_mask = pred_allowed_prob < 0.5
            masked_cls = poses_cls.clone()
            masked_cls[forbidden_mask] = float('-inf')
            # Fallback: if all modes are forbidden, use original scores
            all_forbidden = forbidden_mask.all(dim=-1)  # (B,)
            if all_forbidden.any():
                masked_cls[all_forbidden] = poses_cls[all_forbidden]
            best_mode_idx = torch.argmax(masked_cls, dim=-1)  # (B,)
        else:
            best_mode_idx = torch.argmax(poses_cls, dim=-1)  # (B,)
        horizon = final_abs.shape[2]
        mode_idx_expanded = best_mode_idx.view(bs, 1, 1, 1).expand(-1, 1, horizon, 2)
        best_trajectory = torch.gather(final_abs, 1, mode_idx_expanded).squeeze(1)  # (B, T, 2)

        return best_trajectory, route_pred

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        no_noise: bool = False,
        use_server_style: bool = False,
        server_noise_type: str = "additive",
        gt_trajectory: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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

        # VQA anchor for 33rd mode experiment
        vqa_anchor = None
        if self.use_vqa_anchor and 'vqa_anchor' in nobs:
            vqa_anchor = nobs['vqa_anchor'].to(device=device, dtype=model_dtype)

        # Generate samples using multimodal prediction
        nsample, route_pred = self.conditional_sample(
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            no_noise=no_noise,
            ego_status=ego_status,
            device=device,
            model_dtype=model_dtype,
            use_server_style=use_server_style,
            server_noise_type=server_noise_type,
            gt_trajectory=gt_trajectory,
            vqa_anchor=vqa_anchor,
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
