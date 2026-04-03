"""
Route B+: Compositional Energy-Guided Diffusion Policy

Pure diffusion from N(0,I) with DDIM and compositional energy guidance.
Energy gradients are injected at each denoising step with time-scale scheduling:
  - High noise (t=100->70): Navigation energy pulls trajectories to macro direction
  - Medium noise (t=70->30): Collision energy applies repulsion
  - Low noise (t=30->0): Offroad energy + smoothness for lane-level polish

Dual-optimizer training (GAN-style D/G isolation):
  - Phase 1 (optimizer_energy): Train energy heads on anchor trajectories + GT augmentation
  - Phase 2 (optimizer_diff): Train diffusion decoder (L_diffusion + L_alignment)

Key differences from Route A (DiffusionDiTCarlaPolicy):
  - No anchor centers — start from pure Gaussian noise
  - Model predicts absolute trajectory (no residual from anchor)
  - 10-step DDIM inference with energy gradient guidance
  - Energy heads trained on diverse anchor trajectories (positive + negative)
  - Alignment loss encourages decoder to generate low-energy trajectories
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Callable
import numpy as np
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from model.transformer_for_diffusion_multi_head import TransformerForDiffusion


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


# =============================================================================
# Time-scale Energy Weight Scheduling
# =============================================================================

def get_energy_weights(t: int, T: int = 100):
    """
    Annealed energy weights: different energies activate at different noise levels.

    Schedule (hardcoded; LLM Router can override per-head weights at runtime):
      t=T→0.7T  (high noise):   E_route activates (macro navigation direction)
      t=0.7T→0.3T (mid noise):  E_vehicle (front/left/right) + E_pedestrian activate
      t=0.3T→0    (low noise):  E_offroad activates (lane-level polish)

    Args:
        t: current timestep (higher = more noise)
        T: total timesteps

    Returns:
        (w_route, w_veh, w_off) weight tuple
        w_veh applies to all 4 collision heads (front/left/right/pedestrian)
    """
    progress = 1.0 - t / T  # 0 -> 1 as denoising progresses

    w_route = 1.0                                      # always active (navigation)
    w_veh   = max(0.0, (progress - 0.3) / 0.4)        # activates after t < 0.7*T
    w_off   = max(0.0, (progress - 0.7) / 0.3)        # activates after t < 0.3*T

    return w_route, w_veh, w_off


# =============================================================================
# Route B+ Policy
# =============================================================================

class AnnealedEnergyGuidancePolicy(nn.Module):
    """
    Compositional Energy-Guided Diffusion Policy.

    Training (dual-optimizer, GAN-style):
      Phase 1 (energy heads): Supervised on anchor trajectories + GT augmentation
      Phase 2 (decoder): Standard diffusion loss + alignment loss

    Inference:
      - Start from pure Gaussian noise N(0, I) in normalized [-1, 1] space
      - 10-step DDIM denoising
      - At each step, compute energy gradients and inject them with time-scale weights
      - Select best trajectory from num_samples candidates via energy shielding
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.cfg = config
        policy_cfg = config['policy']
        shape_meta = config['shape_meta']
        action_dim = shape_meta['action']['shape'][0]

        self.enable_action_normalization = config.get('enable_action_normalization', True)
        self.n_obs_steps = policy_cfg.get('n_obs_steps', config.get('obs_horizon', 1))

        # Transfuser feature dimensions
        transfuser_cfg = config.get('transfuser_encoder', {})
        self.bev_feature_dim = transfuser_cfg.get('bev_feature_dim', 1512)
        self.bev_feature_upsample_dim = transfuser_cfg.get('bev_feature_upsample_dim', 64)

        # Route B specific config
        route_b_cfg = config.get('route_b', {})
        self.route_b_cfg = route_b_cfg
        self.num_samples = route_b_cfg.get('num_samples', 1)  # diffusion denoising: single mode
        self.num_energy_modes = route_b_cfg.get('num_energy_modes', 32)  # energy training: multi-anchor
        self.num_inference_steps = route_b_cfg.get('num_inference_steps', 10)
        self.guidance_scale = route_b_cfg.get('guidance_scale', 1.0)  # global energy guidance multiplier
        self.use_split_forward = route_b_cfg.get('use_split_forward', True)
        # Per-head energy weights (used in alignment loss and guidance)
        self.energy_front_weight      = route_b_cfg.get('energy_front_weight', 1.0)
        self.energy_left_weight       = route_b_cfg.get('energy_left_weight', 1.0)
        self.energy_right_weight      = route_b_cfg.get('energy_right_weight', 1.0)
        self.energy_pedestrian_weight = route_b_cfg.get('energy_pedestrian_weight', 1.0)
        self.energy_offroad_weight    = route_b_cfg.get('energy_offroad_weight', 1.0)
        self.energy_route_weight      = route_b_cfg.get('energy_route_weight', 1.0)
        # Route target params
        self.route_energy_margin = route_b_cfg.get('route_energy_margin', 1.0)   # corridor half-width (m)
        self.route_energy_norm   = route_b_cfg.get('route_energy_norm', 5.0)     # normalization factor (m)

        # Route B+ config
        self.energy_grad_clip_norm = route_b_cfg.get('energy_grad_clip_norm', 1.0)
        self.alignment_loss_weight = route_b_cfg.get('alignment_loss_weight', 0.1)
        self.num_gt_augmentations = route_b_cfg.get('num_gt_augmentations', 4)
        self.use_safe_anchors = route_b_cfg.get('use_safe_anchors', False)
        self.energy_noisy_training = route_b_cfg.get('energy_noisy_training', False)
        self.alignment_warmup_epochs = route_b_cfg.get('alignment_warmup_epochs', 0)
        self.train_energy = route_b_cfg.get('train_energy', True)
        self.use_front_route_risk_energy = route_b_cfg.get('use_front_route_risk_energy', False)
        self._current_epoch = 0
        self.route_abs_stats_path = config.get('route_abs_stats_path', None)
        self.use_lidar_bev_detail = route_b_cfg.get('use_lidar_bev_detail', False)

        status_dim = config.get('bev_encoder', {}).get('state_dim', 15)
        ego_status_seq_len = policy_cfg.get('ego_status_seq_len', self.n_obs_steps)
        num_waypoints = policy_cfg.get('num_waypoints', 20)
        self.num_waypoints = num_waypoints
        n_emb = policy_cfg.get('n_emb', 512)

        # Build model with anchor_free=True and energy_heads=True
        model = TransformerForDiffusion(
            input_dim=policy_cfg.get('input_dim', 2),
            output_dim=policy_cfg.get('output_dim', 2),
            horizon=policy_cfg.get('horizon', 6),
            n_obs_steps=self.n_obs_steps,
            cond_dim=256,
            n_layer=policy_cfg.get('n_layer', 8),
            n_head=policy_cfg.get('n_head', 8),
            n_emb=n_emb,
            p_drop_emb=policy_cfg.get('p_drop_emb', 0.1),
            p_drop_attn=policy_cfg.get('p_drop_attn', 0.1),
            causal_attn=policy_cfg.get('causal_attn', True),
            obs_as_cond=policy_cfg.get('obs_as_global_cond', True),
            n_cond_layers=policy_cfg.get('n_cond_layers', 4),
            status_dim=status_dim,
            ego_status_seq_len=ego_status_seq_len,
            transfuser_bev_dim=self.bev_feature_dim,
            transfuser_bev_upsample_dim=self.bev_feature_upsample_dim,
            num_waypoints=num_waypoints,
            num_modes=self.num_energy_modes,
            traj_can_attend_route=policy_cfg.get('traj_can_attend_route', True),
            anchor_free=True,
            energy_heads=True,
            ego_detail_activation_t=policy_cfg.get('ego_detail_activation_t', 400),
            use_lidar_bev_detail=self.use_lidar_bev_detail,
            use_condition_group_dropout=policy_cfg.get('use_condition_group_dropout', False),
        )
        self.model = model

        # ========== Diffusion Configuration ==========
        diffusion_cfg = config.get('truncated_diffusion', {})
        self.num_train_timesteps = diffusion_cfg.get('num_train_timesteps', 1000)
        # Route B uses full diffusion range (not truncated)
        self.train_max_timesteps = route_b_cfg.get('train_max_timesteps',
                                                    diffusion_cfg.get('trunc_timesteps', 100))
        self.prediction_type = diffusion_cfg.get('prediction_type', 'sample')

        # Delta z-score normalization buffers (legacy, for ablation)
        self.register_buffer('delta_mean', None)  # (T, 2)
        self.register_buffer('delta_std', None)   # (T, 2)
        # Per-timestep abs z-score normalization buffers (ablation)
        self.register_buffer('abs_mean', None)  # (T, 2)
        self.register_buffer('abs_std', None)   # (T, 2)
        # Route per-waypoint abs z-score normalization buffers for joint route diffusion
        self.register_buffer('route_abs_mean', None)  # (T_route, 2)
        self.register_buffer('route_abs_std', None)   # (T_route, 2)
        # Global abs z-score normalization buffers
        self.register_buffer('global_abs_mean', None)  # (2,)
        self.register_buffer('global_abs_std', None)   # (2,)

        # Loss weights
        self.cls_loss_weight = config.get('cls_loss_weight', 0.5)
        self.reg_loss_weight = config.get('reg_loss_weight', 1.0)
        self.route_loss_weight = diffusion_cfg.get('route_loss_weight', 0.5)
        self.energy_loss_weight = route_b_cfg.get('energy_loss_weight', 1.0)
        self.speed_loss_weight = route_b_cfg.get('speed_loss_weight', 1.0)

        # DDIM Scheduler
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            steps_offset=1,
            beta_schedule="scaled_linear",
            prediction_type=self.prediction_type,
        )

        self.action_dim = action_dim
        self.horizon = policy_cfg.get('horizon', 6)
        self.n_action_steps = policy_cfg.get('action_horizon', 8)

        # Anchor buffer (registered externally before DDP wrapping)
        self.register_buffer('anchor_centers_abs', None)

    # ========== Anchor Registration ==========
    def register_anchor_centers(self, anchor_centers_abs):
        """Register anchor trajectories for energy head training.
        Must be called before DDP wrapping.
        """
        if isinstance(anchor_centers_abs, np.ndarray):
            anchor_centers_abs = torch.from_numpy(anchor_centers_abs).float()
        device = next(self.parameters()).device
        self.register_buffer('anchor_centers_abs', anchor_centers_abs.to(device))

    def _get_transfuser_lidar_bev(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.use_lidar_bev_detail:
            return None
        transfuser_lidar_bev = tensor_dict.get('transfuser_lidar_bev')
        if transfuser_lidar_bev is None:
            return None
        return transfuser_lidar_bev.to(device=device, dtype=model_dtype)

    def _slice_energy_anchor_inputs(
        self,
        device: torch.device,
        model_dtype: torch.dtype,
        behavior_labels: Optional[torch.Tensor] = None,
        allowed_flags: Optional[torch.Tensor] = None,
        energy_targets: Optional[torch.Tensor] = None,
        energy_active_mask: Optional[torch.Tensor] = None,
    ):
        """Take the first num_energy_modes anchors and aligned supervision tensors."""
        if self.anchor_centers_abs is None:
            raise ValueError("anchor_centers_abs is not registered")

        requested = int(self.num_energy_modes)
        available = int(self.anchor_centers_abs.shape[0])
        if requested > available:
            raise ValueError(
                f"num_energy_modes={requested} exceeds available anchors={available}"
            )

        def _slice_optional(name: str, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            if tensor.shape[1] < requested:
                raise ValueError(
                    f"{name} provides only {tensor.shape[1]} modes, expected at least {requested}"
                )
            return tensor[:, :requested]

        anchor_subset = self.anchor_centers_abs[:requested].to(device=device, dtype=model_dtype)
        behavior_subset = _slice_optional("behavior_labels", behavior_labels)
        allowed_subset = _slice_optional("allowed_flags", allowed_flags)
        energy_targets_subset = _slice_optional("energy_targets", energy_targets)
        energy_active_mask_subset = _slice_optional("energy_active_mask", energy_active_mask)
        return (
            requested,
            anchor_subset,
            behavior_subset,
            allowed_subset,
            energy_targets_subset,
            energy_active_mask_subset,
        )

    # ========== Speed Target Computation ==========
    def _compute_speed_target(self, trajectory, device):
        """Compute two-hot speed target from GT trajectory for speed head training.

        Target speed = displacement magnitude over 1 second (waypoints 0→2 at 0.5s interval).
        Encoded as two-hot distribution over speed bins for cross-entropy loss.

        Args:
            trajectory: (B, T, 2) absolute GT trajectory
            device: torch device

        Returns:
            two_hot: (B, num_classes) soft labels, or None if trajectory too short
        """
        if trajectory.shape[1] < 3:
            return None
        # Speed = ||wp[2] - wp[0]|| / 1.0s  (2 steps × 0.5s)
        displacement = trajectory[:, 2] - trajectory[:, 0]  # (B, 2)
        target_speed = displacement.norm(dim=-1)  # (B,) in m/s

        speed_classes = self.model.speed_classes
        num_classes = len(speed_classes)
        bins = torch.tensor(speed_classes, device=device, dtype=target_speed.dtype)

        # Clamp to valid range
        target_speed = target_speed.clamp(min=bins[0], max=bins[-1])

        # Two-hot encoding: interpolate between adjacent bins
        B = target_speed.shape[0]
        two_hot = torch.zeros(B, num_classes, device=device, dtype=target_speed.dtype)
        for i in range(num_classes - 1):
            mask = (target_speed >= bins[i]) & (target_speed < bins[i + 1])
            if i == num_classes - 2:  # last bin: include upper bound
                mask = mask | (target_speed == bins[i + 1])
            if mask.any():
                ratio = (target_speed[mask] - bins[i]) / (bins[i + 1] - bins[i]).clamp(min=1e-6)
                two_hot[mask, i] = 1.0 - ratio
                two_hot[mask, i + 1] = ratio
        return two_hot

    @staticmethod
    def decode_speed_two_hot(speed_logits, speed_classes):
        """Decode speed logits to scalar m/s via softmax weighted sum.

        Args:
            speed_logits: (B, num_classes) raw logits
            speed_classes: list of float, bin centers

        Returns:
            speed_scalar: (B,) in m/s
        """
        bins = torch.tensor(speed_classes, device=speed_logits.device, dtype=speed_logits.dtype)
        probs = torch.softmax(speed_logits.float(), dim=-1)
        return (probs * bins).sum(dim=-1)

    # ========== GT Augmentation ==========
    def _augment_gt(self, trajectory, K):
        """Generate K augmented GT variants via speed scaling.
        Returns reliable positive samples (safe trajectories).
        """
        B, T, D = trajectory.shape
        aug = trajectory.unsqueeze(1).expand(-1, K, -1, -1).clone()  # (B, K, T, 2)
        for k in range(K):
            # Speed scaling: 0.8-1.0x (slower is safer)
            scale = 0.8 + 0.2 * torch.rand(B, 1, 1, device=trajectory.device)
            # Scale displacements relative to start point
            displacements = aug[:, k, 1:] - aug[:, k, :1]  # (B, T-1, 2)
            aug[:, k, 1:] = aug[:, k, :1] + displacements * scale
        return aug  # (B, K, T, 2)

    # ========== Delta Stats Registration ==========
    def register_delta_stats(self, delta_mean, delta_std):
        """Register per-step delta mean/std for z-score normalization.
        Must be called before DDP wrapping.

        Args:
            delta_mean: (T, 2) per-step delta mean
            delta_std: (T, 2) per-step delta std
        """
        if isinstance(delta_mean, np.ndarray):
            delta_mean = torch.from_numpy(delta_mean).float()
        if isinstance(delta_std, np.ndarray):
            delta_std = torch.from_numpy(delta_std).float()
        device = next(self.parameters()).device
        self.register_buffer('delta_mean', delta_mean.to(device))
        self.register_buffer('delta_std', delta_std.to(device))

    @staticmethod
    def select_norm_stats_config(config: Dict):
        """Select exactly one trajectory normalization source from config.

        Route B should not silently fall back across normalization families.
        The config must explicitly choose exactly one of:
          - global_abs_stats_path
          - abs_stats_path
          - delta_stats_path
        """
        candidates = [
            ('global_abs', config.get('global_abs_stats_path')),
            ('abs', config.get('abs_stats_path')),
            ('delta', config.get('delta_stats_path')),
        ]
        specified = [(name, path) for name, path in candidates if path]
        if len(specified) != 1:
            pretty = {name: path for name, path in candidates}
            raise ValueError(
                "Route B requires exactly one normalization config among "
                "`global_abs_stats_path`, `abs_stats_path`, and `delta_stats_path`. "
                f"Got: {pretty}"
            )
        return specified[0]

    def register_norm_stats_from_config(self, config: Dict):
        """Register the explicitly configured trajectory normalization stats."""
        norm_mode, stats_path = self.select_norm_stats_config(config)
        if norm_mode == 'global_abs':
            gdata = np.load(stats_path)
            self.register_global_abs_stats(gdata['global_abs_mean'], gdata['global_abs_std'])
        elif norm_mode == 'abs':
            adata = np.load(stats_path)
            self.register_abs_stats(adata['abs_mean'], adata['abs_std'])
        else:
            ddata = np.load(stats_path)
            self.register_delta_stats(ddata['delta_mean'], ddata['delta_std'])
        return norm_mode

    # ========== Normalization: Delta Z-Score ==========
    @staticmethod
    def abs_to_delta(abs_traj: torch.Tensor) -> torch.Tensor:
        """Convert absolute trajectory to delta: [p0, p1-p0, p2-p1, ...]"""
        delta = abs_traj.clone()
        delta[..., 1:, :] = abs_traj[..., 1:, :] - abs_traj[..., :-1, :]
        return delta

    @staticmethod
    def delta_to_abs(delta: torch.Tensor) -> torch.Tensor:
        """Convert delta to absolute trajectory via cumulative sum."""
        return delta.cumsum(dim=-2)

    def z_norm(self, delta: torch.Tensor) -> torch.Tensor:
        """Z-score normalize delta using per-step mean/std.
        delta: (..., T, 2)  ->  z: (..., T, 2)
        """
        mean = self.delta_mean.to(delta.device)  # (T, 2)
        std = self.delta_std.to(delta.device)     # (T, 2)
        return (delta - mean) / std.clamp(min=1e-6)

    def z_denorm(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse z-score: z -> delta.
        z: (..., T, 2)  ->  delta: (..., T, 2)
        """
        mean = self.delta_mean.to(z.device)  # (T, 2)
        std = self.delta_std.to(z.device)    # (T, 2)
        return z * std + mean

    def norm_to_abs(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse: z-normed -> absolute trajectory.
        Priority: global_abs > per-step abs > delta (legacy).
        """
        if self.global_abs_mean is not None:
            return self.global_abs_z_denorm(z)
        if self.abs_mean is not None:
            return self.abs_z_denorm(z)
        return self.delta_to_abs(self.z_denorm(z))

    def abs_to_norm(self, abs_traj: torch.Tensor) -> torch.Tensor:
        """Forward: absolute trajectory -> z-normed.
        Priority: global_abs > per-step abs > delta (legacy).
        """
        if self.global_abs_mean is not None:
            return self.global_abs_z_norm(abs_traj)
        if self.abs_mean is not None:
            return self.abs_z_norm(abs_traj)
        return self.z_norm(self.abs_to_delta(abs_traj))

    # ========== Normalization: Per-Timestep Abs Z-Score ==========
    def register_abs_stats(self, abs_mean, abs_std):
        """Register per-step abs mean/std for z-score normalization.
        Must be called before DDP wrapping.
        """
        if isinstance(abs_mean, np.ndarray):
            abs_mean = torch.from_numpy(abs_mean).float()
        if isinstance(abs_std, np.ndarray):
            abs_std = torch.from_numpy(abs_std).float()
        device = next(self.parameters()).device
        self.register_buffer('abs_mean', abs_mean.to(device))
        self.register_buffer('abs_std', abs_std.to(device))

    def abs_z_norm(self, abs_traj: torch.Tensor) -> torch.Tensor:
        """Per-timestep z-score on absolute coordinates.
        abs_traj: (..., T, 2)  ->  z: (..., T, 2)
        """
        mean = self.abs_mean.to(abs_traj.device)  # (T, 2)
        std = self.abs_std.to(abs_traj.device)     # (T, 2)
        return (abs_traj - mean) / std.clamp(min=1e-6)

    def abs_z_denorm(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse per-timestep z-score -> absolute coordinates.
        z: (..., T, 2)  ->  abs_traj: (..., T, 2)
        """
        mean = self.abs_mean.to(z.device)  # (T, 2)
        std = self.abs_std.to(z.device)    # (T, 2)
        return z * std + mean

    # ========== Normalization: Route Per-Waypoint Abs Z-Score ==========
    def register_route_abs_stats(self, route_abs_mean, route_abs_std):
        """Register per-waypoint absolute route stats for route diffusion."""
        if isinstance(route_abs_mean, np.ndarray):
            route_abs_mean = torch.from_numpy(route_abs_mean).float()
        if isinstance(route_abs_std, np.ndarray):
            route_abs_std = torch.from_numpy(route_abs_std).float()
        device = next(self.parameters()).device
        self.register_buffer('route_abs_mean', route_abs_mean.to(device))
        self.register_buffer('route_abs_std', route_abs_std.to(device))

    def _require_route_abs_stats(self):
        if self.route_abs_mean is None or self.route_abs_std is None:
            raise RuntimeError(
                "Joint Route B ego diffusion requires route_abs_stats_path to be configured "
                "and loaded (route_abs_mean/route_abs_std)."
            )

    def route_abs_to_norm(self, abs_route: torch.Tensor) -> torch.Tensor:
        """Per-waypoint z-score on route absolute coordinates."""
        self._require_route_abs_stats()
        mean = self.route_abs_mean.to(abs_route.device)
        std = self.route_abs_std.to(abs_route.device)
        return (abs_route - mean) / std.clamp(min=1e-6)

    def route_norm_to_abs(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse per-waypoint z-score for route absolute coordinates."""
        self._require_route_abs_stats()
        mean = self.route_abs_mean.to(z.device)
        std = self.route_abs_std.to(z.device)
        return z * std + mean

    def joint_abs_to_norm(self, traj_abs: torch.Tensor, route_abs: torch.Tensor) -> torch.Tensor:
        """Concatenate normalized trajectory and route diffusion states."""
        traj_norm = self.abs_to_norm(traj_abs)
        route_norm = self.route_abs_to_norm(route_abs)
        return torch.cat([traj_norm, route_norm], dim=-2)

    def joint_norm_to_abs(self, joint_z: torch.Tensor) -> torch.Tensor:
        """Split a joint traj+route diffusion state back to absolute coordinates."""
        joint_len = self.horizon + self.num_waypoints
        if joint_z.shape[-2] != joint_len:
            raise ValueError(f"Expected joint diffusion length {joint_len}, got {joint_z.shape[-2]}")
        traj_abs = self.norm_to_abs(joint_z[..., :self.horizon, :])
        route_abs = self.route_norm_to_abs(joint_z[..., self.horizon:, :])
        return torch.cat([traj_abs, route_abs], dim=-2)

    # ========== Normalization: Global Abs Z-Score ==========
    def register_global_abs_stats(self, global_abs_mean, global_abs_std):
        """Register global abs mean/std for z-score normalization.
        All timesteps share the same (2,) mean/std — preserves temporal structure.
        """
        if isinstance(global_abs_mean, np.ndarray):
            global_abs_mean = torch.from_numpy(global_abs_mean).float()
        if isinstance(global_abs_std, np.ndarray):
            global_abs_std = torch.from_numpy(global_abs_std).float()
        device = next(self.parameters()).device
        self.register_buffer('global_abs_mean', global_abs_mean.to(device))
        self.register_buffer('global_abs_std', global_abs_std.to(device))

    def global_abs_z_norm(self, abs_traj: torch.Tensor) -> torch.Tensor:
        """Global z-score: all timesteps share same mean/std.
        abs_traj: (..., T, 2)  ->  z: (..., T, 2)
        """
        mean = self.global_abs_mean.to(abs_traj.device)  # (2,)
        std = self.global_abs_std.to(abs_traj.device)     # (2,)
        return (abs_traj - mean) / std.clamp(min=1e-6)

    def global_abs_z_denorm(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse global z-score -> absolute coordinates.
        z: (..., T, 2)  ->  abs_traj: (..., T, 2)
        """
        mean = self.global_abs_mean.to(z.device)  # (2,)
        std = self.global_abs_std.to(z.device)    # (2,)
        return z * std + mean

    # ========== Focal Loss (same as Route A) ==========
    def _focal_loss(self, logits, targets, gamma=2.0, alpha=0.25):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = alpha * (1 - p_t) ** gamma
        return (focal_weight * bce).mean()

    def _get_front_route_energy_targets(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        model_dtype: torch.dtype,
    ):
        """Scene-level front risk supervision from precomputed route-constrained labels.

        The new front/block/risk labels are scene-level, not anchor-level. We therefore
        train the front energy head on the GT slot only instead of forcing the same target
        onto every anchor sample.
        """
        hazard_bin = batch.get('front_route_hazard_bin', None)
        if hazard_bin is not None:
            target = hazard_bin.to(device=device, dtype=model_dtype).clamp(min=0.0, max=4.0) / 4.0
        else:
            risk = batch.get('front_route_risk', None)
            if risk is None:
                return None, None
            target = risk.to(device=device, dtype=model_dtype).clamp(0.0, 1.0)
            block_risk = batch.get('front_route_block_risk', None)
            if block_risk is not None:
                target = torch.maximum(
                    target,
                    block_risk.to(device=device, dtype=model_dtype).clamp(0.0, 1.0),
                )

        actor_weight = batch.get('front_route_actor_weight', None)
        if actor_weight is None:
            sample_weight = torch.ones_like(target)
        else:
            actor_weight = actor_weight.to(device=device, dtype=model_dtype)
            positive_weight = actor_weight.clamp(min=1.0)
            sample_weight = torch.where(
                target > 0,
                positive_weight,
                torch.ones_like(target),
            )
        return target, sample_weight

    def _compute_front_route_energy_loss(
        self,
        front_logits: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> torch.Tensor:
        targets, sample_weight = self._get_front_route_energy_targets(
            batch=batch,
            device=device,
            model_dtype=model_dtype,
        )
        if targets is None:
            return torch.tensor(0.0, device=device, dtype=model_dtype)
        loss = F.binary_cross_entropy_with_logits(
            front_logits.float(),
            targets.float(),
            reduction='none',
        )
        weighted = loss * sample_weight.float()
        return weighted.sum() / sample_weight.float().sum().clamp(min=1.0)

    # ========== Energy Head Eval with Detached Weights ==========
    @staticmethod
    def _eval_energy_head_detached(head: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """Evaluate an energy head using detached weights.
        Gradients flow back to input x, but NOT to head parameters.
        head: nn.Sequential(Linear, SiLU, Linear)
        """
        x = F.linear(x, head[0].weight.detach(), head[0].bias.detach())
        x = F.silu(x)
        x = F.linear(x, head[2].weight.detach(), head[2].bias.detach())
        return x

    # ========== Forward (DDP-compatible) ==========
    def forward(self, batch: Dict[str, torch.Tensor],
                return_loss_dict: bool = False,
                phase: str = 'unified'):
        """
        DDP-compatible forward.
          - 'unified': Single forward pass for both energy + diffusion training (default)
          - 'energy': Phase 1 only — train energy heads (legacy, for ablation)
          - 'diffusion': Phase 2 only — train decoder (legacy, for ablation)
        """
        if phase == 'split' or (phase == 'unified' and self.use_split_forward):
            loss_dict = self.compute_split_loss(batch)
        elif phase == 'unified':
            loss_dict = self.compute_unified_loss(batch)
        elif phase == 'energy':
            loss_dict = self.compute_energy_loss(batch)
        else:
            loss_dict = self.compute_diffusion_loss(batch)

        if return_loss_dict:
            return loss_dict
        else:
            return loss_dict['total_loss']

    # ========== Unified Training: Single Forward Pass ==========
    def compute_split_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Split-forward Route B training.

        1. Ego path: waypoint-token denoising for pred_x0 + route prediction
        2. Energy path: anchor/GT trajectory-level scoring
        3. Alignment path: evaluate pred_x0 as a single trajectory with detached energy heads
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        trajectory = batch['agent_pos'].to(device=device, dtype=model_dtype)  # (B, T, 2)
        B, T, D = trajectory.shape
        M_anchor = self.num_energy_modes

        transfuser_bev_feature = batch['transfuser_bev_feature'].to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = batch['transfuser_bev_feature_upsample'].to(device=device, dtype=model_dtype)
        transfuser_lidar_bev = self._get_transfuser_lidar_bev(batch, device, model_dtype)
        ego_status = batch['ego_status'].to(device=device, dtype=model_dtype)

        route_gt = batch.get('route', None)
        if route_gt is not None:
            route_gt = route_gt.to(device=device, dtype=model_dtype)
        else:
            raise KeyError("Joint Route B ego diffusion requires 'route' in the training batch")

        behavior_labels = batch.get('behavior_labels', None)
        allowed_flags = batch.get('allowed_flags', None)
        energy_targets = batch.get('energy_targets', None)
        energy_active_mask = batch.get('energy_active_mask', None)
        has_energy = (self.anchor_centers_abs is not None
                      and behavior_labels is not None
                      and allowed_flags is not None)

        bev_proj = self.model.decoder.compute_bev_proj(transfuser_bev_feature)

        # ===== Forward 1: Ego denoising (M=1) =====
        if route_gt.shape[1] != self.num_waypoints:
            raise ValueError(f"Expected route_gt with {self.num_waypoints} waypoints, got {route_gt.shape}")
        traj_route_normed = self.joint_abs_to_norm(trajectory, route_gt)  # (B, T_joint, 2)
        diff_timesteps = torch.randint(0, self.train_max_timesteps, (B,), device=device).long()
        noise = torch.randn(B, self.horizon + self.num_waypoints, D, dtype=torch.float32, device=device)
        noisy_flat = self.diffusion_scheduler.add_noise(
            original_samples=traj_route_normed,
            noise=noise,
            timesteps=diff_timesteps,
        )
        noisy_joint = noisy_flat.unsqueeze(1)  # (B, 1, T_joint, 2)
        noisy_joint_abs = self.joint_norm_to_abs(noisy_joint)

        poses_reg, route_pred, _, _, speed_pred = self.model.forward_ego(
            x_t=noisy_joint,
            x_t_abs=noisy_joint_abs,
            timestep=diff_timesteps,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
            bev_proj_cached=bev_proj,
            transfuser_lidar_bev=transfuser_lidar_bev,
        )
        poses_reg_abs = self.norm_to_abs(poses_reg)
        route_pred_abs = self.route_norm_to_abs(route_pred)
        traj_target = trajectory.unsqueeze(1)
        loss_reg = F.l1_loss(poses_reg_abs, traj_target, reduction='mean')

        route_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        if route_pred is not None:
            route_loss = F.l1_loss(route_pred_abs, route_gt, reduction='mean')
            # FDE: extra weight on final route point
            route_loss = route_loss + F.l1_loss(route_pred_abs[:, -1], route_gt[:, -1], reduction='mean')

        # Speed loss: two-hot cross-entropy
        speed_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        if speed_pred is not None:
            speed_target = self._compute_speed_target(trajectory, device)
            if speed_target is not None:
                speed_loss = F.cross_entropy(speed_pred.float(), speed_target)

        # ===== Forward 2: Energy training (anchors + GT) =====
        zero_t = torch.tensor(0.0, device=device, dtype=model_dtype)
        energy_loss = zero_t
        loss_front = loss_left = loss_right = loss_ped = loss_off = zero_t
        loss_route = zero_t

        if has_energy and self.train_energy:
            (
                M_anchor,
                anchor_subset,
                behavior_labels_subset,
                allowed_flags_subset,
                energy_targets_subset,
                energy_active_mask_subset,
            ) = self._slice_energy_anchor_inputs(
                device=device,
                model_dtype=model_dtype,
                behavior_labels=behavior_labels,
                allowed_flags=allowed_flags,
                energy_targets=energy_targets,
                energy_active_mask=energy_active_mask,
            )
            anchor_abs = anchor_subset.unsqueeze(0).expand(B, -1, -1, -1)
            behavior_labels_dev = behavior_labels_subset.to(device=device)
            allowed_flags_dev = allowed_flags_subset.to(device=device, dtype=model_dtype)
            energy_targets_dev = None
            energy_active_mask_dev = None
            if energy_targets_subset is not None:
                energy_targets_dev = energy_targets_subset.to(device=device, dtype=model_dtype)
            if energy_active_mask_subset is not None:
                energy_active_mask_dev = energy_active_mask_subset.to(device=device, dtype=torch.bool)

            K = min(self.num_gt_augmentations, M_anchor)
            if K > 0:
                anchor_abs = anchor_abs.clone()
                behavior_labels_dev = behavior_labels_dev.clone()
                allowed_flags_dev = allowed_flags_dev.clone()
                if energy_targets_dev is not None:
                    energy_targets_dev = energy_targets_dev.clone()
                if energy_active_mask_dev is not None:
                    energy_active_mask_dev = energy_active_mask_dev.clone()
                gt_aug = self._augment_gt(trajectory, K)
                anchor_abs[:, :K] = gt_aug
                behavior_labels_dev[:, :K] = 0
                allowed_flags_dev[:, :K] = 1.0
                if energy_targets_dev is not None:
                    energy_targets_dev[:, :K] = 0.0
                if energy_active_mask_dev is not None:
                    energy_active_mask_dev[:, :K] = True

            gt_abs = trajectory.unsqueeze(1)
            energy_abs = torch.cat([anchor_abs, gt_abs], dim=1)  # (B, 33, T, 2)
            energy_normed = self.abs_to_norm(energy_abs)

            gt_behavior = torch.zeros(B, 1, device=device, dtype=behavior_labels_dev.dtype)
            behavior_all = torch.cat([behavior_labels_dev, gt_behavior], dim=1)
            allowed_all = torch.cat([
                allowed_flags_dev,
                torch.ones(B, 1, device=device, dtype=model_dtype),
            ], dim=1)

            energy_scores, _ = self.model.forward_energy(
                x_t=energy_normed,
                x_t_abs=energy_abs,
                timestep=torch.zeros(B, device=device, dtype=torch.long),
                transfuser_bev_feature=transfuser_bev_feature,
                transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                ego_status=ego_status,
                traj_for_energy=energy_abs,
                behavior_labels=behavior_all,
                allowed_flags=allowed_all,
                bev_proj_cached=bev_proj,
                route_points=route_gt,
                transfuser_lidar_bev=transfuser_lidar_bev,
            )

            if energy_targets_dev is not None:
                gt_targets = torch.zeros(B, 1, energy_targets_dev.shape[-1], device=device, dtype=model_dtype)
                energy_targets_all = torch.cat([energy_targets_dev, gt_targets], dim=1)
                front_target = energy_targets_all[..., 0]
                left_target = energy_targets_all[..., 1]
                right_target = energy_targets_all[..., 2]
                ped_target = energy_targets_all[..., 3]
                offroad_target = energy_targets_all[..., 4]
            else:
                front_target = (behavior_all == 1).float()
                left_target = (behavior_all == 2).float()
                right_target = (behavior_all == 3).float()
                ped_target = (behavior_all == 4).float()
                offroad_target = ((behavior_all >= 5) & (behavior_all <= 6)).float()

            def _sl1e(pred, tgt, mask=None):
                return F.smooth_l1_loss(pred[mask], tgt[mask]) if mask is not None else F.smooth_l1_loss(pred, tgt)

            if not self.use_safe_anchors:
                if energy_active_mask_dev is not None:
                    gt_active = torch.ones(B, 1, device=device, dtype=torch.bool)
                    active_mask = torch.cat([energy_active_mask_dev, gt_active], dim=1)
                else:
                    active_mask = torch.ones(B, M_anchor + 1, device=device, dtype=torch.bool)
                    active_mask[:, K:M_anchor] = (allowed_all[:, K:M_anchor] < 0.5)
                    active_mask[:, M_anchor] = True
                n_active = active_mask.sum()
                if n_active > 0:
                    if self.use_front_route_risk_energy:
                        loss_front = zero_t
                    else:
                        loss_front = _sl1e(energy_scores['front'], front_target, active_mask)
                    loss_left = _sl1e(energy_scores['left'], left_target, active_mask)
                    loss_right = _sl1e(energy_scores['right'], right_target, active_mask)
                    loss_ped = _sl1e(energy_scores['pedestrian'], ped_target, active_mask)
                    loss_off = _sl1e(energy_scores['offroad'], offroad_target, active_mask)
            else:
                if self.use_front_route_risk_energy:
                    loss_front = zero_t
                else:
                    loss_front = _sl1e(energy_scores['front'], front_target)
                loss_left = _sl1e(energy_scores['left'], left_target)
                loss_right = _sl1e(energy_scores['right'], right_target)
                loss_ped = _sl1e(energy_scores['pedestrian'], ped_target)
                loss_off = _sl1e(energy_scores['offroad'], offroad_target)

            energy_loss = loss_front + loss_left + loss_right + loss_ped + loss_off

        if self.use_front_route_risk_energy:
            gt_abs = trajectory.unsqueeze(1)
            gt_normed = self.abs_to_norm(gt_abs)
            front_route_logits, _ = self.model.forward_front_route_risk(
                x_t=gt_normed,
                x_t_abs=gt_abs,
                timestep=torch.zeros(B, device=device, dtype=torch.long),
                transfuser_bev_feature=transfuser_bev_feature,
                transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                ego_status=ego_status,
                traj_for_energy=gt_abs,
                bev_proj_cached=bev_proj,
                route_points=route_gt,
                transfuser_lidar_bev=transfuser_lidar_bev,
            )
            loss_front = self._compute_front_route_energy_loss(
                front_route_logits[:, -1],
                batch,
                device,
                model_dtype,
            )
            energy_loss = loss_front + loss_left + loss_right + loss_ped + loss_off

        # ===== Forward 3: Alignment / guidance eval on pred_x0 =====
        alignment_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        alignment_active = (self._current_epoch >= self.alignment_warmup_epochs)
        if self.alignment_loss_weight > 0 and alignment_active:
            if self.use_front_route_risk_energy:
                _, mode_out_front = self.model.forward_front_route_risk_eval(
                    x_t=poses_reg,
                    x_t_abs=poses_reg_abs,
                    transfuser_bev_feature=transfuser_bev_feature,
                    transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                    ego_status=ego_status,
                    traj_for_energy=poses_reg_abs,
                    bev_proj_cached=bev_proj,
                    route_points=route_gt,
                    transfuser_lidar_bev=transfuser_lidar_bev,
                )
                front_align_input = torch.cat([poses_reg_abs.flatten(-2), mode_out_front], dim=-1)
                a_front = self._eval_energy_head_detached(
                    self.model.front_route_risk_head,
                    front_align_input,
                ).squeeze(-1)
                alignment_loss = alignment_loss + self.energy_front_weight * torch.sigmoid(a_front).mean()

            if has_energy and self.train_energy:
                _, mode_out_clean = self.model.forward_energy_eval(
                    x_t=poses_reg,
                    x_t_abs=poses_reg_abs,
                    transfuser_bev_feature=transfuser_bev_feature,
                    transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                    ego_status=ego_status,
                    traj_for_energy=poses_reg_abs,
                    bev_proj_cached=bev_proj,
                    route_points=route_gt,
                    transfuser_lidar_bev=transfuser_lidar_bev,
                )
                align_input = torch.cat([poses_reg_abs.flatten(-2), mode_out_clean], dim=-1)
                if not self.use_front_route_risk_energy:
                    a_front = self._eval_energy_head_detached(self.model.energy_front_head, align_input).squeeze(-1)
                    alignment_loss = alignment_loss + self.energy_front_weight * torch.sigmoid(a_front).mean()
                a_left = self._eval_energy_head_detached(self.model.energy_left_head, align_input).squeeze(-1)
                a_right = self._eval_energy_head_detached(self.model.energy_right_head, align_input).squeeze(-1)
                a_ped = self._eval_energy_head_detached(self.model.energy_pedestrian_head, align_input).squeeze(-1)
                a_off = self._eval_energy_head_detached(self.model.energy_offroad_head, align_input).squeeze(-1)
                alignment_loss = alignment_loss + (
                    self.energy_left_weight * torch.sigmoid(a_left).mean()
                    + self.energy_right_weight * torch.sigmoid(a_right).mean()
                    + self.energy_pedestrian_weight * torch.sigmoid(a_ped).mean()
                    + self.energy_offroad_weight * torch.sigmoid(a_off).mean()
                )

        total_loss = (
            self.energy_loss_weight * energy_loss
            + self.reg_loss_weight * loss_reg
            + self.route_loss_weight * route_loss
            + self.alignment_loss_weight * alignment_loss
            + self.speed_loss_weight * speed_loss
        )

        return {
            'total_loss': total_loss,
            'energy_loss': energy_loss,
            'energy_front_loss': loss_front,
            'energy_left_loss': loss_left,
            'energy_right_loss': loss_right,
            'energy_ped_loss': loss_ped,
            'energy_off_loss': loss_off,
            'energy_route_loss': loss_route,
            'reg_loss': loss_reg,
            'cls_loss': torch.tensor(0.0, device=device),
            'route_loss': route_loss,
            'speed_loss': speed_loss,
            'alignment_loss': alignment_loss,
        }

    # ========== Unified Training: Single Forward Pass ==========
    def compute_unified_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Unified single-forward training for both energy heads and diffusion decoder.

        Constructs M_total = M_anchor + 1 (GT) + 1 (x_t) modes in one forward pass:
          - Slots [0, M_anchor): anchor trajectories (with behavior labels from dataset)
          - Slot [M_anchor]: GT trajectory (clean, labeled safe)
          - Slot [M_anchor+1]: noisy GT trajectory (x_t for diffusion denoising)

        Mode queries:
          - Slots [0, M_anchor+1): use mode_queries (anchor + GT share anchor query space)
          - Slot [M_anchor+1]: uses diff_mode_query (dedicated diffusion query)

        Energy head evaluates original anchor/GT coords via traj_for_energy (not model output).
        Diffusion loss computed only on the last slot's poses_reg.

        Anchor isolation in decoder self-attention ensures no information leak between modes.
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        trajectory = batch['agent_pos'].to(device=device, dtype=model_dtype)  # (B, T, 2)
        B, T, D = trajectory.shape
        M_anchor = self.num_energy_modes  # 32

        transfuser_bev_feature = batch['transfuser_bev_feature'].to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = batch['transfuser_bev_feature_upsample'].to(device=device, dtype=model_dtype)
        transfuser_lidar_bev = self._get_transfuser_lidar_bev(batch, device, model_dtype)
        ego_status = batch['ego_status'].to(device=device, dtype=model_dtype)

        route_gt = batch.get('route', None)
        if route_gt is not None:
            route_gt = route_gt.to(device=device, dtype=model_dtype)

        behavior_labels = batch.get('behavior_labels', None)  # (B, M_anchor) or None
        allowed_flags = batch.get('allowed_flags', None)      # (B, M_anchor) or None

        has_energy = (self.train_energy
                      and self.anchor_centers_abs is not None
                      and behavior_labels is not None
                      and allowed_flags is not None)

        # ========== Build anchor slots ==========

        # ========== Build GT slot (1) ==========
        gt_abs = trajectory.unsqueeze(1)  # (B, 1, T, 2)

        # ========== Build x_t slot (1): noisy GT for diffusion ==========
        traj_normed = self.abs_to_norm(trajectory)  # (B, T, 2)
        diff_timesteps = torch.randint(0, self.train_max_timesteps, (B,), device=device).long()

        noise = torch.randn(B, T, D, dtype=torch.float32, device=device)
        noisy_flat = self.diffusion_scheduler.add_noise(
            original_samples=traj_normed,
            noise=noise,
            timesteps=diff_timesteps,
        )
        noisy_traj = noisy_flat.unsqueeze(1)  # (B, 1, T, D)

        # ========== Concatenate all slots into unified input ==========
        # IMPORTANT: x_t (diffusion) is at position 0 to match M=1 inference (predict_action).
        # Order: [x_t(0), anchors(1-32), GT(33)]
        if has_energy:
            (
                M_anchor,
                anchor_subset,
                behavior_labels_subset,
                allowed_flags_subset,
                _,
                _,
            ) = self._slice_energy_anchor_inputs(
                device=device,
                model_dtype=model_dtype,
                behavior_labels=behavior_labels,
                allowed_flags=allowed_flags,
            )
            anchor_abs = anchor_subset.unsqueeze(0).expand(B, -1, -1, -1).clone()
            behavior_labels_dev = behavior_labels_subset.to(device=device).clone()
            allowed_flags_dev = allowed_flags_subset.to(device=device, dtype=model_dtype).clone()
            K = min(self.num_gt_augmentations, M_anchor)
            if K > 0:
                gt_aug = self._augment_gt(trajectory, K)
                anchor_abs[:, :K] = gt_aug
                behavior_labels_dev[:, :K] = 0
                allowed_flags_dev[:, :K] = 1.0

            # Energy anchor input: normalize anchors
            anchor_normed = self.abs_to_norm(anchor_abs)  # (B, M_anchor, T, 2)
            gt_normed = self.abs_to_norm(gt_abs)           # (B, 1, T, 2)

            # Unified: [x_t (1), anchors (32), GT (1)] = 34 modes
            x_t_unified = torch.cat([noisy_traj, anchor_normed, gt_normed], dim=1)  # (B, 34, T, 2)

            # Abs coords for BEV grid_sample
            noisy_traj_abs = self.norm_to_abs(noisy_traj)
            x_t_abs_unified = torch.cat([noisy_traj_abs, anchor_abs, gt_abs], dim=1)  # (B, 34, T, 2)

            # traj_for_energy: original anchor/GT abs coords for energy head (slots 1-33)
            # Energy evaluates spatial properties (collision/offroad/target) — abs space is natural.
            # Slot 0 (x_t) uses GT abs as placeholder (energy loss is NOT computed on this slot,
            # alignment loss is computed separately below with the actual diffusion prediction).
            energy_traj_padded = torch.cat([
                gt_abs,       # (B, 1, T, 2) placeholder for x_t slot (not used for energy_loss)
                anchor_abs,   # (B, 32, T, 2) abs
                gt_abs,       # (B, 1, T, 2) abs
            ], dim=1)  # (B, 34, T, 2) abs coords
        else:
            # No anchors: just x_t + GT (2 modes), no energy training
            gt_normed = self.abs_to_norm(gt_abs)
            x_t_unified = torch.cat([noisy_traj, gt_normed], dim=1)  # (B, 2, T, 2)
            noisy_traj_abs = self.norm_to_abs(noisy_traj)
            x_t_abs_unified = torch.cat([noisy_traj_abs, gt_abs], dim=1)
            energy_traj_padded = None

        # ========== Single forward pass ==========
        poses_reg, poses_cls, route_pred, mode_out, energy_scores = self.model(
            x_t=x_t_unified,
            x_t_abs=x_t_abs_unified,
            timestep=diff_timesteps,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
            traj_for_energy=energy_traj_padded,
        )

        # ========== Split outputs ==========
        # Layout: [x_t(0), anchors(1..M_anchor), GT(M_anchor+1)]
        if has_energy:
            M_energy = M_anchor + 1  # anchors + GT = 33 energy slots

            # Energy scores for slots 1-33 (anchors + GT), skip slot 0 (x_t)
            energy_scores_energy = {k: v[:, 1:1+M_energy] for k, v in energy_scores.items()}

            # Diffusion output from first slot (position 0)
            poses_reg_diff = poses_reg[:, :1, :, :]  # (B, 1, T, 2)
        else:
            M_energy = 0
            energy_scores_energy = None
            poses_reg_diff = poses_reg[:, :1, :, :]

        # ========== Energy Loss (on slots 1-33: anchors + GT) ==========
        zero_t = torch.tensor(0.0, device=device, dtype=model_dtype)
        energy_loss = zero_t
        loss_front = loss_left = loss_right = loss_ped = loss_off = loss_route = zero_t

        if has_energy and energy_scores_energy is not None:
            # Build targets: anchors (32) + GT (1)
            # GT slot is safe (all collision/offroad = 0)
            gt_behavior = torch.zeros(B, 1, device=device, dtype=behavior_labels_dev.dtype)
            behavior_all = torch.cat([behavior_labels_dev, gt_behavior], dim=1)  # (B, 33)

            front_target  = (behavior_all == 1).float()
            left_target   = (behavior_all == 2).float()
            right_target  = (behavior_all == 3).float()
            ped_target    = (behavior_all == 4).float()
            offroad_target = ((behavior_all >= 5) & (behavior_all <= 6)).float()

            # Route deviation target for anchors+GT: (B, 33)
            # anchor_abs_for_energy contains anchors+GT in abs coords
            if route_gt is not None:
                # Reconstruct abs traj for energy slots from x_t_abs_unified slots 1-34
                anchor_abs_energy = x_t_abs_unified[:, 1:1+M_energy, :, :]  # (B, 33, T, 2)
                route_target_all = self.compute_route_target(
                    anchor_abs_energy, route_gt, trajectory)   # (B, 33)
            else:
                route_target_all = torch.zeros(B, M_energy, device=device, dtype=model_dtype)

            def _sl1e(pred, tgt, mask=None):
                return F.smooth_l1_loss(pred[mask], tgt[mask]) if mask is not None else F.smooth_l1_loss(pred, tgt)

            if not self.use_safe_anchors:
                active_mask = torch.ones(B, M_energy, device=device, dtype=torch.bool)
                allowed_flags_original = torch.cat([
                    allowed_flags_dev,
                    torch.ones(B, 1, device=device),  # GT always active
                ], dim=1)
                K = min(self.num_gt_augmentations, M_anchor)
                active_mask[:, K:M_anchor] = (allowed_flags_original[:, K:M_anchor] < 0.5)
                active_mask[:, M_anchor] = True  # GT slot always active

                n_active = active_mask.sum()
                if n_active > 0:
                    if self.use_front_route_risk_energy:
                        loss_front = self._compute_front_route_energy_loss(
                            energy_scores_energy['front'][:, -1],
                            batch,
                            device,
                            model_dtype,
                        )
                    else:
                        loss_front = _sl1e(energy_scores_energy['front'],  front_target,  active_mask)
                    loss_left  = _sl1e(energy_scores_energy['left'],   left_target,   active_mask)
                    loss_right = _sl1e(energy_scores_energy['right'],  right_target,  active_mask)
                    loss_ped   = _sl1e(energy_scores_energy['pedestrian'], ped_target, active_mask)
                    loss_off   = _sl1e(energy_scores_energy['offroad'], offroad_target, active_mask)
                # Route loss on ALL slots (continuous metric)
                loss_route = _sl1e(energy_scores_energy['route'], route_target_all)
            else:
                if self.use_front_route_risk_energy:
                    loss_front = self._compute_front_route_energy_loss(
                        energy_scores_energy['front'][:, -1],
                        batch,
                        device,
                        model_dtype,
                    )
                else:
                    loss_front = _sl1e(energy_scores_energy['front'],  front_target)
                loss_left  = _sl1e(energy_scores_energy['left'],   left_target)
                loss_right = _sl1e(energy_scores_energy['right'],  right_target)
                loss_ped   = _sl1e(energy_scores_energy['pedestrian'], ped_target)
                loss_off   = _sl1e(energy_scores_energy['offroad'], offroad_target)
                loss_route = _sl1e(energy_scores_energy['route'], route_target_all)

            energy_loss = loss_front + loss_left + loss_right + loss_ped + loss_off + loss_route

        # ========== Diffusion Loss (on slot 0: x_t) ==========
        poses_reg_diff_abs = self.norm_to_abs(poses_reg_diff)  # (B, 1, T, 2)
        traj_target = trajectory.unsqueeze(1)  # (B, 1, T, 2)
        loss_reg = F.l1_loss(poses_reg_diff_abs, traj_target, reduction='mean')

        # Route loss
        route_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        if route_gt is not None and route_pred is not None:
            route_loss = F.l1_loss(route_pred, route_gt, reduction='mean')

        # Alignment loss: evaluate diffusion prediction with FROZEN energy heads
        # Goal: push diffusion decoder to generate trajectories that energy heads rate as safe.
        # - Use actual prediction (poses_reg_diff_abs), not noisy input or zero placeholder
        # - Gradient flows back to decoder (poses_reg_diff_abs, mode_out are NOT detached)
        # - Energy head weights are detached via _eval_energy_head_detached, so
        #   optimizer_energy sees NO alignment gradient — only energy_loss trains the heads
        # - Sigmoid bounds binary scores to [0,1]; route score left unbounded (already ≥ 0)
        alignment_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        alignment_active = (self._current_epoch >= self.alignment_warmup_epochs)
        if self.alignment_loss_weight > 0 and has_energy and alignment_active:
            diff_mode_out = mode_out[:, :1, :]           # (B, 1, n_emb) — slot 0 is diffusion
            diff_traj_flat = poses_reg_diff_abs.flatten(-2)  # (B, 1, T*2)
            align_input = torch.cat([diff_traj_flat, diff_mode_out], dim=-1)

            # Detached-weight evaluation: grad → decoder, NOT → energy heads
            a_front = self._eval_energy_head_detached(self.model.energy_front_head,      align_input).squeeze(-1)
            a_left  = self._eval_energy_head_detached(self.model.energy_left_head,       align_input).squeeze(-1)
            a_right = self._eval_energy_head_detached(self.model.energy_right_head,      align_input).squeeze(-1)
            a_ped   = self._eval_energy_head_detached(self.model.energy_pedestrian_head, align_input).squeeze(-1)
            a_off   = self._eval_energy_head_detached(self.model.energy_offroad_head,    align_input).squeeze(-1)
            a_rte   = self._eval_energy_head_detached(self.model.energy_route_head,      align_input).squeeze(-1)

            alignment_loss = (
                self.energy_front_weight      * torch.sigmoid(a_front).mean()
                + self.energy_left_weight     * torch.sigmoid(a_left).mean()
                + self.energy_right_weight    * torch.sigmoid(a_right).mean()
                + self.energy_pedestrian_weight * torch.sigmoid(a_ped).mean()
                + self.energy_offroad_weight  * torch.sigmoid(a_off).mean()
                + self.energy_route_weight    * a_rte.mean()  # already ≥ 0, no sigmoid needed
            )

        # ========== Total Loss ==========
        total_loss = (
            self.energy_loss_weight * energy_loss
            + self.reg_loss_weight * loss_reg
            + self.route_loss_weight * route_loss
            + self.alignment_loss_weight * alignment_loss
        )

        return {
            'total_loss': total_loss,
            'energy_loss': energy_loss,
            'energy_front_loss': loss_front,
            'energy_left_loss':  loss_left,
            'energy_right_loss': loss_right,
            'energy_ped_loss':   loss_ped,
            'energy_off_loss':   loss_off,
            'energy_route_loss': loss_route,
            'reg_loss': loss_reg,
            'cls_loss': torch.tensor(0.0, device=device),
            'route_loss': route_loss,
            'alignment_loss': alignment_loss,
        }

    # ========== Route Energy Target ==========
    @staticmethod
    def _point_to_polyline_dist(pts: torch.Tensor, seg_a: torch.Tensor, seg_b: torch.Tensor) -> torch.Tensor:
        """
        Compute mean-over-T minimum distance from trajectory points to a polyline.

        Args:
            pts:   (B, M, T, 2) — trajectory points to evaluate
            seg_a: (B, N, 2)    — polyline segment start points
            seg_b: (B, N, 2)    — polyline segment end points

        Returns:
            (B, M) — mean over T of min-over-N distance to polyline
        """
        # (B, M, T, 1, 2) vs (B, 1, 1, N, 2)
        pts_e = pts.unsqueeze(3)
        a_e = seg_a.unsqueeze(1).unsqueeze(2)
        b_e = seg_b.unsqueeze(1).unsqueeze(2)
        ab = b_e - a_e                                  # (B, 1, 1, N, 2)
        ap = pts_e - a_e                                # (B, M, T, N, 2)
        t = (ap * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-8)
        t = t.clamp(0.0, 1.0)                           # project onto segment
        closest = a_e + t.unsqueeze(-1) * ab            # (B, M, T, N, 2)
        dist = (pts_e - closest).norm(dim=-1)           # (B, M, T, N)
        return dist.min(dim=-1).values.mean(dim=-1)     # (B, M)

    def compute_route_target(
        self,
        anchor_abs: torch.Tensor,
        route: torch.Tensor,
        gt_traj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute continuous route-deviation target for energy heads.

        Defines a "corridor" as the union of the route polyline and the GT traj polyline.
        Trajectories within `route_energy_margin` meters of the corridor get target=0.
        Outside the corridor: target = (dist - margin) / norm, clipped to [0, 2].

        GT trajectory is inside its own corridor by construction → gt_route_target ≈ 0.

        Args:
            anchor_abs: (B, M, T, 2) — anchor/pred trajectories in abs ego-frame coords
            route:      (B, 20, 2)   — route waypoints (ego frame)
            gt_traj:    (B, T, 2)    — GT trajectory (abs ego-frame, used as corridor reference)

        Returns:
            (B, M) route deviation target ≥ 0
        """
        # Route polyline segments: (B, 19, 2)
        route_seg_a = route[:, :-1, :]
        route_seg_b = route[:, 1:, :]

        # GT polyline segments: (B, T-1, 2)
        gt_seg_a = gt_traj[:, :-1, :]
        gt_seg_b = gt_traj[:, 1:, :]

        # Corridor = route + GT polyline (B, 19+T-1, 2)
        corridor_a = torch.cat([route_seg_a, gt_seg_a], dim=1)
        corridor_b = torch.cat([route_seg_b, gt_seg_b], dim=1)

        dist = self._point_to_polyline_dist(anchor_abs, corridor_a, corridor_b)  # (B, M)
        target = F.relu(dist - self.route_energy_margin) / self.route_energy_norm
        return target.clamp(max=2.0)

    # ========== Phase 1: Energy Head Training (legacy) ==========
    def compute_energy_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Train energy heads on anchor trajectories + GT augmentation.
        Provides diverse positive (safe) and negative (forbidden) samples.

        Input composition (M slots):
          - First K slots: GT augmentation (speed-scaled, labeled safe)
          - Remaining M-K slots: anchor trajectories (with behavior_labels from dataset)
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        trajectory = batch['agent_pos'].to(device=device, dtype=model_dtype)  # (B, T, 2)
        B, T, D = trajectory.shape
        M = self.num_energy_modes  # use full anchor count for energy training

        transfuser_bev_feature = batch['transfuser_bev_feature'].to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = batch['transfuser_bev_feature_upsample'].to(device=device, dtype=model_dtype)
        ego_status = batch['ego_status'].to(device=device, dtype=model_dtype)

        behavior_labels = batch.get('behavior_labels', None)  # (B, M) or None
        allowed_flags = batch.get('allowed_flags', None)      # (B, M) or None

        # Fallback: if no anchors or labels, return zero loss (with grad for backward compatibility)
        if self.anchor_centers_abs is None or behavior_labels is None or allowed_flags is None:
            zero = torch.tensor(0.0, device=device, dtype=model_dtype, requires_grad=True)
            return {
                'total_loss': zero,
                'energy_loss': zero.detach(),
                'energy_front_loss': zero.detach(),
                'energy_left_loss': zero.detach(),
                'energy_right_loss': zero.detach(),
                'energy_ped_loss': zero.detach(),
                'energy_off_loss': zero.detach(),
                'energy_route_loss': zero.detach(),
            }

        # --- Build mixed input: GT augmentation + anchors ---
        (
            M,
            anchor_subset,
            behavior_labels_subset,
            allowed_flags_subset,
            _,
            _,
        ) = self._slice_energy_anchor_inputs(
            device=device,
            model_dtype=model_dtype,
            behavior_labels=behavior_labels,
            allowed_flags=allowed_flags,
        )
        anchor_abs = anchor_subset.unsqueeze(0).expand(B, -1, -1, -1).clone()  # (B, M, T, 2)
        behavior_labels_dev = behavior_labels_subset.to(device=device).clone()
        allowed_flags_dev = allowed_flags_subset.to(device=device, dtype=model_dtype).clone()

        # Replace first K slots with GT augmentation (reliable positive samples)
        K = min(self.num_gt_augmentations, M)
        if K > 0:
            gt_aug = self._augment_gt(trajectory, K)  # (B, K, T, 2)
            anchor_abs[:, :K] = gt_aug
            behavior_labels_dev[:, :K] = 0   # safe: follow_road
            allowed_flags_dev[:, :K] = 1.0   # allowed

        # --- Timestep: clean (t=0) or noisy (random t) ---
        if self.energy_noisy_training:
            timesteps = torch.randint(0, self.train_max_timesteps, (B,), device=device).long()
            anchor_normed = self.abs_to_norm(anchor_abs)  # (B, M, T, 2) z-scored delta
            anchor_flat = anchor_normed.contiguous().view(B * M, T, D)
            t_expanded = timesteps.unsqueeze(1).expand(-1, M).reshape(B * M)
            noise = torch.randn_like(anchor_flat)
            noisy_anchor = self.diffusion_scheduler.add_noise(anchor_flat, noise, t_expanded)
            noisy_anchor = noisy_anchor.view(B, M, T, D)
            anchor_input = noisy_anchor
            anchor_abs_input = self.norm_to_abs(anchor_input)
        else:
            timesteps = torch.zeros(B, device=device, dtype=torch.long)
            anchor_input = self.abs_to_norm(anchor_abs)
            anchor_abs_input = anchor_abs

        # --- Forward pass WITH gradients ---
        # Pass original anchor abs coords as traj_for_energy so energy head evaluates
        # spatial properties (collision/offroad) in abs space, consistent with inference.
        _, _, _, _, energy_scores = self.model(
            x_t=anchor_input,
            x_t_abs=anchor_abs_input,
            timestep=timesteps,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
            traj_for_energy=anchor_abs_input,
        )

        # --- Build supervision targets (derived from single-label behavior_labels) ---
        # Vehicle collision direction (mutually exclusive per label priority)
        front_target = (behavior_labels_dev == 1).float()
        left_target  = (behavior_labels_dev == 2).float()
        right_target = (behavior_labels_dev == 3).float()
        ped_target   = (behavior_labels_dev == 4).float()
        offroad_target = ((behavior_labels_dev >= 5) & (behavior_labels_dev <= 6)).float()

        # Route deviation target: continuous distance to route+GT corridor
        route = batch.get('route', None)
        if route is not None:
            route_dev = route.to(device=device, dtype=model_dtype)  # (B, 20, 2)
            trajectory_dev = batch['agent_pos'].to(device=device, dtype=model_dtype)  # (B, T, 2)
            route_target = self.compute_route_target(anchor_abs_input, route_dev, trajectory_dev)  # (B, M)
        else:
            route_target = torch.zeros(B, M, device=device, dtype=model_dtype)

        # --- Compute energy loss with optional masking ---
        def _sl1(pred, tgt, mask=None):
            if mask is not None:
                return F.smooth_l1_loss(pred[mask], tgt[mask])
            return F.smooth_l1_loss(pred, tgt)

        zero = torch.tensor(0.0, device=device, dtype=model_dtype)

        if not self.use_safe_anchors:
            # Binary heads: GT augmentation (first K, always safe) + forbidden anchors only
            active_mask = torch.ones(B, M, device=device, dtype=torch.bool)
            active_mask[:, K:] = (allowed_flags_dev[:, K:] < 0.5)

            n_active = active_mask.sum()
            if n_active > 0:
                if self.use_front_route_risk_energy:
                    gt_like_front = energy_scores['front'].mean(dim=1)
                    loss_front = self._compute_front_route_energy_loss(
                        gt_like_front,
                        batch,
                        device,
                        model_dtype,
                    )
                else:
                    loss_front = _sl1(energy_scores['front'], front_target, active_mask)
                loss_left  = _sl1(energy_scores['left'],  left_target,  active_mask)
                loss_right = _sl1(energy_scores['right'], right_target, active_mask)
                loss_ped   = _sl1(energy_scores['pedestrian'], ped_target, active_mask)
                loss_off   = _sl1(energy_scores['offroad'], offroad_target, active_mask)
            else:
                loss_front = loss_left = loss_right = loss_ped = loss_off = zero
            # Route loss on ALL anchors (continuous metric, not safety-based masking)
            loss_route = _sl1(energy_scores['route'], route_target)
        else:
            if self.use_front_route_risk_energy:
                gt_like_front = energy_scores['front'].mean(dim=1)
                loss_front = self._compute_front_route_energy_loss(
                    gt_like_front,
                    batch,
                    device,
                    model_dtype,
                )
            else:
                loss_front = _sl1(energy_scores['front'], front_target)
            loss_left  = _sl1(energy_scores['left'],  left_target)
            loss_right = _sl1(energy_scores['right'], right_target)
            loss_ped   = _sl1(energy_scores['pedestrian'], ped_target)
            loss_off   = _sl1(energy_scores['offroad'], offroad_target)
            loss_route = _sl1(energy_scores['route'], route_target)

        energy_loss = loss_front + loss_left + loss_right + loss_ped + loss_off + loss_route

        return {
            'total_loss': self.energy_loss_weight * energy_loss,
            'energy_loss': energy_loss,
            'energy_front_loss': loss_front,
            'energy_left_loss': loss_left,
            'energy_right_loss': loss_right,
            'energy_ped_loss': loss_ped,
            'energy_off_loss': loss_off,
            'energy_route_loss': loss_route,
        }

    # ========== Phase 2: Diffusion Training + Alignment (legacy) ==========
    def compute_diffusion_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Single-mode diffusion training + alignment loss.

        Training procedure:
        1. GT trajectory as single mode (M=1), no replication needed
        2. Sample random timestep, add noise to GT
        3. Model predicts clean x_0 from noisy input
        4. Losses: regression (L1), route (L1), alignment (energy)
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        trajectory = batch['agent_pos'].to(device=device, dtype=model_dtype)  # (B, T, 2)
        B, T, D = trajectory.shape

        transfuser_bev_feature = batch['transfuser_bev_feature'].to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = batch['transfuser_bev_feature_upsample'].to(device=device, dtype=model_dtype)
        transfuser_lidar_bev = self._get_transfuser_lidar_bev(batch, device, model_dtype)
        ego_status = batch['ego_status'].to(device=device, dtype=model_dtype)

        route_gt = batch.get('route', None)
        if route_gt is not None:
            route_gt = route_gt.to(device=device, dtype=model_dtype)

        # ========== Prepare noisy trajectory (M=1) ==========
        traj_normed = self.abs_to_norm(trajectory)  # (B, T, 2) z-scored delta
        traj_normed = traj_normed.unsqueeze(1)      # (B, 1, T, 2)

        timesteps = torch.randint(0, self.train_max_timesteps, (B,), device=device).long()

        noise = torch.randn(B, 1, T, D, dtype=torch.float32, device=device)
        traj_flat = traj_normed.view(B, T, D)
        noisy_flat = self.diffusion_scheduler.add_noise(
            original_samples=traj_flat,
            noise=noise.view(B, T, D),
            timesteps=timesteps,
        )
        noisy_traj = noisy_flat.view(B, 1, T, D)

        # ========== Forward pass ==========
        noisy_traj_abs = self.norm_to_abs(noisy_traj)

        poses_reg, poses_cls, route_pred, mode_out, energy_scores = self.model(
            x_t=noisy_traj,
            x_t_abs=noisy_traj_abs,
            timestep=timesteps,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
        )

        # Denorm predictions to absolute space: z-normed delta -> delta -> abs
        poses_reg_abs = self.norm_to_abs(poses_reg)  # (B, 1, T, 2)

        # ========== Regression Loss ==========
        traj_target = trajectory.unsqueeze(1)  # (B, 1, T, 2)
        loss_reg = F.l1_loss(poses_reg_abs, traj_target, reduction='mean')

        # ========== Route Loss ==========
        route_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        if route_gt is not None and route_pred is not None:
            route_loss = F.l1_loss(route_pred, route_gt, reduction='mean')
            # FDE: extra weight on final route point
            route_loss = route_loss + F.l1_loss(route_pred[:, -1], route_gt[:, -1], reduction='mean')

        # ========== Alignment Loss ==========
        alignment_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        alignment_active = (self._current_epoch >= self.alignment_warmup_epochs)
        if self.alignment_loss_weight > 0 and energy_scores is not None and alignment_active:
            alignment_loss = (
                self.energy_front_weight      * torch.sigmoid(energy_scores['front']).mean()
                + self.energy_left_weight     * torch.sigmoid(energy_scores['left']).mean()
                + self.energy_right_weight    * torch.sigmoid(energy_scores['right']).mean()
                + self.energy_pedestrian_weight * torch.sigmoid(energy_scores['pedestrian']).mean()
                + self.energy_offroad_weight  * torch.sigmoid(energy_scores['offroad']).mean()
                + self.energy_route_weight    * energy_scores['route'].mean()
            )

        # ========== Total Loss ==========
        total_loss = (
            self.reg_loss_weight * loss_reg
            + self.route_loss_weight * route_loss
            + self.alignment_loss_weight * alignment_loss
        )

        loss_dict = {
            'total_loss': total_loss,
            'reg_loss': loss_reg,
            'cls_loss': torch.tensor(0.0, device=device),
            'route_loss': route_loss,
            'speed_loss': torch.tensor(0.0, device=device),
            'alignment_loss': alignment_loss,
        }
        return loss_dict

    # ========== Legacy compute_loss (backward compatible) ==========
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Backward compatible: calls compute_diffusion_loss."""
        return self.compute_diffusion_loss(batch)

    # ========== Checkpoint Loading ==========
    @classmethod
    def load_checkpoint(cls, checkpoint_path, config, device='cuda'):
        """Load checkpoint with proper buffer restoration.

        Handles:
        - register_buffer(None) buffers that don't enter state_dict
        - Normalization stats registration from config before loading
        - Best checkpoint already has EMA weights applied (no separate EMA load needed)
        """
        import numpy as np

        policy = cls(config)

        # Register norm stats from config FIRST (makes buffers non-None so load_state_dict can find them)
        policy.register_norm_stats_from_config(config)
        route_abs_stats_path = config.get('route_abs_stats_path')
        if route_abs_stats_path:
            rdata = np.load(route_abs_stats_path)
            policy.register_route_abs_stats(rdata['route_abs_mean'], rdata['route_abs_std'])
        else:
            raise ValueError("route_abs_stats_path is required for joint Route B ego diffusion")

        # Register anchor centers
        anchor_path = config.get('anchor_path')
        if anchor_path:
            if anchor_path.endswith('.npy'):
                ac = np.load(anchor_path)
            else:
                import pickle
                with open(anchor_path, 'rb') as f:
                    ac = pickle.load(f)['centers']
            policy.register_anchor_centers(ac)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)

        # Load state dict — buffers are now non-None so they can be matched
        missing, unexpected = policy.load_state_dict(sd, strict=False)

        if missing:
            print(f"  [load_checkpoint] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  [load_checkpoint] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        policy = policy.to(device)
        policy.eval()
        return policy, ckpt

    def build_roll_timesteps(self, num_steps=None, device=None):
        """Build DDIM rollout timesteps for inference.

        The existing multi-step schedule is kept unchanged. For 1-step inference,
        start from the highest training noise level instead of the degenerate t=0.
        """
        if num_steps is None:
            num_steps = self.num_inference_steps
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        max_t = int(self.train_max_timesteps)
        if max_t <= 0:
            raise ValueError(f"train_max_timesteps must be positive, got {max_t}")

        if num_steps == 1:
            roll_timesteps = np.array([max_t - 1], dtype=np.int64)
        else:
            step_ratio = max_t / num_steps
            roll_timesteps = (
                np.arange(0, num_steps) * step_ratio
            ).round()[::-1].copy().astype(np.int64)
            roll_timesteps = np.clip(roll_timesteps, 0, max_t - 1)

        roll_timesteps = torch.from_numpy(roll_timesteps)
        if device is not None:
            roll_timesteps = roll_timesteps.to(device)
        return roll_timesteps

    # ========== Inference with Energy Guidance ==========
    @torch.no_grad()
    def conditional_sample(
        self,
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        ego_status: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype,
        route_for_guidance: Optional[torch.Tensor] = None,
        energy_weights: Optional[Dict[str, float]] = None,
        transfuser_lidar_bev: Optional[torch.Tensor] = None,
    ):
        """
        DDIM from N(0,I) with annealed energy gradient guidance.

        Args:
            route_for_guidance: (B, num_waypoints, 2) optional route for target energy
            energy_weights: optional dict to override energy weights (LLM Router interface)

        Returns:
            dict with best_trajectory, route_pred, all_trajectories, energy_scores, etc.
        """
        B = transfuser_bev_feature.shape[0]
        M = self.num_samples
        T = self.horizon
        joint_T = self.horizon + self.num_waypoints
        if M != 1:
            raise NotImplementedError(f"Joint Route B ego diffusion currently expects num_samples=1, got {M}")
        self._require_route_abs_stats()

        # Dynamic weight override (LLM Router interface — runtime per-head weight control)
        def _w(key, default):
            return energy_weights.get(key, default) if energy_weights else default
        w_front_cfg = _w('front',      self.energy_front_weight)
        w_left_cfg  = _w('left',       self.energy_left_weight)
        w_right_cfg = _w('right',      self.energy_right_weight)
        w_ped_cfg   = _w('pedestrian', self.energy_pedestrian_weight)
        w_off_cfg   = _w('offroad',    self.energy_offroad_weight)

        # Start from pure Gaussian noise in joint normalized traj+route space
        x_t = torch.randn(B, M, joint_T, 2, device=device, dtype=torch.float32)
        bev_proj = self.model.decoder.compute_bev_proj(
            transfuser_bev_feature.to(device=device, dtype=model_dtype)
        )

        # Set up DDIM timestep schedule
        num_steps = self.num_inference_steps
        roll_timesteps = self.build_roll_timesteps(num_steps=num_steps, device=device)

        alphas_cumprod = self.diffusion_scheduler.alphas_cumprod.to(device)

        poses_cls = None
        route_pred = None
        energy_scores = None
        speed_pred = None

        for step_i, k in enumerate(roll_timesteps):
            t_cur = k.item()
            t_next = roll_timesteps[step_i + 1].item() if step_i + 1 < len(roll_timesteps) else 0

            # Get annealed energy weights for current noise level
            _, w_veh, w_off = get_energy_weights(t_cur, T=self.train_max_timesteps)

            # ========== Forward pass 1: denoise x_t → pred_x0 ==========
            x_input = x_t.to(dtype=model_dtype)
            x_t_abs = self.joint_norm_to_abs(x_input)

            t_tensor = torch.full((B,), t_cur, dtype=torch.long, device=device)

            use_guidance = self.guidance_scale > 0 and (w_veh + w_off) > 0

            if use_guidance:
                # Pass 1: get pred_x0 from denoising (no energy eval yet)
                with torch.no_grad():
                    poses_reg, route_pred, _, _, speed_pred = self.model.forward_ego(
                        x_t=x_input,
                        x_t_abs=x_t_abs,
                        timestep=t_tensor,
                        transfuser_bev_feature=transfuser_bev_feature,
                        transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                        ego_status=ego_status,
                        bev_proj_cached=bev_proj,
                        transfuser_lidar_bev=transfuser_lidar_bev,
                    )
                    pred_x0 = poses_reg.detach()  # (B, M, T, 2)
                    route_context = route_for_guidance if route_for_guidance is not None else self.route_norm_to_abs(route_pred.detach())
                    route_pred_norm = route_pred.detach().unsqueeze(1)  # (B, 1, T_route, 2)

                # Pass 2: re-embed pred_x0 as a trajectory-level energy-eval sample.
                pred_x0_for_grad = pred_x0.clone().requires_grad_(True)

                with torch.enable_grad():
                    pred_x0_abs = self.norm_to_abs(pred_x0_for_grad)  # differentiable: z-denorm + cumsum
                    front_route_scores = None
                    needs_legacy_energy = (
                        (not self.use_front_route_risk_energy and w_front_cfg != 0)
                        or w_left_cfg != 0
                        or w_right_cfg != 0
                        or w_ped_cfg != 0
                        or (w_off_cfg != 0 and w_off > 0)
                    )
                    if needs_legacy_energy:
                        energy_scores, _ = self.model.forward_energy_eval(
                            x_t=pred_x0_for_grad,
                            x_t_abs=pred_x0_abs,
                            transfuser_bev_feature=transfuser_bev_feature,
                            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                            ego_status=ego_status,
                            traj_for_energy=pred_x0_abs,  # abs space for spatial energy evaluation
                            bev_proj_cached=bev_proj,
                            route_points=route_context,
                            transfuser_lidar_bev=transfuser_lidar_bev,
                        )
                    else:
                        energy_scores = None
                    if self.use_front_route_risk_energy and w_front_cfg != 0:
                        front_route_scores, _ = self.model.forward_front_route_risk_eval(
                            x_t=pred_x0_for_grad,
                            x_t_abs=pred_x0_abs,
                            transfuser_bev_feature=transfuser_bev_feature,
                            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                            ego_status=ego_status,
                            traj_for_energy=pred_x0_abs,
                            bev_proj_cached=bev_proj,
                            route_points=route_context,
                            transfuser_lidar_bev=transfuser_lidar_bev,
                        )

                    # Compute total energy
                    total_energy = torch.zeros(1, device=device)
                    if energy_scores is not None:
                        total_energy = (
                            w_veh * (
                                ((0.0 if self.use_front_route_risk_energy else w_front_cfg) * energy_scores['front'].sum())
                                + w_left_cfg  * energy_scores['left'].sum()
                                + w_right_cfg * energy_scores['right'].sum()
                                + w_ped_cfg   * energy_scores['pedestrian'].sum()
                            )
                            + w_off_cfg * w_off * energy_scores['offroad'].sum()
                        )
                    if front_route_scores is not None:
                        total_energy = total_energy + w_veh * w_front_cfg * front_route_scores.sum()

                    # Gradient w.r.t. pred_x0 with clipping
                    if total_energy.requires_grad and pred_x0_for_grad.requires_grad:
                        grad = torch.autograd.grad(total_energy, pred_x0_for_grad, allow_unused=True)[0]
                        if grad is None:
                            grad = torch.zeros_like(pred_x0_for_grad)
                        grad = grad.detach().to(dtype=torch.float32)
                        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        max_norm = self.energy_grad_clip_norm
                        grad = grad * torch.clamp(max_norm / grad_norm, max=1.0)
                    else:
                        grad = torch.zeros_like(pred_x0_for_grad)

                # Correct pred_x0, then DDIM step
                pred_x0_corrected = torch.cat([
                    pred_x0.float() - self.guidance_scale * grad,
                    route_pred_norm.float(),
                ], dim=2)
                poses_cls = None
            else:
                with torch.no_grad():
                    poses_reg, route_pred, _, _, speed_pred = self.model.forward_ego(
                        x_t=x_input,
                        x_t_abs=x_t_abs,
                        timestep=t_tensor,
                        transfuser_bev_feature=transfuser_bev_feature,
                        transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                        ego_status=ego_status,
                        bev_proj_cached=bev_proj,
                        transfuser_lidar_bev=transfuser_lidar_bev,
                    )
                energy_scores = None
                pred_x0_corrected = torch.cat([
                    poses_reg.float(),
                    route_pred.unsqueeze(1).float(),
                ], dim=2)

            # ========== DDIM Step with corrected pred_x0 ==========
            alpha_t = alphas_cumprod[t_cur]
            alpha_next = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)

            pred_eps = (x_t - alpha_t.sqrt() * pred_x0_corrected) / (1 - alpha_t).sqrt().clamp(min=1e-8)
            x_t = alpha_next.sqrt() * pred_x0_corrected + (1 - alpha_next).sqrt() * pred_eps

        # ========== Output trajectory ==========
        final_joint_abs = self.joint_norm_to_abs(pred_x0_corrected)  # (B, 1, T_joint, 2)
        final_traj_abs = final_joint_abs[:, :, :self.horizon, :]
        route_pred = final_joint_abs[:, 0, self.horizon:, :]
        best_trajectory = final_traj_abs.squeeze(1)  # (B, T, 2)

        # Decode speed prediction to scalar m/s
        target_speed_pred = None
        if speed_pred is not None:
            target_speed_pred = self.decode_speed_two_hot(
                speed_pred, self.model.speed_classes
            )  # (B,)

        return {
            'best_trajectory': best_trajectory,       # (B, T, 2)
            'route_pred': route_pred,                 # (B, 20, 2)
            'all_trajectories': final_traj_abs,       # (B, 1, T, 2)
            'energy_scores': energy_scores,           # dict of (B, 1)
            'poses_cls': poses_cls,                   # (B, 1)
            'best_idx': torch.zeros(B, dtype=torch.long, device=device),  # always 0
            'target_speed': target_speed_pred,        # (B,) m/s
        }

    # ========== Predict Action (standard interface) ==========
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        nobs = dict_apply(obs_dict, lambda x: x.to(device))

        transfuser_bev_feature = nobs['transfuser_bev_feature'].to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = nobs['transfuser_bev_feature_upsample'].to(device=device, dtype=model_dtype)
        transfuser_lidar_bev = self._get_transfuser_lidar_bev(nobs, device, model_dtype)
        ego_status = nobs['ego_status'].to(dtype=model_dtype)

        route_for_guidance = nobs.get('route', None)
        if route_for_guidance is not None:
            route_for_guidance = route_for_guidance.to(dtype=model_dtype)

        # Accept dynamic energy weights from kwargs (LLM Router interface)
        energy_weights = kwargs.get('energy_weights', None)

        sample_result = self.conditional_sample(
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
            device=device,
            model_dtype=model_dtype,
            route_for_guidance=route_for_guidance,
            energy_weights=energy_weights,
            transfuser_lidar_bev=transfuser_lidar_bev,
        )

        best_traj = sample_result['best_trajectory']
        action_pred = best_traj[..., :self.action_dim].detach().float().cpu().numpy()

        result = {
            'action': action_pred,
            'action_pred': action_pred,
            'route_pred': sample_result['route_pred'],
            'all_trajectories': sample_result['all_trajectories'].detach().float().cpu().numpy(),
            'best_idx': sample_result['best_idx'].detach().cpu().numpy(),
        }

        # Add predicted target speed (scalar m/s)
        if sample_result.get('target_speed') is not None:
            result['target_speed'] = sample_result['target_speed'].detach().float().cpu().numpy()

        # Add energy scores if available
        if sample_result['energy_scores'] is not None:
            es = sample_result['energy_scores']
            for key in ('front', 'left', 'right', 'pedestrian', 'offroad', 'route'):
                if key in es:
                    result[f'energy_{key}'] = es[key].detach().float().cpu().numpy()

        return result
