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

    Args:
        t: current timestep (higher = more noise)
        T: total timesteps

    Returns:
        (w_nav, w_col, w_off) weight tuple
    """
    progress = 1.0 - t / T  # 0 -> 1 as denoising progresses

    w_nav = 1.0                                       # always active
    w_col = max(0.0, (progress - 0.3) / 0.4)          # activates after t < 0.7*T
    w_off = max(0.0, (progress - 0.7) / 0.3)          # activates after t < 0.3*T

    return w_nav, w_col, w_off


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
        self.num_samples = route_b_cfg.get('num_samples', 32)  # number of noise candidates
        self.num_inference_steps = route_b_cfg.get('num_inference_steps', 10)
        self.guidance_scale = route_b_cfg.get('guidance_scale', 1.0)  # global energy guidance multiplier
        self.energy_collision_weight = route_b_cfg.get('energy_collision_weight', 1.0)
        self.energy_offroad_weight = route_b_cfg.get('energy_offroad_weight', 1.0)
        self.energy_target_weight = route_b_cfg.get('energy_target_weight', 1.0)

        # Route B+ config
        self.energy_grad_clip_norm = route_b_cfg.get('energy_grad_clip_norm', 1.0)
        self.alignment_loss_weight = route_b_cfg.get('alignment_loss_weight', 0.1)
        self.num_gt_augmentations = route_b_cfg.get('num_gt_augmentations', 4)
        self.use_safe_anchors = route_b_cfg.get('use_safe_anchors', False)
        self.energy_noisy_training = route_b_cfg.get('energy_noisy_training', False)

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
            num_modes=self.num_samples,
            traj_can_attend_route=policy_cfg.get('traj_can_attend_route', True),
            anchor_free=True,
            energy_heads=True,
        )
        self.model = model

        # ========== Diffusion Configuration ==========
        diffusion_cfg = config.get('truncated_diffusion', {})
        self.num_train_timesteps = diffusion_cfg.get('num_train_timesteps', 1000)
        # Route B uses full diffusion range (not truncated)
        self.train_max_timesteps = route_b_cfg.get('train_max_timesteps',
                                                    diffusion_cfg.get('trunc_timesteps', 100))
        self.prediction_type = diffusion_cfg.get('prediction_type', 'sample')

        # Coordinate normalization to [-1, 1]
        self.norm_x_offset = diffusion_cfg.get('norm_x_offset', 2.0)
        self.norm_x_range = diffusion_cfg.get('norm_x_range', 80.0)
        self.norm_y_offset = diffusion_cfg.get('norm_y_offset', 20.0)
        self.norm_y_range = diffusion_cfg.get('norm_y_range', 56.0)

        # Loss weights
        self.cls_loss_weight = config.get('cls_loss_weight', 0.5)
        self.reg_loss_weight = config.get('reg_loss_weight', 1.0)
        self.route_loss_weight = diffusion_cfg.get('route_loss_weight', 0.5)
        self.energy_loss_weight = route_b_cfg.get('energy_loss_weight', 1.0)

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
        self.register_buffer('anchor_centers_abs', anchor_centers_abs)

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

    # ========== Normalization ==========
    def norm_odo(self, odo: torch.Tensor) -> torch.Tensor:
        x = odo[..., 0:1]
        y = odo[..., 1:2]
        x = 2 * (x + self.norm_x_offset) / self.norm_x_range - 1
        y = 2 * (y + self.norm_y_offset) / self.norm_y_range - 1
        return torch.cat([x, y], dim=-1)

    def denorm_odo(self, odo: torch.Tensor) -> torch.Tensor:
        x = odo[..., 0:1]
        y = odo[..., 1:2]
        x = (x + 1) / 2 * self.norm_x_range - self.norm_x_offset
        y = (y + 1) / 2 * self.norm_y_range - self.norm_y_offset
        return torch.cat([x, y], dim=-1)

    # ========== Focal Loss (same as Route A) ==========
    def _focal_loss(self, logits, targets, gamma=2.0, alpha=0.25):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = alpha * (1 - p_t) ** gamma
        return (focal_weight * bce).mean()

    # ========== Forward (DDP-compatible) ==========
    def forward(self, batch: Dict[str, torch.Tensor],
                return_loss_dict: bool = False,
                phase: str = 'diffusion'):
        """
        DDP-compatible forward. Use `phase` to select training phase:
          - 'energy': Phase 1 — train energy heads on anchors + GT augmentation
          - 'diffusion': Phase 2 — train decoder (diffusion loss + alignment)
        """
        if phase == 'energy':
            loss_dict = self.compute_energy_loss(batch)
        else:
            loss_dict = self.compute_diffusion_loss(batch)

        if return_loss_dict:
            return loss_dict
        else:
            return loss_dict['total_loss']

    # ========== Phase 1: Energy Head Training ==========
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
        M = self.num_samples

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
                'energy_col_loss': zero.detach(),
                'energy_off_loss': zero.detach(),
                'energy_tgt_loss': zero.detach(),
            }

        # --- Build mixed input: GT augmentation + anchors ---
        anchor_abs = self.anchor_centers_abs.unsqueeze(0).expand(B, -1, -1, -1).clone()  # (B, M, T, 2)
        behavior_labels_dev = behavior_labels.to(device=device).clone()
        allowed_flags_dev = allowed_flags.to(device=device, dtype=model_dtype).clone()

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
            anchor_normed = self.norm_odo(anchor_abs)
            anchor_flat = anchor_normed.contiguous().view(B * M, T, D)
            t_expanded = timesteps.unsqueeze(1).expand(-1, M).reshape(B * M)
            noise = torch.randn_like(anchor_flat)
            noisy_anchor = self.diffusion_scheduler.add_noise(anchor_flat, noise, t_expanded)
            noisy_anchor = noisy_anchor.view(B, M, T, D).clamp(-1, 1)
            anchor_input = noisy_anchor
            anchor_abs_input = self.denorm_odo(anchor_input)
        else:
            timesteps = torch.zeros(B, device=device, dtype=torch.long)
            anchor_input = self.norm_odo(anchor_abs)
            anchor_abs_input = anchor_abs

        # --- Forward pass WITH gradients ---
        _, _, _, _, energy_scores = self.model(
            anchors=anchor_input,
            anchors_abs=anchor_abs_input,
            timestep=timesteps,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
        )

        # --- Build supervision targets ---
        collision_target = ((behavior_labels_dev >= 1) & (behavior_labels_dev <= 4)).float()
        offroad_target = ((behavior_labels_dev >= 5) & (behavior_labels_dev <= 6)).float()
        target_proxy = 1.0 - allowed_flags_dev

        # --- Compute energy loss with optional masking ---
        if not self.use_safe_anchors:
            # Only train on: GT augmentation (first K, safe) + forbidden anchors (rest)
            # Skip allowed anchors (potentially unreliable)
            active_mask = torch.ones(B, M, device=device, dtype=torch.bool)
            allowed_flags_original = allowed_flags.to(device=device)
            active_mask[:, K:] = (allowed_flags_original[:, K:] < 0.5)  # keep forbidden only

            n_active = active_mask.sum()
            if n_active > 0:
                loss_col = F.smooth_l1_loss(
                    energy_scores['collision'][active_mask],
                    collision_target[active_mask],
                )
                loss_off = F.smooth_l1_loss(
                    energy_scores['offroad'][active_mask],
                    offroad_target[active_mask],
                )
                loss_tgt = F.smooth_l1_loss(
                    energy_scores['target'][active_mask],
                    target_proxy[active_mask],
                )
            else:
                loss_col = torch.tensor(0.0, device=device, dtype=model_dtype)
                loss_off = torch.tensor(0.0, device=device, dtype=model_dtype)
                loss_tgt = torch.tensor(0.0, device=device, dtype=model_dtype)
        else:
            # Trust all anchor labels (including safe ones)
            loss_col = F.smooth_l1_loss(energy_scores['collision'], collision_target)
            loss_off = F.smooth_l1_loss(energy_scores['offroad'], offroad_target)
            loss_tgt = F.smooth_l1_loss(energy_scores['target'], target_proxy)

        energy_loss = loss_col + loss_off + loss_tgt

        return {
            'total_loss': self.energy_loss_weight * energy_loss,
            'energy_loss': energy_loss,
            'energy_col_loss': loss_col,
            'energy_off_loss': loss_off,
            'energy_tgt_loss': loss_tgt,
        }

    # ========== Phase 2: Diffusion Training + Alignment ==========
    def compute_diffusion_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Standard diffusion training + alignment loss.

        Training procedure:
        1. Replicate GT trajectory across num_samples slots
        2. Sample random timestep, add noise to GT
        3. Model predicts clean x_0 from noisy input
        4. Losses: regression (L1), classification (focal), route (L1)
        5. Alignment: encourage decoder to produce low-energy trajectories
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        trajectory = batch['agent_pos'].to(device=device, dtype=model_dtype)  # (B, T, 2)
        B, T, D = trajectory.shape

        transfuser_bev_feature = batch['transfuser_bev_feature'].to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = batch['transfuser_bev_feature_upsample'].to(device=device, dtype=model_dtype)
        ego_status = batch['ego_status'].to(device=device, dtype=model_dtype)

        route_gt = batch.get('route', None)
        if route_gt is not None:
            route_gt = route_gt.to(device=device, dtype=model_dtype)

        # ========== Prepare noisy trajectories ==========
        traj_normed = self.norm_odo(trajectory)  # (B, T, 2)
        M = self.num_samples
        traj_normed_expanded = traj_normed.unsqueeze(1).expand(-1, M, -1, -1)

        timesteps = torch.randint(0, self.train_max_timesteps, (B,), device=device).long()

        B_M = B * M
        traj_flat = traj_normed_expanded.contiguous().view(B_M, T, D)
        timesteps_expanded = timesteps.unsqueeze(1).expand(-1, M).reshape(B_M)

        noise = torch.randn(traj_flat.shape, dtype=torch.float32, device=device)
        noisy_traj_flat = self.diffusion_scheduler.add_noise(
            original_samples=traj_flat,
            noise=noise,
            timesteps=timesteps_expanded,
        )
        noisy_traj = noisy_traj_flat.view(B, M, T, D)
        noisy_traj = torch.clamp(noisy_traj, -1, 1)

        # ========== Forward pass ==========
        noisy_traj_abs = self.denorm_odo(noisy_traj)

        poses_reg, poses_cls, route_pred, mode_out, energy_scores = self.model(
            anchors=noisy_traj,
            anchors_abs=noisy_traj_abs,
            timestep=timesteps,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
        )

        # Denorm predictions to absolute space
        poses_reg_abs = self.denorm_odo(poses_reg)  # (B, M, T, 2)

        # ========== Regression Loss ==========
        traj_expanded = trajectory.unsqueeze(1).expand(-1, M, -1, -1)
        loss_reg = F.l1_loss(poses_reg_abs, traj_expanded, reduction='mean')

        # ========== Classification Loss ==========
        dist_per_sample = (poses_reg_abs - traj_expanded).norm(dim=-1).mean(dim=-1)  # (B, M)
        best_idx = dist_per_sample.argmin(dim=-1)  # (B,)
        target_onehot = torch.zeros(B, M, device=device, dtype=model_dtype)
        target_onehot.scatter_(1, best_idx.unsqueeze(1), 1)
        loss_cls = self._focal_loss(poses_cls, target_onehot)

        # ========== Route Loss ==========
        route_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        if route_gt is not None and route_pred is not None:
            route_loss = F.l1_loss(route_pred, route_gt, reduction='mean')

        # ========== Alignment Loss ==========
        # Encourage decoder to generate low-energy (safe) trajectories
        alignment_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        if self.alignment_loss_weight > 0 and energy_scores is not None:
            alignment_loss = (
                self.energy_collision_weight * energy_scores['collision'].mean()
                + self.energy_offroad_weight * energy_scores['offroad'].mean()
                + self.energy_target_weight * energy_scores['target'].mean()
            )

        # ========== Total Loss ==========
        total_loss = (
            self.reg_loss_weight * loss_reg
            + self.cls_loss_weight * loss_cls
            + self.route_loss_weight * route_loss
            + self.alignment_loss_weight * alignment_loss
        )

        loss_dict = {
            'total_loss': total_loss,
            'reg_loss': loss_reg,
            'cls_loss': loss_cls,
            'route_loss': route_loss,
            'alignment_loss': alignment_loss,
        }
        return loss_dict

    # ========== Legacy compute_loss (backward compatible) ==========
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Backward compatible: calls compute_diffusion_loss."""
        return self.compute_diffusion_loss(batch)

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

        # Dynamic weight override (LLM Router interface)
        w_col_cfg = energy_weights.get('collision', self.energy_collision_weight) if energy_weights else self.energy_collision_weight
        w_off_cfg = energy_weights.get('offroad', self.energy_offroad_weight) if energy_weights else self.energy_offroad_weight
        w_tgt_cfg = energy_weights.get('target', self.energy_target_weight) if energy_weights else self.energy_target_weight

        # Start from pure Gaussian noise in normalized space
        x_t = torch.randn(B, M, T, 2, device=device, dtype=torch.float32)
        x_t = torch.clamp(x_t, -1, 1)

        # Set up DDIM timestep schedule
        num_steps = self.num_inference_steps
        step_ratio = self.train_max_timesteps / num_steps
        roll_timesteps = (np.arange(0, num_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        alphas_cumprod = self.diffusion_scheduler.alphas_cumprod.to(device)

        poses_cls = None
        route_pred = None
        energy_scores = None

        for step_i, k in enumerate(roll_timesteps):
            t_cur = k.item()
            t_next = roll_timesteps[step_i + 1].item() if step_i + 1 < len(roll_timesteps) else 0

            # Get annealed energy weights for current noise level
            w_nav, w_col, w_off = get_energy_weights(t_cur, T=self.train_max_timesteps)

            # ========== Forward pass (with gradients for energy guidance) ==========
            x_clamped = torch.clamp(x_t, -1, 1).to(dtype=model_dtype)
            x_abs = self.denorm_odo(x_clamped)

            # Enable gradients for energy guidance
            if self.guidance_scale > 0 and (w_nav + w_col + w_off) > 0:
                x_for_grad = x_clamped.detach().requires_grad_(True)
                x_abs_grad = self.denorm_odo(x_for_grad)

                t_tensor = torch.full((B,), t_cur, dtype=torch.long, device=device)
                poses_reg, poses_cls, route_pred, mode_out, energy_scores = self.model(
                    anchors=x_for_grad,
                    anchors_abs=x_abs_grad,
                    timestep=t_tensor,
                    transfuser_bev_feature=transfuser_bev_feature,
                    transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                    ego_status=ego_status,
                )

                # Compute total energy for gradient
                total_energy = torch.zeros(1, device=device)
                if energy_scores is not None:
                    total_energy = (
                        w_tgt_cfg * w_nav * energy_scores['target'].sum()
                        + w_col_cfg * w_col * energy_scores['collision'].sum()
                        + w_off_cfg * w_off * energy_scores['offroad'].sum()
                    )

                # Compute gradient with clipping
                if total_energy.requires_grad:
                    grad = torch.autograd.grad(total_energy, x_for_grad)[0]
                    grad = grad.detach().to(dtype=torch.float32)
                    # Per-element gradient clipping
                    grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    max_norm = self.energy_grad_clip_norm
                    grad = grad * torch.clamp(max_norm / grad_norm, max=1.0)
                else:
                    grad = torch.zeros_like(x_t)

                # Detach outputs for DDIM step
                poses_reg = poses_reg.detach()
                poses_cls = poses_cls.detach()
            else:
                t_tensor = torch.full((B,), t_cur, dtype=torch.long, device=device)
                with torch.no_grad():
                    poses_reg, poses_cls, route_pred, mode_out, energy_scores = self.model(
                        anchors=x_clamped,
                        anchors_abs=x_abs,
                        timestep=t_tensor,
                        transfuser_bev_feature=transfuser_bev_feature,
                        transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                        ego_status=ego_status,
                    )
                grad = torch.zeros_like(x_t)

            # ========== DDIM Step ==========
            pred_x0_normed = poses_reg.float()  # (B, M, T, 2) in normalized space

            alpha_t = alphas_cumprod[t_cur]
            alpha_next = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)

            pred_eps = (x_t - alpha_t.sqrt() * pred_x0_normed) / (1 - alpha_t).sqrt().clamp(min=1e-8)
            x_t = alpha_next.sqrt() * pred_x0_normed + (1 - alpha_next).sqrt() * pred_eps

            # Inject energy gradient guidance (subtract gradient to minimize energy)
            x_t = x_t - self.guidance_scale * grad

        # ========== Select best trajectory ==========
        final_abs = self.denorm_odo(torch.clamp(pred_x0_normed, -1, 1))  # (B, M, T, 2)

        # Energy shielding: penalize high-energy candidates
        if energy_scores is not None:
            safe_logits = (
                poses_cls
                - w_col_cfg * energy_scores['collision']
                - w_off_cfg * energy_scores['offroad']
                - w_tgt_cfg * energy_scores['target']
            )
            best_idx = safe_logits.argmax(dim=-1)  # (B,)
        else:
            safe_logits = poses_cls
            best_idx = poses_cls.argmax(dim=-1)  # (B,)

        mode_idx_expanded = best_idx.view(B, 1, 1, 1).expand(-1, 1, T, 2)
        best_trajectory = torch.gather(final_abs, 1, mode_idx_expanded).squeeze(1)  # (B, T, 2)

        # Return rich output for visualization
        return {
            'best_trajectory': best_trajectory,       # (B, T, 2)
            'route_pred': route_pred,                 # (B, 20, 2)
            'all_trajectories': final_abs,            # (B, M, T, 2)
            'energy_scores': energy_scores,           # dict of (B, M)
            'poses_cls': poses_cls,                   # (B, M)
            'safe_logits': safe_logits,               # (B, M)
            'best_idx': best_idx,                     # (B,)
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

        # Add energy scores if available
        if sample_result['energy_scores'] is not None:
            result['energy_collision'] = sample_result['energy_scores']['collision'].detach().float().cpu().numpy()
            result['energy_offroad'] = sample_result['energy_scores']['offroad'].detach().float().cpu().numpy()
            result['energy_target'] = sample_result['energy_scores']['target'].detach().float().cpu().numpy()

        return result
