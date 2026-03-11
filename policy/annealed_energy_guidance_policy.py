"""
Route B: Annealed Energy Guidance Policy

Pure diffusion from N(0,I) with 10-step DDIM and classifier guidance.
Energy gradients are injected at each denoising step with time-scale scheduling:
  - High noise (t=100->70): Navigation energy pulls trajectories to macro direction
  - Medium noise (t=70->30): Collision energy applies repulsion
  - Low noise (t=30->0): Offroad energy + smoothness for lane-level polish

Key differences from Route A (DiffusionDiTCarlaPolicy):
  - No anchor centers — start from pure Gaussian noise
  - Model predicts absolute trajectory (no residual from anchor)
  - 10-step DDIM inference with energy gradient guidance
  - Energy heads trained on semantic behavior labels
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
# Route B Policy
# =============================================================================

class AnnealedEnergyGuidancePolicy(nn.Module):
    """
    Annealed Energy Guidance policy for trajectory prediction.

    Training:
      - Standard diffusion: add noise to GT trajectory, model predicts clean x_0
      - Energy heads supervised by semantic behavior labels
      - Classification head supervised by L2 distance to GT

    Inference:
      - Start from pure Gaussian noise N(0, I) in normalized [-1, 1] space
      - 10-step DDIM denoising
      - At each step, compute energy gradients and inject them with time-scale weights
      - Select best trajectory from num_samples candidates
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
    def forward(self, batch: Dict[str, torch.Tensor], return_loss_dict: bool = False):
        loss_dict = self.compute_loss(batch)
        if return_loss_dict:
            return loss_dict
        else:
            return loss_dict['total_loss']

    # ========== Training ==========
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Standard diffusion training + energy head supervision.

        Training procedure:
        1. Replicate GT trajectory across num_samples slots
        2. Sample random timestep, add noise to GT
        3. Model predicts clean x_0 from noisy input
        4. Losses: regression (L1), classification (focal), route (L1), energy (BCE/L1)
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

        # Semantic behavior labels for energy supervision
        behavior_labels = batch.get('behavior_labels', None)  # (B, M) or None
        allowed_flags = batch.get('allowed_flags', None)      # (B, M) or None

        # ========== Prepare noisy trajectories ==========
        # Normalize GT to [-1, 1]
        traj_normed = self.norm_odo(trajectory)  # (B, T, 2)

        # Replicate GT across num_samples slots: (B, M, T, 2)
        M = self.num_samples
        traj_normed_expanded = traj_normed.unsqueeze(1).expand(-1, M, -1, -1)

        # Sample timestep
        timesteps = torch.randint(0, self.train_max_timesteps, (B,), device=device).long()

        # Add noise
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
        # In anchor-free mode, model receives noisy trajectories directly
        # BEV grid_sample uses denormalized (absolute) coordinates
        noisy_traj_abs = self.denorm_odo(noisy_traj)

        poses_reg, poses_cls, route_pred, mode_out, energy_scores = self.model(
            anchors=noisy_traj,          # normalized noisy trajectories
            anchors_abs=noisy_traj_abs,  # absolute coords for BEV grid_sample
            timestep=timesteps,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
        )

        # Denorm predictions to absolute space
        poses_reg_abs = self.denorm_odo(poses_reg)  # (B, M, T, 2)

        # ========== Regression Loss ==========
        # All samples should predict the same GT trajectory
        traj_expanded = trajectory.unsqueeze(1).expand(-1, M, -1, -1)  # (B, M, T, 2)
        loss_reg = F.l1_loss(poses_reg_abs, traj_expanded, reduction='mean')

        # ========== Classification Loss ==========
        # In anchor-free training, all samples see the same noisy GT,
        # but with different noise realizations. The best sample is the one
        # closest to GT in prediction space.
        dist_per_sample = (poses_reg_abs - traj_expanded).norm(dim=-1).mean(dim=-1)  # (B, M)
        best_idx = dist_per_sample.argmin(dim=-1)  # (B,)
        target_onehot = torch.zeros(B, M, device=device, dtype=model_dtype)
        target_onehot.scatter_(1, best_idx.unsqueeze(1), 1)
        loss_cls = self._focal_loss(poses_cls, target_onehot)

        # ========== Route Loss ==========
        route_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        if route_gt is not None and route_pred is not None:
            route_loss = F.l1_loss(route_pred, route_gt, reduction='mean')

        # ========== Energy Loss ==========
        energy_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        if energy_scores is not None and behavior_labels is not None and allowed_flags is not None:
            behavior_labels_dev = behavior_labels.to(device=device)
            allowed_flags_dev = allowed_flags.to(device=device, dtype=model_dtype)

            # Collision target: behavior labels 1-4 indicate collision risk
            collision_target = ((behavior_labels_dev >= 1) & (behavior_labels_dev <= 4)).float()
            # Offroad target: behavior labels 5-6 indicate offroad
            offroad_target = ((behavior_labels_dev >= 5) & (behavior_labels_dev <= 6)).float()
            # Target distance: use 1 - allowed as proxy (allowed=1 means safe/on-target)
            target_proxy = 1.0 - allowed_flags_dev  # higher = worse navigation compliance

            loss_col = F.binary_cross_entropy(energy_scores['collision'], collision_target)
            loss_off = F.binary_cross_entropy(energy_scores['offroad'], offroad_target)
            loss_tgt = F.l1_loss(energy_scores['target'], target_proxy)
            energy_loss = loss_col + loss_off + loss_tgt

        # ========== Total Loss ==========
        total_loss = (
            self.reg_loss_weight * loss_reg
            + self.cls_loss_weight * loss_cls
            + self.route_loss_weight * route_loss
            + self.energy_loss_weight * energy_loss
        )

        loss_dict = {
            'total_loss': total_loss,
            'reg_loss': loss_reg,
            'cls_loss': loss_cls,
            'route_loss': route_loss,
            'energy_loss': energy_loss,
        }
        return loss_dict

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
    ):
        """
        10-step DDIM from N(0,I) with annealed energy gradient guidance.

        Args:
            route_for_guidance: (B, num_waypoints, 2) optional route for target energy

        Returns:
            (best_trajectory, route_pred) - trajectory in absolute coords
        """
        B = transfuser_bev_feature.shape[0]
        M = self.num_samples
        T = self.horizon

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

            # Get energy weights for current noise level
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
                        self.energy_target_weight * w_nav * energy_scores['target'].sum()
                        + self.energy_collision_weight * w_col * energy_scores['collision'].sum()
                        + self.energy_offroad_weight * w_off * energy_scores['offroad'].sum()
                    )

                # Compute gradient w.r.t. noisy input
                if total_energy.requires_grad:
                    grad = torch.autograd.grad(total_energy, x_for_grad)[0]
                    grad = grad.detach().to(dtype=torch.float32)
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
            # Model predicts clean x_0 (in normalized space since anchor_free)
            pred_x0_normed = poses_reg.float()  # (B, M, T, 2) in normalized space

            alpha_t = alphas_cumprod[t_cur]
            alpha_next = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)

            # Compute predicted noise
            pred_eps = (x_t - alpha_t.sqrt() * pred_x0_normed) / (1 - alpha_t).sqrt().clamp(min=1e-8)

            # DDIM deterministic update
            x_t = alpha_next.sqrt() * pred_x0_normed + (1 - alpha_next).sqrt() * pred_eps

            # Inject energy gradient guidance (subtract gradient to minimize energy)
            x_t = x_t - self.guidance_scale * grad

        # ========== Select best trajectory ==========
        # Use final prediction (not x_t) for trajectory selection
        final_abs = self.denorm_odo(torch.clamp(pred_x0_normed, -1, 1))  # (B, M, T, 2)

        # Select via classification scores, optionally filtered by energy
        if energy_scores is not None:
            # Energy shielding: penalize high-energy candidates
            safe_logits = (
                poses_cls
                - self.energy_collision_weight * energy_scores['collision']
                - self.energy_offroad_weight * energy_scores['offroad']
                - self.energy_target_weight * energy_scores['target']
            )
            best_idx = safe_logits.argmax(dim=-1)  # (B,)
        else:
            best_idx = poses_cls.argmax(dim=-1)  # (B,)

        mode_idx_expanded = best_idx.view(B, 1, 1, 1).expand(-1, 1, T, 2)
        best_trajectory = torch.gather(final_abs, 1, mode_idx_expanded).squeeze(1)  # (B, T, 2)

        return best_trajectory, route_pred

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

        nsample, route_pred = self.conditional_sample(
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
            device=device,
            model_dtype=model_dtype,
            route_for_guidance=route_for_guidance,
        )

        action_pred = nsample[..., :self.action_dim].detach().float().cpu().numpy()
        result = {
            'action': action_pred,
            'action_pred': action_pred,
            'route_pred': route_pred,
        }
        return result
