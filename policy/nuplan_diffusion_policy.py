"""White-Noise Diffusion Policy for nuPlan.

Wraps NuPlanDiffusionModel with DDIM scheduler, normalization,
training loss, and inference logic.

Key design:
- Start from pure Gaussian noise N(0, I) in normalized space
- DDIM denoising with prediction_type="sample" (predict x0)
- Joint prediction: ego (index 0) + N neighbors
- Global z-score normalization per [x, y] dimensions
- L1 loss for trajectory regression
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Union, List

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from model.nuplan_diffusion_model import NuPlanDiffusionModel


# =============================================================================
# Normalization helpers
# =============================================================================

def compute_global_norm_stats(data_iterator, key='ego_future', dim=2):
    """Compute global mean/std for z-score normalization.

    Args:
        data_iterator: yields data tuples (see NuPlanDataset.__getitem__)
        key: which output field to compute stats on (index into tuple)
        dim: feature dimension to aggregate

    Returns:
        mean: (dim,) array
        std: (dim,) array
    """
    all_data = []
    for item in data_iterator:
        # item is a tuple; ego_future is at index 1
        traj = item[1]  # (T, 2) numpy
        all_data.append(traj.reshape(-1, dim))

    all_data = np.concatenate(all_data, axis=0)  # (N, dim)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


# =============================================================================
# Policy
# =============================================================================

class NuPlanDiffusionPolicy(nn.Module):
    """White-noise diffusion policy for nuPlan trajectory prediction.

    Training:
        Given ego-centric scene context, denoise a randomly-noised version
        of the ground-truth trajectory. L1 loss over x_start prediction.

    Inference:
        Start from N(0,I), run DDIM denoising for num_inference_steps,
        output the clean trajectory.
    """

    def __init__(self, config: dict):
        super().__init__()

        policy_cfg = config.get('policy', {})
        self.n_obs_steps = policy_cfg.get('n_obs_steps', 1)

        # Model
        self.model = NuPlanDiffusionModel(config)

        # Diffusion scheduler
        self.num_train_timesteps = config.get('num_train_timesteps', 1000)
        self.num_inference_steps = config.get('num_inference_steps', 10)

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule='scaled_linear',
            clip_sample=False,
            prediction_type='sample',  # predict x0 directly
        )

        # Normalization (loaded from stats file)
        self.global_abs_mean = nn.Parameter(
            torch.zeros(2), requires_grad=False
        )
        self.global_abs_std = nn.Parameter(
            torch.ones(2), requires_grad=False
        )

        # Ego future key used by model output reshaping
        self.horizon = config.get('future_len', 80)
        self.predicted_neighbor_num = config.get('predicted_neighbor_num', 10)
        self.P = 1 + self.predicted_neighbor_num
        self.output_dim = self.model.output_dim

    # =========================================================================
    # Normalization
    # =========================================================================

    def abs_to_norm(self, traj_abs: torch.Tensor) -> torch.Tensor:
        """Convert absolute trajectory to normalized space.

        Args:
            traj_abs: (..., dim) in absolute coordinates (meters)

        Returns:
            (..., dim) in normalized [-1, 1] space
        """
        return (traj_abs - self.global_abs_mean) / self.global_abs_std

    def norm_to_abs(self, traj_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized trajectory to absolute space."""
        return traj_norm * self.global_abs_std + self.global_abs_mean

    def load_norm_stats(self, norm_stats: dict):
        """Load normalization statistics.

        Args:
            norm_stats: dict with keys 'mean' and 'std', each (2,) array.
        """
        self.global_abs_mean.copy_(
            torch.tensor(norm_stats['mean'], dtype=torch.float32)
        )
        self.global_abs_std.copy_(
            torch.tensor(norm_stats['std'], dtype=torch.float32)
        )

    # =========================================================================
    # Training
    # =========================================================================

    def forward(self, batch: tuple) -> dict:
        """DDP-friendly training forward that delegates to compute_loss."""
        return self.compute_loss(batch)

    def compute_loss(self, batch: tuple) -> dict:
        """Compute diffusion denoising loss.

        Args:
            batch: tuple from NuPlanDataset: (ego_current, ego_future,
                   neighbor_past, neighbor_future, lanes, lanes_sl, lanes_hsl,
                   route_lanes, static_objs)

        Returns:
            dict with keys: 'loss', 'ego_loss', 'neighbor_loss'
        """
        (ego_current, ego_future, neighbor_past, neighbor_future,
         lanes, lanes_sl, lanes_hsl, route_lanes, static_objs) = batch

        B = ego_current.shape[0]
        device = ego_current.device

        # Build GT trajectories: [current_state, future_1, ..., future_T]
        # Ego: (B, 1, T, 2) -> normalized
        ego_future_norm = self.abs_to_norm(ego_future)  # (B, T, 2)

        # Pad ego_current to 4 dim: [x, y, cos, sin]
        ego_cur = ego_current  # (B, 4)

        # Get neighbor current from past (last timestep, first 4 dims)
        neighbor_cur = neighbor_past[:, :self.predicted_neighbor_num, -1, :4]  # (B, N_pred, 4)
        neighbor_cur_mask = torch.sum(torch.ne(neighbor_cur, 0), dim=-1) == 0  # (B, N_pred)

        # Neighbor futures: (B, N_pred, T, 4) [x, y, cos, sin]
        neighbor_future_norm = neighbor_future.clone()
        neighbor_future_norm[..., :2] = self.abs_to_norm(neighbor_future[..., :2])

        # Mask zero-padded neighbors
        neighbor_future_mask = neighbor_cur_mask.unsqueeze(-1).unsqueeze(-1)  # (B, N_pred, 1, 1)

        # Stack ego + neighbor: (B, P, T, 4)
        ego_future_expanded = torch.cat([
            ego_future_norm,
            torch.zeros(B, self.horizon, 2, device=device)  # cos/sin = 0
        ], dim=-1).unsqueeze(1)  # (B, 1, T, 4)

        gt_future = torch.cat([ego_future_expanded, neighbor_future_norm], dim=1)  # (B, P, T, 4)

        # Current states: (B, P, 4)
        cur_states = torch.cat([ego_cur.unsqueeze(1), neighbor_cur], dim=1)  # (B, P, 4)

        # Full trajectory: (B, P, T+1, 4)
        all_gt = torch.cat([cur_states.unsqueeze(2), gt_future], dim=2)  # (B, P, T+1, 4)

        # Sample noise and timestep for future states only.
        noise = torch.randn_like(gt_future)  # (B, P, T, 4)
        timesteps = torch.randint(
            0, self.num_train_timesteps, (B,), device=device
        ).long()

        # Forward diffusion: keep current states fixed and only noise future states.
        # scheduler.add_noise expects (B, ...) so we flatten the participant dimension temporarily.
        noise_flat = noise.reshape(B * self.P, -1)
        gt_future_flat = gt_future.reshape(B * self.P, -1)

        noisy_future_flat = self.noise_scheduler.add_noise(
            gt_future_flat,
            noise_flat,
            timesteps.repeat_interleave(self.P),
        )
        noisy_future = noisy_future_flat.reshape(B, self.P, self.horizon, 4)
        noisy = torch.cat([cur_states.unsqueeze(2), noisy_future], dim=2)

        # Model forward
        model_inputs = {
            'neighbor_agents_past': neighbor_past,
            'static_objects': static_objs,
            'lanes': lanes,
            'lanes_speed_limit': lanes_sl,
            'lanes_has_speed_limit': lanes_hsl,
            'route_lanes': route_lanes,
            'sampled_trajectories': noisy.reshape(B, self.P, -1),
            'diffusion_time': timesteps,
            'neighbor_current_mask': neighbor_cur_mask,
        }

        outputs = self.model(model_inputs)
        pred = outputs['score']  # (B, P, T+1, 4)

        # L1 loss (predicted x0 vs GT)
        loss_all = F.l1_loss(pred, all_gt, reduction='none')  # (B, P, T+1, 4)
        loss_all = loss_all.mean(dim=[-2, -1])  # (B, P) mean over time and features

        # Split into ego and neighbor
        ego_loss = loss_all[:, 0].mean()  # scalar
        neighbor_loss = loss_all[:, 1:]  # (B, N_pred)
        # Mask padded neighbors
        neighbor_loss = neighbor_loss[~neighbor_cur_mask].mean() if (~neighbor_cur_mask).sum() > 0 else torch.tensor(0.0, device=device)

        total_loss = ego_loss + neighbor_loss

        return {
            'loss': total_loss,
            'ego_loss': ego_loss,
            'neighbor_loss': neighbor_loss,
        }

    # =========================================================================
    # Inference
    # =========================================================================

    @torch.no_grad()
    def conditional_sample(
        self,
        cond_data: dict,
        num_steps: Optional[int] = None,
    ) -> dict:
        """DDIM denoising from pure noise N(0,I).

        Args:
            cond_data: dict with scene context (neighbor_past, lanes, etc.)
            num_steps: override inference steps

        Returns:
            dict with:
                'trajectory': (B, P, T, 4) predicted future trajectories
                'final_noise': optional
        """
        if num_steps is None:
            num_steps = self.num_inference_steps

        neighbor_past = cond_data['neighbor_agents_past']
        B = neighbor_past.shape[0]
        device = neighbor_past.device

        # Derive current state from last timestep
        ego_cur = torch.zeros(B, 4, device=device)  # (B, 4) in normalized space: x=0, y=0, cos=1, sin=0
        ego_cur[:, 2] = 1.0  # cos = 1
        neighbor_cur = neighbor_past[:, :self.predicted_neighbor_num, -1, :4]  # (B, N_pred, 4)
        neighbor_cur_mask = torch.sum(torch.ne(neighbor_cur, 0), dim=-1) == 0

        cur_states = torch.cat([ego_cur.unsqueeze(1), neighbor_cur], dim=1)  # (B, P, 4)

        # Start from pure noise for future steps
        noise = torch.randn(B, self.P, self.horizon, 4, device=device)
        x_t = torch.cat([cur_states.unsqueeze(2), noise], dim=2)  # (B, P, T+1, 4)

        # Setup scheduler
        self.noise_scheduler.set_timesteps(num_steps)

        for t in self.noise_scheduler.timesteps:
            # Model forward — predict x0
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            model_inputs = {
                'neighbor_agents_past': neighbor_past,
                'static_objects': cond_data.get('static_objects', torch.zeros(B, 5, 10, device=device)),
                'lanes': cond_data.get('lanes', torch.zeros(B, 30, 20, 12, device=device)),
                'lanes_speed_limit': cond_data.get('lanes_speed_limit', torch.zeros(B, 30, 1, device=device)),
                'lanes_has_speed_limit': cond_data.get('lanes_has_speed_limit', torch.zeros(B, 30, 1, device=device, dtype=torch.bool)),
                'route_lanes': cond_data.get('route_lanes', torch.zeros(B, 10, 20, 4, device=device)),
                'sampled_trajectories': x_t.reshape(B, self.P, -1),
                'diffusion_time': t_batch,
                'neighbor_current_mask': neighbor_cur_mask,
            }

            outputs = self.model(model_inputs)
            pred_x0 = outputs['score']  # (B, P, T+1, 4)

            # DDIM step
            x_t_flat = x_t.reshape(B * self.P, -1)
            pred_flat = pred_x0.reshape(B * self.P, -1)
            x_next_flat = self.noise_scheduler.step(
                pred_flat, t, x_t_flat, return_dict=False
            )[0]
            x_t = x_next_flat.reshape(B, self.P, self.horizon + 1, 4)

            # Re-apply current state constraint (first frame fixed)
            x_t[:, :, 0, :] = cur_states

        # Remove current frame, return future only
        trajectory = x_t[:, :, 1:, :]  # (B, P, T, 4)

        # Denormalize ego trajectory
        trajectory_norm = trajectory.clone()
        trajectory_norm[:, 0, :, :2] = self.norm_to_abs(trajectory[:, 0, :, :2])

        # Denormalize neighbor trajectories
        trajectory_norm[:, 1:, :, :2] = self.norm_to_abs(trajectory[:, 1:, :, :2])

        return {'trajectory': trajectory_norm}

    # =========================================================================
    # Optimizer
    # =========================================================================

    def configure_optimizers(self, training_config: dict, steps_per_epoch: int = 1):
        """Configure AdamW optimizer with weight decay.

        Args:
            training_config: dict with learning_rate/lr, weight_decay, etc.
            steps_per_epoch: number of optimizer steps per epoch.

        Returns:
            (optimizer, scheduler) tuple
        """
        lr = training_config.get('learning_rate', training_config.get('lr', 5e-4))
        weight_decay = training_config.get('weight_decay', 0.01)
        warmup_epochs = training_config.get('warmup_epochs', 5)
        total_epochs = training_config.get('train_epochs', 500)
        final_lr = training_config.get(
            'final_learning_rate',
            training_config.get('min_learning_rate', 0.0),
        )
        final_lr = max(0.0, float(final_lr))
        final_lr_ratio = min(final_lr / float(lr), 1.0) if lr > 0 else 0.0
        steps_per_epoch = max(1, int(steps_per_epoch))
        warmup_steps = max(0, int(warmup_epochs * steps_per_epoch))
        total_steps = max(1, int(total_epochs * steps_per_epoch))

        # Separate parameters that need weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'LayerNorm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=lr,
        )

        # Cosine decay with linear warmup
        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            if total_steps <= warmup_steps:
                return 1.0
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return final_lr_ratio + (1.0 - final_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scheduler.lr_schedule_unit = 'step'
        scheduler.total_steps = total_steps
        scheduler.warmup_steps = warmup_steps
        scheduler.final_learning_rate = final_lr
        scheduler.final_lr_ratio = final_lr_ratio

        return optimizer, scheduler
