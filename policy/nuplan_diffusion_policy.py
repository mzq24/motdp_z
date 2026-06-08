"""White-Noise Diffusion Policy for nuPlan.

Wraps NuPlanDiffusionModel with DDIM scheduler, normalization,
training loss, and inference logic.

Key design:
- Start from pure Gaussian noise N(0, I) in normalized space
- DDIM denoising with prediction_type="sample" (predict x0)
- Joint prediction: ego (index 0) + N neighbors
- Diffusion-Planner-style fixed normalization by default
- Joint ego + neighbor future denoising loss
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Union, List

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from model.nuplan_diffusion_model import NuPlanDiffusionModel
from model.nuplan_multisource_diffusion_model import NuPlanMultiSourceDiffusionModel


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
        # item is a tuple; ego_future is at index 1. Keep only xy if
        # heading is present.
        traj = item[1][..., :dim]
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
        of the ground-truth trajectory. The current state is a fixed
        condition; loss is computed on future states only.

    Inference:
        Start from N(0,I), run DDIM denoising for num_inference_steps,
        output the clean trajectory in meter space.
    """

    def __init__(self, config: dict):
        super().__init__()

        policy_cfg = config.get('policy', {})
        self.n_obs_steps = policy_cfg.get('n_obs_steps', 1)

        # Model
        self.model_type = str(config.get('model_type', 'simple')).lower()
        if self.model_type in {'n2_multisource', 'multisource', 'n2'}:
            self.model = NuPlanMultiSourceDiffusionModel(config)
        elif self.model_type in {'simple', 'dit', 'diffusion_planner'}:
            self.model = NuPlanDiffusionModel(config)
        else:
            raise ValueError(
                f"Unsupported model_type '{self.model_type}'. "
                "Expected simple or n2_multisource."
            )

        # Diffusion scheduler
        self.num_train_timesteps = config.get('num_train_timesteps', 1000)
        self.num_inference_steps = config.get('num_inference_steps', 10)

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule='scaled_linear',
            clip_sample=False,
            prediction_type='sample',  # predict x0 directly
        )

        # Diffusion-Planner uses a hand-tuned fixed state normalizer by
        # default: x/y are centered at [10, 0] and scaled by [20, 20], while
        # cos/sin are left unchanged. Keep the historical 2D parameter names
        # for checkpoint compatibility.
        norm_cfg = config.get('normalization', {}) or {}
        if not isinstance(norm_cfg, dict):
            norm_cfg = {}
        self.normalization_mode = str(
            config.get(
                'normalization_mode',
                norm_cfg.get('mode', 'diffusion_planner_fixed'),
            )
        )
        xy_mean = config.get('state_xy_mean', norm_cfg.get('xy_mean', [10.0, 0.0]))
        xy_std = config.get('state_xy_std', norm_cfg.get('xy_std', [20.0, 20.0]))
        if len(xy_mean) != 2 or len(xy_std) != 2:
            raise ValueError('state xy normalization mean/std must each have length 2')
        self.global_abs_mean = nn.Parameter(
            torch.tensor(xy_mean, dtype=torch.float32), requires_grad=False
        )
        self.global_abs_std = nn.Parameter(
            torch.tensor(xy_std, dtype=torch.float32), requires_grad=False
        )
        self.observation_scale = nn.Parameter(
            torch.tensor(float(config.get('observation_scale', norm_cfg.get('observation_scale', 20.0))), dtype=torch.float32),
            requires_grad=False,
        )
        self.normalize_observation_inputs = bool(
            config.get(
                'normalize_observations',
                self.normalization_mode == 'diffusion_planner_fixed',
            )
        )
        self.neighbor_loss_weight = float(config.get('neighbor_loss_weight', 1.0))
        self.predict_route_tokens = bool(getattr(self.model, 'predict_route_tokens', False))
        self.route_points = int(config.get('route_points', config.get('route_token_points', 50)))
        self.route_step_m = float(config.get('route_step_m', 1.0))
        self.route_loss_weight = float(config.get('route_loss_weight', 1.0))

        # Ego future key used by model output reshaping
        self.horizon = config.get('future_len', 80)
        self.predicted_neighbor_num = config.get('predicted_neighbor_num', 10)
        self.P = 1 + self.predicted_neighbor_num
        self.output_dim = self.model.output_dim

    def build_ego_route_targets(self, ego_future_xy: torch.Tensor) -> tuple:
        """Build arc-length route geometry targets from ego future xy.

        Returns:
            route_abs: (B, route_points, 2), tail-held beyond valid path length.
            route_mask: (B, route_points), True for supervised points.
        """
        B, T, _ = ego_future_xy.shape
        origin = torch.zeros(B, 1, 2, device=ego_future_xy.device, dtype=ego_future_xy.dtype)
        path = torch.cat([origin, ego_future_xy], dim=1)
        delta = path[:, 1:] - path[:, :-1]
        segment_len = torch.linalg.vector_norm(delta, dim=-1)
        cumdist = torch.cat(
            [torch.zeros(B, 1, device=path.device, dtype=path.dtype), segment_len.cumsum(dim=1)],
            dim=1,
        )

        target_dist = (
            torch.arange(1, self.route_points + 1, device=path.device, dtype=path.dtype)
            * self.route_step_m
        )
        target = target_dist.unsqueeze(0).expand(B, -1)
        total = cumdist[:, -1:]
        route_mask = target <= total

        idx = torch.searchsorted(cumdist.contiguous(), target.contiguous(), right=False)
        idx = idx.clamp(min=1, max=T)
        prev_idx = idx - 1

        gather_idx = idx.unsqueeze(-1).expand(-1, -1, 2)
        gather_prev_idx = prev_idx.unsqueeze(-1).expand(-1, -1, 2)
        p0 = torch.gather(path, dim=1, index=gather_prev_idx)
        p1 = torch.gather(path, dim=1, index=gather_idx)

        cum0 = torch.gather(cumdist, dim=1, index=prev_idx)
        cum1 = torch.gather(cumdist, dim=1, index=idx)
        ratio = ((target - cum0) / (cum1 - cum0).clamp_min(1e-6)).unsqueeze(-1)
        route_abs = p0 + ratio * (p1 - p0)

        final_pose = path[:, -1:, :].expand(-1, self.route_points, -1)
        route_abs = torch.where(route_mask.unsqueeze(-1), route_abs, final_pose)
        return route_abs, route_mask

    # =========================================================================
    # Normalization
    # =========================================================================

    def abs_to_norm(self, traj_abs: torch.Tensor) -> torch.Tensor:
        """Convert xy positions in meters to normalized space."""
        return (traj_abs - self.global_abs_mean) / self.global_abs_std

    def norm_to_abs(self, traj_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized xy positions back to meters."""
        return traj_norm * self.global_abs_std + self.global_abs_mean

    def load_norm_stats(self, norm_stats: dict):
        """Load legacy data-estimated xy normalization statistics.

        Legacy stats only describe future xy targets. Observation inputs are
        therefore left in their original units unless a config explicitly
        enables observation normalization.
        """
        self.global_abs_mean.copy_(
            torch.tensor(norm_stats['mean'], dtype=torch.float32)
        )
        self.global_abs_std.copy_(
            torch.tensor(norm_stats['std'], dtype=torch.float32)
        )
        self.normalization_mode = 'global_stats'
        self.normalize_observation_inputs = False

    def normalization_summary(self) -> dict:
        return {
            'mode': self.normalization_mode,
            'xy_mean': self.global_abs_mean.detach().cpu().tolist(),
            'xy_std': self.global_abs_std.detach().cpu().tolist(),
            'observation_scale': float(self.observation_scale.detach().cpu().item()),
            'normalize_observations': bool(self.normalize_observation_inputs),
            'neighbor_loss_weight': self.neighbor_loss_weight,
        }

    def _zero_padded_rows(self, normalized: torch.Tensor, raw: torch.Tensor, dim: int) -> torch.Tensor:
        mask = torch.sum(torch.ne(raw[..., :dim], 0), dim=-1) == 0
        normalized = normalized.clone()
        normalized[mask] = 0.0
        return normalized

    def state_to_norm(self, state_abs: torch.Tensor) -> torch.Tensor:
        """Normalize [x, y, cos, sin, ...] state tensors.

        For fields beyond x/y we follow Diffusion-Planner's fixed observation
        normalizer: cos/sin and one-hot type channels are unchanged, while
        velocity/shape/vector-like metric channels are divided by 20.
        """
        state_norm = state_abs.clone()
        if state_norm.shape[-1] >= 2:
            state_norm[..., :2] = self.abs_to_norm(state_norm[..., :2])
        if state_norm.shape[-1] >= 6:
            state_norm[..., 4:6] = state_norm[..., 4:6] / self.observation_scale
        if state_norm.shape[-1] >= 8:
            state_norm[..., 6:8] = state_norm[..., 6:8] / self.observation_scale
        return state_norm

    def future_to_state(self, ego_future: torch.Tensor) -> torch.Tensor:
        """Convert ego future [x,y,(heading|cos,sin)] to [x,y,cos,sin]."""
        if ego_future.shape[-1] == 2:
            zeros = torch.zeros(*ego_future.shape[:-1], 2, device=ego_future.device, dtype=ego_future.dtype)
            return torch.cat([ego_future, zeros], dim=-1)
        if ego_future.shape[-1] == 3:
            heading = ego_future[..., 2]
            return torch.cat(
                [
                    ego_future[..., :2],
                    torch.stack([heading.cos(), heading.sin()], dim=-1),
                ],
                dim=-1,
            )
        if ego_future.shape[-1] >= 4:
            return ego_future[..., :4]
        raise ValueError(f'Unsupported ego_future shape: {tuple(ego_future.shape)}')

    def normalize_neighbor_past(self, neighbor_past: torch.Tensor) -> torch.Tensor:
        if not self.normalize_observation_inputs:
            return neighbor_past
        return self._zero_padded_rows(self.state_to_norm(neighbor_past), neighbor_past, dim=8)

    def normalize_lanes(self, lanes: torch.Tensor) -> torch.Tensor:
        if not self.normalize_observation_inputs:
            return lanes
        lanes_norm = lanes.clone()
        lanes_norm[..., :2] = self.abs_to_norm(lanes_norm[..., :2])
        lanes_norm[..., 2:8] = lanes_norm[..., 2:8] / self.observation_scale
        return self._zero_padded_rows(lanes_norm, lanes, dim=8)

    def normalize_route_lanes(self, route_lanes: torch.Tensor) -> torch.Tensor:
        if not self.normalize_observation_inputs:
            return route_lanes
        route_norm = route_lanes.clone()
        route_norm[..., :2] = self.abs_to_norm(route_norm[..., :2])
        route_norm[..., 2:4] = route_norm[..., 2:4] / self.observation_scale
        return self._zero_padded_rows(route_norm, route_lanes, dim=4)

    def normalize_static_objects(self, static_objs: torch.Tensor) -> torch.Tensor:
        if not self.normalize_observation_inputs:
            return static_objs
        static_norm = static_objs.clone()
        static_norm[..., :2] = self.abs_to_norm(static_norm[..., :2])
        if static_norm.shape[-1] >= 6:
            static_norm[..., 4:6] = static_norm[..., 4:6] / self.observation_scale
        return self._zero_padded_rows(static_norm, static_objs, dim=10)

    def normalize_speed_limit(self, speed_limit: torch.Tensor) -> torch.Tensor:
        if not self.normalize_observation_inputs:
            return speed_limit
        speed_norm = speed_limit / self.observation_scale
        speed_norm = speed_norm.clone()
        speed_norm[torch.sum(torch.ne(speed_limit, 0), dim=-1) == 0] = 0.0
        return speed_norm

    def normalize_observations(
        self,
        ego_current: torch.Tensor,
        neighbor_past: torch.Tensor,
        lanes: torch.Tensor,
        lanes_sl: torch.Tensor,
        route_lanes: torch.Tensor,
        static_objs: torch.Tensor,
    ) -> dict:
        if not self.normalize_observation_inputs:
            return {
                'ego_current': ego_current,
                'neighbor_past': neighbor_past,
                'lanes': lanes,
                'lanes_speed_limit': lanes_sl,
                'route_lanes': route_lanes,
                'static_objects': static_objs,
            }
        return {
            'ego_current': self.state_to_norm(ego_current),
            'neighbor_past': self.normalize_neighbor_past(neighbor_past),
            'lanes': self.normalize_lanes(lanes),
            'lanes_speed_limit': self.normalize_speed_limit(lanes_sl),
            'route_lanes': self.normalize_route_lanes(route_lanes),
            'static_objects': self.normalize_static_objects(static_objs),
        }

    # =========================================================================
    # Training
    # =========================================================================

    def forward(self, batch: tuple) -> dict:
        """DDP-friendly training forward that delegates to compute_loss."""
        return self.compute_loss(batch)

    def compute_loss(self, batch: tuple) -> dict:
        """Compute diffusion denoising loss on future states only."""
        (ego_current, ego_future, neighbor_past, neighbor_future,
         lanes, lanes_sl, lanes_hsl, route_lanes, static_objs) = batch

        B = ego_current.shape[0]
        device = ego_current.device

        obs = self.normalize_observations(
            ego_current, neighbor_past, lanes, lanes_sl, route_lanes, static_objs
        )
        ego_cur = obs['ego_current']

        neighbor_cur_raw = neighbor_past[:, :self.predicted_neighbor_num, -1, :4]
        neighbor_cur_mask = torch.sum(torch.ne(neighbor_cur_raw, 0), dim=-1) == 0
        neighbor_cur = obs['neighbor_past'][:, :self.predicted_neighbor_num, -1, :4]

        ego_future_state = self.future_to_state(ego_future)
        ego_future_norm = self.state_to_norm(ego_future_state)

        neighbor_future_raw = neighbor_future[:, :self.predicted_neighbor_num]
        neighbor_future_norm = self.state_to_norm(neighbor_future_raw)
        neighbor_future_valid = torch.sum(torch.ne(neighbor_future_raw[..., :4], 0), dim=-1) != 0
        neighbor_future_norm = neighbor_future_norm.clone()
        neighbor_future_norm[~neighbor_future_valid] = 0.0

        gt_future = torch.cat(
            [ego_future_norm.unsqueeze(1), neighbor_future_norm], dim=1
        )  # (B, P, T, 4)
        cur_states = torch.cat([ego_cur.unsqueeze(1), neighbor_cur], dim=1)

        noise = torch.randn_like(gt_future)
        timesteps = torch.randint(
            0, self.num_train_timesteps, (B,), device=device
        ).long()

        noise_flat = noise.reshape(B * self.P, -1)
        gt_future_flat = gt_future.reshape(B * self.P, -1)
        noisy_future_flat = self.noise_scheduler.add_noise(
            gt_future_flat,
            noise_flat,
            timesteps.repeat_interleave(self.P),
        )
        noisy_future = noisy_future_flat.reshape(B, self.P, self.horizon, 4)
        noisy = torch.cat([cur_states.unsqueeze(2), noisy_future], dim=2)

        route_target_norm = None
        route_target_mask = None
        noisy_route = None
        if self.predict_route_tokens:
            route_target_abs, route_target_mask = self.build_ego_route_targets(ego_future[..., :2])
            route_target_norm = self.abs_to_norm(route_target_abs)
            route_noise = torch.randn_like(route_target_norm)
            noisy_route_flat = self.noise_scheduler.add_noise(
                route_target_norm.reshape(B * self.route_points, 2),
                route_noise.reshape(B * self.route_points, 2),
                timesteps.repeat_interleave(self.route_points),
            )
            noisy_route = noisy_route_flat.reshape(B, self.route_points, 2)

        model_inputs = {
            'neighbor_agents_past': obs['neighbor_past'],
            'static_objects': obs['static_objects'],
            'lanes': obs['lanes'],
            'lanes_speed_limit': obs['lanes_speed_limit'],
            'lanes_has_speed_limit': lanes_hsl,
            'route_lanes': obs['route_lanes'],
            'sampled_trajectories': noisy.reshape(B, self.P, -1),
            'diffusion_time': timesteps,
            'neighbor_current_mask': neighbor_cur_mask,
        }
        if self.predict_route_tokens:
            model_inputs['sampled_route'] = noisy_route

        outputs = self.model(model_inputs)
        pred_future = outputs['score'][:, :, 1:, :]  # (B, P, T, 4)

        loss_all = F.l1_loss(pred_future, gt_future, reduction='none')
        ego_loss = loss_all[:, 0].mean()

        neighbor_loss_all = loss_all[:, 1:]
        if neighbor_future_valid.any():
            neighbor_loss = neighbor_loss_all[neighbor_future_valid].mean()
        else:
            neighbor_loss = torch.tensor(0.0, device=device)

        weighted_neighbor_loss = self.neighbor_loss_weight * neighbor_loss
        route_loss = torch.tensor(0.0, device=device)
        weighted_route_loss = torch.tensor(0.0, device=device)
        if self.predict_route_tokens:
            if 'route_score' not in outputs:
                raise KeyError('predict_route_tokens=True but model output has no route_score')
            route_pred = outputs['route_score']
            route_loss_all = F.l1_loss(route_pred, route_target_norm, reduction='none').mean(dim=-1)
            if route_target_mask.any():
                route_loss = route_loss_all[route_target_mask].mean()
            weighted_route_loss = self.route_loss_weight * route_loss

        total_loss = ego_loss + weighted_neighbor_loss + weighted_route_loss

        return {
            'loss': total_loss,
            'ego_loss': ego_loss,
            'neighbor_loss': neighbor_loss,
            'weighted_neighbor_loss': weighted_neighbor_loss,
            'route_loss': route_loss,
            'weighted_route_loss': weighted_route_loss,
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
        """DDIM denoising from pure noise N(0,I)."""
        if num_steps is None:
            num_steps = self.num_inference_steps

        neighbor_past = cond_data['neighbor_agents_past']
        B = neighbor_past.shape[0]
        device = neighbor_past.device

        ego_cur_raw = cond_data.get('ego_current_state')
        if ego_cur_raw is None:
            ego_cur_raw = torch.zeros(B, 4, device=device, dtype=neighbor_past.dtype)
            ego_cur_raw[:, 2] = 1.0

        static_objs = cond_data.get('static_objects', torch.zeros(B, 5, 10, device=device))
        lanes = cond_data.get('lanes', torch.zeros(B, 30, 20, 12, device=device))
        lanes_sl = cond_data.get('lanes_speed_limit', torch.zeros(B, 30, 1, device=device))
        lanes_hsl = cond_data.get('lanes_has_speed_limit', torch.zeros(B, 30, 1, device=device, dtype=torch.bool))
        route_lanes = cond_data.get('route_lanes', torch.zeros(B, 10, 20, 4, device=device))

        obs = self.normalize_observations(
            ego_cur_raw, neighbor_past, lanes, lanes_sl, route_lanes, static_objs
        )

        neighbor_cur_raw = neighbor_past[:, :self.predicted_neighbor_num, -1, :4]
        neighbor_cur_mask = torch.sum(torch.ne(neighbor_cur_raw, 0), dim=-1) == 0
        neighbor_cur = obs['neighbor_past'][:, :self.predicted_neighbor_num, -1, :4]
        cur_states = torch.cat([obs['ego_current'].unsqueeze(1), neighbor_cur], dim=1)

        noise = torch.randn(B, self.P, self.horizon, 4, device=device)
        x_t = torch.cat([cur_states.unsqueeze(2), noise], dim=2)
        route_t = None
        if self.predict_route_tokens:
            route_t = torch.randn(B, self.route_points, 2, device=device, dtype=x_t.dtype)

        self.noise_scheduler.set_timesteps(num_steps)

        for t in self.noise_scheduler.timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            model_inputs = {
                'neighbor_agents_past': obs['neighbor_past'],
                'static_objects': obs['static_objects'],
                'lanes': obs['lanes'],
                'lanes_speed_limit': obs['lanes_speed_limit'],
                'lanes_has_speed_limit': lanes_hsl,
                'route_lanes': obs['route_lanes'],
                'sampled_trajectories': x_t.reshape(B, self.P, -1),
                'diffusion_time': t_batch,
                'neighbor_current_mask': neighbor_cur_mask,
            }
            if self.predict_route_tokens:
                model_inputs['sampled_route'] = route_t

            outputs = self.model(model_inputs)
            pred_x0 = outputs['score']

            x_t_flat = x_t.reshape(B * self.P, -1)
            pred_flat = pred_x0.reshape(B * self.P, -1)
            x_next_flat = self.noise_scheduler.step(
                pred_flat, t, x_t_flat, return_dict=False
            )[0]
            x_t = x_next_flat.reshape(B, self.P, self.horizon + 1, 4)
            x_t[:, :, 0, :] = cur_states

            if self.predict_route_tokens:
                route_pred = outputs['route_score']
                route_next_flat = self.noise_scheduler.step(
                    route_pred.reshape(B * self.route_points, 2),
                    t,
                    route_t.reshape(B * self.route_points, 2),
                    return_dict=False,
                )[0]
                route_t = route_next_flat.reshape(B, self.route_points, 2)

        trajectory = x_t[:, :, 1:, :]
        trajectory_abs = trajectory.clone()
        trajectory_abs[..., :2] = self.norm_to_abs(trajectory[..., :2])

        result = {'trajectory': trajectory_abs}
        if self.predict_route_tokens and route_t is not None:
            result['route'] = self.norm_to_abs(route_t)
        return result

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
