"""Clean motion-only diffusion policy for paper reproduction."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from model.paper_motion_diffusion_core import PaperMotionDiffusionCore
from model.paper_motion_unified_no_detail_core import PaperMotionUnifiedNoDetailCore


class PaperMotionPolicy(nn.Module):
    """Training and sampling wrapper around :class:`PaperMotionDiffusionCore`."""

    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        policy_cfg = config['policy']
        route_cfg = config.get('route_b', {})
        diffusion_cfg = config.get('truncated_diffusion', {})
        shape_meta = config['shape_meta']
        action_dim = int(shape_meta['action']['shape'][0])
        self.horizon = int(policy_cfg.get('horizon', config.get('action_horizon', 6)))
        self.num_waypoints = int(route_cfg.get('num_waypoints', 20))
        self.num_inference_steps = int(route_cfg.get('num_inference_steps', 10))
        self.train_max_timesteps = int(route_cfg.get('train_max_timesteps', diffusion_cfg.get('num_train_timesteps', 1000)))
        self.num_train_timesteps = int(diffusion_cfg.get('num_train_timesteps', self.train_max_timesteps))
        self.prediction_type = str(diffusion_cfg.get('prediction_type', 'sample'))
        self.paper_sampling_mode = str(route_cfg.get('paper_sampling_mode', 'diffusers_step'))
        self.eta = float(diffusion_cfg.get('eta', 0.0))
        self.speed_profile_dt = float(route_cfg.get('speed_profile_dt', 0.5))
        self.reg_loss_weight = float(config.get('reg_loss_weight', 3.0))
        self.route_loss_weight = float(diffusion_cfg.get('route_loss_weight', 5.0))
        self.route_final_loss_weight = float(diffusion_cfg.get('route_final_loss_weight', 2.0))
        self.speed_loss_weight = float(route_cfg.get('speed_loss_weight', 0.2))
        self.paper_motion_core = str(route_cfg.get('paper_motion_core', 'simple'))

        transfuser_cfg = config.get('transfuser_encoder', {})
        core_kwargs = dict(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=self.horizon,
            num_waypoints=self.num_waypoints,
            n_obs_steps=int(policy_cfg.get('n_obs_steps', config.get('n_obs_steps', 4))),
            status_dim=int(config.get('bev_encoder', {}).get('state_dim', 14)),
            n_layer=int(policy_cfg.get('n_layer', 4)),
            n_head=int(policy_cfg.get('n_head', 8)),
            n_emb=int(policy_cfg.get('n_emb', 512)),
            p_drop_emb=float(policy_cfg.get('p_drop_emb', 0.1)),
            p_drop_attn=float(policy_cfg.get('p_drop_attn', 0.1)),
            transfuser_bev_dim=int(transfuser_cfg.get('bev_feature_dim', 1512)),
            transfuser_bev_upsample_dim=int(transfuser_cfg.get('bev_feature_upsample_dim', 64)),
            traj_can_attend_route=bool(policy_cfg.get('traj_can_attend_route', True)),
        )
        if self.paper_motion_core == 'simple':
            self.model = PaperMotionDiffusionCore(**core_kwargs)
        elif self.paper_motion_core == 'unified_no_detail':
            self.model = PaperMotionUnifiedNoDetailCore(
                **core_kwargs,
                ego_detail_activation_t=int(policy_cfg.get('ego_detail_activation_t', -1)),
            )
        else:
            raise ValueError(
                f"Unsupported paper_motion_core={self.paper_motion_core!r}; "
                "expected 'simple' or 'unified_no_detail'"
            )
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            steps_offset=1,
            beta_schedule='scaled_linear',
            prediction_type=self.prediction_type,
        )
        self.register_buffer('delta_mean', None)
        self.register_buffer('delta_std', None)
        self.register_buffer('abs_mean', None)
        self.register_buffer('abs_std', None)
        self.register_buffer('global_abs_mean', None)
        self.register_buffer('global_abs_std', None)
        self.register_buffer('route_abs_mean', None)
        self.register_buffer('route_abs_std', None)

    def build_roll_timesteps(self, num_steps: Optional[int] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        """Old Route-B DDIM timestep schedule used by the legacy motion-only policy."""
        if num_steps is None:
            num_steps = self.num_inference_steps
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        max_t = int(self.train_max_timesteps)
        if max_t <= 0:
            raise ValueError(f"train_max_timesteps must be positive, got {max_t}")

        if num_steps == 1:
            timesteps = np.array([max_t - 1], dtype=np.int64)
        else:
            step_ratio = max_t / num_steps
            timesteps = (np.arange(0, num_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps = np.clip(timesteps, 0, max_t - 1)

        timesteps = torch.from_numpy(timesteps)
        return timesteps.to(device) if device is not None else timesteps

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return values.mean()
        mask = mask.to(device=values.device, dtype=torch.bool).reshape(-1)
        if not mask.any():
            return values.mean()
        return values.reshape(values.shape[0], -1).mean(dim=1)[mask].mean()

    @staticmethod
    def _reduce_per_sample(values: torch.Tensor) -> torch.Tensor:
        return values.reshape(values.shape[0], -1).mean(dim=1)

    def _good_route_mask(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
        bad = batch.get('is_bad_route')
        if bad is None:
            return None
        return ~bad.to(device=device, dtype=torch.bool).reshape(-1)

    @staticmethod
    def select_norm_stats_config(config: Dict) -> Tuple[str, str]:
        candidates = [
            ('global_abs', config.get('global_abs_stats_path')),
            ('abs', config.get('abs_stats_path')),
            ('delta', config.get('delta_stats_path')),
        ]
        specified = [(name, path) for name, path in candidates if path]
        if len(specified) != 1:
            raise ValueError('Configure exactly one of global_abs_stats_path, abs_stats_path, delta_stats_path')
        return specified[0]

    def register_norm_stats_from_config(self, config: Dict) -> str:
        mode, path = self.select_norm_stats_config(config)
        data = np.load(path)
        if mode == 'global_abs':
            self.register_global_abs_stats(data['global_abs_mean'], data['global_abs_std'])
        elif mode == 'abs':
            self.register_abs_stats(data['abs_mean'], data['abs_std'])
        else:
            self.register_delta_stats(data['delta_mean'], data['delta_std'])
        route_path = config.get('route_abs_stats_path')
        if not route_path:
            raise ValueError('route_abs_stats_path is required for joint trajectory+route diffusion')
        route_data = np.load(route_path)
        self.register_route_abs_stats(route_data['route_abs_mean'], route_data['route_abs_std'])
        return mode

    def register_delta_stats(self, mean, std) -> None:
        self.delta_mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        self.delta_std = torch.as_tensor(std, dtype=torch.float32, device=self.device)

    def register_abs_stats(self, mean, std) -> None:
        self.abs_mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        self.abs_std = torch.as_tensor(std, dtype=torch.float32, device=self.device)

    def register_global_abs_stats(self, mean, std) -> None:
        self.global_abs_mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        self.global_abs_std = torch.as_tensor(std, dtype=torch.float32, device=self.device)

    def register_route_abs_stats(self, mean, std) -> None:
        self.route_abs_mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        self.route_abs_std = torch.as_tensor(std, dtype=torch.float32, device=self.device)

    @staticmethod
    def abs_to_delta(abs_traj: torch.Tensor) -> torch.Tensor:
        delta = abs_traj.clone()
        delta[..., 1:, :] = abs_traj[..., 1:, :] - abs_traj[..., :-1, :]
        return delta

    @staticmethod
    def delta_to_abs(delta: torch.Tensor) -> torch.Tensor:
        return delta.cumsum(dim=-2)

    def abs_to_norm(self, abs_traj: torch.Tensor) -> torch.Tensor:
        if self.global_abs_mean is not None:
            return (abs_traj - self.global_abs_mean.to(abs_traj.device)) / self.global_abs_std.to(abs_traj.device).clamp(min=1e-6)
        if self.abs_mean is not None:
            return (abs_traj - self.abs_mean.to(abs_traj.device)) / self.abs_std.to(abs_traj.device).clamp(min=1e-6)
        delta = self.abs_to_delta(abs_traj)
        return (delta - self.delta_mean.to(abs_traj.device)) / self.delta_std.to(abs_traj.device).clamp(min=1e-6)

    def norm_to_abs(self, z: torch.Tensor) -> torch.Tensor:
        if self.global_abs_mean is not None:
            return z * self.global_abs_std.to(z.device) + self.global_abs_mean.to(z.device)
        if self.abs_mean is not None:
            return z * self.abs_std.to(z.device) + self.abs_mean.to(z.device)
        delta = z * self.delta_std.to(z.device) + self.delta_mean.to(z.device)
        return self.delta_to_abs(delta)

    def route_abs_to_norm(self, route: torch.Tensor) -> torch.Tensor:
        if self.route_abs_mean is None or self.route_abs_std is None:
            raise RuntimeError('route_abs_stats_path must be loaded before training')
        return (route - self.route_abs_mean.to(route.device)) / self.route_abs_std.to(route.device).clamp(min=1e-6)

    def route_norm_to_abs(self, z: torch.Tensor) -> torch.Tensor:
        if self.route_abs_mean is None or self.route_abs_std is None:
            raise RuntimeError('route_abs_stats_path must be loaded before sampling')
        return z * self.route_abs_std.to(z.device) + self.route_abs_mean.to(z.device)

    def joint_abs_to_norm(self, traj: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.abs_to_norm(traj), self.route_abs_to_norm(route)], dim=-2)

    def joint_norm_to_abs(self, joint: torch.Tensor) -> torch.Tensor:
        traj = self.norm_to_abs(joint[..., :self.horizon, :])
        route = self.route_norm_to_abs(joint[..., self.horizon:, :])
        return torch.cat([traj, route], dim=-2)

    def _speed_target(self, trajectory: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        target = batch.get('next_speed_target_mps')
        if target is None:
            if trajectory.shape[1] < 1:
                return None
            target_speed = trajectory[:, 0].norm(dim=-1) / max(self.speed_profile_dt, 1e-6)
        else:
            target_speed = target.to(device=trajectory.device, dtype=trajectory.dtype).reshape(-1)
        bins = torch.tensor(self.model.speed_classes, device=trajectory.device, dtype=trajectory.dtype)
        target_speed = target_speed.clamp(min=bins[0], max=bins[-1])
        two_hot = torch.zeros(target_speed.shape[0], bins.numel(), device=trajectory.device, dtype=trajectory.dtype)
        for i in range(bins.numel() - 1):
            mask = (target_speed >= bins[i]) & (target_speed < bins[i + 1])
            if i == bins.numel() - 2:
                mask = mask | (target_speed == bins[i + 1])
            if mask.any():
                ratio = (target_speed[mask] - bins[i]) / (bins[i + 1] - bins[i]).clamp(min=1e-6)
                two_hot[mask, i] = 1.0 - ratio
                two_hot[mask, i + 1] = ratio
        return two_hot

    @staticmethod
    def decode_speed(speed_logits: torch.Tensor, speed_classes) -> torch.Tensor:
        bins = torch.tensor(speed_classes, device=speed_logits.device, dtype=speed_logits.dtype)
        return (torch.softmax(speed_logits.float(), dim=-1).to(speed_logits.dtype) * bins).sum(dim=-1)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = self.device
        dtype = self.dtype
        trajectory = batch['agent_pos'].to(device=device, dtype=dtype)[:, :self.horizon]
        route = batch['route'].to(device=device, dtype=dtype)[:, :self.num_waypoints]
        ego_status = batch['ego_status'].to(device=device, dtype=dtype)
        bev = batch['transfuser_bev_feature'].to(device=device, dtype=dtype)
        bev_up = batch['transfuser_bev_feature_upsample'].to(device=device, dtype=dtype)
        good_mask = self._good_route_mask(batch, device)
        B = trajectory.shape[0]
        joint_norm = self.joint_abs_to_norm(trajectory, route)
        timesteps = torch.randint(0, self.train_max_timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(joint_norm, dtype=torch.float32)
        noisy_norm = self.diffusion_scheduler.add_noise(joint_norm.float(), noise, timesteps).to(dtype=dtype)
        noisy_abs = self.joint_norm_to_abs(noisy_norm)
        pred = self.model.forward_denoise(noisy_abs, timesteps, bev, bev_up, ego_status)
        traj_abs = self.norm_to_abs(pred['traj_norm'])
        route_abs = self.route_norm_to_abs(pred['route_norm'])
        traj_per_sample = self._reduce_per_sample(F.l1_loss(traj_abs, trajectory, reduction='none'))
        route_recon = self._reduce_per_sample(F.l1_loss(route_abs, route, reduction='none'))
        route_fde = self._reduce_per_sample(F.l1_loss(route_abs[:, -1], route[:, -1], reduction='none'))
        reg_loss = self._masked_mean(traj_per_sample, good_mask)
        route_loss = self._masked_mean(route_recon + self.route_final_loss_weight * route_fde, good_mask)
        speed_loss = torch.zeros((), device=device, dtype=dtype)
        speed_target = self._speed_target(trajectory, batch)
        if speed_target is not None:
            log_probs = F.log_softmax(pred['speed_logits'].float(), dim=-1)
            speed_per_sample = -(speed_target.float() * log_probs).sum(dim=-1)
            speed_loss = self._masked_mean(speed_per_sample.to(dtype), good_mask)
        total = self.reg_loss_weight * reg_loss + self.route_loss_weight * route_loss + self.speed_loss_weight * speed_loss
        return {
            'total_loss': total,
            'reg_loss': reg_loss,
            'route_loss': route_loss,
            'speed_loss': speed_loss,
        }

    def forward(self, batch: Dict[str, torch.Tensor], return_loss_dict: bool = False):
        loss_dict = self.compute_loss(batch)
        return loss_dict if return_loss_dict else loss_dict['total_loss']

    @torch.no_grad()
    def sample(self, batch: Dict[str, torch.Tensor], num_inference_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        device = self.device
        dtype = self.dtype
        bev = batch['transfuser_bev_feature'].to(device=device, dtype=dtype)
        bev_up = batch['transfuser_bev_feature_upsample'].to(device=device, dtype=dtype)
        ego_status = batch['ego_status'].to(device=device, dtype=dtype)
        B = bev.shape[0]
        joint_len = self.horizon + self.num_waypoints
        x = torch.randn(B, joint_len, 2, device=device, dtype=torch.float32)
        steps = int(num_inference_steps or self.num_inference_steps)
        speed_logits = None
        pred_joint = None
        if self.paper_sampling_mode == 'old_pred_x0_ddim':
            timesteps = self.build_roll_timesteps(steps, device=device)
            alphas_cumprod = self.diffusion_scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
            for step_i, t in enumerate(timesteps):
                t_cur = int(t.item())
                t_next = int(timesteps[step_i + 1].item()) if step_i + 1 < len(timesteps) else 0
                t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
                noisy_abs = self.joint_norm_to_abs(x.to(dtype=dtype))
                pred = self.model.forward_denoise(noisy_abs, t_batch, bev, bev_up, ego_status)
                pred_joint = torch.cat([pred['traj_norm'], pred['route_norm']], dim=-2).float()
                speed_logits = pred['speed_logits']

                alpha_t = alphas_cumprod[t_cur]
                alpha_next = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
                pred_eps = (x - alpha_t.sqrt() * pred_joint) / (1 - alpha_t).sqrt().clamp(min=1e-8)
                x = alpha_next.sqrt() * pred_joint + (1 - alpha_next).sqrt() * pred_eps
            if pred_joint is None:
                raise RuntimeError('old_pred_x0_ddim sampling produced no prediction')
            joint_abs = self.joint_norm_to_abs(pred_joint.to(dtype=dtype))
        elif self.paper_sampling_mode == 'diffusers_step':
            self.diffusion_scheduler.set_timesteps(steps, device=device)
            for t in self.diffusion_scheduler.timesteps:
                t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
                noisy_abs = self.joint_norm_to_abs(x.to(dtype=dtype))
                pred = self.model.forward_denoise(noisy_abs, t_batch, bev, bev_up, ego_status)
                pred_joint = torch.cat([pred['traj_norm'], pred['route_norm']], dim=-2).float()
                speed_logits = pred['speed_logits']
                step_out = self.diffusion_scheduler.step(pred_joint, t, x, eta=self.eta)
                x = step_out.prev_sample
            joint_abs = self.joint_norm_to_abs(x.to(dtype=dtype))
        else:
            raise ValueError(
                f"Unsupported paper_sampling_mode={self.paper_sampling_mode!r}; "
                "expected 'diffusers_step' or 'old_pred_x0_ddim'"
            )
        return {
            'trajectory': joint_abs[:, :self.horizon],
            'route': joint_abs[:, self.horizon:],
            'speed_logits': speed_logits,
            'speed_mps': self.decode_speed(speed_logits, self.model.speed_classes) if speed_logits is not None else None,
        }

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        no_noise: bool = True,
        reset_semantic_state_cache: bool = False,
        disable_semantic_state_cache: bool = False,
        num_inference_steps: Optional[int] = None,
        **_: object,
    ) -> Dict[str, object]:
        # Compatibility wrapper for route_b_b2d_agent. Semantic cache arguments are
        # accepted but intentionally ignored by this motion-only policy.
        out = self.sample(obs_dict, num_inference_steps=num_inference_steps)
        traj = out['trajectory'].detach().float().cpu().numpy()
        route = out['route'].detach().float().cpu()
        speed_mps = out.get('speed_mps')
        if speed_mps is not None:
            target_speed = speed_mps.detach().float().cpu().numpy()
        else:
            target_speed = None
        return {
            'action': traj,
            'route_pred': route,
            'target_speed': target_speed,
            'speed_logits': out.get('speed_logits'),
            'speed_mps': target_speed,
        }

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, config: Dict, device: str = 'cuda'):
        policy = cls(config)
        policy.register_norm_stats_from_config(config)
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        policy.load_state_dict(state, strict=True)
        policy.to(device)
        policy.eval()
        return policy, ckpt
