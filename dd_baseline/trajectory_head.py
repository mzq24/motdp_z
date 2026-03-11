"""
Trajectory prediction head with truncated diffusion.

Faithfully reproduces DiffusionDrive's TrajectoryHead architecture:
- Anchor-based multimodal trajectory prediction (20 modes)
- Truncated diffusion training (noise range [0, trunc_timesteps))
- DDIM inference with 2 denoising steps
- CustomTransformerDecoder with GridSampleCrossBEVAttention + FiLM modulation
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from diffusers.schedulers import DDIMScheduler

from .modules.blocks import (
    linear_relu_ln, gen_sineembed_for_position, GridSampleCrossBEVAttention
)
from .modules.modulation import ModulationLayer
from .modules.refinement import DiffMotionPlanningRefinementModule
from .modules.sinusoidal_emb import SinusoidalPosEmb
from .modules.loss import LossComputer
from .config import DDBaselineConfig


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class CustomTransformerDecoderLayer(nn.Module):
    """Single diffusion decoder layer.

    Components:
        1. GridSampleCrossBEVAttention: sample BEV features at trajectory locations
        2. Cross-agent attention: attend to agent queries
        3. Cross-ego attention: attend to ego query
        4. FFN + LayerNorm
        5. ModulationLayer: FiLM timestep conditioning
        6. DiffMotionPlanningRefinementModule: predict cls + reg
    """
    def __init__(self, config: DDBaselineConfig):
        super().__init__()
        d_model = config.tf_d_model
        d_ffn = config.tf_d_ffn
        num_poses = config.num_poses

        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)

        self.cross_bev_attention = GridSampleCrossBEVAttention(
            embed_dims=d_model,
            num_heads=config.tf_num_head,
            num_points=num_poses,
            in_bev_dims=d_model,  # After bev_proj, channels = d_model
            lidar_max_x=config.lidar_max_x,
            lidar_max_y=config.lidar_max_y,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            d_model, config.tf_num_head,
            dropout=config.tf_dropout, batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            d_model, config.tf_num_head,
            dropout=config.tf_dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.time_modulation = ModulationLayer(d_model, d_model)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=config.ego_fut_mode,
        )

    def forward(self, traj_feature, noisy_traj_points, bev_feature,
                bev_spatial_shape, agents_query, ego_query, time_embed,
                status_encoding, predict_delta=False):
        """
        Args:
            traj_feature: (bs, num_modes, d_model)
            noisy_traj_points: (bs, num_modes, num_poses, 2) - BEV sampling locations
            bev_feature: (bs, d_model, H, W)
            bev_spatial_shape: (H, W)
            agents_query: (bs, num_agents, d_model)
            ego_query: (bs, 1, d_model)
            time_embed: (bs, 1, d_model)
            status_encoding: (bs, 1, d_model)
            predict_delta: if True, output is raw delta (not added to noisy_traj_points)
        Returns:
            poses_reg: (bs, num_modes, num_poses, 2)
            poses_cls: (bs, num_modes)
        """
        # 1. Cross-attention with BEV features at trajectory locations
        traj_feature = self.cross_bev_attention(
            traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape
        )

        # 2. Cross-attention with agent queries
        traj_feature = traj_feature + self.dropout(
            self.cross_agent_attention(traj_feature, agents_query, agents_query)[0]
        )
        traj_feature = self.norm1(traj_feature)

        # 3. Cross-attention with ego query
        traj_feature = traj_feature + self.dropout1(
            self.cross_ego_attention(traj_feature, ego_query, ego_query)[0]
        )
        traj_feature = self.norm2(traj_feature)

        # 4. FFN
        traj_feature = self.norm3(self.ffn(traj_feature))

        # 5. Timestep modulation (FiLM)
        traj_feature = self.time_modulation(traj_feature, time_embed)

        # 6. Predict offset & classification
        poses_reg, poses_cls = self.task_decoder(traj_feature)
        if not predict_delta:
            # Absolute mode (default): add offset to noisy trajectory points
            poses_reg = poses_reg + noisy_traj_points

        return poses_reg, poses_cls


class CustomTransformerDecoder(nn.Module):
    """Stack of CustomTransformerDecoderLayer with iterative refinement.

    Each layer predicts trajectory, and the next layer uses the predicted
    trajectory points for GridSample attention (detached).
    """
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, traj_feature, noisy_traj_points, bev_feature,
                bev_spatial_shape, agents_query, ego_query, time_embed,
                status_encoding, clean_anchor=None, predict_delta=False,
                fix_bev_at_anchor=False, delta_to_abs_fn=None):
        poses_reg_list = []
        poses_cls_list = []
        # Initial BEV position: always clean_anchor for delta/fixed modes
        if (predict_delta or fix_bev_at_anchor) and clean_anchor is not None:
            traj_points = clean_anchor.clone().detach()
        else:
            traj_points = noisy_traj_points

        for layer in self.layers:
            poses_reg, poses_cls = layer(
                traj_feature, traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding,
                predict_delta=predict_delta
            )
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)

            # Update BEV sampling positions for next layer
            if delta_to_abs_fn is not None:
                # Experiment 4b: convert delta-normed prediction to abs positions
                traj_points = delta_to_abs_fn(poses_reg).clone().detach()
            elif predict_delta or fix_bev_at_anchor:
                # Fixed BEV: always sample at clean anchor locations
                traj_points = clean_anchor.clone().detach()
            else:
                # Dynamic BEV: next layer uses predicted trajectory for BEV sampling
                traj_points = poses_reg.clone().detach()

        return poses_reg_list, poses_cls_list


class TrajectoryHead(nn.Module):
    """Trajectory prediction head with truncated diffusion.

    Training: add truncated noise to anchors, predict clean trajectory
    Inference: 2-step DDIM denoising from noisy anchors
    """
    def __init__(self, config: DDBaselineConfig):
        super().__init__()
        d_model = config.tf_d_model
        num_poses = config.num_poses
        self.num_poses = num_poses
        self.ego_fut_mode = config.ego_fut_mode
        self.trunc_timesteps = config.trunc_timesteps
        self.num_diffusion_steps = config.num_diffusion_steps

        # Normalization parameters (abs coords)
        self.norm_x_offset = config.norm_x_offset
        self.norm_x_range = config.norm_x_range
        self.norm_y_offset = config.norm_y_offset
        self.norm_y_range = config.norm_y_range
        self.predict_delta = getattr(config, 'predict_delta', False)
        self.use_normalized_forward = getattr(config, 'use_normalized_forward', False)
        self.fix_bev_at_anchor = getattr(config, 'fix_bev_at_anchor', False)
        self.delta_dynamic_bev = getattr(config, 'delta_dynamic_bev', False)
        self.use_full_diffusion = getattr(config, 'use_full_diffusion', False)
        self.num_train_timesteps = config.num_train_timesteps

        # Per-step delta Z-score normalization parameters
        self.norm_delta_x_mean = config.norm_delta_x_mean
        self.norm_delta_x_std = config.norm_delta_x_std
        self.norm_delta_y_mean = config.norm_delta_y_mean
        self.norm_delta_y_std = config.norm_delta_y_std

        # DDIM scheduler
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        # Load anchors (absolute coordinates)
        plan_anchor = np.load(config.plan_anchor_path)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )  # (num_modes, num_poses, 2) - absolute coords

        # Pre-compute anchor deltas for delta mode
        if self.predict_delta:
            anchor_delta = self._traj_to_delta(self.plan_anchor.data)
            self.register_buffer('plan_anchor_delta', anchor_delta)  # (M, T, 2)

        # Trajectory positional encoding: gen_sineembed gives num_poses * 64 = num_poses * hidden_dim
        # With hidden_dim=64: each point -> 64-dim embedding, num_poses points -> num_poses*64
        anchor_embed_dim = num_poses * 64
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, anchor_embed_dim),
            nn.Linear(d_model, d_model),
        )

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        # Diffusion decoder
        diff_decoder_layer = CustomTransformerDecoderLayer(config)
        self.diff_decoder = CustomTransformerDecoder(
            diff_decoder_layer, config.num_decoder_layers
        )

        # Loss
        self.loss_computer = LossComputer(
            cls_loss_weight=config.trajectory_cls_weight,
            reg_loss_weight=config.trajectory_reg_weight,
        )

    def norm_odo(self, odo_info_fut):
        """Normalize trajectory coordinates to [-1, 1]."""
        x = odo_info_fut[..., 0:1]
        y = odo_info_fut[..., 1:2]
        x = 2 * (x + self.norm_x_offset) / self.norm_x_range - 1
        y = 2 * (y + self.norm_y_offset) / self.norm_y_range - 1
        return torch.cat([x, y], dim=-1)

    def denorm_odo(self, odo_info_fut):
        """Denormalize trajectory from [-1, 1] to original coordinates."""
        x = odo_info_fut[..., 0:1]
        y = odo_info_fut[..., 1:2]
        x = (x + 1) / 2 * self.norm_x_range - self.norm_x_offset
        y = (y + 1) / 2 * self.norm_y_range - self.norm_y_offset
        return torch.cat([x, y], dim=-1)

    def norm_delta(self, delta):
        """Z-score normalize per-step delta: (x - mean) / std."""
        dx = (delta[..., 0:1] - self.norm_delta_x_mean) / self.norm_delta_x_std
        dy = (delta[..., 1:2] - self.norm_delta_y_mean) / self.norm_delta_y_std
        return torch.cat([dx, dy], dim=-1)

    def denorm_delta(self, delta_normed):
        """Inverse Z-score: x * std + mean."""
        dx = delta_normed[..., 0:1] * self.norm_delta_x_std + self.norm_delta_x_mean
        dy = delta_normed[..., 1:2] * self.norm_delta_y_std + self.norm_delta_y_mean
        return torch.cat([dx, dy], dim=-1)

    @staticmethod
    def _traj_to_delta(trajectory):
        """Convert absolute trajectory to per-step deltas.
        delta[0] = pos[0], delta[i] = pos[i] - pos[i-1]."""
        delta = torch.zeros_like(trajectory)
        delta[..., 0, :] = trajectory[..., 0, :]
        delta[..., 1:, :] = trajectory[..., 1:, :] - trajectory[..., :-1, :]
        return delta

    @staticmethod
    def _cumulate_trajectory(deltas):
        """Convert per-step deltas to absolute trajectory via cumsum."""
        return torch.cumsum(deltas, dim=-2)

    def forward(self, ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, targets=None):
        if self.training:
            self._test_debug_printed = False  # Reset so next val prints
            return self.forward_train(
                ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, targets
            )
        else:
            return self.forward_test(
                ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, targets=targets
            )

    def forward_train(self, ego_query, agents_query, bev_feature,
                      bev_spatial_shape, status_encoding, targets):
        bs = ego_query.shape[0]
        device = ego_query.device

        # Plan anchors in absolute coords (for BEV sampling and anchor matching)
        plan_anchor = self.plan_anchor.unsqueeze(0).expand(bs, -1, -1, -1)  # (bs, M, T, 2)

        if self.use_full_diffusion:
            timesteps = torch.randint(0, self.num_train_timesteps, (bs,), device=device)
        else:
            timesteps = torch.randint(0, self.trunc_timesteps, (bs,), device=device)

        if self.predict_delta:
            # ========== Delta mode: diffusion in Z-score normalized delta space ==========
            plan_anchor_delta = self.plan_anchor_delta.unsqueeze(0).expand(bs, -1, -1, -1)
            delta_normed = self.norm_delta(plan_anchor_delta)  # (bs, M, T, 2)

            if self.use_full_diffusion:
                # Full diffusion: noise GT (not anchor)
                target_traj = targets["trajectory"][:, :self.num_poses]  # (bs, T, 2)
                gt_delta = self._traj_to_delta(target_traj)
                gt_delta_normed = self.norm_delta(gt_delta)  # (bs, T, 2)
                # Expand GT to M modes (same GT, different noise per mode)
                clean_sample = gt_delta_normed.unsqueeze(1).expand(-1, self.ego_fut_mode, -1, -1)
            else:
                clean_sample = delta_normed

            noise = torch.randn(clean_sample.shape, device=device)
            noisy_delta_normed = self.diffusion_scheduler.add_noise(
                original_samples=clean_sample, noise=noise, timesteps=timesteps
            ).float()

            # Position embedding from noisy normalized deltas
            traj_pos_embed = gen_sineembed_for_position(noisy_delta_normed, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)

            # Timestep embedding
            time_embed = self.time_mlp(timesteps).unsqueeze(1)

            # Decoder: BEV samples at plan_anchor (abs), model predicts clean delta_normed
            # When delta_dynamic_bev: layer 1+ uses cumsum(denorm(pred)) for BEV positions
            delta_to_abs_fn = None
            if self.delta_dynamic_bev:
                delta_to_abs_fn = lambda dn: self._cumulate_trajectory(self.denorm_delta(dn))
            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature, noisy_delta_normed, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding,
                clean_anchor=plan_anchor, predict_delta=True,
                delta_to_abs_fn=delta_to_abs_fn
            )

            # Compute loss: denorm delta → cumsum → abs, compare with GT
            target_traj = targets["trajectory"]
            trajectory_loss_dict = {}
            ret_traj_loss = 0
            for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
                poses_reg_abs = self._cumulate_trajectory(self.denorm_delta(poses_reg))
                layer_loss = self.loss_computer(poses_reg_abs, poses_cls, target_traj, plan_anchor)
                trajectory_loss_dict[f"trajectory_loss_{idx}"] = layer_loss
                ret_traj_loss += layer_loss

            # Select best mode
            mode_idx = poses_cls_list[-1].argmax(dim=-1)
            mode_idx_expanded = mode_idx[:, None, None, None].expand(-1, 1, self.num_poses, 2)
            best_reg_normed = torch.gather(poses_reg_list[-1], 1, mode_idx_expanded).squeeze(1)
            best_reg = self._cumulate_trajectory(self.denorm_delta(best_reg_normed))

        else:
            # ========== Absolute mode (default) ==========
            odo_info_fut = self.norm_odo(plan_anchor)

            if self.use_full_diffusion:
                # Full diffusion: noise GT (not anchor)
                target_traj = targets["trajectory"][:, :self.num_poses]  # (bs, T, 2)
                gt_normed = self.norm_odo(target_traj)  # (bs, T, 2)
                clean_sample = gt_normed.unsqueeze(1).expand(-1, self.ego_fut_mode, -1, -1)
            else:
                clean_sample = odo_info_fut

            noise = torch.randn(clean_sample.shape, device=device)
            noisy_traj_points = self.diffusion_scheduler.add_noise(
                original_samples=clean_sample, noise=noise, timesteps=timesteps
            ).float()
            noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
            if not self.use_normalized_forward:
                noisy_traj_points = self.denorm_odo(noisy_traj_points)

            # Position embedding
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)

            # Timestep embedding
            time_embed = self.time_mlp(timesteps).unsqueeze(1)

            # Decoder: full diffusion always uses anchor for BEV (noisy positions meaningless at high t)
            use_anchor_bev = self.fix_bev_at_anchor or self.use_full_diffusion
            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding,
                clean_anchor=plan_anchor, predict_delta=False,
                fix_bev_at_anchor=use_anchor_bev
            )

            # Compute loss
            target_traj = targets["trajectory"]
            trajectory_loss_dict = {}
            ret_traj_loss = 0
            for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
                if self.use_normalized_forward:
                    poses_reg_abs = self.denorm_odo(poses_reg)
                    layer_loss = self.loss_computer(poses_reg_abs, poses_cls, target_traj, plan_anchor)
                else:
                    layer_loss = self.loss_computer(poses_reg, poses_cls, target_traj, plan_anchor)
                trajectory_loss_dict[f"trajectory_loss_{idx}"] = layer_loss
                ret_traj_loss += layer_loss

            # Select best mode
            mode_idx = poses_cls_list[-1].argmax(dim=-1)
            mode_idx_expanded = mode_idx[:, None, None, None].expand(-1, 1, self.num_poses, 2)
            best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx_expanded).squeeze(1)
            if self.use_normalized_forward:
                best_reg = self.denorm_odo(best_reg)

        return {
            "trajectory": best_reg,
            "trajectory_loss": ret_traj_loss,
            "trajectory_loss_dict": trajectory_loss_dict,
        }

    def forward_test(self, ego_query, agents_query, bev_feature,
                     bev_spatial_shape, status_encoding, targets=None):
        step_num = self.num_diffusion_steps
        bs = ego_query.shape[0]
        device = ego_query.device

        self.diffusion_scheduler.set_timesteps(1000, device)
        if self.use_full_diffusion:
            # Full diffusion: DDIM schedule spans [0, num_train_timesteps)
            total_t = self.num_train_timesteps
        else:
            total_t = self.trunc_timesteps
        step_ratio = total_t / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        plan_anchor = self.plan_anchor.unsqueeze(0).expand(bs, -1, -1, -1)
        alphas_cumprod = self.diffusion_scheduler.alphas_cumprod
        init_t = torch.ones((bs,), device=device, dtype=torch.long) * (total_t - 1)

        if self.predict_delta:
            return self._forward_test_delta(
                ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, targets, plan_anchor, alphas_cumprod,
                init_t, roll_timesteps, step_num, bs, device
            )
        else:
            return self._forward_test_abs(
                ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, targets, plan_anchor, alphas_cumprod,
                init_t, roll_timesteps, step_num, bs, device
            )

    def _forward_test_delta(self, ego_query, agents_query, bev_feature,
                            bev_spatial_shape, status_encoding, targets,
                            plan_anchor, alphas_cumprod, init_t,
                            roll_timesteps, step_num, bs, device):
        """DDIM inference in Z-score normalized delta space."""
        # 1. Init: noise in delta-normed space
        plan_anchor_delta = self.plan_anchor_delta.unsqueeze(0).expand(bs, -1, -1, -1)
        anchor_delta_normed = self.norm_delta(plan_anchor_delta)
        noise = torch.randn(anchor_delta_normed.shape, device=device)
        if self.use_full_diffusion:
            # Pure noise start (anchor info destroyed at t=999)
            img = noise
        else:
            img = self.diffusion_scheduler.add_noise(
                original_samples=anchor_delta_normed, noise=noise, timesteps=init_t
            )
        ego_fut_mode = img.shape[1]

        # Prepare GT delta normed for debug
        gt_delta_normed = None
        if targets is not None and 'trajectory' in targets:
            gt_abs = targets['trajectory'].detach()[:, :self.num_poses]
            gt_delta = self._traj_to_delta(gt_abs)
            gt_delta_normed = self.norm_delta(gt_delta)

        # 2. DDIM denoising loop in delta-normed space
        poses_reg = None
        poses_cls = None
        prev_best_idx = None
        _debug_first_batch = False  # not getattr(self, '_test_debug_printed', False)
        for step_i, k in enumerate(roll_timesteps):
            t_cur = k.item()
            t_next = roll_timesteps[step_i + 1].item() if step_i + 1 < len(roll_timesteps) else 0

            # Position embed from x_t (noisy delta normed)
            traj_pos_embed = gen_sineembed_for_position(img, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

            # Timestep embedding
            timesteps = k
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
            elif timesteps.dim() == 0:
                timesteps = timesteps[None].to(device)
            timesteps = timesteps.expand(bs)
            time_embed = self.time_mlp(timesteps).unsqueeze(1)

            # Decoder: BEV at anchor abs, predict clean delta normed
            # When delta_dynamic_bev: layer 1+ uses cumsum(denorm(pred)) for BEV positions
            delta_to_abs_fn = None
            if self.delta_dynamic_bev:
                delta_to_abs_fn = lambda dn: self._cumulate_trajectory(self.denorm_delta(dn))
            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature, img, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding,
                clean_anchor=plan_anchor, predict_delta=True,
                delta_to_abs_fn=delta_to_abs_fn
            )
            poses_reg = poses_reg_list[-1]  # predicted clean delta normed
            poses_cls = poses_cls_list[-1]

            # DDIM in delta-normed space: model directly predicts x0
            pred_x0_normed = poses_reg
            alpha_t = alphas_cumprod[t_cur]
            alpha_next = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
            pred_eps = (img - alpha_t.sqrt() * pred_x0_normed) / (1 - alpha_t).sqrt().clamp(min=1e-8)

            if _debug_first_batch:
                best_idx = poses_cls.detach().argmax(dim=-1)
                print(f"\n  [DDIM-delta step {step_i}] t={t_cur}->{t_next}, sqrt(1-alpha_t)={(1-alpha_t).sqrt().item():.4f}")
                for b in range(min(bs, 1)):
                    bi = best_idx[b].item()
                    print(f"    batch {b}: best_mode={bi}, cls_top3={torch.topk(poses_cls[b], 3).indices.tolist()}")

                    if gt_delta_normed is not None:
                        gt_dn = gt_delta_normed[b]  # (T, 2)
                        # anchor vs GT in delta-normed space
                        anchor_gt_dist = (anchor_delta_normed[b] - gt_dn.unsqueeze(0)).abs().mean(dim=(-2, -1))
                        # pred_x0 vs GT in delta-normed space
                        pred_error = (pred_x0_normed[b] - gt_dn.unsqueeze(0)).abs().mean(dim=(-2, -1))
                        # x_t vs anchor: drift indicator
                        x_t_drift = (img[b] - anchor_delta_normed[b]).abs().mean(dim=(-2, -1))
                        # Physical L2: convert pred_x0 to abs trajectory
                        pred_x0_abs = self._cumulate_trajectory(self.denorm_delta(pred_x0_normed[b]))
                        gt_meters = targets['trajectory'][b, :self.num_poses].detach().to(device=device, dtype=img.dtype)
                        l2_per_mode = (pred_x0_abs - gt_meters.unsqueeze(0)).norm(dim=-1).mean(dim=-1)

                        print(f"      anchor vs GT (delta-normed MAE):  best={anchor_gt_dist[bi]:.4f}, "
                              f"others_mean={anchor_gt_dist.sum().sub(anchor_gt_dist[bi]).div(self.ego_fut_mode-1):.4f}")
                        print(f"      pred_x0 vs GT (delta-normed MAE): best={pred_error[bi]:.4f}, "
                              f"others_mean={pred_error.sum().sub(pred_error[bi]).div(self.ego_fut_mode-1):.4f}")
                        print(f"      pred_x0 vs GT (L2 meters):        best={l2_per_mode[bi]:.4f}, "
                              f"others_mean={l2_per_mode.sum().sub(l2_per_mode[bi]).div(self.ego_fut_mode-1):.4f}")
                        print(f"      x_t vs anchor (delta-normed MAE): best={x_t_drift[bi]:.4f}, "
                              f"others_mean={x_t_drift.sum().sub(x_t_drift[bi]).div(self.ego_fut_mode-1):.4f}"
                              f"  <- {'~noise level' if step_i == 0 else 'OOD if >> noise level'}")

                    pred_eps_mag = pred_eps[b].abs().mean(dim=(-2, -1))
                    print(f"      pred_eps_mag (should~1.0): best={pred_eps_mag[bi]:.4f}, "
                          f"others_mean={pred_eps_mag.sum().sub(pred_eps_mag[bi]).div(self.ego_fut_mode-1):.4f}, "
                          f"others_max={pred_eps_mag.clone().scatter_(0, best_idx[b:b+1], 0).max():.4f}")

                    if prev_best_idx is not None:
                        print(f"      mode_changed: {bi != prev_best_idx[b].item()} (was {prev_best_idx[b].item()})")

                prev_best_idx = best_idx

            # DDIM step in delta-normed space
            img = alpha_next.sqrt() * pred_x0_normed + (1 - alpha_next).sqrt() * pred_eps

        if _debug_first_batch:
            self._test_debug_printed = True

        # Final: denorm delta → cumsum → abs trajectory
        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx_expanded = mode_idx[:, None, None, None].expand(-1, 1, self.num_poses, 2)
        best_delta_normed = torch.gather(poses_reg, 1, mode_idx_expanded).squeeze(1)
        best_reg = self._cumulate_trajectory(self.denorm_delta(best_delta_normed))

        return {"trajectory": best_reg}

    def _forward_test_abs(self, ego_query, agents_query, bev_feature,
                          bev_spatial_shape, status_encoding, targets,
                          plan_anchor, alphas_cumprod, init_t,
                          roll_timesteps, step_num, bs, device):
        """DDIM inference in absolute normalized space (original behavior)."""
        # 1. Init noise
        anchor_normed = self.norm_odo(plan_anchor)
        noise = torch.randn(anchor_normed.shape, device=device)
        if self.use_full_diffusion:
            # Pure noise start
            img = noise
        else:
            img = self.diffusion_scheduler.add_noise(
                original_samples=anchor_normed, noise=noise, timesteps=init_t
            )
        ego_fut_mode = img.shape[1]

        # Prepare GT in normed space for debug
        gt_normed = None
        if targets is not None and 'trajectory' in targets:
            gt_abs = targets['trajectory'].detach()[:, :self.num_poses]
            gt_normed = self.norm_odo(gt_abs)

        # 2. DDIM denoising loop
        poses_reg = None
        poses_cls = None
        prev_best_idx = None
        _debug_first_batch = False  # not getattr(self, '_test_debug_printed', False)
        for step_i, k in enumerate(roll_timesteps):
            t_cur = k.item()
            t_next = roll_timesteps[step_i + 1].item() if step_i + 1 < len(roll_timesteps) else 0

            x_boxes = torch.clamp(img, min=-1, max=1)
            if self.use_normalized_forward:
                noisy_traj_points = x_boxes
            else:
                noisy_traj_points = self.denorm_odo(x_boxes)

            # Encode trajectory positions
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

            # Timestep embedding
            timesteps = k
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
            elif timesteps.dim() == 0:
                timesteps = timesteps[None].to(device)
            timesteps = timesteps.expand(bs)
            time_embed = self.time_mlp(timesteps).unsqueeze(1)

            # Run decoder: full diffusion always uses anchor for BEV
            use_anchor_bev = self.fix_bev_at_anchor or self.use_full_diffusion
            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding,
                clean_anchor=plan_anchor, predict_delta=False,
                fix_bev_at_anchor=use_anchor_bev
            )
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]

            # Compute pred_x0 in normed space and pred_eps
            if self.use_normalized_forward:
                pred_x0_normed = poses_reg
            else:
                pred_x0_normed = self.norm_odo(poses_reg)
            alpha_t = alphas_cumprod[t_cur]
            alpha_next = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
            pred_eps = (img - alpha_t.sqrt() * pred_x0_normed) / (1 - alpha_t).sqrt().clamp(min=1e-8)

            if _debug_first_batch:
                best_idx = poses_cls.detach().argmax(dim=-1)
                print(f"\n  [DDIM step {step_i}] t={t_cur}->{t_next}, sqrt(1-alpha_t)={(1-alpha_t).sqrt().item():.4f}")
                for b in range(min(bs, 1)):
                    bi = best_idx[b].item()
                    print(f"    batch {b}: best_mode={bi}, cls_top3={torch.topk(poses_cls[b], 3).indices.tolist()}")

                    if gt_normed is not None:
                        gt_n = gt_normed[b]
                        anchor_gt_dist = (anchor_normed[b] - gt_n.unsqueeze(0)).abs().mean(dim=(-2, -1))
                        pred_error_per_mode = (pred_x0_normed[b] - gt_n.unsqueeze(0)).abs().mean(dim=(-2, -1))
                        x_t_anchor_dist = (img[b] - anchor_normed[b]).abs().mean(dim=(-2, -1))
                        pred_x0_abs = self.denorm_odo(pred_x0_normed[b])
                        gt_meters = targets['trajectory'][b, :self.num_poses].detach().to(device=device, dtype=img.dtype)
                        l2_per_mode = (pred_x0_abs - gt_meters.unsqueeze(0)).norm(dim=-1).mean(dim=-1)

                        print(f"      anchor vs GT (normed MAE):  best={anchor_gt_dist[bi]:.4f}, "
                              f"others_mean={anchor_gt_dist.sum().sub(anchor_gt_dist[bi]).div(self.ego_fut_mode-1):.4f}")
                        print(f"      pred_x0 vs GT (normed MAE): best={pred_error_per_mode[bi]:.4f}, "
                              f"others_mean={pred_error_per_mode.sum().sub(pred_error_per_mode[bi]).div(self.ego_fut_mode-1):.4f}")
                        print(f"      pred_x0 vs GT (L2 meters):  best={l2_per_mode[bi]:.4f}, "
                              f"others_mean={l2_per_mode.sum().sub(l2_per_mode[bi]).div(self.ego_fut_mode-1):.4f}")
                        print(f"      x_t vs anchor (normed MAE): best={x_t_anchor_dist[bi]:.4f}, "
                              f"others_mean={x_t_anchor_dist.sum().sub(x_t_anchor_dist[bi]).div(self.ego_fut_mode-1):.4f}"
                              f"  <- {'~noise level' if step_i == 0 else 'OOD if >> noise level'}")

                    pred_eps_mag = pred_eps[b].abs().mean(dim=(-2, -1))
                    print(f"      pred_eps_mag (should~1.0): best={pred_eps_mag[bi]:.4f}, "
                          f"others_mean={pred_eps_mag.sum().sub(pred_eps_mag[bi]).div(self.ego_fut_mode-1):.4f}, "
                          f"others_max={pred_eps_mag.clone().scatter_(0, best_idx[b:b+1], 0).max():.4f}")

                    if prev_best_idx is not None:
                        print(f"      mode_changed: {bi != prev_best_idx[b].item()} (was {prev_best_idx[b].item()})")

                prev_best_idx = best_idx

            # DDIM step
            img = alpha_next.sqrt() * pred_x0_normed + (1 - alpha_next).sqrt() * pred_eps

        if _debug_first_batch:
            self._test_debug_printed = True

        # Select best mode
        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx_expanded = mode_idx[:, None, None, None].expand(-1, 1, self.num_poses, 2)
        best_reg = torch.gather(poses_reg, 1, mode_idx_expanded).squeeze(1)
        if self.use_normalized_forward:
            best_reg = self.denorm_odo(best_reg)

        return {"trajectory": best_reg}
