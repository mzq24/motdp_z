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
                status_encoding):
        """
        Args:
            traj_feature: (bs, num_modes, d_model)
            noisy_traj_points: (bs, num_modes, num_poses, 2)
            bev_feature: (bs, d_model, H, W)
            bev_spatial_shape: (H, W)
            agents_query: (bs, num_agents, d_model)
            ego_query: (bs, 1, d_model)
            time_embed: (bs, 1, d_model)
            status_encoding: (bs, 1, d_model)
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
        # Add offset to noisy trajectory points (residual prediction)
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
                status_encoding):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points

        for layer in self.layers:
            poses_reg, poses_cls = layer(
                traj_feature, traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding
            )
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            # Use predicted trajectory for next layer (detached)
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

        # Normalization parameters
        self.norm_x_offset = config.norm_x_offset
        self.norm_x_range = config.norm_x_range
        self.norm_y_offset = config.norm_y_offset
        self.norm_y_range = config.norm_y_range

        # DDIM scheduler
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        # Load anchors
        plan_anchor = np.load(config.plan_anchor_path)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )  # (num_modes, num_poses, 2)

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

    def forward(self, ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, targets=None):
        if self.training:
            return self.forward_train(
                ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, targets
            )
        else:
            return self.forward_test(
                ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding
            )

    def forward_train(self, ego_query, agents_query, bev_feature,
                      bev_spatial_shape, status_encoding, targets):
        bs = ego_query.shape[0]
        device = ego_query.device

        # 1. Add truncated noise to plan anchors
        plan_anchor = self.plan_anchor.unsqueeze(0).expand(bs, -1, -1, -1)  # (bs, M, T, 2)
        odo_info_fut = self.norm_odo(plan_anchor)

        timesteps = torch.randint(0, self.trunc_timesteps, (bs,), device=device)
        noise = torch.randn(odo_info_fut.shape, device=device)
        noisy_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=odo_info_fut, noise=noise, timesteps=timesteps
        ).float()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(noisy_traj_points)

        ego_fut_mode = noisy_traj_points.shape[1]

        # 2. Encode noisy trajectory positions
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)  # (bs, M, T*64)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)  # (bs, M, d_model)

        # 3. Timestep embedding
        time_embed = self.time_mlp(timesteps).unsqueeze(1)  # (bs, 1, d_model)

        # 4. Run diffusion decoder
        poses_reg_list, poses_cls_list = self.diff_decoder(
            traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding
        )

        # 5. Compute loss for each decoder layer
        target_traj = targets["trajectory"]  # (bs, T, 2)
        trajectory_loss_dict = {}
        ret_traj_loss = 0
        for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
            layer_loss = self.loss_computer(poses_reg, poses_cls, target_traj, plan_anchor)
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = layer_loss
            ret_traj_loss += layer_loss

        # 6. Select best mode from last layer
        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx_expanded = mode_idx[:, None, None, None].expand(-1, 1, self.num_poses, 2)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx_expanded).squeeze(1)

        return {
            "trajectory": best_reg,
            "trajectory_loss": ret_traj_loss,
            "trajectory_loss_dict": trajectory_loss_dict,
        }

    def forward_test(self, ego_query, agents_query, bev_feature,
                     bev_spatial_shape, status_encoding):
        step_num = self.num_diffusion_steps
        bs = ego_query.shape[0]
        device = ego_query.device

        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = self.trunc_timesteps / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        # 1. Add truncated noise to plan anchors
        plan_anchor = self.plan_anchor.unsqueeze(0).expand(bs, -1, -1, -1)
        img = self.norm_odo(plan_anchor)
        noise = torch.randn(img.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = self.diffusion_scheduler.add_noise(
            original_samples=img, noise=noise, timesteps=trunc_timesteps
        )
        ego_fut_mode = img.shape[1]

        # 2. DDIM denoising loop
        poses_reg = None
        poses_cls = None
        for k in roll_timesteps:
            x_boxes = torch.clamp(img, min=-1, max=1)
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

            # Run decoder
            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding
            )
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]

            # DDIM step
            x_start = self.norm_odo(poses_reg)
            img = self.diffusion_scheduler.step(
                model_output=x_start, timestep=k, sample=img
            ).prev_sample

        # Select best mode
        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx_expanded = mode_idx[:, None, None, None].expand(-1, 1, self.num_poses, 2)
        best_reg = torch.gather(poses_reg, 1, mode_idx_expanded).squeeze(1)

        return {"trajectory": best_reg}
