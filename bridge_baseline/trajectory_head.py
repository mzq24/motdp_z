"""
Trajectory prediction head with DDBM (Denoising Diffusion Bridge Model).

Key differences from dd_baseline:
- DDBM scheduler: Brownian Bridge forward/reverse between GT↔anchor (no OOD)
- Dual-branch BEV: noisy trajectory positions + anchor positions, concatenated before FiLM
- Separate classification decoder (diff_decoder_cls) runs at clean anchor without timestep
- Per-waypoint z-score normalization (stats from training data)
- Training: add bridge noise to GT (not anchor); inference: bridge loop starting from anchor
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .modules.blocks import (
    linear_relu_ln, gen_sineembed_for_position, GridSampleCrossBEVAttention
)
from .modules.modulation import ModulationLayer
from .modules.refinement import DiffMotionPlanningRefinementModule
from .modules.sinusoidal_emb import SinusoidalPosEmb
from .modules.loss import LossComputer
from .modules.ddbm_scheduler import DDBMScheduler
from .config import BridgeBaselineConfig


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class CustomTransformerDecoderLayer(nn.Module):
    """Dual-branch diffusion decoder layer.

    Branch 1: GridSample BEV at noisy trajectory positions
    Branch 2: GridSample BEV at clean anchor positions (starts from branch-1 output)
    Both branches: ego cross-attention + FFN + LayerNorm
    Concat → FiLM (timestep) → DiffMotionPlanningRefinementModule
    poses_reg += noisy_traj_points (residual)
    """
    def __init__(self, config: BridgeBaselineConfig):
        super().__init__()
        d_model = config.tf_d_model
        d_ffn = config.tf_d_ffn
        num_poses = config.num_poses

        self.dropout1 = nn.Dropout(0.1)
        self.dropout1_T = nn.Dropout(0.1)

        # Branch 1: BEV attention at noisy trajectory positions
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            embed_dims=d_model,
            num_heads=config.tf_num_head,
            num_points=num_poses,
            in_bev_dims=d_model,
            lidar_max_x=config.lidar_max_x,
            lidar_max_y=config.lidar_max_y,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            d_model, config.tf_num_head,
            dropout=config.tf_dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Branch 2: BEV attention at anchor positions
        self.cross_bev_attention_T = GridSampleCrossBEVAttention(
            embed_dims=d_model,
            num_heads=config.tf_num_head,
            num_points=num_poses,
            in_bev_dims=d_model,
            lidar_max_x=config.lidar_max_x,
            lidar_max_y=config.lidar_max_y,
        )
        self.cross_ego_attention_T = nn.MultiheadAttention(
            d_model, config.tf_num_head,
            dropout=config.tf_dropout, batch_first=True,
        )
        self.ffn_T = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model),
        )
        self.norm1_T = nn.LayerNorm(d_model)
        self.norm2_T = nn.LayerNorm(d_model)
        self.norm3_T = nn.LayerNorm(d_model)

        # Concat [branch1 || branch2] → FiLM → task decoder
        self.time_modulation = ModulationLayer(d_model * 2, d_model)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=d_model * 2,
            ego_fut_ts=num_poses,
            ego_fut_mode=config.ego_fut_mode,
        )

    def forward(
        self,
        traj_feature: torch.Tensor,       # (bs, M, d_model)
        noisy_traj_points: torch.Tensor,  # (bs, M, T, 2) abs coords
        plan_anchor: torch.Tensor,        # (bs, M, T, 2) abs coords
        bev_feature: torch.Tensor,        # (bs, d_model, H, W)
        bev_spatial_shape: Tuple[int, int],
        ego_query: torch.Tensor,          # (bs, 1, d_model)
        time_embed: torch.Tensor,         # (bs, 1, d_model)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- Branch 1: BEV at noisy trajectory ---
        traj_f = self.cross_bev_attention(
            traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape
        )
        # Branch 2 starts from branch-1 output (shares early BEV context)
        traj_f_T = self.cross_bev_attention_T(
            traj_f, plan_anchor, bev_feature, bev_spatial_shape
        )

        # Branch 1: norm1 → ego attention → norm2 → FFN → norm3
        traj_f = self.norm1(traj_f)
        traj_f = traj_f + self.dropout1(
            self.cross_ego_attention(traj_f, ego_query, ego_query)[0]
        )
        traj_f = self.norm2(traj_f)
        traj_f = self.norm3(self.ffn(traj_f))

        # Branch 2: norm1_T → ego attention → norm2_T → FFN_T → norm3_T
        traj_f_T = self.norm1_T(traj_f_T)
        traj_f_T = traj_f_T + self.dropout1_T(
            self.cross_ego_attention_T(traj_f_T, ego_query, ego_query)[0]
        )
        traj_f_T = self.norm2_T(traj_f_T)
        traj_f_T = self.norm3_T(self.ffn_T(traj_f_T))

        # Concat → FiLM → task decode
        combined = torch.cat([traj_f, traj_f_T], dim=-1)  # (bs, M, d_model*2)
        combined = self.time_modulation(combined, time_embed)
        poses_reg, poses_cls = self.task_decoder(combined)

        # Residual: predicted offset + noisy trajectory
        poses_reg = poses_reg + noisy_traj_points

        return poses_reg, poses_cls


class CustomTransformerDecoder(nn.Module):
    """Stack of dual-branch decoder layers with iterative trajectory refinement."""
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(
        self,
        traj_feature: torch.Tensor,
        noisy_traj_points: torch.Tensor,
        plan_anchor: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: Tuple[int, int],
        ego_query: torch.Tensor,
        time_embed: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points

        for layer in self.layers:
            poses_reg, poses_cls = layer(
                traj_feature, traj_points, plan_anchor,
                bev_feature, bev_spatial_shape, ego_query, time_embed,
            )
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            # Next layer samples BEV at the refined trajectory
            traj_points = poses_reg.clone().detach()

        return poses_reg_list, poses_cls_list


class CustomTransformerDecoderLayerCls(nn.Module):
    """Classification-only decoder layer.

    No timestep conditioning; BEV sampled at clean anchor positions.
    Used to select the best mode BEFORE the bridge diffusion loop.
    """
    def __init__(self, config: BridgeBaselineConfig):
        super().__init__()
        d_model = config.tf_d_model
        d_ffn = config.tf_d_ffn
        num_poses = config.num_poses

        self.dropout1 = nn.Dropout(0.1)

        self.cross_bev_attention = GridSampleCrossBEVAttention(
            embed_dims=d_model,
            num_heads=config.tf_num_head,
            num_points=num_poses,
            in_bev_dims=d_model,
            lidar_max_x=config.lidar_max_x,
            lidar_max_y=config.lidar_max_y,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            d_model, config.tf_num_head,
            dropout=config.tf_dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.task_decoder_cls = nn.Sequential(
            *linear_relu_ln(d_model, 1, 2),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        traj_feature: torch.Tensor,  # (bs, M, d_model)
        plan_anchor: torch.Tensor,   # (bs, M, T, 2) abs coords
        bev_feature: torch.Tensor,   # (bs, d_model, H, W)
        bev_spatial_shape: Tuple[int, int],
        ego_query: torch.Tensor,     # (bs, 1, d_model)
    ) -> torch.Tensor:               # (bs, M)
        traj_f = self.cross_bev_attention(
            traj_feature, plan_anchor, bev_feature, bev_spatial_shape
        )
        traj_f = self.norm1(traj_f)
        traj_f = traj_f + self.dropout1(
            self.cross_ego_attention(traj_f, ego_query, ego_query)[0]
        )
        traj_f = self.norm2(traj_f)
        traj_f = self.norm3(self.ffn(traj_f))
        poses_cls = self.task_decoder_cls(traj_f).squeeze(-1)  # (bs, M)
        return poses_cls


class CustomTransformerDecoderCls(nn.Module):
    """Stack of classification-only decoder layers."""
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(
        self,
        traj_feature: torch.Tensor,
        plan_anchor: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: Tuple[int, int],
        ego_query: torch.Tensor,
    ) -> List[torch.Tensor]:
        poses_cls_list = []
        for layer in self.layers:
            poses_cls = layer(
                traj_feature, plan_anchor, bev_feature, bev_spatial_shape, ego_query
            )
            poses_cls_list.append(poses_cls)
        return poses_cls_list


class TrajectoryHead(nn.Module):
    """Trajectory prediction head with DDBM bridge diffusion.

    Architecture:
        - diff_decoder: dual-branch (noisy + anchor BEV), predicts trajectory offsets
        - diff_decoder_cls: single-branch (anchor BEV only), predicts mode classification
        - DDBMScheduler: Brownian Bridge noise schedule (GT↔anchor, not DDPM)

    Training:
        x_0 = norm(GT route[:T])  ← bridge endpoint (data)
        x_T = norm(anchor)         ← bridge endpoint (prior)
        x_t = a_t*x_T + b_t*x_0 + c_t*noise  (bridge forward)
        predict x_0 from x_t

    Inference:
        1. Run diff_decoder_cls at clean anchor → select best mode
        2. x_T = anchor (start of bridge)
        3. For t = T→0: x_{t-1} = bridge_sample_step(x_t, x_0_pred, x_T)
        4. Return x_0 for the selected mode
    """
    def __init__(self, config: BridgeBaselineConfig):
        super().__init__()
        d_model = config.tf_d_model
        num_poses = config.num_poses
        self.num_poses = num_poses
        self.ego_fut_mode = config.ego_fut_mode
        self.step_num = config.step_num

        # Per-waypoint z-score normalization tensors (1, 1, T, 2)
        x_mean = torch.tensor(config.norm_x_mean, dtype=torch.float32)  # (T,)
        x_std  = torch.tensor(config.norm_x_std,  dtype=torch.float32)
        y_mean = torch.tensor(config.norm_y_mean, dtype=torch.float32)
        y_std  = torch.tensor(config.norm_y_std,  dtype=torch.float32)
        mean = torch.stack([x_mean, y_mean], dim=-1).unsqueeze(0).unsqueeze(0)  # (1,1,T,2)
        std  = torch.stack([x_std,  y_std],  dim=-1).unsqueeze(0).unsqueeze(0)
        self.register_buffer('norm_mean', mean)  # (1,1,T,2)
        self.register_buffer('norm_std',  std)   # (1,1,T,2)

        # DDBM scheduler (VP schedule, beta_d=2.0, beta_min=0.1, T=1.0)
        self.diffusion_scheduler = DDBMScheduler(
            beta_d=config.beta_d,
            beta_min=config.beta_min,
            T=1.0,
        )

        # Anchors in absolute coordinates (M, T, 2)
        plan_anchor = np.load(config.plan_anchor_path)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )

        # Trajectory positional encoder: (bs, M, T*64) → (bs, M, d_model)
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

        # Reg decoder: dual-branch, predicts poses_reg + poses_cls
        diff_decoder_layer = CustomTransformerDecoderLayer(config)
        self.diff_decoder = CustomTransformerDecoder(
            diff_decoder_layer, config.num_decoder_layers
        )

        # Cls decoder: single-branch at anchor, predicts mode scores
        diff_decoder_layer_cls = CustomTransformerDecoderLayerCls(config)
        self.diff_decoder_cls = CustomTransformerDecoderCls(
            diff_decoder_layer_cls, config.num_decoder_layers
        )

        # Loss: focal cls + L1 reg
        self.loss_computer = LossComputer(
            cls_loss_weight=config.trajectory_cls_weight,
            reg_loss_weight=config.trajectory_reg_weight,
        )

    def norm_odo(self, traj: torch.Tensor) -> torch.Tensor:
        """Per-waypoint z-score normalization. traj: (..., T, 2)"""
        return (traj - self.norm_mean) / self.norm_std.clamp(min=1e-6)

    def denorm_odo(self, traj_normed: torch.Tensor) -> torch.Tensor:
        """Per-waypoint z-score denormalization. traj_normed: (..., T, 2)"""
        return traj_normed * self.norm_std + self.norm_mean

    def _encode_traj(self, traj_abs: torch.Tensor, bs: int, num_modes: int) -> torch.Tensor:
        """Encode absolute trajectory positions to feature vectors.

        Args:
            traj_abs: (bs, M, T, 2) absolute trajectory coordinates
            bs: batch size
            num_modes: number of modes M
        Returns:
            traj_feature: (bs, M, d_model)
        """
        traj_pos_embed = gen_sineembed_for_position(traj_abs, hidden_dim=64)  # (bs, M, T, 64)
        traj_pos_embed = traj_pos_embed.flatten(-2)                             # (bs, M, T*64)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)                # (bs, M, d_model)
        return traj_feature.view(bs, num_modes, -1)

    def forward(self, ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, targets=None):
        if self.training:
            return self.forward_train(
                ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, targets,
            )
        else:
            return self.forward_test(
                ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, targets=targets,
            )

    def forward_train(self, ego_query, agents_query, bev_feature,
                      bev_spatial_shape, status_encoding, targets):
        """
        Training forward pass.

        1. x_0 = norm(GT route[:T]) broadcast to M modes
        2. x_T = norm(anchor)
        3. x_t = DDBM.add_noise(x_0, x_T, noise, t) with t ~ Uniform[1, 1000]
        4. noisy_abs = denorm(x_t)  ← used for BEV sampling (abs coords)
        5. diff_decoder(noisy_abs, anchor, bev, ego, t) → poses_reg (abs), poses_cls
        6. diff_decoder_cls(anchor, bev, ego) → poses_cls_anchor
        7. Loss = LossComputer(poses_reg, poses_cls_anchor, GT, anchor)
        """
        bs = ego_query.shape[0]
        device = ego_query.device

        # GT trajectory: (bs, T, 2)
        gt_traj = targets["route"][:, :self.num_poses]

        # Anchors: (bs, M, T, 2)
        plan_anchor = self.plan_anchor.unsqueeze(0).expand(bs, -1, -1, -1)

        # x_0 = GT broadcast to M modes: (bs, M, T, 2)
        x_0 = gt_traj.unsqueeze(1).expand(-1, self.ego_fut_mode, -1, -1)

        # Normalize in z-score space
        x_0_norm = self.norm_odo(x_0)
        x_T_norm = self.norm_odo(plan_anchor)

        # DDBM bridge forward: add noise in normalized space
        timesteps = torch.randint(1, 1001, (bs,), device=device)
        noise = torch.randn_like(x_0_norm)
        noisy_norm = self.diffusion_scheduler.add_noise(
            t=timesteps, x0=x_0_norm, xT=x_T_norm, noise=noise,
        ).float()

        # BEV sampling uses absolute coordinates
        noisy_abs = self.denorm_odo(noisy_norm)

        # Encode noisy trajectory for reg decoder
        traj_feature = self._encode_traj(noisy_abs, bs, self.ego_fut_mode)

        # Encode anchor for cls decoder
        ak_traj_feature = self._encode_traj(plan_anchor, bs, self.ego_fut_mode)

        # Timestep embedding
        time_embed = self.time_mlp(timesteps).unsqueeze(1)  # (bs, 1, d_model)

        # Reg decoder: dual-branch, predicts trajectory in abs coords
        poses_reg_list, _ = self.diff_decoder(
            traj_feature, noisy_abs, plan_anchor,
            bev_feature, bev_spatial_shape, ego_query, time_embed,
        )

        # Cls decoder: no time embed, BEV at clean anchor
        poses_cls_list = self.diff_decoder_cls(
            ak_traj_feature, plan_anchor,
            bev_feature, bev_spatial_shape, ego_query,
        )

        # Loss: cls from cls decoder, reg from reg decoder (same number of layers)
        trajectory_loss_dict = {}
        ret_traj_loss = 0
        for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
            layer_loss = self.loss_computer(poses_reg, poses_cls, gt_traj, plan_anchor)
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = layer_loss
            ret_traj_loss = ret_traj_loss + layer_loss

        # Best mode (for monitoring, not for loss)
        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx_exp = mode_idx[:, None, None, None].expand(-1, 1, self.num_poses, 2)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx_exp).squeeze(1)

        return {
            "trajectory": best_reg,
            "trajectory_loss": ret_traj_loss,
            "trajectory_loss_dict": trajectory_loss_dict,
        }

    def forward_test(self, ego_query, agents_query, bev_feature,
                     bev_spatial_shape, status_encoding, targets=None):
        """
        Inference forward pass.

        1. Classify at clean anchor → select best mode index
        2. Start bridge loop from x_T = anchor
        3. For t = T→0 (step_num steps):
               x_0_pred = reg_decoder(x_t, anchor, bev, ego, t)
               x_{t-1} = DDBM.sample_step(t, t-1, norm(x_t), norm(x_0_pred), norm(anchor))
               x_t = denorm(x_{t-1})
        4. Return x_0 at the selected mode
        """
        bs = ego_query.shape[0]
        device = ego_query.device

        # Timestep schedule: [0, 50, 100, ..., 1000] for step_num=20 (21 values)
        ts = torch.arange(0, 1001, 1000 // self.step_num, device=device)  # (step_num+1,)

        # Anchors: (bs, M, T, 2)
        plan_anchor = self.plan_anchor.unsqueeze(0).expand(bs, -1, -1, -1)
        ego_fut_mode = plan_anchor.shape[1]

        # Encode anchor for cls decoder
        ak_traj_feature = self._encode_traj(plan_anchor, bs, ego_fut_mode)

        # Classification: select best mode BEFORE bridge loop
        poses_cls_list = self.diff_decoder_cls(
            ak_traj_feature, plan_anchor,
            bev_feature, bev_spatial_shape, ego_query,
        )
        mode_idx = poses_cls_list[-1].argmax(dim=-1)  # (bs,)
        mode_idx_exp = mode_idx[:, None, None, None].expand(-1, 1, self.num_poses, 2)

        # Bridge loop: start from clean anchor (x_T = anchor)
        xt = plan_anchor.clone()

        for i in range(self.step_num, 0, -1):
            t      = ts[i].float() * torch.ones([bs], device=device)   # e.g. 1000→50
            t_prev = ts[i - 1].float() * torch.ones([bs], device=device)

            # Encode current xt
            traj_feature = self._encode_traj(xt, bs, ego_fut_mode)
            time_embed = self.time_mlp(t).unsqueeze(1)  # (bs, 1, d_model)

            # Reg decoder: predict x_0 in abs coords
            poses_reg_list, _ = self.diff_decoder(
                traj_feature, xt, plan_anchor,
                bev_feature, bev_spatial_shape, ego_query, time_embed,
            )
            x0_pred = poses_reg_list[-1]  # (bs, M, T, 2) abs

            # DDBM bridge sample step (operates in normalized space)
            xt_prev_norm = self.diffusion_scheduler.sample_step(
                t=t,
                t_prev=t_prev,
                xt=self.norm_odo(xt),
                x0=self.norm_odo(x0_pred),
                xT=self.norm_odo(plan_anchor),
            )
            xt = self.denorm_odo(xt_prev_norm)

        # Select best mode
        x0_sample = torch.gather(xt, 1, mode_idx_exp).squeeze(1)  # (bs, T, 2)
        return {"trajectory": x0_sample}
