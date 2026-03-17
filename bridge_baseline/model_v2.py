"""
BDModelV2: Bridge Baseline v2 — proper BridgeDrive structure.

Key changes from model.py (v1):
  - Input BEV: (64, 64, 64) upsample feature only (no 1512-ch small BEV)
  - Ego query: built via PlanningContextEncoder (velocity/command/target_point tokens)
               + 6-layer TransformerDecoder (vs. 3-layer in v1)
  - Agent queries: unchanged (16 learned queries)
  - TrajectoryHead: unchanged (DDBM bridge diffusion)
  - No ego_status (14-dim) dependency

Speed prediction (two methods, both optional):
  - predict_target_speed: independent MLP decoder → 8-class two-hot speed distribution
  - diffusion_speed: speed appended as 11th waypoint (speed, 0) in DDBM route

Batch keys consumed (all available from CARLAImageDataset):
  - transfuser_bev_feature_upsample: (B, 64, 64, 64)
  - speed:                           (B, T) float  — last timestep used
  - command_hist:                    (B, T, 6)
  - target_point_hist:               (B, T, 2)
  - target_point_next_hist:          (B, T, 2)
"""
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .config import BridgeBaselineConfig
from .trajectory_head import TrajectoryHead
from .modules.blocks import linear_relu_ln
from .planning_encoder import PlanningContextEncoder


def _encode_two_hot(
    speed_gt: torch.Tensor,          # (B,) m/s
    classes: torch.Tensor,           # (C,)
    brake: torch.Tensor,             # (B,) bool
) -> torch.Tensor:                   # (B, C) soft labels
    """Two-hot encode scalar speed values into a soft distribution."""
    B, C = speed_gt.shape[0], classes.shape[0]
    labels = torch.zeros(B, C, dtype=speed_gt.dtype, device=speed_gt.device)

    # Brake → encode as class 0 (stopped)
    labels[brake, 0] = 1.0

    non_brake = ~brake
    if non_brake.any():
        scalars = speed_gt[non_brake]
        # Clamp to valid range
        scalars = scalars.clamp(min=0.0, max=classes[-1])
        # Find upper bin index
        upper_idx = torch.searchsorted(classes, scalars, right=False)
        upper_idx = upper_idx.clamp(min=1, max=C - 1)
        lower_idx = upper_idx - 1

        lower_val = classes[lower_idx]
        upper_val = classes[upper_idx]
        width = (upper_val - lower_val).clamp(min=1e-6)
        upper_weight = (scalars - lower_val) / width
        lower_weight = 1.0 - upper_weight

        nb_indices = non_brake.nonzero(as_tuple=True)[0]
        labels[nb_indices, lower_idx] = lower_weight
        labels[nb_indices, upper_idx] = upper_weight

    return labels


def _decode_two_hot(
    dist: torch.Tensor,    # (B, C) after softmax
    classes: torch.Tensor, # (C,)
) -> torch.Tensor:         # (B,)
    return (dist * classes.unsqueeze(0)).sum(dim=-1)


class BDModelV2(nn.Module):
    def __init__(self, config: BridgeBaselineConfig):
        super().__init__()
        self.config = config
        d_model = config.tf_d_model   # 256

        # BEV projection: (64, 64, 64) -> (d_model, 64, 64)
        # Used for GridSampleCrossBEVAttention inside TrajectoryHead
        self.bev_proj = nn.Conv2d(config.bev_feature_upsample_dim, d_model, kernel_size=1)

        # PlanningContextEncoder: BEV tokens + status tokens
        self.context_encoder = PlanningContextEncoder(
            in_bev_channels=config.bev_feature_upsample_dim,
            token_dim=d_model,
            max_speed=config.max_speed,
            tp_norm=tuple(config.tp_norm),
        )

        # Learnable ego query; if predict_target_speed also a speed query
        if config.predict_target_speed:
            self.query_embedding = nn.Embedding(2, d_model)  # [0]=ego, [1]=speed
        else:
            self.query_embedding = nn.Embedding(1, d_model)

        # 6-layer TransformerDecoder: queries attend to context tokens
        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.tf_decoder = nn.TransformerDecoder(tf_decoder_layer, num_layers=6)

        # Learned agent queries
        self.agent_queries = nn.Embedding(config.num_agent_queries, d_model)

        # Method 1: independent speed decoder
        if config.predict_target_speed:
            n_classes = len(config.target_speed_classes)
            self.target_speed_decoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, n_classes),
            )
            self.register_buffer(
                'speed_classes',
                torch.tensor(config.target_speed_classes, dtype=torch.float32),
            )

        # predict_traj: use agent_pos[:6] as GT, dd_baseline anchor, traj norm stats
        if config.predict_traj:
            th_config = copy.copy(config)
            th_config.num_poses = 6
            th_config.plan_anchor_path = config.traj_anchor_path
            th_config.norm_x_mean = list(config.traj_norm_x_mean)
            th_config.norm_x_std  = list(config.traj_norm_x_std)
            th_config.norm_y_mean = list(config.traj_norm_y_mean)
            th_config.norm_y_std  = list(config.traj_norm_y_std)
        else:
            th_config = config

        # Method 2: diffusion_speed — speed as extra DDBM waypoint (num_poses+1 total)
        # Build a config copy based on th_config (respects predict_traj's num_poses)
        if config.diffusion_speed:
            config_sp = copy.copy(th_config)
            config_sp.num_poses = th_config.num_poses + 1  # 6+1=7 or 10+1=11
            config_sp.norm_x_mean = list(th_config.norm_x_mean) + [config.diffusion_speed_mean]
            config_sp.norm_x_std  = list(th_config.norm_x_std)  + [config.diffusion_speed_std]
            config_sp.norm_y_mean = list(th_config.norm_y_mean) + [0.0]
            config_sp.norm_y_std  = list(th_config.norm_y_std)  + [1.0]

            # Build speed-extended anchor from th_config anchor (6 or 10 poses)
            anchor_base = np.load(th_config.plan_anchor_path)   # (M, 6or10, 2)
            speed_col = np.full(
                (anchor_base.shape[0], 1, 2),
                fill_value=[config.diffusion_speed_anchor, 0.0],
                dtype=np.float32,
            )
            anchor_ext = np.concatenate([anchor_base, speed_col], axis=1)
            n_poses = th_config.num_poses
            anchor_ext_path = th_config.plan_anchor_path.replace('.npy', f'_speed{n_poses+1}.npy')
            np.save(anchor_ext_path, anchor_ext)
            config_sp.plan_anchor_path = anchor_ext_path

            self.trajectory_head = TrajectoryHead(config_sp)
        else:
            self.trajectory_head = TrajectoryHead(th_config)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: {
                'transfuser_bev_feature_upsample': (B, 64, 64, 64)
                'speed':                           (B, T) or (B,)
                'command_hist':                    (B, T, 6)
                'target_point_hist':               (B, T, 2)
                'target_point_next_hist':          (B, T, 2)
            }
            targets: { 'route': (B, ≥10, 2) } or None
        """
        bev = features['transfuser_bev_feature_upsample'].float()  # (B, 64, 64, 64)
        B = bev.shape[0]
        bev_spatial_shape: Tuple[int, int] = (bev.shape[2], bev.shape[3])

        # Current speed scalar: (B,)
        speed = features['speed'].float()
        if speed.dim() == 2:
            speed = speed[:, -1]

        # 1. Project BEV for GridSample attention: (B, d_model, 64, 64)
        cross_bev_feature = self.bev_proj(bev)

        # 2. Build context tokens via PlanningContextEncoder
        context_tokens = self.context_encoder(
            bev=bev,
            speed=features['speed'],
            command=features['command_hist'],
            tp=features['target_point_hist'],
            tp_next=features['target_point_next_hist'],
        )  # (B, H*W + 4, d_model)

        # 3. Queries via 6-layer TransformerDecoder
        queries = self.query_embedding.weight[None].expand(B, -1, -1)  # (B, 1or2, d_model)
        decoded = self.tf_decoder(queries, context_tokens)              # (B, 1or2, d_model)

        ego_query = decoded[:, :1]   # (B, 1, d_model)

        # 4. Method 1: target_speed from dedicated query token
        target_speed_dist = None
        target_speed_scalar = None
        if self.config.predict_target_speed:
            speed_token = decoded[:, 1]  # (B, d_model)
            target_speed_dist = self.target_speed_decoder(speed_token)  # (B, C)
            with torch.no_grad():
                target_speed_scalar = _decode_two_hot(
                    torch.softmax(target_speed_dist.float(), dim=-1),
                    self.speed_classes,
                )  # (B,)

        # 5. Agent queries
        agent_q = self.agent_queries.weight[None].expand(B, -1, -1)   # (B, 16, d_model)

        # 6. Method 2: append speed as 11th waypoint when diffusion_speed
        if self.config.diffusion_speed and targets is not None:
            speed_wp = torch.stack(
                [speed, torch.zeros_like(speed)], dim=-1
            ).unsqueeze(1)  # (B, 1, 2)
            n_route = self.trajectory_head.num_poses - 1  # poses before speed (10 or 5)
            route_base = targets['route'][:, :n_route]
            targets = {'route': torch.cat([route_base, speed_wp], dim=1)}  # (B, n+1, 2)

        # 7. DDBM trajectory head
        output = self.trajectory_head(
            ego_query=ego_query,
            agents_query=agent_q,
            bev_feature=cross_bev_feature,
            bev_spatial_shape=bev_spatial_shape,
            status_encoding=ego_query,
            targets=targets,
        )

        # 8. If diffusion_speed, extract speed from last waypoint at inference
        if self.config.diffusion_speed and targets is None:
            traj11 = output['trajectory']        # (B, n+1, 2)
            n_route = self.trajectory_head.num_poses - 1
            output['trajectory'] = traj11[:, :n_route]
            if target_speed_scalar is None:
                output['target_speed_scalar'] = traj11[:, n_route, 0]  # (B,) denormed speed

        if target_speed_dist is not None:
            output['target_speed_dist'] = target_speed_dist
        if target_speed_scalar is not None:
            output['target_speed_scalar'] = target_speed_scalar

        return output
