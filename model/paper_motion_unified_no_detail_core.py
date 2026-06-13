"""Unified-decoder motion-only core without semantic state or detail residuals.

This is the second clean paper ablation step: keep the old Route-B unified
motion decoder skeleton (UnifiedDecoderOnlyTransformer / MultiSourceAttentionBlock
/ base grid BEV point sampling), but expose the same forward_denoise API as the
simple paper motion core. Semantic, graph, branch-condition, lidar, and detail
residual paths stay disabled. Route intent is optional for the N3 ablation.
"""

from __future__ import annotations

from typing import Dict, Union

import torch
import torch.nn as nn

from model.transformer_for_diffusion_multi_head import TransformerForDiffusion


class PaperMotionUnifiedNoDetailCore(nn.Module):
    """Thin wrapper around the legacy Route-B motion-only unified decoder."""

    speed_classes = [0.0, 4.0, 8.0, 10.0, 13.89, 16.0, 17.78, 20.0]

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        horizon: int = 6,
        num_waypoints: int = 20,
        n_obs_steps: int = 4,
        status_dim: int = 14,
        n_layer: int = 4,
        n_head: int = 8,
        n_emb: int = 512,
        n_cond_layers: int = 4,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        transfuser_bev_dim: int = 1512,
        transfuser_bev_upsample_dim: int = 64,
        traj_can_attend_route: bool = True,
        ego_detail_activation_t: int = -1,
        use_route_intent_token: bool = False,
        route_intent_gate_init: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.num_waypoints = num_waypoints
        self.joint_horizon = horizon + num_waypoints
        self.model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=256,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=True,
            obs_as_cond=True,
            n_cond_layers=n_cond_layers,
            status_dim=status_dim,
            ego_status_seq_len=n_obs_steps,
            transfuser_bev_dim=transfuser_bev_dim,
            transfuser_bev_upsample_dim=transfuser_bev_upsample_dim,
            num_waypoints=num_waypoints,
            num_modes=1,
            traj_can_attend_route=traj_can_attend_route,
            anchor_free=True,
            energy_heads=False,
            # Negative threshold keeps the timestep-gated detail residual inactive.
            # Base grid point attention remains active through the legacy ego path.
            ego_detail_activation_t=ego_detail_activation_t,
            use_lidar_bev_detail=False,
            lidar_bev_history_frames=1,
            use_condition_group_dropout=False,
            motion_only_model=True,
            use_chase_front_following_state=False,
            semantic_motion_condition_mode='full',
            semantic_motion_condition_profile='all',
            use_cover_relation_graph_decoder=False,
            cover_graph_use_traj_context=False,
            cover_graph_use_speed_context=False,
            use_route_prev_coarse_memory=False,
            use_route_intent_token=use_route_intent_token,
            route_intent_gate_init=route_intent_gate_init,
        )
        self.speed_classes = list(self.model.speed_classes)

    def forward_denoise(
        self,
        noisy_joint_abs_or_norm: torch.Tensor,
        timestep: Union[torch.Tensor, int, float],
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        ego_status: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict clean normalized trajectory, route, and speed logits."""
        model_dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        joint_abs = noisy_joint_abs_or_norm.to(device=device, dtype=model_dtype)
        if joint_abs.dim() == 4:
            if joint_abs.shape[1] != 1:
                raise ValueError(f"Expected one sample, got {tuple(joint_abs.shape)}")
            joint_abs = joint_abs[:, 0]
        if joint_abs.dim() != 3 or joint_abs.shape[1] != self.joint_horizon:
            raise ValueError(f"Expected (B,{self.joint_horizon},2), got {tuple(joint_abs.shape)}")

        result = self.model.forward_ego(
            x_t=joint_abs,
            x_t_abs=joint_abs,
            timestep=timestep,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
            transfuser_lidar_bev=None,
            branch_condition=None,
            prev_route_coarse_memory=None,
            return_intermediates=True,
        )
        return {
            'traj_norm': result['poses_reg'][:, 0],
            'route_norm': result['route_pred'],
            'speed_logits': result['speed_pred'],
            'traj_tokens': result['traj_out'],
            'route_tokens': result['route_out'],
            'speed_tokens': result['speed_out'],
        }
