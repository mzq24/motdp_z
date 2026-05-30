"""Clean motion-only diffusion core for paper reproduction.

This module intentionally contains no semantic-state, graph, energy, alignment,
route-memory, or route-intent code. It keeps the minimum Route-B motion stack:
BEV context, ego/status/history conditioning, joint trajectory+route denoising,
and a scalar speed classification head.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def gen_sineembed_for_position(pos_tensor: torch.Tensor, hidden_dim: int = 64) -> torch.Tensor:
    """Sinusoidal embedding for 2D waypoint coordinates."""
    if hidden_dim % 2 != 0:
        raise ValueError(f"hidden_dim must be even, got {hidden_dim}")
    orig_dtype = pos_tensor.dtype
    pos = pos_tensor.float()
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos[..., 0] * scale
    y_embed = pos[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1).to(dtype=orig_dtype)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -emb_scale)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class HistoryEncoder(nn.Module):
    """Small GRU encoder for ego-status history."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.summary_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, ego_status: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(ego_status)
        seq, hidden = self.gru(x)
        global_summary = hidden[-1]
        query = self.summary_query.expand(x.shape[0], -1, -1)
        attn_out, _ = self.temporal_attn(query, seq, seq, need_weights=False)
        attn_summary = self.norm(attn_out.squeeze(1))
        return self.fuse(torch.cat([global_summary, attn_summary], dim=-1))


class BEVContextEncoder(nn.Module):
    """Projects TransFuser BEV caches into a compact token set."""

    def __init__(self, bev_dim: int, upsample_dim: int, d_model: int):
        super().__init__()
        self.lowres_proj = nn.Sequential(nn.Linear(bev_dim, d_model), nn.LayerNorm(d_model))
        self.upsample_proj = nn.Sequential(
            nn.Conv2d(upsample_dim, d_model, kernel_size=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, 64, d_model))
        self.fuse = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(
        self,
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
    ) -> torch.Tensor:
        low = transfuser_bev_feature.flatten(2).permute(0, 2, 1)
        low = self.lowres_proj(low)
        up = self.upsample_proj(transfuser_bev_feature_upsample)
        up = F.adaptive_avg_pool2d(up, output_size=(8, 8)).flatten(2).permute(0, 2, 1)
        tokens = self.fuse(low + up)
        return tokens + self.pos_emb[:, :tokens.shape[1], :]


class MotionDecoderBlock(nn.Module):
    """AdaLN self-attention + BEV cross-attention decoder block."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 9 * d_model))
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, bev_tokens: torch.Tensor, cond: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        shift_s, scale_s, gate_s, shift_c, scale_c, gate_c, shift_f, scale_f, gate_f = self.ada(cond).chunk(9, dim=-1)
        x_self = self._modulate(self.norm_self(x), shift_s, scale_s)
        self_out, _ = self.self_attn(x_self, x_self, x_self, attn_mask=attn_mask, need_weights=False)
        x = x + gate_s.unsqueeze(1) * self.dropout(self_out)

        x_cross = self._modulate(self.norm_cross(x), shift_c, scale_c)
        cross_out, _ = self.cross_attn(x_cross, bev_tokens, bev_tokens, need_weights=False)
        x = x + gate_c.unsqueeze(1) * self.dropout(cross_out)

        x_ffn = self._modulate(self.norm_ffn(x), shift_f, scale_f)
        x = x + gate_f.unsqueeze(1) * self.ffn(x_ffn)
        return x


class TrajectoryHead(nn.Module):
    """Trajectory head with route-to-trajectory guidance."""

    def __init__(self, d_model: int, output_dim: int, nhead: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.route_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.route_norm = nn.LayerNorm(d_model)
        self.route_gate = nn.Parameter(torch.zeros(1))
        self.cond_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor, route_tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        route_ctx, _ = self.route_attn(x, route_tokens, route_tokens, need_weights=False)
        x = x + torch.sigmoid(self.route_gate) * self.route_norm(route_ctx)
        x = x + self.cond_proj(cond).unsqueeze(1)
        x = x + self.mlp(x)
        return self.out(x)


class RouteHead(nn.Module):
    """Route head with route-specific ego-status conditioning."""

    def __init__(self, d_model: int, status_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.status_proj = nn.Sequential(nn.Linear(status_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.cond_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.out = nn.Linear(d_model, output_dim)
        nn.init.zeros_(self.final_adaln[-1].weight)
        nn.init.zeros_(self.final_adaln[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, current_status: torch.Tensor) -> torch.Tensor:
        route_cond = self.status_proj(current_status) + self.cond_proj(cond)
        x = self.norm(x) + route_cond.unsqueeze(1)
        x = x + self.mlp(x)
        shift, scale = self.final_adaln(route_cond).chunk(2, dim=-1)
        x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.out(x)


class PaperMotionDiffusionCore(nn.Module):
    """Readable motion-only joint trajectory/route diffusion model."""

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
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        transfuser_bev_dim: int = 1512,
        transfuser_bev_upsample_dim: int = 64,
        traj_can_attend_route: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_waypoints = num_waypoints
        self.n_obs_steps = n_obs_steps
        self.status_dim = status_dim
        self.n_emb = n_emb
        self.joint_horizon = horizon + num_waypoints
        self.traj_can_attend_route = bool(traj_can_attend_route)
        self.anchor_pos_hidden_dim = 64

        self.bev_encoder = BEVContextEncoder(transfuser_bev_dim, transfuser_bev_upsample_dim, n_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.status_proj = nn.Linear(status_dim, n_emb)
        self.history_encoder = HistoryEncoder(status_dim, n_emb)

        self.wp_emb = nn.Sequential(nn.Linear(self.anchor_pos_hidden_dim, n_emb), nn.SiLU(), nn.Linear(n_emb, n_emb))
        self.route_wp_emb = nn.Sequential(nn.Linear(self.anchor_pos_hidden_dim, n_emb), nn.SiLU(), nn.Linear(n_emb, n_emb))
        self.diff_query = nn.Parameter(torch.randn(1, 1, n_emb) * 0.02)
        self.route_diff_query = nn.Parameter(torch.randn(1, 1, n_emb) * 0.02)
        self.speed_query = nn.Parameter(torch.randn(1, 1, n_emb) * 0.02)
        self.traj_pos_emb = nn.Parameter(torch.randn(1, horizon, n_emb) * 0.02)
        self.route_pos_emb = nn.Parameter(torch.randn(1, num_waypoints, n_emb) * 0.02)
        self.traj_segment_emb = nn.Parameter(torch.randn(1, 1, n_emb) * 0.02)
        self.route_segment_emb = nn.Parameter(torch.randn(1, 1, n_emb) * 0.02)
        self.speed_segment_emb = nn.Parameter(torch.randn(1, 1, n_emb) * 0.02)
        self.drop = nn.Dropout(p_drop_emb)
        self.pre_norm = nn.LayerNorm(n_emb)

        self.layers = nn.ModuleList([
            MotionDecoderBlock(n_emb, n_head, 4 * n_emb, p_drop_attn)
            for _ in range(n_layer)
        ])
        self.final_norm = nn.LayerNorm(n_emb)
        self.route_residual = nn.Sequential(nn.LayerNorm(n_emb), nn.Linear(n_emb, n_emb), nn.GELU(), nn.Linear(n_emb, n_emb))
        self.route_residual_gate = nn.Parameter(torch.zeros(1))

        self.traj_head = TrajectoryHead(n_emb, output_dim, n_head, p_drop_emb)
        self.route_head = RouteHead(n_emb, status_dim, output_dim, p_drop_emb)
        self.speed_head = nn.Sequential(
            nn.Linear(2 * n_emb, n_emb // 2),
            nn.ReLU(inplace=True),
            nn.Linear(n_emb // 2, len(self.speed_classes)),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

    def _conditioning(
        self,
        timestep: Union[torch.Tensor, int, float],
        ego_status: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = ego_status.shape[0]
        device = ego_status.device
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep[None].to(device=device)
        timestep = timestep.to(device=device, dtype=torch.long).expand(B)
        current_status = ego_status[:, -1, :]
        return (
            self.time_emb(timestep).to(dtype=ego_status.dtype)
            + self.status_proj(current_status)
            + self.history_encoder(ego_status),
            current_status,
        )

    def _embed_waypoints(self, points: torch.Tensor, is_route: bool = False) -> torch.Tensor:
        emb = gen_sineembed_for_position(points, hidden_dim=self.anchor_pos_hidden_dim)
        return self.route_wp_emb(emb) if is_route else self.wp_emb(emb)

    def _ego_speed_mask(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        T_speed = 1
        T_traj = self.horizon
        T_route = self.num_waypoints
        total = T_speed + T_traj + T_route
        mask = torch.zeros(total, total, device=device, dtype=dtype)
        mask[T_speed:, :T_speed] = float('-inf')
        route_start = T_speed + T_traj
        if not self.traj_can_attend_route:
            mask[T_speed:route_start, route_start:] = float('-inf')
        mask[route_start:, T_speed:route_start] = float('-inf')
        return mask

    def forward_denoise(
        self,
        noisy_joint_abs_or_norm: torch.Tensor,
        timestep: Union[torch.Tensor, int, float],
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        ego_status: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict clean normalized trajectory, route, and speed from noisy joint waypoints."""
        model_dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        joint = noisy_joint_abs_or_norm.to(device=device, dtype=model_dtype)
        if joint.dim() == 4:
            if joint.shape[1] != 1:
                raise ValueError(f"PaperMotionDiffusionCore expects one sample, got {joint.shape}")
            joint = joint[:, 0]
        if joint.dim() != 3 or joint.shape[1] != self.joint_horizon:
            raise ValueError(f"Expected (B,{self.joint_horizon},2), got {tuple(joint.shape)}")
        ego_status = ego_status.to(device=device, dtype=model_dtype)
        bev = transfuser_bev_feature.to(device=device, dtype=model_dtype)
        bev_up = transfuser_bev_feature_upsample.to(device=device, dtype=model_dtype)

        traj_points = joint[:, :self.horizon]
        route_points = joint[:, self.horizon:]
        B = joint.shape[0]
        cond, current_status = self._conditioning(timestep, ego_status)
        bev_tokens = self.bev_encoder(bev, bev_up)

        traj_tokens = self._embed_waypoints(traj_points) + self.diff_query + cond.unsqueeze(1)
        route_tokens = self._embed_waypoints(route_points, is_route=True) + self.route_diff_query + cond.unsqueeze(1)
        speed_token = self.speed_query.expand(B, -1, -1) + cond.unsqueeze(1)
        traj_tokens = traj_tokens + self.traj_pos_emb + self.traj_segment_emb
        route_tokens = route_tokens + self.route_pos_emb + self.route_segment_emb
        speed_token = speed_token + self.speed_segment_emb
        x = torch.cat([speed_token, traj_tokens, route_tokens], dim=1)
        x = self.pre_norm(self.drop(x))
        attn_mask = self._ego_speed_mask(x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, bev_tokens, cond, attn_mask)
        x = self.final_norm(x)

        speed_out = x[:, :1]
        traj_out = x[:, 1:1 + self.horizon]
        route_out = x[:, 1 + self.horizon:]
        route_seed = route_tokens
        route_out = route_out + torch.sigmoid(self.route_residual_gate) * self.route_residual(route_seed)

        traj_norm = self.traj_head(traj_out, route_out, cond)
        route_norm = self.route_head(route_out, cond, current_status)
        speed_logits = self.speed_head(torch.cat([speed_out.squeeze(1), cond], dim=-1))
        return {
            'traj_norm': traj_norm,
            'route_norm': route_norm,
            'speed_logits': speed_logits,
            'traj_tokens': traj_out,
            'route_tokens': route_out,
        }
