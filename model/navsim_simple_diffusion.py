"""
Minimal white-noise diffusion decoder for NavSim.
Stripped from `transformer_for_diffusion_multi_head.py`:
  - No semantic state branch
  - No branch conditioning
  - No energy guidance
  - No speed/route auxiliary heads

Keeps:
  - MultiSourceAttentionBlock (self-attn + BEV cross-attn + AdaLN)
  - GridSampleCrossBEVAttention (BEV spatial sampling)
  - HistoryEncoder (ego status)
  - SinusoidalPosEmb (timestep)
  - DDIM sampling
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Normalization & Position Encoding
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return (self.weight * x).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 256, theta: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dimension must be even, got {dim}")
        self.dim = dim
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)

    def forward(self, position_ids: torch.LongTensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.freqs_cos[position_ids]
        sin = self.freqs_sin[position_ids]
        if position_ids.dim() == 1:
            cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif position_ids.dim() == 2:
            cos = cos.unsqueeze(1)  # (B, 1, T, D/2)
            sin = sin.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported position_ids shape: {tuple(position_ids.shape)}")
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_float = q.float()
    k_float = k.float()
    cos = cos.float()
    sin = sin.float()
    q_even, q_odd = q_float[..., ::2], q_float[..., 1::2]
    k_even, k_odd = k_float[..., ::2], k_float[..., 1::2]
    q_rot = torch.stack((q_even * cos - q_odd * sin,
                         q_odd * cos + q_even * sin), dim=-1).flatten(-2)
    k_rot = torch.stack((k_even * cos - k_odd * sin,
                         k_odd * cos + k_even * sin), dim=-1).flatten(-2)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


# ============================================================
# Attention
# ============================================================

class MultiheadAttentionWithQKNorm(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, use_rope: bool = True):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.use_rope = use_rope
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, cos=None, sin=None, attn_mask=None):
        B, Tq, _ = query.shape
        _, Tk, _ = key.shape
        q = self.q_proj(query).view(B, Tq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Tk, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Tk, self.n_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_rope and cos is not None:
            q, k = apply_rotary_emb(q, k, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, Tq, -1)
        return self.out_proj(y)


# ============================================================
# Grid-Sample BEV Cross-Attention
# ============================================================

class GridSampleCrossBEVAttention(nn.Module):
    """Spatial cross-attention: sample BEV grid at trajectory waypoints."""

    def __init__(self, d_model: int, n_head: int, num_points: int = 8,
                 lidar_max_x: float = 32.0, lidar_max_y: float = 32.0, dropout: float = 0.1):
        super().__init__()
        self.num_points = num_points
        self.lidar_max_x = lidar_max_x
        self.lidar_max_y = lidar_max_y
        self.n_head = n_head

        self.attention_weights = nn.Linear(d_model, num_points)
        self.value_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
        )
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def _match_num_points(self, traj_points: torch.Tensor) -> torch.Tensor:
        n_points = traj_points.shape[-2]
        if n_points == self.num_points:
            return traj_points
        if n_points > self.num_points:
            idx = torch.linspace(0, n_points - 1, self.num_points, device=traj_points.device).long()
            return traj_points.index_select(-2, idx)
        pad = traj_points[..., -1:, :].expand(*traj_points.shape[:-2], self.num_points - n_points, 2)
        return torch.cat([traj_points, pad], dim=-2)

    def _normalize_coords(self, points_xy: torch.Tensor) -> torch.Tensor:
        """Convert ego-frame (x-forward, y-left) meters to grid_sample coordinates."""
        x_norm = points_xy[..., 0] / self.lidar_max_x
        y_norm = points_xy[..., 1] / self.lidar_max_y
        return torch.stack([y_norm, x_norm], dim=-1).clamp(-1.0, 1.0)

    def forward(self, query: torch.Tensor, traj_points: torch.Tensor, bev_grid: torch.Tensor):
        """
        query: (B, T, d_model) - trajectory tokens
        traj_points: (B, T, 2) or (B, T, P, 2) - physical trajectory points in ego frame
        bev_grid: (B, C, H, W) - BEV feature grid (64×64)
        """
        B, Tq, _ = query.shape
        if traj_points.dim() == 3:
            sample_points = self._match_num_points(traj_points)
            sample_points = sample_points.unsqueeze(1).expand(-1, Tq, -1, -1)
        elif traj_points.dim() == 4:
            sample_points = self._match_num_points(traj_points)
        else:
            raise ValueError(f"Unsupported traj_points shape: {tuple(traj_points.shape)}")

        attention_weights = self.attention_weights(query).softmax(dim=-1)  # (B, T, P)
        value = self.value_proj(bev_grid)  # (B, d_model, H, W)
        grid = self._normalize_coords(sample_points).to(dtype=value.dtype)
        sampled_features = F.grid_sample(
            value,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )  # (B, d_model, T, P)

        out = (attention_weights.unsqueeze(1) * sampled_features).sum(dim=-1)
        out = out.permute(0, 2, 1).contiguous()  # (B, T, d_model)
        return self.dropout(self.output_proj(out))


# ============================================================
# Multi-Source Attention Block
# ============================================================

class MultiSourceAttentionBlock(nn.Module):
    """Single decoder block: self-attn + BEV cross-attn + FFN + AdaLN."""

    def __init__(self, d_model: int, n_head: int = 8, d_ffn: int = 2048,
                 p_drop_attn: float = 0.1, p_drop_emb: float = 0.1, bev_num_points: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head

        # Self-attention
        self.self_attn = MultiheadAttentionWithQKNorm(d_model, n_head, p_drop_attn)
        self.self_attn_norm = RMSNorm(d_model)

        # BEV cross-attention
        self.bev_spatial_attn = GridSampleCrossBEVAttention(
            d_model, n_head, num_points=bev_num_points, dropout=p_drop_attn
        )
        self.bev_attn_norm = RMSNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=False),
            nn.GELU(),
            nn.Dropout(p_drop_emb),
            nn.Linear(d_ffn, d_model, bias=False),
            nn.Dropout(p_drop_emb),
        )
        self.ffn_norm = RMSNorm(d_model)

        # AdaLN modulation (conditioned on time + status embedding)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )

    def forward(self, x: torch.Tensor, bev_grid: torch.Tensor,
                traj_points: torch.Tensor,
                conditioning: torch.Tensor, cos=None, sin=None, attn_mask=None):
        """
        x: (B, T, d_model) - trajectory tokens
        bev_grid: (B, C, 64, 64)
        conditioning: (B, d_model) - timestep + status embedding
        """
        # AdaLN: compute shift/scale/gate for norm layers
        mod = self.adaLN_modulation(conditioning)  # (B, 6*d_model)
        shift_self, scale_self, gate_self, shift_bev, scale_bev, gate_ffn = mod.chunk(6, dim=-1)

        # Self-attention with AdaLN
        x_norm = self.self_attn_norm(x)
        x_mod = x_norm * (1 + scale_self.unsqueeze(1)) + shift_self.unsqueeze(1)
        attn_out = self.self_attn(x_mod, x_mod, x_mod, cos=cos, sin=sin, attn_mask=attn_mask)
        x = x + gate_self.unsqueeze(1) * attn_out

        # BEV cross-attention with AdaLN
        x_norm = self.bev_attn_norm(x)
        x_mod = x_norm * (1 + scale_bev.unsqueeze(1)) + shift_bev.unsqueeze(1)
        bev_out = self.bev_spatial_attn(x_mod, traj_points, bev_grid)
        x = x + bev_out

        # FFN with residual
        x_norm = self.ffn_norm(x)
        ffn_out = self.ffn(x_norm)
        x = x + gate_ffn.unsqueeze(1) * ffn_out

        return x


# ============================================================
# History Encoder
# ============================================================

class HistoryEncoder(nn.Module):
    """Encode ego status history (speed, theta, command, etc.)"""

    def __init__(self, input_dim: int = 14, d_model: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=2, batch_first=True, bidirectional=False)
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T_hist, input_dim) - eg: (B, 4, 14)
        returns: (B, d_model) - global history embedding
        """
        x = self.input_proj(x)
        x, _ = self.gru(x)
        x, _ = self.temporal_attn(x, x, x)
        return self.out_proj(x[:, -1])


# ============================================================
# Sinusoidal Timestep Embedding
# ============================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# ============================================================
# Main Diffusion Model
# ============================================================

class NavSimSimpleDiffusion(nn.Module):
    """
    Simple white-noise diffusion decoder for NavSim.
    Input: noisy trajectory x_t + BEV features + ego status + timestep
    Output: denoised trajectory pred_x0
    """

    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8,
        n_layer: int = 4,
        d_ffn: int = 2048,
        p_drop_attn: float = 0.1,
        p_drop_emb: float = 0.1,
        traj_horizon: int = 8,
        traj_dim: int = 2,
        ego_input_dim: int = 8,
        ego_history_frames: int = 4,
        prediction_type: str = "sample",  # "sample" = predict x0, "epsilon" = predict noise
        num_inference_steps: int = 10,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.traj_horizon = traj_horizon
        self.traj_dim = traj_dim
        self.prediction_type = prediction_type
        self.num_inference_steps = num_inference_steps
        self.num_train_timesteps = num_train_timesteps

        # Timestep embedding
        self.time_emb = SinusoidalPosEmb(d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Ego history encoder
        self.history_encoder = HistoryEncoder(input_dim=ego_input_dim, d_model=d_model)
        self.ego_proj = nn.Linear(ego_input_dim, d_model)

        # Noisy trajectory embedding
        self.traj_emb = nn.Sequential(
            nn.Linear(traj_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # BEV feature projection
        self.bev_proj = nn.Conv2d(64, d_model, kernel_size=1)  # upsample BEV → d_model

        # Learnable position embedding for trajectory tokens
        self.pos_emb = nn.Parameter(torch.randn(1, traj_horizon, d_model) * 0.02)

        # Transformer decoder layers
        self.blocks = nn.ModuleList([
            MultiSourceAttentionBlock(
                d_model, n_head, d_ffn, p_drop_attn, p_drop_emb, bev_num_points=traj_horizon
            )
            for _ in range(n_layer)
        ])

        # Rotary embedding over trajectory-token positions.
        self.rotary = RotaryEmbedding(d_model // n_head, max_seq_len=traj_horizon)

        # Output head
        self.traj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p_drop_emb),
            nn.Linear(d_model, traj_dim),
        )

        # Build DDIM scheduler
        self._build_scheduler(beta_schedule)

        # Normalization stats (will be set after data analysis)
        self.register_buffer("traj_mean", torch.zeros(traj_dim))
        self.register_buffer("traj_std", torch.ones(traj_dim))

    def _build_scheduler(self, beta_schedule: str):
        """Build DDIM noise schedule."""
        T = self.num_train_timesteps
        if beta_schedule == "cosine":
            s = 0.008
            steps = T + 1
            x = torch.linspace(0, T, steps)
            alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, max=0.999)
        else:
            betas = torch.linspace(0.0001, 0.02, T)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def normalize_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        return (traj - self.traj_mean.to(traj.device)) / self.traj_std.to(traj.device)

    def unnormalize_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        return traj * self.traj_std.to(traj.device) + self.traj_mean.to(traj.device)

    def _add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        """Forward diffusion: x_t = sqrt(alpha_cum) * x0 + sqrt(1-alpha_cum) * noise."""
        alpha_cum = self.alphas_cumprod.to(x0.device)[t]
        alpha_cum = alpha_cum.view(-1, 1, 1, 1)
        return torch.sqrt(alpha_cum) * x0 + torch.sqrt(1 - alpha_cum) * noise

    def _compute_conditioning(self, ego_status: torch.Tensor, timesteps: torch.Tensor):
        """
        ego_status: (B, T_hist, ego_input_dim) - 4 frames of ego status
        timesteps: (B,) - diffusion timesteps
        returns: (B, d_model) - conditioning vector
        """
        time_emb = self.time_proj(self.time_emb(timesteps))
        hist_emb = self.history_encoder(ego_status)
        ego_emb = self.ego_proj(ego_status[:, -1])  # last frame
        return time_emb + hist_emb + ego_emb

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor,
                bev_grid: torch.Tensor, ego_status: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None):
        """
        x_t: (B, 1, T, 2) - noisy trajectory (normalized)
        timesteps: (B,)
        bev_grid: (B, 64, 64, 64) - BEV upsample features
        ego_status: (B, T_hist, ego_input_dim)
        returns: pred_x0 (B, 1, T, 2) or pred_eps
        """
        B = x_t.shape[0]
        x_norm = x_t.squeeze(1)  # (B, T, 2)
        traj_points = self.unnormalize_trajectory(x_norm)

        # Embed trajectory tokens
        x = self.traj_emb(x_norm) + self.pos_emb[:, :x_norm.shape[1]]  # (B, T, d_model)

        # Conditioning
        cond = self._compute_conditioning(ego_status, timesteps)

        # Project BEV grid
        bev_proj = self.bev_proj(bev_grid)  # (B, d_model, 64, 64)

        position_ids = torch.arange(x.shape[1], device=x.device, dtype=torch.long)
        cos, sin = self.rotary(position_ids, dtype=x.dtype)

        # Decoder layers
        for block in self.blocks:
            x = block(
                x,
                bev_proj,
                traj_points=traj_points,
                conditioning=cond,
                cos=cos,
                sin=sin,
                attn_mask=attn_mask,
            )

        # Output head
        pred = self.traj_head(x)  # (B, T, 2)

        if self.prediction_type == "sample":
            return pred.unsqueeze(1)  # (B, 1, T, 2)
        else:
            return pred.unsqueeze(1)

    # ---- DDIM Sampling ----

    def _get_ddim_timesteps(self, num_steps: int):
        """Compute DDIM timesteps: uniform spacing from T-1 to 0."""
        T = self.num_train_timesteps
        step_ratio = T // num_steps
        timesteps = torch.arange(0, num_steps) * step_ratio
        timesteps = timesteps.flip(0)  # T-1, ..., 0
        return timesteps.to(dtype=torch.long)

    @torch.no_grad()
    def sample(self, bev_grid: torch.Tensor, ego_status: torch.Tensor,
               num_steps: Optional[int] = None, return_trajectory: bool = True):
        """
        DDIM sampling from pure noise N(0, I).
        bev_grid: (B, 64, 64, 64)
        ego_status: (B, T_hist, ego_input_dim)
        returns: trajectory (B, T, 2) in unnormalized space
        """
        num_steps = num_steps or self.num_inference_steps
        B = bev_grid.shape[0]
        device = bev_grid.device

        # Start from pure noise
        x_t = torch.randn(B, 1, self.traj_horizon, self.traj_dim, device=device)
        timesteps = self._get_ddim_timesteps(num_steps)

        for i, t_curr in enumerate(timesteps):
            t_batch = torch.full((B,), t_curr, device=device, dtype=torch.long)

            # Predict x0
            pred_out = self.forward(x_t, t_batch, bev_grid, ego_status)

            if self.prediction_type == "sample":
                pred_x0 = pred_out
                alpha_cum = self.alphas_cumprod.to(device)[t_curr].view(-1, 1, 1, 1)
                if i < len(timesteps) - 1:
                    t_prev = timesteps[i + 1]
                    alpha_cum_prev = self.alphas_cumprod.to(device)[t_prev].view(-1, 1, 1, 1)
                else:
                    alpha_cum_prev = torch.ones_like(alpha_cum)
            else:
                alpha_cum = self.alphas_cumprod.to(device)[t_curr].view(-1, 1, 1, 1)
                pred_x0 = (x_t - torch.sqrt(1 - alpha_cum) * pred_out) / torch.sqrt(alpha_cum)
                if i < len(timesteps) - 1:
                    t_prev = timesteps[i + 1]
                    alpha_cum_prev = self.alphas_cumprod.to(device)[t_prev].view(-1, 1, 1, 1)
                else:
                    alpha_cum_prev = torch.ones_like(alpha_cum)

            # DDIM update
            if i < len(timesteps) - 1:
                noise = torch.randn_like(x_t)
                x_t = (torch.sqrt(alpha_cum_prev) * pred_x0 +
                       torch.sqrt(1 - alpha_cum_prev) * noise)

        if return_trajectory:
            return self.unnormalize_trajectory(pred_x0.squeeze(1))
        return pred_x0.squeeze(1)

    # ---- Training Loss ----

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute diffusion training loss.
        batch:
          bev_grid: (B, 64, 64, 64)
          ego_status: (B, T_hist, ego_input_dim)
          trajectory: (B, T, 2) - ground truth trajectory in physical space
        """
        gt_traj = batch["trajectory"]  # (B, T, 2)
        gt_norm = self.normalize_trajectory(gt_traj)  # (B, T, 2)
        B = gt_traj.shape[0]
        device = gt_traj.device

        # Sample noise
        noise = torch.randn(B, 1, self.traj_horizon, self.traj_dim, device=device)
        x0_norm = gt_norm.unsqueeze(1)  # (B, 1, T, 2)

        # Sample random timesteps
        t = torch.randint(0, self.num_train_timesteps, (B,), device=device)

        # Add noise
        x_t = self._add_noise(x0_norm, noise, t)

        # Predict
        pred = self.forward(x_t, t, batch["bev_grid"], batch["ego_status"])

        if self.prediction_type == "sample":
            target = x0_norm
        else:
            target = noise

        loss = F.mse_loss(pred, target)

        with torch.no_grad():
            # Unnormalized L2 error for logging
            pred_unnorm = self.unnormalize_trajectory(pred.squeeze(1))
            l2_err = torch.norm(pred_unnorm - gt_traj, dim=-1).mean()

        return loss, {"l2_err_m": l2_err.item()}
