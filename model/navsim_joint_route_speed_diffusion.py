"""Joint NAVSIM white-noise diffusion decoder for traj / route / speed.

This model is intentionally NAVSIM-only and has zero state tokens. It keeps the
same ego-status conditioning style as ``NavSimSimpleDiffusion`` while adding
route and speed token groups. BEV conditioning follows the old next-token path:
compact/raw BEV features provide global memory tokens, and top-down BEV grid
features provide spatial sampling around noisy trajectory points.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.navsim_simple_diffusion import (
    HistoryEncoder,
    MultiheadAttentionWithQKNorm,
    MultiSourceAttentionBlock,
    RMSNorm,
    RotaryEmbedding,
    SinusoidalPosEmb,
)


class NavSimJointRouteSpeedDiffusion(nn.Module):
    """White-noise diffusion over [traj8, route50, speed8] tokens."""

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
        route_points: int = 50,
        speed_horizon: int = 8,
        ego_input_dim: int = 8,
        ego_history_frames: int = 4,
        prediction_type: str = "sample",
        num_inference_steps: int = 10,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        route_loss_weight: float = 1.0,
        speed_loss_weight: float = 0.5,
        use_raw_bev_feature: bool = True,
        raw_bev_dim: int = 512,
    ) -> None:
        super().__init__()
        if traj_dim != 2:
            raise ValueError("NAVSIM joint v1 expects traj_dim=2")
        self.d_model = int(d_model)
        self.n_head = int(n_head)
        self.n_layer = int(n_layer)
        self.traj_horizon = int(traj_horizon)
        self.traj_dim = int(traj_dim)
        self.route_points = int(route_points)
        self.speed_horizon = int(speed_horizon)
        self.ego_history_frames = int(ego_history_frames)
        self.prediction_type = str(prediction_type)
        self.num_inference_steps = int(num_inference_steps)
        self.num_train_timesteps = int(num_train_timesteps)
        self.route_loss_weight = float(route_loss_weight)
        self.speed_loss_weight = float(speed_loss_weight)
        self.use_raw_bev_feature = bool(use_raw_bev_feature)
        self.raw_bev_dim = int(raw_bev_dim)

        self.traj_slice = slice(0, self.traj_horizon)
        self.route_slice = slice(self.traj_horizon, self.traj_horizon + self.route_points)
        self.speed_slice = slice(self.traj_horizon + self.route_points, self.total_tokens)

        self.time_emb = SinusoidalPosEmb(d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.history_encoder = HistoryEncoder(input_dim=ego_input_dim, d_model=d_model)
        self.ego_proj = nn.Linear(ego_input_dim, d_model)

        self.traj_emb = nn.Sequential(nn.Linear(2, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, d_model))
        self.route_emb = nn.Sequential(nn.Linear(2, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, d_model))
        self.speed_emb = nn.Sequential(nn.Linear(1, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, d_model))
        self.type_emb = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, self.total_tokens, d_model) * 0.02)

        self.bev_proj = nn.Conv2d(64, d_model, kernel_size=1)
        if self.use_raw_bev_feature:
            self.raw_bev_proj = nn.Sequential(
                nn.Conv2d(self.raw_bev_dim, d_model, kernel_size=1),
                nn.GELU(),
            )
            self.raw_bev_token_norm = RMSNorm(d_model)
            self.raw_bev_pos_emb = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
            self.raw_bev_attn = nn.ModuleList(
                [MultiheadAttentionWithQKNorm(d_model, n_head, p_drop_attn) for _ in range(n_layer)]
            )
            self.raw_bev_attn_norm = nn.ModuleList(
                [RMSNorm(d_model) for _ in range(n_layer)]
            )
        self.blocks = nn.ModuleList(
            [
                MultiSourceAttentionBlock(
                    d_model,
                    n_head,
                    d_ffn,
                    p_drop_attn,
                    p_drop_emb,
                    bev_num_points=self.traj_horizon,
                )
                for _ in range(n_layer)
            ]
        )
        self.rotary = RotaryEmbedding(d_model // n_head, max_seq_len=self.total_tokens)

        self.traj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(p_drop_emb), nn.Linear(d_model, 2))
        self.route_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(p_drop_emb), nn.Linear(d_model, 2))
        self.speed_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(p_drop_emb), nn.Linear(d_model, 1))

        self._build_scheduler(beta_schedule)

        self.register_buffer("traj_mean", torch.zeros(self.traj_horizon, 2))
        self.register_buffer("traj_std", torch.ones(self.traj_horizon, 2))
        self.register_buffer("route_mean", torch.zeros(self.route_points, 2))
        self.register_buffer("route_std", torch.ones(self.route_points, 2))
        self.register_buffer("speed_mean", torch.zeros(self.speed_horizon))
        self.register_buffer("speed_std", torch.ones(self.speed_horizon))

    @property
    def total_tokens(self) -> int:
        return self.traj_horizon + self.route_points + self.speed_horizon

    def _build_scheduler(self, beta_schedule: str) -> None:
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

    def set_normalization_stats(
        self,
        traj_mean,
        traj_std,
        route_mean,
        route_std,
        speed_mean,
        speed_std,
    ) -> None:
        with torch.no_grad():
            self.traj_mean.copy_(torch.as_tensor(traj_mean, dtype=self.traj_mean.dtype, device=self.traj_mean.device))
            self.traj_std.copy_(torch.as_tensor(traj_std, dtype=self.traj_std.dtype, device=self.traj_std.device).clamp_min(1e-6))
            self.route_mean.copy_(torch.as_tensor(route_mean, dtype=self.route_mean.dtype, device=self.route_mean.device))
            self.route_std.copy_(torch.as_tensor(route_std, dtype=self.route_std.dtype, device=self.route_std.device).clamp_min(1e-6))
            self.speed_mean.copy_(torch.as_tensor(speed_mean, dtype=self.speed_mean.dtype, device=self.speed_mean.device))
            self.speed_std.copy_(torch.as_tensor(speed_std, dtype=self.speed_std.dtype, device=self.speed_std.device).clamp_min(1e-6))

    def normalize_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        return (traj - self.traj_mean.to(traj.device)) / self.traj_std.to(traj.device)

    def unnormalize_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        return traj * self.traj_std.to(traj.device) + self.traj_mean.to(traj.device)

    def normalize_route(self, route: torch.Tensor) -> torch.Tensor:
        return (route - self.route_mean.to(route.device)) / self.route_std.to(route.device)

    def unnormalize_route(self, route: torch.Tensor) -> torch.Tensor:
        return route * self.route_std.to(route.device) + self.route_mean.to(route.device)

    def normalize_speed(self, speed: torch.Tensor) -> torch.Tensor:
        return (speed - self.speed_mean.to(speed.device)) / self.speed_std.to(speed.device)

    def unnormalize_speed(self, speed: torch.Tensor) -> torch.Tensor:
        return speed * self.speed_std.to(speed.device) + self.speed_mean.to(speed.device)

    def _add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_cum = self.alphas_cumprod.to(x0.device)[t]
        view_shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
        alpha_cum = alpha_cum.view(view_shape)
        return torch.sqrt(alpha_cum) * x0 + torch.sqrt(1.0 - alpha_cum) * noise

    def _compute_conditioning(self, ego_status: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_proj(self.time_emb(timesteps))
        hist_emb = self.history_encoder(ego_status)
        ego_emb = self.ego_proj(ego_status[:, -1])
        return time_emb + hist_emb + ego_emb

    def _embed_tokens(self, traj_x: torch.Tensor, route_x: torch.Tensor, speed_x: torch.Tensor) -> torch.Tensor:
        traj_tokens = self.traj_emb(traj_x) + self.type_emb[:, 0:1]
        route_tokens = self.route_emb(route_x) + self.type_emb[:, 1:2]
        speed_tokens = self.speed_emb(speed_x) + self.type_emb[:, 2:3]
        x = torch.cat([traj_tokens, route_tokens, speed_tokens], dim=1)
        return x + self.pos_emb[:, : x.shape[1]]

    def _raw_bev_position_embedding(self, height: int, width: int, device, dtype) -> torch.Tensor:
        pos = self.raw_bev_pos_emb
        if height * width == pos.shape[1]:
            return pos.to(device=device, dtype=dtype)
        side = int(round(math.sqrt(pos.shape[1])))
        if side * side != pos.shape[1]:
            return torch.zeros(1, height * width, self.d_model, device=device, dtype=dtype)
        pos_grid = pos.transpose(1, 2).reshape(1, self.d_model, side, side)
        pos_grid = F.interpolate(pos_grid, size=(height, width), mode="bilinear", align_corners=False)
        return pos_grid.flatten(2).transpose(1, 2).to(device=device, dtype=dtype)

    def _compute_raw_bev_tokens(self, bev_feature: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.use_raw_bev_feature:
            return None
        if bev_feature is None:
            raise KeyError(
                "NavSimJointRouteSpeedDiffusion was configured with "
                "use_raw_bev_feature=True but batch has no bev_feature"
            )
        if bev_feature.ndim != 4:
            raise ValueError(f"Expected raw bev_feature shape (B,C,H,W), got {tuple(bev_feature.shape)}")
        if bev_feature.shape[1] != self.raw_bev_dim:
            raise ValueError(f"Expected raw bev_feature channels={self.raw_bev_dim}, got {bev_feature.shape[1]}")
        weight = self.raw_bev_proj[0].weight
        raw = self.raw_bev_proj(bev_feature.to(dtype=weight.dtype))
        height, width = raw.shape[-2:]
        tokens = raw.flatten(2).transpose(1, 2)
        tokens = self.raw_bev_token_norm(tokens)
        return tokens + self._raw_bev_position_embedding(height, width, tokens.device, tokens.dtype)

    def forward(
        self,
        traj_x_t: torch.Tensor,
        route_x_t: torch.Tensor,
        speed_x_t: torch.Tensor,
        timesteps: torch.Tensor,
        bev_grid: torch.Tensor,
        ego_status: torch.Tensor,
        bev_feature: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self._embed_tokens(traj_x_t, route_x_t, speed_x_t)
        cond = self._compute_conditioning(ego_status, timesteps)
        bev_proj = self.bev_proj(bev_grid.to(dtype=self.bev_proj.weight.dtype))
        raw_bev_tokens = self._compute_raw_bev_tokens(bev_feature)
        traj_points = self.unnormalize_trajectory(traj_x_t)

        position_ids = torch.arange(x.shape[1], device=x.device, dtype=torch.long)
        cos, sin = self.rotary(position_ids, dtype=x.dtype)
        for block_idx, block in enumerate(self.blocks):
            if raw_bev_tokens is not None:
                raw_query = self.raw_bev_attn_norm[block_idx](x)
                x = x + self.raw_bev_attn[block_idx](raw_query, raw_bev_tokens, raw_bev_tokens)
            x = block(
                x,
                bev_proj,
                traj_points=traj_points,
                conditioning=cond,
                cos=cos,
                sin=sin,
                attn_mask=attn_mask,
            )

        traj_h = x[:, self.traj_slice]
        route_h = x[:, self.route_slice]
        speed_h = x[:, self.speed_slice]
        return {
            "trajectory": self.traj_head(traj_h),
            "route": self.route_head(route_h),
            "speed_profile": self.speed_head(speed_h).squeeze(-1),
        }

    def _get_ddim_timesteps(self, num_steps: int) -> torch.Tensor:
        """Compute DDIM timesteps: uniform spacing from T-1 to 0."""
        step_ratio = max(1, self.num_train_timesteps // int(num_steps))
        timesteps = torch.arange(0, int(num_steps), device=self.alphas_cumprod.device) * step_ratio
        return timesteps.clamp(max=self.num_train_timesteps - 1).flip(0).to(dtype=torch.long)

    def _predict_x0_from_epsilon(self, x_t: torch.Tensor, eps: torch.Tensor, alpha_cum: torch.Tensor) -> torch.Tensor:
        view_shape = (x_t.shape[0],) + (1,) * (x_t.ndim - 1)
        alpha_cum = alpha_cum.view(view_shape)
        return (x_t - torch.sqrt(1.0 - alpha_cum) * eps) / torch.sqrt(alpha_cum)

    def _ddim_update(self, pred_x0: torch.Tensor, alpha_cum_prev: torch.Tensor, stochastic: bool) -> torch.Tensor:
        view_shape = (pred_x0.shape[0],) + (1,) * (pred_x0.ndim - 1)
        alpha_cum_prev = alpha_cum_prev.view(view_shape)
        if stochastic:
            noise = torch.randn_like(pred_x0)
            return torch.sqrt(alpha_cum_prev) * pred_x0 + torch.sqrt(1.0 - alpha_cum_prev) * noise
        return torch.sqrt(alpha_cum_prev) * pred_x0

    @torch.no_grad()
    def sample(
        self,
        bev_grid: torch.Tensor,
        ego_status: torch.Tensor,
        bev_feature: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        return_trajectory: bool = True,
        stochastic: bool = True,
    ):
        """Sample joint tokens and return the trajectory by default."""
        num_steps = int(num_steps or self.num_inference_steps)
        B = bev_grid.shape[0]
        device = bev_grid.device
        traj_x_t = torch.randn(B, self.traj_horizon, 2, device=device)
        route_x_t = torch.randn(B, self.route_points, 2, device=device)
        speed_x_t = torch.randn(B, self.speed_horizon, 1, device=device)
        timesteps = self._get_ddim_timesteps(num_steps).to(device)

        pred = None
        for i, t_curr in enumerate(timesteps):
            t_batch = torch.full((B,), int(t_curr.item()), device=device, dtype=torch.long)
            pred = self.forward(
                traj_x_t,
                route_x_t,
                speed_x_t,
                t_batch,
                bev_grid,
                ego_status,
                bev_feature=bev_feature,
            )
            alpha_cum = self.alphas_cumprod.to(device)[t_curr]
            if self.prediction_type == "sample":
                pred_traj_x0 = pred["trajectory"]
                pred_route_x0 = pred["route"]
                pred_speed_x0 = pred["speed_profile"].unsqueeze(-1)
            else:
                pred_traj_x0 = self._predict_x0_from_epsilon(traj_x_t, pred["trajectory"], alpha_cum)
                pred_route_x0 = self._predict_x0_from_epsilon(route_x_t, pred["route"], alpha_cum)
                pred_speed_x0 = self._predict_x0_from_epsilon(speed_x_t, pred["speed_profile"].unsqueeze(-1), alpha_cum)

            if i < len(timesteps) - 1:
                alpha_cum_prev = self.alphas_cumprod.to(device)[timesteps[i + 1]]
                traj_x_t = self._ddim_update(pred_traj_x0, alpha_cum_prev, stochastic=stochastic)
                route_x_t = self._ddim_update(pred_route_x0, alpha_cum_prev, stochastic=stochastic)
                speed_x_t = self._ddim_update(pred_speed_x0, alpha_cum_prev, stochastic=stochastic)

        if pred is None:
            raise RuntimeError("sampling produced no prediction")
        traj = self.unnormalize_trajectory(pred_traj_x0)
        route = self.unnormalize_route(pred_route_x0)
        speed = self.unnormalize_speed(pred_speed_x0.squeeze(-1))
        if return_trajectory:
            return traj
        return {"trajectory": traj, "route": route, "speed_profile": speed}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        gt_traj = batch["trajectory"]
        gt_route = batch["route"]
        route_mask = batch["route_mask"].float()
        gt_speed = batch["speed_profile"]

        traj_norm = self.normalize_trajectory(gt_traj)
        route_norm = self.normalize_route(gt_route)
        speed_norm = self.normalize_speed(gt_speed)
        speed_norm_3d = speed_norm.unsqueeze(-1)
        B = gt_traj.shape[0]
        device = gt_traj.device

        traj_noise = torch.randn(B, self.traj_horizon, 2, device=device)
        route_noise = torch.randn(B, self.route_points, 2, device=device)
        speed_noise = torch.randn(B, self.speed_horizon, 1, device=device)
        t = torch.randint(0, self.num_train_timesteps, (B,), device=device)

        traj_x_t = self._add_noise(traj_norm, traj_noise, t)
        route_x_t = self._add_noise(route_norm, route_noise, t)
        speed_x_t = self._add_noise(speed_norm_3d, speed_noise, t)

        pred = self.forward(
            traj_x_t,
            route_x_t,
            speed_x_t,
            t,
            batch["bev_grid"],
            batch["ego_status"],
            bev_feature=batch.get("bev_feature"),
        )
        if self.prediction_type == "sample":
            target_traj = traj_norm
            target_route = route_norm
            target_speed = speed_norm
        else:
            target_traj = traj_noise
            target_route = route_noise
            target_speed = speed_noise.squeeze(-1)

        traj_loss = F.mse_loss(pred["trajectory"], target_traj)
        route_loss_raw = F.mse_loss(pred["route"], target_route, reduction="none").mean(dim=-1)
        route_denom = route_mask.sum().clamp_min(1.0)
        route_loss = (route_loss_raw * route_mask).sum() / route_denom
        speed_loss = F.mse_loss(pred["speed_profile"], target_speed)
        loss = traj_loss + self.route_loss_weight * route_loss + self.speed_loss_weight * speed_loss

        with torch.no_grad():
            if self.prediction_type == "sample":
                pred_traj_unnorm = self.unnormalize_trajectory(pred["trajectory"])
                pred_route_unnorm = self.unnormalize_route(pred["route"])
                pred_speed_unnorm = self.unnormalize_speed(pred["speed_profile"])
            else:
                pred_traj_unnorm = self.unnormalize_trajectory(pred["trajectory"])
                pred_route_unnorm = self.unnormalize_route(pred["route"])
                pred_speed_unnorm = self.unnormalize_speed(pred["speed_profile"])
            l2_err = torch.norm(pred_traj_unnorm - gt_traj, dim=-1).mean()
            route_l2_raw = torch.norm(pred_route_unnorm - gt_route, dim=-1)
            route_l2 = (route_l2_raw * route_mask).sum() / route_denom
            speed_mae = torch.abs(pred_speed_unnorm - gt_speed).mean()

        return loss, {
            "l2_err_m": float(l2_err.item()),
            "traj_loss": float(traj_loss.item()),
            "route_loss": float(route_loss.item()),
            "speed_loss": float(speed_loss.item()),
            "route_l2_m": float(route_l2.item()),
            "speed_mae_mps": float(speed_mae.item()),
        }
