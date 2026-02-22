"""
Decoder-Only Transformer for Diffusion-based Trajectory Prediction.

This module implements a decoder-only architecture for trajectory prediction,
using sequential cross-attention to multiple condition sources:
- Transfuser features (2 types from transfuser backbone, following DiffusionDriveV2)
- State tokens (low-dim ego status)  
- Reasoning tokens (vision-language features)

Key design choices:
- No encoder: removes attention pooling that loses information
- DiffusionDriveV2-style cross-attention for BEV features (GridSampleCrossBEVAttention)
- Standard cross-attention for reasoning tokens
- Unified decoder with heterogeneous queries for trajectory (6) + route (20)
- MLP output heads (no GRU)

Transfuser Features (following DiffusionDriveV2):
- bev_feature: (B, 1512, 8, 8) - Original BEV from lidar, downscaled for cross-attention
- bev_feature_upsample: (B, 64, 64, 64) - FPN output p3 (used for spatial cross-attention via GridSampleCrossBEVAttention)

Note: Unlike the original 4-feature version, we now only use bev_feature and bev_feature_upsample,
following DiffusionDriveV2's approach where fused_features and image_feature_grid are NOT used.
"""

from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

logger = logging.getLogger(__name__)


# =============================================================================
# Basic Components
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) module."""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)

        B, seq_len = position_ids.shape
        cos = self.cos_cached[:, :, position_ids[0], :].expand(B, -1, -1, -1)
        sin = self.sin_cached[:, :, position_ids[0], :].expand(B, -1, -1, -1)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MultiheadAttentionWithQKNorm(nn.Module):
    """MultiheadAttention with Query-Key Normalization and optional RoPE."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        norm_type: str = "rmsnorm",
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, 
            bias=bias, batch_first=batch_first
        )
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        
        if norm_type == "rmsnorm":
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_position_embeddings, rope_theta)
        
        self.batch_first = batch_first
    
    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)
    
    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, seq_len_q, seq_len_k = q.shape[0], q.shape[2], k.shape[2]
        if seq_len_q != seq_len_k:
            return q, k
        
        position_ids = torch.arange(seq_len_q, device=q.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rope(q, position_ids)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = True,
                attn_mask: Optional[torch.Tensor] = None, average_attn_weights: bool = True):
        if self.batch_first:
            B, seq_len_q, embed_dim = query.shape
            _, seq_len_k, _ = key.shape
            
            q = query.view(B, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
            k = key.view(B, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
            
            q, k = self._apply_qk_norm(q, k)
            if self.use_rope:
                q, k = self._apply_rope(q, k)
            
            query = q.transpose(1, 2).reshape(B, seq_len_q, embed_dim)
            key = k.transpose(1, 2).reshape(B, seq_len_k, embed_dim)
        
        return self.attn(query=query, key=key, value=value, key_padding_mask=key_padding_mask,
                        need_weights=need_weights, attn_mask=attn_mask, average_attn_weights=average_attn_weights)


def apply_rope_single(x, cos, sin):
    """Apply RoPE to a single tensor (only K, not Q)."""
    return (x * cos) + (rotate_half(x) * sin)


# =============================================================================
# DiffusionDriveV2-style Grid Sample Cross Attention for BEV Features
# =============================================================================

class GridSampleCrossBEVAttention(nn.Module):
    """
    Grid Sample based Cross Attention for BEV features (DiffusionDriveV2 style).
    
    This module samples BEV features at trajectory point locations and uses 
    attention-weighted aggregation. It's more efficient than standard cross-attention
    for spatial BEV features.
    
    Unlike standard cross-attention which attends to all spatial locations,
    this module:
    1. Samples BEV features at trajectory point locations using grid_sample
    2. Computes attention weights based on query features
    3. Aggregates sampled features using attention weights
    
    Args:
        embed_dims: Embedding dimension of query features
        num_heads: Number of attention heads
        in_bev_dims: Input BEV feature channels (e.g., 64 for bev_feature_upsample)
        num_points: Number of points to sample per query (trajectory length)
        lidar_max_x: Maximum lidar range in x direction (for normalization)
        lidar_max_y: Maximum lidar range in y direction (for normalization)
    """
    def __init__(
        self, 
        embed_dims: int, 
        num_heads: int = 8, 
        in_bev_dims: int = 64, 
        num_points: int = 8,
        lidar_max_x: float = 32.0,  # meters
        lidar_max_y: float = 32.0   # meters
    ):
        super(GridSampleCrossBEVAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.lidar_max_x = lidar_max_x
        self.lidar_max_y = lidar_max_y
        
        # Attention weights projection: query -> num_points attention weights
        self.attention_weights = nn.Linear(embed_dims, num_points)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Value projection from BEV features to embed_dims
        # Conv2d to project BEV features while preserving spatial structure
        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)
    
    def forward(
        self, 
        queries: torch.Tensor,        # (B, num_queries, embed_dims)
        traj_points: torch.Tensor,    # (B, num_queries, num_points, 2) or (B, num_points, 2)
        bev_feature: torch.Tensor,    # (B, in_bev_dims, H, W)
    ) -> torch.Tensor:
        """
        Args:
            queries: Input query features (B, num_queries, embed_dims)
            traj_points: Trajectory points for sampling (B, num_queries, num_points, 2) or (B, num_points, 2)
                         Each point is (x, y) in meters, will be normalized to [-1, 1]
            bev_feature: BEV features (B, in_bev_dims, H, W)
            
        Returns:
            Output features (B, num_queries, embed_dims)
        """
        bs, num_queries, _ = queries.shape
        
        # Handle trajectory points shape
        if traj_points.dim() == 3:
            # (B, num_points, 2) -> expand to (B, num_queries, num_points, 2)
            num_points = traj_points.shape[1]
            traj_points = traj_points.unsqueeze(1).expand(-1, num_queries, -1, -1)
        else:
            num_points = traj_points.shape[2]
        
        # Normalize trajectory points to [-1, 1] range for grid_sample
        # Assuming traj_points are in meters relative to ego vehicle
        normalized_trajectory = traj_points.clone()
        normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.lidar_max_y  # y -> x in grid
        normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.lidar_max_x  # x -> y in grid
        
        # Swap x and y for grid_sample convention (grid_sample expects (x, y) where x is width)
        normalized_trajectory = normalized_trajectory[..., [1, 0]]
        
        # Clamp to valid range
        normalized_trajectory = torch.clamp(normalized_trajectory, -1.0, 1.0)
        
        # Compute attention weights from queries
        attention_weights = self.attention_weights(queries)  # (B, num_queries, num_points)
        attention_weights = attention_weights.softmax(dim=-1)  # Softmax over points
        
        # Project BEV features
        value = self.value_proj(bev_feature)  # (B, embed_dims, H, W)
        
        # Grid for sampling: (B, num_queries, num_points, 2)
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)
        
        # Sample features at trajectory points
        # Reshape grid for grid_sample: (B, num_queries * num_points, 1, 2) -> requires (B, H_out, W_out, 2)
        # We treat num_queries as H_out and num_points as W_out
        grid_for_sample = grid.view(bs, num_queries, num_points, 2)
        
        sampled_features = F.grid_sample(
            value,
            grid_for_sample,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # (B, embed_dims, num_queries, num_points)
        
        # Apply attention weights
        attention_weights = attention_weights.unsqueeze(1)  # (B, 1, num_queries, num_points)
        out = (attention_weights * sampled_features).sum(dim=-1)  # (B, embed_dims, num_queries)
        out = out.permute(0, 2, 1).contiguous()  # (B, num_queries, embed_dims)
        
        # Output projection
        out = self.output_proj(out)
        
        # Residual connection with dropout
        return self.dropout(out) + queries


class MultiSourceAttentionBlock(nn.Module):
    """
    Multi-Source Attention Block for Transfuser Features (DiffusionDriveV2 style).
    
    Adapted for 2 transfuser feature sources following DiffusionDriveV2:
    - bev_feature: (B, 64, d_model) - flattened BEV feature (downscaled)
    - reasoning_tokens: (B, T_r, d_model) - reasoning tokens
    
    Note: fused_feature and image_feature are removed following DiffusionDriveV2's approach.
    
    Architecture:
        1. Self-attention on main sequence with RoPE on both Q and K
        2. Cross-attention to BEV feature with RoPE on K only
        3. Cross-attention to reasoning tokens with RoPE on K only + gating
        4. Residual + FFN
        
    Each source has:
    - Separate K, V projections
    - Source-specific Q adapters
    - Learnable temperature and bias
    - Residual path for guaranteed information flow
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, rope_theta: float = 10000.0, max_seq_len: int = 512,
                 norm_type: str = "rmsnorm"):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        # RoPE embedding
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        
        # ========== QK Normalization ==========
        if norm_type == "rmsnorm":
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        
        # ========== Q projections (BEV only, no reasoning) ==========
        self.q_proj = nn.Linear(d_model, d_model)  # Shared base Q
        # Small adaptation layers for BEV cross-attention queries (low-rank for efficiency)
        self.q_adapter_bev = nn.Linear(d_model, d_model // 4)
        self.q_adapter_out = nn.Linear(d_model // 4, d_model)

        # ========== Self-attention K, V ==========
        self.k_self = nn.Linear(d_model, d_model)
        self.v_self = nn.Linear(d_model, d_model)

        # ========== BEV cross-attention K, V ==========
        self.k_bev = nn.Linear(d_model, d_model)
        self.v_bev = nn.Linear(d_model, d_model)

        # ========== Output projection ==========
        self.o_proj = nn.Linear(d_model, d_model)

        # ========== Per-source learnable temperature (log scale for stability) ==========
        self.temp_self = nn.Parameter(torch.zeros(1))
        self.temp_bev = nn.Parameter(torch.zeros(1))

        # ========== Per-source learnable bias (attention prior) ==========
        self.bias_self = nn.Parameter(torch.zeros(1))
        self.bias_bev = nn.Parameter(torch.zeros(1))

        # ========== BEV Residual Path ==========
        self.bev_residual_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.bev_residual_gate = nn.Parameter(torch.zeros(1))

        # ========== Route-Specific Components (Stability Enhancement) ==========
        # Route-specific Q adapters (BEV only)
        self.route_q_adapter_bev = nn.Linear(d_model, d_model // 4)
        self.route_q_adapter_out = nn.Linear(d_model // 4, d_model)

        # Route-specific attention temperature and bias
        self.route_temp_bev = nn.Parameter(torch.zeros(1))
        self.route_bias_bev = nn.Parameter(torch.zeros(1))
        
        # Route-specific AdaLN modulation
        self.route_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )
        
        # ========== FFN ==========
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
        # ========== Layer Norm for pre-attention ==========
        self.norm_pre = nn.LayerNorm(d_model)
        
        # ========== AdaLN modulation - 6 params: shift, scale, gate ==========
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply QK normalization to query and key tensors."""
        return self.q_norm(q), self.k_norm(k)
    
    def _get_rope_embed(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Generate RoPE cos/sin embeddings."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)
    
    def _reshape_heads(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """Reshape (B, L, d_model) -> (B, nhead, L, head_dim)"""
        return x.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
    
    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
    def forward(
        self,
        x: torch.Tensor,  # (B, T, d_model) - main sequence [trajectory | route]
        bev_tokens: torch.Tensor,     # (B, T_bev, d_model) - BEV tokens (already projected)
        conditioning: torch.Tensor,  # (B, d_model) - conditioning for AdaLN
        self_attn_mask: Optional[torch.Tensor] = None,
        bev_padding_mask: Optional[torch.Tensor] = None,
        route_conditioning: Optional[torch.Tensor] = None,
        T_traj: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with self-attention + BEV cross-attention.

        Cross-attention sources:
        1. bev_tokens: flattened BEV feature (64 tokens)
        """
        B, T, C = x.shape
        T_bev = bev_tokens.shape[1]

        # Determine if we have route-specific processing
        if T_traj is None:
            T_traj = T
        T_route = T - T_traj

        # ========== AdaLN modulation parameters ==========
        mod_params = self.adaLN_modulation(conditioning)
        shift_pre, scale_pre, gate_attn, shift_ffn, scale_ffn, gate_ffn = mod_params.chunk(6, dim=1)

        # Route-specific AdaLN
        if T_route > 0 and route_conditioning is not None:
            route_mod_params = self.route_adaLN_modulation(route_conditioning)
            route_shift_pre, route_scale_pre, route_gate_attn, route_shift_ffn, route_scale_ffn, route_gate_ffn = route_mod_params.chunk(6, dim=1)
        else:
            route_shift_pre, route_scale_pre = shift_pre, scale_pre
            route_gate_attn, route_shift_ffn, route_scale_ffn, route_gate_ffn = gate_attn, shift_ffn, scale_ffn, gate_ffn

        # ========== Pre-LayerNorm with modulation ==========
        x_norm_ln = self.norm_pre(x)

        x_norm_traj = self.modulate(x_norm_ln[:, :T_traj, :], shift_pre, scale_pre)
        if T_route > 0:
            x_norm_route = self.modulate(x_norm_ln[:, T_traj:, :], route_shift_pre, route_scale_pre)
            x_norm = torch.cat([x_norm_traj, x_norm_route], dim=1)
        else:
            x_norm = x_norm_traj

        # ========== Temperature scaling factors ==========
        temp_self = 1.0 + torch.nn.functional.softplus(self.temp_self)
        temp_bev = 1.0 + torch.nn.functional.softplus(self.temp_bev)

        # Route-specific temperatures
        route_temp_bev = 1.0 + torch.nn.functional.softplus(self.route_temp_bev)

        # ========== Q projection ==========
        q_base = self.q_proj(x_norm)

        # Trajectory Q adaptation for BEV
        q_adapt_bev_traj = self.q_adapter_out(torch.tanh(self.q_adapter_bev(x_norm[:, :T_traj, :])))

        # Route Q adaptation for BEV
        if T_route > 0:
            q_adapt_bev_route = self.route_q_adapter_out(torch.tanh(self.route_q_adapter_bev(x_norm[:, T_traj:, :])))
            q_adapt_bev = torch.cat([q_adapt_bev_traj, q_adapt_bev_route], dim=1)
        else:
            q_adapt_bev = q_adapt_bev_traj

        # ========== Self-attention K, V ==========
        k_self = self.k_self(x_norm)
        v_self = self.v_self(x_norm)

        # ========== BEV cross-attention K, V ==========
        k_bev = self.k_bev(bev_tokens)
        v_bev = self.v_bev(bev_tokens)

        # ========== Reshape to multi-head ==========
        q_base = self._reshape_heads(q_base, B, T)
        q_bev = self._reshape_heads(q_base.transpose(1, 2).reshape(B, T, C) + q_adapt_bev, B, T)

        k_self = self._reshape_heads(k_self, B, T)
        v_self = self._reshape_heads(v_self, B, T)
        k_bev = self._reshape_heads(k_bev, B, T_bev)
        v_bev = self._reshape_heads(v_bev, B, T_bev)

        # ========== Apply QK Normalization ==========
        q_base, k_self = self._apply_qk_norm(q_base, k_self)
        q_bev, k_bev = self._apply_qk_norm(q_bev, k_bev)

        # ========== Apply RoPE ==========
        cos_main, sin_main = self._get_rope_embed(T, x.device, x.dtype)
        cos_main = cos_main.unsqueeze(0).unsqueeze(0)
        sin_main = sin_main.unsqueeze(0).unsqueeze(0)
        q_base = apply_rope_single(q_base, cos_main, sin_main)
        k_self = apply_rope_single(k_self, cos_main, sin_main)
        q_bev = apply_rope_single(q_bev, cos_main, sin_main)

        # K gets RoPE for cross-attention
        cos_bev, sin_bev = self._get_rope_embed(T_bev, x.device, x.dtype)
        cos_bev = cos_bev.unsqueeze(0).unsqueeze(0)
        sin_bev = sin_bev.unsqueeze(0).unsqueeze(0)
        k_bev = apply_rope_single(k_bev, cos_bev, sin_bev)

        # ========== Compute attention scores ==========
        scale = math.sqrt(self.head_dim)

        attn_self = torch.matmul(q_base, k_self.transpose(-2, -1)) * temp_self + self.bias_self
        attn_bev_raw = torch.matmul(q_bev, k_bev.transpose(-2, -1))

        if T_route > 0:
            attn_bev_traj = attn_bev_raw[:, :, :T_traj, :] * temp_bev + self.bias_bev
            attn_bev_route = attn_bev_raw[:, :, T_traj:, :] * route_temp_bev + self.route_bias_bev
            attn_bev = torch.cat([attn_bev_traj, attn_bev_route], dim=2)
        else:
            attn_bev = attn_bev_raw * temp_bev + self.bias_bev

        # Concatenate attention scores: [self | bev]
        attn_scores = torch.cat([attn_self, attn_bev], dim=-1)
        attn_scores = attn_scores / scale

        # Apply self-attention mask if provided
        if self_attn_mask is not None:
            total_kv_len = T + T_bev
            full_mask = torch.zeros(T, total_kv_len, device=x.device, dtype=x.dtype)
            full_mask[:, :T] = self_attn_mask
            attn_scores = attn_scores + full_mask.unsqueeze(0).unsqueeze(0)

        # Apply BEV padding mask if provided
        if bev_padding_mask is not None:
            mask = bev_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores[:, :, :, T:T+T_bev] = attn_scores[:, :, :, T:T+T_bev].masked_fill(mask, float('-inf'))

        # Softmax and weighted sum
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        v_combined = torch.cat([v_self, v_bev], dim=2)
        output = torch.matmul(attn_weights, v_combined)

        # Reshape and output projection
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # ========== BEV Residual Path ==========
        bev_gate = torch.sigmoid(self.bev_residual_gate)
        bev_pooled = bev_tokens.mean(dim=1, keepdim=True)
        bev_residual = self.bev_residual_proj(bev_pooled).expand(-1, T, -1)

        output = output + bev_gate * bev_residual
        
        # ========== Residual with segment-specific gate ==========
        if T_route > 0:
            x_traj = x[:, :T_traj, :] + gate_attn.unsqueeze(1) * output[:, :T_traj, :]
            x_route = x[:, T_traj:, :] + route_gate_attn.unsqueeze(1) * output[:, T_traj:, :]
            x = torch.cat([x_traj, x_route], dim=1)
        else:
            x = x + gate_attn.unsqueeze(1) * output
        
        # ========== FFN with segment-specific AdaLN ==========
        x_ln = nn.functional.layer_norm(x, [C])
        
        if T_route > 0:
            x_norm_ffn_traj = self.modulate(x_ln[:, :T_traj, :], shift_ffn, scale_ffn)
            x_norm_ffn_route = self.modulate(x_ln[:, T_traj:, :], route_shift_ffn, route_scale_ffn)
            x_norm_ffn = torch.cat([x_norm_ffn_traj, x_norm_ffn_route], dim=1)
        else:
            x_norm_ffn = self.modulate(x_ln, shift_ffn, scale_ffn)
        
        x_ffn = self.ffn(x_norm_ffn)
        
        if T_route > 0:
            x_traj = x[:, :T_traj, :] + gate_ffn.unsqueeze(1) * x_ffn[:, :T_traj, :]
            x_route = x[:, T_traj:, :] + route_gate_ffn.unsqueeze(1) * x_ffn[:, T_traj:, :]
            x = torch.cat([x_traj, x_route], dim=1)
        else:
            x = x + gate_ffn.unsqueeze(1) * x_ffn
        
        return x


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ModuleAttrMixin(nn.Module):
    """Mixin for device/dtype properties."""
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# =============================================================================
# History Encoder with Attention
# =============================================================================

class HistoryEncoder(nn.Module):
    """
    Encodes the history sequence of ego status with GRU + Temporal Attention.
    
    Improvements over simple GRU:
    1. GRU captures sequential dependencies
    2. Temporal attention allows focusing on important history frames
    3. Combines global (GRU hidden) and selective (attention) information
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GRU for sequential encoding
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # Temporal attention: last frame queries all history
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Learnable query for global summary (alternative to using last frame)
        self.summary_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Combine GRU hidden state and attention output
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) - history sequence of ego status
        Returns:
            (B, hidden_dim) - encoded history representation
        """
        B, T, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (B, T, hidden_dim)
        
        # GRU encoding
        gru_out, h_n = self.gru(x)  # gru_out: (B, T, hidden_dim), h_n: (1, B, hidden_dim)
        gru_hidden = h_n[-1]  # (B, hidden_dim) - global sequential summary
        
        # Temporal attention: summary query attends to all GRU outputs
        query = self.summary_query.expand(B, -1, -1)  # (B, 1, hidden_dim)
        attn_out, _ = self.temporal_attn(
            query=query,
            key=gru_out,
            value=gru_out
        )  # (B, 1, hidden_dim)
        attn_out = self.attn_norm(attn_out).squeeze(1)  # (B, hidden_dim)
        
        # Fuse GRU hidden and attention output
        combined = torch.cat([gru_hidden, attn_out], dim=-1)  # (B, hidden_dim * 2)
        output = self.fusion(combined)  # (B, hidden_dim)
        
        return output


# =============================================================================
# Trajectory Head with Route Guidance
# =============================================================================

class TrajectoryMLPHead(nn.Module):
    """
    MLP-based Trajectory Head with Route-to-Trajectory Guidance.
    
    Key features:
    1. Cross-attention from trajectory to route features (Route Guidance)
    2. Conditioning injection from timestep + ego_status
    3. MLP processing for final output
    
    This allows trajectory prediction to explicitly attend to and be guided by
    the planned route, improving trajectory-route consistency.
    """
    def __init__(self, n_emb: int, output_dim: int, p_drop: float = 0.1, num_heads: int = 8):
        super().__init__()
        self.n_emb = n_emb
        self.ln_f = nn.LayerNorm(n_emb)
        
        # Route Guidance: trajectory attends to route
        self.route_guidance_attn = nn.MultiheadAttention(
            embed_dim=n_emb,
            num_heads=num_heads,
            dropout=p_drop,
            batch_first=True
        )
        self.route_guidance_norm = nn.LayerNorm(n_emb)
        self.route_guidance_gate = nn.Parameter(torch.zeros(1))  # Learnable gate
        
        # MLP layers with conditioning
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Dropout(p_drop),
        )
        
        # Conditioning projection
        self.cond_proj = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.SiLU(),
        )
        
        # Output projection
        self.output_head = nn.Linear(n_emb, output_dim)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor, 
                route_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T_traj, n_emb) - trajectory decoder output
            conditioning: (B, n_emb) - timestep + ego_status conditioning
            route_features: (B, T_route, n_emb) - route decoder output for guidance
        Returns:
            (B, T_traj, output_dim) - trajectory prediction
        """
        x = self.ln_f(x)
        
        # Route Guidance: trajectory attends to route features
        if route_features is not None:
            gate = torch.sigmoid(self.route_guidance_gate)
            route_guided, _ = self.route_guidance_attn(
                query=x,
                key=route_features,
                value=route_features
            )
            route_guided = self.route_guidance_norm(route_guided)
            x = x + gate * route_guided  # Gated residual connection
        
        # Add conditioning as bias
        cond = self.cond_proj(conditioning).unsqueeze(1)  # (B, 1, n_emb)
        x = x + cond
        
        # MLP processing
        x = x + self.mlp(x)
        
        # Output projection
        return self.output_head(x)


# =============================================================================
# Route Head with Conditioning
# =============================================================================

class RouteMLPHead(nn.Module):
    """
    MLP-based Route Head with independent conditioning support.
    
    Key design (inspired by AdaLNRouteHeadDecoderOnly for stability):
    - Route-specific status projection (separate from shared conditioning)
    - Final AdaLN modulation before output (like original stable version)
    - Combined conditioning from both shared and route-specific sources
    
    Unlike trajectory, route does not need to attend to trajectory (unidirectional).
    But it benefits from route-specific conditioning for closed-loop stability.
    """
    def __init__(self, n_emb: int, status_dim: int = 14, output_dim: int = 2, p_drop: float = 0.1):
        super().__init__()
        self.ln_f = nn.LayerNorm(n_emb)
        
        # ========== Route-Specific Status Projection (Key for Stability) ==========
        # This mirrors AdaLNRouteHeadDecoderOnly's independent status_proj
        # Provides route-specific conditioning separate from shared trajectory conditioning
        self.route_status_proj = nn.Sequential(
            nn.Linear(status_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )
        
        # Shared conditioning projection (for timestep + history)
        self.cond_proj = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.SiLU(),
        )
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Dropout(p_drop),
        )
        
        # ========== Final AdaLN Modulation (Key for Stability) ==========
        # This mirrors AdaLNRouteHeadDecoderOnly's final_adaLN
        # Provides fine-grained control over route output based on current ego state
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_emb, 2 * n_emb, bias=True),
        )
        
        # Output projection
        self.output_head = nn.Linear(n_emb, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize final AdaLN to identity (shift=0, scale=0 -> x * 1 + 0)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        conditioning: torch.Tensor,
        ego_status: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T_route, n_emb) - route decoder output
            conditioning: (B, n_emb) - timestep + history conditioning
            ego_status: (B, status_dim) - current ego status for route-specific conditioning
        Returns:
            (B, T_route, output_dim) - route prediction (waypoints)
        """
        x = self.ln_f(x)
        
        # Combine route-specific and shared conditioning
        route_cond = self.route_status_proj(ego_status) + self.cond_proj(conditioning)
        
        # Add conditioning to features
        x = x + route_cond.unsqueeze(1)  # (B, T_route, n_emb)
        
        # MLP processing
        x = x + self.mlp(x)
        
        # ========== Final AdaLN Modulation (Key for Stability) ==========
        # Apply route-specific modulation before output
        shift, scale = self.final_adaLN(route_cond).chunk(2, dim=1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # Output projection
        return self.output_head(x)


# =============================================================================
# Unified Decoder with Heterogeneous Queries
# =============================================================================

class UnifiedDecoderOnlyTransformer(nn.Module):
    """
    Unified Decoder-Only Transformer with heterogeneous queries (DiffusionDriveV2 style).
    
    Combines trajectory prediction (horizon points) and route prediction (20 waypoints)
    into a single decoder, using:
    - Unified sinusoidal position encoding for both trajectory and route
    - Segment embeddings to distinguish query types (trajectory vs route)
    - Multi-source attention with RoPE and gating
    - Separate output heads for trajectory and route
    
    Query structure: [trajectory (horizon) | route (num_waypoints)]
    
    Position Encoding Design:
    - Both trajectory and route share the same sinusoidal position encoding scheme
    - Trajectory positions: 0, 1, 2, ... (horizon-1) representing future time steps
    - Route positions: 0, 1, 2, ... (num_waypoints-1) representing spatial waypoints
    - Segment embeddings differentiate the two modalities
    
    Cross-attention sources (following DiffusionDriveV2):
        - Self-attention with RoPE on Q and K
        - BEV cross-attention with RoPE on K only
        - Reasoning cross-attention with RoPE on K only + gating
    
    Transfuser Feature Processing (following DiffusionDriveV2):
        - bev_feature: (B, 1512, 8, 8) -> flatten to (B, 64, 1512) -> project to (B, 64, d_model)
        - bev_feature_upsample: (B, 64, 64, 64) -> used for GridSampleCrossBEVAttention (spatial attention)
        
    Note: fused_features and image_feature_grid are NOT used, following DiffusionDriveV2's approach.
    """
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        # Transfuser feature dimensions (following DiffusionDriveV2)
        transfuser_bev_dim: int = 1512,       # bev_feature channel dim
        transfuser_bev_upsample_dim: int = 64, # bev_feature_upsample channel dim
        horizon: int = 8,
        num_waypoints: int = 20,
        max_seq_len: int = 64,  # Max length for unified position encoding
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.horizon = horizon
        self.num_waypoints = num_waypoints
        
        # ========== Transfuser Feature Projections (following DiffusionDriveV2) ==========
        # bev_feature: (B, 1512, 8, 8) -> (B, 64, d_model)
        self.bev_feature_proj = nn.Sequential(
            nn.Linear(transfuser_bev_dim, d_model), 
            nn.LayerNorm(d_model)
        )
        
        # ========== GridSampleCrossBEVAttention for spatial BEV features (DiffusionDriveV2 style) ==========
        # Uses bev_feature_upsample: (B, 64, 64, 64) for spatial sampling
        self.bev_spatial_attn = GridSampleCrossBEVAttention(
            embed_dims=d_model,
            num_heads=nhead,
            in_bev_dims=transfuser_bev_upsample_dim,
            num_points=horizon,  # Sample at trajectory points
            lidar_max_x=32.0,
            lidar_max_y=32.0
        )
        
        # Position embeddings for cross-attention sources (BEV tokens only now)
        # Total tokens: 64 (bev)
        self.combined_pos_emb = nn.Parameter(torch.zeros(1, 64, d_model))
        
        # Route queries (learnable)
        self.route_queries = nn.Parameter(torch.randn(1, num_waypoints, d_model))
        
        # ========== Unified Position Encoding ==========
        # Sinusoidal position encoding shared by trajectory and route
        # This provides a consistent spatial/temporal representation
        self.register_buffer(
            "unified_pos_encoding",
            self._create_sinusoidal_pos_encoding(max_seq_len, d_model)
        )
        
        # Learnable position scaling for each modality
        # Allows the model to learn different position importance
        self.traj_pos_scale = nn.Parameter(torch.ones(1, 1, d_model))
        self.route_pos_scale = nn.Parameter(torch.ones(1, 1, d_model))
        
        # Segment embeddings to distinguish query types (trajectory vs route)
        self.traj_segment_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.route_segment_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Decoder blocks - using new MultiSourceAttentionBlock
        self.layers = nn.ModuleList([
            MultiSourceAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # ========== Route Residual Path (Stability Enhancement) ==========
        # This bypasses the shared decoder to preserve route-specific information
        # Key insight: Original AdaLNRouteHeadDecoderOnly had independent decoder,
        # this residual path simulates that independence within unified architecture
        self.route_residual_path = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Learnable gate, initialized to 0 (residual starts inactive, model learns to use it)
        self.route_residual_gate = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _create_sinusoidal_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Create sinusoidal position encoding.
        
        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
            
        Returns:
            pos_encoding: (1, max_len, d_model) position encoding tensor
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def _init_weights(self):
        nn.init.normal_(self.combined_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.route_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.traj_segment_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.route_segment_emb, mean=0.0, std=0.02)
        nn.init.ones_(self.traj_pos_scale)
        nn.init.ones_(self.route_pos_scale)
        # Initialize route residual gate to 0 (starts inactive)
        nn.init.zeros_(self.route_residual_gate)
        for layer in self.layers:
            nn.init.zeros_(layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(layer.adaLN_modulation[-1].bias)
            # Initialize route-specific AdaLN to identity
            nn.init.zeros_(layer.route_adaLN_modulation[-1].weight)
            nn.init.zeros_(layer.route_adaLN_modulation[-1].bias)
    
    def _create_block_diagonal_mask(self, T_traj: int, T_route: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Create a unidirectional block self-attention mask.
        
        This ensures:
        - Trajectory queries can attend to: trajectory + route (full visibility)
        - Route queries can only attend to: route (isolated from trajectory)
        
        Rationale:
        - Trajectory prediction benefits from knowing the planned route
        - Route planning should be independent of specific trajectory details
        
        Attention pattern (0 = allowed, -inf = blocked):
        
                    | Trajectory | Route |
        ------------------------------------
        Trajectory  |     0      |   0   |  <- can see both
        Route       |   -inf     |   0   |  <- can only see route
        
        Args:
            T_traj: Number of trajectory queries
            T_route: Number of route queries
            device: Device for the mask tensor
            dtype: Data type for the mask tensor
            
        Returns:
            mask: (T_total, T_total) mask where -inf blocks attention
        """
        T_total = T_traj + T_route
        # Start with all allowed
        mask = torch.zeros((T_total, T_total), device=device, dtype=dtype)
        
        # Block route-to-trajectory attention (lower-left block)
        # Route queries (rows T_traj:) cannot attend to trajectory keys (cols :T_traj)
        mask[T_traj:, :T_traj] = float('-inf')
        
        return mask
    
    def forward(
        self,
        traj_emb: torch.Tensor,  # (B, T_traj, d_model) - trajectory query embeddings
        transfuser_bev_feature: torch.Tensor,       # (B, 1512, 8, 8)
        transfuser_bev_feature_upsample: torch.Tensor,  # (B, 64, 64, 64)
        conditioning: torch.Tensor,                 # (B, d_model)
        traj_points: Optional[torch.Tensor] = None,  # (B, T_traj, 2) or (B, T_traj, horizon, 2) for GridSampleCrossBEVAttention
        route_conditioning: Optional[torch.Tensor] = None,  # (B, d_model) - route-specific conditioning
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with unified queries and multi-source attention (DiffusionDriveV2 style).
        
        Uses unidirectional self-attention mask:
        - Trajectory queries can see: trajectory + route (route guides trajectory)
        - Route queries can only see: route (independent planning)
        - Both can attend to all cross-attention sources (BEV features, reasoning)
        
        Transfuser Feature Processing (following DiffusionDriveV2):
        - bev_feature: flatten to (B, 64, 1512) -> project to (B, 64, d_model)
        - bev_feature_upsample: used for GridSampleCrossBEVAttention at trajectory points
        
        Note: fused_features and image_feature_grid are NOT used, following DiffusionDriveV2.
        
        Args:
            traj_emb: (B, horizon, d_model) - trajectory query embeddings (noisy traj embedded)
            transfuser_bev_feature: (B, 1512, 8, 8) - BEV feature from transfuser
            transfuser_bev_feature_upsample: (B, 64, 64, 64) - Upsampled BEV for spatial attention
            reasoning_tokens: (B, T_r, reasoning_dim) - reasoning tokens
            conditioning: (B, d_model) - timestep + current_status conditioning
            traj_points: (B, horizon, 2) - trajectory points for spatial BEV attention
            route_conditioning: (B, d_model) - route-specific conditioning (optional)
            
        Returns:
            traj_out: (B, horizon, d_model) - trajectory output
            route_out: (B, num_waypoints, d_model) - route output
        """
        B = traj_emb.shape[0]
        T_traj = traj_emb.shape[1]
        T_route = self.num_waypoints
        
        # ========== Unified Position Encoding ==========
        # Get sinusoidal position encoding for trajectory
        traj_pos = self.unified_pos_encoding[:, :T_traj, :] * self.traj_pos_scale
        # Get sinusoidal position encoding for route  
        route_pos = self.unified_pos_encoding[:, :T_route, :] * self.route_pos_scale
        
        # Add position + segment embeddings to trajectory
        traj_emb = traj_emb + traj_pos + self.traj_segment_emb
        
        # Route queries with position and segment embeddings
        route_emb = self.route_queries.expand(B, -1, -1) + route_pos + self.route_segment_emb
        
        # Concatenate queries: [trajectory | route]
        x = torch.cat([traj_emb, route_emb], dim=1)  # (B, horizon + num_waypoints, d_model)
        
        # Create unidirectional self-attention mask
        self_attn_mask = self._create_block_diagonal_mask(
            T_traj, T_route, device=x.device, dtype=x.dtype
        )
        
        # ========== Process Transfuser Features (DiffusionDriveV2 style) ==========
        # bev_feature: (B, 1512, 8, 8) -> (B, 64, 1512) -> (B, 64, d_model)
        bev_feat = transfuser_bev_feature.flatten(2).permute(0, 2, 1)  # (B, 64, 1512)
        bev_proj = self.bev_feature_proj(bev_feat)  # (B, 64, d_model)
        
        # Add position embeddings to BEV feature
        if bev_proj.shape[1] <= self.combined_pos_emb.shape[1]:
            bev_proj = bev_proj + self.combined_pos_emb[:, :bev_proj.shape[1], :]
        
        # ========== Apply GridSampleCrossBEVAttention for spatial BEV (DiffusionDriveV2 style) ==========
        # This enhances trajectory queries with spatially-sampled BEV features
        if traj_points is not None:
            x_traj = x[:, :T_traj, :]  # (B, T_traj, d_model)
            x_traj = self.bev_spatial_attn(x_traj, traj_points, transfuser_bev_feature_upsample)
            x = torch.cat([x_traj, x[:, T_traj:, :]], dim=1)

        # Decoder layers with multi-source attention
        # Pass separate feature tokens and T_traj for route-specific processing
        for layer in self.layers:
            x = layer(
                x,
                bev_proj,           # bev_tokens: (B, 64, d_model)
                conditioning,
                self_attn_mask,
                None,               # bev_padding_mask
                route_conditioning=route_conditioning,
                T_traj=T_traj,
            )
        
        x = self.final_norm(x)
        
        # Split outputs
        traj_out = x[:, :T_traj, :]
        route_out = x[:, T_traj:, :]
        
        # ========== Route Residual Path (Stability Enhancement) ==========
        # Add route-specific residual from initial queries (bypasses shared decoder)
        # This provides a stable baseline that the shared decoder output modulates
        # Similar to how AdaLNRouteHeadDecoderOnly had independent route_queries
        route_residual = self.route_residual_path(route_emb)
        route_out = route_out + torch.sigmoid(self.route_residual_gate) * route_residual
        
        return traj_out, route_out


# =============================================================================
# Main Model: TransformerForDiffusion (Decoder-Only with Unified Queries)
# =============================================================================

class TransformerForDiffusion(ModuleAttrMixin):
    """
    Multimodal Transformer for Trajectory Prediction (DiffusionDrive style).
    
    Key features:
    - Multimodal prediction: outputs predictions for all anchor modes simultaneously
    - Classification head: predicts which mode is best
    - Regression head: predicts trajectory refinement for each mode
    - Cross-attention to: Transfuser features (bev_feature, bev_feature_upsample)
    - ego_status history used for AdaLN conditioning only
    - MLP output heads
    
    Input:
    - anchors: (B, num_modes, anchor_num_points, 2) - all anchor trajectories
    
    Output:
    - poses_reg: (B, num_modes, horizon, 2) - trajectory predictions for each mode
    - poses_cls: (B, num_modes) - classification logits for mode selection
    - route_pred: (B, num_waypoints, 2) - route prediction
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 2,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        obs_as_cond: bool = False,
        n_cond_layers: int = 4,
        status_dim: int = 15,
        ego_status_seq_len: int = 1,
        # Transfuser feature dimensions (following DiffusionDriveV2)
        transfuser_bev_dim: int = 1512,        # bev_feature channel dim
        transfuser_bev_upsample_dim: int = 64,  # bev_feature_upsample channel dim
        num_waypoints: int = 20,
        num_modes: int = 32,  # Number of anchor modes
    ) -> None:
        super().__init__()
        
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.num_waypoints = num_waypoints
        self.num_modes = num_modes
        self.status_dim = status_dim
        self.transfuser_bev_dim = transfuser_bev_dim
        self.transfuser_bev_upsample_dim = transfuser_bev_upsample_dim
        self.T = horizon
        self.output_dim = output_dim
        
        # ========== Anchor Embedding ==========
        # Embed anchor trajectories: (B, num_modes, anchor_points, 2) -> (B, num_modes, n_emb)
        self.anchor_emb = nn.Sequential(
            nn.Linear(input_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )

        # Learnable mode queries for each anchor
        self.mode_queries = nn.Parameter(torch.randn(1, num_modes, n_emb))

        self.drop = nn.Dropout(p_drop_emb)
        self.pre_decoder_norm = nn.LayerNorm(n_emb)

        # Conditioning: timestep + current_status + GRU-encoded history
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.ego_status_proj = nn.Linear(status_dim, n_emb)
        self.history_encoder = HistoryEncoder(status_dim, n_emb)

        # Route-specific conditioning generator
        self.route_status_proj = nn.Sequential(
            nn.Linear(status_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )

        # ========== Unified Decoder (UnifiedDecoderOnlyTransformer) ==========
        # Handles: BEV feature projection, GridSampleCrossBEVAttention, route queries,
        # position encodings, segment embeddings, and multi-source attention layers.
        self.decoder = UnifiedDecoderOnlyTransformer(
            d_model=n_emb,
            nhead=n_head,
            num_layers=n_layer,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            transfuser_bev_dim=transfuser_bev_dim,
            transfuser_bev_upsample_dim=transfuser_bev_upsample_dim,
            horizon=horizon,        # used for GridSampleCrossBEVAttention.num_points
            num_waypoints=num_waypoints,
        )

        # ========== Output Heads ==========
        # Trajectory regression head: (B, num_modes, n_emb) -> (B, num_modes, horizon, 2)
        self.trajectory_head = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, horizon * output_dim),
        )

        # Classification head: (B, num_modes, n_emb) -> (B, num_modes)
        self.cls_head = nn.Sequential(
            nn.Linear(n_emb, n_emb // 2),
            nn.SiLU(),
            nn.Linear(n_emb // 2, 1),
        )

        # Route head: (B, num_waypoints, n_emb) -> (B, num_waypoints, 2)
        self.route_head = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, 2),
        )
        
        self.apply(self._init_weights)
        
        logger.info("TransformerForDiffusion (Multimodal) - parameters: %e", 
                   sum(p.numel() for p in self.parameters()))
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            pass  # UnifiedDecoderOnlyTransformer handles its own init
    
    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = set()
        no_decay = set()
        whitelist = (nn.Linear, nn.MultiheadAttention, nn.Conv1d, nn.Conv2d, nn.GRU)
        blacklist = (nn.LayerNorm, nn.Embedding, RMSNorm)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias") or "bias" in pn:
                    no_decay.add(fpn)
                elif "weight" in pn and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        for name in param_dict:
            if 'pos_emb' in name or '_dummy_variable' in name or 'segment_emb' in name:
                no_decay.add(name)
            elif 'route_queries' in name or 'pool_query' in name or 'mode_queries' in name:
                no_decay.add(name)
            elif 'gating_factor' in name:
                no_decay.add(name)
            elif 'temp_' in name or 'bias_' in name:
                # Temperature and bias parameters - no weight decay
                no_decay.add(name)
            elif 'inv_freq' in name:
                # inv_freq is a buffer, should not be in param_dict, but handle if exists
                no_decay.add(name)
            # ========== New parameters for stability enhancement ==========
            elif '_residual_gate' in name or '_residual_query' in name:
                # Residual gates and queries - no weight decay (like other gate params)
                no_decay.add(name)
            elif '_pos_scale' in name:
                # Position scaling parameters - no weight decay
                no_decay.add(name)
            elif 'summary_query' in name:
                # History encoder summary query - no weight decay
                no_decay.add(name)
            elif 'guidance_gate' in name:
                # Route guidance gate - no weight decay
                no_decay.add(name)
            elif 'route_temp_' in name or 'route_bias_' in name:
                # Route-specific temperature and bias parameters - no weight decay
                no_decay.add(name)
        
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0, f"Missing params: {param_dict.keys() - union_params}"
        
        return [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
    
    def configure_optimizers(self, learning_rate: float = 1e-4, weight_decay: float = 1e-3,
                            betas: Tuple[float, float] = (0.9, 0.95)):
        return torch.optim.AdamW(self.get_optim_groups(weight_decay), lr=learning_rate, betas=betas)
    
    def forward(
        self,
        anchors: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        ego_status: torch.Tensor,
        **kwargs
    ):
        """
        Multimodal forward pass for trajectory prediction.
        
        Args:
            anchors: (B, num_modes, anchor_num_points, 2) - all anchor trajectories
            timestep: diffusion timestep (for conditioning, can be 0 at inference)
            transfuser_bev_feature: (B, 1512, 8, 8) - BEV feature from transfuser
            transfuser_bev_feature_upsample: (B, 64, 64, 64) - Upsampled BEV for spatial attention
            ego_status: (B, T_obs, status_dim) - ego status history
            
        Returns:
            poses_reg: (B, num_modes, horizon, 2) - trajectory predictions for each mode
            poses_cls: (B, num_modes) - classification logits for mode selection
            route_pred: (B, num_waypoints, 2) - route prediction
        """
        model_dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        
        anchors = anchors.contiguous().to(device=device, dtype=model_dtype)
        transfuser_bev_feature = transfuser_bev_feature.contiguous().to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = transfuser_bev_feature_upsample.contiguous().to(device=device, dtype=model_dtype)
        ego_status = ego_status.to(device=device, dtype=model_dtype)
        
        B = anchors.shape[0]
        num_modes = anchors.shape[1]
        anchor_num_points = anchors.shape[2]
        
        # ========== Timestep handling ==========
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(device)
        timesteps = timestep.expand(B)
        
        # ========== Conditioning ==========
        # 1. Timestep embedding
        time_emb = self.time_emb(timesteps).to(dtype=model_dtype)  # (B, n_emb)
        
        # 2. Current status embedding (last frame only)
        current_status = ego_status[:, -1, :]  # (B, status_dim)
        status_emb = self.ego_status_proj(current_status)  # (B, n_emb)
        
        # 3. GRU-encoded global history
        hist_global_emb = self.history_encoder(ego_status)  # (B, n_emb)
        
        # Combined conditioning
        conditioning = time_emb + status_emb + hist_global_emb  # (B, n_emb)
        
        # ========== Anchor Embedding ==========
        # anchors: (B, num_modes, anchor_num_points, 2)
        # Embed using average anchor position -> (B, num_modes, n_emb)
        anchors_flat = anchors.mean(dim=2)  # (B, num_modes, 2) - average anchor position
        anchor_emb = self.anchor_emb(anchors_flat)  # (B, num_modes, n_emb)

        # Add learnable mode queries
        mode_queries = self.mode_queries.expand(B, -1, -1)  # (B, num_modes, n_emb)

        # Combine: anchor embedding + mode queries + conditioning
        mode_emb = anchor_emb + mode_queries + conditioning.unsqueeze(1)  # (B, num_modes, n_emb)
        mode_emb = self.drop(mode_emb)
        mode_emb = self.pre_decoder_norm(mode_emb)

        # Route-specific conditioning
        route_conditioning = self.route_status_proj(current_status)  # (B, n_emb)

        # ========== UnifiedDecoderOnlyTransformer ==========
        # traj_emb = mode_emb (B, num_modes, n_emb) - each mode is one "trajectory query"
        # traj_points = anchors (B, num_modes, horizon, 2) - each mode samples BEV at its anchor waypoints
        # The decoder internally builds route_queries and returns (mode_out, route_out)
        mode_out, route_out = self.decoder(
            traj_emb=mode_emb,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            conditioning=conditioning,
            traj_points=anchors,  # (B, num_modes, horizon, 2)
            route_conditioning=route_conditioning,
        )
        # mode_out: (B, num_modes, n_emb), route_out: (B, num_waypoints, n_emb)
        
        # ========== Output Heads ==========
        # 1. Trajectory regression: (B, num_modes, n_emb) -> (B, num_modes, horizon * 2)
        traj_flat = self.trajectory_head(mode_out)  # (B, num_modes, horizon * output_dim)
        poses_reg = traj_flat.view(B, num_modes, self.horizon, self.output_dim)  # (B, num_modes, horizon, 2)
        
        # Add anchor as residual (predict refinement)
        # Anchor num_points must match horizon (no interpolation for delta prediction)
        assert anchor_num_points == self.horizon, \
            f"anchor_num_points ({anchor_num_points}) must equal horizon ({self.horizon}). " \
            f"Interpolating deltas is incorrect - ensure config aligns these values."

        poses_reg = poses_reg + anchors  # Residual prediction
        
        # 2. Classification: (B, num_modes, n_emb) -> (B, num_modes, 1) -> (B, num_modes)
        poses_cls = self.cls_head(mode_out).squeeze(-1)  # (B, num_modes)
        
        # 3. Route prediction from unified decoder output
        route_pred = self.route_head(route_out)  # (B, num_waypoints, 2)
        
        return poses_reg, poses_cls, route_pred


# =============================================================================
# Test
# =============================================================================

def test():
    """Test the multimodal architecture."""
    print("=" * 60)
    print("Testing TransformerForDiffusion (Multimodal)")
    print("=" * 60)
    
    transformer = TransformerForDiffusion(
        input_dim=2,
        output_dim=2,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        n_layer=6,
        n_head=8,
        n_emb=512,
        causal_attn=True,
        status_dim=14,
        transfuser_bev_dim=1512,
        transfuser_bev_upsample_dim=64,
        num_waypoints=20,
        num_modes=32,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    B = 4
    timestep = torch.tensor(0)
    anchors = torch.randn((B, 32, 8, 2))  # (B, num_modes, anchor_num_points=horizon, 2)
    
    # Transfuser features (following DiffusionDriveV2: only bev_feature and bev_feature_upsample)
    transfuser_bev_feature = torch.randn((B, 1512, 8, 8))
    transfuser_bev_feature_upsample = torch.randn((B, 64, 64, 64))
    
    ego_status = torch.randn((B, 4, 14))  # 4 frames of history with updated dim
    
    print("\nTest 1: Basic forward pass (multimodal)")
    poses_reg, poses_cls, route_pred = transformer(
        anchors=anchors, timestep=timestep,
        transfuser_bev_feature=transfuser_bev_feature,
        transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
        ego_status=ego_status,
    )
    print(f"  poses_reg: {poses_reg.shape} (trajectory predictions for each mode)")
    print(f"  poses_cls: {poses_cls.shape} (classification logits)")
    print(f"  route_pred: {route_pred.shape} (route prediction)")
    assert poses_reg.shape == (B, 32, 8, 2), f"Expected (B, 32, 8, 2), got {poses_reg.shape}"
    assert poses_cls.shape == (B, 32), f"Expected (B, 32), got {poses_cls.shape}"
    assert route_pred.shape == (B, 20, 2), f"Expected (B, 20, 2), got {route_pred.shape}"
    
    print("\nTest 2: Different BEV features affect output")
    transfuser_bev_feature2 = torch.randn((B, 1512, 8, 8))
    poses_reg2, _, _ = transformer(
        anchors=anchors, timestep=timestep,
        transfuser_bev_feature=transfuser_bev_feature2,
        transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
        ego_status=ego_status,
    )
    diff = torch.abs(poses_reg - poses_reg2).mean()
    print(f"  Difference: {diff:.6f}")
    assert diff > 0, "BEV features should affect output"
    
    print("\nTest 3: Different anchors affect output")
    anchors2 = torch.randn((B, 32, 8, 2))
    poses_reg3, _, _ = transformer(
        anchors=anchors2, timestep=timestep,
        transfuser_bev_feature=transfuser_bev_feature,
        transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
        ego_status=ego_status,
    )
    diff_anchors = torch.abs(poses_reg - poses_reg3).mean()
    print(f"  Difference: {diff_anchors:.6f}")
    assert diff_anchors > 0, "Anchors should affect output"
    
    print("\nTest 4: Optimizer")
    opt = transformer.configure_optimizers()
    print(f"  Optimizer created: {type(opt).__name__}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nArchitecture (Multimodal DiffusionDrive style):")
    print("  1. Input: anchors (B, num_modes, anchor_points, 2)")
    print("  2. Output: ")
    print("     - poses_reg: (B, num_modes, horizon, 2) - trajectory predictions")
    print("     - poses_cls: (B, num_modes) - mode classification logits")
    print("     - route_pred: (B, num_waypoints, 2) - route prediction")
    print("  3. Loss:")
    print("     - Focal loss for classification (select best mode)")
    print("     - L1 loss for regression (only on best mode)")
    print("     - L1 loss for route prediction (optional)")
    print("=" * 60)


if __name__ == "__main__":
    test()
