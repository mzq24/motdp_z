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


def gen_sineembed_for_position(pos_tensor: torch.Tensor, hidden_dim: int = 64) -> torch.Tensor:
    """Sinusoidal position embedding for 2D points.

    Args:
        pos_tensor: (..., 2) tensor of (x, y) coordinates.
        hidden_dim: embedding dimension per point (must be divisible by 2).

    Returns:
        (..., hidden_dim) sinusoidal embedding.
    """
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
    pos_emb = torch.cat((pos_y, pos_x), dim=-1)

    return pos_emb.to(dtype=orig_dtype)


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
        # grid_sample treats dim1 as H_out, dim2 as W_out
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)

        sampled_features = F.grid_sample(
            value,
            grid,
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
        traj_can_attend_route: bool = True,
        ego_detail_activation_t: int = 400,
        use_lidar_bev_detail: bool = False,
        lidar_bev_history_frames: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.horizon = horizon
        self.num_waypoints = num_waypoints
        self.traj_can_attend_route = traj_can_attend_route
        self.ego_detail_activation_t = ego_detail_activation_t
        self.use_lidar_bev_detail = use_lidar_bev_detail
        self.lidar_bev_history_frames = max(int(lidar_bev_history_frames), 1)

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
        # Ego path uses one BEV sample per waypoint token.
        self.bev_point_attn = GridSampleCrossBEVAttention(
            embed_dims=d_model,
            num_heads=nhead,
            in_bev_dims=transfuser_bev_upsample_dim,
            num_points=1,
            lidar_max_x=32.0,
            lidar_max_y=32.0
        )
        self.route_point_attn = GridSampleCrossBEVAttention(
            embed_dims=d_model,
            num_heads=nhead,
            in_bev_dims=transfuser_bev_upsample_dim,
            num_points=1,
            lidar_max_x=32.0,
            lidar_max_y=32.0
        )
        self.traj_detail_attn = GridSampleCrossBEVAttention(
            embed_dims=d_model,
            num_heads=nhead,
            in_bev_dims=transfuser_bev_upsample_dim,
            num_points=14,  # 7 near-field + 7 far-field
            lidar_max_x=32.0,
            lidar_max_y=32.0
        )
        self.route_detail_attn = GridSampleCrossBEVAttention(
            embed_dims=d_model,
            num_heads=nhead,
            in_bev_dims=transfuser_bev_upsample_dim,
            num_points=num_waypoints * 3,  # per route point: center + lateral left/right, flattened
            lidar_max_x=32.0,
            lidar_max_y=32.0
        )
        self.route_far_detail_attn = GridSampleCrossBEVAttention(
            embed_dims=d_model,
            num_heads=nhead,
            in_bev_dims=transfuser_bev_upsample_dim,
            num_points=3,  # farthest route point: longitudinal look-ahead
            lidar_max_x=32.0,
            lidar_max_y=32.0
        )

        if self.use_lidar_bev_detail:
            # ========== LiDAR BEV encoder + detail attention ==========
            # Input: inverted transfuser_lidar_bev (B, H_hist, 2, 256, 256) or (B, 2, 256, 256)
            # Output: (B, 64, 64, 64) matching bev_feature_upsample spatial layout
            lidar_bev_in_channels = 2 * self.lidar_bev_history_frames
            self.lidar_bev_in_channels = lidar_bev_in_channels
            # Learned temporal embedding per history frame so stacked LiDAR channels carry
            # explicit frame order instead of relying only on channel position.
            self.lidar_history_pos_emb = nn.Parameter(
                torch.zeros(1, self.lidar_bev_history_frames, 2, 1, 1)
            )
            self.lidar_bev_encoder = nn.Sequential(
                nn.Conv2d(lidar_bev_in_channels, 32, kernel_size=5, stride=2, padding=2),  # → 128
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → 64
                nn.GroupNorm(8, 64),
                nn.GELU(),
            )  # (B, 64, 64, 64)
            self.traj_lidar_detail_attn = GridSampleCrossBEVAttention(
                embed_dims=d_model,
                num_heads=nhead,
                in_bev_dims=64,  # lidar_bev_encoder output channels
                num_points=14,   # same detail offsets as traj_detail_attn
                lidar_max_x=32.0,
                lidar_max_y=32.0
            )
            self.traj_lidar_spatial_attn = GridSampleCrossBEVAttention(
                embed_dims=d_model,
                num_heads=nhead,
                in_bev_dims=64,
                num_points=horizon,
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
        self.register_buffer(
            "ego_detail_offsets",
            torch.tensor([
                # Near-field: car body envelope (7 points)
                [ 0.0,  1.5],   # left side
                [ 0.0, -1.5],   # right side
                [ 1.5,  1.0],   # front-left
                [ 1.5, -1.0],   # front-right
                [-3.0,  1.5],   # rear-left
                [-3.0, -1.5],   # rear-right
                [-3.0,  0.0],   # rear-center
                # Far-field: lane change & following traffic (7 points)
                [ 0.0,  4.0],   # left lane center (lane change gap)
                [ 0.0, -4.0],   # right lane center
                [ 5.0,  3.5],   # front-left far (merge point)
                [ 5.0, -3.5],   # front-right far
                [-6.0,  3.5],   # rear-left far (approaching traffic)
                [-6.0, -3.5],   # rear-right far
                [-8.0,  0.0],   # rear far (following distance)
            ], dtype=torch.float32),
            persistent=False,
        )
        # Route detail: lateral offsets per route point
        self.register_buffer(
            "route_lateral_offsets",
            torch.tensor([
                [ 0.0,  3.5],   # left adjacent lane
                [ 0.0, -3.5],   # right adjacent lane
            ], dtype=torch.float32),
            persistent=False,
        )
        # Route far: longitudinal look-ahead at farthest route point
        self.register_buffer(
            "route_far_offsets",
            torch.tensor([
                [10.0,  0.0],   # straight ahead
                [10.0,  3.0],   # ahead-left
                [10.0, -3.0],   # ahead-right
            ], dtype=torch.float32),
            persistent=False,
        )
        
        # Learnable position scaling for each modality
        # Allows the model to learn different position importance
        self.traj_pos_scale = nn.Parameter(torch.ones(1, 1, d_model))
        self.route_pos_scale = nn.Parameter(torch.ones(1, 1, d_model))

        # Segment embeddings to distinguish query types (trajectory vs route)
        self.traj_segment_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.route_segment_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.speed_segment_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        
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

    def _prepare_lidar_bev_tensor(self, transfuser_lidar_bev: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.use_lidar_bev_detail or transfuser_lidar_bev is None:
            return None
        if transfuser_lidar_bev.dim() == 5:
            B, H_hist, C, H, W = transfuser_lidar_bev.shape
            if C != 2:
                raise ValueError(
                    f"Expected LiDAR history with 2 channels per frame, got {transfuser_lidar_bev.shape}"
                )
            pos_emb = self.lidar_history_pos_emb[:, :H_hist].to(
                device=transfuser_lidar_bev.device,
                dtype=transfuser_lidar_bev.dtype,
            )
            transfuser_lidar_bev = transfuser_lidar_bev + pos_emb
            transfuser_lidar_bev = transfuser_lidar_bev.reshape(B, H_hist * C, H, W)
        elif transfuser_lidar_bev.dim() != 4:
            raise ValueError(
                f"Expected transfuser_lidar_bev as (B, H_hist, 2, 256, 256) or (B, 2, 256, 256), "
                f"got {transfuser_lidar_bev.shape}"
            )
        elif transfuser_lidar_bev.shape[1] == self.lidar_bev_in_channels:
            pos_emb = self.lidar_history_pos_emb.to(
                device=transfuser_lidar_bev.device,
                dtype=transfuser_lidar_bev.dtype,
            ).reshape(1, self.lidar_bev_in_channels, 1, 1)
            transfuser_lidar_bev = transfuser_lidar_bev + pos_emb
        if transfuser_lidar_bev.shape[1] != self.lidar_bev_in_channels:
            raise ValueError(
                f"Expected LiDAR history channels={self.lidar_bev_in_channels}, got {transfuser_lidar_bev.shape[1]}"
            )
        return transfuser_lidar_bev
    
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
        Create a self-attention mask with anchor isolation.

        This enforces:
        - Trajectory(anchor) queries cannot attend to other anchors
          (only self-anchor attention is allowed).
        - Trajectory(anchor) queries optionally attend route queries
          (controlled by self.traj_can_attend_route).
        - Route queries cannot attend to trajectory(anchor) queries.
        - Route queries can attend to route queries.

        Attention pattern (0 = allowed, -inf = blocked):

                    | Trajectory | Route |
        ------------------------------------
        Trajectory  | diagonal 0 |  0/-inf |
        Route       |   -inf     |   0   |

        Args:
            T_traj: Number of trajectory(anchor) queries
            T_route: Number of route queries
            device: Device for the mask tensor
            dtype: Data type for the mask tensor

        Returns:
            mask: (T_total, T_total) mask where -inf blocks attention
        """
        T_total = T_traj + T_route
        mask = torch.zeros((T_total, T_total), device=device, dtype=dtype)

        # Block anchor-to-anchor interactions (off-diagonal only).
        if T_traj > 1:
            traj_block = torch.full((T_traj, T_traj), float('-inf'), device=device, dtype=dtype)
            traj_block.fill_diagonal_(0)
            mask[:T_traj, :T_traj] = traj_block

        # Optional dd-style strict isolation: anchors cannot attend route tokens.
        if not self.traj_can_attend_route:
            mask[:T_traj, T_traj:] = float('-inf')

        # Block route-to-anchor attention (route rows, trajectory cols).
        mask[T_traj:, :T_traj] = float('-inf')

        return mask

    def _create_ego_mask(self, T_traj: int, T_route: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Ego-path mask for [traj_wp | route].

        - Trajectory waypoints can attend to each other.
        - Trajectory waypoints can optionally attend to route tokens.
        - Route tokens cannot attend to trajectory waypoints.
        - Route tokens can attend to each other.
        """
        T_total = T_traj + T_route
        mask = torch.zeros((T_total, T_total), device=device, dtype=dtype)

        if not self.traj_can_attend_route:
            mask[:T_traj, T_traj:] = float('-inf')

        mask[T_traj:, :T_traj] = float('-inf')
        return mask

    def _create_ego_speed_mask(
        self,
        T_traj: int,
        T_route: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Ego-path mask for [speed | traj_wp | route].

        - Speed token can attend to all tokens.
        - Trajectory/route tokens cannot attend to speed token, to avoid
          perturbing the existing traj/route interaction pattern.
        - Trajectory and route tokens preserve the original ego mask semantics.
        """
        T_speed = 1
        T_total = T_speed + T_traj + T_route
        mask = torch.zeros((T_total, T_total), device=device, dtype=dtype)

        # Trajectory/route tokens do not attend to the prepended speed token.
        mask[T_speed:, :T_speed] = float('-inf')

        traj_start = T_speed
        route_start = T_speed + T_traj

        if not self.traj_can_attend_route:
            mask[traj_start:route_start, route_start:] = float('-inf')

        # Route tokens cannot attend back to trajectory tokens.
        mask[route_start:, traj_start:route_start] = float('-inf')
        return mask

    @staticmethod
    def _create_full_mask(
        total_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Fully-connected self-attention mask (all tokens attend to each other)."""
        return torch.zeros((total_tokens, total_tokens), device=device, dtype=dtype)

    def _build_traj_detail_points(self, traj_points: torch.Tensor) -> torch.Tensor:
        """Expand each traj waypoint with ego detail offsets. (B, T, 2) -> (B, T, 14, 2)"""
        offsets = self.ego_detail_offsets.to(device=traj_points.device, dtype=traj_points.dtype)
        return traj_points.unsqueeze(2) + offsets.view(1, 1, offsets.shape[0], 2)

    def _build_route_detail_points(self, route_points: torch.Tensor) -> torch.Tensor:
        """Expand each route point with center + lateral offsets, rotated to route heading.
        (B, T_route, 2) -> (B, T_route, 3, 2)
        """
        B, T, _ = route_points.shape
        lat_offsets = self.route_lateral_offsets.to(device=route_points.device, dtype=route_points.dtype)  # (2, 2)

        # Compute per-point heading from adjacent route points
        # Forward difference, last point copies from previous
        diff = torch.zeros_like(route_points)  # (B, T, 2)
        diff[:, :-1, :] = route_points[:, 1:, :] - route_points[:, :-1, :]
        diff[:, -1, :] = diff[:, -2, :] if T > 1 else torch.tensor([1.0, 0.0], device=route_points.device)

        heading_norm = diff.norm(dim=-1, keepdim=True).clamp(min=1e-4)  # (B, T, 1)
        dx = diff[..., 0:1] / heading_norm  # (B, T, 1)
        dy = diff[..., 1:2] / heading_norm  # (B, T, 1)

        # Rotation matrix per point: [dx, -dy; dy, dx]
        # Rotate each lateral offset by local heading
        # lat_offsets: (N_off, 2) where [along_route, perpendicular]
        along = lat_offsets[:, 0]  # (N_off,)
        perp = lat_offsets[:, 1]   # (N_off,)

        # Rotated offset = along * heading + perp * heading_perp
        # heading = (dx, dy), heading_perp = (-dy, dx)
        # (B, T, 1) * (N_off,) -> broadcast
        rot_x = dx * along.view(1, 1, -1) + (-dy) * perp.view(1, 1, -1)  # (B, T, N_off)
        rot_y = dy * along.view(1, 1, -1) + dx * perp.view(1, 1, -1)     # (B, T, N_off)
        rotated_offsets = torch.stack([rot_x, rot_y], dim=-1)  # (B, T, N_off, 2)

        # Center point + rotated lateral offsets -> (B, T, 1+N_off, 2)
        center = route_points.unsqueeze(2)  # (B, T, 1, 2)
        detail_points = route_points.unsqueeze(2) + rotated_offsets  # (B, T, N_off, 2)
        return torch.cat([center, detail_points], dim=2)  # (B, T, 3, 2)

    def _build_route_far_detail_points(self, route_points: torch.Tensor) -> torch.Tensor:
        """Build look-ahead points at the farthest route point, rotated to its heading.
        (B, T_route, 2) -> (B, 1, 3, 2)
        """
        far_offsets = self.route_far_offsets.to(device=route_points.device, dtype=route_points.dtype)  # (3, 2)
        T = route_points.shape[1]

        # Heading at farthest point (from second-to-last to last)
        if T > 1:
            diff = route_points[:, -1, :] - route_points[:, -2, :]  # (B, 2)
        else:
            diff = torch.tensor([[1.0, 0.0]], device=route_points.device).expand(route_points.shape[0], -1)

        heading_norm = diff.norm(dim=-1, keepdim=True).clamp(min=1e-4)
        dx = diff[:, 0:1] / heading_norm  # (B, 1)
        dy = diff[:, 1:2] / heading_norm  # (B, 1)

        along = far_offsets[:, 0]  # (3,)
        perp = far_offsets[:, 1]   # (3,)

        rot_x = dx * along.view(1, -1) + (-dy) * perp.view(1, -1)  # (B, 3)
        rot_y = dy * along.view(1, -1) + dx * perp.view(1, -1)     # (B, 3)
        rotated = torch.stack([rot_x, rot_y], dim=-1)  # (B, 3, 2)

        far_point = route_points[:, -1:, :].unsqueeze(2)  # (B, 1, 1, 2)
        return (far_point + rotated.unsqueeze(1))  # (B, 1, 3, 2)

    def _detail_gate(self, timesteps: Optional[torch.Tensor], batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if timesteps is None:
            return torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=device, dtype=torch.long)
        else:
            timesteps = timesteps.to(device=device, dtype=torch.long).reshape(-1)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.numel() != batch_size:
            raise ValueError(f"Expected 1 or {batch_size} timesteps, got {timesteps.numel()}")
        return (timesteps <= self.ego_detail_activation_t).to(dtype=dtype).view(batch_size, 1, 1)

    def compute_bev_proj(self, transfuser_bev_feature: torch.Tensor) -> torch.Tensor:
        """Precompute BEV token projection for reuse across split forwards."""
        bev_feat = transfuser_bev_feature.flatten(2).permute(0, 2, 1)  # (B, 64, 1512)
        bev_proj = self.bev_feature_proj(bev_feat)  # (B, 64, d_model)
        if bev_proj.shape[1] <= self.combined_pos_emb.shape[1]:
            bev_proj = bev_proj + self.combined_pos_emb[:, :bev_proj.shape[1], :]
        return bev_proj
    
    def forward(
        self,
        traj_emb: torch.Tensor,  # (B, T_traj, d_model) - trajectory query embeddings
        transfuser_bev_feature: torch.Tensor,       # (B, 1512, 8, 8)
        transfuser_bev_feature_upsample: torch.Tensor,  # (B, 64, 64, 64)
        conditioning: torch.Tensor,                 # (B, d_model)
        speed_emb: Optional[torch.Tensor] = None,   # (B, 1, d_model) - optional speed token embedding
        extra_emb: Optional[torch.Tensor] = None,   # (B, T_extra, d_model) - optional state tokens
        traj_points: Optional[torch.Tensor] = None,  # (B, T_traj, 2) or (B, T_traj, horizon, 2) for GridSampleCrossBEVAttention
        route_emb: Optional[torch.Tensor] = None,   # (B, T_route, d_model) - route query embeddings for ego diffusion
        route_points: Optional[torch.Tensor] = None,  # (B, T_route, 2) absolute route points for BEV sampling
        timesteps: Optional[torch.Tensor] = None,   # (B,) diffusion timesteps for detail gating
        route_conditioning: Optional[torch.Tensor] = None,  # (B, d_model) - route-specific conditioning
        bev_proj_cached: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        route_pos_offset: int = 0,
        spatial_mode: str = "anchor",
        transfuser_lidar_bev: Optional[torch.Tensor] = None,  # (B, 2, 256, 256) inverted LiDAR BEV
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with unified queries and multi-source attention (DiffusionDriveV2 style).
        
        Uses anchor-isolated self-attention mask:
        - Each trajectory(anchor) query can only see itself
          (or itself + route if traj_can_attend_route=True)
        - Route queries can only see route queries
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
            route_emb: (B, num_waypoints, d_model) - route query embeddings
            route_points: (B, num_waypoints, 2) - route points for fine-BEV sampling
            timesteps: (B,) diffusion timestep used to gate detail branches
            route_conditioning: (B, d_model) - route-specific conditioning (optional)
            
        Returns:
            traj_out: (B, horizon, d_model) - trajectory output
            route_out: (B, num_waypoints, d_model) - route output
            speed_out: (B, 1, d_model) or None - speed token output
        """
        B = traj_emb.shape[0]
        T_traj = traj_emb.shape[1]
        T_route = self.num_waypoints
        T_speed = 1 if speed_emb is not None else 0
        T_extra = extra_emb.shape[1] if extra_emb is not None else 0

        # ========== Unified Position Encoding ==========
        # Get sinusoidal position encoding for trajectory
        traj_pos = self.unified_pos_encoding[:, :T_traj, :] * self.traj_pos_scale
        # Get sinusoidal position encoding for route  
        route_pos = self.unified_pos_encoding[:, route_pos_offset:route_pos_offset + T_route, :] * self.route_pos_scale
        
        # Add position + segment embeddings to trajectory
        traj_emb = traj_emb + traj_pos + self.traj_segment_emb
        
        # Route queries with position and segment embeddings
        if route_emb is None:
            route_emb = self.route_queries.expand(B, -1, -1)
        route_emb = route_emb + route_pos + self.route_segment_emb

        if speed_emb is not None:
            speed_emb = speed_emb + self.speed_segment_emb

        if extra_emb is not None:
            extra_emb = extra_emb + self.speed_segment_emb

        # Concatenate queries: [speed | trajectory | route | extra]
        if speed_emb is not None:
            pieces = [speed_emb, traj_emb, route_emb]
        else:
            pieces = [traj_emb, route_emb]
        if extra_emb is not None:
            pieces.append(extra_emb)
        x = torch.cat(pieces, dim=1)

        if self_attn_mask is None:
            if extra_emb is not None:
                self_attn_mask = self._create_full_mask(
                    x.shape[1], device=x.device, dtype=x.dtype
                )
            elif speed_emb is not None:
                self_attn_mask = self._create_ego_speed_mask(
                    T_traj, T_route, device=x.device, dtype=x.dtype
                )
            else:
                self_attn_mask = self._create_block_diagonal_mask(
                    T_traj, T_route, device=x.device, dtype=x.dtype
                )
        
        # ========== Process Transfuser Features (DiffusionDriveV2 style) ==========
        # bev_feature: (B, 1512, 8, 8) -> (B, 64, 1512) -> (B, 64, d_model)
        if bev_proj_cached is not None:
            bev_proj = bev_proj_cached
        else:
            bev_proj = self.compute_bev_proj(transfuser_bev_feature)
        
        # ========== Apply GridSampleCrossBEVAttention for spatial BEV (DiffusionDriveV2 style) ==========
        # This enhances trajectory queries with spatially-sampled BEV features
        lidar_feat = None
        if self.use_lidar_bev_detail and transfuser_lidar_bev is not None:
            transfuser_lidar_bev = self._prepare_lidar_bev_tensor(transfuser_lidar_bev)
            lidar_feat = self.lidar_bev_encoder(transfuser_lidar_bev)  # (B, 64, 64, 64)

        if traj_points is not None:
            x_speed = x[:, :T_speed, :] if T_speed > 0 else None
            x_traj = x[:, T_speed:T_speed + T_traj, :]  # (B, T_traj, d_model)
            x_extra = x[:, T_speed + T_traj + T_route:, :] if T_extra > 0 else None
            if spatial_mode == "ego":
                if traj_points.dim() != 3:
                    raise ValueError(f"Ego traj_points must be (B, T, 2), got {traj_points.shape}")
                center_points = traj_points.unsqueeze(2)
                x_traj_center = self.bev_point_attn(x_traj, center_points, transfuser_bev_feature_upsample)
                detail_gate = self._detail_gate(timesteps, B, x_traj.device, x_traj.dtype)
                # Ego detail: near-field (car body) + far-field (lane change, following)
                traj_detail_points = self._build_traj_detail_points(traj_points)  # (B, T, 14, 2)
                traj_local_detail = self.traj_detail_attn(x_traj, traj_detail_points, transfuser_bev_feature_upsample) - x_traj
                # Route detail: center + rotated lateral offsets per route point
                if route_points is None:
                    raise ValueError("route_points are required for ego diffusion route detail sampling")
                route_detail_points = self._build_route_detail_points(route_points)  # (B, T_route, 3, 2)
                # Flatten to (B, T_route*3, 2) so all route detail points are shared across traj queries
                route_detail_flat = route_detail_points.view(B, -1, 2)  # (B, T_route*3, 2)
                traj_route_detail = self.route_detail_attn(x_traj, route_detail_flat, transfuser_bev_feature_upsample) - x_traj
                # Route far detail: longitudinal look-ahead at farthest route point
                route_far_points = self._build_route_far_detail_points(route_points)  # (B, 1, 3, 2)
                route_far_flat = route_far_points.view(B, -1, 2)  # (B, 3, 2)
                traj_route_far = self.route_far_detail_attn(x_traj, route_far_flat, transfuser_bev_feature_upsample) - x_traj
                # LiDAR BEV detail: sample obstacle occupancy at same detail points
                traj_lidar_detail = torch.zeros_like(traj_local_detail)
                if lidar_feat is not None:
                    traj_lidar_detail = self.traj_lidar_detail_attn(x_traj, traj_detail_points, lidar_feat) - x_traj
                x_traj = x_traj_center + detail_gate * (traj_local_detail + traj_route_detail + traj_route_far + traj_lidar_detail)
                x_route = x[:, T_speed + T_traj:T_speed + T_traj + T_route, :]
                if route_points is not None:
                    if route_points.dim() != 3:
                        raise ValueError(f"Ego route_points must be (B, T_route, 2), got {route_points.shape}")
                    x_route = self.route_point_attn(
                        x_route,
                        route_points.unsqueeze(2),
                        transfuser_bev_feature_upsample,
                    )
            else:
                x_traj_base = self.bev_spatial_attn(x_traj, traj_points, transfuser_bev_feature_upsample)
                detail_gate = self._detail_gate(timesteps, B, x_traj.device, x_traj.dtype)
                detail_residual = torch.zeros_like(x_traj)
                if route_points is not None:
                    if route_points.dim() != 3:
                        raise ValueError(f"Anchor route_points must be (B, T_route, 2), got {route_points.shape}")
                    route_detail_points = self._build_route_detail_points(route_points)
                    route_detail_flat = route_detail_points.view(B, -1, 2)
                    route_far_points = self._build_route_far_detail_points(route_points)
                    route_far_flat = route_far_points.view(B, -1, 2)
                    detail_residual = detail_residual + (
                        self.route_detail_attn(x_traj, route_detail_flat, transfuser_bev_feature_upsample) - x_traj
                    )
                    detail_residual = detail_residual + (
                        self.route_far_detail_attn(x_traj, route_far_flat, transfuser_bev_feature_upsample) - x_traj
                    )
                if lidar_feat is not None:
                    detail_residual = detail_residual + (
                        self.traj_lidar_spatial_attn(x_traj, traj_points, lidar_feat) - x_traj
                    )
                x_traj = x_traj_base + detail_gate * detail_residual
                x_route = x[:, T_speed + T_traj:T_speed + T_traj + T_route, :]
                if route_points is not None:
                    if route_points.dim() != 3:
                        raise ValueError(f"Anchor route_points must be (B, T_route, 2), got {route_points.shape}")
                    x_route = self.route_point_attn(
                        x_route,
                        route_points.unsqueeze(2),
                        transfuser_bev_feature_upsample,
                    )
            if x_speed is not None:
                pieces = [x_speed, x_traj, x_route]
            else:
                pieces = [x_traj, x_route]
            if x_extra is not None:
                pieces.append(x_extra)
            x = torch.cat(pieces, dim=1)

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
        speed_out = x[:, :T_speed, :] if T_speed > 0 else None
        traj_out = x[:, T_speed:T_speed + T_traj, :]
        route_out = x[:, T_speed + T_traj:T_speed + T_traj + T_route, :]
        extra_out = x[:, T_speed + T_traj + T_route:, :] if T_extra > 0 else None
        
        # ========== Route Residual Path (Stability Enhancement) ==========
        # Add route-specific residual from initial queries (bypasses shared decoder)
        # This provides a stable baseline that the shared decoder output modulates
        # Similar to how AdaLNRouteHeadDecoderOnly had independent route_queries
        route_residual = self.route_residual_path(route_emb)
        route_out = route_out + torch.sigmoid(self.route_residual_gate) * route_residual
        
        if extra_emb is not None:
            return traj_out, route_out, speed_out, extra_out
        return traj_out, route_out, speed_out


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
        num_behaviors: int = 0,  # Number of behavior categories (0 = disabled)
        traj_can_attend_route: bool = True,
        anchor_free: bool = False,  # Route B: no anchor residual, predict absolute trajectory
        energy_heads: bool = False,  # Route B: energy evaluator heads for gradient guidance
        ego_detail_activation_t: int = 400,  # Timestep threshold for detail gate
        use_lidar_bev_detail: bool = False,
        lidar_bev_history_frames: int = 1,
        use_condition_group_dropout: bool = False,
        use_joint_state_diffusion: bool = False,
    ) -> None:
        super().__init__()

        self.anchor_free = anchor_free
        self.energy_heads_enabled = energy_heads

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
        self.ego_joint_horizon = horizon + num_waypoints
        self.use_lidar_bev_detail = use_lidar_bev_detail
        self.lidar_bev_history_frames = max(int(lidar_bev_history_frames), 1)
        self.use_condition_group_dropout = use_condition_group_dropout
        self.use_joint_state_diffusion = use_joint_state_diffusion
        
        # ========== Anchor Embedding ==========
        # Encode full noisy trajectory shape per mode (not just mean point) to preserve
        # mode identity under multi-step denoising.
        self.anchor_pos_hidden_dim = 64
        self.anchor_embed_dim = horizon * self.anchor_pos_hidden_dim
        self.anchor_emb = nn.Sequential(
            nn.Linear(self.anchor_embed_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )
        self.wp_emb = nn.Sequential(
            nn.Linear(self.anchor_pos_hidden_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )
        self.route_wp_emb = nn.Sequential(
            nn.Linear(self.anchor_pos_hidden_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )

        # Learnable mode queries for each anchor (energy training uses all num_modes)
        self.mode_queries = nn.Parameter(torch.randn(1, num_modes, n_emb))
        # Dedicated diffusion mode query (single-mode denoising in Route B)
        self.diff_mode_query = nn.Parameter(torch.randn(1, 1, n_emb))
        self.route_diff_query = nn.Parameter(torch.randn(1, 1, n_emb))
        # Dedicated GT mode query (unified training: GT slot in energy evaluation)
        self.gt_mode_query = nn.Parameter(torch.randn(1, 1, n_emb))
        # Extra learnable query for VLM anchor (33rd mode), used when use_vqa_anchor=True
        self.vqa_mode_query = nn.Parameter(torch.randn(1, 1, n_emb))

        # Semantic behavior conditioning (optional)
        self.num_behaviors = num_behaviors
        if num_behaviors > 0:
            self.behavior_emb = nn.Embedding(num_behaviors, n_emb)
            self.allowed_emb = nn.Embedding(2, n_emb)  # 0=forbidden, 1=allowed

        self.drop = nn.Dropout(p_drop_emb)
        self.pre_decoder_norm = nn.LayerNorm(n_emb)

        # Conditioning: timestep + current_status + GRU-encoded history
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.ego_status_proj = nn.Linear(status_dim, n_emb)
        self.history_encoder = HistoryEncoder(status_dim, n_emb)
        self.traj_window_condition_dim = 4
        self.traj_dir_condition_dim = 4
        self.traj_decision_phase_condition_dim = 2
        self.traj_control_phase_condition_dim = 4
        self.traj_boundary_margin_dim = 2
        self.traj_borrow_aux_dim = 1
        self.traj_branch_condition_dim = (
            self.traj_window_condition_dim
            + self.traj_dir_condition_dim
            + self.traj_decision_phase_condition_dim
            + self.traj_control_phase_condition_dim
            + self.traj_boundary_margin_dim
            + self.traj_borrow_aux_dim
        )
        self.traj_window_condition_proj = nn.Sequential(
            nn.Linear(self.traj_window_condition_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )
        self.traj_dir_condition_proj = nn.Sequential(
            nn.Linear(self.traj_dir_condition_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )
        self.traj_decision_phase_condition_proj = nn.Sequential(
            nn.Linear(self.traj_decision_phase_condition_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )
        self.traj_control_phase_condition_proj = nn.Sequential(
            nn.Linear(self.traj_control_phase_condition_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )
        self.traj_boundary_margin_proj = nn.Sequential(
            nn.Linear(self.traj_boundary_margin_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )
        self.traj_borrow_aux_proj = nn.Sequential(
            nn.Linear(self.traj_borrow_aux_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )

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
            traj_can_attend_route=traj_can_attend_route,
            ego_detail_activation_t=ego_detail_activation_t,
            use_lidar_bev_detail=use_lidar_bev_detail,
            lidar_bev_history_frames=self.lidar_bev_history_frames,
        )

        # ========== Output Heads ==========
        # Trajectory regression head: (B, num_modes, n_emb) -> (B, num_modes, horizon*2)
        # Route guidance cross-attention allows trajectory to attend to route features
        self.trajectory_head = TrajectoryMLPHead(
            n_emb=n_emb,
            output_dim=horizon * output_dim,
            p_drop=p_drop_emb,
            num_heads=n_head,
        )
        self.trajectory_wp_head = TrajectoryMLPHead(
            n_emb=n_emb,
            output_dim=output_dim,
            p_drop=p_drop_emb,
            num_heads=n_head,
        )

        # Classification head: (B, num_modes, n_emb) -> (B, num_modes)
        self.cls_head = nn.Sequential(
            nn.Linear(n_emb, n_emb // 2),
            nn.SiLU(),
            nn.Linear(n_emb // 2, 1),
        )

        # Energy evaluator heads (Route B: evaluate pred_x0 + scene context)
        # Input: concat(pred_x0_flat, mode_out) = (B, M, horizon*output_dim + n_emb)
        #   - pred_x0_flat: trajectory being evaluated (gradient flows back for guidance)
        #   - mode_out: scene context from BEV attention + ego status (provides scene understanding)
        # 6 heads: front/left/right (vehicle collision by direction), pedestrian, offroad, route
        if energy_heads:
            energy_in_dim = self.horizon * self.output_dim + n_emb  # T*2 + n_emb
            def _make_energy_head():
                return nn.Sequential(
                    nn.Linear(energy_in_dim, n_emb // 2), nn.SiLU(),
                    nn.Linear(n_emb // 2, 1),
                )
            self.energy_front_head      = _make_energy_head()  # vehicle collision front (label 1)
            self.energy_left_head       = _make_energy_head()  # vehicle collision left  (label 2)
            self.energy_right_head      = _make_energy_head()  # vehicle collision right (label 3)
            self.energy_pedestrian_head = _make_energy_head()  # pedestrian collision    (label 4)
            self.energy_offroad_head    = _make_energy_head()  # off_road/sidewalk       (label 5-6)
            self.energy_route_head      = _make_energy_head()  # route deviation (continuous, computed in policy)
            self.front_route_risk_head  = _make_energy_head()  # route-conditioned front risk (GT/pred_x0 path)

            self.shared_stage1_route_geom_proj = nn.Sequential(
                nn.Linear(self.num_waypoints * self.output_dim, n_emb),
                nn.SiLU(),
                nn.Linear(n_emb, n_emb),
            )
            # Shared main-path semantic neck (shared path v1)
            self.shared_stage1_pool_attn = nn.MultiheadAttention(
                embed_dim=n_emb,
                num_heads=n_head,
                batch_first=True,
            )
            self.shared_stage1_pool_norm = nn.LayerNorm(n_emb)
            self.shared_stage1_traj_summary_query = nn.Parameter(torch.randn(1, 1, n_emb))
            self.shared_stage1_route_summary_query = nn.Parameter(torch.randn(1, 1, n_emb))
            self.shared_stage1_neck = nn.Sequential(
                nn.Linear(5 * n_emb, n_emb),
                nn.SiLU(),
                nn.Linear(n_emb, n_emb),
                nn.LayerNorm(n_emb),
            )
            self.shared_stage1_speed_query_proj = nn.Sequential(
                nn.Linear(1, n_emb),
                nn.SiLU(),
                nn.Linear(n_emb, n_emb),
            )
            self.shared_stage1_query_token = nn.Parameter(torch.randn(1, 1, n_emb))
            self.shared_stage1_query_attn = nn.MultiheadAttention(
                embed_dim=n_emb,
                num_heads=n_head,
                batch_first=True,
            )
            self.shared_stage1_query_norm = nn.LayerNorm(n_emb)

            def _make_shared_stage1_scalar_head(out_dim: int = 1):
                return nn.Sequential(
                    nn.Linear(n_emb, n_emb // 2), nn.SiLU(),
                    nn.Linear(n_emb // 2, out_dim),
                )

            self.shared_stage1_window_head = _make_shared_stage1_scalar_head(out_dim=4)
            self.shared_stage1_dir_head = _make_shared_stage1_scalar_head(out_dim=4)
            self.shared_stage1_decision_phase_head = _make_shared_stage1_scalar_head(out_dim=2)
            self.shared_stage1_control_phase_head = _make_shared_stage1_scalar_head(out_dim=4)
            self.shared_stage1_conflict_area_status_head = _make_shared_stage1_scalar_head(out_dim=4)
            self.shared_stage1_conflict_timing_head = _make_shared_stage1_scalar_head(out_dim=3)
            self.shared_stage1_go_opportunity_head = _make_shared_stage1_scalar_head(out_dim=2)
            self.shared_stage1_temporary_occupancy_head = _make_shared_stage1_scalar_head(out_dim=13)
            self.shared_stage1_merge_yld_max_head = _make_shared_stage1_scalar_head()
            self.shared_stage1_merge_go_min_head = _make_shared_stage1_scalar_head()
            self.shared_stage1_chase_max_head = _make_shared_stage1_scalar_head()
            self.shared_stage1_junction_yld_max_head = _make_shared_stage1_scalar_head()
            self.shared_stage1_junction_go_min_head = _make_shared_stage1_scalar_head()
            self.shared_stage1_borrow_yld_max_head = _make_shared_stage1_scalar_head()
            self.shared_stage1_borrow_go_min_head = _make_shared_stage1_scalar_head()
            self.shared_stage1_conflict_area_head = nn.Sequential(
                nn.Linear(2 * n_emb, n_emb // 2), nn.SiLU(),
                nn.Linear(n_emb // 2, 1),
            )

        # Route head: (B, num_waypoints, n_emb) -> (B, num_waypoints, 2)
        # AdaLN modulation from ego_status for stable closed-loop route prediction
        self.route_head = RouteMLPHead(
            n_emb=n_emb,
            status_dim=status_dim,
            output_dim=2,
            p_drop=p_drop_emb,
        )
        self.route_norm_head = RouteMLPHead(
            n_emb=n_emb,
            status_dim=status_dim,
            output_dim=2,
            p_drop=p_drop_emb,
        )

        # Speed prediction head: two-hot classification over discrete speed bins
        # Input: global-pooled fine BEV (full scene, incl. behind/sides) + conditioning (ego state + route intent)
        # Decoupled from traj decoder output so it can provide complementary speed signal
        self.speed_classes = [0.0, 4.0, 8.0, 10.0, 13.89, 16.0, 17.78, 20.0]
        self.speed_query = nn.Parameter(torch.randn(1, 1, n_emb))
        self.speed_head = nn.Sequential(
            nn.Linear(n_emb * 2, n_emb // 2),  # concat(speed_token, conditioning)
            nn.ReLU(inplace=True),
            nn.Linear(n_emb // 2, len(self.speed_classes)),
        )
        self.speed_profile_head = nn.Sequential(
            nn.Linear(n_emb * 3, n_emb),
            nn.ReLU(inplace=True),
            nn.Linear(n_emb, horizon),
        )

        self.joint_state_window_dim = 4
        self.joint_state_dir_dim = 4
        self.joint_state_decision_dim = 2
        self.joint_state_control_dim = 4
        self.joint_state_area_status_dim = 4
        self.joint_state_conflict_timing_dim = 3
        self.joint_state_boundary_dim = 7
        self.joint_state_speed_dim = len(self.speed_classes)
        self.joint_state_conflict_area_dim = num_waypoints
        self.joint_state_temporary_occupancy_dim = 13
        self.joint_state_token_count = 9
        self.joint_state_token_names = (
            'window',
            'dir',
            'decision_phase',
            'control_phase',
            'area_status',
            'conflict_timing',
            'boundary_bundle',
            'temporary_occupancy',
            'borrow_time',
        )

        def _make_joint_state_proj(in_dim: int):
            return nn.Sequential(
                nn.Linear(in_dim, n_emb),
                nn.SiLU(),
                nn.Linear(n_emb, n_emb),
            )

        self.joint_state_window_proj = _make_joint_state_proj(self.joint_state_window_dim)
        self.joint_state_dir_proj = _make_joint_state_proj(self.joint_state_dir_dim)
        self.joint_state_decision_proj = _make_joint_state_proj(self.joint_state_decision_dim)
        self.joint_state_control_proj = _make_joint_state_proj(self.joint_state_control_dim)
        self.joint_state_area_status_proj = _make_joint_state_proj(self.joint_state_area_status_dim)
        self.joint_state_conflict_timing_proj = _make_joint_state_proj(self.joint_state_conflict_timing_dim)
        self.joint_state_boundary_proj = _make_joint_state_proj(self.joint_state_boundary_dim)
        self.joint_state_speed_proj = _make_joint_state_proj(self.joint_state_speed_dim)
        self.joint_state_conflict_area_proj = _make_joint_state_proj(1)
        self.joint_state_temporary_occupancy_proj = _make_joint_state_proj(self.joint_state_temporary_occupancy_dim)
        self.joint_state_borrow_time_proj = _make_joint_state_proj(1)

        self.joint_state_window_token_emb = nn.Parameter(torch.zeros(1, 1, n_emb))
        self.joint_state_dir_token_emb = nn.Parameter(torch.zeros(1, 1, n_emb))
        self.joint_state_decision_token_emb = nn.Parameter(torch.zeros(1, 1, n_emb))
        self.joint_state_control_token_emb = nn.Parameter(torch.zeros(1, 1, n_emb))
        self.joint_state_area_status_token_emb = nn.Parameter(torch.zeros(1, 1, n_emb))
        self.joint_state_conflict_timing_token_emb = nn.Parameter(torch.zeros(1, 1, n_emb))
        self.joint_state_boundary_token_emb = nn.Parameter(torch.zeros(1, 1, n_emb))
        self.joint_state_temporary_occupancy_token_emb = nn.Parameter(torch.zeros(1, 1, n_emb))
        self.joint_state_borrow_time_token_emb = nn.Parameter(torch.zeros(1, 1, n_emb))

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
            elif 'route_queries' in name or 'pool_query' in name or 'mode_queries' in name or 'route_diff_query' in name or 'speed_query' in name:
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

    def _compute_conditioning(
        self,
        timestep: Union[torch.Tensor, float, int],
        ego_status: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared conditioning builder for split forwards."""
        B = ego_status.shape[0]
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(device)
        timesteps = timestep.expand(B)

        time_emb = self.time_emb(timesteps).to(dtype=model_dtype)
        ego_status_for_cond = self._apply_condition_group_dropout(ego_status)
        current_status = ego_status_for_cond[:, -1, :]
        status_emb = self.ego_status_proj(current_status)
        hist_global_emb = self.history_encoder(ego_status_for_cond)
        conditioning = time_emb + status_emb + hist_global_emb
        route_conditioning = self.route_status_proj(current_status)
        return conditioning, current_status, route_conditioning

    def _apply_condition_group_dropout(self, ego_status: torch.Tensor) -> torch.Tensor:
        """Drop one condition group per sample to reduce shortcut reliance in ego_status."""
        if not (self.training and self.use_condition_group_dropout):
            return ego_status
        if ego_status.shape[-1] < 14:
            raise ValueError(
                "Condition group dropout expects ego_status layout "
                "[speed, theta, command(6), target_point(2), target_point_next(2), waypoints(2)]."
            )

        dropped = ego_status.clone()
        B = dropped.shape[0]
        probs = torch.tensor([0.7, 0.1, 0.1, 0.1], device=dropped.device)
        group_ids = torch.multinomial(probs, num_samples=B, replacement=True)

        speed_theta_mask = group_ids == 1
        target_mask = group_ids == 2
        waypoints_mask = group_ids == 3

        if speed_theta_mask.any():
            dropped[speed_theta_mask, :, 0:2] = 0.0
        if target_mask.any():
            dropped[target_mask, :, 8:12] = 0.0
        if waypoints_mask.any():
            dropped[waypoints_mask, :, 12:14] = 0.0

        return dropped

    def _embed_trajectory(self, traj_abs: torch.Tensor) -> torch.Tensor:
        """Trajectory-level embedding: (B, M, T, 2) -> (B, M, n_emb)."""
        anchor_pos_embed = gen_sineembed_for_position(
            traj_abs, hidden_dim=self.anchor_pos_hidden_dim
        )
        anchor_pos_embed = anchor_pos_embed.flatten(-2)
        return self.anchor_emb(anchor_pos_embed)

    def _embed_waypoint_tokens(self, traj_abs: torch.Tensor) -> torch.Tensor:
        """Waypoint-level embedding for ego path: (B, T, 2) -> (B, T, n_emb)."""
        wp_pos_embed = gen_sineembed_for_position(
            traj_abs, hidden_dim=self.anchor_pos_hidden_dim
        )
        return self.wp_emb(wp_pos_embed)

    def _embed_route_waypoint_tokens(self, route_abs: torch.Tensor) -> torch.Tensor:
        """Waypoint-level embedding for route diffusion path: (B, T_route, 2) -> (B, T_route, n_emb)."""
        route_pos_embed = gen_sineembed_for_position(
            route_abs, hidden_dim=self.anchor_pos_hidden_dim
        )
        return self.route_wp_emb(route_pos_embed)

    def _compute_energy_scores(self, energy_input: torch.Tensor, include_route: bool = True) -> dict:
        energy_scores = {
            'front': self.energy_front_head(energy_input).squeeze(-1),
            'left': self.energy_left_head(energy_input).squeeze(-1),
            'right': self.energy_right_head(energy_input).squeeze(-1),
            'pedestrian': self.energy_pedestrian_head(energy_input).squeeze(-1),
            'offroad': self.energy_offroad_head(energy_input).squeeze(-1),
        }
        if include_route and hasattr(self, 'energy_route_head'):
            energy_scores['route'] = self.energy_route_head(energy_input).squeeze(-1)
        return energy_scores

    def _attn_pool_stage1_tokens(
        self,
        tokens: torch.Tensor,
        summary_query: torch.Tensor,
    ) -> torch.Tensor:
        B = tokens.shape[0]
        query = summary_query.expand(B, -1, -1)
        attn_out, _ = self.shared_stage1_pool_attn(
            query=query,
            key=tokens,
            value=tokens,
            need_weights=False,
        )
        return self.shared_stage1_pool_norm(query + attn_out).squeeze(1)

    def _build_shared_stage1_context(
        self,
        traj_out: torch.Tensor,
        route_out: torch.Tensor,
        speed_out: torch.Tensor,
        route_points: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> dict:
        if speed_out is None:
            raise RuntimeError("shared stage1 context expects a speed token output")
        if route_points.dim() != 3:
            raise ValueError(f"shared stage1 expects route_points as (B, T_route, 2), got {route_points.shape}")

        route_geom = self.shared_stage1_route_geom_proj(route_points.reshape(route_points.shape[0], -1))
        traj_summary = self._attn_pool_stage1_tokens(
            traj_out, self.shared_stage1_traj_summary_query
        )
        route_summary = self._attn_pool_stage1_tokens(
            route_out, self.shared_stage1_route_summary_query
        )
        speed_summary = speed_out.squeeze(1)
        semantic_feature = self.shared_stage1_neck(
            torch.cat(
                [traj_summary, route_summary, speed_summary, route_geom, conditioning],
                dim=-1,
            )
        )
        curve_memory = torch.stack(
            [traj_summary, route_summary, speed_summary, route_geom],
            dim=1,
        )
        route_geom_tokens = route_geom.unsqueeze(1).expand(-1, route_out.shape[1], -1)
        return {
            'traj_summary': traj_summary,
            'route_summary': route_summary,
            'speed_summary': speed_summary,
            'route_geom': route_geom,
            'semantic_feature': semantic_feature,
            'curve_memory': curve_memory,
            'route_geom_tokens': route_geom_tokens,
        }

    def _compute_shared_stage1_scores(
        self,
        traj_out: torch.Tensor,
        route_out: torch.Tensor,
        speed_out: torch.Tensor,
        route_points: torch.Tensor,
        conditioning: torch.Tensor,
        speed_samples: Optional[torch.Tensor] = None,
    ) -> dict:
        context = self._build_shared_stage1_context(
            traj_out=traj_out,
            route_out=route_out,
            speed_out=speed_out,
            route_points=route_points,
            conditioning=conditioning,
        )
        semantic_feature = context['semantic_feature']
        conflict_area_input = torch.cat([route_out, context['route_geom_tokens']], dim=-1)

        return {
            'window_logits': self.shared_stage1_window_head(semantic_feature),
            'dir_logits': self.shared_stage1_dir_head(semantic_feature),
            'decision_phase_logits': self.shared_stage1_decision_phase_head(semantic_feature),
            'control_phase_logits': self.shared_stage1_control_phase_head(semantic_feature),
            'merge_yld_max': self.shared_stage1_merge_yld_max_head(semantic_feature).squeeze(-1),
            'merge_go_min': self.shared_stage1_merge_go_min_head(semantic_feature).squeeze(-1),
            'chase_max': self.shared_stage1_chase_max_head(semantic_feature).squeeze(-1),
            'junction_yld_max': self.shared_stage1_junction_yld_max_head(semantic_feature).squeeze(-1),
            'junction_go_min': self.shared_stage1_junction_go_min_head(semantic_feature).squeeze(-1),
            'borrow_yld_max': self.shared_stage1_borrow_yld_max_head(semantic_feature).squeeze(-1),
            'borrow_go_min': self.shared_stage1_borrow_go_min_head(semantic_feature).squeeze(-1),
            'conflict_area_logits': self.shared_stage1_conflict_area_head(conflict_area_input).squeeze(-1),
        }

    def compute_shared_stage1_from_ego_outputs(
        self,
        traj_out: torch.Tensor,
        route_out: torch.Tensor,
        speed_out: torch.Tensor,
        route_points: torch.Tensor,
        conditioning: torch.Tensor,
        speed_samples: Optional[torch.Tensor] = None,
    ) -> dict:
        return self._compute_shared_stage1_scores(
            traj_out=traj_out,
            route_out=route_out,
            speed_out=speed_out,
            route_points=route_points,
            conditioning=conditioning,
            speed_samples=speed_samples,
        )

    def _build_joint_state_extra_tokens(
        self,
        state_t: dict,
        borrow_time_s: Optional[torch.Tensor],
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        device = conditioning.device
        model_dtype = conditioning.dtype
        B = conditioning.shape[0]

        def _state_value(key: str, dim: int) -> torch.Tensor:
            value = state_t.get(key, None)
            if value is None:
                return torch.zeros(B, dim, device=device, dtype=model_dtype)
            return value.to(device=device, dtype=model_dtype)

        window_token = (
            self.joint_state_window_proj(_state_value('window_logits', self.joint_state_window_dim))
            + self.joint_state_window_token_emb.expand(B, -1, -1).squeeze(1)
            + conditioning
        )
        dir_token = (
            self.joint_state_dir_proj(_state_value('dir_logits', self.joint_state_dir_dim))
            + self.joint_state_dir_token_emb.expand(B, -1, -1).squeeze(1)
            + conditioning
        )
        decision_token = (
            self.joint_state_decision_proj(_state_value('decision_phase_logits', self.joint_state_decision_dim))
            + self.joint_state_decision_token_emb.expand(B, -1, -1).squeeze(1)
            + conditioning
        )
        control_token = (
            self.joint_state_control_proj(_state_value('control_phase_logits', self.joint_state_control_dim))
            + self.joint_state_control_token_emb.expand(B, -1, -1).squeeze(1)
            + conditioning
        )
        area_status_token = (
            self.joint_state_area_status_proj(_state_value('conflict_area_status_logits', self.joint_state_area_status_dim))
            + self.joint_state_area_status_token_emb.expand(B, -1, -1).squeeze(1)
            + conditioning
        )
        conflict_timing_token = (
            self.joint_state_conflict_timing_proj(_state_value('conflict_timing_values', self.joint_state_conflict_timing_dim))
            + self.joint_state_conflict_timing_token_emb.expand(B, -1, -1).squeeze(1)
            + conditioning
        )
        boundary_token = (
            self.joint_state_boundary_proj(_state_value('boundary_values', self.joint_state_boundary_dim))
            + self.joint_state_boundary_token_emb.expand(B, -1, -1).squeeze(1)
            + conditioning
        )
        temporary_occupancy_token = (
            self.joint_state_temporary_occupancy_proj(
                _state_value('temporary_occupancy_logits', self.joint_state_temporary_occupancy_dim)
            )
            + self.joint_state_temporary_occupancy_token_emb.expand(B, -1, -1).squeeze(1)
            + conditioning
        )
        if borrow_time_s is None:
            borrow_time = torch.zeros(B, 1, device=device, dtype=model_dtype)
        else:
            borrow_time = borrow_time_s.to(device=device, dtype=model_dtype).reshape(-1, 1)
        # Conditioning-only side state: borrow_time is allowed to attend with all
        # joint tokens, but we intentionally do not decode/predict it back.
        borrow_token = (
            self.joint_state_borrow_time_proj(borrow_time)
            + self.joint_state_borrow_time_token_emb.expand(B, -1, -1).squeeze(1)
            + conditioning
        )
        tokens = torch.stack(
            [
                window_token,
                dir_token,
                decision_token,
                control_token,
                area_status_token,
                conflict_timing_token,
                boundary_token,
                temporary_occupancy_token,
                borrow_token,
            ],
            dim=1,
        )
        return self.pre_decoder_norm(self.drop(tokens))

    def _predict_joint_state_outputs(
        self,
        *,
        traj_out: torch.Tensor,
        route_out: torch.Tensor,
        speed_out: torch.Tensor,
        route_points: torch.Tensor,
        conditioning: torch.Tensor,
        extra_out: torch.Tensor,
    ) -> dict:
        if extra_out is None or extra_out.shape[1] != self.joint_state_token_count:
            raise ValueError(
                f"joint state decoding expects extra_out as (B, {self.joint_state_token_count}, n_emb), "
                f"got {None if extra_out is None else extra_out.shape}"
            )
        window_token = extra_out[:, 0]
        dir_token = extra_out[:, 1]
        decision_token = extra_out[:, 2]
        control_token = extra_out[:, 3]
        area_status_token = extra_out[:, 4]
        conflict_timing_token = extra_out[:, 5]
        boundary_token = extra_out[:, 6]
        temporary_occupancy_token = extra_out[:, 7]

        route_geom = self.shared_stage1_route_geom_proj(route_points.reshape(route_points.shape[0], -1))
        route_geom_tokens = route_geom.unsqueeze(1).expand(-1, route_out.shape[1], -1)
        conflict_area_input = torch.cat([route_out, route_geom_tokens], dim=-1)
        decision_phase_logits_base = self.shared_stage1_decision_phase_head(decision_token)

        return {
            'window_logits': self.shared_stage1_window_head(window_token),
            'dir_logits': self.shared_stage1_dir_head(dir_token),
            'decision_phase_logits': decision_phase_logits_base,
            'decision_phase_logits_base': decision_phase_logits_base,
            'control_phase_logits': self.shared_stage1_control_phase_head(control_token),
            'conflict_area_status_logits': self.shared_stage1_conflict_area_status_head(area_status_token),
            'conflict_timing_values': self.shared_stage1_conflict_timing_head(conflict_timing_token),
            'go_opportunity_logits': self.shared_stage1_go_opportunity_head(temporary_occupancy_token),
            'temporary_occupancy_logits': self.shared_stage1_temporary_occupancy_head(temporary_occupancy_token),
            'merge_yld_max': self.shared_stage1_merge_yld_max_head(boundary_token).squeeze(-1),
            'merge_go_min': self.shared_stage1_merge_go_min_head(boundary_token).squeeze(-1),
            'chase_max': self.shared_stage1_chase_max_head(boundary_token).squeeze(-1),
            'junction_yld_max': self.shared_stage1_junction_yld_max_head(boundary_token).squeeze(-1),
            'junction_go_min': self.shared_stage1_junction_go_min_head(boundary_token).squeeze(-1),
            'borrow_yld_max': self.shared_stage1_borrow_yld_max_head(boundary_token).squeeze(-1),
            'borrow_go_min': self.shared_stage1_borrow_go_min_head(boundary_token).squeeze(-1),
            'conflict_area_logits': self.shared_stage1_conflict_area_head(conflict_area_input).squeeze(-1),
        }

    def _forward_traj_energy_context(
        self,
        x_t: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        ego_status: torch.Tensor,
        x_t_abs: Optional[torch.Tensor] = None,
        traj_for_energy: Optional[torch.Tensor] = None,
        behavior_labels: Optional[torch.Tensor] = None,
        allowed_flags: Optional[torch.Tensor] = None,
        bev_proj_cached: Optional[torch.Tensor] = None,
        route_points: Optional[torch.Tensor] = None,
        transfuser_lidar_bev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shared trajectory-context builder for energy-style heads."""
        model_dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        x_t = x_t.contiguous().to(device=device, dtype=model_dtype)
        transfuser_bev_feature = transfuser_bev_feature.contiguous().to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = transfuser_bev_feature_upsample.contiguous().to(device=device, dtype=model_dtype)
        ego_status = ego_status.to(device=device, dtype=model_dtype)
        if route_points is not None:
            route_points = route_points.contiguous().to(device=device, dtype=model_dtype)
        if transfuser_lidar_bev is not None:
            transfuser_lidar_bev = transfuser_lidar_bev.contiguous().to(device=device, dtype=model_dtype)
        bev_traj_points = x_t_abs.contiguous().to(device=device, dtype=model_dtype) if x_t_abs is not None else x_t

        B, M, _, _ = bev_traj_points.shape
        conditioning, _, route_conditioning = self._compute_conditioning(
            timestep, ego_status, device, model_dtype
        )

        traj_emb = self._embed_trajectory(bev_traj_points)
        if M == 1:
            mode_queries = self.diff_mode_query.expand(B, -1, -1)
        elif M <= self.mode_queries.shape[1]:
            mode_queries = self.mode_queries[:, :M, :].expand(B, -1, -1)
        elif M == self.mode_queries.shape[1] + 1:
            anchor_queries = self.mode_queries.expand(B, -1, -1)
            gt_queries = self.gt_mode_query.expand(B, -1, -1)
            mode_queries = torch.cat([anchor_queries, gt_queries], dim=1)
        else:
            raise ValueError(f"Unsupported energy mode count M={M}, expected <= {self.mode_queries.shape[1] + 1}")

        mode_emb = traj_emb + mode_queries + conditioning.unsqueeze(1)
        if self.num_behaviors > 0 and behavior_labels is not None and allowed_flags is not None:
            behavior_emb = self.behavior_emb(behavior_labels.to(device))
            allowed_emb = self.allowed_emb(allowed_flags.long().to(device))
            mode_emb = mode_emb + behavior_emb + allowed_emb

        mode_emb = self.pre_decoder_norm(self.drop(mode_emb))
        route_emb = None
        if route_points is not None:
            if route_points.dim() != 3:
                raise ValueError(
                    f"_forward_traj_energy_context expects route_points as (B, T_route, 2), got {route_points.shape}"
                )
            if route_points.shape[1] != self.num_waypoints:
                raise ValueError(
                    f"_forward_traj_energy_context expects T_route={self.num_waypoints}, got {route_points.shape[1]}"
                )
            route_wp_emb = self._embed_route_waypoint_tokens(route_points)
            route_diff_query = self.route_diff_query.expand(B, route_points.shape[1], -1)
            route_emb = route_wp_emb + route_diff_query + conditioning.unsqueeze(1)
            route_emb = self.pre_decoder_norm(self.drop(route_emb))

        mode_out, route_out, _ = self.decoder(
            traj_emb=mode_emb,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            conditioning=conditioning,
            traj_points=bev_traj_points,
            route_emb=route_emb,
            route_points=route_points,
            timesteps=timestep,
            route_conditioning=route_conditioning,
            bev_proj_cached=bev_proj_cached,
            route_pos_offset=M,
            spatial_mode="anchor",
            transfuser_lidar_bev=transfuser_lidar_bev,
        )

        eval_traj = traj_for_energy if traj_for_energy is not None else bev_traj_points
        eval_traj_flat = eval_traj.flatten(-2)
        energy_input = torch.cat([eval_traj_flat, mode_out], dim=-1)
        return energy_input, mode_out, route_out

    def forward_ego(
        self,
        x_t: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        ego_status: torch.Tensor,
        x_t_abs: Optional[torch.Tensor] = None,
        bev_proj_cached: Optional[torch.Tensor] = None,
        transfuser_lidar_bev: Optional[torch.Tensor] = None,
        branch_condition: Optional[torch.Tensor] = None,
        branch_condition_scale: float = 1.0,
        branch_condition_schedule: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        stage1_speed_samples: Optional[torch.Tensor] = None,
    ):
        """
        Ego denoising path with joint trajectory+route waypoint diffusion.

        Returns:
            poses_reg: (B, 1, T, 2) normalized trajectory prediction
            route_pred: (B, num_waypoints, 2) normalized route prediction
            traj_out: (B, T, n_emb) ego trajectory tokens after decoder
            conditioning: (B, n_emb)
            speed_pred: (B, num_speed_classes) logits over speed bins
            speed_profile_pred: (B, T) direct short-horizon speed profile in m/s
        """
        model_dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        x_t = x_t.contiguous().to(device=device, dtype=model_dtype)
        transfuser_bev_feature = transfuser_bev_feature.contiguous().to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = transfuser_bev_feature_upsample.contiguous().to(device=device, dtype=model_dtype)
        ego_status = ego_status.to(device=device, dtype=model_dtype)

        joint_points = x_t_abs.contiguous().to(device=device, dtype=model_dtype) if x_t_abs is not None else x_t
        if joint_points.dim() == 4:
            if joint_points.shape[1] != 1:
                raise ValueError(f"forward_ego expects a single joint ego sample, got shape {joint_points.shape}")
            joint_points = joint_points[:, 0, :, :]
        if joint_points.dim() != 3:
            raise ValueError(f"forward_ego expects (B, 1, T_joint, 2) or (B, T_joint, 2), got {joint_points.shape}")
        if joint_points.shape[1] != self.ego_joint_horizon:
            raise ValueError(
                f"forward_ego expects T_joint={self.ego_joint_horizon}, got {joint_points.shape[1]}"
            )

        traj_points = joint_points[:, :self.horizon, :]
        route_points = joint_points[:, self.horizon:self.ego_joint_horizon, :]

        B, T_traj, _ = traj_points.shape
        T_route = route_points.shape[1]
        conditioning, current_status, route_conditioning = self._compute_conditioning(
            timestep, ego_status, device, model_dtype
        )

        wp_emb = self._embed_waypoint_tokens(traj_points)
        diff_query = self.diff_mode_query.expand(B, T_traj, -1)
        traj_emb = wp_emb + diff_query + conditioning.unsqueeze(1)
        branch_cond_emb = None
        if branch_condition is not None:
            branch_condition = branch_condition.to(device=device, dtype=model_dtype)
            if branch_condition.dim() != 2 or branch_condition.shape[-1] != self.traj_branch_condition_dim:
                raise ValueError(
                    "forward_ego expects branch_condition as "
                    f"(B, {self.traj_branch_condition_dim}), got {branch_condition.shape}"
                )
            if branch_condition_schedule is None:
                branch_condition_schedule = torch.ones(B, 5, device=device, dtype=model_dtype)
            else:
                branch_condition_schedule = branch_condition_schedule.to(device=device, dtype=model_dtype)
                if branch_condition_schedule.dim() != 2 or branch_condition_schedule.shape[-1] != 5:
                    raise ValueError(
                        "forward_ego expects branch_condition_schedule as "
                        f"(B, 5), got {branch_condition_schedule.shape}"
                    )
            window_cond = branch_condition[:, :self.traj_window_condition_dim]
            dir_start = self.traj_window_condition_dim
            dir_end = dir_start + self.traj_dir_condition_dim
            dir_cond = branch_condition[:, dir_start:dir_end]
            decision_start = dir_end
            decision_end = decision_start + self.traj_decision_phase_condition_dim
            decision_cond = branch_condition[:, decision_start:decision_end]
            control_start = decision_end
            control_end = control_start + self.traj_control_phase_condition_dim
            control_cond = branch_condition[:, control_start:control_end]
            boundary_start = control_end
            boundary_end = boundary_start + self.traj_boundary_margin_dim
            boundary_cond = branch_condition[:, boundary_start:boundary_end]
            borrow_aux = branch_condition[:, boundary_end:]
            gate_window = branch_condition_schedule[:, 0:1]
            gate_dir = branch_condition_schedule[:, 1:2]
            gate_phase = branch_condition_schedule[:, 2:3]
            gate_boundary = branch_condition_schedule[:, 3:4]
            gate_borrow = branch_condition_schedule[:, 4:5]
            branch_cond_emb = (
                self.traj_window_condition_proj(window_cond) * gate_window
                + self.traj_dir_condition_proj(dir_cond) * gate_dir
                + self.traj_decision_phase_condition_proj(decision_cond) * gate_phase
                + self.traj_control_phase_condition_proj(control_cond) * gate_phase
                + self.traj_boundary_margin_proj(boundary_cond) * gate_boundary
                + self.traj_borrow_aux_proj(borrow_aux) * gate_borrow
            ) * float(branch_condition_scale)
            traj_emb = traj_emb + branch_cond_emb.unsqueeze(1)
        traj_emb = self.pre_decoder_norm(self.drop(traj_emb))

        speed_emb = self.speed_query.expand(B, -1, -1) + conditioning.unsqueeze(1)
        if branch_cond_emb is not None:
            speed_emb = speed_emb + branch_cond_emb.unsqueeze(1)
        speed_emb = self.pre_decoder_norm(self.drop(speed_emb))

        route_wp_emb = self._embed_route_waypoint_tokens(route_points)
        route_diff_query = self.route_diff_query.expand(B, T_route, -1)
        route_emb = route_wp_emb + route_diff_query + conditioning.unsqueeze(1)
        route_emb = self.pre_decoder_norm(self.drop(route_emb))

        ego_mask = self.decoder._create_ego_speed_mask(
            T_traj=T_traj,
            T_route=T_route,
            device=traj_emb.device,
            dtype=traj_emb.dtype,
        )
        traj_out, route_out, speed_out = self.decoder(
            speed_emb=speed_emb,
            traj_emb=traj_emb,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            conditioning=conditioning,
            traj_points=traj_points,
            route_emb=route_emb,
            route_points=route_points,
            timesteps=timestep,
            route_conditioning=route_conditioning,
            bev_proj_cached=bev_proj_cached,
            self_attn_mask=ego_mask,
            route_pos_offset=T_traj,
            spatial_mode="ego",
            transfuser_lidar_bev=transfuser_lidar_bev,
        )
        if speed_out is None:
            raise RuntimeError("Decoder ego path expected a speed token output")

        traj_pred = self.trajectory_wp_head(traj_out, conditioning, route_features=route_out)
        poses_reg = traj_pred.unsqueeze(1)
        route_pred = self.route_norm_head(route_out, conditioning, current_status)
        speed_input = torch.cat([
            speed_out.squeeze(1),  # shared token after interacting with traj/route context
            conditioning,
        ], dim=-1)
        speed_pred = self.speed_head(speed_input)  # (B, num_speed_classes)
        speed_profile_input = torch.cat([
            speed_out.squeeze(1),
            traj_out.mean(dim=1),
            conditioning,
        ], dim=-1)
        speed_profile_pred = self.speed_profile_head(speed_profile_input)
        if return_intermediates:
            result = {
                'poses_reg': poses_reg,
                'route_pred': route_pred,
                'traj_out': traj_out,
                'route_out': route_out,
                'speed_out': speed_out,
                'route_points': route_points,
                'conditioning': conditioning,
                'speed_pred': speed_pred,
                'speed_profile_pred': speed_profile_pred,
            }
            if stage1_speed_samples is not None:
                stage1_speed_samples = stage1_speed_samples.to(device=device, dtype=model_dtype)
                result['stage1_raw_scores'] = self._compute_shared_stage1_scores(
                    traj_out=traj_out,
                    route_out=route_out,
                    speed_out=speed_out,
                    route_points=route_points,
                    conditioning=conditioning,
                    speed_samples=stage1_speed_samples,
                )
            return result

        return poses_reg, route_pred, traj_out, conditioning, speed_pred, speed_profile_pred

    def forward_ego_joint(
        self,
        x_t: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        ego_status: torch.Tensor,
        state_t: dict,
        borrow_time_s: Optional[torch.Tensor] = None,
        x_t_abs: Optional[torch.Tensor] = None,
        bev_proj_cached: Optional[torch.Tensor] = None,
        transfuser_lidar_bev: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ):
        model_dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        x_t = x_t.contiguous().to(device=device, dtype=model_dtype)
        transfuser_bev_feature = transfuser_bev_feature.contiguous().to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = transfuser_bev_feature_upsample.contiguous().to(device=device, dtype=model_dtype)
        ego_status = ego_status.to(device=device, dtype=model_dtype)

        joint_points = x_t_abs.contiguous().to(device=device, dtype=model_dtype) if x_t_abs is not None else x_t
        if joint_points.dim() == 4:
            if joint_points.shape[1] != 1:
                raise ValueError(f"forward_ego_joint expects a single joint ego sample, got shape {joint_points.shape}")
            joint_points = joint_points[:, 0, :, :]
        if joint_points.dim() != 3:
            raise ValueError(f"forward_ego_joint expects (B, 1, T_joint, 2) or (B, T_joint, 2), got {joint_points.shape}")
        if joint_points.shape[1] != self.ego_joint_horizon:
            raise ValueError(
                f"forward_ego_joint expects T_joint={self.ego_joint_horizon}, got {joint_points.shape[1]}"
            )

        traj_points = joint_points[:, :self.horizon, :]
        route_points = joint_points[:, self.horizon:self.ego_joint_horizon, :]
        B, T_traj, _ = traj_points.shape
        T_route = route_points.shape[1]

        conditioning, current_status, route_conditioning = self._compute_conditioning(
            timestep, ego_status, device, model_dtype
        )

        wp_emb = self._embed_waypoint_tokens(traj_points)
        diff_query = self.diff_mode_query.expand(B, T_traj, -1)
        traj_emb = self.pre_decoder_norm(self.drop(wp_emb + diff_query + conditioning.unsqueeze(1)))

        route_wp_emb = self._embed_route_waypoint_tokens(route_points)
        route_diff_query = self.route_diff_query.expand(B, T_route, -1)
        conflict_area_logits = state_t.get('conflict_area_logits', None)
        if conflict_area_logits is None:
            conflict_area_logits = torch.zeros(B, T_route, device=device, dtype=model_dtype)
        else:
            conflict_area_logits = conflict_area_logits.to(device=device, dtype=model_dtype)
            if conflict_area_logits.dim() != 2 or conflict_area_logits.shape[1] != T_route:
                raise ValueError(
                    f"forward_ego_joint expects conflict_area_logits as (B, {T_route}), got {conflict_area_logits.shape}"
                )
        route_conflict_emb = self.joint_state_conflict_area_proj(conflict_area_logits.unsqueeze(-1))
        route_emb = self.pre_decoder_norm(
            self.drop(route_wp_emb + route_diff_query + route_conflict_emb + conditioning.unsqueeze(1))
        )

        speed_logits = state_t.get('speed_logits', None)
        if speed_logits is None:
            speed_logits = torch.zeros(B, self.joint_state_speed_dim, device=device, dtype=model_dtype)
        else:
            speed_logits = speed_logits.to(device=device, dtype=model_dtype)
            if speed_logits.dim() != 2 or speed_logits.shape[1] != self.joint_state_speed_dim:
                raise ValueError(
                    f"forward_ego_joint expects speed_logits as (B, {self.joint_state_speed_dim}), got {speed_logits.shape}"
                )
        speed_emb = self.pre_decoder_norm(
            self.drop(
                self.joint_state_speed_proj(speed_logits).unsqueeze(1)
                + conditioning.unsqueeze(1)
            )
        )

        extra_tokens = self._build_joint_state_extra_tokens(
            state_t=state_t,
            borrow_time_s=borrow_time_s,
            conditioning=conditioning,
        )
        full_mask = self.decoder._create_full_mask(
            total_tokens=1 + T_traj + T_route + extra_tokens.shape[1],
            device=device,
            dtype=model_dtype,
        )
        traj_out, route_out, speed_out, extra_out = self.decoder(
            speed_emb=speed_emb,
            extra_emb=extra_tokens,
            traj_emb=traj_emb,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            conditioning=conditioning,
            traj_points=traj_points,
            route_emb=route_emb,
            route_points=route_points,
            timesteps=timestep,
            route_conditioning=route_conditioning,
            bev_proj_cached=bev_proj_cached,
            self_attn_mask=full_mask,
            route_pos_offset=T_traj,
            spatial_mode="ego",
            transfuser_lidar_bev=transfuser_lidar_bev,
        )
        if speed_out is None:
            raise RuntimeError("forward_ego_joint expected a speed token output")

        traj_pred = self.trajectory_wp_head(traj_out, conditioning, route_features=route_out)
        poses_reg = traj_pred.unsqueeze(1)
        route_pred = self.route_norm_head(route_out, conditioning, current_status)
        speed_input = torch.cat([
            speed_out.squeeze(1),
            conditioning,
        ], dim=-1)
        speed_pred = self.speed_head(speed_input)
        speed_profile_input = torch.cat([
            speed_out.squeeze(1),
            traj_out.mean(dim=1),
            conditioning,
        ], dim=-1)
        speed_profile_pred = self.speed_profile_head(speed_profile_input)
        state_pred_dict = self._predict_joint_state_outputs(
            traj_out=traj_out,
            route_out=route_out,
            speed_out=speed_out,
            route_points=route_points,
            conditioning=conditioning,
            extra_out=extra_out,
        )
        state_pred_dict['speed_logits'] = speed_pred
        if return_intermediates:
            return {
                'poses_reg': poses_reg,
                'route_pred': route_pred,
                'traj_out': traj_out,
                'route_out': route_out,
                'speed_out': speed_out,
                'extra_out': extra_out,
                'route_points': route_points,
                'conditioning': conditioning,
                'speed_pred': speed_pred,
                'speed_profile_pred': speed_profile_pred,
                'state_pred_dict': state_pred_dict,
            }
        return poses_reg, route_pred, traj_out, conditioning, speed_pred, speed_profile_pred, state_pred_dict

    def forward_energy(
        self,
        x_t: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        ego_status: torch.Tensor,
        x_t_abs: Optional[torch.Tensor] = None,
        traj_for_energy: Optional[torch.Tensor] = None,
        behavior_labels: Optional[torch.Tensor] = None,
        allowed_flags: Optional[torch.Tensor] = None,
        bev_proj_cached: Optional[torch.Tensor] = None,
        route_points: Optional[torch.Tensor] = None,
        transfuser_lidar_bev: Optional[torch.Tensor] = None,
    ) -> Tuple[dict, torch.Tensor]:
        """
        Trajectory-level energy path. Anchors/GT remain 1 token per trajectory.
        Route is context only; route-token energies are not produced here.
        """
        energy_input, mode_out, _ = self._forward_traj_energy_context(
            x_t=x_t,
            x_t_abs=x_t_abs,
            timestep=timestep,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
            traj_for_energy=traj_for_energy,
            behavior_labels=behavior_labels,
            allowed_flags=allowed_flags,
            bev_proj_cached=bev_proj_cached,
            route_points=route_points,
            transfuser_lidar_bev=transfuser_lidar_bev,
        )
        return self._compute_energy_scores(energy_input, include_route=False), mode_out

    def forward_front_route_risk(
        self,
        x_t: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        ego_status: torch.Tensor,
        x_t_abs: Optional[torch.Tensor] = None,
        traj_for_energy: Optional[torch.Tensor] = None,
        bev_proj_cached: Optional[torch.Tensor] = None,
        route_points: Optional[torch.Tensor] = None,
        transfuser_lidar_bev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Independent route-conditioned front-risk path for GT/pred_x0 evaluation."""
        energy_input, mode_out, _ = self._forward_traj_energy_context(
            x_t=x_t,
            x_t_abs=x_t_abs,
            timestep=timestep,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
            traj_for_energy=traj_for_energy,
            bev_proj_cached=bev_proj_cached,
            route_points=route_points,
            transfuser_lidar_bev=transfuser_lidar_bev,
        )
        return self.front_route_risk_head(energy_input).squeeze(-1), mode_out

    def forward_energy_eval(
        self,
        x_t: torch.Tensor,
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        ego_status: torch.Tensor,
        x_t_abs: Optional[torch.Tensor] = None,
        traj_for_energy: Optional[torch.Tensor] = None,
        bev_proj_cached: Optional[torch.Tensor] = None,
        route_points: Optional[torch.Tensor] = None,
        transfuser_lidar_bev: Optional[torch.Tensor] = None,
    ) -> Tuple[dict, torch.Tensor]:
        """
        Guidance/alignment energy evaluation on a single predicted trajectory.
        """
        B = x_t.shape[0]
        timestep = torch.zeros(B, dtype=torch.long, device=x_t.device)
        return self.forward_energy(
            x_t=x_t,
            x_t_abs=x_t_abs,
            timestep=timestep,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
            traj_for_energy=traj_for_energy,
            bev_proj_cached=bev_proj_cached,
            route_points=route_points,
            transfuser_lidar_bev=transfuser_lidar_bev,
        )

    def forward_front_route_risk_eval(
        self,
        x_t: torch.Tensor,
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        ego_status: torch.Tensor,
        x_t_abs: Optional[torch.Tensor] = None,
        traj_for_energy: Optional[torch.Tensor] = None,
        bev_proj_cached: Optional[torch.Tensor] = None,
        route_points: Optional[torch.Tensor] = None,
        transfuser_lidar_bev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Guidance/alignment evaluation for the independent front-route-risk head."""
        B = x_t.shape[0]
        timestep = torch.zeros(B, dtype=torch.long, device=x_t.device)
        return self.forward_front_route_risk(
            x_t=x_t,
            x_t_abs=x_t_abs,
            timestep=timestep,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
            traj_for_energy=traj_for_energy,
            bev_proj_cached=bev_proj_cached,
            route_points=route_points,
            transfuser_lidar_bev=transfuser_lidar_bev,
        )
    
    def forward(
        self,
        x_t: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        ego_status: torch.Tensor,
        x_t_abs: Optional[torch.Tensor] = None,
        behavior_labels: torch.Tensor = None,
        allowed_flags: torch.Tensor = None,
        traj_for_energy: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Multimodal forward pass for trajectory prediction.

        Args:
            x_t: (B, num_modes, anchor_num_points, 2) - current denoising trajectory in normalized space
            timestep: diffusion timestep (for conditioning, can be 0 at inference)
            transfuser_bev_feature: (B, 1512, 8, 8) - BEV feature from transfuser
            transfuser_bev_feature_upsample: (B, 64, 64, 64) - Upsampled BEV for spatial attention
            ego_status: (B, T_obs, status_dim) - ego status history
            x_t_abs: (B, num_modes, anchor_num_points, 2) - current denoising trajectory in absolute coords for BEV grid_sample.
                     If None, uses `x_t` directly (backward compatible).
            traj_for_energy: (B, M, T, 2) optional - trajectory for energy head evaluation.
                     Training: original anchor coords (labels match these, not model output).
                     Inference: pred_x0 from first forward pass (gradient flows back for guidance).
                     If None, uses poses_reg (model output).

        Returns:
            poses_reg: (B, num_modes, horizon, 2) - trajectory predictions for each mode.
                      Output space follows residual base:
                      - absolute space if x_t_abs is provided
                      - normalized space otherwise (backward compatibility)
            poses_cls: (B, num_modes) - classification logits for mode selection
            route_pred: (B, num_waypoints, 2) - route prediction
        """
        model_dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        x_t = x_t.contiguous().to(device=device, dtype=model_dtype)
        transfuser_bev_feature = transfuser_bev_feature.contiguous().to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = transfuser_bev_feature_upsample.contiguous().to(device=device, dtype=model_dtype)
        ego_status = ego_status.to(device=device, dtype=model_dtype)

        # BEV sampling uses absolute coords; fallback to x_t for backward compat
        bev_traj_points = x_t_abs.contiguous().to(device=device, dtype=model_dtype) if x_t_abs is not None else x_t

        B = x_t.shape[0]
        num_modes = x_t.shape[1]
        anchor_num_points = x_t.shape[2]
        
        # ========== Timestep handling ==========
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(device)
        timesteps = timestep.expand(B)
        
        # ========== Conditioning ==========
        conditioning, current_status, route_conditioning = self._compute_conditioning(
            timestep, ego_status, device, model_dtype
        )
        
        # ========== Anchor Embedding ==========
        # Encode full trajectory geometry for each mode:
        # (B, M, T, 2) -> sine embed (B, M, T, 64) -> flatten (B, M, T*64) -> (B, M, n_emb)
        anchor_pos_embed = gen_sineembed_for_position(
            bev_traj_points, hidden_dim=self.anchor_pos_hidden_dim
        )
        anchor_pos_embed = anchor_pos_embed.flatten(-2)  # (B, M, T * 64)
        anchor_emb = self.anchor_emb(anchor_pos_embed.to(dtype=model_dtype))

        # Add learnable mode queries (select based on input M)
        M = anchor_emb.shape[1]
        M_anchor = self.mode_queries.shape[1]  # num_energy_modes (e.g. 32)
        if M == 1 and self.anchor_free:
            # Single-mode diffusion denoising: use dedicated diff_mode_query
            mode_queries = self.diff_mode_query.expand(B, -1, -1)
        elif self.anchor_free and M == M_anchor + 2:
            # Unified training: M = 1 (x_t) + M_anchor (32) + 1 (GT) = 34
            # Build each block separately to preserve native parameter strides for DDP.
            diff_queries = self.diff_mode_query.expand(B, -1, -1)
            anchor_queries = self.mode_queries.expand(B, -1, -1)
            gt_queries = self.gt_mode_query.expand(B, -1, -1)
            mode_queries = torch.cat([diff_queries, anchor_queries, gt_queries], dim=1)
        elif M > M_anchor:
            if M != M_anchor + 1:
                raise ValueError(f"Unsupported mode count M={M}, expected <= {M_anchor + 2}")
            # VLM anchor added: concatenate vqa_mode_query for the extra mode.
            anchor_queries = self.mode_queries.expand(B, -1, -1)
            vqa_queries = self.vqa_mode_query.expand(B, -1, -1)
            mode_queries = torch.cat([anchor_queries, vqa_queries], dim=1)
        else:
            mode_queries = self.mode_queries[:, :M, :].expand(B, -1, -1)

        # Combine: anchor embedding + mode queries + conditioning
        mode_emb = anchor_emb + mode_queries + conditioning.unsqueeze(1)  # (B, num_modes, n_emb)

        # Add semantic behavior conditioning (if available)
        if self.num_behaviors > 0 and behavior_labels is not None:
            behavior_emb = self.behavior_emb(behavior_labels.to(device))  # (B, num_modes, n_emb)
            allowed_emb = self.allowed_emb(allowed_flags.long().to(device))  # (B, num_modes, n_emb)
            mode_emb = mode_emb + behavior_emb + allowed_emb

        mode_emb = self.drop(mode_emb)
        mode_emb = self.pre_decoder_norm(mode_emb)

        # ========== UnifiedDecoderOnlyTransformer ==========
        # traj_emb = mode_emb (B, num_modes, n_emb) - each mode is one "trajectory query"
        # traj_points = bev_traj_points (B, num_modes, horizon, 2) - each mode samples BEV at its waypoints
        # The decoder internally builds route_queries and returns (mode_out, route_out)
        mode_out, route_out = self.decoder(
            traj_emb=mode_emb,
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            conditioning=conditioning,
            traj_points=bev_traj_points,  # (B, num_modes, horizon, 2) absolute coords for grid_sample
            route_conditioning=route_conditioning,
        )
        # mode_out: (B, num_modes, n_emb), route_out: (B, num_waypoints, n_emb)
        
        # ========== Output Heads ==========
        # 1. Trajectory regression: (B, num_modes, n_emb) -> (B, num_modes, horizon * 2)
        traj_flat = self.trajectory_head(mode_out, conditioning, route_features=route_out)  # (B, num_modes, horizon * output_dim)
        poses_reg = traj_flat.view(B, num_modes, self.horizon, self.output_dim)  # (B, num_modes, horizon, 2)
        
        assert anchor_num_points == self.horizon, \
            f"anchor_num_points ({anchor_num_points}) must equal horizon ({self.horizon})."

        # Add anchor as residual (skip in anchor-free mode where model predicts absolute)
        if not self.anchor_free:
            residual_base = bev_traj_points
            poses_reg = poses_reg + residual_base
        
        # 2. Classification: (B, num_modes, n_emb) -> (B, num_modes, 1) -> (B, num_modes)
        poses_cls = self.cls_head(mode_out).squeeze(-1)  # (B, num_modes)

        # 3. Route prediction from unified decoder output
        route_pred = self.route_head(route_out, conditioning, current_status)  # (B, num_waypoints, 2)

        # 4. Energy scores (Route B: evaluate trajectory + scene context)
        #    traj_for_energy: the trajectory to evaluate (anchor coords for training, pred_x0 for inference)
        #    mode_out: scene context from BEV attention (computed from the evaluated trajectory's path)
        #    At inference, gradient of energy w.r.t. traj_for_energy flows back for guidance.
        if self.energy_heads_enabled:
            # Use external trajectory if provided (training: original anchors), else model output (inference)
            eval_traj = traj_for_energy if traj_for_energy is not None else poses_reg
            eval_traj_flat = eval_traj.flatten(-2)  # (B, M, horizon * 2)
            energy_input = torch.cat([eval_traj_flat, mode_out], dim=-1)  # (B, M, T*2 + n_emb)
            energy_scores = {
                'front':      self.energy_front_head(energy_input).squeeze(-1),      # (B, M) vehicle front
                'left':       self.energy_left_head(energy_input).squeeze(-1),       # (B, M) vehicle left
                'right':      self.energy_right_head(energy_input).squeeze(-1),      # (B, M) vehicle right
                'pedestrian': self.energy_pedestrian_head(energy_input).squeeze(-1), # (B, M) pedestrian
                'offroad':    self.energy_offroad_head(energy_input).squeeze(-1),    # (B, M) offroad
                'route':      self.energy_route_head(energy_input).squeeze(-1),      # (B, M) route deviation
            }
            return poses_reg, poses_cls, route_pred, mode_out, energy_scores

        return poses_reg, poses_cls, route_pred, mode_out


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
    x_t = torch.randn((B, 32, 8, 2))  # (B, num_modes, anchor_num_points=horizon, 2)

    # Transfuser features (following DiffusionDriveV2: only bev_feature and bev_feature_upsample)
    transfuser_bev_feature = torch.randn((B, 1512, 8, 8))
    transfuser_bev_feature_upsample = torch.randn((B, 64, 64, 64))

    ego_status = torch.randn((B, 4, 14))  # 4 frames of history with updated dim

    print("\nTest 1: Basic forward pass (multimodal)")
    poses_reg, poses_cls, route_pred, mode_out = transformer(
        x_t=x_t, timestep=timestep,
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
        x_t=x_t, timestep=timestep,
        transfuser_bev_feature=transfuser_bev_feature2,
        transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
        ego_status=ego_status,
    )
    diff = torch.abs(poses_reg - poses_reg2).mean()
    print(f"  Difference: {diff:.6f}")
    assert diff > 0, "BEV features should affect output"

    print("\nTest 3: Different x_t affect output")
    x_t2 = torch.randn((B, 32, 8, 2))
    poses_reg3, _, _ = transformer(
        x_t=x_t2, timestep=timestep,
        transfuser_bev_feature=transfuser_bev_feature,
        transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
        ego_status=ego_status,
    )
    diff_x_t = torch.abs(poses_reg - poses_reg3).mean()
    print(f"  Difference: {diff_x_t:.6f}")
    assert diff_x_t > 0, "x_t should affect output"
    
    print("\nTest 4: Optimizer")
    opt = transformer.configure_optimizers()
    print(f"  Optimizer created: {type(opt).__name__}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nArchitecture (Multimodal DiffusionDrive style):")
    print("  1. Input: x_t (B, num_modes, anchor_points, 2)")
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
