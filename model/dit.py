"""Simplified DiT decoder for nuPlan diffusion.

Adapted from Diffusion-Planner's DiT architecture:
- adaLN-Zero conditioning: timestep + route encoding modulate self-attention/MLP
- Cross-attention: attend to encoded scene tokens
- Joint prediction: ego + N neighbors

Design notes:
- Uses diffusers.DDIMScheduler for training/inference (no SDE)
- Model predicts x_start (clean trajectory) directly
- No guidance wrapper needed — pure white-noise diffusion
"""

import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp

from model.adapter_layer import AdapterLayer
from model.scene_encoder import MixerBlock


# =============================================================================
# Conditioning helpers
# =============================================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN-Zero modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# =============================================================================
# Timestep Embedder
# =============================================================================

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding -> MLP -> conditioning vector."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# =============================================================================
# Route Encoder
# =============================================================================

class RouteEncoder(nn.Module):
    """Encode route lanes into a global conditioning vector."""

    def __init__(
        self,
        route_num: int = 10,
        lane_len: int = 20,
        hidden_dim: int = 192,
        tokens_mlp_dim: int = 32,
        channels_mlp_dim: int = 64,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(
            in_features=4,  # x, y, dx, dy
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=route_num * lane_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, route_num, lane_len, 4) = [x, y, dx, dy]
        Returns: (B, hidden_dim)
        """
        B, P, V, _ = x.shape

        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0  # (B, P, V)
        mask_p = torch.sum(~mask_v, dim=-1) == 0  # (B, P)
        mask_b = torch.sum(~mask_p, dim=-1) == 0  # (B,)

        x = x.view(B, P * V, -1)
        valid_indices = ~mask_b.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.mixer(x)
        x = torch.mean(x, dim=1)
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B, x.shape[-1]), device=x.device, dtype=x.dtype)
        x_result[valid_indices] = x

        return x_result.view(B, -1)


# =============================================================================
# DiT Block (with adaLN-Zero + Cross-Attention)
# =============================================================================

class DiTBlock(nn.Module):
    """A single DiT block with:
    - adaLN-Zero modulated self-attention + MLP
    - Cross-attention to scene context
    - Second MLP after cross-attention
    """

    def __init__(
        self,
        dim: int = 192,
        heads: int = 6,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        cross_c: torch.Tensor,
        y: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        # adaLN-Zero: chunk conditioning into 6 parts
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(y).chunk(6, dim=1)
        )

        # Self-attention with adaLN-Zero
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulated_x, modulated_x, modulated_x, key_padding_mask=attn_mask
        )[0]
        adapter_self_attn = getattr(self, 'adapter_self_attn', None)
        if adapter_self_attn is not None:
            x = adapter_self_attn(x)

        # MLP with adaLN-Zero
        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)
        adapter_mlp1 = getattr(self, 'adapter_mlp1', None)
        if adapter_mlp1 is not None:
            x = adapter_mlp1(x)

        # Cross-attention to scene context
        x = x + self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        adapter_cross_attn = getattr(self, 'adapter_cross_attn', None)
        if adapter_cross_attn is not None:
            x = adapter_cross_attn(x)

        # Final MLP
        x = x + self.mlp2(self.norm4(x))
        adapter_mlp2 = getattr(self, 'adapter_mlp2', None)
        if adapter_mlp2 is not None:
            x = adapter_mlp2(x)

        return x


# =============================================================================
# Final Layer (with adaLN-Zero)
# =============================================================================

class FinalLayer(nn.Module):
    """Final projection layer with adaLN-Zero conditioning."""

    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x


# =============================================================================
# DiT (Diffusion Transformer)
# =============================================================================

class DiT(nn.Module):
    """Diffusion Transformer for nuPlan trajectory prediction.

    Jointly predicts ego + neighbor trajectories from pure noise,
    conditioned on scene encoding and route information.
    """

    def __init__(
        self,
        hidden_dim: int = 192,
        heads: int = 6,
        depth: int = 3,
        output_dim: int = 324,  # (future_len + 1) * 4
        route_num: int = 10,
        lane_len: int = 20,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection: flatten trajectory -> hidden_dim
        self.preproj = Mlp(
            in_features=output_dim,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        # Agent type embedding: ego=0, neighbor=1
        self.agent_embedding = nn.Embedding(2, hidden_dim)

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(hidden_dim)

        # Route encoder for global conditioning
        self.route_encoder = RouteEncoder(
            route_num=route_num, lane_len=lane_len, hidden_dim=hidden_dim
        )

        # DiT blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for _ in range(depth)]
        )

        # Final projection
        self.final_layer = FinalLayer(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        scene_encoding: torch.Tensor,
        route_lanes: torch.Tensor,
        neighbor_current_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, P, output_dim) - noisy trajectories (flattened current+future)
            t: (B,) - diffusion timestep (0-999 for DDIM, scaled to [0,1])
            scene_encoding: (B, N_scene, hidden_dim) - scene context tokens
            route_lanes: (B, route_num, lane_len, 4) - route lane geometry
            neighbor_current_mask: (B, P-1) - True for PADDED neighbors

        Returns:
            (B, P, output_dim) - predicted clean trajectories (x_start)
        """
        B, P, _ = x.shape

        # Input projection
        x = self.preproj(x)  # (B, P, hidden_dim)

        # Agent type embedding: ego (idx=0) + neighbors (idx=1)
        agent_emb = torch.cat(
            [
                self.agent_embedding.weight[0][None, :],  # (1, D) ego
                self.agent_embedding.weight[1][None, :].expand(P - 1, -1),  # (P-1, D) neighbors
            ],
            dim=0,
        )  # (P, D)
        x = x + agent_emb[None, :, :]  # (B, P, D)

        # Global conditioning: route + timestep
        route_encoding = self.route_encoder(route_lanes)  # (B, D)
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(B)
        y = route_encoding + self.t_embedder(t)  # (B, D)

        # Attention mask: ego always visible, neighbors masked for padding
        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:, 1:] = neighbor_current_mask  # (B, P)

        # DiT blocks
        for block in self.blocks:
            x = block(x, scene_encoding, y, attn_mask)

        # Final projection
        x = self.final_layer(x, y)  # (B, P, output_dim)

        return x
