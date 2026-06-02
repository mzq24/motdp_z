"""Vector-based Scene Encoder for nuPlan.

Adopted from Diffusion-Planner's encoder architecture:
- AgentFusionEncoder: MLP-Mixer for neighbor agent trajectories
- LaneFusionEncoder: MLP-Mixer for vector map lanes
- StaticFusionEncoder: MLP projection for static objects
- FusionEncoder: self-attention over all scene tokens

All inputs are in ego-centric coordinates (x, y, cos_heading, sin_heading, ...).
"""

import torch
import torch.nn as nn
from timm.models.layers import Mlp


# =============================================================================
# Mixer Block
# =============================================================================

class MixerBlock(nn.Module):
    """MLP-Mixer block: channel mixing + token mixing."""

    def __init__(self, tokens_mlp_dim: int, channels_mlp_dim: int, drop_path_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(
            in_features=tokens_mlp_dim,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = y.permute(0, 2, 1)
        y = self.tokens_mlp(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.norm2(x)
        return x + self.channels_mlp(y)


# =============================================================================
# Self-Attention Block (for fusion)
# =============================================================================

class SelfAttentionBlock(nn.Module):
    """LayerNorm -> MultiheadAttention -> DropPath -> LayerNorm -> MLP."""

    def __init__(self, dim: int = 192, heads: int = 6, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=nn.GELU,
            drop=dropout,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# Agent Fusion Encoder
# =============================================================================

class AgentFusionEncoder(nn.Module):
    """Encodes neighbor agent past trajectories using MLP-Mixer.

    Input: (B, num_agents, time_len, D)
      where D = 11: [x, y, cos, sin, vx, vy, w, l, type_vehicle, type_ped, type_bike]
    Output: (B, num_agents, hidden_dim)
    """

    def __init__(
        self,
        time_len: int,
        hidden_dim: int = 192,
        depth: int = 2,
        tokens_mlp_dim: int = 64,
        channels_mlp_dim: int = 128,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.type_emb = nn.Linear(3, channels_mlp_dim)
        self.channel_pre_project = Mlp(
            in_features=8 + 1,  # 8 features + 1 validity mask
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=time_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.blocks = nn.ModuleList(
            [MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, P, V, D) = [x, y, cos, sin, vx, vy, w, l, type_veh, type_ped, type_bike]
        """
        B, P, V, _ = x.shape

        # Save type for later
        neighbor_type = x[:, :, -1, 8:]  # (B, P, 3)

        # Strip type, keep kinematics only
        x = x[..., :8]  # (B, P, V, 8)

        # Position for spatial embedding: x, y, cos, sin, onehot(neighbor=1,0,0)
        pos = x[:, :, -1, :7].clone()  # (B, P, 7)
        pos[..., -3:] = 0.0
        pos[..., -3] = 1.0  # neighbor tag

        # Mask: zero-padded agents (all 8 features are zero) and frames
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0  # (B, P, V)
        mask_p = torch.sum(~mask_v, dim=-1) == 0  # (B, P) - fully padded agents

        # Append validity mask as extra feature channel
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)  # (B, P, V, 9)
        x = x.view(B * P, V, -1)

        # Only process valid agents
        valid_indices = ~mask_p.view(-1)  # (B*P,)
        x = x[valid_indices]  # (N_valid, V, 9)

        # MLP-Mixer
        x = self.channel_pre_project(x)  # (N_valid, V, channels_mlp_dim)
        x = x.permute(0, 2, 1)  # (N_valid, channels_mlp_dim, V)
        x = self.token_pre_project(x)  # (N_valid, channels_mlp_dim, tokens_mlp_dim)
        x = x.permute(0, 2, 1)  # (N_valid, tokens_mlp_dim, channels_mlp_dim)
        for block in self.blocks:
            x = block(x)

        # Mean pool over time
        x = torch.mean(x, dim=1)  # (N_valid, channels_mlp_dim)

        # Add type embedding
        neighbor_type = neighbor_type.view(B * P, -1)
        neighbor_type = neighbor_type[valid_indices]
        x = x + self.type_emb(neighbor_type)

        x = self.emb_project(self.norm(x))  # (N_valid, hidden_dim)

        # Scatter back to full batch
        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device, dtype=x.dtype)
        x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)


# =============================================================================
# Static Fusion Encoder
# =============================================================================

class StaticFusionEncoder(nn.Module):
    """Encodes static objects via MLP projection.

    Input: (B, num_static, 10) = [x, y, cos, sin, w, l, type_czone, type_barrier, type_cone, type_generic]
    Output: (B, num_static, hidden_dim)
    """

    def __init__(self, dim: int = 10, hidden_dim: int = 192, drop_path_rate: float = 0.0):
        super().__init__()
        self._hidden_dim = hidden_dim
        self.projection = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, P, 10) = [x, y, cos, sin, w, l, type(4)]
        """
        B, P, _ = x.shape

        # Position encoding placeholder
        pos = x[:, :, :7].clone()
        pos[..., -3:] = 0.0
        pos[..., -2] = 1.0  # static tag

        # Mask
        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0  # (B, P)

        autocast_dtype = torch.get_autocast_gpu_dtype() if x.is_cuda and torch.is_autocast_enabled() else x.dtype
        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device, dtype=autocast_dtype)
        valid_indices = ~mask_p.view(-1)

        if valid_indices.sum() > 0:
            x = x.view(B * P, -1)
            x = x[valid_indices]
            x = self.projection(x)
            x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)


# =============================================================================
# Lane Fusion Encoder
# =============================================================================

class LaneFusionEncoder(nn.Module):
    """Encodes vector map lanes using MLP-Mixer.

    Input: (B, num_lanes, lane_len, 12) = [x, y, dx, dy, left_dx, left_dy, right_dx, right_dy,
                                            traffic_green, traffic_red, traffic_yellow, traffic_unknown]
    Output: (B, num_lanes, hidden_dim)
    """

    def __init__(
        self,
        lane_len: int,
        hidden_dim: int = 192,
        depth: int = 2,
        tokens_mlp_dim: int = 64,
        channels_mlp_dim: int = 128,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self._lane_len = lane_len
        self._channel = channels_mlp_dim

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)

        self.channel_pre_project = Mlp(
            in_features=8,  # x, y, dx, dy, left_dx, left_dy, right_dx, right_dy
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=lane_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.blocks = nn.ModuleList(
            [MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x, speed_limit, has_speed_limit):
        """
        x: (B, P, V, 12) = [x, y, dx, dy, left_x-x, left_y-y, right_x-x, right_y-y, traffic(4)]
        speed_limit: (B, P, 1)
        has_speed_limit: (B, P, 1)
        """
        B, P, V, _ = x.shape

        # Save traffic and strip to geometric
        traffic = x[:, :, 0, 8:]
        x = x[..., :8]

        # Position: use midpoint
        mid = self._lane_len // 2
        pos = x[:, :, mid, :7].clone()
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        pos[..., -3:] = 0.0
        pos[..., -1] = 1.0  # lane tag

        # Mask
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0  # (B, P, V)
        mask_p = torch.sum(~mask_v, dim=-1) == 0  # (B, P)

        x = x.view(B * P, V, -1)
        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]

        # MLP-Mixer
        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        # Mean pool
        x = torch.mean(x, dim=1)  # (N_valid, channels_mlp_dim)

        # Speed limit embedding
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.view(B * P, -1)

        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)
        speed_limit_embedding = torch.zeros((speed_limit.shape[0], self._channel), device=x.device, dtype=x.dtype)
        if has_speed_limit.sum() > 0:
            speed_limit_embedding[has_speed_limit] = self.speed_limit_emb(
                speed_limit[has_speed_limit].unsqueeze(-1)
            )
        if (~has_speed_limit).sum() > 0:
            speed_limit_embedding[~has_speed_limit] = self.unknown_speed_emb.weight.to(x.dtype).expand(
                (~has_speed_limit).sum().item(), -1
            )

        # Traffic light embedding
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(traffic)

        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device, dtype=x.dtype)
        x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)


# =============================================================================
# Fusion Encoder
# =============================================================================

class FusionEncoder(nn.Module):
    """Self-attention fusion over all scene tokens."""

    def __init__(self, hidden_dim: int = 192, num_heads: int = 6, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # First token (ego) should always be visible
        mask[:, 0] = False
        for b in self.blocks:
            x = b(x, mask)
        return self.norm(x)


# =============================================================================
# Scene Encoder (top-level)
# =============================================================================

class SceneEncoder(nn.Module):
    """Vector-based scene encoder for nuPlan.

    Processes agent history, static objects, and vector map lanes,
    then fuses them via self-attention into a unified scene representation.
    """

    def __init__(self, config):
        super().__init__()

        def g(key, default=None):
            if isinstance(config, dict):
                return config.get(key, default)
            return getattr(config, key, default)

        self.hidden_dim = g('hidden_dim', 192)
        self.agent_num = g('agent_num', 32)
        self.static_objects_num = g('static_objects_num', 5)
        self.lane_num = g('lane_num', 30)
        self.token_num = self.agent_num + self.static_objects_num + self.lane_num

        self.neighbor_encoder = AgentFusionEncoder(
            time_len=g('time_len', 21),
            hidden_dim=self.hidden_dim,
            depth=g('encoder_depth', 2),
            drop_path_rate=g('encoder_drop_path_rate', 0.0),
        )
        self.static_encoder = StaticFusionEncoder(
            dim=g('static_objects_state_dim', 10),
            hidden_dim=self.hidden_dim,
            drop_path_rate=g('encoder_drop_path_rate', 0.0),
        )
        self.lane_encoder = LaneFusionEncoder(
            lane_len=g('lane_len', 20),
            hidden_dim=self.hidden_dim,
            depth=g('encoder_depth', 2),
            drop_path_rate=g('encoder_drop_path_rate', 0.0),
        )
        self.fusion = FusionEncoder(
            hidden_dim=self.hidden_dim,
            num_heads=g('num_heads', 6),
            depth=g('encoder_depth', 2),
        )

        self.pos_emb = nn.Linear(7, self.hidden_dim)

    def forward(self, inputs: dict) -> dict:
        neighbors = inputs['neighbor_agents_past']  # (B, agent_num, time_len, 11)
        static = inputs['static_objects']            # (B, static_num, 10)
        lanes = inputs['lanes']                      # (B, lane_num, lane_len, 12)
        lanes_speed_limit = inputs.get('lanes_speed_limit', torch.zeros(1))
        lanes_has_speed_limit = inputs.get('lanes_has_speed_limit', torch.zeros(1))

        B = neighbors.shape[0]

        encoding_neighbors, neighbors_mask, neighbor_pos = self.neighbor_encoder(neighbors)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(
            lanes, lanes_speed_limit, lanes_has_speed_limit
        )

        # Concatenate all scene tokens
        encoding_input = torch.cat(
            [encoding_neighbors, encoding_static, encoding_lanes], dim=1
        )  # (B, token_num, hidden_dim)

        # Position embeddings
        encoding_pos = torch.cat(
            [neighbor_pos, static_pos, lane_pos], dim=1
        ).view(B * self.token_num, -1)

        encoding_mask = torch.cat(
            [neighbors_mask, static_mask, lanes_mask], dim=1
        ).view(-1)

        # Only embed valid positions
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        encoding_pos_result = torch.zeros(
            (B * self.token_num, self.hidden_dim), device=encoding_pos.device, dtype=encoding_pos.dtype
        )
        encoding_pos_result[~encoding_mask] = encoding_pos

        encoding_input = encoding_input + encoding_pos_result.view(B, self.token_num, -1)

        scene_encoding = self.fusion(encoding_input, encoding_mask.view(B, self.token_num))

        return {'encoding': scene_encoding, 'mask': encoding_mask.view(B, self.token_num)}
