"""
PlanningContextEncoder — adapted from BridgeDrive's planning_decoder_bridgedrive.py.

Encodes BEV feature map + ego status tokens into a flat context token sequence
for the subsequent TransformerDecoder.

Input BEV: (B, 64, H, W) pre-computed transfuser_bev_feature_upsample
Output:    (B, H*W + num_status, token_dim)
"""
import math
import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """Sine-cosine 2-D positional embedding for BEV feature maps."""

    def __init__(self, num_pos_feats: int = 32, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # half of token_dim
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            pos: (B, C, H, W) positional encoding (C = 2 * num_pos_feats)
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        y_embed = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).expand(H, W)

        if self.normalize:
            y_embed = y_embed / (H - 1 + 1e-6) * self.scale
            x_embed = x_embed / (W - 1 + 1e-6) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t   # (H, W, num_pos_feats)
        pos_y = y_embed[:, :, None] / dim_t

        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_y, pos_x], dim=-1)       # (H, W, 2*num_pos_feats)
        pos = pos.permute(2, 0, 1).unsqueeze(0)        # (1, C, H, W)
        return pos.expand(B, -1, -1, -1).to(dtype)


class PlanningContextEncoder(nn.Module):
    """
    Encodes:
      - BEV feature map  → (B, H*W, token_dim)  spatial tokens + sine PE
      - velocity (speed) → (B, 1,   token_dim)  normalized by max_speed
      - command          → (B, 1,   token_dim)  one-hot 6-dim
      - target_point     → (B, 1,   token_dim)  normalized by tp_norm
      - target_point_next→ (B, 1,   token_dim)  same tp_encoder (shared weights)

    Returns: (B, H*W + 4, token_dim)
    """

    NUM_STATUS_TOKENS = 4  # velocity, command, tp_curr, tp_next

    def __init__(
        self,
        in_bev_channels: int = 64,
        token_dim: int = 64,
        max_speed: float = 25.0,
        tp_norm: tuple = (200.0, 50.0),
    ):
        super().__init__()
        self.token_dim = token_dim
        self.max_speed = max_speed
        # tp_norm: [forward_max, lateral_max] — transforms raw target_point to ~[-1, 1]
        self.register_buffer('tp_norm', torch.tensor(tp_norm, dtype=torch.float32))

        # BEV channel projection
        self.dimension_adapter = nn.Conv2d(in_bev_channels, token_dim, kernel_size=1)

        # Positional encoding for BEV tokens
        self.cosine_pos_embedding = PositionEmbeddingSine(num_pos_feats=token_dim // 2, normalize=True)

        # Status encoders
        self.velocity_encoder = nn.Linear(1, token_dim)
        self.command_encoder  = nn.Linear(6, token_dim)
        self.tp_encoder       = nn.Linear(2, token_dim)   # shared for tp_curr + tp_next

        # Learnable positional offsets for the 4 status tokens
        self.status_pos_embedding = nn.Parameter(
            torch.zeros(1, self.NUM_STATUS_TOKENS, token_dim)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.status_pos_embedding)

    def forward(
        self,
        bev: torch.Tensor,     # (B, 64, H, W)
        speed: torch.Tensor,   # (B,) or (B, T) — we take last timestep
        command: torch.Tensor, # (B, 6) or (B, T, 6)
        tp: torch.Tensor,      # (B, 2) or (B, T, 2)
        tp_next: torch.Tensor, # (B, 2) or (B, T, 2)
    ) -> torch.Tensor:
        """Returns context tokens: (B, H*W + NUM_STATUS_TOKENS, token_dim)."""
        B = bev.shape[0]
        bev = bev.float()

        # — BEV tokens —
        bev_proj = self.dimension_adapter(bev)                    # (B, D, H, W)
        bev_proj = bev_proj + self.cosine_pos_embedding(bev_proj) # add spatial PE
        bev_tokens = bev_proj.flatten(2).permute(0, 2, 1)        # (B, H*W, D)

        # — Status tokens —
        # Take last timestep if histories are provided
        if speed.dim() > 1:
            speed = speed[:, -1]
        if command.dim() > 2:
            command = command[:, -1]
        if tp.dim() > 2:
            tp = tp[:, -1]
        if tp_next.dim() > 2:
            tp_next = tp_next[:, -1]

        speed = speed.float().reshape(B, 1) / self.max_speed
        command = command.float()
        tp = tp.float() / self.tp_norm
        tp_next = tp_next.float() / self.tp_norm

        vel_tok = self.velocity_encoder(speed).unsqueeze(1)   # (B, 1, D)
        cmd_tok = self.command_encoder(command).unsqueeze(1)  # (B, 1, D)
        tp_tok  = self.tp_encoder(tp).unsqueeze(1)            # (B, 1, D)
        tp_next_tok = self.tp_encoder(tp_next).unsqueeze(1)   # (B, 1, D)

        status_tokens = torch.cat([vel_tok, cmd_tok, tp_tok, tp_next_tok], dim=1)  # (B, 4, D)
        status_tokens = status_tokens + self.status_pos_embedding

        # Concatenate: BEV tokens first, then status tokens
        context_tokens = torch.cat([bev_tokens, status_tokens], dim=1)  # (B, H*W+4, D)
        return context_tokens
