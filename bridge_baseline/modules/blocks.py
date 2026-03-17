"""
Core building blocks adapted from DiffusionDrive.
- GridSampleCrossBEVAttention: spatial attention via grid_sample on BEV features
- linear_relu_ln: MLP builder utility
- gen_sineembed_for_position: sinusoidal position embedding for 2D points
- bias_init_with_prob: bias initialization helper
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Sinusoidal position embedding for 2D points.
    Adapted from DAB-DETR.

    Args:
        pos_tensor: (..., 2) tensor of (x, y) coordinates
        hidden_dim: embedding dimension per coordinate
    Returns:
        (..., hidden_dim) tensor of position embeddings
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos


def bias_init_with_prob(prior_prob):
    """Initialize conv/fc bias value according to given probability."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


class GridSampleCrossBEVAttention(nn.Module):
    """Spatial cross-attention that samples BEV features at trajectory waypoint locations.

    Uses grid_sample to extract features from BEV map at normalized trajectory coordinates,
    then applies learned attention weights to aggregate across waypoints.
    """
    def __init__(self, embed_dims, num_heads, num_points=6, in_bev_dims=256,
                 lidar_max_x=32.0, lidar_max_y=32.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.lidar_max_x = lidar_max_x
        self.lidar_max_y = lidar_max_y

        self.attention_weights = nn.Linear(embed_dims, num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)

        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, queries, traj_points, bev_feature, spatial_shape):
        """
        Args:
            queries: (bs, num_queries, embed_dims)
            traj_points: (bs, num_queries, num_points, 2) trajectory waypoints in ego frame
            bev_feature: (bs, C, H, W) BEV feature map
            spatial_shape: (H, W) spatial dimensions
        Returns:
            (bs, num_queries, embed_dims) attended features + residual
        """
        bs, num_queries, num_points, _ = traj_points.shape

        # Normalize trajectory points to [-1, 1] for grid_sample
        normalized_trajectory = traj_points.clone()
        normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.lidar_max_y
        normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.lidar_max_x
        normalized_trajectory = normalized_trajectory[..., [1, 0]]  # Swap x and y for grid_sample

        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(bs, num_queries, num_points).softmax(-1)

        value = self.value_proj(bev_feature)
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)
        sampled_features = F.grid_sample(
            value, grid,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )  # (bs, C, num_queries, num_points)

        attention_weights = attention_weights.unsqueeze(1)
        out = (attention_weights * sampled_features).sum(dim=-1)
        out = out.permute(0, 2, 1).contiguous()  # (bs, num_queries, C)
        out = self.output_proj(out)

        return self.dropout(out) + queries
