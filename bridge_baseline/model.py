"""
BDBaselineModel: top-level model for BridgeDrive DDBM baseline.

Identical feature extraction to DDBaselineModel (TransFuser BEV → d_model, ego_query, agent_queries),
only the trajectory head is replaced with the DDBM-based TrajectoryHead.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .config import BridgeBaselineConfig
from .trajectory_head import TrajectoryHead
from .modules.blocks import linear_relu_ln


class BDBaselineModel(nn.Module):
    def __init__(self, config: BridgeBaselineConfig):
        super().__init__()
        self.config = config
        d_model = config.tf_d_model

        # BEV feature projection: (1512, 8, 8) -> (d_model, 8, 8)
        self.bev_downscale = nn.Conv2d(config.bev_feature_dim, d_model, kernel_size=1)

        # Status encoding: 14-dim ego_status -> d_model
        self.status_encoding = nn.Linear(config.status_dim, d_model)

        # Key-value embedding: 8x8 BEV tokens + 1 status token = 65 tokens
        self.keyval_embedding = nn.Embedding(8 * 8 + 1, d_model)

        # Query embedding: 1 ego query
        self.query_embedding = nn.Embedding(1, d_model)

        # Standard transformer decoder to produce ego_query
        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)

        # BEV projection: concat upsampled bev_down (d_model) + bev_upsample (64) -> d_model
        bev_concat_dim = d_model + config.bev_feature_upsample_dim
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, bev_concat_dim),
        )

        # Learned agent queries (replacing agent detection head)
        self.agent_queries = nn.Embedding(config.num_agent_queries, d_model)

        # DDBM trajectory head
        self.trajectory_head = TrajectoryHead(config)

    def forward(self, features: Dict[str, torch.Tensor],
                targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: {
                'transfuser_bev_feature': (B, 1512, 8, 8),
                'transfuser_bev_feature_upsample': (B, 64, 64, 64),
                'ego_status': (B, T_obs, 14)
            }
            targets: {
                'route': (B, T_route, 2)   ← full 20-pt route; head uses [:num_poses]
            } or None for inference
        """
        bev_feature = features['transfuser_bev_feature']           # (B, 1512, 8, 8)
        bev_upsample = features['transfuser_bev_feature_upsample']  # (B, 64, 64, 64)
        ego_status = features['ego_status']                         # (B, T_obs, 14)

        B = bev_feature.shape[0]

        # 1. Downscale BEV: (B, 1512, 8, 8) -> (B, d_model, 8, 8)
        bev_down = self.bev_downscale(bev_feature)

        # 2. Status encoding: use last observation step
        status_enc = self.status_encoding(ego_status[:, -1, :])  # (B, d_model)

        # 3. Build key-value sequence: (B, 65, d_model)
        bev_flat = bev_down.flatten(-2, -1).permute(0, 2, 1)     # (B, 64, d_model)
        keyval = torch.cat([bev_flat, status_enc[:, None]], dim=1)  # (B, 65, d_model)
        keyval = keyval + self.keyval_embedding.weight[None, ...]

        # 4. Get ego query via standard transformer decoder
        query = self.query_embedding.weight[None, ...].expand(B, -1, -1)  # (B, 1, d_model)
        ego_query = self.tf_decoder(query, keyval)                          # (B, 1, d_model)

        # 5. Build cross_bev_feature for GridSample attention (64x64 resolution)
        bev_spatial_shape = bev_upsample.shape[2:]  # (64, 64)
        bev_down_up = F.interpolate(
            bev_down, size=bev_spatial_shape, mode='bilinear', align_corners=False
        )  # (B, d_model, 64, 64)
        cross_bev = torch.cat([bev_down_up, bev_upsample], dim=1)  # (B, d_model+64, 64, 64)

        cross_bev_flat = cross_bev.flatten(-2, -1).permute(0, 2, 1)  # (B, 4096, d_model+64)
        cross_bev_proj = self.bev_proj(cross_bev_flat)                # (B, 4096, d_model)
        cross_bev_feature = cross_bev_proj.permute(0, 2, 1).contiguous().view(
            B, -1, bev_spatial_shape[0], bev_spatial_shape[1]
        )  # (B, d_model, 64, 64)

        # 6. Agent queries (learned, batch-expanded)
        agent_q = self.agent_queries.weight[None, ...].expand(B, -1, -1)  # (B, N_agent, d_model)

        # 7. DDBM trajectory head
        output = self.trajectory_head(
            ego_query=ego_query,
            agents_query=agent_q,
            bev_feature=cross_bev_feature,
            bev_spatial_shape=bev_spatial_shape,
            status_encoding=status_enc[:, None],
            targets=targets,
        )

        return output
