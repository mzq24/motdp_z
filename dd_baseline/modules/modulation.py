"""FiLM-style modulation layer conditioned on diffusion timestep."""
import torch
import torch.nn as nn


class ModulationLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioned on timestep embedding.

    Applies: out = feature * (1 + scale) + shift
    where (scale, shift) are predicted from the conditioning signal.
    """
    def __init__(self, embed_dims: int, condition_dims: int):
        super().__init__()
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims * 2),
        )

    def forward(self, traj_feature, time_embed, global_cond=None):
        if global_cond is not None:
            global_feature = torch.cat([global_cond, time_embed], dim=-1)
        else:
            global_feature = time_embed

        scale_shift = self.scale_shift_mlp(global_feature)
        scale, shift = scale_shift.chunk(2, dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature
