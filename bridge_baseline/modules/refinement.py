"""Trajectory refinement module: predicts classification logits and regression offsets."""
import torch
import torch.nn as nn
from .blocks import linear_relu_ln, bias_init_with_prob


class DiffMotionPlanningRefinementModule(nn.Module):
    """Per-decoder-layer prediction head.

    Predicts:
        - plan_cls: (bs, num_modes) classification logits
        - plan_reg: (bs, num_modes, num_poses, 2) trajectory offsets (x, y)
    """
    def __init__(self, embed_dims=256, ego_fut_ts=6, ego_fut_mode=20):
        super().__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),  # Only predict (x, y), no heading
        )

        self._init_weight()

    def _init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(self, traj_feature):
        """
        Args:
            traj_feature: (bs, num_modes, embed_dims)
        Returns:
            plan_reg: (bs, num_modes, ego_fut_ts, 2)
            plan_cls: (bs, num_modes)
        """
        bs, ego_fut_mode, _ = traj_feature.shape

        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)  # (bs, num_modes)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 2)

        return plan_reg, plan_cls
