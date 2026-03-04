"""Loss computation for multimodal trajectory prediction.

Focal loss for mode classification + L1 loss for trajectory regression.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss: Tensor,
                       weight: Optional[Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> Tensor:
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def py_sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25,
                          reduction='mean', avg_factor=None):
    """PyTorch implementation of Focal Loss."""
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class LossComputer(nn.Module):
    """Computes multimodal trajectory prediction loss.

    1. Find best matching mode (anchor closest to GT trajectory)
    2. Focal loss for mode classification
    3. L1 loss for trajectory regression on best mode
    """
    def __init__(self, cls_loss_weight=10.0, reg_loss_weight=8.0):
        super().__init__()
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight

    def forward(self, poses_reg, poses_cls, target_traj, plan_anchor):
        """
        Args:
            poses_reg: (bs, num_modes, T, 2) predicted trajectories
            poses_cls: (bs, num_modes) classification logits
            target_traj: (bs, T, 2) ground truth trajectory
            plan_anchor: (bs, num_modes, T, 2) anchor centers
        Returns:
            total_loss: scalar
        """
        bs, num_mode, ts, d = poses_reg.shape

        # Find best matching mode based on anchor distance to GT
        dist = torch.linalg.norm(target_traj.unsqueeze(1) - plan_anchor, dim=-1)  # (bs, num_modes, T)
        dist = dist.mean(dim=-1)  # (bs, num_modes)
        mode_idx = torch.argmin(dist, dim=-1)  # (bs,)

        # Get best mode prediction
        cls_target = mode_idx
        mode_idx_expanded = mode_idx[:, None, None, None].expand(-1, 1, ts, d)
        best_reg = torch.gather(poses_reg, 1, mode_idx_expanded).squeeze(1)  # (bs, T, 2)

        # Focal loss for classification
        target_classes_onehot = torch.zeros(
            [bs, num_mode], dtype=poses_cls.dtype,
            layout=poses_cls.layout, device=poses_cls.device
        )
        target_classes_onehot.scatter_(1, cls_target.unsqueeze(1), 1)
        loss_cls = self.cls_loss_weight * py_sigmoid_focal_loss(
            poses_cls, target_classes_onehot,
            gamma=2.0, alpha=0.25, reduction='mean'
        )

        # L1 regression loss on best mode
        reg_loss = self.reg_loss_weight * F.l1_loss(best_reg, target_traj)

        return loss_cls + reg_loss
