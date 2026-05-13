"""Adapters for feeding NAVSIM/LEAD BEV features into MoT-DP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn


TensorDict = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class NavsimLeadBEVFeatures:
    """MoT-DP BEV feature contract produced from a LEAD-style backbone."""

    transfuser_bev_feature: torch.Tensor
    transfuser_bev_feature_upsample: torch.Tensor
    transfuser_bev_feature_raw: torch.Tensor
    lead_image_feature: Optional[torch.Tensor] = None

    def as_batch(self) -> TensorDict:
        batch = {
            "transfuser_bev_feature": self.transfuser_bev_feature,
            "transfuser_bev_feature_upsample": self.transfuser_bev_feature_upsample,
            "transfuser_bev_feature_raw": self.transfuser_bev_feature_raw,
        }
        if self.lead_image_feature is not None:
            batch["lead_image_feature"] = self.lead_image_feature
        return batch


class NavsimLeadBEVAdapter(nn.Module):
    """Wrap a LEAD backbone and emit the BEV tensors expected by MoT-DP."""

    def __init__(
        self,
        lead_module: nn.Module,
        global_in_channels: int = 512,
        global_out_channels: int = 1512,
        upsample_channels: int = 64,
        freeze_lead: bool = True,
    ) -> None:
        super().__init__()
        self.lead_module = lead_module
        self.global_in_channels = int(global_in_channels)
        self.global_out_channels = int(global_out_channels)
        self.upsample_channels = int(upsample_channels)

        if self.global_in_channels == self.global_out_channels:
            self.global_adapter = nn.Identity()
        else:
            self.global_adapter = nn.Conv2d(
                self.global_in_channels,
                self.global_out_channels,
                kernel_size=1,
            )

        if freeze_lead:
            self.freeze_lead()

    @property
    def backbone(self) -> nn.Module:
        return getattr(self.lead_module, "backbone", self.lead_module)

    def freeze_lead(self) -> None:
        for param in self.lead_module.parameters():
            param.requires_grad_(False)
        self.lead_module.eval()

    def extract_raw_features(
        self,
        lead_input: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        backbone = self.backbone
        raw_output = backbone(lead_input)
        if not isinstance(raw_output, (tuple, list)) or len(raw_output) < 1:
            raise TypeError(
                "Expected LEAD backbone to return (bev_features, image_features)."
            )

        bev_feature = raw_output[0]
        image_feature = raw_output[1] if len(raw_output) > 1 else None
        if bev_feature.ndim != 4:
            raise ValueError(
                f"Expected compact BEV shape [B,C,H,W], got {tuple(bev_feature.shape)}"
            )
        if bev_feature.shape[1] != self.global_in_channels:
            raise ValueError(
                f"Expected compact BEV channels={self.global_in_channels}, "
                f"got {bev_feature.shape[1]}"
            )
        return bev_feature, image_feature

    def forward(self, lead_input: Mapping[str, torch.Tensor]) -> TensorDict:
        raw_bev_feature, image_feature = self.extract_raw_features(lead_input)

        backbone = self.backbone
        if not hasattr(backbone, "top_down"):
            raise AttributeError("LEAD backbone must expose a top_down(bev) method.")
        upsample_bev_feature = backbone.top_down(raw_bev_feature)
        if upsample_bev_feature.ndim != 4:
            raise ValueError(
                "Expected top_down BEV shape [B,C,H,W], "
                f"got {tuple(upsample_bev_feature.shape)}"
            )
        if upsample_bev_feature.shape[1] != self.upsample_channels:
            raise ValueError(
                f"Expected top_down BEV channels={self.upsample_channels}, "
                f"got {upsample_bev_feature.shape[1]}"
            )

        motdp_bev_feature = self.global_adapter(raw_bev_feature)
        return NavsimLeadBEVFeatures(
            transfuser_bev_feature=motdp_bev_feature,
            transfuser_bev_feature_upsample=upsample_bev_feature,
            transfuser_bev_feature_raw=raw_bev_feature,
            lead_image_feature=image_feature,
        ).as_batch()


def lead_carla_to_navsim_trajectory(
    waypoints: torch.Tensor,
    headings: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Convert LEAD/CARLA y-right trajectories back to NAVSIM y-left."""

    navsim_waypoints = waypoints.clone()
    navsim_waypoints[..., 1] = -navsim_waypoints[..., 1]
    if headings is None:
        return navsim_waypoints

    navsim_headings = -headings.clone()
    return navsim_waypoints, navsim_headings


def adapt_waypoint_horizon(
    waypoints: torch.Tensor,
    target_horizon: int = 8,
    pad_mode: str = "repeat_last",
) -> torch.Tensor:
    """Truncate or pad waypoint tensors on the horizon dimension."""

    if target_horizon <= 0:
        raise ValueError(f"target_horizon must be positive, got {target_horizon}")
    if waypoints.ndim < 2:
        raise ValueError(f"Expected waypoint tensor with horizon dim, got {waypoints.ndim}D")

    horizon = waypoints.shape[-2]
    if horizon == target_horizon:
        return waypoints
    if horizon > target_horizon:
        return waypoints[..., :target_horizon, :]
    if pad_mode != "repeat_last":
        raise ValueError(f"Unsupported pad_mode={pad_mode!r}")

    pad_count = target_horizon - horizon
    last = waypoints[..., -1:, :].expand(*waypoints.shape[:-2], pad_count, waypoints.shape[-1])
    return torch.cat([waypoints, last], dim=-2)
