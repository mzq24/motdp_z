"""
Simple white-noise diffusion policy for NavSim.
No energy guidance, no semantic state, no branch conditioning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from model.navsim_lead_bev_adapter import NavsimLeadBEVAdapter


class NavSimDiffusionPolicy:
    """
    Minimal diffusion policy: LEAD backbone → BEV adapter → diffusion decoder.
    """

    def __init__(
        self,
        model: nn.Module,  # NavSimSimpleDiffusion
        bev_adapter: nn.Module,  # NavsimLeadBEVAdapter
        device: torch.device,
        traj_mean: Optional[np.ndarray] = None,
        traj_std: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.bev_adapter = bev_adapter
        self.device = device

        # Set normalization stats if provided
        if traj_mean is not None:
            model.traj_mean.copy_(torch.tensor(traj_mean))
        if traj_std is not None:
            model.traj_std.copy_(torch.tensor(traj_std))

        self.model.eval()
        self.bev_adapter.eval()

    @torch.no_grad()
    def predict(self, lead_input: Dict[str, torch.Tensor],
                ego_status: torch.Tensor) -> np.ndarray:
        """
        Run diffusion sampling end-to-end.
        lead_input: dict with "rgb", "command", "speed", "acceleration" for LEAD backbone
        ego_status: (T_hist, 14) - ego history in NavSim format
        returns: (T, 2) numpy array - predicted trajectory
        """
        # LEAD backbone → BEV features
        bev_dict = self.bev_adapter(lead_input)  # on device

        bev_grid = bev_dict["transfuser_bev_feature_upsample"]  # (1, 64, 64, 64)
        ego_batch = ego_status.unsqueeze(0).to(self.device)  # (1, T_hist, 14)

        # Diffusion sampling
        traj = self.model.sample(bev_grid, ego_batch, return_trajectory=True)  # (1, T, 2)
        return traj[0].cpu().numpy()

    def train(self):
        self.model.train()
        self.bev_adapter.eval()  # backbone stays frozen

    def eval(self):
        self.model.eval()
        self.bev_adapter.eval()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "traj_mean": self.model.traj_mean.cpu().numpy(),
            "traj_std": self.model.traj_std.cpu().numpy(),
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.model.traj_mean.copy_(torch.tensor(state_dict["traj_mean"]))
        self.model.traj_std.copy_(torch.tensor(state_dict["traj_std"]))


def build_ego_status_from_navsim(agent_input) -> np.ndarray:
    """
    Build ego status tensor from NavSim AgentInput.
    Returns (T_hist, 14) numpy array: [speed(2), accel(2), command(4), velocity_norm(1),
                                        steer(1), yaw_rate(1), ???(3)]

    Simplified version: [speed(2), accel(2), command(4), zeros(6)]
    """
    ego_statuses = agent_input.ego_statuses
    features = []
    for es in ego_statuses:
        vel = np.array(es.ego_velocity, dtype=np.float32)  # (2,)
        acc = np.array(es.ego_acceleration, dtype=np.float32)  # (2,)
        cmd = np.array(es.driving_command, dtype=np.float32)  # (4,)
        padding = np.zeros(6, dtype=np.float32)
        feat = np.concatenate([vel, acc, cmd, padding])
        features.append(feat)
    return np.stack(features, axis=0)  # (T_hist, 14)
