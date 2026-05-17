"""
Pure LEAD agent: uses LEAD's own planning_decoder (no diffusion model).
Benchmark: what does the off-the-shelf LEAD checkpoint score on navtest?
"""

from __future__ import annotations

import hashlib, sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory
from navsim_motdp.lead_preprocessing import (
    build_official_lead_rgb_tensor,
    build_official_lead_sensor_config,
)


class PureLeadAgent(AbstractAgent):
    requires_scene = False

    def __init__(
        self,
        lead_ckpt_path: str = "/workspace1/z_project/models/navsim_backbones/tfv6_navsim/model_0060.pth",
        device: str = "cuda",
        deterministic_seed: int = 20260516,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ) -> None:
        super().__init__(trajectory_sampling=trajectory_sampling)
        self.lead_ckpt_path = str(lead_ckpt_path)
        self.device_name = device
        self.deterministic_seed = int(deterministic_seed)
        self.device: Optional[torch.device] = None
        self.lead_model = None

    def name(self) -> str:
        return self.__class__.__name__

    def get_sensor_config(self):
        return build_official_lead_sensor_config()

    def initialize(self) -> None:
        self.device = torch.device(self.device_name if torch.cuda.is_available() else "cpu")
        sys.path.insert(0, str(Path("/workspace1/z_project/models/navsim_backbones/tfv6_navsim")))
        from ltfv6 import load_tf
        self.lead_model = load_tf(self.lead_ckpt_path, self.device)
        self.lead_model = self.lead_model.to(dtype=torch.bfloat16)
        self.lead_model.eval()

        # Monkey-patch LTF grid
        def patched_forward(data):
            rgb_in = data["rgb"].to(self.device, dtype=torch.bfloat16)
            cfg = self.lead_model.config
            x = torch.linspace(0, 1, cfg.lidar_width_pixel, device=self.device, dtype=torch.bfloat16)
            y = torch.linspace(0, 1, cfg.lidar_height_pixel, device=self.device, dtype=torch.bfloat16)
            y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
            lidar = torch.zeros((rgb_in.shape[0], 2, cfg.lidar_height_pixel, cfg.lidar_width_pixel),
                               device=self.device, dtype=torch.bfloat16)
            lidar[:, 0] = y_grid.unsqueeze(0); lidar[:, 1] = x_grid.unsqueeze(0)
            return self.lead_model.backbone._forward(rgb_in, lidar)
        self.lead_model.backbone.forward = patched_forward

    def _build_lead_input(self, agent_input: AgentInput):
        return build_official_lead_rgb_tensor(agent_input, batched=True)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        rgb = self._build_lead_input(agent_input)
        ego = agent_input.ego_statuses[-1]
        speed = float(np.linalg.norm(ego.ego_velocity))
        accel = float(np.linalg.norm(ego.ego_acceleration))
        cmd = torch.tensor([ego.driving_command.tolist()], device=self.device, dtype=torch.bfloat16)

        data = {
            "rgb": rgb.to(self.device, dtype=torch.bfloat16),
            "command": cmd,
            "speed": torch.tensor([speed], device=self.device, dtype=torch.bfloat16),
            "acceleration": torch.tensor([accel], device=self.device, dtype=torch.bfloat16),
        }

        with torch.no_grad():
            pred = self.lead_model(data)

        # LEAD outputs CARLA left-handed coords -> convert to ISO (y-flip)
        waypoints = pred.pred_future_waypoints[0].float().cpu().numpy().copy()
        headings = pred.pred_headings[0].float().cpu().numpy().copy()
        waypoints[:, 1] *= -1.0
        headings *= -1.0

        # Combine to (N, 3) poses
        poses = np.concatenate([waypoints, headings[:, None]], axis=-1).astype(np.float32)
        return Trajectory(poses, self._trajectory_sampling)
