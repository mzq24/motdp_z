"""Cached LEAD agent: reads NPY BEV cache, runs LEAD planning_decoder."""
import glob, hashlib, sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np, torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory

class CachedLeadAgent(AbstractAgent):
    requires_scene = True

    def __init__(self, checkpoint_path, cache_dir, device='cuda',
                 trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5)):
        try:
            super().__init__(requires_scene=True)
        except TypeError:
            super().__init__(trajectory_sampling=trajectory_sampling, requires_scene=True)
        self._trajectory_sampling = trajectory_sampling
        self.checkpoint_path = str(checkpoint_path)
        self.cache_dir = Path(cache_dir)
        self.device_name = device
        self.device = None; self.model = None

    def name(self): return 'CachedLeadAgent'
    def get_sensor_config(self): return SensorConfig.build_no_sensors()

    def initialize(self):
        self.device = torch.device(self.device_name if torch.cuda.is_available() else 'cpu')
        # Load cache
        idx = np.load(self.cache_dir / 'cache_index.npz')
        self.tokens = idx['tokens'].astype(str)
        self.token_to_index = {t: i for i, t in enumerate(self.tokens)}
        self.bf_mmap = np.load(self.cache_dir / 'bev_feature.npy', mmap_mode='r')
        self.eg_mmap = np.load(self.cache_dir / 'ego_status.npy', mmap_mode='r')
        # Load LEAD model
        sys.path.insert(0, '/workspace1/z_project/models/navsim_backbones/tfv6_navsim')
        from ltfv6 import load_tf
        self.model = load_tf(self.checkpoint_path, self.device)
        self.model = self.model.to(dtype=torch.bfloat16)
        self.model.eval()

    def compute_trajectory(self, agent_input, scene):
        token = scene.scene_metadata.initial_token
        idx = self.token_to_index.get(token)
        if idx is None:
            token = scene.frames[scene.scene_metadata.num_history_frames - 1].token
            idx = self.token_to_index.get(token)
        if idx is None:
            ego_speed = float(np.linalg.norm(agent_input.ego_statuses[-1].ego_velocity))
            dt = self._trajectory_sampling.interval_length
            n = self._trajectory_sampling.num_poses
            return Trajectory(np.asarray([[(i+1)*dt*ego_speed,0.0,0.0] for i in range(n)], dtype=np.float32), self._trajectory_sampling)

        # Get cached BEV + ego
        bf = torch.from_numpy(np.asarray(self.bf_mmap[idx]).copy()).unsqueeze(0).to(self.device).bfloat16()
        eg = self.eg_mmap[idx]
        cmd = torch.tensor([eg[-1, 4:8].tolist()], device=self.device, dtype=torch.bfloat16)
        speed = float(np.linalg.norm(eg[-1, :2]))
        accel = float(np.linalg.norm(eg[-1, 2:4]))
        data = {
            'command': cmd,
            'speed': torch.tensor([speed], device=self.device, dtype=torch.bfloat16),
            'acceleration': torch.tensor([accel], device=self.device, dtype=torch.bfloat16),
        }
        with torch.no_grad():
            waypoints, headings = self.model.planning_decoder(bf, data, {})
        wp = waypoints[0].float().cpu().numpy().copy()
        hd = headings[0].float().cpu().numpy().copy() if headings is not None else np.zeros(wp.shape[0])
        wp[:,1] *= -1; hd *= -1
        return Trajectory(np.concatenate([wp, hd[:, None]], -1).astype(np.float32), self._trajectory_sampling)
