"""Online LEAD Diffusion Agent with official LEAD NAVSIM preprocessing."""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory
from model.navsim_joint_route_speed_diffusion import NavSimJointRouteSpeedDiffusion
from model.navsim_simple_diffusion import NavSimSimpleDiffusion


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
from navsim_motdp.lead_preprocessing import (
    build_official_lead_rgb_tensor,
    build_official_lead_sensor_config,
)


class OnlineLeadDiffusionAgent(AbstractAgent):
    requires_scene = False

    def __init__(
        self,
        checkpoint_path,
        lead_ckpt_path="/workspace1/z_project/models/navsim_backbones/tfv6_navsim/model_0060.pth",
        device="cuda",
        num_inference_steps=10,
        deterministic_seed=20260516,
        sampler_stochastic=False,
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        try:
            super().__init__(requires_scene=False)
        except TypeError:
            super().__init__(trajectory_sampling=trajectory_sampling)
        self._trajectory_sampling = trajectory_sampling
        self.checkpoint_path = str(checkpoint_path)
        self.lead_ckpt_path = str(lead_ckpt_path)
        self.device_name = device
        self.num_inference_steps = int(num_inference_steps)
        self.deterministic_seed = int(deterministic_seed)
        self.sampler_stochastic = _as_bool(sampler_stochastic)
        self.device = None
        self.model = None
        self.lead_model = None

    def name(self):
        return self.__class__.__name__

    def get_sensor_config(self):
        return build_official_lead_sensor_config()

    @staticmethod
    def _infer_ego_input_dim(cfg, state):
        ego_input_dim = cfg.get("ego_input_dim")
        if ego_input_dim is None and isinstance(state, dict) and "ego_proj.weight" in state:
            ego_input_dim = int(state["ego_proj.weight"].shape[1])
        return int(ego_input_dim or 8)

    def _build_diffusion_model(self, cfg, state, ego_input_dim):
        model_type = str(cfg.get("model_type", "navsim_simple_diffusion"))
        if model_type == "navsim_joint_route_speed_diffusion":
            return NavSimJointRouteSpeedDiffusion(
                d_model=cfg.get("d_model", 512),
                n_head=cfg.get("n_head", 8),
                n_layer=cfg.get("n_layer", 4),
                d_ffn=cfg.get("d_ffn", 2048),
                p_drop_attn=cfg.get("p_drop_attn", 0.1),
                p_drop_emb=cfg.get("p_drop_emb", 0.1),
                traj_horizon=cfg.get("traj_horizon", 8),
                traj_dim=cfg.get("traj_dim", 2),
                route_points=cfg.get("route_points", 50),
                speed_horizon=cfg.get("speed_horizon", 8),
                ego_input_dim=ego_input_dim,
                ego_history_frames=cfg.get("ego_history_frames", 4),
                prediction_type=cfg.get("prediction_type", "sample"),
                num_inference_steps=self.num_inference_steps,
                num_train_timesteps=cfg.get("num_train_timesteps", 1000),
                beta_schedule=cfg.get("beta_schedule", "cosine"),
                route_loss_weight=cfg.get("route_loss_weight", 1.0),
                speed_loss_weight=cfg.get("speed_loss_weight", 0.5),
                use_raw_bev_feature=cfg.get("use_raw_bev_feature", True),
                raw_bev_dim=cfg.get("raw_bev_dim", 512),
            ).to(self.device)
        return NavSimSimpleDiffusion(
            d_model=cfg.get("d_model", 512),
            n_head=cfg.get("n_head", 8),
            n_layer=cfg.get("n_layer", 4),
            d_ffn=cfg.get("d_ffn", 2048),
            traj_horizon=cfg.get("traj_horizon", 8),
            traj_dim=cfg.get("traj_dim", 2),
            ego_input_dim=ego_input_dim,
            ego_history_frames=cfg.get("ego_history_frames", 4),
            prediction_type=cfg.get("prediction_type", "sample"),
            num_inference_steps=self.num_inference_steps,
            num_train_timesteps=cfg.get("num_train_timesteps", 1000),
            beta_schedule=cfg.get("beta_schedule", "cosine"),
        ).to(self.device)

    def _load_diffusion_stats(self, ckpt):
        if not isinstance(ckpt, dict):
            return
        if "traj_mean" in ckpt:
            self.model.traj_mean.copy_(torch.as_tensor(ckpt["traj_mean"], device=self.device, dtype=torch.float32))
            self.model.traj_std.copy_(torch.as_tensor(ckpt["traj_std"], device=self.device, dtype=torch.float32))
        if isinstance(self.model, NavSimJointRouteSpeedDiffusion) and "route_mean" in ckpt:
            self.model.route_mean.copy_(torch.as_tensor(ckpt["route_mean"], device=self.device, dtype=torch.float32))
            self.model.route_std.copy_(torch.as_tensor(ckpt["route_std"], device=self.device, dtype=torch.float32))
            self.model.speed_mean.copy_(torch.as_tensor(ckpt["speed_mean"], device=self.device, dtype=torch.float32))
            self.model.speed_std.copy_(torch.as_tensor(ckpt["speed_std"], device=self.device, dtype=torch.float32))

    def initialize(self):
        self.device = torch.device(self.device_name if torch.cuda.is_available() else "cpu")
        sys.path.insert(0, "/workspace1/z_project/models/navsim_backbones/tfv6_navsim")
        from ltfv6 import load_tf

        self.lead_model = load_tf(self.lead_ckpt_path, self.device)
        self.lead_model = self.lead_model.to(dtype=torch.bfloat16)
        self.lead_model.eval()

        def patched_forward(data):
            rgb_in = data["rgb"].to(self.device, dtype=torch.bfloat16)
            cfg = self.lead_model.config
            x = torch.linspace(0, 1, cfg.lidar_width_pixel, device=self.device, dtype=torch.bfloat16)
            y = torch.linspace(0, 1, cfg.lidar_height_pixel, device=self.device, dtype=torch.bfloat16)
            y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
            lidar = torch.zeros((rgb_in.shape[0], 2, cfg.lidar_height_pixel, cfg.lidar_width_pixel), device=self.device, dtype=torch.bfloat16)
            lidar[:, 0] = y_grid.unsqueeze(0)
            lidar[:, 1] = x_grid.unsqueeze(0)
            return self.lead_model.backbone._forward(rgb_in, lidar)

        self.lead_model.backbone.forward = patched_forward
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        ego_input_dim = self._infer_ego_input_dim(cfg, state)
        self.model = self._build_diffusion_model(cfg, state, ego_input_dim)
        self.model.load_state_dict(state, strict=True)
        self._load_diffusion_stats(ckpt)
        self.model.num_inference_steps = self.num_inference_steps
        self.ego_input_dim = int(self.model.ego_proj.in_features)
        self.model.eval()

    def _build_lead_input(self, agent_input):
        return build_official_lead_rgb_tensor(agent_input, batched=True)

    def _build_ego_status(self, agent_input):
        features = []
        for es in agent_input.ego_statuses:
            vel = np.array(es.ego_velocity, dtype=np.float32)
            acc = np.array(es.ego_acceleration, dtype=np.float32)
            cmd = np.array(es.driving_command, dtype=np.float32)
            features.append(np.concatenate([vel, acc, cmd]))
        status = np.stack(features, axis=0).astype(np.float32)
        cur_dim = status.shape[-1]
        target_dim = int(getattr(self, "ego_input_dim", cur_dim))
        if cur_dim > target_dim:
            status = status[..., :target_dim]
        elif cur_dim < target_dim:
            pad = np.zeros((*status.shape[:-1], target_dim - cur_dim), dtype=np.float32)
            status = np.concatenate([status, pad], axis=-1)
        return torch.from_numpy(status).unsqueeze(0)

    @staticmethod
    def _xy_to_se2(xy):
        xy = np.asarray(xy, dtype=np.float32)
        headings = np.zeros((xy.shape[0],), dtype=np.float32)
        prev = np.zeros((2,), dtype=np.float32)
        for i, p in enumerate(xy):
            d = p - prev
            if float(np.linalg.norm(d)) > 1e-3:
                headings[i] = np.arctan2(float(d[1]), float(d[0]))
            elif i > 0:
                headings[i] = headings[i - 1]
            prev = p
        return np.concatenate([xy, headings[:, None]], axis=-1).astype(np.float32)

    def _seed_from_agent_input(self, agent_input: AgentInput) -> int:
        token = getattr(agent_input, "token", None)
        if token is not None:
            key = str(token).encode("utf-8")
        else:
            pose = np.asarray(agent_input.ego_statuses[-1].ego_pose, dtype=np.float64)
            key = pose.tobytes()
        seed_key = str(self.deterministic_seed).encode("utf-8") + b":" + key
        return int.from_bytes(hashlib.sha1(seed_key).digest()[:4], "little") & 0x7FFFFFFF

    def compute_trajectory(self, agent_input):
        rgb = self._build_lead_input(agent_input)
        lead_input = {"rgb": rgb.to(self.device, dtype=torch.bfloat16)}
        with torch.no_grad():
            bev_feat, _ = self.lead_model.backbone(lead_input)
            bev_grid = self.lead_model.backbone.top_down(bev_feat)
        ego_status = self._build_ego_status(agent_input).to(self.device).float()
        with torch.no_grad():
            seed = self._seed_from_agent_input(agent_input)
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            if isinstance(self.model, NavSimJointRouteSpeedDiffusion):
                pred_xy = self.model.sample(
                    bev_grid.float(),
                    ego_status,
                    bev_feature=bev_feat.float(),
                    return_trajectory=True,
                    stochastic=self.sampler_stochastic,
                )[0].cpu().numpy()
            else:
                try:
                    pred_xy = self.model.sample(
                        bev_grid.float(),
                        ego_status,
                        return_trajectory=True,
                        stochastic=self.sampler_stochastic,
                    )[0].cpu().numpy()
                except TypeError:
                    pred_xy = self.model.sample(bev_grid.float(), ego_status, return_trajectory=True)[0].cpu().numpy()
        return Trajectory(self._xy_to_se2(pred_xy), self._trajectory_sampling)
