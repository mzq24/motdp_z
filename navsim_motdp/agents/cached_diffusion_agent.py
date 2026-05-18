"""NAVSIM CachedDiffusionAgent - supports NPY/NPZ cache and simple/joint ckpts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory
from model.navsim_joint_route_speed_diffusion import NavSimJointRouteSpeedDiffusion
from model.navsim_simple_diffusion import NavSimSimpleDiffusion


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class CachedDiffusionAgent(AbstractAgent):
    requires_scene = True

    def __init__(
        self,
        checkpoint_path,
        cache_dir="/workspace2/z_project/motdp_bev_cache_official_npy",
        device="cuda",
        num_inference_steps=None,
        deterministic_seed=20260516,
        fallback="constant_velocity",
        sampler_stochastic=False,
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        try:
            super().__init__(requires_scene=True)
        except TypeError:
            super().__init__(trajectory_sampling=trajectory_sampling, requires_scene=True)
        self._trajectory_sampling = trajectory_sampling
        self.checkpoint_path = str(checkpoint_path)
        self.cache_dir = Path(cache_dir)
        self.device_name = device
        self.num_inference_steps = num_inference_steps
        self.deterministic_seed = int(deterministic_seed)
        self.fallback = fallback
        self.sampler_stochastic = _as_bool(sampler_stochastic)
        self.device = None
        self.model = None

    def name(self):
        return self.__class__.__name__

    def get_sensor_config(self):
        return SensorConfig.build_no_sensors()

    @staticmethod
    def _infer_ego_input_dim(cfg, state):
        ego_input_dim = cfg.get("ego_input_dim")
        if ego_input_dim is None and isinstance(state, dict) and "ego_proj.weight" in state:
            ego_input_dim = int(state["ego_proj.weight"].shape[1])
        return int(ego_input_dim or 8)

    def _build_model(self, cfg, state, ego_input_dim, num_inference_steps):
        model_type = str(cfg.get("model_type", "navsim_simple_diffusion"))
        if model_type == "navsim_joint_route_speed_diffusion" or "route_mean" in cfg:
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
                num_inference_steps=num_inference_steps,
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
            p_drop_attn=cfg.get("p_drop_attn", 0.1),
            p_drop_emb=cfg.get("p_drop_emb", 0.1),
            traj_horizon=cfg.get("traj_horizon", 8),
            traj_dim=cfg.get("traj_dim", 2),
            ego_input_dim=ego_input_dim,
            ego_history_frames=cfg.get("ego_history_frames", 4),
            prediction_type=cfg.get("prediction_type", "sample"),
            num_inference_steps=num_inference_steps,
            num_train_timesteps=cfg.get("num_train_timesteps", 1000),
            beta_schedule=cfg.get("beta_schedule", "cosine"),
        ).to(self.device)

    def _load_checkpoint_stats(self, ckpt):
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
        self._load_cache_index()
        self.token_to_index = {token: index for index, token in enumerate(self.tokens)}

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        ego_input_dim = self._infer_ego_input_dim(cfg, state)
        num_inference_steps = int(self.num_inference_steps or cfg.get("num_inference_steps", 10))
        self.model = self._build_model(cfg, state, ego_input_dim, num_inference_steps)
        self.model.load_state_dict(state, strict=True)
        self._load_checkpoint_stats(ckpt)
        self.model.num_inference_steps = num_inference_steps
        self.ego_input_dim = int(self.model.ego_proj.in_features)
        self.model.eval()

    def _load_cache_index(self):
        index_path = self.cache_dir / "cache_index.npz"
        if index_path.is_file():
            index = np.load(index_path, allow_pickle=False)
            self.tokens = index["tokens"].astype(str)
            self.bev_grid = np.load(self.cache_dir / "bev_grid.npy", mmap_mode="r")
            self.bev_feature = np.load(self.cache_dir / "bev_feature.npy", mmap_mode="r")
            self.ego_status = np.load(self.cache_dir / "ego_status.npy", mmap_mode="r")
            return

        shard_paths = sorted(self.cache_dir.glob("bev_cache_shard[0-9][0-9][0-9].npz"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No NPY cache_index.npz or final bev_cache_shardNNN.npz files found in {self.cache_dir}"
            )
        self._load_npz_shards(shard_paths)

    def _load_npz_shards(self, shard_paths):
        self._shard_data = []
        all_tokens = []
        self._offsets = [0]
        for shard_path in shard_paths:
            shard = np.load(shard_path, allow_pickle=False)
            self._shard_data.append(shard)
            all_tokens.extend(str(token) for token in shard["tokens"])
            self._offsets.append(self._offsets[-1] + len(shard["tokens"]))
        self.tokens = np.array(all_tokens)

    @staticmethod
    def _xy_to_se2(xy):
        xy = np.asarray(xy, dtype=np.float32)
        headings = np.zeros((xy.shape[0],), dtype=np.float32)
        prev = np.zeros((2,), dtype=np.float32)
        for i, point in enumerate(xy):
            delta = point - prev
            if float(np.linalg.norm(delta)) > 1e-3:
                headings[i] = np.arctan2(float(delta[1]), float(delta[0]))
            elif i > 0:
                headings[i] = headings[i - 1]
            prev = point
        return np.concatenate([xy, headings[:, None]], axis=-1).astype(np.float32)

    def _seed_for_token(self, token):
        digest = hashlib.sha1(str(token).encode("utf-8")).hexdigest()[:8]
        return (self.deterministic_seed + int(digest, 16)) % (2**31 - 1)

    def _fallback_trajectory(self, agent_input, token):
        if self.fallback == "raise":
            raise KeyError(f"Token {token} not found in cached BEV directory {self.cache_dir}")
        if self.fallback != "constant_velocity":
            raise ValueError(f"Unsupported fallback mode: {self.fallback}")
        ego_speed = float(np.linalg.norm(agent_input.ego_statuses[-1].ego_velocity))
        dt = self._trajectory_sampling.interval_length
        poses = np.asarray(
            [[(i + 1) * dt * ego_speed, 0.0, 0.0] for i in range(self._trajectory_sampling.num_poses)],
            dtype=np.float32,
        )
        return Trajectory(poses, self._trajectory_sampling)

    def _lookup_token(self, scene):
        token = scene.scene_metadata.initial_token
        index = self.token_to_index.get(token)
        if index is not None:
            return token, index
        if scene.frames:
            token = scene.frames[scene.scene_metadata.num_history_frames - 1].token
            index = self.token_to_index.get(token)
        return token, index

    def _get_features(self, index):
        if hasattr(self, "_shard_data"):
            for shard_index in range(len(self._offsets) - 2, -1, -1):
                if index >= self._offsets[shard_index]:
                    shard = self._shard_data[shard_index]
                    local_index = index - self._offsets[shard_index]
                    return (
                        np.array(shard["bev_grid"][local_index], copy=True),
                        np.array(shard["bev_feature"][local_index], copy=True),
                        np.array(shard["ego_status"][local_index], dtype=np.float32, copy=True),
                    )
            raise IndexError(f"Cache index {index} outside shard offsets")
        return (
            np.array(self.bev_grid[index], copy=True),
            np.array(self.bev_feature[index], copy=True),
            np.array(self.ego_status[index], dtype=np.float32, copy=True),
        )

    def _fit_ego_status(self, ego_status):
        ego_status = np.asarray(ego_status, dtype=np.float32)
        cur_dim = ego_status.shape[-1]
        target_dim = int(getattr(self, "ego_input_dim", cur_dim))
        if cur_dim == target_dim:
            return ego_status
        if cur_dim > target_dim:
            return ego_status[..., :target_dim]
        pad = np.zeros((*ego_status.shape[:-1], target_dim - cur_dim), dtype=np.float32)
        return np.concatenate([ego_status, pad], axis=-1)

    def compute_trajectory(self, agent_input, scene):
        token, index = self._lookup_token(scene)
        if index is None:
            return self._fallback_trajectory(agent_input, token)

        bev_grid_np, bev_feature_np, ego_status_np = self._get_features(index)
        bev_grid = torch.from_numpy(bev_grid_np).unsqueeze(0).to(self.device).float()
        bev_feature = torch.from_numpy(bev_feature_np).unsqueeze(0).to(self.device).float()
        ego_status = torch.from_numpy(self._fit_ego_status(ego_status_np)).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            torch.manual_seed(self._seed_for_token(token))
            if isinstance(self.model, NavSimJointRouteSpeedDiffusion):
                pred_xy = self.model.sample(
                    bev_grid,
                    ego_status,
                    bev_feature=bev_feature,
                    return_trajectory=True,
                    stochastic=self.sampler_stochastic,
                )[0].cpu().numpy()
            else:
                try:
                    pred_xy = self.model.sample(
                        bev_grid,
                        ego_status,
                        return_trajectory=True,
                        stochastic=self.sampler_stochastic,
                    )[0].cpu().numpy()
                except TypeError:
                    pred_xy = self.model.sample(bev_grid, ego_status, return_trajectory=True)[0].cpu().numpy()
        return Trajectory(self._xy_to_se2(pred_xy), self._trajectory_sampling)
