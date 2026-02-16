"""
Scene-level cached CARLA dataset for HPC environments.

Instead of loading per-sample pkl files + heavy transfuser features (~1.4MB each),
this dataset:
  1. Enumerates raw scenes (routes) from the dataset directory
  2. Loads lightweight raw data: BEV semantics (~2KB), measurements (~1KB), boxes (~2KB)
  3. Loads raw RGB images + LiDAR BEV for on-the-fly TransFuser feature extraction
  4. Caches loaded data on local SSD via diskcache to avoid repeated I/O
  5. Reconstructs all training signals (ego_status, behavior labels, etc.) from raw data

The TransFuser backbone runs on GPU in the policy forward pass (not in DataLoader workers).

Usage:
    dataset = CachedCARLADataset(
        raw_data_root='/shared/dataset',
        cache_dir='$SCRATCH/mot_dp_cache',
        split='train',
    )
"""

import os
import sys
import glob
import gzip
import json
import pickle
import re

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


class CachedCARLADataset(torch.utils.data.Dataset):
    """
    Scene-level dataset that loads raw CARLA data with diskcache.

    Raw data per frame:
        - bev_semantics/*.png (256x256 uint8)
        - measurements/*.json.gz (ego state, hazards, etc.)
        - boxes/*.json.gz (3D bounding boxes)
        - rgb/*.jpg (only for validation visualization)

    From measurements we reconstruct:
        - ego_waypoints (from ego_matrix across frames)
        - speed_hist, theta_hist, command_hist, waypoints_hist
        - target_point, route, etc.
    """

    def __init__(
        self,
        raw_data_root: str,
        split: str = 'train',
        mode: str = 'train',
        cache_dir: str = None,
        cache_size_limit_gb: int = 100,
        obs_horizon: int = 4,
        pred_horizon: int = 6,
        skip_first_n_frames: int = 3,
        anchor_centers_abs: np.ndarray = None,
        semantic_behavior_cfg: dict = None,
        val_towns: list = None,
        prefetch_scene: bool = True,
        transfuser_config_path: str = None,
    ):
        self.raw_data_root = raw_data_root
        self.split = split
        self.mode = mode
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.skip_first_n_frames = skip_first_n_frames
        self.prefetch_scene = prefetch_scene

        # --- TransFuser config for raw image/LiDAR preprocessing ---
        self.tf_config = None
        if transfuser_config_path is not None:
            self.tf_config = self._load_transfuser_config(transfuser_config_path)
            print(f"[CachedCARLADataset] TransFuser config loaded: "
                  f"RGB crop=({self.tf_config.cropped_height},{self.tf_config.cropped_width}), "
                  f"LiDAR=({self.tf_config.lidar_resolution_height},{self.tf_config.lidar_resolution_width})")

        # Semantic behavior labeling
        self.anchor_centers_abs = anchor_centers_abs
        self.semantic_behavior_enabled = (
            anchor_centers_abs is not None
            and semantic_behavior_cfg is not None
            and semantic_behavior_cfg.get('enabled', False)
        )
        self.semantic_behavior_cfg = semantic_behavior_cfg or {}

        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 928)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # --- Setup diskcache ---
        self.data_cache = None
        if cache_dir is not None:
            # Expand environment variables like $SCRATCH
            cache_dir = os.path.expandvars(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            try:
                from diskcache import Cache
                self.data_cache = Cache(
                    directory=cache_dir,
                    size_limit=int(cache_size_limit_gb * 1024**3),
                )
                print(f"[CachedCARLADataset] diskcache at {cache_dir} "
                      f"(limit={cache_size_limit_gb}GB)")
            except ImportError:
                print("[CachedCARLADataset] WARNING: diskcache not installed, "
                      "running without cache. Install with: pip install diskcache")

        # --- Enumerate routes (scenes) ---
        if val_towns is None:
            val_towns = [13]  # Default: Town13 for validation

        self.routes = []       # List of route directory paths
        self.route_frames = [] # Number of valid frames per route
        self.samples = []      # List of (route_idx, frame_id) tuples

        scenario_dirs = sorted(glob.glob(os.path.join(raw_data_root, '*')))
        for scenario_dir in scenario_dirs:
            if not os.path.isdir(scenario_dir):
                continue
            route_dirs = sorted(glob.glob(os.path.join(scenario_dir, '*')))
            for route_dir in route_dirs:
                if not os.path.isdir(route_dir):
                    continue
                route_name = os.path.basename(route_dir)

                # Extract town number for train/val split
                town_match = re.search(r'Town(\d+)', route_name)
                if town_match:
                    town = int(town_match.group(1))
                    if split == 'val' and town not in val_towns:
                        continue
                    elif split == 'train' and town in val_towns:
                        continue

                # Check required subdirectories
                meas_dir = os.path.join(route_dir, 'measurements')
                bev_dir = os.path.join(route_dir, 'bev_semantics')
                if not os.path.isdir(meas_dir) or not os.path.isdir(bev_dir):
                    continue

                # Count frames from measurements directory
                meas_files = sorted(glob.glob(os.path.join(meas_dir, '*.json.gz')))
                num_frames = len(meas_files)
                if num_frames == 0:
                    continue

                # Need at least obs_horizon past + pred_horizon future frames
                min_frame = self.skip_first_n_frames + self.obs_horizon - 1
                max_frame = num_frames - self.pred_horizon

                if min_frame >= max_frame:
                    continue

                route_idx = len(self.routes)
                self.routes.append(route_dir)
                self.route_frames.append(num_frames)

                for frame_id in range(min_frame, max_frame):
                    self.samples.append((route_idx, frame_id))

        # Store as numpy for memory efficiency (prevent multiprocessing leak)
        self.routes = np.array(self.routes, dtype=np.bytes_)
        self.route_frames = np.array(self.route_frames, dtype=np.int32)
        self.samples = np.array(self.samples, dtype=np.int32)

        print(f"[CachedCARLADataset] {split}: {len(self.samples)} samples "
              f"from {len(self.routes)} routes")

    def __len__(self):
        return len(self.samples)

    def _load_json_gz(self, path):
        """Load a gzipped JSON file, with optional caching."""
        if self.data_cache is not None and path in self.data_cache:
            return self.data_cache[path]

        with gzip.open(path, 'rt', encoding='utf-8') as f:
            data = json.load(f)

        if self.data_cache is not None:
            self.data_cache[path] = data
        return data

    def _load_bev(self, path):
        """Load BEV semantic PNG, with optional caching."""
        cache_key = 'bev:' + path
        if self.data_cache is not None and cache_key in self.data_cache:
            buf = self.data_cache[cache_key]
            return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

        bev = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if self.data_cache is not None and bev is not None:
            _, buf = cv2.imencode('.png', bev)
            self.data_cache[cache_key] = buf
        return bev

    def _load_rgb(self, path):
        """Load RGB image for validation visualization."""
        cache_key = 'rgb:' + path
        if self.data_cache is not None and cache_key in self.data_cache:
            buf = self.data_cache[cache_key]
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        if self.data_cache is not None:
            _, buf = cv2.imencode('.jpg', img)
            self.data_cache[cache_key] = buf

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_transfuser_config(config_path):
        """Load TransFuser config from saved config.json (via jsonpickle)."""
        import jsonpickle
        import jsonpickle.ext.numpy as jsonpickle_numpy
        jsonpickle_numpy.register_handlers()

        sys.path.insert(0, os.path.join(project_root, 'model', 'transfuser_extractor'))
        from config import GlobalConfig

        config_file = os.path.join(config_path, 'config.json')
        with open(config_file, 'rt', encoding='utf-8') as f:
            loaded = jsonpickle.decode(f.read())

        config = GlobalConfig()
        config.__dict__.update(loaded.__dict__)
        return config

    def _load_rgb_for_backbone(self, path):
        """Load and crop RGB image for TransFuser backbone input.

        Returns:
            np.ndarray (H, W, 3) float32 range [0, 255], or None if load fails.
        """
        cache_key = 'rgb_bb:' + path
        if self.data_cache is not None and cache_key in self.data_cache:
            buf = self.data_cache[cache_key]
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop (same as transfuser_utils.crop_array)
        if self.tf_config is not None and self.tf_config.crop_image:
            # crop from top
            crop_h = img.shape[0] - self.tf_config.cropped_height
            img = img[crop_h:, :, :]

        if self.data_cache is not None:
            _, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.data_cache[cache_key] = buf

        return img.astype(np.float32)

    def _load_lidar_bev(self, path):
        """Load LiDAR .laz file and convert to histogram BEV.

        Returns:
            np.ndarray (C, H, W) float32, or None if load fails.
        """
        cache_key = 'lidar:' + path
        if self.data_cache is not None and cache_key in self.data_cache:
            return self.data_cache[cache_key]

        try:
            import laspy
            las = laspy.read(path)
            lidar = las.xyz
        except Exception:
            return None

        cfg = self.tf_config
        bev = self._lidar_to_histogram(lidar, cfg)

        if self.data_cache is not None:
            self.data_cache[cache_key] = bev
        return bev

    @staticmethod
    def _lidar_to_histogram(lidar, cfg):
        """Convert LiDAR point cloud to BEV histogram (matches TransFuser preprocessing)."""
        def splat(pc):
            xbins = np.linspace(cfg.min_x, cfg.max_x,
                                (cfg.max_x - cfg.min_x) * int(cfg.pixels_per_meter) + 1)
            ybins = np.linspace(cfg.min_y, cfg.max_y,
                                (cfg.max_y - cfg.min_y) * int(cfg.pixels_per_meter) + 1)
            hist = np.histogramdd(pc[:, :2], bins=(xbins, ybins))[0]
            hist[hist > cfg.hist_max_per_pixel] = cfg.hist_max_per_pixel
            return (hist / cfg.hist_max_per_pixel).T

        lidar = lidar[lidar[..., 2] < cfg.max_height_lidar]
        below = lidar[lidar[..., 2] <= cfg.lidar_split_height]
        above = lidar[lidar[..., 2] > cfg.lidar_split_height]

        if cfg.use_ground_plane:
            features = np.stack([splat(below), splat(above)], axis=-1)
        else:
            features = np.stack([splat(above)], axis=-1)

        return np.transpose(features, (2, 0, 1)).astype(np.float32)

    def _prefetch_route(self, route_dir, num_frames):
        """Prefetch all frames of a route into cache."""
        if self.data_cache is None:
            return

        # Check if already cached (spot-check first and last frame)
        first_key = os.path.join(route_dir, 'measurements', '0000.json.gz')
        last_key = os.path.join(route_dir, 'measurements', f'{num_frames-1:04d}.json.gz')
        if first_key in self.data_cache and last_key in self.data_cache:
            return  # Already cached

        for fid in range(num_frames):
            frame_str = f'{fid:04d}'

            # Measurements
            meas_path = os.path.join(route_dir, 'measurements', f'{frame_str}.json.gz')
            if os.path.exists(meas_path):
                self._load_json_gz(meas_path)

            # Boxes
            boxes_path = os.path.join(route_dir, 'boxes', f'{frame_str}.json.gz')
            if os.path.exists(boxes_path):
                self._load_json_gz(boxes_path)

            # BEV semantics
            bev_path = os.path.join(route_dir, 'bev_semantics', f'{frame_str}.png')
            if os.path.exists(bev_path):
                self._load_bev(bev_path)

    def _get_waypoints_from_measurements(self, measurements_list):
        """
        Compute waypoints in ego frame from a list of measurements.
        Origin is the first measurement's ego_matrix.

        Args:
            measurements_list: list of measurement dicts (temporal sequence)

        Returns:
            waypoints: (N, 2) array in ego frame [x_forward, y_lateral]
        """
        origin = measurements_list[0]
        origin_matrix = np.array(origin['ego_matrix'])[:3]
        origin_translation = origin_matrix[:, 3:4]
        origin_rotation = origin_matrix[:, :3]

        waypoints = []
        for meas in measurements_list:
            pos = np.array(meas['ego_matrix'])[:3, 3:4]
            wp_ego = origin_rotation.T @ (pos - origin_translation)
            waypoints.append(wp_ego[:2, 0])  # [x_forward, y_lateral]

        return np.array(waypoints)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)

        route_idx, frame_id = self.samples[idx]
        route_dir = str(self.routes[route_idx], encoding='utf-8')
        num_frames = int(self.route_frames[route_idx])

        # Optional: prefetch entire scene on first access
        if self.prefetch_scene:
            self._prefetch_route(route_dir, num_frames)

        # ============ Load measurements for temporal window ============
        # History: [frame_id - obs_horizon + 1, ..., frame_id] (obs_horizon frames)
        # Future:  [frame_id + 1, ..., frame_id + pred_horizon] (pred_horizon frames)

        hist_start = frame_id - self.obs_horizon + 1
        fut_end = frame_id + self.pred_horizon

        all_measurements = []
        for fid in range(hist_start, fut_end + 1):
            meas_path = os.path.join(route_dir, 'measurements', f'{fid:04d}.json.gz')
            meas = self._load_json_gz(meas_path)
            all_measurements.append(meas)

        # Split into history and future
        hist_measurements = all_measurements[:self.obs_horizon]  # obs_horizon frames
        current_meas = hist_measurements[-1]                     # current frame
        future_measurements = all_measurements[self.obs_horizon:]  # pred_horizon frames

        # ============ Reconstruct ego_status components ============

        # Speed history
        speed_hist = np.array([m['speed'] for m in hist_measurements], dtype=np.float64)

        # Theta history (steering angle / heading relative angle)
        theta_hist = np.array([m.get('theta', 0.0) for m in hist_measurements], dtype=np.float64)

        # Command history (one-hot, 6 classes)
        def command_to_onehot(cmd_val):
            oh = np.zeros(6, dtype=np.float64)
            cmd_int = int(cmd_val) if isinstance(cmd_val, (int, float)) else 4
            if 0 <= cmd_int < 6:
                oh[cmd_int] = 1.0
            else:
                oh[4] = 1.0  # default: lane follow
            return oh

        command_hist = np.stack([
            command_to_onehot(m.get('command', 4)) for m in hist_measurements
        ])

        # Waypoints history (position of each history frame in current ego frame)
        waypoints_hist_all = self._get_waypoints_from_measurements(hist_measurements)
        # Relative to current frame (last in history)
        current_origin_meas = [current_meas]  # single element
        waypoints_hist = self._get_waypoints_from_measurements(hist_measurements)
        # Re-center: express all history positions relative to current frame
        origin_matrix = np.array(current_meas['ego_matrix'])[:3]
        origin_translation = origin_matrix[:, 3:4]
        origin_rotation = origin_matrix[:, :3]
        waypoints_hist_ego = []
        for m in hist_measurements:
            pos = np.array(m['ego_matrix'])[:3, 3:4]
            wp = origin_rotation.T @ (pos - origin_translation)
            waypoints_hist_ego.append(wp[:2, 0])
        waypoints_hist = np.array(waypoints_hist_ego, dtype=np.float64)

        # Target point history
        target_point_hist = np.array([
            m.get('target_point', [0.0, 0.0]) for m in hist_measurements
        ], dtype=np.float64)

        # Target point next history
        target_point_next_hist = np.array([
            m.get('target_point_next', m.get('target_point', [0.0, 0.0]))
            for m in hist_measurements
        ], dtype=np.float64)

        # Current target point
        target_point = np.array(
            current_meas.get('target_point', [0.0, 0.0]), dtype=np.float64
        )

        # ============ GT trajectory (future waypoints) ============
        all_for_gt = [current_meas] + future_measurements
        ego_waypoints = self._get_waypoints_from_measurements(all_for_gt)
        # ego_waypoints[0] = origin (0,0), ego_waypoints[1:] = future positions

        # ============ Route ============
        route_raw = current_meas.get('route', None)
        if route_raw is not None:
            route_points = np.array(route_raw, dtype=np.float64)
            # Pad or truncate to 20 points
            if len(route_points) < 20:
                pad = np.tile(route_points[-1:], (20 - len(route_points), 1))
                route_points = np.concatenate([route_points, pad], axis=0)
            else:
                route_points = route_points[:20]
        else:
            route_points = np.zeros((20, 2), dtype=np.float64)

        # ============ Current commands ============
        command = command_to_onehot(current_meas.get('command', 4))
        next_command = command_to_onehot(current_meas.get('next_command', 4))

        # ============ Build sample dict ============
        final_sample = {}
        final_sample['speed'] = torch.from_numpy(speed_hist).float()
        final_sample['theta_hist'] = torch.from_numpy(theta_hist).float()
        final_sample['command_hist'] = torch.from_numpy(command_hist).float()
        final_sample['waypoints_hist'] = torch.from_numpy(waypoints_hist).float()
        final_sample['target_point_hist'] = torch.from_numpy(target_point_hist).float()
        final_sample['target_point_next_hist'] = torch.from_numpy(target_point_next_hist).float()
        final_sample['target_point'] = torch.from_numpy(target_point).float()
        final_sample['agent_pos'] = torch.from_numpy(ego_waypoints[1:]).float()
        final_sample['route'] = torch.from_numpy(route_points).float()
        final_sample['command'] = torch.from_numpy(command).float()
        final_sample['next_command'] = torch.from_numpy(next_command).float()

        # Extra metadata
        route_name = os.path.basename(route_dir)
        scenario_name = os.path.basename(os.path.dirname(route_dir))
        final_sample['town_name'] = scenario_name
        final_sample['route_name'] = route_name
        final_sample['frame_id'] = frame_id

        # ============ Load RGB images (validation only) ============
        if self.mode == 'val':
            images = []
            for fid_hist in range(hist_start, frame_id + 1):
                rgb_path = os.path.join(route_dir, 'rgb', f'{fid_hist:04d}.jpg')
                if os.path.exists(rgb_path):
                    img = self._load_rgb(rgb_path)
                    if img is not None:
                        img_tensor = self.image_transform(img)
                        images.append(img_tensor)
                    else:
                        images.append(torch.zeros(3, 256, 928))
                else:
                    images.append(torch.zeros(3, 256, 928))
            final_sample['image'] = torch.stack(images)

        # ============ Load raw RGB + LiDAR for TransFuser backbone ============
        if self.tf_config is not None:
            frame_str = f'{frame_id:04d}'
            # RGB: (3, H, W) float32 [0, 255]
            rgb_path = os.path.join(route_dir, 'rgb', f'{frame_str}.jpg')
            rgb_raw = self._load_rgb_for_backbone(rgb_path)
            if rgb_raw is not None:
                # (H, W, 3) -> (3, H, W)
                final_sample['rgb_raw'] = torch.from_numpy(
                    np.transpose(rgb_raw, (2, 0, 1)))
            else:
                ch, cw = self.tf_config.cropped_height, self.tf_config.cropped_width
                final_sample['rgb_raw'] = torch.zeros(3, ch, cw)

            # LiDAR BEV: (C, H, W) float32
            lidar_path = os.path.join(route_dir, 'lidar', f'{frame_str}.laz')
            lidar_bev = self._load_lidar_bev(lidar_path)
            if lidar_bev is not None:
                final_sample['lidar_bev'] = torch.from_numpy(lidar_bev)
            else:
                lh = self.tf_config.lidar_resolution_height
                lw = self.tf_config.lidar_resolution_width
                lc = 2 if self.tf_config.use_ground_plane else 1
                final_sample['lidar_bev'] = torch.zeros(lc, lh, lw)

        # ============ Semantic Behavior Labeling ============
        if self.semantic_behavior_enabled:
            from tools.anchor_semantic_labeler import (
                label_anchors_semantic, classify_scene_buckets,
                NUM_BUCKET_CATEGORIES,
            )

            frame_str = f'{frame_id:04d}'
            bev_path = os.path.join(route_dir, 'bev_semantics', f'{frame_str}.png')
            bev_semantic = self._load_bev(bev_path)

            if bev_semantic is not None:
                # Current boxes
                boxes_path = os.path.join(route_dir, 'boxes', f'{frame_str}.json.gz')
                boxes = None
                if os.path.exists(boxes_path):
                    boxes = self._load_json_gz(boxes_path)

                # Current measurements already loaded
                measurements = current_meas
                ego_matrix_current = measurements.get('ego_matrix', None)

                # Future frames for dynamic collision
                future_frames_data = None
                num_anchor_pts = self.anchor_centers_abs.shape[1]
                if ego_matrix_current is not None:
                    future_frames_data = []
                    for k in range(1, num_anchor_pts + 1):
                        fut_fid = frame_id + k
                        fut_boxes_path = os.path.join(
                            route_dir, 'boxes', f'{fut_fid:04d}.json.gz')
                        fut_meas_path = os.path.join(
                            route_dir, 'measurements', f'{fut_fid:04d}.json.gz')
                        if os.path.exists(fut_boxes_path) and os.path.exists(fut_meas_path):
                            fut_boxes = self._load_json_gz(fut_boxes_path)
                            fut_meas = self._load_json_gz(fut_meas_path)
                            fut_ego = fut_meas.get('ego_matrix', None)
                            if fut_ego is not None:
                                future_frames_data.append((fut_boxes, fut_ego))
                            else:
                                future_frames_data.append(None)
                        else:
                            future_frames_data.append(None)

                # GT trajectory for false-positive suppression
                gt_traj = ego_waypoints[1:]  # skip origin

                behavior_labels, allowed_flags, _ = label_anchors_semantic(
                    self.anchor_centers_abs, bev_semantic,
                    ppm=self.semantic_behavior_cfg.get('bev_ppm', 2.0),
                    bev_size=self.semantic_behavior_cfg.get('bev_size', 256),
                    boxes=boxes,
                    measurements=measurements,
                    ego_matrix_current=ego_matrix_current,
                    future_frames_data=future_frames_data,
                    gt_trajectory=gt_traj,
                )
                final_sample['behavior_labels'] = torch.from_numpy(behavior_labels).long()
                final_sample['allowed_flags'] = torch.from_numpy(
                    allowed_flags.astype(np.float32))

                # Scene buckets
                bucket_flags = classify_scene_buckets(
                    measurements=measurements,
                    boxes=boxes,
                    ego_waypoints=gt_traj,
                )
                final_sample['scene_buckets'] = torch.from_numpy(
                    bucket_flags.astype(np.float32))
            else:
                n_modes = self.anchor_centers_abs.shape[0]
                final_sample['behavior_labels'] = torch.zeros(n_modes, dtype=torch.long)
                final_sample['allowed_flags'] = torch.ones(n_modes, dtype=torch.float32)
                from tools.anchor_semantic_labeler import NUM_BUCKET_CATEGORIES
                final_sample['scene_buckets'] = torch.zeros(
                    NUM_BUCKET_CATEGORIES, dtype=torch.float32)

        # ============ Build ego_status ============
        ego_status_components = [
            final_sample['speed'].unsqueeze(-1),           # (obs_horizon, 1)
            final_sample['theta_hist'].unsqueeze(-1),      # (obs_horizon, 1)
            final_sample['command_hist'],                   # (obs_horizon, 6)
            final_sample['target_point_hist'],              # (obs_horizon, 2)
            final_sample['target_point_next_hist'],         # (obs_horizon, 2)
            final_sample['waypoints_hist'],                 # (obs_horizon, 2)
        ]
        final_sample['ego_status'] = torch.cat(ego_status_components, dim=-1)

        return final_sample
