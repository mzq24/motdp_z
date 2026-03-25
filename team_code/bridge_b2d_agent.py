"""
Bridge Baseline closed-loop agent for Bench2Drive.

Uses BDBaselinePolicyV2 (DDBM bridge model) with TransFuser backbone
for online BEV feature extraction.

Model config:  bridge_baseline/bd_config.yaml
Checkpoint:    defined in yaml's training.checkpoint_dir
"""
import os
import sys
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import imageio
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

project_root = str(pathlib.Path(__file__).parent.parent.parent)
leaderboard_root = str(os.path.join(project_root, 'leaderboard'))
scenario_runner_root = str(os.path.join(project_root, 'scenario_runner'))
mot_dp_root = str(os.path.join(project_root, 'MoT-DP'))
carla_api_root = str(os.path.join(project_root.replace('Bench2Drive', 'carla'), 'PythonAPI', 'carla'))

for path in [project_root, leaderboard_root, scenario_runner_root, mot_dp_root, carla_api_root]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

sys.path = [str(p) for p in sys.path]

from leaderboard.autoagents import autonomous_agent
from team_code.simlingo.nav_planner import RoutePlanner, LateralPIDController, get_throttle
from team_code.simlingo.birds_eye_view.run_stop_sign import RunStopSign
from agents.navigation.local_planner import RoadOption
import team_code.simlingo.transfuser_utils as t_u
from team_code.render import render, render_self_car, render_waypoints
from dataset.generate_lidar_bev_b2d import generate_lidar_bev_images
from scipy.interpolate import PchipInterpolator
import xml.etree.ElementTree as ET
from scipy.optimize import fsolve
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# TransFuser backbone for online BEV feature extraction
from model.transfuser_extractor.backbone_extractor import TransFuserBackboneExtractor
from model.transfuser_extractor.config import GlobalConfig as TransfuserConfig
import model.transfuser_extractor.transfuser_utils as transfuser_t_u

# Bridge Baseline policy
# __file__ resolves to the symlink path (Bench2Drive side), so go up 4 levels to reach
# /media/.../others/, then join MoT-DP — same pattern as mot_b2d_agent.py
_agent_dir = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # leaderboard/
mot_dp_path = str(os.path.join(os.path.dirname(os.path.dirname(_agent_dir)), 'MoT-DP'))
if mot_dp_path not in sys.path:
    sys.path.insert(0, mot_dp_path)
from bridge_baseline.policy import BDBaselinePolicyV2

# TransfuserData for lidar histogram
import importlib.util
team_code_transfuser_path = os.path.join(mot_dp_path, 'team_code', 'team_code_transfuser')
if team_code_transfuser_path not in sys.path:
    sys.path.insert(0, team_code_transfuser_path)

_transfuser_data_spec = importlib.util.spec_from_file_location(
    "transfuser_data_module",
    os.path.join(team_code_transfuser_path, "data.py")
)
_transfuser_data_module = importlib.util.module_from_spec(_transfuser_data_spec)
_transfuser_data_spec.loader.exec_module(_transfuser_data_module)
TransfuserData = _transfuser_data_module.CARLA_Data

from team_code.lidar_utils import lidar_to_ego_coordinate, algin_lidar
from team_code.ukf_utils import (
    bicycle_model_forward, measurement_function_hx,
    state_mean, measurement_mean,
    residual_state_x, residual_measurement_h
)

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
PLANNER_TYPE = os.environ.get('PLANNER_TYPE', None)
EARTH_RADIUS_EQUA = 6378137.0
USE_UKF = True


def get_entry_point():
    return 'BridgeAgent'


def create_carla_config(config_path=None):
    """Load config from yaml. config_path is passed from TEAM_CONFIG via setup()."""
    if config_path is None:
        config_path = "/media/z/data/mzq/others/MoT-DP/bridge_baseline/bd_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_bridge_model(checkpoint_path, config, device):
    """Load BDBaselinePolicyV2 from checkpoint."""
    print(f"Loading bridge model from: {checkpoint_path}")
    if not hasattr(np, '_core'):
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    policy = BDBaselinePolicyV2(config)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    policy.load_state_dict(state_dict)

    epoch = checkpoint.get('epoch', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')
    del checkpoint
    import gc; gc.collect()

    policy = policy.to(device)
    policy.eval()

    print(f"✓ Bridge model loaded! Epoch: {epoch}"
          + (f", val_loss: {val_loss:.4f}" if isinstance(val_loss, float) else ""))
    return policy


class BridgeAgent(autonomous_agent.AutonomousAgent):

    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        if IS_BENCH2DRIVE:
            self.save_name = path_to_conf_file.split('+')[-1]
            self.config_path = path_to_conf_file.split('+')[0]
        else:
            now = datetime.datetime.now()
            self.config_path = path_to_conf_file
            self.save_name = '_'.join(map(lambda x: '%02d' % x,
                                          (now.month, now.day, now.hour, now.minute, now.second)))
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        import gc

        # Load config
        print("Loading bridge config...")
        self.config = create_carla_config(self.config_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load bridge model
        checkpoint_dir = self.config.get('training', {}).get(
            'checkpoint_dir',
            "/media/z/data/mzq/others/MoT-DP/checkpoints/bridge_traj_diffsped_20"
        )
        checkpoint_path = os.path.join(checkpoint_dir, "bd_baseline_best.pt")
        self.net = load_bridge_model(checkpoint_path, self.config, device)
        print("✓ Bridge policy loaded.")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[GPU Memory] After bridge model: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

        # Load TransFuser backbone for online BEV extraction
        print("Loading TransFuser backbone...")
        transfuser_config_path = "/media/z/data/models/garage2/pretrained_models/all_towns"
        transfuser_model_path = os.path.join(transfuser_config_path, "model_0030_1.pth")
        self.transfuser_backbone = TransFuserBackboneExtractor(
            config_path=transfuser_config_path,
            model_path=transfuser_model_path,
            device='cuda:0'
        )
        self.transfuser_backbone.eval()
        self.transfuser_config = self.transfuser_backbone.config
        self.transfuser_data = TransfuserData(root=[], config=self.transfuser_config, shared_dict=None)
        print("✓ TransFuser backbone loaded.")

        self.transfuser_lidar_buffer = deque(maxlen=self.transfuser_config.lidar_seq_len * self.transfuser_config.data_save_freq)
        self.transfuser_lidar_last = None
        self.transfuser_state_log = deque(maxlen=max((self.transfuser_config.lidar_seq_len * self.transfuser_config.data_save_freq), 2))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # PID controllers
        self.turn_controller = LateralPIDController(
            inference_mode=False,
            k_p=3.118,
            speed_offset=1.195,
            default_lookahead=24
        )
        self.speed_controller = t_u.PIDController(k_p=1.75, k_i=1.0, k_d=2.0, n=20)

        # Control config
        self.carla_fps = 20
        self.brake_speed = 0.4
        self.brake_ratio = 1.1
        self.clip_delta = 1.0
        self.clip_throttle = 1.0
        self.stuck_threshold = 300
        self.creep_duration = 15
        self.creep_throttle = 0.4

        # Stuck detection
        self.stuck_detector = 0
        self.force_move = 0

        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steers = 0

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.save_path = None
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.lat_ref, self.lon_ref = 42.0, 2.0
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        self.prev_control = control
        self.control = control

        # UKF
        self.carla_frame_rate = 1.0 / 20.0
        if USE_UKF:
            self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
            self.ukf = UKF(dim_x=4, dim_z=4,
                           fx=bicycle_model_forward, hx=measurement_function_hx,
                           dt=self.carla_frame_rate, points=self.points,
                           x_mean_fn=state_mean, z_mean_fn=measurement_mean,
                           residual_x=residual_state_x, residual_z=residual_measurement_h)
            self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
            self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
            self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])
            self.filter_initialized = False
            self.state_log = deque(maxlen=20)

        if SAVE_PATH is not None:
            string = self.save_name
            print(string)

        self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
        self.save_path.mkdir(parents=True, exist_ok=False)
        (self.save_path / 'rgb_front').mkdir()
        (self.save_path / 'meta').mkdir()
        (self.save_path / 'bev').mkdir()

        # History buffers for bridge obs (obs_horizon=4, sampled every 10 frames)
        self.obs_horizon = self.config.get('obs_horizon', 4)
        buf_len = self.obs_horizon * 10
        self.speed_history = deque(maxlen=buf_len)
        self.next_command_history = deque(maxlen=buf_len)
        self.target_point_history = deque(maxlen=buf_len)
        self.next_target_point_history = deque(maxlen=buf_len)

        # LiDAR buffers
        self.lidar_buffer = deque(maxlen=2)
        self.last_ego_transform = None
        self.last_lidar = None

        # Visualization
        self.last_pred_traj = None

    def _init(self):
        try:
            world_map = CarlaDataProvider.get_map()
            xodr = world_map.to_opendrive()
            tree = ET.ElementTree(ET.fromstring(xodr))
            self.lat_ref = 42.0
            self.lon_ref = 2.0
            for opendrive in tree.iter('OpenDRIVE'):
                for header in opendrive.iter('header'):
                    for georef in header.iter('geoReference'):
                        if georef.text:
                            for item in georef.text.split(' '):
                                if '+lat_0' in item:
                                    self.lat_ref = float(item.split('=')[1])
                                if '+lon_0' in item:
                                    self.lon_ref = float(item.split('=')[1])
        except Exception:
            try:
                locx = self._global_plan_world_coord[0][0].location.x
                locy = self._global_plan_world_coord[0][0].location.y
                lon = self._global_plan[0][0]['lon']
                lat = self._global_plan[0][0]['lat']
                def equations(variables):
                    x, y = variables
                    eq1 = (lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (math.pi * EARTH_RADIUS_EQUA)
                           - math.cos(x * math.pi / 180.0) * y)
                    eq2 = (math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * EARTH_RADIUS_EQUA
                           * math.cos(x * math.pi / 180.0) + locy - math.cos(x * math.pi / 180.0) * EARTH_RADIUS_EQUA
                           * math.log(math.tan((90.0 + x) * math.pi / 360.0)))
                    return [eq1, eq2]
                solution = fsolve(equations, [0.0, 0.0])
                self.lat_ref, self.lon_ref = solution[0], solution[1]
            except Exception:
                self.lat_ref, self.lon_ref = 0.0, 0.0

        self.route_planner_min_distance = 7.5
        self.route_planner_max_distance = 50.0
        self._route_planner = RoutePlanner(self.route_planner_min_distance, self.route_planner_max_distance,
                                           self.lat_ref, self.lon_ref)
        self._route_planner.set_route(self._global_plan_world_coord, gps=False)

        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]
        self.last_command = -1
        self.last_command_tmp = -1

        world = CarlaDataProvider.get_world()
        self.stop_sign_criteria = RunStopSign(world)

        self.initialized = True
        self.metric_info = {}

    def sensors(self):
        sensors = [
            {'type': 'sensor.camera.rgb',
             'x': -1.50, 'y': 0.0, 'z': 2.0,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 1024, 'height': 512, 'fov': 110,
             'id': 'CAM_FRONT'},
            {'type': 'sensor.lidar.ray_cast',
             'x': 0.0, 'y': 0.0, 'z': 2.5,
             'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
             'id': 'LIDAR'},
            {'type': 'sensor.other.imu',
             'x': 0.0, 'y': 0.0, 'z': 0.0,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'sensor_tick': 0.05, 'id': 'IMU'},
            {'type': 'sensor.other.gnss',
             'x': 0.0, 'y': 0.0, 'z': 0.0,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'sensor_tick': 0.01, 'id': 'GPS'},
            {'type': 'sensor.speedometer',
             'reading_frequency': 20, 'id': 'SPEED'},
        ]
        if IS_BENCH2DRIVE:
            sensors += [{'type': 'sensor.camera.rgb',
                          'x': 0.0, 'y': 0.0, 'z': 50.0,
                          'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                          'width': 512, 'height': 512, 'fov': 5 * 10.0,
                          'id': 'bev'}]
        return sensors

    def tick(self, input_data):
        self.step += 1
        rgb_front = cv2.cvtColor(input_data['CAM_FRONT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        lidar_ego = lidar_to_ego_coordinate(input_data['LIDAR'])

        gps_full = input_data['GPS'][1]
        gps_pos = self._route_planner.convert_gps_to_carla(gps_full)

        compass_raw = input_data['IMU'][1][-1]
        if math.isnan(compass_raw):
            compass_raw = 0.0
        compass = t_u.preprocess_compass(compass_raw)
        speed = input_data['SPEED'][1]['speed']

        if USE_UKF:
            if not self.filter_initialized:
                self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
                self.filter_initialized = True
            self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
            self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
            filtered_state = self.ukf.x
            self.state_log.append(filtered_state)
            gps_filtered = filtered_state[0:2]
            compass_filtered = filtered_state[2]
        else:
            gps_filtered = np.array([gps_pos[0], gps_pos[1]])
            compass_filtered = compass

        # Combine two frames of LiDAR
        if self.last_lidar is not None and self.last_ego_transform is not None:
            current_pos = np.array([gps_filtered[0], gps_filtered[1], 0.0])
            last_pos = np.array([self.last_ego_transform['gps'][0], self.last_ego_transform['gps'][1], 0.0])
            relative_translation = current_pos - last_pos
            current_yaw = compass_filtered
            last_yaw = self.last_ego_transform['compass']
            relative_rotation = current_yaw - last_yaw
            rotation_matrix = np.array([[np.cos(current_yaw), -np.sin(current_yaw), 0.0],
                                         [np.sin(current_yaw), np.cos(current_yaw), 0.0],
                                         [0.0, 0.0, 1.0]])
            relative_translation_local = rotation_matrix.T @ relative_translation
            lidar_last = algin_lidar(self.last_lidar, relative_translation_local, relative_rotation)
            lidar_combined = np.concatenate((lidar_ego, lidar_last), axis=0)
        else:
            lidar_combined = lidar_ego

        self.last_lidar = lidar_ego
        self.last_ego_transform = {'gps': gps_filtered, 'compass': compass_filtered}

        # TransFuser-style processing
        transfuser_rgb = input_data['CAM_FRONT'][1][:, :, :3]
        _, compressed_image = cv2.imencode('.jpg', transfuser_rgb)
        transfuser_rgb = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
        transfuser_rgb = cv2.cvtColor(transfuser_rgb, cv2.COLOR_BGR2RGB)
        transfuser_rgb = transfuser_t_u.crop_array(self.transfuser_config, transfuser_rgb)
        transfuser_rgb = np.transpose(transfuser_rgb, (2, 0, 1))
        transfuser_rgb_tensor = torch.from_numpy(transfuser_rgb).float().unsqueeze(0).to('cuda')

        transfuser_lidar = transfuser_t_u.lidar_to_ego_coordinate(self.transfuser_config, input_data['LIDAR'])
        self.transfuser_state_log.append([gps_filtered[0], gps_filtered[1], compass_filtered, speed])

        if self.transfuser_lidar_last is not None and len(self.transfuser_state_log) >= 2:
            ego_x, ego_y, ego_theta = [self.transfuser_state_log[-1][i] for i in range(3)]
            ego_x_last, ego_y_last, ego_theta_last = [self.transfuser_state_log[-2][i] for i in range(3)]
            transfuser_lidar_last_aligned = self._align_lidar_transfuser(
                self.transfuser_lidar_last, ego_x_last, ego_y_last, ego_theta_last,
                ego_x, ego_y, ego_theta
            )
            transfuser_lidar_full = np.concatenate((transfuser_lidar, transfuser_lidar_last_aligned), axis=0)
        else:
            transfuser_lidar_full = transfuser_lidar

        self.transfuser_lidar_last = transfuser_lidar.copy()
        self.transfuser_lidar_buffer.append(transfuser_lidar_full)

        transfuser_lidar_bev = self.transfuser_data.lidar_to_histogram_features(
            transfuser_lidar_full, use_ground_plane=self.transfuser_config.use_ground_plane
        )
        transfuser_lidar_bev_tensor = torch.from_numpy(transfuser_lidar_bev).float().unsqueeze(0).to('cuda')

        if IS_BENCH2DRIVE:
            bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            bev = np.zeros((512, 512, 3), dtype=np.uint8)

        result = {
            'rgb_front': rgb_front,
            'gps': gps_filtered,
            'speed': speed,
            'compass': compass_filtered,
            'bev': bev,
            'transfuser_rgb': transfuser_rgb_tensor,
            'transfuser_lidar_bev': transfuser_lidar_bev_tensor,
        }

        waypoint_route = self._route_planner.run_step(np.append(result['gps'], gps_pos[2]))

        # Use the farthest waypoint (waypoint_route[-1]) as target_point for stability.
        # Close waypoints (e.g. waypoint_route[1] at ~7.5m) are too sensitive to GPS noise,
        # causing target_point angle to jitter. The farthest point (~50m) is much more stable.
        # This follows the proven HPC agent convention.
        if len(waypoint_route) > 0:
            target_point, far_command = waypoint_route[-1]
            next_target_point = waypoint_route[-1][0]
        else:
            target_point, far_command = (result['gps'][:2], RoadOption.LANEFOLLOW)
            next_target_point = result['gps'][:2]

        if self.last_command_tmp != far_command:
            self.last_command = self.last_command_tmp
        self.last_command_tmp = far_command

        if hasattr(target_point, '__iter__') and len(target_point) >= 2:
            if (target_point[:2] != self.target_point_prev[:2]).any() if isinstance(target_point, np.ndarray) else (list(target_point[:2]) != list(self.target_point_prev[:2])):
                self.target_point_prev = target_point
                self.commands.append(far_command.value)

        result['next_command'] = self.commands[-2]
        ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])
        ego_next_target_point = t_u.inverse_conversion_2d(next_target_point[:2], result['gps'], result['compass'])
        result['target_point'] = ego_target_point
        result['next_target_point'] = ego_next_target_point
        result['theta'] = compass_filtered

        return result

    def _align_lidar_transfuser(self, lidar, x, y, orientation, x_target, y_target, orientation_target):
        pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
        rot_diff = transfuser_t_u.normalize_angle(orientation_target - orientation)
        rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                     [np.sin(orientation_target), np.cos(orientation_target), 0.0],
                                     [0.0, 0.0, 1.0]])
        pos_diff = rotation_matrix.T @ pos_diff
        return transfuser_t_u.algin_lidar(lidar, pos_diff, rot_diff)

    def interpolate_waypoints(self, waypoints):
        waypoints = waypoints.copy()
        waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
        shift = np.roll(waypoints, 1, axis=0)
        shift[0] = shift[1]
        dists = np.linalg.norm(waypoints - shift, axis=1)
        dists = np.cumsum(dists)
        dists += np.arange(0, len(dists)) * 1e-4
        interp = PchipInterpolator(dists, waypoints, axis=0)
        x = np.arange(0.1, dists[-1], 0.1)
        interp_points = interp(x)
        if interp_points.shape[0] == 0:
            interp_points = waypoints[None, -1]
        return interp_points

    def control_pid(self, route_waypoints, velocity, speed_waypoints):
        """PID control from route waypoints. route_waypoints: (1, N, 2) tensor."""
        assert route_waypoints.size(0) == 1
        route_waypoints_np = route_waypoints[0].data.cpu().numpy()
        speed_waypoints_np = speed_waypoints[0].data.cpu().numpy()

        # Desired speed from waypoint displacement (0.5s to 1.0s)
        one_second_idx, half_second_idx = 2, 0
        if speed_waypoints_np.shape[0] >= 3:
            desired_speed = np.linalg.norm(speed_waypoints_np[one_second_idx] - speed_waypoints_np[half_second_idx])
        else:
            desired_speed = np.linalg.norm(speed_waypoints_np[0]) * 2.0

        brake = (desired_speed < self.brake_speed) or ((velocity / max(desired_speed, 1e-5)) > self.brake_ratio)
        throttle, brake = get_throttle(brake, desired_speed, velocity)

        route_interp = self.interpolate_waypoints(route_waypoints_np)
        steer = self.turn_controller.step(route_interp, velocity)
        steer = float(np.clip(steer, -1.0, 1.0))
        steer = round(steer, 3)

        return steer, throttle, brake

    def _build_bridge_obs(self, transfuser_bev_feature_upsample):
        """Build obs_dict for BDBaselinePolicyV2 from history buffers."""
        speed_hist = list(self.speed_history)
        cmd_hist = list(self.next_command_history)
        tp_hist = list(self.target_point_history)
        ntp_hist = list(self.next_target_point_history)

        # Sample every 10 frames, obs_horizon=4, reverse to chronological
        def _sample(buf):
            return [buf[-1 - i * 10] for i in range(self.obs_horizon)
                    if -1 - i * 10 >= -len(buf)][::-1]

        speed_list = _sample(speed_hist)    # each: (1, 1)
        cmd_list   = _sample(cmd_hist)      # each: (1, 6)
        tp_list    = _sample(tp_hist)       # each: (1, 2)
        ntp_list   = _sample(ntp_hist)      # each: (1, 2)

        # Stack and squeeze to (1, T, dim)
        speed_stacked = torch.cat(speed_list, dim=0).unsqueeze(0)   # (1, T, 1)
        cmd_stacked   = torch.cat(cmd_list,   dim=0).unsqueeze(0)   # (1, T, 6)
        tp_stacked    = torch.cat(tp_list,    dim=0).unsqueeze(0)   # (1, T, 2)
        ntp_stacked   = torch.cat(ntp_list,   dim=0).unsqueeze(0)   # (1, T, 2)

        # Bridge model expects speed as (B, T) — squeeze last dim
        speed_stacked = speed_stacked.squeeze(-1)  # (1, T)

        return {
            'transfuser_bev_feature_upsample': transfuser_bev_feature_upsample,
            'speed':                  speed_stacked,
            'command_hist':           cmd_stacked,
            'target_point_hist':      tp_stacked,
            'target_point_next_hist': ntp_stacked,
        }

    def get_metric_info(self):
        vehicle = CarlaDataProvider.get_hero_actor()
        transform = vehicle.get_transform()
        vel = vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        return {
            'x': transform.location.x,
            'y': transform.location.y,
            'speed': speed,
        }

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)

        one_hot_command = t_u.command_to_one_hot(self.commands[-2])
        cmd_one_hot = torch.from_numpy(one_hot_command[np.newaxis]).to('cuda', dtype=torch.float32)
        command = tick_data['next_command']
        if command < 0:
            command = 4
        command -= 1

        speed = torch.FloatTensor([float(tick_data['speed'])]).view(1, 1).to('cuda', dtype=torch.float32)
        target_point = torch.from_numpy(tick_data['target_point']).unsqueeze(0).float().to('cuda', dtype=torch.float32)
        next_target_point = torch.from_numpy(tick_data['next_target_point']).unsqueeze(0).float().to('cuda', dtype=torch.float32)

        # Append to history buffers (every frame)
        self.speed_history.append(speed)
        self.next_command_history.append(cmd_one_hot)
        self.target_point_history.append(target_point)
        self.next_target_point_history.append(next_target_point)

        gt_velocity = tick_data['speed']

        # Buffer warmup phase: need at least obs_horizon*10 frames
        BUFFER_PHASE = self.obs_horizon * 10 + 1

        if self.step < BUFFER_PHASE:
            control = self.prev_control
            self.pid_metadata = {'agent': 'warmup_phase', 'step': self.step}
        else:
            # Extract TransFuser BEV features
            with torch.no_grad():
                transfuser_output = self.transfuser_backbone(
                    rgb=tick_data['transfuser_rgb'].to(torch.float32),
                    lidar_bev=tick_data['transfuser_lidar_bev'].to(torch.float32)
                )
            transfuser_bev_feature_upsample = transfuser_output['bev_feature_upscale']  # (1, 64, 64, 64)

            # Build bridge obs_dict
            bd_obs_dict = self._build_bridge_obs(transfuser_bev_feature_upsample)

            # Run bridge model inference
            bd_pred = self.net.predict_action(bd_obs_dict)
            action = bd_pred['action']  # (1, N, 2) numpy, N=6 (predict_traj) or 10 (route)
            self.last_pred_traj = action.squeeze(0).copy()

            # Use action as both route (steering) and speed waypoints
            route_waypoints = torch.from_numpy(action).float()   # (1, N, 2)
            speed_waypoints = torch.from_numpy(action).float()   # (1, N, 2)

            steer, throttle, brake = self.control_pid(route_waypoints, gt_velocity, speed_waypoints)

            # Stuck detection
            if gt_velocity < 0.1:
                self.stuck_detector += 1
            elif gt_velocity >= 1.0:
                self.stuck_detector = 0

            if self.stuck_detector > self.stuck_threshold:
                self.force_move = self.creep_duration

            if self.force_move > 0:
                throttle = max(self.creep_throttle, throttle)
                brake = False
                self.force_move -= 1

            # Traffic light stop (CARLA API)
            vehicle = CarlaDataProvider.get_hero_actor()
            if vehicle.is_at_traffic_light():
                tl = vehicle.get_traffic_light()
                tl_state = tl.get_state()
                if tl_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
                    throttle = 0.0
                    brake = 1.0
                    self.stuck_detector = 0

            # Stop sign
            self.stop_sign_criteria.tick(vehicle)
            if self.stop_sign_criteria.target_stop_sign is not None and not self.stop_sign_criteria.stop_completed:
                throttle = 0.0
                brake = 1.0
                self.stuck_detector = 0

            # Speed limit: 35 km/h cap
            if gt_velocity * 3.6 > 35:
                throttle = 0.0
                brake = 1.0

            control = carla.VehicleControl()
            control.steer = float(steer)
            control.throttle = float(throttle)
            control.brake = float(brake)

            self.pid_metadata = {
                'agent': 'bridge',
                'steer': control.steer,
                'throttle': control.throttle,
                'brake': control.brake,
                'speed': gt_velocity,
                'command': command,
            }

            self.prev_control = control
            self.control = control
            self.metric_info[self.step] = self.get_metric_info()

            if SAVE_PATH is not None:
                self.save(tick_data)

        return control

    def save(self, tick_data):
        frame = self.step
        Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))

        bev_img = tick_data['bev'].copy()
        if self.last_pred_traj is not None:
            bev_img = self._draw_traj_on_bev(bev_img, self.last_pred_traj)
        Image.fromarray(bev_img).save(self.save_path / 'bev' / ('%04d.png' % frame))

        with open(self.save_path / 'meta' / ('%04d.json' % frame), 'w') as f:
            json.dump(self.pid_metadata, f, indent=4)

        with open(self.save_path / 'metric_info.json', 'w') as f:
            json.dump(self.metric_info, f, indent=4)

    def _draw_traj_on_bev(self, bev_img, traj):
        """Draw predicted trajectory on BEV image (simple version)."""
        img_h, img_w = bev_img.shape[:2]
        fov_deg = 50.0
        fov_rad = math.radians(fov_deg)
        meters_per_pixel = (2 * 50.0 * math.tan(fov_rad / 2)) / img_w
        cx, cy = img_w // 2, img_h // 2

        for pt in traj:
            x_ego, y_ego = float(pt[0]), float(pt[1])
            # ego frame: x=forward, y=left. BEV: up=forward, right=positive
            px = int(cx - y_ego / meters_per_pixel)
            py = int(cy - x_ego / meters_per_pixel)
            if 0 <= px < img_w and 0 <= py < img_h:
                cv2.circle(bev_img, (px, py), 3, (255, 0, 0), -1)

        return bev_img
