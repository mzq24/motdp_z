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
import random
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
from policy.annealed_energy_guidance_policy import AnnealedEnergyGuidancePolicy
from team_code.simlingo.nav_planner import RoutePlanner, LateralPIDController, get_throttle
from team_code.simlingo.birds_eye_view.run_stop_sign import RunStopSign
from agents.navigation.local_planner import RoadOption
import team_code.simlingo.transfuser_utils as t_u  
from team_code.render import render, render_self_car, render_waypoints
from dataset.generate_lidar_bev_b2d import generate_lidar_bev_images
from scipy.optimize import fsolve
from scipy.interpolate import PchipInterpolator
import xml.etree.ElementTree as ET  
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  

# TransFuser backbone for DP features
from model.transfuser_extractor.backbone_extractor import TransFuserBackboneExtractor
from model.transfuser_extractor.config import GlobalConfig as TransfuserConfig
import model.transfuser_extractor.transfuser_utils as transfuser_t_u

# mot dependencies
project_root = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
mot_dp_path = str(os.path.join(os.path.dirname(os.path.dirname(project_root)), 'MoT-DP'))
mot_path = str(os.path.join(mot_dp_path, 'mot'))
sys.path.append(mot_dp_path)
sys.path.append(mot_path)
sys.path = [str(p) for p in sys.path]

# ===== MoT LLM switch: set False to run DP-only without loading LLM =====
USE_MOT = False

if USE_MOT:
    from transformers import HfArgumentParser
    from dataclasses import dataclass, field
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from safetensors.torch import load_file
    import glob
    from data.reasoning.data_utils import add_special_tokens
    from mot.modeling.automotive import (
        AutoMoTConfig, AutoMoT,
        Qwen3VLTextConfig, Qwen3VLTextModel, Qwen3VLForConditionalGenerationMoT
    )
    from dataset.unified_carla_dataset import CARLAImageDataset
    from policy.annealed_energy_guidance_policy import AnnealedEnergyGuidancePolicy
    from mot.evaluation.inference import InterleaveInferencer
    from transformers import AutoTokenizer

# Import TransfuserData using importlib to avoid conflicts with mot/data
import importlib.util
team_code_transfuser_path = os.path.join(mot_dp_path, 'team_code', 'team_code_transfuser')
# Add team_code_transfuser to sys.path so its internal imports (like gaussian_target) work
if team_code_transfuser_path not in sys.path:
    sys.path.insert(0, team_code_transfuser_path)

_transfuser_data_spec = importlib.util.spec_from_file_location(
    "transfuser_data_module", 
    os.path.join(team_code_transfuser_path, "data.py")
)
_transfuser_data_module = importlib.util.module_from_spec(_transfuser_data_spec)
_transfuser_data_spec.loader.exec_module(_transfuser_data_module)
TransfuserData = _transfuser_data_module.CARLA_Data

# Import utility modules
if USE_MOT:
    from team_code.mot_utils import (
        ModelArguments, InferenceArguments,
        load_model_mot, build_cleaned_prompt_and_modes,
        parse_decision_sequence, split_prompt
    )
from team_code.lidar_utils import lidar_to_ego_coordinate, algin_lidar
from team_code.ukf_utils import (
    bicycle_model_forward, measurement_function_hx,
    state_mean, measurement_mean,
    residual_state_x, residual_measurement_h
)
# from team_code.display_interface import DisplayInterface

# try:
#     import pygame
# except ImportError:
#     raise RuntimeError("cannot import pygame, make sure pygame package is installed")

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
PLANNER_TYPE = os.environ.get('PLANNER_TYPE', None)
EARTH_RADIUS_EQUA = 6378137.0
USE_UKF = True  # Enable Unscented Kalman Filter for GPS/compass smoothing
TARGET_POSE_SOURCE = os.environ.get('TARGET_POSE_SOURCE', 'filtered').lower()
STEER_SIGN_SCALE = float(os.environ.get('STEER_SIGN_SCALE', '1.0'))
TARGET_YAW_SIGN = float(os.environ.get('TARGET_YAW_SIGN', '1.0'))
TARGET_GEOM_YAW_SIGN = float(os.environ.get('TARGET_GEOM_YAW_SIGN', '1.0'))

# Entry point
def get_entry_point():
	return 'MOTAgent'

def create_carla_config(config_path=None):
    """Load CARLA configuration from YAML file."""
    if config_path is None:
        config_path = "/media/z/data/mzq/others/MoT-DP/config/pdm_local_route_b.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    mot_dp_root = os.path.dirname(config_dir)

    def _resolve_path(path_value):
        if not path_value or os.path.isabs(path_value):
            return path_value

        candidate_from_config = os.path.abspath(os.path.join(config_dir, path_value))
        if os.path.exists(candidate_from_config):
            return candidate_from_config

        candidate_from_project = os.path.abspath(os.path.join(mot_dp_root, path_value))
        return candidate_from_project

    for key in ['anchor_path', 'abs_stats_path', 'delta_stats_path', 'global_abs_stats_path']:
        if key in config:
            config[key] = _resolve_path(config.get(key))

    training_cfg = config.get('training', {})
    if 'checkpoint_dir' in training_cfg:
        training_cfg['checkpoint_dir'] = _resolve_path(training_cfg.get('checkpoint_dir'))
    if 'checkpoint_path' in training_cfg:
        training_cfg['checkpoint_path'] = _resolve_path(training_cfg.get('checkpoint_path'))

    logging_cfg = config.get('logging', {})
    if 'checkpoint_dir' in logging_cfg:
        logging_cfg['checkpoint_dir'] = _resolve_path(logging_cfg.get('checkpoint_dir'))
    if 'checkpoint_path' in logging_cfg:
        logging_cfg['checkpoint_path'] = _resolve_path(logging_cfg.get('checkpoint_path'))

    return config

def load_best_model(checkpoint_path, config, device):
    """Load the DP (Diffusion Policy) best model checkpoint."""
    print(f"Loading best model from: {checkpoint_path}")
    if not hasattr(np, '_core'):
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath

    policy, ckpt = AnnealedEnergyGuidancePolicy.load_checkpoint(checkpoint_path, config, device)

    epoch = ckpt.get('epoch', 'N/A')
    val_loss = ckpt.get('val_loss', 'N/A')
    train_loss = ckpt.get('train_loss', 'N/A')

    del ckpt
    import gc
    gc.collect()

    print(f"✓ Model loaded: epoch={epoch}")
    if isinstance(val_loss, float):
        print(f"  - Validation Loss: {val_loss:.4f}")
    if isinstance(train_loss, float):
        print(f"  - Training Loss: {train_loss:.4f}")

    return policy


class MOTAgent(autonomous_agent.AutonomousAgent):
	def get_default_config_path(self):
		return "/media/z/data/mzq/others/MoT-DP/config/pdm_local_route_b.yaml"

	def get_checkpoint_filename(self):
		return "dit_policy_best.pt"

	def resolve_checkpoint_path(self):
		training_cfg = self.config.get('training', {})
		logging_cfg = self.config.get('logging', {})
		checkpoint_path = training_cfg.get('checkpoint_path') or logging_cfg.get('checkpoint_path')
		if checkpoint_path:
			return checkpoint_path

		checkpoint_base_path = training_cfg.get(
			'checkpoint_dir',
			"/media/z/data/mzq/others/MoT-DP/checkpoints/add_noise_multi_infer_trunc20"
		)
		return os.path.join(checkpoint_base_path, self.get_checkpoint_filename())

	def _predict_dp_action(self, dp_obs_dict):
		return self.net.predict_action(dp_obs_dict, no_noise=True)

	def setup(self, path_to_conf_file):
		self.track = autonomous_agent.Track.SENSORS
		if IS_BENCH2DRIVE:
			self.save_name = path_to_conf_file.split('+')[-1]
			self.config_path = path_to_conf_file.split('+')[0]
		else:
			now = datetime.datetime.now()
			self.config_path = path_to_conf_file
			self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		import gc
		
		# Load diffusion policy first (smaller model)
		print("Loading diffusion policy...")
		self.config = create_carla_config(self.config_path)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		checkpoint_path = self.resolve_checkpoint_path()
		self.net = load_best_model(checkpoint_path, self.config, device)
		print("✓ Diffusion policy loaded (float32).")
		
		# Aggressive memory cleanup before loading MoT model
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.synchronize()
		
		# Print GPU memory status
		if torch.cuda.is_available():
			allocated = torch.cuda.memory_allocated() / 1024**3
			reserved = torch.cuda.memory_reserved() / 1024**3
			print(f"[GPU Memory] After DP: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

		# Load MoT model
		if USE_MOT:
			print("Loading MoT model...")
			parser = HfArgumentParser((ModelArguments, InferenceArguments))
			model_args, inference_args = parser.parse_args_into_dataclasses(args=[])
			self.inference_args = inference_args
			self.AutoMoT = load_model_mot(device)
			tokenizer = AutoTokenizer.from_pretrained(model_args.qwen3vl_path)
			tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
			self.AutoMoT.language_model.tokenizer = tokenizer
			self.inferencer = InterleaveInferencer(
				model=self.AutoMoT,
				vae_model=None,
				tokenizer=tokenizer,
				vae_transform=None,
				vit_transform=None,
				new_token_ids=new_token_ids,
				max_num_tokens=inference_args.max_num_tokens,
				visual_gen=True,
				visual_und=True,
			)
			print("✓ MoT model loaded.")
		else:
			print("[USE_MOT=False] Skipping MoT model loading.")

		# ========== Load TransFuser Backbone for DP features ==========
		print("Loading TransFuser backbone for DP features...")
		transfuser_config_path = "/media/z/data/models/garage2/pretrained_models/all_towns"
		transfuser_model_path = os.path.join(transfuser_config_path, "model_0030_1.pth")
		self.transfuser_backbone = TransFuserBackboneExtractor(
			config_path=transfuser_config_path,
			model_path=transfuser_model_path,
			device='cuda:0'
		)
		# Backbone is already frozen in TransFuserBackboneExtractor
		self.transfuser_backbone.eval()
		# Get transfuser config for lidar processing
		self.transfuser_config = self.transfuser_backbone.config
		# Initialize TransfuserData for lidar histogram conversion
		self.transfuser_data = TransfuserData(root=[], config=self.transfuser_config, shared_dict=None)
		print("✓ TransFuser backbone loaded, frozen, and using float32.")
		
		# Initialize transfuser lidar buffer for temporal alignment
		self.transfuser_lidar_buffer = deque(maxlen=self.transfuser_config.lidar_seq_len * self.transfuser_config.data_save_freq)
		self.transfuser_lidar_last = None
		self.transfuser_state_log = deque(maxlen=max((self.transfuser_config.lidar_seq_len * self.transfuser_config.data_save_freq), 2))
		
		# Print GPU memory status
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			allocated = torch.cuda.memory_allocated() / 1024**3
			reserved = torch.cuda.memory_reserved() / 1024**3
			print(f"[GPU Memory] After TransFuser: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

		# route_pred有20个稀疏点(~20m, 1m间距)，插值后约200点(0.1m间距)
		# inference_mode=False: lookahead范围24-105个点，对应2.4m-10.5m前方
		# 使用自定义的LateralPIDController，增加低速时的最小前视距离
		# 默认 speed_offset=1.915, 低速时 lookahead = 0.9755*speed_kmh + 1.915，最小24
		# 增加 speed_offset 和最小lookahead，让低速时看得更远，有助于直行时回正
		self.turn_controller = LateralPIDController(
			inference_mode=False, 
			k_p=3.118,  # 增加P增益，提高回正响应（默认3.118）
			speed_offset=1.195,  # 增加offset，低速时看更远（默认1.915）
			default_lookahead=24  # 增加最小前视距离到4m（默认24=2.4m）
		)
		self.speed_controller = t_u.PIDController(k_p=1.75, k_i=1.0, k_d=2.0, n=20)  
		
		# Control config 
		self.carla_fps = 20
		self.wp_dilation = 1
		self.data_save_freq = 5
		self.brake_speed = 0.4
		self.brake_ratio = 1.1
		self.clip_delta = 1.0
		self.clip_throttle = 1.0
		self.stuck_threshold = 300 #800
		self.stuck_helper_threshold = 100
		self.creep_duration = 15
		self.creep_throttle = 0.4
		
		# Stuck detection
		self.stuck_detector = 0
		self.stuck_helper = 0
		self.force_move = 0

		self.steer_step = 0
		self.last_moving_status = 0
		self.last_moving_step = -1
		self.last_steers = 0
		
		self.takeover = False
		self.stop_time = 0
		self.takeover_time = 0
		self.save_path = None
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
		self.lat_ref, self.lon_ref = 42.0, 2.0
		control = carla.VehicleControl()
		control.steer = 0.0
		control.throttle = 0.0
		control.brake = 0.0	
		self.prev_control = control
		self.control = control  # Store control for UKF prediction
		
		# Initialize Unscented Kalman Filter 
		self.carla_frame_rate = 1.0 / 20.0  # CARLA frame rate
		if USE_UKF:
			self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
			self.ukf = UKF(dim_x=4,
						   dim_z=4,
						   fx=bicycle_model_forward,
						   hx=measurement_function_hx,
						   dt=self.carla_frame_rate,
						   points=self.points,
						   x_mean_fn=state_mean,
						   z_mean_fn=measurement_mean,
						   residual_x=residual_state_x,
						   residual_z=residual_measurement_h)
			# State noise, same as measurement because we initialize with the first measurement later
			self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
			# Measurement noise
			self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
			self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
			# Used to set the filter state equal the first measurement
			self.filter_initialized = False
			# Stores the last filtered positions of the ego vehicle
			self.state_log = deque(maxlen=20)

		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = self.save_name
			print (string)

		self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
		self.save_path.mkdir(parents=True, exist_ok=False)

		(self.save_path / 'rgb_front').mkdir()
		(self.save_path / 'meta').mkdir()
		(self.save_path / 'bev').mkdir()
		(self.save_path / 'lidar_bev').mkdir()
		(self.save_path / 'debug_vis').mkdir()
		
		# Initialize lidar buffer for combining two frames
		self.lidar_buffer = deque(maxlen=2)
		self.lidar_step_counter = 0
		self.last_ego_transform = None
		self.last_lidar = None
		
		obs_horizon = self.config.get('obs_horizon', 4)   
		self.obs_horizon = obs_horizon
		self.lidar_bev_history = deque(maxlen=obs_horizon*10) 
		self.rgb_history = deque(maxlen=obs_horizon*10)
		self.speed_history = deque(maxlen=obs_horizon*10)
		self.theta_history = deque(maxlen=obs_horizon*10)
		self.throttle_history = deque(maxlen=obs_horizon*10)
		self.next_command_history = deque(maxlen=obs_horizon*10)
		self.target_point_history = deque(maxlen=obs_horizon*10)
		self.next_target_point_history = deque(maxlen=obs_horizon*10)  # For DP model input
		self.waypoint_history = deque(maxlen=obs_horizon*10)
		self.throttle_history = deque(maxlen=obs_horizon*10) 
		self.brake_history = deque(maxlen=obs_horizon*10) 
		self.obs_accumulate_counter = 0
		
		# Store predicted trajectory for BEV visualization
		self.last_pred_traj = None  # Store the last predicted trajectory (in ego frame)
		self.last_dp_pred_traj = None  # Store the last DP refined trajectory (in ego frame)
		self.last_target_point = None  # Store the last target point (in ego frame)
		self.last_next_target_point = None  # Store the last next target point (in ego frame)
		self.last_waypoint_route = None  # Store the planner waypoint route (in ego frame)
		self.last_route_pred = None  # Store the last route prediction (20 waypoints for lateral control)
		self.last_energy_debug = {}
		self.prev_debug_planner_xy = None
		self.prev_debug_filtered_xy = None
		self.prev_debug_raw_xy = None
		self.prev_debug_hero_xy = None
		self.last_steer_debug = {}

	def _init(self):
		# Use _global_plan_world_coord directly (already in CARLA coordinates)
		# This avoids the GPS-to-CARLA conversion which can fail when fsolve doesn't converge
		# Get lat_ref/lon_ref from CARLA map directly
		try:
			world_map = CarlaDataProvider.get_map()
			xodr = world_map.to_opendrive()
			tree = ET.ElementTree(ET.fromstring(xodr))
			
			# Default values if not found in OpenDRIVE
			self.lat_ref = 42.0
			self.lon_ref = 2.0
			
			for opendrive in tree.iter('OpenDRIVE'):
				for header in opendrive.iter('header'):
					for georef in header.iter('geoReference'):
						if georef.text:
							str_list = georef.text.split(' ')
							for item in str_list:
								if '+lat_0' in item:
									self.lat_ref = float(item.split('=')[1])
								if '+lon_0' in item:
									self.lon_ref = float(item.split('=')[1])
		except Exception as e:
			# Fallback: try fsolve (might not converge)
			try:
				locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
				lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
				earth_radius_equa = 6378137.0
				def equations(variables):
					x, y = variables
					eq1 = (lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (math.pi * earth_radius_equa)
								 - math.cos(x * math.pi / 180.0) * y)
					eq2 = (math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * earth_radius_equa
								 * math.cos(x * math.pi / 180.0) + locy - math.cos(x * math.pi / 180.0) * earth_radius_equa
								 * math.log(math.tan((90.0 + x) * math.pi / 360.0)))
					return [eq1, eq2]
				initial_guess = [0.0, 0.0]
				solution = fsolve(equations, initial_guess)
				self.lat_ref, self.lon_ref = solution[0], solution[1]
			except Exception as e2:
				self.lat_ref, self.lon_ref = 0.0, 0.0
		

		self.route_planner_min_distance = 7.5
		self.route_planner_max_distance = 50.0
		self._route_planner = RoutePlanner(self.route_planner_min_distance, self.route_planner_max_distance,
										   self.lat_ref, self.lon_ref)
		
		if len(self._global_plan_world_coord) > 0:
			first_wp = self._global_plan_world_coord[0]
		
		# Use _global_plan_world_coord with gps=False (recommended, GPS is deprecated in nav_planner.py)
		self._route_planner.set_route(self._global_plan_world_coord, gps=False)
		
				
		# Initialize command tracking 
		self.commands = deque(maxlen=2)
		self.commands.append(4)
		self.commands.append(4)
		self.target_point_prev = [1e5, 1e5, 1e5]
		self.last_command = -1
		self.last_command_tmp = -1
		
		# Stop sign post-processor (CARLA API-based, no model needed)
		world = CarlaDataProvider.get_world()
		self.stop_sign_criteria = RunStopSign(world)

		self.initialized = True
		self.metric_info = {}
		# self._hic = DisplayInterface()

	def _build_obs_dict(self, tick_data, lidar, rgb_front, speed, theta, target_point, next_target_point, cmd_one_hot, waypoint):
		"""
		Build observation dictionaries from historical data.
		
		Args:
			target_point: Current target point in ego frame
			next_target_point: Next target point in ego frame (used together with target_point for DP model)
		
		Returns:
			lidar_stacked: (1, obs_horizon, C, H, W) stacked lidar BEV images
			ego_status_stacked: (1, obs_horizon, 14) concatenated ego status features
			                    Order: speed(1) + theta(1) + cmd(6) + target_point(2) + next_target_point(2) + waypoint_relative(2)
			rgb_stacked: (1, 5, C, H, W) stacked RGB images
		"""
		# Build obs_dict from historical observations
		lidar_history_list = list(self.lidar_bev_history)
		speed_history_list = list(self.speed_history)
		target_point_history_list = list(self.target_point_history)
		next_target_point_history_list = list(self.next_target_point_history)  # For DP model input
		cmd_history_list = list(self.next_command_history)
		theta_history_list = list(self.theta_history)
		throttle_history_list = list(self.throttle_history)
		brake_history_list = list(self.brake_history)
		rgb_history_list = list(self.rgb_history)
		waypoint_history_list = list(self.waypoint_history)
		
		lidar_list = [lidar_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(lidar_history_list)]
		speed_list = [speed_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(speed_history_list)]
		target_point_list = [target_point_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(target_point_history_list)]
		next_target_point_list = [next_target_point_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(next_target_point_history_list)]  # For DP model input
		cmd_list = [cmd_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(cmd_history_list)]
		theta_list = [theta_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(theta_history_list)]
		throttle_list = [throttle_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(throttle_history_list)]
		brake_list = [brake_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(brake_history_list)]
		waypoint_list = [waypoint_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(waypoint_history_list)]
		
		# Reverse to get chronological order (oldest to newest)
		lidar_list = lidar_list[::-1]
		speed_list = speed_list[::-1]
		target_point_list = target_point_list[::-1]
		next_target_point_list = next_target_point_list[::-1]  # For DP model input
		cmd_list = cmd_list[::-1]
		theta_list = theta_list[::-1]
		throttle_list = throttle_list[::-1]
		brake_list = brake_list[::-1]
		waypoint_list = waypoint_list[::-1]
		
		# sample every 5 for rgb from the end (t0, t-5, t-10, t-15)
		rgb_list = [rgb_history_list[-1 - i*5] for i in range(4) if -1 - i*5 >= -len(rgb_history_list)]
		rgb_list = rgb_list[::-1]  
		
		# Stack along time dimension
		lidar_stacked = torch.stack(lidar_list, dim=0).unsqueeze(0)  # (1, obs_horizon, C, H, W)
		speed_stacked = torch.cat(speed_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 1)
		theta_stacked = torch.cat(theta_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 1)
		cmd_stacked = torch.cat(cmd_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 6)
		target_point_stacked = torch.cat(target_point_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		waypoint_stacked = torch.stack(waypoint_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		rgb_stacked = torch.stack(rgb_list, dim=0).unsqueeze(0)  # (1, 5, C, H, W)
		

		current_pos = tick_data['gps']  
		current_theta = tick_data['theta']  # This is already preprocessed: compass - 90°
		
		# Build rotation matrix for world-to-ego transformation
		# R = [[cos, -sin], [sin, cos]] is the ego-to-world rotation
		# R.T = [[cos, sin], [-sin, cos]] is the world-to-ego rotation
		# This matches inverse_conversion_2d: R.T @ (point - translation)
		cos_theta = np.cos(current_theta)
		sin_theta = np.sin(current_theta)
		current_R = np.array([
			[cos_theta, sin_theta],
			[-sin_theta, cos_theta]
		])  # This is R.T, for world-to-ego transformation
		
		# Transform each historical waypoint to current frame
		waypoint_relative_list = []
		for i in range(self.obs_horizon):
			past_waypoint = waypoint_stacked[0, i].cpu().numpy()  # [x, y] in global coordinates
			# Transform from world to ego frame: R.T @ (past - current)
			relative_waypoint = current_R @ (past_waypoint - current_pos)
			waypoint_relative_list.append(torch.from_numpy(relative_waypoint).float().to('cuda'))
		
		waypoint_relative_stacked = torch.stack(waypoint_relative_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		
		# Transform target_point history to current ego frame (same logic as next_target_point)
		target_point_transformed_list = []
		for i in range(self.obs_horizon):
			past_world_pos = waypoint_list[i].cpu().numpy()  # [x, y] in world coordinates
			past_theta = theta_list[i].squeeze().cpu().item()  # theta value for past frame (already preprocessed)
			target_in_past_ego = target_point_list[i].squeeze().cpu().numpy()  # [x, y] in past ego frame
			
			# Build past ego-to-world rotation matrix
			cos_past = np.cos(past_theta)
			sin_past = np.sin(past_theta)
			past_R_ego_to_world = np.array([
				[cos_past, -sin_past],
				[sin_past, cos_past]
			])  # This is R, for ego-to-world transformation
			
			# Step 1: Transform target_point from past ego frame to world frame
			target_world = past_R_ego_to_world @ target_in_past_ego + past_world_pos
			
			# Step 2: Transform from world frame to current ego frame
			target_in_current_ego = current_R @ (target_world - current_pos)
			
			target_point_transformed_list.append(torch.from_numpy(target_in_current_ego).float().to('cuda'))
		
		target_point_transformed_stacked = torch.stack(target_point_transformed_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		
		# Transform next_target_point history to current ego frame
		next_target_point_transformed_list = []
		for i in range(self.obs_horizon):
			past_world_pos = waypoint_list[i].cpu().numpy()  # [x, y] in world coordinates
			past_theta = theta_list[i].squeeze().cpu().item()  # theta value for past frame (already preprocessed)
			next_target_in_past_ego = next_target_point_list[i].squeeze().cpu().numpy()  # [x, y] in past ego frame
			
			# Build past ego-to-world rotation matrix
			cos_past = np.cos(past_theta)
			sin_past = np.sin(past_theta)
			past_R_ego_to_world = np.array([
				[cos_past, -sin_past],
				[sin_past, cos_past]
			])  # This is R, for ego-to-world transformation
			
			# Step 1: Transform next_target_point from past ego frame to world frame
			next_target_world = past_R_ego_to_world @ next_target_in_past_ego + past_world_pos
			
			# Step 2: Transform from world frame to current ego frame
			next_target_in_current_ego = current_R @ (next_target_world - current_pos)
			
			next_target_point_transformed_list.append(torch.from_numpy(next_target_in_current_ego).float().to('cuda'))
		
		next_target_point_transformed_stacked = torch.stack(next_target_point_transformed_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		
		
		# Concatenate all ego status features following unified_carla_dataset order:
		# speed(1) + theta(1) + cmd(6) + target_point(2) + next_target_point(2) + waypoint_relative(2) = 14
		ego_status_stacked = torch.cat([
			speed_stacked,                            # (1, obs_horizon, 1)
			theta_stacked,                            # (1, obs_horizon, 1)
			cmd_stacked,                              # (1, obs_horizon, 6)
			target_point_transformed_stacked,         # (1, obs_horizon, 2) - target_points transformed to current ego frame
			next_target_point_transformed_stacked,    # (1, obs_horizon, 2) - next_target_points transformed to current ego frame
			waypoint_relative_stacked                 # (1, obs_horizon, 2) - ego positions in current frame
		], dim=-1)  # Concatenate along feature dimension
		
		return lidar_stacked, ego_status_stacked, rgb_stacked

	def _resolve_target_pose(self, gps_raw, gps_filtered, compass_raw, compass_filtered):
		"""Choose the pose source used by planner/target projection/model theta."""
		if TARGET_POSE_SOURCE == 'hero':
			try:
				vehicle = CarlaDataProvider.get_hero_actor()
				if vehicle is not None:
					vehicle_transform = vehicle.get_transform()
					hero_xy = np.array(
						[vehicle_transform.location.x, vehicle_transform.location.y],
						dtype=np.float32,
					)
					hero_yaw = t_u.normalize_angle(np.deg2rad(vehicle_transform.rotation.yaw))
					return hero_xy, hero_yaw, 'hero'
			except Exception:
				pass

		if TARGET_POSE_SOURCE == 'raw':
			return (
				np.asarray(gps_raw, dtype=np.float32),
				float(t_u.normalize_angle(compass_raw)),
				'raw',
			)

		return (
			np.asarray(gps_filtered, dtype=np.float32),
			float(t_u.normalize_angle(compass_filtered)),
			'filtered',
		)


	def sensors(self):
		sensors =  [
				{
					'type': 'sensor.camera.rgb',
					'x': -1.50, 'y': 0.0, 'z': 2.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 1024, 'height': 512, 'fov': 110,
					'id': 'CAM_FRONT'
					},
				# lidar
				{
          			'type': 'sensor.lidar.ray_cast',
          			'x': 0.0, 'y': 0.0, 'z': 2.5,
          			'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
          			'id': 'LIDAR'
      				},
				# imu
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'IMU'
					},
				# gps
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'GPS'
					},
				# speed
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'SPEED'
					},
				]
		
		if IS_BENCH2DRIVE:
			sensors += [
					{	
						'type': 'sensor.camera.rgb',
						'x': 0.0, 'y': 0.0, 'z': 50.0,
						'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
						'width': 512, 'height': 512, 'fov': 5 * 10.0,
						'id': 'bev'
					}]
		return sensors

	def tick(self, input_data):
		self.step += 1
		rgb_front = cv2.cvtColor(input_data['CAM_FRONT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		lidar_ego = lidar_to_ego_coordinate(input_data['LIDAR'])
		
		gps_full = input_data['GPS'][1]  # [lat, lon, altitude]
		gps_pos = self._route_planner.convert_gps_to_carla(gps_full)
		
		# Handle compass NaN 
		compass_raw = input_data['IMU'][1][-1]
		if math.isnan(compass_raw):
			print("compass sends nan!!!")
			compass_raw = 0.0
		
		# Preprocess compass to CARLA coordinate system
		compass = t_u.preprocess_compass(compass_raw)
		
		# Get speed for UKF
		speed = input_data['SPEED'][1]['speed']
		
		# Apply Unscented Kalman Filter 
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

		gps_target_pose, compass_target_pose, target_pose_source = self._resolve_target_pose(
			gps_raw=np.array([gps_pos[0], gps_pos[1]], dtype=np.float32),
			gps_filtered=gps_filtered,
			compass_raw=compass,
			compass_filtered=compass_filtered,
		)

		# Combine two frames of lidar data using algin_lidar
		# Use filtered GPS for lidar alignment
		if self.last_lidar is not None and self.last_ego_transform is not None:
			# Calculate relative transformation between current and last frame
			current_pos = np.array([gps_filtered[0], gps_filtered[1], 0.0])
			last_pos = np.array([self.last_ego_transform['gps'][0], self.last_ego_transform['gps'][1], 0.0])
			relative_translation = current_pos - last_pos
			
			# Calculate relative rotation using filtered compass
			current_yaw = compass_filtered
			last_yaw = self.last_ego_transform['compass']
			relative_rotation = current_yaw - last_yaw
			
			# Rotate difference vector from global to local coordinate system
			rotation_matrix = np.array([[np.cos(current_yaw), -np.sin(current_yaw), 0.0],
										[np.sin(current_yaw), np.cos(current_yaw), 0.0], 
										[0.0, 0.0, 1.0]])
			relative_translation_local = rotation_matrix.T @ relative_translation
			
			# Align the last lidar to current coordinate system
			lidar_last = algin_lidar(self.last_lidar, relative_translation_local, relative_rotation)
			# Combine lidar frames
			lidar_combined = np.concatenate((lidar_ego, lidar_last), axis=0)
		else:
			lidar_combined = lidar_ego
		
		# Store current frame for next iteration (use filtered values)
		self.last_lidar = lidar_ego
		self.last_ego_transform = {'gps': gps_filtered, 'compass': compass_filtered}
		
		# Generate lidar BEV image from combined lidar data
		lidar_bev_img = generate_lidar_bev_images(
			np.copy(lidar_combined), 
			saving_name=None, 
			img_height=448, 
			img_width=448
		)
		# Convert BEV image to tensor format for interfuser_bev_encoder backbone
		lidar_bev_tensor = torch.from_numpy(lidar_bev_img).permute(2, 0, 1).float() / 255.0
		
		# ========== TransFuser style processing for DP features ==========
		# Process RGB for TransFuser (same as team_code_transfuser/sensor_agent.py)
		transfuser_rgb = input_data['CAM_FRONT'][1][:, :, :3]
		# Add jpg artifacts at test time, because the training data was saved as jpg
		_, compressed_image = cv2.imencode('.jpg', transfuser_rgb)
		transfuser_rgb = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
		transfuser_rgb = cv2.cvtColor(transfuser_rgb, cv2.COLOR_BGR2RGB)
		# Crop RGB image (same as transfuser training)
		transfuser_rgb = transfuser_t_u.crop_array(self.transfuser_config, transfuser_rgb)
		# Convert to PyTorch format (C, H, W) and batch
		transfuser_rgb = np.transpose(transfuser_rgb, (2, 0, 1))
		transfuser_rgb_tensor = torch.from_numpy(transfuser_rgb).float().unsqueeze(0).to('cuda')
		
		# Process LiDAR for TransFuser (same as team_code_transfuser/sensor_agent.py)
		transfuser_lidar = transfuser_t_u.lidar_to_ego_coordinate(self.transfuser_config, input_data['LIDAR'])
		
		# Store state for lidar alignment
		self.transfuser_state_log.append([gps_filtered[0], gps_filtered[1], compass_filtered, speed])
		
		# We only get half a LiDAR at every time step. Align the last half into the current frame.
		if self.transfuser_lidar_last is not None and len(self.transfuser_state_log) >= 2:
			ego_x = self.transfuser_state_log[-1][0]
			ego_y = self.transfuser_state_log[-1][1]
			ego_theta = self.transfuser_state_log[-1][2]
			
			ego_x_last = self.transfuser_state_log[-2][0]
			ego_y_last = self.transfuser_state_log[-2][1]
			ego_theta_last = self.transfuser_state_log[-2][2]
			
			transfuser_lidar_last_aligned = self._align_lidar_transfuser(
				self.transfuser_lidar_last, 
				ego_x_last, ego_y_last, ego_theta_last,
				ego_x, ego_y, ego_theta
			)
			transfuser_lidar_full = np.concatenate((transfuser_lidar, transfuser_lidar_last_aligned), axis=0)
		else:
			transfuser_lidar_full = transfuser_lidar
		
		self.transfuser_lidar_last = transfuser_lidar.copy()
		self.transfuser_lidar_buffer.append(transfuser_lidar_full)
		
		# Convert to histogram BEV (same as sensor_agent.py)
		transfuser_lidar_bev = self.transfuser_data.lidar_to_histogram_features(
			transfuser_lidar_full,
			use_ground_plane=self.transfuser_config.use_ground_plane
		)
		transfuser_lidar_bev_tensor = torch.from_numpy(transfuser_lidar_bev).float().unsqueeze(0).to('cuda')
		
		# Process other sensors
		if IS_BENCH2DRIVE:
			bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		else:
			bev = np.zeros((512, 512, 3), dtype=np.uint8)
		
		result = {
				'rgb_front': rgb_front,
				'lidar_bev': lidar_bev_tensor,
				'gps': gps_target_pose,  # Pose used by planner/model target projection
				'gps_raw': np.array([gps_pos[0], gps_pos[1]], dtype=np.float32),
				'speed': speed,
				'compass': compass_target_pose,  # Heading used by planner/model target projection
				'compass_raw': compass,
				'bev': bev,
				'gps_filtered': gps_filtered,
				'compass_filtered': compass_filtered,
				'target_pose_source': target_pose_source,
				# TransFuser processed data for DP
				'transfuser_rgb': transfuser_rgb_tensor,  # (1, 3, H, W) on GPU
				'transfuser_lidar_bev': transfuser_lidar_bev_tensor,  # (1, C, H, W) on GPU
				}
		
		waypoint_route = self._route_planner.run_step(np.append(result['gps'], gps_pos[2]))
		

		
		# Follow the newer hpc_agent_1 logic:
		# - use the near-future route point as target_point
		# - use the next route point as next_target_point when available
		# - if route points are insufficient, synthesize a farther next_target_point in world frame
		if len(waypoint_route) > 2:
			target_point, far_command = waypoint_route[1]
			next_target_point, next_far_command = waypoint_route[2]
		elif len(waypoint_route) > 1:
			target_point, far_command = waypoint_route[1]
			ego_pos = result['gps'][:2]
			direction = target_point[:2] - ego_pos
			dist = np.linalg.norm(direction)
			if dist > 1e-3:
				direction_normalized = direction / dist
			else:
				direction_normalized = np.array([np.cos(result['compass']), np.sin(result['compass'])])
			next_target_point = target_point[:2] + direction_normalized * 50.0
			next_far_command = far_command
		elif len(waypoint_route) > 0:
			target_point, far_command = waypoint_route[0]
			ego_pos = result['gps'][:2]
			direction = target_point[:2] - ego_pos
			dist = np.linalg.norm(direction)
			if dist > 1e-3:
				direction_normalized = direction / dist
			else:
				direction_normalized = np.array([np.cos(result['compass']), np.sin(result['compass'])])
			next_target_point = target_point[:2] + direction_normalized * 50.0
			next_far_command = far_command
		else:
			target_point, far_command = (result['gps'][:2], RoadOption.LANEFOLLOW)
			direction_normalized = np.array([np.cos(result['compass']), np.sin(result['compass'])])
			next_target_point = result['gps'][:2] + direction_normalized * 50.0
			next_far_command = RoadOption.LANEFOLLOW

		if self.last_command_tmp != far_command:
			self.last_command = self.last_command_tmp
		self.last_command_tmp = far_command
		
		if hasattr(target_point, '__iter__') and len(target_point) >= 2:
			if (target_point[:2] != self.target_point_prev[:2]).any() if isinstance(target_point, np.ndarray) else (list(target_point[:2]) != list(self.target_point_prev[:2])):
				self.target_point_prev = target_point
				self.commands.append(far_command.value)
		
		result['next_command'] = self.commands[-2]
		ego_waypoint_route = []
		for route_item in waypoint_route:
			route_point = route_item[0]
			ego_route_point = t_u.inverse_conversion_2d(route_point[:2], result['gps'], result['compass'])
			ego_waypoint_route.append(ego_route_point)
		if ego_waypoint_route:
			self.last_waypoint_route = np.asarray(ego_waypoint_route, dtype=np.float32)
		else:
			self.last_waypoint_route = None

		ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])
		ego_next_target_point = t_u.inverse_conversion_2d(next_target_point[:2], result['gps'], result['compass'])

		forward_vec_world = np.array([
			np.cos(result['compass']),
			np.sin(result['compass']),
		], dtype=np.float32)
		target_delta_world = np.asarray(target_point[:2], dtype=np.float32) - np.asarray(result['gps'][:2], dtype=np.float32)
		next_target_delta_world = np.asarray(next_target_point[:2], dtype=np.float32) - np.asarray(result['gps'][:2], dtype=np.float32)

		def _angle_and_dot(delta_world):
			dist = float(np.linalg.norm(delta_world))
			if dist < 1e-6:
				return 0.0, 0.0, 0.0
			dot = float(np.dot(forward_vec_world, delta_world))
			cross = float(forward_vec_world[0] * delta_world[1] - forward_vec_world[1] * delta_world[0])
			cos_val = float(np.clip(dot / dist, -1.0, 1.0))
			angle_deg = float(np.rad2deg(np.arccos(cos_val)))
			return angle_deg, dot, cross

		target_angle_deg, target_dot_forward, target_cross_forward = _angle_and_dot(target_delta_world)
		next_target_angle_deg, next_target_dot_forward, next_target_cross_forward = _angle_and_dot(next_target_delta_world)

		# Debug: print target point transformation
		# if self.step <= 5:
		# 	print(f"  target_point (world): {target_point[:2]}")
		# 	print(f"  ego position (gps): {result['gps']}")
		# 	print(f"  compass (heading): {result['compass']:.4f} rad ({np.rad2deg(result['compass']):.2f} deg)")
		# 	print(f"  ego_target_point: {ego_target_point}")
		
		result['target_point'] = ego_target_point  # numpy array (2,)
		result['next_target_point'] = ego_next_target_point  # numpy array (2,)
		result['target_point_world'] = np.asarray(target_point[:2], dtype=np.float32)
		result['next_target_point_world'] = np.asarray(next_target_point[:2], dtype=np.float32)
		result['target_dot_forward'] = target_dot_forward
		result['next_target_dot_forward'] = next_target_dot_forward
		result['target_cross_forward'] = target_cross_forward
		result['next_target_cross_forward'] = next_target_cross_forward
		result['target_angle_deg'] = target_angle_deg
		result['next_target_angle_deg'] = next_target_angle_deg
		result['target_is_behind'] = bool(target_dot_forward < 0.0)
		result['next_target_is_behind'] = bool(next_target_dot_forward < 0.0)
		# Docs/data convention: ego frame uses [x_forward, y_lateral] with y > 0 = right.
		result['target_is_right'] = bool(target_cross_forward > 0.0)
		result['next_target_is_right'] = bool(next_target_cross_forward > 0.0)
		result['target_is_left'] = bool(target_cross_forward < 0.0)
		result['next_target_is_left'] = bool(next_target_cross_forward < 0.0)
		if len(waypoint_route) > 0:
			result['waypoint_route_world'] = np.asarray([route_item[0][:2] for route_item in waypoint_route], dtype=np.float32)
			result['waypoint_route_ego'] = self.last_waypoint_route.copy() if self.last_waypoint_route is not None else None
		else:
			result['waypoint_route_world'] = None
			result['waypoint_route_ego'] = None
		result['theta'] = compass_filtered

		return result

	def _align_lidar_transfuser(self, lidar, x, y, orientation, x_target, y_target, orientation_target):
		"""
		Align lidar from past frame to current frame (same as sensor_agent.py).
		
		Args:
			lidar: numpy LiDAR point cloud (N, 3)
			x, y, orientation: past frame ego pose
			x_target, y_target, orientation_target: current frame ego pose
			
		Returns:
			aligned_lidar: numpy LiDAR point cloud in current frame coordinates
		"""
		pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
		rot_diff = transfuser_t_u.normalize_angle(orientation_target - orientation)
		
		# Rotate difference vector from global to local coordinate system.
		rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
		                            [np.sin(orientation_target), np.cos(orientation_target), 0.0], 
		                            [0.0, 0.0, 1.0]])
		pos_diff = rotation_matrix.T @ pos_diff
		
		return transfuser_t_u.algin_lidar(lidar, pos_diff, rot_diff)
	
	def _truncate_route_by_target_point(self, route_waypoints_np, target_point_np):
		"""
		Truncate route_pred based on target_point projection.
		
		Logic:
		- Project target_point onto the polyline formed by route_pred
		- If projection falls inside route_pred (route is truncated by target_point),
		  then the portion after projection is inaccurate and should not be used
		- If projection falls beyond route_pred's end, the entire route is valid
		
		Protection mechanism:
		- If truncated route has too few points (< MIN_POINTS_THRESHOLD) or
		  is too short (< MIN_LENGTH_THRESHOLD), skip truncation and use original route
		- This handles edge cases near the destination where target_point is very close
		
		Args:
			route_waypoints_np: (N, 2) numpy array in ego frame [x_forward, y_right]
			target_point_np: (2,) numpy array in ego frame [x_forward, y_right]
		
		Returns:
			truncated_route: (M, 2) numpy array, M <= N, the valid portion of route_pred
			truncation_idx: int, the index up to which the route is valid (-1 if no truncation)
		"""
		# Protection thresholds
		MIN_POINTS_THRESHOLD = 5  # Minimum number of points needed for reliable control
		MIN_LENGTH_THRESHOLD = 3.0  # Minimum route length in meters for reliable lookahead
		
		if len(route_waypoints_np) < 2:
			return route_waypoints_np, -1
		
		# Find the closest segment to target_point
		min_dist = float('inf')
		best_segment_idx = -1
		best_t = 0.0  # Parameter along segment [0, 1]
		best_proj_point = None
		
		for i in range(len(route_waypoints_np) - 1):
			p1 = route_waypoints_np[i]
			p2 = route_waypoints_np[i + 1]
			
			# Vector from p1 to p2
			v = p2 - p1
			# Vector from p1 to target_point
			w = target_point_np - p1
			
			# Length squared of segment
			l2 = np.dot(v, v)
			if l2 < 1e-10:  # Degenerate segment
				t = 0.0
				proj = p1
			else:
				# Project target_point onto the line containing the segment
				t = np.dot(w, v) / l2
				proj = p1 + t * v
			
			# Distance from target_point to projection
			dist = np.linalg.norm(target_point_np - proj)
			
			# We consider projections within or beyond the segment
			# t < 0: projection is before p1
			# 0 <= t <= 1: projection is within segment
			# t > 1: projection is beyond p2
			
			if dist < min_dist:
				min_dist = dist
				best_segment_idx = i
				best_t = t
				best_proj_point = proj
		
		# Determine truncation based on projection position
		# If best_t is within [0, 1], the projection is inside the route segment
		# If best_t > 1, check if we're on the last segment - if so, projection is beyond route
		
		if best_segment_idx == -1:
			# No valid segment found, return original route
			return route_waypoints_np, -1
		
		# Calculate the "arc length" position of the projection along the route
		# If projection is beyond the last point, no truncation needed
		is_on_last_segment = (best_segment_idx == len(route_waypoints_np) - 2)
		
		if best_t > 1.0 and is_on_last_segment:
			# Projection is beyond the end of route_pred
			# The entire route is valid for lookahead calculation
			return route_waypoints_np, -1
		
		# Projection is within route_pred or before it (shouldn't happen normally)
		# Truncate the route at the projection point
		if best_t <= 0.0:
			# Projection is at or before the start of this segment
			# Keep points up to and including segment start
			truncation_idx = best_segment_idx
		elif best_t >= 1.0:
			# Projection is at or beyond the end of this segment
			# Keep points up to and including segment end
			truncation_idx = best_segment_idx + 1
		else:
			# Projection is within the segment
			# Keep points up to segment start, then add the projection point
			truncation_idx = best_segment_idx
		
		# Build truncated route
		if truncation_idx >= len(route_waypoints_np) - 1:
			# No truncation needed
			return route_waypoints_np, -1
		
		# Include points up to truncation_idx, then add projection point
		truncated = route_waypoints_np[:truncation_idx + 1].copy()
		
		# Add the projection point if it's meaningfully different from the last included point
		if best_proj_point is not None and len(truncated) > 0:
			dist_to_last = np.linalg.norm(best_proj_point - truncated[-1])
			if dist_to_last > 0.1:  # Only add if more than 0.1m away
				truncated = np.vstack([truncated, best_proj_point])
		
		# ============ Protection mechanism ============
		# Calculate the total length of truncated route
		truncated_length = 0.0
		for i in range(len(truncated) - 1):
			truncated_length += np.linalg.norm(truncated[i + 1] - truncated[i])
		
		# Check if truncated route meets minimum requirements
		if len(truncated) < MIN_POINTS_THRESHOLD or truncated_length < MIN_LENGTH_THRESHOLD:
			# Truncated route is too short, skip truncation and use original route
			# This handles edge cases near destination where target_point is very close
			# print(f"[Lateral] Skip truncation: points={len(truncated)}, length={truncated_length:.2f}m "
			# 	  f"(thresholds: {MIN_POINTS_THRESHOLD} points, {MIN_LENGTH_THRESHOLD}m)")
			return route_waypoints_np, -1
		
		return truncated, truncation_idx
	
	def control_pid(self, route_waypoints, velocity, speed_waypoints, target_point=None):
		"""
		Predicts vehicle control with a PID controller.
		
		Args:
			route_waypoints: (1, N, 2) tensor in ego frame [x_forward, y_right]
			velocity: float, current speed in m/s
			speed_waypoints: (1, N, 2) tensor for speed calculation
			target_point: (1, 2) tensor in ego frame [x_forward, y_right], used for route truncation
		"""
		assert route_waypoints.size(0) == 1
		route_waypoints_np = route_waypoints[0].data.cpu().numpy()  # (N, 2)
		speed = velocity  # Already a float
		speed_waypoints_np = speed_waypoints[0].data.cpu().numpy()  # (N, 2)
		
		# Truncate route using target_point projection
		if target_point is not None:
			target_point_np = target_point[0].data.cpu().numpy()  # (2,)
			route_waypoints_np, truncation_idx = self._truncate_route_by_target_point(route_waypoints_np, target_point_np)
			# if truncation_idx >= 0:
			# 	print(f"[Lateral] Route truncated at index {truncation_idx}, remaining points: {len(route_waypoints_np)}")
		
		# MoT trajectory: 6 points, 0.5s interval each, total 3s
		# Point indices: 0(0.5s), 1(1.0s), 2(1.5s), 3(2.0s), 4(2.5s), 5(3.0s)
		mot_waypoint_interval = 0.5  # seconds between waypoints
		one_second_idx = 2 #1  # point[1] is at 1.0s
		half_second_idx = 0  # point[0] is at 0.5s
		
		if speed_waypoints_np.shape[0] >= 2:
			# Displacement from 0.5s to 1.0s position, multiply by 2 to get m/s
			# desired_speed = np.linalg.norm(speed_waypoints_np[one_second_idx] - speed_waypoints_np[half_second_idx]) * 2.0
			desired_speed = np.linalg.norm(speed_waypoints_np[one_second_idx] - speed_waypoints_np[half_second_idx])
		else:
			# Fallback: use first point distance, assuming it represents 0.5s travel
			desired_speed = np.linalg.norm(speed_waypoints_np[0]) * 2.0

		# Speed limit: cap desired_speed at 35 km/h = 35/3.6 ≈ 9.72 m/s
		# max_desired_speed_ms = 35.0 / 3.6  # 35 km/h in m/s
		# desired_speed = min(desired_speed, max_desired_speed_ms)

		# OLD PID throttle (kept for reference):
		# brake = ((desired_speed < self.brake_speed) or ((speed / max(desired_speed, 1e-5)) > self.brake_ratio))
		# delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
		# throttle = self.speed_controller.step(delta)
		# throttle = np.clip(throttle, 0.0, self.clip_throttle)
		# throttle = throttle if not brake else 0.0

		# BridgeDrive post-processing: learned polynomial throttle
		brake = (desired_speed < self.brake_speed) or ((speed / max(desired_speed, 1e-5)) > self.brake_ratio)
		throttle, brake = get_throttle(brake, desired_speed, speed)
		

		route_interp = self.interpolate_waypoints(route_waypoints_np)
		
		
		steer = self._compute_lateral_steer(route_interp, speed)
		steer = np.clip(steer, -1.0, 1.0)
		steer = round(steer, 3)
		
		
		return steer, throttle, brake
	
	def interpolate_waypoints(self, waypoints):
		"""
		Interpolate waypoints to be 0.1m apart
		
		Args:
			waypoints: (N, 2) numpy array in ego frame [x_forward, y_right]
			
		Returns:
			interp_points: (M, 2) numpy array with points 0.1m apart
		"""
		waypoints = waypoints.copy()
		# Add origin point at the beginning
		waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
		shift = np.roll(waypoints, 1, axis=0)
		shift[0] = shift[1]
		
		dists = np.linalg.norm(waypoints - shift, axis=1)
		dists = np.cumsum(dists)
		dists += np.arange(0, len(dists)) * 1e-4  # Prevents dists not being strictly increasing
		
		interp = PchipInterpolator(dists, waypoints, axis=0)
		
		x = np.arange(0.1, dists[-1], 0.1)
		
		interp_points = interp(x)
		
		if interp_points.shape[0] == 0:
			interp_points = waypoints[None, -1]
		
		return interp_points

	def _compute_lateral_steer(self, route_interp, current_speed):
		"""Compute a smoother steering command from forward-valid route points."""
		ctrl = self.turn_controller
		speed_kmh = current_speed * 3.6

		if ctrl.inference_mode:
			requested_idx = np.clip(
				ctrl.speed_scale * speed_kmh + ctrl.speed_offset,
				ctrl.default_lookahead,
				105,
			) / 10.0
			requested_idx = max(requested_idx - 2.0, 0.0)
		else:
			requested_idx = float(np.clip(
				ctrl.speed_scale * speed_kmh + ctrl.speed_offset,
				ctrl.default_lookahead,
				105,
			))

		requested_idx = min(requested_idx, max(len(route_interp) - 1, 0))
		target_idx = int(round(requested_idx))

		# Ignore route points that are effectively behind the ego; these can flip
		# the heading target during tight turns or route truncation.
		forward_valid_idx = np.flatnonzero(route_interp[:, 0] > 0.5)
		if forward_valid_idx.size > 0:
			candidate_idx = forward_valid_idx[forward_valid_idx >= target_idx]
			if candidate_idx.size > 0:
				target_idx = int(candidate_idx[0])
			else:
				target_idx = int(forward_valid_idx[-1])

		window_start = max(0, target_idx - 2)
		window_end = min(len(route_interp), target_idx + 3)
		desired_heading_vec = route_interp[window_start:window_end].mean(axis=0)

		yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
		heading_error_rad = (yaw_path) % (2 * np.pi)
		heading_error_rad = heading_error_rad if heading_error_rad < np.pi else heading_error_rad - 2 * np.pi
		heading_error = heading_error_rad * 180.0 / np.pi / 90.0

		ctrl._window.append(heading_error)
		ctrl._window = ctrl._window[-ctrl.n:]

		derivative = 0.0 if len(ctrl._window) == 1 else ctrl._window[-1] - ctrl._window[-2]
		integral = float(np.mean(ctrl._window))
		steering = np.clip(
			ctrl.k_p * heading_error + ctrl.k_d * derivative + ctrl.k_i * integral,
			-1.0,
			1.0,
		).item()

		self.last_steer_debug = {
			'requested_lookahead_idx': float(requested_idx),
			'target_idx': int(target_idx),
			'forward_valid_count': int(forward_valid_idx.size),
			'desired_heading_vec': desired_heading_vec.tolist(),
			'yaw_path_deg': float(np.rad2deg(yaw_path)),
			'heading_error_norm': float(heading_error),
			'heading_error_deg': float(np.rad2deg(heading_error_rad)),
			'steer_raw': float(steering),
		}
		return steering
	
	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()
		tick_data = self.tick(input_data)

		# Prepare current observations
		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		# Use the same method as agent_simlingo.py: t_u.command_to_one_hot with self.commands[-2]
		one_hot_command = t_u.command_to_one_hot(self.commands[-2])
		cmd_one_hot = torch.from_numpy(one_hot_command[np.newaxis]).to('cuda', dtype=torch.float32)
		# Keep command variable for metadata (convert from 1-6 to 0-5 range)
		command = tick_data['next_command']
		if command < 0:
			command = 4
		command -= 1
		speed = torch.FloatTensor([float(tick_data['speed'])]).view(1,1).to('cuda', dtype=torch.float32)
		theta = torch.FloatTensor([float(tick_data['theta'])]).view(1,1).to('cuda', dtype=torch.float32)
		lidar = tick_data['lidar_bev'].to('cuda', dtype=torch.float32)
		
		rgb_front = torch.from_numpy(tick_data['rgb_front']).permute(2, 0, 1).float() / 255.0
		rgb_front = rgb_front.to('cuda', dtype=torch.float32)
		waypoint = torch.from_numpy(tick_data['gps']).float().to('cuda', dtype=torch.float32)
		target_point = torch.from_numpy(tick_data['target_point']).unsqueeze(0).float().to('cuda', dtype=torch.float32)
		next_target_point = torch.from_numpy(tick_data['next_target_point']).unsqueeze(0).float().to('cuda', dtype=torch.float32)
		
		# For debugging: print target point info occasionally
		if self.step % 20 == 0:
			tp = tick_data['target_point']
			ntp = tick_data['next_target_point']
			# print(f"[Target Points] TP=({tp[0]:.1f},{tp[1]:.1f}), NTP=({ntp[0]:.1f},{ntp[1]:.1f})")

		# Accumulate observation history into buffers 
		self.lidar_bev_history.append(lidar)
		self.rgb_history.append(rgb_front)
		self.speed_history.append(speed)
		self.target_point_history.append(target_point)
		self.next_target_point_history.append(next_target_point)  # Both target_point and next_target_point are used for DP model
		self.next_command_history.append(cmd_one_hot)
		self.theta_history.append(theta)
		self.waypoint_history.append(waypoint)
		
		# Append throttle and brake from previous step (or 0 for first step)
		if self.step < 1:
			# First step: initialize with 0
			self.throttle_history.append(torch.tensor(0.0).view(1, 1).to('cuda'))
			self.brake_history.append(torch.tensor(0.0).view(1, 1).to('cuda'))
		else:
			# Use control from previous step
			prev_control = self.prev_control if self.prev_control is not None else carla.VehicleControl()
			self.throttle_history.append(torch.tensor(prev_control.throttle).view(1, 1).to('cuda'))
			self.brake_history.append(torch.tensor(prev_control.brake).view(1, 1).to('cuda'))
		
		# Buffer size = 31 frames (for obs_horizon with 10x sampling)
		BUFFER_PHASE = 31     # Fill buffer to minimum required size
		
		if self.step < BUFFER_PHASE:
			# Warmup phase: use previous control or default
			control = self.prev_control
			self.pid_metadata = {}
			self.pid_metadata['agent'] = 'warmup_phase'
			self.pid_metadata['step'] = self.step
		else:
			# Build observation dict
			# Both target_point and next_target_point are used for DP model in ego_status_stacked
			lidar_stacked, ego_status_stacked, rgb_stacked = self._build_obs_dict(
				tick_data, lidar, rgb_front, speed, theta, target_point, next_target_point,
				cmd_one_hot, waypoint
			)
			
			if USE_MOT:
				rgb_pil_list = []
				for i in range(rgb_stacked.shape[1]):
					rgb_tensor = rgb_stacked[0, i]  # (C, H, W)
					rgb_np = (rgb_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
					rgb_pil = Image.fromarray(rgb_np, mode='RGB')
					rgb_pil_list.append(rgb_pil)

				lidar_tensor = lidar_stacked[0, -1]  # (C, H, W) - last frame
				lidar_np = (lidar_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
				lidar_pil = Image.fromarray(lidar_np, mode='RGB')
				lidar_pil_list = [lidar_pil]

				if self.stuck_helper > 0:
					target_point_speed = torch.cat([speed, next_target_point], dim=-1)
					# print("Get stucked! Trigger the stuck helper!")
				else:
					target_point_speed = torch.cat([speed, target_point], dim=-1)  # (1, 3)

				prompt_cleaned, understanding_output, reasoning_output = build_cleaned_prompt_and_modes(target_point_speed)

				predicted_answer = self.inferencer(
					image=rgb_pil_list,
					front=[rgb_pil_list[-1]],
					lidar=lidar_pil_list,
					v_target_point=target_point_speed,
					text=prompt_cleaned,
					understanding_output=understanding_output,
					reasoning_output=reasoning_output,
					max_think_token_n=self.inference_args.max_num_tokens,
					do_sample=False,
					text_temperature=0.0,
				)

				pred_traj = predicted_answer['traj']  # (1, 6, 2)
				pred_decision = predicted_answer['text']
				self.last_pred_traj = pred_traj.squeeze(0).float().cpu().numpy()
			else:
				pred_traj = None
				pred_decision = ""
				prompt_cleaned = ""

			self.last_target_point = target_point.squeeze(0).float().cpu().numpy()
			self.last_next_target_point = next_target_point.squeeze(0).float().cpu().numpy()

			# ========== Run TransFuser backbone to get BEV features ==========
			with torch.no_grad():
				# Keep float32 inputs for full precision inference
				transfuser_rgb_fp32 = tick_data['transfuser_rgb'].to(torch.float32)
				transfuser_lidar_bev_fp32 = tick_data['transfuser_lidar_bev'].to(torch.float32)
				transfuser_output = self.transfuser_backbone(
					rgb=transfuser_rgb_fp32,  # (1, 3, H, W) on GPU, float32
					lidar_bev=transfuser_lidar_bev_fp32  # (1, C, H, W) on GPU, float32
				)
			
			# Extract transfuser features (following DiffusionDriveV2: only 2 features)
			# bev_feature: (B, 1512, 8, 8) - original BEV feature (x4)
			# bev_feature_upscale: (B, 64, 64, 64) - upsampled BEV (p3)
			transfuser_bev_feature = transfuser_output['bev_feature']  # (1, 1512, 8, 8)
			transfuser_bev_feature_upsample = transfuser_output['bev_feature_upscale']  # (1, 64, 64, 64)
			
			# Build dp_obs_dict with transfuser features
			dp_obs_dict = {
				'ego_status': ego_status_stacked,
				'transfuser_bev_feature': transfuser_bev_feature,  # (B, 1512, 8, 8)
				'transfuser_bev_feature_upsample': transfuser_bev_feature_upsample,  # (B, 64, 64, 64)
			}
			dp_pred_traj = self._predict_dp_action(dp_obs_dict)
			# self.last_dp_pred_traj = dp_pred_traj['action'].squeeze(0).copy()  # (6, 2) in [x, y] format
			# if self.step % 20 == 0:
			# 	bev_f = transfuser_bev_feature.float()
			# 	print(f"[bev_feature] dtype={transfuser_bev_feature.dtype}, mean={bev_f.mean().item():.3f}, std={bev_f.std().item():.3f}, min={bev_f.min().item():.3f}, max={bev_f.max().item():.3f}")

			# ================== control_pid method ==================
			# Following agent_simlingo convention:
			# - speed_waypoints: use pred_traj from MoT model for speed control (throttle/brake)
			# - route_waypoints: use 'route_pred' from DP model for lateral angle control (steering)
			# speed_waypoints = pred_traj.float()  # (1, 6, 2) for speed control - use MoT prediction
			speed_waypoints = torch.from_numpy(dp_pred_traj['action']).float() # - use DP prediction
			# Get DP prediction for route_pred
			self.last_dp_pred_traj = dp_pred_traj['action'].squeeze(0).copy()  # (6, 2) in [x, y] format
			self.last_energy_debug = {}
			for energy_key in ['energy_front', 'energy_left', 'energy_right',
								'energy_pedestrian', 'energy_offroad', 'energy_route']:
				energy_value = dp_pred_traj.get(energy_key)
				if energy_value is None:
					continue
				energy_array = np.asarray(energy_value).reshape(-1)
				if energy_array.size > 0:
					self.last_energy_debug[energy_key] = float(energy_array[0])
			
			# route_pred is 20 waypoints with equal intervals for lateral control
			route_pred = dp_pred_traj['route_pred']  # tensor (B, 20, 2)
			if isinstance(route_pred, torch.Tensor):
				route_waypoints = route_pred.float().cpu()  # (1, 20, 2) for steering control
			else:
				route_waypoints = torch.from_numpy(route_pred).float()
			self.last_route_pred = route_waypoints.squeeze(0).numpy().copy()  # (20, 2) for visualization
			# if self.step % 20 == 0:
			# 	rp = self.last_route_pred
			# 	print(f"[route_pred] x_range=[{rp[:,0].min():.2f}, {rp[:,0].max():.2f}], y_range=[{rp[:,1].min():.2f}, {rp[:,1].max():.2f}]")
			# 	print(f"  x values: {rp[:,0].round(1).tolist()}")
			# 	dp_a = dp_pred_traj['action'].squeeze(0)
			# 	print(f"[dp_traj]  x_range=[{dp_a[:,0].min():.2f}, {dp_a[:,0].max():.2f}], y_range=[{dp_a[:,1].min():.2f}, {dp_a[:,1].max():.2f}]")
			# 	print(f"  ego_status last: speed={ego_status_stacked[0,-1,0].item():.2f}, tp={ego_status_stacked[0,-1,6:8].cpu().numpy().round(2).tolist()}")

			
			gt_velocity = tick_data['speed']
			
			# Use target_point to truncate route for lateral control
			steer, throttle, brake = self.control_pid(route_waypoints, gt_velocity, speed_waypoints, target_point=target_point)
			
			# Restart mechanism in case the car got stuck (following simlingo logic)
			# 0.1 is just an arbitrary low number to threshold when the car is stopped
			if gt_velocity < 0.1:
				self.stuck_detector += 1
			
			elif gt_velocity >= 1.0:
				self.stuck_detector = 0
			
			# If stuck for too long, trigger force_move
			if self.stuck_detector > self.stuck_threshold:
				self.force_move = self.creep_duration
			
			# Force move: override throttle and brake to get unstuck
			if self.force_move > 0:
				throttle = max(self.creep_throttle, throttle)
				brake = False
				self.force_move -= 1
				# print(f"force_move: {self.force_move}")

			# print(f"stuck_detector: {self.stuck_detector}")

			# ---- BridgeDrive-style post-processing ----

			# Traffic light: stop on red/yellow (CARLA API, no model needed)
			vehicle = CarlaDataProvider.get_hero_actor()
			if vehicle.is_at_traffic_light():
				tl = vehicle.get_traffic_light()
				tl_state = tl.get_state()
				if tl_state == carla.TrafficLightState.Red or tl_state == carla.TrafficLightState.Yellow:
					throttle = 0.0
					brake = 1.0
					self.stuck_detector = 0  # Waiting at TL is not stuck

			# Stop sign: brake to a full stop, then continue
			self.stop_sign_criteria.tick(vehicle)
			if self.stop_sign_criteria.target_stop_sign is not None and not self.stop_sign_criteria.stop_completed:
				throttle = 0.0
				brake = 1.0
				self.stuck_detector = 0  # Waiting at stop sign is not stuck

			applied_steer = float(np.clip(STEER_SIGN_SCALE * steer, -1.0, 1.0))
			control = carla.VehicleControl()
			control.steer = applied_steer
			control.throttle = float(throttle)
			control.brake = float(brake)
			
			# Speed limit enforcement: if current speed > 35 km/h, force brake
			# gt_velocity is in m/s, convert to km/h by multiplying 3.6
			if gt_velocity * 3.6 > 35:
				control.throttle = 0.0
				control.brake = 1.0
				# print(f"[Speed Limit] Speed {gt_velocity * 3.6:.2f} km/h > 35 km/h, forcing brake!")
			
			# Store metadata
			self.pid_metadata = {
				'agent': 'mot',
				'steer': control.steer,
				'steer_controller': float(steer),
				'steer_sign_scale': STEER_SIGN_SCALE,
				'throttle': control.throttle,
				'brake': control.brake,
				'speed': gt_velocity,
				'command': command,
			}
			vehicle_transform = vehicle.get_transform()
			hero_xy = np.array([
				float(vehicle_transform.location.x),
				float(vehicle_transform.location.y),
			], dtype=np.float32)
			planner_xy = np.asarray(tick_data.get('gps'), dtype=np.float32)
			filtered_xy = np.asarray(tick_data.get('gps_filtered'), dtype=np.float32)
			raw_xy = np.asarray(tick_data.get('gps_raw'), dtype=np.float32)
			expected_step_distance = float(gt_velocity) * self.carla_frame_rate

			def _step_distance(current_xy, previous_xy):
				if current_xy is None or previous_xy is None:
					return None
				return float(np.linalg.norm(current_xy - previous_xy))

			planner_step_distance = _step_distance(planner_xy, self.prev_debug_planner_xy)
			filtered_step_distance = _step_distance(filtered_xy, self.prev_debug_filtered_xy)
			raw_step_distance = _step_distance(raw_xy, self.prev_debug_raw_xy)
			hero_step_distance = _step_distance(hero_xy, self.prev_debug_hero_xy)

			self.pid_metadata.update({
				'gps_raw': tick_data['gps_raw'].tolist() if isinstance(tick_data.get('gps_raw'), np.ndarray) else tick_data.get('gps_raw'),
				'gps_filtered': tick_data['gps_filtered'].tolist() if isinstance(tick_data.get('gps_filtered'), np.ndarray) else tick_data.get('gps_filtered'),
				'gps_planner': tick_data['gps'].tolist() if isinstance(tick_data.get('gps'), np.ndarray) else tick_data.get('gps'),
				'target_pose_source': tick_data.get('target_pose_source'),
				'compass_raw': float(tick_data.get('compass_raw', 0.0)),
				'compass_filtered': float(tick_data.get('compass_filtered', 0.0)),
				'compass_planner': float(tick_data.get('compass', 0.0)),
				'target_geom_theta': float(tick_data.get('target_geom_theta', tick_data.get('compass', 0.0))),
				'target_yaw_sign': TARGET_YAW_SIGN,
				'target_geom_yaw_sign': TARGET_GEOM_YAW_SIGN,
				'theta_model': float(tick_data.get('theta', 0.0)),
				'hero_location_world': [
					float(vehicle_transform.location.x),
					float(vehicle_transform.location.y),
					float(vehicle_transform.location.z),
				],
				'hero_yaw_world_deg': float(vehicle_transform.rotation.yaw),
				'planner_vs_hero_xy_error': float(np.linalg.norm(planner_xy - hero_xy)),
				'filtered_vs_hero_xy_error': float(np.linalg.norm(filtered_xy - hero_xy)),
				'raw_vs_hero_xy_error': float(np.linalg.norm(raw_xy - hero_xy)),
				'planner_step_distance': planner_step_distance,
				'filtered_step_distance': filtered_step_distance,
				'raw_step_distance': raw_step_distance,
				'hero_step_distance': hero_step_distance,
				'expected_step_distance': expected_step_distance,
				'target_point_ego': tick_data['target_point'].tolist() if isinstance(tick_data.get('target_point'), np.ndarray) else tick_data.get('target_point'),
				'next_target_point_ego': tick_data['next_target_point'].tolist() if isinstance(tick_data.get('next_target_point'), np.ndarray) else tick_data.get('next_target_point'),
				'target_point_world': tick_data['target_point_world'].tolist() if isinstance(tick_data.get('target_point_world'), np.ndarray) else tick_data.get('target_point_world'),
				'next_target_point_world': tick_data['next_target_point_world'].tolist() if isinstance(tick_data.get('next_target_point_world'), np.ndarray) else tick_data.get('next_target_point_world'),
				'model_route_pred_ego': self.last_route_pred.tolist() if isinstance(self.last_route_pred, np.ndarray) else self.last_route_pred,
				'model_dp_traj_ego': self.last_dp_pred_traj.tolist() if isinstance(self.last_dp_pred_traj, np.ndarray) else self.last_dp_pred_traj,
				'target_dot_forward': float(tick_data.get('target_dot_forward', 0.0)),
				'next_target_dot_forward': float(tick_data.get('next_target_dot_forward', 0.0)),
				'target_cross_forward': float(tick_data.get('target_cross_forward', 0.0)),
				'next_target_cross_forward': float(tick_data.get('next_target_cross_forward', 0.0)),
				'target_angle_deg': float(tick_data.get('target_angle_deg', 0.0)),
				'next_target_angle_deg': float(tick_data.get('next_target_angle_deg', 0.0)),
				'target_is_behind': bool(tick_data.get('target_is_behind', False)),
				'next_target_is_behind': bool(tick_data.get('next_target_is_behind', False)),
				'target_is_right': bool(tick_data.get('target_is_right', False)),
				'next_target_is_right': bool(tick_data.get('next_target_is_right', False)),
				'target_is_left': bool(tick_data.get('target_is_left', False)),
				'next_target_is_left': bool(tick_data.get('next_target_is_left', False)),
				'steer_requested_lookahead_idx': self.last_steer_debug.get('requested_lookahead_idx'),
				'steer_target_idx': self.last_steer_debug.get('target_idx'),
				'steer_forward_valid_count': self.last_steer_debug.get('forward_valid_count'),
				'steer_desired_heading_vec': self.last_steer_debug.get('desired_heading_vec'),
				'steer_yaw_path_deg': self.last_steer_debug.get('yaw_path_deg'),
				'steer_heading_error_deg': self.last_steer_debug.get('heading_error_deg'),
				'steer_heading_error_norm': self.last_steer_debug.get('heading_error_norm'),
				'steer_raw_before_round': self.last_steer_debug.get('steer_raw'),
			})
			if isinstance(self.last_route_pred, np.ndarray) and len(self.last_route_pred) > 0:
				self.pid_metadata['model_route_pred_first'] = self.last_route_pred[0].tolist()
				self.pid_metadata['model_route_pred_last'] = self.last_route_pred[-1].tolist()
			if isinstance(self.last_dp_pred_traj, np.ndarray) and len(self.last_dp_pred_traj) > 0:
				self.pid_metadata['model_dp_traj_first'] = self.last_dp_pred_traj[0].tolist()
				self.pid_metadata['model_dp_traj_last'] = self.last_dp_pred_traj[-1].tolist()
			self.prev_debug_planner_xy = planner_xy.copy()
			self.prev_debug_filtered_xy = filtered_xy.copy()
			self.prev_debug_raw_xy = raw_xy.copy()
			self.prev_debug_hero_xy = hero_xy.copy()
			waypoint_route_world = tick_data.get('waypoint_route_world')
			waypoint_route_ego = tick_data.get('waypoint_route_ego')
			if isinstance(waypoint_route_world, np.ndarray) and len(waypoint_route_world) > 0:
				self.pid_metadata['waypoint_route_world_first'] = waypoint_route_world[0].tolist()
				self.pid_metadata['waypoint_route_world_last'] = waypoint_route_world[-1].tolist()
				self.pid_metadata['waypoint_route_world'] = waypoint_route_world.tolist()
			if isinstance(waypoint_route_ego, np.ndarray) and len(waypoint_route_ego) > 0:
				self.pid_metadata['waypoint_route_ego_first'] = waypoint_route_ego[0].tolist()
				self.pid_metadata['waypoint_route_ego_last'] = waypoint_route_ego[-1].tolist()
				self.pid_metadata['waypoint_route_ego'] = waypoint_route_ego.tolist()

			self.prev_control = control
			self.control = control  # Update control for UKF prediction in next tick
			metric_info = self.get_metric_info()
			self.metric_info[self.step] = metric_info

			if SAVE_PATH is not None:
				self.save(tick_data)

			##### Rendering ####
			ego_car_map = render_self_car(
				loc=np.array([0, 0]),
				ori=np.array([0, -1]),
				box=np.array([2.45, 1.0]),
				color=[1, 1, 0], pixels_per_meter=10, max_distance=30,
			)

			tp_for_render = target_point.cpu().float().numpy().copy()
			if tp_for_render.ndim == 2:
				tp_for_render = tp_for_render.squeeze(0)

			# Prepare dp_pred_traj for rendering (red)
			dp_traj_for_render = dp_pred_traj['action'].squeeze(0).copy()  # (6, 2) - already numpy
			dp_trajectory = np.concatenate((dp_traj_for_render, tp_for_render.reshape(1, 2)), axis=0)
			dp_trajectory = dp_trajectory[:, [1, 0]]
			dp_trajectory[:, 0] = -dp_trajectory[:, 0]
			dp_trajectory[:, 1] = -dp_trajectory[:, 1]
			render_dp_trajectory = render_waypoints(dp_trajectory, pixels_per_meter=30, max_distance=20, color=(255, 0, 0))

			ego_car_map = cv2.resize(ego_car_map, (200, 200))
			render_dp_trajectory = cv2.resize(render_dp_trajectory, (200, 200))

			if USE_MOT and pred_traj is not None:
				# Prepare MoT pred_traj for rendering (green)
				traj_for_render = pred_traj.squeeze(0).cpu().float().numpy().copy()
				trajectory = np.concatenate((traj_for_render, tp_for_render.reshape(1, 2)), axis=0)
				trajectory = trajectory[:, [1, 0]]
				trajectory[:, 0] = -trajectory[:, 0]
				trajectory[:, 1] = -trajectory[:, 1]
				render_trajectory = cv2.resize(
					render_waypoints(trajectory, pixels_per_meter=30, max_distance=20, color=(0, 255, 0)),
					(200, 200)
				)
				surround_map = np.clip(
					ego_car_map.astype(np.float32) + render_trajectory.astype(np.float32) + render_dp_trajectory.astype(np.float32),
					0, 255,
				).astype(np.uint8)
				decision_1s, decision_2s, decision_3s = parse_decision_sequence(pred_decision)
			else:
				surround_map = np.clip(
					ego_car_map.astype(np.float32) + render_dp_trajectory.astype(np.float32),
					0, 255,
				).astype(np.uint8)
				decision_1s, decision_2s, decision_3s = "", "", ""

			tick_data["predicted_trajectory"] = surround_map
			tick_data["decision_1s"] = decision_1s
			tick_data["decision_2s"] = decision_2s
			tick_data["decision_3s"] = decision_3s

			tick_data["rgb_raw"] = tick_data["rgb_front"]

			tick_data["rgb"] = cv2.resize(tick_data["rgb_front"], (800, 600))
			if 'bev_traj' not in tick_data:
				tick_data["bev_traj"] = np.zeros((400, 400, 3), dtype=np.uint8)
			tick_data["bev_traj"] = cv2.resize(tick_data["bev_traj"], (400, 400))

			tick_data["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
				control.throttle,
				control.steer,
				control.brake,
			)
			tick_data["speed"] = "speed: %.2f Km/h, target point x: %.2f m, target point y: %.2f m" % (gt_velocity*3.6, target_point.squeeze(0).cpu().float().numpy()[0], target_point.squeeze(0).cpu().float().numpy()[1])
			
			if USE_MOT:
				sentence1, sentence2 = split_prompt(prompt_cleaned)
				tick_data["language_1"] = "Instruction: " + sentence1
				tick_data["language_2"] = sentence2
			else:
				tick_data["language_1"] = ""
				tick_data["language_2"] = ""

			tick_data["mes"] = "speed: %.2f" % gt_velocity
			tick_data["time"] = "time: %.3f" % timestamp

			# surface = self._hic.run_interface(tick_data)
			# tick_data["surface"] = surface

		return control

	def save(self, tick_data):
		frame = self.step 
		Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
		
		# Draw trajectory on BEV image if available
		bev_img = tick_data['bev'].copy()
		if self.last_pred_traj is not None or self.last_dp_pred_traj is not None or self.last_route_pred is not None:
			# Pass last_route_pred for visualization (20 waypoints for lateral control, blue points)
			# Pass both target_point and next_target_point for visualization
			bev_img = self._draw_trajectory_on_bev(bev_img, self.last_pred_traj, self.last_target_point,
			                                        self.last_next_target_point, self.last_dp_pred_traj, self.last_route_pred,
			                                        self.last_waypoint_route)
		tick_data['bev_traj'] = bev_img
		Image.fromarray(bev_img).save(self.save_path / 'bev' / ('%04d.png' % frame))
		debug_img = self._compose_debug_visualization(tick_data['rgb_front'], bev_img)
		Image.fromarray(debug_img).save(self.save_path / 'debug_vis' / ('%04d.png' % frame))
		
		if 'lidar_bev' in tick_data:
			lidar_bev_tensor = tick_data['lidar_bev']
			if isinstance(lidar_bev_tensor, torch.Tensor):
				lidar_bev_tensor = lidar_bev_tensor.cpu().numpy()
			# Remove batch dim if present: (1, C, H, W) -> (C, H, W)
			while lidar_bev_tensor.ndim > 3:
				lidar_bev_tensor = lidar_bev_tensor[0]
			if lidar_bev_tensor.ndim == 3:
				# Take first 3 channels for RGB visualization
				lidar_bev_img = (lidar_bev_tensor[:3].transpose(1, 2, 0) * 255).astype(np.uint8)
			elif lidar_bev_tensor.ndim == 2:
				lidar_bev_img = (lidar_bev_tensor * 255).astype(np.uint8)
			else:
				lidar_bev_img = None
			if lidar_bev_img is not None:
				imageio.imwrite(str(self.save_path / 'lidar_bev' / (f'{frame:04d}.png')), lidar_bev_img)

		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

		# metric info
		outfile = open(self.save_path / 'metric_info.json', 'w')
		json.dump(self.metric_info, outfile, indent=4)
		outfile.close()

	def _compose_debug_visualization(self, rgb_img, bev_img):
		left = cv2.resize(rgb_img, (800, 600))
		right = cv2.resize(bev_img, (800, 600))

		overlay = right.copy()
		panel_top = 350
		cv2.rectangle(overlay, (20, panel_top), (780, 580), (20, 20, 20), -1)
		right = cv2.addWeighted(overlay, 0.45, right, 0.55, 0.0)

		status_lines = [
			f"frame: {self.step}",
			f"speed_kmh: {float(self.pid_metadata.get('speed', 0.0)) * 3.6:.2f}",
			f"steer: {float(self.pid_metadata.get('steer', 0.0)):.3f}",
			f"steer_ctrl: {float(self.pid_metadata.get('steer_controller', 0.0)):.3f}",
			f"throttle: {float(self.pid_metadata.get('throttle', 0.0)):.3f}",
			f"brake: {float(self.pid_metadata.get('brake', 0.0)):.3f}",
			f"command: {self.pid_metadata.get('command', 'N/A')}",
			f"target_pose: {self.pid_metadata.get('target_pose_source', 'N/A')}",
		]

		if self.last_target_point is not None:
			status_lines.append(
				f"target_point: [{self.last_target_point[0]:.2f}, {self.last_target_point[1]:.2f}]"
			)
		if self.last_next_target_point is not None:
			status_lines.append(
				f"next_target_point: [{self.last_next_target_point[0]:.2f}, {self.last_next_target_point[1]:.2f}]"
			)

		status_lines.extend([
			f"target_angle_deg: {float(self.pid_metadata.get('target_angle_deg', 0.0)):.2f}",
			f"target_is_right: {self.pid_metadata.get('target_is_right', False)}",
			f"target_is_left: {self.pid_metadata.get('target_is_left', False)}",
			f"target_is_behind: {self.pid_metadata.get('target_is_behind', False)}",
			f"steer_heading_error_deg: {float(self.pid_metadata.get('steer_heading_error_deg', 0.0)):.2f}",
			f"steer_target_idx: {self.pid_metadata.get('steer_target_idx', 'N/A')}",
			f"energy_front: {self.last_energy_debug.get('energy_front', float('nan')):.4f}",
			f"energy_left: {self.last_energy_debug.get('energy_left', float('nan')):.4f}",
			f"energy_right: {self.last_energy_debug.get('energy_right', float('nan')):.4f}",
			f"energy_ped: {self.last_energy_debug.get('energy_pedestrian', float('nan')):.4f}",
			f"energy_offroad: {self.last_energy_debug.get('energy_offroad', float('nan')):.4f}",
			f"energy_route: {self.last_energy_debug.get('energy_route', float('nan')):.4f}",
		])

		for idx, line in enumerate(status_lines):
			y = panel_top + 30 + idx * 22
			cv2.putText(
				right, line, (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
				(255, 255, 255), 1, cv2.LINE_AA
			)

		legend_items = [
			("traj", (0, 255, 0)),
			("dp_traj", (255, 0, 0)),
			("route", (0, 0, 255)),
			("wp_route", (255, 255, 255)),
			("target", (0, 255, 255)),
			("next_target", (255, 0, 255)),
		]
		legend_x = 35
		legend_y = 330
		for label, color in legend_items:
			cv2.circle(right, (legend_x, legend_y), 7, color, -1)
			cv2.putText(
				right, label, (legend_x + 18, legend_y + 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
			)
			legend_x += 145

		canvas = np.zeros((600, 1600, 3), dtype=np.uint8)
		canvas[:, :800] = left
		canvas[:, 800:] = right
		cv2.putText(canvas, "RGB", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(canvas, "BEV + Status", (820, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
		return canvas

	def _draw_trajectory_on_bev(self, bev_img, traj, target_point=None, next_target_point=None, dp_traj=None, route_pred=None, waypoint_route=None):
		"""
		Draw predicted trajectory on BEV image.
		
		BEV camera parameters:
		- Position: x=0, y=0, z=50 (50m height, looking down)
		- FOV: 50 degrees
		- Image size: 512x512
		
		Trajectory is in ego frame: [x, y] where x is forward, y is lateral with
		y > 0 meaning right (training-data / model convention from docs).
		BEV image: center is ego position, up is forward (negative x in image coords)
		
		Args:
			bev_img: numpy array (512, 512, 3) RGB image
			traj: numpy array (6, 2) trajectory points in ego frame [x_forward, y_right]
			target_point: numpy array (2,) target point in ego frame [x_forward, y_right], optional
			next_target_point: numpy array (2,) next target point in ego frame [x_forward, y_right], optional
			dp_traj: numpy array (6, 2) DP refined trajectory points in ego frame, optional
			route_pred: numpy array (20, 2) route waypoints for lateral control, optional
			waypoint_route: numpy array (N, 2) planner waypoint route in ego frame, optional
		
		Returns:
			bev_img: numpy array with trajectory drawn
		"""
		img_h, img_w = bev_img.shape[:2]  # 512, 512
		
		# BEV camera: z=50m, FOV=50 degrees
		# Calculate meters per pixel
		# FOV = 50 deg means the camera sees 50 degrees width/height
		# At z=50m, the ground coverage is: 2 * z * tan(FOV/2)
		fov_rad = np.deg2rad(50.0)
		ground_size = 2 * 50.0 * np.tan(fov_rad / 2)  # meters covered by the image
		meters_per_pixel = ground_size / img_w  # ~0.093 m/pixel
		
		# Image center is ego position
		cx, cy = img_w // 2, img_h // 2
		
		# Convert trajectory points to pixel coordinates
		# Model/data ego frame: x is forward, y is RIGHT (positive y = right)
		# BEV image: center is ego, up (-row) is forward, right (+col) is right
		# So: pixel_col = cx + y / meters_per_pixel
		#     pixel_row = cy - x / meters_per_pixel (x forward -> -row, i.e., up)
		
		pixels = []
		if traj is not None:
			for i in range(len(traj)):
				x, y = traj[i]  # x: forward, y: right (data/model convention)
				pixel_col = int(cx + y / meters_per_pixel)
				pixel_row = int(cy - x / meters_per_pixel)
				pixels.append((pixel_col, pixel_row))
		
		# Draw trajectory using cv2
		# Draw lines connecting waypoints
		for i in range(len(pixels) - 1):
			pt1 = pixels[i]
			pt2 = pixels[i + 1]
			# Check if points are within image bounds
			if (0 <= pt1[0] < img_w and 0 <= pt1[1] < img_h and
				0 <= pt2[0] < img_w and 0 <= pt2[1] < img_h):
				cv2.line(bev_img, pt1, pt2, (0, 255, 0), 2)  # Green line
		
		# Draw waypoints as circles
		for i, (col, row) in enumerate(pixels):
			if 0 <= col < img_w and 0 <= row < img_h:
				# Color gradient: start (red) -> end (blue)
				color_r = int(255 * (1 - i / (len(pixels) - 1)))
				color_b = int(255 * (i / (len(pixels) - 1)))
				cv2.circle(bev_img, (col, row), 5, (color_r, 0, color_b), -1)
		
		# Draw DP trajectory if provided (red color)
		if dp_traj is not None:
			dp_pixels = []
			for i in range(len(dp_traj)):
				x, y = dp_traj[i]  # x: forward, y: right (data/model convention)
				pixel_col = int(cx + y / meters_per_pixel)
				pixel_row = int(cy - x / meters_per_pixel)
				dp_pixels.append((pixel_col, pixel_row))
			
			# Draw DP trajectory lines (red)
			for i in range(len(dp_pixels) - 1):
				pt1 = dp_pixels[i]
				pt2 = dp_pixels[i + 1]
				if (0 <= pt1[0] < img_w and 0 <= pt1[1] < img_h and
					0 <= pt2[0] < img_w and 0 <= pt2[1] < img_h):
					cv2.line(bev_img, pt1, pt2, (255, 0, 0), 2)  # Red line
			
			# Draw DP waypoints as circles (red with gradient to orange)
			for i, (col, row) in enumerate(dp_pixels):
				if 0 <= col < img_w and 0 <= row < img_h:
					# Color gradient: start (red) -> end (orange)
					color_g = int(128 * (i / (len(dp_pixels) - 1))) if len(dp_pixels) > 1 else 0
					cv2.circle(bev_img, (col, row), 4, (255, color_g, 0), -1)
		
		# Draw route_pred waypoints if provided (blue color) - used for lateral/steering control
		if route_pred is not None:
			route_pixels = []
			for i in range(len(route_pred)):
				x, y = route_pred[i]  # x: forward, y: right (data/model convention)
				pixel_col = int(cx + y / meters_per_pixel)
				pixel_row = int(cy - x / meters_per_pixel)
				route_pixels.append((pixel_col, pixel_row))
			
			# Draw route_pred trajectory lines (blue)
			for i in range(len(route_pixels) - 1):
				pt1 = route_pixels[i]
				pt2 = route_pixels[i + 1]
				if (0 <= pt1[0] < img_w and 0 <= pt1[1] < img_h and
					0 <= pt2[0] < img_w and 0 <= pt2[1] < img_h):
					cv2.line(bev_img, pt1, pt2, (255, 165, 0), 1)  # Orange line (thinner)
			
			# Draw route_pred waypoints as circles (blue)
			for i, (col, row) in enumerate(route_pixels):
				if 0 <= col < img_w and 0 <= row < img_h:
					# Solid blue points for route_pred
					cv2.circle(bev_img, (col, row), 3, (0, 0, 255), -1)  # Blue circles (smaller)

		# Draw planner waypoint_route if provided (white color)
		if waypoint_route is not None:
			wp_pixels = []
			for i in range(len(waypoint_route)):
				x, y = waypoint_route[i]
				pixel_col = int(cx + y / meters_per_pixel)
				pixel_row = int(cy - x / meters_per_pixel)
				wp_pixels.append((pixel_col, pixel_row))

			for i in range(len(wp_pixels) - 1):
				pt1 = wp_pixels[i]
				pt2 = wp_pixels[i + 1]
				if (0 <= pt1[0] < img_w and 0 <= pt1[1] < img_h and
					0 <= pt2[0] < img_w and 0 <= pt2[1] < img_h):
					cv2.line(bev_img, pt1, pt2, (255, 255, 255), 1)

			for i, (col, row) in enumerate(wp_pixels):
				if 0 <= col < img_w and 0 <= row < img_h:
					radius = 4 if i in (0, len(wp_pixels) - 1) else 2
					cv2.circle(bev_img, (col, row), radius, (255, 255, 255), -1)
		
		target_overlaps_next = (
			target_point is not None and
			next_target_point is not None and
			np.linalg.norm(np.asarray(target_point) - np.asarray(next_target_point)) < 1e-3
		)

		# Draw target point if provided (cyan/aqua color with larger circle)
		if target_point is not None:
			x, y = target_point[0], target_point[1]  # x: forward, y: right (data/model convention)
			tp_col = int(cx + y / meters_per_pixel)
			tp_row = int(cy - x / meters_per_pixel)
			if 0 <= tp_col < img_w and 0 <= tp_row < img_h:
				if target_overlaps_next:
					cv2.circle(bev_img, (tp_col, tp_row), 13, (255, 0, 255), -1)
					cv2.circle(bev_img, (tp_col, tp_row), 8, (0, 255, 255), -1)
					cv2.circle(bev_img, (tp_col, tp_row), 15, (255, 255, 255), 2)
					cv2.putText(bev_img, "TP/NTP", (tp_col + 10, tp_row - 10),
					            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
				else:
					cv2.circle(bev_img, (tp_col, tp_row), 10, (0, 255, 255), -1)
					cv2.circle(bev_img, (tp_col, tp_row), 12, (255, 255, 255), 2)
		
		# Draw next target point if provided (magenta/pink color with larger circle)
		if next_target_point is not None and not target_overlaps_next:
			x, y = next_target_point[0], next_target_point[1]  # x: forward, y: right (data/model convention)
			ntp_col = int(cx + y / meters_per_pixel)
			ntp_row = int(cy - x / meters_per_pixel)
			if 0 <= ntp_col < img_w and 0 <= ntp_row < img_h:
				cv2.circle(bev_img, (ntp_col, ntp_row), 10, (255, 0, 255), -1)  # Magenta circle for next target point
				cv2.circle(bev_img, (ntp_col, ntp_row), 12, (255, 255, 255), 2)  # White border
		
		# Draw ego position (center)
		cv2.circle(bev_img, (cx, cy), 8, (255, 255, 0), -1)  # Yellow circle for ego
		
		# Draw ego-frame axes to make the BEV convention explicit.
		# In this visualization: forward is up, right is right.
		axis_len = 42
		forward_end = (cx, cy - axis_len)
		right_end = (cx + axis_len, cy)
		left_end = (cx - axis_len, cy)
		cv2.arrowedLine(bev_img, (cx, cy), forward_end, (255, 255, 0), 2, tipLength=0.25)
		cv2.arrowedLine(bev_img, (cx, cy), right_end, (0, 255, 255), 2, tipLength=0.25)
		cv2.arrowedLine(bev_img, (cx, cy), left_end, (160, 160, 160), 1, tipLength=0.2)
		cv2.putText(bev_img, "FWD", (forward_end[0] + 8, forward_end[1] - 6),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
		cv2.putText(bev_img, "RIGHT(+y)", (right_end[0] + 6, right_end[1] - 8),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
		cv2.putText(bev_img, "LEFT", (left_end[0] - 46, left_end[1] - 8),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
		
		return bev_img

	def destroy(self):
		del self.net
		torch.cuda.empty_cache()

	def gps_to_location(self, gps):
		# gps content: numpy array: [lat, lon, alt]
		lat, lon = gps
		scale = math.cos(self.lat_ref * math.pi / 180.0)
		my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
		mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
		y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
		x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
		return np.array([x, y])
