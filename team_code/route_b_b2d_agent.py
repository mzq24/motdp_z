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
# This agent file now lives inside the MoT-DP repo itself, so the repo root is
# simply the parent directory of team_code/. Do not reconstruct a sibling
# "MoT-DP" path from a Bench2Drive-style layout.
project_root = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
mot_dp_path = project_root
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
from team_code.ego_localizer import EgoLocalizer
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
LOCALIZER_STRATEGY = os.environ.get('LOCALIZER_STRATEGY', 'complementary').lower()
LOCALIZER_ALPHA = float(os.environ.get('LOCALIZER_ALPHA', '0.5'))
LIDAR_POSE_SOURCE = os.environ.get('LIDAR_POSE_SOURCE', 'ukf').lower()
STEER_SIGN_SCALE = float(os.environ.get('STEER_SIGN_SCALE', '1.0'))
TARGET_YAW_SIGN = float(os.environ.get('TARGET_YAW_SIGN', '1.0'))
TARGET_GEOM_YAW_SIGN = float(os.environ.get('TARGET_GEOM_YAW_SIGN', '1.0'))
SOFT_SPEED_LIMIT_MS = float(os.environ.get('SOFT_SPEED_LIMIT_MS', '0.0'))
HARD_SPEED_LIMIT_MS = float(os.environ.get('HARD_SPEED_LIMIT_MS', str(35.0 / 3.6)))
NUM_INFERENCE_STEPS_OVERRIDE = os.environ.get('NUM_INFERENCE_STEPS_OVERRIDE', '').strip()
TERMINAL_ROUTE_ACTIVE_POINTS_MAX = int(os.environ.get('TERMINAL_ROUTE_ACTIVE_POINTS_MAX', '2'))
TERMINAL_ROUTE_NEAR_DISTANCE_M = float(os.environ.get('TERMINAL_ROUTE_NEAR_DISTANCE_M', '3.0'))
TERMINAL_ROUTE_SPEED_CAP_MS = float(os.environ.get('TERMINAL_ROUTE_SPEED_CAP_MS', '1.2'))
TERMINAL_ROUTE_BEHIND_SPEED_CAP_MS = float(os.environ.get('TERMINAL_ROUTE_BEHIND_SPEED_CAP_MS', '0.8'))
SPEED_SOURCE = os.environ.get('SPEED_SOURCE', 'speed_head').lower()  # 'speed_head', 'traj', 'traj_05s', 'fuse', 'fuse_traj', 'fuse3_median', 'fuse3_adaptive'
STUCK_HELPER_TARGET_INSERT_ENABLE = os.environ.get('STUCK_HELPER_TARGET_INSERT_ENABLE', '1').lower() in (
    '1', 'true', 'yes', 'on'
)
STUCK_HELPER_TARGET_1_FORWARD_M = float(os.environ.get('STUCK_HELPER_TARGET_1_FORWARD_M', '3.63'))
STUCK_HELPER_TARGET_2_FORWARD_M = float(os.environ.get('STUCK_HELPER_TARGET_2_FORWARD_M', '25.63'))
STUCK_HELPER_TARGET_LATERAL_M = float(os.environ.get('STUCK_HELPER_TARGET_LATERAL_M', '-3.145'))
STUCK_HELPER_STARTUP_RECOVERY_MODE = os.environ.get(
    'STUCK_HELPER_STARTUP_RECOVERY_MODE', 'control'
).lower()
STUCK_HELPER_CONTROL_THROTTLE = float(os.environ.get('STUCK_HELPER_CONTROL_THROTTLE', '0.5'))
STUCK_HELPER_CONTROL_STEER = float(os.environ.get('STUCK_HELPER_CONTROL_STEER', '-1.0'))
STUCK_HELPER_RELEASE_HEADING_DEG = float(os.environ.get('STUCK_HELPER_RELEASE_HEADING_DEG', '20.0'))
STUCK_HELPER_STARTUP_DISTANCE_M = float(os.environ.get('STUCK_HELPER_STARTUP_DISTANCE_M', '10.0'))
STUCK_HELPER_STARTUP_THRESHOLD = int(os.environ.get('STUCK_HELPER_STARTUP_THRESHOLD', '100'))
STUCK_HELPER_POSTSTART_THRESHOLD = int(os.environ.get('STUCK_HELPER_POSTSTART_THRESHOLD', '300'))
JUNCTION_WINDOW_SOFT_SPEED_CAP_ENABLE = os.environ.get('JUNCTION_WINDOW_SOFT_SPEED_CAP_ENABLE', '1').lower() in (
    '1', 'true', 'yes', 'on'
)
JUNCTION_WINDOW_SOFT_SPEED_CAP_MS = float(os.environ.get('JUNCTION_WINDOW_SOFT_SPEED_CAP_MS', '20.0'))
JUNCTION_WINDOW_SOFT_SPEED_CAP_SINGLE_THRESHOLD = float(os.environ.get('JUNCTION_WINDOW_SOFT_SPEED_CAP_SINGLE_THRESHOLD', '0.2'))
JUNCTION_WINDOW_SOFT_SPEED_CAP_CONSEC_THRESHOLD = float(os.environ.get('JUNCTION_WINDOW_SOFT_SPEED_CAP_CONSEC_THRESHOLD', '0.1'))
JUNCTION_WINDOW_SOFT_SPEED_CAP_CONSEC_FRAMES = max(1, int(os.environ.get('JUNCTION_WINDOW_SOFT_SPEED_CAP_CONSEC_FRAMES', '3')))
EARLY_TARGET_PROMOTE_ENABLE = os.environ.get('EARLY_TARGET_PROMOTE_ENABLE', '1').lower() in (
    '1', 'true', 'yes', 'on'
)
EARLY_TARGET_PROMOTE_CUR_FORWARD_MAX_M = float(os.environ.get('EARLY_TARGET_PROMOTE_CUR_FORWARD_MAX_M', '4.0'))
EARLY_TARGET_PROMOTE_CUR_LATERAL_MIN_M = float(os.environ.get('EARLY_TARGET_PROMOTE_CUR_LATERAL_MIN_M', '8.0'))
EARLY_TARGET_PROMOTE_CUR_ANGLE_MIN_DEG = float(os.environ.get('EARLY_TARGET_PROMOTE_CUR_ANGLE_MIN_DEG', '45.0'))
EARLY_TARGET_PROMOTE_NEXT_FORWARD_MIN_M = float(os.environ.get('EARLY_TARGET_PROMOTE_NEXT_FORWARD_MIN_M', '20.0'))
EARLY_TARGET_PROMOTE_NEXT_ANGLE_MAX_DEG = float(os.environ.get('EARLY_TARGET_PROMOTE_NEXT_ANGLE_MAX_DEG', '30.0'))
SAVE_TRANSFUSER_BEV_DEBUG = os.environ.get('SAVE_TRANSFUSER_BEV_DEBUG', '0').lower() in (
    '1', 'true', 'yes', 'on'
)
FRONT_ROUTE_RISK_SPEED_CAP_ENABLE = os.environ.get('FRONT_ROUTE_RISK_SPEED_CAP_ENABLE', '1').lower() in (
	'1', 'true', 'yes', 'on'
)
FRONT_ROUTE_RISK_RECENT_WINDOW = max(1, int(os.environ.get('FRONT_ROUTE_RISK_RECENT_WINDOW', '8')))
FRONT_ROUTE_RISK_HOLD_FRAMES = max(0, int(os.environ.get('FRONT_ROUTE_RISK_HOLD_FRAMES', '8')))
FRONT_ROUTE_RISK_MIN_SPEED_MS = float(os.environ.get('FRONT_ROUTE_RISK_MIN_SPEED_MS', '5.0'))
FRONT_ROUTE_RISK_GEOM_ANGLE_DEG = float(os.environ.get('FRONT_ROUTE_RISK_GEOM_ANGLE_DEG', '12.0'))
FRONT_ROUTE_RISK_GEOM_LATERAL_M = float(os.environ.get('FRONT_ROUTE_RISK_GEOM_LATERAL_M', '2.5'))
FRONT_ROUTE_RISK_MED_SIGMOID = float(os.environ.get('FRONT_ROUTE_RISK_MED_SIGMOID', '0.76'))
FRONT_ROUTE_RISK_HIGH_SIGMOID = float(os.environ.get('FRONT_ROUTE_RISK_HIGH_SIGMOID', '0.86'))
FRONT_ROUTE_RISK_HIGH_EMA = float(os.environ.get('FRONT_ROUTE_RISK_HIGH_EMA', '1.35'))
FRONT_ROUTE_RISK_MED_CAP_MS = float(os.environ.get('FRONT_ROUTE_RISK_MED_CAP_MS', '6.5'))
FRONT_ROUTE_RISK_HIGH_CAP_MS = float(os.environ.get('FRONT_ROUTE_RISK_HIGH_CAP_MS', '4.5'))
FRONT_ROUTE_RISK_EARLY_SPEED_START_MS = float(os.environ.get('FRONT_ROUTE_RISK_EARLY_SPEED_START_MS', '6.5'))
FRONT_ROUTE_RISK_EARLY_SPEED_FULL_MS = float(os.environ.get('FRONT_ROUTE_RISK_EARLY_SPEED_FULL_MS', '9.0'))
FRONT_ROUTE_RISK_MED_SIGMOID_SPEED_DELTA = float(os.environ.get('FRONT_ROUTE_RISK_MED_SIGMOID_SPEED_DELTA', '0.16'))
FRONT_ROUTE_RISK_HIGH_SIGMOID_SPEED_DELTA = float(os.environ.get('FRONT_ROUTE_RISK_HIGH_SIGMOID_SPEED_DELTA', '0.14'))
FRONT_ROUTE_RISK_HIGH_EMA_SPEED_DELTA = float(os.environ.get('FRONT_ROUTE_RISK_HIGH_EMA_SPEED_DELTA', '0.50'))
FRONT_ROUTE_RISK_MED_SIGMOID_MIN = float(os.environ.get('FRONT_ROUTE_RISK_MED_SIGMOID_MIN', '0.60'))
FRONT_ROUTE_RISK_HIGH_SIGMOID_MIN = float(os.environ.get('FRONT_ROUTE_RISK_HIGH_SIGMOID_MIN', '0.72'))
FRONT_ROUTE_RISK_HIGH_EMA_MIN = float(os.environ.get('FRONT_ROUTE_RISK_HIGH_EMA_MIN', '0.80'))
FRONT_ROUTE_RISK_RAW_OVERRIDE_SPEED_MS = float(os.environ.get('FRONT_ROUTE_RISK_RAW_OVERRIDE_SPEED_MS', '8.0'))
FRONT_ROUTE_RISK_RAW_OVERRIDE_SCORE = float(os.environ.get('FRONT_ROUTE_RISK_RAW_OVERRIDE_SCORE', '1.6'))
STAGE1_ENERGY_SPEED_CAP_ENABLE = os.environ.get('STAGE1_ENERGY_SPEED_CAP_ENABLE', '0').lower() in (
	'1', 'true', 'yes', 'on'
)
STAGE1_ENERGY_SPEED_CONTROL_MODE = os.environ.get('STAGE1_ENERGY_SPEED_CONTROL_MODE', 'cap').strip().lower()
STAGE1_ENERGY_CAP_HOLD_FRAMES = max(0, int(os.environ.get('STAGE1_ENERGY_CAP_HOLD_FRAMES', '6')))
STAGE1_ENERGY_CAP_MIN_SPEED_MS = float(os.environ.get('STAGE1_ENERGY_CAP_MIN_SPEED_MS', '1.0'))
STAGE1_ENERGY_CAP_LOOKAHEAD_BINS = max(0, int(os.environ.get('STAGE1_ENERGY_CAP_LOOKAHEAD_BINS', '2')))
STAGE1_ENERGY_CAP_CUR_WARN = float(os.environ.get('STAGE1_ENERGY_CAP_CUR_WARN', '0.20'))
STAGE1_ENERGY_CAP_CUR_HIGH = float(os.environ.get('STAGE1_ENERGY_CAP_CUR_HIGH', '0.40'))
STAGE1_ENERGY_CAP_PEAK_WARN = float(os.environ.get('STAGE1_ENERGY_CAP_PEAK_WARN', '0.35'))
STAGE1_ENERGY_CAP_PEAK_HIGH = float(os.environ.get('STAGE1_ENERGY_CAP_PEAK_HIGH', '0.55'))
STAGE1_ENERGY_CAP_SAFE_WARN = float(os.environ.get('STAGE1_ENERGY_CAP_SAFE_WARN', '0.15'))
STAGE1_ENERGY_CAP_SAFE_HIGH = float(os.environ.get('STAGE1_ENERGY_CAP_SAFE_HIGH', '0.08'))
STAGE1_ENERGY_CAP_TARGET_WARN = float(os.environ.get('STAGE1_ENERGY_CAP_TARGET_WARN', '0.35'))
STAGE1_ENERGY_CAP_TARGET_HIGH = float(os.environ.get('STAGE1_ENERGY_CAP_TARGET_HIGH', '0.55'))
STAGE1_ENERGY_CAP_TREND_MARGIN = float(os.environ.get('STAGE1_ENERGY_CAP_TREND_MARGIN', '0.05'))
STAGE1_ENERGY_CAP_SPEED_EPS = float(os.environ.get('STAGE1_ENERGY_CAP_SPEED_EPS', '0.20'))
STAGE1_ENERGY_GRADIENT_GAIN = float(os.environ.get('STAGE1_ENERGY_GRADIENT_GAIN', '2.0'))
STAGE1_ENERGY_GRADIENT_MAX_DELTA_MS = float(os.environ.get('STAGE1_ENERGY_GRADIENT_MAX_DELTA_MS', '1.5'))
STAGE1_ENERGY_GRADIENT_MIN_SCORE = float(os.environ.get('STAGE1_ENERGY_GRADIENT_MIN_SCORE', '0.15'))
STAGE1_ENERGY_GRADIENT_SLOPE_EPS = float(os.environ.get('STAGE1_ENERGY_GRADIENT_SLOPE_EPS', '0.03'))
STAGE1_ENERGY_GRADIENT_CHASE_WEIGHT = float(os.environ.get('STAGE1_ENERGY_GRADIENT_CHASE_WEIGHT', '0.10'))
STAGE1_ENERGY_GRADIENT_MEET_WEIGHT = float(os.environ.get('STAGE1_ENERGY_GRADIENT_MEET_WEIGHT', '1.00'))
STAGE1_ENERGY_GRADIENT_PEDESTRIAN_WEIGHT = float(os.environ.get('STAGE1_ENERGY_GRADIENT_PEDESTRIAN_WEIGHT', '2.00'))
STAGE1_ENERGY_GRADIENT_CHASE_MIN_SCORE = float(os.environ.get('STAGE1_ENERGY_GRADIENT_CHASE_MIN_SCORE', '0.25'))
STAGE1_ENERGY_GRADIENT_MEET_MIN_SCORE = float(os.environ.get('STAGE1_ENERGY_GRADIENT_MEET_MIN_SCORE', '0.15'))
STAGE1_ENERGY_GRADIENT_PEDESTRIAN_MIN_SCORE = float(os.environ.get('STAGE1_ENERGY_GRADIENT_PEDESTRIAN_MIN_SCORE', '0.08'))

ROAD_OPTION_TEXT = {
	1: 'left',
	2: 'right',
	3: 'straight',
	4: 'lane_follow',
	5: 'change_left',
	6: 'change_right',
}

SEMANTIC_STOP_SIGN_CLASS = 5
SEMANTIC_LIGHT_GREEN_CLASS = 6
SEMANTIC_LIGHT_YELLOW_CLASS = 7
SEMANTIC_LIGHT_RED_CLASS = 8
SEMANTIC_VEHICLE_CLASS = 9

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

    for key in ['anchor_path', 'abs_stats_path', 'delta_stats_path', 'global_abs_stats_path', 'route_abs_stats_path']:
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
	def _command_value_to_text(self, command_value):
		try:
			command_int = int(command_value)
		except (TypeError, ValueError):
			return 'unknown'
		return ROAD_OPTION_TEXT.get(command_int, f'unknown({command_int})')

	def _format_debug_value(self, value, fmt=".3f"):
		try:
			value_float = float(value)
		except (TypeError, ValueError):
			return "NA"
		if not np.isfinite(value_float):
			return "NA"
		return format(value_float, fmt)

	def _format_energy_peak_with_speed(self):
		peak_value = self.pid_metadata.get('speed_energy_max')
		peak_idx = self.pid_metadata.get('speed_energy_argmax')
		samples = self.pid_metadata.get('speed_energy_samples')
		peak_str = self._format_debug_value(peak_value, '.1f')
		try:
			peak_idx_int = int(peak_idx)
		except (TypeError, ValueError):
			peak_idx_int = None

		if isinstance(samples, (list, tuple)) and peak_idx_int is not None and 0 <= peak_idx_int < len(samples):
			speed_str = self._format_debug_value(samples[peak_idx_int], '.1f')
			return f"{peak_str}@{speed_str}m/s"
		if peak_idx_int is not None:
			return f"{peak_str}@idx{peak_idx_int}"
		return peak_str

	def _build_transfuser_bev_semantic_decoder(self, model_path, device):
		decoder = torch.nn.Sequential(
			torch.nn.Conv2d(
				self.transfuser_config.bev_features_chanels,
				self.transfuser_config.bev_features_chanels,
				kernel_size=3,
				stride=1,
				padding=1,
				bias=True,
			),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(
				self.transfuser_config.bev_features_chanels,
				self.transfuser_config.num_bev_semantic_classes,
				kernel_size=1,
				stride=1,
				padding=0,
				bias=True,
			),
			torch.nn.Upsample(
				size=(
					self.transfuser_config.lidar_resolution_height,
					self.transfuser_config.lidar_resolution_width,
				),
				mode='bilinear',
				align_corners=False,
			),
		).to(device)

		state_dict = torch.load(model_path, map_location='cpu')
		prefix = 'bev_semantic_decoder.'
		decoder_state = {
			key[len(prefix):]: value
			for key, value in state_dict.items()
			if key.startswith(prefix)
		}
		if not decoder_state:
			raise RuntimeError(f"Missing BEV semantic decoder weights in {model_path}")
		decoder.load_state_dict(decoder_state, strict=True)
		for param in decoder.parameters():
			param.requires_grad = False
		decoder.eval()
		return decoder

	def _init_semantic_hazard_state(self):
		self.semantic_bev_pixels_per_meter = 2.0
		self.semantic_tl_min_pixels = 6
		self.semantic_tl_brake_distance_m = float(
			os.environ.get('SEMANTIC_TL_BRAKE_DISTANCE_M', '0.1')
		)
		self.semantic_stop_min_pixels = 3
		self.semantic_tl_min_models = max(
			1,
			int(os.environ.get('SEMANTIC_TL_MIN_MODELS', '2'))
		)
		self.semantic_stop_min_models = max(
			1,
			int(os.environ.get('SEMANTIC_STOP_MIN_MODELS', '1'))
		)
		self.semantic_tl_roi = (0.0, 25.0, -10.0, 10.0)
		self.semantic_tl_green_release_delay_frames = int(
			os.environ.get('SEMANTIC_TL_GREEN_RELEASE_DELAY_FRAMES', '0')
		)
		self.semantic_tl_green_release_speed_threshold = float(
			os.environ.get('SEMANTIC_TL_GREEN_RELEASE_SPEED_THRESHOLD', '0.5')
		)
		self.semantic_stop_roi = (0.0, 25.0, -10.0, 10.0)
		self.semantic_stop_brake_distance_m = 6.0
		self.semantic_stop_min_stop_frames = 10
		self.semantic_stop_reset_missing_frames = 5
		self.semantic_stop_arm_speed_threshold = float(
			os.environ.get('SEMANTIC_STOP_ARM_SPEED_THRESHOLD', '0.1')
		)
		self.semantic_stop_max_apply_events = max(
			0,
			int(os.environ.get('SEMANTIC_STOP_MAX_APPLY_EVENTS', '2'))
		)
		self.semantic_planner_stop_speed_threshold = 0.05

		self.semantic_tl_prev_state = 'none'
		self.semantic_tl_green_hold_frames = 0
		self.semantic_stop_state = 'NONE'
		self.semantic_stop_stopped_frames = 0
		self.semantic_stop_missing_frames = 0
		self.semantic_stop_apply_events = 0
		self.semantic_stop_prev_apply_stop = False
		self.semantic_stop_disabled = False
		self.last_semantic_debug = {}

	def _decode_bev_semantic_classes(self, bev_feature_upscale):
		if bev_feature_upscale is None:
			return None
		if isinstance(bev_feature_upscale, (list, tuple)):
			decoded_classes = []
			for feature_upscale, decoder in zip(
				bev_feature_upscale,
				self.transfuser_bev_semantic_decoders,
			):
				decoder_device = next(decoder.parameters()).device
				with torch.no_grad():
					semantic_logits = decoder(
						feature_upscale.to(device=decoder_device, dtype=torch.float32)
					)
				decoded_classes.append(
					semantic_logits.argmax(dim=1)
					.squeeze(0)
					.detach()
					.cpu()
					.numpy()
					.astype(np.uint8)
				)
			return decoded_classes

		decoder_device = next(self.transfuser_bev_semantic_decoder.parameters()).device
		with torch.no_grad():
			semantic_logits = self.transfuser_bev_semantic_decoder(
				bev_feature_upscale.to(device=decoder_device, dtype=torch.float32)
			)
		return (
			semantic_logits.argmax(dim=1)
			.squeeze(0)
			.detach()
			.cpu()
			.numpy()
			.astype(np.uint8)
		)

	def _get_semantic_vote_threshold(self, class_id, num_models):
		if num_models <= 1:
			return 1
		if class_id == SEMANTIC_STOP_SIGN_CLASS:
			return min(num_models, self.semantic_stop_min_models)
		if class_id in (
			SEMANTIC_LIGHT_RED_CLASS,
			SEMANTIC_LIGHT_YELLOW_CLASS,
			SEMANTIC_LIGHT_GREEN_CLASS,
		):
			return min(num_models, self.semantic_tl_min_models)
		return max(1, (num_models + 1) // 2)

	def _fuse_bev_semantic_classes_for_visualization(self, bev_classes):
		if bev_classes is None:
			return None
		if not isinstance(bev_classes, (list, tuple)):
			return np.asarray(bev_classes, dtype=np.uint8)
		if len(bev_classes) == 0:
			return None
		if len(bev_classes) == 1:
			return np.asarray(bev_classes[0], dtype=np.uint8)

		stack = np.stack([np.asarray(bc, dtype=np.uint8) for bc in bev_classes], axis=0)
		fused = np.zeros_like(stack[0], dtype=np.uint8)
		num_models = stack.shape[0]
		num_classes = int(getattr(self.transfuser_config, 'num_bev_semantic_classes', 11))

		for class_id in range(num_classes):
			threshold = self._get_semantic_vote_threshold(class_id, num_models)
			class_votes = (stack == class_id).sum(axis=0)
			fused[class_votes >= threshold] = class_id

		return fused

	def _rotate_bev_ego_up(self, bev_img):
		if bev_img is None:
			return None
		return np.rot90(bev_img, k=1).copy()

	def _render_transfuser_lidar_bev_image(self, transfuser_lidar_bev_tensor):
		if transfuser_lidar_bev_tensor is None:
			return None
		if isinstance(transfuser_lidar_bev_tensor, torch.Tensor):
			lidar_bev = transfuser_lidar_bev_tensor.detach().cpu().numpy()
		else:
			lidar_bev = np.asarray(transfuser_lidar_bev_tensor)

		while lidar_bev.ndim > 3:
			lidar_bev = lidar_bev[0]
		if lidar_bev.ndim != 3:
			return None

		channels = min(3, lidar_bev.shape[0])
		lidar_vis = lidar_bev[:channels].transpose(1, 2, 0)
		lidar_vis = np.clip(lidar_vis * 255.0, 0, 255).astype(np.uint8)
		if lidar_vis.shape[2] == 1:
			lidar_vis = np.repeat(lidar_vis, 3, axis=2)
		elif lidar_vis.shape[2] == 2:
			third = np.zeros_like(lidar_vis[:, :, :1])
			lidar_vis = np.concatenate([lidar_vis, third], axis=2)

		return self._rotate_bev_ego_up(lidar_vis)

	def _render_bev_semantic_image(self, bev_classes):
		fused_classes = self._fuse_bev_semantic_classes_for_visualization(bev_classes)
		if fused_classes is None:
			return None, None

		palette = np.array([
			[0, 0, 0],         # 0 unlabeled
			[90, 90, 90],      # 1 road
			[160, 160, 160],   # 2 sidewalk
			[255, 215, 0],     # 3 lane_solid
			[255, 255, 255],   # 4 lane_broken
			[255, 140, 0],     # 5 stop_sign
			[0, 200, 0],       # 6 light_green
			[255, 215, 0],     # 7 light_yellow
			[220, 30, 30],     # 8 light_red
			[0, 120, 255],     # 9 vehicle
			[255, 0, 255],     # 10 walker
		], dtype=np.uint8)

		semantic_vis = palette[np.clip(fused_classes, 0, len(palette) - 1)]
		vehicle_mask = (fused_classes == SEMANTIC_VEHICLE_CLASS).astype(np.uint8) * 255
		vehicle_vis = np.stack([vehicle_mask, vehicle_mask, vehicle_mask], axis=-1)

		return (
			self._rotate_bev_ego_up(semantic_vis),
			self._rotate_bev_ego_up(vehicle_vis),
		)

	def _jsonify_debug_value(self, value):
		if isinstance(value, np.ndarray):
			return value.tolist()
		if isinstance(value, torch.Tensor):
			return value.detach().cpu().tolist()
		if isinstance(value, (np.floating,)):
			return float(value)
		if isinstance(value, (np.integer,)):
			return int(value)
		if isinstance(value, (list, tuple)):
			return [self._jsonify_debug_value(v) for v in value]
		if isinstance(value, dict):
			return {k: self._jsonify_debug_value(v) for k, v in value.items()}
		return value

	def _format_debug_curve(self, values, fmt='.2f', max_items=7):
		if values is None:
			return 'NA'
		if isinstance(values, torch.Tensor):
			arr = values.detach().cpu().float().reshape(-1).tolist()
		else:
			arr = np.asarray(values, dtype=np.float32).reshape(-1).tolist()
		if len(arr) == 0:
			return '[]'
		arr = arr[:max_items]
		parts = [format(float(v), fmt) for v in arr]
		return '[' + ', '.join(parts) + ']'

	def _format_debug_first(self, values, fmt='.2f'):
		if values is None:
			return 'NA'
		if isinstance(values, torch.Tensor):
			arr = values.detach().cpu().float().reshape(-1).tolist()
		else:
			arr = np.asarray(values, dtype=np.float32).reshape(-1).tolist()
		if len(arr) == 0:
			return 'NA'
		return self._format_debug_value(arr[0], fmt)

	def _format_debug_speed_curve_kmh(self, values, fmt='.1f', max_items=7):
		if values is None:
			return 'NA'
		if isinstance(values, torch.Tensor):
			arr = values.detach().cpu().float().reshape(-1).tolist()
		else:
			arr = np.asarray(values, dtype=np.float32).reshape(-1).tolist()
		if len(arr) == 0:
			return '[]'
		arr = arr[:max_items]
		parts = [format(float(v) * 3.6, fmt) for v in arr]
		return '[' + ', '.join(parts) + ']'

	def _bev_roi_bounds(self, bev_classes, x_min_m, x_max_m, y_min_m, y_max_m):
		if isinstance(bev_classes, (list, tuple)):
			if len(bev_classes) == 0:
				return 0, 0, 0, 0
			bev_reference = bev_classes[0]
		else:
			bev_reference = bev_classes

		height, width = bev_reference.shape
		center_col = width / 2.0
		center_row = height / 2.0
		ppm = self.semantic_bev_pixels_per_meter

		col_start = int(np.floor(center_col + x_min_m * ppm))
		col_end = int(np.ceil(center_col + x_max_m * ppm))
		row_start = int(np.floor(center_row + y_min_m * ppm))
		row_end = int(np.ceil(center_row + y_max_m * ppm))

		col_start = max(0, min(width, col_start))
		col_end = max(0, min(width, col_end))
		row_start = max(0, min(height, row_start))
		row_end = max(0, min(height, row_end))
		return row_start, row_end, col_start, col_end

	def _semantic_class_stats(self, bev_classes, class_id, roi):
		x_min_m, x_max_m, y_min_m, y_max_m = roi
		row_start, row_end, col_start, col_end = self._bev_roi_bounds(
			bev_classes, x_min_m, x_max_m, y_min_m, y_max_m
		)
		if row_end <= row_start or col_end <= col_start:
			return {
				'count': 0,
				'closest_forward_m': None,
				'mean_forward_m': None,
				'mean_lateral_m': None,
			}

		if isinstance(bev_classes, (list, tuple)):
			if len(bev_classes) == 0:
				return {
					'count': 0,
					'closest_forward_m': None,
					'mean_forward_m': None,
					'mean_lateral_m': None,
				}
			masks = np.stack(
				[
					bev_class[row_start:row_end, col_start:col_end] == class_id
					for bev_class in bev_classes
				],
				axis=0,
			)
			vote_threshold = self._get_semantic_vote_threshold(class_id, masks.shape[0])
			mask = masks.sum(axis=0) >= vote_threshold
			bev_reference = bev_classes[0]
		else:
			roi_classes = bev_classes[row_start:row_end, col_start:col_end]
			mask = roi_classes == class_id
			bev_reference = bev_classes

		count = int(mask.sum())
		if count == 0:
			return {
				'count': 0,
				'closest_forward_m': None,
				'mean_forward_m': None,
				'mean_lateral_m': None,
			}

		rows, cols = np.nonzero(mask)
		rows = rows.astype(np.float32) + float(row_start)
		cols = cols.astype(np.float32) + float(col_start)
		center_col = bev_reference.shape[1] / 2.0
		center_row = bev_reference.shape[0] / 2.0
		ppm = self.semantic_bev_pixels_per_meter
		x_forward = (cols - center_col) / ppm
		y_lateral = (rows - center_row) / ppm
		return {
			'count': count,
			'closest_forward_m': float(np.min(x_forward)),
			'mean_forward_m': float(np.mean(x_forward)),
			'mean_lateral_m': float(np.mean(y_lateral)),
		}

	def _get_semantic_traffic_light_debug(self, bev_classes):
		red_stats = self._semantic_class_stats(
			bev_classes, SEMANTIC_LIGHT_RED_CLASS, self.semantic_tl_roi
		)
		yellow_stats = self._semantic_class_stats(
			bev_classes, SEMANTIC_LIGHT_YELLOW_CLASS, self.semantic_tl_roi
		)
		green_stats = self._semantic_class_stats(
			bev_classes, SEMANTIC_LIGHT_GREEN_CLASS, self.semantic_tl_roi
		)

		state = 'none'
		block_force_move = False
		closest_forward_m = None
		if red_stats['count'] >= self.semantic_tl_min_pixels:
			state = 'red'
			block_force_move = True
			closest_forward_m = red_stats['closest_forward_m']
		elif yellow_stats['count'] >= self.semantic_tl_min_pixels:
			state = 'yellow'
			block_force_move = True
			closest_forward_m = yellow_stats['closest_forward_m']
		elif green_stats['count'] >= self.semantic_tl_min_pixels:
			state = 'green'
			closest_forward_m = green_stats['closest_forward_m']

		apply_stop = (
			state in ('red', 'yellow')
			and closest_forward_m is not None
			and closest_forward_m <= self.semantic_tl_brake_distance_m
		)

		return {
			'state': state,
			'block_force_move': block_force_move,
			'apply_stop': apply_stop,
			'closest_forward_m': closest_forward_m,
			'red_pixels': red_stats['count'],
			'yellow_pixels': yellow_stats['count'],
			'green_pixels': green_stats['count'],
			'red_closest_forward_m': red_stats['closest_forward_m'],
			'yellow_closest_forward_m': yellow_stats['closest_forward_m'],
			'green_closest_forward_m': green_stats['closest_forward_m'],
		}

	def _update_semantic_stop_sign_debug(self, bev_classes, ego_speed):
		stop_stats = self._semantic_class_stats(
			bev_classes, SEMANTIC_STOP_SIGN_CLASS, self.semantic_stop_roi
		)
		stop_detected = stop_stats['count'] >= self.semantic_stop_min_pixels
		arm_speed_ok = ego_speed > self.semantic_stop_arm_speed_threshold

		if self.semantic_stop_disabled:
			self.semantic_stop_state = 'NONE'
			self.semantic_stop_stopped_frames = 0
			self.semantic_stop_missing_frames = 0
			return {
				'state': self.semantic_stop_state,
				'hold_force_move': False,
				'apply_stop': False,
				'pixels': stop_stats['count'],
				'closest_forward_m': stop_stats['closest_forward_m'],
				'mean_lateral_m': stop_stats['mean_lateral_m'],
				'stopped_frames': int(self.semantic_stop_stopped_frames),
				'disabled': True,
				'apply_events': int(self.semantic_stop_apply_events),
				'max_apply_events': int(self.semantic_stop_max_apply_events),
				'arm_speed_ok': bool(arm_speed_ok),
			}

		if stop_detected:
			self.semantic_stop_missing_frames = 0
		else:
			self.semantic_stop_missing_frames += 1

		if self.semantic_stop_state == 'CLEARED':
			if self.semantic_stop_missing_frames >= self.semantic_stop_reset_missing_frames:
				self.semantic_stop_state = 'NONE'
				self.semantic_stop_stopped_frames = 0
		elif self.semantic_stop_state == 'NONE':
			# Only arm stop-sign handling when we are actually arriving with non-trivial speed.
			if stop_detected and arm_speed_ok:
				self.semantic_stop_state = 'APPROACHING'
				self.semantic_stop_stopped_frames = 0
		elif self.semantic_stop_state == 'APPROACHING':
			if not stop_detected and self.semantic_stop_missing_frames >= self.semantic_stop_reset_missing_frames:
				self.semantic_stop_state = 'NONE'
				self.semantic_stop_stopped_frames = 0
			elif ego_speed < 0.1:
				self.semantic_stop_stopped_frames += 1
				if self.semantic_stop_stopped_frames >= self.semantic_stop_min_stop_frames:
					self.semantic_stop_state = 'STOPPED'
			else:
				self.semantic_stop_stopped_frames = 0
		elif self.semantic_stop_state == 'STOPPED':
			self.semantic_stop_state = 'CLEARED'
			self.semantic_stop_stopped_frames = 0

		closest_forward_m = stop_stats['closest_forward_m']
		apply_stop = (
			self.semantic_stop_state == 'APPROACHING'
			and stop_detected
			and closest_forward_m is not None
			and closest_forward_m <= self.semantic_stop_brake_distance_m
		)
		hold_force_move = self.semantic_stop_state in ('APPROACHING', 'STOPPED')

		return {
			'state': self.semantic_stop_state,
			'hold_force_move': hold_force_move,
			'apply_stop': apply_stop,
			'pixels': stop_stats['count'],
			'closest_forward_m': closest_forward_m,
			'mean_lateral_m': stop_stats['mean_lateral_m'],
			'stopped_frames': int(self.semantic_stop_stopped_frames),
			'disabled': False,
			'apply_events': int(self.semantic_stop_apply_events),
			'max_apply_events': int(self.semantic_stop_max_apply_events),
			'arm_speed_ok': bool(arm_speed_ok),
		}

	def _apply_semantic_hazard_postprocess(
		self,
		ego_speed,
		current_heading,
		desired_speed_capped,
		throttle,
		brake,
		bev_classes,
		external_force_move_block_reason=None,
	):
		if bev_classes is None:
			return throttle, brake, {
				'traffic_light_state': 'none',
				'traffic_light_block_force_move': False,
				'traffic_light_green_hold_active': False,
				'traffic_light_green_hold_frames_remaining': int(self.semantic_tl_green_hold_frames),
				'traffic_light_red_pixels': 0,
				'traffic_light_yellow_pixels': 0,
				'traffic_light_green_pixels': 0,
				'stop_sign_state': self.semantic_stop_state,
				'stop_sign_hold_force_move': False,
				'stop_sign_apply_stop': False,
				'stop_sign_pixels': 0,
				'stop_sign_closest_forward_m': None,
				'stop_sign_disabled': bool(self.semantic_stop_disabled),
				'stop_sign_apply_events': int(self.semantic_stop_apply_events),
				'stop_sign_max_apply_events': int(self.semantic_stop_max_apply_events),
				'stop_sign_arm_speed_ok': False,
				'stuck_helper_active': bool(self.stuck_helper_active),
				'stuck_helper_mode': self.stuck_helper_mode,
				'stuck_helper_recovery_mode': STUCK_HELPER_STARTUP_RECOVERY_MODE,
				'stuck_helper_frames_remaining': int(self.stuck_helper),
				'stuck_helper_heading_delta_deg': float(self.stuck_helper_heading_delta_deg),
				'stuck_helper_release_heading_deg': float(STUCK_HELPER_RELEASE_HEADING_DEG),
				'stuck_helper_in_startup_zone': bool(self.stuck_helper_in_startup_zone),
				'stuck_helper_distance_from_start_m': float(self.stuck_helper_distance_from_start_m),
				'stuck_helper_startup_distance_m': float(self.stuck_helper_startup_distance_m),
				'stuck_helper_startup_threshold': int(self.stuck_helper_startup_threshold),
				'stuck_helper_poststart_threshold': int(self.stuck_helper_poststart_threshold),
				'stuck_helper_startup_reference_xy': (
					self.stuck_helper_startup_reference_xy.tolist()
					if isinstance(self.stuck_helper_startup_reference_xy, np.ndarray)
					else self.stuck_helper_startup_reference_xy
				),
				'force_move_blocked_reason': None,
				'planner_wants_stop': False,
			}

		traffic_light_debug = self._get_semantic_traffic_light_debug(bev_classes)
		stop_sign_debug = self._update_semantic_stop_sign_debug(bev_classes, ego_speed)
		stop_sign_apply_rising = bool(stop_sign_debug['apply_stop']) and not self.semantic_stop_prev_apply_stop
		self.semantic_stop_prev_apply_stop = bool(stop_sign_debug['apply_stop'])
		traffic_light_state = traffic_light_debug['state']
		if (
			traffic_light_state == 'green'
			and self.semantic_tl_prev_state in ('red', 'yellow')
			and ego_speed < self.semantic_tl_green_release_speed_threshold
		):
			self.semantic_tl_green_hold_frames = self.semantic_tl_green_release_delay_frames

		green_hold_active = False
		if traffic_light_state == 'green':
			if self.semantic_tl_green_hold_frames > 0:
				green_hold_active = True
				self.semantic_tl_green_hold_frames -= 1
		else:
			self.semantic_tl_green_hold_frames = 0

		self.semantic_tl_prev_state = traffic_light_state
		planner_wants_stop = (
			desired_speed_capped is not None
			and desired_speed_capped < self.semantic_planner_stop_speed_threshold
			and ego_speed < 0.1
		)

		force_move_blocked_reason = external_force_move_block_reason
		if force_move_blocked_reason is not None:
			self.force_move = 0
			self._reset_stuck_helper_state()
		elif green_hold_active:
			force_move_blocked_reason = 'traffic_light_green_hold'
		elif traffic_light_debug['block_force_move']:
			force_move_blocked_reason = f"traffic_light_{traffic_light_debug['state']}"
		# Only block stuck-helper during the actual stop-apply window.
		# A long APPROACHING phase should not keep resetting the helper.
		elif stop_sign_debug['apply_stop']:
			force_move_blocked_reason = 'stop_sign'

		if force_move_blocked_reason is None:
			self._update_stuck_helper_release(current_heading)

		current_stuck_helper_threshold = (
			self.stuck_helper_startup_threshold
			if self.stuck_helper_in_startup_zone
			else self.stuck_helper_poststart_threshold
		)
		self.stuck_helper_threshold = int(current_stuck_helper_threshold)

		if ego_speed < 0.1:
			if force_move_blocked_reason is None:
				self.stuck_detector += 1
				if (
					self.stuck_helper_in_startup_zone
					and self.stuck_detector > current_stuck_helper_threshold
				):
					self._activate_stuck_helper(current_heading, mode='startup')
			else:
				self.stuck_detector = 0
				self.force_move = 0
				self._reset_stuck_helper_state()
		elif ego_speed >= 1.0 and not self.stuck_helper_active:
			self.stuck_detector = 0
			self._reset_stuck_helper_state()

		# Away from the startup point, keep the old stuck recovery path: creep only,
		# no target-point rewrite.
		if (
			force_move_blocked_reason is None
			and (not self.stuck_helper_in_startup_zone)
			and self.stuck_detector > self.stuck_threshold
		):
			self.force_move = self.creep_duration

		if self.force_move > 0:
			throttle = max(self.creep_throttle, throttle)
			brake = False
			self.force_move -= 1

		if green_hold_active:
			throttle = 0.0
			brake = 1.0
			self.stuck_detector = 0
			self.force_move = 0
			self._reset_stuck_helper_state()

		if traffic_light_debug['apply_stop']:
			throttle = 0.0
			brake = 1.0
			self.stuck_detector = 0
			self.force_move = 0
			self._reset_stuck_helper_state()
			if force_move_blocked_reason is None:
				force_move_blocked_reason = f"traffic_light_{traffic_light_state}"

		if stop_sign_debug['apply_stop']:
			throttle = 0.0
			brake = 1.0
			self.stuck_detector = 0
			self.force_move = 0
			self._reset_stuck_helper_state()
			if force_move_blocked_reason is None:
				force_move_blocked_reason = 'stop_sign'
			if stop_sign_apply_rising:
				self.semantic_stop_apply_events += 1
				if (
					self.semantic_stop_max_apply_events > 0
					and self.semantic_stop_apply_events >= self.semantic_stop_max_apply_events
				):
					self.semantic_stop_disabled = True
					self.semantic_stop_state = 'NONE'
					self.semantic_stop_stopped_frames = 0
					self.semantic_stop_missing_frames = 0

		debug = {
			'traffic_light_state': traffic_light_debug['state'],
			'traffic_light_block_force_move': traffic_light_debug['block_force_move'],
			'traffic_light_apply_stop': bool(traffic_light_debug['apply_stop']),
			'traffic_light_closest_forward_m': traffic_light_debug['closest_forward_m'],
			'traffic_light_green_hold_active': bool(green_hold_active),
			'traffic_light_green_hold_frames_remaining': int(self.semantic_tl_green_hold_frames),
			'traffic_light_red_pixels': int(traffic_light_debug['red_pixels']),
			'traffic_light_yellow_pixels': int(traffic_light_debug['yellow_pixels']),
			'traffic_light_green_pixels': int(traffic_light_debug['green_pixels']),
			'traffic_light_red_closest_forward_m': traffic_light_debug['red_closest_forward_m'],
			'traffic_light_yellow_closest_forward_m': traffic_light_debug['yellow_closest_forward_m'],
			'traffic_light_green_closest_forward_m': traffic_light_debug['green_closest_forward_m'],
			'stop_sign_state': stop_sign_debug['state'],
			'stop_sign_hold_force_move': stop_sign_debug['hold_force_move'],
			'stop_sign_apply_stop': stop_sign_debug['apply_stop'],
			'stop_sign_pixels': int(stop_sign_debug['pixels']),
			'stop_sign_closest_forward_m': stop_sign_debug['closest_forward_m'],
			'stop_sign_mean_lateral_m': stop_sign_debug['mean_lateral_m'],
			'stop_sign_stopped_frames': int(stop_sign_debug['stopped_frames']),
			'stop_sign_disabled': bool(self.semantic_stop_disabled),
			'stop_sign_apply_events': int(self.semantic_stop_apply_events),
			'stop_sign_max_apply_events': int(self.semantic_stop_max_apply_events),
			'stop_sign_arm_speed_ok': bool(stop_sign_debug.get('arm_speed_ok', False)),
			'stuck_helper_active': bool(self.stuck_helper_active),
			'stuck_helper_mode': self.stuck_helper_mode,
			'stuck_helper_recovery_mode': STUCK_HELPER_STARTUP_RECOVERY_MODE,
			'stuck_helper_frames_remaining': int(self.stuck_helper),
			'stuck_helper_heading_delta_deg': float(self.stuck_helper_heading_delta_deg),
			'stuck_helper_release_heading_deg': float(STUCK_HELPER_RELEASE_HEADING_DEG),
			'stuck_helper_in_startup_zone': bool(self.stuck_helper_in_startup_zone),
			'stuck_helper_distance_from_start_m': float(self.stuck_helper_distance_from_start_m),
			'stuck_helper_startup_distance_m': float(self.stuck_helper_startup_distance_m),
			'stuck_helper_startup_threshold': int(self.stuck_helper_startup_threshold),
			'stuck_helper_poststart_threshold': int(self.stuck_helper_poststart_threshold),
			'stuck_helper_startup_reference_xy': (
				self.stuck_helper_startup_reference_xy.tolist()
				if isinstance(self.stuck_helper_startup_reference_xy, np.ndarray)
				else self.stuck_helper_startup_reference_xy
			),
			'force_move_blocked_reason': force_move_blocked_reason,
			'planner_wants_stop': bool(planner_wants_stop),
		}
		self.last_semantic_debug = debug
		return throttle, brake, debug

	def _get_terminal_route_speed_cap(self, tick_data):
		waypoint_route_ego = tick_data.get('waypoint_route_ego')
		target_point_ego = tick_data.get('target_point')

		debug = {
			'active': False,
			'reason': None,
			'speed_cap_ms': None,
			'remaining_route_points': 0,
			'target_distance_m': None,
			'target_forward_m': None,
		}

		if waypoint_route_ego is None or target_point_ego is None:
			return debug

		waypoint_route_ego = np.asarray(waypoint_route_ego, dtype=np.float32)
		target_point_ego = np.asarray(target_point_ego[:2], dtype=np.float32)

		if waypoint_route_ego.ndim != 2 or waypoint_route_ego.shape[0] == 0:
			return debug

		remaining_points = int(waypoint_route_ego.shape[0])
		target_distance = float(np.linalg.norm(target_point_ego))
		target_forward = float(target_point_ego[0])

		debug.update({
			'remaining_route_points': remaining_points,
			'target_distance_m': target_distance,
			'target_forward_m': target_forward,
		})

		if remaining_points > TERMINAL_ROUTE_ACTIVE_POINTS_MAX:
			return debug

		if target_forward < 0.0:
			debug.update({
				'active': True,
				'reason': 'terminal_target_behind',
				'speed_cap_ms': float(TERMINAL_ROUTE_BEHIND_SPEED_CAP_MS),
			})
			return debug

		if target_distance <= TERMINAL_ROUTE_NEAR_DISTANCE_M:
			debug.update({
				'active': True,
				'reason': 'terminal_target_near',
				'speed_cap_ms': float(TERMINAL_ROUTE_SPEED_CAP_MS),
			})

		return debug

	def _get_front_route_risk_speed_cap(self, tick_data, ego_speed):
		debug = {
			'active': False,
			'reason': None,
			'speed_cap_ms': None,
			'raw_score': None,
			'sigmoid': None,
			'ema': None,
			'recent_max_sigmoid': None,
			'geometry_active': False,
			'gate_active': False,
			'raw_override_active': False,
			'geometry_max_angle_deg': 0.0,
			'geometry_lateral_mag_m': 0.0,
			'speed_bias_ratio': 0.0,
			'effective_med_sigmoid_threshold': float(FRONT_ROUTE_RISK_MED_SIGMOID),
			'effective_high_sigmoid_threshold': float(FRONT_ROUTE_RISK_HIGH_SIGMOID),
			'effective_high_ema_threshold': float(FRONT_ROUTE_RISK_HIGH_EMA),
			'hold_frames_remaining': int(self.front_route_risk_cap_hold_frames),
		}

		if not self.use_front_route_risk_energy or not FRONT_ROUTE_RISK_SPEED_CAP_ENABLE:
			self.front_route_risk_cap_hold_frames = 0
			self.front_route_risk_cap_value_ms = None
			return debug

		raw_score = self.last_energy_debug.get('front_route_risk_score')
		sigmoid = self.last_energy_debug.get('front_route_risk_sigmoid')
		ema = self.last_energy_debug.get('front_route_risk_ema')
		if raw_score is not None:
			raw_score = float(raw_score)
		if sigmoid is not None:
			sigmoid = float(sigmoid)
			self.front_route_risk_sigmoid_history.append(sigmoid)
		if ema is not None:
			ema = float(ema)

		recent_max_sigmoid = None
		if len(self.front_route_risk_sigmoid_history) > 0:
			recent_max_sigmoid = float(max(self.front_route_risk_sigmoid_history))

		target_angle = abs(float(tick_data.get('target_angle_deg', 0.0)))
		next_target_angle = abs(float(tick_data.get('next_target_angle_deg', 0.0)))
		target_point = tick_data.get('target_point')
		next_target_point = tick_data.get('next_target_point')
		target_lat = abs(float(target_point[1])) if target_point is not None else 0.0
		next_target_lat = abs(float(next_target_point[1])) if next_target_point is not None else 0.0
		max_angle = max(target_angle, next_target_angle)
		lateral_mag = max(target_lat, next_target_lat)
		geometry_active = (
			max_angle >= FRONT_ROUTE_RISK_GEOM_ANGLE_DEG
			or lateral_mag >= FRONT_ROUTE_RISK_GEOM_LATERAL_M
		)

		debug.update({
			'raw_score': raw_score,
			'sigmoid': sigmoid,
			'ema': ema,
			'recent_max_sigmoid': recent_max_sigmoid,
			'geometry_active': bool(geometry_active),
			'geometry_max_angle_deg': float(max_angle),
			'geometry_lateral_mag_m': float(lateral_mag),
		})

		speed_bias_ratio = 0.0
		if FRONT_ROUTE_RISK_EARLY_SPEED_FULL_MS > FRONT_ROUTE_RISK_EARLY_SPEED_START_MS:
			speed_bias_ratio = float(np.clip(
				(ego_speed - FRONT_ROUTE_RISK_EARLY_SPEED_START_MS) /
				(FRONT_ROUTE_RISK_EARLY_SPEED_FULL_MS - FRONT_ROUTE_RISK_EARLY_SPEED_START_MS),
				0.0,
				1.0,
			))
		effective_med_sigmoid = max(
			FRONT_ROUTE_RISK_MED_SIGMOID_MIN,
			FRONT_ROUTE_RISK_MED_SIGMOID - FRONT_ROUTE_RISK_MED_SIGMOID_SPEED_DELTA * speed_bias_ratio,
		)
		effective_high_sigmoid = max(
			FRONT_ROUTE_RISK_HIGH_SIGMOID_MIN,
			FRONT_ROUTE_RISK_HIGH_SIGMOID - FRONT_ROUTE_RISK_HIGH_SIGMOID_SPEED_DELTA * speed_bias_ratio,
		)
		effective_high_ema = max(
			FRONT_ROUTE_RISK_HIGH_EMA_MIN,
			FRONT_ROUTE_RISK_HIGH_EMA - FRONT_ROUTE_RISK_HIGH_EMA_SPEED_DELTA * speed_bias_ratio,
		)
		debug.update({
			'speed_bias_ratio': float(speed_bias_ratio),
			'effective_med_sigmoid_threshold': float(effective_med_sigmoid),
			'effective_high_sigmoid_threshold': float(effective_high_sigmoid),
			'effective_high_ema_threshold': float(effective_high_ema),
		})

		gate_active = (
			ego_speed >= FRONT_ROUTE_RISK_MIN_SPEED_MS
			and recent_max_sigmoid is not None
		)
		debug.update({
			'gate_active': bool(gate_active),
			'raw_override_active': bool(gate_active),
		})

		candidate_cap_ms = None
		reason = None
		if gate_active:
			if recent_max_sigmoid >= effective_high_sigmoid or (
				ema is not None and ema >= effective_high_ema
			):
				candidate_cap_ms = float(FRONT_ROUTE_RISK_HIGH_CAP_MS)
				reason = 'front_route_risk_high'
			elif recent_max_sigmoid >= effective_med_sigmoid:
				candidate_cap_ms = float(FRONT_ROUTE_RISK_MED_CAP_MS)
				reason = 'front_route_risk_medium'

		if candidate_cap_ms is not None:
			self.front_route_risk_cap_value_ms = candidate_cap_ms
			self.front_route_risk_cap_hold_frames = FRONT_ROUTE_RISK_HOLD_FRAMES
			debug.update({
				'active': True,
				'reason': reason,
				'speed_cap_ms': candidate_cap_ms,
				'hold_frames_remaining': int(self.front_route_risk_cap_hold_frames),
			})
			return debug

		if (
			gate_active
			and self.front_route_risk_cap_hold_frames > 0
			and self.front_route_risk_cap_value_ms is not None
		):
			self.front_route_risk_cap_hold_frames -= 1
			debug.update({
				'active': True,
				'reason': 'front_route_risk_hold',
				'speed_cap_ms': float(self.front_route_risk_cap_value_ms),
				'hold_frames_remaining': int(self.front_route_risk_cap_hold_frames),
			})
			return debug

		self.front_route_risk_cap_hold_frames = 0
		self.front_route_risk_cap_value_ms = None
		return debug

	def get_default_config_path(self):
		return "/media/z/data/mzq/others/MoT-DP/config/pdm_local_route_b.yaml"

	def get_checkpoint_filename(self):
		return "dit_policy_best.pt"

	def resolve_checkpoint_path(self):
		checkpoint_path_override = os.environ.get('CHECKPOINT_PATH_OVERRIDE', '').strip()
		if checkpoint_path_override:
			return checkpoint_path_override

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

	def _get_observed_borrow_time_s(self, current_time_s):
		if self.borrow_latched and self.borrow_candidate_start_time_s is not None:
			return max(float(current_time_s) - float(self.borrow_candidate_start_time_s), 0.0)
		return 0.0

	def _update_borrow_semantic_state(self, dp_pred_traj, current_time_s):
		relation_probs = dp_pred_traj.get('lane_dir_relation_probs')
		if relation_probs is not None:
			relation_probs = np.asarray(relation_probs, dtype=np.float32).reshape(-1)
			if relation_probs.size == 2:
				relation_probs = np.clip(relation_probs, 1e-6, None)
				relation_probs = relation_probs / np.sum(relation_probs)
				prev_relation = np.asarray(self.prev_lane_dir_relation_probs, dtype=np.float32).reshape(-1)
				if prev_relation.size != 2:
					prev_relation = np.array([0.5, 0.5], dtype=np.float32)
				same_smoothed = 0.7 * relation_probs[0] + 0.3 * prev_relation[0]
				opposite_smoothed = 0.45 * relation_probs[1] + 0.55 * prev_relation[1]
				relation_smoothed = np.array([same_smoothed, opposite_smoothed], dtype=np.float32)
				relation_smoothed /= max(float(np.sum(relation_smoothed)), 1e-6)
				self.prev_lane_dir_relation_probs = relation_smoothed

		relation_smoothed = np.asarray(self.prev_lane_dir_relation_probs, dtype=np.float32).reshape(-1)
		if relation_smoothed.size != 2:
			relation_smoothed = np.array([0.5, 0.5], dtype=np.float32)
			self.prev_lane_dir_relation_probs = relation_smoothed

		borrow_prob = None
		traj_window_probs = dp_pred_traj.get('traj_window_condition_probs')
		if traj_window_probs is not None:
			traj_window_probs = np.asarray(traj_window_probs, dtype=np.float32).reshape(-1)
			if traj_window_probs.size >= 4:
				borrow_prob = float(traj_window_probs[3])
		if borrow_prob is None:
			speed_borrow_prob = dp_pred_traj.get('speed_energy_borrow_active_prob')
			if speed_borrow_prob is not None:
				speed_borrow_prob = np.asarray(speed_borrow_prob, dtype=np.float32).reshape(-1)
				if speed_borrow_prob.size > 0:
					borrow_prob = float(speed_borrow_prob[0])
		if borrow_prob is None:
			borrow_prob = 0.0

		opposite_prob = float(relation_smoothed[1])
		current_time_s = float(current_time_s)
		if self.borrow_candidate_start_time_s is None and (borrow_prob > 0.25 or opposite_prob > 0.65):
			self.borrow_candidate_start_time_s = current_time_s

		if borrow_prob > 0.5:
			self.borrow_latch_counter += 1
		else:
			self.borrow_latch_counter = 0

		if (not self.borrow_latched) and self.borrow_latch_counter >= 2:
			self.borrow_latched = True
			self.borrow_release_counter = 0
			if self.borrow_candidate_start_time_s is None:
				self.borrow_candidate_start_time_s = current_time_s

		if self.borrow_latched:
			if borrow_prob < 0.2 and opposite_prob < 0.55:
				self.borrow_release_counter += 1
			else:
				self.borrow_release_counter = 0
			if self.borrow_release_counter >= 4:
				self.borrow_latched = False
				self.borrow_release_counter = 0
				self.borrow_latch_counter = 0
				self.borrow_candidate_start_time_s = None
		elif borrow_prob < 0.1 and opposite_prob < 0.45:
			self.borrow_candidate_start_time_s = None

		self.borrow_observed_time_s = self._get_observed_borrow_time_s(current_time_s)

	def _compute_front_route_risk_debug(
		self,
		dp_pred_traj,
		transfuser_bev_feature,
		transfuser_bev_feature_upsample,
		ego_status_stacked,
		transfuser_lidar_bev_detail,
	):
		if not getattr(self, 'use_front_route_risk_energy', False):
			return None
		if not hasattr(self, 'net') or not hasattr(self.net, 'model'):
			return None
		if not hasattr(self.net.model, 'forward_energy_eval'):
			return None

		try:
			device = next(self.net.parameters()).device
			model_dtype = next(self.net.parameters()).dtype

			traj_abs = dp_pred_traj.get('action')
			route_pred = dp_pred_traj.get('route_pred')
			if traj_abs is None or route_pred is None:
				return None

			if isinstance(traj_abs, np.ndarray):
				traj_abs = torch.from_numpy(traj_abs)
			if isinstance(route_pred, np.ndarray):
				route_pred = torch.from_numpy(route_pred)

			traj_abs = traj_abs.to(device=device, dtype=model_dtype)
			route_pred = route_pred.to(device=device, dtype=model_dtype)
			if traj_abs.dim() == 2:
				traj_abs = traj_abs.unsqueeze(0)
			if route_pred.dim() == 2:
				route_pred = route_pred.unsqueeze(0)

			x_t_abs = traj_abs.unsqueeze(1)  # (B, 1, T, 2)
			x_t = self.net.abs_to_norm(traj_abs).unsqueeze(1)

			bev_proj = self.net.model.decoder.compute_bev_proj(
				transfuser_bev_feature.to(device=device, dtype=model_dtype)
			)

			with torch.no_grad():
				energy_scores, _ = self.net.model.forward_energy_eval(
					x_t=x_t,
					x_t_abs=x_t_abs,
					transfuser_bev_feature=transfuser_bev_feature.to(device=device, dtype=model_dtype),
					transfuser_bev_feature_upsample=transfuser_bev_feature_upsample.to(device=device, dtype=model_dtype),
					ego_status=ego_status_stacked.to(device=device, dtype=model_dtype),
					traj_for_energy=x_t_abs,
					bev_proj_cached=bev_proj,
					route_points=route_pred,
					transfuser_lidar_bev=transfuser_lidar_bev_detail.to(device=device, dtype=model_dtype),
				)

			if energy_scores is None or 'front' not in energy_scores:
				return None
			front_score = energy_scores['front']
			front_array = front_score.detach().float().reshape(-1)
			if front_array.numel() == 0:
				return None
			return float(front_array[0].item())
		except Exception as exc:
			print(f"[front_route_risk_debug] failed: {exc}")
			return None

	def _get_junction_window_soft_speed_cap(self):
		debug = {
			'active': False,
			'reason': None,
			'speed_cap_ms': None,
			'junction_prob': None,
			'streak_frames': int(self.junction_window_soft_cap_streak),
			'source': None,
		}

		if not JUNCTION_WINDOW_SOFT_SPEED_CAP_ENABLE:
			self.junction_window_soft_cap_streak = 0
			return debug

		window_probs = self.last_branch_condition_debug.get('traj_window_condition_probs')
		source = 'traj_window_condition_probs'
		if window_probs is None:
			window_probs = self.last_branch_condition_debug.get('speed_energy_window_probs')
			source = 'speed_energy_window_probs'
		if window_probs is None:
			self.junction_window_soft_cap_streak = 0
			return debug

		window_probs = np.asarray(window_probs, dtype=np.float32).reshape(-1)
		if window_probs.size < 3:
			self.junction_window_soft_cap_streak = 0
			return debug

		junction_prob = float(window_probs[2])
		if junction_prob > JUNCTION_WINDOW_SOFT_SPEED_CAP_CONSEC_THRESHOLD:
			self.junction_window_soft_cap_streak += 1
		else:
			self.junction_window_soft_cap_streak = 0

		debug.update({
			'junction_prob': junction_prob,
			'streak_frames': int(self.junction_window_soft_cap_streak),
			'source': source,
		})

		if junction_prob > JUNCTION_WINDOW_SOFT_SPEED_CAP_SINGLE_THRESHOLD:
			debug.update({
				'active': True,
				'reason': 'junction_window_single_high',
				'speed_cap_ms': float(JUNCTION_WINDOW_SOFT_SPEED_CAP_MS),
			})
			return debug

		if self.junction_window_soft_cap_streak >= JUNCTION_WINDOW_SOFT_SPEED_CAP_CONSEC_FRAMES:
			debug.update({
				'active': True,
				'reason': 'junction_window_streak',
				'speed_cap_ms': float(JUNCTION_WINDOW_SOFT_SPEED_CAP_MS),
			})

		return debug

	def _get_stage1_energy_speed_cap(self, tick_data, ego_speed):
		debug = {
			'active': False,
			'control_mode': STAGE1_ENERGY_SPEED_CONTROL_MODE,
			'reason': None,
			'speed_cap_ms': None,
			'speed_adjust_ms': None,
			'current_score': None,
			'target_score': None,
			'local_peak_score': None,
			'query_center_ms': None,
			'current_index': None,
			'target_index': None,
			'target_speed_ms': None,
			'target_source': None,
			'lookahead_bins': int(STAGE1_ENERGY_CAP_LOOKAHEAD_BINS),
			'safe_threshold': None,
			'gradient_dedv': None,
			'gradient_gate_dedv': None,
			'gradient_dominant_component': None,
			'gradient_raw_adjust_ms': None,
			'gradient_score_gate': False,
			'gradient_slope_gate': False,
			'gradient_chase_dedv': None,
			'gradient_merge_dedv': None,
			'gradient_cross_dedv': None,
			'gradient_pedestrian_dedv': None,
			'gradient_chase_score': None,
			'gradient_merge_score': None,
			'gradient_cross_score': None,
			'gradient_pedestrian_score': None,
			'gradient_chase_active': False,
			'gradient_merge_active': False,
			'gradient_cross_active': False,
			'gradient_pedestrian_active': False,
			'hold_frames_remaining': int(self.stage1_energy_cap_hold_frames),
		}

		if not STAGE1_ENERGY_SPEED_CAP_ENABLE:
			self.stage1_energy_cap_hold_frames = 0
			self.stage1_energy_cap_value_ms = None
			return debug

		samples = self.last_energy_debug.get('speed_energy_samples')
		total_curve = self.last_energy_debug.get('speed_energy_total')
		query_center = self.last_energy_debug.get('speed_energy_query_center')
		if samples is None or total_curve is None:
			self.stage1_energy_cap_hold_frames = 0
			self.stage1_energy_cap_value_ms = None
			return debug

		samples = np.asarray(samples, dtype=np.float32).reshape(-1)
		total_curve = np.asarray(total_curve, dtype=np.float32).reshape(-1)
		if samples.size == 0 or total_curve.size == 0 or samples.size != total_curve.size:
			self.stage1_energy_cap_hold_frames = 0
			self.stage1_energy_cap_value_ms = None
			return debug

		center_speed = float(query_center) if query_center is not None else float(ego_speed)
		center_speed = max(0.0, center_speed)
		debug['query_center_ms'] = center_speed
		if center_speed < STAGE1_ENERGY_CAP_MIN_SPEED_MS:
			self.stage1_energy_cap_hold_frames = 0
			self.stage1_energy_cap_value_ms = None
			return debug

		current_idx = int(np.argmin(np.abs(samples - center_speed)))
		peak_end = min(total_curve.size, current_idx + 1 + STAGE1_ENERGY_CAP_LOOKAHEAD_BINS)
		current_score = float(total_curve[current_idx])
		local_peak_score = float(np.max(total_curve[current_idx:peak_end]))
		debug.update({
			'current_index': current_idx,
			'current_score': current_score,
			'local_peak_score': local_peak_score,
		})

		target_speed = center_speed
		target_score = current_score
		target_idx = current_idx
		target_source = 'query_center'
		ref_speeds = self.last_energy_debug.get('speed_energy_ref_speeds')
		ref_totals = self.last_energy_debug.get('speed_energy_ref_total')
		if ref_speeds is not None:
			ref_speeds = np.asarray(ref_speeds, dtype=np.float32).reshape(-1)
			ref_totals = np.asarray(ref_totals if ref_totals is not None else [], dtype=np.float32).reshape(-1)
			speed_source_to_idx = {
				'speed_head': 0,
				'traj': 1,
				'traj_05s': 2,
				'traj05': 2,
			}
			ref_idx = speed_source_to_idx.get(SPEED_SOURCE)
			if ref_idx is not None and ref_idx < ref_speeds.size and np.isfinite(ref_speeds[ref_idx]):
				target_speed = max(0.0, float(ref_speeds[ref_idx]))
				target_idx = int(np.argmin(np.abs(samples - target_speed)))
				if ref_idx < ref_totals.size and np.isfinite(ref_totals[ref_idx]):
					target_score = float(ref_totals[ref_idx])
				else:
					target_score = float(total_curve[target_idx])
				target_source = SPEED_SOURCE
		debug.update({
			'target_index': int(target_idx),
			'target_speed_ms': float(target_speed),
			'target_score': float(target_score),
			'target_source': target_source,
		})

		path_lo = min(current_idx, target_idx)
		path_hi = max(current_idx, target_idx) + 1
		path_peak_score = float(np.max(total_curve[path_lo:path_hi]))
		debug['local_peak_score'] = path_peak_score

		speed_delta = float(target_speed - center_speed)
		moving_toward_danger = (
			target_score >= current_score + STAGE1_ENERGY_CAP_TREND_MARGIN
			or path_peak_score >= current_score + STAGE1_ENERGY_CAP_TREND_MARGIN
		)

		if STAGE1_ENERGY_SPEED_CONTROL_MODE == 'gradient':
			self.stage1_energy_cap_hold_frames = 0
			self.stage1_energy_cap_value_ms = None

			component_specs = {
				'chase': (
					STAGE1_ENERGY_GRADIENT_CHASE_WEIGHT,
					STAGE1_ENERGY_GRADIENT_CHASE_MIN_SCORE,
				),
				'merge': (
					STAGE1_ENERGY_GRADIENT_MEET_WEIGHT,
					STAGE1_ENERGY_GRADIENT_MEET_MIN_SCORE,
				),
				'cross': (
					STAGE1_ENERGY_GRADIENT_MEET_WEIGHT,
					STAGE1_ENERGY_GRADIENT_MEET_MIN_SCORE,
				),
				'pedestrian': (
					STAGE1_ENERGY_GRADIENT_PEDESTRIAN_WEIGHT,
					STAGE1_ENERGY_GRADIENT_PEDESTRIAN_MIN_SCORE,
				),
			}
			component_grad_sum = 0.0
			component_weight_sum = 0.0
			component_active = False
			dominant_component = None
			dominant_component_value = 0.0
			dominant_component_grad = None
			for energy_key, (component_weight, component_min_score) in component_specs.items():
				curve_values = self.last_energy_debug.get(f'speed_energy_{energy_key}')
				if curve_values is None:
					continue
				component_curve = np.asarray(curve_values, dtype=np.float32).reshape(-1)
				if component_curve.size != total_curve.size:
					continue
				component_score = float(component_curve[current_idx])
				component_grad_curve = np.gradient(component_curve.astype(np.float32), samples.astype(np.float32))
				component_grad = float(component_grad_curve[current_idx]) if component_grad_curve.size > current_idx else 0.0
				component_is_active = (
					component_weight > 0.0
					and component_score >= component_min_score
					and abs(component_grad) >= STAGE1_ENERGY_GRADIENT_SLOPE_EPS
				)
				debug.update({
					f'gradient_{energy_key}_dedv': component_grad,
					f'gradient_{energy_key}_score': component_score,
					f'gradient_{energy_key}_active': bool(component_is_active),
				})
				if component_is_active:
					component_active = True
					component_contribution = component_weight * component_grad
					component_grad_sum += component_contribution
					component_weight_sum += component_weight
					if abs(component_contribution) >= dominant_component_value:
						dominant_component_value = abs(component_contribution)
						dominant_component = energy_key
						dominant_component_grad = component_grad

			if component_active:
				current_grad = float(component_grad_sum)
				gate_grad = float(dominant_component_grad) if dominant_component_grad is not None else current_grad
			else:
				energy_grad = np.gradient(total_curve.astype(np.float32), samples.astype(np.float32))
				current_grad = float(energy_grad[current_idx]) if energy_grad.size > current_idx else 0.0
				gate_grad = current_grad
			raw_adjust_ms = float(-STAGE1_ENERGY_GRADIENT_GAIN * current_grad)
			clipped_adjust_ms = float(np.clip(
				raw_adjust_ms,
				-STAGE1_ENERGY_GRADIENT_MAX_DELTA_MS,
				STAGE1_ENERGY_GRADIENT_MAX_DELTA_MS,
			))
			score_gate = component_active or max(current_score, path_peak_score) >= STAGE1_ENERGY_GRADIENT_MIN_SCORE
			slope_gate = abs(gate_grad) >= STAGE1_ENERGY_GRADIENT_SLOPE_EPS
			debug.update({
				'gradient_dedv': current_grad,
				'gradient_gate_dedv': gate_grad,
				'gradient_dominant_component': dominant_component,
				'gradient_raw_adjust_ms': raw_adjust_ms,
				'gradient_score_gate': bool(score_gate),
				'gradient_slope_gate': bool(slope_gate),
			})

			if (
				score_gate
				and slope_gate
				and np.isfinite(clipped_adjust_ms)
				and abs(clipped_adjust_ms) > 1e-4
			):
				reason_base = 'stage1_energy_gradient'
				if dominant_component is not None:
					reason_base = f'{reason_base}_{dominant_component}'
				debug.update({
					'active': True,
					'reason': f'{reason_base}_accel' if clipped_adjust_ms > 0.0 else f'{reason_base}_brake',
					'speed_adjust_ms': clipped_adjust_ms,
				})
			return debug

		if STAGE1_ENERGY_SPEED_CONTROL_MODE != 'cap':
			self.stage1_energy_cap_hold_frames = 0
			self.stage1_energy_cap_value_ms = None
			debug['reason'] = f'invalid_mode:{STAGE1_ENERGY_SPEED_CONTROL_MODE}'
			return debug

		level = None
		if speed_delta >= STAGE1_ENERGY_CAP_SPEED_EPS and moving_toward_danger:
			if target_score >= STAGE1_ENERGY_CAP_TARGET_HIGH or path_peak_score >= STAGE1_ENERGY_CAP_PEAK_HIGH:
				level = 'high'
				safe_threshold = float(STAGE1_ENERGY_CAP_SAFE_HIGH)
			elif target_score >= STAGE1_ENERGY_CAP_TARGET_WARN or path_peak_score >= STAGE1_ENERGY_CAP_TARGET_WARN:
				level = 'warn'
				safe_threshold = float(STAGE1_ENERGY_CAP_SAFE_WARN)
			else:
				safe_threshold = None
		elif speed_delta <= -STAGE1_ENERGY_CAP_SPEED_EPS:
			# If the chosen target speed is already safer than the current speed,
			# do not add extra intervention on top of the nominal slowdown.
			if target_score >= current_score - STAGE1_ENERGY_CAP_TREND_MARGIN:
				if target_score >= STAGE1_ENERGY_CAP_TARGET_HIGH:
					level = 'high'
					safe_threshold = float(STAGE1_ENERGY_CAP_SAFE_HIGH)
				elif target_score >= STAGE1_ENERGY_CAP_TARGET_WARN:
					level = 'warn'
					safe_threshold = float(STAGE1_ENERGY_CAP_SAFE_WARN)
				else:
					safe_threshold = None
			else:
				safe_threshold = None
		else:
			safe_threshold = None
		debug['safe_threshold'] = safe_threshold

		if level is not None:
			speed_ceiling = max(0.0, target_speed)
			decel_indices = np.where(samples <= speed_ceiling + 1e-3)[0]
			if decel_indices.size == 0:
				decel_indices = np.arange(samples.size)
			safe_indices = [int(i) for i in decel_indices if float(total_curve[i]) <= safe_threshold]
			if len(safe_indices) > 0:
				chosen_idx = max(safe_indices, key=lambda i: float(samples[i]))
				reason = f'stage1_energy_{level}'
			else:
				chosen_idx = int(decel_indices[np.argmin(total_curve[decel_indices])])
				reason = f'stage1_energy_{level}_fallback'
			candidate_cap_ms = float(samples[chosen_idx])
			self.stage1_energy_cap_value_ms = candidate_cap_ms
			self.stage1_energy_cap_hold_frames = int(STAGE1_ENERGY_CAP_HOLD_FRAMES)
			debug.update({
				'active': True,
				'reason': reason,
				'speed_cap_ms': candidate_cap_ms,
				'hold_frames_remaining': int(self.stage1_energy_cap_hold_frames),
			})
			return debug

		if (
			speed_delta <= -STAGE1_ENERGY_CAP_SPEED_EPS
			and target_score < current_score - STAGE1_ENERGY_CAP_TREND_MARGIN
		):
			self.stage1_energy_cap_hold_frames = 0
			self.stage1_energy_cap_value_ms = None
			return debug

		if self.stage1_energy_cap_hold_frames > 0 and self.stage1_energy_cap_value_ms is not None:
			self.stage1_energy_cap_hold_frames -= 1
			debug.update({
				'active': True,
				'reason': 'stage1_energy_hold',
				'speed_cap_ms': float(self.stage1_energy_cap_value_ms),
				'hold_frames_remaining': int(self.stage1_energy_cap_hold_frames),
			})
			return debug

		self.stage1_energy_cap_hold_frames = 0
		self.stage1_energy_cap_value_ms = None
		return debug

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
		self.use_lidar_bev_detail = bool(
			self.config.get('route_b', {}).get('use_lidar_bev_detail', False)
		)
		self.use_front_route_risk_energy = bool(
			self.config.get('route_b', {}).get('use_front_route_risk_energy', False)
		)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		checkpoint_path = self.resolve_checkpoint_path()
		self.net = load_best_model(checkpoint_path, self.config, device)
		if NUM_INFERENCE_STEPS_OVERRIDE:
			override_steps = int(NUM_INFERENCE_STEPS_OVERRIDE)
			self.net.num_inference_steps = override_steps
			self.config.setdefault('route_b', {})['num_inference_steps'] = override_steps
			if hasattr(self.net, 'route_b_cfg') and isinstance(self.net.route_b_cfg, dict):
				self.net.route_b_cfg['num_inference_steps'] = override_steps
			print(f"Overriding Route-B num_inference_steps -> {override_steps}")
		print("✓ Diffusion policy loaded (float32).")
		print(f"  - use_lidar_bev_detail: {self.use_lidar_bev_detail}")
		print(f"  - use_front_route_risk_energy: {self.use_front_route_risk_energy}")
		
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

			# ========== Load TransFuser Backbone(s) for DP features ==========
			transfuser_config_path = "/media/z/data/models/garage2/pretrained_models/all_towns"
			transfuser_model_paths = [
				os.path.join(transfuser_config_path, "model_0030_0.pth"),
				os.path.join(transfuser_config_path, "model_0030_1.pth"),
				os.path.join(transfuser_config_path, "model_0030_2.pth"),
			]
			self.transfuser_backbones = []
			self.transfuser_bev_semantic_decoders = []
			for mp in transfuser_model_paths:
				print(f"Loading TransFuser backbone: {os.path.basename(mp)}")
				bb = TransFuserBackboneExtractor(
					config_path=transfuser_config_path,
					model_path=mp,
					device='cuda:0'
				)
				bb.eval()
				self.transfuser_backbones.append(bb)
				if not hasattr(self, 'transfuser_config'):
					self.transfuser_config = bb.config
				self.transfuser_bev_semantic_decoders.append(
					self._build_transfuser_bev_semantic_decoder(
						model_path=mp,
						device='cuda:0',
					)
				)
			# Keep first backbone's config (all share same architecture)
			self.transfuser_config = self.transfuser_backbones[0].config
			self.transfuser_bev_semantic_decoder = self.transfuser_bev_semantic_decoders[0]
			self._init_semantic_hazard_state()
			# Initialize TransfuserData for lidar histogram conversion
			self.transfuser_data = TransfuserData(root=[], config=self.transfuser_config, shared_dict=None)
			print("✓ TransFuser backbone and BEV semantic decoder loaded, frozen, and using float32.")
		
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
		self.stuck_helper_startup_threshold = STUCK_HELPER_STARTUP_THRESHOLD
		self.stuck_helper_poststart_threshold = STUCK_HELPER_POSTSTART_THRESHOLD
		# Startup stuck uses helper target override; once ego leaves the startup zone we
		# fall back to the legacy creep-based stuck recovery.
		self.stuck_threshold = int(self.stuck_helper_poststart_threshold)
		self.stuck_helper_threshold = STUCK_HELPER_STARTUP_THRESHOLD
		self.stuck_helper_startup_distance_m = STUCK_HELPER_STARTUP_DISTANCE_M
		self.creep_duration = 15
		self.creep_throttle = 0.4
		
		# Stuck detection
		self.stuck_detector = 0
		self.stuck_helper = 0
		self.stuck_helper_active = False
		self.stuck_helper_mode = None
		self.stuck_helper_start_heading = None
		self.stuck_helper_heading_delta_deg = 0.0
		self.stuck_helper_startup_reference_xy = None
		self.stuck_helper_distance_from_start_m = 0.0
		self.stuck_helper_in_startup_zone = True
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
		self.localizer = EgoLocalizer(
			strategy=LOCALIZER_STRATEGY,
			dt=self.carla_frame_rate,
			gps_alpha=LOCALIZER_ALPHA,
			latency_compensation=False,
		)
		print(
			f"[EgoLocalizer] strategy={LOCALIZER_STRATEGY}, "
			f"alpha={LOCALIZER_ALPHA}, lidar_pose={LIDAR_POSE_SOURCE}"
		)
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
		if SAVE_TRANSFUSER_BEV_DEBUG:
			(self.save_path / 'transfuser_lidar_bev').mkdir()
			(self.save_path / 'transfuser_bev_semantic').mkdir()
			(self.save_path / 'transfuser_bev_vehicle').mkdir()
		(self.save_path / 'debug_vis').mkdir()
		
		# Initialize lidar buffer for combining two frames
		self.lidar_buffer = deque(maxlen=2)
		self.lidar_step_counter = 0
		self.last_ego_transform = None
		self.last_lidar = None
		
		obs_horizon = self.config.get('obs_horizon', 4)   
		self.obs_horizon = obs_horizon
		self.transfuser_lidar_history_frames = int(
			self.config.get('route_b', {}).get('lidar_history_frames', 1)
		)
		self.transfuser_lidar_bev_detail_history = deque(
			maxlen=max(self.transfuser_lidar_history_frames, 1)
		)
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
		self.last_branch_condition_debug = {}
		self.last_model_input_debug = {}
		self.last_speed_debug = {}
		self.prev_lane_dir_relation_probs = np.array([0.5, 0.5], dtype=np.float32)
		self.borrow_candidate_start_time_s = None
		self.borrow_latched = False
		self.borrow_latch_counter = 0
		self.borrow_release_counter = 0
		self.borrow_observed_time_s = 0.0
		self.front_route_risk_ema = None
		self.front_route_risk_sigmoid_history = deque(maxlen=FRONT_ROUTE_RISK_RECENT_WINDOW)
		self.front_route_risk_cap_hold_frames = 0
		self.front_route_risk_cap_value_ms = None
		self.last_front_route_risk_cap_debug = {}
		self.stage1_energy_cap_hold_frames = 0
		self.stage1_energy_cap_value_ms = None
		self.last_stage1_energy_cap_debug = {}
		self.last_terminal_route_debug = {}
		self.last_junction_window_soft_cap_debug = {}
		self.junction_window_soft_cap_streak = 0
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
		

		self.route_planner_min_distance = float(os.environ.get('ROUTE_PLANNER_MIN_DISTANCE', '7.5'))
		self.route_planner_max_distance = float(os.environ.get('ROUTE_PLANNER_MAX_DISTANCE', '50.0'))
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

	def _ego_points_to_world(self, ego_points, ego_xy, yaw):
		ego_points = np.asarray(ego_points, dtype=np.float32)
		ego_xy = np.asarray(ego_xy, dtype=np.float32)
		cos_yaw = np.cos(yaw)
		sin_yaw = np.sin(yaw)
		rotation = np.array([
			[cos_yaw, -sin_yaw],
			[sin_yaw, cos_yaw],
		], dtype=np.float32)
		return (ego_points @ rotation.T) + ego_xy

	def _get_stuck_helper_target_override(self, ego_xy, yaw):
		if (
			not STUCK_HELPER_TARGET_INSERT_ENABLE
			or STUCK_HELPER_STARTUP_RECOVERY_MODE != 'target'
			or not self.stuck_helper_active
			or self.stuck_helper_mode != 'startup'
		):
			return None

		# Only the startup stuck helper is allowed to rewrite target points. Once ego
		# has moved away from the startup point, stuck handling falls back to legacy
		# creep recovery without touching the planner targets.
		helper_ego_points = np.array([
			[STUCK_HELPER_TARGET_1_FORWARD_M, STUCK_HELPER_TARGET_LATERAL_M],
			[STUCK_HELPER_TARGET_2_FORWARD_M, STUCK_HELPER_TARGET_LATERAL_M],
		], dtype=np.float32)
		helper_world_points = self._ego_points_to_world(helper_ego_points, ego_xy, yaw)
		frames_remaining = int(self.stuck_helper)
		return {
			'active': True,
			'frames_remaining': frames_remaining,
			'target_point_ego': helper_ego_points[0].copy(),
			'next_target_point_ego': helper_ego_points[1].copy(),
			'target_point_world': helper_world_points[0].copy(),
			'next_target_point_world': helper_world_points[1].copy(),
		}

	def _get_stuck_helper_control_override(self):
		if (
			STUCK_HELPER_STARTUP_RECOVERY_MODE != 'control'
			or not self.stuck_helper_active
			or self.stuck_helper_mode != 'startup'
		):
			return None
		return {
			'active': True,
			'frames_remaining': int(self.stuck_helper),
			'throttle': float(STUCK_HELPER_CONTROL_THROTTLE),
			'steer': float(STUCK_HELPER_CONTROL_STEER),
			'brake': 0.0,
		}

	def _reset_stuck_helper_state(self):
		self.stuck_helper = 0
		self.stuck_helper_active = False
		self.stuck_helper_mode = None
		self.stuck_helper_start_heading = None
		self.stuck_helper_heading_delta_deg = 0.0

	def _activate_stuck_helper(self, current_heading, mode='startup'):
		self.stuck_helper_active = True
		self.stuck_helper = 1
		self.stuck_helper_mode = mode
		self.stuck_helper_start_heading = float(current_heading)
		self.stuck_helper_heading_delta_deg = 0.0

	def _update_stuck_helper_startup_state(self, ego_xy):
		if ego_xy is None:
			return
		ego_xy = np.asarray(ego_xy, dtype=np.float32)
		if self.stuck_helper_startup_reference_xy is None:
			self.stuck_helper_startup_reference_xy = ego_xy.copy()
		delta_xy = ego_xy - self.stuck_helper_startup_reference_xy
		self.stuck_helper_distance_from_start_m = float(np.linalg.norm(delta_xy))
		self.stuck_helper_in_startup_zone = (
			self.stuck_helper_distance_from_start_m <= self.stuck_helper_startup_distance_m
		)
		self.stuck_helper_threshold = (
			self.stuck_helper_startup_threshold
			if self.stuck_helper_in_startup_zone
			else self.stuck_helper_poststart_threshold
		)
		if not self.stuck_helper_in_startup_zone and self.stuck_helper_mode == 'startup':
			self._reset_stuck_helper_state()

	def _update_stuck_helper_release(self, current_heading):
		if not self.stuck_helper_active or self.stuck_helper_start_heading is None:
			return
		delta_rad = float(current_heading) - float(self.stuck_helper_start_heading)
		delta_deg = float(np.rad2deg(np.arctan2(np.sin(delta_rad), np.cos(delta_rad))))
		self.stuck_helper_heading_delta_deg = abs(delta_deg)
		if self.stuck_helper_heading_delta_deg >= STUCK_HELPER_RELEASE_HEADING_DEG:
			self.stuck_detector = 0
			self._reset_stuck_helper_state()


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
		
		# Get speed
		speed = input_data['SPEED'][1]['speed']
		
		# ---- Localizer pose: used for target point / planner pose when requested ----
		gps_raw_xy = np.array([gps_pos[0], gps_pos[1]], dtype=np.float64)
		if hasattr(self, 'localizer') and self.localizer is not None:
			localized_pos, localized_yaw = self.localizer.update(
				gps_xy=gps_raw_xy,
				compass=compass,
				speed=speed,
				steer=self.control.steer,
				throttle=self.control.throttle,
				brake=float(self.control.brake > 0.5),
				imu_data=input_data['IMU'][1],
			)
			gps_filtered = np.asarray(localized_pos, dtype=np.float64)
			compass_filtered = float(localized_yaw)
		elif USE_UKF:
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

		# ---- Lidar alignment pose: keep the old controllable split between ukf/localizer ----
		if LIDAR_POSE_SOURCE == 'ukf' and USE_UKF:
			if not self.filter_initialized:
				self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
				self.filter_initialized = True

			self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
			self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
			lidar_state = self.ukf.x
			self.state_log.append(lidar_state)
			gps_lidar = np.asarray(lidar_state[0:2], dtype=np.float64)
			compass_lidar = float(lidar_state[2])
		else:
			gps_lidar = np.asarray(gps_filtered, dtype=np.float64)
			compass_lidar = float(compass_filtered)

		gps_target_pose, compass_target_pose, target_pose_source = self._resolve_target_pose(
			gps_raw=np.array([gps_pos[0], gps_pos[1]], dtype=np.float32),
			gps_filtered=gps_filtered,
			compass_raw=compass,
			compass_filtered=compass_filtered,
		)

		# Combine two frames of lidar data using the selected lidar pose.
		if self.last_lidar is not None and self.last_ego_transform is not None:
			# Calculate relative transformation between current and last frame
			current_pos = np.array([gps_lidar[0], gps_lidar[1], 0.0])
			last_pos = np.array([self.last_ego_transform['gps'][0], self.last_ego_transform['gps'][1], 0.0])
			relative_translation = current_pos - last_pos
			
			# Calculate relative rotation using lidar-alignment yaw
			current_yaw = compass_lidar
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
		
		# Store current frame for next iteration (use lidar-alignment pose)
		self.last_lidar = lidar_ego
		self.last_ego_transform = {'gps': gps_lidar, 'compass': compass_lidar}
		
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
		self.transfuser_state_log.append([gps_lidar[0], gps_lidar[1], compass_lidar, speed])
		
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
		# LiDAR-detail path follows the 2-channel split histogram convention from docs:
		# ch0 = below-split histogram, ch1 = above-split histogram, then inverted.
		transfuser_lidar_bev_detail = self.transfuser_data.lidar_to_histogram_features(
			transfuser_lidar_full,
			use_ground_plane=True
		)
		transfuser_lidar_bev_detail_tensor = torch.from_numpy(
			transfuser_lidar_bev_detail
		).float().unsqueeze(0).to('cuda')
		transfuser_lidar_bev_inv = 1.0 - transfuser_lidar_bev_detail_tensor
		
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
				'gps_lidar': gps_lidar,
				'compass_lidar': compass_lidar,
				'target_pose_source': target_pose_source,
				# TransFuser processed data for DP
				'transfuser_rgb': transfuser_rgb_tensor,  # (1, 3, H, W) on GPU
				'transfuser_lidar_bev': transfuser_lidar_bev_tensor,  # (1, C, H, W) on GPU, raw for backbone
				'transfuser_lidar_bev_inv': transfuser_lidar_bev_inv,  # (1, C, H, W) inverted for DiT detail
				}
		self._update_stuck_helper_startup_state(result['gps'][:2])
		
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

		prepromote_target_point_world = np.asarray(target_point[:2], dtype=np.float32)
		prepromote_next_target_point_world = np.asarray(next_target_point[:2], dtype=np.float32)
		prepromote_target_point_ego = t_u.inverse_conversion_2d(
			prepromote_target_point_world, result['gps'], result['compass']
		).astype(np.float32)
		prepromote_next_target_point_ego = t_u.inverse_conversion_2d(
			prepromote_next_target_point_world, result['gps'], result['compass']
		).astype(np.float32)
		early_target_promote_active = False
		early_target_promote_reason = None

		if (
			EARLY_TARGET_PROMOTE_ENABLE
			and len(waypoint_route) > 2
		):
			cur_forward = float(prepromote_target_point_ego[0])
			cur_lateral = float(prepromote_target_point_ego[1])
			next_forward = float(prepromote_next_target_point_ego[0])
			next_lateral = float(prepromote_next_target_point_ego[1])
			cur_angle_deg = float(np.rad2deg(np.arctan2(abs(cur_lateral), max(cur_forward, 1e-6))))
			next_angle_deg = float(np.rad2deg(np.arctan2(abs(next_lateral), max(next_forward, 1e-6))))
			if (
				cur_forward <= EARLY_TARGET_PROMOTE_CUR_FORWARD_MAX_M
				and abs(cur_lateral) >= EARLY_TARGET_PROMOTE_CUR_LATERAL_MIN_M
				and cur_angle_deg >= EARLY_TARGET_PROMOTE_CUR_ANGLE_MIN_DEG
				and next_forward >= EARLY_TARGET_PROMOTE_NEXT_FORWARD_MIN_M
				and next_angle_deg <= EARLY_TARGET_PROMOTE_NEXT_ANGLE_MAX_DEG
			):
				early_target_promote_active = True
				early_target_promote_reason = (
					f"cur_fwd={cur_forward:.2f},cur_lat={cur_lateral:.2f},"
					f"cur_ang={cur_angle_deg:.1f},next_fwd={next_forward:.2f},"
					f"next_ang={next_angle_deg:.1f}"
				)
				target_point, far_command = waypoint_route[2]
				if len(waypoint_route) > 3:
					next_target_point, next_far_command = waypoint_route[3]
				else:
					promoted_target_world = np.asarray(target_point[:2], dtype=np.float32)
					promoted_direction = promoted_target_world - prepromote_next_target_point_world
					promoted_dist = float(np.linalg.norm(promoted_direction))
					if promoted_dist > 1e-3:
						promoted_direction = promoted_direction / promoted_dist
					else:
						promoted_direction = np.array(
							[np.cos(result['compass']), np.sin(result['compass'])],
							dtype=np.float32,
						)
					next_target_point = promoted_target_world + promoted_direction * 50.0
					next_far_command = far_command

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

		raw_target_point_world = np.asarray(target_point[:2], dtype=np.float32)
		raw_next_target_point_world = np.asarray(next_target_point[:2], dtype=np.float32)
		raw_target_point_ego = t_u.inverse_conversion_2d(
			raw_target_point_world, result['gps'], result['compass']
		).astype(np.float32)
		raw_next_target_point_ego = t_u.inverse_conversion_2d(
			raw_next_target_point_world, result['gps'], result['compass']
		).astype(np.float32)

		stuck_helper_debug = {
			'active': bool(self.stuck_helper_active),
			'frames_remaining': int(self.stuck_helper),
			'target_points_ego': None,
			'target_points_world': None,
			'control_override': None,
			'recovery_mode': (
				STUCK_HELPER_STARTUP_RECOVERY_MODE
				if self.stuck_helper_mode == 'startup'
				else None
			),
		}
		ego_target_point = raw_target_point_ego
		ego_next_target_point = raw_next_target_point_ego
		target_point_world = raw_target_point_world
		next_target_point_world = raw_next_target_point_world
		stuck_helper_override = self._get_stuck_helper_target_override(
			result['gps'][:2],
			result['compass'],
		)
		if stuck_helper_override is not None:
			ego_target_point = stuck_helper_override['target_point_ego']
			ego_next_target_point = stuck_helper_override['next_target_point_ego']
			target_point_world = stuck_helper_override['target_point_world']
			next_target_point_world = stuck_helper_override['next_target_point_world']
			stuck_helper_debug = {
				'active': True,
				'frames_remaining': int(stuck_helper_override['frames_remaining']),
				'target_points_ego': [
					stuck_helper_override['target_point_ego'].tolist(),
					stuck_helper_override['next_target_point_ego'].tolist(),
				],
				'target_points_world': [
					stuck_helper_override['target_point_world'].tolist(),
					stuck_helper_override['next_target_point_world'].tolist(),
				],
				'control_override': None,
				'recovery_mode': 'target',
			}

		stuck_helper_control_override = self._get_stuck_helper_control_override()
		if stuck_helper_control_override is not None:
			stuck_helper_debug['active'] = True
			stuck_helper_debug['control_override'] = {
				'throttle': float(stuck_helper_control_override['throttle']),
				'steer': float(stuck_helper_control_override['steer']),
				'brake': float(stuck_helper_control_override['brake']),
			}
			stuck_helper_debug['recovery_mode'] = 'control'

		forward_vec_world = np.array([
			np.cos(result['compass']),
			np.sin(result['compass']),
		], dtype=np.float32)
		target_delta_world = target_point_world - np.asarray(result['gps'][:2], dtype=np.float32)
		next_target_delta_world = next_target_point_world - np.asarray(result['gps'][:2], dtype=np.float32)

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
		result['target_point_world'] = target_point_world
		result['next_target_point_world'] = next_target_point_world
		result['target_point_prepromote_ego'] = prepromote_target_point_ego
		result['next_target_point_prepromote_ego'] = prepromote_next_target_point_ego
		result['target_point_prepromote_world'] = prepromote_target_point_world
		result['next_target_point_prepromote_world'] = prepromote_next_target_point_world
		result['early_target_promote_active'] = bool(early_target_promote_active)
		result['early_target_promote_reason'] = early_target_promote_reason
		result['target_point_raw_ego'] = raw_target_point_ego
		result['next_target_point_raw_ego'] = raw_next_target_point_ego
		result['target_point_raw_world'] = raw_target_point_world
		result['next_target_point_raw_world'] = raw_next_target_point_world
		result['stuck_helper_active'] = bool(stuck_helper_debug['active'])
		result['stuck_helper_mode'] = self.stuck_helper_mode
		result['stuck_helper_recovery_mode'] = stuck_helper_debug['recovery_mode']
		result['stuck_helper_control_override'] = stuck_helper_debug['control_override']
		result['stuck_helper_frames_remaining'] = int(stuck_helper_debug['frames_remaining'])
		result['stuck_helper_heading_delta_deg'] = float(self.stuck_helper_heading_delta_deg)
		result['stuck_helper_release_heading_deg'] = float(STUCK_HELPER_RELEASE_HEADING_DEG)
		result['stuck_helper_threshold'] = int(self.stuck_helper_threshold)
		result['stuck_helper_in_startup_zone'] = bool(self.stuck_helper_in_startup_zone)
		result['stuck_helper_distance_from_start_m'] = float(self.stuck_helper_distance_from_start_m)
		result['stuck_helper_startup_distance_m'] = float(self.stuck_helper_startup_distance_m)
		result['stuck_helper_startup_threshold'] = int(self.stuck_helper_startup_threshold)
		result['stuck_helper_poststart_threshold'] = int(self.stuck_helper_poststart_threshold)
		result['stuck_helper_startup_reference_xy'] = (
			self.stuck_helper_startup_reference_xy.tolist()
			if isinstance(self.stuck_helper_startup_reference_xy, np.ndarray)
			else self.stuck_helper_startup_reference_xy
		)
		result['stuck_helper_target_points_ego'] = stuck_helper_debug['target_points_ego']
		result['stuck_helper_target_points_world'] = stuck_helper_debug['target_points_world']
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
	
	def control_pid(
		self,
		route_waypoints,
		velocity,
		speed_waypoints,
		target_point=None,
		junction_window_soft_cap_ms=None,
		terminal_speed_cap_ms=None,
		front_route_risk_cap_ms=None,
		stage1_energy_cap_ms=None,
		stage1_energy_speed_adjust_ms=None,
	):
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
		
		# Trajectory-based speed estimates (always computed)
		# MoT trajectory: 6 points, 0.5s interval each, total 3s
		# traj_1s: ||wp[2] - wp[0]|| — 1.0s window, smoother, better at high speed
		# traj_05s: ||wp[1] - wp[0]|| * 2 — 0.5s window, more reactive, better at low speed
		if speed_waypoints_np.shape[0] >= 3:
			traj_speed_1s = float(np.linalg.norm(speed_waypoints_np[2] - speed_waypoints_np[0]))
		elif speed_waypoints_np.shape[0] >= 2:
			traj_speed_1s = float(np.linalg.norm(speed_waypoints_np[1] - speed_waypoints_np[0]) * 2.0)
		else:
			traj_speed_1s = float(np.linalg.norm(speed_waypoints_np[0]) * 2.0)

		if speed_waypoints_np.shape[0] >= 2:
			traj_speed_05s = float(np.linalg.norm(speed_waypoints_np[1] - speed_waypoints_np[0]) * 2.0)
		else:
			traj_speed_05s = float(np.linalg.norm(speed_waypoints_np[0]) * 2.0)

		traj_speed = traj_speed_1s  # default traj speed for legacy modes

		# Speed source selection via SPEED_SOURCE env var
		speed_head_speed = float(self._last_target_speed) if hasattr(self, '_last_target_speed') and self._last_target_speed is not None else None
		fusion_regime = 'single_source'
		fusion_rough_speed = None
		fusion_weights = None

		if SPEED_SOURCE == 'traj':
			desired_speed = traj_speed
		elif SPEED_SOURCE in ('traj_05s', 'traj05'):
			desired_speed = traj_speed_05s
		elif SPEED_SOURCE == 'fuse' and speed_head_speed is not None:
			# speed head primary, traj lower bound
			desired_speed = speed_head_speed
			desired_speed = max(desired_speed, traj_speed * 0.5)
		elif SPEED_SOURCE == 'fuse_traj' and speed_head_speed is not None:
			# traj primary, speed head lower bound
			desired_speed = traj_speed
			desired_speed = max(desired_speed, speed_head_speed * 0.5)
		elif SPEED_SOURCE == 'fuse3_median' and speed_head_speed is not None:
			# Median of three: robust to any single source outlier
			desired_speed = float(np.median([speed_head_speed, traj_speed_1s, traj_speed_05s]))
			fusion_regime = 'median'
			fusion_rough_speed = desired_speed
			fusion_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
		elif SPEED_SOURCE == 'fuse3_adaptive' and speed_head_speed is not None:
			# Adaptive three-way fusion based on measured MAE by regime.
			# Sources: [speed_head, traj_1s, traj_0.5s*2]
			# startup from standstill: bias toward speed_head + traj_1s for quicker launch
			# stop/near-stop: [0.29, 0.30, 0.41]
			# medium speed:   [0.36, 0.34, 0.30]
			# high speed:     [0.30, 0.37, 0.33]
			# Use the median as a robust rough estimate to choose the regime.
			rough = float(np.median([speed_head_speed, traj_speed_1s, traj_speed_05s]))
			fusion_rough_speed = rough
			if rough < 2.5:
				if speed < 0.2:
					fusion_regime = 'startup'
					fusion_weights = [0.50, 0.35, 0.15]
				else:
					fusion_regime = 'low'
					fusion_weights = [0.29, 0.30, 0.41]
				desired_speed = (
					fusion_weights[0] * speed_head_speed
					+ fusion_weights[1] * traj_speed_1s
					+ fusion_weights[2] * traj_speed_05s
				)
			elif rough < 10.0:
				fusion_regime = 'medium'
				fusion_weights = [0.36, 0.34, 0.30]
				desired_speed = (
					fusion_weights[0] * speed_head_speed
					+ fusion_weights[1] * traj_speed_1s
					+ fusion_weights[2] * traj_speed_05s
				)
			else:
				fusion_regime = 'high'
				fusion_weights = [0.30, 0.37, 0.33]
				desired_speed = (
					fusion_weights[0] * speed_head_speed
					+ fusion_weights[1] * traj_speed_1s
					+ fusion_weights[2] * traj_speed_05s
				)
		elif speed_head_speed is not None:
			# 'speed_head': speed head only
			desired_speed = speed_head_speed
		else:
			desired_speed = traj_speed

		desired_speed_raw = float(desired_speed)
		desired_speed_after_soft_cap = float(desired_speed)
		if SOFT_SPEED_LIMIT_MS > 0.0:
			desired_speed_after_soft_cap = min(desired_speed_after_soft_cap, SOFT_SPEED_LIMIT_MS)
		desired_speed_after_junction_window_soft_cap = float(desired_speed_after_soft_cap)
		junction_window_soft_cap_applied = False
		if junction_window_soft_cap_ms is not None and junction_window_soft_cap_ms > 0.0:
			junction_window_soft_cap_applied = (
				desired_speed_after_junction_window_soft_cap > float(junction_window_soft_cap_ms)
			)
			desired_speed_after_junction_window_soft_cap = min(
				desired_speed_after_junction_window_soft_cap,
				junction_window_soft_cap_ms,
			)
		desired_speed_after_terminal_cap = float(desired_speed_after_junction_window_soft_cap)
		terminal_speed_cap_applied = False
		if terminal_speed_cap_ms is not None and terminal_speed_cap_ms > 0.0:
			terminal_speed_cap_applied = desired_speed_after_terminal_cap > float(terminal_speed_cap_ms)
			desired_speed_after_terminal_cap = min(desired_speed_after_terminal_cap, terminal_speed_cap_ms)
		desired_speed_after_front_route_risk_cap = float(desired_speed_after_terminal_cap)
		front_route_risk_cap_applied = False
		if front_route_risk_cap_ms is not None and front_route_risk_cap_ms > 0.0:
			front_route_risk_cap_applied = (
				desired_speed_after_front_route_risk_cap > float(front_route_risk_cap_ms)
			)
			desired_speed_after_front_route_risk_cap = min(
				desired_speed_after_front_route_risk_cap,
				front_route_risk_cap_ms,
			)
		desired_speed_after_stage1_energy_adjust = float(desired_speed_after_front_route_risk_cap)
		stage1_energy_speed_adjust_applied = False
		if stage1_energy_speed_adjust_ms is not None and abs(float(stage1_energy_speed_adjust_ms)) > 1e-5:
			adjusted_speed = max(
				0.0,
				desired_speed_after_stage1_energy_adjust + float(stage1_energy_speed_adjust_ms),
			)
			stage1_energy_speed_adjust_applied = (
				abs(adjusted_speed - desired_speed_after_stage1_energy_adjust) > 1e-5
			)
			desired_speed_after_stage1_energy_adjust = adjusted_speed
		desired_speed_after_stage1_energy_cap = float(desired_speed_after_stage1_energy_adjust)
		stage1_energy_cap_applied = False
		if stage1_energy_cap_ms is not None and stage1_energy_cap_ms > 0.0:
			stage1_energy_cap_applied = (
				desired_speed_after_stage1_energy_cap > float(stage1_energy_cap_ms)
			)
			desired_speed_after_stage1_energy_cap = min(
				desired_speed_after_stage1_energy_cap,
				stage1_energy_cap_ms,
			)
		desired_speed = float(desired_speed_after_stage1_energy_cap)
		self.last_speed_debug = {
			'speed_source': SPEED_SOURCE,
			'speed_head_speed': speed_head_speed,
			'traj_speed_1s': traj_speed_1s,
			'traj_speed_05s': traj_speed_05s,
			'fusion_regime': fusion_regime,
			'fusion_rough_speed': fusion_rough_speed,
			'fusion_weights': fusion_weights,
			'desired_speed_raw': desired_speed_raw,
			'desired_speed_capped': float(desired_speed),
			'desired_speed_after_soft_cap': float(desired_speed_after_soft_cap),
			'desired_speed_after_junction_window_soft_cap': float(desired_speed_after_junction_window_soft_cap),
			'junction_window_soft_cap_ms': float(junction_window_soft_cap_ms) if junction_window_soft_cap_ms is not None else None,
			'junction_window_soft_cap_applied': bool(junction_window_soft_cap_applied),
			'desired_speed_after_terminal_cap': float(desired_speed_after_terminal_cap),
			'desired_speed_after_front_route_risk_cap': float(desired_speed_after_front_route_risk_cap),
			'desired_speed_after_stage1_energy_adjust': float(desired_speed_after_stage1_energy_adjust),
			'desired_speed_after_stage1_energy_cap': float(desired_speed_after_stage1_energy_cap),
			'soft_speed_limit_ms': float(SOFT_SPEED_LIMIT_MS),
			'terminal_speed_cap_ms': float(terminal_speed_cap_ms) if terminal_speed_cap_ms is not None else None,
			'terminal_speed_cap_applied': bool(terminal_speed_cap_applied),
			'front_route_risk_cap_ms': float(front_route_risk_cap_ms) if front_route_risk_cap_ms is not None else None,
			'front_route_risk_cap_applied': bool(front_route_risk_cap_applied),
			'stage1_energy_control_mode': STAGE1_ENERGY_SPEED_CONTROL_MODE,
			'stage1_energy_speed_adjust_ms': float(stage1_energy_speed_adjust_ms) if stage1_energy_speed_adjust_ms is not None else None,
			'stage1_energy_speed_adjust_applied': bool(stage1_energy_speed_adjust_applied),
			'stage1_energy_cap_ms': float(stage1_energy_cap_ms) if stage1_energy_cap_ms is not None else None,
			'stage1_energy_cap_applied': bool(stage1_energy_cap_applied),
			'hard_speed_limit_ms': float(HARD_SPEED_LIMIT_MS),
		}

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
		command_value = tick_data['next_command']
		if command_value < 0:
			command_value = 4
		command = command_value - 1
		command_text = self._command_value_to_text(command_value)
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

				if self.stuck_helper_active and STUCK_HELPER_STARTUP_RECOVERY_MODE == 'target':
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
				# Ensemble: run all backbones and average BEV features
				bev_features = []
				bev_upsamples = []
				for bb in self.transfuser_backbones:
					out = bb(
						rgb=transfuser_rgb_fp32,
						lidar_bev=transfuser_lidar_bev_fp32
					)
					bev_features.append(out['bev_feature'])
					bev_upsamples.append(out['bev_feature_upscale'])
				transfuser_bev_feature = torch.stack(bev_features).mean(dim=0)  # (1, 1512, 8, 8)
				transfuser_bev_feature_upsample = torch.stack(bev_upsamples).mean(dim=0)  # (1, 64, 64, 64)
				bev_semantic_classes = self._decode_bev_semantic_classes(
					bev_upsamples
				)
				tick_data['bev_semantic_classes'] = bev_semantic_classes
				
				# Keep predict_action batch schema stable on the non-LiDAR path.
				# The policy will ignore this tensor when use_lidar_bev_detail is false,
				# but dict_apply() still requires a real tensor instead of None.
				transfuser_lidar_bev_raw = tick_data['transfuser_lidar_bev']
				transfuser_lidar_bev_detail = tick_data.get('transfuser_lidar_bev_inv')
				if transfuser_lidar_bev_detail is None:
					batch_size = transfuser_lidar_bev_raw.shape[0]
					height = transfuser_lidar_bev_raw.shape[-2]
					width = transfuser_lidar_bev_raw.shape[-1]
					transfuser_lidar_bev_detail = torch.zeros(
						(batch_size, 2, height, width),
						dtype=transfuser_lidar_bev_raw.dtype,
						device=transfuser_lidar_bev_raw.device,
					)
				detail_input_fallback = not self.use_lidar_bev_detail
				if detail_input_fallback:
					transfuser_lidar_bev_detail = torch.zeros_like(
						transfuser_lidar_bev_detail
					)
				current_lidar_detail = transfuser_lidar_bev_detail
				if current_lidar_detail.dim() == 4 and current_lidar_detail.shape[0] == 1:
					current_lidar_detail = current_lidar_detail.squeeze(0)
				self.transfuser_lidar_bev_detail_history.append(
					current_lidar_detail.detach()
				)
				lidar_detail_history = list(self.transfuser_lidar_bev_detail_history)
				hist_len = max(int(self.transfuser_lidar_history_frames), 1)
				while len(lidar_detail_history) < hist_len:
					lidar_detail_history.insert(0, torch.zeros_like(current_lidar_detail))
				lidar_detail_history = lidar_detail_history[-hist_len:]
				transfuser_lidar_bev_detail = torch.stack(
					lidar_detail_history, dim=0
				).unsqueeze(0)

				def _tensor_debug_stats(value):
					if value is None:
						return {
							'mean': None,
							'max': None,
							'nonzero_ratio': None,
						}
					if isinstance(value, torch.Tensor):
						value_cpu = value.detach().float().cpu()
					else:
						value_cpu = torch.as_tensor(value).float()
					if value_cpu.numel() == 0:
						return {
							'mean': None,
							'max': None,
							'nonzero_ratio': None,
						}
					return {
						'mean': float(value_cpu.mean().item()),
						'max': float(value_cpu.max().item()),
						'nonzero_ratio': float((value_cpu != 0).float().mean().item()),
					}

				raw_lidar_stats = _tensor_debug_stats(transfuser_lidar_bev_raw)
				detail_lidar_stats = _tensor_debug_stats(transfuser_lidar_bev_detail)
				self.last_model_input_debug = {
					'use_lidar_bev_detail': bool(self.use_lidar_bev_detail),
					'use_front_route_risk_energy': bool(self.use_front_route_risk_energy),
					'lidar_bev_detail_zero_fallback': bool(detail_input_fallback),
					'lidar_bev_raw_mean': raw_lidar_stats['mean'],
					'lidar_bev_raw_max': raw_lidar_stats['max'],
					'lidar_bev_raw_nonzero_ratio': raw_lidar_stats['nonzero_ratio'],
					'lidar_bev_detail_mean': detail_lidar_stats['mean'],
					'lidar_bev_detail_max': detail_lidar_stats['max'],
					'lidar_bev_detail_nonzero_ratio': detail_lidar_stats['nonzero_ratio'],
				}

				# Build dp_obs_dict with transfuser features
				current_time_s = float(self.step) * float(self.carla_frame_rate)
				borrow_time_cond_s = self._get_observed_borrow_time_s(current_time_s)
				dp_obs_dict = {
					'ego_status': ego_status_stacked,
					'transfuser_bev_feature': transfuser_bev_feature,  # (B, 1512, 8, 8)
					'transfuser_bev_feature_upsample': transfuser_bev_feature_upsample,  # (B, 64, 64, 64)
					'transfuser_lidar_bev': transfuser_lidar_bev_detail,  # (1, 2, 256, 256)
					'prev_lane_dir_relation_probs': torch.from_numpy(
						self.prev_lane_dir_relation_probs.astype(np.float32)
					).unsqueeze(0).to('cuda', dtype=torch.float32),
					'borrow_cross_active_time_s': torch.tensor(
						[borrow_time_cond_s], device='cuda', dtype=torch.float32
					),
				}
			dp_pred_traj = self._predict_dp_action(dp_obs_dict)
			self._update_borrow_semantic_state(dp_pred_traj, current_time_s)
			# Store predicted target speed for control_pid
			self._last_target_speed = dp_pred_traj.get('target_speed', None)
			if self._last_target_speed is not None:
				self._last_target_speed = float(np.asarray(self._last_target_speed).reshape(-1)[0])
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
			traj_decision_phase_condition_names = list(
				getattr(
					self.net,
					'traj_decision_phase_condition_names',
					getattr(self.net, 'traj_phase_condition_names', ('yld', 'go')),
				)
			)
			self.last_branch_condition_debug = {
				'traj_branch_condition_enabled': bool(getattr(self.net, 'use_traj_branch_condition', False)),
				'traj_branch_condition_scale': float(getattr(self.net, 'traj_branch_condition_scale', 0.0) or 0.0),
				'traj_branch_condition_detach': bool(getattr(self.net, 'traj_branch_condition_detach', False)),
				'traj_branch_condition_affinity_scale': float(
					getattr(self.net, 'traj_branch_condition_affinity_scale', 0.0) or 0.0
				),
				'traj_branch_condition_borrow_time_scale': float(
					getattr(self.net, 'traj_branch_condition_borrow_time_scale', 0.0) or 0.0
				),
				'traj_branch_condition_boundary_margin_scale': float(
					getattr(self.net, 'traj_branch_condition_boundary_margin_scale', 0.0) or 0.0
				),
				'stage1_boundary_norm_scale': float(
					getattr(self.net, 'stage1_boundary_norm_scale', 0.0) or 0.0
				),
				'traj_branch_condition_names': list(
					getattr(self.net, 'traj_branch_condition_names', ())
				),
				'traj_window_condition_names': list(
					getattr(self.net, 'traj_window_condition_names', ('none', 'merge', 'junction', 'borrow'))
				),
				'traj_dir_condition_names': list(
					getattr(self.net, 'traj_dir_condition_names', ('none', 'same', 'opposite', 'cross'))
				),
				'traj_decision_phase_condition_names': traj_decision_phase_condition_names,
				'traj_phase_condition_names': traj_decision_phase_condition_names,
				'traj_control_phase_condition_names': list(
					getattr(self.net, 'traj_control_phase_condition_names', ('coast_yld', 'slow_yld', 'stop_yld', 'go'))
				),
				'traj_boundary_condition_names': list(
					getattr(self.net, 'traj_boundary_condition_names', ('yld_margin', 'go_margin'))
				),
				'traj_phase_energy_condition_names': list(
					getattr(self.net, 'traj_phase_energy_condition_names', ('phase_energy_yld', 'phase_energy_go'))
				),
			}
			target_speed_profile_list = None
			for semantic_key in [
				'traj_branch_condition_probs',
				'traj_window_condition_probs',
				'traj_dir_condition_probs',
				'traj_decision_phase_condition_probs',
				'traj_control_phase_condition_probs',
				'traj_boundary_margin_condition',
				'traj_phase_condition_probs',
				'traj_phase_energy_summary',
				'lane_dir_relation_probs',
			]:
				semantic_value = dp_pred_traj.get(semantic_key)
				if semantic_value is None:
					continue
				semantic_array = np.asarray(semantic_value).reshape(-1)
				if semantic_array.size > 0:
					self.last_branch_condition_debug[semantic_key] = semantic_array.astype(np.float32).tolist()
			if (
				'traj_phase_condition_probs' not in self.last_branch_condition_debug
				and 'traj_decision_phase_condition_probs' in self.last_branch_condition_debug
			):
				self.last_branch_condition_debug['traj_phase_condition_probs'] = (
					self.last_branch_condition_debug['traj_decision_phase_condition_probs']
				)
			traj_borrow_time_condition = dp_pred_traj.get('traj_borrow_time_condition')
			if traj_borrow_time_condition is not None:
				traj_borrow_time_condition = np.asarray(traj_borrow_time_condition).reshape(-1)
				if traj_borrow_time_condition.size > 0:
					self.last_branch_condition_debug['traj_borrow_time_condition'] = float(traj_borrow_time_condition[0])
			self.last_branch_condition_debug['prev_lane_dir_relation_probs'] = (
				self.prev_lane_dir_relation_probs.astype(np.float32).tolist()
			)
			self.last_branch_condition_debug['borrow_candidate_start_time_s'] = (
				None if self.borrow_candidate_start_time_s is None else float(self.borrow_candidate_start_time_s)
			)
			self.last_branch_condition_debug['borrow_latched'] = bool(self.borrow_latched)
			self.last_branch_condition_debug['borrow_observed_time_s'] = float(self.borrow_observed_time_s)
			for energy_key in ['energy_front', 'energy_left', 'energy_right',
								'energy_pedestrian', 'energy_offroad', 'energy_route']:
				energy_value = dp_pred_traj.get(energy_key)
				if energy_value is None:
					continue
				energy_array = np.asarray(energy_value).reshape(-1)
				if energy_array.size > 0:
					self.last_energy_debug[energy_key] = float(energy_array[0])
			speed_energy_samples = dp_pred_traj.get('speed_energy_samples')
			if speed_energy_samples is not None:
				speed_energy_samples = np.asarray(speed_energy_samples).reshape(-1)
				if speed_energy_samples.size > 0:
					self.last_energy_debug['speed_energy_samples'] = speed_energy_samples.astype(np.float32).tolist()
			speed_energy_query_center = dp_pred_traj.get('speed_energy_query_center')
			if speed_energy_query_center is not None:
				speed_energy_query_center = np.asarray(speed_energy_query_center).reshape(-1)
				if speed_energy_query_center.size > 0:
					self.last_energy_debug['speed_energy_query_center'] = float(speed_energy_query_center[0])
			speed_energy_ref_speeds = dp_pred_traj.get('speed_energy_ref_speeds')
			if speed_energy_ref_speeds is not None:
				speed_energy_ref_speeds = np.asarray(speed_energy_ref_speeds).reshape(-1)
				if speed_energy_ref_speeds.size > 0:
					self.last_energy_debug['speed_energy_ref_speeds'] = speed_energy_ref_speeds.astype(np.float32).tolist()
			if speed_energy_ref_speeds is None or speed_energy_ref_speeds.size == 0:
				ref_speed_values = []
				for speed_key in ['speed_head_speed', 'traj_speed_1s', 'traj_speed_05s']:
					speed_value = self.last_speed_debug.get(speed_key)
					if speed_value is None:
						ref_speed_values.append(np.nan)
					else:
						ref_speed_values.append(float(speed_value))
				speed_energy_ref_speeds = np.asarray(ref_speed_values, dtype=np.float32)
				self.last_energy_debug['speed_energy_ref_speeds'] = speed_energy_ref_speeds.tolist()
			for scalar_key in [
				'merge_active_prob',
				'junction_active_prob',
				'borrow_active_prob',
				'cross_active_prob',
				'merge_active_logits',
				'junction_active_logits',
				'borrow_active_logits',
				'cross_active_logits',
			]:
				scalar_value = dp_pred_traj.get(f'speed_energy_{scalar_key}')
				if scalar_value is None:
					continue
				scalar_array = np.asarray(scalar_value).reshape(-1)
				if scalar_array.size > 0:
					self.last_branch_condition_debug[f'speed_energy_{scalar_key}'] = float(scalar_array[0])
			for stage1_key in [
				'window_probs',
				'dir_probs',
				'decision_phase_probs',
				'control_phase_probs',
				'conflict_area_probs',
				'merge_yld_max_mps',
				'merge_go_min_mps',
				'junction_yld_max_mps',
				'junction_go_min_mps',
				'borrow_yld_max_mps',
				'borrow_go_min_mps',
				'selected_yld_max_mps',
				'selected_go_min_mps',
			]:
				for prefix in ['speed_energy', 'speed_energy_ref']:
					stage1_value = dp_pred_traj.get(f'{prefix}_{stage1_key}')
					if stage1_value is None:
						continue
					stage1_array = np.asarray(stage1_value).reshape(-1)
					if stage1_array.size == 0:
						continue
					out_key = f'{prefix}_{stage1_key}'
					if stage1_array.size == 1:
						self.last_branch_condition_debug[out_key] = float(stage1_array[0])
					else:
						self.last_branch_condition_debug[out_key] = stage1_array.astype(np.float32).tolist()
			for raw_curve_key in [
				'merge_yld',
				'merge_go',
				'junction_yld',
				'junction_go',
				'borrow_yld',
				'borrow_go',
				'cross_yld',
				'cross_go',
			]:
				raw_curve_value = dp_pred_traj.get(f'speed_energy_{raw_curve_key}')
				if raw_curve_value is None:
					continue
				raw_curve_array = np.asarray(raw_curve_value).reshape(-1)
				if raw_curve_array.size > 0:
					self.last_branch_condition_debug[f'speed_energy_{raw_curve_key}'] = raw_curve_array.astype(np.float32).tolist()
			speed_energy_curves = {}
			speed_energy_total_curve = None
			for speed_energy_key in ['chase', 'merge', 'cross', 'pedestrian']:
				speed_energy_value = dp_pred_traj.get(f'speed_energy_{speed_energy_key}')
				if speed_energy_value is None:
					continue
				speed_energy_array = np.asarray(speed_energy_value).reshape(-1)
				if speed_energy_array.size > 0:
					speed_energy_curves[speed_energy_key] = speed_energy_array.astype(np.float32)
					self.last_energy_debug[f'speed_energy_{speed_energy_key}'] = speed_energy_array.astype(np.float32).tolist()
			explicit_total_curve = dp_pred_traj.get('speed_energy_total')
			if explicit_total_curve is not None:
				explicit_total_curve = np.asarray(explicit_total_curve).reshape(-1)
				if explicit_total_curve.size > 0:
					speed_energy_total_curve = explicit_total_curve.astype(np.float32)
					self.last_energy_debug['speed_energy_total'] = speed_energy_total_curve.tolist()
			if len(speed_energy_curves) > 0:
				total_curve = None
				for curve in speed_energy_curves.values():
					total_curve = curve.copy() if total_curve is None else np.maximum(total_curve, curve)
				if total_curve is not None:
					if speed_energy_total_curve is None or speed_energy_total_curve.size != total_curve.size:
						speed_energy_total_curve = total_curve.astype(np.float32)
						self.last_energy_debug['speed_energy_total'] = total_curve.astype(np.float32).tolist()
					self.last_energy_debug['speed_energy_max'] = float(np.max(total_curve))
					self.last_energy_debug['speed_energy_argmax'] = int(np.argmax(total_curve))
					if speed_energy_samples is not None and speed_energy_samples.size == total_curve.size:
						samples_np = speed_energy_samples.astype(np.float32)
						global_min_idx = int(np.argmin(total_curve))
						self.last_energy_debug['speed_energy_global_min_index'] = global_min_idx
						self.last_energy_debug['speed_energy_global_min_speed'] = float(samples_np[global_min_idx])
						self.last_energy_debug['speed_energy_global_min_value'] = float(total_curve[global_min_idx])

						local_min_indices = []
						for idx in range(total_curve.size):
							left_ok = idx == 0 or total_curve[idx] <= total_curve[idx - 1]
							right_ok = idx == total_curve.size - 1 or total_curve[idx] <= total_curve[idx + 1]
							if left_ok and right_ok:
								local_min_indices.append(idx)
						self.last_energy_debug['speed_energy_local_min_indices'] = [int(i) for i in local_min_indices]
						self.last_energy_debug['speed_energy_local_min_speeds'] = [float(samples_np[i]) for i in local_min_indices]
						self.last_energy_debug['speed_energy_local_min_values'] = [float(total_curve[i]) for i in local_min_indices]

						if len(local_min_indices) > 0:
							center_speed_value = None
							if speed_energy_query_center is not None and speed_energy_query_center.size > 0:
								center_speed_value = float(speed_energy_query_center[0])
							if center_speed_value is None:
								center_speed_value = float(samples_np[global_min_idx])
							local_choice_idx = min(
								local_min_indices,
								key=lambda i: (abs(float(samples_np[i]) - center_speed_value), float(total_curve[i]))
							)
						else:
							local_choice_idx = global_min_idx
						self.last_energy_debug['speed_energy_local_choice_index'] = int(local_choice_idx)
						self.last_energy_debug['speed_energy_local_choice_speed'] = float(samples_np[local_choice_idx])
						self.last_energy_debug['speed_energy_local_choice_value'] = float(total_curve[local_choice_idx])
			speed_energy_ref_curves = {}
			for speed_energy_key in ['chase', 'merge', 'cross', 'pedestrian']:
				speed_energy_value = dp_pred_traj.get(f'speed_energy_ref_{speed_energy_key}')
				if speed_energy_value is None:
					continue
				speed_energy_array = np.asarray(speed_energy_value).reshape(-1)
				if speed_energy_array.size > 0:
					speed_energy_ref_curves[speed_energy_key] = speed_energy_array.astype(np.float32)
					self.last_energy_debug[f'speed_energy_ref_{speed_energy_key}'] = speed_energy_array.astype(np.float32).tolist()
			explicit_ref_total = dp_pred_traj.get('speed_energy_ref_total')
			if explicit_ref_total is not None:
				explicit_ref_total = np.asarray(explicit_ref_total).reshape(-1)
				if explicit_ref_total.size > 0:
					self.last_energy_debug['speed_energy_ref_total'] = explicit_ref_total.astype(np.float32).tolist()
			if len(speed_energy_ref_curves) > 0:
				total_curve = None
				for curve in speed_energy_ref_curves.values():
					total_curve = curve.copy() if total_curve is None else np.maximum(total_curve, curve)
				if total_curve is not None:
					if 'speed_energy_ref_total' not in self.last_energy_debug:
						self.last_energy_debug['speed_energy_ref_total'] = total_curve.astype(np.float32).tolist()
			for ref_curve_key in [
				'merge_yld',
				'merge_go',
				'junction_yld',
				'junction_go',
				'borrow_yld',
				'borrow_go',
				'cross_yld',
				'cross_go',
			]:
				ref_curve_value = dp_pred_traj.get(f'speed_energy_ref_{ref_curve_key}')
				if ref_curve_value is None:
					continue
				ref_curve_array = np.asarray(ref_curve_value).reshape(-1)
				if ref_curve_array.size > 0:
					self.last_branch_condition_debug[f'speed_energy_ref_{ref_curve_key}'] = (
						ref_curve_array.astype(np.float32).tolist()
					)
			if (
				len(speed_energy_ref_curves) == 0
				and speed_energy_ref_speeds is not None
				and speed_energy_ref_speeds.size > 0
				and speed_energy_samples is not None
				and speed_energy_samples.size > 0
				and speed_energy_total_curve is not None
				and speed_energy_samples.size == speed_energy_total_curve.size
			):
				samples_np = speed_energy_samples.astype(np.float32)
				for speed_energy_key, curve in speed_energy_curves.items():
					ref_lookup = []
					for ref_speed in speed_energy_ref_speeds.astype(np.float32):
						if not np.isfinite(ref_speed):
							ref_lookup.append(np.nan)
							continue
						ref_idx = int(np.argmin(np.abs(samples_np - float(ref_speed))))
						ref_lookup.append(float(curve[ref_idx]))
					self.last_energy_debug[f'speed_energy_ref_{speed_energy_key}'] = ref_lookup
				ref_total_lookup = []
				for ref_speed in speed_energy_ref_speeds.astype(np.float32):
					if not np.isfinite(ref_speed):
						ref_total_lookup.append(np.nan)
						continue
					ref_idx = int(np.argmin(np.abs(samples_np - float(ref_speed))))
					ref_total_lookup.append(float(speed_energy_total_curve[ref_idx]))
				self.last_energy_debug['speed_energy_ref_total'] = ref_total_lookup
			front_route_risk_score = self._compute_front_route_risk_debug(
				dp_pred_traj=dp_pred_traj,
				transfuser_bev_feature=transfuser_bev_feature,
				transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
				ego_status_stacked=ego_status_stacked,
				transfuser_lidar_bev_detail=transfuser_lidar_bev_detail,
			)
			if front_route_risk_score is not None:
				front_route_risk_score = float(front_route_risk_score)
				front_route_risk_sigmoid = float(1.0 / (1.0 + np.exp(-front_route_risk_score)))
				if self.front_route_risk_ema is None:
					self.front_route_risk_ema = front_route_risk_score
				else:
					self.front_route_risk_ema = 0.85 * self.front_route_risk_ema + 0.15 * front_route_risk_score
				self.last_energy_debug['front_route_risk_score'] = front_route_risk_score
				self.last_energy_debug['front_route_risk_sigmoid'] = front_route_risk_sigmoid
				self.last_energy_debug['front_route_risk_ema'] = float(self.front_route_risk_ema)
			elif 'energy_front' in self.last_energy_debug and self.use_front_route_risk_energy:
				front_route_risk_score = float(self.last_energy_debug['energy_front'])
				front_route_risk_sigmoid = float(1.0 / (1.0 + np.exp(-front_route_risk_score)))
				if self.front_route_risk_ema is None:
					self.front_route_risk_ema = front_route_risk_score
				else:
					self.front_route_risk_ema = 0.85 * self.front_route_risk_ema + 0.15 * front_route_risk_score
				self.last_energy_debug['front_route_risk_score'] = front_route_risk_score
				self.last_energy_debug['front_route_risk_sigmoid'] = front_route_risk_sigmoid
				self.last_energy_debug['front_route_risk_ema'] = float(self.front_route_risk_ema)
			target_speed_profile = dp_pred_traj.get('target_speed_profile')
			if target_speed_profile is not None:
				target_speed_profile = np.asarray(target_speed_profile).reshape(-1)
				if target_speed_profile.size > 0:
					target_speed_profile_list = target_speed_profile.astype(np.float32).tolist()
			
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
			
			terminal_route_debug = self._get_terminal_route_speed_cap(tick_data)
			self.last_terminal_route_debug = terminal_route_debug
			junction_window_soft_cap_debug = self._get_junction_window_soft_speed_cap()
			self.last_junction_window_soft_cap_debug = junction_window_soft_cap_debug
			front_route_risk_cap_debug = self._get_front_route_risk_speed_cap(
				tick_data,
				gt_velocity,
			)
			self.last_front_route_risk_cap_debug = front_route_risk_cap_debug
			stage1_energy_cap_debug = {}
			if STAGE1_ENERGY_SPEED_CAP_ENABLE:
				stage1_energy_cap_debug = self._get_stage1_energy_speed_cap(
					tick_data,
					gt_velocity,
				)
			self.last_stage1_energy_cap_debug = stage1_energy_cap_debug

			# Use target_point to truncate route for lateral control
			steer, throttle, brake = self.control_pid(
				route_waypoints,
				gt_velocity,
				speed_waypoints,
				target_point=target_point,
				junction_window_soft_cap_ms=(
					junction_window_soft_cap_debug.get('speed_cap_ms')
					if junction_window_soft_cap_debug.get('active')
					else None
				),
				terminal_speed_cap_ms=terminal_route_debug.get('speed_cap_ms'),
				front_route_risk_cap_ms=front_route_risk_cap_debug.get('speed_cap_ms'),
				stage1_energy_cap_ms=stage1_energy_cap_debug.get('speed_cap_ms') if STAGE1_ENERGY_SPEED_CAP_ENABLE else None,
				stage1_energy_speed_adjust_ms=stage1_energy_cap_debug.get('speed_adjust_ms') if STAGE1_ENERGY_SPEED_CAP_ENABLE else None,
			)
			if target_speed_profile_list is not None:
				self.last_speed_debug['target_speed_profile'] = target_speed_profile_list
			throttle, brake, semantic_debug = self._apply_semantic_hazard_postprocess(
				ego_speed=gt_velocity,
				current_heading=float(tick_data.get('compass', 0.0)),
				desired_speed_capped=self.last_speed_debug.get('desired_speed_capped'),
				throttle=throttle,
				brake=brake,
				bev_classes=bev_semantic_classes,
				external_force_move_block_reason=(
					terminal_route_debug.get('reason') if terminal_route_debug.get('active') else None
				),
				)
			stuck_helper_control_override = tick_data.get('stuck_helper_control_override')
			if stuck_helper_control_override is not None:
				steer = float(stuck_helper_control_override.get('steer', steer))
				throttle = float(stuck_helper_control_override.get('throttle', throttle))
				brake = float(stuck_helper_control_override.get('brake', 0.0))

			if stuck_helper_control_override is not None:
				applied_steer = float(np.clip(
					stuck_helper_control_override.get('steer', steer),
					-1.0,
					1.0,
				))
			else:
				applied_steer = float(np.clip(STEER_SIGN_SCALE * steer, -1.0, 1.0))
			control = carla.VehicleControl()
			control.steer = applied_steer
			control.throttle = float(throttle)
			control.brake = float(brake)
			vehicle = CarlaDataProvider.get_hero_actor()
			
			# Optional hard speed limit: force brake only if explicitly enabled.
			if HARD_SPEED_LIMIT_MS > 0.0 and gt_velocity > HARD_SPEED_LIMIT_MS:
				control.throttle = 0.0
				control.brake = 1.0
			
			# Store metadata
			self.pid_metadata = {
				'agent': 'mot',
				'steer': control.steer,
				'steer_controller': float(steer),
				'steer_sign_scale': STEER_SIGN_SCALE,
				'stuck_helper_recovery_mode': tick_data.get('stuck_helper_recovery_mode'),
				'stuck_helper_control_override_active': bool(stuck_helper_control_override is not None),
				'stuck_helper_control_steer': (
					float(stuck_helper_control_override.get('steer'))
					if stuck_helper_control_override is not None
					else None
				),
				'stuck_helper_control_throttle': (
					float(stuck_helper_control_override.get('throttle'))
					if stuck_helper_control_override is not None
					else None
				),
				'stuck_helper_control_brake': (
					float(stuck_helper_control_override.get('brake'))
					if stuck_helper_control_override is not None
					else None
				),
				'throttle': control.throttle,
				'brake': control.brake,
				'speed': gt_velocity,
				'command': command,
				'command_value': int(command_value),
				'command_text': command_text,
				'speed_source': self.last_speed_debug.get('speed_source'),
				'speed_head_speed': self.last_speed_debug.get('speed_head_speed'),
				'traj_speed_1s': self.last_speed_debug.get('traj_speed_1s'),
				'traj_speed_05s': self.last_speed_debug.get('traj_speed_05s'),
				'desired_speed_raw': self.last_speed_debug.get('desired_speed_raw'),
				'desired_speed_capped': self.last_speed_debug.get('desired_speed_capped'),
				'soft_speed_limit_ms': self.last_speed_debug.get('soft_speed_limit_ms'),
				'junction_window_soft_cap_active': bool(junction_window_soft_cap_debug.get('active', False)),
				'junction_window_soft_cap_reason': junction_window_soft_cap_debug.get('reason'),
				'junction_window_soft_cap_ms': self.last_speed_debug.get('junction_window_soft_cap_ms'),
				'junction_window_soft_cap_applied': bool(self.last_speed_debug.get('junction_window_soft_cap_applied', False)),
				'junction_window_soft_cap_junction_prob': junction_window_soft_cap_debug.get('junction_prob'),
				'junction_window_soft_cap_streak_frames': int(junction_window_soft_cap_debug.get('streak_frames', 0)),
				'junction_window_soft_cap_source': junction_window_soft_cap_debug.get('source'),
				'terminal_speed_cap_ms': self.last_speed_debug.get('terminal_speed_cap_ms'),
				'hard_speed_limit_ms': self.last_speed_debug.get('hard_speed_limit_ms'),
				'terminal_route_speed_cap_active': bool(terminal_route_debug.get('active', False)),
				'terminal_route_speed_cap_reason': terminal_route_debug.get('reason'),
				'terminal_route_remaining_points': terminal_route_debug.get('remaining_route_points'),
				'terminal_route_target_distance_m': terminal_route_debug.get('target_distance_m'),
				'terminal_route_target_forward_m': terminal_route_debug.get('target_forward_m'),
				'front_route_risk_speed_cap_active': bool(front_route_risk_cap_debug.get('active', False)),
				'front_route_risk_speed_cap_reason': front_route_risk_cap_debug.get('reason'),
				'front_route_risk_speed_cap_ms': front_route_risk_cap_debug.get('speed_cap_ms'),
				'front_route_risk_speed_cap_applied': bool(self.last_speed_debug.get('front_route_risk_cap_applied', False)),
				'front_route_risk_raw_score': front_route_risk_cap_debug.get('raw_score'),
				'front_route_risk_recent_max_sigmoid': front_route_risk_cap_debug.get('recent_max_sigmoid'),
				'front_route_risk_gate_active': bool(front_route_risk_cap_debug.get('gate_active', False)),
				'front_route_risk_geometry_active': bool(front_route_risk_cap_debug.get('geometry_active', False)),
				'front_route_risk_geometry_max_angle_deg': front_route_risk_cap_debug.get('geometry_max_angle_deg'),
				'front_route_risk_geometry_lateral_mag_m': front_route_risk_cap_debug.get('geometry_lateral_mag_m'),
				'front_route_risk_speed_bias_ratio': front_route_risk_cap_debug.get('speed_bias_ratio'),
				'front_route_risk_effective_med_sigmoid_threshold': front_route_risk_cap_debug.get('effective_med_sigmoid_threshold'),
				'front_route_risk_effective_high_sigmoid_threshold': front_route_risk_cap_debug.get('effective_high_sigmoid_threshold'),
				'front_route_risk_effective_high_ema_threshold': front_route_risk_cap_debug.get('effective_high_ema_threshold'),
				'front_route_risk_cap_hold_frames_remaining': int(front_route_risk_cap_debug.get('hold_frames_remaining', 0)),
				'stage1_energy_control_mode': stage1_energy_cap_debug.get('control_mode'),
				'stage1_energy_speed_control_active': bool(stage1_energy_cap_debug.get('active', False)),
				'stage1_energy_speed_control_reason': stage1_energy_cap_debug.get('reason'),
				'stage1_energy_speed_cap_active': bool(
					stage1_energy_cap_debug.get('active', False)
					and stage1_energy_cap_debug.get('control_mode') == 'cap'
				),
				'stage1_energy_speed_cap_reason': stage1_energy_cap_debug.get('reason') if stage1_energy_cap_debug.get('control_mode') == 'cap' else None,
				'stage1_energy_speed_cap_ms': stage1_energy_cap_debug.get('speed_cap_ms'),
				'stage1_energy_speed_cap_applied': bool(self.last_speed_debug.get('stage1_energy_cap_applied', False)),
				'stage1_energy_speed_adjust_ms': self.last_speed_debug.get('stage1_energy_speed_adjust_ms'),
				'stage1_energy_speed_adjust_applied': bool(self.last_speed_debug.get('stage1_energy_speed_adjust_applied', False)),
				'stage1_energy_current_score': stage1_energy_cap_debug.get('current_score'),
				'stage1_energy_target_score': stage1_energy_cap_debug.get('target_score'),
				'stage1_energy_local_peak_score': stage1_energy_cap_debug.get('local_peak_score'),
				'stage1_energy_current_index': stage1_energy_cap_debug.get('current_index'),
				'stage1_energy_target_index': stage1_energy_cap_debug.get('target_index'),
				'stage1_energy_query_center_ms': stage1_energy_cap_debug.get('query_center_ms'),
				'stage1_energy_target_speed_ms': stage1_energy_cap_debug.get('target_speed_ms'),
				'stage1_energy_target_source': stage1_energy_cap_debug.get('target_source'),
				'stage1_energy_safe_threshold': stage1_energy_cap_debug.get('safe_threshold'),
				'stage1_energy_gradient_dedv': stage1_energy_cap_debug.get('gradient_dedv'),
				'stage1_energy_gradient_gate_dedv': stage1_energy_cap_debug.get('gradient_gate_dedv'),
				'stage1_energy_gradient_dominant_component': stage1_energy_cap_debug.get('gradient_dominant_component'),
				'stage1_energy_gradient_raw_adjust_ms': stage1_energy_cap_debug.get('gradient_raw_adjust_ms'),
				'stage1_energy_gradient_score_gate': bool(stage1_energy_cap_debug.get('gradient_score_gate', False)),
				'stage1_energy_gradient_slope_gate': bool(stage1_energy_cap_debug.get('gradient_slope_gate', False)),
				'stage1_energy_gradient_chase_dedv': stage1_energy_cap_debug.get('gradient_chase_dedv'),
				'stage1_energy_gradient_merge_dedv': stage1_energy_cap_debug.get('gradient_merge_dedv'),
				'stage1_energy_gradient_cross_dedv': stage1_energy_cap_debug.get('gradient_cross_dedv'),
				'stage1_energy_gradient_pedestrian_dedv': stage1_energy_cap_debug.get('gradient_pedestrian_dedv'),
				'stage1_energy_gradient_chase_score': stage1_energy_cap_debug.get('gradient_chase_score'),
				'stage1_energy_gradient_merge_score': stage1_energy_cap_debug.get('gradient_merge_score'),
				'stage1_energy_gradient_cross_score': stage1_energy_cap_debug.get('gradient_cross_score'),
				'stage1_energy_gradient_pedestrian_score': stage1_energy_cap_debug.get('gradient_pedestrian_score'),
				'stage1_energy_gradient_chase_active': bool(stage1_energy_cap_debug.get('gradient_chase_active', False)),
				'stage1_energy_gradient_merge_active': bool(stage1_energy_cap_debug.get('gradient_merge_active', False)),
				'stage1_energy_gradient_cross_active': bool(stage1_energy_cap_debug.get('gradient_cross_active', False)),
				'stage1_energy_gradient_pedestrian_active': bool(stage1_energy_cap_debug.get('gradient_pedestrian_active', False)),
				'stage1_energy_cap_hold_frames_remaining': int(stage1_energy_cap_debug.get('hold_frames_remaining', 0)),
				'stuck_detector': int(self.stuck_detector),
				'stuck_helper': int(self.stuck_helper),
				'stuck_helper_active_state': bool(self.stuck_helper_active),
				'stuck_helper_mode': self.stuck_helper_mode,
				'stuck_helper_heading_delta_deg': float(self.stuck_helper_heading_delta_deg),
				'stuck_helper_release_heading_deg': float(STUCK_HELPER_RELEASE_HEADING_DEG),
				'stuck_helper_in_startup_zone': bool(self.stuck_helper_in_startup_zone),
				'stuck_helper_distance_from_start_m': float(self.stuck_helper_distance_from_start_m),
				'stuck_helper_startup_distance_m': float(self.stuck_helper_startup_distance_m),
				'stuck_helper_startup_threshold': int(self.stuck_helper_startup_threshold),
				'stuck_helper_poststart_threshold': int(self.stuck_helper_poststart_threshold),
				'stuck_helper_startup_reference_xy': (
					self.stuck_helper_startup_reference_xy.tolist()
					if isinstance(self.stuck_helper_startup_reference_xy, np.ndarray)
					else self.stuck_helper_startup_reference_xy
				),
				'force_move': int(self.force_move),
				'traffic_light_semantic_state': semantic_debug.get('traffic_light_state'),
				'traffic_light_semantic_block_force_move': bool(semantic_debug.get('traffic_light_block_force_move', False)),
				'traffic_light_semantic_apply_stop': bool(semantic_debug.get('traffic_light_apply_stop', False)),
				'traffic_light_semantic_closest_forward_m': semantic_debug.get('traffic_light_closest_forward_m'),
				'traffic_light_semantic_green_hold_active': bool(semantic_debug.get('traffic_light_green_hold_active', False)),
				'traffic_light_semantic_green_hold_frames_remaining': int(semantic_debug.get('traffic_light_green_hold_frames_remaining', 0)),
				'traffic_light_semantic_red_pixels': int(semantic_debug.get('traffic_light_red_pixels', 0)),
				'traffic_light_semantic_yellow_pixels': int(semantic_debug.get('traffic_light_yellow_pixels', 0)),
				'traffic_light_semantic_green_pixels': int(semantic_debug.get('traffic_light_green_pixels', 0)),
				'traffic_light_semantic_red_closest_forward_m': semantic_debug.get('traffic_light_red_closest_forward_m'),
				'traffic_light_semantic_yellow_closest_forward_m': semantic_debug.get('traffic_light_yellow_closest_forward_m'),
				'traffic_light_semantic_green_closest_forward_m': semantic_debug.get('traffic_light_green_closest_forward_m'),
				'stop_sign_semantic_state': semantic_debug.get('stop_sign_state'),
				'stop_sign_semantic_hold_force_move': bool(semantic_debug.get('stop_sign_hold_force_move', False)),
				'stop_sign_semantic_apply_stop': bool(semantic_debug.get('stop_sign_apply_stop', False)),
				'stop_sign_semantic_pixels': int(semantic_debug.get('stop_sign_pixels', 0)),
				'stop_sign_semantic_closest_forward_m': semantic_debug.get('stop_sign_closest_forward_m'),
				'stop_sign_semantic_mean_lateral_m': semantic_debug.get('stop_sign_mean_lateral_m'),
				'stop_sign_semantic_stopped_frames': int(semantic_debug.get('stop_sign_stopped_frames', 0)),
				'stop_sign_semantic_disabled': bool(semantic_debug.get('stop_sign_disabled', False)),
				'stop_sign_semantic_apply_events': int(semantic_debug.get('stop_sign_apply_events', 0)),
				'stop_sign_semantic_max_apply_events': int(semantic_debug.get('stop_sign_max_apply_events', 0)),
				'stop_sign_semantic_arm_speed_ok': bool(semantic_debug.get('stop_sign_arm_speed_ok', False)),
				'stuck_helper_active': bool(semantic_debug.get('stuck_helper_active', False)),
				'stuck_helper_mode_tick': semantic_debug.get('stuck_helper_mode'),
				'stuck_helper_frames_remaining': int(semantic_debug.get('stuck_helper_frames_remaining', 0)),
				'stuck_helper_in_startup_zone_tick': bool(semantic_debug.get('stuck_helper_in_startup_zone', False)),
				'stuck_helper_distance_from_start_m_tick': semantic_debug.get('stuck_helper_distance_from_start_m'),
				'force_move_blocked_reason': semantic_debug.get('force_move_blocked_reason'),
				'planner_wants_stop_for_force_move': bool(semantic_debug.get('planner_wants_stop', False)),
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
				'target_point_prepromote_ego': tick_data['target_point_prepromote_ego'].tolist() if isinstance(tick_data.get('target_point_prepromote_ego'), np.ndarray) else tick_data.get('target_point_prepromote_ego'),
				'next_target_point_prepromote_ego': tick_data['next_target_point_prepromote_ego'].tolist() if isinstance(tick_data.get('next_target_point_prepromote_ego'), np.ndarray) else tick_data.get('next_target_point_prepromote_ego'),
				'target_point_prepromote_world': tick_data['target_point_prepromote_world'].tolist() if isinstance(tick_data.get('target_point_prepromote_world'), np.ndarray) else tick_data.get('target_point_prepromote_world'),
				'next_target_point_prepromote_world': tick_data['next_target_point_prepromote_world'].tolist() if isinstance(tick_data.get('next_target_point_prepromote_world'), np.ndarray) else tick_data.get('next_target_point_prepromote_world'),
				'early_target_promote_active': bool(tick_data.get('early_target_promote_active', False)),
				'early_target_promote_reason': tick_data.get('early_target_promote_reason'),
				'target_point_raw_ego': tick_data['target_point_raw_ego'].tolist() if isinstance(tick_data.get('target_point_raw_ego'), np.ndarray) else tick_data.get('target_point_raw_ego'),
				'next_target_point_raw_ego': tick_data['next_target_point_raw_ego'].tolist() if isinstance(tick_data.get('next_target_point_raw_ego'), np.ndarray) else tick_data.get('next_target_point_raw_ego'),
				'target_point_raw_world': tick_data['target_point_raw_world'].tolist() if isinstance(tick_data.get('target_point_raw_world'), np.ndarray) else tick_data.get('target_point_raw_world'),
				'next_target_point_raw_world': tick_data['next_target_point_raw_world'].tolist() if isinstance(tick_data.get('next_target_point_raw_world'), np.ndarray) else tick_data.get('next_target_point_raw_world'),
				'tick_stuck_helper_active': bool(tick_data.get('stuck_helper_active', False)),
				'tick_stuck_helper_mode': tick_data.get('stuck_helper_mode'),
				'tick_stuck_helper_frames_remaining': int(tick_data.get('stuck_helper_frames_remaining', 0)),
				'tick_stuck_helper_heading_delta_deg': float(tick_data.get('stuck_helper_heading_delta_deg', 0.0)),
				'tick_stuck_helper_release_heading_deg': float(tick_data.get('stuck_helper_release_heading_deg', 0.0)),
				'tick_stuck_helper_threshold': int(tick_data.get('stuck_helper_threshold', 0)),
				'tick_stuck_helper_in_startup_zone': bool(tick_data.get('stuck_helper_in_startup_zone', False)),
				'tick_stuck_helper_distance_from_start_m': float(tick_data.get('stuck_helper_distance_from_start_m', 0.0)),
				'tick_stuck_helper_startup_distance_m': float(tick_data.get('stuck_helper_startup_distance_m', 0.0)),
				'tick_stuck_helper_startup_threshold': int(tick_data.get('stuck_helper_startup_threshold', 0)),
				'tick_stuck_helper_poststart_threshold': int(tick_data.get('stuck_helper_poststart_threshold', 0)),
				'tick_stuck_helper_startup_reference_xy': tick_data.get('stuck_helper_startup_reference_xy'),
				'tick_stuck_helper_target_points_ego': tick_data.get('stuck_helper_target_points_ego'),
				'tick_stuck_helper_target_points_world': tick_data.get('stuck_helper_target_points_world'),
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
			for energy_key, energy_value in self.last_energy_debug.items():
				self.pid_metadata[energy_key] = self._jsonify_debug_value(energy_value)
			for branch_key, branch_value in self.last_branch_condition_debug.items():
				self.pid_metadata[branch_key] = self._jsonify_debug_value(branch_value)
			for semantic_key, semantic_value in self.last_semantic_debug.items():
				self.pid_metadata[semantic_key] = self._jsonify_debug_value(semantic_value)
			for input_key, input_value in self.last_model_input_debug.items():
				self.pid_metadata[input_key] = self._jsonify_debug_value(input_value)
			for speed_key, speed_value in self.last_speed_debug.items():
				self.pid_metadata[speed_key] = self._jsonify_debug_value(speed_value)
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

		if SAVE_TRANSFUSER_BEV_DEBUG and 'transfuser_lidar_bev' in tick_data:
			transfuser_lidar_bev_img = self._render_transfuser_lidar_bev_image(
				tick_data['transfuser_lidar_bev']
			)
			if transfuser_lidar_bev_img is not None:
				imageio.imwrite(
					str(self.save_path / 'transfuser_lidar_bev' / (f'{frame:04d}.png')),
					transfuser_lidar_bev_img
				)

		if SAVE_TRANSFUSER_BEV_DEBUG and 'bev_semantic_classes' in tick_data:
			semantic_vis, vehicle_vis = self._render_bev_semantic_image(
				tick_data['bev_semantic_classes']
			)
			if semantic_vis is not None:
				imageio.imwrite(
					str(self.save_path / 'transfuser_bev_semantic' / (f'{frame:04d}.png')),
					semantic_vis
				)
			if vehicle_vis is not None:
				imageio.imwrite(
					str(self.save_path / 'transfuser_bev_vehicle' / (f'{frame:04d}.png')),
					vehicle_vis
				)

		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

		# metric info
		outfile = open(self.save_path / 'metric_info.json', 'w')
		json.dump(self.metric_info, outfile, indent=4)
		outfile.close()

	def _compose_debug_visualization(self, rgb_img, bev_img):
		left_img = cv2.resize(rgb_img, (800, 600))
		right_img = cv2.resize(bev_img, (800, 600))

		speed_mps = float(self.pid_metadata.get('speed', 0.0))
		command_value = self.pid_metadata.get('command_value', 'N/A')
		command_text = self.pid_metadata.get('command_text', 'unknown')
		cond_enabled = int(bool(self.pid_metadata.get('traj_branch_condition_enabled', False)))
		cond_scale = self._format_debug_value(self.pid_metadata.get('traj_branch_condition_scale'), '.1f')
		cond_detach = int(bool(self.pid_metadata.get('traj_branch_condition_detach', False)))
		cond_boundary_scale = self._format_debug_value(
			self.pid_metadata.get('traj_branch_condition_boundary_margin_scale'), '.1f'
		)
		cond_borrow_scale = self._format_debug_value(
			self.pid_metadata.get('traj_branch_condition_borrow_time_scale'), '.1f'
		)

		left_status_lines = [
			f"frm: {self.step}",
			f"v: {speed_mps:.2f} m/s",
			f"src: {self.pid_metadata.get('speed_source', 'N/A')}",
			f"fus: {self.pid_metadata.get('fusion_regime', 'N/A')}",
			f"vd0(raw): {self._format_debug_value(self.pid_metadata.get('desired_speed_raw'), '.2f')}",
			f"vd(cap): {self._format_debug_value(self.pid_metadata.get('desired_speed_capped'), '.2f')}",
			f"Estg: {self.pid_metadata.get('stage1_energy_control_mode', 'off')}/{int(bool(self.pid_metadata.get('stage1_energy_speed_control_active', False)))}",
			f"dV(E): {self._format_debug_value(self.pid_metadata.get('stage1_energy_speed_adjust_ms'), '.2f')}/{self._format_debug_value(self.pid_metadata.get('stage1_energy_speed_cap_ms'), '.1f')}",
			f"Et c/t/p: {self._format_debug_value(self.pid_metadata.get('stage1_energy_current_score'), '.1f')}/{self._format_debug_value(self.pid_metadata.get('stage1_energy_target_score'), '.1f')}/{self._format_debug_value(self.pid_metadata.get('stage1_energy_local_peak_score'), '.1f')}",
			f"v_tgt: {self._format_debug_value(self.pid_metadata.get('stage1_energy_target_speed_ms'), '.1f')} ({self.pid_metadata.get('stage1_energy_target_source', 'NA')})",
			f"vY/G(q): {self._format_debug_first(self.pid_metadata.get('speed_energy_selected_yld_max_mps'), '.1f')}/{self._format_debug_first(self.pid_metadata.get('speed_energy_selected_go_min_mps'), '.1f')}",
			f"vY@ref: {self._format_debug_curve(self.pid_metadata.get('speed_energy_ref_selected_yld_max_mps'), fmt='.1f', max_items=3)}",
			f"vG@ref: {self._format_debug_curve(self.pid_metadata.get('speed_energy_ref_selected_go_min_mps'), fmt='.1f', max_items=3)}",
			f"E*: {self._format_energy_peak_with_speed()}",
			f"steer: {float(self.pid_metadata.get('steer', 0.0)):.3f}",
			f"thr: {float(self.pid_metadata.get('throttle', 0.0)):.2f}",
			f"brk: {float(self.pid_metadata.get('brake', 0.0)):.2f}",
		]

		mid_status_lines = [
			f"cmd: {command_text} ({command_value})",
			f"pose: {self.pid_metadata.get('target_pose_source', 'N/A')}",
			f"ang: {float(self.pid_metadata.get('target_angle_deg', 0.0)):.1f}",
			f"tp_L/R/B: {int(bool(self.pid_metadata.get('target_is_left', False)))}/{int(bool(self.pid_metadata.get('target_is_right', False)))}/{int(bool(self.pid_metadata.get('target_is_behind', False)))}",
			f"herr: {float(self.pid_metadata.get('steer_heading_error_deg', 0.0)):.1f}",
			f"sidx: {self.pid_metadata.get('steer_target_idx', 'N/A')}",
			f"sctl: {float(self.pid_metadata.get('steer_controller', 0.0)):.2f}",
			f"cond cfg: {cond_enabled} w+d+dp+cp+bm",
			f"cond sc/dt: {cond_scale}/{cond_detach}",
			f"bm/bt sc: {cond_boundary_scale}/{cond_borrow_scale}",
			f"sW[nmjb]: {self._format_debug_curve(self.pid_metadata.get('speed_energy_window_probs'), fmt='.2f', max_items=4)}",
			f"sP[y/g]: {self._format_debug_curve(self.pid_metadata.get('speed_energy_decision_phase_probs'), fmt='.2f', max_items=2)}",
			f"sC[c/s/t/g]: {self._format_debug_curve(self.pid_metadata.get('speed_energy_control_phase_probs'), fmt='.2f', max_items=4)}",
			f"lidar: {int(bool(self.pid_metadata.get('use_lidar_bev_detail', False)))}/{int(bool(self.pid_metadata.get('lidar_bev_detail_zero_fallback', False)))}",
		]

		if self.last_target_point is not None:
			mid_status_lines.append(
				f"tp: [{self.last_target_point[0]:.2f}, {self.last_target_point[1]:.2f}]"
			)
		if self.last_next_target_point is not None:
			mid_status_lines.append(
				f"ntp: [{self.last_next_target_point[0]:.2f}, {self.last_next_target_point[1]:.2f}]"
			)

		right_status_lines = [
			f"pred h/1/.5: {self._format_debug_value(self.pid_metadata.get('speed_head_speed'), '.1f')}/{self._format_debug_value(self.pid_metadata.get('traj_speed_1s'), '.1f')}/{self._format_debug_value(self.pid_metadata.get('traj_speed_05s'), '.1f')} m/s",
			f"vs m/s: {self._format_debug_curve(self.pid_metadata.get('speed_energy_samples'), fmt='.1f', max_items=7)}",
			f"sD[n/s/o/c]: {self._format_debug_curve(self.pid_metadata.get('speed_energy_dir_probs'), fmt='.2f', max_items=4)}",
			f"sA: {self._format_debug_curve(self.pid_metadata.get('speed_energy_conflict_area_probs'), fmt='.2f', max_items=7)}",
			f"vY*: {self._format_debug_curve(self.pid_metadata.get('speed_energy_selected_yld_max_mps'), fmt='.1f', max_items=7)}",
			f"vG*: {self._format_debug_curve(self.pid_metadata.get('speed_energy_selected_go_min_mps'), fmt='.1f', max_items=7)}",
			f"bcW[nmjb]: {self._format_debug_curve(self.pid_metadata.get('traj_window_condition_probs'), fmt='.2f', max_items=4)}",
			f"bcD[n/s/o/c]: {self._format_debug_curve(self.pid_metadata.get('traj_dir_condition_probs'), fmt='.2f', max_items=4)}",
			f"bcP[y/g]/bt: {self._format_debug_curve(self.pid_metadata.get('traj_decision_phase_condition_probs'), fmt='.2f', max_items=2)}/{self._format_debug_value(self.pid_metadata.get('traj_borrow_time_condition'), '.2f')}",
			f"bcC[c/s/t/g]: {self._format_debug_curve(self.pid_metadata.get('traj_control_phase_condition_probs'), fmt='.2f', max_items=4)}",
			f"bm[y/g]/lr: {self._format_debug_curve(self.pid_metadata.get('traj_boundary_margin_condition'), fmt='.2f', max_items=2)}/{self._format_debug_curve(self.pid_metadata.get('lane_dir_relation_probs'), fmt='.2f', max_items=2)}",
		]

		line_gap = 18
		font_scale = 0.39
		col1_x = 26
		col2_x = 220
		col3_x = 415
		panel_top = 290
		start_y = panel_top + 28
		max_line_count = max(len(left_status_lines), len(mid_status_lines), len(right_status_lines))
		panel_bottom = max(580, start_y + max_line_count * line_gap + 12)
		canvas_height = max(600, panel_bottom + 20)
		left = np.zeros((canvas_height, 800, 3), dtype=np.uint8)
		right = np.zeros((canvas_height, 800, 3), dtype=np.uint8)
		left[:600] = left_img
		right[:600] = right_img
		overlay = right.copy()
		cv2.rectangle(overlay, (20, panel_top), (780, panel_bottom), (20, 20, 20), -1)
		right = cv2.addWeighted(overlay, 0.45, right, 0.55, 0.0)

		for idx, line in enumerate(left_status_lines):
			y = start_y + idx * line_gap
			cv2.putText(
				right, line, (col1_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
				(255, 255, 255), 1, cv2.LINE_AA
			)

		for idx, line in enumerate(mid_status_lines):
			y = start_y + idx * line_gap
			cv2.putText(
				right, line, (col2_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
				(255, 255, 255), 1, cv2.LINE_AA
			)

		for idx, line in enumerate(right_status_lines):
			y = start_y + idx * line_gap
			cv2.putText(
				right, line, (col3_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
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
		legend_y = 270
		for label, color in legend_items:
			cv2.circle(right, (legend_x, legend_y), 7, color, -1)
			cv2.putText(
				right, label, (legend_x + 18, legend_y + 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
			)
			legend_x += 145

		canvas = np.zeros((canvas_height, 1600, 3), dtype=np.uint8)
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
