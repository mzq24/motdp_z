"""
Route B semantic-state diffusion policy.

This branch keeps the Route B anchor-free traj/route/speed path plus independent
and transition semantic-state heads. Legacy Route-A anchors and anchor-energy
training/guidance have been removed; some ``energy_*`` log aliases remain only
for backward-compatible dashboards.
"""

import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Callable, Tuple
import numpy as np
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from model.transformer_for_diffusion_multi_head import TransformerForDiffusion


def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.lower() in ('1', 'true', 'yes', 'on')


# =============================================================================
# Route B+ Policy
# =============================================================================

class AnnealedEnergyGuidancePolicy(nn.Module):
    """
    Compositional Energy-Guided Diffusion Policy.

    Training (dual-optimizer, GAN-style):
      Phase 1 (energy heads): Supervised on anchor trajectories + GT augmentation
      Phase 2 (decoder): Standard diffusion loss + alignment loss

    Inference:
      - Start from pure Gaussian noise N(0, I) in normalized [-1, 1] space
      - 10-step DDIM denoising
      - At each step, compute energy gradients and inject them with time-scale weights
      - Select best trajectory from num_samples candidates via energy shielding
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.cfg = config
        policy_cfg = config['policy']
        shape_meta = config['shape_meta']
        action_dim = shape_meta['action']['shape'][0]

        self.enable_action_normalization = config.get('enable_action_normalization', True)
        self.n_obs_steps = policy_cfg.get('n_obs_steps', config.get('obs_horizon', 1))

        # Transfuser feature dimensions
        transfuser_cfg = config.get('transfuser_encoder', {})
        self.bev_feature_dim = transfuser_cfg.get('bev_feature_dim', 1512)
        self.bev_feature_upsample_dim = transfuser_cfg.get('bev_feature_upsample_dim', 64)

        # Route B specific config
        route_b_cfg = config.get('route_b', {})
        self.route_b_cfg = route_b_cfg
        self.motion_only_model = bool(route_b_cfg.get('motion_only_model', False))
        self.num_samples = route_b_cfg.get('num_samples', 1)  # diffusion denoising: single mode
        self.num_inference_steps = route_b_cfg.get('num_inference_steps', 10)
        self.guidance_scale = route_b_cfg.get('guidance_scale', 0.0)
        self.use_split_forward = route_b_cfg.get('use_split_forward', True)

        self.train_energy = False if self.motion_only_model else route_b_cfg.get('train_stage1', route_b_cfg.get('train_energy', True))
        self.use_stage1_state = False if self.motion_only_model else route_b_cfg.get(
            'use_stage1_state',
            route_b_cfg.get('use_stage1_speed_energy', True),
        )
        # Backward-compatible alias for older checkpoints/configs.
        self.use_stage1_speed_energy = self.use_stage1_state
        self.shared_stage1_training_source = str(
            route_b_cfg.get('shared_stage1_training_source', 'clean')
        ).lower()
        if self.shared_stage1_training_source not in ('clean', 'noisy', 'clean_then_noisy'):
            raise ValueError(
                f"Unsupported shared_stage1_training_source={self.shared_stage1_training_source}"
            )
        self.shared_stage1_training_source_switch_epoch = int(
            route_b_cfg.get('shared_stage1_training_source_switch_epoch', 30)
        )
        self.use_speed_profile_head = route_b_cfg.get('use_speed_profile_head', False)
        self._current_epoch = 0
        self._current_batch_idx = 0
        self.route_abs_stats_path = config.get('route_abs_stats_path', None)
        self.use_lidar_bev_detail = bool(route_b_cfg.get('use_lidar_bev_detail', False)) and not self.motion_only_model
        self.lidar_history_frames = max(int(route_b_cfg.get('lidar_history_frames', self.n_obs_steps)), 1)

        self.energy_chase_weight = route_b_cfg.get(
            'stage1_chase_weight',
            route_b_cfg.get('energy_chase_weight', route_b_cfg.get('energy_front_weight', 1.0)),
        )
        self.energy_meet_weight = route_b_cfg.get(
            'stage1_meet_weight',
            route_b_cfg.get('energy_meet_weight', route_b_cfg.get('energy_left_weight', 1.0)),
        )
        self.energy_merge_weight = route_b_cfg.get('energy_merge_weight', self.energy_meet_weight)
        self.energy_cross_weight = route_b_cfg.get('energy_cross_weight', self.energy_meet_weight)
        self.energy_junction_weight = route_b_cfg.get('energy_junction_weight', self.energy_cross_weight)
        self.energy_borrow_weight = route_b_cfg.get('energy_borrow_weight', self.energy_cross_weight)
        self.energy_ped_weight = route_b_cfg.get(
            'stage1_ped_weight',
            route_b_cfg.get('energy_ped_weight', route_b_cfg.get('energy_pedestrian_weight', 1.0)),
        )
        self.energy_merge_active_weight = route_b_cfg.get('energy_merge_active_weight', 0.25)
        self.energy_cross_active_weight = route_b_cfg.get('energy_cross_active_weight', 0.25)
        self.energy_junction_active_weight = route_b_cfg.get(
            'energy_junction_active_weight', self.energy_cross_active_weight
        )
        self.energy_borrow_active_weight = route_b_cfg.get(
            'energy_borrow_active_weight', self.energy_cross_active_weight
        )
        self.energy_relation_weight = float(route_b_cfg.get('energy_relation_weight', 0.25))
        self.stage1_loss_weight = float(route_b_cfg.get(
            'stage1_loss_weight',
            route_b_cfg.get('energy_loss_weight', 1.0),
        ))
        self.energy_loss_weight = self.stage1_loss_weight
        self.energy_window_weight = float(route_b_cfg.get('stage1_window_weight', route_b_cfg.get('energy_window_weight', 0.25)))
        self.energy_phase_weight = float(route_b_cfg.get('stage1_phase_weight', route_b_cfg.get('energy_phase_weight', 0.25)))
        self.energy_conflict_area_weight = float(route_b_cfg.get('stage1_conflict_area_weight', route_b_cfg.get('energy_conflict_area_weight', 0.25)))
        self.train_stage1_speed_energy_until_epoch = route_b_cfg.get(
            'train_stage1_until_epoch',
            route_b_cfg.get('train_stage1_speed_energy_until_epoch', None),
        )
        self.train_speed_head_until_epoch = route_b_cfg.get('train_speed_head_until_epoch', None)
        self.train_stage1_speed_energy_after_update_every = route_b_cfg.get(
            'train_stage1_after_update_every',
            route_b_cfg.get('train_stage1_speed_energy_after_update_every', None),
        )
        self.train_speed_head_after_update_every = route_b_cfg.get(
            'train_speed_head_after_update_every', None
        )
        self.speed_profile_dt = float(route_b_cfg.get('speed_profile_dt', 0.5))
        self.stage1_query_dt = float(route_b_cfg.get('stage1_query_dt', 1.0))
        self.stage1_query_brake_mps2 = float(route_b_cfg.get('stage1_query_brake_mps2', 6.0))
        self.stage1_query_accel_mps2 = float(route_b_cfg.get('stage1_query_accel_mps2', 2.5))
        self.stage1_speed_offsets = torch.tensor([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0], dtype=torch.float32)
        self.use_traj_branch_condition = bool(route_b_cfg.get('use_traj_branch_condition', False)) and not self.motion_only_model
        self.traj_branch_condition_scale = float(route_b_cfg.get('traj_branch_condition_scale', 2.0))
        self.traj_branch_condition_detach = bool(route_b_cfg.get('traj_branch_condition_detach', True))
        self.traj_branch_condition_gt_prob_start = float(route_b_cfg.get('traj_branch_condition_gt_prob_start', 1.0))
        self.traj_branch_condition_gt_prob_end = float(
            route_b_cfg.get('traj_branch_condition_gt_prob_end', self.traj_branch_condition_gt_prob_start)
        )
        self.traj_branch_condition_gt_decay_epochs = max(
            int(route_b_cfg.get('traj_branch_condition_gt_decay_epochs', 0)), 0
        )
        self.traj_branch_condition_affinity_scale = float(
            route_b_cfg.get('traj_branch_condition_affinity_scale', 6.0)
        )
        self.traj_branch_condition_borrow_time_scale = float(
            route_b_cfg.get('traj_branch_condition_borrow_time_scale', 8.0)
        )
        self.traj_branch_condition_boundary_margin_scale = float(
            route_b_cfg.get('traj_branch_condition_boundary_margin_scale', 5.0)
        )
        self.semantic_motion_condition_mode = str(
            route_b_cfg.get('semantic_motion_condition_mode', 'full')
        ).lower()
        if self.semantic_motion_condition_mode not in ('full', 'compact_graph'):
            raise ValueError(
                "semantic_motion_condition_mode must be 'full' or 'compact_graph', "
                f"got {self.semantic_motion_condition_mode}"
            )
        self.semantic_motion_condition_profile = str(
            route_b_cfg.get('semantic_motion_condition_profile', 'all')
        ).lower()
        semantic_motion_condition_profile_choices = {
            'all',
            'window_only',
            'window_decision',
            'window_decision_control',
            'window_phase_opportunity',
            'compact_safe',
        }
        if self.semantic_motion_condition_profile not in semantic_motion_condition_profile_choices:
            raise ValueError(
                "semantic_motion_condition_profile must be one of "
                f"{sorted(semantic_motion_condition_profile_choices)}, "
                f"got {self.semantic_motion_condition_profile}"
            )
        self.semantic_state_supervision_profile = str(
            route_b_cfg.get('semantic_state_supervision_profile', 'all')
        ).lower()
        semantic_state_supervision_profile_choices = {
            'all',
            'window_only',
            'window_decision',
            'window_decision_control',
            'window_phase_opportunity',
            'compact_safe',
        }
        if self.semantic_state_supervision_profile not in semantic_state_supervision_profile_choices:
            raise ValueError(
                "semantic_state_supervision_profile must be one of "
                f"{sorted(semantic_state_supervision_profile_choices)}, "
                f"got {self.semantic_state_supervision_profile}"
            )
        self.semantic_state_supervision_groups = self._semantic_state_supervision_groups(
            self.semantic_state_supervision_profile
        )
        self.use_cover_relation_graph_decoder = bool(
            route_b_cfg.get('use_cover_relation_graph_decoder', False)
        ) and not self.motion_only_model
        self.cover_graph_use_traj_context = bool(
            route_b_cfg.get('cover_graph_use_traj_context', False)
        ) and not self.motion_only_model
        self.cover_graph_use_speed_context = bool(
            route_b_cfg.get('cover_graph_use_speed_context', False)
        ) and not self.motion_only_model
        self.use_route_prev_coarse_memory = bool(
            route_b_cfg.get('use_route_prev_coarse_memory', False)
        ) and not self.motion_only_model
        self.use_route_intent_token = bool(
            route_b_cfg.get('use_route_intent_token', False)
        ) and not self.motion_only_model
        self.route_intent_gate_init = float(
            route_b_cfg.get('route_intent_gate_init', 0.1)
        )
        self.current_edge_valid_loss_weight = float(
            route_b_cfg.get('current_edge_valid_loss_weight', 0.10)
        )
        self.current_edge_mode_loss_weight = float(
            route_b_cfg.get('current_edge_mode_loss_weight', 0.10)
        )
        self.future_edge_valid_loss_weight = float(
            route_b_cfg.get('future_edge_valid_loss_weight', 0.10)
        )
        self.future_edge_mode_loss_weight = float(
            route_b_cfg.get('future_edge_mode_loss_weight', 0.10)
        )
        self.current_cover_upper_loss_weight = float(
            route_b_cfg.get('current_cover_upper_loss_weight', 0.15)
        )
        self.future_cover_lower_loss_weight = float(
            route_b_cfg.get('future_cover_lower_loss_weight', 0.15)
        )
        self.front_follow_upper_loss_weight = float(
            route_b_cfg.get('front_follow_upper_loss_weight', 0.10)
        )
        self.merge_flow_lower_loss_weight = float(
            route_b_cfg.get('merge_flow_lower_loss_weight', 0.10)
        )
        self.use_edge_speed_consistency_loss = bool(
            route_b_cfg.get('use_edge_speed_consistency_loss', False)
        )
        self.edge_speed_consistency_loss_weight = float(
            route_b_cfg.get('edge_speed_consistency_loss_weight', 0.05)
        )
        self.stage1_boundary_norm_scale = float(
            route_b_cfg.get('stage1_boundary_norm_scale', 30.0)
        )
        self.use_temporary_occupancy_phase = bool(
            route_b_cfg.get('use_temporary_occupancy_phase', False)
        )
        self.temporary_occupancy_dim = int(route_b_cfg.get('temporary_occupancy_dim', 13))
        self.temporary_occupancy_loss_weight = float(
            route_b_cfg.get('temporary_occupancy_loss_weight', 0.25)
        )
        self.go_opportunity_loss_weight = float(
            route_b_cfg.get('go_opportunity_loss_weight', 0.25)
        )
        # Independent-state v1 keeps expert phase as the supervision target.
        # This alpha is reserved for eval/debug ablations and defaults to off.
        self.temporary_occupancy_phase_alpha = float(
            route_b_cfg.get('temporary_occupancy_phase_alpha', 0.0)
        )
        self.phase_go_smoothing_enable = _env_bool(
            'PHASE_GO_SMOOTHING_ENABLE',
            bool(route_b_cfg.get('phase_go_smoothing_enable', False)),
        )
        self.phase_go_smoothing_window = max(
            1,
            int(os.environ.get(
                'PHASE_GO_SMOOTHING_WINDOW',
                route_b_cfg.get('phase_go_smoothing_window', 5),
            )),
        )
        self.phase_go_smoothing_threshold = float(os.environ.get(
            'PHASE_GO_SMOOTHING_THRESHOLD',
            route_b_cfg.get('phase_go_smoothing_threshold', 0.8),
        ))
        self.phase_go_smoothing_source = str(os.environ.get(
            'PHASE_GO_SMOOTHING_SOURCE',
            route_b_cfg.get('phase_go_smoothing_source', 'go_opportunity'),
        )).lower()
        self.phase_go_smoothing_history = deque(maxlen=self.phase_go_smoothing_window)
        self.use_conflict_timing_state = bool(
            route_b_cfg.get('use_conflict_timing_state', False)
        )
        self.conflict_area_status_loss_weight = float(
            route_b_cfg.get('conflict_area_status_loss_weight', 0.25)
        )
        self.conflict_timing_loss_weight = float(
            route_b_cfg.get('conflict_timing_loss_weight', 0.25)
        )
        self.conflict_timing_dist_norm_scale = float(
            route_b_cfg.get('conflict_timing_dist_norm_scale', 30.0)
        )
        self.conflict_timing_time_norm_scale = float(
            route_b_cfg.get('conflict_timing_time_norm_scale', 10.0)
        )
        self.use_independent_state_consistency_loss = bool(
            route_b_cfg.get('use_independent_state_consistency_loss', False)
        )
        self.state_consistency_loss_weight = float(
            route_b_cfg.get('state_consistency_loss_weight', 0.02)
        )
        self.state_consistency_prob = float(
            route_b_cfg.get('state_consistency_prob', 0.25)
        )
        self.state_consistency_window_weight = float(
            route_b_cfg.get('state_consistency_window_weight', 1.0)
        )
        self.state_consistency_phase_weight = float(
            route_b_cfg.get('state_consistency_phase_weight', 1.0)
        )
        self.state_consistency_boundary_weight = float(
            route_b_cfg.get('state_consistency_boundary_weight', 1.0)
        )
        self.state_consistency_area_weight = float(
            route_b_cfg.get('state_consistency_area_weight', 1.0)
        )
        self.state_consistency_tempocc_weight = float(
            route_b_cfg.get('state_consistency_tempocc_weight', 1.0)
        )
        self.state_consistency_opportunity_weight = float(
            route_b_cfg.get('state_consistency_opportunity_weight', 1.0)
        )
        self.state_consistency_timing_weight = float(
            route_b_cfg.get('state_consistency_timing_weight', 1.0)
        )
        self.inside_area_go_loss_weight = float(
            route_b_cfg.get('inside_area_go_loss_weight', 0.0)
        )
        self.use_chase_front_following_state = _env_bool(
            'USE_CHASE_FRONT_FOLLOWING_STATE',
            bool(route_b_cfg.get('use_chase_front_following_state', False)),
        ) and not self.motion_only_model
        self.chase_has_lead_loss_weight = float(
            route_b_cfg.get('chase_has_lead_loss_weight', 0.05)
        )
        self.chase_speed_max_loss_weight = float(
            route_b_cfg.get('chase_speed_max_loss_weight', 0.15)
        )
        self.chase_speed_norm_scale = float(
            route_b_cfg.get('chase_speed_norm_scale', self.stage1_boundary_norm_scale)
        )
        self.traj_branch_condition_chase_margin_scale = float(
            route_b_cfg.get('traj_branch_condition_chase_margin_scale', 5.0)
        )
        self.use_semantic_state_transition = bool(
            route_b_cfg.get('use_semantic_state_transition', False)
        ) and not self.motion_only_model
        semantic_state_predictor_mode = str(
            route_b_cfg.get('semantic_state_predictor_mode', 'direct_plus_transition')
        ).lower()
        if semantic_state_predictor_mode == 'direct_transition':
            semantic_state_predictor_mode = 'direct_plus_transition'
        self.semantic_state_predictor_mode = semantic_state_predictor_mode
        if self.semantic_state_predictor_mode not in (
            'direct_plus_transition',
            'direct_prev_modulated',
            'direct_only',
            'transition_only',
        ):
            raise ValueError(
                "semantic_state_predictor_mode must be one of "
                "'direct_plus_transition', 'direct_prev_modulated', "
                f"'direct_only', or 'transition_only', got {self.semantic_state_predictor_mode}"
            )
        if (
            self.semantic_state_predictor_mode in ('transition_only', 'direct_prev_modulated')
            and not self.use_semantic_state_transition
        ):
            raise ValueError(
                f"{self.semantic_state_predictor_mode} semantic state predictor "
                "requires use_semantic_state_transition=true"
            )
        self.semantic_transition_prev_source = str(
            route_b_cfg.get('semantic_transition_prev_source', 'offline_gt')
        ).lower()
        if self.semantic_transition_prev_source != 'offline_gt':
            raise ValueError(
                "V1 semantic state transition only supports "
                "semantic_transition_prev_source='offline_gt'"
            )
        self.semantic_transition_loss_weight = float(
            route_b_cfg.get('semantic_transition_loss_weight', 1.0)
        )
        self.semantic_direct_aux_loss_weight = float(
            route_b_cfg.get('semantic_direct_aux_loss_weight', 0.25)
        )
        self.semantic_transition_consistency_weight = float(
            route_b_cfg.get('semantic_transition_consistency_weight', 0.05)
        )
        self.semantic_transition_prev_dropout_prob = float(
            route_b_cfg.get('semantic_transition_prev_dropout_prob', 0.0)
        )
        self.semantic_prev_token_dropout_prob = float(
            route_b_cfg.get('semantic_prev_token_dropout_prob', 0.0)
        )
        self.semantic_prev_window_dropout_prob = float(
            route_b_cfg.get('semantic_prev_window_dropout_prob', self.semantic_prev_token_dropout_prob)
        )
        self.semantic_prev_dir_dropout_prob = float(
            route_b_cfg.get('semantic_prev_dir_dropout_prob', self.semantic_prev_token_dropout_prob)
        )
        self.semantic_prev_area_dropout_prob = float(
            route_b_cfg.get('semantic_prev_area_dropout_prob', self.semantic_prev_token_dropout_prob)
        )
        self.semantic_prev_tempocc_dropout_prob = float(
            route_b_cfg.get('semantic_prev_tempocc_dropout_prob', self.semantic_prev_token_dropout_prob)
        )
        self.semantic_prev_phase_dropout_prob = float(
            route_b_cfg.get('semantic_prev_phase_dropout_prob', self.semantic_prev_token_dropout_prob)
        )
        self.semantic_prev_graph_dropout_prob = float(
            route_b_cfg.get('semantic_prev_graph_dropout_prob', self.semantic_prev_token_dropout_prob)
        )
        self.semantic_prev_boundary_dropout_prob = float(
            route_b_cfg.get('semantic_prev_boundary_dropout_prob', self.semantic_prev_token_dropout_prob)
        )
        self.semantic_prev_chase_dropout_prob = float(
            route_b_cfg.get('semantic_prev_chase_dropout_prob', self.semantic_prev_token_dropout_prob)
        )
        self.semantic_prev_random_replace_prob = float(
            route_b_cfg.get('semantic_prev_random_replace_prob', 0.0)
        )
        self.semantic_prev_replace_to_none_prob = float(
            route_b_cfg.get('semantic_prev_replace_to_none_prob', 1.0)
        )
        self.semantic_prev_prefix_dropout_frames = int(
            route_b_cfg.get('semantic_prev_prefix_dropout_frames', 0) or 0
        )
        self.semantic_prev_prefix_dropout_prob = float(
            route_b_cfg.get('semantic_prev_prefix_dropout_prob', 0.0)
        )
        self.semantic_prev_modulation_scale = float(
            route_b_cfg.get('semantic_prev_modulation_scale', 0.2)
        )
        self.semantic_prev_modulation_dropout_prob = float(
            route_b_cfg.get('semantic_prev_modulation_dropout_prob', 0.0)
        )
        self.use_semantic_state_fusion = bool(
            route_b_cfg.get('use_semantic_state_fusion', False)
        )
        self.semantic_state_fusion_alpha = float(
            route_b_cfg.get('semantic_state_fusion_alpha', 0.35)
        )
        self.semantic_state_fusion_warmup_frames = int(
            route_b_cfg.get('semantic_state_fusion_warmup_frames', 0) or 0
        )
        self.semantic_state_fusion_update_cache = bool(
            route_b_cfg.get('semantic_state_fusion_update_cache', True)
        )
        self._semantic_state_cache: Optional[dict] = None
        self._semantic_state_cache_frame: int = 0
        self.traj_phase_energy_band_offsets = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float32)
        self.traj_window_condition_names = (
            'none',
            'merge',
            'junction',
            'borrow',
        )
        self.traj_dir_condition_names = (
            'none',
            'same',
            'opposite',
            'cross',
        )
        self.traj_decision_phase_condition_names = (
            'yld',
            'go',
        )
        self.traj_control_phase_condition_names = (
            'coast_yld',
            'slow_yld',
            'stop_yld',
            'go',
        )
        self.traj_boundary_condition_names = (
            'yld_margin',
            'go_margin',
        )
        self.traj_opportunity_condition_names = (
            'yld_pressure',
            'go_opportunity',
        )
        self.traj_area_status_condition_names = (
            'none',
            'approaching',
            'inside',
            'past',
        )
        self.traj_timing_condition_names = (
            'dist_to_entry',
            'dist_to_exit',
            'time_to_entry',
        )
        self.traj_chase_condition_names = (
            'chase_has_lead',
            'chase_speed_margin',
        )
        self.traj_edge_condition_names = (
            'current_edge_valid',
            'current_edge_none',
            'current_edge_pass_after_current',
            'current_edge_go_before_future',
            'current_edge_yield_after_future',
            'current_edge_ambiguous',
            'future_edge_valid',
            'future_edge_none',
            'future_edge_pass_after_current',
            'future_edge_go_before_future',
            'future_edge_yield_after_future',
            'future_edge_ambiguous',
            'current_upper_margin',
            'future_lower_margin',
            'front_follow_upper_margin',
            'merge_flow_lower_margin',
            'current_upper_valid',
            'future_lower_valid',
            'front_follow_upper_valid',
            'merge_flow_lower_valid',
        )
        if self.semantic_motion_condition_mode == 'compact_graph':
            self.traj_branch_condition_names = (
                *self.traj_window_condition_names,
                *self.traj_decision_phase_condition_names,
                *self.traj_control_phase_condition_names,
                *self.traj_opportunity_condition_names,
                *self.traj_edge_condition_names,
                'borrow_time',
            )
        else:
            self.traj_branch_condition_names = (
                *self.traj_window_condition_names,
                *self.traj_dir_condition_names,
                *self.traj_decision_phase_condition_names,
                *self.traj_control_phase_condition_names,
                *self.traj_boundary_condition_names,
                *self.traj_opportunity_condition_names,
                *self.traj_area_status_condition_names,
                *self.traj_timing_condition_names,
                *self.traj_chase_condition_names,
                'borrow_time',
            )

        status_dim = config.get('bev_encoder', {}).get('state_dim', 15)
        ego_status_seq_len = policy_cfg.get('ego_status_seq_len', self.n_obs_steps)
        num_waypoints = policy_cfg.get('num_waypoints', 20)
        self.num_waypoints = num_waypoints
        n_emb = policy_cfg.get('n_emb', 512)

        # Build the Route B semantic-state model.
        model = TransformerForDiffusion(
            input_dim=policy_cfg.get('input_dim', 2),
            output_dim=policy_cfg.get('output_dim', 2),
            horizon=policy_cfg.get('horizon', 6),
            n_obs_steps=self.n_obs_steps,
            cond_dim=256,
            n_layer=policy_cfg.get('n_layer', 8),
            n_head=policy_cfg.get('n_head', 8),
            n_emb=n_emb,
            p_drop_emb=policy_cfg.get('p_drop_emb', 0.1),
            p_drop_attn=policy_cfg.get('p_drop_attn', 0.1),
            causal_attn=policy_cfg.get('causal_attn', True),
            obs_as_cond=policy_cfg.get('obs_as_global_cond', True),
            n_cond_layers=policy_cfg.get('n_cond_layers', 4),
            status_dim=status_dim,
            ego_status_seq_len=ego_status_seq_len,
            transfuser_bev_dim=self.bev_feature_dim,
            transfuser_bev_upsample_dim=self.bev_feature_upsample_dim,
            num_waypoints=num_waypoints,
            num_modes=1,
            traj_can_attend_route=policy_cfg.get('traj_can_attend_route', True),
            anchor_free=True,
            energy_heads=False,
            ego_detail_activation_t=policy_cfg.get('ego_detail_activation_t', 400),
            use_lidar_bev_detail=self.use_lidar_bev_detail,
            lidar_bev_history_frames=self.lidar_history_frames,
            use_condition_group_dropout=policy_cfg.get('use_condition_group_dropout', False),
            motion_only_model=self.motion_only_model,
            use_chase_front_following_state=self.use_chase_front_following_state,
            semantic_motion_condition_mode=self.semantic_motion_condition_mode,
            semantic_motion_condition_profile=self.semantic_motion_condition_profile,
            use_cover_relation_graph_decoder=self.use_cover_relation_graph_decoder,
            cover_graph_use_traj_context=self.cover_graph_use_traj_context,
            cover_graph_use_speed_context=self.cover_graph_use_speed_context,
            use_route_prev_coarse_memory=self.use_route_prev_coarse_memory,
            use_route_intent_token=self.use_route_intent_token,
            route_intent_gate_init=self.route_intent_gate_init,
        )
        self.model = model

        # ========== Diffusion Configuration ==========
        diffusion_cfg = config.get('truncated_diffusion', {})
        self.num_train_timesteps = diffusion_cfg.get('num_train_timesteps', 1000)
        # Route B uses full diffusion range (not truncated)
        self.train_max_timesteps = route_b_cfg.get('train_max_timesteps',
                                                    diffusion_cfg.get('trunc_timesteps', 100))
        self.prediction_type = diffusion_cfg.get('prediction_type', 'sample')

        # Delta z-score normalization buffers (legacy, for ablation)
        self.register_buffer('delta_mean', None)  # (T, 2)
        self.register_buffer('delta_std', None)   # (T, 2)
        # Per-timestep abs z-score normalization buffers (ablation)
        self.register_buffer('abs_mean', None)  # (T, 2)
        self.register_buffer('abs_std', None)   # (T, 2)
        # Route per-waypoint abs z-score normalization buffers for joint route diffusion
        self.register_buffer('route_abs_mean', None)  # (T_route, 2)
        self.register_buffer('route_abs_std', None)   # (T_route, 2)
        # Global abs z-score normalization buffers
        self.register_buffer('global_abs_mean', None)  # (2,)
        self.register_buffer('global_abs_std', None)   # (2,)

        # Loss weights
        self.cls_loss_weight = config.get('cls_loss_weight', 0.5)
        self.reg_loss_weight = config.get('reg_loss_weight', 1.0)
        self.route_loss_weight = diffusion_cfg.get('route_loss_weight', 0.5)
        self.route_final_loss_weight = diffusion_cfg.get('route_final_loss_weight', 1.0)
        self.stage1_loss_weight = float(route_b_cfg.get(
            'stage1_loss_weight',
            route_b_cfg.get('energy_loss_weight', getattr(self, 'stage1_loss_weight', 1.0)),
        ))
        self.energy_loss_weight = self.stage1_loss_weight
        self.speed_loss_weight = route_b_cfg.get('speed_loss_weight', 1.0)
        self.speed_profile_loss_weight = route_b_cfg.get('speed_profile_loss_weight', 1.0)

        # DDIM Scheduler
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            steps_offset=1,
            beta_schedule="scaled_linear",
            prediction_type=self.prediction_type,
        )

        self.action_dim = action_dim
        self.horizon = policy_cfg.get('horizon', 6)
        self.n_action_steps = policy_cfg.get('action_horizon', 8)
        default_profile_weights = [1.0, 0.7, 0.5, 0.35, 0.25, 0.2]
        if self.horizon <= len(default_profile_weights):
            speed_profile_weights = default_profile_weights[:self.horizon]
        else:
            speed_profile_weights = default_profile_weights + [default_profile_weights[-1]] * (self.horizon - len(default_profile_weights))
        self.register_buffer(
            'speed_profile_step_weights',
            torch.tensor(speed_profile_weights, dtype=torch.float32),
        )

    def _get_transfuser_lidar_bev(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.use_lidar_bev_detail:
            return None
        transfuser_lidar_bev = tensor_dict.get('transfuser_lidar_bev')
        if transfuser_lidar_bev is None:
            return None
        return transfuser_lidar_bev.to(device=device, dtype=model_dtype)

    # ========== Speed Target Computation ==========
    def _compute_speed_target(self, trajectory, device, batch: Optional[Dict[str, torch.Tensor]] = None):
        """Compute two-hot speed target for scalar speed head training.

        Preferred target is the exact next-step future speed read from raw measurements
        (`next_speed_target_mps`, t+0.5s). When unavailable, fall back to the first
        0.5s average speed derived from GT trajectory.
        """
        target_speed = None
        if batch is not None:
            next_speed_target = batch.get('next_speed_target_mps')
            if next_speed_target is not None:
                target_speed = next_speed_target.to(device=device, dtype=trajectory.dtype).reshape(-1)

        if target_speed is None:
            if trajectory.shape[1] < 1:
                return None
            target_speed = trajectory[:, 0].norm(dim=-1) / max(self.speed_profile_dt, 1e-6)

        speed_classes = self.model.speed_classes
        num_classes = len(speed_classes)
        bins = torch.tensor(speed_classes, device=device, dtype=target_speed.dtype)

        # Clamp to valid range
        target_speed = target_speed.clamp(min=bins[0], max=bins[-1])

        # Two-hot encoding: interpolate between adjacent bins
        B = target_speed.shape[0]
        two_hot = torch.zeros(B, num_classes, device=device, dtype=target_speed.dtype)
        for i in range(num_classes - 1):
            mask = (target_speed >= bins[i]) & (target_speed < bins[i + 1])
            if i == num_classes - 2:  # last bin: include upper bound
                mask = mask | (target_speed == bins[i + 1])
            if mask.any():
                ratio = (target_speed[mask] - bins[i]) / (bins[i + 1] - bins[i]).clamp(min=1e-6)
                two_hot[mask, i] = 1.0 - ratio
                two_hot[mask, i + 1] = ratio
        return two_hot

    def _compute_speed_profile_target(self, trajectory, device, model_dtype):
        """Compute per-step short-horizon speed profile from GT trajectory."""
        if trajectory.shape[1] < 1:
            return None
        first_disp = trajectory[:, :1, :]
        if trajectory.shape[1] > 1:
            future_delta = trajectory[:, 1:, :] - trajectory[:, :-1, :]
            displacements = torch.cat([first_disp, future_delta], dim=1)
        else:
            displacements = first_disp
        speed_profile = displacements.norm(dim=-1) / max(self.speed_profile_dt, 1e-6)
        return speed_profile.to(device=device, dtype=model_dtype).clamp(min=0.0, max=20.0)

    def _has_stage1_labels(self, batch: Dict[str, torch.Tensor]) -> bool:
        if not self.use_stage1_speed_energy:
            return False
        required = (
            self._resolve_stage1_batch_key(batch, 'conflict_area_family'),
            self._resolve_stage1_batch_key(batch, 'conflict_area_dir'),
            self._resolve_stage1_batch_key(batch, 'conflict_area_active'),
            self._resolve_stage1_batch_key(batch, 'conflict_area_start_frame'),
            self._resolve_stage1_batch_key(batch, 'conflict_area_end_frame'),
            self._resolve_stage1_batch_key(batch, 'conflict_decision_phase'),
            self._resolve_stage1_batch_key(batch, 'conflict_control_phase'),
            self._resolve_stage1_batch_key(batch, 'merge_yld_max_speed'),
            self._resolve_stage1_batch_key(batch, 'merge_go_min_speed'),
            self._resolve_stage1_batch_key(batch, 'junction_yld_max_speed'),
            self._resolve_stage1_batch_key(batch, 'junction_go_min_speed'),
            self._resolve_stage1_batch_key(batch, 'borrow_yld_max_speed'),
            self._resolve_stage1_batch_key(batch, 'borrow_go_min_speed'),
        )
        if any(key is None for key in required):
            return False
        if self.use_temporary_occupancy_phase:
            temp_required = (
                self._resolve_stage1_batch_key(batch, 'temporary_occupancy_cover_bins'),
                self._resolve_stage1_batch_key(batch, 'temporary_occupancy_cover_valid'),
                self._resolve_stage1_batch_key(batch, 'go_opportunity_prob'),
                self._resolve_stage1_batch_key(batch, 'yld_pressure_prob'),
                self._resolve_stage1_batch_key(batch, 'go_opportunity_valid'),
            )
            if any(key is None for key in temp_required):
                raise ValueError(
                    "use_temporary_occupancy_phase=true requires "
                    "temporary_occupancy_cover_bins/valid, go_opportunity_prob, "
                    "yld_pressure_prob, and go_opportunity_valid in every batch"
                )
        if self.use_conflict_timing_state:
            timing_required = (
                self._resolve_stage1_batch_key(batch, 'conflict_area_status'),
                self._resolve_stage1_batch_key(batch, 'conflict_dist_to_entry_m'),
                self._resolve_stage1_batch_key(batch, 'conflict_dist_to_exit_m'),
                self._resolve_stage1_batch_key(batch, 'conflict_time_to_entry_s'),
            )
            if any(key is None for key in timing_required):
                raise ValueError(
                    "use_conflict_timing_state=true requires conflict_area_status, "
                    "conflict_dist_to_entry_m, conflict_dist_to_exit_m, "
                    "and conflict_time_to_entry_s in every batch"
                )
        if self.use_chase_front_following_state:
            chase_required = (
                self._resolve_stage1_batch_key(batch, 'chase_has_lead'),
                self._resolve_stage1_batch_key(batch, 'chase_speed_max'),
            )
            if any(key is None for key in chase_required):
                raise ValueError(
                    "use_chase_front_following_state=true requires chase_has_lead, "
                    "and chase_speed_max in every batch"
                )
        if self.use_cover_relation_graph_decoder:
            graph_required = (
                self._resolve_stage1_batch_key(batch, 'current_cover_edge_valid'),
                self._resolve_stage1_batch_key(batch, 'current_cover_edge_occupied'),
                self._resolve_stage1_batch_key(batch, 'current_cover_edge_mode'),
                self._resolve_stage1_batch_key(batch, 'current_cover_edge_mode_valid'),
                self._resolve_stage1_batch_key(batch, 'current_cover_upper_speed_mps'),
                self._resolve_stage1_batch_key(batch, 'current_cover_upper_speed_valid'),
                self._resolve_stage1_batch_key(batch, 'future_cover_edge_valid'),
                self._resolve_stage1_batch_key(batch, 'future_cover_edge_mode'),
                self._resolve_stage1_batch_key(batch, 'future_cover_edge_mode_valid'),
                self._resolve_stage1_batch_key(batch, 'future_cover_lower_speed_mps'),
                self._resolve_stage1_batch_key(batch, 'future_cover_lower_speed_valid'),
                self._resolve_stage1_batch_key(batch, 'front_follow_upper_speed_mps'),
                self._resolve_stage1_batch_key(batch, 'front_follow_upper_speed_valid'),
                self._resolve_stage1_batch_key(batch, 'merge_flow_lower_speed_mps'),
                self._resolve_stage1_batch_key(batch, 'merge_flow_lower_speed_valid'),
            )
            if any(key is None for key in graph_required):
                raise ValueError(
                    "use_cover_relation_graph_decoder=true requires edge-aware "
                    "cover relation labels. Run the graph postprocess plus project/split."
                )
        if self.use_semantic_state_transition:
            prev_required = (
                self._resolve_stage1_batch_key(batch, 'prev_semantic_state_valid'),
                self._resolve_stage1_batch_key(batch, 'prev_conflict_area_family'),
                self._resolve_stage1_batch_key(batch, 'prev_conflict_area_dir'),
                self._resolve_stage1_batch_key(batch, 'prev_conflict_area_status'),
                self._resolve_stage1_batch_key(batch, 'prev_conflict_decision_phase'),
                self._resolve_stage1_batch_key(batch, 'prev_conflict_control_phase'),
                self._resolve_stage1_batch_key(batch, 'prev_conflict_area_route_mask'),
                self._resolve_stage1_batch_key(batch, 'prev_temporary_occupancy_cover_bins'),
                self._resolve_stage1_batch_key(batch, 'prev_temporary_occupancy_cover_valid'),
                self._resolve_stage1_batch_key(batch, 'prev_go_opportunity_prob'),
                self._resolve_stage1_batch_key(batch, 'prev_yld_pressure_prob'),
                self._resolve_stage1_batch_key(batch, 'prev_go_opportunity_valid'),
                self._resolve_stage1_batch_key(batch, 'prev_conflict_dist_to_entry_m'),
                self._resolve_stage1_batch_key(batch, 'prev_conflict_dist_to_exit_m'),
                self._resolve_stage1_batch_key(batch, 'prev_conflict_time_to_entry_s'),
                self._resolve_stage1_batch_key(batch, 'prev_conflict_timing_valid'),
                self._resolve_stage1_batch_key(batch, 'prev_merge_yld_max_speed'),
                self._resolve_stage1_batch_key(batch, 'prev_merge_go_min_speed'),
                self._resolve_stage1_batch_key(batch, 'prev_junction_yld_max_speed'),
                self._resolve_stage1_batch_key(batch, 'prev_junction_go_min_speed'),
                self._resolve_stage1_batch_key(batch, 'prev_borrow_yld_max_speed'),
                self._resolve_stage1_batch_key(batch, 'prev_borrow_go_min_speed'),
                self._resolve_stage1_batch_key(batch, 'prev_chase_has_lead'),
                self._resolve_stage1_batch_key(batch, 'prev_chase_speed_max'),
            )
            if any(key is None for key in prev_required):
                raise ValueError(
                    "use_semantic_state_transition=true requires offline prev_* "
                    "semantic state fields. Run postprocess_semantic_prev_state.py "
                    "on train/val packed files after split."
                )
        return True

    @staticmethod
    def _resolve_stage1_batch_key(batch: Dict[str, torch.Tensor], base_key: str):
        dense_key = f'{base_key}_dense'
        if dense_key in batch:
            return dense_key
        if base_key in batch:
            return base_key
        return None

    def _get_stage1_batch_tensor(
        self,
        batch: Dict[str, torch.Tensor],
        base_key: str,
        device: Optional[torch.device],
        model_dtype: Optional[torch.dtype],
    ):
        resolved_key = self._resolve_stage1_batch_key(batch, base_key)
        if resolved_key is None:
            return None
        value = batch[resolved_key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if device is None and model_dtype is None:
            return value
        kwargs = {}
        if device is not None:
            kwargs['device'] = device
        if model_dtype is not None:
            kwargs['dtype'] = model_dtype
        return value.to(**kwargs)

    def _get_stage1_borrow_time_target(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if 'borrow_cross_active_time_s' not in batch:
            return None
        value = batch['borrow_cross_active_time_s']
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        return value.to(device=device, dtype=model_dtype).reshape(-1)

    def _get_stage1_long_target(
        self,
        batch: Dict[str, torch.Tensor],
        base_key: str,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        value = self._get_stage1_batch_tensor(batch, base_key, device=device, model_dtype=None)
        if value is None:
            return None
        return value.reshape(-1).long()

    def _get_temporary_occupancy_targets(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        model_dtype: torch.dtype,
        require: bool = False,
    ):
        bins = self._get_stage1_batch_tensor(
            batch, 'temporary_occupancy_cover_bins', device=device, model_dtype=model_dtype
        )
        valid = self._get_stage1_batch_tensor(
            batch, 'temporary_occupancy_cover_valid', device=device, model_dtype=model_dtype
        )
        go_prob = self._get_stage1_batch_tensor(
            batch, 'go_opportunity_prob', device=device, model_dtype=model_dtype
        )
        yld_prob = self._get_stage1_batch_tensor(
            batch, 'yld_pressure_prob', device=device, model_dtype=model_dtype
        )
        go_valid = self._get_stage1_batch_tensor(
            batch, 'go_opportunity_valid', device=device, model_dtype=model_dtype
        )
        missing = bins is None or valid is None or go_prob is None or yld_prob is None or go_valid is None
        if missing:
            if require:
                raise ValueError(
                    "temporary occupancy training requires "
                    "temporary_occupancy_cover_bins/valid and go/yld opportunity labels"
                )
            return None

        B = go_prob.reshape(-1).shape[0]
        bins = bins.reshape(B, -1)
        valid = valid.reshape(B, -1)
        if bins.shape[1] != self.temporary_occupancy_dim or valid.shape[1] != self.temporary_occupancy_dim:
            raise ValueError(
                "temporary occupancy labels must have shape "
                f"(B, {self.temporary_occupancy_dim}), got bins={bins.shape}, valid={valid.shape}"
            )

        yld_prob = torch.nan_to_num(yld_prob.reshape(-1), nan=0.5, posinf=1.0, neginf=0.0)
        go_prob = torch.nan_to_num(go_prob.reshape(-1), nan=0.5, posinf=1.0, neginf=0.0)
        target_probs = torch.stack([yld_prob, go_prob], dim=-1).clamp(0.0, 1.0)
        prob_sum = target_probs.sum(dim=-1, keepdim=True)
        target_probs = torch.where(
            prob_sum > 1e-6,
            target_probs / prob_sum.clamp(min=1e-6),
            torch.full_like(target_probs, 0.5),
        )

        return {
            'bins': bins.clamp(0.0, 1.0),
            'valid': valid > 0.5,
            'go_opportunity_target': target_probs,
            'go_opportunity_valid': go_valid.reshape(-1) > 0.5,
        }

    def _get_conflict_timing_targets(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        model_dtype: torch.dtype,
        family_codes: Optional[torch.Tensor] = None,
        require: bool = False,
    ):
        status = self._get_stage1_long_target(batch, 'conflict_area_status', device=device)
        dist_entry = self._get_stage1_batch_tensor(
            batch, 'conflict_dist_to_entry_m', device=device, model_dtype=model_dtype
        )
        dist_exit = self._get_stage1_batch_tensor(
            batch, 'conflict_dist_to_exit_m', device=device, model_dtype=model_dtype
        )
        time_entry = self._get_stage1_batch_tensor(
            batch, 'conflict_time_to_entry_s', device=device, model_dtype=model_dtype
        )
        timing_valid_explicit = self._get_stage1_batch_tensor(
            batch, 'conflict_timing_valid', device=device, model_dtype=model_dtype
        )
        if timing_valid_explicit is None:
            timing_valid_explicit = self._get_stage1_batch_tensor(
                batch, 'conflict_area_timing_valid', device=device, model_dtype=model_dtype
            )
        status_valid_explicit = self._get_stage1_batch_tensor(
            batch, 'conflict_area_status_valid', device=device, model_dtype=model_dtype
        )
        scalar_valids = []
        for key in (
            'conflict_dist_to_entry_valid',
            'conflict_dist_to_exit_valid',
            'conflict_time_to_entry_valid',
        ):
            value = self._get_stage1_batch_tensor(batch, key, device=device, model_dtype=model_dtype)
            if value is not None:
                scalar_valids.append(value.reshape(-1) > 0.5)
        missing = status is None or dist_entry is None or dist_exit is None or time_entry is None
        if missing:
            if require:
                raise ValueError(
                    "conflict timing training requires conflict_area_status, "
                    "conflict_dist_to_entry_m, conflict_dist_to_exit_m, conflict_time_to_entry_s"
                )
            return None

        status = status.reshape(-1).clamp(min=0, max=3)
        dist_entry = torch.nan_to_num(dist_entry.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        dist_exit = torch.nan_to_num(dist_exit.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        time_entry = torch.nan_to_num(time_entry.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        B = status.shape[0]

        def _valid_vector(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if value is None:
                return None
            value = value.to(device=device).reshape(B, -1)
            return (value > 0.5).all(dim=-1)

        timing_valid_vec = _valid_vector(timing_valid_explicit)
        status_valid_vec = _valid_vector(status_valid_explicit)
        values = torch.stack(
            [
                dist_entry / max(self.conflict_timing_dist_norm_scale, 1e-6),
                dist_exit / max(self.conflict_timing_dist_norm_scale, 1e-6),
                time_entry / max(self.conflict_timing_time_norm_scale, 1e-6),
            ],
            dim=-1,
        ).clamp(-2.0, 2.0)
        if timing_valid_vec is not None:
            valid = timing_valid_vec
        elif scalar_valids:
            valid = torch.stack(scalar_valids, dim=-1).all(dim=-1)
        elif status_valid_vec is not None:
            valid = status_valid_vec
        elif family_codes is None:
            valid = status > 0
        else:
            valid = family_codes.reshape(-1).to(device=device) > 0
        if status_valid_vec is not None:
            status_valid = status_valid_vec
        else:
            status_valid = valid
        finite_valid = torch.isfinite(values).all(dim=-1)
        return {
            'status': status,
            'values': values,
            'valid': valid & finite_valid,
            'status_valid': status_valid & finite_valid,
        }

    @staticmethod
    def _window_target_from_family_codes(family_codes: torch.Tensor) -> torch.Tensor:
        window_target = torch.zeros_like(family_codes)
        window_target[family_codes == 1] = 3  # borrow -> borrow
        window_target[family_codes == 2] = 1  # merge -> merge
        window_target[family_codes == 3] = 2  # junction -> junction
        return window_target

    def _boundary_norm_to_mps(self, value: torch.Tensor) -> torch.Tensor:
        return value.clamp(0.0, 1.0) * float(self.stage1_boundary_norm_scale)

    def _chase_norm_to_mps(self, value: torch.Tensor) -> torch.Tensor:
        return value.clamp(0.0, 1.0) * float(self.chase_speed_norm_scale)

    def _get_chase_targets(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        model_dtype: torch.dtype,
        require: bool = False,
    ):
        has_lead = self._get_stage1_batch_tensor(
            batch, 'chase_has_lead', device=device, model_dtype=model_dtype
        )
        speed_max = self._get_stage1_batch_tensor(
            batch, 'chase_speed_max', device=device, model_dtype=model_dtype
        )
        speed_valid = self._get_stage1_batch_tensor(
            batch, 'chase_speed_max_valid', device=device, model_dtype=model_dtype
        )
        missing = has_lead is None or speed_max is None
        if missing:
            if require:
                raise ValueError(
                    "chase/front-following training requires chase_has_lead, "
                    "and chase_speed_max"
                )
            return None
        has_lead = torch.nan_to_num(
            has_lead.reshape(-1), nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        speed_max = torch.nan_to_num(
            speed_max.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(min=0.0)
        if speed_valid is None:
            speed_valid = torch.ones_like(speed_max, dtype=torch.bool)
        else:
            speed_valid = speed_valid.reshape(-1) > 0.5
        return {
            'has_lead': has_lead,
            'speed_max': speed_max,
            'speed_max_valid': speed_valid,
        }

    def _get_cover_relation_graph_targets(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        model_dtype: torch.dtype,
        require: bool = False,
    ) -> Optional[dict]:
        def _float(name: str) -> Optional[torch.Tensor]:
            value = self._get_stage1_batch_tensor(
                batch, name, device=device, model_dtype=model_dtype
            )
            if value is None:
                return None
            return torch.nan_to_num(
                value.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0
            )

        def _long(name: str) -> Optional[torch.Tensor]:
            value = self._get_stage1_long_target(batch, name, device=device)
            if value is None:
                return None
            return value.reshape(-1)

        current_valid = _float('current_cover_edge_valid')
        current_occupied = _float('current_cover_edge_occupied')
        current_mode = _long('current_cover_edge_mode')
        current_mode_valid = _float('current_cover_edge_mode_valid')
        current_upper = _float('current_cover_upper_speed_mps')
        current_upper_valid = _float('current_cover_upper_speed_valid')
        future_valid = _float('future_cover_edge_valid')
        future_mode = _long('future_cover_edge_mode')
        future_mode_valid = _float('future_cover_edge_mode_valid')
        future_lower = _float('future_cover_lower_speed_mps')
        future_lower_valid = _float('future_cover_lower_speed_valid')
        front_follow_upper = _float('front_follow_upper_speed_mps')
        front_follow_upper_valid = _float('front_follow_upper_speed_valid')
        merge_flow_lower = _float('merge_flow_lower_speed_mps')
        merge_flow_lower_valid = _float('merge_flow_lower_speed_valid')

        required_values = (
            current_valid,
            current_occupied,
            current_mode,
            current_mode_valid,
            current_upper,
            current_upper_valid,
            future_valid,
            future_mode,
            future_mode_valid,
            future_lower,
            future_lower_valid,
            front_follow_upper,
            front_follow_upper_valid,
            merge_flow_lower,
            merge_flow_lower_valid,
        )
        if any(value is None for value in required_values):
            if require:
                raise ValueError(
                    "cover relation graph training requires current/future edge "
                    "valid/mode/speed labels plus front-follow and merge-flow speeds"
                )
            return None

        return {
            'current_valid': current_valid.clamp(0.0, 1.0),
            'current_occupied': current_occupied.clamp(0.0, 1.0),
            'current_mode': current_mode.clamp(min=0, max=4),
            'current_mode_valid': current_mode_valid.clamp(0.0, 1.0) > 0.5,
            'current_upper': current_upper.clamp(min=0.0),
            'current_upper_valid': current_upper_valid.clamp(0.0, 1.0) > 0.5,
            'future_valid': future_valid.clamp(0.0, 1.0),
            'future_mode': future_mode.clamp(min=0, max=4),
            'future_mode_valid': future_mode_valid.clamp(0.0, 1.0) > 0.5,
            'future_lower': future_lower.clamp(min=0.0),
            'future_lower_valid': future_lower_valid.clamp(0.0, 1.0) > 0.5,
            'front_follow_upper': front_follow_upper.clamp(min=0.0),
            'front_follow_upper_valid': front_follow_upper_valid.clamp(0.0, 1.0) > 0.5,
            'merge_flow_lower': merge_flow_lower.clamp(min=0.0),
            'merge_flow_lower_valid': merge_flow_lower_valid.clamp(0.0, 1.0) > 0.5,
        }

    def _get_semantic_transition_prev_state(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        model_dtype: torch.dtype,
        route_steps: int,
        require: bool = False,
    ) -> Optional[dict]:
        valid = self._get_stage1_batch_tensor(
            batch, 'prev_semantic_state_valid', device=device, model_dtype=model_dtype
        )
        categorical = {}
        for out_key, batch_key in (
            ('family', 'prev_conflict_area_family'),
            ('dir', 'prev_conflict_area_dir'),
            ('status', 'prev_conflict_area_status'),
            ('decision_phase', 'prev_conflict_decision_phase'),
            ('control_phase', 'prev_conflict_control_phase'),
        ):
            value = self._get_stage1_long_target(batch, batch_key, device=device)
            if value is None:
                if require:
                    raise ValueError(f"semantic transition requires {batch_key}")
                return None
            categorical[out_key] = value.reshape(-1)

        if valid is None:
            if require:
                raise ValueError("semantic transition requires prev_semantic_state_valid")
            return None
        valid = torch.nan_to_num(valid.reshape(-1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        B = valid.shape[0]

        def _float_field(key: str, *, width: int = 1, default: float = 0.0) -> Optional[torch.Tensor]:
            value = self._get_stage1_batch_tensor(batch, key, device=device, model_dtype=model_dtype)
            if value is None:
                if require:
                    raise ValueError(f"semantic transition requires {key}")
                return None
            value = value.reshape(B, -1)
            if value.shape[1] != width:
                raise ValueError(f"{key} expects width={width}, got {value.shape}")
            return torch.nan_to_num(value, nan=default, posinf=default, neginf=default)

        def _optional_float_field(key: str, *, width: int = 1, default: float = 0.0) -> torch.Tensor:
            value = self._get_stage1_batch_tensor(batch, key, device=device, model_dtype=model_dtype)
            if value is None:
                return torch.full((B, width), default, device=device, dtype=model_dtype)
            value = value.reshape(B, -1)
            if value.shape[1] != width:
                raise ValueError(f"{key} expects width={width}, got {value.shape}")
            return torch.nan_to_num(value, nan=default, posinf=default, neginf=default)

        def _optional_long_field(key: str, *, default: int = 0, max_value: int = 4) -> torch.Tensor:
            value = self._get_stage1_long_target(batch, key, device=device)
            if value is None:
                return torch.full((B,), default, device=device, dtype=torch.long)
            return value.reshape(B).long().clamp(min=0, max=max_value)

        area_mask = _float_field('prev_conflict_area_route_mask', width=route_steps)
        temp_bins = _float_field('prev_temporary_occupancy_cover_bins', width=self.temporary_occupancy_dim)
        temp_valid = _float_field('prev_temporary_occupancy_cover_valid', width=self.temporary_occupancy_dim)
        go_prob = _float_field('prev_go_opportunity_prob', default=0.5)
        yld_prob = _float_field('prev_yld_pressure_prob', default=0.5)
        go_valid = _float_field('prev_go_opportunity_valid')
        dist_entry = _float_field('prev_conflict_dist_to_entry_m')
        dist_exit = _float_field('prev_conflict_dist_to_exit_m')
        time_entry = _float_field('prev_conflict_time_to_entry_s')
        timing_valid = _float_field('prev_conflict_timing_valid')
        boundary_values = []
        for key in (
            'prev_merge_yld_max_speed',
            'prev_merge_go_min_speed',
            'prev_junction_yld_max_speed',
            'prev_junction_go_min_speed',
            'prev_borrow_yld_max_speed',
            'prev_borrow_go_min_speed',
        ):
            value = _float_field(key)
            if value is None:
                return None
            boundary_values.append(value.reshape(B))
        chase_has_lead = _float_field('prev_chase_has_lead')
        chase_speed = _float_field('prev_chase_speed_max')
        if any(
            value is None
            for value in (
                area_mask,
                temp_bins,
                temp_valid,
                go_prob,
                yld_prob,
                go_valid,
                dist_entry,
                dist_exit,
                time_entry,
                timing_valid,
                chase_has_lead,
                chase_speed,
            )
        ):
            return None

        timing_values = torch.cat(
            [
                dist_entry / max(self.conflict_timing_dist_norm_scale, 1e-6),
                dist_exit / max(self.conflict_timing_dist_norm_scale, 1e-6),
                time_entry / max(self.conflict_timing_time_norm_scale, 1e-6),
            ],
            dim=-1,
        ).clamp(-2.0, 2.0)
        boundary = (
            torch.stack(boundary_values, dim=-1)
            / max(self.stage1_boundary_norm_scale, 1e-6)
        ).clamp(0.0, 1.0)
        chase_values = torch.cat(
            [
                chase_has_lead.clamp(0.0, 1.0),
                (chase_speed / max(self.chase_speed_norm_scale, 1e-6)).clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        prev_current_mode = _optional_long_field('prev_current_cover_edge_mode', max_value=4)
        prev_future_mode = _optional_long_field('prev_future_cover_edge_mode', max_value=4)
        prev_current_mode_oh = F.one_hot(prev_current_mode, num_classes=5).to(
            device=device, dtype=model_dtype
        )
        prev_future_mode_oh = F.one_hot(prev_future_mode, num_classes=5).to(
            device=device, dtype=model_dtype
        )
        graph_values = torch.cat(
            [
                _optional_float_field('prev_current_cover_edge_valid').clamp(0.0, 1.0),
                prev_current_mode_oh,
                _optional_float_field('prev_future_cover_edge_valid').clamp(0.0, 1.0),
                prev_future_mode_oh,
                (
                    _optional_float_field('prev_current_cover_upper_speed_mps')
                    / max(self.stage1_boundary_norm_scale, 1e-6)
                ).clamp(0.0, 1.0),
                (
                    _optional_float_field('prev_future_cover_lower_speed_mps')
                    / max(self.stage1_boundary_norm_scale, 1e-6)
                ).clamp(0.0, 1.0),
                (
                    _optional_float_field('prev_front_follow_upper_speed_mps')
                    / max(self.stage1_boundary_norm_scale, 1e-6)
                ).clamp(0.0, 1.0),
                (
                    _optional_float_field('prev_merge_flow_lower_speed_mps')
                    / max(self.stage1_boundary_norm_scale, 1e-6)
                ).clamp(0.0, 1.0),
                _optional_float_field('prev_current_cover_upper_speed_valid').clamp(0.0, 1.0),
                _optional_float_field('prev_future_cover_lower_speed_valid').clamp(0.0, 1.0),
                _optional_float_field('prev_front_follow_upper_speed_valid').clamp(0.0, 1.0),
                _optional_float_field('prev_merge_flow_lower_speed_valid').clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        prev_state = {
            **categorical,
            'valid': valid,
            'conflict_area_route_mask': area_mask.clamp(0.0, 1.0),
            'temporary_occupancy_cover_bins': temp_bins.clamp(0.0, 1.0),
            'temporary_occupancy_cover_valid': temp_valid.clamp(0.0, 1.0),
            'go_opportunity_prob': go_prob.clamp(0.0, 1.0),
            'yld_pressure_prob': yld_prob.clamp(0.0, 1.0),
            'go_opportunity_valid': go_valid.clamp(0.0, 1.0),
            'conflict_timing_values': timing_values,
            'conflict_timing_valid': timing_valid.clamp(0.0, 1.0),
            'boundary_values': boundary,
            'chase_values': chase_values,
            'graph_values': graph_values,
        }
        # Invalid prev samples are true sequence starts; hide semantic content but
        # keep the validity bit so the transition branch can learn the reset case.
        for key, value in list(prev_state.items()):
            if key in ('valid',):
                continue
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                gate = valid
                while gate.dim() < value.dim():
                    gate = gate.unsqueeze(-1)
                prev_state[key] = value * gate
        return prev_state

    def _apply_semantic_transition_prev_dropout(
        self,
        prev_state: Optional[dict],
        *,
        route_steps: int,
        device: torch.device,
        model_dtype: torch.dtype,
        batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Optional[dict]:
        """Training-time corruption for offline-GT prev semantic state.

        Sample-level dropout simulates route/cache reset. Group-level dropout
        and replacement prevent the transition branch from learning a pure
        previous-token copy prior.
        """
        if prev_state is None or (not self.training):
            return prev_state
        all_probs = (
            self.semantic_transition_prev_dropout_prob,
            self.semantic_prev_token_dropout_prob,
            self.semantic_prev_window_dropout_prob,
            self.semantic_prev_dir_dropout_prob,
            self.semantic_prev_area_dropout_prob,
            self.semantic_prev_tempocc_dropout_prob,
            self.semantic_prev_phase_dropout_prob,
            self.semantic_prev_graph_dropout_prob,
            self.semantic_prev_boundary_dropout_prob,
            self.semantic_prev_chase_dropout_prob,
            self.semantic_prev_prefix_dropout_prob,
            self.semantic_prev_random_replace_prob,
        )
        if max(float(p) for p in all_probs) <= 0.0:
            return prev_state
        valid = prev_state.get('valid')
        if not isinstance(valid, torch.Tensor):
            return prev_state
        valid = valid.to(device=device, dtype=model_dtype).reshape(-1).clamp(0.0, 1.0)
        B = valid.shape[0]
        neutral = self._build_neutral_semantic_prev_state(
            B, route_steps, device=device, model_dtype=model_dtype
        )

        sample_drop_p = min(max(float(self.semantic_transition_prev_dropout_prob), 0.0), 1.0)
        sample_drop = torch.rand((B,), device=device) < sample_drop_p
        prefix_frames = max(int(self.semantic_prev_prefix_dropout_frames), 0)
        prefix_p = min(max(float(self.semantic_prev_prefix_dropout_prob), 0.0), 1.0)
        if batch is not None and prefix_frames > 0 and prefix_p > 0.0:
            prefix_index = None
            for key in (
                'semantic_route_frame_index',
                'semantic_prev_route_frame_index',
                'route_local_frame_index',
                'route_frame_index',
                'frame_in_route',
                'sample_route_index',
            ):
                value = self._get_stage1_batch_tensor(
                    batch, key, device=device, model_dtype=model_dtype
                )
                if value is not None:
                    prefix_index = value.reshape(-1)
                    break
            if prefix_index is not None and prefix_index.shape[0] == B:
                prefix_mask = prefix_index < float(prefix_frames)
                prefix_drop = prefix_mask & (torch.rand((B,), device=device) < prefix_p)
                sample_drop = sample_drop | prefix_drop

        keep = (~sample_drop).to(dtype=model_dtype)
        dropped = {}
        for key, value in prev_state.items():
            if not isinstance(value, torch.Tensor):
                dropped[key] = value
                continue
            neutral_value = neutral.get(key)
            if key == 'valid':
                dropped[key] = valid * keep
                continue
            if neutral_value is None:
                neutral_value = torch.zeros_like(value)
            if value.dtype.is_floating_point:
                value_t = value.to(device=device, dtype=model_dtype)
                neutral_t = neutral_value.to(device=device, dtype=model_dtype)
                gate = keep
                while gate.dim() < value_t.dim():
                    gate = gate.unsqueeze(-1)
                dropped[key] = value_t * gate + neutral_t * (1.0 - gate)
            else:
                value_t = value.to(device=device)
                neutral_t = neutral_value.to(device=device, dtype=value_t.dtype)
                gate_bool = keep.to(dtype=torch.bool)
                while gate_bool.dim() < value_t.dim():
                    gate_bool = gate_bool.unsqueeze(-1)
                dropped[key] = torch.where(gate_bool, value_t, neutral_t)

        def _prob(value: float) -> float:
            return min(max(float(value), 0.0), 1.0)

        def _mask(prob: float) -> torch.Tensor:
            prob = _prob(max(prob, self.semantic_prev_token_dropout_prob))
            if prob <= 0.0:
                return torch.zeros((B,), device=device, dtype=torch.bool)
            return torch.rand((B,), device=device) < prob

        def _replace_categorical(key: str, mask: torch.Tensor, max_value: int) -> None:
            value = dropped.get(key)
            neutral_value = neutral.get(key)
            if not isinstance(value, torch.Tensor) or neutral_value is None or not mask.any():
                return
            value_t = value.to(device=device)
            neutral_t = neutral_value.to(device=device, dtype=value_t.dtype)
            random_p = _prob(self.semantic_prev_random_replace_prob)
            none_p = _prob(self.semantic_prev_replace_to_none_prob)
            random_mask = torch.zeros_like(mask)
            if random_p > 0.0:
                random_mask = mask & (torch.rand((B,), device=device) < random_p)
                random_values = torch.randint(
                    low=0,
                    high=max_value + 1,
                    size=value_t.reshape(B).shape,
                    device=device,
                    dtype=value_t.dtype,
                )
                none_mask = random_mask & (torch.rand((B,), device=device) < none_p)
                replacement = torch.where(none_mask, neutral_t.reshape(B), random_values)
                value_t = torch.where(random_mask, replacement, value_t.reshape(B))
            value_t = torch.where(mask & ~random_mask, neutral_t.reshape(B), value_t.reshape(B))
            dropped[key] = value_t

        def _replace_float(key: str, mask: torch.Tensor) -> None:
            value = dropped.get(key)
            neutral_value = neutral.get(key)
            if not isinstance(value, torch.Tensor) or neutral_value is None or not mask.any():
                return
            value_t = value.to(device=device, dtype=model_dtype)
            neutral_t = neutral_value.to(device=device, dtype=model_dtype)
            gate = mask.to(dtype=model_dtype)
            while gate.dim() < value_t.dim():
                gate = gate.unsqueeze(-1)
            dropped[key] = value_t * (1.0 - gate) + neutral_t * gate

        group_masks = {
            'window': _mask(self.semantic_prev_window_dropout_prob),
            'dir': _mask(self.semantic_prev_dir_dropout_prob),
            'phase': _mask(self.semantic_prev_phase_dropout_prob),
            'area': _mask(self.semantic_prev_area_dropout_prob),
            'tempocc': _mask(self.semantic_prev_tempocc_dropout_prob),
            'graph': _mask(self.semantic_prev_graph_dropout_prob),
            'boundary': _mask(self.semantic_prev_boundary_dropout_prob),
            'chase': _mask(self.semantic_prev_chase_dropout_prob),
        }
        _replace_categorical('family', group_masks['window'], 3)
        _replace_categorical('dir', group_masks['dir'], 3)
        _replace_categorical('decision_phase', group_masks['phase'], 2)
        _replace_categorical('control_phase', group_masks['phase'], 4)
        _replace_categorical('status', group_masks['area'], 3)
        for key in ('conflict_area_route_mask', 'conflict_timing_values', 'conflict_timing_valid'):
            _replace_float(key, group_masks['area'])
        for key in (
            'temporary_occupancy_cover_bins',
            'temporary_occupancy_cover_valid',
            'go_opportunity_prob',
            'yld_pressure_prob',
            'go_opportunity_valid',
        ):
            _replace_float(key, group_masks['tempocc'])
        _replace_float('graph_values', group_masks['graph'])
        _replace_float('boundary_values', group_masks['boundary'])
        _replace_float('chase_values', group_masks['chase'])
        return dropped

    def _get_route_prev_coarse_memory(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        model_dtype: torch.dtype,
        require: bool = False,
    ) -> Optional[torch.Tensor]:
        valid = self._get_stage1_batch_tensor(
            batch, 'prev_semantic_state_valid', device=device, model_dtype=model_dtype
        )
        family = self._get_stage1_long_target(batch, 'prev_conflict_area_family', device=device)
        direction = self._get_stage1_long_target(batch, 'prev_conflict_area_dir', device=device)
        if valid is None or family is None or direction is None:
            if require:
                raise ValueError(
                    "route previous coarse memory requires prev_semantic_state_valid, "
                    "prev_conflict_area_family, and prev_conflict_area_dir"
                )
            return None
        valid = torch.nan_to_num(valid.reshape(-1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        window = self._window_target_from_family_codes(family.reshape(-1))
        window_oh = F.one_hot(window.clamp(min=0, max=3), num_classes=4).to(
            device=device, dtype=model_dtype
        )
        dir_oh = F.one_hot(direction.reshape(-1).clamp(min=0, max=3), num_classes=4).to(
            device=device, dtype=model_dtype
        )
        memory = torch.cat([valid.unsqueeze(-1), window_oh, dir_oh], dim=-1)
        return memory * valid.unsqueeze(-1)

    def _route_prev_coarse_memory_from_prev_state(
        self,
        prev_state: Optional[dict],
        *,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Build route-token coarse memory from the closed-loop semantic cache."""
        if not self.use_route_prev_coarse_memory or prev_state is None:
            return None
        valid = prev_state.get('valid')
        family = prev_state.get('family')
        direction = prev_state.get('dir')
        if not (
            isinstance(valid, torch.Tensor)
            and isinstance(family, torch.Tensor)
            and isinstance(direction, torch.Tensor)
        ):
            return None

        valid = torch.nan_to_num(
            valid.to(device=device, dtype=model_dtype).reshape(-1),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        family = family.to(device=device, dtype=torch.long).reshape(-1)
        direction = direction.to(device=device, dtype=torch.long).reshape(-1)

        window = self._window_target_from_family_codes(family)
        window_oh = F.one_hot(window.clamp(min=0, max=3), num_classes=4).to(
            device=device,
            dtype=model_dtype,
        )
        dir_oh = F.one_hot(direction.clamp(min=0, max=3), num_classes=4).to(
            device=device,
            dtype=model_dtype,
        )
        memory = torch.cat([valid.unsqueeze(-1), window_oh, dir_oh], dim=-1)
        return memory * valid.unsqueeze(-1)

    def _compute_semantic_transition_scores_from_shared(
        self,
        shared_forward: dict,
        prev_state: dict,
        speed_samples: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute the temporal semantic branch for the resolved predictor mode."""
        kwargs = dict(
            traj_out=shared_forward['traj_out'],
            route_out=shared_forward['route_out'],
            speed_out=shared_forward['speed_out'],
            route_points=shared_forward['route_points'],
            conditioning=shared_forward['conditioning'],
            prev_state=prev_state,
        )
        if self.semantic_state_predictor_mode == 'direct_prev_modulated':
            return self.model.compute_shared_stage1_prev_modulated_from_ego_outputs(
                **kwargs,
                prev_modulation_scale=self.semantic_prev_modulation_scale,
                prev_modulation_dropout_prob=self.semantic_prev_modulation_dropout_prob,
                speed_samples=speed_samples,
            )
        return self.model.compute_shared_stage1_transition_from_ego_outputs(**kwargs)

    def _build_conflict_area_route_target(
        self,
        batch: Dict[str, torch.Tensor],
        route_steps: int,
        device: torch.device,
        model_dtype: torch.dtype,
    ):
        offline_mask = self._get_stage1_batch_tensor(
            batch, 'conflict_area_route_mask', device=device, model_dtype=model_dtype
        )
        B = None
        fallback_rows = None
        target = None
        valid_target_mask = None
        if offline_mask is not None:
            B = offline_mask.reshape(offline_mask.shape[0], -1).shape[0]
            target = offline_mask.reshape(B, -1)
            if target.shape[1] != route_steps:
                raise ValueError(
                    f"conflict_area_route_mask expects {route_steps} route bins, got {target.shape}"
                )
            target = target.clamp(0.0, 1.0)
            valid_target_mask = torch.ones_like(target, dtype=torch.bool)
            fallback_rows = torch.zeros(B, device=device, dtype=torch.bool)
            return target, valid_target_mask

        conflict_active = self._get_stage1_batch_tensor(
            batch, 'conflict_area_active', device=device, model_dtype=model_dtype
        )
        start_frames = self._get_stage1_long_target(batch, 'conflict_area_start_frame', device=device)
        end_frames = self._get_stage1_long_target(batch, 'conflict_area_end_frame', device=device)
        frame_ids = batch.get('frame_id')
        if conflict_active is None or start_frames is None or end_frames is None or frame_ids is None:
            if target is not None:
                return target, valid_target_mask
            return None, None
        if not isinstance(frame_ids, torch.Tensor):
            frame_ids = torch.as_tensor(frame_ids)
        frame_ids = frame_ids.reshape(-1).to(device=device).long()
        legacy_target = torch.zeros((frame_ids.shape[0], route_steps), device=device, dtype=model_dtype)
        legacy_valid_mask = torch.ones_like(legacy_target, dtype=torch.bool)
        valid_mask = (
            (conflict_active.reshape(-1) > 0.5)
            & (start_frames >= 0)
            & (end_frames >= 0)
            & (frame_ids >= 0)
        )
        if valid_mask.any():
            rel_start = start_frames - frame_ids
            rel_end = end_frames - frame_ids
            active_indices = torch.nonzero(valid_mask, as_tuple=False).flatten().tolist()
            for idx in active_indices:
                start_idx = int(rel_start[idx].item())
                end_idx = int(rel_end[idx].item())
                if end_idx < 0 or start_idx >= route_steps:
                    continue
                # Keep a small tolerance because route tokens may be offset by one step
                # relative to frame-based stage1 annotations.
                start_idx = max(start_idx - 1, 0)
                end_idx = min(end_idx + 1, route_steps - 1)
                if start_idx <= end_idx:
                    legacy_target[idx, start_idx:end_idx + 1] = 1.0
        if target is None:
            return legacy_target, legacy_valid_mask
        target = target.clone()
        valid_target_mask = valid_target_mask.clone()
        target[fallback_rows] = legacy_target[fallback_rows]
        valid_target_mask[fallback_rows] = legacy_valid_mask[fallback_rows]
        return target, valid_target_mask

    def _compose_stage1_outputs(self, raw_scores: dict) -> dict:
        window_probs = torch.softmax(raw_scores['window_logits'], dim=-1)
        dir_probs = torch.softmax(raw_scores['dir_logits'], dim=-1)
        decision_phase_base_logits = raw_scores.get(
            'decision_phase_logits_base',
            raw_scores['decision_phase_logits'],
        )
        decision_phase_probs = torch.softmax(raw_scores['decision_phase_logits'], dim=-1)
        decision_phase_base_probs = torch.softmax(decision_phase_base_logits, dim=-1)
        control_phase_probs = torch.softmax(raw_scores['control_phase_logits'], dim=-1)
        conflict_area_probs = torch.sigmoid(raw_scores['conflict_area_logits'])
        if 'temporary_occupancy_logits' in raw_scores:
            temporary_occupancy_probs = torch.sigmoid(raw_scores['temporary_occupancy_logits'])
        else:
            temporary_occupancy_probs = None
        if 'go_opportunity_logits' in raw_scores:
            go_opportunity_probs = torch.softmax(raw_scores['go_opportunity_logits'], dim=-1)
            yld_pressure_probs = go_opportunity_probs[:, 0]
        else:
            go_opportunity_probs = None
            yld_pressure_probs = None
        if 'conflict_area_status_logits' in raw_scores:
            conflict_area_status_probs = torch.softmax(raw_scores['conflict_area_status_logits'], dim=-1)
        else:
            conflict_area_status_probs = None
        conflict_timing_values = raw_scores.get('conflict_timing_values')
        if conflict_timing_values is not None:
            conflict_dist_to_entry_m = (
                conflict_timing_values[:, 0] * float(self.conflict_timing_dist_norm_scale)
            )
            conflict_dist_to_exit_m = (
                conflict_timing_values[:, 1] * float(self.conflict_timing_dist_norm_scale)
            )
            conflict_time_to_entry_s = (
                conflict_timing_values[:, 2] * float(self.conflict_timing_time_norm_scale)
            )
        else:
            conflict_dist_to_entry_m = None
            conflict_dist_to_exit_m = None
            conflict_time_to_entry_s = None
        if self.use_chase_front_following_state and 'chase_has_lead_logit' in raw_scores:
            chase_has_lead_prob = torch.sigmoid(raw_scores['chase_has_lead_logit'])
        else:
            chase_has_lead_prob = None
        if self.use_chase_front_following_state and 'chase_speed_max' in raw_scores:
            chase_speed_max_mps = self._chase_norm_to_mps(raw_scores['chase_speed_max'])
        else:
            chase_speed_max_mps = None
        current_edge_valid_prob = (
            torch.sigmoid(raw_scores['current_cover_edge_valid_logit'])
            if 'current_cover_edge_valid_logit' in raw_scores else None
        )
        current_edge_mode_probs = (
            torch.softmax(raw_scores['current_cover_edge_mode_logits'], dim=-1)
            if 'current_cover_edge_mode_logits' in raw_scores else None
        )
        future_edge_valid_prob = (
            torch.sigmoid(raw_scores['future_cover_edge_valid_logit'])
            if 'future_cover_edge_valid_logit' in raw_scores else None
        )
        future_edge_mode_probs = (
            torch.softmax(raw_scores['future_cover_edge_mode_logits'], dim=-1)
            if 'future_cover_edge_mode_logits' in raw_scores else None
        )
        current_cover_upper_speed_mps = (
            self._boundary_norm_to_mps(raw_scores['current_cover_upper_speed'])
            if 'current_cover_upper_speed' in raw_scores else None
        )
        future_cover_lower_speed_mps = (
            self._boundary_norm_to_mps(raw_scores['future_cover_lower_speed'])
            if 'future_cover_lower_speed' in raw_scores else None
        )
        front_follow_upper_speed_mps = (
            self._boundary_norm_to_mps(raw_scores['front_follow_upper_speed'])
            if 'front_follow_upper_speed' in raw_scores else None
        )
        merge_flow_lower_speed_mps = (
            self._boundary_norm_to_mps(raw_scores['merge_flow_lower_speed'])
            if 'merge_flow_lower_speed' in raw_scores else None
        )

        same_opp = dir_probs[:, 1:3]
        lane_dir_relation_probs = torch.where(
            same_opp.sum(dim=-1, keepdim=True) > 1e-6,
            same_opp / same_opp.sum(dim=-1, keepdim=True).clamp(min=1e-6),
            torch.full_like(same_opp, 0.5),
        )

        family_probs = window_probs[:, 1:]
        merge_yld_max = self._boundary_norm_to_mps(raw_scores['merge_yld_max'])
        merge_go_min = self._boundary_norm_to_mps(raw_scores['merge_go_min'])
        junction_yld_max = self._boundary_norm_to_mps(raw_scores['junction_yld_max'])
        junction_go_min = self._boundary_norm_to_mps(raw_scores['junction_go_min'])
        borrow_yld_max = self._boundary_norm_to_mps(raw_scores['borrow_yld_max'])
        borrow_go_min = self._boundary_norm_to_mps(raw_scores['borrow_go_min'])
        selected_yld_max = (
            family_probs[:, 0] * merge_yld_max
            + family_probs[:, 1] * junction_yld_max
            + family_probs[:, 2] * borrow_yld_max
        )
        selected_go_min = (
            family_probs[:, 0] * merge_go_min
            + family_probs[:, 1] * junction_go_min
            + family_probs[:, 2] * borrow_go_min
        )

        composed = dict(raw_scores)
        composed.update({
            'window_probs': window_probs,
            'dir_probs': dir_probs,
            'decision_phase_probs': decision_phase_probs,
            'decision_phase_base_probs': decision_phase_base_probs,
            'control_phase_probs': control_phase_probs,
            'conflict_area_probs': conflict_area_probs,
            'temporary_occupancy_probs': temporary_occupancy_probs,
            'go_opportunity_probs': go_opportunity_probs,
            'yld_pressure_probs': yld_pressure_probs,
            'conflict_area_status_probs': conflict_area_status_probs,
            'conflict_timing_values': conflict_timing_values,
            'conflict_dist_to_entry_m': conflict_dist_to_entry_m,
            'conflict_dist_to_exit_m': conflict_dist_to_exit_m,
            'conflict_time_to_entry_s': conflict_time_to_entry_s,
            'chase_has_lead_prob': chase_has_lead_prob,
            'chase_speed_max_mps': chase_speed_max_mps,
            'current_cover_edge_valid_prob': current_edge_valid_prob,
            'current_cover_edge_mode_probs': current_edge_mode_probs,
            'future_cover_edge_valid_prob': future_edge_valid_prob,
            'future_cover_edge_mode_probs': future_edge_mode_probs,
            'current_cover_upper_speed_mps': current_cover_upper_speed_mps,
            'future_cover_lower_speed_mps': future_cover_lower_speed_mps,
            'front_follow_upper_speed_mps': front_follow_upper_speed_mps,
            'merge_flow_lower_speed_mps': merge_flow_lower_speed_mps,
            'lane_dir_relation_probs': lane_dir_relation_probs,
            'merge_yld_max_mps': merge_yld_max,
            'merge_go_min_mps': merge_go_min,
            'junction_yld_max_mps': junction_yld_max,
            'junction_go_min_mps': junction_go_min,
            'borrow_yld_max_mps': borrow_yld_max,
            'borrow_go_min_mps': borrow_go_min,
            'selected_yld_max_mps': selected_yld_max,
            'selected_go_min_mps': selected_go_min,
        })
        return composed

    def _get_traj_branch_condition_gt_prob(self) -> float:
        start = float(self.traj_branch_condition_gt_prob_start)
        end = float(self.traj_branch_condition_gt_prob_end)
        decay_epochs = int(self.traj_branch_condition_gt_decay_epochs)
        if decay_epochs <= 0:
            return start
        progress = min(max(float(self._current_epoch), 0.0) / float(decay_epochs), 1.0)
        return start + (end - start) * progress

    @staticmethod
    def _semantic_state_supervision_groups(profile: str) -> set:
        all_groups = {
            'window',
            'dir',
            'decision',
            'control',
            'boundary',
            'conflict_area',
            'tempocc',
            'opportunity',
            'area_status',
            'timing',
            'inside_area_go',
            'chase',
            'graph',
            'edge_speed_consistency',
        }
        profile_groups = {
            'all': all_groups,
            'window_only': {'window'},
            'window_decision': {'window', 'decision'},
            'window_decision_control': {'window', 'decision', 'control'},
            'window_phase_opportunity': {'window', 'decision', 'control', 'opportunity'},
            'compact_safe': {
                'window',
                'decision',
                'control',
                'opportunity',
                'chase',
                'graph',
                'edge_speed_consistency',
            },
        }
        if profile not in profile_groups:
            raise ValueError(f"Unknown semantic_state_supervision_profile: {profile}")
        return set(profile_groups[profile])

    def _supervise_state_group(self, group: str) -> bool:
        return group in self.semantic_state_supervision_groups

    def _build_traj_condition_schedule(
        self,
        timestep: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> torch.Tensor:
        timestep = timestep.to(device=device, dtype=model_dtype).reshape(-1)
        max_t = max(float(self.train_max_timesteps - 1), 1.0)
        denoise_progress = 1.0 - (timestep / max_t)

        gate_window = torch.where(
            denoise_progress <= 0.35,
            torch.ones_like(denoise_progress),
            1.0 - 0.75 * ((denoise_progress - 0.35) / 0.65).clamp(min=0.0, max=1.0),
        ).clamp_(0.25, 1.0)
        gate_dir = gate_window
        gate_phase = (denoise_progress / 0.6).clamp_(0.0, 1.0)
        gate_boundary = ((denoise_progress - 0.5) / 0.5).clamp_(0.0, 1.0)
        gate_borrow = ((denoise_progress - 0.7) / 0.3).clamp_(0.0, 1.0)
        return torch.stack(
            [
                gate_window,
                gate_dir,
                gate_phase,
                gate_boundary,
                gate_phase,     # go/yld opportunity prior
                gate_window,    # area status is coarse spatial state
                gate_boundary,  # dist/time matters more in refinement
                gate_borrow,
            ],
            dim=-1,
        )

    def _build_local_phase_energy_samples(
        self,
        center_speed: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> torch.Tensor:
        center_speed = center_speed.to(device=device, dtype=model_dtype).reshape(-1, 1)
        band = center_speed + self.traj_phase_energy_band_offsets.to(
            device=device, dtype=model_dtype
        ).view(1, -1)
        return band.clamp_(0.0, 20.0)

    def _pool_curve_near_speed_band(
        self,
        curve: Optional[torch.Tensor],
        speed_samples: torch.Tensor,
        center_speed: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if curve is None:
            return None
        curve = curve.to(device=device, dtype=model_dtype)
        speed_samples = speed_samples.to(device=device, dtype=model_dtype)
        query_band = self._build_local_phase_energy_samples(center_speed, device, model_dtype)
        nearest_index = torch.argmin(
            torch.abs(speed_samples.unsqueeze(-1) - query_band.unsqueeze(1)),
            dim=1,
        )
        pooled = torch.gather(curve, dim=-1, index=nearest_index).mean(dim=-1)
        return (1.0 - pooled.clamp(0.0, 1.0)).clamp_(0.0, 1.0)

    def _normalize_prob_rows(
        self,
        scores: torch.Tensor,
        fallback_index: int = 0,
    ) -> torch.Tensor:
        score_sum = scores.sum(dim=-1, keepdim=True)
        fallback = torch.zeros_like(scores)
        fallback[:, fallback_index] = 1.0
        return torch.where(
            score_sum > 1e-6,
            scores / score_sum.clamp(min=1e-6),
            fallback,
        )

    def _compose_stage1_branch_condition(
        self,
        *,
        window_probs: torch.Tensor,
        dir_probs: torch.Tensor,
        decision_phase_probs: torch.Tensor,
        control_phase_probs: torch.Tensor,
        merge_yld_max_mps: torch.Tensor,
        merge_go_min_mps: torch.Tensor,
        junction_yld_max_mps: torch.Tensor,
        junction_go_min_mps: torch.Tensor,
        borrow_yld_max_mps: torch.Tensor,
        borrow_go_min_mps: torch.Tensor,
        merge_yld_valid: torch.Tensor,
        merge_go_valid: torch.Tensor,
        junction_yld_valid: torch.Tensor,
        junction_go_valid: torch.Tensor,
        borrow_yld_valid: torch.Tensor,
        borrow_go_valid: torch.Tensor,
        center_speed: torch.Tensor,
        go_opportunity_probs: Optional[torch.Tensor],
        conflict_area_status_probs: Optional[torch.Tensor],
        conflict_timing_values: Optional[torch.Tensor],
        chase_has_lead_prob: Optional[torch.Tensor],
        chase_speed_max_mps: Optional[torch.Tensor],
        borrow_time_s: Optional[torch.Tensor],
        device: torch.device,
        model_dtype: torch.dtype,
        current_edge_valid_prob: Optional[torch.Tensor] = None,
        current_edge_mode_probs: Optional[torch.Tensor] = None,
        current_cover_upper_speed_mps: Optional[torch.Tensor] = None,
        current_cover_upper_valid: Optional[torch.Tensor] = None,
        future_edge_valid_prob: Optional[torch.Tensor] = None,
        future_edge_mode_probs: Optional[torch.Tensor] = None,
        future_cover_lower_speed_mps: Optional[torch.Tensor] = None,
        future_cover_lower_valid: Optional[torch.Tensor] = None,
        front_follow_upper_speed_mps: Optional[torch.Tensor] = None,
        front_follow_upper_valid: Optional[torch.Tensor] = None,
        merge_flow_lower_speed_mps: Optional[torch.Tensor] = None,
        merge_flow_lower_valid: Optional[torch.Tensor] = None,
    ):
        window_probs = self._normalize_prob_rows(
            window_probs.to(device=device, dtype=model_dtype),
            fallback_index=0,
        )
        dir_probs = self._normalize_prob_rows(
            dir_probs.to(device=device, dtype=model_dtype),
            fallback_index=0,
        )
        decision_phase_probs = decision_phase_probs.to(device=device, dtype=model_dtype)
        control_phase_probs = control_phase_probs.to(device=device, dtype=model_dtype)
        center_speed = center_speed.to(device=device, dtype=model_dtype).reshape(-1)
        B = window_probs.shape[0]
        if go_opportunity_probs is None:
            go_opportunity_probs = torch.full((B, 2), 0.5, device=device, dtype=model_dtype)
        else:
            go_opportunity_probs = self._normalize_prob_rows(
                go_opportunity_probs.to(device=device, dtype=model_dtype),
                fallback_index=1,
            )
        if conflict_area_status_probs is None:
            conflict_area_status_probs = torch.zeros((B, 4), device=device, dtype=model_dtype)
            conflict_area_status_probs[:, 0] = 1.0
        else:
            conflict_area_status_probs = self._normalize_prob_rows(
                conflict_area_status_probs.to(device=device, dtype=model_dtype),
                fallback_index=0,
            )
        if conflict_timing_values is None:
            conflict_timing_values = torch.zeros((B, 3), device=device, dtype=model_dtype)
        else:
            conflict_timing_values = torch.nan_to_num(
                conflict_timing_values.to(device=device, dtype=model_dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).reshape(B, -1)
            if conflict_timing_values.shape[-1] != 3:
                raise ValueError(
                    f"conflict_timing_values expects 3 dims, got {conflict_timing_values.shape}"
            )
            conflict_timing_values = conflict_timing_values.clamp(-2.0, 2.0)
        if chase_has_lead_prob is None:
            chase_has_lead_prob = torch.zeros((B,), device=device, dtype=model_dtype)
        else:
            chase_has_lead_prob = torch.nan_to_num(
                chase_has_lead_prob.to(device=device, dtype=model_dtype).reshape(-1),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
        if chase_speed_max_mps is None:
            chase_speed_margin = torch.zeros((B,), device=device, dtype=model_dtype)
        else:
            chase_speed_max_mps = torch.nan_to_num(
                chase_speed_max_mps.to(device=device, dtype=model_dtype).reshape(-1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            chase_speed_margin = (
                (chase_speed_max_mps - center_speed)
                / max(self.traj_branch_condition_chase_margin_scale, 1e-6)
            ).clamp(-1.0, 1.0)
        chase_condition = torch.stack(
            [chase_has_lead_prob, chase_speed_margin],
            dim=-1,
        )

        def _prob1(value: Optional[torch.Tensor]) -> torch.Tensor:
            if value is None:
                return torch.zeros((B,), device=device, dtype=model_dtype)
            return torch.nan_to_num(
                value.to(device=device, dtype=model_dtype).reshape(-1),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)

        def _mode_probs(value: Optional[torch.Tensor]) -> torch.Tensor:
            if value is None:
                out = torch.zeros((B, 5), device=device, dtype=model_dtype)
                out[:, 0] = 1.0
                return out
            value = value.to(device=device, dtype=model_dtype).reshape(B, -1)
            if value.shape[-1] != 5:
                raise ValueError(f"edge mode probs expects (B, 5), got {value.shape}")
            return self._normalize_prob_rows(value, fallback_index=0)

        def _speed_margin(
            speed: Optional[torch.Tensor],
            valid: Optional[torch.Tensor],
            *,
            upper: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            valid_prob = _prob1(valid)
            if speed is None:
                margin = torch.zeros((B,), device=device, dtype=model_dtype)
            else:
                speed = torch.nan_to_num(
                    speed.to(device=device, dtype=model_dtype).reshape(-1),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if upper:
                    margin = (
                        (speed - center_speed)
                        / max(self.traj_branch_condition_boundary_margin_scale, 1e-6)
                    )
                else:
                    margin = (
                        (center_speed - speed)
                        / max(self.traj_branch_condition_boundary_margin_scale, 1e-6)
                    )
                margin = margin.clamp(-1.0, 1.0)
            return margin * valid_prob, valid_prob

        current_edge_valid_prob = _prob1(current_edge_valid_prob)
        future_edge_valid_prob = _prob1(future_edge_valid_prob)
        current_edge_mode_probs = _mode_probs(current_edge_mode_probs)
        future_edge_mode_probs = _mode_probs(future_edge_mode_probs)
        current_upper_margin, current_upper_valid_prob = _speed_margin(
            current_cover_upper_speed_mps,
            current_cover_upper_valid,
            upper=True,
        )
        future_lower_margin, future_lower_valid_prob = _speed_margin(
            future_cover_lower_speed_mps,
            future_cover_lower_valid,
            upper=False,
        )
        front_follow_upper_margin, front_follow_upper_valid_prob = _speed_margin(
            front_follow_upper_speed_mps,
            front_follow_upper_valid,
            upper=True,
        )
        merge_flow_lower_margin, merge_flow_lower_valid_prob = _speed_margin(
            merge_flow_lower_speed_mps,
            merge_flow_lower_valid,
            upper=False,
        )
        current_edge_condition = torch.cat(
            [current_edge_valid_prob.unsqueeze(-1), current_edge_mode_probs],
            dim=-1,
        )
        future_edge_condition = torch.cat(
            [future_edge_valid_prob.unsqueeze(-1), future_edge_mode_probs],
            dim=-1,
        )
        edge_speed_margins = torch.stack(
            [
                current_upper_margin,
                future_lower_margin,
                front_follow_upper_margin,
                merge_flow_lower_margin,
            ],
            dim=-1,
        )
        edge_speed_valids = torch.stack(
            [
                current_upper_valid_prob,
                future_lower_valid_prob,
                front_follow_upper_valid_prob,
                merge_flow_lower_valid_prob,
            ],
            dim=-1,
        )

        family_probs = window_probs[:, 1:]
        yld_stack = torch.stack(
            [merge_yld_max_mps, junction_yld_max_mps, borrow_yld_max_mps],
            dim=-1,
        ).to(device=device, dtype=model_dtype)
        go_stack = torch.stack(
            [merge_go_min_mps, junction_go_min_mps, borrow_go_min_mps],
            dim=-1,
        ).to(device=device, dtype=model_dtype)
        yld_valid_stack = torch.stack(
            [merge_yld_valid, junction_yld_valid, borrow_yld_valid],
            dim=-1,
        ).to(device=device, dtype=model_dtype).clamp(0.0, 1.0)
        go_valid_stack = torch.stack(
            [merge_go_valid, junction_go_valid, borrow_go_valid],
            dim=-1,
        ).to(device=device, dtype=model_dtype).clamp(0.0, 1.0)
        # Boundary labels use NaN outside valid regions. Sanitize them before any
        # weighted aggregation; otherwise 0 * NaN still becomes NaN and pollutes
        # the branch condition fed back into the main diffusion path.
        yld_stack = torch.where(
            yld_valid_stack > 0.5,
            torch.nan_to_num(yld_stack, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(yld_stack),
        )
        go_stack = torch.where(
            go_valid_stack > 0.5,
            torch.nan_to_num(go_stack, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(go_stack),
        )
        weighted_family_yld = family_probs * yld_valid_stack
        weighted_family_go = family_probs * go_valid_stack
        weighted_yld_sum = weighted_family_yld.sum(dim=-1, keepdim=True)
        weighted_go_sum = weighted_family_go.sum(dim=-1, keepdim=True)
        zero = torch.zeros_like(center_speed)
        selected_yld_max = torch.where(
            weighted_yld_sum.squeeze(-1) > 1e-6,
            (weighted_family_yld * yld_stack).sum(dim=-1) / weighted_yld_sum.squeeze(-1).clamp(min=1e-6),
            zero,
        )
        selected_go_min = torch.where(
            weighted_go_sum.squeeze(-1) > 1e-6,
            (weighted_family_go * go_stack).sum(dim=-1) / weighted_go_sum.squeeze(-1).clamp(min=1e-6),
            zero,
        )
        boundary_margins = torch.stack(
            [
                (selected_yld_max - center_speed) / max(self.traj_branch_condition_boundary_margin_scale, 1e-6),
                (center_speed - selected_go_min) / max(self.traj_branch_condition_boundary_margin_scale, 1e-6),
            ],
            dim=-1,
        ).clamp_(-1.0, 1.0)
        no_valid_boundary = (weighted_yld_sum.squeeze(-1) <= 1e-6) & (weighted_go_sum.squeeze(-1) <= 1e-6)
        if no_valid_boundary.any():
            boundary_margins = boundary_margins.clone()
            boundary_margins[no_valid_boundary] = 0.0

        same_opp = dir_probs[:, 1:3]
        lane_dir_relation_probs = torch.where(
            same_opp.sum(dim=-1, keepdim=True) > 1e-6,
            same_opp / same_opp.sum(dim=-1, keepdim=True).clamp(min=1e-6),
            torch.full_like(same_opp, 0.5),
        )

        borrow_window_score = window_probs[:, 3]
        if borrow_time_s is None:
            borrow_time_cond = zero
        else:
            borrow_time_cond = borrow_time_s.to(device=device, dtype=model_dtype).reshape(-1)
            borrow_time_cond = torch.clamp(
                borrow_time_cond / max(self.traj_branch_condition_borrow_time_scale, 1e-6),
                min=0.0,
                max=1.0,
            ) * borrow_window_score

        if self.semantic_motion_condition_mode == 'compact_graph':
            branch_condition = torch.cat(
                [
                    window_probs,
                    decision_phase_probs,
                    control_phase_probs,
                    go_opportunity_probs,
                    current_edge_condition,
                    future_edge_condition,
                    edge_speed_margins,
                    edge_speed_valids,
                    borrow_time_cond.unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            branch_condition = torch.cat(
                [
                    window_probs,
                    dir_probs,
                    decision_phase_probs,
                    control_phase_probs,
                    boundary_margins,
                    go_opportunity_probs,
                    conflict_area_status_probs,
                    conflict_timing_values,
                    chase_condition,
                    borrow_time_cond.unsqueeze(-1),
                ],
                dim=-1,
            )
        details = {
            'window_probs': window_probs,
            'dir_probs': dir_probs,
            'decision_phase_probs': decision_phase_probs,
            'control_phase_probs': control_phase_probs,
            'boundary_margins': boundary_margins,
            'go_opportunity_probs': go_opportunity_probs,
            'conflict_area_status_probs': conflict_area_status_probs,
            'conflict_timing_values': conflict_timing_values,
            'chase_has_lead_prob': chase_has_lead_prob,
            'chase_speed_margin': chase_speed_margin,
            'chase_speed_max_mps': chase_speed_max_mps,
            'current_edge_valid_prob': current_edge_valid_prob,
            'current_edge_mode_probs': current_edge_mode_probs,
            'future_edge_valid_prob': future_edge_valid_prob,
            'future_edge_mode_probs': future_edge_mode_probs,
            'edge_speed_margins': edge_speed_margins,
            'edge_speed_valids': edge_speed_valids,
            'current_upper_margin': current_upper_margin,
            'future_lower_margin': future_lower_margin,
            'front_follow_upper_margin': front_follow_upper_margin,
            'merge_flow_lower_margin': merge_flow_lower_margin,
            'borrow_time_condition': borrow_time_cond,
            'lane_dir_relation_probs': lane_dir_relation_probs,
            'selected_yld_max_mps': selected_yld_max,
            'selected_go_min_mps': selected_go_min,
        }
        return branch_condition, details

    def _apply_phase_go_smoothing_override(
        self,
        branch_condition: Optional[torch.Tensor],
        details: Optional[Dict[str, torch.Tensor]],
        *,
        update_history: bool,
        device: torch.device,
        model_dtype: torch.dtype,
    ):
        debug = {
            'enabled': float(self.phase_go_smoothing_enable),
            'applied': 0.0,
            'raw_go_prob': float('nan'),
            'smoothed_go_prob': float('nan'),
            'history_len': float(len(self.phase_go_smoothing_history)),
            'threshold': float(self.phase_go_smoothing_threshold),
        }
        if not self.phase_go_smoothing_enable or branch_condition is None or details is None:
            return branch_condition, details, debug, False

        decision_phase_probs = details.get('decision_phase_probs')
        if decision_phase_probs is None or decision_phase_probs.shape[-1] < 2:
            return branch_condition, details, debug, False

        source = self.phase_go_smoothing_source
        go_prob_tensor = None
        if source in ('decision', 'decision_phase', 'phase'):
            go_prob_tensor = decision_phase_probs[:, 1]
        else:
            go_opportunity_probs = details.get('go_opportunity_probs')
            if go_opportunity_probs is not None and go_opportunity_probs.shape[-1] >= 2:
                go_prob_tensor = go_opportunity_probs[:, 1]
            else:
                go_prob_tensor = decision_phase_probs[:, 1]

        raw_go_prob = float(go_prob_tensor.detach().float().reshape(-1)[0].item())
        if update_history:
            self.phase_go_smoothing_history.append(raw_go_prob)
        if len(self.phase_go_smoothing_history) > 0:
            smoothed_go_prob = float(
                sum(self.phase_go_smoothing_history) / len(self.phase_go_smoothing_history)
            )
        else:
            smoothed_go_prob = raw_go_prob

        applied = smoothed_go_prob > self.phase_go_smoothing_threshold
        debug.update({
            'applied': float(applied),
            'raw_go_prob': raw_go_prob,
            'smoothed_go_prob': smoothed_go_prob,
            'history_len': float(len(self.phase_go_smoothing_history)),
        })
        if not applied:
            return branch_condition, details, debug, update_history

        if self.semantic_motion_condition_mode == 'compact_graph':
            decision_start = len(self.traj_window_condition_names)
        else:
            decision_start = len(self.traj_window_condition_names) + len(self.traj_dir_condition_names)
        decision_end = decision_start + len(self.traj_decision_phase_condition_names)
        decision_override = torch.zeros_like(decision_phase_probs)
        decision_override[:, 1] = 1.0

        new_details = dict(details)
        new_details['decision_phase_probs_before_go_smoothing'] = decision_phase_probs
        new_details['decision_phase_probs'] = decision_override

        new_branch_condition = branch_condition.clone()
        new_branch_condition[:, decision_start:decision_end] = decision_override.to(
            device=device,
            dtype=model_dtype,
        )
        return new_branch_condition, new_details, debug, update_history

    def _build_stage1_branch_condition_gt(
        self,
        batch: Dict[str, torch.Tensor],
        ego_status: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype,
    ):
        family_codes = self._get_stage1_long_target(batch, 'conflict_area_family', device=device)
        dir_codes = self._get_stage1_long_target(batch, 'conflict_area_dir', device=device)
        decision_phase_codes = self._get_stage1_long_target(batch, 'conflict_decision_phase', device=device)
        control_phase_codes = self._get_stage1_long_target(batch, 'conflict_control_phase', device=device)
        if family_codes is None or dir_codes is None or decision_phase_codes is None or control_phase_codes is None:
            return None, None

        window_target = self._window_target_from_family_codes(family_codes)
        window_probs = F.one_hot(window_target, num_classes=4).to(device=device, dtype=model_dtype)
        dir_probs = F.one_hot(dir_codes.clamp(min=0, max=3), num_classes=4).to(device=device, dtype=model_dtype)

        decision_phase_probs = torch.full(
            (family_codes.shape[0], 2), 0.5, device=device, dtype=model_dtype
        )
        decision_active = decision_phase_codes > 0
        if decision_active.any():
            decision_phase_probs[decision_active] = F.one_hot(
                (decision_phase_codes[decision_active] - 1).clamp(min=0, max=1),
                num_classes=2,
            ).to(device=device, dtype=model_dtype)

        control_phase_probs = torch.full(
            (family_codes.shape[0], 4), 0.25, device=device, dtype=model_dtype
        )
        control_active = control_phase_codes > 0
        if control_active.any():
            control_phase_probs[control_active] = F.one_hot(
                (control_phase_codes[control_active] - 1).clamp(min=0, max=3),
                num_classes=4,
            ).to(device=device, dtype=model_dtype)

        merge_yld_max = self._get_stage1_batch_tensor(batch, 'merge_yld_max_speed', device=device, model_dtype=model_dtype)
        merge_go_min = self._get_stage1_batch_tensor(batch, 'merge_go_min_speed', device=device, model_dtype=model_dtype)
        junction_yld_max = self._get_stage1_batch_tensor(batch, 'junction_yld_max_speed', device=device, model_dtype=model_dtype)
        junction_go_min = self._get_stage1_batch_tensor(batch, 'junction_go_min_speed', device=device, model_dtype=model_dtype)
        borrow_yld_max = self._get_stage1_batch_tensor(batch, 'borrow_yld_max_speed', device=device, model_dtype=model_dtype)
        borrow_go_min = self._get_stage1_batch_tensor(batch, 'borrow_go_min_speed', device=device, model_dtype=model_dtype)
        boundary_targets = (
            merge_yld_max,
            merge_go_min,
            junction_yld_max,
            junction_go_min,
            borrow_yld_max,
            borrow_go_min,
        )
        if any(target is None for target in boundary_targets):
            return None, None
        merge_yld_valid = torch.ones_like(merge_yld_max, device=device, dtype=model_dtype)
        merge_go_valid = torch.ones_like(merge_go_min, device=device, dtype=model_dtype)
        junction_yld_valid = torch.ones_like(junction_yld_max, device=device, dtype=model_dtype)
        junction_go_valid = torch.ones_like(junction_go_min, device=device, dtype=model_dtype)
        borrow_yld_valid = torch.ones_like(borrow_yld_max, device=device, dtype=model_dtype)
        borrow_go_valid = torch.ones_like(borrow_go_min, device=device, dtype=model_dtype)
        borrow_time_s = self._get_stage1_borrow_time_target(batch, device=device, model_dtype=model_dtype)
        center_speed = ego_status[:, -1, 0].to(device=device, dtype=model_dtype)
        temp_targets = self._get_temporary_occupancy_targets(
            batch,
            device=device,
            model_dtype=model_dtype,
            require=self.use_temporary_occupancy_phase,
        )
        go_opportunity_probs = (
            temp_targets['go_opportunity_target'] if temp_targets is not None else None
        )
        timing_targets = self._get_conflict_timing_targets(
            batch,
            device=device,
            model_dtype=model_dtype,
            family_codes=family_codes,
            require=self.use_conflict_timing_state,
        )
        if timing_targets is not None:
            conflict_area_status_probs = F.one_hot(
                timing_targets['status'].clamp(min=0, max=3),
                num_classes=4,
            ).to(device=device, dtype=model_dtype)
            conflict_timing_values = timing_targets['values']
        else:
            conflict_area_status_probs = None
            conflict_timing_values = None
        chase_targets = (
            self._get_chase_targets(
                batch,
                device=device,
                model_dtype=model_dtype,
                require=True,
            )
            if self.use_chase_front_following_state else None
        )
        graph_targets = self._get_cover_relation_graph_targets(
            batch,
            device=device,
            model_dtype=model_dtype,
            require=self.use_cover_relation_graph_decoder,
        )
        if graph_targets is not None:
            current_edge_mode_probs = F.one_hot(
                graph_targets['current_mode'].clamp(min=0, max=4),
                num_classes=5,
            ).to(device=device, dtype=model_dtype)
            future_edge_mode_probs = F.one_hot(
                graph_targets['future_mode'].clamp(min=0, max=4),
                num_classes=5,
            ).to(device=device, dtype=model_dtype)
        else:
            current_edge_mode_probs = None
            future_edge_mode_probs = None

        return self._compose_stage1_branch_condition(
            window_probs=window_probs,
            dir_probs=dir_probs,
            decision_phase_probs=decision_phase_probs,
            control_phase_probs=control_phase_probs,
            merge_yld_max_mps=merge_yld_max,
            merge_go_min_mps=merge_go_min,
            junction_yld_max_mps=junction_yld_max,
            junction_go_min_mps=junction_go_min,
            borrow_yld_max_mps=borrow_yld_max,
            borrow_go_min_mps=borrow_go_min,
            merge_yld_valid=merge_yld_valid,
            merge_go_valid=merge_go_valid,
            junction_yld_valid=junction_yld_valid,
            junction_go_valid=junction_go_valid,
            borrow_yld_valid=borrow_yld_valid,
            borrow_go_valid=borrow_go_valid,
            center_speed=center_speed,
            go_opportunity_probs=go_opportunity_probs,
            conflict_area_status_probs=conflict_area_status_probs,
            conflict_timing_values=conflict_timing_values,
            chase_has_lead_prob=(
                chase_targets['has_lead'] if chase_targets is not None else None
            ),
            chase_speed_max_mps=(
                chase_targets['speed_max'] if chase_targets is not None else None
            ),
            borrow_time_s=borrow_time_s,
            device=device,
            model_dtype=model_dtype,
            current_edge_valid_prob=(
                graph_targets['current_valid'] if graph_targets is not None else None
            ),
            current_edge_mode_probs=current_edge_mode_probs,
            current_cover_upper_speed_mps=(
                graph_targets['current_upper'] if graph_targets is not None else None
            ),
            current_cover_upper_valid=(
                graph_targets['current_upper_valid'].to(dtype=model_dtype)
                if graph_targets is not None else None
            ),
            future_edge_valid_prob=(
                graph_targets['future_valid'] if graph_targets is not None else None
            ),
            future_edge_mode_probs=future_edge_mode_probs,
            future_cover_lower_speed_mps=(
                graph_targets['future_lower'] if graph_targets is not None else None
            ),
            future_cover_lower_valid=(
                graph_targets['future_lower_valid'].to(dtype=model_dtype)
                if graph_targets is not None else None
            ),
            front_follow_upper_speed_mps=(
                graph_targets['front_follow_upper'] if graph_targets is not None else None
            ),
            front_follow_upper_valid=(
                graph_targets['front_follow_upper_valid'].to(dtype=model_dtype)
                if graph_targets is not None else None
            ),
            merge_flow_lower_speed_mps=(
                graph_targets['merge_flow_lower'] if graph_targets is not None else None
            ),
            merge_flow_lower_valid=(
                graph_targets['merge_flow_lower_valid'].to(dtype=model_dtype)
                if graph_targets is not None else None
            ),
        )

    def _build_traj_branch_condition_from_stage1_raw(
        self,
        *,
        raw_scores: dict,
        speed_ref: torch.Tensor,
        borrow_time_s: Optional[torch.Tensor],
        prev_relation_probs: Optional[torch.Tensor],
        device: torch.device,
        model_dtype: torch.dtype,
    ):
        if 'dir_logits' not in raw_scores:
            raise ValueError("shared stage1 branch condition expects dir_logits in raw_scores")

        window_probs = torch.softmax(raw_scores['window_logits'], dim=-1)
        dir_probs = torch.softmax(raw_scores['dir_logits'], dim=-1)
        relation_probs = torch.where(
            dir_probs[:, 1:3].sum(dim=-1, keepdim=True) > 1e-6,
            dir_probs[:, 1:3] / dir_probs[:, 1:3].sum(dim=-1, keepdim=True).clamp(min=1e-6),
            torch.full_like(dir_probs[:, 1:3], 0.5),
        )
        if prev_relation_probs is not None:
            prev_relation_probs = prev_relation_probs.to(device=device, dtype=model_dtype)
            if prev_relation_probs.dim() == 2 and prev_relation_probs.shape[-1] == 2:
                same_smoothed = 0.7 * relation_probs[:, 0] + 0.3 * prev_relation_probs[:, 0]
                opposite_smoothed = 0.45 * relation_probs[:, 1] + 0.55 * prev_relation_probs[:, 1]
                relation_probs = torch.stack([same_smoothed, opposite_smoothed], dim=-1)
                relation_probs = relation_probs / relation_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                dir_probs = dir_probs.clone()
                same_opp_mass = dir_probs[:, 1:3].sum(dim=-1, keepdim=True)
                dir_probs[:, 1:3] = relation_probs * same_opp_mass

        return self._compose_stage1_branch_condition(
            window_probs=window_probs,
            dir_probs=dir_probs,
            decision_phase_probs=torch.softmax(raw_scores['decision_phase_logits'], dim=-1),
            control_phase_probs=torch.softmax(raw_scores['control_phase_logits'], dim=-1),
            merge_yld_max_mps=self._boundary_norm_to_mps(raw_scores['merge_yld_max']),
            merge_go_min_mps=self._boundary_norm_to_mps(raw_scores['merge_go_min']),
            junction_yld_max_mps=self._boundary_norm_to_mps(raw_scores['junction_yld_max']),
            junction_go_min_mps=self._boundary_norm_to_mps(raw_scores['junction_go_min']),
            borrow_yld_max_mps=self._boundary_norm_to_mps(raw_scores['borrow_yld_max']),
            borrow_go_min_mps=self._boundary_norm_to_mps(raw_scores['borrow_go_min']),
            merge_yld_valid=torch.ones_like(raw_scores['merge_yld_max'], device=device, dtype=model_dtype),
            merge_go_valid=torch.ones_like(raw_scores['merge_go_min'], device=device, dtype=model_dtype),
            junction_yld_valid=torch.ones_like(raw_scores['junction_yld_max'], device=device, dtype=model_dtype),
            junction_go_valid=torch.ones_like(raw_scores['junction_go_min'], device=device, dtype=model_dtype),
            borrow_yld_valid=torch.ones_like(raw_scores['borrow_yld_max'], device=device, dtype=model_dtype),
            borrow_go_valid=torch.ones_like(raw_scores['borrow_go_min'], device=device, dtype=model_dtype),
            center_speed=speed_ref.to(device=device, dtype=model_dtype).reshape(-1),
            go_opportunity_probs=(
                torch.softmax(raw_scores['go_opportunity_logits'], dim=-1)
                if 'go_opportunity_logits' in raw_scores else None
            ),
            conflict_area_status_probs=(
                torch.softmax(raw_scores['conflict_area_status_logits'], dim=-1)
                if 'conflict_area_status_logits' in raw_scores else None
            ),
            conflict_timing_values=raw_scores.get('conflict_timing_values'),
            chase_has_lead_prob=(
                torch.sigmoid(raw_scores['chase_has_lead_logit'])
                if self.use_chase_front_following_state and 'chase_has_lead_logit' in raw_scores else None
            ),
            chase_speed_max_mps=(
                self._chase_norm_to_mps(raw_scores['chase_speed_max'])
                if self.use_chase_front_following_state and 'chase_speed_max' in raw_scores else None
            ),
            borrow_time_s=borrow_time_s,
            device=device,
            model_dtype=model_dtype,
            current_edge_valid_prob=(
                torch.sigmoid(raw_scores['current_cover_edge_valid_logit'])
                if 'current_cover_edge_valid_logit' in raw_scores else None
            ),
            current_edge_mode_probs=(
                torch.softmax(raw_scores['current_cover_edge_mode_logits'], dim=-1)
                if 'current_cover_edge_mode_logits' in raw_scores else None
            ),
            current_cover_upper_speed_mps=(
                self._boundary_norm_to_mps(raw_scores['current_cover_upper_speed'])
                if 'current_cover_upper_speed' in raw_scores else None
            ),
            current_cover_upper_valid=(
                torch.sigmoid(raw_scores['current_cover_edge_valid_logit'])
                if 'current_cover_edge_valid_logit' in raw_scores else None
            ),
            future_edge_valid_prob=(
                torch.sigmoid(raw_scores['future_cover_edge_valid_logit'])
                if 'future_cover_edge_valid_logit' in raw_scores else None
            ),
            future_edge_mode_probs=(
                torch.softmax(raw_scores['future_cover_edge_mode_logits'], dim=-1)
                if 'future_cover_edge_mode_logits' in raw_scores else None
            ),
            future_cover_lower_speed_mps=(
                self._boundary_norm_to_mps(raw_scores['future_cover_lower_speed'])
                if 'future_cover_lower_speed' in raw_scores else None
            ),
            future_cover_lower_valid=(
                torch.sigmoid(raw_scores['future_cover_edge_valid_logit'])
                if 'future_cover_edge_valid_logit' in raw_scores else None
            ),
            front_follow_upper_speed_mps=(
                self._boundary_norm_to_mps(raw_scores['front_follow_upper_speed'])
                if 'front_follow_upper_speed' in raw_scores else None
            ),
            front_follow_upper_valid=(
                torch.sigmoid(raw_scores['chase_has_lead_logit'])
                if 'chase_has_lead_logit' in raw_scores else None
            ),
            merge_flow_lower_speed_mps=(
                self._boundary_norm_to_mps(raw_scores['merge_flow_lower_speed'])
                if 'merge_flow_lower_speed' in raw_scores else None
            ),
            merge_flow_lower_valid=(
                torch.sigmoid(raw_scores['future_cover_edge_valid_logit'])
                if 'future_cover_edge_valid_logit' in raw_scores else None
            ),
        )

    def reset_semantic_state_cache(self) -> None:
        """Clear the inference-only semantic transition memory."""
        self._semantic_state_cache = None
        self._semantic_state_cache_frame = 0

    def _build_neutral_semantic_prev_state(
        self,
        batch_size: int,
        route_steps: int,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> dict:
        zeros_1 = torch.zeros((batch_size,), device=device, dtype=model_dtype)
        zeros_route = torch.zeros((batch_size, route_steps), device=device, dtype=model_dtype)
        zeros_temp = torch.zeros((batch_size, self.temporary_occupancy_dim), device=device, dtype=model_dtype)
        return {
            'valid': zeros_1,
            'family': torch.zeros((batch_size,), device=device, dtype=torch.long),
            'dir': torch.zeros((batch_size,), device=device, dtype=torch.long),
            'status': torch.zeros((batch_size,), device=device, dtype=torch.long),
            'decision_phase': torch.zeros((batch_size,), device=device, dtype=torch.long),
            'control_phase': torch.zeros((batch_size,), device=device, dtype=torch.long),
            'conflict_area_route_mask': zeros_route,
            'temporary_occupancy_cover_bins': zeros_temp,
            'temporary_occupancy_cover_valid': zeros_temp,
            'go_opportunity_prob': torch.full((batch_size, 1), 0.5, device=device, dtype=model_dtype),
            'yld_pressure_prob': torch.full((batch_size, 1), 0.5, device=device, dtype=model_dtype),
            'go_opportunity_valid': torch.zeros((batch_size, 1), device=device, dtype=model_dtype),
            'conflict_timing_values': torch.zeros((batch_size, 3), device=device, dtype=model_dtype),
            'conflict_timing_valid': torch.zeros((batch_size, 1), device=device, dtype=model_dtype),
            'boundary_values': torch.zeros((batch_size, 6), device=device, dtype=model_dtype),
            'chase_values': torch.zeros((batch_size, 2), device=device, dtype=model_dtype),
            'graph_values': torch.zeros((batch_size, 20), device=device, dtype=model_dtype),
        }

    def _semantic_state_cache_to_device(
        self,
        cache: dict,
        *,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> dict:
        out = {}
        for key, value in cache.items():
            if not isinstance(value, torch.Tensor):
                out[key] = value
            elif value.dtype.is_floating_point:
                out[key] = value.to(device=device, dtype=model_dtype)
            else:
                out[key] = value.to(device=device)
        return out

    def _get_inference_semantic_prev_state(
        self,
        batch_size: int,
        route_steps: int,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> dict:
        cache = self._semantic_state_cache
        if cache is None:
            return self._build_neutral_semantic_prev_state(
                batch_size, route_steps, device, model_dtype
            )
        valid = cache.get('valid')
        if not isinstance(valid, torch.Tensor) or valid.reshape(-1).shape[0] != batch_size:
            return self._build_neutral_semantic_prev_state(
                batch_size, route_steps, device, model_dtype
            )
        return self._semantic_state_cache_to_device(
            cache, device=device, model_dtype=model_dtype
        )

    def _stage1_raw_scores_to_prev_state(
        self,
        raw_scores: dict,
        *,
        route_steps: int,
        device: torch.device,
        model_dtype: torch.dtype,
        valid: Optional[torch.Tensor] = None,
    ) -> dict:
        window_probs = torch.softmax(raw_scores['window_logits'].detach().float(), dim=-1)
        window_class = window_probs.argmax(dim=-1)
        # Window head order is none/merge/junction/borrow, while label-family
        # order is none/borrow/merge/junction.
        family_lookup = torch.tensor([0, 2, 3, 1], device=window_class.device, dtype=torch.long)
        family = family_lookup[window_class.clamp(min=0, max=3)]
        active = (window_class > 0).long()

        dir_class = torch.softmax(raw_scores['dir_logits'].detach().float(), dim=-1).argmax(dim=-1)
        status_class = torch.softmax(
            raw_scores['conflict_area_status_logits'].detach().float(), dim=-1
        ).argmax(dim=-1) if 'conflict_area_status_logits' in raw_scores else torch.zeros_like(window_class)
        decision_class = torch.softmax(
            raw_scores['decision_phase_logits'].detach().float(), dim=-1
        ).argmax(dim=-1) + 1
        control_class = torch.softmax(
            raw_scores['control_phase_logits'].detach().float(), dim=-1
        ).argmax(dim=-1) + 1
        decision_class = decision_class * active
        control_class = control_class * active

        B = window_class.shape[0]
        if valid is None:
            valid = torch.ones((B,), device=device, dtype=model_dtype)
        else:
            valid = valid.to(device=device, dtype=model_dtype).reshape(-1).clamp(0.0, 1.0)

        def _sigmoid_field(key: str, width: int) -> torch.Tensor:
            value = raw_scores.get(key)
            if value is None:
                return torch.zeros((B, width), device=device, dtype=model_dtype)
            return torch.sigmoid(value.detach().to(device=device, dtype=model_dtype)).reshape(B, width)

        def _norm_field(key: str) -> torch.Tensor:
            value = raw_scores.get(key)
            if value is None:
                return torch.zeros((B,), device=device, dtype=model_dtype)
            return torch.nan_to_num(
                value.detach().to(device=device, dtype=model_dtype).reshape(-1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)

        def _predicted_speed_valid(key: str) -> torch.Tensor:
            # Predicted-cache has no separate speed-valid heads. Preserve the
            # graph_values slot semantics by marking availability of the
            # corresponding scalar prediction, instead of reusing unrelated
            # edge/chase probabilities.
            if key not in raw_scores:
                return torch.zeros((B, 1), device=device, dtype=model_dtype)
            return torch.ones((B, 1), device=device, dtype=model_dtype)

        temp_bins = _sigmoid_field('temporary_occupancy_logits', self.temporary_occupancy_dim)
        temp_valid = torch.ones_like(temp_bins) if 'temporary_occupancy_logits' in raw_scores else torch.zeros_like(temp_bins)
        if 'go_opportunity_logits' in raw_scores:
            opportunity = torch.softmax(
                raw_scores['go_opportunity_logits'].detach().to(device=device, dtype=model_dtype),
                dim=-1,
            )
            yld_prob = opportunity[:, 0:1]
            go_prob = opportunity[:, 1:2]
            go_valid = torch.ones((B, 1), device=device, dtype=model_dtype)
        else:
            yld_prob = torch.full((B, 1), 0.5, device=device, dtype=model_dtype)
            go_prob = torch.full((B, 1), 0.5, device=device, dtype=model_dtype)
            go_valid = torch.zeros((B, 1), device=device, dtype=model_dtype)

        timing_values = raw_scores.get('conflict_timing_values')
        if timing_values is None:
            timing_values = torch.zeros((B, 3), device=device, dtype=model_dtype)
            timing_valid = torch.zeros((B, 1), device=device, dtype=model_dtype)
        else:
            timing_values = torch.nan_to_num(
                timing_values.detach().to(device=device, dtype=model_dtype).reshape(B, 3),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(-2.0, 2.0)
            timing_valid = (status_class.to(device=device).reshape(B, 1) > 0).to(dtype=model_dtype)

        boundary_values = torch.stack(
            [
                _norm_field('merge_yld_max'),
                _norm_field('merge_go_min'),
                _norm_field('junction_yld_max'),
                _norm_field('junction_go_min'),
                _norm_field('borrow_yld_max'),
                _norm_field('borrow_go_min'),
            ],
            dim=-1,
        )
        chase_has_lead = (
            torch.sigmoid(raw_scores['chase_has_lead_logit'].detach().to(device=device, dtype=model_dtype))
            if 'chase_has_lead_logit' in raw_scores else
            torch.zeros((B,), device=device, dtype=model_dtype)
        ).reshape(B, 1)
        chase_speed = _norm_field('chase_speed_max').reshape(B, 1)
        current_edge_valid = (
            torch.sigmoid(raw_scores['current_cover_edge_valid_logit'].detach().to(device=device, dtype=model_dtype))
            if 'current_cover_edge_valid_logit' in raw_scores else
            torch.zeros((B,), device=device, dtype=model_dtype)
        ).reshape(B, 1)
        current_edge_mode = (
            torch.softmax(raw_scores['current_cover_edge_mode_logits'].detach().to(device=device, dtype=model_dtype), dim=-1)
            if 'current_cover_edge_mode_logits' in raw_scores else
            F.one_hot(torch.zeros((B,), device=device, dtype=torch.long), num_classes=5).to(dtype=model_dtype)
        )
        future_edge_valid = (
            torch.sigmoid(raw_scores['future_cover_edge_valid_logit'].detach().to(device=device, dtype=model_dtype))
            if 'future_cover_edge_valid_logit' in raw_scores else
            torch.zeros((B,), device=device, dtype=model_dtype)
        ).reshape(B, 1)
        future_edge_mode = (
            torch.softmax(raw_scores['future_cover_edge_mode_logits'].detach().to(device=device, dtype=model_dtype), dim=-1)
            if 'future_cover_edge_mode_logits' in raw_scores else
            F.one_hot(torch.zeros((B,), device=device, dtype=torch.long), num_classes=5).to(dtype=model_dtype)
        )
        current_upper = _norm_field('current_cover_upper_speed').reshape(B, 1)
        future_lower = _norm_field('future_cover_lower_speed').reshape(B, 1)
        front_follow_upper = _norm_field('front_follow_upper_speed').reshape(B, 1)
        merge_flow_lower = _norm_field('merge_flow_lower_speed').reshape(B, 1)
        current_upper_valid = _predicted_speed_valid('current_cover_upper_speed')
        future_lower_valid = _predicted_speed_valid('future_cover_lower_speed')
        front_follow_upper_valid = _predicted_speed_valid('front_follow_upper_speed')
        merge_flow_lower_valid = _predicted_speed_valid('merge_flow_lower_speed')
        graph_values = torch.cat(
            [
                current_edge_valid,
                current_edge_mode,
                future_edge_valid,
                future_edge_mode,
                current_upper,
                future_lower,
                front_follow_upper,
                merge_flow_lower,
                current_upper_valid,
                future_lower_valid,
                front_follow_upper_valid,
                merge_flow_lower_valid,
            ],
            dim=-1,
        ).clamp(0.0, 1.0)

        prev_state = {
            'valid': valid.detach(),
            'family': family.to(device=device).detach(),
            'dir': dir_class.to(device=device).detach(),
            'status': status_class.to(device=device).detach(),
            'decision_phase': decision_class.to(device=device).detach(),
            'control_phase': control_class.to(device=device).detach(),
            'conflict_area_route_mask': _sigmoid_field('conflict_area_logits', route_steps).detach(),
            'temporary_occupancy_cover_bins': temp_bins.detach(),
            'temporary_occupancy_cover_valid': temp_valid.detach(),
            'go_opportunity_prob': go_prob.detach(),
            'yld_pressure_prob': yld_prob.detach(),
            'go_opportunity_valid': go_valid.detach(),
            'conflict_timing_values': timing_values.detach(),
            'conflict_timing_valid': timing_valid.detach(),
            'boundary_values': boundary_values.detach(),
            'chase_values': torch.cat([chase_has_lead, chase_speed], dim=-1).detach(),
            'graph_values': graph_values.detach(),
        }
        return self._semantic_state_cache_to_device(
            prev_state, device=torch.device('cpu'), model_dtype=torch.float32
        )

    def _fuse_stage1_raw_scores(
        self,
        direct_scores: dict,
        transition_scores: Optional[dict],
        prev_valid: Optional[torch.Tensor],
    ) -> Tuple[dict, Dict[str, torch.Tensor]]:
        if (
            not self.use_semantic_state_fusion
            or transition_scores is None
            or prev_valid is None
        ):
            device = next(iter(direct_scores.values())).device
            dtype = next(iter(direct_scores.values())).dtype
            zero_gate = torch.zeros((next(iter(direct_scores.values())).shape[0],), device=device, dtype=dtype)
            return direct_scores, {
                'enabled': zero_gate.new_tensor(float(self.use_semantic_state_fusion)),
                'gate': zero_gate,
            }

        fused = dict(direct_scores)
        first_tensor = next(value for value in direct_scores.values() if isinstance(value, torch.Tensor))
        gate = prev_valid.to(device=first_tensor.device, dtype=first_tensor.dtype).reshape(-1)
        gate = gate.clamp(0.0, 1.0) * float(self.semantic_state_fusion_alpha)
        warmup_frames = max(int(self.semantic_state_fusion_warmup_frames), 0)
        if warmup_frames > 0:
            warmup_scale = min(
                max(float(getattr(self, '_semantic_state_cache_frame', 0)) / float(warmup_frames), 0.0),
                1.0,
            )
            gate = gate * warmup_scale
        for key, direct_value in direct_scores.items():
            transition_value = transition_scores.get(key)
            if (
                not isinstance(direct_value, torch.Tensor)
                or not isinstance(transition_value, torch.Tensor)
                or direct_value.shape != transition_value.shape
                or not direct_value.dtype.is_floating_point
            ):
                continue
            gate_view = gate
            while gate_view.dim() < direct_value.dim():
                gate_view = gate_view.unsqueeze(-1)
            transition_value = transition_value.to(device=direct_value.device, dtype=direct_value.dtype)
            fused[key] = direct_value * (1.0 - gate_view) + transition_value * gate_view
        return fused, {
            'enabled': gate.new_tensor(1.0),
            'gate': gate.detach(),
        }

    def _infer_traj_branch_condition(
        self,
        *,
        stage1_raw_scores: dict,
        speed_ref: torch.Tensor,
        borrow_time_s: Optional[torch.Tensor],
        prev_relation_probs: Optional[torch.Tensor],
        device: torch.device,
        model_dtype: torch.dtype,
    ):
        if not self.use_stage1_speed_energy:
            return None, None
        if stage1_raw_scores is None:
            raise ValueError("shared stage1 branch conditioning requires stage1_raw_scores")

        return self._build_traj_branch_condition_from_stage1_raw(
            raw_scores=stage1_raw_scores,
            speed_ref=speed_ref,
            borrow_time_s=borrow_time_s,
            prev_relation_probs=prev_relation_probs,
            device=device,
            model_dtype=model_dtype,
        )

    def _build_stage1_speed_samples(
        self,
        center_speed: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> torch.Tensor:
        center_speed = center_speed.to(device=device, dtype=model_dtype)
        brake_span = center_speed.new_tensor(self.stage1_query_brake_mps2 * self.stage1_query_dt)
        accel_span = center_speed.new_tensor(self.stage1_query_accel_mps2 * self.stage1_query_dt)
        samples = torch.stack([
            center_speed - brake_span,
            center_speed - brake_span * (2.0 / 3.0),
            center_speed - brake_span * (1.0 / 3.0),
            center_speed,
            center_speed + accel_span * (1.0 / 3.0),
            center_speed + accel_span * (2.0 / 3.0),
            center_speed + accel_span,
        ], dim=-1)
        return samples.clamp_(0.0, 20.0)

    def _compute_inference_traj_speed_refs(
        self,
        trajectory: torch.Tensor,
        model_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the short-horizon trajectory speed refs used by the local PID agent."""
        traj = trajectory.to(dtype=model_dtype)
        if traj.shape[1] >= 3:
            traj_speed_1s = (traj[:, 2] - traj[:, 0]).norm(dim=-1)
        elif traj.shape[1] >= 2:
            traj_speed_1s = (traj[:, 1] - traj[:, 0]).norm(dim=-1) * 2.0
        else:
            traj_speed_1s = traj[:, 0].norm(dim=-1) * 2.0

        if traj.shape[1] >= 2:
            traj_speed_05s = (traj[:, 1] - traj[:, 0]).norm(dim=-1) * 2.0
        else:
            traj_speed_05s = traj[:, 0].norm(dim=-1) * 2.0
        return traj_speed_1s.clamp(0.0, 20.0), traj_speed_05s.clamp(0.0, 20.0)

    @staticmethod
    def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = valid_mask.to(dtype=torch.bool)
        if valid_mask.sum() <= 0:
            return pred.new_tensor(0.0)
        return F.smooth_l1_loss(pred[valid_mask], target[valid_mask])

    @staticmethod
    def _reduce_per_sample(loss_tensor: torch.Tensor) -> torch.Tensor:
        if loss_tensor.dim() <= 1:
            return loss_tensor
        return loss_tensor.reshape(loss_tensor.shape[0], -1).mean(dim=1)

    @staticmethod
    def _masked_batch_mean(loss_per_sample: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        active_mask = active_mask.to(dtype=torch.bool, device=loss_per_sample.device)
        if active_mask.numel() == 0 or active_mask.sum() <= 0:
            return loss_per_sample.new_tensor(0.0)
        return loss_per_sample[active_mask].mean()

    def _get_good_route_mask(self, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        bad_mask = batch.get('is_bad_route', None)
        if bad_mask is None:
            return torch.ones(batch['agent_pos'].shape[0], device=device, dtype=torch.bool)
        if not isinstance(bad_mask, torch.Tensor):
            bad_mask = torch.as_tensor(bad_mask, device=device)
        return ~bad_mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def decode_speed_two_hot(speed_logits, speed_classes):
        """Decode speed logits to scalar m/s via softmax weighted sum.

        Args:
            speed_logits: (B, num_classes) raw logits
            speed_classes: list of float, bin centers

        Returns:
            speed_scalar: (B,) in m/s
        """
        bins = torch.tensor(speed_classes, device=speed_logits.device, dtype=speed_logits.dtype)
        probs = torch.softmax(speed_logits.float(), dim=-1)
        return (probs * bins).sum(dim=-1)

    # ========== GT Augmentation ==========
    def _augment_gt(self, trajectory, K):
        """Generate K augmented GT variants via speed scaling.
        Returns reliable positive samples (safe trajectories).
        """
        B, T, D = trajectory.shape
        aug = trajectory.unsqueeze(1).expand(-1, K, -1, -1).clone()  # (B, K, T, 2)
        for k in range(K):
            # Speed scaling: 0.8-1.0x (slower is safer)
            scale = 0.8 + 0.2 * torch.rand(B, 1, 1, device=trajectory.device)
            # Scale displacements relative to start point
            displacements = aug[:, k, 1:] - aug[:, k, :1]  # (B, T-1, 2)
            aug[:, k, 1:] = aug[:, k, :1] + displacements * scale
        return aug  # (B, K, T, 2)

    # ========== Delta Stats Registration ==========
    def register_delta_stats(self, delta_mean, delta_std):
        """Register per-step delta mean/std for z-score normalization.
        Must be called before DDP wrapping.

        Args:
            delta_mean: (T, 2) per-step delta mean
            delta_std: (T, 2) per-step delta std
        """
        if isinstance(delta_mean, np.ndarray):
            delta_mean = torch.from_numpy(delta_mean).float()
        if isinstance(delta_std, np.ndarray):
            delta_std = torch.from_numpy(delta_std).float()
        device = next(self.parameters()).device
        self.register_buffer('delta_mean', delta_mean.to(device))
        self.register_buffer('delta_std', delta_std.to(device))

    @staticmethod
    def select_norm_stats_config(config: Dict):
        """Select exactly one trajectory normalization source from config.

        Route B should not silently fall back across normalization families.
        The config must explicitly choose exactly one of:
          - global_abs_stats_path
          - abs_stats_path
          - delta_stats_path
        """
        candidates = [
            ('global_abs', config.get('global_abs_stats_path')),
            ('abs', config.get('abs_stats_path')),
            ('delta', config.get('delta_stats_path')),
        ]
        specified = [(name, path) for name, path in candidates if path]
        if len(specified) != 1:
            pretty = {name: path for name, path in candidates}
            raise ValueError(
                "Route B requires exactly one normalization config among "
                "`global_abs_stats_path`, `abs_stats_path`, and `delta_stats_path`. "
                f"Got: {pretty}"
            )
        return specified[0]

    def register_norm_stats_from_config(self, config: Dict):
        """Register the explicitly configured trajectory normalization stats."""
        norm_mode, stats_path = self.select_norm_stats_config(config)
        if norm_mode == 'global_abs':
            gdata = np.load(stats_path)
            self.register_global_abs_stats(gdata['global_abs_mean'], gdata['global_abs_std'])
        elif norm_mode == 'abs':
            adata = np.load(stats_path)
            self.register_abs_stats(adata['abs_mean'], adata['abs_std'])
        else:
            ddata = np.load(stats_path)
            self.register_delta_stats(ddata['delta_mean'], ddata['delta_std'])
        return norm_mode

    # ========== Normalization: Delta Z-Score ==========
    @staticmethod
    def abs_to_delta(abs_traj: torch.Tensor) -> torch.Tensor:
        """Convert absolute trajectory to delta: [p0, p1-p0, p2-p1, ...]"""
        delta = abs_traj.clone()
        delta[..., 1:, :] = abs_traj[..., 1:, :] - abs_traj[..., :-1, :]
        return delta

    @staticmethod
    def delta_to_abs(delta: torch.Tensor) -> torch.Tensor:
        """Convert delta to absolute trajectory via cumulative sum."""
        return delta.cumsum(dim=-2)

    def z_norm(self, delta: torch.Tensor) -> torch.Tensor:
        """Z-score normalize delta using per-step mean/std.
        delta: (..., T, 2)  ->  z: (..., T, 2)
        """
        mean = self.delta_mean.to(delta.device)  # (T, 2)
        std = self.delta_std.to(delta.device)     # (T, 2)
        return (delta - mean) / std.clamp(min=1e-6)

    def z_denorm(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse z-score: z -> delta.
        z: (..., T, 2)  ->  delta: (..., T, 2)
        """
        mean = self.delta_mean.to(z.device)  # (T, 2)
        std = self.delta_std.to(z.device)    # (T, 2)
        return z * std + mean

    def norm_to_abs(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse: z-normed -> absolute trajectory.
        Priority: global_abs > per-step abs > delta (legacy).
        """
        if self.global_abs_mean is not None:
            return self.global_abs_z_denorm(z)
        if self.abs_mean is not None:
            return self.abs_z_denorm(z)
        return self.delta_to_abs(self.z_denorm(z))

    def abs_to_norm(self, abs_traj: torch.Tensor) -> torch.Tensor:
        """Forward: absolute trajectory -> z-normed.
        Priority: global_abs > per-step abs > delta (legacy).
        """
        if self.global_abs_mean is not None:
            return self.global_abs_z_norm(abs_traj)
        if self.abs_mean is not None:
            return self.abs_z_norm(abs_traj)
        return self.z_norm(self.abs_to_delta(abs_traj))

    # ========== Normalization: Per-Timestep Abs Z-Score ==========
    def register_abs_stats(self, abs_mean, abs_std):
        """Register per-step abs mean/std for z-score normalization.
        Must be called before DDP wrapping.
        """
        if isinstance(abs_mean, np.ndarray):
            abs_mean = torch.from_numpy(abs_mean).float()
        if isinstance(abs_std, np.ndarray):
            abs_std = torch.from_numpy(abs_std).float()
        device = next(self.parameters()).device
        self.register_buffer('abs_mean', abs_mean.to(device))
        self.register_buffer('abs_std', abs_std.to(device))

    def abs_z_norm(self, abs_traj: torch.Tensor) -> torch.Tensor:
        """Per-timestep z-score on absolute coordinates.
        abs_traj: (..., T, 2)  ->  z: (..., T, 2)
        """
        mean = self.abs_mean.to(abs_traj.device)  # (T, 2)
        std = self.abs_std.to(abs_traj.device)     # (T, 2)
        return (abs_traj - mean) / std.clamp(min=1e-6)

    def abs_z_denorm(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse per-timestep z-score -> absolute coordinates.
        z: (..., T, 2)  ->  abs_traj: (..., T, 2)
        """
        mean = self.abs_mean.to(z.device)  # (T, 2)
        std = self.abs_std.to(z.device)    # (T, 2)
        return z * std + mean

    # ========== Normalization: Route Per-Waypoint Abs Z-Score ==========
    def register_route_abs_stats(self, route_abs_mean, route_abs_std):
        """Register per-waypoint absolute route stats for route diffusion."""
        if isinstance(route_abs_mean, np.ndarray):
            route_abs_mean = torch.from_numpy(route_abs_mean).float()
        if isinstance(route_abs_std, np.ndarray):
            route_abs_std = torch.from_numpy(route_abs_std).float()
        device = next(self.parameters()).device
        self.register_buffer('route_abs_mean', route_abs_mean.to(device))
        self.register_buffer('route_abs_std', route_abs_std.to(device))

    def _require_route_abs_stats(self):
        if self.route_abs_mean is None or self.route_abs_std is None:
            raise RuntimeError(
                "Joint Route B ego diffusion requires route_abs_stats_path to be configured "
                "and loaded (route_abs_mean/route_abs_std)."
            )

    def route_abs_to_norm(self, abs_route: torch.Tensor) -> torch.Tensor:
        """Per-waypoint z-score on route absolute coordinates."""
        self._require_route_abs_stats()
        mean = self.route_abs_mean.to(abs_route.device)
        std = self.route_abs_std.to(abs_route.device)
        return (abs_route - mean) / std.clamp(min=1e-6)

    def route_norm_to_abs(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse per-waypoint z-score for route absolute coordinates."""
        self._require_route_abs_stats()
        mean = self.route_abs_mean.to(z.device)
        std = self.route_abs_std.to(z.device)
        return z * std + mean

    def joint_abs_to_norm(self, traj_abs: torch.Tensor, route_abs: torch.Tensor) -> torch.Tensor:
        """Concatenate normalized trajectory and route diffusion states."""
        traj_norm = self.abs_to_norm(traj_abs)
        route_norm = self.route_abs_to_norm(route_abs)
        return torch.cat([traj_norm, route_norm], dim=-2)

    def joint_norm_to_abs(self, joint_z: torch.Tensor) -> torch.Tensor:
        """Split a joint traj+route diffusion state back to absolute coordinates."""
        joint_len = self.horizon + self.num_waypoints
        if joint_z.shape[-2] != joint_len:
            raise ValueError(f"Expected joint diffusion length {joint_len}, got {joint_z.shape[-2]}")
        traj_abs = self.norm_to_abs(joint_z[..., :self.horizon, :])
        route_abs = self.route_norm_to_abs(joint_z[..., self.horizon:, :])
        return torch.cat([traj_abs, route_abs], dim=-2)

    # ========== Normalization: Global Abs Z-Score ==========
    def register_global_abs_stats(self, global_abs_mean, global_abs_std):
        """Register global abs mean/std for z-score normalization.
        All timesteps share the same (2,) mean/std — preserves temporal structure.
        """
        if isinstance(global_abs_mean, np.ndarray):
            global_abs_mean = torch.from_numpy(global_abs_mean).float()
        if isinstance(global_abs_std, np.ndarray):
            global_abs_std = torch.from_numpy(global_abs_std).float()
        device = next(self.parameters()).device
        self.register_buffer('global_abs_mean', global_abs_mean.to(device))
        self.register_buffer('global_abs_std', global_abs_std.to(device))

    def global_abs_z_norm(self, abs_traj: torch.Tensor) -> torch.Tensor:
        """Global z-score: all timesteps share same mean/std.
        abs_traj: (..., T, 2)  ->  z: (..., T, 2)
        """
        mean = self.global_abs_mean.to(abs_traj.device)  # (2,)
        std = self.global_abs_std.to(abs_traj.device)     # (2,)
        return (abs_traj - mean) / std.clamp(min=1e-6)

    def global_abs_z_denorm(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse global z-score -> absolute coordinates.
        z: (..., T, 2)  ->  abs_traj: (..., T, 2)
        """
        mean = self.global_abs_mean.to(z.device)  # (2,)
        std = self.global_abs_std.to(z.device)    # (2,)
        return z * std + mean

    # ========== Focal Loss (same as Route A) ==========
    def _focal_loss(self, logits, targets, gamma=2.0, alpha=0.25):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = alpha * (1 - p_t) ** gamma
        return (focal_weight * bce).mean()

    def _compute_shared_stage1_loss(
        self,
        batch: Dict[str, torch.Tensor],
        trajectory: torch.Tensor,
        route_gt: torch.Tensor,
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        transfuser_lidar_bev: Optional[torch.Tensor],
        ego_status: torch.Tensor,
        bev_proj: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype,
        noisy_joint: Optional[torch.Tensor] = None,
        noisy_joint_abs: Optional[torch.Tensor] = None,
        diff_timesteps: Optional[torch.Tensor] = None,
        branch_condition: Optional[torch.Tensor] = None,
        branch_condition_schedule: Optional[torch.Tensor] = None,
        prev_route_coarse_memory: Optional[torch.Tensor] = None,
        speed_pred_for_consistency: Optional[torch.Tensor] = None,
    ):
        gt_abs = trajectory.unsqueeze(1)
        gt_joint_normed = self.joint_abs_to_norm(trajectory, route_gt).unsqueeze(1)
        gt_joint_abs = self.joint_norm_to_abs(gt_joint_normed)
        family_codes = self._get_stage1_long_target(batch, 'conflict_area_family', device=device)
        dir_target = self._get_stage1_long_target(batch, 'conflict_area_dir', device=device)
        decision_phase_codes = self._get_stage1_long_target(batch, 'conflict_decision_phase', device=device)
        control_phase_codes = self._get_stage1_long_target(batch, 'conflict_control_phase', device=device)
        if family_codes is None or dir_target is None or decision_phase_codes is None or control_phase_codes is None:
            raise ValueError("shared stage1 training requires new conflict family/dir/phase labels")

        window_target = self._window_target_from_family_codes(family_codes)
        merge_yld_target = self._get_stage1_batch_tensor(batch, 'merge_yld_max_speed', device=device, model_dtype=model_dtype)
        merge_go_target = self._get_stage1_batch_tensor(batch, 'merge_go_min_speed', device=device, model_dtype=model_dtype)
        junction_yld_target = self._get_stage1_batch_tensor(batch, 'junction_yld_max_speed', device=device, model_dtype=model_dtype)
        junction_go_target = self._get_stage1_batch_tensor(batch, 'junction_go_min_speed', device=device, model_dtype=model_dtype)
        borrow_yld_target = self._get_stage1_batch_tensor(batch, 'borrow_yld_max_speed', device=device, model_dtype=model_dtype)
        borrow_go_target = self._get_stage1_batch_tensor(batch, 'borrow_go_min_speed', device=device, model_dtype=model_dtype)
        boundary_targets = (
            merge_yld_target,
            merge_go_target,
            junction_yld_target,
            junction_go_target,
            borrow_yld_target,
            borrow_go_target,
        )
        if any(target is None for target in boundary_targets):
            raise ValueError("shared stage1 training requires all boundary speed labels")
        merge_yld_valid = torch.ones_like(merge_yld_target, device=device, dtype=torch.bool)
        merge_go_valid = torch.ones_like(merge_go_target, device=device, dtype=torch.bool)
        junction_yld_valid = torch.ones_like(junction_yld_target, device=device, dtype=torch.bool)
        junction_go_valid = torch.ones_like(junction_go_target, device=device, dtype=torch.bool)
        borrow_yld_valid = torch.ones_like(borrow_yld_target, device=device, dtype=torch.bool)
        borrow_go_valid = torch.ones_like(borrow_go_target, device=device, dtype=torch.bool)
        temp_targets = self._get_temporary_occupancy_targets(
            batch,
            device=device,
            model_dtype=model_dtype,
            require=self.use_temporary_occupancy_phase,
        )
        timing_targets = self._get_conflict_timing_targets(
            batch,
            device=device,
            model_dtype=model_dtype,
            family_codes=family_codes,
            require=self.use_conflict_timing_state,
        )
        chase_targets = (
            self._get_chase_targets(
                batch,
                device=device,
                model_dtype=model_dtype,
                require=True,
            )
            if self.use_chase_front_following_state else None
        )
        graph_targets = self._get_cover_relation_graph_targets(
            batch,
            device=device,
            model_dtype=model_dtype,
            require=self.use_cover_relation_graph_decoder,
        )

        stage1_training_source = self.shared_stage1_training_source
        if stage1_training_source == 'clean_then_noisy':
            stage1_training_source = (
                'noisy'
                if self._current_epoch >= self.shared_stage1_training_source_switch_epoch
                else 'clean'
            )

        if stage1_training_source == 'noisy':
            if noisy_joint is None or noisy_joint_abs is None or diff_timesteps is None:
                raise ValueError("shared_stage1_training_source='noisy' requires noisy_joint inputs")
            shared_forward = self.model.forward_ego(
                x_t=noisy_joint,
                x_t_abs=noisy_joint_abs,
                timestep=diff_timesteps,
                transfuser_bev_feature=transfuser_bev_feature,
                transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                ego_status=ego_status,
                bev_proj_cached=bev_proj,
                transfuser_lidar_bev=transfuser_lidar_bev,
                branch_condition=branch_condition,
                branch_condition_scale=self.traj_branch_condition_scale,
                branch_condition_schedule=branch_condition_schedule if branch_condition is not None else None,
                prev_route_coarse_memory=prev_route_coarse_memory,
                return_intermediates=True,
            )
        else:
            zero_timestep = torch.zeros(gt_abs.shape[0], device=device, dtype=torch.long)
            shared_forward = self.model.forward_ego(
                x_t=gt_joint_normed,
                x_t_abs=gt_joint_abs,
                timestep=zero_timestep,
                transfuser_bev_feature=transfuser_bev_feature,
                transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                ego_status=ego_status,
                bev_proj_cached=bev_proj,
                transfuser_lidar_bev=transfuser_lidar_bev,
                prev_route_coarse_memory=prev_route_coarse_memory,
                return_intermediates=True,
            )
        transition_only_state = self.semantic_state_predictor_mode == 'transition_only'
        direct_only_state = self.semantic_state_predictor_mode == 'direct_only'
        use_transition_state = self.use_semantic_state_transition and not direct_only_state
        route_steps = self.num_waypoints
        raw_stage1_scores = None
        if not transition_only_state:
            raw_stage1_scores = self.model.compute_shared_stage1_from_ego_outputs(
                traj_out=shared_forward['traj_out'],
                route_out=shared_forward['route_out'],
                speed_out=shared_forward['speed_out'],
                route_points=shared_forward['route_points'],
                conditioning=shared_forward['conditioning'],
            )
            route_steps = raw_stage1_scores['conflict_area_logits'].shape[1]
        transition_stage1_scores = None
        if use_transition_state:
            prev_state = self._get_semantic_transition_prev_state(
                batch=batch,
                device=device,
                model_dtype=model_dtype,
                route_steps=route_steps,
                require=True,
            )
            prev_state = self._apply_semantic_transition_prev_dropout(
                prev_state,
                route_steps=route_steps,
                device=device,
                model_dtype=model_dtype,
                batch=batch,
            )
            transition_stage1_scores = self._compute_semantic_transition_scores_from_shared(
                shared_forward,
                prev_state=prev_state,
            )
            if transition_only_state:
                raw_stage1_scores = transition_stage1_scores
                transition_stage1_scores = None
        if raw_stage1_scores is None:
            raise RuntimeError("semantic state training produced no state scores")

        zero = gt_abs.new_tensor(0.0)
        supervise_window = self._supervise_state_group('window')
        supervise_dir = self._supervise_state_group('dir')
        supervise_decision = self._supervise_state_group('decision')
        supervise_control = self._supervise_state_group('control')
        supervise_boundary = self._supervise_state_group('boundary')
        supervise_conflict_area = self._supervise_state_group('conflict_area')
        supervise_tempocc = self._supervise_state_group('tempocc')
        supervise_opportunity = self._supervise_state_group('opportunity')
        supervise_area_status = self._supervise_state_group('area_status')
        supervise_timing = self._supervise_state_group('timing')
        supervise_inside_area_go = self._supervise_state_group('inside_area_go')
        supervise_chase = self._supervise_state_group('chase')
        supervise_graph = self._supervise_state_group('graph')
        supervise_edge_speed_consistency = self._supervise_state_group('edge_speed_consistency')

        def _boundary_loss(pred_norm: torch.Tensor, target_mps: torch.Tensor) -> torch.Tensor:
            target_norm = torch.nan_to_num(
                target_mps,
                nan=0.0,
                posinf=float(self.stage1_boundary_norm_scale),
                neginf=0.0,
            )
            target_norm = (target_norm / max(self.stage1_boundary_norm_scale, 1e-6)).clamp(0.0, 1.0)
            return F.smooth_l1_loss(pred_norm.float(), target_norm.float())

        loss_merge_yld = _boundary_loss(raw_stage1_scores['merge_yld_max'], merge_yld_target)
        loss_merge_go = _boundary_loss(raw_stage1_scores['merge_go_min'], merge_go_target)
        loss_junction_yld = _boundary_loss(raw_stage1_scores['junction_yld_max'], junction_yld_target)
        loss_junction_go = _boundary_loss(raw_stage1_scores['junction_go_min'], junction_go_target)
        loss_borrow_yld = _boundary_loss(raw_stage1_scores['borrow_yld_max'], borrow_yld_target)
        loss_borrow_go = _boundary_loss(raw_stage1_scores['borrow_go_min'], borrow_go_target)
        loss_merge = 0.5 * (loss_merge_yld + loss_merge_go)
        loss_junction = 0.5 * (loss_junction_yld + loss_junction_go)
        loss_borrow = 0.5 * (loss_borrow_yld + loss_borrow_go)
        loss_cross = loss_junction + loss_borrow

        loss_window = F.cross_entropy(raw_stage1_scores['window_logits'].float(), window_target)
        loss_dir = F.cross_entropy(raw_stage1_scores['dir_logits'].float(), dir_target.clamp(min=0, max=3))
        conflict_area_target, conflict_area_valid_mask = self._build_conflict_area_route_target(
            batch=batch,
            route_steps=raw_stage1_scores['conflict_area_logits'].shape[1],
            device=device,
            model_dtype=model_dtype,
        )
        loss_conflict_area = zero
        if conflict_area_target is not None:
            area_loss_raw = F.binary_cross_entropy_with_logits(
                raw_stage1_scores['conflict_area_logits'].float(),
                conflict_area_target.float(),
                reduction='none',
            )
            if conflict_area_valid_mask is None:
                conflict_area_valid_mask = torch.ones_like(conflict_area_target, dtype=torch.bool)
            pos_mask = (conflict_area_target > 0.5) & conflict_area_valid_mask
            neg_mask = (conflict_area_target <= 0.5) & conflict_area_valid_mask
            if pos_mask.any() and neg_mask.any():
                loss_conflict_area = 0.5 * (area_loss_raw[pos_mask].mean() + area_loss_raw[neg_mask].mean())
            elif pos_mask.any():
                loss_conflict_area = area_loss_raw[pos_mask].mean()
            elif neg_mask.any():
                loss_conflict_area = area_loss_raw[neg_mask].mean()

        decision_phase_valid_mask = decision_phase_codes > 0
        decision_phase_target = (decision_phase_codes - 1).clamp(min=0, max=1)
        loss_decision_phase = zero
        if decision_phase_valid_mask.any():
            loss_decision_phase = F.cross_entropy(
                raw_stage1_scores['decision_phase_logits'][decision_phase_valid_mask].float(),
                decision_phase_target[decision_phase_valid_mask],
            )

        control_phase_valid_mask = control_phase_codes > 0
        control_phase_target = (control_phase_codes - 1).clamp(min=0, max=3)
        loss_control_phase = zero
        if control_phase_valid_mask.any():
            loss_control_phase = F.cross_entropy(
                raw_stage1_scores['control_phase_logits'][control_phase_valid_mask].float(),
                control_phase_target[control_phase_valid_mask],
            )

        loss_phase = loss_decision_phase + loss_control_phase
        loss_temporary_occupancy = zero
        loss_go_opportunity = zero
        if temp_targets is not None:
            temp_valid = temp_targets['valid']
            if temp_valid.any():
                temp_loss_raw = F.binary_cross_entropy_with_logits(
                    raw_stage1_scores['temporary_occupancy_logits'].float(),
                    temp_targets['bins'].float(),
                    reduction='none',
                )
                loss_temporary_occupancy = temp_loss_raw[temp_valid].mean()
            go_valid = temp_targets['go_opportunity_valid']
            if go_valid.any():
                target_probs = temp_targets['go_opportunity_target'][go_valid].float()
                log_probs = F.log_softmax(
                    raw_stage1_scores['go_opportunity_logits'][go_valid].float(),
                    dim=-1,
                )
                loss_go_opportunity = -(target_probs * log_probs).sum(dim=-1).mean()

        loss_conflict_area_status = zero
        loss_conflict_timing = zero
        loss_inside_area_go = zero
        if timing_targets is not None:
            status_target = timing_targets['status'].clamp(min=0, max=3)
            loss_conflict_area_status = F.cross_entropy(
                raw_stage1_scores['conflict_area_status_logits'].float(),
                status_target,
            )
            timing_valid = timing_targets['valid']
            if timing_valid.any():
                loss_conflict_timing = F.smooth_l1_loss(
                    raw_stage1_scores['conflict_timing_values'][timing_valid].float(),
                    timing_targets['values'][timing_valid].float(),
                )
            inside_mask = timing_targets['status'] == 2
            if inside_mask.any():
                inside_decision_target = torch.ones(
                    int(inside_mask.sum().item()), device=device, dtype=torch.long
                )
                inside_control_target = torch.full(
                    (int(inside_mask.sum().item()),),
                    3,
                    device=device,
                    dtype=torch.long,
                )
                loss_inside_area_go = (
                    F.cross_entropy(
                        raw_stage1_scores['decision_phase_logits'][inside_mask].float(),
                        inside_decision_target,
                    )
                    + F.cross_entropy(
                        raw_stage1_scores['control_phase_logits'][inside_mask].float(),
                        inside_control_target,
                    )
                )

        loss_chase_has_lead = zero
        loss_chase_speed_max = zero
        if self.use_chase_front_following_state and chase_targets is not None:
            loss_chase_has_lead = F.binary_cross_entropy_with_logits(
                raw_stage1_scores['chase_has_lead_logit'].float(),
                chase_targets['has_lead'].float(),
            )
            chase_speed_target_norm = (
                chase_targets['speed_max'] / max(self.chase_speed_norm_scale, 1e-6)
            ).clamp(0.0, 1.0)
            loss_chase_speed_max = F.smooth_l1_loss(
                raw_stage1_scores['chase_speed_max'].float(),
                chase_speed_target_norm.float(),
            )

        loss_current_edge_valid = zero
        loss_current_edge_mode = zero
        loss_future_edge_valid = zero
        loss_future_edge_mode = zero
        loss_current_cover_upper = zero
        loss_future_cover_lower = zero
        loss_front_follow_upper = zero
        loss_merge_flow_lower = zero
        loss_edge_speed_consistency = zero
        if graph_targets is not None:
            loss_current_edge_valid = F.binary_cross_entropy_with_logits(
                raw_stage1_scores['current_cover_edge_valid_logit'].float(),
                graph_targets['current_valid'].float(),
            )
            if graph_targets['current_mode_valid'].any():
                loss_current_edge_mode = F.cross_entropy(
                    raw_stage1_scores['current_cover_edge_mode_logits'][
                        graph_targets['current_mode_valid']
                    ].float(),
                    graph_targets['current_mode'][graph_targets['current_mode_valid']],
                )
            loss_future_edge_valid = F.binary_cross_entropy_with_logits(
                raw_stage1_scores['future_cover_edge_valid_logit'].float(),
                graph_targets['future_valid'].float(),
            )
            if graph_targets['future_mode_valid'].any():
                loss_future_edge_mode = F.cross_entropy(
                    raw_stage1_scores['future_cover_edge_mode_logits'][
                        graph_targets['future_mode_valid']
                    ].float(),
                    graph_targets['future_mode'][graph_targets['future_mode_valid']],
                )

            def _graph_speed_loss(pred_norm: torch.Tensor, target_mps: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
                if not valid.any():
                    return zero
                target_norm = (
                    target_mps / max(self.stage1_boundary_norm_scale, 1e-6)
                ).clamp(0.0, 1.0)
                return F.smooth_l1_loss(
                    pred_norm[valid].float(),
                    target_norm[valid].float(),
                )

            loss_current_cover_upper = _graph_speed_loss(
                raw_stage1_scores['current_cover_upper_speed'],
                graph_targets['current_upper'],
                graph_targets['current_upper_valid'],
            )
            loss_future_cover_lower = _graph_speed_loss(
                raw_stage1_scores['future_cover_lower_speed'],
                graph_targets['future_lower'],
                graph_targets['future_lower_valid'],
            )
            loss_front_follow_upper = _graph_speed_loss(
                raw_stage1_scores['front_follow_upper_speed'],
                graph_targets['front_follow_upper'],
                graph_targets['front_follow_upper_valid'],
            )
            loss_merge_flow_lower = _graph_speed_loss(
                raw_stage1_scores['merge_flow_lower_speed'],
                graph_targets['merge_flow_lower'],
                graph_targets['merge_flow_lower_valid'],
            )
            if self.use_edge_speed_consistency_loss and speed_pred_for_consistency is not None:
                pred_speed = self.decode_speed_two_hot(
                    speed_pred_for_consistency, self.model.speed_classes
                ).to(device=device, dtype=model_dtype)

                def _upper_violation(upper_speed: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
                    if not valid.any():
                        return zero
                    return F.smooth_l1_loss(
                        pred_speed[valid],
                        torch.minimum(pred_speed[valid], upper_speed[valid]),
                    )

                def _lower_violation(lower_speed: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
                    if not valid.any():
                        return zero
                    return F.smooth_l1_loss(
                        pred_speed[valid],
                        torch.maximum(pred_speed[valid], lower_speed[valid]),
                    )

                future_go_before_mask = (
                    graph_targets['future_lower_valid']
                    & (graph_targets['future_mode'] == 2)
                )
                loss_edge_speed_consistency = torch.stack([
                    _upper_violation(
                        graph_targets['current_upper'],
                        graph_targets['current_upper_valid'],
                    ),
                    _lower_violation(
                        graph_targets['future_lower'],
                        future_go_before_mask,
                    ),
                    _upper_violation(
                        graph_targets['front_follow_upper'],
                        graph_targets['front_follow_upper_valid'],
                    ),
                    _lower_violation(
                        graph_targets['merge_flow_lower'],
                        graph_targets['merge_flow_lower_valid'],
                    ),
                ]).mean()

        loss_merge_active = zero
        loss_junction_active = zero
        loss_borrow_active = zero
        loss_cross_active = zero
        loss_chase = (
            self.chase_has_lead_loss_weight * loss_chase_has_lead
            + self.chase_speed_max_loss_weight * loss_chase_speed_max
        )
        loss_ped = zero
        loss_state_consistency = zero
        loss_state_consistency_window = zero
        loss_state_consistency_phase = zero
        loss_state_consistency_timing = zero
        loss_state_consistency_boundary = zero
        loss_state_consistency_area = zero
        loss_state_consistency_tempocc = zero
        loss_state_consistency_opportunity = zero
        if (
            self.use_independent_state_consistency_loss
            and not transition_only_state
            and self.state_consistency_loss_weight > 0
            and self.state_consistency_prob > 0
            and noisy_joint is not None
            and diff_timesteps is not None
            and torch.rand((), device=device).item() < self.state_consistency_prob
        ):
            def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
                if mask is None:
                    return values.mean()
                mask = mask.to(device=values.device, dtype=torch.bool)
                while mask.dim() < values.dim():
                    mask = mask.unsqueeze(-1)
                mask = mask.expand_as(values)
                if not mask.any():
                    return zero
                return values[mask].mean()

            def _sym_kl_logits(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
                a = a.float()
                b = b.float()
                log_a = F.log_softmax(a, dim=-1)
                log_b = F.log_softmax(b, dim=-1)
                prob_a = log_a.exp()
                prob_b = log_b.exp()
                loss_ab = (prob_a.detach() * (log_a.detach() - log_b)).sum(dim=-1)
                loss_ba = (prob_b.detach() * (log_b.detach() - log_a)).sum(dim=-1)
                return _masked_mean(0.5 * (loss_ab + loss_ba), mask)

            def _sym_mse(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
                loss = 0.5 * (
                    F.mse_loss(a.float(), b.detach().float(), reduction='none')
                    + F.mse_loss(b.float(), a.detach().float(), reduction='none')
                )
                return _masked_mean(loss, mask)

            def _sym_smooth_l1(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
                loss = 0.5 * (
                    F.smooth_l1_loss(a.float(), b.detach().float(), reduction='none')
                    + F.smooth_l1_loss(b.float(), a.detach().float(), reduction='none')
                )
                return _masked_mean(loss, mask)

            def _stage1_scores_for_noisy_view(
                noisy_joint_view: torch.Tensor,
                timesteps_view: torch.Tensor,
            ) -> dict:
                noisy_joint_abs_view = self.joint_norm_to_abs(noisy_joint_view)
                view_forward = self.model.forward_ego(
                    x_t=noisy_joint_view,
                    x_t_abs=noisy_joint_abs_view,
                    timestep=timesteps_view,
                    transfuser_bev_feature=transfuser_bev_feature,
                    transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                    ego_status=ego_status,
                    bev_proj_cached=bev_proj,
                    transfuser_lidar_bev=transfuser_lidar_bev,
                    prev_route_coarse_memory=prev_route_coarse_memory,
                    return_intermediates=True,
                )
                return self.model.compute_shared_stage1_from_ego_outputs(
                    traj_out=view_forward['traj_out'],
                    route_out=view_forward['route_out'],
                    speed_out=view_forward['speed_out'],
                    route_points=view_forward['route_points'],
                    conditioning=view_forward['conditioning'],
                )

            timesteps_view2 = torch.randint(0, self.train_max_timesteps, (gt_abs.shape[0],), device=device).long()
            noise_view2 = torch.randn_like(gt_joint_normed.squeeze(1), dtype=torch.float32)
            noisy_view2_flat = self.diffusion_scheduler.add_noise(
                original_samples=gt_joint_normed.squeeze(1),
                noise=noise_view2,
                timesteps=timesteps_view2,
            )
            scores_view1 = _stage1_scores_for_noisy_view(noisy_joint.detach(), diff_timesteps.detach())
            scores_view2 = _stage1_scores_for_noisy_view(noisy_view2_flat.unsqueeze(1), timesteps_view2)

            window_consistency_terms = []
            if supervise_window:
                window_consistency_terms.append(
                    _sym_kl_logits(scores_view1['window_logits'], scores_view2['window_logits'])
                )
            if supervise_dir:
                window_consistency_terms.append(
                    _sym_kl_logits(scores_view1['dir_logits'], scores_view2['dir_logits'])
                )
            loss_state_consistency_window = (
                torch.stack(window_consistency_terms).mean()
                if window_consistency_terms else zero
            )
            phase_consistency_terms = []
            if supervise_decision:
                phase_consistency_terms.append(_sym_kl_logits(
                    scores_view1['decision_phase_logits'],
                    scores_view2['decision_phase_logits'],
                    decision_phase_valid_mask,
                ))
            if supervise_control:
                phase_consistency_terms.append(_sym_kl_logits(
                    scores_view1['control_phase_logits'],
                    scores_view2['control_phase_logits'],
                    control_phase_valid_mask,
                ))
            loss_state_consistency_phase = (
                torch.stack(phase_consistency_terms).mean()
                if phase_consistency_terms else zero
            )
            boundary_consistency = zero
            if supervise_boundary:
                boundary_consistency = (
                    _sym_smooth_l1(scores_view1['merge_yld_max'], scores_view2['merge_yld_max'], merge_yld_valid)
                    + _sym_smooth_l1(scores_view1['merge_go_min'], scores_view2['merge_go_min'], merge_go_valid)
                    + _sym_smooth_l1(scores_view1['junction_yld_max'], scores_view2['junction_yld_max'], junction_yld_valid)
                    + _sym_smooth_l1(scores_view1['junction_go_min'], scores_view2['junction_go_min'], junction_go_valid)
                    + _sym_smooth_l1(scores_view1['borrow_yld_max'], scores_view2['borrow_yld_max'], borrow_yld_valid)
                    + _sym_smooth_l1(scores_view1['borrow_go_min'], scores_view2['borrow_go_min'], borrow_go_valid)
                ) / 6.0
            loss_state_consistency_boundary = boundary_consistency
            area_consistency = zero
            if supervise_conflict_area and conflict_area_valid_mask is not None:
                area_consistency = _sym_mse(
                    scores_view1['conflict_area_logits'],
                    scores_view2['conflict_area_logits'],
                    conflict_area_valid_mask,
                )
            loss_state_consistency_area = area_consistency
            temp_consistency = zero
            go_consistency = zero
            if temp_targets is not None:
                if supervise_tempocc:
                    temp_consistency = _sym_mse(
                        scores_view1['temporary_occupancy_logits'],
                        scores_view2['temporary_occupancy_logits'],
                        temp_targets['valid'],
                    )
                if supervise_opportunity:
                    go_consistency = _sym_kl_logits(
                        scores_view1['go_opportunity_logits'],
                        scores_view2['go_opportunity_logits'],
                        temp_targets['go_opportunity_valid'],
                    )
            loss_state_consistency_tempocc = temp_consistency
            loss_state_consistency_opportunity = go_consistency
            status_consistency = zero
            timing_consistency = zero
            if timing_targets is not None:
                if supervise_area_status:
                    status_consistency = _sym_kl_logits(
                        scores_view1['conflict_area_status_logits'],
                        scores_view2['conflict_area_status_logits'],
                        None,
                    )
                if supervise_timing:
                    timing_consistency = _sym_smooth_l1(
                        scores_view1['conflict_timing_values'],
                        scores_view2['conflict_timing_values'],
                        timing_targets['valid'],
                    )
            loss_state_consistency_timing = status_consistency + timing_consistency
            if supervise_graph and graph_targets is not None:
                graph_consistency = (
                    _sym_mse(
                        scores_view1['current_cover_edge_valid_logit'],
                        scores_view2['current_cover_edge_valid_logit'],
                    )
                    + _sym_kl_logits(
                        scores_view1['current_cover_edge_mode_logits'],
                        scores_view2['current_cover_edge_mode_logits'],
                        graph_targets['current_mode_valid'],
                    )
                    + _sym_mse(
                        scores_view1['future_cover_edge_valid_logit'],
                        scores_view2['future_cover_edge_valid_logit'],
                    )
                    + _sym_kl_logits(
                        scores_view1['future_cover_edge_mode_logits'],
                        scores_view2['future_cover_edge_mode_logits'],
                        graph_targets['future_mode_valid'],
                    )
                    + _sym_smooth_l1(
                        scores_view1['current_cover_upper_speed'],
                        scores_view2['current_cover_upper_speed'],
                        graph_targets['current_upper_valid'],
                    )
                    + _sym_smooth_l1(
                        scores_view1['future_cover_lower_speed'],
                        scores_view2['future_cover_lower_speed'],
                        graph_targets['future_lower_valid'],
                    )
                    + _sym_smooth_l1(
                        scores_view1['front_follow_upper_speed'],
                        scores_view2['front_follow_upper_speed'],
                        graph_targets['front_follow_upper_valid'],
                    )
                    + _sym_smooth_l1(
                        scores_view1['merge_flow_lower_speed'],
                        scores_view2['merge_flow_lower_speed'],
                        graph_targets['merge_flow_lower_valid'],
                    )
                ) / 8.0
                loss_state_consistency_timing = loss_state_consistency_timing + graph_consistency
            loss_state_consistency = (
                self.state_consistency_window_weight * loss_state_consistency_window
                + self.state_consistency_phase_weight * loss_state_consistency_phase
                + self.state_consistency_boundary_weight * loss_state_consistency_boundary
                + self.state_consistency_area_weight * loss_state_consistency_area
                + self.state_consistency_tempocc_weight * loss_state_consistency_tempocc
                + self.state_consistency_opportunity_weight * loss_state_consistency_opportunity
                + self.state_consistency_timing_weight * loss_state_consistency_timing
            )

        loss_phase_supervised = (
            (loss_decision_phase if supervise_decision else zero)
            + (loss_control_phase if supervise_control else zero)
        )
        direct_stage1_base_loss = zero
        if supervise_boundary:
            direct_stage1_base_loss = direct_stage1_base_loss + (
                self.energy_merge_weight * loss_merge
                + self.energy_junction_weight * loss_junction
                + self.energy_borrow_weight * loss_borrow
            )
        if supervise_dir:
            direct_stage1_base_loss = direct_stage1_base_loss + self.energy_relation_weight * loss_dir
        if supervise_window:
            direct_stage1_base_loss = direct_stage1_base_loss + self.energy_window_weight * loss_window
        if supervise_decision or supervise_control:
            direct_stage1_base_loss = direct_stage1_base_loss + self.energy_phase_weight * loss_phase_supervised
        if supervise_conflict_area:
            direct_stage1_base_loss = direct_stage1_base_loss + self.energy_conflict_area_weight * loss_conflict_area
        if supervise_tempocc:
            direct_stage1_base_loss = (
                direct_stage1_base_loss
                + self.temporary_occupancy_loss_weight * loss_temporary_occupancy
            )
        if supervise_opportunity:
            direct_stage1_base_loss = direct_stage1_base_loss + self.go_opportunity_loss_weight * loss_go_opportunity
        if supervise_area_status:
            direct_stage1_base_loss = (
                direct_stage1_base_loss
                + self.conflict_area_status_loss_weight * loss_conflict_area_status
            )
        if supervise_timing:
            direct_stage1_base_loss = direct_stage1_base_loss + self.conflict_timing_loss_weight * loss_conflict_timing
        if supervise_chase:
            direct_stage1_base_loss = direct_stage1_base_loss + self.energy_chase_weight * loss_chase
        if supervise_inside_area_go:
            direct_stage1_base_loss = direct_stage1_base_loss + self.inside_area_go_loss_weight * loss_inside_area_go
        if supervise_graph:
            direct_stage1_base_loss = direct_stage1_base_loss + (
                self.current_edge_valid_loss_weight * loss_current_edge_valid
                + self.current_edge_mode_loss_weight * loss_current_edge_mode
                + self.future_edge_valid_loss_weight * loss_future_edge_valid
                + self.future_edge_mode_loss_weight * loss_future_edge_mode
                + self.current_cover_upper_loss_weight * loss_current_cover_upper
                + self.future_cover_lower_loss_weight * loss_future_cover_lower
                + self.front_follow_upper_loss_weight * loss_front_follow_upper
                + self.merge_flow_lower_loss_weight * loss_merge_flow_lower
            )
        if supervise_edge_speed_consistency:
            direct_stage1_base_loss = (
                direct_stage1_base_loss
                + self.edge_speed_consistency_loss_weight * loss_edge_speed_consistency
            )
        semantic_transition_loss = zero
        semantic_transition_consistency_loss = zero
        if transition_stage1_scores is not None:
            def _transition_boundary_loss(pred_norm: torch.Tensor, target_mps: torch.Tensor) -> torch.Tensor:
                return _boundary_loss(pred_norm, target_mps)

            transition_loss_merge = 0.5 * (
                _transition_boundary_loss(transition_stage1_scores['merge_yld_max'], merge_yld_target)
                + _transition_boundary_loss(transition_stage1_scores['merge_go_min'], merge_go_target)
            )
            transition_loss_junction = 0.5 * (
                _transition_boundary_loss(transition_stage1_scores['junction_yld_max'], junction_yld_target)
                + _transition_boundary_loss(transition_stage1_scores['junction_go_min'], junction_go_target)
            )
            transition_loss_borrow = 0.5 * (
                _transition_boundary_loss(transition_stage1_scores['borrow_yld_max'], borrow_yld_target)
                + _transition_boundary_loss(transition_stage1_scores['borrow_go_min'], borrow_go_target)
            )
            transition_loss_window = F.cross_entropy(
                transition_stage1_scores['window_logits'].float(),
                window_target,
            )
            transition_loss_dir = F.cross_entropy(
                transition_stage1_scores['dir_logits'].float(),
                dir_target.clamp(min=0, max=3),
            )
            transition_loss_conflict_area = zero
            if conflict_area_target is not None:
                transition_area_raw = F.binary_cross_entropy_with_logits(
                    transition_stage1_scores['conflict_area_logits'].float(),
                    conflict_area_target.float(),
                    reduction='none',
                )
                transition_valid = conflict_area_valid_mask
                if transition_valid is None:
                    transition_valid = torch.ones_like(conflict_area_target, dtype=torch.bool)
                transition_pos = (conflict_area_target > 0.5) & transition_valid
                transition_neg = (conflict_area_target <= 0.5) & transition_valid
                if transition_pos.any() and transition_neg.any():
                    transition_loss_conflict_area = 0.5 * (
                        transition_area_raw[transition_pos].mean()
                        + transition_area_raw[transition_neg].mean()
                    )
                elif transition_pos.any():
                    transition_loss_conflict_area = transition_area_raw[transition_pos].mean()
                elif transition_neg.any():
                    transition_loss_conflict_area = transition_area_raw[transition_neg].mean()

            transition_loss_decision_phase = zero
            if decision_phase_valid_mask.any():
                transition_loss_decision_phase = F.cross_entropy(
                    transition_stage1_scores['decision_phase_logits'][decision_phase_valid_mask].float(),
                    decision_phase_target[decision_phase_valid_mask],
                )
            transition_loss_control_phase = zero
            if control_phase_valid_mask.any():
                transition_loss_control_phase = F.cross_entropy(
                    transition_stage1_scores['control_phase_logits'][control_phase_valid_mask].float(),
                    control_phase_target[control_phase_valid_mask],
                )
            transition_loss_phase = transition_loss_decision_phase + transition_loss_control_phase

            transition_loss_tempocc = zero
            transition_loss_go_opp = zero
            if temp_targets is not None:
                transition_temp_valid = temp_targets['valid']
                if transition_temp_valid.any():
                    transition_temp_raw = F.binary_cross_entropy_with_logits(
                        transition_stage1_scores['temporary_occupancy_logits'].float(),
                        temp_targets['bins'].float(),
                        reduction='none',
                    )
                    transition_loss_tempocc = transition_temp_raw[transition_temp_valid].mean()
                transition_go_valid = temp_targets['go_opportunity_valid']
                if transition_go_valid.any():
                    transition_target_probs = temp_targets['go_opportunity_target'][transition_go_valid].float()
                    transition_log_probs = F.log_softmax(
                        transition_stage1_scores['go_opportunity_logits'][transition_go_valid].float(),
                        dim=-1,
                    )
                    transition_loss_go_opp = -(
                        transition_target_probs * transition_log_probs
                    ).sum(dim=-1).mean()

            transition_loss_status = zero
            transition_loss_timing = zero
            transition_loss_inside_go = zero
            if timing_targets is not None:
                transition_loss_status = F.cross_entropy(
                    transition_stage1_scores['conflict_area_status_logits'].float(),
                    timing_targets['status'].clamp(min=0, max=3),
                )
                if timing_targets['valid'].any():
                    transition_loss_timing = F.smooth_l1_loss(
                        transition_stage1_scores['conflict_timing_values'][timing_targets['valid']].float(),
                        timing_targets['values'][timing_targets['valid']].float(),
                    )
                transition_inside_mask = timing_targets['status'] == 2
                if transition_inside_mask.any():
                    transition_inside_decision_target = torch.ones(
                        int(transition_inside_mask.sum().item()), device=device, dtype=torch.long
                    )
                    transition_inside_control_target = torch.full(
                        (int(transition_inside_mask.sum().item()),),
                        3,
                        device=device,
                        dtype=torch.long,
                    )
                    transition_loss_inside_go = (
                        F.cross_entropy(
                            transition_stage1_scores['decision_phase_logits'][transition_inside_mask].float(),
                            transition_inside_decision_target,
                        )
                        + F.cross_entropy(
                            transition_stage1_scores['control_phase_logits'][transition_inside_mask].float(),
                            transition_inside_control_target,
                        )
                    )

            transition_loss_chase = zero
            if self.use_chase_front_following_state and chase_targets is not None:
                transition_chase_has_lead = F.binary_cross_entropy_with_logits(
                    transition_stage1_scores['chase_has_lead_logit'].float(),
                    chase_targets['has_lead'].float(),
                )
                transition_chase_speed_target = (
                    chase_targets['speed_max'] / max(self.chase_speed_norm_scale, 1e-6)
                ).clamp(0.0, 1.0)
                transition_chase_speed = F.smooth_l1_loss(
                    transition_stage1_scores['chase_speed_max'].float(),
                    transition_chase_speed_target.float(),
                )
                transition_loss_chase = (
                    self.chase_has_lead_loss_weight * transition_chase_has_lead
                    + self.chase_speed_max_loss_weight * transition_chase_speed
                )

            transition_loss_current_edge_valid = zero
            transition_loss_current_edge_mode = zero
            transition_loss_future_edge_valid = zero
            transition_loss_future_edge_mode = zero
            transition_loss_current_cover_upper = zero
            transition_loss_future_cover_lower = zero
            transition_loss_front_follow_upper = zero
            transition_loss_merge_flow_lower = zero
            if graph_targets is not None:
                transition_loss_current_edge_valid = F.binary_cross_entropy_with_logits(
                    transition_stage1_scores['current_cover_edge_valid_logit'].float(),
                    graph_targets['current_valid'].float(),
                )
                if graph_targets['current_mode_valid'].any():
                    transition_loss_current_edge_mode = F.cross_entropy(
                        transition_stage1_scores['current_cover_edge_mode_logits'][
                            graph_targets['current_mode_valid']
                        ].float(),
                        graph_targets['current_mode'][graph_targets['current_mode_valid']],
                    )
                transition_loss_future_edge_valid = F.binary_cross_entropy_with_logits(
                    transition_stage1_scores['future_cover_edge_valid_logit'].float(),
                    graph_targets['future_valid'].float(),
                )
                if graph_targets['future_mode_valid'].any():
                    transition_loss_future_edge_mode = F.cross_entropy(
                        transition_stage1_scores['future_cover_edge_mode_logits'][
                            graph_targets['future_mode_valid']
                        ].float(),
                        graph_targets['future_mode'][graph_targets['future_mode_valid']],
                    )

                def _transition_graph_speed_loss(pred_norm: torch.Tensor, target_mps: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
                    if not valid.any():
                        return zero
                    target_norm = (
                        target_mps / max(self.stage1_boundary_norm_scale, 1e-6)
                    ).clamp(0.0, 1.0)
                    return F.smooth_l1_loss(
                        pred_norm[valid].float(),
                        target_norm[valid].float(),
                    )

                transition_loss_current_cover_upper = _transition_graph_speed_loss(
                    transition_stage1_scores['current_cover_upper_speed'],
                    graph_targets['current_upper'],
                    graph_targets['current_upper_valid'],
                )
                transition_loss_future_cover_lower = _transition_graph_speed_loss(
                    transition_stage1_scores['future_cover_lower_speed'],
                    graph_targets['future_lower'],
                    graph_targets['future_lower_valid'],
                )
                transition_loss_front_follow_upper = _transition_graph_speed_loss(
                    transition_stage1_scores['front_follow_upper_speed'],
                    graph_targets['front_follow_upper'],
                    graph_targets['front_follow_upper_valid'],
                )
                transition_loss_merge_flow_lower = _transition_graph_speed_loss(
                    transition_stage1_scores['merge_flow_lower_speed'],
                    graph_targets['merge_flow_lower'],
                    graph_targets['merge_flow_lower_valid'],
                )

            transition_loss_phase_supervised = (
                (transition_loss_decision_phase if supervise_decision else zero)
                + (transition_loss_control_phase if supervise_control else zero)
            )
            semantic_transition_loss = zero
            if supervise_boundary:
                semantic_transition_loss = semantic_transition_loss + (
                    self.energy_merge_weight * transition_loss_merge
                    + self.energy_junction_weight * transition_loss_junction
                    + self.energy_borrow_weight * transition_loss_borrow
                )
            if supervise_dir:
                semantic_transition_loss = (
                    semantic_transition_loss
                    + self.energy_relation_weight * transition_loss_dir
                )
            if supervise_window:
                semantic_transition_loss = (
                    semantic_transition_loss
                    + self.energy_window_weight * transition_loss_window
                )
            if supervise_decision or supervise_control:
                semantic_transition_loss = (
                    semantic_transition_loss
                    + self.energy_phase_weight * transition_loss_phase_supervised
                )
            if supervise_conflict_area:
                semantic_transition_loss = (
                    semantic_transition_loss
                    + self.energy_conflict_area_weight * transition_loss_conflict_area
                )
            if supervise_tempocc:
                semantic_transition_loss = (
                    semantic_transition_loss
                    + self.temporary_occupancy_loss_weight * transition_loss_tempocc
                )
            if supervise_opportunity:
                semantic_transition_loss = (
                    semantic_transition_loss
                    + self.go_opportunity_loss_weight * transition_loss_go_opp
                )
            if supervise_area_status:
                semantic_transition_loss = (
                    semantic_transition_loss
                    + self.conflict_area_status_loss_weight * transition_loss_status
                )
            if supervise_timing:
                semantic_transition_loss = (
                    semantic_transition_loss
                    + self.conflict_timing_loss_weight * transition_loss_timing
                )
            if supervise_chase:
                semantic_transition_loss = (
                    semantic_transition_loss
                    + self.energy_chase_weight * transition_loss_chase
                )
            if supervise_inside_area_go:
                semantic_transition_loss = (
                    semantic_transition_loss
                    + self.inside_area_go_loss_weight * transition_loss_inside_go
                )
            if supervise_graph:
                semantic_transition_loss = semantic_transition_loss + (
                    self.current_edge_valid_loss_weight * transition_loss_current_edge_valid
                    + self.current_edge_mode_loss_weight * transition_loss_current_edge_mode
                    + self.future_edge_valid_loss_weight * transition_loss_future_edge_valid
                    + self.future_edge_mode_loss_weight * transition_loss_future_edge_mode
                    + self.current_cover_upper_loss_weight * transition_loss_current_cover_upper
                    + self.future_cover_lower_loss_weight * transition_loss_future_cover_lower
                    + self.front_follow_upper_loss_weight * transition_loss_front_follow_upper
                    + self.merge_flow_lower_loss_weight * transition_loss_merge_flow_lower
                )

            def _transition_masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
                if mask is None:
                    return values.mean()
                mask = mask.to(device=values.device, dtype=torch.bool)
                while mask.dim() < values.dim():
                    mask = mask.unsqueeze(-1)
                mask = mask.expand_as(values)
                if not mask.any():
                    return zero
                return values[mask].mean()

            def _transition_sym_kl(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
                a = a.float()
                b = b.float()
                log_a = F.log_softmax(a, dim=-1)
                log_b = F.log_softmax(b, dim=-1)
                prob_a = log_a.exp()
                prob_b = log_b.exp()
                loss_ab = (prob_a.detach() * (log_a.detach() - log_b)).sum(dim=-1)
                loss_ba = (prob_b.detach() * (log_b.detach() - log_a)).sum(dim=-1)
                return _transition_masked_mean(0.5 * (loss_ab + loss_ba), mask)

            def _transition_sym_mse(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
                loss = 0.5 * (
                    F.mse_loss(a.float(), b.detach().float(), reduction='none')
                    + F.mse_loss(b.float(), a.detach().float(), reduction='none')
                )
                return _transition_masked_mean(loss, mask)

            transition_consistency_terms = []
            if supervise_window:
                transition_consistency_terms.append(
                    _transition_sym_kl(raw_stage1_scores['window_logits'], transition_stage1_scores['window_logits'])
                )
            if supervise_dir:
                transition_consistency_terms.append(
                    _transition_sym_kl(raw_stage1_scores['dir_logits'], transition_stage1_scores['dir_logits'])
                )
            if supervise_decision:
                transition_consistency_terms.append(_transition_sym_kl(
                    raw_stage1_scores['decision_phase_logits'],
                    transition_stage1_scores['decision_phase_logits'],
                    decision_phase_valid_mask,
                ))
            if supervise_control:
                transition_consistency_terms.append(_transition_sym_kl(
                    raw_stage1_scores['control_phase_logits'],
                    transition_stage1_scores['control_phase_logits'],
                    control_phase_valid_mask,
                ))
            if supervise_conflict_area:
                transition_consistency_terms.append(_transition_sym_mse(
                    raw_stage1_scores['conflict_area_logits'],
                    transition_stage1_scores['conflict_area_logits'],
                    conflict_area_valid_mask,
                ))
            if temp_targets is not None:
                if supervise_tempocc:
                    transition_consistency_terms.append(_transition_sym_mse(
                        raw_stage1_scores['temporary_occupancy_logits'],
                        transition_stage1_scores['temporary_occupancy_logits'],
                        temp_targets['valid'],
                    ))
                if supervise_opportunity:
                    transition_consistency_terms.append(_transition_sym_kl(
                        raw_stage1_scores['go_opportunity_logits'],
                        transition_stage1_scores['go_opportunity_logits'],
                        temp_targets['go_opportunity_valid'],
                    ))
            if timing_targets is not None:
                if supervise_area_status:
                    transition_consistency_terms.append(_transition_sym_kl(
                        raw_stage1_scores['conflict_area_status_logits'],
                        transition_stage1_scores['conflict_area_status_logits'],
                    ))
                if supervise_timing:
                    transition_consistency_terms.append(_transition_sym_mse(
                        raw_stage1_scores['conflict_timing_values'],
                        transition_stage1_scores['conflict_timing_values'],
                        timing_targets['valid'],
                    ))
            if supervise_boundary:
                for key in (
                    'merge_yld_max',
                    'merge_go_min',
                    'junction_yld_max',
                    'junction_go_min',
                    'borrow_yld_max',
                    'borrow_go_min',
                ):
                    transition_consistency_terms.append(
                        _transition_sym_mse(raw_stage1_scores[key], transition_stage1_scores[key])
                    )
            if supervise_chase and self.use_chase_front_following_state and chase_targets is not None:
                transition_consistency_terms.extend([
                    _transition_sym_mse(
                        raw_stage1_scores['chase_has_lead_logit'],
                        transition_stage1_scores['chase_has_lead_logit'],
                    ),
                    _transition_sym_mse(
                        raw_stage1_scores['chase_speed_max'],
                        transition_stage1_scores['chase_speed_max'],
                    ),
                ])
            if supervise_graph and graph_targets is not None:
                transition_consistency_terms.extend([
                    _transition_sym_mse(
                        raw_stage1_scores['current_cover_edge_valid_logit'],
                        transition_stage1_scores['current_cover_edge_valid_logit'],
                    ),
                    _transition_sym_kl(
                        raw_stage1_scores['current_cover_edge_mode_logits'],
                        transition_stage1_scores['current_cover_edge_mode_logits'],
                        graph_targets['current_mode_valid'],
                    ),
                    _transition_sym_mse(
                        raw_stage1_scores['future_cover_edge_valid_logit'],
                        transition_stage1_scores['future_cover_edge_valid_logit'],
                    ),
                    _transition_sym_kl(
                        raw_stage1_scores['future_cover_edge_mode_logits'],
                        transition_stage1_scores['future_cover_edge_mode_logits'],
                        graph_targets['future_mode_valid'],
                    ),
                    _transition_sym_mse(
                        raw_stage1_scores['current_cover_upper_speed'],
                        transition_stage1_scores['current_cover_upper_speed'],
                        graph_targets['current_upper_valid'],
                    ),
                    _transition_sym_mse(
                        raw_stage1_scores['future_cover_lower_speed'],
                        transition_stage1_scores['future_cover_lower_speed'],
                        graph_targets['future_lower_valid'],
                    ),
                    _transition_sym_mse(
                        raw_stage1_scores['front_follow_upper_speed'],
                        transition_stage1_scores['front_follow_upper_speed'],
                        graph_targets['front_follow_upper_valid'],
                    ),
                    _transition_sym_mse(
                        raw_stage1_scores['merge_flow_lower_speed'],
                        transition_stage1_scores['merge_flow_lower_speed'],
                        graph_targets['merge_flow_lower_valid'],
                    ),
                ])
            semantic_transition_consistency_loss = (
                torch.stack(transition_consistency_terms).mean()
                if transition_consistency_terms else zero
            )

        if transition_stage1_scores is not None:
            stage1_loss = (
                self.semantic_direct_aux_loss_weight * direct_stage1_base_loss
                + self.semantic_transition_loss_weight * semantic_transition_loss
                + self.semantic_transition_consistency_weight * semantic_transition_consistency_loss
                + self.state_consistency_loss_weight * loss_state_consistency
            )
            semantic_next_token_loss = semantic_transition_loss
            semantic_direct_aux_log_loss = direct_stage1_base_loss
        else:
            stage1_loss = (
                direct_stage1_base_loss
                + self.state_consistency_loss_weight * loss_state_consistency
            )
            semantic_next_token_loss = direct_stage1_base_loss if transition_only_state else zero
            semantic_direct_aux_log_loss = zero if transition_only_state else direct_stage1_base_loss
        return {
            'stage1_loss': stage1_loss,
            # Backward-compatible alias while downstream logs/agents migrate.
            'energy_loss': stage1_loss,
            'chase_loss': loss_chase,
            'chase_has_lead_loss': loss_chase_has_lead,
            'chase_speed_max_loss': loss_chase_speed_max,
            'current_edge_valid_loss': loss_current_edge_valid,
            'current_edge_mode_loss': loss_current_edge_mode,
            'future_edge_valid_loss': loss_future_edge_valid,
            'future_edge_mode_loss': loss_future_edge_mode,
            'current_cover_upper_loss': loss_current_cover_upper,
            'future_cover_lower_loss': loss_future_cover_lower,
            'front_follow_upper_loss': loss_front_follow_upper,
            'merge_flow_lower_loss': loss_merge_flow_lower,
            'edge_speed_consistency_loss': loss_edge_speed_consistency,
            'merge_loss': loss_merge,
            'junction_loss': loss_junction,
            'borrow_loss': loss_borrow,
            'cross_loss': loss_cross,
            'pedestrian_loss': loss_ped,
            'merge_yld_loss': loss_merge_yld,
            'merge_go_loss': loss_merge_go,
            'junction_yld_loss': loss_junction_yld,
            'junction_go_loss': loss_junction_go,
            'borrow_yld_loss': loss_borrow_yld,
            'borrow_go_loss': loss_borrow_go,
            'cross_yld_loss': loss_junction_yld + loss_borrow_yld,
            'cross_go_loss': loss_junction_go + loss_borrow_go,
            'merge_active_loss': loss_merge_active,
            'junction_active_loss': loss_junction_active,
            'borrow_active_loss': loss_borrow_active,
            'cross_active_loss': loss_cross_active,
            'relation_loss': loss_dir,
            'dir_loss': loss_dir,
            'conflict_area_loss': loss_conflict_area,
            'window_loss': loss_window,
            'phase_loss': loss_phase,
            'decision_phase_loss': loss_decision_phase,
            'control_phase_loss': loss_control_phase,
            'temporary_occupancy_loss': loss_temporary_occupancy,
            'go_opportunity_loss': loss_go_opportunity,
            'conflict_area_status_loss': loss_conflict_area_status,
            'conflict_timing_loss': loss_conflict_timing,
            'inside_area_go_loss': loss_inside_area_go,
            'state_consistency_loss': loss_state_consistency,
            'state_consistency_window_loss': loss_state_consistency_window,
            'state_consistency_phase_loss': loss_state_consistency_phase,
            'state_consistency_timing_loss': loss_state_consistency_timing,
            'state_consistency_boundary_loss': loss_state_consistency_boundary,
            'state_consistency_area_loss': loss_state_consistency_area,
            'state_consistency_tempocc_loss': loss_state_consistency_tempocc,
            'state_consistency_opportunity_loss': loss_state_consistency_opportunity,
            'semantic_direct_aux_loss': semantic_direct_aux_log_loss,
            'semantic_next_token_loss': semantic_next_token_loss,
            'semantic_transition_loss': semantic_transition_loss,
            'semantic_transition_consistency_loss': semantic_transition_consistency_loss,
            'merge_yld_max_loss': loss_merge_yld,
            'merge_go_min_loss': loss_merge_go,
            'junction_yld_max_loss': loss_junction_yld,
            'junction_go_min_loss': loss_junction_go,
            'borrow_yld_max_loss': loss_borrow_yld,
            'borrow_go_min_loss': loss_borrow_go,
        }

    def _train_branch_enabled_with_schedule(self, until_epoch, after_update_every) -> bool:
        """1-based epoch cutoff with optional lower-frequency updates after the cutoff."""
        if until_epoch is None:
            return True
        try:
            until_epoch = int(until_epoch)
        except (TypeError, ValueError):
            return True
        if until_epoch <= 0:
            return True
        current_epoch_1based = self._current_epoch + 1
        if current_epoch_1based <= until_epoch:
            return True
        if after_update_every is None:
            return False
        try:
            after_update_every = int(after_update_every)
        except (TypeError, ValueError):
            return False
        if after_update_every <= 1:
            return True
        return (self._current_batch_idx % after_update_every) == 0

    # ========== Forward (DDP-compatible) ==========
    def forward(self, batch: Dict[str, torch.Tensor],
                return_loss_dict: bool = False,
                phase: str = 'unified'):
        """DDP-compatible Route B semantic-state forward."""
        loss_dict = self.compute_split_loss(batch)

        if return_loss_dict:
            return loss_dict
        else:
            return loss_dict['total_loss']

    # ========== Unified Training: Single Forward Pass ==========
    def compute_split_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Split-forward Route B training.

        1. Ego path: waypoint-token denoising for pred_x0 + route prediction
        2. Shared stage1 semantic-state heads on clean/noisy Route B context
        """
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        trajectory = batch['agent_pos'].to(device=device, dtype=model_dtype)  # (B, T, 2)
        B, T, D = trajectory.shape
        good_route_mask = self._get_good_route_mask(batch, device)

        transfuser_bev_feature = batch['transfuser_bev_feature'].to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = batch['transfuser_bev_feature_upsample'].to(device=device, dtype=model_dtype)
        transfuser_lidar_bev = self._get_transfuser_lidar_bev(batch, device, model_dtype)
        ego_status = batch['ego_status'].to(device=device, dtype=model_dtype)

        route_gt = batch.get('route', None)
        if route_gt is not None:
            route_gt = route_gt.to(device=device, dtype=model_dtype)
        else:
            raise KeyError("Joint Route B ego diffusion requires 'route' in the training batch")

        has_stage1_labels = self._has_stage1_labels(batch)
        train_speed_head_active = self._train_branch_enabled_with_schedule(
            self.train_speed_head_until_epoch,
            self.train_speed_head_after_update_every,
        )
        train_stage1_active = (
            has_stage1_labels
            and self.train_energy
            and self._train_branch_enabled_with_schedule(
                self.train_stage1_speed_energy_until_epoch,
                self.train_stage1_speed_energy_after_update_every,
            )
        )

        bev_proj = self.model.decoder.compute_bev_proj(transfuser_bev_feature)
        prev_route_coarse_memory = (
            self._get_route_prev_coarse_memory(
                batch,
                device=device,
                model_dtype=model_dtype,
                require=False,
            )
            if self.use_route_prev_coarse_memory else None
        )

        # ===== Forward 1: Ego denoising (M=1) =====
        if route_gt.shape[1] != self.num_waypoints:
            raise ValueError(f"Expected route_gt with {self.num_waypoints} waypoints, got {route_gt.shape}")
        traj_route_normed = self.joint_abs_to_norm(trajectory, route_gt)  # (B, T_joint, 2)
        diff_timesteps = torch.randint(0, self.train_max_timesteps, (B,), device=device).long()
        noise = torch.randn(B, self.horizon + self.num_waypoints, D, dtype=torch.float32, device=device)
        noisy_flat = self.diffusion_scheduler.add_noise(
            original_samples=traj_route_normed,
            noise=noise,
            timesteps=diff_timesteps,
        )
        noisy_joint = noisy_flat.unsqueeze(1)  # (B, 1, T_joint, 2)
        noisy_joint_abs = self.joint_norm_to_abs(noisy_joint)

        traj_branch_condition = None
        traj_branch_condition_details = None
        branch_condition_schedule = None
        if self.use_traj_branch_condition and has_stage1_labels:
            traj_branch_condition, traj_branch_condition_details = self._build_stage1_branch_condition_gt(
                batch=batch,
                ego_status=ego_status,
                device=device,
                model_dtype=model_dtype,
            )
            if traj_branch_condition is not None:
                branch_condition_schedule = self._build_traj_condition_schedule(
                    diff_timesteps, device=device, model_dtype=model_dtype
                )

        def _forward_ego_with_branch(branch_condition_tensor: Optional[torch.Tensor]):
            branch_input = branch_condition_tensor
            if branch_input is not None and self.traj_branch_condition_detach:
                branch_input = branch_input.detach()
            return self.model.forward_ego(
                x_t=noisy_joint,
                x_t_abs=noisy_joint_abs,
                timestep=diff_timesteps,
                transfuser_bev_feature=transfuser_bev_feature,
                transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                ego_status=ego_status,
                bev_proj_cached=bev_proj,
                transfuser_lidar_bev=transfuser_lidar_bev,
                branch_condition=branch_input,
                branch_condition_scale=self.traj_branch_condition_scale,
                branch_condition_schedule=branch_condition_schedule if branch_input is not None else None,
                prev_route_coarse_memory=prev_route_coarse_memory,
            )

        poses_reg, route_pred, _, _, speed_pred, speed_profile_pred = _forward_ego_with_branch(
            traj_branch_condition
        )
        poses_reg_abs = self.norm_to_abs(poses_reg)
        route_pred_abs = self.route_norm_to_abs(route_pred)
        traj_target = trajectory.unsqueeze(1)
        reg_per_sample = self._reduce_per_sample(
            F.l1_loss(poses_reg_abs, traj_target, reduction='none')
        )
        loss_reg = self._masked_batch_mean(reg_per_sample, good_route_mask)

        route_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        if route_pred is not None:
            route_recon = self._reduce_per_sample(
                F.l1_loss(route_pred_abs, route_gt, reduction='none')
            )
            route_fde = self._reduce_per_sample(
                F.l1_loss(route_pred_abs[:, -1], route_gt[:, -1], reduction='none')
            )
            route_loss = self._masked_batch_mean(
                route_recon + self.route_final_loss_weight * route_fde,
                good_route_mask,
            )

        # Speed loss: two-hot cross-entropy
        speed_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        speed_profile_loss = torch.tensor(0.0, device=device, dtype=model_dtype)
        speed_profile_step_losses = []
        if speed_pred is not None and train_speed_head_active:
            speed_target = self._compute_speed_target(trajectory, device, batch=batch)
            if speed_target is not None:
                speed_per_sample = F.cross_entropy(
                    speed_pred.float(),
                    speed_target,
                    reduction='none',
                )
                speed_loss = self._masked_batch_mean(speed_per_sample, good_route_mask)
        if self.use_speed_profile_head and speed_profile_pred is not None and train_speed_head_active:
            speed_profile_target = self._compute_speed_profile_target(trajectory, device, model_dtype)
            if speed_profile_target is not None:
                step_weights = self.speed_profile_step_weights[:speed_profile_pred.shape[1]].to(
                    device=device, dtype=model_dtype
                )
                profile_per_step = F.smooth_l1_loss(
                    speed_profile_pred,
                    speed_profile_target,
                    reduction='none',
                )
                speed_profile_step_losses = [
                    self._masked_batch_mean(profile_per_step[:, step_idx], good_route_mask)
                    for step_idx in range(profile_per_step.shape[1])
                ]
                profile_per_sample = (
                    profile_per_step * step_weights.unsqueeze(0)
                ).sum(dim=-1) / step_weights.sum().clamp(min=1e-6)
                speed_profile_loss = self._masked_batch_mean(profile_per_sample, good_route_mask)

        # ===== Forward 2: Stage1 state training (clean/noisy shared path) =====
        zero_t = torch.tensor(0.0, device=device, dtype=model_dtype)
        energy_loss = zero_t
        loss_front = loss_left = loss_right = loss_ped = loss_off = zero_t
        loss_route = zero_t
        stage1_extra_losses = {}

        if train_stage1_active:
            stage1_loss_dict = self._compute_shared_stage1_loss(
                batch=batch,
                trajectory=trajectory,
                route_gt=route_gt,
                transfuser_bev_feature=transfuser_bev_feature,
                transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                transfuser_lidar_bev=transfuser_lidar_bev,
                ego_status=ego_status,
                bev_proj=bev_proj,
                device=device,
                model_dtype=model_dtype,
                noisy_joint=noisy_joint,
                noisy_joint_abs=noisy_joint_abs,
                diff_timesteps=diff_timesteps,
                branch_condition=traj_branch_condition.detach() if (
                    traj_branch_condition is not None and self.traj_branch_condition_detach
                ) else traj_branch_condition,
                branch_condition_schedule=branch_condition_schedule,
                prev_route_coarse_memory=prev_route_coarse_memory,
                speed_pred_for_consistency=speed_pred,
            )
            energy_loss = stage1_loss_dict['stage1_loss']
            loss_front = stage1_loss_dict['chase_loss']
            loss_left = stage1_loss_dict['merge_loss']
            loss_right = stage1_loss_dict['cross_loss']
            loss_ped = stage1_loss_dict['pedestrian_loss']
            loss_off = zero_t
            stage1_extra_losses = {
                'stage1_merge_yld_loss': stage1_loss_dict['merge_yld_loss'],
                'stage1_merge_go_loss': stage1_loss_dict['merge_go_loss'],
                'stage1_junction_yld_loss': stage1_loss_dict['junction_yld_loss'],
                'stage1_junction_go_loss': stage1_loss_dict['junction_go_loss'],
                'stage1_borrow_yld_loss': stage1_loss_dict['borrow_yld_loss'],
                'stage1_borrow_go_loss': stage1_loss_dict['borrow_go_loss'],
                'stage1_cross_yld_loss': stage1_loss_dict['cross_yld_loss'],
                'stage1_cross_go_loss': stage1_loss_dict['cross_go_loss'],
                'stage1_merge_active_loss': stage1_loss_dict['merge_active_loss'],
                'stage1_junction_active_loss': stage1_loss_dict['junction_active_loss'],
                'stage1_borrow_active_loss': stage1_loss_dict['borrow_active_loss'],
                'stage1_cross_active_loss': stage1_loss_dict['cross_active_loss'],
                'stage1_relation_loss': stage1_loss_dict['relation_loss'],
                'stage1_dir_loss': stage1_loss_dict['dir_loss'],
                'stage1_conflict_area_loss': stage1_loss_dict.get('conflict_area_loss', zero_t),
                'stage1_window_loss': stage1_loss_dict.get('window_loss', zero_t),
                'stage1_phase_loss': stage1_loss_dict.get('phase_loss', zero_t),
                'stage1_decision_phase_loss': stage1_loss_dict.get('decision_phase_loss', zero_t),
                'stage1_control_phase_loss': stage1_loss_dict.get('control_phase_loss', zero_t),
                'stage1_temporary_occupancy_loss': stage1_loss_dict.get('temporary_occupancy_loss', zero_t),
                'stage1_go_opportunity_loss': stage1_loss_dict.get('go_opportunity_loss', zero_t),
                'stage1_conflict_area_status_loss': stage1_loss_dict.get('conflict_area_status_loss', zero_t),
                'stage1_conflict_timing_loss': stage1_loss_dict.get('conflict_timing_loss', zero_t),
                'stage1_inside_area_go_loss': stage1_loss_dict.get('inside_area_go_loss', zero_t),
                'stage1_chase_loss': stage1_loss_dict.get('chase_loss', zero_t),
                'stage1_chase_has_lead_loss': stage1_loss_dict.get('chase_has_lead_loss', zero_t),
                'stage1_chase_speed_max_loss': stage1_loss_dict.get('chase_speed_max_loss', zero_t),
                'stage1_current_edge_valid_loss': stage1_loss_dict.get('current_edge_valid_loss', zero_t),
                'stage1_current_edge_mode_loss': stage1_loss_dict.get('current_edge_mode_loss', zero_t),
                'stage1_future_edge_valid_loss': stage1_loss_dict.get('future_edge_valid_loss', zero_t),
                'stage1_future_edge_mode_loss': stage1_loss_dict.get('future_edge_mode_loss', zero_t),
                'stage1_current_cover_upper_loss': stage1_loss_dict.get('current_cover_upper_loss', zero_t),
                'stage1_future_cover_lower_loss': stage1_loss_dict.get('future_cover_lower_loss', zero_t),
                'stage1_front_follow_upper_loss': stage1_loss_dict.get('front_follow_upper_loss', zero_t),
                'stage1_merge_flow_lower_loss': stage1_loss_dict.get('merge_flow_lower_loss', zero_t),
                'stage1_edge_speed_consistency_loss': stage1_loss_dict.get('edge_speed_consistency_loss', zero_t),
                'stage1_state_consistency_loss': stage1_loss_dict.get('state_consistency_loss', zero_t),
                'stage1_state_consistency_window_loss': stage1_loss_dict.get('state_consistency_window_loss', zero_t),
                'stage1_state_consistency_phase_loss': stage1_loss_dict.get('state_consistency_phase_loss', zero_t),
                'stage1_state_consistency_timing_loss': stage1_loss_dict.get('state_consistency_timing_loss', zero_t),
                'stage1_state_consistency_boundary_loss': stage1_loss_dict.get('state_consistency_boundary_loss', zero_t),
                'stage1_state_consistency_area_loss': stage1_loss_dict.get('state_consistency_area_loss', zero_t),
                'stage1_state_consistency_tempocc_loss': stage1_loss_dict.get('state_consistency_tempocc_loss', zero_t),
                'stage1_state_consistency_opportunity_loss': stage1_loss_dict.get('state_consistency_opportunity_loss', zero_t),
                'stage1_semantic_direct_aux_loss': stage1_loss_dict.get('semantic_direct_aux_loss', zero_t),
                'stage1_semantic_next_token_loss': stage1_loss_dict.get('semantic_next_token_loss', zero_t),
                'stage1_semantic_transition_loss': stage1_loss_dict.get('semantic_transition_loss', zero_t),
                'stage1_semantic_transition_consistency_loss': stage1_loss_dict.get('semantic_transition_consistency_loss', zero_t),
                'stage1_merge_yld_max_loss': stage1_loss_dict.get('merge_yld_max_loss', zero_t),
                'stage1_merge_go_min_loss': stage1_loss_dict.get('merge_go_min_loss', zero_t),
                'stage1_junction_yld_max_loss': stage1_loss_dict.get('junction_yld_max_loss', zero_t),
                'stage1_junction_go_min_loss': stage1_loss_dict.get('junction_go_min_loss', zero_t),
                'stage1_borrow_yld_max_loss': stage1_loss_dict.get('borrow_yld_max_loss', zero_t),
                'stage1_borrow_go_min_loss': stage1_loss_dict.get('borrow_go_min_loss', zero_t),
            }
            stage1_extra_losses.update({
                key.replace('stage1_', 'energy_', 1): value
                for key, value in stage1_extra_losses.items()
                if key.startswith('stage1_')
            })
        alignment_loss = torch.tensor(0.0, device=device, dtype=model_dtype)

        total_loss = (
            self.energy_loss_weight * energy_loss
            + self.reg_loss_weight * loss_reg
            + self.route_loss_weight * route_loss
            + self.speed_loss_weight * speed_loss
            + self.speed_profile_loss_weight * speed_profile_loss
        )

        loss_dict = {
            'total_loss': total_loss,
            'energy_loss': energy_loss,
            'energy_front_loss': loss_front,
            'energy_left_loss': loss_left,
            'energy_right_loss': loss_right,
            'energy_chase_loss': loss_front,
            'energy_merge_loss': loss_left,
            'energy_cross_loss': loss_right,
            'energy_ped_loss': loss_ped,
            'energy_pedestrian_loss': loss_ped,
            'energy_off_loss': loss_off,
            'energy_route_loss': loss_route,
            'reg_loss': loss_reg,
            'cls_loss': torch.tensor(0.0, device=device),
            'route_loss': route_loss,
            'speed_loss': speed_loss,
            'alignment_loss': alignment_loss,
        }
        if self.use_speed_profile_head:
            loss_dict['speed_profile_loss'] = speed_profile_loss
            for step_idx, step_loss in enumerate(speed_profile_step_losses):
                loss_dict[f'speed_profile_step{step_idx}_loss'] = step_loss
        if stage1_extra_losses:
            loss_dict['stage1_loss'] = energy_loss
        loss_dict.update(stage1_extra_losses)
        return loss_dict

    # ========== Legacy compute_loss (backward compatible) ==========
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Backward compatible: Route B semantic-state split loss."""
        return self.compute_split_loss(batch)

    # ========== Checkpoint Loading ==========
    @classmethod
    def load_checkpoint(cls, checkpoint_path, config, device='cuda'):
        """Load checkpoint with proper buffer restoration.

        Handles:
        - register_buffer(None) buffers that don't enter state_dict
        - Normalization stats registration from config before loading
        - Best checkpoint already has EMA weights applied (no separate EMA load needed)
        """
        import numpy as np

        policy = cls(config)

        # Register norm stats from config FIRST (makes buffers non-None so load_state_dict can find them)
        policy.register_norm_stats_from_config(config)
        route_abs_stats_path = config.get('route_abs_stats_path')
        if route_abs_stats_path:
            rdata = np.load(route_abs_stats_path)
            policy.register_route_abs_stats(rdata['route_abs_mean'], rdata['route_abs_std'])
        else:
            raise ValueError("route_abs_stats_path is required for joint Route B ego diffusion")

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)

        # Load state dict — buffers are now non-None so they can be matched
        missing, unexpected = policy.load_state_dict(sd, strict=False)

        if missing:
            print(f"  [load_checkpoint] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  [load_checkpoint] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        policy = policy.to(device)
        policy.eval()
        return policy, ckpt

    def build_roll_timesteps(self, num_steps=None, device=None):
        """Build DDIM rollout timesteps for inference.

        The existing multi-step schedule is kept unchanged. For 1-step inference,
        start from the highest training noise level instead of the degenerate t=0.
        """
        if num_steps is None:
            num_steps = self.num_inference_steps
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        max_t = int(self.train_max_timesteps)
        if max_t <= 0:
            raise ValueError(f"train_max_timesteps must be positive, got {max_t}")

        if num_steps == 1:
            roll_timesteps = np.array([max_t - 1], dtype=np.int64)
        else:
            step_ratio = max_t / num_steps
            roll_timesteps = (
                np.arange(0, num_steps) * step_ratio
            ).round()[::-1].copy().astype(np.int64)
            roll_timesteps = np.clip(roll_timesteps, 0, max_t - 1)

        roll_timesteps = torch.from_numpy(roll_timesteps)
        if device is not None:
            roll_timesteps = roll_timesteps.to(device)
        return roll_timesteps

    # ========== Inference with Energy Guidance ==========
    @torch.no_grad()
    def conditional_sample(
        self,
        transfuser_bev_feature: torch.Tensor,
        transfuser_bev_feature_upsample: torch.Tensor,
        ego_status: torch.Tensor,
        device: torch.device,
        model_dtype: torch.dtype,
        route_for_guidance: Optional[torch.Tensor] = None,
        energy_weights: Optional[Dict[str, float]] = None,
        transfuser_lidar_bev: Optional[torch.Tensor] = None,
        borrow_time_s: Optional[torch.Tensor] = None,
        prev_relation_probs: Optional[torch.Tensor] = None,
        disable_semantic_state_cache: bool = False,
    ):
        """
        DDIM from N(0,I) with annealed energy gradient guidance.

        Args:
            route_for_guidance: (B, num_waypoints, 2) optional route for target energy
            energy_weights: optional dict to override energy weights (LLM Router interface)

        Returns:
            dict with best_trajectory, route_pred, all_trajectories, energy_scores, etc.
        """
        B = transfuser_bev_feature.shape[0]
        M = self.num_samples
        T = self.horizon
        joint_T = self.horizon + self.num_waypoints
        if M != 1:
            raise NotImplementedError(f"Joint Route B ego diffusion currently expects num_samples=1, got {M}")
        self._require_route_abs_stats()

        # Start from pure Gaussian noise in joint normalized traj+route space
        x_t = torch.randn(B, M, joint_T, 2, device=device, dtype=torch.float32)
        bev_proj = self.model.decoder.compute_bev_proj(
            transfuser_bev_feature.to(device=device, dtype=model_dtype)
        )

        # Set up DDIM timestep schedule
        num_steps = self.num_inference_steps
        roll_timesteps = self.build_roll_timesteps(num_steps=num_steps, device=device)

        alphas_cumprod = self.diffusion_scheduler.alphas_cumprod.to(device)

        poses_cls = None
        route_pred = None
        energy_scores = None
        speed_pred = None
        speed_profile_pred = None
        traj_branch_condition_probs = None
        traj_branch_condition_details = None
        pass1_trajectory = None
        phase_go_smoothing_debug = None
        phase_go_smoothing_history_updated = False
        semantic_prev_state = None
        semantic_prev_valid = None
        semantic_fusion_debug = None
        transition_only_state = self.semantic_state_predictor_mode == 'transition_only'
        direct_only_state = self.semantic_state_predictor_mode == 'direct_only'
        direct_prev_modulated_state = self.semantic_state_predictor_mode == 'direct_prev_modulated'
        use_infer_semantic_transition = (
            (not disable_semantic_state_cache)
            and (not direct_only_state)
            and
            (self.use_semantic_state_fusion or transition_only_state or direct_prev_modulated_state)
            and self.use_semantic_state_transition
            and self.use_stage1_speed_energy
        )
        if use_infer_semantic_transition:
            semantic_prev_state = self._get_inference_semantic_prev_state(
                B, self.num_waypoints, device, model_dtype
            )
            semantic_prev_valid = semantic_prev_state['valid'].to(
                device=device, dtype=model_dtype
            ).reshape(-1)
        prev_route_coarse_memory = self._route_prev_coarse_memory_from_prev_state(
            semantic_prev_state,
            device=device,
            model_dtype=model_dtype,
        )

        for step_i, k in enumerate(roll_timesteps):
            t_cur = k.item()
            t_next = roll_timesteps[step_i + 1].item() if step_i + 1 < len(roll_timesteps) else 0

            # Get annealed energy weights for current noise level
            # ========== Forward pass 1: denoise x_t → pred_x0 ==========
            x_input = x_t.to(dtype=model_dtype)
            x_t_abs = self.joint_norm_to_abs(x_input)

            t_tensor = torch.full((B,), t_cur, dtype=torch.long, device=device)

            # Legacy anchor-energy guidance was removed in this Route B-only cleanup.
            with torch.no_grad():
                pass1_shared = None
                if self.use_traj_branch_condition and self.use_stage1_speed_energy:
                    pass1_shared = self.model.forward_ego(
                        x_t=x_input,
                        x_t_abs=x_t_abs,
                        timestep=t_tensor,
                        transfuser_bev_feature=transfuser_bev_feature,
                        transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                        ego_status=ego_status,
                        bev_proj_cached=bev_proj,
                        transfuser_lidar_bev=transfuser_lidar_bev,
                        return_intermediates=True,
                        prev_route_coarse_memory=prev_route_coarse_memory,
                    )
                    poses_reg = pass1_shared['poses_reg']
                    route_pred = pass1_shared['route_pred']
                    speed_pred = pass1_shared['speed_pred']
                    speed_profile_pred = pass1_shared['speed_profile_pred']
                else:
                    poses_reg, route_pred, _, _, speed_pred, speed_profile_pred = self.model.forward_ego(
                        x_t=x_input,
                        x_t_abs=x_t_abs,
                        timestep=t_tensor,
                        transfuser_bev_feature=transfuser_bev_feature,
                        transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                        ego_status=ego_status,
                        bev_proj_cached=bev_proj,
                        transfuser_lidar_bev=transfuser_lidar_bev,
                        prev_route_coarse_memory=prev_route_coarse_memory,
                    )
                pass1_trajectory = self.norm_to_abs(poses_reg.detach())
                if self.use_traj_branch_condition and self.use_stage1_speed_energy:
                    route_pred_abs = self.route_norm_to_abs(route_pred.detach())
                    branch_speed_ref = (
                        self.decode_speed_two_hot(speed_pred, self.model.speed_classes)
                        if speed_pred is not None else
                        ego_status[:, -1, 0].to(device=device, dtype=model_dtype)
                    )
                    stage1_raw_scores = None
                    if pass1_shared is not None:
                        stage1_speed_samples = self._build_local_phase_energy_samples(
                            branch_speed_ref.detach(), device, model_dtype
                        )
                        if not transition_only_state:
                            stage1_raw_scores = self.model.compute_shared_stage1_from_ego_outputs(
                                traj_out=pass1_shared['traj_out'],
                                route_out=pass1_shared['route_out'],
                                speed_out=pass1_shared['speed_out'],
                                route_points=pass1_shared['route_points'],
                                conditioning=pass1_shared['conditioning'],
                                speed_samples=stage1_speed_samples,
                            )
                        if use_infer_semantic_transition and semantic_prev_state is not None:
                            transition_stage1_raw_scores = (
                                self._compute_semantic_transition_scores_from_shared(
                                    pass1_shared,
                                    prev_state=semantic_prev_state,
                                    speed_samples=stage1_speed_samples,
                                )
                            )
                            if transition_only_state:
                                stage1_raw_scores = transition_stage1_raw_scores
                                semantic_fusion_debug = {
                                    'enabled': torch.ones((), device=device, dtype=model_dtype),
                                    'gate': torch.ones((B,), device=device, dtype=model_dtype),
                                }
                            elif direct_prev_modulated_state:
                                stage1_raw_scores = transition_stage1_raw_scores
                                semantic_fusion_debug = {
                                    'enabled': torch.zeros((), device=device, dtype=model_dtype),
                                    'gate': semantic_prev_valid.detach() if semantic_prev_valid is not None else torch.zeros((B,), device=device, dtype=model_dtype),
                                }
                            else:
                                stage1_raw_scores, semantic_fusion_debug = self._fuse_stage1_raw_scores(
                                    stage1_raw_scores,
                                    transition_stage1_raw_scores,
                                    semantic_prev_valid,
                                )
                    traj_branch_condition_probs, traj_branch_condition_details = self._infer_traj_branch_condition(
                        stage1_raw_scores=stage1_raw_scores,
                        speed_ref=branch_speed_ref.detach(),
                        borrow_time_s=borrow_time_s,
                        prev_relation_probs=prev_relation_probs,
                        device=device,
                        model_dtype=model_dtype,
                    )
                    (
                        traj_branch_condition_probs,
                        traj_branch_condition_details,
                        phase_go_smoothing_debug,
                        phase_go_smoothing_history_just_updated,
                    ) = self._apply_phase_go_smoothing_override(
                        traj_branch_condition_probs,
                        traj_branch_condition_details,
                        update_history=not phase_go_smoothing_history_updated,
                        device=device,
                        model_dtype=model_dtype,
                    )
                    phase_go_smoothing_history_updated = (
                        phase_go_smoothing_history_updated
                        or phase_go_smoothing_history_just_updated
                    )
                    branch_schedule = self._build_traj_condition_schedule(
                        t_tensor, device=device, model_dtype=model_dtype
                    )
                    branch_input = traj_branch_condition_probs.detach() if self.traj_branch_condition_detach else traj_branch_condition_probs
                    poses_reg, route_pred, _, _, speed_pred, speed_profile_pred = self.model.forward_ego(
                        x_t=x_input,
                        x_t_abs=x_t_abs,
                        timestep=t_tensor,
                        transfuser_bev_feature=transfuser_bev_feature,
                        transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                        ego_status=ego_status,
                        bev_proj_cached=bev_proj,
                        transfuser_lidar_bev=transfuser_lidar_bev,
                        branch_condition=branch_input,
                        branch_condition_scale=self.traj_branch_condition_scale,
                        branch_condition_schedule=branch_schedule,
                        prev_route_coarse_memory=prev_route_coarse_memory,
                    )
            energy_scores = None
            pred_x0_corrected = torch.cat([
                poses_reg.float(),
                route_pred.unsqueeze(1).float(),
            ], dim=2)

            # ========== DDIM Step with corrected pred_x0 ==========
            alpha_t = alphas_cumprod[t_cur]
            alpha_next = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)

            pred_eps = (x_t - alpha_t.sqrt() * pred_x0_corrected) / (1 - alpha_t).sqrt().clamp(min=1e-8)
            x_t = alpha_next.sqrt() * pred_x0_corrected + (1 - alpha_next).sqrt() * pred_eps

        # ========== Output trajectory ==========
        final_joint_abs = self.joint_norm_to_abs(pred_x0_corrected)  # (B, 1, T_joint, 2)
        final_traj_abs = final_joint_abs[:, :, :self.horizon, :]
        route_pred = final_joint_abs[:, 0, self.horizon:, :]
        best_trajectory = final_traj_abs.squeeze(1)  # (B, T, 2)

        # Decode speed prediction to scalar m/s
        target_speed_pred = None
        if speed_pred is not None:
            target_speed_pred = self.decode_speed_two_hot(
                speed_pred, self.model.speed_classes
            )  # (B,)

        stage1_scores = None
        stage1_speed_samples = None
        stage1_speed_query_center = None
        stage1_speed_ref_speeds = None
        stage1_ref_scores = None
        if self.use_stage1_speed_energy:
            # Query stage1 state heads around the current feasible ego speed rather than
            # around the nominal speed-head output. This keeps the queried bucket locally
            # reachable during closed-loop control, especially when the speed head wants to
            # stop but the vehicle is still moving quickly.
            center_speed = ego_status[:, -1, 0].to(device=device, dtype=model_dtype)
            stage1_speed_query_center = center_speed
            stage1_speed_samples = self._build_stage1_speed_samples(center_speed, device, model_dtype)
            best_joint_norm = self.joint_abs_to_norm(best_trajectory, route_pred.detach()).unsqueeze(1)
            best_joint_abs = torch.cat(
                [best_trajectory.unsqueeze(1), route_pred.detach().unsqueeze(1)],
                dim=2,
            )
            shared_eval = self.model.forward_ego(
                x_t=best_joint_norm,
                x_t_abs=best_joint_abs,
                timestep=torch.zeros(B, dtype=torch.long, device=device),
                transfuser_bev_feature=transfuser_bev_feature,
                transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                ego_status=ego_status,
                bev_proj_cached=bev_proj,
                transfuser_lidar_bev=transfuser_lidar_bev,
                return_intermediates=True,
                prev_route_coarse_memory=prev_route_coarse_memory,
            )
            stage1_scores_raw = None
            if not transition_only_state:
                stage1_scores_raw = self.model.compute_shared_stage1_from_ego_outputs(
                    traj_out=shared_eval['traj_out'],
                    route_out=shared_eval['route_out'],
                    speed_out=shared_eval['speed_out'],
                    route_points=shared_eval['route_points'],
                    conditioning=shared_eval['conditioning'],
                    speed_samples=stage1_speed_samples,
                )
            if use_infer_semantic_transition and semantic_prev_state is not None:
                transition_stage1_scores_raw = (
                    self._compute_semantic_transition_scores_from_shared(
                        shared_eval,
                        prev_state=semantic_prev_state,
                        speed_samples=stage1_speed_samples,
                    )
                )
                if transition_only_state:
                    stage1_scores_raw = transition_stage1_scores_raw
                    semantic_fusion_debug = {
                        'enabled': torch.ones((), device=device, dtype=model_dtype),
                        'gate': torch.ones((B,), device=device, dtype=model_dtype),
                    }
                elif direct_prev_modulated_state:
                    stage1_scores_raw = transition_stage1_scores_raw
                    semantic_fusion_debug = {
                        'enabled': torch.zeros((), device=device, dtype=model_dtype),
                        'gate': semantic_prev_valid.detach() if semantic_prev_valid is not None else torch.zeros((B,), device=device, dtype=model_dtype),
                    }
                else:
                    stage1_scores_raw, semantic_fusion_debug = self._fuse_stage1_raw_scores(
                        stage1_scores_raw,
                        transition_stage1_scores_raw,
                        semantic_prev_valid,
                    )
            if stage1_scores_raw is None:
                raise RuntimeError("semantic inference produced no stage1 scores")
            stage1_scores = self._compose_stage1_outputs(stage1_scores_raw)
            if use_infer_semantic_transition and self.semantic_state_fusion_update_cache:
                self._semantic_state_cache = self._stage1_raw_scores_to_prev_state(
                    stage1_scores_raw,
                    route_steps=self.num_waypoints,
                    device=device,
                    model_dtype=model_dtype,
                    valid=torch.ones((B,), device=device, dtype=model_dtype),
                )
                self._semantic_state_cache_frame += 1
            traj_speed_1s_ref, traj_speed_05s_ref = self._compute_inference_traj_speed_refs(
                best_trajectory, model_dtype
            )
            head_speed_ref = target_speed_pred if target_speed_pred is not None else center_speed
            stage1_speed_ref_speeds = torch.stack(
                [head_speed_ref, traj_speed_1s_ref, traj_speed_05s_ref], dim=-1
            ).clamp_(0.0, 20.0)
            stage1_ref_scores_raw = None
            if not transition_only_state:
                stage1_ref_scores_raw = self.model.compute_shared_stage1_from_ego_outputs(
                    traj_out=shared_eval['traj_out'],
                    route_out=shared_eval['route_out'],
                    speed_out=shared_eval['speed_out'],
                    route_points=shared_eval['route_points'],
                    conditioning=shared_eval['conditioning'],
                    speed_samples=stage1_speed_ref_speeds,
                )
            if use_infer_semantic_transition and semantic_prev_state is not None:
                transition_stage1_ref_scores_raw = (
                    self._compute_semantic_transition_scores_from_shared(
                        shared_eval,
                        prev_state=semantic_prev_state,
                        speed_samples=stage1_speed_ref_speeds,
                    )
                )
                if transition_only_state:
                    stage1_ref_scores_raw = transition_stage1_ref_scores_raw
                elif direct_prev_modulated_state:
                    stage1_ref_scores_raw = transition_stage1_ref_scores_raw
                else:
                    stage1_ref_scores_raw, _ = self._fuse_stage1_raw_scores(
                        stage1_ref_scores_raw,
                        transition_stage1_ref_scores_raw,
                        semantic_prev_valid,
                    )
            if stage1_ref_scores_raw is None:
                raise RuntimeError("semantic inference produced no stage1 ref scores")
            stage1_ref_scores = self._compose_stage1_outputs(stage1_ref_scores_raw)

        if phase_go_smoothing_debug is None:
            phase_go_smoothing_debug = {
                'enabled': float(self.phase_go_smoothing_enable),
                'applied': 0.0,
                'raw_go_prob': float('nan'),
                'smoothed_go_prob': float('nan'),
                'history_len': float(len(self.phase_go_smoothing_history)),
                'threshold': float(self.phase_go_smoothing_threshold),
            }
        phase_go_smoothing_tensors = {
            key: torch.tensor([value], device=device, dtype=model_dtype)
            for key, value in phase_go_smoothing_debug.items()
        }
        if semantic_fusion_debug is None:
            semantic_fusion_enabled = torch.tensor(
                [float(use_infer_semantic_transition)], device=device, dtype=model_dtype
            )
            semantic_fusion_gate = torch.zeros((B,), device=device, dtype=model_dtype)
        else:
            semantic_fusion_enabled = semantic_fusion_debug['enabled'].reshape(1).to(
                device=device, dtype=model_dtype
            )
            semantic_fusion_gate = semantic_fusion_debug['gate'].to(
                device=device, dtype=model_dtype
            ).reshape(-1)
        mode_id_map = {
            'direct_only': 0.0,
            'direct_plus_transition': 1.0,
            'transition_only': 2.0,
            'direct_prev_modulated': 3.0,
        }
        semantic_mode_id = torch.full(
            (B,),
            mode_id_map.get(self.semantic_state_predictor_mode, -1.0),
            device=device,
            dtype=model_dtype,
        )
        semantic_prev_corruption_enabled = torch.full(
            (B,),
            float(
                max(
                    self.semantic_transition_prev_dropout_prob,
                    self.semantic_prev_token_dropout_prob,
                    self.semantic_prev_random_replace_prob,
                    self.semantic_prev_prefix_dropout_prob,
                )
                > 0.0
            ),
            device=device,
            dtype=model_dtype,
        )
        profile_id_map = {
            'all': 0.0,
            'window_only': 1.0,
            'window_decision': 2.0,
            'window_decision_control': 3.0,
            'window_phase_opportunity': 4.0,
            'compact_safe': 5.0,
        }
        semantic_motion_profile_id = torch.full(
            (B,),
            profile_id_map.get(self.semantic_motion_condition_profile, -1.0),
            device=device,
            dtype=model_dtype,
        )
        semantic_state_supervision_profile_id = torch.full(
            (B,),
            profile_id_map.get(self.semantic_state_supervision_profile, -1.0),
            device=device,
            dtype=model_dtype,
        )
        group_mask = getattr(self.model, 'semantic_motion_condition_group_mask', None)
        if group_mask is not None:
            semantic_motion_group_mask = group_mask.to(device=device, dtype=model_dtype).reshape(1, -1).expand(B, -1)
        else:
            semantic_motion_group_mask = None

        return {
            'best_trajectory': best_trajectory,       # (B, T, 2)
            'route_pred': route_pred,                 # (B, 20, 2)
            'all_trajectories': final_traj_abs,       # (B, 1, T, 2)
            'energy_scores': energy_scores,           # dict of (B, 1)
            'stage1_scores': stage1_scores,
            'stage1_speed_samples': stage1_speed_samples,
            'stage1_speed_query_center': stage1_speed_query_center,
            'stage1_speed_ref_speeds': stage1_speed_ref_speeds,
            'stage1_ref_scores': stage1_ref_scores,
            # Backward-compatible aliases for current close-loop debug code.
            'speed_energy_scores': stage1_scores,
            'speed_energy_samples': stage1_speed_samples,
            'speed_energy_query_center': stage1_speed_query_center,
            'speed_energy_ref_speeds': stage1_speed_ref_speeds,
            'speed_energy_ref_scores': stage1_ref_scores,
            'traj_branch_condition_probs': traj_branch_condition_probs,
            'traj_window_condition_probs': (
                traj_branch_condition_details['window_probs']
                if traj_branch_condition_details is not None else None
            ),
            'traj_dir_condition_probs': (
                traj_branch_condition_details['dir_probs']
                if traj_branch_condition_details is not None else None
            ),
            'traj_decision_phase_condition_probs': (
                traj_branch_condition_details['decision_phase_probs']
                if traj_branch_condition_details is not None else None
            ),
            'traj_control_phase_condition_probs': (
                traj_branch_condition_details['control_phase_probs']
                if traj_branch_condition_details is not None else None
            ),
            'traj_boundary_margin_condition': (
                traj_branch_condition_details['boundary_margins']
                if traj_branch_condition_details is not None else None
            ),
            'traj_go_opportunity_condition_probs': (
                traj_branch_condition_details['go_opportunity_probs']
                if traj_branch_condition_details is not None else None
            ),
            'traj_conflict_area_status_condition_probs': (
                traj_branch_condition_details['conflict_area_status_probs']
                if traj_branch_condition_details is not None else None
            ),
            'traj_conflict_timing_condition': (
                traj_branch_condition_details['conflict_timing_values']
                if traj_branch_condition_details is not None else None
            ),
            'traj_chase_has_lead_condition': (
                traj_branch_condition_details['chase_has_lead_prob']
                if traj_branch_condition_details is not None else None
            ),
            'traj_chase_speed_margin_condition': (
                traj_branch_condition_details['chase_speed_margin']
                if traj_branch_condition_details is not None else None
            ),
            'traj_current_edge_condition_probs': (
                traj_branch_condition_details['current_edge_mode_probs']
                if traj_branch_condition_details is not None else None
            ),
            'traj_future_edge_condition_probs': (
                traj_branch_condition_details['future_edge_mode_probs']
                if traj_branch_condition_details is not None else None
            ),
            'traj_edge_speed_margin_condition': (
                traj_branch_condition_details['edge_speed_margins']
                if traj_branch_condition_details is not None else None
            ),
            'traj_edge_speed_valid_condition': (
                traj_branch_condition_details['edge_speed_valids']
                if traj_branch_condition_details is not None else None
            ),
            'traj_borrow_time_condition': (
                traj_branch_condition_details['borrow_time_condition']
                if traj_branch_condition_details is not None else None
            ),
            'lane_dir_relation_probs': (
                traj_branch_condition_details['lane_dir_relation_probs']
                if traj_branch_condition_details is not None else None
            ),
            'traj_phase_go_smoothing_enabled': phase_go_smoothing_tensors['enabled'],
            'traj_phase_go_smoothing_applied': phase_go_smoothing_tensors['applied'],
            'traj_phase_go_smoothing_raw_go_prob': phase_go_smoothing_tensors['raw_go_prob'],
            'traj_phase_go_smoothing_smoothed_go_prob': phase_go_smoothing_tensors['smoothed_go_prob'],
            'traj_phase_go_smoothing_history_len': phase_go_smoothing_tensors['history_len'],
            'traj_phase_go_smoothing_threshold': phase_go_smoothing_tensors['threshold'],
            'semantic_state_fusion_enabled': semantic_fusion_enabled,
            'semantic_state_fusion_gate': semantic_fusion_gate,
            'semantic_state_predictor_mode_resolved': semantic_mode_id,
            'semantic_prev_corruption_enabled': semantic_prev_corruption_enabled,
            'semantic_motion_condition_profile': semantic_motion_profile_id,
            'semantic_motion_condition_profile_id': semantic_motion_profile_id,
            'semantic_state_supervision_profile': semantic_state_supervision_profile_id,
            'semantic_state_supervision_profile_id': semantic_state_supervision_profile_id,
            'semantic_motion_condition_group_mask': semantic_motion_group_mask,
            'pass1_trajectory': pass1_trajectory,
            'pass2_trajectory': best_trajectory,
            'poses_cls': poses_cls,                   # (B, 1)
            'best_idx': torch.zeros(B, dtype=torch.long, device=device),  # always 0
            'target_speed': target_speed_pred,        # (B,) m/s
            'target_speed_profile': speed_profile_pred if self.use_speed_profile_head else None,
        }

    # ========== Predict Action (standard interface) ==========
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        nobs = dict_apply(obs_dict, lambda x: x.to(device))

        transfuser_bev_feature = nobs['transfuser_bev_feature'].to(device=device, dtype=model_dtype)
        transfuser_bev_feature_upsample = nobs['transfuser_bev_feature_upsample'].to(device=device, dtype=model_dtype)
        transfuser_lidar_bev = self._get_transfuser_lidar_bev(nobs, device, model_dtype)
        ego_status = nobs['ego_status'].to(dtype=model_dtype)

        route_for_guidance = nobs.get('route', None)
        if route_for_guidance is not None:
            route_for_guidance = route_for_guidance.to(dtype=model_dtype)
        borrow_time_s = nobs.get('borrow_cross_active_time_s', None)
        if borrow_time_s is not None:
            borrow_time_s = borrow_time_s.to(device=device, dtype=model_dtype).reshape(-1)
        prev_relation_probs = nobs.get('prev_lane_dir_relation_probs', None)
        if prev_relation_probs is not None:
            prev_relation_probs = prev_relation_probs.to(device=device, dtype=model_dtype)
            if prev_relation_probs.dim() == 1:
                prev_relation_probs = prev_relation_probs.unsqueeze(0)
        reset_semantic_state_cache = bool(kwargs.get('reset_semantic_state_cache', False))
        semantic_reset = nobs.get('semantic_state_reset', None)
        if semantic_reset is not None:
            reset_semantic_state_cache = reset_semantic_state_cache or bool(
                (semantic_reset.to(device=device).reshape(-1) > 0.5).any().item()
            )
        if reset_semantic_state_cache:
            self.reset_semantic_state_cache()
        disable_semantic_state_cache = bool(kwargs.get('disable_semantic_state_cache', False))

        # Accept dynamic energy weights from kwargs (LLM Router interface)
        energy_weights = kwargs.get('energy_weights', None)

        sample_result = self.conditional_sample(
            transfuser_bev_feature=transfuser_bev_feature,
            transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
            ego_status=ego_status,
            device=device,
            model_dtype=model_dtype,
            route_for_guidance=route_for_guidance,
            energy_weights=energy_weights,
            transfuser_lidar_bev=transfuser_lidar_bev,
            borrow_time_s=borrow_time_s,
            prev_relation_probs=prev_relation_probs,
            disable_semantic_state_cache=disable_semantic_state_cache,
        )

        best_traj = sample_result['best_trajectory']
        action_pred = best_traj[..., :self.action_dim].detach().float().cpu().numpy()

        result = {
            'action': action_pred,
            'action_pred': action_pred,
            'route_pred': sample_result['route_pred'],
            'all_trajectories': sample_result['all_trajectories'].detach().float().cpu().numpy(),
            'best_idx': sample_result['best_idx'].detach().cpu().numpy(),
        }

        # Add predicted target speed (scalar m/s)
        if sample_result.get('target_speed') is not None:
            result['target_speed'] = sample_result['target_speed'].detach().float().cpu().numpy()
        if sample_result.get('target_speed_profile') is not None:
            result['target_speed_profile'] = sample_result['target_speed_profile'].detach().float().cpu().numpy()
        if sample_result.get('traj_branch_condition_probs') is not None:
            result['traj_branch_condition_probs'] = (
                sample_result['traj_branch_condition_probs'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_window_condition_probs') is not None:
            result['traj_window_condition_probs'] = (
                sample_result['traj_window_condition_probs'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_dir_condition_probs') is not None:
            result['traj_dir_condition_probs'] = (
                sample_result['traj_dir_condition_probs'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_decision_phase_condition_probs') is not None:
            result['traj_decision_phase_condition_probs'] = (
                sample_result['traj_decision_phase_condition_probs'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_control_phase_condition_probs') is not None:
            result['traj_control_phase_condition_probs'] = (
                sample_result['traj_control_phase_condition_probs'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_boundary_margin_condition') is not None:
            result['traj_boundary_margin_condition'] = (
                sample_result['traj_boundary_margin_condition'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_go_opportunity_condition_probs') is not None:
            result['traj_go_opportunity_condition_probs'] = (
                sample_result['traj_go_opportunity_condition_probs'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_conflict_area_status_condition_probs') is not None:
            result['traj_conflict_area_status_condition_probs'] = (
                sample_result['traj_conflict_area_status_condition_probs'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_conflict_timing_condition') is not None:
            result['traj_conflict_timing_condition'] = (
                sample_result['traj_conflict_timing_condition'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_chase_has_lead_condition') is not None:
            result['traj_chase_has_lead_condition'] = (
                sample_result['traj_chase_has_lead_condition'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_chase_speed_margin_condition') is not None:
            result['traj_chase_speed_margin_condition'] = (
                sample_result['traj_chase_speed_margin_condition'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_current_edge_condition_probs') is not None:
            result['traj_current_edge_condition_probs'] = (
                sample_result['traj_current_edge_condition_probs'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_future_edge_condition_probs') is not None:
            result['traj_future_edge_condition_probs'] = (
                sample_result['traj_future_edge_condition_probs'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_edge_speed_margin_condition') is not None:
            result['traj_edge_speed_margin_condition'] = (
                sample_result['traj_edge_speed_margin_condition'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_edge_speed_valid_condition') is not None:
            result['traj_edge_speed_valid_condition'] = (
                sample_result['traj_edge_speed_valid_condition'].detach().float().cpu().numpy()
            )
        if sample_result.get('traj_borrow_time_condition') is not None:
            result['traj_borrow_time_condition'] = (
                sample_result['traj_borrow_time_condition'].detach().float().cpu().numpy()
            )
        if sample_result.get('lane_dir_relation_probs') is not None:
            result['lane_dir_relation_probs'] = (
                sample_result['lane_dir_relation_probs'].detach().float().cpu().numpy()
            )
        for key in (
            'traj_phase_go_smoothing_enabled',
            'traj_phase_go_smoothing_applied',
            'traj_phase_go_smoothing_raw_go_prob',
            'traj_phase_go_smoothing_smoothed_go_prob',
            'traj_phase_go_smoothing_history_len',
            'traj_phase_go_smoothing_threshold',
            'semantic_state_fusion_enabled',
            'semantic_state_fusion_gate',
            'semantic_state_predictor_mode_resolved',
            'semantic_prev_corruption_enabled',
            'semantic_motion_condition_profile',
            'semantic_motion_condition_profile_id',
            'semantic_state_supervision_profile',
            'semantic_state_supervision_profile_id',
        ):
            if sample_result.get(key) is not None:
                result[key] = sample_result[key].detach().float().cpu().numpy()
        if sample_result.get('semantic_motion_condition_group_mask') is not None:
            result['semantic_motion_condition_group_mask'] = (
                sample_result['semantic_motion_condition_group_mask'].detach().float().cpu().numpy()
            )
        if sample_result.get('pass1_trajectory') is not None:
            result['pass1_trajectory'] = (
                sample_result['pass1_trajectory'].detach().float().cpu().numpy()
            )
        if sample_result.get('pass2_trajectory') is not None:
            result['pass2_trajectory'] = (
                sample_result['pass2_trajectory'].detach().float().cpu().numpy()
            )

        # Add energy scores if available
        if sample_result['energy_scores'] is not None:
            es = sample_result['energy_scores']
            for key in ('front', 'left', 'right', 'pedestrian', 'offroad', 'route'):
                if key in es:
                    result[f'energy_{key}'] = es[key].detach().float().cpu().numpy()

        if sample_result.get('stage1_scores') is not None:
            ses = sample_result['stage1_scores']
            if sample_result.get('stage1_speed_samples') is not None:
                value_np = sample_result['stage1_speed_samples'].detach().float().cpu().numpy()
                result['stage1_speed_samples'] = value_np
                result['speed_energy_samples'] = value_np
            if sample_result.get('stage1_speed_query_center') is not None:
                value_np = sample_result['stage1_speed_query_center'].detach().float().cpu().numpy()
                result['stage1_speed_query_center'] = value_np
                result['speed_energy_query_center'] = value_np
            for key in (
                'window_probs',
                'dir_probs',
                'decision_phase_probs',
                'decision_phase_base_probs',
                'control_phase_probs',
                'conflict_area_probs',
                'temporary_occupancy_probs',
                'go_opportunity_probs',
                'yld_pressure_probs',
                'conflict_area_status_probs',
                'conflict_timing_values',
                'conflict_dist_to_entry_m',
                'conflict_dist_to_exit_m',
                'conflict_time_to_entry_s',
                'chase_has_lead_prob',
                'chase_speed_max_mps',
                'current_cover_edge_valid_prob',
                'current_cover_edge_mode_probs',
                'future_cover_edge_valid_prob',
                'future_cover_edge_mode_probs',
                'current_cover_upper_speed_mps',
                'future_cover_lower_speed_mps',
                'front_follow_upper_speed_mps',
                'merge_flow_lower_speed_mps',
                'merge_yld_max_mps',
                'merge_go_min_mps',
                'junction_yld_max_mps',
                'junction_go_min_mps',
                'borrow_yld_max_mps',
                'borrow_go_min_mps',
                'selected_yld_max_mps',
                'selected_go_min_mps',
            ):
                if key in ses and ses.get(key) is not None:
                    value_np = ses[key].detach().float().cpu().numpy()
                    result[f'stage1_{key}'] = value_np
                    result[f'speed_energy_{key}'] = value_np
            for key in (
                'window_logits',
                'dir_logits',
                'decision_phase_logits',
                'decision_phase_logits_base',
                'control_phase_logits',
                'temporary_occupancy_logits',
                'go_opportunity_logits',
                'conflict_area_status_logits',
                'conflict_timing_values',
                'chase_has_lead_logit',
                'chase_speed_max',
                'current_cover_edge_valid_logit',
                'current_cover_edge_mode_logits',
                'future_cover_edge_valid_logit',
                'future_cover_edge_mode_logits',
                'current_cover_upper_speed',
                'future_cover_lower_speed',
                'front_follow_upper_speed',
                'merge_flow_lower_speed',
                'conflict_area_logits',
                'merge_yld_max',
                'merge_go_min',
                'junction_yld_max',
                'junction_go_min',
                'borrow_yld_max',
                'borrow_go_min',
            ):
                if key in ses and ses.get(key) is not None:
                    value_np = ses[key].detach().float().cpu().numpy()
                    result[f'stage1_{key}'] = value_np
                    result[f'speed_energy_{key}'] = value_np
        if sample_result.get('stage1_ref_scores') is not None:
            ref = sample_result['stage1_ref_scores']
            if sample_result.get('stage1_speed_ref_speeds') is not None:
                value_np = sample_result['stage1_speed_ref_speeds'].detach().float().cpu().numpy()
                result['stage1_speed_ref_speeds'] = value_np
                result['speed_energy_ref_speeds'] = value_np
            for key in (
                'window_probs',
                'dir_probs',
                'decision_phase_probs',
                'decision_phase_base_probs',
                'control_phase_probs',
                'conflict_area_probs',
                'temporary_occupancy_probs',
                'go_opportunity_probs',
                'yld_pressure_probs',
                'conflict_area_status_probs',
                'conflict_timing_values',
                'conflict_dist_to_entry_m',
                'conflict_dist_to_exit_m',
                'conflict_time_to_entry_s',
                'chase_has_lead_prob',
                'chase_speed_max_mps',
                'current_cover_edge_valid_prob',
                'current_cover_edge_mode_probs',
                'future_cover_edge_valid_prob',
                'future_cover_edge_mode_probs',
                'current_cover_upper_speed_mps',
                'future_cover_lower_speed_mps',
                'front_follow_upper_speed_mps',
                'merge_flow_lower_speed_mps',
                'merge_yld_max_mps',
                'merge_go_min_mps',
                'junction_yld_max_mps',
                'junction_go_min_mps',
                'borrow_yld_max_mps',
                'borrow_go_min_mps',
                'selected_yld_max_mps',
                'selected_go_min_mps',
            ):
                if key in ref and ref.get(key) is not None:
                    value_np = ref[key].detach().float().cpu().numpy()
                    result[f'stage1_ref_{key}'] = value_np
                    result[f'speed_energy_ref_{key}'] = value_np
            for key in (
                'window_logits',
                'dir_logits',
                'decision_phase_logits',
                'decision_phase_logits_base',
                'control_phase_logits',
                'temporary_occupancy_logits',
                'go_opportunity_logits',
                'conflict_area_status_logits',
                'conflict_timing_values',
                'chase_has_lead_logit',
                'chase_speed_max',
                'current_cover_edge_valid_logit',
                'current_cover_edge_mode_logits',
                'future_cover_edge_valid_logit',
                'future_cover_edge_mode_logits',
                'current_cover_upper_speed',
                'future_cover_lower_speed',
                'front_follow_upper_speed',
                'merge_flow_lower_speed',
                'conflict_area_logits',
                'merge_yld_max',
                'merge_go_min',
                'junction_yld_max',
                'junction_go_min',
                'borrow_yld_max',
                'borrow_go_min',
            ):
                if key in ref and ref.get(key) is not None:
                    value_np = ref[key].detach().float().cpu().numpy()
                    result[f'stage1_ref_{key}'] = value_np
                    result[f'speed_energy_ref_{key}'] = value_np

        return result
