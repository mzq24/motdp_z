"""
BDBaselinePolicy: wraps BDBaselineModel with training/inference interface.

Compatible with MoT-DP's training pipeline (compute_loss / predict_action).
Key difference from DDBaselinePolicy: GT trajectory is `route[:num_poses]`
(equidistant geometric route) instead of `agent_pos` (time-spaced waypoints).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

from .config import BridgeBaselineConfig
from .model import BDBaselineModel
from .model_v2 import BDModelV2, _encode_two_hot


class BDBaselinePolicy(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config

        # Build BridgeBaselineConfig from yaml config
        bd_cfg_dict = config.get('bridge_baseline', {})
        self.bd_config = BridgeBaselineConfig(**bd_cfg_dict)
        self.model = BDBaselineModel(self.bd_config)

        self.n_obs_steps = config.get('n_obs_steps', config.get('obs_horizon', 4))
        self.n_action_steps = config.get('action_horizon', 6)

    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_loss(self, batch):
        """Compute training loss.

        Args:
            batch: dict with keys:
                - transfuser_bev_feature: (B, 1512, 8, 8)
                - transfuser_bev_feature_upsample: (B, 64, 64, 64)
                - ego_status: (B, T, 14)
                - route: (B, 20, 2)  ← equidistant geometric route
        Returns:
            dict with 'total_loss' and per-layer losses
        """
        features = {
            'transfuser_bev_feature': batch['transfuser_bev_feature'],
            'transfuser_bev_feature_upsample': batch['transfuser_bev_feature_upsample'],
            'ego_status': batch['ego_status'][:, :self.n_obs_steps],
        }
        targets = {
            'route': batch['route'],  # (B, 20, 2); head slices [:num_poses]
        }

        output = self.model(features, targets=targets)

        loss_dict = {
            'total_loss': output['trajectory_loss'],
        }
        if 'trajectory_loss_dict' in output:
            for k, v in output['trajectory_loss_dict'].items():
                loss_dict[k] = v

        loss_dict['cls_loss'] = loss_dict.get('trajectory_loss_0', output['trajectory_loss'])
        loss_dict['reg_loss'] = loss_dict.get('trajectory_loss_1', torch.tensor(0.0))

        return loss_dict

    @torch.no_grad()
    def predict_action(self, obs_dict, targets=None, **kwargs):
        """Inference: predict trajectory from observations.

        Args:
            obs_dict: dict with BEV features and ego_status
            targets: optional dict (ignored during inference)
        Returns:
            dict with 'action': (B, T, 2) numpy array (route coordinate space)
        """
        self.eval()
        features = {
            'transfuser_bev_feature': obs_dict['transfuser_bev_feature'].float(),
            'transfuser_bev_feature_upsample': obs_dict['transfuser_bev_feature_upsample'].float(),
            'ego_status': obs_dict['ego_status'].float(),
        }
        output = self.model(features, targets=None)
        trajectory = output['trajectory']  # (B, T, 2)
        return {'action': trajectory.cpu().numpy()}


class BDBaselinePolicyV2(nn.Module):
    """Bridge Baseline v2: PlanningContextEncoder + BDModelV2.

    Uses transfuser_bev_feature_upsample (64, 64, 64) directly.
    No ego_status needed — velocity/command/target_point tokens instead.
    Compatible with CARLAImageDataset batch keys.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        bd_cfg_dict = config.get('bridge_baseline', {})
        self.bd_config = BridgeBaselineConfig(**bd_cfg_dict)
        self.model = BDModelV2(self.bd_config)

    def forward(self, batch):
        return self.compute_loss(batch)

    def _extract_features(self, batch):
        return {
            'transfuser_bev_feature_upsample': batch['transfuser_bev_feature_upsample'],
            'speed':                   batch['speed'],
            'command_hist':            batch['command_hist'],
            'target_point_hist':       batch['target_point_hist'],
            'target_point_next_hist':  batch['target_point_next_hist'],
        }

    def compute_loss(self, batch):
        features = self._extract_features(batch)
        if self.bd_config.predict_traj:
            targets = {'route': batch['agent_pos']}   # (B, ≥6, 2) time-spaced
        else:
            targets = {'route': batch['route']}       # (B, 20, 2) equidistant route
        output = self.model(features, targets=targets)

        loss_dict = {'total_loss': output['trajectory_loss']}
        if 'trajectory_loss_dict' in output:
            for k, v in output['trajectory_loss_dict'].items():
                loss_dict[k] = v
        loss_dict['cls_loss'] = loss_dict.get('trajectory_loss_0', output['trajectory_loss'])
        loss_dict['reg_loss'] = loss_dict.get('trajectory_loss_1', torch.tensor(0.0))

        # Method 1: speed CE loss
        if self.bd_config.predict_target_speed and 'target_speed_dist' in output:
            speed_gt = features['speed']
            if speed_gt.dim() == 2:
                speed_gt = speed_gt[:, -1]
            brake = speed_gt < 0.5
            speed_dist_gt = _encode_two_hot(
                speed_gt, self.model.speed_classes, brake
            )
            speed_loss = F.cross_entropy(output['target_speed_dist'].float(), speed_dist_gt)
            loss_dict['speed_loss'] = speed_loss
            loss_dict['total_loss'] = loss_dict['total_loss'] + self.bd_config.speed_loss_weight * speed_loss

        return loss_dict

    @torch.no_grad()
    def predict_action(self, obs_dict, targets=None, **kwargs):
        self.eval()
        features = self._extract_features(obs_dict)
        # Convert to float
        for k in features:
            if isinstance(features[k], torch.Tensor):
                features[k] = features[k].float()
        output = self.model(features, targets=None)
        result = {'action': output['trajectory'].cpu().numpy()}
        if 'target_speed_scalar' in output and output['target_speed_scalar'] is not None:
            result['target_speed'] = output['target_speed_scalar'].cpu().numpy()
        return result
