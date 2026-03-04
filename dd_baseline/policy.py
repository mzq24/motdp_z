"""
DDBaselinePolicy: wraps DDBaselineModel with training/inference interface.

Compatible with MoT-DP's training pipeline (compute_loss / predict_action).
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict

from .config import DDBaselineConfig
from .model import DDBaselineModel


class DDBaselinePolicy(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config

        # Build DDBaselineConfig from yaml config
        dd_cfg_dict = config.get('dd_baseline', {})
        # Override normalization from truncated_diffusion section if present
        trunc_cfg = config.get('truncated_diffusion', {})
        if 'norm_x_offset' in trunc_cfg:
            dd_cfg_dict.setdefault('norm_x_offset', trunc_cfg['norm_x_offset'])
            dd_cfg_dict.setdefault('norm_x_range', trunc_cfg['norm_x_range'])
            dd_cfg_dict.setdefault('norm_y_offset', trunc_cfg['norm_y_offset'])
            dd_cfg_dict.setdefault('norm_y_range', trunc_cfg['norm_y_range'])
        if 'trunc_timesteps' in trunc_cfg:
            dd_cfg_dict.setdefault('trunc_timesteps', trunc_cfg['trunc_timesteps'])
        if 'num_diffusion_steps' in trunc_cfg:
            dd_cfg_dict.setdefault('num_diffusion_steps', trunc_cfg['num_diffusion_steps'])

        self.dd_config = DDBaselineConfig(**dd_cfg_dict)
        self.model = DDBaselineModel(self.dd_config)

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
                - agent_pos: (B, T_pred, 2)
        Returns:
            dict with 'total_loss', 'cls_loss', 'reg_loss'
        """
        features = {
            'transfuser_bev_feature': batch['transfuser_bev_feature'],
            'transfuser_bev_feature_upsample': batch['transfuser_bev_feature_upsample'],
            'ego_status': batch['ego_status'][:, :self.n_obs_steps],
        }
        targets = {
            'trajectory': batch['agent_pos'][:, :self.dd_config.num_poses, :],
        }

        output = self.model(features, targets=targets)

        loss_dict = {
            'total_loss': output['trajectory_loss'],
        }
        # Add per-layer losses for logging
        if 'trajectory_loss_dict' in output:
            for k, v in output['trajectory_loss_dict'].items():
                loss_dict[k] = v

        # Extract cls and reg from first layer loss for compatibility
        # (The total loss already includes both)
        loss_dict['cls_loss'] = loss_dict.get('trajectory_loss_0', output['trajectory_loss'])
        loss_dict['reg_loss'] = loss_dict.get('trajectory_loss_1', torch.tensor(0.0))

        return loss_dict

    @torch.no_grad()
    def predict_action(self, obs_dict, **kwargs):
        """Inference: predict trajectory from observations.

        Args:
            obs_dict: dict with BEV features and ego_status
        Returns:
            dict with 'action': (B, T, 2) numpy array
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
