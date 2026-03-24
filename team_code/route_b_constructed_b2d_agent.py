import os

import numpy as np
import torch

from route_b_b2d_agent import MOTAgent as BaseRouteBAgent


def get_entry_point():
	return 'RouteBConstructedAgent'


class RouteBConstructedAgent(BaseRouteBAgent):
	def resolve_checkpoint_path(self):
		training_cfg = self.config.get('training', {})
		logging_cfg = self.config.get('logging', {})
		checkpoint_path = training_cfg.get('checkpoint_path') or logging_cfg.get('checkpoint_path')
		if checkpoint_path:
			return checkpoint_path

		return "/media/z/data/mzq/others/MoT-DP/checkpoints/hpc/dit_policy_best.pt"

	def _predict_dp_action(self, dp_obs_dict):
		policy = self.net
		device = dp_obs_dict['transfuser_bev_feature'].device
		model_dtype = next(policy.parameters()).dtype
		batch_size = dp_obs_dict['transfuser_bev_feature'].shape[0]
		horizon = policy.horizon

		transfuser_bev_feature = dp_obs_dict['transfuser_bev_feature'].to(device=device, dtype=model_dtype)
		transfuser_bev_feature_upsample = dp_obs_dict['transfuser_bev_feature_upsample'].to(device=device, dtype=model_dtype)
		ego_status = dp_obs_dict['ego_status'].to(device=device, dtype=model_dtype)

		if getattr(policy, 'anchor_centers_abs', None) is None:
			raise RuntimeError("Constructed M=34 inference requires anchor_centers_abs in the loaded checkpoint.")

		anchor_abs = policy.anchor_centers_abs.unsqueeze(0).expand(batch_size, -1, -1, -1)
		anchor_normed = policy.abs_to_norm(anchor_abs)
		gt_dummy = torch.zeros(batch_size, 1, horizon, 2, device=device, dtype=model_dtype)
		x_t = torch.randn(batch_size, 1, horizon, 2, device=device, dtype=torch.float32)

		num_steps = policy.num_inference_steps
		step_ratio = policy.train_max_timesteps / num_steps
		roll_timesteps = (np.arange(0, num_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
		roll_timesteps_t = torch.from_numpy(roll_timesteps).to(device)
		alphas_cumprod = policy.diffusion_scheduler.alphas_cumprod.to(device)

		route_pred = None
		pred_x0 = None
		last_energy_scores = None

		for step_i, k in enumerate(roll_timesteps_t):
			t_cur = k.item()
			t_next = roll_timesteps_t[step_i + 1].item() if step_i + 1 < len(roll_timesteps_t) else 0

			full_input = torch.cat([x_t.to(dtype=model_dtype), anchor_normed, gt_dummy], dim=1)
			full_abs = policy.norm_to_abs(full_input)
			t_tensor = torch.full((batch_size,), t_cur, dtype=torch.long, device=device)

			result_tuple = policy.model(
				x_t=full_input,
				timestep=t_tensor,
				transfuser_bev_feature=transfuser_bev_feature,
				transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
				ego_status=ego_status,
				x_t_abs=full_abs,
			)
			poses_reg = result_tuple[0]
			route_pred = result_tuple[2]
			if len(result_tuple) > 4:
				last_energy_scores = result_tuple[4]
			pred_x0 = poses_reg[:, 0:1]

			alpha_t = alphas_cumprod[t_cur]
			alpha_next = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
			pred_eps = (x_t - alpha_t.sqrt() * pred_x0.float()) / (1 - alpha_t).sqrt().clamp(min=1e-8)
			x_t = alpha_next.sqrt() * pred_x0.float() + (1 - alpha_next).sqrt() * pred_eps

		final_abs = policy.norm_to_abs(pred_x0)
		best_trajectory = final_abs.squeeze(1)
		action_pred = best_trajectory[..., :policy.action_dim].detach().float().cpu().numpy()

		result = {
			'action': action_pred,
			'route_pred': route_pred,
		}
		if last_energy_scores is not None:
			result['energy_collision'] = last_energy_scores['collision'][:, 0].detach().float().cpu().numpy()
			result['energy_offroad'] = last_energy_scores['offroad'][:, 0].detach().float().cpu().numpy()
			result['energy_target'] = last_energy_scores['target'][:, 0].detach().float().cpu().numpy()
		return result
