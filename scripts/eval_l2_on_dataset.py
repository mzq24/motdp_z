#!/usr/bin/env python3
"""Quick eval: load checkpoint, run DDIM inference on train/val, report L2."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


def _record_l2(result_dict, pred, gt):
    """Compute and record metrics with the same definition as train-time validation."""
    gt_trim = gt[:, :pred.shape[1]]
    T_len = gt_trim.shape[1]
    l2_per_t = ((pred - gt_trim) ** 2).sum(dim=-1).sqrt()
    result_dict['ade'].append(l2_per_t.mean(dim=1).cpu().numpy())

    selected = []
    if T_len >= 2:
        l2_1s = l2_per_t[:, 1].cpu().numpy()
        result_dict['l2_1s'].append(l2_1s)
        selected.append(l2_1s)
    if T_len >= 4:
        l2_2s = l2_per_t[:, 3].cpu().numpy()
        result_dict['l2_2s'].append(l2_2s)
        selected.append(l2_2s)
    if T_len >= 6:
        l2_3s = l2_per_t[:, 5].cpu().numpy()
        result_dict['l2_3s'].append(l2_3s)
        selected.append(l2_3s)
    if selected:
        result_dict['l2_avg'].append(np.stack(selected, axis=0).mean(axis=0))


def _record_route_l2(result_dict, result, batch, device):
    """Record route L2 if available."""
    route_pred = result.get('route_pred', None)
    route_gt = batch.get('route', None)
    if route_pred is not None and route_gt is not None:
        if isinstance(route_pred, np.ndarray):
            route_pred = torch.from_numpy(route_pred).to(device)
        route_l2 = ((route_pred - route_gt) ** 2).sum(dim=-1).sqrt().mean(dim=-1)
        result_dict['route_l2'].append(route_l2.cpu().numpy())


def _predict_action_m34_constructed(policy, obs_dict, device):
    """M=34 constructed DDIM inference using unified forward (for old checkpoints).

    Pads x_t with anchor contexts to form M=34 input, then extracts
    the last slot (x_t at pos=33) as the prediction.
    """
    model_dtype = next(policy.parameters()).dtype
    B = obs_dict['transfuser_bev_feature'].shape[0]
    T = policy.horizon
    M_anchor = policy.num_energy_modes  # 32

    transfuser_bev_feature = obs_dict['transfuser_bev_feature'].to(device=device, dtype=model_dtype)
    transfuser_bev_feature_upsample = obs_dict['transfuser_bev_feature_upsample'].to(device=device, dtype=model_dtype)
    ego_status = obs_dict['ego_status'].to(device=device, dtype=model_dtype)

    anchor_abs = policy.anchor_centers_abs.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 32, T, 2)
    anchor_normed = policy.abs_to_norm(anchor_abs)

    # Dummy GT slot (zeros, won't affect x_t prediction due to block diagonal mask)
    gt_normed = torch.zeros(B, 1, T, 2, device=device, dtype=model_dtype)

    # x_t: start from noise
    x_t = torch.randn(B, 1, T, 2, device=device, dtype=torch.float32)

    # DDIM schedule
    num_steps = policy.num_inference_steps
    roll_timesteps_t = policy.build_roll_timesteps(num_steps=num_steps, device=device)
    alphas_cumprod = policy.diffusion_scheduler.alphas_cumprod.to(device)

    route_pred = None
    for step_i, k in enumerate(roll_timesteps_t):
        t_cur = k.item()
        t_next = roll_timesteps_t[step_i + 1].item() if step_i + 1 < len(roll_timesteps_t) else 0

        x_input = x_t.to(dtype=model_dtype)

        # Construct M=34: [x_t(1), anchors(32), gt_dummy(1)] — matches unified training layout
        full_input = torch.cat([x_input, anchor_normed, gt_normed], dim=1)  # (B, 34, T, 2)
        full_abs = policy.norm_to_abs(full_input)

        t_tensor = torch.full((B,), t_cur, dtype=torch.long, device=device)

        with torch.no_grad():
            result_tuple = policy.model(
                x_t=full_input,
                timestep=t_tensor,
                transfuser_bev_feature=transfuser_bev_feature,
                transfuser_bev_feature_upsample=transfuser_bev_feature_upsample,
                ego_status=ego_status,
                x_t_abs=full_abs,
            )
            # forward() returns (poses_reg, poses_cls, route_pred, mode_out[, energy_scores])
            poses_reg = result_tuple[0]   # (B, M, T, 2)
            route_pred = result_tuple[2]  # (B, num_waypoints, 2)

        # x_t is at pos 0 in unified layout: [x_t(0), anchors(1-32), GT(33)]
        pred_x0 = poses_reg[:, 0:1]  # (B, 1, T, 2)

        # DDIM step
        alpha_t = alphas_cumprod[t_cur]
        alpha_next = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
        pred_eps = (x_t - alpha_t.sqrt() * pred_x0.float()) / (1 - alpha_t).sqrt().clamp(min=1e-8)
        x_t = alpha_next.sqrt() * pred_x0.float() + (1 - alpha_next).sqrt() * pred_eps

    final_abs = policy.norm_to_abs(pred_x0)
    best_trajectory = final_abs.squeeze(1)
    action_pred = best_trajectory[..., :policy.action_dim].detach().float().cpu().numpy()

    return {
        'action': action_pred,
        'route_pred': route_pred,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/pdm_local_route_b.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to .pt checkpoint (default: best)')
    parser.add_argument('--max_batches', type=int, default=10)
    parser.add_argument('--split', type=str, default='both', choices=['train', 'val', 'both'])
    parser.add_argument('--num_inference_steps', type=int, default=None,
                        help='Override policy.num_inference_steps for eval (e.g. 1 for corrected 1-step DDIM)')
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda:0')

    # Build dataset
    from dataset.unified_carla_dataset import CARLAImageDataset
    training_cfg = config.get('training', {})
    dataset_path = training_cfg.get('dataset_path')
    image_data_root = training_cfg.get('image_data_root', dataset_path)

    splits = []
    if args.split in ('train', 'both'):
        splits.append('train')
    if args.split in ('val', 'both'):
        splits.append('val')

    # Load policy via load_checkpoint (handles stats, anchors, buffer restoration)
    from policy.annealed_energy_guidance_policy import AnnealedEnergyGuidancePolicy
    ckpt_path = args.checkpoint or os.path.join(
        training_cfg.get('checkpoint_dir', config.get('logging', {}).get('checkpoint_dir', '.')),
        'dit_policy_best.pt'
    )
    print(f"Loading checkpoint: {ckpt_path}")
    policy, ckpt = AnnealedEnergyGuidancePolicy.load_checkpoint(ckpt_path, config, device)
    print(f"Loaded: epoch={ckpt.get('epoch', 'N/A')}, "
          f"train_loss={ckpt.get('train_loss', 'N/A')}, "
          f"val_loss={ckpt.get('val_loss', 'N/A')}")
    if args.num_inference_steps is not None:
        policy.num_inference_steps = int(args.num_inference_steps)
        print(f"Overriding num_inference_steps -> {policy.num_inference_steps}")

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Evaluating on {split} set (max {args.max_batches} batches)")
        print(f"{'='*60}")

        split_path = os.path.join(dataset_path, split)
        anchor_centers_abs = None
        anchor_path = config.get('anchor_path', None)
        if anchor_path and anchor_path.endswith('.npy'):
            anchor_centers_abs = np.load(anchor_path)

        ds = CARLAImageDataset(
            dataset_path=split_path,
            image_data_root=image_data_root,
            mode=split,
            anchor_centers_abs=anchor_centers_abs,
            skip_memmap=True,
            use_per_frame=config.get('dataset', {}).get('use_per_frame', False),
        )
        loader = DataLoader(ds, batch_size=config['dataloader']['batch_size'],
                           shuffle=False, num_workers=4, pin_memory=True)

        all_reg_loss = []
        all_reg_loss_unified = []
        modes = ['M=1', 'M=34_constructed']
        all_ddim_results = {
            m: {'ade': [], 'l2_1s': [], 'l2_2s': [], 'l2_3s': [], 'l2_avg': [], 'route_l2': []}
            for m in modes
        }

        has_anchors = getattr(policy, 'anchor_centers_abs', None) is not None

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"{split}", total=min(args.max_batches, len(loader)))):
            if batch_idx >= args.max_batches:
                break
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            target_actions = batch['agent_pos']  # (B, T, 2)

            # 1a) Single-step reg_loss via compute_diffusion_loss (M=1)
            with torch.no_grad():
                loss_dict = policy.compute_loss(batch)
                all_reg_loss.append(loss_dict['reg_loss'].item())

            # 1b) Single-step reg_loss via compute_unified_loss (M=34)
            if has_anchors:
                with torch.no_grad():
                    loss_dict_unified = policy.compute_unified_loss(batch)
                    all_reg_loss_unified.append(loss_dict_unified['reg_loss'].item())

            obs_dict = {
                'transfuser_bev_feature': batch['transfuser_bev_feature'],
                'transfuser_bev_feature_upsample': batch['transfuser_bev_feature_upsample'],
                'ego_status': batch['ego_status'][:, :policy.n_obs_steps],
            }
            gt = target_actions

            # 2a) DDIM M=1 (split forward, default)
            try:
                result = policy.predict_action(obs_dict)
                pred = torch.from_numpy(result['action']).to(device)
                _record_l2(all_ddim_results['M=1'], pred, gt)
                _record_route_l2(all_ddim_results['M=1'], result, batch, device)
            except Exception as e:
                print(f"  Inference error (M=1): {e}")
                import traceback; traceback.print_exc()

            # 2b) DDIM M=34 constructed (unified forward, for old checkpoints)
            if has_anchors:
                try:
                    result34 = _predict_action_m34_constructed(policy, obs_dict, device)
                    pred34 = torch.from_numpy(result34['action']).to(device)
                    _record_l2(all_ddim_results['M=34_constructed'], pred34, gt)
                    _record_route_l2(all_ddim_results['M=34_constructed'], result34, batch, device)
                except Exception as e:
                    print(f"  Inference error (M=34): {e}")
                    import traceback; traceback.print_exc()

        # Report
        reg = np.mean(all_reg_loss)
        print(f"\n  reg_loss (M=1):  {reg:.4f}")
        if all_reg_loss_unified:
            reg_unified = np.mean(all_reg_loss_unified)
            print(f"  reg_loss (M=34): {reg_unified:.4f}")

        for mode_name in modes:
            r = all_ddim_results[mode_name]
            if r['ade']:
                ade = np.mean(np.concatenate(r['ade']))
                print(f"  ADE (DDIM {mode_name}):    {ade:.4f}")
            if r['l2_avg']:
                l2_1s = np.mean(np.concatenate(r['l2_1s'])) if r['l2_1s'] else float('nan')
                l2_2s = np.mean(np.concatenate(r['l2_2s'])) if r['l2_2s'] else float('nan')
                l2_3s = np.mean(np.concatenate(r['l2_3s'])) if r['l2_3s'] else float('nan')
                l2_avg = np.mean(np.concatenate(r['l2_avg']))
                print(f"  L2_1s (DDIM {mode_name}):  {l2_1s:.4f}")
                print(f"  L2_2s (DDIM {mode_name}):  {l2_2s:.4f}")
                print(f"  L2_3s (DDIM {mode_name}):  {l2_3s:.4f}")
                print(f"  L2_avg (DDIM {mode_name}): {l2_avg:.4f}")
                if r['route_l2']:
                    route_l2 = np.mean(np.concatenate(r['route_l2']))
                    print(f"  Route_L2 (DDIM {mode_name}): {route_l2:.4f}")

if __name__ == '__main__':
    main()
