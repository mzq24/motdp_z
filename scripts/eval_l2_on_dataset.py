#!/usr/bin/env python3
"""Quick eval: load checkpoint, run DDIM inference on train/val, report L2."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/pdm_local_route_b.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to .pt checkpoint (default: best)')
    parser.add_argument('--max_batches', type=int, default=10)
    parser.add_argument('--split', type=str, default='both', choices=['train', 'val', 'both'])
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

        all_l2_1s, all_l2_2s, all_l2_3s = [], [], []
        all_reg_loss = []
        all_reg_loss_unified = []

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
            if getattr(policy, 'anchor_centers_abs', None) is not None:
                with torch.no_grad():
                    loss_dict_unified = policy.compute_unified_loss(batch)
                    all_reg_loss_unified.append(loss_dict_unified['reg_loss'].item())

            # 2) Full DDIM inference
            obs_dict = {
                'transfuser_bev_feature': batch['transfuser_bev_feature'],
                'transfuser_bev_feature_upsample': batch['transfuser_bev_feature_upsample'],
                'ego_status': batch['ego_status'][:, :policy.n_obs_steps],
            }
            try:
                result = policy.predict_action(obs_dict, no_noise=False, use_server_style=False, gt_trajectory=target_actions)
                pred = torch.from_numpy(result['action']).to(device)  # (B, T, 2)

                gt = target_actions[:, :pred.shape[1]]  # (B, T, 2)
                T = gt.shape[1]

                l2_per_t = ((pred - gt) ** 2).sum(dim=-1).sqrt()  # (B, T)
                steps_1s = min(2, T)
                steps_2s = min(4, T)
                steps_3s = min(6, T)

                all_l2_1s.append(l2_per_t[:, :steps_1s].mean(dim=1).cpu().numpy())
                all_l2_2s.append(l2_per_t[:, :steps_2s].mean(dim=1).cpu().numpy())
                all_l2_3s.append(l2_per_t[:, :steps_3s].mean(dim=1).cpu().numpy())
            except Exception as e:
                print(f"  Inference error: {e}")
                continue

        # Report
        reg = np.mean(all_reg_loss)
        print(f"\n  reg_loss (M=1):  {reg:.4f}")
        if all_reg_loss_unified:
            reg_unified = np.mean(all_reg_loss_unified)
            print(f"  reg_loss (M=34): {reg_unified:.4f}")

        if all_l2_1s:
            l2_1s = np.mean(np.concatenate(all_l2_1s))
            l2_2s = np.mean(np.concatenate(all_l2_2s))
            l2_3s = np.mean(np.concatenate(all_l2_3s))
            l2_avg = (l2_1s + l2_2s + l2_3s) / 3
            print(f"  L2_1s (DDIM):  {l2_1s:.4f}")
            print(f"  L2_2s (DDIM):  {l2_2s:.4f}")
            print(f"  L2_3s (DDIM):  {l2_3s:.4f}")
            print(f"  L2_avg (DDIM): {l2_avg:.4f}")

if __name__ == '__main__':
    main()
