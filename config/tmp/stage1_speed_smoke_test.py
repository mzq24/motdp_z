#!/usr/bin/env python3
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_SCRIPT_DIR) == 'tmp' and os.path.basename(os.path.dirname(_SCRIPT_DIR)) == 'config':
    PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
else:
    PROJECT_ROOT = _SCRIPT_DIR
sys.path.append(PROJECT_ROOT)

from dataset.unified_carla_dataset import CARLAImageDataset
from policy.annealed_energy_guidance_policy import AnnealedEnergyGuidancePolicy
from training.train_carla_bev import load_config


def _resolve_repo_path(path_str: str) -> str:
    if path_str is None:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(PROJECT_ROOT, path_str)


def _build_dataset(config):
    dataset_root = config['training']['dataset_path']
    image_data_root = config['training']['image_data_root']
    val_dataset_path = os.path.join(dataset_root, 'val')

    route_b_cfg = config.get('route_b', {})
    policy_cfg = config.get('policy', {})
    use_lidar_bev_detail = route_b_cfg.get('use_lidar_bev_detail', False)
    lidar_history_frames = max(
        int(route_b_cfg.get('lidar_history_frames', policy_cfg.get('ego_status_seq_len', config.get('obs_horizon', 1)))),
        1,
    )

    dataset_cfg = config.get('dataset', {})
    ds = CARLAImageDataset(
        dataset_path=val_dataset_path,
        image_data_root=image_data_root,
        mode='val',
        skip_memmap=False,
        use_per_frame=dataset_cfg.get('use_per_frame', False),
        use_vqa_anchor=config.get('use_vqa_anchor', False),
        anchor_centers_abs=None,
        semantic_behavior_cfg={},
        cache_dir=dataset_cfg.get('cache_dir', None),
        feature_suffix=dataset_cfg.get('feature_suffix', ''),
        load_transfuser_lidar_bev=use_lidar_bev_detail,
        lidar_history_frames=lidar_history_frames,
    )
    return ds


def _build_policy(config, device):
    policy = AnnealedEnergyGuidancePolicy(config).to(device)
    policy.register_norm_stats_from_config(config)

    route_abs_stats_path = config.get('route_abs_stats_path', None)
    if not route_abs_stats_path:
        raise ValueError('route_abs_stats_path is required for smoke test')
    route_abs_stats_path = _resolve_repo_path(route_abs_stats_path)
    route_data = np.load(route_abs_stats_path)
    policy.register_route_abs_stats(route_data['route_abs_mean'], route_data['route_abs_std'])

    anchor_path = config.get('anchor_path', None)
    if anchor_path:
        anchor_path = _resolve_repo_path(anchor_path)
        if os.path.exists(anchor_path):
            if anchor_path.endswith('.npy'):
                anchor_centers = np.load(anchor_path)
            else:
                with open(anchor_path, 'rb') as f:
                    anchor_centers = pickle.load(f)['centers']
            policy.register_anchor_centers(anchor_centers)
        else:
            print(f"[smoke] anchor_path missing, skip anchor registration: {anchor_path}")

    policy.eval()
    return policy


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, 'config', 'pdm_hpc_route_b_lidar_bev_trainonly.yaml')
    config = load_config(config_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[smoke] device={device}")
    print(f"[smoke] config={config_path}")

    dataset = _build_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=default_collate,
    )
    batch = next(iter(loader))
    print(f"[smoke] batch keys={sorted(batch.keys())}")
    print(f"[smoke] lidar shape={tuple(batch['transfuser_lidar_bev'].shape)}")
    print(f"[smoke] route shape={tuple(batch['route'].shape)}")
    if 'speed_sample_values' in batch:
        print(f"[smoke] stage1 speed shape={tuple(batch['speed_sample_values'].shape)}")
    else:
        print("[smoke] stage1 speed shape=<not present>")

    policy = _build_policy(config, device)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)

    with torch.no_grad():
        loss_dict = policy(batch, return_loss_dict=True, phase='split')

    print("[smoke] loss keys:", sorted(loss_dict.keys()))
    for key in sorted(loss_dict.keys()):
        value = loss_dict[key]
        if isinstance(value, torch.Tensor):
            print(f"[smoke] {key}={float(value.detach().cpu())}")
        else:
            print(f"[smoke] {key}={value}")


if __name__ == '__main__':
    main()
