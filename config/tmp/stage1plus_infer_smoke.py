#!/usr/bin/env python3
import os
import sys

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_SCRIPT_DIR) == 'tmp' and os.path.basename(os.path.dirname(_SCRIPT_DIR)) == 'config':
    PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
else:
    PROJECT_ROOT = _SCRIPT_DIR
sys.path.append(PROJECT_ROOT)

from config.tmp.stage1_speed_smoke_test import _build_dataset, _build_policy
from training.train_carla_bev import load_config


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        PROJECT_ROOT, 'config', 'tmp', 'route_b_hierarchical.yaml'
    )
    config = load_config(config_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[infer-smoke] device={device}")
    print(f"[infer-smoke] config={config_path}")

    dataset = _build_dataset(config)
    sample = dataset[0]
    batch = default_collate([sample])
    print(f"[infer-smoke] sample keys={sorted(batch.keys())}")

    policy = _build_policy(config, device)
    policy.eval()

    obs_dict = {
        'ego_status': batch['ego_status'].to(device),
        'transfuser_bev_feature': batch['transfuser_bev_feature'].to(device),
        'transfuser_bev_feature_upsample': batch['transfuser_bev_feature_upsample'].to(device),
        'transfuser_lidar_bev': batch['transfuser_lidar_bev'].to(device),
    }
    if 'route' in batch:
        obs_dict['route'] = batch['route'].to(device)
    if 'borrow_cross_active_time_s' in batch:
        obs_dict['borrow_cross_active_time_s'] = batch['borrow_cross_active_time_s'].to(device)
    obs_dict['prev_lane_dir_relation_probs'] = torch.tensor(
        [[0.5, 0.5]], device=device, dtype=torch.float32
    )

    with torch.no_grad():
        pred = policy.predict_action(obs_dict)

    print("[infer-smoke] result keys:", sorted(pred.keys()))
    print("[infer-smoke] action shape:", np.asarray(pred['action']).shape)
    if 'traj_window_condition_probs' in pred:
        print("[infer-smoke] traj_window_condition_probs:", np.asarray(pred['traj_window_condition_probs']).reshape(-1).tolist())
    if 'traj_phase_condition_probs' in pred:
        print("[infer-smoke] traj_phase_condition_probs:", np.asarray(pred['traj_phase_condition_probs']).reshape(-1).tolist())
    if 'traj_phase_energy_summary' in pred:
        print("[infer-smoke] traj_phase_energy_summary:", np.asarray(pred['traj_phase_energy_summary']).reshape(-1).tolist())
    if 'lane_dir_relation_probs' in pred:
        print("[infer-smoke] lane_dir_relation_probs:", np.asarray(pred['lane_dir_relation_probs']).reshape(-1).tolist())
    if 'traj_borrow_time_condition' in pred:
        print("[infer-smoke] traj_borrow_time_condition:", np.asarray(pred['traj_borrow_time_condition']).reshape(-1).tolist())


if __name__ == '__main__':
    main()
