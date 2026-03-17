#!/usr/bin/env python3
"""
Compute per-waypoint mean and std of route[:10] and/or agent_pos[:6] from training dataset.

These statistics are used for per-waypoint z-score normalization in the
Bridge Baseline's DDBM trajectory head.

Usage:
    # Both route and traj stats in one pass:
    python bridge_baseline/scripts/compute_route_stats.py \
        --dataset_path /media/z/data/dataset/pdm_lite_mini/train \
        --key all --output_yaml bridge_baseline/bd_config.yaml

    # Route only:
    python bridge_baseline/scripts/compute_route_stats.py \
        --dataset_path /media/z/data/dataset/pdm_lite_mini/train \
        --key route --num_poses 10

The script updates norm_x_mean / norm_x_std / norm_y_mean / norm_y_std
(and traj_norm_* for agent_pos) in the bridge_baseline section of the output YAML.
"""
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from dataset.unified_carla_dataset import CARLAImageDataset


def _finalize(accum, keys_and_poses):
    """Compute mean/std from accumulators and print results."""
    results = {}
    for key, _ in keys_and_poses:
        a = accum[key]
        np_ = a['num_poses']
        x_mean = [float(np.mean(a['all_x'][t])) for t in range(np_)]
        x_std  = [float(np.std(a['all_x'][t]))  for t in range(np_)]
        y_mean = [float(np.mean(a['all_y'][t])) for t in range(np_)]
        y_std  = [float(np.std(a['all_y'][t]))  for t in range(np_)]
        x_std = [max(s, 1e-3) for s in x_std]
        y_std = [max(s, 1e-3) for s in y_std]

        print(f"\n[{key}] Per-waypoint statistics ({np_} poses):")
        print(f"  x_mean: {[f'{v:.4f}' for v in x_mean]}")
        print(f"  x_std:  {[f'{v:.4f}' for v in x_std]}")
        print(f"  y_mean: {[f'{v:.4f}' for v in y_mean]}")
        print(f"  y_std:  {[f'{v:.4f}' for v in y_std]}")

        results[key] = (x_mean, x_std, y_mean, y_std)
    return results


# pkl key mapping: pkl uses raw keys, dataset.__getitem__ renames them
_PKL_KEY_MAP = {'route': 'route', 'agent_pos': 'ego_waypoints'}


def compute_stats_fast(dataset_path, keys_and_poses):
    """Fast path: read samples_packed.pkl directly, skip CARLAImageDataset entirely.

    No BEV loading, no tensor conversion — pure numpy on raw pkl dicts.
    """
    import pickle, time

    packed_path = os.path.join(dataset_path, 'samples_packed.pkl')
    if not os.path.exists(packed_path):
        raise FileNotFoundError(
            f"samples_packed.pkl not found at {packed_path}. "
            "Run training once to generate it, or use --no-fast.")

    print(f"[fast] Loading {packed_path} ...")
    t0 = time.time()
    with open(packed_path, 'rb') as f:
        all_samples = pickle.load(f)
    print(f"[fast] Loaded {len(all_samples)} samples in {time.time()-t0:.1f}s")

    accum = {}
    for key, num_poses in keys_and_poses:
        accum[key] = {
            'num_poses': num_poses,
            'all_x': [[] for _ in range(num_poses)],
            'all_y': [[] for _ in range(num_poses)],
        }

    keys_str = ', '.join(f'{k}[:{n}]' for k, n in keys_and_poses)
    print(f"[fast] Computing stats for {keys_str} ...")

    for sample in tqdm(all_samples):
        for key, num_poses in keys_and_poses:
            pkl_key = _PKL_KEY_MAP.get(key, key)
            traj = sample.get(pkl_key)
            if traj is None:
                continue
            # ego_waypoints -> agent_pos: skip first row (current pos)
            if pkl_key == 'ego_waypoints':
                traj = traj[1:]
            if traj.shape[0] >= num_poses:
                for t in range(num_poses):
                    accum[key]['all_x'][t].append(float(traj[t, 0]))
                    accum[key]['all_y'][t].append(float(traj[t, 1]))

    return _finalize(accum, keys_and_poses)


def compute_stats(dataset_path, image_data_root, keys_and_poses):
    """Original path: uses CARLAImageDataset.__getitem__ (slower due to BEV loading)."""
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        image_data_root=image_data_root,
        skip_memmap=True,
    )

    accum = {}
    for key, num_poses in keys_and_poses:
        accum[key] = {
            'num_poses': num_poses,
            'all_x': [[] for _ in range(num_poses)],
            'all_y': [[] for _ in range(num_poses)],
        }

    keys_str = ', '.join(f'{k}[:{n}]' for k, n in keys_and_poses)
    print(f"Computing stats for {keys_str} from {len(dataset)} samples...")

    for i in tqdm(range(len(dataset))):
        try:
            sample = dataset[i]
        except Exception:
            continue
        for key, num_poses in keys_and_poses:
            traj = sample.get(key)
            if traj is not None and traj.shape[0] >= num_poses:
                traj_np = traj[:num_poses].numpy()
                for t in range(num_poses):
                    accum[key]['all_x'][t].append(float(traj_np[t, 0]))
                    accum[key]['all_y'][t].append(float(traj_np[t, 1]))

    return _finalize(accum, keys_and_poses)


def update_yaml(yaml_path, x_mean, x_std, y_mean, y_std, key='route'):
    """Update norm stats in the bridge_baseline section of the YAML file.

    For key='route':   writes norm_x_mean / norm_x_std / norm_y_mean / norm_y_std
    For key='agent_pos': writes traj_norm_x_mean / ... (predict_traj mode)
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'bridge_baseline' not in config:
        config['bridge_baseline'] = {}

    prefix = 'traj_norm' if key == 'agent_pos' else 'norm'
    config['bridge_baseline'][f'{prefix}_x_mean'] = [round(v, 6) for v in x_mean]
    config['bridge_baseline'][f'{prefix}_x_std']  = [round(v, 6) for v in x_std]
    config['bridge_baseline'][f'{prefix}_y_mean'] = [round(v, 6) for v in y_mean]
    config['bridge_baseline'][f'{prefix}_y_std']  = [round(v, 6) for v in y_std]

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Updated {yaml_path} ({prefix}_*) with per-waypoint norm stats.")


def main():
    parser = argparse.ArgumentParser(description="Compute per-waypoint route stats for Bridge Baseline")
    parser.add_argument('--dataset_path', type=str,
                        default='/media/z/data/dataset/pdm_lite_mini/train')
    parser.add_argument('--image_data_root', type=str, default=None)
    parser.add_argument('--num_poses', type=int, default=10,
                        help='Number of poses (only used when --key is route or agent_pos)')
    parser.add_argument('--key', type=str, default='all', choices=['route', 'agent_pos', 'all'],
                        help='all: compute both route(10) and agent_pos(6) in one pass')
    parser.add_argument('--output_yaml', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bd_config.yaml'))
    parser.add_argument('--fast', action='store_true',
                        help='Read samples_packed.pkl directly, skip BEV loading (much faster)')
    args = parser.parse_args()

    if args.image_data_root is None:
        args.image_data_root = os.path.dirname(args.dataset_path)

    if args.key == 'all':
        keys_and_poses = [('route', 10), ('agent_pos', 6)]
    else:
        keys_and_poses = [(args.key, args.num_poses)]

    if args.fast:
        results = compute_stats_fast(args.dataset_path, keys_and_poses)
    else:
        results = compute_stats(args.dataset_path, args.image_data_root, keys_and_poses)

    if os.path.exists(args.output_yaml):
        for key, (x_mean, x_std, y_mean, y_std) in results.items():
            update_yaml(args.output_yaml, x_mean, x_std, y_mean, y_std, key=key)
    else:
        print(f"\nYAML not found at {args.output_yaml}, printing stats only.")


if __name__ == '__main__':
    main()
