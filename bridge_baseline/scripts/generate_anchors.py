#!/usr/bin/env python3
"""
Generate K-means trajectory anchors from CARLA dataset.

Supports both route (geometric waypoints) and agent_pos (ego trajectory) keys.
--fast reads samples_packed.pkl directly, skipping BEV loading (~7s vs ~1hr).

Usage:
    # Route anchors (bridge_baseline), fast mode:
    python bridge_baseline/scripts/generate_anchors.py \
        --dataset_path /path/to/pdm_lite/train \
        --key route --num_poses 10 --num_modes 20 --fast \
        --output bridge_baseline/anchors/carla_kmeans_20_route10.npy

    # Traj anchors (dd_baseline):
    python bridge_baseline/scripts/generate_anchors.py \
        --dataset_path /path/to/pdm_lite/train \
        --key agent_pos --num_poses 6 --num_modes 20 --fast \
        --output dd_baseline/anchors/carla_kmeans_20.npy
"""
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# pkl key mapping: pkl uses raw keys, dataset.__getitem__ renames them
_PKL_KEY_MAP = {'route': 'route', 'agent_pos': 'ego_waypoints'}


def collect_trajectories_fast(dataset_path, key, num_poses):
    """Fast path: read samples_packed.pkl directly, skip CARLAImageDataset."""
    import pickle, time

    packed_path = os.path.join(dataset_path, 'samples_packed.pkl')
    if not os.path.exists(packed_path):
        raise FileNotFoundError(
            f"samples_packed.pkl not found at {packed_path}. "
            "Run training once to generate it, or use without --fast.")

    print(f"[fast] Loading {packed_path} ...")
    t0 = time.time()
    with open(packed_path, 'rb') as f:
        all_samples = pickle.load(f)
    print(f"[fast] Loaded {len(all_samples)} samples in {time.time()-t0:.1f}s")

    pkl_key = _PKL_KEY_MAP.get(key, key)
    trajectories = []
    for sample in tqdm(all_samples, desc=f"Collecting {key}[:{num_poses}]"):
        traj = sample.get(pkl_key)
        if traj is None:
            continue
        if pkl_key == 'ego_waypoints':
            traj = traj[1:]  # skip current pos
        if traj.shape[0] >= num_poses:
            trajectories.append(traj[:num_poses].astype(np.float32))

    trajectories = np.array(trajectories)  # (N, num_poses, 2)
    print(f"Collected {len(trajectories)} valid trajectories")
    return trajectories


def collect_trajectories(dataset_path, image_data_root, key, num_poses):
    """Original path: uses CARLAImageDataset.__getitem__ (slower)."""
    from dataset.unified_carla_dataset import CARLAImageDataset

    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        image_data_root=image_data_root,
        skip_memmap=True,
    )

    trajectories = []
    print(f"Collecting {key}[:{num_poses}] from {len(dataset)} samples...")

    for i in tqdm(range(len(dataset))):
        try:
            sample = dataset[i]
            traj = sample.get(key)
            if traj is not None and traj.shape[0] >= num_poses:
                trajectories.append(traj[:num_poses].numpy())
        except Exception:
            continue

    trajectories = np.array(trajectories)  # (N, num_poses, 2)
    print(f"Collected {len(trajectories)} valid trajectories")
    return trajectories


def cluster_trajectories(trajectories, num_modes, seed=42):
    """Run K-means clustering on trajectories."""
    N, T, D = trajectories.shape
    flat = trajectories.reshape(N, T * D)  # (N, T*2)

    print(f"Running K-means with {num_modes} clusters on {N} trajectories...")
    kmeans = KMeans(n_clusters=num_modes, random_state=seed, n_init=10, max_iter=300)
    kmeans.fit(flat)

    centers = kmeans.cluster_centers_.reshape(num_modes, T, D)  # (K, T, 2)
    labels = kmeans.labels_

    print(f"\nCluster statistics:")
    for k in range(num_modes):
        count = np.sum(labels == k)
        center = centers[k]
        x_range = f"[{center[:, 0].min():.1f}, {center[:, 0].max():.1f}]"
        y_range = f"[{center[:, 1].min():.1f}, {center[:, 1].max():.1f}]"
        print(f"  Mode {k:2d}: {count:6d} samples ({count/N*100:5.1f}%), "
              f"x={x_range}, y={y_range}")

    return centers


def main():
    parser = argparse.ArgumentParser(description="Generate K-means trajectory anchors")
    parser.add_argument('--dataset_path', type=str,
                        default='/media/z/data/dataset/pdm_lite_mini/train',
                        help='Path to training dataset (containing samples_packed.pkl for --fast)')
    parser.add_argument('--image_data_root', type=str, default=None,
                        help='Image data root (defaults to dataset_path parent, only used without --fast)')
    parser.add_argument('--key', type=str, default='route', choices=['route', 'agent_pos'],
                        help='route: geometric waypoints (bridge), agent_pos: ego trajectory (dd)')
    parser.add_argument('--num_modes', type=int, default=20,
                        help='Number of K-means clusters')
    parser.add_argument('--num_poses', type=int, default=None,
                        help='Number of waypoints (default: 10 for route, 6 for agent_pos)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .npy file path (auto-generated if not set)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fast', action='store_true',
                        help='Read samples_packed.pkl directly, skip BEV loading (much faster)')
    args = parser.parse_args()

    # Defaults based on key
    if args.num_poses is None:
        args.num_poses = 10 if args.key == 'route' else 6

    if args.output is None:
        if args.key == 'route':
            args.output = os.path.join(project_root,
                'bridge_baseline', 'anchors', f'carla_kmeans_{args.num_modes}_route{args.num_poses}.npy')
        else:
            args.output = os.path.join(project_root,
                'dd_baseline', 'anchors', f'carla_kmeans_{args.num_modes}.npy')

    if args.image_data_root is None:
        args.image_data_root = os.path.dirname(args.dataset_path)

    # Collect
    if args.fast:
        trajectories = collect_trajectories_fast(args.dataset_path, args.key, args.num_poses)
    else:
        trajectories = collect_trajectories(args.dataset_path, args.image_data_root, args.key, args.num_poses)

    if len(trajectories) < args.num_modes:
        raise ValueError(f"Not enough trajectories ({len(trajectories)}) for {args.num_modes} clusters")

    # Cluster
    centers = cluster_trajectories(trajectories, args.num_modes, args.seed)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, centers.astype(np.float32))
    print(f"\nAnchors saved to {args.output}")
    print(f"Shape: {centers.shape} (num_modes={args.num_modes}, num_poses={args.num_poses}, 2)")


if __name__ == '__main__':
    main()
