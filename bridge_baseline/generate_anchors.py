#!/usr/bin/env python3
"""
Generate K-means trajectory anchors from CARLA dataset for Bridge Baseline.

Uses route[:10] (equidistant geometric route, first 10 of 20 points) as GT,
matching BridgeDrive's num_poses=10 convention.

Usage:
    python bridge_baseline/generate_anchors.py \
        --dataset_path /media/z/data/dataset/pdm_lite_mini/train \
        --num_modes 20 \
        --num_poses 10 \
        --output bridge_baseline/anchors/carla_kmeans_20_route10.npy
"""
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from dataset.unified_carla_dataset import CARLAImageDataset


def collect_trajectories(dataset_path, image_data_root, num_poses):
    """Collect route[:num_poses] trajectories from the dataset."""
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        image_data_root=image_data_root,
        skip_memmap=True,
    )

    trajectories = []
    print(f"Collecting route trajectories from {len(dataset)} samples...")

    for i in tqdm(range(len(dataset))):
        try:
            sample = dataset[i]
            # Use geometric route (equidistant), not agent_pos (time-spaced)
            route = sample['route']  # (20, 2) tensor
            if route.shape[0] >= num_poses:
                traj_np = route[:num_poses].numpy()  # (num_poses, 2)
                trajectories.append(traj_np)
        except Exception:
            continue

    trajectories = np.array(trajectories)  # (N, num_poses, 2)
    print(f"Collected {len(trajectories)} valid route trajectories")
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
    parser = argparse.ArgumentParser(description="Generate K-means anchors for Bridge Baseline")
    parser.add_argument('--dataset_path', type=str,
                        default='/media/z/data/dataset/pdm_lite_mini/train',
                        help='Path to training dataset')
    parser.add_argument('--image_data_root', type=str, default=None,
                        help='Image data root (defaults to dataset_path parent)')
    parser.add_argument('--num_modes', type=int, default=20,
                        help='Number of K-means clusters')
    parser.add_argument('--num_poses', type=int, default=10,
                        help='Number of route waypoints per trajectory (use route[:num_poses])')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'anchors', 'carla_kmeans_20_route10.npy'),
                        help='Output .npy file path')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.image_data_root is None:
        args.image_data_root = os.path.dirname(args.dataset_path)

    trajectories = collect_trajectories(args.dataset_path, args.image_data_root, args.num_poses)

    if len(trajectories) < args.num_modes:
        raise ValueError(f"Not enough trajectories ({len(trajectories)}) for {args.num_modes} clusters")

    centers = cluster_trajectories(trajectories, args.num_modes, args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, centers.astype(np.float32))
    print(f"\nAnchors saved to {args.output}")
    print(f"Shape: {centers.shape} (num_modes={args.num_modes}, num_poses={args.num_poses}, 2)")


if __name__ == '__main__':
    main()
