#!/usr/bin/env python3
"""
计算数据集中的action_stats (agent_pos和anchor统计信息)
适用于CARLA数据集，统计ego_waypoints/agent_pos和anchor(pred_traj)的min, max, mean, std
最后取两者的最小min和最大max，并更新yaml中truncated_diffusion的归一化参数
"""
import os
import sys
import pickle
import glob
import numpy as np
import torch
from tqdm import tqdm
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def compute_global_abs_stats_from_dataset(dataset_path, max_samples=None):
    """
    统计所有 timestep 汇总的 global abs mean/std，shape=(2,)。
    所有 timestep 共享同一组 mean/std，保留轨迹时序结构。

    Returns:
        global_abs_mean: (2,)  x/y 全局均值
        global_abs_std:  (2,)  x/y 全局标准差
    """
    print(f"\n{'='*60}")
    print(f"Computing global absolute coordinate statistics...")
    print(f"Dataset path: {dataset_path}")
    print(f"{'='*60}\n")

    train_files = glob.glob(os.path.join(dataset_path, "train", "*.pkl"))
    val_files = glob.glob(os.path.join(dataset_path, "val", "*.pkl"))
    direct_files = glob.glob(os.path.join(dataset_path, "*.pkl"))

    if train_files or val_files:
        all_files = sorted(train_files + val_files)
        print(f"Found {len(all_files)} samples ({len(train_files)} train, {len(val_files)} val)")
    elif direct_files:
        all_files = sorted(direct_files)
        print(f"Found {len(all_files)} samples")
    else:
        raise FileNotFoundError(f"No pkl files found in {dataset_path}")

    if max_samples is not None:
        all_files = all_files[:max_samples]
        print(f"Limiting to {len(all_files)} samples")

    all_points = []  # collect all (x, y) points across all timesteps
    failed = 0

    for pkl_file in tqdm(all_files, desc="Processing"):
        try:
            with open(pkl_file, 'rb') as f:
                sample = pickle.load(f)

            ego_waypoints = sample.get('ego_waypoints')
            if ego_waypoints is None:
                continue

            if isinstance(ego_waypoints, torch.Tensor):
                if ego_waypoints.dtype == torch.bfloat16:
                    ego_waypoints = ego_waypoints.float()
                ego_waypoints = ego_waypoints.cpu().numpy()
            elif not isinstance(ego_waypoints, np.ndarray):
                ego_waypoints = np.array(ego_waypoints)

            if len(ego_waypoints) <= 1:
                continue
            abs_traj = ego_waypoints[1:]  # (T, 2)
            all_points.append(abs_traj)  # keep (T, 2) shape, flatten later

        except Exception as e:
            failed += 1
            continue

    if failed > 0:
        print(f"Failed to load {failed} samples")

    if len(all_points) == 0:
        raise ValueError("No valid trajectories found!")

    # Flatten all timesteps: (N*T, 2)
    all_points = np.concatenate(all_points, axis=0)  # each is (T, 2), concat along axis=0
    print(f"Collected {all_points.shape[0]} total points (N*T)")

    # Global stats: (2,)
    global_abs_mean = all_points.mean(axis=0)
    global_abs_std = all_points.std(axis=0)
    global_abs_std = np.maximum(global_abs_std, 1e-6)

    print(f"\nGlobal absolute coordinate statistics:")
    print(f"  mean=({global_abs_mean[0]:+.4f}, {global_abs_mean[1]:+.4f})")
    print(f"  std=({global_abs_std[0]:.4f}, {global_abs_std[1]:.4f})")

    return global_abs_mean.astype(np.float32), global_abs_std.astype(np.float32)


def compute_abs_stats_from_dataset(dataset_path, max_samples=None):
    """
    统计 per-step absolute coordinate 的 mean 和 std，用于 abs z-score 归一化。
    每个 timestep 独立统计，不做 delta 差分。

    Returns:
        abs_mean: (T, 2) 每步绝对坐标的均值
        abs_std:  (T, 2) 每步绝对坐标的标准差
    """
    print(f"\n{'='*60}")
    print(f"Computing per-step absolute coordinate statistics...")
    print(f"Dataset path: {dataset_path}")
    print(f"{'='*60}\n")

    train_files = glob.glob(os.path.join(dataset_path, "train", "*.pkl"))
    val_files = glob.glob(os.path.join(dataset_path, "val", "*.pkl"))
    direct_files = glob.glob(os.path.join(dataset_path, "*.pkl"))

    if train_files or val_files:
        all_files = sorted(train_files + val_files)
        print(f"Found {len(all_files)} samples ({len(train_files)} train, {len(val_files)} val)")
    elif direct_files:
        all_files = sorted(direct_files)
        print(f"Found {len(all_files)} samples")
    else:
        raise FileNotFoundError(f"No pkl files found in {dataset_path}")

    if max_samples is not None:
        all_files = all_files[:max_samples]
        print(f"Limiting to {len(all_files)} samples")

    all_abs = []
    failed = 0

    for pkl_file in tqdm(all_files, desc="Processing"):
        try:
            with open(pkl_file, 'rb') as f:
                sample = pickle.load(f)

            ego_waypoints = sample.get('ego_waypoints')
            if ego_waypoints is None:
                continue

            if isinstance(ego_waypoints, torch.Tensor):
                if ego_waypoints.dtype == torch.bfloat16:
                    ego_waypoints = ego_waypoints.float()
                ego_waypoints = ego_waypoints.cpu().numpy()
            elif not isinstance(ego_waypoints, np.ndarray):
                ego_waypoints = np.array(ego_waypoints)

            # Skip first point: ego_waypoints[1:] is agent_pos (relative to ego)
            if len(ego_waypoints) <= 1:
                continue
            abs_traj = ego_waypoints[1:]  # (T, 2)
            all_abs.append(abs_traj)

        except Exception as e:
            failed += 1
            continue

    if failed > 0:
        print(f"Failed to load {failed} samples")

    if len(all_abs) == 0:
        raise ValueError("No valid trajectories found!")

    # Stack: (N, T, 2)
    all_abs = np.stack(all_abs, axis=0)
    print(f"Collected {all_abs.shape[0]} trajectories, T={all_abs.shape[1]}")

    # Per-step stats: (T, 2)
    abs_mean = all_abs.mean(axis=0)
    abs_std = all_abs.std(axis=0)

    # Clamp std to avoid division by zero
    abs_std = np.maximum(abs_std, 1e-6)

    print(f"\nPer-step absolute coordinate statistics:")
    for t in range(abs_mean.shape[0]):
        print(f"  step {t}: mean=({abs_mean[t, 0]:+.4f}, {abs_mean[t, 1]:+.4f})  "
              f"std=({abs_std[t, 0]:.4f}, {abs_std[t, 1]:.4f})")

    return abs_mean.astype(np.float32), abs_std.astype(np.float32)


def compute_route_abs_stats_from_dataset(dataset_path, max_samples=None, num_points=20):
    """
    统计 route 的 per-waypoint absolute coordinate mean/std，用于 Route B route diffusion。

    Returns:
        route_abs_mean: (T_route, 2)
        route_abs_std:  (T_route, 2)
    """
    print(f"\n{'='*60}")
    print(f"Computing route per-waypoint absolute coordinate statistics...")
    print(f"Dataset path: {dataset_path}")
    print(f"Route points: {num_points}")
    print(f"{'='*60}\n")

    all_routes = []
    failed = 0
    train_packed = os.path.join(dataset_path, "train", "samples_packed.pkl")
    val_packed = os.path.join(dataset_path, "val", "samples_packed.pkl")
    packed_paths = [p for p in (train_packed, val_packed) if os.path.exists(p)]

    def _append_route(sample):
        route = sample.get('route')
        if route is None:
            return
        if isinstance(route, torch.Tensor):
            if route.dtype == torch.bfloat16:
                route = route.float()
            route = route.cpu().numpy()
        elif not isinstance(route, np.ndarray):
            route = np.array(route)
        if route.shape[0] < num_points:
            return
        all_routes.append(route[:num_points])

    if packed_paths:
        print(f"Using packed samples: {packed_paths}")
        remaining = max_samples
        total_loaded = 0
        for packed_path in packed_paths:
            with open(packed_path, 'rb') as f:
                packed_samples = pickle.load(f)
            samples = packed_samples
            if remaining is not None:
                samples = packed_samples[:remaining]
            print(f"Loaded {len(samples)} packed samples from {packed_path}")
            total_loaded += len(samples)

            split_name = os.path.basename(os.path.dirname(packed_path))
            for sample in tqdm(samples, desc=f"Processing {split_name} packed"):
                try:
                    _append_route(sample)
                except Exception:
                    failed += 1
                    continue

            if remaining is not None:
                remaining -= len(samples)
                if remaining <= 0:
                    break
        print(f"Found {total_loaded} packed samples")
    else:
        train_files = glob.glob(os.path.join(dataset_path, "train", "*.pkl"))
        val_files = glob.glob(os.path.join(dataset_path, "val", "*.pkl"))
        direct_files = glob.glob(os.path.join(dataset_path, "*.pkl"))

        if train_files or val_files:
            all_files = sorted(train_files + val_files)
            print(f"Found {len(all_files)} samples ({len(train_files)} train, {len(val_files)} val)")
        elif direct_files:
            all_files = sorted(direct_files)
            print(f"Found {len(all_files)} samples")
        else:
            raise FileNotFoundError(f"No pkl files found in {dataset_path}")

        if max_samples is not None:
            all_files = all_files[:max_samples]
            print(f"Limiting to {len(all_files)} samples")

        for pkl_file in tqdm(all_files, desc="Processing"):
            try:
                with open(pkl_file, 'rb') as f:
                    sample = pickle.load(f)
                _append_route(sample)
            except Exception:
                failed += 1
                continue

    if failed > 0:
        print(f"Failed to load {failed} samples")

    if len(all_routes) == 0:
        raise ValueError("No valid route trajectories found!")

    all_routes = np.stack(all_routes, axis=0)
    print(f"Collected {all_routes.shape[0]} routes, T_route={all_routes.shape[1]}")

    route_abs_mean = all_routes.mean(axis=0)
    route_abs_std = all_routes.std(axis=0)
    route_abs_std = np.maximum(route_abs_std, 1e-6)

    print(f"\nPer-waypoint route absolute coordinate statistics:")
    for t in range(route_abs_mean.shape[0]):
        print(
            f"  step {t}: mean=({route_abs_mean[t, 0]:+.4f}, {route_abs_mean[t, 1]:+.4f})  "
            f"std=({route_abs_std[t, 0]:.4f}, {route_abs_std[t, 1]:.4f})"
        )

    return route_abs_mean.astype(np.float32), route_abs_std.astype(np.float32)


def compute_delta_stats_from_dataset(dataset_path, max_samples=None):
    """
    统计 per-step delta 的 mean 和 std，用于 delta z-score 归一化。

    delta 定义: [p0, p1-p0, p2-p1, ...]  (p0 是相对 ego 的位移)

    Returns:
        delta_mean: (T, 2) 每步 delta 的均值
        delta_std:  (T, 2) 每步 delta 的标准差
    """
    print(f"\n{'='*60}")
    print(f"Computing per-step delta statistics...")
    print(f"Dataset path: {dataset_path}")
    print(f"{'='*60}\n")

    train_files = glob.glob(os.path.join(dataset_path, "train", "*.pkl"))
    val_files = glob.glob(os.path.join(dataset_path, "val", "*.pkl"))
    direct_files = glob.glob(os.path.join(dataset_path, "*.pkl"))

    if train_files or val_files:
        all_files = sorted(train_files + val_files)
        print(f"Found {len(all_files)} samples ({len(train_files)} train, {len(val_files)} val)")
    elif direct_files:
        all_files = sorted(direct_files)
        print(f"Found {len(all_files)} samples")
    else:
        raise FileNotFoundError(f"No pkl files found in {dataset_path}")

    if max_samples is not None:
        all_files = all_files[:max_samples]
        print(f"Limiting to {len(all_files)} samples")

    all_deltas = []
    failed = 0

    for pkl_file in tqdm(all_files, desc="Processing"):
        try:
            with open(pkl_file, 'rb') as f:
                sample = pickle.load(f)

            ego_waypoints = sample.get('ego_waypoints')
            if ego_waypoints is None:
                continue

            if isinstance(ego_waypoints, torch.Tensor):
                if ego_waypoints.dtype == torch.bfloat16:
                    ego_waypoints = ego_waypoints.float()
                ego_waypoints = ego_waypoints.cpu().numpy()
            elif not isinstance(ego_waypoints, np.ndarray):
                ego_waypoints = np.array(ego_waypoints)

            # Skip first point: ego_waypoints[1:] is agent_pos (relative to ego)
            if len(ego_waypoints) <= 1:
                continue
            abs_traj = ego_waypoints[1:]  # (T, 2)

            # abs -> delta: [p0, p1-p0, p2-p1, ...]
            delta = abs_traj.copy()
            delta[1:] = abs_traj[1:] - abs_traj[:-1]
            all_deltas.append(delta)

        except Exception as e:
            failed += 1
            continue

    if failed > 0:
        print(f"Failed to load {failed} samples")

    if len(all_deltas) == 0:
        raise ValueError("No valid trajectories found!")

    # Stack: (N, T, 2)
    all_deltas = np.stack(all_deltas, axis=0)
    print(f"Collected {all_deltas.shape[0]} trajectories, T={all_deltas.shape[1]}")

    # Per-step stats: (T, 2)
    delta_mean = all_deltas.mean(axis=0)
    delta_std = all_deltas.std(axis=0)

    # Clamp std to avoid division by zero
    delta_std = np.maximum(delta_std, 1e-6)

    print(f"\nPer-step delta statistics:")
    for t in range(delta_mean.shape[0]):
        print(f"  step {t}: mean=({delta_mean[t, 0]:+.4f}, {delta_mean[t, 1]:+.4f})  "
              f"std=({delta_std[t, 0]:.4f}, {delta_std[t, 1]:.4f})")

    return delta_mean.astype(np.float32), delta_std.astype(np.float32)


def compute_action_stats_from_dataset(dataset_path, image_data_root, max_samples=None):
    """
    从预处理的pkl文件中统计action (agent_pos和anchor) 的统计信息
    
    Args:
        dataset_path: 数据集路径，包含train/val子目录或直接包含pkl文件
        image_data_root: 图像数据根目录，用于加载vqa特征文件
        max_samples: 最多处理多少个样本（None表示处理全部）
    
    Returns:
        stats: 包含min, max, mean, std的字典（合并agent_pos和anchor范围）
    """
    print(f"\n{'='*60}")
    print(f"Computing action statistics from dataset...")
    print(f"Dataset path: {dataset_path}")
    print(f"Image data root: {image_data_root}")
    print(f"{'='*60}\n")
    
    train_files = glob.glob(os.path.join(dataset_path, "train", "*.pkl"))
    val_files = glob.glob(os.path.join(dataset_path, "val", "*.pkl"))
    direct_files = glob.glob(os.path.join(dataset_path, "*.pkl"))
    
    if train_files or val_files:
        all_files = sorted(train_files + val_files)
        print(f"✓ Found {len(all_files)} samples ({len(train_files)} train, {len(val_files)} val)")
    elif direct_files:
        all_files = sorted(direct_files)
        print(f"✓ Found {len(all_files)} samples")
    else:
        raise FileNotFoundError(f"No pkl files found in {dataset_path} or its train/val subdirectories")
    
    if max_samples is not None:
        all_files = all_files[:max_samples]
        print(f"⚠ Limiting to {len(all_files)} samples for statistics computation")
    
    all_agent_pos = []
    all_anchor = []
    failed_samples = 0
    
    print("\nLoading samples...")
    for pkl_file in tqdm(all_files, desc="Processing"):
        try:
            with open(pkl_file, 'rb') as f:
                sample = pickle.load(f)
            
            # 获取ego_waypoints (agent_pos)
            ego_waypoints = sample.get('ego_waypoints')
            
            if ego_waypoints is not None:
                if isinstance(ego_waypoints, torch.Tensor):
                    # Convert BFloat16 to Float32 if necessary
                    if ego_waypoints.dtype == torch.bfloat16:
                        ego_waypoints = ego_waypoints.float()
                    ego_waypoints = ego_waypoints.cpu().numpy()
                elif not isinstance(ego_waypoints, np.ndarray):
                    ego_waypoints = np.array(ego_waypoints)
                
                # 跳过第一个点 (ego_waypoints[1:])
                if len(ego_waypoints) > 1:
                    agent_pos = ego_waypoints[1:]
                    all_agent_pos.append(agent_pos)
            
            # 获取anchor (from vqa feature: pred_traj)
            vqa_path = sample.get('vqa', None)
            if vqa_path is not None and image_data_root is not None:
                full_vqa_path = os.path.join(image_data_root, vqa_path)
                if os.path.exists(full_vqa_path):
                    try:
                        # Try loading with weights_only=True first
                        vqa_feature = torch.load(full_vqa_path, weights_only=True, map_location='cpu')
                    except Exception as e:
                        # Fall back to loading without weights_only if BFloat16 issues occur
                        if 'BFloat16' in str(e):
                            vqa_feature = torch.load(full_vqa_path, weights_only=False, map_location='cpu')
                        else:
                            raise
                    
                    if 'pred_traj' in vqa_feature:
                        anchor = vqa_feature['pred_traj']
                        if isinstance(anchor, torch.Tensor):
                            # Convert BFloat16 to Float32 if necessary
                            if anchor.dtype == torch.bfloat16:
                                anchor = anchor.float()
                            anchor = anchor.cpu().numpy()
                        all_anchor.append(anchor)
            
        except Exception as e:
            print(f"\n⚠ Error loading {pkl_file}: {e}")
            failed_samples += 1
            continue
    
    if failed_samples > 0:
        print(f"\n⚠ Failed to load {failed_samples} samples")
    
    if len(all_agent_pos) == 0 and len(all_anchor) == 0:
        raise ValueError("No valid actions found in dataset!")
    
    print(f"\n✓ Successfully loaded {len(all_agent_pos)} agent_pos samples")
    print(f"✓ Successfully loaded {len(all_anchor)} anchor samples")
    
    print("Computing statistics...")
    
    # Compute agent_pos stats
    agent_pos_stats = None
    if len(all_agent_pos) > 0:
        all_agent_pos_flat = np.concatenate(all_agent_pos, axis=0)
        print(f"\nTotal agent_pos waypoints: {len(all_agent_pos_flat)}")
        print(f"agent_pos shape: {all_agent_pos_flat.shape}")
        
        agent_pos_stats = {
            'min': np.min(all_agent_pos_flat, axis=0),
            'max': np.max(all_agent_pos_flat, axis=0),
            'mean': np.mean(all_agent_pos_flat, axis=0),
            'std': np.std(all_agent_pos_flat, axis=0),
        }
        print(f"\nagent_pos stats:")
        print(f"  Min:  {agent_pos_stats['min']}")
        print(f"  Max:  {agent_pos_stats['max']}")
        print(f"  Mean: {agent_pos_stats['mean']}")
        print(f"  Std:  {agent_pos_stats['std']}")
    
    # Compute anchor stats
    anchor_stats = None
    if len(all_anchor) > 0:
        all_anchor_concat = np.concatenate(all_anchor, axis=0)
        print(f"\nTotal anchor samples: {len(all_anchor_concat)}")
        print(f"anchor shape before reshape: {all_anchor_concat.shape}")
        
        # If anchor has 3 dimensions (e.g., [N, seq_len, 2]), flatten to 2D [N*seq_len, 2]
        # This way stats will be computed across ALL waypoints, same as agent_pos
        if all_anchor_concat.ndim == 3:
            original_shape = all_anchor_concat.shape
            all_anchor_flat = all_anchor_concat.reshape(-1, all_anchor_concat.shape[-1])
            print(f"anchor shape after reshape: {all_anchor_flat.shape} (flattened from {original_shape})")
        else:
            all_anchor_flat = all_anchor_concat
            print(f"anchor shape: {all_anchor_flat.shape}")
        
        print(f"Total anchor waypoints (after flatten): {len(all_anchor_flat)}")
        
        anchor_stats = {
            'min': np.min(all_anchor_flat, axis=0),
            'max': np.max(all_anchor_flat, axis=0),
            'mean': np.mean(all_anchor_flat, axis=0),
            'std': np.std(all_anchor_flat, axis=0),
        }
        print(f"\nanchor stats:")
        print(f"  Min:  {anchor_stats['min']}")
        print(f"  Max:  {anchor_stats['max']}")
        print(f"  Mean: {anchor_stats['mean']}")
        print(f"  Std:  {anchor_stats['std']}")
    
    # Combine stats: take min of mins and max of maxs
    if agent_pos_stats is not None and anchor_stats is not None:
        combined_min = np.minimum(agent_pos_stats['min'], anchor_stats['min'])
        combined_max = np.maximum(agent_pos_stats['max'], anchor_stats['max'])
        # For mean and std, use the combined data
        # Now both should have the same number of dimensions (2D)
        all_combined = np.concatenate([all_agent_pos_flat, all_anchor_flat], axis=0)
        combined_mean = np.mean(all_combined, axis=0)
        combined_std = np.std(all_combined, axis=0)
    elif agent_pos_stats is not None:
        combined_min = agent_pos_stats['min']
        combined_max = agent_pos_stats['max']
        combined_mean = agent_pos_stats['mean']
        combined_std = agent_pos_stats['std']
    else:
        combined_min = anchor_stats['min']
        combined_max = anchor_stats['max']
        combined_mean = anchor_stats['mean']
        combined_std = anchor_stats['std']
    
    stats = {
        'min': torch.tensor(combined_min, dtype=torch.float32),
        'max': torch.tensor(combined_max, dtype=torch.float32),
        'mean': torch.tensor(combined_mean, dtype=torch.float32),
        'std': torch.tensor(combined_std, dtype=torch.float32),
    }
    
    print(f"\n{'='*60}")
    print("Combined Action Statistics (agent_pos + anchor):")
    print(f"{'='*60}")
    print(f"Dimensions: {combined_min.shape[0]} (x, y)")
    print(f"\nCombined Min:  {stats['min'].numpy()}")
    print(f"Combined Max:  {stats['max'].numpy()}")
    print(f"Combined Mean: {stats['mean'].numpy()}")
    print(f"Combined Std:  {stats['std'].numpy()}")
    print(f"{'='*60}\n")
    
    print("Python code format (copy to your training script):")
    print("-" * 60)
    print("action_stats = {")
    print(f"    'min': torch.tensor({stats['min'].tolist()}),")
    print(f"    'max': torch.tensor({stats['max'].tolist()}),")
    print(f"    'mean': torch.tensor({stats['mean'].tolist()}),")
    print(f"    'std': torch.tensor({stats['std'].tolist()}),")
    print("}")
    print("-" * 60)
    
    return stats


def save_stats_to_config(stats, config_path, output_path=None):
    if output_path is None:
        output_path = config_path
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'action_stats' not in config:
            config['action_stats'] = {}
        
        config['action_stats']['min'] = stats['min'].tolist()
        config['action_stats']['max'] = stats['max'].tolist()
        config['action_stats']['mean'] = stats['mean'].tolist()
        config['action_stats']['std'] = stats['std'].tolist()
        
        # Update truncated_diffusion normalization parameters
        # Normalization: 2*(x + offset)/range - 1, mapping data to [-1, 1]
        # offset = -x_min (with margin), range = x_max - x_min (with margin)
        x_min = stats['min'][0].item()
        x_max = stats['max'][0].item()
        y_min = stats['min'][1].item()
        y_max = stats['max'][1].item()
        
        # Add margin (round to nice numbers)
        margin = 1.0
        x_min_margin = np.floor(x_min - margin)
        x_max_margin = np.ceil(x_max + margin)
        y_min_margin = np.floor(y_min - margin)
        y_max_margin = np.ceil(y_max + margin)
        
        # Calculate normalization parameters
        norm_x_offset = -x_min_margin  # offset to make min become 0
        norm_x_range = x_max_margin - x_min_margin
        norm_y_offset = -y_min_margin
        norm_y_range = y_max_margin - y_min_margin
        
        if 'truncated_diffusion' not in config:
            config['truncated_diffusion'] = {}
        
        config['truncated_diffusion']['norm_x_offset'] = float(norm_x_offset)
        config['truncated_diffusion']['norm_x_range'] = float(norm_x_range)
        config['truncated_diffusion']['norm_y_offset'] = float(norm_y_offset)
        config['truncated_diffusion']['norm_y_range'] = float(norm_y_range)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✓ Action stats saved to config: {output_path}")
        print(f"\ntruncated_diffusion normalization parameters updated:")
        print(f"  Data range: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
        print(f"  Extended range: x=[{x_min_margin}, {x_max_margin}], y=[{y_min_margin}, {y_max_margin}]")
        print(f"  norm_x_offset: {norm_x_offset}")
        print(f"  norm_x_range: {norm_x_range}")
        print(f"  norm_y_offset: {norm_y_offset}")
        print(f"  norm_y_range: {norm_y_range}")
        
    except Exception as e:
        print(f"\n⚠ Failed to save stats to config: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compute action statistics from CARLA dataset')
    parser.add_argument(
        '--mode', type=str, default='legacy', choices=['legacy', 'delta', 'abs', 'global_abs', 'route_abs'],
        help='legacy: min/max; delta: per-step delta z-score; abs: per-step abs z-score; global_abs: global abs z-score; route_abs: per-waypoint route abs z-score'
    )
    parser.add_argument(
        '--dataset_path', type=str, default='/share-data/pdm_lite/tmp_data',
        help='Path to processed dataset (containing train/val folders or pkl files)'
    )
    parser.add_argument(
        '--image_data_root', type=str, default='/share-data/pdm_lite/',
        help='Root directory for image data (to load vqa feature files for anchor)'
    )
    parser.add_argument(
        '--config_path', type=str, default=None,
        help='Path to config file to save stats (legacy mode only)'
    )
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='Maximum number of samples to process (None = all)'
    )
    parser.add_argument(
        '--no_save', action='store_true',
        help='Do NOT save computed stats'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Output path (legacy: config yaml; delta: .npz file)'
    )
    parser.add_argument(
        '--route_points', type=int, default=20,
        help='Number of route waypoints to include when --mode route_abs'
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"Dataset path not found: {args.dataset_path}")
        return

    try:
        if args.mode == 'global_abs':
            g_mean, g_std = compute_global_abs_stats_from_dataset(
                dataset_path=args.dataset_path,
                max_samples=args.max_samples,
            )
            if not args.no_save:
                out = args.output_path or os.path.join(args.dataset_path, 'global_abs_stats.npz')
                np.savez(out, global_abs_mean=g_mean, global_abs_std=g_std)
                print(f"\nSaved global abs stats to {out}")
            print("\nGlobal abs statistics computation completed!")
        elif args.mode == 'abs':
            abs_mean, abs_std = compute_abs_stats_from_dataset(
                dataset_path=args.dataset_path,
                max_samples=args.max_samples,
            )
            if not args.no_save:
                out = args.output_path or os.path.join(args.dataset_path, 'abs_stats.npz')
                np.savez(out, abs_mean=abs_mean, abs_std=abs_std)
                print(f"\nSaved abs stats to {out}")
            print("\nAbs statistics computation completed!")
        elif args.mode == 'delta':
            delta_mean, delta_std = compute_delta_stats_from_dataset(
                dataset_path=args.dataset_path,
                max_samples=args.max_samples,
            )
            if not args.no_save:
                out = args.output_path or os.path.join(args.dataset_path, 'delta_stats.npz')
                np.savez(out, delta_mean=delta_mean, delta_std=delta_std)
                print(f"\nSaved delta stats to {out}")
            print("\nDelta statistics computation completed!")
        elif args.mode == 'route_abs':
            route_abs_mean, route_abs_std = compute_route_abs_stats_from_dataset(
                dataset_path=args.dataset_path,
                max_samples=args.max_samples,
                num_points=args.route_points,
            )
            if not args.no_save:
                out = args.output_path or os.path.join(args.dataset_path, 'route_abs_stats.npz')
                np.savez(out, route_abs_mean=route_abs_mean, route_abs_std=route_abs_std)
                print(f"\nSaved route abs stats to {out}")
            print("\nRoute abs statistics computation completed!")
        else:
            stats = compute_action_stats_from_dataset(
                dataset_path=args.dataset_path,
                image_data_root=args.image_data_root,
                max_samples=args.max_samples,
            )
            if not args.no_save and args.config_path and os.path.exists(args.config_path):
                save_stats_to_config(stats, args.config_path, args.output_path)
            print("\nStatistics computation completed!")

    except Exception as e:
        print(f"\nError computing statistics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
