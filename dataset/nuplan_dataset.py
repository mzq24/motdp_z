"""PyTorch Dataset that reads PlanTF caches directly (no .npz preprocessing).

Structure: cache_plantf_{split}/{log_name}/{scenario_type}/{token}/feature.gz + trajectory.gz

Each sample is converted on-the-fly to our model format.
Memory-mapped where possible; all coordinates are ego-centric.
"""

import gzip
import hashlib
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# PlanTF agent category enum
AGENT_EGO = 0
AGENT_VEHICLE = 1
AGENT_PEDESTRIAN = 2
AGENT_BICYCLE = 3


SCENARIO_TYPE_TO_TARGET_TYPE_14 = {
    'behind_long_vehicle': 'behind_long_vehicle',
    'changing_lane': 'changing_lane',
    'following_lane_with_lead': 'following_lane_with_lead',
    'high_lateral_acceleration': 'high_lateral_acceleration',
    'high_magnitude_speed': 'high_magnitude_speed',
    'low_magnitude_speed': 'low_magnitude_speed',
    'near_multiple_vehicles': 'near_multiple_vehicles',
    'starting_left_turn': 'starting_left_turn',
    'starting_right_turn': 'starting_right_turn',
    'starting_straight_traffic_light_intersection_traversal': 'starting_straight_traffic_light_intersection_traversal',
    'stationary_in_traffic': 'stationary_in_traffic',
    'stopping_with_lead': 'stopping_with_lead',
    'traversing_pickup_dropoff': 'traversing_pickup_dropoff',
    'waiting_for_pedestrian_to_cross': 'waiting_for_pedestrian_to_cross',
    'accelerating_at_crosswalk': 'waiting_for_pedestrian_to_cross',
    'accelerating_at_traffic_light_with_lead': 'starting_straight_traffic_light_intersection_traversal',
    'behind_bike': 'near_multiple_vehicles',
    'behind_pedestrian_on_driveable': 'waiting_for_pedestrian_to_cross',
    'changing_lane_to_left': 'changing_lane',
    'changing_lane_to_right': 'changing_lane',
    'crossed_by_bike': 'near_multiple_vehicles',
    'crossed_by_vehicle': 'near_multiple_vehicles',
    'following_lane_with_slow_lead': 'following_lane_with_lead',
    'following_lane_without_lead': 'following_lane_with_lead',
    'high_magnitude_jerk': 'high_lateral_acceleration',
    'near_high_speed_vehicle': 'near_multiple_vehicles',
    'near_long_vehicle': 'behind_long_vehicle',
    'near_multiple_bikes': 'near_multiple_vehicles',
    'near_multiple_pedestrians': 'waiting_for_pedestrian_to_cross',
    'near_pedestrian_on_crosswalk': 'waiting_for_pedestrian_to_cross',
    'on_carpark': 'traversing_pickup_dropoff',
    'on_intersection': 'starting_straight_traffic_light_intersection_traversal',
    'on_stopline_crosswalk': 'waiting_for_pedestrian_to_cross',
    'on_stopline_stop_sign': 'stopping_with_lead',
    'on_stopline_traffic_light': 'starting_straight_traffic_light_intersection_traversal',
    'on_traffic_light_intersection': 'starting_straight_traffic_light_intersection_traversal',
    'starting_protected_cross_turn': 'starting_right_turn',
    'starting_protected_noncross_turn': 'starting_left_turn',
    'starting_straight_stop_sign_intersection_traversal': 'starting_straight_traffic_light_intersection_traversal',
    'starting_u_turn': 'high_lateral_acceleration',
    'starting_unprotected_cross_turn': 'starting_right_turn',
    'starting_unprotected_noncross_turn': 'starting_left_turn',
    'stationary': 'stationary_in_traffic',
    'stationary_at_crosswalk': 'waiting_for_pedestrian_to_cross',
    'stationary_at_traffic_light_with_lead': 'stationary_in_traffic',
    'stationary_at_traffic_light_without_lead': 'stationary_in_traffic',
    'stopping_at_crosswalk': 'waiting_for_pedestrian_to_cross',
    'stopping_at_stop_sign_no_crosswalk': 'stopping_with_lead',
    'stopping_at_stop_sign_without_lead': 'stopping_with_lead',
    'stopping_at_traffic_light_without_lead': 'stopping_with_lead',
    'traversing_crosswalk': 'waiting_for_pedestrian_to_cross',
    'traversing_intersection': 'starting_straight_traffic_light_intersection_traversal',
    'traversing_narrow_lane': 'following_lane_with_lead',
    'traversing_traffic_light_intersection': 'starting_straight_traffic_light_intersection_traversal',
}


def map_scenario_type_to_target_type_14(scenario_type: str) -> str:
    """Map raw PlanTF cache directory names to the merged official CL14 task label."""
    return SCENARIO_TYPE_TO_TARGET_TYPE_14.get(scenario_type, scenario_type)


def build_target_type_rng(sampling_seed: int, target_type: str) -> random.Random:
    """Create a deterministic RNG per target type so partitions stay stable across configs."""
    digest = hashlib.sha256(f"{sampling_seed}:{target_type}".encode('utf-8')).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder='big', signed=False))


def convert_plantf_to_tensors(feature: dict, trajectory: dict) -> tuple:
    """Convert PlanTF feature/trajectory to model input tensors.

    Returns a tuple matching the collate function expectation:
      (ego_cur, ego_fut, neighbor_past, neighbor_fut,
       lanes, lanes_sl, lanes_hsl, route_lanes, static_objs)
    """
    agent = feature['agent']
    map_data = feature['map']

    # === Ego current (always at origin in ego-centric) ===
    ego_cur = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    # === Ego future ===
    ego_target = agent['target'][0]  # (80, 3) = [x, y, heading]
    ego_fut = torch.tensor(ego_target[:, :2], dtype=torch.float32)

    # === Neighbor agents past (32, 21, 11) ===
    num_agents_total = agent['position'].shape[0]
    past_start = 101 - 21

    neighbor_past = torch.zeros(32, 21, 11)
    for i in range(1, min(num_agents_total, 33)):
        out_idx = i - 1
        if out_idx >= 32:
            break

        pos = torch.tensor(agent['position'][i, past_start:, :], dtype=torch.float32)
        heading = agent['heading'][i, past_start:]
        vel = torch.tensor(agent['velocity'][i, past_start:, :], dtype=torch.float32)
        shape_info = torch.tensor(agent['shape'][i, past_start:, :2], dtype=torch.float32)
        valid = agent['valid_mask'][i, past_start:]
        cat = agent['category'][i]

        neighbor_past[out_idx, :, 0] = pos[:, 0]
        neighbor_past[out_idx, :, 1] = pos[:, 1]
        neighbor_past[out_idx, :, 2] = torch.cos(torch.tensor(heading, dtype=torch.float32))
        neighbor_past[out_idx, :, 3] = torch.sin(torch.tensor(heading, dtype=torch.float32))
        neighbor_past[out_idx, :, 4] = vel[:, 0]
        neighbor_past[out_idx, :, 5] = vel[:, 1]
        neighbor_past[out_idx, :, 6] = shape_info[:, 0]
        neighbor_past[out_idx, :, 7] = shape_info[:, 1]
        if cat == AGENT_VEHICLE:
            neighbor_past[out_idx, :, 8] = 1.0
        elif cat == AGENT_PEDESTRIAN:
            neighbor_past[out_idx, :, 9] = 1.0
        elif cat == AGENT_BICYCLE:
            neighbor_past[out_idx, :, 10] = 1.0
        neighbor_past[out_idx, ~valid] = 0.0

    # === Neighbors future (10, 80, 4) ===
    neighbor_fut = torch.zeros(10, 80, 4)
    for i in range(1, min(num_agents_total, 11)):
        out_idx = i - 1
        target = agent['target'][i]  # (80, 3)
        neighbor_fut[out_idx, :, 0] = torch.tensor(target[:, 0], dtype=torch.float32)
        neighbor_fut[out_idx, :, 1] = torch.tensor(target[:, 1], dtype=torch.float32)
        neighbor_fut[out_idx, :, 2] = torch.cos(torch.tensor(target[:, 2], dtype=torch.float32))
        neighbor_fut[out_idx, :, 3] = torch.sin(torch.tensor(target[:, 2], dtype=torch.float32))

    # === Lanes (30, 20, 12) ===
    M = min(map_data['point_position'].shape[0], 30)
    lanes = torch.zeros(30, 20, 12)
    lanes_sl = torch.zeros(30, 1)
    lanes_hsl = torch.zeros(30, 1, dtype=torch.bool)

    for i in range(M):
        center_pos = torch.tensor(map_data['point_position'][i, 0], dtype=torch.float32)
        center_vec = torch.tensor(map_data['point_vector'][i, 0], dtype=torch.float32)
        left_pos = torch.tensor(map_data['point_position'][i, 1], dtype=torch.float32)
        right_pos = torch.tensor(map_data['point_position'][i, 2], dtype=torch.float32)

        left_vec = left_pos - center_pos
        right_vec = right_pos - center_pos

        lanes[i, :, 0] = center_pos[:, 0]
        lanes[i, :, 1] = center_pos[:, 1]
        lanes[i, :, 2] = center_vec[:, 0]
        lanes[i, :, 3] = center_vec[:, 1]
        lanes[i, :, 4] = left_vec[:, 0]
        lanes[i, :, 5] = left_vec[:, 1]
        lanes[i, :, 6] = right_vec[:, 0]
        lanes[i, :, 7] = right_vec[:, 1]

        valid = map_data['valid_mask'][i]
        lanes[i, ~valid] = 0.0

        if i < len(map_data.get('polygon_has_speed_limit', [])):
            lanes_hsl[i] = torch.tensor(map_data['polygon_has_speed_limit'][i], dtype=torch.bool)
            lanes_sl[i] = torch.tensor(map_data['polygon_speed_limit'][i], dtype=torch.float32)

    # === Route lanes (10, 20, 4) ===
    route_lanes = torch.zeros(10, 20, 4)
    src_idx = 0
    if 'polygon_on_route' in map_data:
        route_mask = map_data['polygon_on_route']
        for i in range(min(M, 30)):
            if route_mask[i] and src_idx < 10:
                route_lanes[src_idx, :, 0] = lanes[i, :, 0]
                route_lanes[src_idx, :, 1] = lanes[i, :, 1]
                route_lanes[src_idx, :, 2] = lanes[i, :, 2]
                route_lanes[src_idx, :, 3] = lanes[i, :, 3]
                src_idx += 1

    # === Static objects (5, 10) ===
    static_objs = torch.zeros(5, 10)

    return (ego_cur, ego_fut, neighbor_past, neighbor_fut,
            lanes, lanes_sl, lanes_hsl, route_lanes, static_objs)


class NuPlanDataset(Dataset):
    """Direct-loading dataset from PlanTF caches.

    Args:
        cache_dirs: List of PlanTF cache directory paths.
        max_samples: Optionally limit total samples.
    """

    def __init__(
        self,
        cache_dirs: List[str],
        max_samples: Optional[int] = None,
        allowed_scenario_types: Optional[List[str]] = None,
        allowed_target_types: Optional[List[str]] = None,
        samples_per_target_type: Optional[int] = None,
        repeat_small_target_types: bool = False,
        sampling_seed: int = 0,
        default_target_type_partition_index: int = 0,
        target_type_partition_indices: Optional[Dict[str, int]] = None,
    ):
        self.samples: List[Tuple[str, str]] = []  # (feature_path, trajectory_path)
        self.available_sample_counts_by_target_type: Dict[str, int] = {}
        self.available_partition_sample_counts_by_target_type: Dict[str, int] = {}
        self.partition_index_by_target_type: Dict[str, int] = {}
        self.selected_sample_counts_by_target_type: Dict[str, int] = {}

        allowed_scenario_set = set(allowed_scenario_types) if allowed_scenario_types else None
        allowed_target_set = set(allowed_target_types) if allowed_target_types else None
        target_type_partition_indices = target_type_partition_indices or {}
        should_group_samples = allowed_target_set is not None or samples_per_target_type is not None
        grouped_samples: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for cache_dir in cache_dirs:
            cache_path = Path(cache_dir)
            if not cache_path.is_dir():
                continue
            for log_dir in sorted(cache_path.iterdir()):
                if not log_dir.is_dir():
                    continue
                for st_dir in sorted(log_dir.iterdir()):
                    if not st_dir.is_dir():
                        continue
                    scenario_type = st_dir.name
                    if allowed_scenario_set is not None and scenario_type not in allowed_scenario_set:
                        continue

                    target_type = map_scenario_type_to_target_type_14(scenario_type)
                    if allowed_target_set is not None and target_type not in allowed_target_set:
                        continue

                    for token_dir in sorted(st_dir.iterdir()):
                        if not token_dir.is_dir():
                            continue
                        feat_path = token_dir / 'feature.gz'
                        traj_path = token_dir / 'trajectory.gz'
                        if feat_path.exists() and traj_path.exists():
                            sample = (str(feat_path), str(traj_path))
                            if should_group_samples:
                                grouped_samples[target_type].append(sample)
                            else:
                                self.samples.append(sample)

        if should_group_samples:
            for target_type in sorted(grouped_samples):
                type_rng = build_target_type_rng(sampling_seed, target_type)
                target_samples = list(grouped_samples[target_type])
                type_rng.shuffle(target_samples)
                self.available_sample_counts_by_target_type[target_type] = len(target_samples)
                partition_index = target_type_partition_indices.get(target_type, default_target_type_partition_index)
                self.partition_index_by_target_type[target_type] = partition_index

                if samples_per_target_type is None:
                    selected_samples = list(target_samples)
                else:
                    partition_start = partition_index * samples_per_target_type
                    partition_end = partition_start + samples_per_target_type
                    partition_samples = target_samples[partition_start:partition_end]
                    self.available_partition_sample_counts_by_target_type[target_type] = max(0, len(target_samples) - partition_start)

                    if repeat_small_target_types and 0 < len(partition_samples) < samples_per_target_type:
                        repeat_count = samples_per_target_type - len(partition_samples)
                        selected_samples = list(partition_samples)
                        selected_samples.extend(type_rng.choices(partition_samples, k=repeat_count))
                        type_rng.shuffle(selected_samples)
                    else:
                        selected_samples = list(partition_samples)

                self.selected_sample_counts_by_target_type[target_type] = len(selected_samples)
                self.samples.extend(selected_samples)

        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, traj_path = self.samples[idx]
        with gzip.open(feat_path, 'rb') as f:
            feature = pickle.load(f)
        with gzip.open(traj_path, 'rb') as f:
            trajectory = pickle.load(f)
        return convert_plantf_to_tensors(feature, trajectory)
