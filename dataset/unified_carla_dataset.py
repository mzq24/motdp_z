
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gzip
import json
import os
import io
import sys
import pickle
import glob
import random
import time
from collections import defaultdict
from typing import Optional
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import textwrap


def _apply_stage1_near_zero_speed_snap(final_sample, eps_mps=0.1):
    if 'speed_sample_values' not in final_sample:
        return

    speed_samples = final_sample['speed_sample_values']
    if not isinstance(speed_samples, torch.Tensor) or speed_samples.numel() == 0:
        return

    near_zero_mask = torch.abs(speed_samples) <= float(eps_mps)
    if not torch.any(near_zero_mask):
        return

    speed_samples = speed_samples.clone()
    speed_samples[near_zero_mask] = 0.0
    final_sample['speed_sample_values'] = speed_samples

    for key in (
        'speed_risk_chase_values',
        'speed_risk_meet_values',
        'speed_risk_cross_yld_values',
        'speed_risk_cross_go_values',
        'speed_risk_junction_cross_yld_values',
        'speed_risk_junction_cross_go_values',
        'speed_risk_merge_yld_values',
        'speed_risk_merge_go_values',
        'speed_risk_borrow_yld_values',
        'speed_risk_borrow_go_values',
        'speed_risk_ped_values',
    ):
        value = final_sample.get(key)
        if isinstance(value, torch.Tensor) and value.shape == speed_samples.shape:
            value = value.clone()
            value[near_zero_mask] = 0.0
            final_sample[key] = value


def _ensure_stage1_legacy_curve_defaults(final_sample):
    """Keep legacy stage1 curve keys structurally present for collate stability.

    New direct-label training no longer relies on these keys, but some packed
    samples still contain a subset of the old curve fields. Dataloader
    `default_collate` requires per-sample dict keys to match, so we inject
    harmless zero defaults when any legacy field is missing.
    """
    legacy_curve_keys = (
        'speed_risk_chase_values',
        'speed_risk_meet_values',
        'speed_risk_cross_yld_values',
        'speed_risk_cross_go_values',
        'speed_risk_junction_cross_yld_values',
        'speed_risk_junction_cross_go_values',
        'speed_risk_merge_yld_values',
        'speed_risk_merge_go_values',
        'speed_risk_borrow_yld_values',
        'speed_risk_borrow_go_values',
        'speed_risk_ped_values',
    )
    scalar_defaults = {
        'speed_cross_wait_time_s': 0.0,
        'speed_cross_wait_valid': 0.0,
        'borrow_cross_active_time_s': 0.0,
        'chase_max_speed': float('nan'),
        'chase_max_speed_valid': 0.0,
    }

    speed_samples = final_sample.get('speed_sample_values')
    if isinstance(speed_samples, torch.Tensor):
        curve_template = torch.zeros_like(speed_samples)
    else:
        curve_template = torch.zeros(7, dtype=torch.float32)
        final_sample.setdefault('speed_sample_values', curve_template.clone())

    for key in legacy_curve_keys:
        if key not in final_sample:
            final_sample[key] = curve_template.clone()

    for key, default_value in scalar_defaults.items():
        if key not in final_sample:
            final_sample[key] = torch.tensor(default_value, dtype=torch.float32)

    legacy_merge_float_defaults = {
        'merge_episode_active': 0.0,
        'merge_episode_no_go': 0.0,
    }
    legacy_merge_long_defaults = {
        'merge_decision_phase': 0,
        'merge_end_state': 0,
        'merge_episode_id': -1,
        'merge_episode_start_frame': -1,
        'merge_episode_end_frame': -1,
        'merge_go_frame': -1,
        'merge_resolution_actor_id': -1,
    }

    for key, default_value in legacy_merge_float_defaults.items():
        if key not in final_sample:
            final_sample[key] = torch.tensor(default_value, dtype=torch.float32)

    for key, default_value in legacy_merge_long_defaults.items():
        if key not in final_sample:
            final_sample[key] = torch.tensor(default_value, dtype=torch.long)


class RouteBatchSampler:
    """Batch sampler that groups samples by route to maximize route_features.pt cache hits.

    Each batch's samples come from the same or adjacent routes, so only 1-2 packs
    need to be loaded instead of ~batch_size packs with random shuffling.
    Route order is shuffled each epoch for training randomness.
    """
    def __init__(self, route_groups, batch_size, shuffle=True, drop_last=False):
        self.route_groups = route_groups  # list of list[int]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        groups = [list(g) for g in self.route_groups]
        if self.shuffle:
            random.shuffle(groups)
            for g in groups:
                random.shuffle(g)
        # Flatten: samples from the same route are consecutive
        all_indices = []
        for g in groups:
            all_indices.extend(g)
        # Yield consecutive batches
        batch = []
        for idx in all_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = sum(len(g) for g in self.route_groups)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class CARLAImageDataset(torch.utils.data.Dataset):
    
    
    def __init__(self,
                 dataset_path: str,
                 image_data_root: str,
                 mode: str = 'train',        # train or val
                 anchor_centers_abs: np.ndarray = None,  # (num_modes, num_points, 2)
                 semantic_behavior_cfg: dict = None,      # semantic behavior config
                 skip_memmap: bool = False,   # True for val: skip memmap, use inject_ram_features() later
                 use_per_frame: bool = False, # True for local SSD: load individual .pt files directly (no pack/memmap)
                 use_vqa_anchor: bool = False, # True to load VLM-predicted anchor from dp_vl_feature/*.pt
                 cache_dir: str = None,       # Override memmap cache dir (e.g. /tmp/tmp_data for tmpfs)
                 feature_suffix: str = '',    # Suffix for feature files (e.g. 'ensemble' → bev_features_fp16_ensemble.bin)
                 gps_noise_cfg: dict = None,  # GPS noise augmentation config
                 load_transfuser_lidar_bev: bool = False,  # Load LiDAR BEV detail input
                 lidar_history_frames: int = 1,
                 filter_bad_routes: bool = True,
                 retain_bad_routes_for_energy: bool = False,
                 load_exact_next_speed_online: bool = False,
                 next_speed_frame_offset: int = 2,
                 ):

        self.image_data_root = os.path.realpath(image_data_root)
        self.dataset_path = dataset_path
        self.mode = mode
        # Feature loading: memmap (shared across DDP ranks, zero-copy) or LRU fallback
        self._feat_mmap = None       # numpy memmap for bev_features
        self._ups_mmap = None        # numpy memmap for bev_upsamples
        self._feat_index = None      # dict: packed_path -> {offset, n_frames, frame_num_to_idx}
        # LRU fallback (used when memmap cache not built yet)
        self._route_pack_cache = {}
        self._route_pack_cache_maxsize = 32
        self._ram_features = None    # dict: abs_idx -> (feat_tensor, ups_tensor), set by preload_to_ram()
        self._use_per_frame = use_per_frame  # Local SSD mode: read individual .pt files
        self._use_vqa_anchor = use_vqa_anchor  # Load VLM anchor from dp_vl_feature
        self._load_transfuser_lidar_bev = load_transfuser_lidar_bev
        self._lidar_history_frames = max(int(lidar_history_frames), 1)
        self._filter_bad_routes = bool(filter_bad_routes)
        self._retain_bad_routes_for_energy = bool(retain_bad_routes_for_energy)
        self._load_exact_next_speed_online = bool(load_exact_next_speed_online)
        self._next_speed_frame_offset = max(int(next_speed_frame_offset), 1)
        self._lidar_bev_mmap = None     # numpy memmap for lidar_bev_fp16.bin
        self._lidar_bev_index = None    # dict: route_rel -> {offset, n_frames, frame_ids}
        self._route_speed_cache = {}
        self._route_speed_cache_maxsize = 64

        # Semantic behavior labeling
        self.anchor_centers_abs = anchor_centers_abs
        self.semantic_behavior_enabled = (
            anchor_centers_abs is not None
            and semantic_behavior_cfg is not None
            and semantic_behavior_cfg.get('enabled', False)
        )
        self.semantic_behavior_cfg = semantic_behavior_cfg or {}

        # GPS noise augmentation (train only)
        gps_cfg = gps_noise_cfg or {}
        self._gps_noise_enabled = gps_cfg.get('enabled', False)
        self._gps_noise_sigma = gps_cfg.get('sigma', 0.4)
        self._gps_noise_prob = gps_cfg.get('probability', 0.8)
        self._gps_noise_apply_to_route = gps_cfg.get('apply_to_route', True)

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 928)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.lidar_bev_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Processed data directory not found: {dataset_path}")

        # Detect DDP rank: only rank 0 does slow IO, others wait for packed file
        ddp_active = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if ddp_active else 0

        packed_path = os.path.join(dataset_path, 'samples_packed.pkl')

        if not os.path.exists(packed_path):
            if rank == 0:
                # Rank 0: load individual pkl files and save packed file
                train_files = glob.glob(os.path.join(dataset_path, "train", "*.pkl"))
                val_files = glob.glob(os.path.join(dataset_path, "val", "*.pkl"))
                direct_files = glob.glob(os.path.join(dataset_path, "*.pkl"))

                if train_files or val_files:
                    sample_files = sorted(train_files + val_files)
                    print(f"Found {len(sample_files)} preprocessed samples in '{dataset_path}' "
                          f"({len(train_files)} train, {len(val_files)} val).")
                elif direct_files:
                    sample_files = sorted(direct_files)
                    print(f"Found {len(sample_files)} preprocessed samples in '{dataset_path}'.")
                else:
                    raise FileNotFoundError(f"No pkl files found in {dataset_path} or its train/val subdirectories.")

                print(f"[Rank 0] Preloading {len(sample_files)} pkl files into memory...")
                cache = [None] * len(sample_files)
                for i, path in enumerate(tqdm(sample_files, desc="Loading pkl", leave=False)):
                    with open(path, 'rb') as f:
                        cache[i] = pickle.load(f)

                # Save packed file for all ranks (atomic write)
                tmp_path = packed_path + f'.tmp.{os.getpid()}'
                print(f"[Rank 0] Saving packed samples to {packed_path}...")
                with open(tmp_path, 'wb') as f:
                    pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.rename(tmp_path, packed_path)
                print(f"[Rank 0] Packed file saved ({os.path.getsize(packed_path) / 1e6:.1f} MB).")
                del cache
            else:
                # Other ranks: wait for rank 0 to create packed file (no NCCL timeout issue)
                print(f"[Rank {rank}] Waiting for rank 0 to create {packed_path}...")
                while not os.path.exists(packed_path):
                    time.sleep(5)
                # Small delay to ensure rank 0 has finished rename
                time.sleep(2)

        # All ranks load from packed file (single large sequential read, fast on Lustre)
        print(f"[Rank {rank}] Loading packed samples from {packed_path}...")
        with open(packed_path, 'rb') as f:
            all_samples = pickle.load(f)

        # Build route_name -> event_name mapping from disk (event_name is None in old pkl)
        route_name_to_event = {}
        for entry in os.scandir(image_data_root):
            if not entry.is_dir() or entry.name == 'tmp_data':
                continue
            for route_entry in os.scandir(entry.path):
                if route_entry.is_dir():
                    route_name_to_event[route_entry.name] = entry.name
        print(f"[Rank {rank}] Scanned {len(route_name_to_event)} routes on disk.")

        # Build set of routes that have route_features.pt
        route_has_features = set()
        for rn, ev in route_name_to_event.items():
            feat_pt = os.path.join(image_data_root, ev, rn, 'transfuser_feature', 'route_features.pt')
            if os.path.exists(feat_pt):
                route_has_features.add(rn)
        print(f"[Rank {rank}] Routes with route_features.pt: {len(route_has_features)}/{len(route_name_to_event)}")

        # Load bad routes exclude list if it exists
        bad_routes_path = os.path.join(image_data_root, 'bad_routes.txt')
        bad_routes_set = set()
        if os.path.exists(bad_routes_path):
            with open(bad_routes_path) as f:
                bad_routes_set = {line.strip() for line in f if line.strip()}
            if rank == 0:
                print(f"[Rank {rank}] Loaded {len(bad_routes_set)} bad routes from {bad_routes_path}")

        # Patch transfuser_bev_feature path and filter samples
        before_count = len(all_samples)
        patched = 0
        missing_routes = set()
        bad_route_dropped = 0
        self._sample_cache = []
        for s in all_samples:
            route = s.get('route_name', '')
            fid = s.get('frame_id', None)

            # Skip if route has no route_features.pt
            if route not in route_has_features:
                missing_routes.add(route)
                continue

            # Skip or retain bad routes depending on training objective.
            if route in bad_routes_set:
                if self._retain_bad_routes_for_energy:
                    s = dict(s)
                    s['is_bad_route'] = True
                elif self._filter_bad_routes:
                    bad_route_dropped += 1
                    continue

            if 'is_bad_route' not in s:
                s = dict(s)
                s['is_bad_route'] = False

            # Derive transfuser_bev_feature path if missing
            if not s.get('transfuser_bev_feature', ''):
                if fid is not None:
                    event = route_name_to_event[route]
                    s['transfuser_bev_feature'] = os.path.join(
                        event, route, 'transfuser_feature', f'{int(fid):04d}_feature.pt')
                    patched += 1
                else:
                    continue

            self._sample_cache.append(s)

        dropped = before_count - len(self._sample_cache)
        del all_samples
        if patched > 0:
            print(f"[Rank {rank}] Patched {patched} samples with derived transfuser_bev_feature path.")
        if bad_route_dropped > 0:
            print(f"[Rank {rank}] Filtered {bad_route_dropped} samples from {len(bad_routes_set)} bad routes (collisions/crashes).")
        if dropped > 0:
            print(f"[Rank {rank}] WARNING: Dropped {dropped}/{before_count} samples "
                  f"({len(missing_routes)} routes missing route_features.pt).")
            if len(missing_routes) <= 20:
                for r in sorted(missing_routes):
                    print(f"  [Rank {rank}]   missing: {r}")
            else:
                for r in sorted(missing_routes)[:10]:
                    print(f"  [Rank {rank}]   missing: {r}")
                print(f"  [Rank {rank}]   ... and {len(missing_routes) - 10} more")

        self.sample_files = list(range(len(self._sample_cache)))
        print(f"[Rank {rank}] Loaded {len(self._sample_cache)} samples from packed file.")

        # Build route groups: group sample indices by route directory for batch sampling
        route_to_indices = defaultdict(list)
        for i, sample in enumerate(self._sample_cache):
            feat_rel = sample.get('transfuser_bev_feature', '')
            route_key = os.path.dirname(feat_rel)  # e.g. "Accident/Town12_.../transfuser_feature"
            route_to_indices[route_key].append(i)
        self._route_groups = list(route_to_indices.values())
        print(f"Grouped into {len(self._route_groups)} routes for batch sampling.")

        # ===== Load feature cache (memmap, shared across DDP ranks) =====
        if cache_dir is None:
            cache_dir = os.path.join(image_data_root, 'tmp_data')
        sfx = f'_{feature_suffix}' if feature_suffix else ''
        index_path = os.path.join(cache_dir, f'feature_index{sfx}.pkl')
        feat_bin = os.path.join(cache_dir, f'bev_features_fp16{sfx}.bin')
        ups_bin = os.path.join(cache_dir, f'bev_upsamples_fp16{sfx}.bin')

        if skip_memmap:
            print(f"[Rank {rank}] Skipping memmap (will use inject_ram_features later).")
        elif os.path.exists(index_path) and os.path.exists(feat_bin) and os.path.exists(ups_bin):
            with open(index_path, 'rb') as f:
                cache_meta = pickle.load(f)
            self._feat_index = cache_meta['index']
            self._feat_mmap = np.memmap(feat_bin, dtype=np.float16, mode='r',
                                        shape=tuple(cache_meta['bev_feat_shape']))
            self._ups_mmap = np.memmap(ups_bin, dtype=np.float16, mode='r',
                                       shape=tuple(cache_meta['bev_ups_shape']))
            print(f"[Rank {rank}] Feature memmap loaded: {len(self._feat_index)} routes, "
                  f"{cache_meta['total_frames']} frames (shared across ranks).")
        else:
            print(f"[Rank {rank}] WARNING: Feature memmap cache not found. "
                  f"Using LRU fallback (slow). Run: python scripts/data_tools/build_feature_cache_fp16.py")

        # ===== Load LiDAR BEV memmap (if enabled and available) =====
        if self._load_transfuser_lidar_bev and not skip_memmap:
            lidar_bin = os.path.join(cache_dir, 'lidar_bev_fp16.bin')
            lidar_idx = os.path.join(cache_dir, 'lidar_bev_index.pkl')
            if os.path.exists(lidar_bin) and os.path.exists(lidar_idx):
                with open(lidar_idx, 'rb') as f:
                    lidar_meta = pickle.load(f)
                self._lidar_bev_index = lidar_meta['index']
                self._lidar_bev_mmap = np.memmap(
                    lidar_bin, dtype=np.float16, mode='r',
                    shape=tuple(lidar_meta['shape']))
                print(f"[Rank {rank}] LiDAR BEV memmap loaded: "
                      f"{len(self._lidar_bev_index)} routes, "
                      f"{lidar_meta['total_frames']} frames "
                      f"(inverted={lidar_meta.get('inverted', False)}).")
            else:
                print(f"[Rank {rank}] WARNING: LiDAR BEV memmap not found. "
                      f"Falling back to per-frame .npy loading. "
                      f"Run: python scripts/data_tools/build_lidar_bev_cache.py")

        # ===== Pre-load VQA anchors into RAM (tiny: ~48 bytes each) =====
        self._vqa_anchor_cache = {}  # sample_idx -> tensor (6, 2)
        if self._use_vqa_anchor:
            loaded = 0
            for i, s in enumerate(tqdm(self._sample_cache, desc="Loading VQA anchors", disable=(rank != 0))):
                feat_rel = s.get('transfuser_bev_feature', '')
                if not feat_rel:
                    continue
                base_dir = os.path.dirname(os.path.dirname(feat_rel))
                frame_str = os.path.basename(feat_rel).replace('_feature.pt', '')
                vqa_path = os.path.join(image_data_root, base_dir, 'dp_vl_feature', f'{frame_str}.pt')
                if os.path.exists(vqa_path):
                    vf = torch.load(vqa_path, weights_only=True)
                    if 'pred_traj' in vf:
                        anchor = vf['pred_traj']
                        if anchor.dim() == 3:
                            anchor = anchor.squeeze(0)
                        self._vqa_anchor_cache[i] = anchor[:6].float()
                        loaded += 1
            print(f"[Rank {rank}] Pre-loaded {loaded}/{len(self._sample_cache)} VQA anchors into RAM.")

    def get_route_batch_sampler(self, batch_size, shuffle=True, drop_last=False):
        """Return a RouteBatchSampler for this dataset."""
        return RouteBatchSampler(self._route_groups, batch_size, shuffle=shuffle, drop_last=drop_last)

    def __len__(self):
        return len(self.sample_files)

    def inject_ram_features(self, train_dataset, rank=0, world_size=1, max_val_samples=None, sample_indices=None):
        """Pre-load val features into RAM using train dataset's memmap.
        Only loads the samples this rank will access via DistributedSampler(shuffle=False).
        Two-pass: first collect needed abs_idx, then read sorted (sequential IO).
        Args:
            train_dataset: CARLAImageDataset with loaded memmap
            rank: DDP rank (determines which sample indices this rank gets)
            world_size: total DDP ranks
            max_val_samples: max samples per rank (max_batches * batch_size), None = all
            sample_indices: optional explicit dataset indices to preload for this rank.
                When provided, bypasses the default rank/world_size slicing logic.
        """
        if train_dataset._feat_mmap is None or train_dataset._feat_index is None:
            print(f"[Rank {rank}] inject_ram_features: train has no memmap, skipping.")
            return

        # Compute which sample indices this rank will access
        # DistributedSampler(shuffle=False): rank k gets indices [k, k+W, k+2W, ...]
        if sample_indices is not None:
            my_indices = list(sample_indices)
        else:
            n_dataset = len(self.sample_files)
            my_indices = list(range(rank, n_dataset, world_size))
        if max_val_samples is not None:
            my_indices = my_indices[:max_val_samples]

        # Pass 1: collect sample_idx -> abs_idx mapping (no IO)
        sidx_to_abs = {}
        for idx in my_indices:
            sample = self._sample_cache[idx]
            if 'transfuser_bev_feature' not in sample:
                continue
            bev_feature_path = os.path.join(self.image_data_root, sample['transfuser_bev_feature'])
            packed_path = os.path.join(os.path.dirname(bev_feature_path), 'route_features.pt')
            frame_id = sample.get('frame_id')
            route_info = train_dataset._feat_index.get(packed_path)
            if route_info is None or frame_id is None:
                continue
            n_frames = route_info.get('n_frames', len(route_info['frame_num_to_idx']))
            if frame_id >= n_frames:
                continue
            sidx_to_abs[idx] = route_info['offset'] + frame_id

        # Pass 2: read unique abs_idx in sorted order (sequential memmap access = fast IO)
        unique_abs = sorted(set(sidx_to_abs.values()))
        ram = {}
        desc = f"[Rank {rank}] Preloading val features to RAM"
        for abs_idx in tqdm(unique_abs, desc=desc, disable=(rank != 0)):
            feat = torch.from_numpy(train_dataset._feat_mmap[abs_idx].copy()).clone()
            ups = torch.from_numpy(train_dataset._ups_mmap[abs_idx].copy()).clone()
            ram[abs_idx] = (feat, ups)

        # Add sample_idx -> abs_idx mappings
        for sidx, abs_idx in sidx_to_abs.items():
            ram[('sidx', sidx)] = abs_idx

        self._ram_features = ram
        mem_mb = sum(v[0].nbytes + v[1].nbytes for v in ram.values() if isinstance(v, tuple)) / 1e6
        print(f"[Rank {rank}] inject_ram_features: {len(unique_abs)} unique frames "
              f"({len(sidx_to_abs)} samples) -> {mem_mb:.0f} MB in RAM")

    def _get_route_pack_lru(self, packed_path: str) -> dict:
        """Fallback: load route_features.pt into LRU cache (used when memmap not available)."""
        if packed_path not in self._route_pack_cache:
            if len(self._route_pack_cache) >= self._route_pack_cache_maxsize:
                self._route_pack_cache.pop(next(iter(self._route_pack_cache)))
            pack = torch.load(packed_path, weights_only=True)
            self._route_pack_cache[packed_path] = {
                'frame_num_to_idx': {fn: i for i, fn in enumerate(pack['frame_nums'])},
                'bev_features': pack['bev_features'].half(),
                'bev_upsamples': pack['bev_upsamples'].half(),
            }
        return self._route_pack_cache[packed_path]

    def _load_single_lidar_bev_frame(self, route_rel: str, frame_id: int, feat_rel: str) -> Optional[torch.Tensor]:
        if frame_id is None:
            return None

        if self._lidar_bev_mmap is not None and self._lidar_bev_index is not None:
            route_info = self._lidar_bev_index.get(route_rel)
            if route_info is not None:
                local_idx = route_info.get('fid_to_local', {}).get(int(frame_id))
                if local_idx is not None:
                    abs_idx = route_info['offset'] + local_idx
                    return torch.from_numpy(self._lidar_bev_mmap[abs_idx].copy())  # (2, 256, 256) fp16
            return None

        lidar_bev_rel = feat_rel.replace(
            'transfuser_feature/', 'transfuser_lidar_bev/').replace(
            '_feature.pt', '.npy')
        cur_frame = os.path.basename(lidar_bev_rel).replace('.npy', '')
        hist_frame = f'{int(frame_id):04d}'
        lidar_bev_rel = lidar_bev_rel.replace(f'/{cur_frame}.npy', f'/{hist_frame}.npy')
        lidar_bev_path = os.path.join(self.image_data_root, lidar_bev_rel)
        if not os.path.exists(lidar_bev_path):
            return None
        raw = np.load(lidar_bev_path)  # (2, 256, 256) float16
        return torch.from_numpy((1.0 - raw.astype(np.float32))).half()

    def _load_lidar_bev_history(self, feat_rel: str, route_rel: str, frame_id: int) -> Optional[torch.Tensor]:
        if frame_id is None:
            return None

        frames = []
        start_frame = int(frame_id) - (self._lidar_history_frames - 1)
        for hist_frame_id in range(start_frame, int(frame_id) + 1):
            frame_tensor = self._load_single_lidar_bev_frame(route_rel, hist_frame_id, feat_rel)
            if frame_tensor is None:
                frame_tensor = torch.zeros(2, 256, 256, dtype=torch.float16)
            frames.append(frame_tensor)

        if not frames:
            return None
        return torch.stack(frames, dim=0)  # (H_hist, 2, 256, 256)

    def _get_route_rel_from_sample(self, sample: dict) -> Optional[str]:
        feat_rel = sample.get('transfuser_bev_feature', '')
        if isinstance(feat_rel, str) and feat_rel:
            return os.path.dirname(os.path.dirname(feat_rel))
        route_name = sample.get('route_name')
        town_name = sample.get('town_name')
        if route_name and town_name:
            return os.path.join(str(town_name), str(route_name))
        return None

    def _get_route_speed_map(self, route_rel: str) -> dict:
        cached = self._route_speed_cache.get(route_rel)
        if cached is not None:
            return cached

        if len(self._route_speed_cache) >= self._route_speed_cache_maxsize:
            self._route_speed_cache.pop(next(iter(self._route_speed_cache)))

        measurements_dir = os.path.join(self.image_data_root, route_rel, 'measurements')
        speed_map = {}
        if os.path.isdir(measurements_dir):
            try:
                for fname in sorted(os.listdir(measurements_dir)):
                    if not fname.endswith('.json.gz'):
                        continue
                    try:
                        frame_id = int(fname.split('.')[0])
                    except ValueError:
                        continue
                    try:
                        with gzip.open(os.path.join(measurements_dir, fname), 'rt', encoding='utf-8') as gz_file:
                            measurement = json.load(gz_file)
                        speed_map[frame_id] = float(measurement.get('speed', 0.0))
                    except Exception:
                        continue
            except Exception:
                speed_map = {}

        self._route_speed_cache[route_rel] = speed_map
        return speed_map

    def _get_exact_next_speed_target(self, sample: dict) -> Optional[float]:
        frame_id = sample.get('frame_id')
        if frame_id is None:
            return None
        route_rel = self._get_route_rel_from_sample(sample)
        if not route_rel:
            return None
        speed_map = self._get_route_speed_map(route_rel)
        target_frame_id = int(frame_id) + self._next_speed_frame_offset
        target_speed = speed_map.get(target_frame_id)
        if target_speed is None:
            return None
        return float(target_speed)

    def __getitem__(self, idx):
        sample = self._sample_cache[idx]
        clone_keys = set()

        def _from_numpy(value, key, dtype=torch.float32):
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
            else:
                tensor = torch.as_tensor(value)
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            clone_keys.add(key)
            return tensor

        # --- Load Transfuser Features ---
        # Memmap path: zero-copy from shared memory (all DDP ranks share same pages)
        # LRU fallback: per-worker cache (slow, only if memmap cache not built)
        transfuser_bev_feature = None
        transfuser_bev_feature_upsample = None

        if 'transfuser_bev_feature' in sample:
            bev_feature_path = os.path.join(self.image_data_root, sample['transfuser_bev_feature'])
            packed_path = os.path.join(os.path.dirname(bev_feature_path), 'route_features.pt')
            frame_id = sample.get('frame_id')

            if self._use_per_frame:
                # Local SSD mode: load individual per-frame .pt files directly
                # e.g. "0010_feature.pt" -> (1, 1512, 8, 8), "0010_feature_upsample.pt" -> (1, 64, 64, 64)
                feat_path = bev_feature_path  # already points to {frame_id}_feature.pt
                ups_path = bev_feature_path.replace('_feature.pt', '_feature_upsample.pt')
                if os.path.exists(feat_path):
                    transfuser_bev_feature = torch.load(feat_path, weights_only=True).squeeze(0).half()
                if os.path.exists(ups_path):
                    transfuser_bev_feature_upsample = torch.load(ups_path, weights_only=True).squeeze(0).half()
            elif self._ram_features is not None:
                # RAM pre-loaded path (val dataset): pure memory lookup, zero disk IO
                abs_idx = self._ram_features.get(('sidx', idx))
                if abs_idx is not None:
                    cached = self._ram_features.get(abs_idx)
                    if cached is not None:
                        transfuser_bev_feature = cached[0]
                        ups_ds = cached[1]
                        transfuser_bev_feature_upsample = F.interpolate(
                            ups_ds.unsqueeze(0).float(), size=(64, 64),
                            mode='bilinear', align_corners=False).squeeze(0).half()
            elif self._feat_index is not None:
                # Fast path: memmap (zero IO after pages are faulted in)
                route_info = self._feat_index.get(packed_path)
                if route_info is not None:
                    n_frames = route_info.get('n_frames', len(route_info['frame_num_to_idx']))
                    if frame_id is not None and frame_id < n_frames:
                        abs_idx = route_info['offset'] + frame_id
                        transfuser_bev_feature = torch.from_numpy(
                            self._feat_mmap[abs_idx].copy())  # (1512, 8, 8) float16
                        clone_keys.add('transfuser_bev_feature')
                        # Stored as (64, 32, 32) after 2x downsample, interpolate back
                        ups_ds = torch.from_numpy(
                            self._ups_mmap[abs_idx].copy())           # (64, 32, 32) float16
                        transfuser_bev_feature_upsample = F.interpolate(
                            ups_ds.unsqueeze(0).float(), size=(64, 64),
                            mode='bilinear', align_corners=False).squeeze(0).half()
                    else:
                        import warnings
                        warnings.warn(
                            f"[Dataset] frame_id {frame_id} out of range "
                            f"(n_frames={n_frames}) for {packed_path}",
                            stacklevel=2)
                else:
                    import warnings
                    warnings.warn(
                        f"[Dataset] route not in memmap index: {packed_path}",
                        stacklevel=2)
            else:
                # Slow fallback: LRU cache with disk IO
                pack = self._get_route_pack_lru(packed_path)
                n_frames = len(pack['bev_features'])
                if frame_id is not None and frame_id < n_frames:
                    transfuser_bev_feature = pack['bev_features'][frame_id]
                    transfuser_bev_feature_upsample = pack['bev_upsamples'][frame_id]
                else:
                    import warnings
                    warnings.warn(
                        f"[Dataset] frame_id {frame_id} out of range "
                        f"(n_frames={n_frames}) for {packed_path}",
                        stacklevel=2)
        
        # Load VQA anchor from pre-loaded RAM cache (zero IO)
        vqa_anchor_cached = None
        if self._use_vqa_anchor:
            vqa_anchor_cached = self._vqa_anchor_cache.get(idx)  # (6, 2) or None

        # Convert sample data
        final_sample = dict()
        next_speed_target_mps = sample.get('next_speed_target_mps')
        if next_speed_target_mps is None and self._load_exact_next_speed_online:
            next_speed_target_mps = self._get_exact_next_speed_target(sample)
        for key, value in sample.items():
            if key == 'rgb_hist_jpg':
                continue
            elif key == 'stage1_speed_debug':
                # Debug payload has variable nested keys across frames/routes and is
                # only meant for inspection/video replay, not training batches.
                continue
            elif key == 'next_speed_target_mps':
                if value is not None:
                    final_sample['next_speed_target_mps'] = torch.tensor(float(value), dtype=torch.float32)
            elif key == 'is_bad_route':
                final_sample['is_bad_route'] = torch.tensor(bool(value), dtype=torch.bool)
            elif key == 'speed_hist':
                speed_data = sample['speed_hist']
                final_sample['speed'] = _from_numpy(speed_data, 'speed')
            elif key == 'ego_waypoints':
                ego_waypoints = _from_numpy(sample['ego_waypoints'][1:], 'agent_pos')
                final_sample['agent_pos'] = ego_waypoints
            elif key == 'vqa':
                if self._use_vqa_anchor and vqa_anchor_cached is not None:
                    final_sample['vqa_anchor'] = vqa_anchor_cached.clone()
                continue
            elif key == 'route':
                # Load route waypoints (expected shape: (20, 2))
                route_data = _from_numpy(value, 'route')
                final_sample['route'] = route_data
            elif key == 'target_point_hist':
                # Two data formats exist:
                #   Old HPC packed: (T, 4) = [target_point, target_point_next] concatenated
                #   Standard:       (T, 2) = target_point only (target_point_next is separate key)
                tp = _from_numpy(value, 'target_point_hist')
                final_sample['target_point_hist'] = tp[..., :2]
                if tp.shape[-1] == 4:
                    clone_keys.add('target_point_next_hist')
                    final_sample['target_point_next_hist'] = tp[..., 2:]
            elif key == 'target_point_next_hist':
                final_sample['target_point_next_hist'] = _from_numpy(value, 'target_point_next_hist')
            elif key == 'ego_status':
                final_sample['ego_status'] = _from_numpy(value, 'ego_status')
            elif key == 'energy_targets':
                final_sample['energy_targets'] = _from_numpy(value, 'energy_targets')
            elif key == 'energy_active_mask':
                final_sample['energy_active_mask'] = _from_numpy(
                    value.astype(np.bool_) if isinstance(value, np.ndarray) else value,
                    'energy_active_mask',
                    dtype=None,
                ).bool()
            elif key == 'speed_sample_valid_mask':
                final_sample['speed_sample_valid_mask'] = _from_numpy(value, 'speed_sample_valid_mask')
            elif key == 'speed_sample_exp_index':
                if isinstance(value, np.ndarray):
                    final_sample['speed_sample_exp_index'] = torch.from_numpy(value).long()
                else:
                    final_sample['speed_sample_exp_index'] = torch.tensor(value).long()
            elif key in {
                'speed_sample_values',
                'speed_risk_chase_values',
                'speed_risk_meet_values',
                'speed_risk_cross_yld_values',
                'speed_risk_cross_go_values',
                'speed_risk_junction_cross_yld_values',
                'speed_risk_junction_cross_go_values',
                'speed_risk_merge_yld_values',
                'speed_risk_merge_go_values',
                'speed_risk_borrow_yld_values',
                'speed_risk_borrow_go_values',
                'speed_risk_ped_values',
                'speed_cross_wait_time_s',
                'speed_cross_wait_valid',
                'borrow_cross_active_time_s',
                'chase_max_speed',
                'chase_max_speed_valid',
                'merge_yld_max_speed',
                'merge_go_min_speed',
                'merge_yld_max_speed_valid',
                'merge_go_min_speed_valid',
                'borrow_yld_max_speed',
                'borrow_go_min_speed',
                'borrow_yld_max_speed_valid',
                'borrow_go_min_speed_valid',
                'junction_yld_max_speed',
                'junction_go_min_speed',
                'junction_yld_max_speed_valid',
                'junction_go_min_speed_valid',
                'merge_threshold_train_only_negative_tail',
            }:
                final_sample[key] = _from_numpy(value, key)
            elif key in {
                'conflict_area_family',
                'conflict_area_dir',
                'conflict_decision_phase',
                'conflict_control_phase',
                'conflict_go_frame',
                'conflict_area_start_frame',
                'conflict_area_end_frame',
            }:
                if isinstance(value, np.ndarray):
                    final_sample[key] = torch.from_numpy(value).long()
                else:
                    final_sample[key] = torch.tensor(value).long()
            elif key == 'conflict_area_active':
                final_sample[key] = torch.as_tensor(value, dtype=torch.float32)
            elif key in {
                'merge_active',
                'merge_episode_active',
                'cross_active',
                'cross_episode_active',
                'junction_cross_episode_active',
                'borrow_cross_episode_active',
            }:
                final_sample[key] = torch.as_tensor(value, dtype=torch.float32)
            elif key.startswith('transfuser_'):
                # Skip transfuser paths, we already loaded them as tensors
                continue
            elif value is None:
                # Skip None values to avoid DataLoader collate errors
                continue
            elif isinstance(value, np.ndarray):
                final_sample[key] = _from_numpy(value, key)
            elif isinstance(value, np.generic):
                if np.issubdtype(value.dtype, np.bool_):
                    final_sample[key] = torch.tensor(bool(value), dtype=torch.bool)
                elif np.issubdtype(value.dtype, np.integer):
                    final_sample[key] = torch.tensor(int(value), dtype=torch.long)
                elif np.issubdtype(value.dtype, np.floating):
                    final_sample[key] = torch.tensor(float(value), dtype=torch.float32)
                else:
                    final_sample[key] = value.item()
            else:
                final_sample[key] = value

        if next_speed_target_mps is not None and 'next_speed_target_mps' not in final_sample:
            final_sample['next_speed_target_mps'] = torch.tensor(next_speed_target_mps, dtype=torch.float32)

        # Ensure target_point_next_hist always exists (fallback to target_point_hist)
        if 'target_point_next_hist' not in final_sample:
            final_sample['target_point_next_hist'] = final_sample['target_point_hist'].clone()

        # Set vqa_anchor from RAM cache (or fallback to zeros)
        if self._use_vqa_anchor:
            if vqa_anchor_cached is not None:
                final_sample['vqa_anchor'] = vqa_anchor_cached.clone()
            elif 'vqa_anchor' not in final_sample:
                final_sample['vqa_anchor'] = torch.zeros(6, 2)

        _apply_stage1_near_zero_speed_snap(final_sample)
        _ensure_stage1_legacy_curve_defaults(final_sample)

        # Add transfuser features to final_sample
        # Following DiffusionDriveV2: only use bev_feature and bev_feature_upsample
        # Always include keys to avoid KeyError in collate when batch has mixed samples
        if transfuser_bev_feature is not None:
            final_sample['transfuser_bev_feature'] = transfuser_bev_feature
        else:
            final_sample['transfuser_bev_feature'] = torch.zeros(1512, 8, 8, dtype=torch.float16)
        if transfuser_bev_feature_upsample is not None:
            final_sample['transfuser_bev_feature_upsample'] = transfuser_bev_feature_upsample
        else:
            final_sample['transfuser_bev_feature_upsample'] = torch.zeros(64, 64, 64, dtype=torch.float16)

        # --- Load TransFuser LiDAR BEV histogram history (H_hist, 2, 256, 256) ---
        transfuser_lidar_bev = None
        if self._load_transfuser_lidar_bev and 'transfuser_bev_feature' in sample:
            feat_rel = sample['transfuser_bev_feature']
            # Derive route_rel and frame_id from feat_rel
            # e.g. "Accident/Town12_.../transfuser_feature/0006_feature.pt"
            #   route_rel = "Accident/Town12_..."
            #   frame_id  = 6
            route_rel = os.path.dirname(os.path.dirname(feat_rel))
            frame_id = sample.get('frame_id')
            transfuser_lidar_bev = self._load_lidar_bev_history(feat_rel, route_rel, frame_id)
        if transfuser_lidar_bev is not None:
            final_sample['transfuser_lidar_bev'] = transfuser_lidar_bev
        else:
            final_sample['transfuser_lidar_bev'] = torch.zeros(
                self._lidar_history_frames, 2, 256, 256, dtype=torch.float16)

        # ========== Semantic Behavior Labeling ==========
        if self.semantic_behavior_enabled:
            # Fast path: use pre-computed labels from samples_packed.pkl
            if 'behavior_labels' in sample and 'allowed_flags' in sample:
                bl = sample['behavior_labels']
                af = sample['allowed_flags']
                final_sample['behavior_labels'] = torch.from_numpy(bl).long() if isinstance(bl, np.ndarray) else torch.tensor(bl).long()
                final_sample['allowed_flags'] = torch.from_numpy(af).float() if isinstance(af, np.ndarray) else torch.tensor(af).float()
                if 'scene_buckets' in sample:
                    sb = sample['scene_buckets']
                    final_sample['scene_buckets'] = torch.from_numpy(sb).float() if isinstance(sb, np.ndarray) else torch.tensor(sb).float()
                else:
                    from tools.anchor_semantic_labeler import NUM_BUCKET_CATEGORIES
                    final_sample['scene_buckets'] = torch.zeros(NUM_BUCKET_CATEGORIES, dtype=torch.float32)
            else:
                # Slow path: on-the-fly computation (fallback if not pre-computed)
                from tools.anchor_semantic_labeler import label_anchors_semantic, classify_scene_buckets
                import json, gzip

                feature_rel = sample.get('transfuser_bev_feature', '')
                base_dir = os.path.dirname(os.path.dirname(feature_rel))
                frame_str = os.path.basename(feature_rel).replace('_feature.pt', '')

                bev_rel = feature_rel.replace('transfuser_feature/', 'bev_semantics/').replace('_feature.pt', '.png')
                bev_path = os.path.join(self.image_data_root, bev_rel)

                if os.path.exists(bev_path):
                    bev_semantic = np.array(Image.open(bev_path))

                    boxes = None
                    boxes_rel = feature_rel.replace('transfuser_feature/', 'boxes/').replace('_feature.pt', '.json.gz')
                    boxes_path = os.path.join(self.image_data_root, boxes_rel)
                    if os.path.exists(boxes_path):
                        try:
                            with gzip.open(boxes_path, 'rt') as bf:
                                boxes = json.load(bf)
                        except Exception:
                            boxes = None

                    measurements = None
                    meas_rel = feature_rel.replace('transfuser_feature/', 'measurements/').replace('_feature.pt', '.json.gz')
                    meas_path = os.path.join(self.image_data_root, meas_rel)
                    if os.path.exists(meas_path):
                        try:
                            with gzip.open(meas_path, 'rt') as mf:
                                measurements = json.load(mf)
                        except Exception:
                            measurements = None

                    ego_matrix_current = None
                    future_frames_data = None
                    if measurements is not None:
                        ego_matrix_current = measurements.get('ego_matrix', None)

                    if ego_matrix_current is not None:
                        frame_id = int(frame_str)
                        num_points = self.anchor_centers_abs.shape[1]
                        future_frames_data = []
                        for k in range(1, num_points + 1):
                            future_frame_str = f"{frame_id + k:04d}"
                            fut_boxes_path = os.path.join(
                                self.image_data_root, base_dir,
                                'boxes', f'{future_frame_str}.json.gz')
                            fut_meas_path = os.path.join(
                                self.image_data_root, base_dir,
                                'measurements', f'{future_frame_str}.json.gz')
                            if os.path.exists(fut_boxes_path) and os.path.exists(fut_meas_path):
                                try:
                                    with gzip.open(fut_boxes_path, 'rt') as bf:
                                        fut_boxes = json.load(bf)
                                    with gzip.open(fut_meas_path, 'rt') as mf:
                                        fut_meas = json.load(mf)
                                    fut_ego_matrix = fut_meas.get('ego_matrix', None)
                                    if fut_ego_matrix is not None:
                                        future_frames_data.append((fut_boxes, fut_ego_matrix))
                                    else:
                                        future_frames_data.append(None)
                                except Exception:
                                    future_frames_data.append(None)
                            else:
                                future_frames_data.append(None)

                    gt_traj = sample.get('ego_waypoints', None)
                    if gt_traj is not None:
                        gt_traj = gt_traj[1:]

                    behavior_labels, allowed_flags, _ = label_anchors_semantic(
                        self.anchor_centers_abs, bev_semantic,
                        ppm=self.semantic_behavior_cfg.get('bev_ppm', 2.0),
                        bev_size=self.semantic_behavior_cfg.get('bev_size', 256),
                        boxes=boxes,
                        measurements=measurements,
                        ego_matrix_current=ego_matrix_current,
                        future_frames_data=future_frames_data,
                        gt_trajectory=gt_traj,
                    )
                    final_sample['behavior_labels'] = torch.from_numpy(behavior_labels).long()
                    final_sample['allowed_flags'] = torch.from_numpy(allowed_flags.astype(np.float32))

                    bucket_flags = classify_scene_buckets(
                        measurements=measurements,
                        boxes=boxes,
                        ego_waypoints=gt_traj,
                    )
                    final_sample['scene_buckets'] = torch.from_numpy(bucket_flags.astype(np.float32))
                else:
                    n_modes = self.anchor_centers_abs.shape[0]
                    final_sample['behavior_labels'] = torch.zeros(n_modes, dtype=torch.long)
                    final_sample['allowed_flags'] = torch.ones(n_modes, dtype=torch.float32)
                    from tools.anchor_semantic_labeler import NUM_BUCKET_CATEGORIES
                    final_sample['scene_buckets'] = torch.zeros(NUM_BUCKET_CATEGORIES, dtype=torch.float32)

        # ========== GPS Noise Augmentation ==========
        gps_noise_applied = False
        if self._gps_noise_enabled and self.mode == 'train':
            if random.random() < self._gps_noise_prob:
                sigma = self._gps_noise_sigma
                obs_horizon = final_sample['target_point_hist'].shape[0]
                frame_noise = torch.randn(obs_horizon, 2) * sigma

                # target_point and next_target_point share same noise per frame (same GPS reading)
                final_sample['target_point_hist'] = final_sample['target_point_hist'] + frame_noise
                final_sample['target_point_next_hist'] = final_sample['target_point_next_hist'] + frame_noise

                # waypoints_hist: relative displacement between frames
                # noise = epsilon_t - epsilon_current (current frame cancels to 0)
                current_noise = frame_noise[-1:]  # (1, 2)
                final_sample['waypoints_hist'] = final_sample['waypoints_hist'] + (frame_noise - current_noise)

                # route is expressed in the current ego frame, so under the current
                # target-point noise convention it should receive the same current-frame
                # offset as the latest target point.
                if self._gps_noise_apply_to_route and 'route' in final_sample:
                    final_sample['route'] = final_sample['route'] + current_noise
                gps_noise_applied = True

        # Build or update ego_status: [speed | theta | command | tp | tp_next | waypoints]
        if 'ego_status' in final_sample:
            if gps_noise_applied:
                if 'ego_status' in clone_keys:
                    final_sample['ego_status'] = final_sample['ego_status'].clone()
                    clone_keys.discard('ego_status')
                final_sample['ego_status'][..., 8:10] = final_sample['target_point_hist']
                final_sample['ego_status'][..., 10:12] = final_sample['target_point_next_hist']
                final_sample['ego_status'][..., 12:14] = final_sample['waypoints_hist']
        else:
            ego_status_components = []

            speed_data = final_sample.get('speed', final_sample.get('speed_hist'))
            if speed_data is None:
                raise KeyError("Neither 'speed' nor 'speed_hist' found in sample")
            ego_status_components.append(speed_data.unsqueeze(-1))  # (obs_horizon, 1)

            theta_data = final_sample['theta_hist']
            ego_status_components.append(theta_data.unsqueeze(-1))  # (obs_horizon, 1)

            command_data = final_sample['command_hist']
            ego_status_components.append(command_data)  # (obs_horizon, 6)

            target_point_data = final_sample['target_point_hist']
            ego_status_components.append(target_point_data)  # (obs_horizon, 2)

            target_point_next_data = final_sample['target_point_next_hist']
            ego_status_components.append(target_point_next_data)  # (obs_horizon, 2)

            waypoints_data = final_sample['waypoints_hist']
            ego_status_components.append(waypoints_data)  # (obs_horizon, 2)

            final_sample['ego_status'] = torch.cat(ego_status_components, dim=-1)  # (obs_horizon, 14)


        # Only tensors created from numpy-backed storage need a clone for safe multi-worker collate.
        for key in clone_keys:
            value = final_sample.get(key)
            if isinstance(value, torch.Tensor) and not value.is_cuda:
                final_sample[key] = value.clone()

        return final_sample



    def load_image(self, image_paths):
        images = []
        for img_path in image_paths:
            full_img_path = os.path.join(self.image_data_root, img_path)
            try:
                img = Image.open(full_img_path)
                img_tensor = self.image_transform(img)
                images.append(img_tensor)
                img.close()  # Close image object to prevent file handle leak
            except Exception as e:
                print(f"Error loading image {full_img_path}: {e}")
                images.append(torch.zeros(3, 256, 928))
        
        if len(images) > 0:
            images_tensor = torch.stack(images)
        else:
            images_tensor = torch.zeros(2, 3, 256, 928)  # 默认obs_horizon=2
        
        images.clear()  # Clear list to release references

        return images_tensor

    def load_lidar_bev(self, bev_paths, sample_path):
        images = []
        for bev_path in bev_paths:
            full_bev_path = os.path.join(self.image_data_root, bev_path)
            bev_image = Image.open(full_bev_path)
            bev_tensor = self.lidar_bev_transform(bev_image)
            images.append(bev_tensor)
            bev_image.close()  # Close image object to prevent file handle leak

        images_tensor = torch.stack(images)
        images.clear()  # Clear list to release references

        return images_tensor

# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_trajectory(sample, obs_horizon, rand_idx, save_dir='/home/wang/Project/MoT-DP/image'):

    agent_pos = sample.get('agent_pos')
    waypoints_hist = sample.get('waypoints_hist')
    target_point = sample.get('target_point')
    target_point_hist = sample.get('target_point_hist')
    anchor = sample.get('anchor')
    route = sample.get('route')
    
    if isinstance(agent_pos, torch.Tensor):
        agent_pos = agent_pos.float().numpy()
    if isinstance(waypoints_hist, torch.Tensor):
        waypoints_hist = waypoints_hist.float().numpy()
    if isinstance(target_point, torch.Tensor):
        target_point = target_point.float().numpy()
    if isinstance(target_point_hist, torch.Tensor):
        target_point_hist = target_point_hist.float().numpy()
    if isinstance(anchor, torch.Tensor):
        # Handle bfloat16 by converting to float32 first
        anchor = anchor.float().numpy()
        # Remove extra dimension if shape is (1, N, 2)
        if anchor.ndim == 3 and anchor.shape[0] == 1:
            anchor = anchor[0]
    if isinstance(route, torch.Tensor):
        route = route.float().numpy()
    
    plt.figure(figsize=(12, 12))
    
    # ========== Plot Historical Waypoints ==========
    if waypoints_hist is not None and len(waypoints_hist) > 0:
        # waypoints_hist shape: (obs_horizon, 2)
        # Plot line connecting historical points
        plt.plot(waypoints_hist[:, 1], waypoints_hist[:, 0], 'b-', 
                linewidth=1.5, alpha=0.5, label='History trajectory', zorder=2)
        
        # Plot each historical waypoint as discrete point
        for i, waypoint in enumerate(waypoints_hist[:-1]):  # Exclude last (current frame)
            plt.plot(waypoint[1], waypoint[0], 'bo', markersize=8, zorder=3)
    
    # ========== Plot Historical Target Points ==========
    if target_point_hist is not None and len(target_point_hist) > 0:
        # target_point_hist shape: (obs_horizon, 2)
        # Plot each historical target point with different colors based on time
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(target_point_hist)))
        for i, target_pt in enumerate(target_point_hist):
            plt.plot(target_pt[1], target_pt[0], 'D', color=colors[i], 
                    markersize=6, alpha=0.7, zorder=3)
            # Add text label for frame index
            plt.text(target_pt[1], target_pt[0] + 1.5, f't-{len(target_point_hist)-1-i}', 
                    fontsize=8, ha='center', alpha=0.6)
    
    # ========== Plot Current Position (Origin) ==========
    plt.plot(0, 0, 'ko', markersize=15, label='Current position (t=0)', 
            markeredgecolor='yellow', markeredgewidth=2, zorder=5)
    
    # ========== Plot Future Waypoints (GT trajectory - agent_pos) ==========
    if agent_pos is not None and len(agent_pos) > 0:
        # Plot line connecting future points (RED for GT)
        future_waypoints = np.vstack([[[0, 0]], agent_pos])
        plt.plot(future_waypoints[:, 1], future_waypoints[:, 0], 'r-', 
                linewidth=2.5, alpha=0.8, label='GT trajectory (agent_pos)', zorder=4)
        
        # Plot each future waypoint as discrete point
        for i, waypoint in enumerate(agent_pos, 1):
            plt.plot(waypoint[1], waypoint[0], 'ro', markersize=10, 
                    markeredgecolor='darkred', markeredgewidth=1.5, zorder=5)
    
    # ========== Plot Anchor Waypoints (predicted trajectory) ==========
    if anchor is not None and len(anchor) > 0:
        # Plot line connecting anchor points (CYAN for anchor/prediction)
        # Ensure anchor has shape (N, 2)
        if anchor.ndim == 1:
            anchor = anchor.reshape(-1, 2)
        anchor_waypoints = np.vstack([[0, 0], anchor])
        plt.plot(anchor_waypoints[:, 1], anchor_waypoints[:, 0], 'c-', 
                linewidth=2.5, alpha=0.7, label='Anchor trajectory (pred_traj)', zorder=3)
        
        # Plot each anchor waypoint as discrete point
        for i, waypoint in enumerate(anchor, 1):
            plt.plot(waypoint[1], waypoint[0], 'c^', markersize=8, 
                    markeredgecolor='darkcyan', markeredgewidth=1.5, zorder=4)
    
    # ========== Plot Route Waypoints (planned route) ==========
    if route is not None and len(route) > 0:
        # Plot line connecting route points (MAGENTA for route)
        # route shape: (20, 2) - [x, y] in ego frame
        plt.plot(route[:, 1], route[:, 0], 'm--', 
                linewidth=2.0, alpha=0.6, label='Route waypoints', zorder=2)
        
        # Plot discrete route points
        for i, waypoint in enumerate(route):
            plt.plot(waypoint[1], waypoint[0], 'ms', markersize=6, 
                    markeredgecolor='darkmagenta', markeredgewidth=1.0, alpha=0.6, zorder=3)
    
    # ========== Plot Current Target Point ==========
    if target_point is not None:
        if target_point.ndim == 2:
            target_point = target_point[0]
        plt.plot(target_point[1], target_point[0], 'g*', markersize=30, 
                label='Current target point (t=0)', markeredgecolor='darkgreen', markeredgewidth=2, zorder=6)
    
    # ========== Formatting ==========
    plt.xlabel('Y (ego frame, lateral / m)', fontsize=12, fontweight='bold')
    plt.ylabel('X (ego frame, longitudinal / m)', fontsize=12, fontweight='bold')
    plt.title(f'Sample {rand_idx}: Trajectory with History Target Points', fontsize=13, fontweight='bold')
    
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Add legend with custom formatting
    plt.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='black')
    
    # Add axis line at origin
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set axis range to be square and centered at origin
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    
    # Calculate the maximum range needed
    x_range = max(abs(xlim[0]), abs(xlim[1]))
    y_range = max(abs(ylim[0]), abs(ylim[1]))
    max_range = max(x_range, y_range, 1.0)
    
    # Set symmetric square limits around origin
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    # Skip tight_layout due to numpy/matplotlib compatibility issues
    # plt.tight_layout() is already handled by bbox_inches='tight' in savefig
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'sample_{rand_idx}_trajectory.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✓ 保存轨迹可视化到: {save_path}")


def visualize_observation_images(sample, obs_horizon, rand_idx, save_dir='/root/z_projects/code/MoT-DP-1/image'):

    images = sample['image']  # shape: (obs_horizon, C, H, W)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for t, img_tensor in enumerate(images):
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor * std + mean
            img_tensor = torch.clamp(img_tensor, 0, 1)
            img_arr = img_tensor.numpy()
        else:
            img_arr = img_tensor
            
        if img_arr.shape[0] == 3:
            img_vis = np.moveaxis(img_arr, 0, -1)
            img_vis = (img_vis * 255).astype(np.uint8)
        else:
            img_vis = img_arr.astype(np.uint8)
        
        plt.figure(figsize=(20, 5))
        plt.imshow(img_vis)
        plt.title(f'Random Sample {rand_idx} - Obs Image t={t}', fontsize=12)
        plt.axis('off')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)

        save_path = os.path.join(save_dir, f'sample_{rand_idx}_obs_image_t{t}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存观测图像到: {save_path}")




def print_sample_details(sample, dataset, rand_idx, obs_horizon, 
                        save_dir='/home/wang/Project/MoT-DP/image'):
    print(f"\n样本 {rand_idx} 的详细信息:")
    
    agent_pos = sample.get('agent_pos')
    if agent_pos is not None:
        obs_agent_pos = agent_pos[:obs_horizon]
        pred_agent_pos = agent_pos[obs_horizon:]
        print(f"观测位置: {obs_agent_pos}")
        print(f"预测位置: {pred_agent_pos}")

    target_point = sample.get('target_point')
    if target_point is not None:
        if target_point.ndim == 2:
            target_point = target_point[0]
        print(f"目标点 (相对): {target_point}")
        if agent_pos is not None and len(agent_pos) > obs_horizon - 1:
            last_obs = agent_pos[obs_horizon - 1]
            distance_to_target = np.linalg.norm(target_point - last_obs)
            print(f"目标点距离最后观测点的距离: {distance_to_target:.3f}")

    target_point_hist = sample.get('target_point_hist')
    if target_point_hist is not None:
        print(f"\n历史目标点 (target_point_hist):")
        print(f"  形状: {target_point_hist.shape}")
        for i, tp in enumerate(target_point_hist):
            frame_idx = len(target_point_hist) - 1 - i
            print(f"  t-{frame_idx}: {tp}")

    print(f"\nDataset length: {len(dataset)}")
    first_sample = dataset[0]
    print("Sample keys:", first_sample.keys())
    if 'image' in first_sample:
        print("Image shape:", first_sample['image'].shape)
    if 'agent_pos' in first_sample:
        print("Agent pos shape:", first_sample['agent_pos'].shape)

    print("\nKey fields:")
    for key in ['town_name', 'speed', 'command', 'next_command', 'target_point', 
                'target_point_hist', 'ego_waypoints', 'image', 'agent_pos', 'meta_action_direction', 'meta_action_speed', 'gen_vit_tokens', 'route']:
        if key in first_sample:
            value = first_sample[key]
            print(f"  {key}: shape={getattr(value, 'shape', 'N/A')}, type={type(value)}")
            if hasattr(value, '__len__') and not isinstance(value, str):
                print(f"    Length: {len(value)}")



def test_pdm():
    """Test with PDM Lite dataset."""
    import random
    
    dataset_path = '/home/wang/Dataset/pdm_lite_mini/tmp_data/val'
    obs_horizon = 4
    """
    Prints detailed information about a sample and the dataset.
    
    Args:
        sample (dict): Data sample
        dataset (CARLAImageDataset): Dataset object
        rand_idx (int): Sample index
        obs_horizon (int): Number of observation frames
        save_dir (str): Directory for saving visualizations
    """
    print("\n========== Testing PDM Lite Dataset ==========")
    dataset = CARLAImageDataset(
        dataset_path=dataset_path,
        image_data_root='/home/wang/Dataset/pdm_lite_mini'
    )
    
    print(f"\n总样本数: {len(dataset)}")
    if len(dataset) == 0:
        print("数据为空，无法进行测试。")
        return

    rand_idx = random.choice(range(len(dataset)))
    rand_sample = dataset[rand_idx]
    print(f"\n随机选择的样本索引: {rand_idx}")

    print("\nSample keys:", rand_sample.keys())
    for key, value in rand_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}")

    visualize_trajectory(rand_sample, obs_horizon, rand_idx)
    print_sample_details(rand_sample, dataset, rand_idx, obs_horizon)

    
if __name__ == "__main__":
    test_pdm()
    
