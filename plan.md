# Plan: Scene-Level Cached Dataset for HPC

## Background

Current `unified_carla_dataset.py` loads **sample-level pkl files** + **transfuser features** (~1.4MB/sample).
On HPC with slow shared filesystem and limited memory (110GB/GPU, 440GB for 4 GPUs), this causes I/O bottleneck.

**Key insight**: Raw data (bev_semantics + measurements + boxes) is only ~10KB/sample vs 1404KB with transfuser features (139x reduction). Total raw dataset = ~0.2GB vs 25.4GB.

## Design

New file: `dataset/cached_carla_dataset.py` — scene-level dataset with diskcache.

### Architecture

```
Raw dataset on shared FS:
  /shared/dataset/
    AccidentTwoWays/
      Town12_Rep0_.../
        bev_semantics/*.png      (2.1KB each)
        measurements/*.json.gz   (1.2KB each)
        boxes/*.json.gz          (1.8KB each)
        rgb/*.jpg                (162KB each, for val only)

                    ↓ (first epoch: load from shared FS)
                    ↓ (cached: load from local SSD)

diskcache.Cache on local SSD ($SCRATCH):
  scene_key → {all frames of a scene packed together}
```

### Core Differences from Sample-Level Dataset

| Aspect | Current (sample-level) | New (scene-level) |
|--------|----------------------|-------------------|
| Index unit | Individual pkl file | (route_idx, frame_idx) |
| Feature source | Transfuser .pt files (1.4MB) | Raw BEV/boxes/measurements (~10KB) |
| I/O pattern | 18936 small files | 184 scene reads, then cache |
| Cache | None | diskcache on local SSD |
| Memory footprint | ~25GB features | ~0.2GB raw data |
| Scene context | Lost (independent samples) | Preserved (temporal neighbors available) |

### Data Loading Flow

1. **Init**: Scan raw dataset directory → enumerate routes → build `(route_path, frame_id)` index
2. **`__getitem__`**:
   - Compute cache key = `f"{route_path}/{frame_id:04d}"`
   - If in cache → deserialize and return
   - If not → load from shared FS, cache it, return
3. **Scene-level caching**: When loading a frame, optionally prefetch entire scene (all frames in route) to minimize I/O round-trips

### What Each Sample Contains (no transfuser features)

From measurements + existing pkl metadata reconstruction:
- `ego_waypoints`, `speed_hist`, `theta_hist`, `command_hist`, `waypoints_hist`
- `target_point_hist`, `target_point_next_hist`, `route`

Loaded from raw data on-the-fly:
- `bev_semantics` (256×256 uint8 PNG → numpy)
- `measurements` (json.gz → dict, for behavior labeling + ego status)
- `boxes` (json.gz → list, for dynamic collision)
- `rgb` (jpg, validation only)
- Future frames boxes/measurements (for semantic behavior labeling)

### Config Addition

```yaml
# In carla.yaml, add:
scene_dataset:
  enabled: false              # Toggle between sample-level and scene-level
  raw_data_root: '/path/to/raw/dataset'
  cache_dir: '$SCRATCH/mot_dp_cache'
  cache_size_limit_gb: 100    # Fits in 110GB/GPU
  prefetch_scene: true        # Load full scene on first access
  skip_first_n_frames: 3      # Skip initial frames (need obs_horizon history)
  obs_horizon: 4              # History frames for speed/theta/command/waypoints
  pred_horizon: 6             # Future frames for GT waypoints
```

### Training Script Change

Minimal change in `train_carla_bev.py`:
```python
if config.get('scene_dataset', {}).get('enabled', False):
    from dataset.cached_carla_dataset import CachedCARLADataset
    train_dataset = CachedCARLADataset(config=config, split='train', ...)
else:
    from dataset.unified_carla_dataset import CARLAImageDataset
    train_dataset = CARLAImageDataset(...)
```

### Key Implementation Details

1. **Route enumeration**: Walk `raw_data_root/{scenario}/{route}/measurements/` to count frames per route, skip first N frames
2. **Index mapping**: Flat index → (route_idx, frame_within_route) via cumulative sum
3. **Cache key**: Use file path string (same pattern as simlingo)
4. **Cache value**: Compressed tuple — `cv2.imencode('.png', bev)` for BEV, raw bytes for json.gz (already compressed)
5. **Memory leak prevention**: Store route paths as `np.string_`, frame counts as `np.int32` (same pattern as simlingo)
6. **Semantic behavior**: Computed on-the-fly (same as current), using cached raw data
7. **Scene buckets**: Computed on-the-fly from cached measurements

### HPC Memory Budget

Per GPU (110GB):
- Cache: ~100GB limit → holds entire raw dataset (0.2GB) easily, plus precomputed behavior labels
- Model + gradients: ~5-8GB
- DataLoader buffers: ~2GB

This is very comfortable. Even with RGB images cached for validation, total would be ~3GB.

## Files to Create/Modify

1. **CREATE** `dataset/cached_carla_dataset.py` — New scene-level cached dataset class
2. **MODIFY** `config/carla.yaml` — Add `scene_dataset` config section
3. **MODIFY** `training/train_carla_bev.py` — Add conditional import for new dataset

No changes to `unified_carla_dataset.py` (preserved as-is).
