# NAVSIM Dataset Structure Notes

Date: 2026-05-13

This note records what we have confirmed about NAVSIM raw logs, official scene loading, and how this should affect MoT-DP label migration.

## 1. Official Structure

NAVSIM uses OpenScene logs as the underlying data source. The standard downloadable splits (`mini`, `trainval`, `test`) are log splits, while NAVSIM-specific splits such as `navtrain` and `navtest` are implemented as **scene filters** over those logs.

Official references:

- NAVSIM split docs: https://github.com/autonomousvision/navsim/blob/main/docs/splits.md
- NAVSIM cache/data-format docs: https://github.com/autonomousvision/navsim/blob/main/docs/cache.md
- Official dataloader: https://github.com/autonomousvision/navsim/blob/main/navsim/common/dataloader.py
- Official dataclasses / `SceneFilter`: https://github.com/autonomousvision/navsim/blob/main/navsim/common/dataclasses.py

Key official points:

- OpenScene/NAVSIM logs are downsampled from nuPlan to 2Hz.
- `navtrain/navtest` are filtered scene splits, not separate raw-log formats.
- NAVSIM splits can include overlapping scenes.
- `SceneFilter` defines:
  - `num_history_frames`
  - `num_future_frames`
  - `frame_interval`
  - `has_route`
  - optional `log_names` / `tokens`

Official `SceneFilter` defaults in `dataclasses.py`:

```python
num_history_frames: int = 4
num_future_frames: int = 10
frame_interval: Optional[int] = None
has_route: bool = True
```

If `frame_interval is None`, the filter sets `frame_interval = num_frames`, so extraction is non-overlapping. If `frame_interval=1`, extraction is sliding/overlapping.

## 2. How Official Loader Extracts Scenes

Official `filter_scenes()` loads each raw log pickle as a list of frames:

```python
scene_dict_list = pickle.load(open(log_pickle_path, "rb"))
```

It then slices the frame list directly:

```python
input_list[i : i + num_frames]
```

The scene token used by NAVSIM is the token of the current frame:

```python
token = frame_list[num_history_frames - 1]["token"]
```

Important implication:

- The official public dataloader does **not** split raw logs by `scene_token`.
- `scene_token` is metadata for the extracted NAVSIM scene/window.
- The raw `.pkl` should be treated as a sequential frame list first, then sliced by `SceneFilter`.

## 3. Raw `.pkl` Anatomy

Each raw `.pkl` under paths like:

```text
/workspace2/data/navsim/navsim_logs/mini/*.pkl
/workspace2/data/navsim/navsim_logs/trainval/*.pkl
```

contains a Python list of frame dictionaries. Each frame includes fields such as:

```text
token
timestamp
sample_prev
sample_next
scene_token
scene_name
log_name
frame_idx
ego2global
ego2global_translation
ego2global_rotation
roadblock_ids
traffic_lights
driving_command
anns.gt_boxes
anns.gt_names
anns.gt_velocity_3d
anns.instance_tokens
anns.track_tokens
cams
lidar_path
```

The important thing we observed is that a single `.pkl` often contains many `scene_token` runs. In many cases, `frame_idx` resets at `scene_token` boundaries, but `sample_prev/sample_next` and `timestamp` can remain continuous across those boundaries.

Example from mini:

```text
index 23: frame_idx=23, scene_token=A, token=b1..., sample_next=f1...
index 24: frame_idx=0,  scene_token=B, token=f1..., sample_prev=b1...
timestamp delta ~= 0.5s
```

So `scene_token` changes do not necessarily mean the physical/log sequence is broken.

## 4. Empirical Split Statistics

These stats were computed on newhpc using raw logs under `/workspace2/data/navsim/navsim_logs`.

### mini

```text
pkls: 64
total_frames: 51,867
bad token/timestamp links inside pkl: 33
```

Per `.pkl` frame count:

```text
min: 172
p25: 737
p50: 800
p75: 936.5
p90: 997.2
p95: 1017.75
max: 1132
mean: 810.42
```

`scene_token` runs per `.pkl`:

```text
min: 5
p25: 19
p50: 21
p75: 24
p90: 26
max: 29
mean: 21.28
```

`scene_token` run length:

```text
min: 1
p25: 40
p50: 40
p75: 40
p90: 40
p95: 40
max: 41
mean: 38.08
```

Most common run lengths:

```text
40: 1178 runs
39: 44
41: 16
```

### trainval

```text
pkls: 1,310
total_frames: 723,019
bad token/timestamp links inside pkl: 675
```

Per `.pkl` frame count:

```text
min: 119
p25: 196.25
p50: 330
p75: 600
p90: 1118.6
p95: 1698.85
max: 5962
mean: 551.92
```

`scene_token` runs per `.pkl`:

```text
min: 3
p25: 6
p50: 9
p75: 16
p90: 29
p95: 43.55
max: 150
mean: 14.79
```

`scene_token` run length:

```text
min: 1
p25: 40
p50: 40
p75: 40
p90: 40
p95: 40
max: 41
mean: 37.32
```

Continuous chain length when splitting by `sample_next/sample_prev` and timestamp continuity:

```text
min: 1
p25: 108
p50: 198
p75: 384
p90: 776.4
p95: 1199.2
max: 5962
mean: 364.24
```

This explains the confusion:

- `scene_token` fragments are usually ~40 frames, about 20s at 2Hz.
- Continuous chains inside `.pkl` are often much longer, commonly 100-400+ frames.

## 5. Recommended Unit For MoT-DP Labels

For MoT-DP label migration, especially cover-based labels, the correct processing unit should be:

```text
raw .pkl
  -> split into continuous chains by sample_prev/sample_next + timestamp
  -> choose training current token/window inside each chain
  -> build long future route from the same chain
  -> compute bbox current_cover / future_cover
```

Do not use `scene_token` as the route/label boundary.

Do not blindly treat the whole `.pkl` as a continuous sequence either. There are some broken links/gaps, so the robust rule should be:

```python
prev["sample_next"] == cur["token"]
cur["sample_prev"] == prev["token"]
0.35 <= (cur["timestamp"] - prev["timestamp"]) / 1e6 <= 0.75
```

This is conservative for 2Hz data and avoids crossing discontinuities.

## 6. Token Lookup Strategy

If using official split tokens, e.g. `navtrain.yaml`, the token corresponds to the current/initial frame selected by the official loader:

```python
token = frame_list[num_history_frames - 1]["token"]
```

For our label builder:

1. Load each raw `.pkl`.
2. Build `token_to_idx`.
3. For each official token or sliding current token:
   - find `idx = token_to_idx[token]`
   - find the continuous chain containing `idx`
   - take history/future from that chain
   - build long route from future ego poses in the same chain
   - compute `current_cover/future_cover`

This lets us keep official NAVSIM sample selection while using longer route context for labels.

## 7. Implication For Current Prototype Work

The earlier `navsim_label_stats.py` script was only a coarse distribution scan:

- `path -> route`
- trajectory/speed/object counts
- rough `junction_like`
- rough `merge_like` based on lateral approach to route

That rough `merge_like` is **not** the old MoT-DP label logic.

The correct next prototype should be cover-based:

```text
long route from continuous chain
current boxes in current ego frame
future boxes transformed into current ego frame
_find_route_cover_point()
_compute_front_route_label()
current_cover / future_cover
then family-specific merge/junction/borrow decisions
```

Borrow should still remain out of main supervision until full trainval confirms enough stable two-way borrow-like examples.

## 8. Practical Next Steps

1. Keep a small script that only validates dataset structure:
   - pkl lengths
   - scene_token run lengths
   - continuous chain lengths
   - route-valid counts

2. Refactor the label prototype around continuous chains:
   - no `scene_token` truncation
   - no cross-pkl stitching initially
   - split only at broken token/timestamp links

3. Implement NAVSIM adapter into old cover logic:
   - convert NAVSIM `anns.gt_boxes` into MoT-DP box dicts
   - preserve stable numeric actor ids from `track_tokens`
   - use `ego2global` to transform future boxes into current frame

4. Generate cover-based videos only after the cover stats are nontrivial:
   - visualize `current_cover`
   - visualize `future_cover`
   - show cover point on route
   - show selected actor id / track token
