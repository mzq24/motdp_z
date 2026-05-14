# DataLoader CPU Bottleneck Analysis

Date: 2026-05-09

## Summary

2 GPU × batch_size 176, GPU util 66-68%. 瓶颈不在 worker 数量/prefetch，
在 `__getitem__` 的 per-sample CPU 处理。

## Per-Sample CPU Pipeline

| 步骤 | 操作 | 数据量 |
|---|---|---|
| 1 | memmap `.copy()` — BEV feature | (1512, 8, 8) = 192KB |
| 2 | memmap `.copy()` — BEV upsample | (64, 32, 32) = 128KB |
| 3 | `F.interpolate` CPU bilinear upsampling | 32²→64² |
| 4 | LiDAR BEV: 4-frame memmap load | 4×(2, 256, 256) = 1MB |
| 5 | `_from_numpy` — stage1 fields | ~50+ tensors |
| 6 | `_ensure_stage1_legacy_curve_defaults` | 11 curve template clones |
| 7 | `torch.cat` build ego_status | (4, 14) |
| 8 | GPS noise random + tensor addition | |
| 9 | `.clone()` loop — all numpy-backed tensors | ~50+ tensors, 含 BEV/lidar |

## Extra Overhead vs Independent-State Branch

New fields added: graph edges (current_cover_*, future_cover_*),
prev state fields (prev_conflict_*, prev_merge_*, etc.), chase fields.
~20 extra fields × `_from_numpy` + clone = ~3520 extra clone ops per batch (176×20).

Also graph labels and transition fields increase per-sample tensor count in collate.

## Identified Issues

### 1. Double copy on BEV/LiDAR features

```
memmap[abs_idx] → .copy() [numpy copy #1]
→ torch.from_numpy() → .clone() [tensor copy #2]
```

`.copy()` on numpy side + `.clone()` on tensor side = 2 copies. The first copy is already in-memory (not mmap-backed). The clone is redundant for these large tensors.

Potential fix: do only one copy. Either skip the numpy `.copy()` and clone the tensor, or keep the numpy copy and skip the clone.

### 2. Clone loop clones everything

Line 981-984 clones every tensor that came from numpy, including large BEV features that already did a numpy copy. For large tensors this is significant CPU overhead.

### 3. `_ensure_stage1_legacy_curve_defaults` may be obsolete

If the new relabel/project/split pipeline guarantees all legacy curve keys are present, this function and its 11 template clones per sample can be removed. Otherwise it adds ~constant overhead.

### 4. Per-sample field count keeps growing

Each new relabel field adds: numpy→tensor → clone flag → collate → batch tensor. CPU cost scales with field count. This is a structural scaling issue — more fields = more CPU time per sample.

## Historical Speed Reference

| config | GPUs | batch | forwards | time/epoch |
|---|---|---|---|---|
| 1-forward era | 5 | 128 | 1 | ~5 min |
| 2-forward era | 4 | 128 | 2 | ~10 min |
| independent-state | 2 | 128→176 | 4 (2 ego + 2 consistency) | 26-30 min |
| graph-decoder | 2 | 176 | 4 | current |

## Current Known Non-Code Optimizations Applied

- num_workers: 4 (train), 1 (val)
- prefetch_factor: 2 (train), 1 (val)
- pin_memory: false (disabled due to OOM with large batches)
- persistent_workers: false (disabled due to memory leak)
- batch_size pushed to 176 (near VRAM limit)

## 2026-05-11 Update: RAM-Backed Feature Cache

Warmup-only page-cache staging was not reliable on shared new_hpc nodes: other
jobs can evict the warmed pages during training, causing sudden multi-second
stalls again.

Implemented a stronger path in `training/train_carla_bev.py`:

```yaml
dataset:
  stage_feature_cache_to_ram: true
  cache_source_dir: /workspace1/z_project/dataset/pdm_lite/tmp_data
  cache_dir: /dev/shm/motdp_feature_cache_ensemble
  train_warmup_memmap_page_cache: false
```

Behavior:

- rank 0 copies the shared feature memmap files into `/dev/shm`;
- other DDP ranks wait at a barrier;
- train/val datasets still use the normal memmap path, but the backing files
  are RAM-backed rather than disk/page-cache-backed;
- this avoids per-rank Python tensor duplication while reducing cold page-fault
  stalls.

Notes:

- This is intended for nodes with enough free RAM.
- If `/dev/shm` is too small on a specific machine, set `cache_dir` to another
  RAM-backed or local-NVMe path.
- `validation.preload_to_ram` can stay false because validation also reads the
  memmap from the RAM-backed `cache_dir`.

## Recommended Code-Level Changes

1. Remove consistency loss (`use_independent_state_consistency_loss: false`) — 2 forwards saved
2. Merge ego denoising forward with stage1 semantic forward — 4→3→2 forwards
3. Eliminate double copy on BEV/lidar features
4. Remove `_ensure_stage1_legacy_curve_defaults` if all relabeled data has the fields
5. Skip clone on large tensors that already own their memory

## Note

This analysis was discussed on 2026-05-09 in the context of the
`semantic_state_next_token_rl_v1` worktree (cover relation graph + compact
semantic conditioning).
