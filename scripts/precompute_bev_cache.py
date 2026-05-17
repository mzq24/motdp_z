"""
Offline BEV feature precomputation using LEAD backbone.
Shards by pkl files (not tokens) to avoid per-process full dataset scan.

Run on newhpc:
    OPENSCENE_DATA_ROOT=/workspace2/data/navsim \
    NUPLAN_MAPS_ROOT=/workspace2/data/navsim/maps \
    conda run -n z_navsim_motdp python scripts/precompute_bev_cache.py --shard 0 --num_shards 4
"""

import argparse, gzip, os, pickle, sys, time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from navsim_motdp.lead_preprocessing import (
    LEAD_NAVSIM_RAW_CAMERA_KEYS,
    build_official_lead_rgb_tensor_from_arrays,
)

# Config
CACHE_DIR = "/workspace2/z_project/motdp_bev_cache"
LOG_ROOT_DEFAULT = "/workspace2/data/navsim/navsim_logs/trainval"
SENSOR_ROOT_DEFAULT = "/workspace2/data/navsim/sensor_blobs/trainval"
FILTER_DEFAULT = "/home/z/code/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
TRAINVAL_LOGS = "/workspace2/data/navsim/navsim_logs/trainval"
SENSOR_BLOBS = "/workspace2/data/navsim/sensor_blobs/trainval"
LEAD_CKPT = "/workspace1/z_project/models/navsim_backbones/tfv6_navsim/model_0060.pth"
FRAME_INTERVAL = 1  # sliding window
NUM_HISTORY, NUM_FUTURE = 4, 10
WINDOW_SIZE = NUM_HISTORY + NUM_FUTURE

# Load navtrain split.  SparseDrive/NAVSIM training uses both log_names and
# explicit scene_filter.tokens, so the default precompute path follows that
# same token-level filter.
import yaml
with open("/home/z/code/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml") as f:
    _nav = yaml.safe_load(f)
NAVTRAIN_LOGS = set(_nav["log_names"])
NAVTRAIN_TOKENS = set(_nav.get("tokens") or [])


def load_lead_backbone(device):
    sys.path.insert(0, "/workspace1/z_project/models/navsim_backbones/tfv6_navsim")
    from ltfv6 import load_tf
    model = load_tf(LEAD_CKPT, device)
    model = model.to(dtype=torch.bfloat16)
    model.eval()
    original_forward = model.backbone.forward
    def patched_forward(data):
        rgb_in = data["rgb"].to(device, dtype=torch.bfloat16)
        x = torch.linspace(0, 1, model.config.lidar_width_pixel, device=device, dtype=torch.bfloat16)
        y = torch.linspace(0, 1, model.config.lidar_height_pixel, device=device, dtype=torch.bfloat16)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        lidar = torch.zeros((rgb_in.shape[0], 2, model.config.lidar_height_pixel, model.config.lidar_width_pixel),
                           device=device, dtype=torch.bfloat16)
        lidar[:, 0] = y_grid.unsqueeze(0); lidar[:, 1] = x_grid.unsqueeze(0)
        return model.backbone._forward(rgb_in, lidar)
    model.backbone.forward = patched_forward
    return model


def build_lead_input(raw_frame, device):
    images = []
    for cam_name in LEAD_NAVSIM_RAW_CAMERA_KEYS:
        img_path = Path(SENSOR_BLOBS) / raw_frame["cams"][cam_name]["data_path"]
        if not img_path.exists():
            return None
        images.append(np.asarray(Image.open(img_path).convert("RGB")))

    rgb_tensor = build_official_lead_rgb_tensor_from_arrays(images, batched=True)
    return {"rgb": rgb_tensor.to(device, dtype=torch.bfloat16)}


def build_ego_status(raw_frames, current_idx):
    feats = []
    for i in range(current_idx - 3, current_idx + 1):
        f = raw_frames[i]
        vel = np.array(f["ego_dynamic_state"][:2], dtype=np.float32)
        acc = np.zeros(2, dtype=np.float32)
        if i > current_idx - 3:
            pv = np.array(raw_frames[i-1]["ego_dynamic_state"][:2], dtype=np.float32)
            dt = (f["timestamp"] - raw_frames[i-1]["timestamp"]) / 1e6
            if dt > 0: acc = (vel - pv) / dt
        cmd = np.array(f["driving_command"], dtype=np.float32)
        feats.append(np.concatenate([vel, acc, cmd, np.zeros(6, dtype=np.float32)]))
    return np.stack(feats, axis=0).astype(np.float32)


def build_trajectory(raw_frames, current_idx, horizon=8):
    cur_f = raw_frames[current_idx]
    ego_t = np.array(cur_f["ego2global_translation"][:2], dtype=np.float64)
    q = Quaternion(*cur_f["ego2global_rotation"])
    ego_yaw = q.yaw_pitch_roll[0]
    cos_y, sin_y = np.cos(ego_yaw), np.sin(ego_yaw)
    waypoints = []
    for fi in range(current_idx + 1, min(current_idx + horizon + 1, len(raw_frames))):
        f = raw_frames[fi]
        t = np.array(f["ego2global_translation"][:2], dtype=np.float64)
        dx, dy = t[0] - ego_t[0], t[1] - ego_t[1]
        waypoints.append([dx * cos_y + dy * sin_y, -dx * sin_y + dy * cos_y])
    while len(waypoints) < horizon:
        waypoints.append(waypoints[-1] if waypoints else [0.0, 0.0])
    return np.array(waypoints, dtype=np.float32)


def is_continuous(prev, cur):
    if prev["sample_next"] != cur["token"]: return False
    if cur["sample_prev"] != prev["token"]: return False
    dt = (cur["timestamp"] - prev["timestamp"]) / 1e6
    return 0.35 <= dt <= 0.75


def process_pkl(pkl_path, model, device, shard_id, total_shards, token_filter=None, require_continuity=False):
    """Process one pkl: sliding windows → backbone → accumulate results."""
    pkl_name = pkl_path.stem
    print(f"  [{pkl_name}] loading...", flush=True)
    with open(pkl_path, "rb") as f:
        frames = pickle.load(f)

    n_frames = len(frames)
    if n_frames < WINDOW_SIZE:
        print(f"  [{pkl_name}] too short ({n_frames} frames), skip", flush=True)
        return None, None, None, None, None, None, None, 0, 0

    all_bev_grid, all_bev_feat, all_ego, all_traj = [], [], [], []
    all_tokens, all_log_names, all_frame_indices = [], [], []
    n_ok, n_skip = 0, 0

    for start in range(0, n_frames - WINDOW_SIZE + 1, FRAME_INTERVAL):
        window = frames[start:start + WINDOW_SIZE]
        current_idx = NUM_HISTORY - 1
        cur_raw = window[current_idx]

        if token_filter is not None and cur_raw["token"] not in token_filter:
            n_skip += 1
            continue

        # Check continuity within window
        if require_continuity:
            continuous = True
            for i in range(len(window) - 1):
                if not is_continuous(window[i], window[i + 1]):
                    continuous = False
                    break
            if not continuous:
                n_skip += 1
                continue

        lead_input = build_lead_input(cur_raw, device)
        if lead_input is None:
            n_skip += 1
            continue

        with torch.no_grad():
            bev_feat, _ = model.backbone(lead_input)
            bev_grid = model.backbone.top_down(bev_feat)

        ego_status = build_ego_status(frames, start + current_idx)
        trajectory = build_trajectory(frames, start + current_idx)

        all_bev_grid.append(bev_grid[0].float().cpu().numpy())
        all_bev_feat.append(bev_feat[0].float().cpu().numpy())
        all_ego.append(ego_status)
        all_traj.append(trajectory)
        all_tokens.append(cur_raw["token"])
        all_log_names.append(pkl_name)
        all_frame_indices.append(start + current_idx)
        n_ok += 1

    if n_ok == 0:
        print(f"  [{pkl_name}] no valid tokens", flush=True)
        return None, None, None, None, None, None, None, 0, n_skip

    print(f"  [{pkl_name}] {n_ok} tokens cached, {n_skip} skipped", flush=True)
    return (np.stack(all_bev_grid).astype(np.float16),
            np.stack(all_bev_feat).astype(np.float16),
            np.stack(all_ego).astype(np.float32),
            np.stack(all_traj).astype(np.float32),
            np.asarray(all_tokens),
            np.asarray(all_log_names),
            np.asarray(all_frame_indices, dtype=np.int32),
            n_ok, n_skip)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--limit_pkls", type=int, default=-1)
    parser.add_argument("--pkl_offset", type=int, default=0, help="start pkl index")
    parser.add_argument("--pkl_count", type=int, default=0, help="number of pkls (0=all)")
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--split_mode", choices=["official_tokens", "log_only"], default="official_tokens",
                        help="official_tokens matches SparseDrive/NAVSIM navtrain; log_only keeps the old scan mode")
    parser.add_argument("--token_list", type=str, default="",
                        help="optional newline token list; intersected with split_mode token filter")
    parser.add_argument("--require_continuity", action="store_true",
                        help="optionally require sample_prev/sample_next continuity inside each window")
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Device: {device}, shard {args.shard}/{args.num_shards}", flush=True)

    # Load LEAD backbone
    print("Loading LEAD backbone...", flush=True)
    model = load_lead_backbone(device)
    print(f"  OK, params={sum(p.numel() for p in model.backbone.parameters())/1e6:.1f}M", flush=True)

    # List pkl files
    all_pkls = sorted(Path(TRAINVAL_LOGS).glob("*.pkl"))
    # Filter to navtrain logs
    my_pkls = [p for p in all_pkls if p.stem in NAVTRAIN_LOGS]
    print(f"Found {len(my_pkls)} navtrain pkls (from {len(all_pkls)} total)", flush=True)
    token_filter = NAVTRAIN_TOKENS if args.split_mode == "official_tokens" else None
    if args.token_list:
        listed_tokens = set(Path(args.token_list).read_text().split())
        token_filter = listed_tokens if token_filter is None else token_filter & listed_tokens
    print(f"Split mode: {args.split_mode}, token_filter={len(token_filter) if token_filter is not None else 'off'}, "
          f"require_continuity={args.require_continuity}", flush=True)

    if args.limit_pkls > 0:
        my_pkls = my_pkls[:args.limit_pkls]

    # Apply pkl_offset/pkl_count if set
    if args.pkl_count > 0:
        my_pkls = my_pkls[args.pkl_offset:args.pkl_offset + args.pkl_count]
        print(f"Sub-range pkgs [{args.pkl_offset}, {args.pkl_offset+args.pkl_count}): {len(my_pkls)} pkls", flush=True)
    else:
        # Shard by pkl
        n_total = len(my_pkls)
        shard_size = (n_total + args.num_shards - 1) // args.num_shards
        start = args.shard * shard_size
        end = min(start + shard_size, n_total)
        my_pkls = my_pkls[start:end]
        print(f"Shard pkgs [{start}, {end}): {len(my_pkls)} pkls", flush=True)

    # Process each pkl
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    all_bev_grid, all_bev_feat, all_ego, all_traj = [], [], [], []
    all_tokens, all_log_names, all_frame_indices = [], [], []
    total_ok, total_skip = 0, 0
    t0 = time.time()

    for pi, pkl_path in enumerate(my_pkls):
        bg, bf, eg, tr, tok, log_names, frame_indices, n_ok, n_skip = process_pkl(
            pkl_path,
            model,
            device,
            args.shard,
            args.num_shards,
            token_filter=token_filter,
            require_continuity=args.require_continuity,
        )
        if bg is not None:
            all_bev_grid.append(bg)
            all_bev_feat.append(bf)
            all_ego.append(eg)
            all_traj.append(tr)
            all_tokens.append(tok)
            all_log_names.append(log_names)
            all_frame_indices.append(frame_indices)
        total_ok += n_ok
        total_skip += n_skip

        # Save intermediate every 10 pkls
        sfx = args.output_suffix
        if (pi + 1) % 10 == 0 and all_bev_grid:
            tmp = f"{cache_dir}/bev_cache_shard{args.shard:03d}{sfx}_tmp.npz"
            np.savez_compressed(tmp,
                bev_grid=np.concatenate(all_bev_grid, axis=0).astype(np.float16),
                bev_feature=np.concatenate(all_bev_feat, axis=0).astype(np.float16),
                ego_status=np.concatenate(all_ego, axis=0).astype(np.float32),
                trajectory=np.concatenate(all_traj, axis=0).astype(np.float32),
                tokens=np.concatenate(all_tokens, axis=0),
                log_names=np.concatenate(all_log_names, axis=0),
                frame_indices=np.concatenate(all_frame_indices, axis=0),
            )
            print(f"  [interim save] {total_ok} tokens, {os.path.getsize(tmp)/1e6:.0f}MB", flush=True)

    # Final save
    sfx = args.output_suffix
    if all_bev_grid:
        out_path = f"{cache_dir}/bev_cache_shard{args.shard:03d}{sfx}.npz"
        np.savez_compressed(out_path,
            bev_grid=np.concatenate(all_bev_grid, axis=0).astype(np.float16),
            bev_feature=np.concatenate(all_bev_feat, axis=0).astype(np.float16),
            ego_status=np.concatenate(all_ego, axis=0).astype(np.float32),
            trajectory=np.concatenate(all_traj, axis=0).astype(np.float32),
            tokens=np.concatenate(all_tokens, axis=0),
            log_names=np.concatenate(all_log_names, axis=0),
            frame_indices=np.concatenate(all_frame_indices, axis=0),
        )
        size_mb = os.path.getsize(out_path) / 1e6
        elapsed = (time.time() - t0) / 60
        print(f"DONE: {total_ok} tokens, {total_skip} skipped, {size_mb:.0f}MB, {elapsed:.1f}min", flush=True)
    else:
        print(f"DONE: no tokens produced", flush=True)


if __name__ == "__main__":
    main()
