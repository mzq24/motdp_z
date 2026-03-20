#!/usr/bin/env python3
"""Generate sample videos for each scenario type in the dataset.

For each scenario type directory (e.g. Accident, ConstructionObstacle),
pick N sample routes and render a video showing:
  - Left: RGB front camera
  - Center: BEV semantic map (colorized)
  - Right: BEV semantic map + 2D bounding box overlays

Usage:
    python tools/generate_scenario_videos.py \
        --dataset_root /media/z/data/dataset/pdm_lite_mini \
        --output_dir /tmp/scenario_videos \
        --n_per_type 3 --fps 10 --max_frames 200
"""

import argparse
import gzip
import json
import os

import cv2
import numpy as np

# ── BEV semantic class colorization (BGR) ──────────────────────────────────
BEV_COLORS = np.array([
    [40, 40, 40],       # 0  background
    [128, 128, 128],    # 1  road
    [180, 0, 180],      # 2  sidewalk
    [255, 255, 255],    # 3  solid_line
    [0, 255, 255],      # 4  dashed_line
    [0, 0, 255],        # 5  stop_sign
    [0, 255, 0],        # 6  green_light
    [0, 200, 255],      # 7  yellow_light
    [0, 0, 255],        # 8  red_light
    [255, 100, 0],      # 9  vehicle
    [255, 255, 0],      # 10 pedestrian
], dtype=np.uint8)

COMMAND_MAP = {
    1: "LEFT",
    2: "RIGHT",
    3: "STRAIGHT",
    4: "LANE_FOLLOW",
    5: "CHANGE_LEFT",
    6: "CHANGE_RIGHT",
}

# BEV coordinate mapping constants
BEV_SIZE = 256
BEV_PPM = 2.0   # pixels per meter
BEV_CENTER = BEV_SIZE // 2  # 128

SKIP_DIRS = {'tmp_data', 'tar', 'train', 'val'}


def colorize_bev(bev_gray):
    """Convert grayscale BEV class IDs to BGR color image."""
    bev_gray = np.clip(bev_gray, 0, len(BEV_COLORS) - 1)
    return BEV_COLORS[bev_gray]


def load_measurements(path):
    """Load measurements json.gz, return dict or None."""
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    except Exception:
        return None


def load_boxes(path):
    """Load boxes json.gz, return list or empty list."""
    if not os.path.exists(path):
        return []
    try:
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    except Exception:
        return []


def box_corners_2d(position, extent, yaw):
    """Compute 4 BEV corners of a box given ego-centric position, extent, yaw.

    position: [x_forward, y_right, z_up]
    extent:   [half_length, half_width, half_height]
    yaw:      rotation in radians

    Returns: (4, 2) array of [col, row] in BEV 256×256 coords.
    """
    cx, cy = position[0], position[1]
    hl, hw = extent[0], extent[1]

    # 4 corners in local frame (length along x, width along y)
    corners_local = np.array([
        [hl, hw],
        [hl, -hw],
        [-hl, -hw],
        [-hl, hw],
    ])

    # Rotation matrix
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    R = np.array([[cos_y, -sin_y],
                   [sin_y, cos_y]])

    corners_ego = (R @ corners_local.T).T + np.array([cx, cy])

    # Ego coords → BEV pixel coords
    # x_forward → col (right in image), y_right → row (down in image)
    # BEV convention: col = center + x * ppm, row = center - y * ppm
    # But checking semantic_behavior_labeling.md: col = 128 + x_forward * 2.0, row = 128 + y_lateral * 2.0
    # Note: in CARLA ego frame, x=forward, y=right
    # In BEV image: col direction = forward (right), row direction = lateral
    # Actually the BEV mapping from labeler: col = 128 + x * 2, row = 128 + y * 2
    # where x=forward, y=right(lateral)
    # This means: forward→right in image, right→down in image
    cols = BEV_CENTER + corners_ego[:, 0] * BEV_PPM
    rows = BEV_CENTER + corners_ego[:, 1] * BEV_PPM

    return np.stack([cols, rows], axis=1).astype(np.int32)


def draw_boxes_on_bev(bev_color, boxes):
    """Draw 2D bounding box rectangles on a colorized BEV image (256×256)."""
    for box in boxes:
        cls = box.get('class', '')
        if cls == 'ego_car':
            continue
        pos = box.get('position')
        ext = box.get('extent')
        yaw = box.get('yaw', 0.0)
        if pos is None or ext is None:
            continue

        # Skip boxes far outside BEV range (±64m)
        if abs(pos[0]) > 64 or abs(pos[1]) > 64:
            continue

        corners = box_corners_2d(pos, ext, yaw)

        # Choose color based on class + direction + state
        speed = box.get('speed', 0)
        brake = box.get('brake', 0)
        same_dir = np.cos(yaw) > 0  # cos(yaw)>0 → same direction as ego
        if 'walker' in cls or 'pedestrian' in cls:
            color = (255, 255, 0)   # yellow = pedestrian
        elif speed is not None and abs(speed) < 0.5:
            color = (0, 0, 255)     # red = stopped
        elif not same_dir:
            color = (255, 0, 255)   # magenta = oncoming (opposite direction)
        elif brake is not None and brake > 0.5:
            color = (0, 100, 255)   # orange = same-dir braking
        else:
            color = (0, 255, 0)     # green = same-dir moving

        cv2.polylines(bev_color, [corners], isClosed=True, color=color, thickness=1)

        # Draw small arrow for heading direction
        front_mid = ((corners[0] + corners[1]) // 2).astype(int)
        center = np.mean(corners, axis=0).astype(int)
        cv2.arrowedLine(bev_color, tuple(center), tuple(front_mid), color, 1, tipLength=0.3)

    # Draw ego car as a small filled rectangle at center
    ego_corners = np.array([
        [BEV_CENTER + 5, BEV_CENTER - 2],
        [BEV_CENTER + 5, BEV_CENTER + 2],
        [BEV_CENTER - 5, BEV_CENTER + 2],
        [BEV_CENTER - 5, BEV_CENTER - 2],
    ], dtype=np.int32)
    cv2.fillPoly(bev_color, [ego_corners], (255, 255, 255))

    return bev_color


def compose_text_bar(width, scenario_type, meas):
    """Create a text bar image with measurement overlay."""
    bar_h = 80
    bar = np.zeros((bar_h, width, 3), dtype=np.uint8)

    if meas is None:
        cv2.putText(bar, f'[{scenario_type}] No measurements', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return bar

    spd = meas.get('speed', 0)
    tgt = meas.get('target_speed', 0)
    lim = meas.get('speed_limit', 0)
    cmd_id = int(meas.get('command', 0))
    cmd_str = COMMAND_MAP.get(cmd_id, f'UNK({cmd_id})')
    steer = meas.get('steer', 0)
    thr = meas.get('throttle', 0)
    brk = meas.get('brake', 0)

    veh_h = int(meas.get('vehicle_hazard', 0))
    wal_h = int(meas.get('walker_hazard', 0))
    lit_h = int(meas.get('light_hazard', 0))
    stp_h = int(meas.get('stop_sign_hazard', 0))
    junc = int(meas.get('junction', 0))

    line1 = f'[{scenario_type}]  spd:{spd:.1f}  tgt:{tgt:.1f}  lim:{lim:.0f}  cmd:{cmd_str}'
    line2 = f'steer:{steer:.3f}  thr:{thr:.2f}  brk:{brk:.2f}'
    line3 = f'veh_haz:{veh_h}  ped_haz:{wal_h}  light:{lit_h}  stop:{stp_h}  junc:{junc}'

    cv2.putText(bar, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, line2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(bar, line3, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    return bar


def compose_frame(rgb, bev_box_color, scenario_type, meas):
    """Compose the full video frame: RGB | BEV+boxes, plus text bar."""
    # Resize BEV from 256→512 to match RGB height
    bev_up = cv2.resize(bev_box_color, (512, 512), interpolation=cv2.INTER_NEAREST)

    # RGB is 512×1024, BEV is 512×512 → top row = 1536×512
    top = np.concatenate([rgb, bev_up], axis=1)

    # Text bar
    text_bar = compose_text_bar(top.shape[1], scenario_type, meas)

    # Full frame
    frame = np.concatenate([top, text_bar], axis=0)
    return frame


def get_frame_ids(route_dir):
    """Get sorted frame IDs from the rgb directory."""
    rgb_dir = os.path.join(route_dir, 'rgb')
    if not os.path.isdir(rgb_dir):
        return []
    ids = []
    for fname in os.listdir(rgb_dir):
        if fname.endswith(('.jpg', '.png')):
            fid = os.path.splitext(fname)[0]
            ids.append(fid)
    ids.sort()
    return ids


def generate_video(route_dir, output_path, scenario_type, fps, max_frames):
    """Generate a single video for one route."""
    frame_ids = get_frame_ids(route_dir)
    if not frame_ids:
        print(f'  [SKIP] No RGB frames in {route_dir}')
        return False

    frame_ids = frame_ids[:max_frames]

    # Probe first frame to determine video size
    first_rgb_path = os.path.join(route_dir, 'rgb', f'{frame_ids[0]}.jpg')
    if not os.path.exists(first_rgb_path):
        first_rgb_path = os.path.join(route_dir, 'rgb', f'{frame_ids[0]}.png')
    first_rgb = cv2.imread(first_rgb_path)
    if first_rgb is None:
        print(f'  [SKIP] Cannot read first RGB frame')
        return False

    # Expected: 512×1024×3
    h, w = first_rgb.shape[:2]
    frame_w = w + 512  # RGB + BEV
    frame_h = h + 80          # + text bar

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    n_written = 0
    for fid in frame_ids:
        # Read RGB
        rgb_path = os.path.join(route_dir, 'rgb', f'{fid}.jpg')
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(route_dir, 'rgb', f'{fid}.png')
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            continue

        # Read BEV semantic
        bev_path = os.path.join(route_dir, 'bev_semantics', f'{fid}.png')
        bev_gray = cv2.imread(bev_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(bev_path) else None
        if bev_gray is None:
            bev_gray = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.uint8)

        bev_color = colorize_bev(bev_gray)

        # Read boxes and draw on BEV
        boxes_path = os.path.join(route_dir, 'boxes', f'{fid}.json.gz')
        boxes = load_boxes(boxes_path)
        if boxes:
            draw_boxes_on_bev(bev_color, boxes)

        # Read measurements
        meas_path = os.path.join(route_dir, 'measurements', f'{fid}.json.gz')
        meas = load_measurements(meas_path)

        # Compose and write frame
        frame = compose_frame(rgb, bev_color, scenario_type, meas)
        video.write(frame)
        n_written += 1

    video.release()
    if n_written == 0:
        # Remove empty video file
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    return True


def scan_scenarios(dataset_root):
    """Scan dataset root for scenario type directories. Returns {type: [route_dirs]}."""
    scenarios = {}
    for entry in sorted(os.scandir(dataset_root), key=lambda e: e.name):
        if not entry.is_dir() or entry.name in SKIP_DIRS:
            continue
        routes = []
        for route_entry in sorted(os.scandir(entry.path), key=lambda e: e.name):
            if route_entry.is_dir():
                routes.append(route_entry.path)
        if routes:
            scenarios[entry.name] = routes
    return scenarios


def main():
    parser = argparse.ArgumentParser(description='Generate scenario type sample videos')
    parser.add_argument('--dataset_root', required=True, help='Dataset root directory')
    parser.add_argument('--output_dir', default='/tmp/scenario_videos', help='Output directory')
    parser.add_argument('--n_per_type', type=int, default=3, help='Max videos per scenario type')
    parser.add_argument('--fps', type=int, default=10, help='Video FPS')
    parser.add_argument('--max_frames', type=int, default=200, help='Max frames per video')
    parser.add_argument('--scenario', type=str, default=None, help='Only process this scenario type')
    args = parser.parse_args()

    scenarios = scan_scenarios(args.dataset_root)
    if not scenarios:
        print(f'No scenario directories found in {args.dataset_root}')
        return

    # Filter if specific scenario requested
    if args.scenario:
        if args.scenario not in scenarios:
            print(f'Scenario "{args.scenario}" not found. Available: {list(scenarios.keys())}')
            return
        scenarios = {args.scenario: scenarios[args.scenario]}

    print(f'Found {len(scenarios)} scenario types:')
    for stype, routes in scenarios.items():
        print(f'  {stype}: {len(routes)} routes')

    summary_lines = []
    total_videos = 0

    for stype, routes in scenarios.items():
        selected = routes[:args.n_per_type]
        print(f'\n=== {stype} ({len(selected)}/{len(routes)} routes) ===')

        n_generated = 0
        for route_dir in selected:
            route_name = os.path.basename(route_dir)
            output_path = os.path.join(args.output_dir, stype, f'{route_name}.mp4')
            print(f'  Generating {route_name}...', end=' ', flush=True)

            ok = generate_video(route_dir, output_path, stype, args.fps, args.max_frames)
            if ok:
                print('OK')
                n_generated += 1
            else:
                print('SKIPPED')

        summary_lines.append(f'{stype}: {n_generated}/{len(selected)} generated ({len(routes)} available)')
        total_videos += n_generated

    # Write summary
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write(f'Total: {total_videos} videos\n\n')
        for line in summary_lines:
            f.write(line + '\n')

    print(f'\nDone! {total_videos} videos generated → {args.output_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
