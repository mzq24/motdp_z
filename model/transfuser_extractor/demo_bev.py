"""
TransFuser BEV Demo: GT vs Prediction side-by-side visualization
================================================================
Usage:
    python demo_bev.py [--frame FRAME_IDX] [--route ROUTE_DIR]

Loads the full TransFuser model (backbone + BEV semantic decoder + CenterNet head),
runs inference on a single frame, and produces a side-by-side image:
  LEFT  = GT BEV semantics
  RIGHT = Predicted BEV semantics + CenterNet detections overlay
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

from config import GlobalConfig
from transfuser import TransfuserBackbone
import transfuser_utils as t_u


# ── BEV semantic color map (BGR→RGB) ──
BEV_CLASS_NAMES = [
    'unlabeled', 'road', 'sidewalk', 'lane_solid', 'lane_broken',
    'stop_sign', 'light_green', 'light_yellow', 'light_red',
    'vehicle', 'walker'
]
BEV_CLASS_COLORS_RGB = np.array([
    [0,   0,   0],    # unlabeled
    [200, 200, 200],  # road
    [255, 255, 255],  # sidewalk
    [0,   255, 255],  # lane markers solid
    [157, 234, 50],   # lane markers broken
    [255, 100, 0],    # stop sign (orange)
    [0,   200, 0],    # light green
    [255, 255, 0],    # light yellow
    [255, 0,   0],    # light red  (RED, not blue!)
    [0,   150, 255],  # vehicle (blue-ish)
    [255, 0,   255],  # walker (magenta)
], dtype=np.uint8)


# ── CenterNet bbox class names ──
BB_CLASS_NAMES = ['vehicle', 'pedestrian', 'cyclist', 'motorcycle', 'emergency']
BB_CLASS_COLORS = [(30, 170, 250), (0, 255, 0), (255, 0, 255), (255, 128, 0), (0, 0, 255)]


class TransFuserFullModel(nn.Module):
    """TransFuser backbone + BEV semantic decoder + CenterNet detection head."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = TransfuserBackbone(config)

        # BEV semantic decoder (same as in carla_garage model.py)
        self.bev_semantic_decoder = nn.Sequential(
            nn.Conv2d(config.bev_features_chanels, config.bev_features_chanels,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.bev_features_chanels, config.num_bev_semantic_classes,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.Upsample(size=(config.lidar_resolution_height, config.lidar_resolution_width),
                         mode='bilinear', align_corners=False),
        )

        # CenterNet detection head
        self.head_heatmap = self._build_head(config.bb_input_channel, config.num_bb_classes)
        self.head_wh = self._build_head(config.bb_input_channel, 2)
        self.head_offset = self._build_head(config.bb_input_channel, 2)
        self.head_yaw_class = self._build_head(config.bb_input_channel, config.num_dir_bins)
        self.head_yaw_res = self._build_head(config.bb_input_channel, 1)

    @staticmethod
    def _build_head(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
        )

    def forward(self, rgb, lidar_bev):
        bev_feature_grid, fused_features, image_feature_grid = self.backbone(rgb, lidar_bev)

        # BEV semantic segmentation
        pred_bev_semantic = self.bev_semantic_decoder(bev_feature_grid)

        # CenterNet detection
        heatmap = self.head_heatmap(bev_feature_grid).sigmoid()
        wh = self.head_wh(bev_feature_grid)
        offset = self.head_offset(bev_feature_grid)
        yaw_class = self.head_yaw_class(bev_feature_grid)
        yaw_res = self.head_yaw_res(bev_feature_grid)

        return {
            'bev_semantic': pred_bev_semantic,       # (B, 11, 256, 256)
            'heatmap': heatmap,                       # (B, 5, H', W')
            'wh': wh,                                 # (B, 2, H', W')
            'offset': offset,                         # (B, 2, H', W')
            'yaw_class': yaw_class,                   # (B, 12, H', W')
            'yaw_res': yaw_res,                       # (B, 1, H', W')
            'bev_feature_grid': bev_feature_grid,     # for reference
        }


def load_config(config_path):
    config_file = os.path.join(config_path, 'config.json')
    with open(config_file, 'rt', encoding='utf-8') as f:
        loaded_config = jsonpickle.decode(f.read())
    config = GlobalConfig()
    config.__dict__.update(loaded_config.__dict__)
    return config


def load_model(config_path, model_path, device='cuda:0'):
    config = load_config(config_path)
    model = TransFuserFullModel(config)

    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

    # Map ckpt keys to our model
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_sd[k] = v  # backbone.* maps directly
        elif k.startswith('bev_semantic_decoder.'):
            new_sd[k] = v  # bev_semantic_decoder.* maps directly
        elif k.startswith('head.heatmap_head.'):
            new_sd[k.replace('head.heatmap_head.', 'head_heatmap.')] = v
        elif k.startswith('head.wh_head.'):
            new_sd[k.replace('head.wh_head.', 'head_wh.')] = v
        elif k.startswith('head.offset_head.'):
            new_sd[k.replace('head.offset_head.', 'head_offset.')] = v
        elif k.startswith('head.yaw_class_head.'):
            new_sd[k.replace('head.yaw_class_head.', 'head_yaw_class.')] = v
        elif k.startswith('head.yaw_res_head.'):
            new_sd[k.replace('head.yaw_res_head.', 'head_yaw_res.')] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    model.to(device).eval()
    return model, config


def preprocess_rgb(config, rgb_path):
    image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = t_u.crop_array(config, image)
    image = np.transpose(image, (2, 0, 1))  # (3, H, W)
    return torch.from_numpy(image).float().unsqueeze(0)


def preprocess_lidar(config, lidar_path):
    las = laspy.read(lidar_path)
    lidar = las.xyz.copy()

    def splat_points(pc):
        xbins = np.linspace(config.min_x, config.max_x,
                            (config.max_x - config.min_x) * int(config.pixels_per_meter) + 1)
        ybins = np.linspace(config.min_y, config.max_y,
                            (config.max_y - config.min_y) * int(config.pixels_per_meter) + 1)
        hist = np.histogramdd(pc[:, :2], bins=(xbins, ybins))[0]
        hist[hist > config.hist_max_per_pixel] = config.hist_max_per_pixel
        return (hist / config.hist_max_per_pixel).T

    lidar = lidar[lidar[..., 2] < config.max_height_lidar]
    below = lidar[lidar[..., 2] <= config.lidar_split_height]
    above = lidar[lidar[..., 2] > config.lidar_split_height]

    if config.use_ground_plane:
        features = np.stack([splat_points(below), splat_points(above)], axis=-1)
    else:
        features = np.stack([splat_points(above)], axis=-1)

    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return torch.from_numpy(features).unsqueeze(0)


def load_gt_bev_semantic(config, bev_sem_path):
    """Load GT BEV semantic label and apply the same preprocessing as training."""
    bev_sem = cv2.imread(bev_sem_path, cv2.IMREAD_UNCHANGED)  # (256, 256), uint8 class ids

    # Match training preprocessing: crop center 128x128, then repeat 2x
    if config.pixels_per_meter == 4.0:
        bev_sem = bev_sem[64:192, 64:192].repeat(2, axis=0).repeat(2, axis=1)

    # Apply bev_converter mapping
    bev_converter = np.array(config.bev_converter)
    bev_sem = bev_converter[bev_sem]
    return bev_sem  # (256, 256), class indices 0-10


def semantic_to_rgb(class_map, color_lut=BEV_CLASS_COLORS_RGB):
    """Convert class index map to RGB image."""
    return color_lut[class_map]


def rotate_bev_ego_up(img):
    """
    Rotate BEV image so ego faces UP.

    Raw BEV convention (after lidar histogram .T):
      row = y (left→right),  col = x (behind→ahead)
      → ego faces RIGHT in the raw image.

    After np.rot90(k=1) (90° CCW):
      UP = ahead,  DOWN = behind,  LEFT = left,  RIGHT = right
    """
    return np.rot90(img, k=1).copy()


def draw_ego(bev_rgb, config):
    """Draw ego vehicle at BEV center (assumes ego-up orientation)."""
    h, w = bev_rgb.shape[:2]
    cx, cy = w // 2, h // 2
    ppm = config.pixels_per_meter  # 4.0
    ego_half_l = int(4.5 / 2 * ppm)  # ~9 px  (length along forward = vertical)
    ego_half_w = int(2.0 / 2 * ppm)  # ~4 px  (width along lateral = horizontal)
    # Ego rectangle (length vertical, width horizontal)
    pt1 = (cx - ego_half_w, cy - ego_half_l)
    pt2 = (cx + ego_half_w, cy + ego_half_l)
    overlay = bev_rgb.copy()
    cv2.rectangle(overlay, pt1, pt2, (255, 80, 80), -1)
    cv2.addWeighted(overlay, 0.5, bev_rgb, 0.5, 0, bev_rgb)
    cv2.rectangle(bev_rgb, pt1, pt2, (255, 50, 50), 2)
    # Forward arrow (pointing up)
    arrow_start = (cx, cy)
    arrow_end = (cx, cy - ego_half_l - 8)
    cv2.arrowedLine(bev_rgb, arrow_start, arrow_end, (255, 50, 50), 2, tipLength=0.4)
    cv2.putText(bev_rgb, "EGO", (cx - 14, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return bev_rgb


def decode_centernet_detections(output, config, bev_size, score_thresh=0.3):
    """
    Decode CenterNet heatmap + regression outputs into bounding boxes.
    Returns detections with coordinates in the ROTATED (ego-up) BEV image space.

    Raw feature grid convention:  row=y(left→right), col=x(behind→ahead)
    After 90° CCW rotation to ego-up:
        new_col = old_row,  new_row = (W-1) - old_col
    """
    heatmap = output['heatmap'][0].cpu().numpy()   # (5, H, W)
    wh = output['wh'][0].cpu().numpy()             # (2, H, W)
    offset = output['offset'][0].cpu().numpy()     # (2, H, W)
    yaw_class = output['yaw_class'][0].cpu().numpy()  # (12, H, W)
    yaw_res = output['yaw_res'][0].cpu().numpy()      # (1, H, W)

    num_classes, feat_h, feat_w = heatmap.shape
    # Scale from feature grid to full BEV image (before rotation)
    scale_row = bev_size / feat_h
    scale_col = bev_size / feat_w

    detections = []

    for cls_id in range(num_classes):
        hm = heatmap[cls_id]
        hm_pool = _pool_nms(hm)
        mask = (hm == hm_pool) & (hm >= score_thresh)
        ys, xs = np.where(mask)

        for y, x in zip(ys, xs):
            score = hm[y, x]
            # Raw feature grid coords (row=y_lateral, col=x_forward)
            raw_col = x + offset[0, y, x]  # x-offset in col direction
            raw_row = y + offset[1, y, x]  # y-offset in row direction
            box_w = abs(wh[0, y, x])  # width in col (forward) direction
            box_h = abs(wh[1, y, x])  # height in row (lateral) direction

            # Scale to full BEV image coords
            raw_col *= scale_col
            raw_row *= scale_row
            box_w *= scale_col
            box_h *= scale_row

            # Apply 90° CCW rotation: (row, col) → (new_row, new_col)
            # new_col = raw_row,  new_row = (bev_size - 1) - raw_col
            rot_cx = raw_row
            rot_cy = (bev_size - 1) - raw_col
            # box dimensions swap: forward-length → vertical, lateral-width → horizontal
            rot_w = box_h  # lateral becomes horizontal
            rot_h = box_w  # forward becomes vertical

            # Decode yaw (in raw BEV frame) and rotate by -90°
            yaw_cls = np.argmax(yaw_class[:, y, x])
            yaw_r = yaw_res[0, y, x]
            num_bins = config.num_dir_bins
            bin_size = 2 * np.pi / num_bins
            yaw = yaw_cls * bin_size + yaw_r - np.pi
            yaw_rotated = yaw - np.pi / 2  # 90° CCW rotation

            detections.append({
                'cx': rot_cx, 'cy': rot_cy,
                'w': rot_w, 'h': rot_h,
                'yaw': yaw_rotated,
                'cls': cls_id, 'score': float(score),
            })

    return detections


def _pool_nms(heatmap, kernel=3):
    """Simple max-pool NMS on a 2D heatmap."""
    from scipy.ndimage import maximum_filter
    return maximum_filter(heatmap, size=kernel)


def draw_detections_on_bev(bev_rgb, detections, config):
    """Draw CenterNet detections as oriented bboxes on the rotated (ego-up) BEV image."""
    for det in detections:
        cx, cy = det['cx'], det['cy']
        w, h = det['w'], det['h']
        yaw = det['yaw']
        cls_id = det['cls']
        score = det['score']
        color = BB_CLASS_COLORS[cls_id]

        cos_a, sin_a = np.cos(yaw), np.sin(yaw)
        dx, dy = w / 2, h / 2
        corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = (rot @ corners.T).T + np.array([cx, cy])
        corners = corners.astype(np.int32)

        cv2.polylines(bev_rgb, [corners], isClosed=True, color=color, thickness=2)
        label = f"{BB_CLASS_NAMES[cls_id]} {score:.2f}"
        cv2.putText(bev_rgb, label, (int(cx) - 10, int(cy) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return bev_rgb


def make_legend(class_names, class_colors, title="BEV Classes"):
    """Create a vertical legend image."""
    line_h = 22
    pad = 10
    w = 180
    h = pad * 2 + len(class_names) * line_h
    legend = np.ones((h, w, 3), dtype=np.uint8) * 30  # dark bg

    for i, (name, color) in enumerate(zip(class_names, class_colors)):
        y = pad + i * line_h
        cv2.rectangle(legend, (pad, y), (pad + 14, y + 14), tuple(int(c) for c in color), -1)
        cv2.putText(legend, name, (pad + 20, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    return legend


def find_route(dataset_root):
    """Find the first valid route directory."""
    for scenario in sorted(Path(dataset_root).iterdir()):
        if not scenario.is_dir() or scenario.name.startswith(('.', 'tmp', 'tar', 'inspect')):
            continue
        for route in sorted(scenario.iterdir()):
            if not route.is_dir():
                continue
            if (route / 'rgb').exists() and (route / 'lidar').exists():
                return str(route)
    return None


def main():
    parser = argparse.ArgumentParser(description='TransFuser BEV Demo')
    parser.add_argument('--config_path', type=str,
                        default='/media/z/data/models/garage2/pretrained_models/all_towns')
    parser.add_argument('--model_path', type=str,
                        default='/media/z/data/models/garage2/pretrained_models/all_towns/model_0030_1.pth')
    parser.add_argument('--dataset', type=str,
                        default='/media/z/data/dataset/pdm_lite_mini')
    parser.add_argument('--route', type=str, default=None,
                        help='Specific route directory. Auto-detected if not given.')
    parser.add_argument('--frame', type=int, default=10,
                        help='Frame index to visualize')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--score_thresh', type=float, default=0.2)
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path. Default: demo_bev_output.png')
    args = parser.parse_args()

    # ── Find route ──
    if args.route:
        route_dir = args.route
    else:
        route_dir = find_route(args.dataset)
    if route_dir is None:
        print("No valid route found!")
        return
    print(f"Route: {route_dir}")

    # ── Check files exist ──
    frame_str = f"{args.frame:04d}"
    rgb_path = os.path.join(route_dir, 'rgb', f'{frame_str}.jpg')
    lidar_path = os.path.join(route_dir, 'lidar', f'{frame_str}.laz')
    bev_sem_path = os.path.join(route_dir, 'bev_semantics', f'{frame_str}.png')

    for p in [rgb_path, lidar_path, bev_sem_path]:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            return

    # ── Load model ──
    print("Loading model...")
    model, config = load_model(args.config_path, args.model_path, args.device)
    print(f"Model loaded. BEV semantic classes: {config.num_bev_semantic_classes}, "
          f"Detection classes: {config.num_bb_classes}")

    # ── Preprocess inputs ──
    print("Preprocessing inputs...")
    rgb = preprocess_rgb(config, rgb_path)
    lidar_bev = preprocess_lidar(config, lidar_path)
    print(f"  RGB: {rgb.shape}, LiDAR BEV: {lidar_bev.shape}")

    # ── Inference ──
    print("Running inference...")
    with torch.no_grad():
        output = model(rgb.to(args.device), lidar_bev.to(args.device))

    # ── Decode predictions ──
    # BEV semantic
    pred_bev_sem = output['bev_semantic'][0].cpu().numpy()  # (11, 256, 256)
    pred_classes = np.argmax(pred_bev_sem, axis=0)  # (256, 256)
    bev_size = pred_classes.shape[0]  # 256

    # Rotate to ego-up then colorize
    pred_classes_rot = rotate_bev_ego_up(pred_classes)
    pred_rgb = semantic_to_rgb(pred_classes_rot)

    # CenterNet detections (coords already in rotated ego-up space)
    detections = decode_centernet_detections(output, config, bev_size, args.score_thresh)
    print(f"  Detections: {len(detections)} objects")
    for det in detections:
        print(f"    {BB_CLASS_NAMES[det['cls']]:12s} score={det['score']:.3f} "
              f"pos=({det['cx']:.1f},{det['cy']:.1f}) wh=({det['w']:.1f},{det['h']:.1f})")

    # Draw ego + detections on predicted BEV
    pred_with_det = pred_rgb.copy()
    draw_ego(pred_with_det, config)
    draw_detections_on_bev(pred_with_det, detections, config)

    # ── Load GT ──
    gt_classes = load_gt_bev_semantic(config, bev_sem_path)
    gt_classes_rot = rotate_bev_ego_up(gt_classes)
    gt_rgb = semantic_to_rgb(gt_classes_rot)
    draw_ego(gt_rgb, config)

    # ── Load RGB for context ──
    rgb_vis = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB)
    bev_h = gt_rgb.shape[0]
    rgb_scale = bev_h / rgb_vis.shape[0]
    rgb_vis = cv2.resize(rgb_vis, (int(rgb_vis.shape[1] * rgb_scale), bev_h))

    # ── Load LiDAR BEV for context ──
    lidar_vis = lidar_bev[0, 0].numpy()  # first channel
    lidar_vis = rotate_bev_ego_up(lidar_vis)  # rotate to ego-up
    lidar_vis = (lidar_vis * 255).clip(0, 255).astype(np.uint8)
    lidar_vis = cv2.applyColorMap(lidar_vis, cv2.COLORMAP_HOT)
    lidar_vis = cv2.cvtColor(lidar_vis, cv2.COLOR_BGR2RGB)
    lidar_vis = cv2.resize(lidar_vis, (bev_h, bev_h))

    # ── Create legend ──
    legend = make_legend(BEV_CLASS_NAMES, BEV_CLASS_COLORS_RGB, "BEV Semantic Classes")
    # Resize legend to match BEV height
    legend_scale = bev_h / legend.shape[0]
    legend = cv2.resize(legend, (int(legend.shape[1] * legend_scale), bev_h),
                        interpolation=cv2.INTER_NEAREST)

    # ── Compose final figure ──
    # Row 1: RGB camera image (full width)
    # Row 2: LiDAR BEV | GT BEV semantic | Pred BEV semantic + detections | Legend
    row2 = np.concatenate([lidar_vis, gt_rgb, pred_with_det, legend], axis=1)

    # Resize RGB to match row2 width
    rgb_vis = cv2.resize(rgb_vis, (row2.shape[1], bev_h))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rgb_vis, f"RGB Frame {frame_str}", (10, 25), font, 0.7, (255, 255, 0), 2)
    cv2.putText(row2, "LiDAR BEV", (10, 25), font, 0.6, (255, 255, 0), 2)
    cv2.putText(row2, "GT BEV Semantic", (bev_h + 10, 25), font, 0.6, (255, 255, 0), 2)
    cv2.putText(row2, f"Pred BEV + Det (n={len(detections)})", (bev_h * 2 + 10, 25),
                font, 0.6, (255, 255, 0), 2)

    final = np.concatenate([rgb_vis, row2], axis=0)

    # ── Save ──
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), 'demo_bev_output.png')
    cv2.imwrite(output_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    print(f"\nSaved to: {output_path}")
    print(f"Image size: {final.shape[1]}x{final.shape[0]}")

    # Also save individual panels for inspection
    panels_dir = os.path.join(os.path.dirname(output_path), 'demo_panels')
    os.makedirs(panels_dir, exist_ok=True)
    cv2.imwrite(os.path.join(panels_dir, 'gt_bev_semantic.png'),
                cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(panels_dir, 'pred_bev_semantic.png'),
                cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(panels_dir, 'pred_bev_with_det.png'),
                cv2.cvtColor(pred_with_det, cv2.COLOR_RGB2BGR))
    print(f"Individual panels saved to: {panels_dir}")


if __name__ == '__main__':
    main()
