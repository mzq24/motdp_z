"""
Offline evaluation of ego localization strategies.

Reads meta/*.json from an existing eval run (which logged gps_raw, gps_filtered,
hero_location_world, compass_raw, speed, steer, throttle, brake) and simulates
each EgoLocalizer strategy to compute error vs GT.

Usage:
    python eval_localizer_offline.py \
        --base_folder /path/to/Bench2Drive/eval_M_route_b_constructed_gtpose_0325_gt \
        --max_scenes 10

Output: per-scene and overall error statistics for each strategy.
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

# Add parent dirs so ego_localizer can be imported
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ego_localizer import EgoLocalizer, _normalize_angle


def preprocess_compass(compass_raw):
    """Same as transfuser_utils.preprocess_compass."""
    if math.isnan(compass_raw):
        compass_raw = 0.0
    return _normalize_angle(compass_raw - np.deg2rad(90.0))


def evaluate_scene(scene_meta_dir, strategies, dt=0.05):
    """
    Run all strategies on one scene's meta files.
    Returns dict: strategy_name -> list of xy errors vs GT.
    """
    meta_files = sorted(glob.glob(os.path.join(scene_meta_dir, '*.json')))
    if not meta_files:
        return None

    # Initialize localizers
    localizers = {}
    for name, kwargs in strategies.items():
        localizers[name] = EgoLocalizer(dt=dt, **kwargs)

    errors = {name: [] for name in strategies}
    errors['raw'] = []
    errors['filtered'] = []

    for f in meta_files:
        with open(f) as fp:
            d = json.load(fp)

        # Need these fields
        if 'hero_location_world' not in d or 'gps_raw' not in d:
            continue

        hero_xy = np.array(d['hero_location_world'][:2], dtype=np.float64)
        gps_raw = np.array(d['gps_raw'], dtype=np.float64)
        speed = d.get('speed', 0.0)
        compass_raw = d.get('compass_raw', 0.0)
        compass_filtered = d.get('compass_filtered', compass_raw)
        steer = d.get('steer', 0.0)
        throttle = d.get('throttle', 0.0)
        brake = d.get('brake', 0.0)

        # Use compass_raw after preprocessing (same pipeline as agent)
        compass = compass_raw  # already preprocessed in agent before saving to meta

        # Baseline errors
        gps_filtered = np.array(d.get('gps_filtered', gps_raw), dtype=np.float64)
        errors['raw'].append(float(np.linalg.norm(gps_raw - hero_xy)))
        errors['filtered'].append(float(np.linalg.norm(gps_filtered - hero_xy)))

        # Test each strategy
        for name, loc in localizers.items():
            pos, yaw = loc.update(
                gps_xy=gps_raw,
                compass=compass,
                speed=speed,
                steer=steer,
                throttle=throttle,
                brake=brake,
            )
            err = float(np.linalg.norm(pos - hero_xy))
            errors[name].append(err)

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, required=True,
                        help='Path to eval_M_* directory with scene folders')
    parser.add_argument('--max_scenes', type=int, default=20)
    parser.add_argument('--dt', type=float, default=0.05)
    args = parser.parse_args()

    # Define strategies to test
    strategies = {
        'raw': dict(strategy='raw', latency_compensation=False),
        'comp_a50': dict(strategy='complementary', gps_alpha=0.50,
                          latency_compensation=False),
        'comp_a30': dict(strategy='complementary', gps_alpha=0.30,
                          latency_compensation=False),
        'comp_a10': dict(strategy='complementary', gps_alpha=0.10,
                          latency_compensation=False),
        'comp_a05': dict(strategy='complementary', gps_alpha=0.05,
                          latency_compensation=False),
        'comp_a02': dict(strategy='complementary', gps_alpha=0.02,
                          latency_compensation=False),
        'comp_a01': dict(strategy='complementary', gps_alpha=0.01,
                          latency_compensation=False),
        'ema': dict(strategy='ema', latency_compensation=False),
        'ema_dr_a02': dict(strategy='ema_dr', gps_alpha=0.02,
                            latency_compensation=False),
        'ema_dr_a05': dict(strategy='ema_dr', gps_alpha=0.05,
                            latency_compensation=False),
        'ukf_tuned': dict(strategy='ukf_tuned', latency_compensation=False),
    }

    scene_dirs = sorted(glob.glob(os.path.join(args.base_folder, '*/meta')))
    if not scene_dirs:
        print(f"No scene/meta dirs found in {args.base_folder}")
        return

    scene_dirs = scene_dirs[:args.max_scenes]

    # Collect all errors
    all_labels = ['raw', 'filtered'] + list(strategies.keys())
    all_errors = {label: [] for label in all_labels}

    for scene_meta in scene_dirs:
        scene_name = os.path.basename(os.path.dirname(scene_meta))
        result = evaluate_scene(scene_meta, strategies, dt=args.dt)
        if result is None:
            continue

        print(f"\n{'='*70}")
        print(f"Scene: {scene_name[:65]}")
        print(f"  Frames: {len(result['raw'])}")
        print(f"  {'Strategy':<25s} {'Mean':>7s} {'Max':>7s} {'Std':>7s}")
        print(f"  {'-'*50}")
        for label in all_labels:
            errs = result[label]
            if errs:
                all_errors[label].extend(errs)
                print(f"  {label:<25s} {np.mean(errs):7.3f} {np.max(errs):7.3f} {np.std(errs):7.3f}")

    # Overall summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY ({len(scene_dirs)} scenes)")
    print(f"  {'Strategy':<25s} {'Mean':>7s} {'Max':>7s} {'Std':>7s} {'Median':>7s}")
    print(f"  {'-'*55}")
    for label in all_labels:
        errs = all_errors[label]
        if errs:
            print(f"  {label:<25s} {np.mean(errs):7.3f} {np.max(errs):7.3f} "
                  f"{np.std(errs):7.3f} {np.median(errs):7.3f}")


if __name__ == '__main__':
    main()
