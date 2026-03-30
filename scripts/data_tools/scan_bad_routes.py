#!/usr/bin/env python3
"""Scan dataset routes for collisions/violations and generate an exclude list.

Usage:
    python scripts/data_tools/scan_bad_routes.py --dataset_root /media/z/data/dataset/pdm_lite_mini
    python scripts/data_tools/scan_bad_routes.py --dataset_root /workspace1/z_project/dataset/pdm_lite

Output: <dataset_root>/bad_routes.txt  (one route_name per line)

Filtering logic (follows LEAD route_filtering.py):
- EXCLUDE routes with collisions_vehicle, collisions_pedestrian, collisions_layout
- EXCLUDE routes with route_dev (severe route deviation)
- KEEP routes with only min_speed_infractions (common, not harmful for learning)
- KEEP routes with only outside_route_lanes if status=Completed (minor)
"""
import argparse
import gzip
import json
import os


COLLISION_KEYS = {'collisions_vehicle', 'collisions_pedestrian', 'collisions_layout'}
SEVERE_KEYS = COLLISION_KEYS | {'route_dev', 'vehicle_blocked', 'outside_route_lanes'}
# Only tolerable: min_speed_infractions


def scan_route(results_path):
    """Return (route_name, is_bad, violations) for a results.json.gz file."""
    route_dir = os.path.dirname(results_path)
    route_name = os.path.basename(route_dir)

    try:
        with gzip.open(results_path, 'rt') as f:
            data = json.load(f)
    except Exception:
        return route_name, True, ['unreadable_results']

    infractions = data.get('infractions', {})
    status = data.get('status', '')

    # Agent/sim crash = exclude
    if 'crashed' in status.lower() or "couldn't be set up" in status.lower():
        return route_name, True, [f'status:{status}']

    violations = []
    for key, vals in infractions.items():
        if vals and key in SEVERE_KEYS:
            violations.append(f'{key}({len(vals)})')

    is_bad = len(violations) > 0
    return route_name, is_bad, violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--output', default=None, help='Output file (default: <dataset_root>/bad_routes.txt)')
    args = parser.parse_args()

    # Find all results.json.gz
    results_files = []
    for root, dirs, files in os.walk(args.dataset_root):
        for f in files:
            if f == 'results.json.gz':
                results_files.append(os.path.join(root, f))

    print(f"Found {len(results_files)} routes with results.json.gz")

    bad_routes = []
    all_violations = {}
    for rpath in sorted(results_files):
        route_name, is_bad, violations = scan_route(rpath)
        if is_bad:
            bad_routes.append(route_name)
            all_violations[route_name] = violations

    print(f"\nBad routes: {len(bad_routes)}/{len(results_files)}")
    for rn in bad_routes:
        print(f"  {rn}: {all_violations[rn]}")

    # Write exclude list
    output_path = args.output or os.path.join(args.dataset_root, 'bad_routes.txt')
    with open(output_path, 'w') as f:
        for rn in sorted(bad_routes):
            f.write(rn + '\n')
    print(f"\nExclude list written to: {output_path}")


if __name__ == '__main__':
    main()
